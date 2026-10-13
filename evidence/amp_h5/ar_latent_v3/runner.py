#!/usr/bin/env python3
"""Single authorized natural run. External supervisor owns H5 lock and 660s kill."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import time
import traceback
import numpy as np
import torch
from core import AR, configure_cpu, initial_models, teacher_batch, extend_z, digest, state_digest, loss_parts, decode
from contracts import admission, prepare, dump, record, sha, verify_bindings, load_json
from scoring import reconstruction, score_generation, information_gate

class BudgetExpired(RuntimeError): pass

class Runtime:
    def __init__(self,out):
        self.out=out;self.started=time.monotonic();self.deadline=self.started+660;self.phase_deadline=self.deadline
        self.counts={'updates':0,'training_forward':0,'dev_teacher_forced_forward':0,'dev_one_token_forward':0,
            'generation_forward':0,'encoder_forward':0,'DDPM_forward':0,'D_forward':0,'new_generation_requests':0}
        self.phase='admission'
    def phase_start(self,name,seconds):
        self.phase=name;self.phase_deadline=min(self.deadline,time.monotonic()+seconds)
        signal.setitimer(signal.ITIMER_REAL,max(.001,self.phase_deadline-time.monotonic()))
        self.save()
    def check(self):
        if time.monotonic()>=self.phase_deadline: raise BudgetExpired(self.phase)
    def forward(self,kind):
        self.check();self.counts[kind]+=1
        limits={'training_forward':3000,'dev_teacher_forced_forward':28,'dev_one_token_forward':1372,'generation_forward':147}
        assert self.counts[kind]<=limits[kind]
        assert sum(self.counts[k] for k in limits)<=4547
    def save(self):
        dump(self.out/'COUNTS.json', {'phase':self.phase,'counts':self.counts,'elapsed_seconds':time.monotonic()-self.started})

def alarm(_signum,_frame): raise BudgetExpired('phase hard timer')

def condition_names():
    return ([f'{arm}_{mode}' for arm in 'AB' for mode in ['mu']+[f'S{s}' for s in range(1422,1430)]]+
        ['C_plain','A_zero']+[f'A_mismatch_S{s}' for s in range(1422,1430)])

def placeholders():
    return [{'arm':a,'terminal':t,'slot':i,'sequence':None,'status':'not_executed'} for a in 'ABCD' for t in ('FREE','AMD') for i in range(16)]

def save_decode(path,result):
    np.savez(path,**{k:v.numpy() for k,v in result.items() if isinstance(v,torch.Tensor)})

def train_arm(arm,model,data,out,rt,expected_z):
    started=time.monotonic();inner_deadline=started+120
    armout=out/arm;armout.mkdir()
    batch=teacher_batch([r['sequence'] for r in data['train']],data['labels'][:279])
    q=data['q'][:279]
    mu,sigma=q[:,0],q[:,1].exp()
    assert bool(torch.isfinite(sigma).all())
    generator=torch.Generator(device='cpu').manual_seed(4422)
    optimizer=torch.optim.Adam(model.parameters(),lr=.001,betas=(.9,.999),eps=1e-8,weight_decay=0,foreach=False,fused=False)
    completed=0;attempted=0;failure=None;zrecords=[]
    initial_rng=generator.get_state().clone()
    torch.save({'model':model.state_dict(),'optimizer':optimizer.state_dict(),'updates':0,
        'train_rng_state':initial_rng,'state_sha256':state_digest(model)},armout/'initial.pt')
    with (armout/'steps.jsonl').open('x') as stream:
        try:
            model.train()
            for step,ids in enumerate(data['batches'],1):
                rt.check()
                if time.monotonic()>=inner_deadline: raise BudgetExpired('training inner120')
                attempted=step
                rng_before=digest(generator.get_state()) if arm!='C' else None
                if arm!='C':
                    zbank=mu+sigma*torch.randn((279,16,48),generator=generator,dtype=torch.float32)
                    assert bool(torch.isfinite(zbank).all()),'nonfinite train zbank'
                    zdigest=digest(zbank)
                    if arm=='B': assert zdigest==expected_z[step-1], 'A/B sampled bank mismatch'
                    zrecords.append(zdigest)
                    z=extend_z(zbank[data['maps'][step-1]][ids] if arm=='B' else zbank[ids])
                else: z=None;zdigest=None
                optimizer.zero_grad(set_to_none=True)
                rt.forward('training_forward')
                logits,_=model(batch['inputs'][ids],batch['labels'][ids],z)
                assert bool(torch.isfinite(logits).all()),'nonfinite training logits'
                parts,_=loss_parts(logits,batch['targets'][ids],batch['mask'][ids])
                parts['loss'].backward()
                grad=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
                optimizer.step();completed=step;rt.counts['updates']+=1
                assert all(bool(torch.isfinite(p).all()) for p in model.parameters()),'nonfinite post-update parameters'
                log={'step':step,'batch_indices':ids.tolist(),'batch_digest':digest(ids),'zbank_digest':zdigest,
                    'mapping_digest':digest(data['maps'][step-1]) if arm=='B' else None,
                    'train_rng_before_digest':rng_before,'train_rng_after_digest':digest(generator.get_state()) if arm!='C' else None,
                    'preclip_grad_l2':float(grad),'elapsed_seconds':time.monotonic()-started,
                    **{k:v.detach().item() for k,v in parts.items()}}
                stream.write(json.dumps(log,allow_nan=False)+'\n');stream.flush()
                rt.save()
            if time.monotonic()>=inner_deadline: raise BudgetExpired('training inner120 crossed on last update')
            status='COMPLETE'
        except BaseException as exc:
            status='FAILED';failure={'type':type(exc).__name__,'message':str(exc),'traceback':traceback.format_exc()}
        finally:
            # SIGALRM is one-shot. After it fires, global supervisor660 remains
            # the hard fallback; elapsed outer150 overrun is a resource failure.
            torch.save({'model':model.state_dict(),'optimizer':optimizer.state_dict(),'updates':completed,'attempted':attempted,
                'train_rng_initial':initial_rng,'train_rng_final':generator.get_state(),'state_sha256':state_digest(model)},armout/'final.pt')
            elapsed=time.monotonic()-started
            outer_overrun=time.monotonic()>rt.phase_deadline
            if elapsed>120 or outer_overrun:
                status='FAILED'
                failure=failure or {'type':'BudgetExpired','message':'training elapsed budget exceeded including checkpoint I/O'}
            result={'status':status,'inner120_exceeded':elapsed>120,'outer150_exceeded':outer_overrun,'completed_updates':completed,'attempted_updates':attempted,'failure':failure,
                'seconds':time.monotonic()-started,'final_state_sha256':state_digest(model),'checkpoint_selection':'final_only'}
            dump(armout/'TRAIN_RESULT.json',result)
    return result,zrecords

def evaluate(models,data,out,rt,slots,dev_results):
    directory=out/'evaluation';directory.mkdir()
    q=data['q'][279:];mu=q[:,0];sigma=q[:,1].exp()
    labels=data['labels'][279:]
    batch=teacher_batch([r['sequence'] for r in data['dev']],labels)
    for name in condition_names():
        rt.check();arm,mode=name.split('_',1);model=models[arm];model.eval()
        if mode=='plain': z=None
        elif mode=='mu': z=mu
        elif mode=='zero': z=torch.zeros_like(mu)
        else:
            seed=int(mode.rsplit('S',1)[1]);z=mu+sigma*data['eps'][seed]
            if mode.startswith('mismatch'): z=z[data['devmaps'][seed-1422]]
        if z is not None: assert bool(torch.isfinite(z).all())
        with torch.no_grad():
            rt.forward('dev_teacher_forced_forward')
            logits,_=model(batch['inputs'],labels,None if z is None else extend_z(z))
            assert bool(torch.isfinite(logits).all())
            parts,tokenloss=loss_parts(logits,batch['targets'],batch['mask'])
            np.savez(directory/(name+'_teacher.npz'),logits=logits.numpy(),probabilities=logits.softmax(-1).numpy(),
                target_tokens=batch['targets'].numpy(),inputs=batch['inputs'].numpy(),observed_mask=batch['mask'].numpy(),
                token_ce=tokenloss.numpy(),labels=labels.numpy())
            result=decode(model,labels,z=z,before_forward=lambda:rt.forward('dev_one_token_forward'))
        save_decode(directory/(name+'_free.npz'),result)
        dev_results[name]={'status':'FAILED' if result['failure'] else 'COMPLETE','failure':result['failure'],
            'NLL':{k:v.item() for k,v in parts.items()},
            'reconstruction':reconstruction([r['sequence'] for r in data['dev']],result),
            'z_digest':digest(z) if z is not None else None,'free_steps':result['steps']}
        dump(out/'DEVELOPMENT.json',dev_results);rt.save()
        if result['failure']: raise RuntimeError('dev decode failure: '+name)
    generation_labels=torch.tensor([0]*16+[1]*16)
    for arm in 'ABC':
        rt.check();rt.counts['new_generation_requests']+=32
        result=decode(models[arm],generation_labels,z=data['generation_z'] if arm!='C' else None,
            uniforms=data['uniforms'],before_forward=lambda:rt.forward('generation_forward'))
        save_decode(directory/(arm+'_generation.npz'),result)
        for i,(seq,status) in enumerate(zip(result['sequences'],result['status'])):
            slot=next(r for r in slots if r['arm']==arm and r['terminal']==('FREE' if i<16 else 'AMD') and r['slot']==i%16)
            slot.update(sequence=seq,status=status,terminated=bool(result['terminated'][i]),steps=result['steps'])
        dump(out/'GENERATION_SLOTS.json',slots);rt.save()
        if result['failure']: raise RuntimeError('generation decode failure: '+arm)
    rt.check()
    return score_generation(slots,data['train'],data['dev'],rt.check)

def run(args):
    out=Path(args.output).resolve()
    here=Path(__file__).resolve().parent
    assert out.parent==here/'runs', 'output must be a new child of v3/runs'
    # Root supervisor's all-H5 lock is mandatory; no direct run authorization.
    lock=here.parent.parent/'H5_PROCESS.lock'
    assert lock.read_text().strip()==str(os.getppid()),'supervisor lock not held by parent'
    c=admission(args.config,args.receipt,args.receipt_sha256)
    out.mkdir(parents=True,exist_ok=False)
    rt=Runtime(out);signal.signal(signal.SIGALRM,alarm)
    slots=placeholders();dev_results={name:{'status':'not_executed','sequences_requested':41} for name in condition_names()}
    results={'A':{'status':'not_executed'},'B':{'status':'not_executed'},'C':{'status':'not_executed'}}
    dump(out/'GENERATION_SLOTS.json',slots);dump(out/'DEVELOPMENT.json',dev_results)
    dump(out/'ADMISSION.json',{'config':record(args.config),'receipt':record(args.receipt),'code':[record(here/n) for n in ('core.py','contracts.py','runner.py','scoring.py')]})
    status='FAILED';failure=None;generation=None;models={};data=None
    try:
        rt.phase_start('preparation',60);configure_cpu();data=prepare(c,out)
        for d in data['D']:
            next(r for r in slots if r['arm']=='D' and r['terminal']==d['terminal'] and r['slot']==d['slot']).update(d)
        dump(out/'GENERATION_SLOTS.json',slots)
        models,identity=initial_models()
        torch.save({a:m.state_dict() for a,m in models.items()},out/'INITIAL_MODELS.pt')
        torch.save(identity,out/'INITIAL_IDENTITY.pt')
        dump(out/'INITIAL_SEAL.json',{'states':{a:state_digest(m) for a,m in models.items()},
            'initial_models_sha256':sha(out/'INITIAL_MODELS.pt'),'identity_sha256':sha(out/'INITIAL_IDENTITY.pt'),
            'shared_common_sha256':identity['common_sha256'],'shared_A_B_sha256':identity['A_B_sha256'],
            'before_any_model_forward':True})
        actual_preparation = {
            'initial_state_sha256': {a:state_digest(m) for a,m in models.items()},
            'batch_indices_sha256': hashlib.sha256(json.dumps([b.tolist() for b in data['batches']],separators=(',',':')).encode()).hexdigest(),
            'train_mappings_digest': digest(data['maps']),
            'dev_mappings_digest': digest(data['devmaps']),
            'uniforms_digest': digest(data['uniforms'])}
        assert load_json(args.receipt)['deterministic_preparation'] == actual_preparation, 'pre-forward preparation differs from independent seal'
        dump(out/'INDEPENDENT_PREPARATION_PARITY.json',actual_preparation)
        zrecords=None
        for arm in 'ABC':
            rt.phase_start('training_'+arm,150)
            results[arm],zr=train_arm(arm,models[arm],data,out,rt,zrecords)
            dump(out/'TRAINING.json',results)
            if arm=='A':zrecords=zr
            if results[arm]['status']!='COMPLETE' or results[arm]['completed_updates']!=1000:
                raise RuntimeError('training failed; no retry or unequal-update comparison')
        rt.phase_start('evaluation_and_scoring',120)
        generation=evaluate(models,data,out,rt,slots,dev_results)
        dump(out/'GENERATION_METRICS.json',generation)
        status='COMPLETE'
    except BaseException as exc:
        failure={'type':type(exc).__name__,'message':str(exc),'traceback':traceback.format_exc()}
        dump(out/'FAILURE.json',failure)
    finally:
        rt.phase_start('closure',30)
        try:
            verify_bindings(c)
            assert sha(args.receipt)==args.receipt_sha256
            assert sha(args.config)==load_json(args.receipt)['config_sha256']
            assert load_json(args.receipt)['code_sha256']=={str(here/n):sha(here/n) for n in ('core.py','contracts.py','runner.py','scoring.py')}
            inf=information_gate(dev_results)
            summary={'status':status,'failure':failure,'training':results,'information_gate':inf,
                'generation_gate':generation['status'] if generation else 'NOT_EVALUABLE',
                'equal_update_comparison':all(r.get('completed_updates')==1000 for r in results.values()),
                'counts':rt.counts,'elapsed_seconds':time.monotonic()-rt.started,'no_retry':True,
                'PROMOTE':'HOLD','BACKUP':'BACKUP_BLOCKED','test_reads':0,
                'interpretation':'One train seed, fixed dev41, 32 cached z; no biological efficacy or submission readiness.'}
            dump(out/'SUMMARY.json',summary);dump(out/'TRAINING.json',results)
            dump(out/'GENERATION_SLOTS.json',slots);dump(out/'DEVELOPMENT.json',dev_results);rt.save()
            dump(out/'MANIFEST.json',{'files':[record(p) for p in sorted(out.rglob('*')) if p.is_file() and p.name!='MANIFEST.json'],
                'input_hashes_verified_at_closure':True,'receipt_sha256':args.receipt_sha256})
        finally:signal.setitimer(signal.ITIMER_REAL,0)
    return 0 if status=='COMPLETE' else 1

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--receipt',required=True)
    p.add_argument('--receipt-sha256',required=True);p.add_argument('--output',required=True)
    raise SystemExit(run(p.parse_args()))
