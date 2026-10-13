#!/usr/bin/env python3
"""One authorized, hash-bound direct-z natural run. Parent owns hard kills/lock."""
import argparse
import json
import os
from pathlib import Path
import signal
import time
import traceback
import numpy as np
import torch
from contracts import HERE,ROOT,CODE_NAMES,read,dump,sha,record,code_hashes,verify,admit,verify_lock,load_inputs,load_ar_core
from mechanics import (initialize_models,normalization,noise_schedule,optimizer_and_scheduler,scheduler_seal,
    seal_randomness,TrainingRandom,GenerationRandom,random_record,tensor_digest,state_digest,training_step,
    reverse_step,inverse_normalize,to_decoder_z,finite)
from score import score_generation

CAPS={'prepare':60,'R_training':720,'Z_training':720,'R_generation':1440,'Z_generation':1440,'decode_score':120,'closure':60}

class StopResource(RuntimeError):pass

def alarm(_s,_f):raise StopResource('phase timer expired')

class Runtime:
    def __init__(self,out):
        self.out=out;self.start=time.monotonic();self.global_deadline=self.start+4560
        self.phase=None;self.phase_start=None;self.timings={};self.resource_exceeded=False;self.deadline=self.global_deadline
        self.counts={'denoiser_train_forward':0,'denoiser_generation_forward':0,'A_decoder_forward':0,
            'updates':0,'new_requests':0,'encoder_forward':0,'old_DDPM_forward':0,'C_forward':0}
        self.arm_updates={'R':0,'Z':0}
    def finish_phase(self):
        if self.phase is None:return
        elapsed=time.monotonic()-self.phase_start
        exceeded=elapsed>CAPS[self.phase] or time.monotonic()>self.global_deadline
        self.timings[self.phase]={'wall_seconds':elapsed,'cap_seconds':CAPS[self.phase],'exceeded':exceeded}
        self.resource_exceeded |= exceeded
        dump(self.out/'PHASE_TIMINGS.json',self.timings)
    def enter(self,name):
        self.finish_phase()
        if self.resource_exceeded and name!='closure':raise StopResource('completed phase overrun')
        self.phase=name;started=time.monotonic();self.phase_start=started;self.deadline=min(started+CAPS[name],self.global_deadline)
        dump(self.out/'PHASE.json',{'phase':name,'started_monotonic':started,'cap_seconds':CAPS[name],
            'deadline_monotonic':started+CAPS[name],'total_started_monotonic':self.start})
        signal.setitimer(signal.ITIMER_REAL,max(.001,self.deadline-time.monotonic()));self.save()
    def check(self):
        if time.monotonic()>=self.deadline:raise StopResource(self.phase+' budget exhausted')
    def forward(self,kind):
        self.check();self.counts[kind]+=1
        limits={'denoiser_train_forward':2000,'denoiser_generation_forward':3992,'A_decoder_forward':98}
        assert self.counts[kind]<=limits[kind]
        assert sum(self.counts[k] for k in limits)<=6090
        self.save()
    def update(self,arm):
        self.counts['updates']+=1;self.arm_updates[arm]+=1
        assert self.counts['updates']<=2000 and self.arm_updates[arm]<=1000
        self.save()
    def save(self):
        dump(self.out/'COUNTS.json',{'phase':self.phase,'counts':self.counts,'updates_by_arm':self.arm_updates,
            'elapsed_seconds':time.monotonic()-self.start})

def initial_slots():
    return [{'arm':a,'terminal':t,'slot':i,'sequence':None,'status':'not_executed'} for a in ('R','Z','C') for t in ('FREE','AMD') for i in range(16)]

def json_digest(value):
    import hashlib
    return hashlib.sha256(json.dumps(value,separators=(',',':'),sort_keys=True).encode()).hexdigest()

def prepare(c,r,out,rt,slots):
    data=load_inputs(c);rt.check()
    for row in data['C']:
        next(s for s in slots if s['arm']=='C' and s['terminal']==row['terminal'] and s['slot']==row['slot']).update(row)
    dump(out/'GENERATION_SLOTS.json',slots)
    random_seal,rngstates=seal_randomness();rt.check()
    dump(out/'RANDOM_SEAL.json',random_seal)
    np.savez(out/'RNG_STATES.npz',**{k:v.numpy() for k,v in rngstates.items()})
    scheduler_rows,scheduler_hash=scheduler_seal();dump(out/'SCHEDULER_SEQUENCE.json',scheduler_rows)
    models,identity=initialize_models()
    for arm in ('R','Z'):torch.save(models[arm].state_dict(),out/(arm+'_initial.pt'))
    torch.save(identity,out/'INITIAL_IDENTITY.pt')
    ar=load_ar_core(c)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0);decoder=ar.AR(True)
    checkpoint=torch.load(c['inputs']['A_checkpoint'],map_location='cpu',weights_only=True)
    assert checkpoint['updates']==1000
    decoder.load_state_dict(checkpoint['model'],strict=True);decoder.eval().requires_grad_(False)
    assert ar.state_digest(decoder)==checkpoint['state_sha256']
    schedule=noise_schedule()
    deterministic={'R_initial_sha256':identity['R_sha256'],'Z_initial_sha256':identity['Z_sha256'],
        'shared_trunk_sha256':identity['common_sha256'],'random_seal_sha256':json_digest(random_seal),
        'initial_rng_sha256':{k:tensor_digest(v) for k,v in identity.items() if torch.is_tensor(v)},
        'scheduler_sequence_sha256':scheduler_hash,'noise_schedule_sha256':[tensor_digest(t) for t in schedule],
        'normalization':data['normalization'],'torch_version':torch.__version__,
        'A_state_sha256':ar.state_digest(decoder),'uniforms_sha256':ar.digest(data['uniforms']),
        'posterior279_sha256':tensor_digest(data['q'])}
    dump(out/'DETERMINISTIC_PREPARATION.json',deterministic)
    assert deterministic==r['deterministic_preparation'],'independent pre-forward seal differs'
    dump(out/'PRE_FORWARD_SEAL.json',{'deterministic':deterministic,'bindings':c['bindings'],
        'random_seal_file':record(out/'RANDOM_SEAL.json'),'models':{a:record(out/(a+'_initial.pt')) for a in ('R','Z')},
        'before_any_natural_forward':True})
    rt.check()
    return data,models,decoder,ar,schedule,random_seal,rngstates,scheduler_rows

def train(arm,model,data,schedule,sealed,rngstates,scheduler_rows,out,rt):
    folder=out/arm;folder.mkdir();started=time.monotonic();completed=0;failure=None
    opt,sch=optimizer_and_scheduler(model);stream=TrainingRandom(arm)
    for k,s in stream.states().items():assert torch.equal(s,rngstates[f'{arm}_train_initial_{k}'])
    observations={};monitors=[]
    with (folder/'training_steps.jsonl').open('x') as log:
        try:
            for step in range(1000):
                rt.check();t0=time.monotonic();random=stream.next();observations=random
                rec=random_record(random);assert rec==sealed['training'][arm][step],'training RNG replay differs'
                stats=training_step(model,opt,sch,data['q'],data['labels'],arm,data['normalization'],random,schedule,
                    before_forward=lambda:rt.forward('denoiser_train_forward'),after_update=lambda:rt.update(arm))
                completed=step+1
                assert stats['lr_used']==scheduler_rows[step]['lr'] and stats['beta1_used']==scheduler_rows[step]['beta1']
                item={'step':completed,**rec,**stats,'seconds':time.monotonic()-t0,
                    'elapsed_stage_seconds':time.monotonic()-started,'rng_after_sha256':{k:tensor_digest(s) for k,s in stream.states().items()}}
                log.write(json.dumps(item,allow_nan=False)+'\n');log.flush()
                if completed%100==0:
                    monitors.append({'step':completed,'current_batch_MSE':stats['mse'],'cumulative_objects':sum(len(x['ids']) for x in sealed['training'][arm][:completed])})
                    dump(folder/'MONITORS.json',monitors)
            for k,s in stream.states().items():assert torch.equal(s,rngstates[f'{arm}_train_final_{k}'])
            rt.check()
        except BaseException as exc:
            failure={'type':type(exc).__name__,'message':str(exc),'traceback':traceback.format_exc()}
            np.savez(folder/'training_failure_observations.npz',**{k:v.numpy() for k,v in observations.items() if torch.is_tensor(v)})
        finally:
            # Parent enforces phase720 even if checkpoint I/O outlasts one-shot SIGALRM.
            torch.save({'model':model.state_dict(),'optimizer':opt.state_dict(),'scheduler':sch.state_dict(),
                'updates':rt.arm_updates[arm],'logged_complete_steps':completed,'rng_states':stream.states(),
                'state_sha256':state_digest(model),'ema_state_sha256':state_digest(model),
                'ema_semantics':'exact current model copy below2000; alias, no extra EMA gain'},folder/'final.pt')
            status='COMPLETE' if completed==1000 and failure is None and time.monotonic()<rt.deadline else 'FAILED'
            result={'status':status,'completed_updates':rt.arm_updates[arm],'logged_complete_steps':completed,
                'failure':failure,'elapsed_seconds':time.monotonic()-started,'parameters':sum(p.numel() for p in model.parameters()),
                'final_state_sha256':state_digest(model),'checkpoint_selection':'final1000_only','no_retry':True}
            dump(folder/'TRAIN_RESULT.json',result);rt.save()
    if status!='COMPLETE':raise RuntimeError('training failed '+arm+': '+str(failure))
    return result

def generate(arm,model,stats,schedule,sealed,rngstates,out,rt,slots):
    folder=out/arm;model.eval();zs=[];rt.counts['new_requests']+=32;rt.save()
    for terminal,label in [('FREE',0),('AMD',1)]:
        selected=[s for s in slots if s['arm']==arm and s['terminal']==terminal]
        for s in selected:s['status']='latent_requested'
        dump(out/'GENERATION_SLOTS.json',slots)
        stream=GenerationRandom(terminal);observed={};step=None
        assert torch.equal(stream.state(),rngstates[terminal+'_generation_initial'])
        try:
            full=stream.noise();assert tensor_digest(full)==sealed['generation'][terminal]['initial_noise_sha256']
            x=full if arm=='R' else full[:,:1];observed={'initial_u':x.clone(),'u':x}
            labels=torch.full((16,),label,dtype=torch.long)
            with (folder/('reverse_'+terminal+'.jsonl')).open('x') as log:
                for step in range(998,0,-1):
                    rt.check();started=time.monotonic()
                    if step>1:
                        full=stream.noise();h=tensor_digest(full)
                        assert h==sealed['generation'][terminal]['reverse_noise_sha256'][str(step)]
                        noise=full if arm=='R' else full[:,:1]
                    else:h=None;noise=torch.zeros_like(x)
                    observed.update(u=x,noise=noise)
                    x=reverse_step(model,x,step,labels,noise,schedule,before_forward=lambda:rt.forward('denoiser_generation_forward'),observations=observed)
                    observed['u']=x
                    log.write(json.dumps({'t':step,'full_noise_sha256':h,'u_sha256':tensor_digest(x),
                        'finite':True,'seconds':time.monotonic()-started})+'\n');log.flush()
            assert torch.equal(stream.state(),rngstates[terminal+'_generation_final'])
            target=inverse_normalize(x,arm,stats);observed['target']=target
            if arm=='R':observed['sigma']=target[:,1].exp()
            if arm=='R':
                g=torch.Generator(device='cpu').manual_seed(8422 if terminal=='FREE' else 8423)
                assert torch.equal(g.get_state(),rngstates[terminal+'_reparam_initial'])
                epsilon=torch.randn((16,16,48),generator=g,dtype=torch.float32);observed['reparameterization_epsilon']=epsilon
                assert tensor_digest(epsilon)==sealed['reparameterization'][terminal]
                assert torch.equal(g.get_state(),rngstates[terminal+'_reparam_final'])
            else:epsilon=None
            z=to_decoder_z(target,arm,epsilon);observed['decoder_z']=z
            np.savez(folder/('generated_latent_'+terminal+'.npz'),**{k:v.numpy() for k,v in observed.items()})
            zs.append(z)
            for s in selected:s['status']='latent_complete_pending_decode'
            dump(out/'GENERATION_SLOTS.json',slots);rt.check()
        except BaseException as exc:
            for s in selected:s['status']='latent_failed';s['failure']=str(exc)
            np.savez(folder/('generation_failure_'+terminal+'.npz'),**{k:v.numpy() for k,v in observed.items()})
            dump(folder/('generation_failure_'+terminal+'.json'),{'t':step,'error_type':type(exc).__name__,'error':str(exc),'traceback':traceback.format_exc()})
            dump(out/'GENERATION_SLOTS.json',slots);raise
    return torch.cat(zs)

def decode_score(decoder,ar,z_by_arm,data,slots,out,rt):
    labels=torch.tensor([0]*16+[1]*16,dtype=torch.long)
    for arm in ('R','Z'):
        result=ar.decode(decoder,labels,z=z_by_arm[arm],uniforms=data['uniforms'],before_forward=lambda:rt.forward('A_decoder_forward'))
        np.savez(out/(arm+'_decoder_evidence.npz'),**{k:v.numpy() for k,v in result.items() if torch.is_tensor(v)})
        for i,(seq,status) in enumerate(zip(result['sequences'],result['status'])):
            s=next(s for s in slots if s['arm']==arm and s['terminal']==('FREE' if i<16 else 'AMD') and s['slot']==i%16)
            s.update(sequence=seq,status=status,terminated=bool(result['terminated'][i]),steps=result['steps'])
        dump(out/'GENERATION_SLOTS.json',slots)
        if result['failure']:raise RuntimeError('decoder failure '+arm+': '+str(result['failure']))
    return score_generation(slots,data['train'],data['dev'],rt.check)

def run(args):
    lock=verify_lock();c,r=admit(args.config,args.receipt,args.receipt_sha256)
    out=Path(args.output).resolve();assert out.parent==HERE/'runs'
    out.mkdir(parents=True,exist_ok=False);rt=Runtime(out);signal.signal(signal.SIGALRM,alarm)
    slots=initial_slots();dump(out/'GENERATION_SLOTS.json',slots);rt.enter('prepare')
    dump(out/'ADMISSION.json',{'config':record(args.config),'receipt':record(args.receipt),'code':code_hashes(),'lock':lock})
    status='FAILED';failure=None;training={a:{'status':'not_executed'} for a in ('R','Z')};metrics=None
    try:
        torch.set_num_threads(1)
        if torch.get_num_interop_threads()!=1:torch.set_num_interop_threads(1)
        torch.use_deterministic_algorithms(True)
        assert torch.__version__==c['torch_version']
        data,models,decoder,ar,schedule,sealed,rngstates,scheduler_rows=prepare(c,r,out,rt,slots)
        for arm in ('R','Z'):
            rt.enter(arm+'_training');training[arm]=train(arm,models[arm],data,schedule,sealed,rngstates,scheduler_rows,out,rt)
            dump(out/'TRAINING.json',training)
        z_by_arm={}
        for arm in ('R','Z'):
            rt.enter(arm+'_generation');z_by_arm[arm]=generate(arm,models[arm],data['normalization'],schedule,sealed,rngstates,out,rt,slots)
        rt.enter('decode_score');metrics=decode_score(decoder,ar,z_by_arm,data,slots,out,rt)
        dump(out/'METRICS.json',metrics);rt.check();status='COMPLETE'
    except BaseException as exc:
        status='STOP_RESOURCE' if isinstance(exc,(StopResource,TimeoutError)) else 'FAILED'
        failure={'type':type(exc).__name__,'message':str(exc),'traceback':traceback.format_exc()}
        dump(out/'FAILURE.json',failure)
    finally:
        rt.enter('closure')
        if rt.resource_exceeded:status='STOP_RESOURCE'
        try:
            verify(c);assert sha(args.receipt)==args.receipt_sha256
            assert r['config_sha256']==sha(args.config) and r['code_sha256']==code_hashes()
            for arm in ('R','Z'):
                p=out/arm/'TRAIN_RESULT.json'
                if p.exists():training[arm]=read(p)
            dump(out/'GENERATION_SLOTS.json',slots);dump(out/'TRAINING.json',training);rt.save()
            dump(out/'SUMMARY.json',{'status':status,'failure':failure,'training':training,'counts':rt.counts,
                'screen_status':metrics['status'] if metrics else 'NOT_EVALUABLE','elapsed_seconds':time.monotonic()-rt.start,
                'all96_slots_retained':len(slots)==96,'test_reads':0,'configuration_count':1,'training_seed_count':1,
                'PROMOTE':'HOLD','BACKUP':'BACKUP_BLOCKED','no_retry':True,
                'scope':'Finite operational screen only. No biological efficacy or submission readiness.'})
            rt.check()
            dump(out/'MANIFEST.json',{'files':[record(p) for p in sorted(out.rglob('*')) if p.is_file() and p.name not in ('MANIFEST.json','PHASE.json','PHASE_TIMINGS.json')],
                'bindings':c['bindings'],'input_hashes_verified_at_closure':True,
                'exclusions':{'MANIFEST.json':'self hash omitted','PHASE.json':'lifecycle control finalized after manifest; outer audit binds final bytes',
                    'PHASE_TIMINGS.json':'closure timing finalized after manifest; outer audit binds final bytes'}})
            rt.check();rt.finish_phase()
            if rt.resource_exceeded:raise StopResource('closure overrun')
            dump(out/'PHASE.json',{'phase':'completed','status':status})
        finally:signal.setitimer(signal.ITIMER_REAL,0)
    return 0 if status=='COMPLETE' else 1

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--receipt',required=True)
    p.add_argument('--receipt-sha256',required=True);p.add_argument('--output',required=True)
    raise SystemExit(run(p.parse_args()))
