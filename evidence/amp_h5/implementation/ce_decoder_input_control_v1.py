#!/usr/bin/env python3
"""Candidate only: a new receipt-bound CE decoder diagnostic, never a retry.

Importing this module imports stdlib only. No execution is possible without the
future exact root plan, independent receipt, scalar terminal audit and task lock.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import sys
import time
import traceback

ROOT=Path('/Users/loctek/Documents/Codex/2026-10-11/task-4/amp_h5')
CONTRACT=ROOT/'protocol/CE_DECODER_INPUT_CONTROL_CONTRACT_v1.json'
PLAN=ROOT/'protocol/CE_DECODER_INPUT_CONTROL_PLAN_v1.json'
RECEIPT=ROOT/'review/independent/CE_DECODER_INPUT_CONTROL_PREFLIGHT_RECEIPT_v1.json'
OUTPUT=ROOT/'runs/ce_decoder_input_control_v1'
SUPERVISOR_OUTPUT=ROOT/'runs/ce_decoder_input_control_execution_v1'
SCALAR_STATUS=ROOT/'runs/single_scalar_legacy_execution_v2/SUPERVISOR_STATUS.json'
SCALAR_REVIEW=ROOT/'review/independent/SCALAR_V2_POSTFLIGHT_v1.json'
FROZEN=ROOT/'implementation/vae_pipeline_formal_v2.py'
TRAIN=ROOT/'runs/formal_v2/categorical_ce'
GEN=ROOT/'runs/ce_checkpoint_recovery_v1'
SCOPE='AMP_H5_CE_DECODER_INPUT_CONTROL_V1'
BRANCHES=('A_STANDARD_NORMAL_DECODER_SPACE','B_TRAIN_EMPIRICAL_POSTERIOR')
FIELDS=['sample_id','paired_slot_id','branch','terminal','terminal_role','terminal_sample_id',
        'reference_train_row_id','sequence','length','status','error']

class ControlInterrupt(BaseException):
    pass

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()

def read(path):
    return json.loads(Path(path).read_text())

def dump(path,value):
    path=Path(path);tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n');tmp.replace(path)

def canonical(value,expected):
    p=Path(value)
    if not p.is_absolute() or p.is_symlink() or str(p)!=str(p.resolve()) or p!=expected:
        raise ValueError('exact canonical scope path required: '+str(expected))

def verify(bindings):
    errors=[]
    for b in bindings:
        p=Path(b['path'])
        try:
            canonical(p,p)
            if sha(p)!=b['sha256']:raise ValueError('SHA256 differs')
        except Exception as e:errors.append({'path':str(p),'error':repr(e)})
    return errors

def allocated_rows():
    return [{'sample_id':f'CE_CONTROL_{branch}_{terminal}_{i:03d}',
        'paired_slot_id':f'CE_CONTROL_{terminal}_{i:03d}','branch':branch,
        'terminal':terminal,'terminal_role':'report_group_only' if branch==BRANCHES[0] else 'train_pool_selector',
        'terminal_sample_id':i,'reference_train_row_id':'','sequence':'','length':'',
        'status':'not_executed','error':''}
        for terminal in ('FREE','AMD') for branch in BRANCHES for i in range(16)]

def write_rows(out,rows):
    path=out/'controls.csv';tmp=path.with_suffix('.csv.tmp')
    with tmp.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=FIELDS);w.writeheader();w.writerows(rows)
    tmp.replace(path)

def validate_request(contract_path,plan_path,receipt_path,output):
    for value,expected in [(contract_path,CONTRACT),(plan_path,PLAN),(receipt_path,RECEIPT),(output,OUTPUT)]:canonical(value,expected)
    if OUTPUT.exists():raise ValueError('fresh output required; no overwrite/resume/retry')
    c=read(CONTRACT)
    if c['scope_id']!=SCOPE or c['script_sha256']!=sha(__file__):raise ValueError('contract scope/script differs')
    if c['budgets_seconds']!={'total_inner':120,'outer':180,'training_updates':0,'DDPM_forwards':0}:raise ValueError('budget differs')
    if c['slots']!={'paired_slots':32,'FREE':16,'AMD':16,'branches':2,'decoder_inputs':64,'decoder_batches':4}:raise ValueError('slots differ')
    if c['epsilon_seeds']!=[422,423] or c['selector_seeds']!=[523,524]:raise ValueError('fixed seeds differ')
    if c['selector_algorithm']!='numpy1.26.4.Generator(PCG64(seed)).integers(0,n,16,endpoint=False,dtype=int64)':raise ValueError('selector differs')
    for key,expected in [('output_directory',OUTPUT),('supervisor_output_directory',SUPERVISOR_OUTPUT),
                         ('frozen_runner',FROZEN),('training_root',TRAIN),('generation_root',GEN)]:canonical(c[key],expected)
    # Deliberately require the future execution authorization before any scalar file read.
    p=read(PLAN);r=read(RECEIPT)
    if p.get('scope_id')!=SCOPE or p.get('root_execution_authorized') is not True:raise PermissionError('new root execution scope required')
    if r.get('authorized') is not True or r.get('scope_id')!=SCOPE or r.get('scalar_terminal_review_checked') is not True:
        raise PermissionError('new independent exact-scope execution receipt required')
    for key,path in [('plan_sha256',PLAN),('contract_sha256',CONTRACT),('control_script_sha256',Path(__file__).resolve())]:
        if r.get(key)!=sha(path):raise PermissionError('receipt binding differs: '+key)
    if p.get('contract_sha256')!=sha(CONTRACT):raise ValueError('plan contract differs')
    canonical(p['workspace'],ROOT.parent);canonical(p['output_root'],SUPERVISOR_OUTPUT)
    canonical(p['supervisor_path'],ROOT/'implementation/supervise.py')
    canonical(p['launcher_path'],ROOT/'implementation/launch_detached_supervisor_v1.py')
    if r.get('supervisor_sha256')!=sha(p['supervisor_path']) or r.get('launcher_sha256')!=sha(p['launcher_path']):
        raise PermissionError('supervisor/launcher receipt differs')
    argv=[c['python_executable'],'-u','-B',str(Path(__file__).resolve()),'--contract',str(CONTRACT),
          '--plan',str(PLAN),'--receipt',str(RECEIPT),'--output',str(OUTPUT)]
    if p.get('jobs')!=[{'id':'ce_decoder_input_control_v1','argv':argv,'outer_timeout_seconds':180}]:raise ValueError('exact one-job invocation required')
    bound={x['path']:x['sha256'] for x in p['bindings']}
    for path in (CONTRACT,Path(__file__).resolve(),SCALAR_STATUS,SCALAR_REVIEW):
        if bound.get(str(path))!=sha(path):raise ValueError('missing exact final-plan binding: '+str(path))
    scalar=read(SCALAR_STATUS)
    if scalar.get('state') not in ('completed','failed','interrupted') or any(j.get('state')=='running' for j in scalar.get('jobs',[])):
        raise PermissionError('scalar must be terminal before the new control')
    if not read(SCALAR_REVIEW).get('status'):raise PermissionError('scalar terminal review missing')
    if (ROOT.parent/'H5_PROCESS.lock').read_text().strip()!=str(os.getppid()):
        raise PermissionError('only child of the serial locked supervisor may execute')
    errors=verify(c['input_bindings'])+verify(p['bindings'])
    if errors:raise ValueError('input binding failure: '+json.dumps(errors))
    return c,p,r

def digest_tensor(t):
    return hashlib.sha256(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest()

def prepare_inputs(m,c,out,check):
    """Future authorized execution only: no model is constructed here."""
    torch,np=m.torch,m.np
    if torch.__version__!='2.2.2' or np.__version__!='1.26.4':raise ValueError('fixed RNG implementation/version differs')
    with np.load(TRAIN/'train_posterior_parameters.npz',allow_pickle=False) as a:
        train=a['values'].copy();labels=a['labels'].copy();row_ids=a['row_ids'].copy()
    if train.shape!=(279,2,16,48) or train.dtype!=np.float32 or not np.array_equal(row_ids,np.arange(279)):
        raise ValueError('train cache exact identity/shape required')
    if set(labels.tolist())!={0,1} or not np.isfinite(train).all():raise ValueError('train cache labels/finite failure')
    tape={};metadata={'source':'reconstructed original RNG stream, not inverse residual','terminals':{},'all_saved_z_bitwise_equal':False}
    global_before=torch.get_rng_state().clone()
    with torch.random.fork_rng(devices=[]),torch.no_grad():
        for label,terminal in enumerate(('FREE','AMD')):
            check();seed=c['epsilon_seeds'][label]
            torch.manual_seed(seed);initial_state=torch.get_rng_state().clone();stream=hashlib.sha256()
            x=torch.randn(16,2,16,48);stream.update(x.numpy().tobytes())
            for t in reversed(range(2,999)):
                check();noise=torch.randn_like(x);stream.update(noise.numpy().tobytes())
            pre_eps=torch.get_rng_state().clone()
            epsilon=torch.randn(16,16,48);post_eps=torch.get_rng_state().clone()
            with np.load(GEN/f'generated_{terminal}.npz',allow_pickle=False) as a:
                generated=torch.from_numpy(a['posterior_parameters'].copy());saved_z=torch.from_numpy(a['z'].copy())
            if tuple(generated.shape)!=(16,2,16,48) or tuple(saved_z.shape)!=(16,16,48):raise ValueError('saved generated shape differs')
            mu,logs=generated[:,0],generated[:,1];sigma=logs.exp()
            m.finite(epsilon,'reconstructed epsilon');m.finite(sigma,'saved generated sigma')
            if not bool((sigma>0).all()) or not torch.equal(mu+epsilon*sigma,saved_z):
                raise ValueError('exact original epsilon provenance check FAILED; zero decoder calls allowed')
            pool=row_ids[labels==label]
            if len(pool)!=(126 if label==0 else 153):raise ValueError('terminal train pool count differs')
            selector=np.random.Generator(np.random.PCG64(c['selector_seeds'][label]));selector_before=selector.bit_generator.state
            local=selector.integers(0,len(pool),size=16,endpoint=False,dtype=np.int64)
            selected=pool[local];selector_after=selector.bit_generator.state
            posterior=torch.from_numpy(train[selected].copy());qmu,qlogs=posterior[:,0],posterior[:,1];qsigma=qlogs.exp()
            m.finite(qsigma,'selected train sigma')
            if not bool((qsigma>0).all()):raise ValueError('selected train sigma underflow')
            za=epsilon.clone();zb=qmu+epsilon*qsigma
            m.finite(za,'all A inputs');m.finite(zb,'all B inputs')
            np.savez_compressed(out/f'prepared_{terminal}.npz',epsilon=epsilon.numpy(),
                A_z=za.numpy(),B_z=zb.numpy(),local_indices=local,selected_train_row_ids=selected,
                selected_mu=qmu.numpy(),selected_log_sigma=qlogs.numpy(),selected_sigma=qsigma.numpy(),
                initial_rng_state=initial_state.numpy(),pre_epsilon_rng_state=pre_eps.numpy(),post_epsilon_rng_state=post_eps.numpy())
            metadata['terminals'][terminal]={'epsilon_seed':seed,'consumed_gaussian_tensors':998,
                'consumed_shapes':[16,2,16,48],'consumed_stream_sha256':stream.hexdigest(),
                'epsilon_shape':[16,16,48],'epsilon_sha256':digest_tensor(epsilon),
                'initial_rng_sha256':digest_tensor(initial_state),'pre_epsilon_rng_sha256':digest_tensor(pre_eps),'post_epsilon_rng_sha256':digest_tensor(post_eps),
                'saved_z_bitwise_equal':True,'selector_seed':c['selector_seeds'][label],
                'selector_algorithm':c['selector_algorithm'],'selector_state_before':selector_before,'selector_state_after':selector_after,
                'local_indices':local.tolist(),'selected_train_row_ids':selected.tolist(),'with_replacement':True,
                'duplicate_count':int(16-len(set(selected.tolist()))),'prepared_npz_sha256':sha(out/f'prepared_{terminal}.npz')}
            tape[terminal]={BRANCHES[0]:za,BRANCHES[1]:zb,'selected':selected}
    if not torch.equal(torch.get_rng_state(),global_before):raise ValueError('RNG preparation failed to restore global state')
    metadata['global_rng_restored']=True;metadata['all_saved_z_bitwise_equal']=True
    metadata['sealed_before_any_model_construction_or_forward']=True
    dump(out/'PREPARED_INPUTS_SEAL.json',metadata)
    return tape,metadata

def interruption(signum,frame):
    raise ControlInterrupt('control interrupted by signal '+str(signum)+'; no retry')

def execute(args):
    started=time.perf_counter();c,p,r=validate_request(args.contract,args.plan,args.receipt,args.output)
    out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    rows=allocated_rows();write_rows(out,rows)
    state={'state':'running','scope_id':SCOPE,'attempt_id':'ce_decoder_input_control_v1','posthoc_exploratory':True,
        'original_CE_E_L_G':'NOT_EVALUABLE_INTERRUPTION','PROMOTE':'HOLD','new_training_updates':0,
        'DDPM_forwards':0,'encoder_forwards':0,'flow_forwards':0,'decoder_forward_attempts':0,
        'planned_decoder_inputs':64,'paired_slots':32,'test_read':False,'dev_read':False,
        'source_cohort_read':False,'live_scalar_read':False,'phase':'setup','plan_sha256':sha(PLAN),
        'contract_sha256':sha(CONTRACT),'receipt_sha256':sha(RECEIPT),'script_sha256':sha(__file__),
        'budget_seconds':c['budgets_seconds'],'inputs_unchanged':None,'loaded_model_unchanged':None}
    dump(out/'status.json',state);dump(out/'contract.json',c)
    old_handlers={s:signal.getsignal(s) for s in (signal.SIGTERM,signal.SIGINT,signal.SIGALRM)}
    for s in old_handlers:signal.signal(s,interruption)
    signal.setitimer(signal.ITIMER_REAL,max(.001,120-(time.perf_counter()-started)))
    model=None;m=None;before=None;active=[];observations={}
    def check():
        if time.perf_counter()-started>=120:raise TimeoutError('total control cap reached; no retry')
    try:
        check()
        for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[key]='1'
        spec=importlib.util.spec_from_file_location('frozen_h5_ce_control',FROZEN);m=importlib.util.module_from_spec(spec)
        sys.dont_write_bytecode=True;spec.loader.exec_module(m)
        config=read(TRAIN/'config.json')
        if config['arm']!='categorical_ce':raise ValueError('CE-only scope')
        runtime=m.init_runtime(config['source_root']);original=read(GEN/'runtime_and_source.json')
        for key in ('torch','numpy','python','threads','interop_threads','device','source_manifest_sha256'):
            if runtime[key]!=original[key]:raise ValueError('runtime differs: '+key)
        dump(out/'runtime_and_source.json',runtime)
        state['phase']='prepare_and_seal_all64_inputs';dump(out/'status.json',state)
        tick=time.perf_counter();tape,metadata=prepare_inputs(m,c,out,check)
        state['preparation_seconds']=time.perf_counter()-tick
        state['prepared_inputs_seal_sha256']=sha(out/'PREPARED_INPUTS_SEAL.json')
        # Nothing below is reachable until both original epsilon streams and all64 inputs pass.
        check();meta=read(TRAIN/'vae_state.json')
        if not(meta['updates']==meta['requested_updates']==1000 and meta['fixed_endpoint_reached']):raise ValueError('not final1000 checkpoint')
        if meta['sha256']!=sha(TRAIN/'vae_state.pt') or meta['tensor_state_sha256']!=c['checkpoint_tensor_digest']:raise ValueError('checkpoint identity differs')
        with m.torch.random.fork_rng(devices=[]):model=m.make_model('categorical_ce',222)
        model.load_state_dict(m.torch.load(TRAIN/'vae_state.pt',map_location='cpu',weights_only=True),strict=True)
        model.eval().requires_grad_(False);before=m.state_hash(model)
        if before!=c['checkpoint_tensor_digest']:raise ValueError('loaded checkpoint tensor digest differs')
        for name,tensor in model.state_dict().items():m.finite(tensor,'checkpoint '+name)
        dump(out/'model_identity_before.json',{'tensor_digest':before,'checkpoint_sha256':sha(TRAIN/'vae_state.pt')})
        state['phase']='fixed_decoder_calls';dump(out/'status.json',state)
        tick=time.perf_counter()
        with m.torch.no_grad():
            for terminal in ('FREE','AMD'):
                for branch in BRANCHES:
                    check();active=[row for row in rows if row['terminal']==terminal and row['branch']==branch]
                    for i,row in enumerate(active):
                        if branch==BRANCHES[1]:row['reference_train_row_id']=int(tape[terminal]['selected'][i])
                    observations={};state['decoder_forward_attempts']+=1
                    seqs,tokens,lengths,raw,scores=m.decode(model,tape[terminal][branch],'categorical_ce',observations)
                    if len(seqs)!=16 or tuple(raw.shape)!=(16,21,48) or tuple(tokens.shape)!=(16,48) or tuple(lengths.shape)!=(16,) or tuple(scores.shape)!=(16,49):
                        raise ValueError('fixed decoder output shape differs')
                    m.finite(scores,'joint prefix length scores')
                    if not bool(((lengths>=0)&(lengths<=48)).all()):raise ValueError('invalid length output')
                    check()
                    m.np.savez_compressed(out/f'{terminal}_{branch}.npz',z=tape[terminal][branch].numpy(),
                        raw=raw.numpy(),tokens=tokens.numpy(),lengths=lengths.numpy(),length_scores=scores.numpy())
                    for row,seq,n in zip(active,seqs,lengths.tolist()):
                        row.update(sequence=seq,length=n,status=('empty_sequence' if n==0 else 'no_eos_at_limit' if n==48 else 'ok'))
                    write_rows(out,rows);active=[];observations={}
        state['decoder_and_export_seconds']=time.perf_counter()-tick;check();state['state']='completed'
    except (Exception,ControlInterrupt) as exc:
        state.update(state='failed',error=repr(exc),traceback=traceback.format_exc())
        print(state['traceback'],file=sys.stderr,flush=True)
        for row in active:row.update(status='failed',error=repr(exc))
        if m is not None and observations:
            try:m.np.savez_compressed(out/'failed_available_tensors.npz',**{k:v.detach().cpu().numpy() for k,v in observations.items() if m.torch.is_tensor(v)})
            except Exception as save_exc:state['failed_tensor_save_error']=repr(save_exc)
    finally:
        signal.setitimer(signal.ITIMER_REAL,0)
        for s,handler in old_handlers.items():signal.signal(s,handler)
        errors=[]
        if model is not None and before is not None:
            try:
                after=m.state_hash(model);state['loaded_model_unchanged']=after==before==c['checkpoint_tensor_digest']
                dump(out/'model_identity_after.json',{'before':before,'after':after,'unchanged':state['loaded_model_unchanged']})
            except Exception as exc:errors.append({'phase':'model_identity','error':repr(exc)})
        try:
            input_errors=verify(c['input_bindings'])+verify(p['bindings']);state['inputs_unchanged']=not input_errors
            dump(out/'input_integrity_after.json',{'errors':input_errors})
        except Exception as exc:errors.append({'phase':'input_identity','error':repr(exc)})
        state['completed_inputs']=sum(row['status'] in ('ok','empty_sequence','no_eos_at_limit') for row in rows)
        state['failed_inputs']=sum(row['status']=='failed' for row in rows)
        state['not_executed_inputs']=sum(row['status']=='not_executed' for row in rows)
        state['process_seconds_before_final_io']=time.perf_counter()-started
        if state['completed_inputs']!=64 or state['loaded_model_unchanged'] is not True or state['inputs_unchanged'] is not True or errors or state['process_seconds_before_final_io']>120:
            state['state']='failed'
        state['finalization_errors']=errors;write_rows(out,rows);dump(out/'status.json',state)
        files=[{'path':str(path),'sha256':sha(path),'bytes':path.stat().st_size} for path in sorted(out.iterdir()) if path.is_file() and path.name!='manifest.json']
        dump(out/'manifest.json',{'files':files,'state':state['state'],'posthoc_exploratory_only':True,'original_CE_E_L_G':'NOT_EVALUABLE_INTERRUPTION'})
    print(json.dumps({'state':state['state'],'completed_inputs':state['completed_inputs'],'decoder_forward_attempts':state['decoder_forward_attempts']}))
    return 0 if state['state']=='completed' else 2

def main():
    ap=argparse.ArgumentParser()
    for name in ('contract','plan','receipt','output'):ap.add_argument('--'+name,required=True)
    args=ap.parse_args()
    try:return execute(args)
    except Exception as exc:
        print(json.dumps({'state':'preflight_failed_no_decoder','error':repr(exc),'traceback':traceback.format_exc()}),file=sys.stderr)
        return 2

if __name__=='__main__':raise SystemExit(main())
