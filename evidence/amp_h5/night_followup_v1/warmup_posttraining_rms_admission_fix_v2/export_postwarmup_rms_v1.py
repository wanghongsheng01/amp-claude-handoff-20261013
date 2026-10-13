#!/usr/bin/env python3
"""Conditional post-warmup R/M/S export. New q320, same saved eight epsilon banks. No work on import."""
from __future__ import annotations
import time
EARLIEST_PYTHON_CLOCK=time.perf_counter()
import argparse,csv,hashlib,importlib.util,json,os,signal,sys,traceback
from pathlib import Path
ROOT=Path('/Users/loctek/Documents/Codex/2026-10-11/task-4/amp_h5')
HERE=ROOT/'night_followup_v1/warmup_posttraining_rms_admission_fix_v2'
CONTRACT=HERE/'CONTRACT.json'
PLAN=HERE/'ROOT_PLAN.json'
RECEIPT=HERE/'INDEPENDENT_PREFLIGHT.json'
LATE=HERE/'LATE_BOUND_TRAINING.json'
SLOTS=ROOT/'protocol/CE_RMS_RECONSTRUCTION_REQUEST_SLOTS_v2.json'
OUTPUT=HERE/'run_v1'
SUPERVISOR_OUTPUT=HERE/'execution_v1'
FROZEN=ROOT/'implementation/vae_pipeline_formal_v2.py'
TRAIN=ROOT/'runs/formal_v2/categorical_ce'
BASELINE=ROOT/'runs/ce_rms_reconstruction_v2'
SOURCE_REVIEW=ROOT/'review/independent/CE_RMS_RECONSTRUCTION_POSTFLIGHT_v2.json'
SCOPE='AMP_H5_WARMUP_POSTTRAIN_RMS_V2'
SEEDS=tuple(range(1422,1430))
SPLITS=(('train',279,0),('development',41,279))
FIELDS=['sample_id','row_key','split','row_id','source_row_id','terminal','class','mode','draw_index','seed',
        'truth_sequence','target_length','prediction_sequence','length','status','prediction_finite',
        'selected_length','first_pad_index','length48_cap','prediction_tokens48','error']
REF_FIELDS=['row_key','split','row_id','source_row_id','terminal','class','truth_sequence','target_length']
class AdmissionFailure(Exception):pass
class ExportInterrupt(BaseException):pass

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def dump(p,x):
    p=Path(p);t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n');t.replace(p)
def canonical(value,expected):
    p=Path(value)
    if not p.is_absolute() or p.is_symlink() or str(p)!=str(p.resolve()) or p!=expected:raise ValueError('exact canonical path required: '+str(expected))
def verify(bindings):
    errors=[]
    for b in bindings:
        try:
            p=Path(b['path']);canonical(p,p)
            if sha(p)!=b['sha256']:raise ValueError('hash differs')
        except Exception as exc:errors.append({'path':b['path'],'error':repr(exc)})
    return errors

def slot_id(mode,seed,split,row):return f'RMS_{mode}{seed if mode=="S" else ""}_{split}_{row:03d}'
def slot_intent():
    return [{'sample_id':slot_id(mode,seed,split,i),'row_key':f'{split}_{i:03d}','split':split,'row_id':i,
        'source_row_id':offset+i,'mode':mode,'draw_index':draw,'seed':seed}
        for mode,draw,seed in [('M',-1,''),*[('S',j,seed) for j,seed in enumerate(SEEDS)]]
        for split,n,offset in SPLITS for i in range(n)]
def allocated_rows():
    return [{**r,'terminal':'','class':'','truth_sequence':'','target_length':'','prediction_sequence':'','length':0,
        'status':'not_executed','prediction_finite':'false','selected_length':'','first_pad_index':'','length48_cap':'',
        'prediction_tokens48':'[]','error':''} for r in slot_intent()]
def csv_write(path,rows,fields):
    temp=path.with_suffix(path.suffix+'.tmp')
    with temp.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
    temp.replace(path)
def write_rows(rows):csv_write(OUTPUT/'reconstructions.csv',rows,FIELDS)
def tensor_digest(t):return hashlib.sha256(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
def bits_equal(a,b):return a.dtype==b.dtype and tuple(a.shape)==tuple(b.shape) and tensor_digest(a)==tensor_digest(b)

def nested_bindings(value):
    found={}
    def walk(x):
        if isinstance(x,dict):
            if isinstance(x.get('path'),str) and isinstance(x.get('sha256'),str):
                if x['path'] in found and found[x['path']]!=x['sha256']:raise ValueError('conflicting binding')
                found[x['path']]=x['sha256']
            for v in x.values():walk(v)
        elif isinstance(x,list):
            for v in x:walk(v)
    walk(value);return found

def validate_late(late):
    if late.get('scope_id')!=SCOPE or late.get('status')!='BOUND_SUCCESSFUL_WARMUP1000':raise PermissionError('successful warmup late binding required')
    keys=('checkpoint','checkpoint_metadata','training_summary','training_status','training_supervisor','training_independent_postflight','training_execution_contract')
    for key in keys:
        ref=late[key];path=Path(ref['path']);canonical(path,path)
        if ROOT not in path.parents or any('test' in part.lower() for part in path.parts):raise ValueError('latebinding outside AMP H5 or test path')
        if sha(path)!=ref['sha256']:raise ValueError('latebinding hash differs '+key)
    metadata=read(late['checkpoint_metadata']['path']);summary=read(late['training_summary']['path'])
    status=read(late['training_status']['path']);supervisor=read(late['training_supervisor']['path'])
    audit=read(late['training_independent_postflight']['path'])
    if metadata.get('updates')!=1000 or metadata.get('fixed_endpoint_reached') is not True or metadata.get('sha256')!=late['checkpoint']['sha256']:raise ValueError('not warmup fixed1000 checkpoint')
    if summary.get('updates')!=1000 or summary.get('fixed_endpoint_reached') is not True or not 0<summary.get('elapsed_seconds',float('inf'))<=1920:raise ValueError('warmup VAE1000 budget/summary')
    if metadata.get('tensor_state_sha256')!=late.get('checkpoint_tensor_digest') or not isinstance(late.get('checkpoint_tensor_digest'),str) or len(late['checkpoint_tensor_digest'])!=64:raise ValueError('late tensor hash')
    if status.get('state')!='completed_vae_only_not_scientific_success' or status.get('vae_updates')!=1000:raise ValueError('training not completed')
    jobs=supervisor.get('jobs',[])
    if supervisor.get('state')!='completed' or len(jobs)!=1 or jobs[0].get('state')!='completed' or jobs[0].get('exit_code')!=0 or jobs[0].get('outer_timeout') or not 0<jobs[0].get('seconds',float('inf'))<=2100:raise ValueError('training supervisor actual complete wall gate')
    if audit.get('status')!='PASS' or audit.get('blocked') is not False:raise PermissionError('independent warmup postflight required')
    bound=nested_bindings(audit)
    for key in keys:
        if key=='training_independent_postflight':continue
        ref=late[key]
        if bound.get(ref['path'])!=ref['sha256']:raise PermissionError('warmup audit missing binding '+key)
    if late.get('independent_paired_trace_pass') is not True or late.get('root_confirms_single_beta_schedule_only') is not True:raise PermissionError('warmup scope/coupling gate')
    return late

def validate_request(args):
    for key,path in [('contract',CONTRACT),('plan',PLAN),('receipt',RECEIPT),('output',OUTPUT)]:canonical(getattr(args,key),path)
    if OUTPUT.exists():raise ValueError('new output required; no resume/retry')
    c,p,r=read(CONTRACT),read(PLAN),read(RECEIPT)
    if c['scope_id']!=SCOPE or c['script_sha256']!=sha(__file__):raise ValueError('contract/script differs')
    if c['seeds']!=list(SEEDS) or c['batch_size']!=32:raise ValueError('fixed banks/batching differs')
    if c['budgets_seconds']!={'inner':120,'outer':180} or c['model_calls']!={'encoder_batches':11,'decoder_batches':99,'training_updates':0,'DDPM':0,'prior_flow':0}:raise ValueError('fixed budget/calls differs')
    if c['pilot']!={'batch':'first M train rows0..31, included once in99','safety_multiplier':8,'fixed_remaining_IO_reserve_seconds':10,'criterion':'elapsed + 8*observed_first_M_batch_seconds*98 + 10 <=120; finite shapes only, no historical M parity or quality gate'}:raise ValueError('pilot policy differs')
    for key,path in [('output_directory',OUTPUT),('supervisor_output_directory',SUPERVISOR_OUTPUT),('request_slots_path',SLOTS),('frozen_runner',FROZEN),('late_bound_training_path',LATE)]:canonical(c[key],path)
    if read(SLOTS)['slots']!=slot_intent():raise ValueError('prelocked2880 slots differ')
    if p.get('scope_id')!=SCOPE or p.get('root_execution_authorized') is not True:raise PermissionError('new posttrain RMS root plan required')
    if r.get('scope_id')!=SCOPE or r.get('authorized') is not True or r.get('scoring_and_model_scope_checked') is not True:raise PermissionError('new posttrain RMS independent receipt required')
    for key,path in [('plan_sha256',PLAN),('contract_sha256',CONTRACT),('export_script_sha256',Path(__file__).resolve()),('late_bound_training_sha256',LATE)]:
        if r.get(key)!=sha(path):raise PermissionError('receipt mismatch: '+key)
    if p.get('contract_sha256')!=sha(CONTRACT):raise PermissionError('plan contract mismatch')
    canonical(p['workspace'],ROOT.parent);canonical(p['output_root'],SUPERVISOR_OUTPUT)
    canonical(p['supervisor_path'],ROOT/'implementation/supervise.py');canonical(p['launcher_path'],ROOT/'implementation/launch_detached_supervisor_v1.py')
    if r.get('supervisor_sha256')!=sha(p['supervisor_path']) or r.get('launcher_sha256')!=sha(p['launcher_path']):raise PermissionError('launcher/supervisor differs')
    argv=[c['python_executable'],'-u','-B',str(Path(__file__).resolve()),'--contract',str(CONTRACT),'--plan',str(PLAN),'--receipt',str(RECEIPT),'--output',str(OUTPUT)]
    if p.get('jobs')!=[{'id':'warmup_posttrain_rms_v1','argv':argv,'outer_timeout_seconds':180}]:raise ValueError('one exact posttrain RMS export job required')
    bound={b['path']:b['sha256'] for b in p['bindings']}
    for path in (CONTRACT,Path(__file__).resolve(),SLOTS,SOURCE_REVIEW,LATE):
        if bound.get(str(path))!=sha(path):raise PermissionError('required plan input unbound: '+str(path))
    source_review=read(SOURCE_REVIEW)
    if (sha(SOURCE_REVIEW)!='9786d0be1c9d1fa345f598c626cb6d9b284f63aa41e60b8a706bc6772d5e5666' or source_review.get('scope_id')!='AMP_H5_CE_RMS_RECONSTRUCTION_V2' or source_review.get('status')!='PASS' or ('blocked' in source_review and source_review['blocked'] is not False)):raise PermissionError('baseline RMS independent closure required')
    late=validate_late(read(LATE))
    for ref in late['bindings_for_integrity']:
        if bound.get(ref['path'])!=ref['sha256']:raise PermissionError('late input missing plan binding')
    expected_late={late[k]['path']:late[k]['sha256'] for k in ('checkpoint','checkpoint_metadata','training_summary','training_status','training_supervisor','training_independent_postflight','training_execution_contract')}
    if {ref['path']:ref['sha256'] for ref in late['bindings_for_integrity']}!=expected_late:raise ValueError('exact late integrity set required')
    if (ROOT.parent/'H5_PROCESS.lock').read_text().strip()!=str(os.getppid()):raise PermissionError('serial supervisor lock required')
    return c,p,r,late

def init_runtime_code_only(m,source_root):
    """Same runtime imports/globals as frozen bootstrap; verify code only, never open legacy datasets."""
    source_root=Path(source_root).resolve();code=source_root/'corrected_code'/'AMP_VAEflow'
    manifest_path=source_root/'runner'/'source_copy_manifest.json';manifest=read(manifest_path)
    checked=[];skipped=[]
    for entry in manifest['files']:
        relative=Path(entry['relative_path']);path=source_root/'corrected_code'/relative
        if relative.suffix not in ('.py','.yaml','.yml','.pyc'):
            skipped.append({'path':str(path),'declared_sha256_not_verified':entry['corrected_sha256'],'opened_or_hashed':False,'reason':'non-code legacy artifact excluded from new runtime verification'})
            continue
        if relative.is_absolute() or '..' in relative.parts:raise ValueError('source manifest path traversal')
        digest=sha(path)
        if digest!=entry['corrected_sha256']:raise ValueError('frozen executable code drift: '+str(path))
        checked.append({'path':str(path),'sha256':digest})
    sys.dont_write_bytecode=True;sys.path.insert(0,str(code))
    import torch
    import numpy as np
    from modules.vae import VAEModel
    from modules.diff import UNet_conditional,EMA
    from tasks.dataset_loader.dataset_utils.amp_input_contract import validate_file
    m.torch=torch;m.np=np;m.VAEModel=VAEModel;m.UNet_conditional=UNet_conditional;m.EMA=EMA;m.validate_file=validate_file
    torch.set_num_threads(1)
    try:torch.set_num_interop_threads(1)
    except RuntimeError:
        if torch.get_num_interop_threads()!=1:raise
    return {'torch':torch.__version__,'numpy':np.__version__,'python':sys.version,'threads':torch.get_num_threads(),
        'interop_threads':torch.get_num_interop_threads(),'device':'cpu','source_manifest_sha256':sha(manifest_path),
        'source_files':checked,'verified_code_count':len(checked),'skipped_noncode_count':len(skipped),'skipped_noncode':skipped,
        'verification_scope':'manifest itself plus .py/.yaml/.yml/.pyc (including readable existing bytecode); non-code contents not read or hashed; no fake sha substitution',
        'legacy_dataset_contents_read':False}

def build_reference(records):
    return [{'row_key':f'{split}_{i:03d}','split':split,'row_id':i,'source_row_id':record.idx,
             'terminal':('FREE','AMD')[record.class_index],'class':record.class_index,'truth_sequence':record.protein,'target_length':len(record.protein)}
            for split,n,offset in SPLITS for i,record in enumerate(records[split])]

def epsilon_banks(m,out,check):
    """Load byte-bound original banks; no generator construction, reseeding or draw."""
    source=read(BASELINE/'EPSILON_BANKS_SEAL.json');banks={};before=m.torch.get_rng_state().clone()
    if source['shape']!=[320,16,48] or source['dtype']!='float32' or source['row_order']!='train0..278 then development0..40' or [x['seed'] for x in source['banks']]!=list(SEEDS):raise ValueError('baseline bank identity/order')
    for entry in source['banks']:
        check();path=Path(entry['path']);canonical(path,BASELINE/f"epsilon_seed_{entry['seed']}.npz")
        if sha(path)!=entry['file_sha256']:raise ValueError('epsilon file drift')
        with m.np.load(path,allow_pickle=False) as saved:
            epsilon=m.torch.from_numpy(saved['epsilon'].copy())
            initial=m.torch.from_numpy(saved['generator_initial_state'].copy());final=m.torch.from_numpy(saved['generator_final_state'].copy())
        if tuple(epsilon.shape)!=(320,16,48) or epsilon.dtype!=m.torch.float32:raise ValueError('epsilon array shape/dtype')
        m.finite(epsilon,'reused S bank')
        if tensor_digest(epsilon)!=entry['epsilon_sha256'] or tensor_digest(initial)!=entry['initial_generator_sha256'] or tensor_digest(final)!=entry['final_generator_sha256']:raise ValueError('epsilon tensor/state seal')
        banks[entry['seed']]=epsilon
    if not bits_equal(before,m.torch.get_rng_state()):raise ValueError('bank loading changed RNG')
    dump(out/'EPSILON_BANKS_SEAL.json',{'reused_original_seal_path':str(BASELINE/'EPSILON_BANKS_SEAL.json'),'reused_original_seal_sha256':sha(BASELINE/'EPSILON_BANKS_SEAL.json'),'source_seal':source,'new_random_draws':0,'global_rng_unchanged':True,'loaded_before_model_construction_or_forward':True})
    return banks

def decode_batch(m,model,z,batch,outfile,mode,state,observations):
    """One retained formal batch; no score-based decisions."""
    state['decoder_forward_attempts']+=1;dump(OUTPUT/'status.json',state)
    seqs,tokens,lengths,raw,scores=m.decode(model,z,'categorical_ce',observations)
    if tuple(raw.shape)!=(len(batch),21,48) or tuple(tokens.shape)!=(len(batch),48) or tuple(scores.shape)!=(len(batch),49):raise ValueError('decoder output shape')
    m.finite(scores,'J scores')
    m.np.savez_compressed(outfile,z=z.numpy(),raw=raw.numpy(),tokens=tokens.numpy(),lengths=lengths.numpy(),length_scores=scores.numpy())
    for row,seq,tok,n in zip(batch,seqs,tokens.tolist(),lengths.tolist()):
        if not 0<=n<=48 or len(seq)!=n:raise ValueError('rendered length invalid')
        firstpad=tok.index(20) if 20 in tok else -1
        if firstpad!=(n if n<48 else -1) or any(x!=20 for x in tok[n:]):raise ValueError('J PAD boundary differs')
        row.update(prediction_sequence=seq,length=n,status=('empty_sequence' if n==0 else 'no_eos_at_limit' if n==48 else 'ok'),prediction_finite='true',selected_length=n,first_pad_index=firstpad,length48_cap=str(n==48).lower(),prediction_tokens48=json.dumps(tok))
    return observations

def interrupt(signum,frame):raise ExportInterrupt('signal'+str(signum)+'; no retry')

def execute(args):
    started=EARLIEST_PYTHON_CLOCK
    try:c,p,r,late=validate_request(args)
    except Exception as exc:raise AdmissionFailure(repr(exc)) from exc
    OUTPUT.mkdir(parents=True,exist_ok=False);rows=allocated_rows();write_rows(rows);dump(OUTPUT/'REQUEST_SLOTS.json',read(SLOTS));dump(OUTPUT/'contract.json',c)
    state={'state':'running','scope_id':SCOPE,'phase':'source_setup','requested_rows':2880,'reference_rows':320,'M_rows':320,'S_rows':2560,'S_seeds':list(SEEDS),
        'decoder_forward_attempts':0,'encoder_forward_attempts':0,'discarded_encoder_internal_z_rows':0,'new_training_updates':0,'DDPM_forward_attempts':0,'prior_flow_forward_attempts':0,
        'test_read':False,'train_read_started':False,'dev_read_started':False,'fit_scaler_calls':0,'historical_CE_E_L_G':'NOT_EVALUABLE_INTERRUPTION','PROMOTE':'HOLD',
        'plan_sha256':sha(PLAN),'contract_sha256':sha(CONTRACT),'receipt_sha256':sha(RECEIPT),'script_sha256':sha(__file__),'inputs_unchanged':None,'model_unchanged':None,'historical_M_parity':'NOT_APPLICABLE_CHANGED_WEIGHTS'}
    dump(OUTPUT/'status.json',state)
    handlers={s:signal.getsignal(s) for s in (signal.SIGTERM,signal.SIGINT,signal.SIGALRM)}
    for s in handlers:signal.signal(s,interrupt)
    signal.setitimer(signal.ITIMER_REAL,max(.001,120-(time.perf_counter()-started)))
    model=None;m=None;before=None;active=[];observations={};timings={};hook_handles=[]
    def check():
        if time.perf_counter()-started>=120:raise TimeoutError('inner120 exhausted; no retry')
    try:
        check();errors=verify(c['input_bindings'])+verify(p['bindings'])+verify(late['bindings_for_integrity'])
        if errors:raise ValueError('input binding failure '+json.dumps(errors))
        for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[key]='1'
        sys.dont_write_bytecode=True;spec=importlib.util.spec_from_file_location('frozen_ce_rms',FROZEN);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
        cfg=read(TRAIN/'config.json');runtime=init_runtime_code_only(m,cfg['source_root']);old_runtime=read(BASELINE/'runtime_and_source.json')
        for key in ('torch','numpy','python','threads','interop_threads','device','source_manifest_sha256'):
            if runtime[key]!=old_runtime[key]:raise ValueError('runtime differs '+key)
        dump(OUTPUT/'runtime_and_source.json',runtime)
        state['train_read_started']=True;dump(OUTPUT/'status.json',state)
        tr=m.read_records(cfg['train'],cfg['train_sha256'],279,cfg['train_cohort'],cfg['train_cohort_sha256'],0)
        state['dev_read_started']=True;dump(OUTPUT/'status.json',state)
        dv=m.read_records(cfg['dev'],cfg['dev_sha256'],41,cfg['dev_cohort'],cfg['dev_cohort_sha256'],279)
        records={'train':tr,'development':dv};reference=build_reference(records);csv_write(OUTPUT/'reference_rows.csv',reference,REF_FIELDS);refmap={r['row_key']:r for r in reference}
        if sha(OUTPUT/'reference_rows.csv')!=c['baseline_reference_rows_sha256']:raise ValueError('candidate reference bytes differ from full baseline')
        for row in rows:row.update(refmap[row['row_key']])
        write_rows(rows)
        scaler=m.FeatureScaler.__new__(m.FeatureScaler);scaler.stats=read(TRAIN/'train_feature_scaler.json')
        if scaler.stats['fit_record_ids']!=list(range(279)) or scaler.stats['fit_scope']!='train only':raise ValueError('original scaler identity')
        check();tick=time.perf_counter();banks=epsilon_banks(m,OUTPUT,check);timings['epsilon_banks']=time.perf_counter()-tick
        dump(OUTPUT/'RAW_INPUTS_SEAL.json',{'reference_rows_sha256':sha(OUTPUT/'reference_rows.csv'),'epsilon_banks_seal_sha256':sha(OUTPUT/'EPSILON_BANKS_SEAL.json'),'scaler_sha256':sha(TRAIN/'train_feature_scaler.json'),'train_descriptor_sha256':cfg['train_sha256'],'dev_descriptor_sha256':cfg['dev_sha256'],'sealed_before_any_model_construction_or_forward':True,'all320_q_not_yet_encoded':True,'old_q_cache_reused':False,'late_training_binding_sha256':sha(LATE)})
        check();state['phase']='model_load';dump(OUTPUT/'status.json',state);tick=time.perf_counter();meta=read(late['checkpoint_metadata']['path'])
        if meta['updates']!=1000 or not meta['fixed_endpoint_reached'] or meta['sha256']!=sha(late['checkpoint']['path']):raise ValueError('final warmup checkpoint identity')
        global_before=m.torch.get_rng_state().clone()
        with m.torch.random.fork_rng(devices=[]):model=m.make_model('categorical_ce',222)
        if not bits_equal(global_before,m.torch.get_rng_state()):raise ValueError('modelinit RNG not restored')
        model.load_state_dict(m.torch.load(late['checkpoint']['path'],map_location='cpu',weights_only=True),strict=True);model.eval().requires_grad_(False);before=m.state_hash(model)
        if before!=late['checkpoint_tensor_digest']:raise ValueError('loaded checkpoint tensor digest')
        dump(OUTPUT/'model_identity_before.json',{'sha256':sha(late['checkpoint']['path']),'tensor_digest':before,'modelinit_seed':222,'modelinit_global_rng_restored':True,'global_rng_sha256':tensor_digest(global_before)})
        timings['model_load']=time.perf_counter()-tick
        # Reuse exact frozen encoder routine. Hooks only record calls and discarded internal draw provenance.
        encoder_log=[]
        def before_encoder(module,args):
            check();state['encoder_forward_attempts']+=1;dump(OUTPUT/'status.json',state)
            encoder_log.append({'batch':state['encoder_forward_attempts'],'split':state['encoding_split'],'rows':int(args[0].shape[0]),'rng_before':tensor_digest(m.torch.get_rng_state())})
        def after_encoder(module,args,output):
            discarded,mu,logs=output;entry=encoder_log[-1]
            entry.update(rng_after=tensor_digest(m.torch.get_rng_state()),discarded_internal_z_shape=list(discarded.shape),discarded_internal_z_sha256=tensor_digest(discarded),mu_shape=list(mu.shape),log_sigma_shape=list(logs.shape))
            state['discarded_encoder_internal_z_rows']+=int(discarded.shape[0])
            dump(OUTPUT/'ALL320_ENCODING_WORK_RNG.json',{'work_seed':0,'frozen_routine':'posterior_parameters uses fork_rng and manual_seed0','batches':encoder_log,'internal_sample_is_not_an_S_seed':True})
        hook_handles=[model.vae.encoder.register_forward_pre_hook(before_encoder),model.vae.encoder.register_forward_hook(after_encoder)]
        state['phase']='all320_q_encode_once';dump(OUTPUT/'status.json',state);tick=time.perf_counter()
        state['encoding_split']='train';train_values,train_labels=m.posterior_parameters(model,tr,scaler,32)
        if not bits_equal(global_before,m.torch.get_rng_state()):raise ValueError('train encoder work RNG not restored')
        state['encoding_split']='development';dev_values,dev_labels=m.posterior_parameters(model,dv,scaler,32)
        for h in hook_handles:h.remove()
        hook_handles=[];timings['all320_q_encoding']=time.perf_counter()-tick
        if state['encoder_forward_attempts']!=11 or state['discarded_encoder_internal_z_rows']!=320 or train_labels.tolist()!=[r.class_index for r in tr] or dev_labels.tolist()!=[r.class_index for r in dv]:raise ValueError('all320 q exact calls/labels')
        if not bits_equal(global_before,m.torch.get_rng_state()):raise ValueError('dev encoder work RNG not restored')
        q=m.torch.cat([train_values,dev_values],dim=0);m.finite(q,'all q parameters');sigma=q[:,1].exp();m.finite(sigma,'q sigma')
        if tuple(q.shape)!=(320,2,16,48) or not bool((sigma>0).all()):raise ValueError('q shape/sigma')
        m.np.savez_compressed(OUTPUT/'posterior_parameters320.npz',values=q.numpy(),labels=m.np.asarray([r['class'] for r in reference],dtype=m.np.int64),source_row_ids=m.np.arange(320,dtype=m.np.int64))
        m.np.savez_compressed(OUTPUT/'dev_posterior_parameters.npz',values=dev_values.numpy(),labels=dev_labels.numpy(),source_row_ids=m.np.arange(279,320,dtype=m.np.int64))
        dump(OUTPUT/'POSTERIOR_INPUTS_SEAL.json',{'raw_inputs_sha256':sha(OUTPUT/'RAW_INPUTS_SEAL.json'),'posterior_parameters_sha256':sha(OUTPUT/'posterior_parameters320.npz'),'dev_q_sha256':sha(OUTPUT/'dev_posterior_parameters.npz'),'work_rng_sha256':sha(OUTPUT/'ALL320_ENCODING_WORK_RNG.json'),'global_work_rng_restored':True,'no_posterior_flow':True,'before_any_decoder':True})
        index={row['sample_id']:row for row in rows};state['phase']='fixed_M_then_S_decode';dump(OUTPUT/'status.json',state)
        decoder_global_before=m.torch.get_rng_state().clone();tick_all=time.perf_counter()
        with m.torch.no_grad():
            for mode,draw,seed in [('M',-1,''),*[('S',j,s) for j,s in enumerate(SEEDS)]]:
                for split,n,offset in SPLITS:
                    for first in range(0,n,32):
                        check();count=min(32,n-first);indices=slice(offset+first,offset+first+count)
                        active=[index[slot_id(mode,seed,split,i)] for i in range(first,first+count)]
                        z=q[indices,0] if mode=='M' else q[indices,0]+banks[seed][indices]*sigma[indices]
                        observations={'z':z};batch_start=time.perf_counter()
                        decode_batch(m,model,z,active,OUTPUT/f'{mode}{seed}_{split}_{first:03d}.npz',mode,state,observations)
                        write_rows(rows);batch_seconds=time.perf_counter()-batch_start;active=[];observations={}
                        if state['decoder_forward_attempts']==1:
                            estimate=time.perf_counter()-started+8*batch_seconds*98+10
                            pilot={'included_in_formal_M_rows':[0,31],'additional_requests':0,'batch_seconds':batch_seconds,'elapsed_seconds':time.perf_counter()-started,'remaining_batches':98,'safety_multiplier':8,'reserve_seconds':10,'estimated_total_seconds':estimate,'inner_cap':120,'continue_by_time_and_finite_only':estimate<=120,'quality_scores_computed':False,'repeated_pilot':False}
                            dump(OUTPUT/'FIRST_BATCH_THROUGHPUT_PILOT.json',pilot)
                            if estimate>120:raise TimeoutError('prelocked pilot time projection exceeds120; retain first formal batch, no retry')
            if not bits_equal(decoder_global_before,m.torch.get_rng_state()):raise ValueError('decoder unexpectedly consumed RNG')
        timings['M_S_decode_and_export']=time.perf_counter()-tick_all
        if state['decoder_forward_attempts']!=99:raise ValueError('decoder fixed count')
        check();state['state']='completed'
    except (Exception,ExportInterrupt) as exc:
        state.update(state='failed',error=repr(exc),traceback=traceback.format_exc());print(state['traceback'],file=sys.stderr,flush=True)
        for row in active:row['status']='failed';row['error']=repr(exc)
        if m is not None and observations:
            try:m.np.savez_compressed(OUTPUT/'failed_available_tensors.npz',**{k:v.detach().cpu().numpy() for k,v in observations.items() if m.torch.is_tensor(v)})
            except Exception as save_exc:state['failure_tensor_save_error']=repr(save_exc)
    finally:
        signal.setitimer(signal.ITIMER_REAL,0)
        for s,h in handlers.items():signal.signal(s,h)
        for h in hook_handles:h.remove()
        final_errors=[]
        if model is not None and before is not None:
            try:
                after=m.state_hash(model);state['model_unchanged']=after==before==late['checkpoint_tensor_digest'];dump(OUTPUT/'model_identity_after.json',{'before':before,'after':after,'unchanged':state['model_unchanged']})
            except Exception as exc:final_errors.append({'phase':'model_integrity','error':repr(exc)})
        try:
            errors=verify(c['input_bindings'])+verify(p['bindings'])+verify(late['bindings_for_integrity']);state['inputs_unchanged']=not errors;dump(OUTPUT/'input_integrity_after.json',{'errors':errors})
        except Exception as exc:final_errors.append({'phase':'input_integrity','error':repr(exc)})
        state['completed_rows']=sum(row['status'] in ('ok','empty_sequence','no_eos_at_limit') for row in rows)
        state['failed_rows']=sum(row['status']=='failed' for row in rows);state['not_executed_rows']=sum(row['status']=='not_executed' for row in rows)
        state['process_seconds_before_final_io']=time.perf_counter()-started;state['phase_seconds']=timings;state['finalization_errors']=final_errors
        if state['completed_rows']!=2880 or state['model_unchanged'] is not True or state['inputs_unchanged'] is not True or state['process_seconds_before_final_io']>120 or final_errors:state['state']='failed'
        write_rows(rows);dump(OUTPUT/'status.json',state)
        files=[{'path':str(f),'sha256':sha(f),'bytes':f.stat().st_size} for f in sorted(OUTPUT.iterdir()) if f.is_file() and f.name!='manifest.json']
        dump(OUTPUT/'manifest.json',{'files':files,'state':state['state'],'scope_id':SCOPE,'historical_CE_E_L_G':'NOT_EVALUABLE_INTERRUPTION'})
        closing_elapsed=time.perf_counter()-started
        if closing_elapsed>120:
            state.update(state='failed',inner_budget_exceeded_during_finalization=True);dump(OUTPUT/'status.json',state)
            files=[{'path':str(f),'sha256':sha(f),'bytes':f.stat().st_size} for f in sorted(OUTPUT.iterdir()) if f.is_file() and f.name!='manifest.json']
            dump(OUTPUT/'manifest.json',{'files':files,'state':'failed','scope_id':SCOPE,'historical_CE_E_L_G':'NOT_EVALUABLE_INTERRUPTION'})
            closing_elapsed=time.perf_counter()-started
        dump(OUTPUT/'CLOSING_LEDGER.json',{'state':state['state'],'manifest_sha256':sha(OUTPUT/'manifest.json'),'process_seconds_after_manifest_before_closing_ledger':closing_elapsed,'inner_seconds_cap':120,'outer_seconds_cap':180,'closing_not_self_in_manifest':True,'supervisor_wall_separate':True})
    print(json.dumps({'state':state['state'],'completed_rows':state['completed_rows'],'encoder_calls':state['encoder_forward_attempts'],'decoder_calls':state['decoder_forward_attempts']}))
    return 0 if state['state']=='completed' else 2

def main():
    ap=argparse.ArgumentParser()
    for key in ('contract','plan','receipt','output'):ap.add_argument('--'+key,required=True)
    args=ap.parse_args()
    try:return execute(args)
    except AdmissionFailure as exc:
        print(json.dumps({'state':'admission_failed_no_model','error':repr(exc),'traceback':traceback.format_exc(),'requested':2880,'completed':0,'not_executed':2880,'model_calls':0,'request_slots_path':str(SLOTS),'request_slots_sha256':sha(SLOTS) if SLOTS.is_file() else None}),file=sys.stderr);return 2
    except (Exception,ExportInterrupt) as exc:
        report={'state':'execution_or_finalization_failure_phase_unknown','error':repr(exc),'traceback':traceback.format_exc(),'requested':2880,'model_calls':None,'completed':None,'not_executed':None,'request_slots_path':str(SLOTS),'last_persisted_status_path':str(OUTPUT/'status.json'),'persisted_counts_may_be_stale':True,'persisted_completed_is_not_success':True,'no_retry':True}
        try:
            if (OUTPUT/'status.json').is_file():report.update(last_persisted_status=read(OUTPUT/'status.json'),last_persisted_status_sha256=sha(OUTPUT/'status.json'))
        except Exception as error:report['last_persisted_status_read_error']=repr(error)
        print(json.dumps(report),file=sys.stderr);return 2
if __name__=='__main__':raise SystemExit(main())
