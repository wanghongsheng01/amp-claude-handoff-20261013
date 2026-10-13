#!/usr/bin/env python3
"""Unexecuted candidate: receipt-bound CE B/C/D source diagnostics. Stdlib import only."""
from __future__ import annotations
import time
EARLIEST_PYTHON_CLOCK=time.perf_counter()
import argparse
import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import sys
import traceback

ROOT=Path('/Users/loctek/Documents/Codex/2026-10-11/task-4/amp_h5')
CONTRACT=ROOT/'protocol/CE_SOURCE_REPLACEMENT_CONTRACT_v1.json'
PLAN=ROOT/'protocol/CE_SOURCE_REPLACEMENT_PLAN_v1.json'
RECEIPT=ROOT/'review/independent/CE_SOURCE_REPLACEMENT_PREFLIGHT_v1.json'
OUTPUT=ROOT/'runs/ce_source_replacement_v1'
SUPERVISOR_OUTPUT=ROOT/'runs/ce_source_replacement_execution_v1'
SEEDS=ROOT/'protocol/CE_SOURCE_REPLACEMENT_SAMPLING_SEEDS_v1.json'
SLOTS=ROOT/'protocol/CE_SOURCE_REPLACEMENT_REQUEST_SLOTS_v1.json'
SCALAR_STATUS=ROOT/'runs/single_scalar_legacy_execution_v2/SUPERVISOR_STATUS.json'
SCALAR_REVIEW=ROOT/'review/independent/SCALAR_V2_POSTFLIGHT_v1.json'
FROZEN=ROOT/'implementation/vae_pipeline_formal_v2.py'
TRAIN=ROOT/'runs/formal_v2/categorical_ce'
GEN=ROOT/'runs/ce_checkpoint_recovery_v1'
SCOPE='AMP_H5_CE_SOURCE_REPLACEMENT_V1'
JOB='ce_source_replacement_v1'
TERMINALS=('FREE','AMD')
METHODS=('EMPIRICAL_POSTERIOR_B','COORDINATE_PERMUTATION_C','REFERENCE_G_PRIOR_D')
REUSED='CE_DDPM_REUSED'
FIELDS=['sample_id','method','paired_slot_id','terminal','terminal_sample_id','class','seed',
        'selection_seed','permutation_seed','condition_scope','ranking_eligible','reference_train_row_id',
        'sequence','length','status','error']

class DiagnosticInterrupt(BaseException): pass

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()

def read(path): return json.loads(Path(path).read_text())

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
        try:
            p=Path(b['path']);canonical(p,p)
            if sha(p)!=b['sha256']:raise ValueError('SHA256 differs')
        except Exception as e:errors.append({'path':b['path'],'error':repr(e)})
    return errors

def allocated_rows():
    result=[]
    conditions=('empirical_train_posterior_reference','permuted_train_posterior_decoder_z','reference_g_extra_residue_and_length')
    for method,condition in zip(METHODS,conditions):
        for label,terminal in enumerate(TERMINALS):
            for i in range(16):
                result.append({'sample_id':f'{method}_{terminal}_{i:03d}','method':method,
                    'paired_slot_id':f'CE_SOURCE_{terminal}_{i:03d}','terminal':terminal,'terminal_sample_id':i,
                    'class':label,'seed':422+label,'selection_seed':523+label,
                    'permutation_seed':625+label if method==METHODS[1] else '',
                    'condition_scope':condition,'ranking_eligible':'false','reference_train_row_id':'',
                    'sequence':'','length':0,'status':'not_executed','error':''})
    return result

def csv_write(path,rows,fields):
    tmp=path.with_suffix(path.suffix+'.tmp')
    with tmp.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(rows)
    tmp.replace(path)

def write_rows(out,rows):csv_write(out/'generated.csv',rows,FIELDS)

def validate_request(args):
    for name,path in [('contract',CONTRACT),('plan',PLAN),('receipt',RECEIPT),('output',OUTPUT)]:canonical(getattr(args,name),path)
    if OUTPUT.exists():raise ValueError('fresh output required; no overwrite/resume/retry')
    c=read(CONTRACT);p=read(PLAN);r=read(RECEIPT)
    if c['scope_id']!=SCOPE or c['script_sha256']!=sha(__file__):raise ValueError('contract scope/script differs')
    if c['budgets_seconds']!={'total_inner':120,'outer':180}:raise ValueError('fixed budget differs')
    if c['methods']!=list(METHODS) or c['D_enabled'] is not True:raise ValueError('all B/C/D required; no optional fallback')
    if c['slots']!={'paired_slots':32,'per_terminal':16,'new_decoder_inputs':96,'decoder_batches':6,'D_inverse_batches':2,'D_roundtrip_forward_batches':2,'reused_DDPM_rows':32}:raise ValueError('fixed slots/calls differ')
    if c['epsilon_seeds']!=[422,423] or c['selector_seeds']!=[523,524] or c['permutation_seeds']!=[625,626]:raise ValueError('fixed seeds differ')
    if c['roundtrip_tolerance']!={'atol':1e-5,'rtol':1e-5}:raise ValueError('fixed tolerance differs')
    for key,path in [('output_directory',OUTPUT),('supervisor_output_directory',SUPERVISOR_OUTPUT),('frozen_runner',FROZEN),('training_root',TRAIN),('generation_root',GEN),('sampling_seeds_path',SEEDS),('request_slots_path',SLOTS)]:canonical(c[key],path)
    if read(SLOTS)['rows']!=allocated_rows():raise ValueError('prelocked96 request slots differ')
    if p.get('scope_id')!=SCOPE or p.get('root_execution_authorized') is not True or p.get('D_enabled') is not True:raise PermissionError('new all-B/C/D root execution scope required')
    if r.get('authorized') is not True or r.get('scope_id')!=SCOPE or r.get('four_arm_scientific_review_checked') is not True or r.get('scalar_terminal_review_checked') is not True:raise PermissionError('new independent exact-scope receipt required')
    for key,path in [('plan_sha256',PLAN),('contract_sha256',CONTRACT),('control_script_sha256',Path(__file__).resolve())]:
        if r.get(key)!=sha(path):raise PermissionError('receipt binding differs: '+key)
    if p.get('contract_sha256')!=sha(CONTRACT):raise ValueError('plan contract differs')
    canonical(p['workspace'],ROOT.parent);canonical(p['output_root'],SUPERVISOR_OUTPUT)
    canonical(p['supervisor_path'],ROOT/'implementation/supervise.py');canonical(p['launcher_path'],ROOT/'implementation/launch_detached_supervisor_v1.py')
    if r.get('supervisor_sha256')!=sha(p['supervisor_path']) or r.get('launcher_sha256')!=sha(p['launcher_path']):raise PermissionError('supervisor/launcher receipt differs')
    argv=[c['python_executable'],'-u','-B',str(Path(__file__).resolve()),'--contract',str(CONTRACT),'--plan',str(PLAN),'--receipt',str(RECEIPT),'--output',str(OUTPUT)]
    if p.get('jobs')!=[{'id':JOB,'argv':argv,'outer_timeout_seconds':180}]:raise ValueError('exact single job required')
    # Scientific document path and its exact decision fields are locked in the new contract, never guessed.
    sci=c['scientific_review'];sp=Path(sci['path']);canonical(sp,sp)
    if ROOT/'review' not in sp.parents or not sci['required_fields']:raise PermissionError('scientific review scope missing')
    sj=read(sp)
    if sha(sp)!=sci['sha256'] or any(sj.get(k)!=v for k,v in sci['required_fields'].items()):raise PermissionError('required scientific decision differs')
    if r.get('scientific_review_sha256')!=sci['sha256']:raise PermissionError('independent scientific review binding differs')
    bound={x['path']:x['sha256'] for x in p['bindings']}
    for path in (CONTRACT,Path(__file__).resolve(),SCALAR_STATUS,SCALAR_REVIEW,sp,SEEDS,SLOTS):
        if bound.get(str(path))!=sha(path):raise PermissionError('missing exact final-plan binding: '+str(path))
    scalar=read(SCALAR_STATUS)
    if scalar.get('state') not in ('completed','completed_with_failures','failed','interrupted') or any(j.get('state')=='running' for j in scalar.get('jobs',[])):raise PermissionError('scalar must be terminal')
    if read(SCALAR_REVIEW).get('status')!='PASS':raise PermissionError('scalar engineering independent PASS required')
    if (ROOT.parent/'H5_PROCESS.lock').read_text().strip()!=str(os.getppid()):raise PermissionError('must be child of serial locked supervisor')
    return c,p,r

def digest_tensor(t):return hashlib.sha256(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest()

def tensor_bitwise_equal(a,b):return a.dtype==b.dtype and tuple(a.shape)==tuple(b.shape) and digest_tensor(a)==digest_tensor(b)

def permute_columns(np,zb,seed):
    """Only coordinate permutation; inputs are toy arrays in synthetic verification."""
    if zb.shape!=(16,16,48) or zb.dtype!=np.float32:raise ValueError('B z shape/dtype')
    rng=np.random.Generator(np.random.PCG64(seed));before=rng.bit_generator.state
    permutations=np.empty((16,48,16),dtype=np.int64);zc=np.empty_like(zb)
    for ch in range(16):
        for pos in range(48):
            perm=rng.permutation(16);permutations[ch,pos]=perm;zc[:,ch,pos]=zb[perm,ch,pos]
    verify_permutation(np,zb,zc,permutations)
    return zc,permutations,before,rng.bit_generator.state

def verify_permutation(np,zb,zc,permutations):
    if zb.shape!=zc.shape or zb.dtype!=zc.dtype or zb.dtype!=np.float32 or permutations.shape!=(16,48,16) or permutations.dtype!=np.int64:raise ValueError('permutation shape/dtype')
    for ch in range(16):
        for pos in range(48):
            perm=permutations[ch,pos]
            if not np.array_equal(np.sort(perm),np.arange(16)):raise ValueError('permutation not bijective')
            if zb[perm,ch,pos].tobytes()!=zc[:,ch,pos].tobytes():raise ValueError('C mapping differs in raw bytes')
            if not np.array_equal(np.sort(zb[:,ch,pos].copy().view(np.uint32)),np.sort(zc[:,ch,pos].copy().view(np.uint32))):raise ValueError('marginal bit-pattern multiset differs')

def prepare_inputs(m,c,out,check):
    torch,np=m.torch,m.np
    if torch.__version__!='2.2.2' or np.__version__!='1.26.4':raise ValueError('fixed RNG versions differ')
    cfg=read(TRAIN/'config.json')
    # No dev reads; this is the exact frozen train-only row identity contract.
    records=m.read_records(cfg['train'],cfg['train_sha256'],279,cfg['train_cohort'],cfg['train_cohort_sha256'],0)
    scaler=m.FeatureScaler.__new__(m.FeatureScaler);scaler.stats=read(TRAIN/'train_feature_scaler.json')
    if scaler.stats['fit_record_ids']!=list(range(279)) or scaler.stats['fit_scope']!='train only':raise ValueError('saved train scaler identity')
    with np.load(TRAIN/'train_posterior_parameters.npz',allow_pickle=False) as a:train=a['values'].copy();labels=a['labels'].copy();row_ids=a['row_ids'].copy()
    if train.shape!=(279,2,16,48) or train.dtype!=np.float32 or row_ids.dtype!=np.int64 or labels.dtype!=np.int64 or not np.array_equal(row_ids,np.arange(279)):raise ValueError('train cache identity/shape')
    if not np.isfinite(train).all() or labels.tolist()!=[r.class_index for r in records]:raise ValueError('train cache/descriptor labels differ')
    tape={};metadata={'terminals':{},'source':'exact original RNG reconstruction, no inverse residual','g_fit_calls':0}
    global_before=torch.get_rng_state().clone()
    with torch.random.fork_rng(devices=[]),torch.no_grad():
        for label,terminal in enumerate(TERMINALS):
            check();torch.manual_seed(c['epsilon_seeds'][label]);initial_state=torch.get_rng_state().clone();stream=hashlib.sha256()
            x=torch.randn(16,2,16,48);stream.update(x.numpy().tobytes())
            for timestep in reversed(range(2,999)):
                check();noise=torch.randn_like(x);stream.update(noise.numpy().tobytes())
            pre_eps=torch.get_rng_state().clone();epsilon=torch.randn(16,16,48);post_eps=torch.get_rng_state().clone()
            with np.load(GEN/f'generated_{terminal}.npz',allow_pickle=False) as a:generated=torch.from_numpy(a['posterior_parameters'].copy());saved_z=torch.from_numpy(a['z'].copy())
            if tuple(generated.shape)!=(16,2,16,48) or tuple(saved_z.shape)!=(16,16,48) or generated.dtype!=torch.float32 or saved_z.dtype!=torch.float32:raise ValueError('saved generation shape/dtype')
            mu,logs=generated[:,0],generated[:,1];sigma=logs.exp();m.finite(epsilon,'epsilon');m.finite(sigma,'saved sigma')
            if not bool((sigma>0).all()) or not tensor_bitwise_equal(mu+epsilon*sigma,saved_z):raise ValueError('saved-z raw-byte reconstruction failed; zero forwards allowed')
            pool=row_ids[labels==label]
            if len(pool)!=(126 if label==0 else 153):raise ValueError('terminal train pool differs')
            rng=np.random.Generator(np.random.PCG64(c['selector_seeds'][label]));selector_before=rng.bit_generator.state
            local=rng.integers(0,len(pool),size=16,endpoint=False,dtype=np.int64);selected=pool[local];selector_after=rng.bit_generator.state
            posterior=torch.from_numpy(train[selected].copy());qmu,qlogs=posterior[:,0],posterior[:,1];qsigma=qlogs.exp();m.finite(qsigma,'selected sigma')
            if not bool((qsigma>0).all()):raise ValueError('selected sigma underflow')
            zb=qmu+epsilon*qsigma;m.finite(zb,'B z')
            zc_array,permutations,perm_before,perm_after=permute_columns(np,zb.numpy(),c['permutation_seeds'][label]);zc=torch.from_numpy(zc_array);m.finite(zc,'C z')
            selected_records=[records[int(j)] for j in selected]
            _,g,_,y=scaler.tensorize(selected_records)
            if tuple(g.shape)!=(16,3,48) or g.dtype!=torch.float32 or y.tolist()!=[label]*16:raise ValueError('reference g shape/dtype/terminal')
            # Independent scalar-loop formula, not inverse flow or learned condition inference.
            expected=np.zeros((16,3,48),dtype=np.float32);st=scaler.stats['residue']
            for i,record in enumerate(selected_records):
                if record.idx!=int(selected[i]):raise ValueError('reference row binding')
                for pos,values in enumerate(record.residue_descriptors):
                    for ch,value in enumerate(values):expected[i,ch,pos]=np.float32((float(value)-st['mean'][ch])/st['scale'][ch])
                if g[i,:,len(record.protein):].contiguous().numpy().tobytes()!=np.zeros((3,48-len(record.protein)),dtype=np.float32).tobytes():raise ValueError('g PAD not exact positive zero')
            if not tensor_bitwise_equal(g,torch.from_numpy(expected)):raise ValueError('reference-g independent bytes disagree')
            m.finite(g,'reference g')
            prepared=out/f'prepared_{terminal}.npz'
            np.savez_compressed(prepared,epsilon=epsilon.numpy(),B_z=zb.numpy(),C_z=zc.numpy(),g=g.numpy(),permutations=permutations,
                local_indices=local,selected_train_row_ids=selected,pool_row_ids=pool,selected_mu=qmu.numpy(),selected_log_sigma=qlogs.numpy(),selected_sigma=qsigma.numpy(),
                reference_lengths=np.asarray([len(r.protein) for r in selected_records],dtype=np.int64),initial_rng_state=initial_state.numpy(),pre_epsilon_rng_state=pre_eps.numpy(),post_epsilon_rng_state=post_eps.numpy())
            metadata['terminals'][terminal]={'saved_z_raw_bytes_equal':True,'epsilon_seed':c['epsilon_seeds'][label],'epsilon_digest':digest_tensor(epsilon),
                'preceding_gaussian_tensor_count':998,'preceding_gaussian_stream_sha256':stream.hexdigest(),'initial_rng_digest':digest_tensor(initial_state),'pre_epsilon_rng_digest':digest_tensor(pre_eps),'post_epsilon_rng_digest':digest_tensor(post_eps),
                'selector_seed':c['selector_seeds'][label],'selector_state_before':selector_before,'selector_state_after':selector_after,'selected_row_ids':selected.tolist(),'local_indices':local.tolist(),'duplicate_count':16-len(set(selected.tolist())),
                'permutation_seed':c['permutation_seeds'][label],'permutation_state_before':perm_before,'permutation_state_after':perm_after,'per_coordinate_mapping_and_bit_multiset_checks':768,
                'g_digest':digest_tensor(g),'g_independent_formula_bytes_equal':True,'g_source_row_ids':[r.idx for r in selected_records],'g_reference_lengths':[len(r.protein) for r in selected_records],
                'g_reference_sequence_sha256':[hashlib.sha256(r.protein.encode()).hexdigest() for r in selected_records],'g_padding_positive_zero':True,'scaler_sha256':sha(TRAIN/'train_feature_scaler.json'),'prepared_npz_sha256':sha(prepared)}
            tape[terminal]={'epsilon':epsilon,'g':g,'selected':selected,METHODS[0]:zb,METHODS[1]:zc}
    if not tensor_bitwise_equal(torch.get_rng_state(),global_before):raise ValueError('global RNG restoration failed')
    metadata.update(global_rng_restored=True,all_saved_z_raw_bytes_equal=True,sealed_before_any_model_construction_or_forward=True,D_z_not_yet_computed=True,
                    train_descriptor_path=cfg['train'],train_descriptor_sha256=cfg['train_sha256'],train_cohort_path=cfg['train_cohort'],train_cohort_sha256=cfg['train_cohort_sha256'],scaler_sha256=sha(TRAIN/'train_feature_scaler.json'))
    dump(out/'BASE_INPUTS_SEAL.json',metadata)
    return tape,metadata

def reused_rows(out):
    src=GEN/'generated.csv';records=list(csv.DictReader(src.open(newline='')))
    if len(records)!=32:raise ValueError('existing DDPM request count differs')
    byid={r['sample_id']:r for r in records};result=[]
    for label,terminal in enumerate(TERMINALS):
        for i in range(16):
            original=f'CE21_DDPM_{terminal}_{i:03d}'
            if original not in byid:raise ValueError('existing DDPM request identity missing')
            old=byid[original]
            if old['method']!='CE21_DDPM' or old['terminal']!=terminal or int(old['seed'])!=422+label:raise ValueError('existing DDPM row metadata differs')
            result.append({'sample_id':f'{REUSED}_{terminal}_{i:03d}','method':REUSED,'terminal':terminal,'terminal_sample_id':i,
                'seed':422+label,'sequence':old['sequence'],'status':old['status'],'source_method':old['method'],'source_sample_id':original,
                'source_csv':str(src),'source_csv_sha256':sha(src),'new_model_calls':0,'condition_scope':'original_terminal_only','ranking_eligible':'false'})
    csv_write(out/'reused_ddpm.csv',result,list(result[0]))
    dump(out/'REUSED_DDPM_SLOTS.json',{'rows':[{k:v for k,v in r.items() if k not in ('sequence','status')} for r in result],'old_rows_referenced':32,'new_generation_calls':0,'source_csv_sha256':sha(src)})

def interruption(signum,frame):raise DiagnosticInterrupt('signal '+str(signum)+'; no retry')

def execute(args):
    started=EARLIEST_PYTHON_CLOCK;c,p,r=validate_request(args)
    out=OUTPUT;out.mkdir(parents=True,exist_ok=False);rows=allocated_rows();write_rows(out,rows)
    dump(out/'REQUEST_SLOTS.json',{'scope_id':SCOPE,'count':96,'rows':rows,'D_enabled':True})
    state={'state':'running','scope_id':SCOPE,'attempt_id':JOB,'phase':'input_identity','posthoc_exploratory':True,'original_CE_E_L_G':'NOT_EVALUABLE_INTERRUPTION','PROMOTE':'HOLD',
        'exploratory_gates':'NOT_APPLICABLE_SOURCE_DIAGNOSTIC','new_training_updates':0,'DDPM_forward_attempts':0,'encoder_forward_attempts':0,'decoder_forward_attempts':0,
        'prior_inverse_attempts':0,'prior_roundtrip_forward_attempts':0,'planned_decoder_inputs':96,'reused_DDPM_rows':32,'test_read':False,'dev_read':False,
        'train_source_read_started':False,'train_source_read_validated':False,'plan_sha256':sha(PLAN),'contract_sha256':sha(CONTRACT),'receipt_sha256':sha(RECEIPT),
        'script_sha256':sha(__file__),'budget_seconds':c['budgets_seconds'],'inputs_unchanged':None,'loaded_model_unchanged':None,'D_enabled':True}
    dump(out/'status.json',state);dump(out/'contract.json',c)
    old_handlers={s:signal.getsignal(s) for s in (signal.SIGTERM,signal.SIGINT,signal.SIGALRM)}
    for s in old_handlers:signal.signal(s,interruption)
    signal.setitimer(signal.ITIMER_REAL,max(.001,120-(time.perf_counter()-started)))
    model=None;m=None;before=None;active=[];observations={};timings={}
    def check():
        if time.perf_counter()-started>=120:raise TimeoutError('total diagnostic cap reached; no retry')
    try:
        check();errors=verify(c['input_bindings'])+verify(p['bindings'])
        if errors:raise ValueError('input hash failure: '+json.dumps(errors))
        reused_rows(out)
        for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[key]='1'
        spec=importlib.util.spec_from_file_location('frozen_h5_source_diagnostic',FROZEN);m=importlib.util.module_from_spec(spec);sys.dont_write_bytecode=True;spec.loader.exec_module(m)
        cfg=read(TRAIN/'config.json')
        if cfg['arm']!='categorical_ce':raise ValueError('CE-only scope')
        runtime=m.init_runtime(cfg['source_root']);original=read(GEN/'runtime_and_source.json')
        for key in ('torch','numpy','python','threads','interop_threads','device','source_manifest_sha256'):
            if runtime[key]!=original[key]:raise ValueError('runtime differs: '+key)
        dump(out/'runtime_and_source.json',runtime)
        state.update(phase='prepare_base_inputs',train_source_read_started=True);dump(out/'status.json',state)
        tick=time.perf_counter();tape,metadata=prepare_inputs(m,c,out,check);timings['base_input_preparation']=time.perf_counter()-tick
        state.update(train_source_read_validated=True,base_input_seal_sha256=sha(out/'BASE_INPUTS_SEAL.json'))
        for row in rows:
            if row['method'] in (METHODS[0],METHODS[2]):row['reference_train_row_id']=int(tape[row['terminal']]['selected'][row['terminal_sample_id']])
        write_rows(out,rows);check();state['phase']='load_frozen_model';dump(out/'status.json',state)
        tick=time.perf_counter();meta=read(TRAIN/'vae_state.json')
        if not(meta['updates']==meta['requested_updates']==1000 and meta['fixed_endpoint_reached']):raise ValueError('not final1000 checkpoint')
        if meta['sha256']!=sha(TRAIN/'vae_state.pt') or meta['tensor_state_sha256']!=c['checkpoint_tensor_digest']:raise ValueError('checkpoint identity differs')
        with m.torch.random.fork_rng(devices=[]):model=m.make_model('categorical_ce',222)
        model.load_state_dict(m.torch.load(TRAIN/'vae_state.pt',map_location='cpu',weights_only=True),strict=True)
        model.eval().requires_grad_(False);before=m.state_hash(model)
        if before!=c['checkpoint_tensor_digest']:raise ValueError('loaded tensor identity differs')
        for name,tensor in model.state_dict().items():m.finite(tensor,'checkpoint '+name)
        dump(out/'model_identity_before.json',{'tensor_digest':before,'checkpoint_sha256':sha(TRAIN/'vae_state.pt')});timings['model_load']=time.perf_counter()-tick
        state['phase']='D_inverse_and_roundtrip';dump(out/'status.json',state);tick=time.perf_counter();roundtrips={}
        rng_before=m.torch.get_rng_state().clone()
        with m.torch.no_grad():
            for terminal in TERMINALS:
                check();active=[row for row in rows if row['method']==METHODS[2] and row['terminal']==terminal]
                epsilon,g=tape[terminal]['epsilon'],tape[terminal]['g'];observations={'epsilon':epsilon,'g':g}
                state['prior_inverse_attempts']+=1;dump(out/'status.json',state)
                zd=model.vae.prior_flow(epsilon,g,reverse=True);observations['D_z']=zd;m.finite(zd,'D inverse output')
                if zd.dtype!=m.torch.float32 or tuple(zd.shape)!=(16,16,48):raise ValueError('D z shape/dtype')
                m.np.savez_compressed(out/f'D_inverse_{terminal}.npz',epsilon=epsilon.numpy(),g=g.numpy(),z=zd.numpy())
                check();state['prior_roundtrip_forward_attempts']+=1;dump(out/'status.json',state)
                returned=model.vae.prior_flow(zd,g,reverse=False);observations['returned_zp']=returned;m.finite(returned,'D roundtrip output')
                if returned.dtype!=epsilon.dtype or tuple(returned.shape)!=tuple(epsilon.shape):raise ValueError('roundtrip shape/dtype')
                residual=returned-epsilon;abs_error=residual.abs();tolerance=1e-5+1e-5*epsilon.abs();normalized=abs_error/tolerance
                observations.update(roundtrip_residual=residual,normalized_error=normalized)
                m.finite(normalized,'roundtrip normalized error')
                passed=bool((abs_error<=tolerance).all())
                m.np.savez_compressed(out/f'D_roundtrip_{terminal}.npz',epsilon=epsilon.numpy(),z=zd.numpy(),returned_zp=returned.numpy(),residual=residual.numpy(),normalized_error=normalized.numpy())
                roundtrips[terminal]={'pass':passed,'atol':1e-5,'rtol':1e-5,'max_absolute_error':float(abs_error.max()),'max_normalized_error':float(normalized.max()),
                    'failed_coordinates':int((abs_error>tolerance).sum()),'per_slot_max_absolute_error':abs_error.flatten(1).max(1).values.tolist(),
                    'per_slot_rms_error':residual.square().flatten(1).mean(1).sqrt().tolist(),'per_slot_max_normalized_error':normalized.flatten(1).max(1).values.tolist()}
                dump(out/'D_ROUNDTRIP.json',roundtrips)
                if not passed:raise FloatingPointError('D roundtrip tolerance failed; no threshold relaxation')
                tape[terminal][METHODS[2]]=zd;active=[];observations={}
            if not tensor_bitwise_equal(m.torch.get_rng_state(),rng_before):raise ValueError('flow unexpectedly consumed RNG')
            timings['D_inverse_and_roundtrip']=time.perf_counter()-tick
            seal={'base_input_seal_sha256':sha(out/'BASE_INPUTS_SEAL.json'),'D_roundtrip_sha256':sha(out/'D_ROUNDTRIP.json'),'all_96_decoder_inputs':{},'sealed_before_any_decoder':True,'D_z_created_after_inverse':True}
            for method in METHODS:
                for terminal in TERMINALS:
                    value=tape[terminal][method];m.finite(value,'sealed decoder input')
                    seal['all_96_decoder_inputs'][method+'_'+terminal]={'dtype':str(value.dtype),'shape':list(value.shape),'sha256':digest_tensor(value)}
            dump(out/'DECODER_INPUTS_SEAL.json',seal);state['decoder_input_seal_sha256']=sha(out/'DECODER_INPUTS_SEAL.json')
            state['phase']='fixed_decoder_calls';dump(out/'status.json',state);tick=time.perf_counter()
            for method in METHODS:
                for terminal in TERMINALS:
                    check();active=[row for row in rows if row['method']==method and row['terminal']==terminal]
                    observations={};state['decoder_forward_attempts']+=1;dump(out/'status.json',state)
                    seqs,tokens,lengths,raw,scores=m.decode(model,tape[terminal][method],'categorical_ce',observations)
                    if len(seqs)!=16 or tuple(raw.shape)!=(16,21,48) or tuple(tokens.shape)!=(16,48) or tuple(lengths.shape)!=(16,) or tuple(scores.shape)!=(16,49):raise ValueError('decode fixed shape differs')
                    m.finite(scores,'J scores')
                    if not bool(((lengths>=0)&(lengths<=48)).all()):raise ValueError('J length')
                    m.np.savez_compressed(out/f'{method}_{terminal}.npz',z=tape[terminal][method].numpy(),raw=raw.numpy(),tokens=tokens.numpy(),lengths=lengths.numpy(),length_scores=scores.numpy())
                    for row,seq,n in zip(active,seqs,lengths.tolist()):
                        if len(seq)!=n:raise ValueError('sequence length differs')
                        row.update(sequence=seq,length=n,status=('empty_sequence' if n==0 else 'no_eos_at_limit' if n==48 else 'ok'))
                    write_rows(out,rows);active=[];observations={};check()
            timings['decoder_and_export']=time.perf_counter()-tick
            if not tensor_bitwise_equal(m.torch.get_rng_state(),rng_before):raise ValueError('decoder unexpectedly consumed RNG')
        check();state['state']='completed'
    except (Exception,DiagnosticInterrupt) as exc:
        state.update(state='failed',error=repr(exc),traceback=traceback.format_exc());print(state['traceback'],file=sys.stderr,flush=True)
        for row in active:row.update(status='failed',error=repr(exc))
        if m is not None and observations:
            try:m.np.savez_compressed(out/'failed_available_tensors.npz',**{k:v.detach().cpu().numpy() for k,v in observations.items() if m.torch.is_tensor(v)})
            except Exception as save_exc:state['failed_tensor_save_error']=repr(save_exc)
    finally:
        signal.setitimer(signal.ITIMER_REAL,0)
        for s,handler in old_handlers.items():signal.signal(s,handler)
        final_errors=[]
        if model is not None and before is not None:
            try:
                after=m.state_hash(model);state['loaded_model_unchanged']=after==before==c['checkpoint_tensor_digest']
                dump(out/'model_identity_after.json',{'before':before,'after':after,'unchanged':state['loaded_model_unchanged']})
            except Exception as exc:final_errors.append({'phase':'model_identity','error':repr(exc)})
        try:
            errors=verify(c['input_bindings'])+verify(p['bindings']);state['inputs_unchanged']=not errors;dump(out/'input_integrity_after.json',{'errors':errors})
        except Exception as exc:final_errors.append({'phase':'input_identity','error':repr(exc)})
        state['completed_inputs']=sum(row['status'] in ('ok','empty_sequence','no_eos_at_limit') for row in rows)
        state['failed_inputs']=sum(row['status']=='failed' for row in rows);state['not_executed_inputs']=sum(row['status']=='not_executed' for row in rows)
        state['process_seconds_before_final_io']=time.perf_counter()-started;state['phase_seconds']=timings;state['finalization_errors']=final_errors
        if state['completed_inputs']!=96 or state['loaded_model_unchanged'] is not True or state['inputs_unchanged'] is not True or final_errors or state['process_seconds_before_final_io']>120 or (state['decoder_forward_attempts'],state['prior_inverse_attempts'],state['prior_roundtrip_forward_attempts'])!=(6,2,2):state['state']='failed'
        write_rows(out,rows);dump(out/'status.json',state)
        dump(out/'generation_summary.json',{'state':state['state'],'requested':96,'completed':state['completed_inputs'],'failed':state['failed_inputs'],'not_executed':state['not_executed_inputs'],'reused_prior_rows':32,'new_training_updates':0,'E_L_G':'NOT_APPLICABLE_SOURCE_DIAGNOSTIC'})
        files=[{'path':str(path),'sha256':sha(path),'bytes':path.stat().st_size} for path in sorted(out.iterdir()) if path.is_file() and path.name!='manifest.json']
        dump(out/'manifest.json',{'files':files,'state':state['state'],'scope_id':SCOPE,'posthoc_exploratory_only':True,'original_CE_E_L_G':'NOT_EVALUABLE_INTERRUPTION'})
        # Separate closing ledger includes final status/manifest export cost; no claim
        # that the earlier before-final-I/O timer was the supervisor wall time.
        closed_elapsed=time.perf_counter()-started
        if closed_elapsed>120:
            state['state']='failed';state['inner_budget_exceeded_during_finalization']=True
            dump(out/'status.json',state)
            summary=read(out/'generation_summary.json');summary['state']='failed';dump(out/'generation_summary.json',summary)
            files=[{'path':str(path),'sha256':sha(path),'bytes':path.stat().st_size} for path in sorted(out.iterdir()) if path.is_file() and path.name!='manifest.json']
            dump(out/'manifest.json',{'files':files,'state':'failed','scope_id':SCOPE,'posthoc_exploratory_only':True,'original_CE_E_L_G':'NOT_EVALUABLE_INTERRUPTION'})
        dump(out/'CLOSING_LEDGER.json',{'state':state['state'],'manifest_sha256':sha(out/'manifest.json'),
            'process_seconds_after_manifest_before_closing_ledger':time.perf_counter()-started,
            'inner_seconds_cap':120,'outer_seconds_cap':180,'supervisor_wall_time_is_separate':True,
            'closing_ledger_not_self_included_in_manifest':True})
    print(json.dumps({'state':state['state'],'completed_inputs':state['completed_inputs'],'decoder_forward_attempts':state['decoder_forward_attempts'],'prior_inverse_attempts':state['prior_inverse_attempts'],'prior_roundtrip_forward_attempts':state['prior_roundtrip_forward_attempts']}))
    return 0 if state['state']=='completed' else 2

def main():
    ap=argparse.ArgumentParser()
    for name in ('contract','plan','receipt','output'):ap.add_argument('--'+name,required=True)
    args=ap.parse_args()
    try:return execute(args)
    except Exception as exc:
        print(json.dumps({'state':'admission_failed_no_model','error':repr(exc),'traceback':traceback.format_exc(),
            'requested':96,'completed':0,'failed':0,'not_executed':96,
            'request_slots_path':str(SLOTS),'request_slots_sha256':sha(SLOTS) if SLOTS.is_file() else None,
            'all_requested_slots':allocated_rows(),'model_calls':0,'D_enabled':True,
            'process_seconds_since_earliest_python_clock':time.perf_counter()-EARLIEST_PYTHON_CLOCK,
            'admission_evidence_destination':'supervisor-captured stderr; no unauthorized output creation'}),file=sys.stderr);return 2

if __name__=='__main__':raise SystemExit(main())
