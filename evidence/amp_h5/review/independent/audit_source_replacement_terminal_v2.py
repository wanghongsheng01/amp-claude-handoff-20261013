#!/usr/bin/env python3
"""Independent saved-artifact audit. No model construction/import/forward."""
import os
for _k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_k]='1'
import time
START=time.perf_counter()
import csv, hashlib, json, re, traceback
from decimal import Decimal
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import torch
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
R=Path('/Users/loctek/Documents/Codex/2026-10-11/task-4/amp_h5')
O=R/'review/independent'
RUN=R/'runs/ce_source_replacement_v2'
TRAIN=R/'runs/formal_v2/categorical_ce'
OLD=R/'runs/ce_checkpoint_recovery_v1'
PLAN=R/'protocol/CE_SOURCE_REPLACEMENT_PLAN_v2.json'
CONTRACT=R/'protocol/CE_SOURCE_REPLACEMENT_CONTRACT_v2.json'
PREFLIGHT=O/'CE_SOURCE_REPLACEMENT_PREFLIGHT_v2.json'
REPORT=O/'CE_SOURCE_REPLACEMENT_TERMINAL_AUDIT_v2.json'
METHODS=('EMPIRICAL_POSTERIOR_B','COORDINATE_PERMUTATION_C','REFERENCE_G_PRIOR_D')
TERMS=('FREE','AMD')
AA='ARNDCEQGHILKMFPSTWYV'
checks=[];facts={};tracked={}
def sha(p):
    p=Path(p);h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def track(p):
    p=Path(p);s=sha(p);tracked.setdefault(str(p),s);return p
def read(p):return json.loads(track(p).read_text())
def rows(p):
    with track(p).open(newline='') as f:return list(csv.DictReader(f))
def arrays(p):
    with np.load(track(p),allow_pickle=False) as f:return {k:f[k].copy() for k in f.files}
def ck(name,ok,detail=None):
    checks.append({'check':name,'pass':bool(ok),'detail':detail})
    if not ok:raise AssertionError(name+': '+str(detail))
def bits(a,b):
    if torch.is_tensor(a):a=a.detach().contiguous().numpy()
    if torch.is_tensor(b):b=b.detach().contiguous().numpy()
    return a.dtype==b.dtype and a.shape==b.shape and a.tobytes()==b.tobytes()
def digest(a):
    if torch.is_tensor(a):a=a.detach().contiguous().numpy()
    return hashlib.sha256(a.tobytes()).hexdigest()
def bindings(items):
    for b in items:
        p=track(b['path'])
        if sha(p)!=b['sha256'] or ('bytes' in b and p.stat().st_size!=b['bytes']):return False
    return True
def shape(a,s,dtype):return a.shape==s and a.dtype==dtype and bool(np.isfinite(a).all())
def numerical_summary(a):
    return {'dtype':str(a.dtype),'shape':list(a.shape),'min':float(a.min()),'max':float(a.max()),'rms':float(np.sqrt(np.mean(a.astype(np.float64)**2))),'sha256_raw_bytes':digest(a)}
def main():
    ck('fixed_arithmetic_runtime',torch.__version__=='2.2.2' and np.__version__=='1.26.4')
    c,p,pre=read(CONTRACT),read(PLAN),read(PREFLIGHT)
    ck('exact_locked_plan',sha(PLAN)=='91e64d4ca473c1a56b2c66b285f0f5b8fd0ccba3884fe15ba465e574108bb46f')
    ck('plan_and_contract_bindings',bindings(p['bindings']) and bindings(c['input_bindings']),{'plan':len(p['bindings']),'contract':len(c['input_bindings'])})
    ck('receipt_exact_plan_contract_script',pre['status']=='PASS' and pre['authorized'] and pre['plan_sha256']==sha(PLAN) and pre['contract_sha256']==sha(CONTRACT) and pre['control_script_sha256']==sha(R/'implementation/ce_source_replacement_v2.py'))
    state=read(RUN/'status.json');summary=read(RUN/'generation_summary.json');manifest=read(RUN/'manifest.json');closing=read(RUN/'CLOSING_LEDGER.json')
    sup=read(R/'runs/ce_source_replacement_execution_v2/SUPERVISOR_STATUS.json');launch=read(R/'runs/ce_source_replacement_launch_v2/LAUNCH.json')
    ck('five_final_states_and_single_exit0',all(x['state']=='completed' for x in (state,summary,manifest,closing,sup)) and len(sup['jobs'])==1 and sup['jobs'][0]['state']=='completed' and sup['jobs'][0]['exit_code']==0)
    ck('exact_executed_argv_and_admission',sup['jobs'][0]['argv']==p['jobs'][0]['argv'] and sup['plan_sha256']==sha(PLAN)==launch['plan_sha256']==state['plan_sha256'] and launch['receipt_sha256']==sha(PREFLIGHT)==state['receipt_sha256'] and state['contract_sha256']==sha(CONTRACT) and state['script_sha256']==c['script_sha256'] and launch['supervisor_pid']==sup['pid'])
    ck('manifest_members_hash_bytes_and_unique',bindings(manifest['files']) and len({x['path'] for x in manifest['files']})==len(manifest['files']))
    members={Path(x['path']) for x in manifest['files']}
    ck('manifest_complete_exact_directory_and_separate_closing',members=={x for x in RUN.iterdir() if x.is_file() and x.name not in ('manifest.json','CLOSING_LEDGER.json')} and all(x.parent==RUN for x in members) and closing['manifest_sha256']==sha(RUN/'manifest.json'))
    inner=closing['process_seconds_after_manifest_before_closing_ledger'];outer=sup['jobs'][0]['seconds'];before=state['process_seconds_before_final_io']
    ck('three_clock_order_and_locked_caps',0<before<=inner<=120 and inner<=outer<=180 and c['budgets_seconds']==state['budget_seconds']=={'total_inner':120,'outer':180} and closing['inner_seconds_cap']==120 and closing['outer_seconds_cap']==180 and p['jobs'][0]['outer_timeout_seconds']==180)
    ck('exact_call_ledger_and_no_training',tuple(state[k] for k in ('decoder_forward_attempts','prior_inverse_attempts','prior_roundtrip_forward_attempts','new_training_updates','encoder_forward_attempts','DDPM_forward_attempts'))==(6,2,2,0,0,0))
    ck('all96_completed_zero_failed_pending',tuple(state[k] for k in ('planned_decoder_inputs','completed_inputs','failed_inputs','not_executed_inputs','reused_DDPM_rows'))==(96,96,0,0,32) and tuple(summary[k] for k in ('requested','completed','failed','not_executed','reused_prior_rows'))==(96,96,0,0,32))
    ck('frozen_scope_gates_not_repaired',state['original_CE_E_L_G']=='NOT_EVALUABLE_INTERRUPTION' and state['PROMOTE']=='HOLD' and state['exploratory_gates']==summary['E_L_G']=='NOT_APPLICABLE_SOURCE_DIAGNOSTIC' and state['posthoc_exploratory'] and state['D_enabled'])
    ck('no_dev_test_read_path_and_source_declaration',state['test_read'] is False and state['dev_read'] is False and state['train_source_read_started'] and state['train_source_read_validated'])
    ck('no_finalization_error_and_original_inputs_unchanged',state['finalization_errors']==[] and state['inputs_unchanged'] and read(RUN/'input_integrity_after.json')['errors']==[] and sha(RUN/'contract.json')==sha(CONTRACT))
    runtime=read(RUN/'runtime_and_source.json')
    ck('exact_original_runtime_and_source_files',sha(RUN/'runtime_and_source.json')==sha(OLD/'runtime_and_source.json') and bindings(runtime['source_files']) and runtime['threads']==runtime['interop_threads']==1 and runtime['device']=='cpu')
    meta=read(TRAIN/'vae_state.json');mb=read(RUN/'model_identity_before.json');ma=read(RUN/'model_identity_after.json')
    ck('final1000_checkpoint_and_hash_before_after',meta['updates']==meta['requested_updates']==1000 and meta['fixed_endpoint_reached'] and meta['sha256']==c['checkpoint_sha256']==sha(track(TRAIN/'vae_state.pt'))==mb['checkpoint_sha256'] and meta['tensor_state_sha256']==c['checkpoint_tensor_digest']==mb['tensor_digest']==ma['before']==ma['after'] and ma['unchanged'] and state['loaded_model_unchanged'])
    # Data-only restricted checkpoint load; never instantiate the model.
    weights=torch.load(TRAIN/'vae_state.pt',map_location='cpu',weights_only=True);h=hashlib.sha256()
    for name,tensor in sorted(weights.items()):
        a=tensor.detach().contiguous().numpy();assert np.isfinite(a).all()
        h.update(name.encode());h.update(str(a.dtype).encode());h.update(str(a.shape).encode());h.update(a.tobytes())
    ck('independent_checkpoint_tensor_digest',h.hexdigest()==c['checkpoint_tensor_digest']);del weights
    base=read(RUN/'BASE_INPUTS_SEAL.json');seal=read(RUN/'DECODER_INPUTS_SEAL.json');rt=read(RUN/'D_ROUNDTRIP.json')
    ck('cross_seal_hashes_and_precedence_contract',state['base_input_seal_sha256']==seal['base_input_seal_sha256']==sha(RUN/'BASE_INPUTS_SEAL.json') and state['decoder_input_seal_sha256']==sha(RUN/'DECODER_INPUTS_SEAL.json') and seal['D_roundtrip_sha256']==sha(RUN/'D_ROUNDTRIP.json') and base['sealed_before_any_model_construction_or_forward'] and seal['sealed_before_any_decoder'] and base['D_z_not_yet_computed'] and seal['D_z_created_after_inverse'])
    ck('seal_file_timestamps_consistent_with_code_order',(RUN/'BASE_INPUTS_SEAL.json').stat().st_mtime_ns<=(RUN/'model_identity_before.json').stat().st_mtime_ns<=(RUN/'D_ROUNDTRIP.json').stat().st_mtime_ns<=(RUN/'DECODER_INPUTS_SEAL.json').stat().st_mtime_ns<=min((RUN/f'{m}_{t}.npz').stat().st_mtime_ns for m in METHODS for t in TERMS))
    cfg=read(TRAIN/'config.json');st=read(TRAIN/'train_feature_scaler.json');q=arrays(TRAIN/'train_posterior_parameters.npz');desc=rows(cfg['train']);cohort=rows(cfg['train_cohort'])
    ck('train279_cache_scaler_and_order',len(desc)==len(cohort)==279 and shape(q['values'],(279,2,16,48),np.float32) and bits(q['row_ids'],np.arange(279,dtype=np.int64)) and q['labels'].dtype==np.int64 and st['fit_record_ids']==list(range(279)) and st['fit_scope']=='train only')
    ck('descriptor_cohort_cache_identity_all279',all(int(d['idx'])==i and d['protein']==cohort[i]['sequence'] and int(d['class'])==int(cohort[i]['class'])==int(q['labels'][i]) and len(d['protein'])==int(cohort[i]['length']) and cohort[i]['terminal_state']==('Free' if int(d['class'])==0 else 'Amide') for i,d in enumerate(desc)))
    new=rows(RUN/'generated.csv');reuse=rows(RUN/'reused_ddpm.csv');oldrows=rows(OLD/'generated.csv');slot=read(RUN/'REQUEST_SLOTS.json');intent=read(R/'protocol/CE_SOURCE_REPLACEMENT_REQUEST_SLOTS_v2.json')
    ck('96_prelocked_ids_order_rows',len(new)==96 and len({r['sample_id'] for r in new})==96 and slot['rows']==intent['rows'] and [r['sample_id'] for r in new]==[r['sample_id'] for r in intent['rows']])
    mutable={'reference_train_row_id','sequence','length','status','error'}
    ck('96_immutable_request_metadata_and_ranking',all(all(str(row[k])==str(w[k]) for k in w if k not in mutable) and row['ranking_eligible']=='false' and row['status'] in ('ok','empty_sequence','no_eos_at_limit') and row['error']=='' for row,w in zip(new,intent['rows'])))
    oldby={r['sample_id']:r for r in oldrows}
    ck('reused32_exact_source_identity_no_new_calls',len(reuse)==32 and len({r['sample_id'] for r in reuse})==32 and all(r['sample_id']==f"CE_DDPM_REUSED_{r['terminal']}_{int(r['terminal_sample_id']):03d}" and r['method']=='CE_DDPM_REUSED' and r['source_csv']==str(OLD/'generated.csv') and r['source_csv_sha256']==sha(OLD/'generated.csv') and r['new_model_calls']=='0' and r['ranking_eligible']=='false' and all(r[k]==oldby[r['source_sample_id']][k] for k in ('terminal','seed','sequence','status')) for r in reuse))
    facts['terminals']={};global_rng=torch.get_rng_state().clone()
    for label,t in enumerate(TERMS):
        a=arrays(RUN/f'prepared_{t}.npz');b=base['terminals'][t];saved=arrays(OLD/f'generated_{t}.npz')
        ck(t+'_prepared_shapes_types_finite',all(shape(a[k],s,d) for k,s,d in [('epsilon',(16,16,48),np.float32),('B_z',(16,16,48),np.float32),('C_z',(16,16,48),np.float32),('g',(16,3,48),np.float32),('permutations',(16,48,16),np.int64),('selected_train_row_ids',(16,),np.int64),('local_indices',(16,),np.int64),('selected_mu',(16,16,48),np.float32),('selected_log_sigma',(16,16,48),np.float32),('selected_sigma',(16,16,48),np.float32)]))
        with torch.random.fork_rng(devices=[]),torch.no_grad():
            torch.manual_seed(422+label);initial=torch.get_rng_state().clone();stream=hashlib.sha256();x=torch.randn(16,2,16,48);stream.update(x.numpy().tobytes())
            for _ in range(997):stream.update(torch.randn_like(x).numpy().tobytes())
            before_eps=torch.get_rng_state().clone();eps=torch.randn(16,16,48);after_eps=torch.get_rng_state().clone()
        ck(t+'_exact_global_rng_tape998_then_epsilon',bits(eps,a['epsilon']) and bits(initial,a['initial_rng_state']) and bits(before_eps,a['pre_epsilon_rng_state']) and bits(after_eps,a['post_epsilon_rng_state']) and b['preceding_gaussian_tensor_count']==998 and b['preceding_gaussian_stream_sha256']==stream.hexdigest() and b['epsilon_seed']==422+label and b['epsilon_digest']==digest(eps) and b['initial_rng_digest']==digest(initial) and b['pre_epsilon_rng_digest']==digest(before_eps) and b['post_epsilon_rng_digest']==digest(after_eps))
        oldq=torch.from_numpy(saved['posterior_parameters']);oldsigma=oldq[:,1].exp();oldz=oldq[:,0]+eps*oldsigma
        ck(t+'_saved_CE_z_exact_dtype_shape_bytes',bits(oldz,saved['z']) and bool(torch.isfinite(oldsigma).all()) and bool((oldsigma>0).all()))
        pool=np.flatnonzero(q['labels']==label).astype(np.int64);rng=np.random.Generator(np.random.PCG64(523+label));selbefore=rng.bit_generator.state;local=rng.integers(0,len(pool),16,endpoint=False,dtype=np.int64);ids=pool[local]
        ck(t+'_selector_state_and_all_indices',len(pool)==(126,153)[label] and bits(pool,a['pool_row_ids']) and bits(local,a['local_indices']) and bits(ids,a['selected_train_row_ids']) and b['selector_seed']==523+label and b['selector_state_before']==selbefore and b['selector_state_after']==rng.bit_generator.state and b['selected_row_ids']==ids.tolist() and b['duplicate_count']==16-len(set(ids.tolist())))
        qm=torch.from_numpy(q['values'][ids,0].copy());ql=torch.from_numpy(q['values'][ids,1].copy());sigma=ql.exp();bz=qm+eps*sigma
        ck(t+'_B_same_eps_paired_posterior_positive_sigma_bytes',bits(qm,a['selected_mu']) and bits(ql,a['selected_log_sigma']) and bits(sigma,a['selected_sigma']) and bits(bz,a['B_z']) and bool(torch.isfinite(sigma).all()) and bool((sigma>0).all()))
        prng=np.random.Generator(np.random.PCG64(625+label));permbefore=prng.bit_generator.state;perms=np.empty((16,48,16),np.int64);zc=np.empty_like(a['B_z']);valid=True
        for ch in range(16):
            for pos in range(48):
                pi=prng.permutation(16);perms[ch,pos]=pi;zc[:,ch,pos]=a['B_z'][pi,ch,pos]
                valid &= bool(np.array_equal(np.sort(a['permutations'][ch,pos]),np.arange(16)))
                valid &= np.sort(a['B_z'][:,ch,pos].copy().view(np.uint32)).tobytes()==np.sort(a['C_z'][:,ch,pos].copy().view(np.uint32)).tobytes()
        ck(t+'_C768_bijections_column_bit_margins_and_seed_states',valid and bits(perms,a['permutations']) and bits(zc,a['C_z']) and b['permutation_seed']==625+label and b['permutation_state_before']==permbefore and b['permutation_state_after']==prng.bit_generator.state)
        eg=np.zeros((16,3,48),np.float32);lengths=[];sequence_hashes=[];padding=True
        for i,j in enumerate(ids.tolist()):
            d=desc[j];lengths.append(len(d['protein']));sequence_hashes.append(hashlib.sha256(d['protein'].encode()).hexdigest());vectors=re.findall(r'\[([^\]]+)\]',d['describe']);assert len(vectors)==lengths[-1]
            for pos,vector in enumerate(vectors):
                values=vector.split();assert len(values)==3
                for ch,v in enumerate(values):eg[i,ch,pos]=np.float32((float(Decimal(v))-st['residue']['mean'][ch])/st['residue']['scale'][ch])
            padding &= a['g'][i,:,lengths[-1]:].tobytes()==np.zeros((3,48-lengths[-1]),np.float32).tobytes()
        ck(t+'_D_reference_g_independent_raw_source_formula_PAD_and_row_bindings',bits(eg,a['g']) and padding and lengths==a['reference_lengths'].tolist()==b['g_reference_lengths'] and b['g_source_row_ids']==ids.tolist() and b['g_reference_sequence_sha256']==sequence_hashes and b['g_digest']==digest(eg) and b['prepared_npz_sha256']==sha(RUN/f'prepared_{t}.npz'))
        inv=arrays(RUN/f'D_inverse_{t}.npz');roundtrip=arrays(RUN/f'D_roundtrip_{t}.npz')
        ck(t+'_D_saved_inverse_forward_input_chain',bits(inv['epsilon'],a['epsilon']) and bits(inv['g'],a['g']) and bits(roundtrip['epsilon'],a['epsilon']) and bits(roundtrip['z'],inv['z']) and shape(inv['z'],(16,16,48),np.float32) and shape(roundtrip['returned_zp'],(16,16,48),np.float32))
        residual=torch.from_numpy(roundtrip['returned_zp'])-eps;err=residual.abs();tol=1e-5+1e-5*eps.abs();norm=err/tol
        actual={'pass':bool((err<=tol).all()),'atol':1e-5,'rtol':1e-5,'max_absolute_error':float(err.max()),'max_normalized_error':float(norm.max()),'failed_coordinates':int((err>tol).sum()),'per_slot_max_absolute_error':err.flatten(1).max(1).values.tolist(),'per_slot_rms_error':residual.square().flatten(1).mean(1).sqrt().tolist(),'per_slot_max_normalized_error':norm.flatten(1).max(1).values.tolist()}
        ck(t+'_D_saved_roundtrip_residual_tolerance_and_all_slot_stats',bits(residual,roundtrip['residual']) and bits(norm,roundtrip['normalized_error']) and actual==rt[t] and actual['pass'] and actual['failed_coordinates']==0)
        facts['terminals'][t]={'selected_train_row_ids':ids.tolist(),'duplicate_count':b['duplicate_count'],'roundtrip':actual,'source_z':{},'selected_sigma':numerical_summary(a['selected_sigma'])}
        for mi,m in enumerate(METHODS):
            output=arrays(RUN/f'{m}_{t}.npz');expected_z=(a['B_z'],a['C_z'],inv['z'])[mi];ss=seal['all_96_decoder_inputs'][m+'_'+t]
            ck(m+'_'+t+'_sealed_input_all_output_shapes_finite',bits(expected_z,output['z']) and ss=={'dtype':'torch.float32','shape':[16,16,48],'sha256':digest(expected_z)} and shape(output['raw'],(16,21,48),np.float32) and shape(output['tokens'],(16,48),np.int64) and shape(output['lengths'],(16,),np.int64) and shape(output['length_scores'],(16,49),np.float32))
            raw=torch.from_numpy(output['raw']);mx,aatok=raw[:,:20,:].max(dim=1);pad=raw[:,20,:];basepad=pad.sum(dim=1,keepdim=True);js=basepad+torch.cat((torch.zeros((16,1),dtype=torch.float32),(mx-pad).cumsum(dim=1)),dim=1);lens=js.argmax(dim=1);tokens=torch.where(torch.arange(48)[None,:]<lens[:,None],aatok,torch.full_like(aatok,20))
            ck(m+'_'+t+'_independent_J_scores_argmax_tokens_bytes',bits(js,output['length_scores']) and bits(lens,output['lengths']) and bits(tokens,output['tokens']))
            selected_rows=[r for r in new if r['method']==m and r['terminal']==t];assert len(selected_rows)==16
            for i,row in enumerate(selected_rows):
                n=int(lens[i]);seq=''.join(AA[int(v)] for v in tokens[i,:n]);status='empty_sequence' if n==0 else 'no_eos_at_limit' if n==48 else 'ok'
                assert row['sequence']==seq and int(row['length'])==n and row['status']==status and int(row['terminal_sample_id'])==i
                assert row['reference_train_row_id']==('' if mi==1 else str(ids[i]))
            ck(m+'_'+t+'_all16_csv_sequences_lengths_status_and_reference_rows',True)
            facts['terminals'][t]['source_z'][m]=numerical_summary(output['z'])
    ck('audit_global_rng_restored',bits(torch.get_rng_state(),global_rng))
    ck('input_seal_exact6_groups_and_all128_slot_denominator',set(seal['all_96_decoder_inputs'])=={m+'_'+t for m in METHODS for t in TERMS} and len(new)+len(reuse)==128)
    ck('all_tracked_sources_unchanged_after_readonly_audit',all(sha(p)==s for p,s in tracked.items()),{'files':len(tracked)})
    facts['costs_seconds']={'source_before_final_io':before,'source_after_manifest_before_closing_ledger':inner,'supervisor_outer':outer,'source_phase_components':state['phase_seconds']}
    facts['observations']={'new':96,'reused':32,'total':128,'new_status_counts':{s:sum(r['status']==s for r in new) for s in ('ok','empty_sequence','no_eos_at_limit','failed','not_executed')},'reused_status_counts':{s:sum(r['status']==s for r in reuse) for s in ('ok','empty_sequence','no_eos_at_limit')}}

error=None
try:main()
except Exception:
    error=traceback.format_exc()
report={'created_utc':datetime.now(timezone.utc).isoformat(),'status':'PASS_CHECKS_PENDING_INDEPENDENT_INTERPRETATION' if error is None else 'FAIL','scope_id':'AMP_H5_CE_SOURCE_REPLACEMENT_V2','generation_attempt_id':'ce_source_replacement_v2','generation_attempt_root':str(RUN),'checks':checks,'passed_checks':sum(x['pass'] for x in checks),'failed_checks':[x for x in checks if not x['pass']],'error':error,'facts':facts,'audit_seconds':time.perf_counter()-START,'audit_model_forwards':0,'audit_model_constructions':0,'audit_new_sampling_requests':0,'audit_rng_replay':'Only the fixed already-used streams 422/423,523/524,625/626; no new seeds or selected samples.','audit_checkpoint_loading':'torch.load(weights_only=True), state tensors only; no model import or construction.','test37_reads':0,'dev41_reads':0,'evidence_limits':['Model calls and seal ordering supported by exact bound code, terminal ledger, saved intermediates and filesystem timestamps; no independent model-forward replay.','D roundtrip residuals recomputed from saved returned_zp and epsilon; this is not independent flow execution and does not prove conditional-information fairness.','No sequence quality or new method superiority assessed.'],'script_sha256':sha(__file__),'source_bindings':[{'path':p,'sha256':s,'bytes':Path(p).stat().st_size} for p,s in sorted(tracked.items())]}
if REPORT.exists():raise RuntimeError('Refuse overwrite of prior audit; choose explicit new report/version')
REPORT.write_text(json.dumps(report,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
print(json.dumps({'report':str(REPORT),'sha256':sha(REPORT),'status':report['status'],'passed_checks':report['passed_checks'],'error':error,'audit_seconds':report['audit_seconds']}))
raise SystemExit(0 if error is None else 2)
