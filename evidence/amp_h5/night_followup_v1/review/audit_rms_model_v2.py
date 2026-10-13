"""Independent saved-artifact audit. No model construction, forward or scoring.

Run only after root confirms the exact RMS v2 job is terminal. Existing epsilon
banks may be replayed with private generators; no new prediction is produced.
"""
import os
for _k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[_k]='1'
import time
START=time.perf_counter()
import csv,hashlib,json,sys,traceback
from pathlib import Path
sys.dont_write_bytecode=True
R=Path('/Users/loctek/Documents/Codex/2026-10-11/task-4/amp_h5')
O=R/'runs/ce_rms_reconstruction_v2'
P=R/'protocol/CE_RMS_RECONSTRUCTION_PLAN_v2.json'
C=R/'protocol/CE_RMS_RECONSTRUCTION_CONTRACT_v2.json'
SUP=R/'runs/ce_rms_reconstruction_execution_v2/SUPERVISOR_STATUS.json'
DEST=R/'review/independent/CE_RMS_RECONSTRUCTION_POSTFLIGHT_v2.json'
AA='ARNDCEQGHILKMFPSTWYV'
SEEDS=list(range(1422,1430))
CHECKS=[]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(p):return json.loads(Path(p).read_text())
def record(p):
    p=Path(p);return {'path':str(p),'bytes':p.stat().st_size,'sha256':sha(p)}
def check(name,condition,detail=None):
    CHECKS.append({'name':name,'passed':bool(condition),'detail':detail})
    if not condition:raise AssertionError(name)
def csvrows(p):
    with Path(p).open(newline='') as f:return list(csv.DictReader(f))
def digest(a):return hashlib.sha256(a.tobytes(order='C')).hexdigest()
def bits(a,b):return a.dtype==b.dtype and a.shape==b.shape and digest(a)==digest(b)
def binding(b):
    p=Path(b['path']);return sha(p)==b['sha256'] and ('bytes' not in b or p.stat().st_size==b['bytes'])
def run():
    sup=load(SUP);state=load(O/'status.json');closing=load(O/'CLOSING_LEDGER.json')
    check('terminal_before_any_array_audit',sup['state']=='completed' and state['state']=='completed' and closing['state']=='completed')
    job=sup['jobs'][0]
    check('single_job_exit0_outer_budget',len(sup['jobs'])==1 and job['id']=='ce_rms_reconstruction_v2' and job['exit_code']==0 and not job.get('outer_timeout',False) and job['seconds']<=180)
    check('inner_closing_budget_and_manifest',closing['process_seconds_after_manifest_before_closing_ledger']<=120 and closing['manifest_sha256']==sha(O/'manifest.json'))
    plan,contract=load(P),load(C)
    check('exact_plan_identity',sha(P)=='b917c14e8db3e89f5a6a862bda5d3318bb202ed295244f04e80beee5ff5c909d' and sup['plan_sha256']==sha(P))
    check('all_original_bound_inputs_preserved',all(binding(b) for b in plan['bindings']) and all(binding(b) for b in contract['input_bindings']))
    manifest=load(O/'manifest.json');files=manifest['files']
    check('manifest_all_members_hash_bytes',len(files)==len({b['path'] for b in files}) and all(binding(b) for b in files))
    check('status_plan_receipt_contract_script_bound',state['plan_sha256']==sha(P) and state['contract_sha256']==sha(C) and state['receipt_sha256']==sha(R/'review/independent/CE_RMS_RECONSTRUCTION_PREFLIGHT_v2.json') and state['script_sha256']==sha(R/'implementation/ce_rms_reconstruction_v2.py'))
    check('declared_exact_calls_and_counts',state['completed_rows']==2880 and state['failed_rows']==state['not_executed_rows']==0 and state['encoder_forward_attempts']==2 and state['decoder_forward_attempts']==99 and state['discarded_encoder_internal_z_rows']==41 and state['new_training_updates']==state['DDPM_forward_attempts']==state['prior_flow_forward_attempts']==0)
    check('input_model_integrity_and_historical_parity',state['inputs_unchanged'] is True and state['model_unchanged'] is True and state['historical_M_parity'] is True and not state['finalization_errors'] and not load(O/'input_integrity_after.json')['errors'])
    before,after=load(O/'model_identity_before.json'),load(O/'model_identity_after.json')
    check('checkpoint_identity_evidence',before['sha256']==sha(R/'runs/formal_v2/categorical_ce/vae_state.pt') and before['tensor_digest']==after['before']==after['after']==contract['checkpoint_tensor_digest'] and after['unchanged'] is True and before['modelinit_global_rng_restored'] is True)
    runtime=load(O/'runtime_and_source.json');source=Path(load(R/'runs/formal_v2/categorical_ce/config.json')['source_root'])
    old=load(source/'runner/source_copy_manifest.json');code=[e for e in old['files'] if Path(e['relative_path']).suffix in ('.py','.pyc','.yaml','.yml')]
    skip=[e for e in old['files'] if e not in code]
    code_map={str(source/'corrected_code'/e['relative_path']):e['corrected_sha256'] for e in code}
    check('code_only_runtime_no_legacy_data_hash',len(code)==runtime['verified_code_count']==82 and len(skip)==runtime['skipped_noncode_count']==9 and runtime['legacy_dataset_contents_read'] is False and {x['path']:x['sha256'] for x in runtime['source_files']}==code_map and all(sha(q)==h for q,h in code_map.items()) and all(x['opened_or_hashed'] is False for x in runtime['skipped_noncode']))
    refs=csvrows(O/'reference_rows.csv');rows=csvrows(O/'reconstructions.csv')
    ref={(r['split'],int(r['row_id'])):r for r in refs};pred={(r['split'],int(r['row_id']),r['mode'],int(r['draw_index'])):r for r in rows}
    slots=load(R/'protocol/CE_RMS_RECONSTRUCTION_REQUEST_SLOTS_v2.json')['slots']
    check('all_reference_and_prediction_identities',len(refs)==len(ref)==320 and len(rows)==len(pred)==2880 and {r['sample_id'] for r in rows}=={x['sample_id'] for x in slots})
    for split,n,offset in [('train',279,0),('development',41,279)]:
        for i in range(n):
            rr=ref[(split,i)]
            assert int(rr['source_row_id'])==offset+i and int(rr['target_length'])==len(rr['truth_sequence']) and rr['terminal']==('FREE','AMD')[int(rr['class'])]
            for mode,draw,seed in [('M',-1,'')]+[('S',j,str(seed)) for j,seed in enumerate(SEEDS)]:
                row=pred[(split,i,mode,draw)]
                assert row['seed']==seed and all(row[k]==rr[k] for k in ('source_row_id','row_key','terminal','class','truth_sequence','target_length'))
    check('reference_truth_row_map_and_all_seeds',True)
    import numpy as np
    import torch
    torch.set_num_threads(1)
    check('runtime_versions',np.__version__=='1.26.4' and torch.__version__=='2.2.2')
    with np.load(O/'posterior_parameters320.npz',allow_pickle=False) as a:q=a['values'].copy();qlabels=a['labels'].copy();qids=a['source_row_ids'].copy()
    with np.load(R/'runs/formal_v2/categorical_ce/train_posterior_parameters.npz',allow_pickle=False) as a:train=a['values'].copy()
    with np.load(O/'dev_posterior_parameters.npz',allow_pickle=False) as a:dev=a['values'].copy();devlabels=a['labels'].copy();devids=a['source_row_ids'].copy()
    check('saved_q320_train_cache_dev41_full_identity',q.shape==(320,2,16,48) and q.dtype==np.float32 and bits(q[:279],train) and bits(q[279:],dev) and qids.tolist()==list(range(320)) and devids.tolist()==list(range(279,320)) and qlabels.tolist()==[int(r['class']) for r in refs] and devlabels.tolist()==qlabels[279:].tolist())
    qt=torch.from_numpy(q);sigma=qt[:,1].exp()
    check('posterior_mu_logsigma_sigma_finite_positive',np.isfinite(q).all() and bool(torch.isfinite(sigma).all()) and bool((sigma>0).all()))
    work=load(O/'DEV_ENCODING_WORK_RNG.json')
    check('dev_encoder_work_calls_only_discarded_samples',work['work_seed']==0 and [b['rows'] for b in work['batches']]==[32,9] and work['internal_sample_is_not_an_S_seed'] is True)
    epsseal=load(O/'EPSILON_BANKS_SEAL.json');banks={};rng_before=torch.get_rng_state().clone()
    for seed in SEEDS:
        with np.load(O/f'epsilon_seed_{seed}.npz',allow_pickle=False) as a:eps=a['epsilon'].copy();initial=a['generator_initial_state'].copy();final=a['generator_final_state'].copy()
        g=torch.Generator(device='cpu').manual_seed(seed)
        assert bits(g.get_state().numpy(),initial)
        replay=torch.randn((320,16,48),dtype=torch.float32,device='cpu',generator=g)
        assert bits(replay.numpy(),eps) and bits(g.get_state().numpy(),final)
        e=next(x for x in epsseal['banks'] if x['seed']==seed)
        assert e['epsilon_sha256']==digest(eps) and e['initial_generator_sha256']==digest(initial) and e['final_generator_sha256']==digest(final) and e['file_sha256']==sha(O/f'epsilon_seed_{seed}.npz')
        banks[seed]=torch.from_numpy(eps)
    check('all_eight_existing_private_epsilon_banks_byte_replayed',torch.equal(rng_before,torch.get_rng_state()) and epsseal['global_rng_unchanged'] is True and epsseal['generated_before_any_model_construction_or_forward'] is True)
    rawseal=load(O/'RAW_INPUTS_SEAL.json');qseal=load(O/'POSTERIOR_INPUTS_SEAL.json')
    check('saved_input_seals_bound',rawseal['reference_rows_sha256']==sha(O/'reference_rows.csv') and rawseal['epsilon_banks_seal_sha256']==sha(O/'EPSILON_BANKS_SEAL.json') and qseal['posterior_parameters_sha256']==sha(O/'posterior_parameters320.npz') and qseal['dev_q_sha256']==sha(O/'dev_posterior_parameters.npz') and qseal['before_any_decoder'] is True and qseal['global_work_rng_restored'] is True)
    legacy={}
    for split,f in [('train','train_reconstruction.csv'),('development','dev_reconstruction.csv')]:
        for row in csvrows(R/'runs/ce_checkpoint_recovery_v1'/f):legacy[(split,int(row['row_id']))]=row
    batch_count=0;row_count=0;status_counts={};zmin=1e99;zmax=-1e99
    for mode,draw,seed in [('M',-1,'')]+[('S',j,s) for j,s in enumerate(SEEDS)]:
        for split,n,offset in [('train',279,0),('development',41,279)]:
            for first in range(0,n,32):
                count=min(32,n-first);ix=slice(offset+first,offset+first+count)
                with np.load(O/f'{mode}{seed}_{split}_{first:03d}.npz',allow_pickle=False) as a:
                    saved={k:a[k].copy() for k in ('z','raw','tokens','lengths','length_scores')}
                assert saved['raw'].shape==(count,21,48) and saved['tokens'].shape==(count,48) and saved['length_scores'].shape==(count,49)
                assert all(np.isfinite(a).all() for a in saved.values())
                z=qt[ix,0] if mode=='M' else qt[ix,0]+banks[seed][ix]*sigma[ix]
                assert bits(z.numpy(),saved['z'])
                scores=torch.from_numpy(saved['raw']).transpose(1,2)
                body,aa=scores[:,:,:20].max(-1);pad=scores[:,:,20];base=pad.sum(1,keepdim=True)
                js=base+torch.cat((torch.zeros_like(base),(body-pad).cumsum(1)),dim=1)
                lengths=js.argmax(1);tokens=torch.where(torch.arange(48)[None,:]<lengths[:,None],aa,torch.full_like(aa,20))
                assert bits(js.numpy(),saved['length_scores']) and bits(tokens.numpy(),saved['tokens']) and bits(lengths.numpy(),saved['lengths'])
                for j,i in enumerate(range(first,first+count)):
                    row=pred[(split,i,mode,draw)];tok=tokens[j].tolist();nseq=int(lengths[j]);seq=''.join(AA[t] for t in tok[:nseq]);status='empty_sequence' if nseq==0 else 'no_eos_at_limit' if nseq==48 else 'ok'
                    assert row['prediction_sequence']==seq and row['prediction_finite']=='true' and row['status']==status and int(row['length'])==int(row['selected_length'])==nseq and json.loads(row['prediction_tokens48'])==tok and int(row['first_pad_index'])==(nseq if nseq<48 else -1) and row['length48_cap']==str(nseq==48).lower()
                    if mode=='M':
                        old=legacy[(split,i)];assert row['truth_sequence']==old['truth_sequence'] and seq==old['prediction_sequence'] and tok==json.loads(old['prediction_tokens48'])==json.loads(old['raw_prediction_tokens48'])
                    status_counts[status]=status_counts.get(status,0)+1;row_count+=1
                zmin=min(zmin,float(z.min()));zmax=max(zmax,float(z.max()));batch_count+=1
    check('all99_saved_batches_all2880_reparameterizations_and_J_byte_equal',batch_count==99 and row_count==2880)
    check('all320_historical_M_token_truth_parity',True)
    pilot=load(O/'FIRST_BATCH_THROUGHPUT_PILOT.json')
    check('retained_first_batch_throughput_rule',pilot['additional_requests']==0 and pilot['remaining_batches']==98 and pilot['safety_multiplier']==8 and pilot['estimated_total_seconds']<=120 and pilot['continue_by_time_and_finite_only'] is True and not pilot['quality_scores_computed'])
    check('final_bound_inputs_still_unchanged',all(binding(b) for b in plan['bindings']) and all(binding(b) for b in files))
    return {'outer_seconds':job['seconds'],'inner_after_manifest_seconds':closing['process_seconds_after_manifest_before_closing_ledger'],'forward_calls_recorded':{'encoder':2,'decoder':99,'DDPM':0,'prior_flow':0,'training':0},'predictions':row_count,'status_counts':status_counts,'scale_description':{'mu_min':float(qt[:,0].min()),'mu_max':float(qt[:,0].max()),'logsigma_min':float(qt[:,1].min()),'logsigma_max':float(qt[:,1].max()),'sigma_min':float(sigma.min()),'sigma_max':float(sigma.max()),'all_saved_z_min':zmin,'all_saved_z_max':zmax,'scope':'finite continuous diagnostics only; no learned-distribution adequacy claim'}}
if __name__=='__main__':
    if DEST.exists():raise FileExistsError('preserve previous audit; new version needed')
    result={'status':'FAIL','scope_id':'AMP_H5_CE_RMS_RECONSTRUCTION_V2','checks':CHECKS,'model_calls_by_reviewer':0,'new_prediction_rows_by_reviewer':0,'independent_additional_encoder_decoder_flow_DDPM_calls':0,'test_access':False,'original_CE_E_L_G':'NOT_EVALUABLE_PERMANENTLY','PROMOTE':'HOLD','training_authorized':False}
    try:result.update(evidence=run(),status='PASS')
    except Exception as exc:result.update(error=repr(exc),traceback=traceback.format_exc())
    result['source_bindings']=[record(p) for p in [P,C,R/'implementation/ce_rms_reconstruction_v2.py',SUP,O/'manifest.json',O/'CLOSING_LEDGER.json',O/'status.json',O/'reference_rows.csv',O/'reconstructions.csv',Path(__file__)] if p.is_file()]
    result['inference_boundaries']=['Saved logits/J/reparameterization and RNG banks were independently checked; there was no extra model forward.','New dev posterior parameters have recorded encoder provenance and historical M token parity, but were not recomputed with a second independent encoder forward.','Before/after model tensor identities are runtime evidence; reviewer rechecked checkpoint file hashes without constructing or deserializing model weights.','This engineering PASS only permits the separately bound scorer; investment gate and any training permission are separate.']
    result['check_count']=len(CHECKS);result['audit_seconds_before_receipt_io']=time.perf_counter()-START
    with DEST.open('x') as f:json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps({'status':result['status'],'receipt':str(DEST),'sha256':sha(DEST),'checks':len(CHECKS),'audit_seconds':result['audit_seconds_before_receipt_io']}))
    raise SystemExit(0 if result['status']=='PASS' else 2)
