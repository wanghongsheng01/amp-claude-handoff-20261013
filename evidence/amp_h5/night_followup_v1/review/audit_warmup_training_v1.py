"""Single saved-artifact audit; no model construction/forward, cohort or test read."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[k]='1'
import time
START=time.perf_counter()
import json,hashlib,math,datetime
from pathlib import Path
import numpy as np
import torch
torch.set_num_threads(1)
BASE=Path('/Users/loctek/Documents/Codex/2026-10-11/task-4/amp_h5')
C=BASE/'night_followup_v1/warmup_candidate'
R=C/'run_v1'
OLD=BASE/'runs/formal_v2/categorical_ce'
OUT=BASE/'review/independent/CE_BETA_WARMUP_VAE_ONLY_POSTFLIGHT_v1.json'
PLAN=BASE/'protocol/CE_BETA_WARMUP_VAE_ONLY_PLAN_v1.json'
SUP=BASE/'night_followup_v1/warmup_execution_v1/SUPERVISOR_STATUS.json'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def lines(p):return [json.loads(x) for x in Path(p).read_text().splitlines()]
def bind(p):return dict(path=str(p),sha256=sha(p),bytes=p.stat().st_size)
def finite_json(v):
    if isinstance(v,float):assert math.isfinite(v)
    elif isinstance(v,dict):
        for x in v.values():finite_json(x)
    elif isinstance(v,list):
        for x in v:finite_json(x)
def tensor_hash(state):
    h=hashlib.sha256()
    for name,value in sorted(state.items()):
        a=value.detach().cpu().contiguous().numpy()
        assert np.isfinite(a).all(),name
        h.update(name.encode());h.update(str(a.dtype).encode());h.update(str(a.shape).encode());h.update(a.tobytes())
    return h.hexdigest()
assert not OUT.exists(),'no overwrite'
assert sha(PLAN)=='575bab69d2de163c92aee20f52a64a02faf158d2355850a6e0b83bee94a702d6'
assert sha(C/'CONTRACT.json')=='733ad7dab4c3dd1bfad010f07b3fb5ebf8e32f0c0f60ce71521caaed0fdafc9a'
contract=read(C/'CONTRACT.json');plan=read(PLAN);supervisor=read(SUP);status=read(R/'status.json')
meta=read(R/'vae_state.json');summary=read(R/'vae_training_summary.json');manifest=read(R/'manifest.json')
assert supervisor['state']=='completed' and supervisor['plan_sha256']==sha(PLAN) and len(supervisor['jobs'])==1
job=supervisor['jobs'][0]
assert job['state']=='completed' and job['exit_code']==0 and not job.get('outer_timeout')
assert job['argv']==plan['jobs'][0]['argv'] and 0<job['seconds']<=2100
assert status['state']=='completed_vae_only_not_scientific_success'
for k in ('vae_updates','objective_forward_attempts','completed_updates_logged'):assert status[k]==1000
for k in ('fixed_monitor_calls','fixed_monitors_logged'):assert status[k]==10
for k in ('ddpm_updates','generation_requests','posttraining_q_or_rms_calls','test_reads'):assert status[k]==0
assert status['checkpoint_selection'] is False and status['partial_forward_or_monitor_possible'] is False
assert summary['updates']==summary['requested_updates']==meta['updates']==1000
assert summary['fixed_endpoint_reached'] and meta['fixed_endpoint_reached']
assert 0<summary['elapsed_seconds']<=1920 and summary['elapsed_seconds']<job['seconds']
members={x['path']:x for x in manifest['files']}
assert len(members)==len(manifest['files'])==16
assert set(members)=={p.name for p in R.iterdir() if p.is_file() and p.name!='manifest.json'}
before={}
for name,ref in members.items():
    p=R/name;assert p.parent==R and sha(p)==ref['sha256'] and p.stat().st_size==ref['bytes']
    before[str(p)]=sha(p)
for ref in contract['frozen_metadata_and_weights'].values():
    # Only prebound metadata/weights/code; none of these records is a cohort or test dataset.
    p=Path(ref['path']);assert p.suffix.lower()!='.csv';assert sha(p)==ref['sha256']
    before[str(p)]=sha(p)
assert read(R/'root_launch_receipt.json')==read(C/'ROOT_LAUNCH_AUTHORIZATION.json')
assert status['root_receipt_sha256']==sha(C/'ROOT_LAUNCH_AUTHORIZATION.json')
effective=read(R/'effective_vae_config.json');expected=dict(contract['baseline_config'])
expected.update(training_seconds_cap=2100,vae_seconds_cap=1920)
assert effective==expected
assert sha(R/'train_feature_scaler.json')==contract['frozen_metadata_and_weights']['baseline_scaler']['sha256']
runtime=read(R/'runtime_and_source.json');oldruntime=read(OLD/'runtime_and_source.json')
for key in ('torch','numpy','python','threads','interop_threads','device'):assert runtime[key]==oldruntime[key]
assert runtime['code_only_bootstrap'] is True and runtime['excluded_manifest_members_not_accessed']==9
t=lines(R/'vae_train.jsonl');beta=lines(R/'beta_loss_split.jsonl');attempt=lines(R/'paired_forward_attempts.jsonl');old=lines(OLD/'vae_train.jsonl')
assert len(t)==len(beta)==len(attempt)==len(old)==1000
for i,(v,w,a,h) in enumerate(zip(t,beta,attempt,old),1):
    finite_json(v);finite_json(w)
    assert v['step']==w['step']==a['step']==h['step']==i and a['zero_based_update']==i-1
    for key in ('rows','record_indices','forward_rng_sha256'):
        assert v[key]==a[key]==h[key]
    assert v['lr']==w['lr']==a['expected_lr']==h['lr']==max(.0005*max(0,i-2)/1000,1e-7)
    assert len(v['record_indices'])==v['rows'] and all(0<=x<279 for x in v['record_indices'])
    assert hashlib.sha256(bytes.fromhex(a['forward_rng_state_hex'])).hexdigest()==v['forward_rng_sha256']
    assert a['beta']==w['beta']==min(1.,(i-1)/200.) and a['initialization_batch_rng_verified'] is True
    assert w['paired_trace_fields_equal'] is True and w['normalization_denominator']==v['rows']*48
    assert w['unweighted_kl']==v['kl'] and w['loss']==v['loss'] and w['reconstruction']==v['reconstruction'] and w['density_sum']==v['density_sum']
    weighted=np.float32(w['beta'])*np.float32(v['kl'])
    assert float(weighted)==w['weighted_kl']
    assert float(np.float32(v['reconstruction'])+weighted)==v['loss']
    assert float(np.float32(v['density_sum'])/np.float32(v['rows']*48))==v['kl']
    if i<1000:assert w['postforward_rng_sha256']==attempt[i]['forward_rng_sha256']
    assert len(a['target_g_h_sha256'])==64
for key in ('reconstruction','kl','density_sum'):assert t[0][key]==old[0][key]
mon=lines(R/'dev_monitor.jsonl');oldmon=lines(OLD/'dev_monitor.jsonl')
assert [x['step'] for x in mon]==list(range(100,1001,100)) and len(oldmon)==10
for v,h in zip(mon,oldmon):
    finite_json(v);assert v['rows']==41 and v['nonpad']==701
    assert v['rng_preserved'] is True and v['checkpoint_selection'] is False
    assert v['rng_sha256']==beta[v['step']-1]['postforward_rng_sha256']==h['rng_sha256']
    assert v['nonpad_token_accuracy']==v['correct_nonpad']/701 and v['length_mae']==v['absolute_length_error']/41
assert sha(R/'vae_initial_state.pt')==sha(OLD/'vae_initial_state.pt')
initial=torch.load(R/'vae_initial_state.pt',map_location='cpu',weights_only=True)
final=torch.load(R/'vae_state.pt',map_location='cpu',weights_only=True)
assert set(initial)==set(final)
initial_digest=tensor_hash(initial);final_digest=tensor_hash(final)
assert initial_digest==contract['initial_tensor_state_sha256']==summary['initial_tensor_state_sha256']==meta['initial_tensor_state_sha256']
assert final_digest==meta['tensor_state_sha256']=='85f650713eadc074b126c7082903fa4153cc07d1ab2b20d53e5a94913d97a89a'
assert meta['sha256']==sha(R/'vae_state.pt')=='05b9497ec78697dafdbceaae0cac5ef54e8b157b3b3f9cf417053b6d7e9e96d6'
for key in initial:assert initial[key].shape==final[key].shape and initial[key].dtype==final[key].dtype
training_state=torch.load(R/'vae_training_state.pt',map_location='cpu',weights_only=True)
assert training_state['completed_updates']==1000
assert hashlib.sha256(training_state['torch_rng'].numpy().tobytes()).hexdigest()==beta[-1]['postforward_rng_sha256']
for slot in training_state['optimizer']['state'].values():
    assert float(slot['step'])==1000
    for v in slot.values():
        if torch.is_tensor(v):assert torch.isfinite(v).all()
for p,h in before.items():assert sha(p)==h
paths=[R/'vae_state.pt',R/'vae_state.json',R/'vae_training_summary.json',R/'status.json',SUP,C/'CONTRACT.json',R/'manifest.json',PLAN,C/'warmup_vae_only.py',R/'vae_initial_state.pt',R/'vae_initialization.json',R/'vae_train.jsonl',R/'beta_loss_split.jsonl',R/'paired_forward_attempts.jsonl',R/'dev_monitor.jsonl',R/'vae_training_state.pt',R/'train_feature_scaler.json',R/'runtime_and_source.json',R/'effective_vae_config.json',OLD/'vae_train.jsonl',OLD/'dev_monitor.jsonl',BASE/'review/independent/CE_BETA_WARMUP_VAE_ONLY_PREFLIGHT_v1.json',BASE/'night_followup_v1/review/WARMUP_POSTTRAIN_PREPARATION_REVIEW_v1.json']
checks=['Single exact supervisor job completed exit0 under actual2100s cap','Final1000 VAE stage under1920s including checkpoint IO','All16 manifest members hashes/bytes and no extra run files','Frozen metadata/code/weights unchanged; run receipt/config/scaler/runtime match','All1000 baseline batch indices/rows/LR/preforward RNG hashes equal','All1000 stored full RNG states hash correctly; postforward-to-next-step chain preserved','All1000 beta schedule and float32 weighted objective/KL normalization recomputed','First unweighted reconstruction/KL/density exactly match baseline','All10 fixed100step dev monitors preserve RNG/no selection/41rows701targets','Initial checkpoint byte-identical and actual tensor digest verified','Final saved tensors finite with metadata digest/hash and shape identity verified','Optimizer saved1000 counters and finite state; saved final RNG matches trace','Before/after consumed file hashes unchanged; no extra model or data access']
report={'status':'PASS','blocked':False,'scope_id':'AMP_H5_CE_BETA_WARMUP_VAE_ONLY_V1','created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'checks':[{'check':x,'pass':True}for x in checks],'independent_paired_trace_pass':True,'single_beta_schedule_only':True,'vae_updates':1000,'checkpoint_tensor_digest':final_digest,'initial_tensor_digest':initial_digest,'actual_vae_stage_seconds':summary['elapsed_seconds'],'actual_supervisor_seconds':job['seconds'],'monitor_calls':10,'model_forward_calls_by_audit':0,'model_constructions_by_audit':0,'new_random_draws_by_audit':0,'cohort_reads_by_audit':0,'test_reads':0,'checkpoint_loads':'weights_only=True CPU for own hash-bound saved initial/final/optimizer artifacts; no model instantiated','historical_pairing_limit':'Historical baseline lacks raw epsilon/full preforward states and input tensor hashes. Verified baseline indices/RNG hashes/LR and initial weights/first unweighted components; new stored state bytes validate their own hashes. No claim of independent historical raw epsilon replay or rematerialized batch tensors.','scope_boundary':'Training engineering PASS only. Original CE E/L/G remain NE and PROMOTE HOLD. Same-eight-epsilon endpoint requires its own late-bound exact plan; no retraining, DDPM or generation authorized.','last_existing_monitor_descriptive_only':{k:mon[-1][k]for k in ['step','correct_nonpad','nonpad','nonpad_token_accuracy','absolute_length_error','length_mae','exact_sequences']},'source_bindings':[bind(p)for p in paths],'audit_seconds_before_receipt_io':time.perf_counter()-START}
OUT.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'path':str(OUT),'sha256':sha(OUT),'checks':len(checks),'audit_seconds':report['audit_seconds_before_receipt_io'],'VAE_seconds':summary['elapsed_seconds'],'supervisor_seconds':job['seconds']}))
