#!/usr/bin/env python3
"""Saved-output audit for the one recovery attempt. No model imports or forwards.
Prepared before terminal notification; require both terminal status files.
The automatic report is not a scientific PASS or a following-arm authorization.
"""
from pathlib import Path
import argparse,collections,csv,datetime,hashlib,json,math,os
W=Path('/Users/loctek/Documents/Codex/2026-10-11/task-4'); H=W/'amp_h5'; O=H/'review/independent'
B=H/'runs/ce_checkpoint_recovery_v1'; ORIGINAL=H/'runs/formal_v2/categorical_ce'
S=H/'runs/ce_checkpoint_recovery_execution_v1/SUPERVISOR_STATUS.json'
CP=H/'protocol/CE_CHECKPOINT_RECOVERY_CONTRACT_v1.json'; PP=H/'protocol/CE_CHECKPOINT_RECOVERY_PLAN_v1.json'
RP=O/'CE_CHECKPOINT_RECOVERY_PREFLIGHT_RECEIPT_v1.json'
AA='ARNDCEQGHILKMFPSTWYV'; SUCCESS={'ok','empty_sequence','no_eos_at_limit'}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def lines(p):return [json.loads(s) for s in Path(p).read_text().splitlines()]
def summary(a):
 a=np.asarray(a,dtype=np.float64); finite=np.isfinite(a); f=a[finite]
 return {'shape':list(a.shape),'elements':int(a.size),'finite_elements':int(f.size),'nan':int(np.isnan(a).sum()),
         'positive_inf':int(np.isposinf(a).sum()),'negative_inf':int(np.isneginf(a).sum()),
         'min':float(f.min()) if f.size else None,'max':float(f.max()) if f.size else None,
         'mean':float(f.mean()) if f.size else None,'std':float(f.std()) if f.size else None,
         'rms':float(np.sqrt(np.mean(f*f))) if f.size else None}
def main():
 parser=argparse.ArgumentParser();parser.add_argument('--terminal-confirmed',action='store_true',required=True);parser.parse_args()
 dest=O/'CE_CHECKPOINT_RECOVERY_TERMINAL_AUDIT_v1.json'
 if dest.exists():raise FileExistsError('Preserve the existing audit; use a new version for any correction')
 sup=read(S); status=read(B/'status.json')
 if sup.get('state')=='running' or any(j.get('state')=='running' for j in sup.get('jobs',[])) or str(status.get('state','')).startswith('running'):
  raise RuntimeError('Live attempt: no artifact audit is allowed yet')
 for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):os.environ[k]='1'
 global np,torch
 import numpy as np
 import torch
 torch.set_num_threads(1)
 cfg=read(ORIGINAL/'config.json');contract=read(CP);plan=read(PP);receipt=read(RP)
 checks=[];errors=[];facts={'raw_status':status,'raw_supervisor':sup}
 def ck(name,ok,detail=None):checks.append({'name':name,'pass':bool(ok),**({'detail':detail} if detail is not None else {})})
 def section(name,fn):
  try:fn()
  except Exception as exc:errors.append({'section':name,'error':repr(exc)});ck(name+'_section_complete',False,repr(exc))
 def binding_checks():
  ck('exact_frozen_plan_contract_receipt',sha(PP)=='3e66ffa415dc759e396d25d4ff9c445c7c855fb0e48aaf96b834d6da1b1ae1e4' and
     sha(CP)=='bf610d590f06fa21e4d1b7fc2459ec62d3564bdd5bd32fdbe1fbdfb8e5a71a00' and sha(RP)=='0e9e4d79e34a7e56dee012a0716c240fc63e6b6dcb7492a5c28ad4274b1f3e56')
  ck('all_preflight_plan_core_bindings_unchanged',all(sha(x['path'])==x['sha256'] for x in plan['bindings']+contract['core_bindings']))
  ck('new_recorded_bindings_exact',status['recovery_contract_sha256']==sha(CP) and status['recovery_plan_sha256']==sha(PP) and
     status['independent_receipt_sha256']==sha(RP) and status['script_sha256']==contract['recovery_script_sha256'])
  ck('new_config_and_provenance_exact',read(B/'config.json')==cfg and read(B/'recovery_contract.json')==contract and read(B/'phase_provenance.json')==contract['phase_provenance'])
  ck('exact_supervisor_job_completed',sup['state']=='completed' and sup['plan_sha256']==sha(PP) and len(sup['jobs'])==1 and
     sup['jobs'][0]['argv']==plan['jobs'][0]['argv'] and sup['jobs'][0]['exit_code']==0)
  ck('new_recovery_completed_zero_training_original_NE',status['state']=='recovery_completed' and status['attempt_id']=='ce_checkpoint_recovery_v1' and
     status['scope_id']=='AMP_H5_CE_CHECKPOINT_RECOVERY_V1' and status['training_updates_this_attempt']==0 and
     status['original_v2_engineering_gate']=='NOT_EVALUABLE_INTERRUPTION' and status['PROMOTE']=='HOLD')
  manifest=read(B/'manifest.json')['files']; actual={p.name for p in B.iterdir() if p.is_file() and p.name!='manifest.json'}
  ck('manifest_inventory_and_hashes',{x['path'] for x in manifest}==actual and all(Path(x['path']).name==x['path'] and
     sha(B/x['path'])==x['sha256'] and (B/x['path']).stat().st_size==x['bytes'] for x in manifest))
  ck('no_fabricated_training_or_new_checkpoint_files',not any((B/n).exists() for n in ('vae_train.jsonl','ddpm_train.jsonl','vae_training_summary.json','ddpm_training_summary.json','vae_state.pt','ddpm_ema_state.pt','train_posterior_parameters.npz')))
  facts['manifest_files']=len(manifest);facts['manifest_sha256']=sha(B/'manifest.json')
  facts['output_snapshot_start']=[{'path':str(B/x['path']),'sha256':x['sha256']} for x in manifest]+[{'path':str(B/'manifest.json'),'sha256':sha(B/'manifest.json')}]
 section('bindings',binding_checks)
 def training_and_identity():
  original_items=contract['original_25_file_snapshot']; original_dirs=[Path(contract[k]) for k in ('original_run_directory','original_supervisor_directory')]
  ck('original25_membership_and_hashes_unchanged',len(original_items)==25 and {str(p) for d in original_dirs for p in d.iterdir() if p.is_file()}=={x['path'] for x in original_items} and
     all(sha(x['path'])==x['sha256'] and Path(x['path']).stat().st_size==x['bytes'] for x in original_items))
  forensic=read(O/'CE_V2_INTERRUPTION_FORENSIC_v2.json')
  ck('bound_original_training_forensic41_pass',forensic['status']=='TRAINING_CHECKPOINTS_VERIFIED_ORIGINAL_ATTEMPT_INCOMPLETE' and
     len(forensic['checks'])==41 and all(x['pass'] for x in forensic['checks']) and
     all(forensic['training_summary'][stage]['updates']==1000 and forensic['training_summary'][stage]['fixed_endpoint_reached'] for stage in ('vae','ddpm')))
  expected=contract['checkpoint_tensor_digests']; before=read(B/'loaded_state_identity.json');after=read(B/'model_identity_after.json');integrity=read(B/'input_integrity_after.json')
  ck('strict_loaded_and_final_tensor_identity',before['before']==before['expected']==after['before']==after['after']==after['expected']==expected and
     after['both_loaded_verified_and_unchanged'] and status['model_states_unchanged'])
  ck('original_checkpoint_sha_and_digest_link',all(sha(ORIGINAL/(name+'.pt'))==read(ORIGINAL/(name+'.json'))['sha256'] and
     read(ORIGINAL/(name+'.json'))['tensor_state_sha256']==expected[name]==forensic['state_digests'][name] for name in ('vae_state','ddpm_ema_state')))
  ck('final_input_integrity_record_clean',status['original_inputs_unchanged'] and integrity['original_errors']==integrity['core_errors']==[] and status['finalization_errors']==[])
  runtime=read(B/'runtime_and_source.json');old=read(ORIGINAL/'runtime_and_source.json')
  ck('same_runtime_singlethread_cpu_and_source_manifest',all(runtime[k]==old[k] for k in ('torch','numpy','python','threads','interop_threads','device','source_manifest_sha256')) and runtime['threads']==runtime['interop_threads']==1 and runtime['device']=='cpu' and runtime['source_files']==old['source_files'])
  restored=read(B/'restored_scaler_identity.json')
  ck('original_scaler_restored_no_fit',restored['source']==str(ORIGINAL/'train_feature_scaler.json') and restored['sha256']==sha(ORIGINAL/'train_feature_scaler.json') and restored['fit_called'] is False)
  facts['training_provenance']={'directory':str(ORIGINAL),'forensic_sha256':sha(O/'CE_V2_INTERRUPTION_FORENSIC_v2.json'),'updates':{'VAE':1000,'DDPM':1000},'new_training_updates':0,'original_status_preserved':read(ORIGINAL/'status.json')['state']}
 section('training_and_identity',training_and_identity)
 def budgets_and_exports():
  gen=read(B/'generation_summary.json')
  ck('new_generation_and_outer_caps',status['generation_cap_exceeded'] is False and status['outer_cap_exceeded'] is False and gen['elapsed_seconds']<=1800 and
     status['process_seconds_before_final_status_io']<=1980 and sup['jobs'][0]['seconds']<=1980 and status['budget_seconds']==contract['budget_seconds'])
  ck('only_bound_train_dev_validated_no_test',all(status[k] is True for k in ('source_train_read_attempted','source_train_read_validated','source_dev_read_attempted','source_dev_read_validated')) and status['test_read'] is False)
  ck('all_four_export_input_hashes_unchanged',all(cfg[k]==contract['allowed_export_source_paths'][k] and sha(cfg[k])==cfg[k+'_sha256'] for k in ('train','train_cohort','dev','dev_cohort')))
  cohorts={}
  for role,n in (('train',279),('dev',41)):
   p=Path(cfg[role+'_cohort'])
   if sha(p)!=cfg[role+'_cohort_sha256'] or 'test' in str(p).lower():raise ValueError('Refuse unbound natural cohort read')
   cohorts[role]=list(csv.DictReader(p.open(newline='')))
   rows=list(csv.DictReader((B/(role+'_reconstruction.csv')).open(newline='')))
   ck(role+'_export_exact_identity_and_rendering',len(rows)==len(cohorts[role])==n and all(
      int(r['row_id'])==i and r['terminal']==c['terminal_state'] and r['truth_sequence']==c['sequence'] and r['status']=='ok' and
      len(r['prediction_sequence'])<=48 and all(x in AA for x in r['prediction_sequence']) and
      json.loads(r['prediction_tokens48'])==[AA.index(x) for x in r['prediction_sequence']]+[20]*(48-len(r['prediction_sequence'])) and
      len(json.loads(r['raw_prediction_tokens48']))==48 and all(type(x) is int and 0<=x<=20 for x in json.loads(r['raw_prediction_tokens48']))
      for i,(r,c) in enumerate(zip(rows,cohorts[role]))))
  ck('train_dev_sequence_disjoint',not({r['sequence'] for r in cohorts['train']} & {r['sequence'] for r in cohorts['dev']}))
  facts['costs']={'original_training_stage_subtotal':1179.0319216251373+564.3461695411243,'original_generation_lower_bound':451.9357427917421,'original_exact_total':None,
     'recovery_setup':status['setup_seconds'],'recovery_generation':gen['elapsed_seconds'],'recovery_export':status['export_seconds'],
     'recovery_process_before_final_io':status['process_seconds_before_final_status_io'],'recovery_supervisor_outer':sup['jobs'][0]['seconds'],
     'additional_cost_separate':True}
 section('budgets_and_exports',budgets_and_exports)
 def generation():
  rows=list(csv.DictReader((B/'generated.csv').open(newline=''))); gen=read(B/'generation_summary.json')
  ck('32_exact_ids_terminal_seeds',[r['sample_id'] for r in rows]==[f'CE21_DDPM_{t}_{i:03d}' for t in ('FREE','AMD') for i in range(16)] and
     all(r['method']=='CE21_DDPM' and r['terminal']==('FREE' if i<16 else 'AMD') and int(r['seed'])==(422 if i<16 else 423) and int(r['terminal_sample_id'])==i%16 for i,r in enumerate(rows)))
  ck('32_actual_completed_requests',gen['requested']==gen['successful_execution']==32 and gen['failed_execution']==gen['not_executed']==0 and all(r['status'] in SUCCESS for r in rows))
  ck('terminal_only_no_retry_generation',gen['condition_inputs']==['terminal class only'] and gen['no_retry_or_filter'] and gen==status['generation'])
  progress=lines(B/'generation_progress.jsonl')
  ck('both_reverse_chains_complete_to_t1',all([x['timestep'] for x in progress if x['terminal']==t]==[900,800,700,600,500,400,300,200,100,1] for t in ('FREE','AMD')))
  with np.load(ORIGINAL/'train_posterior_parameters.npz',allow_pickle=False) as a:values=a['values'].copy();labels=a['labels'].copy();ids=a['row_ids'].copy()
  ck('fixed_train_cache_finite279',values.shape==(279,2,16,48) and values.dtype==np.float32 and np.isfinite(values).all() and
     np.array_equal(ids,np.arange(279)) and np.bincount(labels,minlength=2).tolist()==[126,153])
  def train_scale(v):
   mu=v[:,0].astype(np.float64);sigma=torch.from_numpy(v[:,1].copy()).exp().numpy().astype(np.float64);q2=float(np.mean(mu*mu+sigma*sigma));mean=float(mu.mean())
   return {'records':len(v),'mu':summary(mu),'logsigma':summary(v[:,1]),'sigma':summary(sigma),'analytic_qz_mean':mean,'analytic_qz_rms':math.sqrt(q2),'analytic_qz_std':math.sqrt(max(0,q2-mean*mean))}
  scales={'pooled_train':train_scale(values),'terminal_train':{},'generated':{}}
  ck('train_sigma_finite_positive',scales['pooled_train']['sigma']['finite_elements']==scales['pooled_train']['sigma']['elements'] and scales['pooled_train']['sigma']['min']>0)
  for label,t in enumerate(('FREE','AMD')):
   scales['terminal_train'][t]=train_scale(values[labels==label])
   with np.load(B/f'generated_{t}.npz',allow_pickle=False) as a:data={k:a[k].copy() for k in a.files}
   with np.load(B/f'generated_posterior_{t}.npz',allow_pickle=False) as a:posterior=a['values'].copy()
   ck(t+'_finite_generated_arrays',all(np.isfinite(v).all() for v in data.values()) and np.isfinite(posterior).all())
   ck(t+'_saved_shapes_and_posterior_exact',posterior.shape==(16,2,16,48) and data['z'].shape==(16,16,48) and data['raw'].shape==(16,21,48) and
      data['tokens'].shape==(16,48) and data['length_scores'].shape==(16,49) and np.array_equal(posterior,data['posterior_parameters']))
   mu=torch.from_numpy(posterior[:,0].copy());logsig=torch.from_numpy(posterior[:,1].copy());sigma=logsig.exp()
   ck(t+'_sigma_strict_positive_finite',bool(torch.isfinite(sigma).all()) and bool((sigma>0).all()))
   # RNG reproduction only: one initial draw and997 reverse draws, no model.
   local=torch.Generator().manual_seed(422+label)
   for unused in range(998):torch.randn((16,2,16,48),generator=local)
   epsilon=torch.randn((16,16,48),generator=local);z=mu+epsilon*sigma
   ck(t+'_same_seed_reparameterization_bitwise',z.numpy().tobytes()==data['z'].tobytes())
   raw=torch.from_numpy(data['raw']);body,aa=raw[:,:20,:].max(1);pad=raw[:,20,:]
   scores=pad.sum(1,keepdim=True)+torch.cat((torch.zeros((16,1)),(body-pad).cumsum(1)),1);length=scores.argmax(1)
   tokens=torch.where(torch.arange(48)[None,:]<length[:,None],aa,torch.full_like(aa,20))
   ck(t+'_raw_J_decode_exact',np.array_equal(scores.numpy(),data['length_scores']) and np.array_equal(length.numpy(),data['lengths']) and np.array_equal(tokens.numpy(),data['tokens']))
   rendered=[''.join(AA[x] for x in tok[:n]) for tok,n in zip(tokens.tolist(),length.tolist())]
   group=[r for r in rows if r['terminal']==t]
   ck(t+'_slot_rendering_and_status_exact',all(r['sequence']==s and int(r['length'])==len(s)==n and
      r['status']==('empty_sequence' if n==0 else ('ok' if n<48 else 'no_eos_at_limit')) for r,s,n in zip(group,rendered,length.tolist())))
   numeric={'mu':summary(mu.numpy()),'logsigma':summary(logsig.numpy()),'sigma':summary(sigma.numpy()),'z':summary(data['z']),'decoder_raw':summary(data['raw'])}
   numeric.update(z_rms_ratio_to_pooled_train=numeric['z']['rms']/scales['pooled_train']['analytic_qz_rms'],
      z_rms_ratio_to_terminal_train=numeric['z']['rms']/scales['terminal_train'][t]['analytic_qz_rms'],
      per_request_z_rms=np.sqrt(np.mean(data['z'].astype(np.float64)**2,axis=(1,2))).tolist())
   scales['generated'][t]=numeric
  facts['scales']=scales;facts['slot_status_counts']=dict(collections.Counter(r['status'] for r in rows));facts['requested_slots']=len(rows)
 section('generation',generation)
 def failures_and_stability():
  available={}
  for t in ('FREE','AMD'):
   f=B/f'generation_failure_{t}.json'
   if f.exists():
    available[t]={'record':read(f)};n=B/f'generation_failure_{t}.npz'
    if n.exists():
     with np.load(n,allow_pickle=False) as a:available[t]['saved_tensor_summaries']={k:summary(a[k]) for k in a.files}
  facts['failure_evidence']=available
  facts['raw_slot_status_inventory']=dict(collections.Counter(r['status'] for r in csv.DictReader((B/'generated.csv').open(newline=''))))
  ck('new_artifacts_stable_during_readonly_audit',all(sha(x['path'])==x['sha256'] for x in facts['output_snapshot_start']))
  ck('original25_still_unchanged_after_audit',all(sha(x['path'])==x['sha256'] for x in contract['original_25_file_snapshot']))
 section('failures_and_stability',failures_and_stability)
 passed=all(x['pass'] for x in checks)
 report={'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'status':'PASS_CHECKS_PENDING_INDEPENDENT_SCALE_ASSESSMENT' if passed else 'RECOVERY_ENGINEERING_FINDINGS_HOLD',
   'checks':checks,'failed_checks':[x for x in checks if not x['pass']],'section_errors':errors,'facts':facts,
   'scope':'RECOVERY_ENGINEERING_ONLY','generation_attempt_id':'ce_checkpoint_recovery_v1','generation_attempt_root':str(B),'training_source_root':str(ORIGINAL),
   'original_v2_engineering_gate':'NOT_EVALUABLE_INTERRUPTION','original_protocol_E_L_G':'NOT_EVALUABLE','PROMOTE':'HOLD','further_arms_authorized':False,
   'audit_script_sha256':sha(__file__),'source_bindings':[{'path':str(p),'sha256':sha(p)} for p in (B/'status.json',B/'generated.csv',ORIGINAL/'vae_state.pt',ORIGINAL/'ddpm_ema_state.pt') if p.exists()],
   'reviewer_actions':{'model_constructions':0,'model_forwards':0,'training_updates':0,'new_generation_requests':0,'test37_reads':0,
     'scope':'Saved tensors and reports; two hash-bound cohort CSVs for export identity; local RNG/arithmetic only; no model or optimizer deserialization'},
   'limitations':['Original interruption and protocol deviation remain; no retroactive original E/L/G success.','Numeric scale comparisons are descriptive and not independent-draw distributional tests; no posthoc quality threshold.','No-test evidence is input/config/source/output audit, not OS file-access tracing.','This automated audit requires separate reviewer scale judgment and does not authorize further arms.']}
 with dest.open('x') as f:json.dump(report,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
 print(json.dumps({'status':report['status'],'checks':len(checks),'failed_checks':report['failed_checks'],'report':str(dest)},ensure_ascii=False))
if __name__=='__main__':main()
