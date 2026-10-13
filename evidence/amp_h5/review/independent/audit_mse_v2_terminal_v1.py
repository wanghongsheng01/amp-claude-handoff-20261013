#!/usr/bin/env python3
"""Prepared before MSE terminal result: immutable saved artifacts only, no model calls.
Run only following parent terminal notification. No generation quality metrics.
"""
from pathlib import Path
import argparse, collections, csv, datetime, hashlib, json, math, os, random
W=Path('/Users/loctek/Documents/Codex/2026-10-11/task-4'); H=W/'amp_h5'; O=H/'review/independent'
B=H/'runs/formal_v2/raw_onehot_mse'; CE=H/'runs/formal_v2/categorical_ce'; CR=H/'runs/ce_checkpoint_recovery_v1'
S=H/'runs/single_raw_onehot_mse_execution_v2/SUPERVISOR_STATUS.json'
P=H/'protocol/SINGLE_MSE_V2_PLAN.json'; R=O/'SINGLE_MSE_V2_PREFLIGHT_RECEIPT.json'; C=H/'protocol/formal_configs_v2/raw_onehot_mse.json'
AA='ARNDCEQGHILKMFPSTWYV'; SUCCESS={'ok','empty_sequence','no_eos_at_limit'}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def lines(p):return [json.loads(s) for s in Path(p).read_text().splitlines()]
def nums(x):
 if isinstance(x,(int,float)):return math.isfinite(x)
 if isinstance(x,dict):return all(nums(v) for v in x.values())
 if isinstance(x,(list,tuple)):return all(nums(v) for v in x)
 return True
def digest(st):
 h=hashlib.sha256()
 for name,t in sorted(st.items()):
  a=t.detach().cpu().contiguous().numpy();h.update(name.encode());h.update(str(a.dtype).encode());h.update(str(a.shape).encode());h.update(a.tobytes())
 return h.hexdigest()
def summary(a):
 a=np.asarray(a,dtype=np.float64);f=a[np.isfinite(a)]
 return {'shape':list(a.shape),'elements':int(a.size),'finite_elements':int(f.size),'nan':int(np.isnan(a).sum()),'positive_inf':int(np.isposinf(a).sum()),'negative_inf':int(np.isneginf(a).sum()),'min':float(f.min()) if f.size else None,'max':float(f.max()) if f.size else None,'mean':float(f.mean()) if f.size else None,'std':float(f.std()) if f.size else None,'rms':float(np.sqrt(np.mean(f*f))) if f.size else None}
def main():
 p=argparse.ArgumentParser();p.add_argument('--terminal-confirmed',action='store_true',required=True);p.parse_args()
 dest=O/'MSE_V2_TERMINAL_AUDIT_v1.json'
 if dest.exists():raise FileExistsError('Existing audit must remain unchanged; version any corrections')
 sup=read(S);status=read(B/'status.json')
 if sup.get('state')=='running' or any(j.get('state')=='running' for j in sup.get('jobs',[])) or str(status.get('state','')).startswith('running'):raise RuntimeError('Live/stale-running attempt requires separate interruption handling; no terminal audit')
 for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):os.environ[k]='1'
 global np,torch
 import numpy as np
 import torch
 torch.set_num_threads(1)
 cfg=read(C);plan=read(P);receipt=read(R);checks=[];errors=[];facts={'raw_status':status,'raw_supervisor':sup}
 def ck(n,x,d=None):checks.append({'name':n,'pass':bool(x),**({'detail':d} if d is not None else {})})
 def section(n,fn):
  try:fn()
  except Exception as e:errors.append({'section':n,'error':repr(e)});ck(n+'_section_complete',False,repr(e))
 snapshot=[{'path':str(f),'sha256':sha(f),'bytes':f.stat().st_size} for f in sorted(B.iterdir()) if f.is_file()]+[{'path':str(S),'sha256':sha(S),'bytes':S.stat().st_size}]
 def bindings():
  ck('fixed_plan_receipt_config',sha(P)=='a1c24efa2164c5cd9fcca2e46154fac8af44df2aec5f363226da168a374ae802' and sha(R)=='29be127a09ac695f38d9fee442e7c998e1c6ab1a829e599fac0f4efc615e8768' and sha(C)=='861d530bce7529826ed481697083a4efee016cce24878d85dced240b28d9d004')
  ck('all_plan_bindings_unchanged',all(sha(b['path'])==b['sha256'] for b in plan['bindings']))
  ck('exact_job_terminal_success',sup['state']=='completed' and sup['plan_sha256']==sha(P) and len(sup['jobs'])==1 and sup['jobs'][0]['argv']==plan['jobs'][0]['argv'] and sup['jobs'][0]['exit_code']==0)
  ck('fixed_config_runner_status',read(B/'config.json')==cfg and status['config_sha256']==sha(C) and status['runner_sha256']==cfg['implementation_sha256']==sha(H/'implementation/vae_pipeline_formal_v2.py') and status['state']=='completed')
  manifest=read(B/'manifest.json')['files'];actual={f.name for f in B.iterdir() if f.is_file() and f.name!='manifest.json'}
  ck('manifest_members_unique_exact_and_hashes',len({m['path'] for m in manifest})==len(manifest) and {m['path'] for m in manifest}==actual and all(Path(m['path']).name==m['path'] and sha(B/m['path'])==m['sha256'] and (B/m['path']).stat().st_size==m['bytes'] for m in manifest))
  runtime=read(B/'runtime_and_source.json')
  ck('single_thread_CPU_and_source_hashes',runtime['device']=='cpu' and runtime['threads']==runtime['interop_threads']==1 and all(sha(s['path'])==s['sha256'] for s in runtime['source_files']))
  ck('exact_allowed_train_dev_hashes',all('test' not in cfg[k].lower() and sha(cfg[k])==cfg[k+'_sha256'] for k in ('train','train_cohort','dev','dev_cohort')))
  ck('validated_train279_dev41_no_test',status['test_read'] is False and all(status[k] is True for k in ('train_read_attempted','train_read_validated','development_read','development_read_attempted','development_read_validated')) and cfg['train_rows']==279 and cfg['dev_rows']==41)
 section('bindings',bindings)
 def training():
  logs={s:lines(B/(s+'_train.jsonl')) for s in ('vae','ddpm')};sums={s:read(B/(s+'_training_summary.json')) for s in ('vae','ddpm')}
  for s in ('vae','ddpm'):
   ck(s+'_1000_finite_updates',[r['step'] for r in logs[s]]==list(range(1,1001)) and nums(logs[s]) and sums[s]['updates']==sums[s]['requested_updates']==status[s+'_updates']==1000 and sums[s]['fixed_endpoint_reached'])
   ck(s+'_stage_cap',sums[s]['elapsed_seconds']<=cfg[s+'_seconds_cap'])
  ck('shared_training_cap',not status['training_cap_exceeded'] and status['training_seconds']<=2940)
  ck('process_outer_cap',status['process_seconds']<=4920 and sup['jobs'][0]['seconds']<=4920 and not sup['jobs'][0].get('outer_timeout',False))
  rng=random.Random(222);order=[];cursor=279;good=[]
  for r in logs['vae']:
   if cursor>=279:order=list(range(279));rng.shuffle(order);cursor=0
   ids=order[cursor:cursor+32];cursor+=len(ids);good.append(r['record_indices']==ids and r['rows']==len(ids))
  ck('all_VAE_seed222_batch_order',all(good))
  ck('VAE_fixed_lr',all(r['lr']==max(.0005*max(0,r['step']-2)/1000,1e-7) for r in logs['vae']))
  ck('MSE_objective_accounting',all(abs(r['loss']-r['reconstruction']-r['kl'])<=2e-6*max(1,abs(r['reconstruction'])+abs(r['kl'])) and math.isclose(r['kl'],r['density_sum']/(r['rows']*48),rel_tol=2e-6,abs_tol=2e-6) for r in logs['vae']))
  mon=lines(B/'dev_monitor.jsonl');ck('ten_fixed_rng_preserving_descriptive_monitors',[r['step'] for r in mon]==list(range(100,1001,100)) and nums(mon) and all(r['rows']==41 and r['rng_preserved'] and r['checkpoint_selection'] is False for r in mon))
  digests={}
  for name in ('vae_initial_state','vae_state','ddpm_state','ddpm_ema_state'):
   st=torch.load(B/(name+'.pt'),map_location='cpu',weights_only=True);meta=read(B/(name+'.json'));digests[name]=digest(st)
   ck(name+'_file_tensor_hash_and_finite',meta['sha256']==sha(B/(name+'.pt')) and meta['tensor_state_sha256']==digests[name] and all(bool(torch.isfinite(t).all()) for t in st.values()))
   if name!='vae_initial_state':ck(name+'_1000_endpoint',meta['updates']==meta['requested_updates']==1000 and meta['fixed_endpoint_reached'])
   del st
  ck('DDPM_frozen_EMA_copy_rule',digests['ddpm_state']==digests['ddpm_ema_state'] and sums['ddpm']['ema_step']==1000)
  ck('fresh_VAE_parameters_changed',digests['vae_initial_state']!=digests['vae_state'])
  scaler=read(B/'train_feature_scaler.json');ck('train_only_finite_scaler',scaler['fit_scope']=='train only' and scaler['fit_record_ids']==list(range(279)) and scaler['global']['n']==279 and nums(scaler) and all(v>0 for t in ('global','residue') for v in scaler[t]['scale']))
  facts['training_summary']=sums;facts['state_digests']=digests;facts['costs']={'training_seconds':status['training_seconds'],'generation_seconds':read(B/'generation_summary.json')['elapsed_seconds'],'process_seconds_before_final_io':status['process_seconds'],'supervisor_outer_seconds':sup['jobs'][0]['seconds'],'CE_recovery_costs_separate':True,'original_CE_exact_wall_unknown':True}
 section('training',training)
 def coupling():
  # Compare fixed computational exposures; losses/outcomes intentionally excluded.
  ce_cfg=read(CE/'config.json');ck('CE_MSE_config_diff_only_arm',[k for k in sorted(set(cfg)|set(ce_cfg)) if cfg.get(k)!=ce_cfg.get(k)]==['arm'])
  ck('CE_MSE_identical_initialization',read(B/'vae_initialization.json')==read(CE/'vae_initialization.json'))
  ck('CE_MSE_train_scaler_identical',read(B/'train_feature_scaler.json')==read(CE/'train_feature_scaler.json'))
  m=lines(B/'vae_train.jsonl');c=lines(CE/'vae_train.jsonl');vk=('step','record_indices','rows','forward_rng_sha256','lr')
  ck('CE_MSE_all1000_VAE_batches_forward_RNG_LR',len(m)==len(c)==1000 and all(all(a[k]==b[k] for k in vk) for a,b in zip(m,c)))
  m=lines(B/'ddpm_train.jsonl');c=lines(CE/'ddpm_train.jsonl');dk=('step','record_indices','rows','label_dropped','noise_timestep_sha256','lr')
  ck('CE_MSE_all1000_DDPM_batches_noise_timestep_dropout_LR',len(m)==len(c)==1000 and all(all(a[k]==b[k] for k in dk) for a,b in zip(m,c)))
  ck('CE_MSE_same_fresh_DDPM_initialization',read(B/'ddpm_training_summary.json')['initial_tensor_state_sha256']==read(CE/'ddpm_training_summary.json')['initial_tensor_state_sha256'])
  facts['coupling_limits']='Same initialization/data/batches/RNG/optimizer exposures are checked. Learned posteriors and outcomes are expected to differ. Actual costs differ; adaptive CE diagnostic/recovery costs remain separate.'
 section('coupling',coupling)
 def exports():
  cohorts={}
  for role,n in (('train',279),('dev',41)):
   src=Path(cfg[role+'_cohort'])
   if 'test' in str(src).lower() or sha(src)!=cfg[role+'_cohort_sha256']:raise ValueError('Refuse unbound cohort read')
   cohorts[role]=list(csv.DictReader(src.open(newline='')));rows=list(csv.DictReader((B/(role+'_reconstruction.csv')).open(newline='')))
   ck(role+'_export_exact_source_and_rendering',len(rows)==len(cohorts[role])==n and all(int(r['row_id'])==i and r['terminal']==c['terminal_state'] and r['truth_sequence']==c['sequence'] and r['status']=='ok' and len(r['prediction_sequence'])<=48 and all(x in AA for x in r['prediction_sequence']) and json.loads(r['prediction_tokens48'])==[AA.index(x) for x in r['prediction_sequence']]+[20]*(48-len(r['prediction_sequence'])) and len(json.loads(r['raw_prediction_tokens48']))==48 and all(type(x) is int and 0<=x<=20 for x in json.loads(r['raw_prediction_tokens48'])) for i,(r,c) in enumerate(zip(rows,cohorts[role]))))
  ck('train_dev_disjoint',not ({r['sequence'] for r in cohorts['train']}&{r['sequence'] for r in cohorts['dev']}))
 section('exports',exports)
 def generation():
  rows=list(csv.DictReader((B/'generated.csv').open(newline='')));gen=read(B/'generation_summary.json')
  ck('32_exact_request_id_seeds',[r['sample_id'] for r in rows]==[f'MSE21_DDPM_{t}_{i:03d}' for t in ('FREE','AMD') for i in range(16)] and all(r['method']=='MSE21_DDPM' and r['terminal']==('FREE' if i<16 else 'AMD') and int(r['seed'])==(422 if i<16 else 423) and int(r['terminal_sample_id'])==i%16 for i,r in enumerate(rows)))
  ck('32_completed_no_retry_or_filter',gen['requested']==gen['successful_execution']==32 and gen['failed_execution']==gen['not_executed']==0 and all(r['status'] in SUCCESS for r in rows) and gen['no_retry_or_filter'] and gen['condition_inputs']==['terminal class only'] and gen==status['generation'])
  ck('generation_cap',gen['elapsed_seconds']<=1800)
  progress=lines(B/'generation_progress.jsonl');ck('both_complete_to_t1',all([r['timestep'] for r in progress if r['terminal']==t]==[900,800,700,600,500,400,300,200,100,1] for t in ('FREE','AMD')))
  with np.load(B/'train_posterior_parameters.npz',allow_pickle=False) as a:values=a['values'].copy();labels=a['labels'].copy();ids=a['row_ids'].copy()
  ck('own_train_cache_schema_finite',values.shape==(279,2,16,48) and values.dtype==np.float32 and np.isfinite(values).all() and np.array_equal(ids,np.arange(279)) and np.bincount(labels,minlength=2).tolist()==[126,153])
  def train_scale(v):
   mu=v[:,0].astype(np.float64);sigma=torch.from_numpy(v[:,1].copy()).exp().numpy().astype(np.float64);q2=float(np.mean(mu*mu+sigma*sigma));mean=float(mu.mean())
   return {'records':len(v),'mu':summary(mu),'logsigma':summary(v[:,1]),'sigma':summary(sigma),'analytic_qz_mean':mean,'analytic_qz_rms':math.sqrt(q2),'analytic_qz_std':math.sqrt(max(0,q2-mean*mean))}
  scales={'pooled_train':train_scale(values),'terminal_train':{},'generated':{}}
  ck('train_sigma_strict_positive_finite',scales['pooled_train']['sigma']['elements']==scales['pooled_train']['sigma']['finite_elements'] and scales['pooled_train']['sigma']['min']>0)
  for label,t in enumerate(('FREE','AMD')):
   scales['terminal_train'][t]=train_scale(values[labels==label])
   with np.load(B/f'generated_{t}.npz',allow_pickle=False) as a:data={k:a[k].copy() for k in a.files}
   with np.load(B/f'generated_posterior_{t}.npz',allow_pickle=False) as a:posterior=a['values'].copy()
   ck(t+'_finite_arrays',all(np.isfinite(v).all() for v in data.values()) and np.isfinite(posterior).all())
   ck(t+'_shapes_and_posterior_exact',posterior.shape==(16,2,16,48) and data['z'].shape==(16,16,48) and data['raw'].shape==(16,21,48) and data['length_scores'].shape==(16,49) and data['tokens'].shape==(16,48) and np.array_equal(posterior,data['posterior_parameters']))
   mu=torch.from_numpy(posterior[:,0].copy());logsigma=torch.from_numpy(posterior[:,1].copy());sigma=logsigma.exp();ck(t+'_sigma_strict_positive_finite',bool(torch.isfinite(sigma).all()) and bool((sigma>0).all()))
   local=torch.Generator().manual_seed(422+label)
   for unused in range(998):torch.randn((16,2,16,48),generator=local)
   eps=torch.randn((16,16,48),generator=local);z=mu+eps*sigma
   ck(t+'_seed_reparameterization_bitwise',z.numpy().tobytes()==data['z'].tobytes())
   raw=torch.from_numpy(data['raw']);body,aa=raw[:,:20,:].max(1);pad=raw[:,20,:];scores=pad.sum(1,keepdim=True)+torch.cat((torch.zeros((16,1)),(body-pad).cumsum(1)),1);length=scores.argmax(1);tokens=torch.where(torch.arange(48)[None,:]<length[:,None],aa,torch.full_like(aa,20))
   ck(t+'_J_decode_exact',np.array_equal(scores.numpy(),data['length_scores']) and np.array_equal(length.numpy(),data['lengths']) and np.array_equal(tokens.numpy(),data['tokens']))
   rendered=[''.join(AA[x] for x in tok[:n]) for tok,n in zip(tokens.tolist(),length.tolist())];group=[r for r in rows if r['terminal']==t]
   ck(t+'_slot_rendering_exact',all(r['sequence']==s and int(r['length'])==len(s)==n and r['status']==('empty_sequence' if n==0 else ('ok' if n<48 else 'no_eos_at_limit')) for r,s,n in zip(group,rendered,length.tolist())))
   numeric={'mu':summary(mu.numpy()),'logsigma':summary(logsigma.numpy()),'sigma':summary(sigma.numpy()),'z':summary(data['z']),'decoder_raw':summary(data['raw'])};numeric.update(z_rms_ratio_to_terminal_train=numeric['z']['rms']/scales['terminal_train'][t]['analytic_qz_rms'],z_rms_ratio_to_pooled_train=numeric['z']['rms']/scales['pooled_train']['analytic_qz_rms'],per_request_z_rms=np.sqrt(np.mean(data['z'].astype(np.float64)**2,axis=(1,2))).tolist());scales['generated'][t]=numeric
  facts['scales']=scales;facts['slot_status_counts']=dict(collections.Counter(r['status'] for r in rows))
 section('generation',generation)
 def failure_evidence():
  facts['raw_slot_status_inventory']=dict(collections.Counter(r['status'] for r in csv.DictReader((B/'generated.csv').open(newline=''))));facts['failure_evidence']={}
  for t in ('FREE','AMD'):
   f=B/f'generation_failure_{t}.json'
   if f.exists():
    e={'record':read(f)};n=B/f'generation_failure_{t}.npz'
    if n.exists():
     with np.load(n,allow_pickle=False) as a:e['saved_tensor_summaries']={k:summary(a[k]) for k in a.files}
    facts['failure_evidence'][t]=e
 section('failure_evidence',failure_evidence)
 ck('all_MSE_artifacts_stable_during_audit',all(sha(b['path'])==b['sha256'] and Path(b['path']).stat().st_size==b['bytes'] for b in snapshot))
 ck('CE_postflight_bindings_still_unchanged',all(sha(b['path'])==b['sha256'] for b in read(O/'CE_CHECKPOINT_RECOVERY_POSTFLIGHT_v1.json')['source_bindings']))
 report={'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'status':'PASS_CHECKS_PENDING_INDEPENDENT_SCALE_ASSESSMENT' if all(c['pass'] for c in checks) else 'MSE_ENGINEERING_FINDINGS_HOLD','checks':checks,'failed_checks':[c for c in checks if not c['pass']],'section_errors':errors,'facts':facts,'source_bindings':snapshot,'audit_script_sha256':sha(__file__),'scope':'SINGLE_MSE_ENGINEERING_AND_CE_COUPLING_ONLY','original_CE_E_L_G':'NOT_EVALUABLE_INTERRUPTION','PROMOTE':'HOLD','AR_ADEQUACY':'NOT_ESTABLISHED','scalar_authorized':False,'reviewer_actions':{'model_constructions':0,'model_forwards':0,'training_updates':0,'new_generation_requests':0,'test37_reads':0,'scope':'Immutable saved arrays/logs; weights_only tensor-state loading; two exact-bound cohort CSVs for export identity; local RNG/arithmetic only.'},'limitations':['No generation quality, biology or goodness-of-fit decision. Scale values are descriptive without posthoc thresholds.','No independent998-step DDPM replay; source/hash/progress and final arithmetic checks.','This audit alone does not authorize scalar; separate manual engineering decision and exact new plan/receipt required.','No OS syscall tracing; no-test evidence is exact config/path/source/status audit.','Optimizer objects are not deserialized here; finite optimizer execution comes from frozen per-update guards and full finite logs.']}
 with dest.open('x') as f:json.dump(report,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
 print(json.dumps({'status':report['status'],'checks':len(checks),'failed_checks':report['failed_checks'],'path':str(dest)},ensure_ascii=False))
if __name__=='__main__':main()
