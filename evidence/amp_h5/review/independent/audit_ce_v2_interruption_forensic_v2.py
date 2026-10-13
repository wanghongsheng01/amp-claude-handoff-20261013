"""AMP H5 interruption forensics: saved data only, no model construction/forward."""
from pathlib import Path
import os, json, csv, math, random, hashlib, datetime, collections, pickle, types
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):os.environ[k]='1'
import numpy as np
import torch
torch.set_num_threads(1)
W=Path('/Users/loctek/Documents/Codex/2026-10-11/task-4'); H=W/'amp_h5'; B=H/'runs/formal_v2/categorical_ce'; O=H/'review/independent'
P=O/'CE_V2_INTERRUPTION_FORENSIC_v2.json'
if P.exists():raise FileExistsError('Preserve existing forensic report')
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text())
lines=lambda p:[json.loads(s) for s in Path(p).read_text().splitlines()]
checks=[]
def ck(name,ok,detail=None):
 checks.append({'name':name,'pass':bool(ok),**({'detail':detail} if detail is not None else {})})
def tensors(x):
 if torch.is_tensor(x):yield x
 elif isinstance(x,dict):
  for v in x.values():yield from tensors(v)
 elif isinstance(x,(list,tuple)):
  for v in x:yield from tensors(v)
def digest(st):
 h=hashlib.sha256()
 for name,t in sorted(st.items()):
  a=t.detach().cpu().contiguous().numpy();h.update(name.encode());h.update(str(a.dtype).encode());h.update(str(a.shape).encode());h.update(a.tobytes())
 return h.hexdigest()
def nums(x):
 if isinstance(x,(int,float)):return math.isfinite(x)
 if isinstance(x,dict):return all(nums(v) for v in x.values())
 if isinstance(x,(list,tuple)):return all(nums(v) for v in x)
 return True
def inventory():
 return [{'path':p.name,'bytes':p.stat().st_size,'sha256':sha(p)} for p in sorted(B.iterdir()) if p.is_file()]
before=inventory(); cfg=read(B/'config.json'); status=read(B/'status.json')
planpath=H/'protocol/SINGLE_CE_V2_PLAN.json';plan=read(planpath);supath=H/'runs/single_ce_execution_v2/SUPERVISOR_STATUS.json';su=read(supath);susha=sha(supath)
ck('plan_and_command_bound',sha(planpath)=='ce193e133e86d995c21e4596412ce08ee61bb2be3cb47b2549d4f11f5c86f2d6' and su['plan_sha256']==sha(planpath) and len(su['jobs'])==1 and su['jobs'][0]['argv']==plan['jobs'][0]['argv'])
ck('plan_bindings_unchanged',all(sha(x['path'])==x['sha256'] for x in plan['bindings']))
ck('config_identical_to_frozen',cfg==read(H/'protocol/formal_configs_v2/categorical_ce.json') and status['config_sha256']==sha(H/'protocol/formal_configs_v2/categorical_ce.json'))
ck('runner_hash_unchanged',status['runner_sha256']==cfg['implementation_sha256']==sha(H/'implementation/vae_pipeline_formal_v2.py'))
runtime=read(B/'runtime_and_source.json')
ck('recorded_CPU_singlethread_sources_bound',runtime['device']=='cpu' and runtime['threads']==runtime['interop_threads']==1 and all(sha(x['path'])==x['sha256'] for x in runtime['source_files']))
ck('exact_four_train_dev_input_hashes',all(sha(cfg[k])==cfg[k+'_sha256'] and 'test' not in cfg[k].lower() for k in ('train','dev','train_cohort','dev_cohort')))
ck('recorded_train_dev_contract_validated_no_test',status['test_read'] is False and all(status[k] is True for k in ('train_read_attempted','train_read_validated','development_read','development_read_attempted','development_read_validated')) and cfg['train_rows']==279 and cfg['dev_rows']==41 and cfg['train_idx_start']==0 and cfg['dev_idx_start']==279)
scaler=read(B/'train_feature_scaler.json')
ck('train_only_scaler_finite',scaler['fit_scope']=='train only' and scaler['fit_record_ids']==list(range(279)) and scaler['global']['n']==279 and nums(scaler) and all(x>0 for typ in ('global','residue') for x in scaler[typ]['scale']))
logs={s:lines(B/(s+'_train.jsonl')) for s in ('vae','ddpm')};sums={s:read(B/(s+'_training_summary.json')) for s in ('vae','ddpm')}
for s in ('vae','ddpm'):
 ck(s+'_exact1000_finite_updates',[r['step'] for r in logs[s]]==list(range(1,1001)) and nums(logs[s]) and sums[s]['updates']==sums[s]['requested_updates']==1000 and sums[s]['fixed_endpoint_reached'])
 ck(s+'_stage_cap_met',sums[s]['elapsed_seconds']<=cfg[s+'_seconds_cap'])
ck('summed_stage_training_times_within_shared_cap',sum(x['elapsed_seconds'] for x in sums.values())<cfg['training_seconds_cap'],{'scope':'Sum of stage measurements; top-level training_seconds was not flushed. Not a recovered exact total.'})
order=[];cursor=279;rng=random.Random(222);ok=[]
for r in logs['vae']:
 if cursor>=279:order=list(range(279));rng.shuffle(order);cursor=0
 idx=order[cursor:cursor+32];cursor+=len(idx);ok.append(r['record_indices']==idx and r['rows']==len(idx))
ck('vae_all_seed222_batch_order',all(ok))
ck('vae_original_lr',all(r['lr']==max(.0005*max(0,r['step']-2)/1000,1e-7) for r in logs['vae']))
ck('vae_objective_accounting',all(abs(r['loss']-(r['reconstruction']+r['kl']))<=2e-6*max(1,abs(r['reconstruction'])+abs(r['kl'])) and math.isclose(r['kl'],r['density_sum']/(r['rows']*48),rel_tol=2e-6,abs_tol=2e-6) for r in logs['vae']))
old=lines(H/'runs/smoke_v1/categorical_ce/vae_train.jsonl')
ck('vae_first20_rng_batch_match_smoke',len(old)==20 and all(all(a[k]==b[k] for k in ('step','record_indices','rows','forward_rng_sha256')) for a,b in zip(logs['vae'][:20],old)))
mature=lines(H/'feasibility/ddpm_maturity_v1/ddpm_train.jsonl')
ck('ddpm1000_same_batch_noise_drop_lr_as_frozen_maturity',len(mature)==1000 and all(all(a[k]==b[k] for k in ('step','record_indices','rows','label_dropped','noise_timestep_sha256','lr')) for a,b in zip(logs['ddpm'],mature)))
mon=lines(B/'dev_monitor.jsonl')
ck('ten_fixed_rng_preserving_dev_monitors',[r['step'] for r in mon]==list(range(100,1001,100)) and nums(mon) and all(r['rows']==41 and r['rng_preserved'] and r['checkpoint_selection'] is False for r in mon))
state_digests={}
for name in ('vae_initial_state','vae_state','ddpm_state','ddpm_ema_state'):
 st=torch.load(B/(name+'.pt'),map_location='cpu',weights_only=True);meta=read(B/(name+'.json'));state_digests[name]=digest(st)
 ck(name+'_sha_digest_all_finite',meta['sha256']==sha(B/(name+'.pt')) and meta['tensor_state_sha256']==state_digests[name] and all(bool(torch.isfinite(t).all()) for t in tensors(st)))
 if name!='vae_initial_state':ck(name+'_1000_endpoint_metadata',meta['updates']==meta['requested_updates']==1000 and meta['fixed_endpoint_reached'])
 del st
ck('vae_fresh_init_and_changed_final',state_digests['vae_initial_state']==read(H/'runs/smoke_v1/categorical_ce/vae_initialization.json')['tensor_state_sha256'] and state_digests['vae_initial_state']!=state_digests['vae_state'])
ck('ddpm_original_ema_copy_semantics',state_digests['ddpm_state']==state_digests['ddpm_ema_state'] and sums['ddpm']['ema_step']==1000)
ck('ddpm_seed322_init_matches_fixed_diagnostic',sums['ddpm']['initial_tensor_state_sha256']==read(H/'feasibility/ddpm_maturity_v1/ddpm_initial_state.json')['tensor_state_sha256'])
def inert(cls,name):
 if cls is torch.optim.lr_scheduler.OneCycleLR and name=='_annealing_cos':return 'INERT_ONECYCLE_ANNEALING_COS_REFERENCE'
 raise pickle.UnpicklingError('unexpected serialized attribute')
class Restricted(pickle.Unpickler):
 def find_class(self,module,name):
  allowed={('torch._utils','_rebuild_tensor_v2'):torch._utils._rebuild_tensor_v2,('torch','FloatStorage'):torch.FloatStorage,('torch','ByteStorage'):torch.ByteStorage,('collections','OrderedDict'):collections.OrderedDict,('__builtin__','getattr'):inert,('torch.optim.lr_scheduler','OneCycleLR'):torch.optim.lr_scheduler.OneCycleLR}
  if (module,name) not in allowed:raise pickle.UnpicklingError('unapproved '+module+'.'+name)
  return allowed[(module,name)]
restricted=types.SimpleNamespace(__name__='restricted_local_training_state',Unpickler=Restricted)
vae=torch.load(B/'vae_training_state.pt',map_location='cpu',weights_only=True)
ddpm=torch.load(B/'ddpm_training_state.pt',map_location='cpu',pickle_module=restricted,weights_only=False)
ck('saved_optimizers_finite_and_completed1000',all(st['completed_updates']==1000 and all(bool(torch.isfinite(t).all()) for t in tensors(st)) for st in (vae,ddpm)))
ck('vae_existing_optimizer_steps1000',all(float(v['step'])==1000 for v in vae['optimizer']['state'].values()))
ddpm_model=torch.load(B/'ddpm_state.pt',map_location='cpu',weights_only=True)
parameter_order=list(ddpm_model); nondropped=sum(not r['label_dropped'] for r in logs['ddpm'])
step_evidence=[{'parameter_id':k,'parameter_name':parameter_order[k],'step':float(v['step']),
               'expected_step':nondropped if parameter_order[k]=='label_emb.weight' else 1000,
               'shape_matches':tuple(v['exp_avg'].shape)==tuple(ddpm_model[parameter_order[k]].shape)}
               for k,v in ddpm['optimizer']['state'].items()]
ck('ddpm_optimizer_steps_match_actual_conditional_usage',len(ddpm_model)==len(ddpm['optimizer']['param_groups'][0]['params']) and all(r['shape_matches'] and r['step']==r['expected_step'] for r in step_evidence))
ck('ddpm_scheduler_1000',ddpm['scheduler']['last_epoch']==1000 and ddpm['ema_step']==1000)
ck('last_monitor_rng_equals_saved_vae_rng',hashlib.sha256(vae['torch_rng'].numpy().tobytes()).hexdigest()==mon[-1]['rng_sha256'])
with np.load(B/'train_posterior_parameters.npz',allow_pickle=False) as z:values=z['values'].copy();labels=z['labels'].copy();rowids=z['row_ids'].copy()
ck('posterior_train279_finite_schema',values.shape==(279,2,16,48) and values.dtype==np.float32 and np.isfinite(values).all() and np.array_equal(rowids,np.arange(279)) and np.bincount(labels,minlength=2).tolist()==[126,153])
sigma=torch.from_numpy(values[:,1].copy()).exp().numpy();mu=values[:,0].astype(np.float64)
ck('train_posterior_sigma_finite_positive',np.isfinite(sigma).all() and (sigma>0).all())
scales={}
for label,terminal in ((None,'pooled'),(0,'FREE'),(1,'AMD')):
 mask=np.ones(len(mu),dtype=bool) if label is None else labels==label
 m=mu[mask];sg=sigma[mask].astype(np.float64);ls=values[:,1][mask]
 scales[terminal]={'records':int(mask.sum()),'mu_min':float(m.min()),'mu_max':float(m.max()),'logsigma_min':float(ls.min()),'logsigma_max':float(ls.max()),'sigma_min':float(sg.min()),'sigma_max':float(sg.max()),'analytic_qz_rms':float(np.sqrt(np.mean(m*m+sg*sg)))}
rows=list(csv.DictReader((B/'generated.csv').open(newline='')));progress=lines(B/'generation_progress.jsonl')
ck('all32_original_preallocated_slots_retained',[r['sample_id'] for r in rows]==[f'CE21_DDPM_{t}_{i:03d}' for t in ('FREE','AMD') for i in range(16)] and all(r['status']=='not_executed' and r['sequence']=='' and int(r['seed'])==(422 if i<16 else 423) for i,r in enumerate(rows)))
ck('observed_incomplete_FREE_chain_only',[r['terminal'] for r in progress]==['FREE']*8 and [r['timestep'] for r in progress]==[900,800,700,600,500,400,300,200])
missing=['manifest.json','generation_summary.json','train_reconstruction.csv','dev_reconstruction.csv','generated_FREE.npz','generated_AMD.npz','generated_posterior_FREE.npz','generated_posterior_AMD.npz','generation_failure_FREE.npz','generation_failure_AMD.npz','generation_errors.jsonl']
ck('no_generated_final_or_failure_artifacts',all(not (B/n).exists() for n in missing))
ck('original_unclosed_status_retained',status['state']==su['state']==su['jobs'][0]['state']=='running' and 'exit_code' not in su['jobs'][0])
after=inventory();ck('all_original_run_files_unchanged_during_audit',before==after and sha(supath)==susha)
report={'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'status':'TRAINING_CHECKPOINTS_VERIFIED_ORIGINAL_ATTEMPT_INCOMPLETE' if all(c['pass'] for c in checks) else 'FORENSIC_FINDINGS_HOLD','checks':checks,'failed_checks':[c for c in checks if not c['pass']],
 'original_attempt_outcome':'PROCESS_ABSENT_PER_SEPARATE_READONLY_PS; STALE_RUNNING_RECORDS; NO_FINAL_GENERATION_OR_EXIT_STATUS; CAUSE_UNKNOWN',
 'process_evidence':{'tool':'exec_command ps -p 60744,60747,60887 -o pid,ppid,etime,%cpu,stat,command','reviewer_observation':'Approved read-only ps returned exit1 and only column header, no rows for these three PIDs. No signals sent. First unprivileged ps was blocked by sandbox.','parent_also_reported_absence':True},
 'audit_correction':{'preserved_previous_report':'CE_V2_INTERRUPTION_FORENSIC_v1.json','correction':'v1 wrongly required every Adam state step=1000. Conditional label_emb.weight receives gradients only on nondropped label steps; observed894 matches894 nondropped updates. This is an audit-assumption correction, not a model or protocol repair.','nondropped_condition_updates':nondropped,'ddpm_step_distribution':dict(collections.Counter(str(x['step']) for x in step_evidence)),'less_than1000_states':[x for x in step_evidence if x['step']!=1000]},
 'training_summary':sums,'training_stage_sum_seconds':sum(x['elapsed_seconds'] for x in sums.values()),'state_digests':state_digests,'train_posterior_scales':scales,
 'generation_observation':{'progress':progress,'last_recorded_elapsed_seconds':progress[-1]['elapsed_seconds'],'recorded_completed_sequences':0,'external_attempt_accounting':{'FREE':{'requests':16,'state':'partial_interrupted','minimum_reverse_steps_per_request':799},'AMD':{'requests':16,'state':'no_start_evidence'}},'retained_slots':32,'all_slots_status':'not_executed','interpretation':'CSV is preallocated and updated only after a terminal batch; it is not evidence that no generation computation occurred. FREE chain reached at least t200; unknown additional work after last flushed log.','absent_final_artifacts':missing},
 'original_status':status,'original_supervisor':su,'frozen_run_files':before,'frozen_supervisor_sha256':susha,'audit_script_sha256':sha(__file__),
 'reviewer_actions':{'model_constructions':0,'model_forwards':0,'training_updates':0,'new_generation_requests':0,'natural_CSV_parsing':0,'sealed_test37_reads':0,'saved_tensor_loading':'weights_only model/Adam or strict OneCycle metadata allowlist with inert callable; no unrestricted pickle','original_files_written':0},
 'limitations':['No terminal exit code, traceback, crash report or final sampler state exists here; transport disconnection is temporally associated, not proven causal.','Stage durations sum to a recorded training subtotal; exact top-level process and generation duration were not flushed.','No generation finite/scale claim and no original v2 engineering PASS.','No OS file-access tracing; no-test evidence is exact input hashes/path/source and recorded read flags.'],
 'recovery_authorized':False,'original_v2_PASS':False,'MSE_scalar_authorized':False,'PROMOTE':'HOLD'}
with P.open('x') as f:json.dump(report,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
print(json.dumps({'status':report['status'],'checks':len(checks),'failed_checks':report['failed_checks'],'training_stage_sum_seconds':report['training_stage_sum_seconds'],'scales':scales,'manifest_file_count':len(before),'report':str(P)},ensure_ascii=False))
