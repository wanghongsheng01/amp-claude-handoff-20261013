"""Independent saved-artifact audit. No model construction/forward or new sample."""
from pathlib import Path
import os,json,csv,hashlib,math,datetime,random,collections,pickle,types
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS'): os.environ[k]='1'
import numpy as np
import torch
torch.set_num_threads(1)
W=Path('/Users/loctek/Documents/Codex/2026-10-11/task-4');B=W/'amp_h5/feasibility/ddpm_maturity_v1';O=W/'amp_h5/review/independent'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda n:json.loads((B/n).read_text())
checks=[]
def ck(name,ok,detail=None):
 checks.append({'name':name,'pass':bool(ok),**({'detail':detail} if detail is not None else {})})
def arrsha(x):return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()
def tensor_digest(state):
 h=hashlib.sha256()
 for name,t in sorted(state.items()):
  a=t.detach().cpu().contiguous().numpy();h.update(name.encode());h.update(str(a.dtype).encode());h.update(str(a.shape).encode());h.update(a.tobytes())
 return h.hexdigest()
def tensors(x):
 if torch.is_tensor(x):yield x
 elif isinstance(x,dict):
  for y in x.values():yield from tensors(y)
 elif isinstance(x,(list,tuple)):
  for y in x:yield from tensors(y)
def summary(x):
 a=np.asarray(x,dtype=np.float64);f=a[np.isfinite(a)]
 return {'elements':int(a.size),'finite':int(np.isfinite(a).sum()),'nan':int(np.isnan(a).sum()),'positive_inf':int(np.isposinf(a).sum()),'negative_inf':int(np.isneginf(a).sum()),'min':float(f.min()) if f.size else None,'max':float(f.max()) if f.size else None,'mean':float(f.mean()) if f.size else None,'std':float(f.std()) if f.size else None,'rms':float(np.sqrt(np.mean(f*f))) if f.size else None,'q01_q50_q99':np.quantile(f,[.01,.5,.99]).tolist() if f.size else None}
status=read('status.json');cfg=read('config.json');manifest=read('manifest.json')['files']
ck('saved_manifest_hashes',all(sha(B/e['path'])==e['sha256'] for e in manifest))
ck('config_matches_preflight',sha(B/'config.json')==sha(W/'amp_h5/feasibility/DDPM_MATURITY_CONFIG.json'))
receipt=json.loads((O/'DDPM_MATURITY_PREFLIGHT_RECEIPT.json').read_text())
ck('all_preflight_bindings_unchanged',all(sha(x['path'])==x['sha256'] for x in receipt['bindings_verified']))
ck('fixed_posterior_unchanged',sha(cfg['posterior_npz'])==cfg['posterior_npz_sha256'] and status['frozen_posterior_unchanged'])
ck('recorded_scope',all(status[k]==0 for k in ('train_vae_updates','encoder_forwards','decoder_forwards','dev_test_reads')))
train=read('training_summary.json');generation=read('generation_summary.json');steps=[json.loads(x) for x in (B/'ddpm_train.jsonl').read_text().splitlines()]
ck('1000_contiguous_updates',train['fixed_endpoint_reached'] and train['updates']==1000 and [s['step'] for s in steps]==list(range(1,1001)))
ck('all_training_numeric_records_finite',all(math.isfinite(v) for s in steps for v in s.values() if isinstance(v,(int,float))))
ck('caps_met',train['training_seconds']<=1020 and generation['seconds']<=240)
g=torch.Generator().manual_seed(322);py=random.Random(322);order=[];cursor=279;order_ok=[];drop_ok=[]
for s in steps:
 if cursor>=279:order=torch.randperm(279,generator=g).tolist();cursor=0
 ids=order[cursor:cursor+8];cursor+=len(ids);order_ok.append(ids==s['record_indices'] and len(ids)==s['rows']);drop_ok.append((py.random()<.1)==s['label_dropped'])
ck('all_seed322_batch_orders',all(order_ok));ck('all_seed322_drop_decisions',all(drop_ok))
smoke=W/'amp_h5/runs/smoke_v1/categorical_ce';old_steps=[json.loads(x) for x in (smoke/'ddpm_train.jsonl').read_text().splitlines()]
ck('first20_rng_noise_and_batch_match_smoke',len(old_steps)==20 and all(all(a[k]==b[k] for k in ('step','record_indices','rows','label_dropped','noise_timestep_sha256')) for a,b in zip(steps[:20],old_steps)))
checkpoints={}
for u in (20,1000):
 cp=B/f'ddpm_ema_update_{u:04d}.pt';info=read(cp.with_suffix('.json').name);st=torch.load(cp,map_location='cpu',weights_only=True);checkpoints[u]=info
 ck(f'checkpoint{u}_hash_digest_finite',sha(cp)==info['sha256'] and tensor_digest(st)==info['tensor_state_sha256'] and all(bool(torch.isfinite(t).all()) for t in tensors(st)))
 ck(f'checkpoint{u}_metadata',info['update']==u and info['fresh_total_OneCycle_horizon']==1000)
# PyTorch2.2 OneCycle state serializes getattr(OneCycleLR, '_annealing_cos').
# Numeric audit does not need this callable; decode only the explicit known globals
# and replace that one function reference with an inert string. No unrestricted pickle.
def inert_annealing_reference(cls,name):
 if cls is torch.optim.lr_scheduler.OneCycleLR and name=='_annealing_cos':return 'INERT_ONECYCLE_ANNEALING_COS_REFERENCE'
 raise pickle.UnpicklingError('unexpected serialized attribute')
class RestrictedTrainingStateUnpickler(pickle.Unpickler):
 def find_class(self,module,name):
  allowed={('torch._utils','_rebuild_tensor_v2'):torch._utils._rebuild_tensor_v2,('torch','FloatStorage'):torch.FloatStorage,('torch','ByteStorage'):torch.ByteStorage,('collections','OrderedDict'):collections.OrderedDict,('__builtin__','getattr'):inert_annealing_reference,('torch.optim.lr_scheduler','OneCycleLR'):torch.optim.lr_scheduler.OneCycleLR}
  if (module,name) not in allowed:raise pickle.UnpicklingError('unapproved training-state global: '+module+'.'+name)
  return allowed[(module,name)]
restricted=types.SimpleNamespace(__name__='restricted_local_training_state',Unpickler=RestrictedTrainingStateUnpickler)
final_training=torch.load(B/'training_final_state.pt',map_location='cpu',pickle_module=restricted,weights_only=False)
ck('final_optimizer_and_scheduler_finite1000',final_training['completed_updates']==1000 and final_training['ema_step']==1000 and final_training['scheduler']['last_epoch']==1000 and all(bool(torch.isfinite(t).all()) for t in tensors(final_training)))
initial=torch.load(B/'ddpm_initial_state.pt',map_location='cpu',weights_only=True);ii=read('ddpm_initial_state.json');old_initial=json.loads((smoke/'ddpm_training_summary.json').read_text())['initial_tensor_state_sha256']
ck('fresh_initial_state_matches_seed322_smoke',ii['tensor_state_sha256']==tensor_digest(initial)==old_initial)
slots=list(csv.DictReader((B/'ATTEMPTS.csv').open()));expected=[f'UPDATE{u:04d}_{t}_000' for u in (20,1000) for t in ('FREE','AMD')]
ck('exact4_prespecified_attempts',[s['attempt_id'] for s in slots]==expected and all(int(s['seed'])==(422 if s['terminal']=='FREE' else 423) and int(s['checkpoint_update']) in (20,1000) for s in slots))
ck('slots_match_summary',all(all(str(generation['attempts'][i][k])==s[k] for k in s) for i,s in enumerate(slots)))
ck('diagnostic_execution_complete',status['state'] in ('COMPLETE','COMPLETE_WITH_NUMERICAL_FAILURES') and generation['not_executed_count']==generation['execution_failure_count']==0)
ck('failure_counts_reconcile',generation['numerical_success_count']==sum(s['status']=='finite_latent_no_decoder' for s in slots) and generation['numerical_failure_count']==sum(s['status']=='error:FloatingPointError' for s in slots))
ck('sampling_global_rng_unchanged',generation['global_rng_unchanged'])
noise_records=[json.loads(x) for x in (B/'generation_noise_hashes.jsonl').read_text().splitlines()];grouped=collections.defaultdict(list)
for r in noise_records:grouped[r['attempt_id']].append(r)
noise_banks={}
for t,seed in [('FREE',422),('AMD',423)]:
 with np.load(B/f'frozen_noise_{t}.npz',allow_pickle=False) as a:bank={k:a[k].copy() for k in a.files}
 noise_banks[t]=bank;meta=read(f'frozen_noise_{t}.json');gen=torch.Generator().manual_seed(seed)
 init=torch.randn((1,2,16,48),generator=gen).numpy();rev=torch.stack([torch.randn((1,2,16,48),generator=gen) for _ in range(997)]).numpy();eps=torch.randn((1,16,48),generator=gen).numpy()
 ck(f'{t}_bank_reproduced_seed_exact',init.tobytes()==bank['initial'].tobytes() and rev.tobytes()==bank['reverse_noise_t998_to2'].tobytes() and eps.tobytes()==bank['posterior_epsilon'].tobytes())
 ck(f'{t}_bank_hashes',meta['file_sha256']==sha(B/f'frozen_noise_{t}.npz') and arrsha(init)==meta['initial_sha256'] and arrsha(rev)==meta['reverse_noise_sha256'] and arrsha(eps)==meta['epsilon_sha256'])
 for u in (20,1000):
  name=f'UPDATE{u:04d}_{t}_000';rr=grouped[name]
  expected_ts=list(reversed(range(1,999)))
  failure_path=B/f'{name}_failure.json'
  if failure_path.exists():
   failure=json.loads(failure_path.read_text())
   if failure['phase']=='reverse_ddpm':expected_ts=list(reversed(range(failure['timestep']+1,999)))
  ck(f'{name}_noise_record_count_matches_reached_phase',[r['timestep'] for r in rr]==expected_ts)
  ck(f'{name}_noise_hashes_match_shared_bank',all(r['noise_sha256']==arrsha(rev[998-r['timestep']] if r['timestep']>1 else np.zeros((1,2,16,48),np.float32)) for r in rr))
 twenty={x['timestep']:x['noise_sha256'] for x in grouped[f'UPDATE0020_{t}_000']};thousand={x['timestep']:x['noise_sha256'] for x in grouped[f'UPDATE1000_{t}_000']}
 ck(f'{t}_checkpoint_used_noise_pairing_on_shared_timesteps',all(twenty[k]==thousand[k] for k in twenty.keys()&thousand.keys()))
with np.load(cfg['posterior_npz'],allow_pickle=False) as a:values=a['values'].copy();labels=a['labels'].copy();rowids=a['row_ids'].copy()
ck('fixed_train_cache_identity_schema',values.shape==(279,2,16,48) and values.dtype==np.float32 and labels.shape==(279,) and np.array_equal(rowids,np.arange(279)) and np.bincount(labels,minlength=2).tolist()==[126,153])
mu=values[:,0].astype(np.float64);sigma=torch.from_numpy(values[:,1]).exp().numpy().astype(np.float64)
target={'mu':summary(values[:,0]),'logsigma':summary(values[:,1]),'sigma':summary(sigma),'analytic_z_mean':float(mu.mean()),'analytic_z_rms':float(np.sqrt(np.mean(mu*mu+sigma*sigma))),'analytic_z_std':float(np.sqrt(np.mean(mu*mu+sigma*sigma)-mu.mean()**2))}
scales={}
for s in slots:
 name=s['attempt_id'];t=s['terminal']
 with np.load(B/f'{name}_available_tensors.npz',allow_pickle=False) as a:arrays={k:a[k].copy() for k in a.files}
 ck(name+'_bank_identity',arrays['initial'].tobytes()==noise_banks[t]['initial'].tobytes() and arrays['posterior_epsilon'].tobytes()==noise_banks[t]['posterior_epsilon'].tobytes())
 if all(k in arrays for k in ('mu','logsigma','sigma','latent')):
  logs=torch.from_numpy(arrays['logsigma']);tm=torch.from_numpy(arrays['mu']);ep=torch.from_numpy(arrays['posterior_epsilon']);sg=logs.exp();zz=tm+ep*sg
  ck(name+'_saved_reparam_exact',sg.numpy().tobytes()==arrays['sigma'].tobytes() and zz.numpy().tobytes()==arrays['latent'].tobytes())
  finite=all(np.isfinite(arrays[k]).all() for k in ('mu','logsigma','sigma','latent')) and bool((arrays['sigma']>0).all())
  ck(name+'_finite_gate_matches_slot',finite==(s['status']=='finite_latent_no_decoder'))
  scales[name]={k:summary(arrays[k]) for k in ('mu','logsigma','sigma','latent')};scales[name]['sigma_zero_count']=int((arrays['sigma']==0).sum())
  scales[name]['ratios_to_train']={'mu_rms':scales[name]['mu']['rms']/target['mu']['rms'],'logsigma_rms':scales[name]['logsigma']['rms']/target['logsigma']['rms'],'sigma_finite_subset_rms':scales[name]['sigma']['rms']/target['sigma']['rms'] if scales[name]['sigma']['rms'] is not None else None,'latent_finite_subset_rms_vs_analytic_train':scales[name]['latent']['rms']/target['analytic_z_rms'] if scales[name]['latent']['rms'] is not None else None}
 else:scales[name]={'no_complete_posterior':'failure occurred before full reverse chain'}
supervisor=json.loads((W/'amp_h5/runs/ddpm_maturity_execution_v1/SUPERVISOR_STATUS.json').read_text())
ck('supervisor_bound_plan_completed',supervisor['plan_sha256']==receipt['plan_sha256'] and supervisor['jobs'][0]['exit_code']==0)
finite1000=all(s['status']=='finite_latent_no_decoder' for s in slots if s['checkpoint_update']=='1000')
report={'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'status':'PASS_RECORDED_DIAGNOSTIC_EXECUTION_AUDIT' if all(x['pass'] for x in checks) else 'REVIEW_FINDINGS','checks':checks,'failed_checks':[x for x in checks if not x['pass']],'status_sha256':sha(B/'status.json'),'manifest_sha256':sha(B/'manifest.json'),'train_updates':train['updates'],'recorded_execution_state':status['state'],'attempts':slots,'both1000_endpoints_finite':finite1000,'target_scale_independent':target,'generated_scales_independent':scales,'scale_review':'PENDING_HUMAN_READABLE_INDEPENDENT_JUDGMENT; no automatic numerical threshold or authorization','single_CE_authorized':False,'full_VAE_formal_authorized':False,'PROMOTE':'HOLD','reviewer_actions':{'training_state_loader':'Explicit global allowlist; OneCycle annealing callable represented by inert string; no unrestricted pickle','model_constructions':0,'model_forwards':0,'training_updates':0,'new_generation_requests':0,'natural_CSV_reads':0,'test_reads':0,'actions':'Inspect saved tensors, recompute numeric summaries and reproduce noise banks without model calls'},'limitations':['One fixedCE20 representation/seed; total1000 schedule20 endpoint is not originaltotal20 smoke.','Any numeric failure remains a result; failed/missing reverse-step telemetry needs scope-specific interpretation.','Finite-subset statistics on arrays containingInf/NaN are not whole-array moments and must not be interpreted as numerical success.','Extra adaptive diagnostic cost is separate from method comparison.']}
p=O/'DDPM_MATURITY_POSTFLIGHT_REVIEW.json';p.write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
print(json.dumps({'status':report['status'],'checks':len(checks),'failed_checks':report['failed_checks'],'both1000_endpoints_finite':finite1000,'attempts':slots,'scale_summary':{k:{'mu':v.get('mu'),'logsigma':v.get('logsigma'),'sigma':v.get('sigma'),'latent':v.get('latent'),'ratios_to_train':v.get('ratios_to_train')} for k,v in scales.items()}},allow_nan=False))
