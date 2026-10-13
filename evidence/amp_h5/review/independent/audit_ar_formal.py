from pathlib import Path
import os,json,csv,hashlib,math,datetime,collections
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):os.environ[k]='1'
import torch
torch.set_num_threads(1)
W=Path('/Users/loctek/Documents/Codex/2026-10-11/task-4'); B=W/'amp_h5/runs/formal_v1/terminal_lstm'; O=W/'amp_h5/review/independent'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda n:json.loads((B/n).read_text())
checks=[]
def ck(name,ok):
 checks.append({'name':name,'pass':bool(ok)})
 if not ok: raise AssertionError(name)
status=read('status.json');run=read('training/RUN.json'); cfg=read('config.json');gen=read('generation/GENERATION.json');mon=read('training/MONITORS.json')
manifest=read('manifest.json')['files']
ck('all_manifest_files_hash_and_size',all(sha(B/r['path'])==r['sha256'] and (B/r['path']).stat().st_size==r['bytes'] for r in manifest))
ck('config_matches_frozen',sha(B/'config.json')==sha(W/'amp_h5/protocol/formal_configs/terminal_lstm.json'))
ck('bound_implementation_unchanged',cfg['implementation_sha256']==sha(W/'amp_h5/implementation/ar_pipeline.py') and cfg['entry_sha256']==sha(W/'amp_h5/implementation/ar_entry.py'))
ck('recorded_completed_fixed1000',status['state']=='completed' and run['status']=='COMPLETED_FIXED_STEPS' and run['completed_steps']==run['attempted_steps']==1000 and run['retries']==0 and run['failure'] is None)
steps=[json.loads(x) for x in (B/'training/steps.jsonl').read_text().splitlines()]
ck('exact1000_contiguous_step_records',[r['step'] for r in steps]==list(range(1,1001)))
ck('all_step_numeric_finite',all(math.isfinite(v) for r in steps for v in r.values() if isinstance(v,(int,float))))
ck('loss_times_token_count_matches_nll',all(math.isclose(r['loss']*r['observed_tokens'],r['nll_sum'],rel_tol=2e-7,abs_tol=1e-4) and r['observed_tokens']==r['aa_tokens']+r['eos_tokens'] for r in steps))
g=torch.Generator().manual_seed(222); expected=[];epoch=-1;order=[];cursor=279
for r in steps:
 if cursor>=279: order=torch.randperm(279,generator=g).tolist();cursor=0;epoch+=1
 idx=order[cursor:cursor+32];cursor+=len(idx)
 expected.append(r['row_indices']==idx and r['examples']==len(idx) and r['epoch_zero_based']==epoch)
ck('seed222_exact_batch_order_all_steps',all(expected))
ck('aggregate_exposure_matches_steps',all(run[k]==sum(r[v] for r in steps) for k,v in [('examples_presented','examples'),('observed_tokens_presented','observed_tokens'),('aa_tokens_presented','aa_tokens'),('eos_tokens_presented','eos_tokens')]))
ck('fixed_monitor100_to1000',[r['step'] for r in mon]==list(range(100,1001,100)) and all(r['used_for_selection'] is False for r in mon))
ck('monitor_same_model_train_dev',all(r['train']['tensor_state_sha256']==r['development']['tensor_state_sha256'] and r['train']['objects']==279 and r['development']['objects']==41 for r in mon))
ck('all_monitor_metrics_finite',all(math.isfinite(v) for r in mon for role in ('train','development') for v in r[role].values() if isinstance(v,(int,float))))
final=torch.load(B/'training/final.pt',map_location='cpu',weights_only=True);initial=torch.load(B/'training/initial.pt',map_location='cpu',weights_only=True)
def tensors(v):
 if torch.is_tensor(v):yield v
 elif isinstance(v,dict):
  for x in v.values():yield from tensors(x)
 elif isinstance(v,(list,tuple)):
  for x in v:yield from tensors(x)
ck('all_saved_initial_final_model_optimizer_tensors_finite',all(bool(torch.isfinite(t).all()) for x in (initial,final) for t in tensors(x)))
ck('checkpoint_and_adam_step1000',initial['updates']==0 and final['updates']==1000 and all(float(s['step'])==1000 for s in final['optimizer']['state'].values()))
h=hashlib.sha256()
for key,t in sorted(final['model'].items()):
 t=t.detach().cpu().contiguous();h.update(key.encode()+b'\0');h.update(str(t.dtype).encode()+b'\0');h.update(str(tuple(t.shape)).encode()+b'\0');h.update(t.numpy().tobytes())
ck('final_state_digest_linked_everywhere',h.hexdigest()==final['tensor_state_sha256']==run['final_tensor_state_sha256']==gen['tensor_state_sha256']==mon[-1]['train']['tensor_state_sha256'])
rows=list(csv.DictReader((B/'generated.csv').open())); archived=[json.loads(x) for x in (B/'generation/samples.jsonl').read_text().splitlines()]
ck('exact32_ids_seeds_and_complete_slots',[r['sample_id'] for r in rows]==[f'terminal_LSTM_{t}_{i:03d}' for t in ('FREE','AMD') for i in range(16)] and all(int(r['seed'])==(422 if r['terminal']=='FREE' else 423) and int(r['terminal_sample_id'])==i%16 and r['status'] in ('ok','empty_sequence') for i,r in enumerate(rows)))
ck('csv_matches_archived_sequence_status',all(r['sequence']==a['sequence'] and r['status']==a['status'] and int(r['length'])==a['length']==len(a['sequence']) and a['complete_attempt'] for r,a in zip(rows,archived)))
raw=torch.load(B/'generation/raw_sampling.pt',map_location='cpu',weights_only=True)
ck('all_raw_sampling_tensors_finite',all(bool(torch.isfinite(t).all()) for t in tensors(raw)))
ck('raw_shape_ids_and_seeds',list(raw['tokens'].shape)==[32,48] and list(raw['logits'].shape)==[32,48,21] and raw['sample_ids']==[r['sample_id'] for r in rows] and raw['seeds']=={'FREE':422,'AMD':423})
AA='ARNDCEQGHILKMFPSTWYV';actual=[]
for ts in raw['tokens'].tolist():
 s=''
 for x in ts:
  if x==20:break
  s+=AA[x]
 actual.append(s)
ck('rendered_raw_tokens_match_all32',actual==[r['sequence'] for r in rows])
ck('probabilities_normalized_on_evaluated_steps',all(torch.allclose(raw['probabilities'][i*16:(i+1)*16,:s].sum(-1),torch.ones(16,s),atol=2e-7,rtol=2e-7) for i,s in enumerate(raw['evaluated_steps'])))
ck('raw_observed_mask_total',int(raw['active_before'].sum())==gen['observed_sampling_tokens'])
ck('no_retry_or_cap',all(v is None for v in gen['failures_by_terminal'].values()) and not any(gen['capped_by_terminal'].values()) and gen['no_rejection_or_resampling'] and run['cap_overrun_seconds']==0)
ck('only_bound_train_dev_recorded',run['train_objects']==279 and run['train_path']==cfg['train'] and run['train_sha256']==cfg['train_sha256'] and run['development_path']==cfg['dev'] and run['development_sha256']==cfg['dev_sha256'] and status['test_read'] is False and run['test_accessed'] is False and gen['test_accessed'] is False and not gen['source_sequence_accessed'] and not gen['true_length_accessed'])
nll={r:read(r+'_nll.json') for r in ('train','dev')}
ck('final_nll_matches_monitor1000',all(nll[k]['nll_sum']==mon[-1][v]['nll_sum'] and nll[k]['tensor_state_sha256']==h.hexdigest() for k,v in [('train','train'),('dev','development')]))
supervisor=json.loads((W/'amp_h5/runs/ar_formal_execution_v1/SUPERVISOR_STATUS.json').read_text())
ck('supervisor_exact_plan_completed',supervisor['plan_sha256']=='f873246b1e3c434dbd64058e85e49d62adba691fc9785ad58a9ee92f65c946c6' and len(supervisor['jobs'])==1 and supervisor['jobs'][0]['exit_code']==0)
best=min(mon,key=lambda r:r['development']['mean_nll'])
report={'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'status':'PASS_AR_EXECUTION_POSTFLIGHT','checks':checks,'manifest_files':len(manifest),'output_manifest_sha256':sha(B/'manifest.json'),'implementation_sha256':sha(W/'amp_h5/implementation/ar_pipeline.py'),'config_sha256':sha(B/'config.json'),'supervisor_sha256':sha(W/'amp_h5/runs/ar_formal_execution_v1/SUPERVISOR_STATUS.json'),'update_count':len(steps),'requested_slots':32,'terminal_counts':dict(collections.Counter(r['terminal'] for r in rows)),'slot_status_counts':dict(collections.Counter(r['status'] for r in rows)),'valid1to48':sum(1<=int(r['length'])<=48 for r in rows),'valid5to48':sum(5<=int(r['length'])<=48 for r in rows),'monitor_curve':[{'step':r['step'],'train_nll':r['train']['mean_nll'],'dev_nll':r['development']['mean_nll']} for r in mon],'development_curve_descriptive':{'smallest_recorded_step':best['step'],'smallest_recorded_mean_nll':best['development']['mean_nll'],'final_train_mean_nll':nll['train']['mean_nll'],'final_development_mean_nll':nll['dev']['mean_nll'],'interpretation':'Train NLL falls while development NLL worsens after step200, evidence of overfitting in this fixed recipe. Final1000 was predeclared and remains the comparison endpoint; no retrospective checkpoint swap authorized.'},'AR_ADEQUACY':'NOT_ESTABLISHED','PROMOTE':'HOLD','no_test_evidence_scope':'Exact command/config and bound source use train279/dev41 only; recorded status confirms no test. This is source/output audit, not OS-level file-access tracing. Reviewer opened no natural split.','reviewer_actions':{'model_instantiations':0,'model_forwards':0,'training_updates':0,'new_sampling':0,'natural_split_reads':0,'test_reads':0,'saved_local_tensor_files_inspected':['initial.pt','final.pt','raw_sampling.pt']},'remaining_boundaries':['Single seed and one fixed recipe; no strongest-modern-baseline or tuning adequacy claim.','Teacher-forced NLL is not generation quality or directly comparable to VAE objective.','32 slots are execution evidence; no comparative distributional screen has been run by this reviewer.','Three VAE long runs remain HOLD.']}
(O/'AR_FORMAL_POSTFLIGHT_REVIEW.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
print(json.dumps({k:report[k] for k in ['status','update_count','requested_slots','slot_status_counts','valid1to48','valid5to48','development_curve_descriptive','AR_ADEQUACY','PROMOTE']}));print('checks',len(checks))
