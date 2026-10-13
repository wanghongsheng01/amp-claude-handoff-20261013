"""One bounded native-label development run. No protected data or validation-driven fitting."""
import os
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[key]='1'
import collections,csv,datetime,hashlib,json,math,resource,signal,sys,time,traceback
from pathlib import Path
import numpy as np
import torch
from model import WittenCNN,KerasStyleAdam,encode,ALPHABET
R=Path(__file__).resolve().parent

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,d):p.write_text(json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def rows(p):return [json.loads(s) for s in p.read_text().splitlines()]
def weights(rs):
 groups=collections.defaultdict(lambda:collections.Counter())
 for r in rs:groups[r['source_family_group']][r['peptide_id']]+=1
 return np.array([1/(len(groups)*len(groups[r['source_family_group']])*groups[r['source_family_group']][r['peptide_id']]) for r in rs],dtype=np.float64)
def features(rs):
 return np.array([[r['sequence'].count(a)/len(r['sequence']) for a in ALPHABET]+[math.log(len(r['sequence']))] for r in rs],dtype=np.float64)
def ridge_fit(x,y,w,alpha):
 mu=w@x;scale=np.sqrt(w@((x-mu)**2));scale=np.where(scale>1e-12,scale,1.0);z=(x-mu)/scale;intercept=float(w@y)
 beta=np.linalg.solve(z.T@(w[:,None]*z)+alpha*np.eye(x.shape[1]),z.T@(w*(y-intercept)))
 return {'mu':mu,'scale':scale,'beta':beta,'intercept':intercept}
def ridge_predict(state,x):return state['intercept']+(x-state['mu'])/state['scale']@state['beta']
def metrics(rs):
 y=np.array([r['target_log10'] for r in rs]);p=np.array([r['prediction_log10'] for r in rs]);e=p-y;w=weights(rs)
 bypid=collections.Counter(r['peptide_id'] for r in rs);pw=np.array([1/(len(bypid)*bypid[r['peptide_id']]) for r in rs])
 return {'rows':len(rs),'peptide_IDs':len(bypid),'families':len({r['sequence_family']for r in rs}),'groups':len({r['source_family_group']for r in rs}),'weight_sum':float(w.sum()),'group_peptide_macro_MAE':float(w@np.abs(e)),'group_peptide_macro_RMSE':float(np.sqrt(w@(e*e))),'peptide_macro_MAE':float(pw@np.abs(e)),'peptide_macro_RMSE':float(np.sqrt(pw@(e*e))),'row_MAE':float(np.mean(np.abs(e))),'row_RMSE':float(np.sqrt(np.mean(e*e)))}
def main():
 cfg=json.loads((R/'PROTOCOL.json').read_text());freeze=json.loads((R/'FREEZE.json').read_text());review=json.loads((R/'independent/SIGNED_PREAUDIT.json').read_text())
 assert review['status']=='PASS','Independent engineering preaudit missing'
 assert review['freeze_sha256']==sha(R/'FREEZE.json'),'Review is not bound to this freeze'
 for f,h in freeze['local_sha256'].items():assert sha(R/f)==h,f
 for f,h in freeze['source_sha256'].items():assert sha(R.parent/f)==h,f
 out=R/'run';out.mkdir(exist_ok=False);dump(out/'STARTED.json',{'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'pid':os.getpid(),'protocol_sha256':sha(R/'PROTOCOL.json'),'freeze_sha256':sha(R/'FREEZE.json'),'preaudit_sha256':sha(R/'independent/SIGNED_PREAUDIT.json')})
 started=time.monotonic();steps=0;completed=[];preds=[];os.nice(cfg['limits']['process_nice_increment']);torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
 def limit(*args):raise TimeoutError('Frozen600s total run deadline')
 signal.signal(signal.SIGALRM,limit);signal.alarm(cfg['limits']['wall_seconds'])
 dump(out/'EXECUTION_ENV.json',{'python':sys.version,'torch':torch.__version__,'numpy':np.__version__,'torch_threads':torch.get_num_threads(),'torch_interop_threads':torch.get_num_interop_threads(),'BLAS_env':{k:os.environ[k]for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS']},'device':'cpu','nice':os.getpriority(os.PRIO_PROCESS,0),'PID':os.getpid()})
 assert torch.get_num_threads()==torch.get_num_interop_threads()==1
 def resources():
  rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
  if sys.platform!='darwin':rss*=1024
  if rss>cfg['limits']['peak_rss_bytes']:raise MemoryError('Frozen RSS limit')
  if time.monotonic()-started>=cfg['limits']['wall_seconds']:raise TimeoutError('Frozen wall limit')
  if steps>cfg['limits']['maximum_CNN_updates']:raise RuntimeError('Frozen step limit')
  return rss
 try:
  data=rows(R/'DATA.jsonl');fm=json.loads((R/'FOLDS.json').read_text());lookup={r['activity_id']:r for r in data};assert len(lookup)==len(data)==85
  with (out/'PREDICTIONS.jsonl').open('x') as handle:
   for unit in cfg['units_separate']:
    for fold in fm['units'][unit]['folds']:
     foldstart=time.monotonic();fid=fold['fold_id'];train=[lookup[i] for i in fold['train_activity_ids']];val=[lookup[i] for i in fold['validation_activity_ids']]
     assert all(r['native_unit_group']==unit and r['plain_Witten_primary_point_eligible'] for r in train+val)
     for key in ['peptide_id','sequence','annotation_fingerprint','sequence_family','source_family_group']:
      assert not ({r[key]for r in train}&{r[key]for r in val}),(unit,fid,key)
     assert not ({s for r in train for s in r['source_keys']}&{s for r in val for s in r['source_keys']})
     w=weights(train);assert abs(w.sum()-1)<1e-12
     y=np.log10([float(r['lower_raw_unit'])for r in train]);center=float(w@y)
     ridge=ridge_fit(features(train),y,w,cfg['ridge']['alpha']);ridgepred=ridge_predict(ridge,features(val))
     torch.manual_seed(cfg['seed']);np.random.seed(cfg['seed']);model=WittenCNN().cpu();assert sum(p.numel()for p in model.parameters())==cfg['cnn']['parameters']
     opt=KerasStyleAdam(model.parameters());x=torch.from_numpy(encode([r['sequence']for r in train]));target=torch.tensor(y-center,dtype=torch.float32).reshape(-1,1);wt=torch.tensor(w,dtype=torch.float32).reshape(-1,1)
     model.train();trace=[]
     for epoch in range(cfg['cnn']['epochs']):
      resources();opt.zero_grad(set_to_none=True);p=model(x);loss=torch.sum(wt*(p-target)**2)
      if not torch.isfinite(loss):raise ValueError('Nonfinite loss')
      loss.backward()
      if any(q.grad is None or not torch.isfinite(q.grad).all() for q in model.parameters()):raise ValueError('Invalid gradient')
      opt.step();steps+=1
      trace.append(float(loss.detach()))
     # Heldout features evaluated only after fixed training endpoint; no validation-based changes.
     model.eval()
     with torch.no_grad():cnnpred=model(torch.from_numpy(encode([r['sequence']for r in val]))).reshape(-1).numpy().astype(float)+center
     predictions={'training_weighted_mean':np.full(len(val),center),'composition_ridge':ridgepred,'witten_CNN':cnnpred}
     assert all(np.isfinite(p).all() for p in predictions.values())
     weights_val=weights(val)
     for name,values in predictions.items():
      for i,(r,pr) in enumerate(zip(val,values)):
       rr={k:r[k] for k in ['activity_id','peptide_id','sequence','annotation_fingerprint','sequence_family','source_family_group','source_keys','concentration_raw','unit_raw','operator_or_semantics','format','target_raw','medium','cfu','cfu_group','ph','ionic_strength','salt','both_termini_observed','modification_annotated','structure_status','raw_sha256']}
       rr.update(native_unit_group=unit,fold_id=fid,model=name,target_log10=math.log10(float(r['lower_raw_unit'])),prediction_log10=float(pr),native_reported_point=float(r['lower_raw_unit']),within_fold_weight=float(weights_val[i]),unit_primary_weight=float(weights_val[i]/fm['units'][unit]['groups']),exposure='OOF_from_already_exposed_development_cache_not_independent_confirmation')
       handle.write(json.dumps(rr,ensure_ascii=False,allow_nan=False)+'\n');preds.append(rr)
     handle.flush();os.fsync(handle.fileno())
     tag=unit+'__'+fid
     torch.save({'state_dict':model.state_dict(),'training_label_center':center,'protocol_sha256':sha(R/'PROTOCOL.json'),'purpose':'development_only_do_not_reuse_in_protected_H5_evaluation'},out/(tag+'.pt'))
     receipt={'unit':unit,'fold':fold,'seed':cfg['seed'],'training_activity_ids':fold['train_activity_ids'],'training_weights':w.tolist(),'training_label_center':center,'ridge':{k:v.tolist()if isinstance(v,np.ndarray)else v for k,v in ridge.items()},'CNN_updates':cfg['cnn']['epochs'],'CNN_training_loss':trace,'validation_used_for_endpoint':False,'checkpoint_sha256':sha(out/(tag+'.pt')),'elapsed_seconds':time.monotonic()-foldstart,'peak_rss_bytes':resources()}
     dump(out/(tag+'.json'),receipt);completed.append(tag);print(json.dumps({'completed_fold':tag,'train_rows':len(train),'val_rows':len(val),'updates':steps,'elapsed_seconds':time.monotonic()-started}),flush=True)
     del model,opt,x,target,wt
  assert len(preds)==255 and steps==1700
  summary={'status':'COMPLETE_DEVELOPMENT_ONLY','units':{},'fold_metrics':[],'total_CNN_updates':steps,'completed_folds':completed,'elapsed_seconds':time.monotonic()-started,'peak_rss_bytes':resources(),'seed':cfg['seed'],'test37_access':False,'no_tuning':True,'claim':'Finite native-unit pipeline and baseline diagnostic; no novelty/biological efficacy/independent confirmation claim'}
  for unit in cfg['units_separate']:
   summary['units'][unit]={name:metrics([p for p in preds if p['native_unit_group']==unit and p['model']==name])for name in cfg['models']}
   for f in fm['units'][unit]['folds']:
    summary['fold_metrics'].append({'unit':unit,'fold_id':f['fold_id'],'metrics':{name:metrics([p for p in preds if p['native_unit_group']==unit and p['fold_id']==f['fold_id'] and p['model']==name])for name in cfg['models']}})
  dump(out/'RESULT.json',summary)
  with (out/'PREDICTIONS.csv').open('w',newline='')as f:
   writer=csv.DictWriter(f,fieldnames=list(preds[0]));writer.writeheader()
   for p in preds:writer.writerow({k:json.dumps(v,ensure_ascii=False)if isinstance(v,(dict,list))else v for k,v in p.items()})
  print(json.dumps({'status':summary['status'],'elapsed_seconds':summary['elapsed_seconds'],'units':summary['units']},ensure_ascii=False),flush=True)
 except BaseException as exc:
  dump(out/'FAILURE.json',{'status':'FAILED_OR_PARTIAL_PRESERVED','error':repr(exc),'traceback':traceback.format_exc(),'completed_folds':completed,'steps':steps,'elapsed_seconds':time.monotonic()-started});raise
 finally:signal.alarm(0)
if __name__=='__main__':main()
