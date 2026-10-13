import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[k]='1'
from pathlib import Path
import json,hashlib,time,datetime,signal,resource,traceback,sys
import numpy as np
from model import weights,features,scale_fit,design,fit,metrics
R=Path(__file__).resolve().parent

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,d):p.write_text(json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def main():
 freeze=json.loads((R/'FREEZE.json').read_text())
 for name,h in freeze['local'].items():assert sha(R/name)==h,name
 for name,h in freeze['sources'].items():assert sha(R.parent/name)==h,name
 for name in ['SCIENCE_PREAUDIT.json','ENGINEERING_PREAUDIT.json']:
  review=json.loads((R/'independent'/name).read_text());assert review['status']=='PASS';assert review['freeze_sha256']==sha(R/'FREEZE.json')
 out=R/'run';out.mkdir(exist_ok=False);t=time.monotonic();np.random.seed(1729)
 dump(out/'STARTED.json',{'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'pid':os.getpid(),'freeze_sha256':sha(R/'FREEZE.json'),'seed':1729,'threads':{k:os.environ[k]for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS']}})
 def deadline(*args):raise TimeoutError('Total600seconds')
 signal.signal(signal.SIGALRM,deadline);signal.alarm(600);pred=[];completed=[]
 def guard():
  rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin' else 1024)
  if rss>1500000000:raise MemoryError('RSS limit')
  if time.monotonic()-t>=600:raise TimeoutError('Total600seconds')
  return rss
 try:
  rs=[json.loads(l)for l in (R/'DATA.jsonl').read_text().splitlines()];lookup={r['activity_id']:r for r in rs};folds=json.loads((R/'FOLDS.json').read_text());arms=['point_mean','point_ridge','endpoint_ridge','tail_ridge']
  with (out/'PREDICTIONS.jsonl').open('x')as fh:
   for unit,fs in folds.items():
    for f in fs:
     guard();tr=[lookup[i]for i in f['train']];va=[lookup[i]for i in f['validation']];tp=[r for r in tr if r['label_kind']=='point'];w=weights(tr);wp=weights(tp)
     for key in ['peptide_id','sequence','annotation_fingerprint','sequence_family','source_family_group']:
      assert not ({r[key]for r in tr}&{r[key]for r in va})
     assert not ({s for r in tr for s in r['source_keys']}&{s for r in va for s in r['source_keys']})
     mean,scale=scale_fit(features(tr),w);z=design(features(tr),mean,scale);zp=design(features(tp),mean,scale);zv=design(features(va),mean,scale);y=np.array([r['target_log10']for r in tr]);yp=np.array([r['target_log10']for r in tp]);k=np.array([r['label_kind']for r in tr]);kp=np.array(['point']*len(tp));center=float(wp@yp);start=np.zeros(22);start[0]=center
     states={};diagnostics={};values={'point_mean':np.full(len(va),center)}
     for arm,zz,yy,ww,kk,tail in [('point_ridge',zp,yp,wp,kp,False),('endpoint_ridge',z,y,w,k,False),('tail_ridge',z,y,w,k,True)]:
      guard();states[arm],diagnostics[arm]=fit(zz,yy,ww,kk,tail,start);values[arm]=zv@states[arm]
     assert all(np.isfinite(v).all()for v in values.values());assert all(r['label_kind']=='point'for r in va)
     receipt={'unit':unit,'fold':f,'train_weights_all':w.tolist(),'train_point_activity_ids':[r['activity_id']for r in tp],'train_weights_point':wp.tolist(),'mean':mean.tolist(),'scale':scale.tolist(),'start':start.tolist(),'states':{k:v.tolist()for k,v in states.items()},'fit_diagnostics':diagnostics,'point_mean':center}
     dump(out/(unit+'__'+f['group']+'.json'),receipt)
     for arm in arms:
      for r,p in zip(va,values[arm]):
       row={key:r[key]for key in ['activity_id','peptide_id','sequence_family','source_family_group','native_unit_group','concentration_raw','target_log10','label_kind']};row.update(model=arm,prediction_log10=float(p));pred.append(row);fh.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')
     fh.flush();os.fsync(fh.fileno());completed.append(unit+'__'+f['group']);print(completed[-1],flush=True)
  result={'status':'COMPLETE_DEVELOPMENT_ONLY','seed':1729,'elapsed_seconds':time.monotonic()-t,'peak_rss_bytes':guard(),'completed_folds':completed,'units':{},'fold_metrics':{},'no_test37':True,'no_tuning':True}
  for u in folds:
   result['units'][u]={a:metrics([r for r in pred if r['native_unit_group']==u and r['model']==a])for a in arms}
   result['fold_metrics'][u]={f['group']:{a:metrics([r for r in pred if r['native_unit_group']==u and r['source_family_group']==f['group'] and r['model']==a])for a in arms}for f in folds[u]}
  assert len(pred)==4*147
  dump(out/'RESULT.json',result);print(json.dumps({'status':result['status'],'seconds':result['elapsed_seconds'],'units':result['units']}),flush=True)
 except BaseException as e:
  dump(out/'FAILURE.json',{'error':repr(e),'traceback':traceback.format_exc(),'completed':completed,'elapsed_seconds':time.monotonic()-t});raise
 finally:signal.alarm(0)
if __name__=='__main__':main()
