import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[k]='1'
from pathlib import Path
import json,hashlib,collections,math,datetime
import numpy as np
from scipy.stats import norm
P=Path(__file__).resolve().parent;R=P.parent;H=R.parent;O=R/'run'
read=lambda p:json.loads(p.read_text());rows=lambda p:[json.loads(s)for s in p.read_text().splitlines()];sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
checks=[]
def check(n,c):checks.append({'name':n,'pass':bool(c)})
def close(a,b,tol=1e-9):return abs(float(a)-float(b))<=tol*max(1,abs(float(a)),abs(float(b)))
def group(rs):
 d=collections.defaultdict(lambda:collections.defaultdict(list))
 for r in rs:d[r['source_family_group']][r['peptide_id']].append(r)
 return d

def weight(rs):
 gs=group(rs);return np.array([1/len(gs)/len(gs[r['source_family_group']])/len(gs[r['source_family_group']][r['peptide_id']])for r in rs])
AA='ACDEFGHIKLMNPQRSTVWY'
def feat(rs):return np.asarray([[r['sequence'].count(a)/len(r['sequence'])for a in AA]+[math.log(len(r['sequence']))]for r in rs])
def metrics(rs):
 gs=group(rs);ga=[];gq=[];pa=[];pq=[];errs=[]
 for peps in gs.values():
  a=[];q=[]
  for pp in peps.values():
   e=[r['prediction_log10']-data[r['activity_id']]['target_log10']for r in pp];errs+=e;a.append(sum(map(abs,e))/len(e));q.append(sum(v*v for v in e)/len(e))
  pa+=a;pq+=q;ga.append(sum(a)/len(a));gq.append(sum(q)/len(q))
 return {'rows':len(rs),'peptide_IDs':len(pa),'families':len({r['sequence_family']for r in rs}),'source_groups':len(gs),'group_peptide_MAE':sum(ga)/len(ga),'group_peptide_RMSE':math.sqrt(sum(gq)/len(gq)),'peptide_MAE':sum(pa)/len(pa),'peptide_RMSE':math.sqrt(sum(pq)/len(pq)),'row_MAE':sum(map(abs,errs))/len(errs),'row_RMSE':math.sqrt(sum(v*v for v in errs)/len(errs))}

def independent_objective(theta,z,ys,ws,kinds,tail):
 mu=z@theta;loss=[];grad=[]
 for m,y,kind in zip(mu,ys,kinds):
  if kind=='point'or not tail:loss.append(.5*(m-y)**2);grad.append(m-y)
  elif kind=='right':loss.append(-norm.logsf(y-m));grad.append(-math.exp(norm.logpdf(y-m)-norm.logsf(y-m)))
  elif kind=='left':loss.append(-norm.logcdf(y-m));grad.append(math.exp(norm.logpdf(y-m)-norm.logcdf(y-m)))
  else:raise ValueError(kind)
 penalty=theta.copy();penalty[0]=0
 return float(ws@loss+.5*np.dot(theta[1:],theta[1:])),z.T@(ws*np.array(grad))+penalty

cfg=read(R/'PROTOCOL.json');fr=read(R/'FREEZE.json');result=read(O/'RESULT.json');folds=read(R/'FOLDS.json');data={r['activity_id']:r for r in rows(R/'DATA.jsonl')};pred=rows(O/'PREDICTIONS.jsonl');arms=['point_mean','point_ridge','endpoint_ridge','tail_ridge']
check('freeze_local_unchanged',all(sha(R/n)==h for n,h in fr['local'].items()))
check('freeze_sources_unchanged',all(sha(H/n)==h for n,h in fr['sources'].items()))
for name in ['SCIENCE_PREAUDIT.json','ENGINEERING_PREAUDIT.json']:
 rev=read(P/name);check(name+'_pass_and_binding',rev['status']=='PASS'and rev['freeze_sha256']==sha(R/'FREEZE.json'))
check('started_binding_threads_seed',read(O/'STARTED.json')['freeze_sha256']==sha(R/'FREEZE.json')and read(O/'STARTED.json')['seed']==1729 and set(read(O/'STARTED.json')['threads'].values())=={'1'})
check('complete_no_failure',result['status']=='COMPLETE_DEVELOPMENT_ONLY'and not(O/'FAILURE.json').exists())
check('runtime_budget',result['elapsed_seconds']<600 and result['peak_rss_bytes']<1500000000)
check('exact588_unique_predictions',len(pred)==588 and len({(r['model'],r['activity_id'])for r in pred})==588)
point_ids={r['activity_id']for r in data.values()if r['label_kind']=='point'}
check('each_arm147_points_exact',all({r['activity_id']for r in pred if r['model']==a}==point_ids for a in arms))
fields=['peptide_id','sequence_family','source_family_group','native_unit_group','concentration_raw','target_log10','label_kind']
check('all_prediction_metadata_and_point_targets_unchanged',all(all(r[k]==data[r['activity_id']][k]for k in fields)and r['label_kind']=='point'and math.isfinite(r['prediction_log10'])for r in pred))
recomputed={'units':{},'fold_metrics':{},'training_audits':[]};alltags=[];fitcount=0
for unit,fs in folds.items():
 recomputed['units'][unit]={};recomputed['fold_metrics'][unit]={}
 for arm in arms:
  mm=metrics([r for r in pred if r['native_unit_group']==unit and r['model']==arm]);recomputed['units'][unit][arm]=mm
  check(unit+'_'+arm+'_all_unit_metrics_denominators',set(mm)==set(result['units'][unit][arm])and all(close(v,result['units'][unit][arm][k])for k,v in mm.items()))
 for fold in fs:
  fid=fold['group'];tag=unit+'__'+fid;alltags.append(tag);receipt=read(O/(tag+'.json'));tr=[data[i]for i in fold['train']];va=[data[i]for i in fold['validation']];tp=[r for r in tr if r['label_kind']=='point'];w=weight(tr);wp=weight(tp)
  check(tag+'_memberships',receipt['unit']==unit and receipt['fold']==fold and receipt['train_point_activity_ids']==[r['activity_id']for r in tp])
  check(tag+'_both_weights_exact',len(w)==len(receipt['train_weights_all'])and len(wp)==len(receipt['train_weights_point'])and np.allclose(w,receipt['train_weights_all'],atol=1e-14,rtol=0)and np.allclose(wp,receipt['train_weights_point'],atol=1e-14,rtol=0)and close(w.sum(),1)and close(wp.sum(),1))
  for key in ['peptide_id','sequence','annotation_fingerprint','sequence_family','source_family_group']:
   check(tag+'_no_overlap_'+key,not({r[key]for r in tr}&{r[key]for r in va}))
  check(tag+'_no_source_overlap',not({s for r in tr for s in r['source_keys']}&{s for r in va for s in r['source_keys']}))
  x=feat(tr);mu=np.sum(w[:,None]*x,axis=0);sc=np.sqrt(np.sum(w[:,None]*(x-mu)**2,axis=0));sc=np.where(sc>1e-12,sc,1)
  check(tag+'_scaler_train_X_only',np.allclose(mu,receipt['mean'],atol=1e-12,rtol=0)and np.allclose(sc,receipt['scale'],atol=1e-12,rtol=0))
  def z(rs):return np.column_stack([np.ones(len(rs)),(feat(rs)-mu)/sc])
  y=np.array([r['target_log10']for r in tr]);yp=np.array([r['target_log10']for r in tp]);center=float(wp@yp);start=np.zeros(22);start[0]=center
  check(tag+'_mean_and_identical_start',close(center,receipt['point_mean'])and np.allclose(start,receipt['start'],atol=1e-12,rtol=0))
  max_gradient=0.;closederr={}
  for arm,rr,ww in [('point_ridge',tp,wp),('endpoint_ridge',tr,w),('tail_ridge',tr,w)]:
   fitcount+=1;theta=np.array(receipt['states'][arm]);zz=z(rr);yy=np.array([r['target_log10']for r in rr]);obj,grad=independent_objective(theta,zz,yy,ww,[r['label_kind']for r in rr],arm=='tail_ridge');diag=receipt['fit_diagnostics'][arm];max_gradient=max(max_gradient,float(np.max(abs(grad))))
   check(tag+'_'+arm+'_independent_stationarity_and_objective',theta.shape==(22,)and np.isfinite(theta).all()and np.max(abs(grad))<=1e-5 and close(obj,diag['objective_within_arm_only'])and close(np.max(abs(grad)),diag['max_abs_gradient']))
   check(tag+'_'+arm+'_optimizer_limits_success',diag['success']is True and diag['iterations']<=500 and diag['evaluations']<=10000)
   if arm!='tail_ridge':
    closed=np.linalg.solve(zz.T@(ww[:,None]*zz)+np.diag([0]+[1]*21),zz.T@(ww*yy));closederr[arm]=float(np.max(abs(closed-theta)));check(tag+'_'+arm+'_independent_closedform',closederr[arm]<1e-5)
   pp={r['activity_id']:r['prediction_log10']for r in pred if r['native_unit_group']==unit and r['source_family_group']==fid and r['model']==arm}
   check(tag+'_'+arm+'_prediction_from_saved_state',set(pp)==set(fold['validation'])and all(close(pp[r['activity_id']],v)for r,v in zip(va,z(va)@theta)))
  cp=[r for r in pred if r['native_unit_group']==unit and r['source_family_group']==fid and r['model']=='point_mean'];check(tag+'_pointmean_predictions',len(cp)==len(va)and all(close(r['prediction_log10'],center)for r in cp))
  recomputed['fold_metrics'][unit][fid]={}
  for arm in arms:
   pp=[r for r in pred if r['native_unit_group']==unit and r['source_family_group']==fid and r['model']==arm];mm=metrics(pp);recomputed['fold_metrics'][unit][fid][arm]=mm
   check(tag+'_'+arm+'_all_fold_metrics_denominators',all(close(v,result['fold_metrics'][unit][fid][arm][k])for k,v in mm.items()))
  recomputed['training_audits'].append({'unit':unit,'group':fid,'train_all_rows':len(tr),'train_point_rows':len(tp),'train_censor_rows':len(tr)-len(tp),'train_groups_all':len(group(tr)),'train_groups_point':len(group(tp)),'validation_rows':len(va),'validation_peptides':len({r['peptide_id']for r in va}),'max_abs_gradient_across_arms':max_gradient,'closedform_coefficient_max_error':closederr})
check('34folds_102fits_exact',result['completed_folds']==alltags and len(alltags)==34 and fitcount==102)
contrasts={}
for unit,foldmetrics in recomputed['fold_metrics'].items():
 contrasts[unit]={}
 for name,a,b in [('tail_minus_endpoint','tail_ridge','endpoint_ridge'),('endpoint_minus_point','endpoint_ridge','point_ridge'),('tail_minus_point','tail_ridge','point_ridge'),('tail_minus_mean','tail_ridge','point_mean')]:
  vals=[{'group':g,'delta_group_peptide_MAE':m[a]['group_peptide_MAE']-m[b]['group_peptide_MAE'],'a_MAE':m[a]['group_peptide_MAE'],'b_MAE':m[b]['group_peptide_MAE'],'point_rows':m[a]['rows'],'point_peptides':m[a]['peptide_IDs']}for g,m in foldmetrics.items()];mean=sum(v['delta_group_peptide_MAE']for v in vals)/len(vals)
  top=recomputed['units'][unit];check(unit+'_'+name+'_paired_mean_equals_primary_delta',close(mean,top[a]['group_peptide_MAE']-top[b]['group_peptide_MAE']))
  contrasts[unit][name]={'delta_group_peptide_MAE':mean,'delta_group_peptide_RMSE':top[a]['group_peptide_RMSE']-top[b]['group_peptide_RMSE'],'delta_peptide_MAE':top[a]['peptide_MAE']-top[b]['peptide_MAE'],'delta_row_MAE':top[a]['row_MAE']-top[b]['row_MAE'],'win_tie_loss_tolerance1e-12':{'wins':sum(v['delta_group_peptide_MAE']< -1e-12 for v in vals),'ties':sum(abs(v['delta_group_peptide_MAE'])<=1e-12 for v in vals),'losses':sum(v['delta_group_peptide_MAE']>1e-12 for v in vals)},'groups':vals}
recomputed['paired_contrasts']=contrasts
(P/'RECOMPUTED_RESULTS.json').write_text(json.dumps(recomputed,indent=2)+'\n')
report={'status':'PASS'if all(c['pass']for c in checks)else'FAIL','reviewer':'/root/native_baseline_engineering_audit','utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'checks':checks,'prediction_rows':len(pred),'folds':len(alltags),'fitted_states_independently_verified':fitcount,'units':recomputed['units'],'paired_contrasts':{u:{n:{k:v for k,v in d.items()if k!='groups'}for n,d in cc.items()}for u,cc in contrasts.items()},'sha256':{'freeze':sha(R/'FREEZE.json'),'predictions':sha(O/'PREDICTIONS.jsonl'),'result':sha(O/'RESULT.json'),'audit_script':sha(P/'audit_results.py'),'recomputed_results':sha(P/'RECOMPUTED_RESULTS.json')},'method':'No model module imported and no optimizer called. Separate scipy.stats survival/CDF objective and derivatives evaluate all saved states; closed normal equations independently verify68ridge states; predictions reconstructed from102saved states; group/peptide/record nested means recomputed from147rawpoint labels; all paired group contrasts retained.','limitations':['Point-report selection, fixed Gaussian working scale and omitted covariates constrain interpretation; no latent censored-outcome validation.','Operational source-family groups and already-exposed cache are not independent blind confirmation.','No natural refitting or new data/seed/tuning by auditor; protected H5 data and test37 untouched.']}
(P/'RESULT_AUDIT.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({'status':report['status'],'checks':len(checks),'failed':[c['name']for c in checks if not c['pass']],'units':report['units'],'paired':report['paired_contrasts']}));assert report['status']=='PASS'
