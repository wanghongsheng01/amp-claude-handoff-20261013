"""Independent recomputation without importing runner or any model; no training."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[k]='1'
from pathlib import Path
import json,math,hashlib,collections,datetime,csv
import numpy as np
P=Path(__file__).resolve().parent;R=P.parent;H=R.parent;O=R/'run'
read=lambda p:json.loads(p.read_text())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
rows=lambda p:[json.loads(s) for s in p.read_text().splitlines()]
checks=[]
def check(n,c):checks.append({'name':n,'pass':bool(c)})
def close(a,b):return abs(float(a)-float(b))<=1e-9*max(1,abs(float(a)),abs(float(b)))
def nest(rs):
 d=collections.defaultdict(lambda:collections.defaultdict(list))
 for r in rs:d[r['source_family_group']][r['peptide_id']].append(r)
 return d

def summarize(rs):
 bygroup=nest(rs);allerr=[];pepabs=[];pepsq=[];groupabs=[];groupsq=[]
 for g,peps in bygroup.items():
  a=[];s=[]
  for pid,rr in peps.items():
   es=[r['prediction_log10']-math.log10(data[r['activity_id']]['lower_raw_unit']) for r in rr]
   allerr+=es;a.append(sum(map(abs,es))/len(es));s.append(sum(e*e for e in es)/len(es))
  pepabs+=a;pepsq+=s;groupabs.append(sum(a)/len(a));groupsq.append(sum(s)/len(s))
 return {'rows':len(rs),'peptide_IDs':len({r['peptide_id'] for r in rs}),'families':len({r['sequence_family'] for r in rs}),'groups':len(bygroup),'weight_sum':1.,'group_peptide_macro_MAE':sum(groupabs)/len(groupabs),'group_peptide_macro_RMSE':math.sqrt(sum(groupsq)/len(groupsq)),'peptide_macro_MAE':sum(pepabs)/len(pepabs),'peptide_macro_RMSE':math.sqrt(sum(pepsq)/len(pepsq)),'row_MAE':sum(map(abs,allerr))/len(allerr),'row_RMSE':math.sqrt(sum(e*e for e in allerr)/len(allerr))}

cfg=read(R/'PROTOCOL.json');freeze=read(R/'FREEZE.json');summary=read(O/'RESULT.json');fm=read(R/'FOLDS.json')
data={r['activity_id']:r for r in rows(R/'DATA.jsonl')};preds=rows(O/'PREDICTIONS.jsonl')
check('local_freeze_hashes_unchanged',all(sha(R/f)==h for f,h in freeze['local_sha256'].items()))
check('source_freeze_hashes_unchanged',all(sha(H/f)==h for f,h in freeze['source_sha256'].items()))
check('preaudit_bound_to_freeze',read(P/'SIGNED_PREAUDIT.json')['freeze_sha256']==sha(R/'FREEZE.json'))
started=read(O/'STARTED.json')
check('run_start_bindings',started['freeze_sha256']==sha(R/'FREEZE.json') and started['protocol_sha256']==sha(R/'PROTOCOL.json') and started['preaudit_sha256']==sha(P/'SIGNED_PREAUDIT.json'))
check('complete_without_failure',summary['status']=='COMPLETE_DEVELOPMENT_ONLY' and not (O/'FAILURE.json').exists())
check('255_prediction_rows',len(preds)==255)
check('255_unique_model_activity_pairs',len({(p['model'],p['activity_id']) for p in preds})==255)
check('all_models_complete_exact85',all({r['activity_id'] for r in preds if r['model']==m}==set(data) for m in cfg['models']))
rawfields=['peptide_id','sequence','annotation_fingerprint','sequence_family','source_family_group','source_keys','concentration_raw','unit_raw','operator_or_semantics','format','target_raw','medium','cfu','cfu_group','ph','ionic_strength','salt','both_termini_observed','modification_annotated','structure_status','raw_sha256','native_unit_group']
check('all_raw_prediction_fields_unchanged',all(all(r[k]==data[r['activity_id']][k] for k in rawfields) for r in preds))
check('all_labels_exact_log_original',all(close(r['target_log10'],math.log10(data[r['activity_id']]['lower_raw_unit'])) and r['native_reported_point']==data[r['activity_id']]['lower_raw_unit'] for r in preds))
check('all_predictions_finite',all(math.isfinite(r['prediction_log10']) for r in preds))
check('single_thread_CPU_environment',read(O/'EXECUTION_ENV.json')['torch_threads']==read(O/'EXECUTION_ENV.json')['torch_interop_threads']==1 and read(O/'EXECUTION_ENV.json')['device']=='cpu' and set(read(O/'EXECUTION_ENV.json')['BLAS_env'].values())=={'1'})
check('budget_and_updates',summary['total_CNN_updates']==1700 and summary['elapsed_seconds']<600 and summary['peak_rss_bytes']<1500000000)
recomputed={'units':{},'fold_metrics':[],'fold_training_audits':[]};expected_tags=[]
AA='ACDEFGHIKLMNPQRSTVWY'
def feature(rs):return np.asarray([[r['sequence'].count(a)/len(r['sequence']) for a in AA]+[math.log(len(r['sequence']))] for r in rs])
for unit,defs in fm['units'].items():
 recomputed['units'][unit]={}
 for model in cfg['models']:
  rr=[r for r in preds if r['native_unit_group']==unit and r['model']==model];own=summarize(rr);given=summary['units'][unit][model]
  check(unit+'_'+model+'_all_unit_metrics_and_denominators',set(own)==set(given) and all(close(v,given[k]) for k,v in own.items()))
  recomputed['units'][unit][model]=own
 for fold in defs['folds']:
  fid=fold['fold_id'];tag=unit+'__'+fid;expected_tags.append(tag);receipt=read(O/(tag+'.json'))
  tr=[data[i] for i in fold['train_activity_ids']];val=[data[i] for i in fold['validation_activity_ids']];trn=nest(tr);vn=nest(val)
  weights=[1/len(trn)/len(trn[r['source_family_group']])/len(trn[r['source_family_group']][r['peptide_id']]) for r in tr]
  y=np.asarray([math.log10(r['lower_raw_unit']) for r in tr]);w=np.asarray(weights);center=sum(q*t for q,t in zip(weights,y))
  check(tag+'_train_weights_all_and_center',len(weights)==len(receipt['training_weights']) and all(close(a,b) for a,b in zip(weights,receipt['training_weights'])) and close(center,receipt['training_label_center']))
  check(tag+'_train_val_memberships',receipt['training_activity_ids']==fold['train_activity_ids'] and receipt['fold']==fold)
  for key in ['peptide_id','sequence','annotation_fingerprint','sequence_family','source_family_group']:
   check(tag+'_zero_overlap_'+key,not({r[key] for r in tr}&{r[key] for r in val}))
  check(tag+'_zero_overlap_source',not({s for r in tr for s in r['source_keys']}&{s for r in val for s in r['source_keys']}))
  check(tag+'_fixed_CNN_receipt',receipt['CNN_updates']==100 and len(receipt['CNN_training_loss'])==100 and all(math.isfinite(x) for x in receipt['CNN_training_loss']) and receipt['seed']==1729 and receipt['validation_used_for_endpoint'] is False)
  check(tag+'_checkpoint_hash',receipt['checkpoint_sha256']==sha(O/(tag+'.pt')))
  x=feature(tr);xv=feature(val);mu=np.sum(w[:,None]*x,axis=0);sd=np.sqrt(np.sum(w[:,None]*(x-mu)**2,axis=0));sd=np.where(sd>1e-12,sd,1.)
  z=np.column_stack([np.ones(len(tr)),(x-mu)/sd]);penalty=np.eye(22)*cfg['ridge']['alpha'];penalty[0,0]=0
  b=np.linalg.solve((z*w[:,None]).T@z+penalty,(z*w[:,None]).T@y)
  pr=np.column_stack([np.ones(len(val)),(xv-mu)/sd])@b
  rp={r['activity_id']:r['prediction_log10'] for r in preds if r['native_unit_group']==unit and r['fold_id']==fid and r['model']=='composition_ridge'}
  check(tag+'_ridge_independent_augmented_system_predictions',all(close(rp[r['activity_id']],p) for r,p in zip(val,pr)))
  cp=[r['prediction_log10'] for r in preds if r['native_unit_group']==unit and r['fold_id']==fid and r['model']=='training_weighted_mean']
  check(tag+'_constant_independent',len(cp)==len(val) and all(close(p,center) for p in cp))
  ownfold={}
  givenfold=next(f for f in summary['fold_metrics'] if f['unit']==unit and f['fold_id']==fid)
  for model in cfg['models']:
   rr=[r for r in preds if r['native_unit_group']==unit and r['fold_id']==fid and r['model']==model]
   check(tag+'_'+model+'_exact_heldout_activity_set',collections.Counter(r['activity_id'] for r in rr)==collections.Counter(fold['validation_activity_ids']))
   check(tag+'_'+model+'_prediction_weights',all(close(r['within_fold_weight'],1/len(vn[r['source_family_group']])/len(vn[r['source_family_group']][r['peptide_id']])) and close(r['unit_primary_weight'],r['within_fold_weight']/defs['groups']) for r in rr))
   metric=summarize(rr);ownfold[model]=metric
   check(tag+'_'+model+'_all_fold_metrics_and_denominators',all(close(v,givenfold['metrics'][model][k]) for k,v in metric.items()))
  recomputed['fold_metrics'].append({'unit':unit,'fold_id':fid,'metrics':ownfold})
  recomputed['fold_training_audits'].append({'unit':unit,'fold_id':fid,'train_rows':len(tr),'train_peptides':len({r['peptide_id'] for r in tr}),'train_groups':len(trn),'held_rows':len(val),'held_peptides':len(vn[fid]),'training_weight_sum':sum(weights),'training_center':center})
check('all17_completed_folds_exact',summary['completed_folds']==expected_tags and len(expected_tags)==17)
with (O/'PREDICTIONS.csv').open() as f: cr=list(csv.DictReader(f))
check('CSV255_matches_JSONL_identity_predictions',len(cr)==len(preds) and all(int(a['activity_id'])==b['activity_id'] and a['model']==b['model'] and close(a['prediction_log10'],b['prediction_log10']) and close(a['target_log10'],b['target_log10']) for a,b in zip(cr,preds)))
(P/'RECOMPUTED_METRICS.json').write_text(json.dumps(recomputed,indent=2)+'\n')
report={'schema':'AMP_H5_NATIVE_BASELINE_INDEPENDENT_RESULT_AUDIT_V1','status':'PASS' if all(c['pass'] for c in checks) else 'FAIL','reviewer':'/root/native_baseline_engineering_audit','utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'checks':checks,'recomputed_predictions':255,'folds':17,'units':recomputed['units'],'reviewed_sha256':{'freeze':sha(R/'FREEZE.json'),'predictions_jsonl':sha(O/'PREDICTIONS.jsonl'),'predictions_csv':sha(O/'PREDICTIONS.csv'),'result':sha(O/'RESULT.json'),'audit_script':sha(P/'audit_results.py'),'recomputed_metrics':sha(P/'RECOMPUTED_METRICS.json')},'scope':'Independent metric/denominator/group weight recomputation from raw labels and OOF predictions; separate augmented-matrix ridge/constant recomputation; no torch/model import or natural retraining. CNN checkpoint hashes verified but forward pass not rerun.','limitations':['All material already exposed; source-family components operational, not proven independent biology.','Exactly one seed and100epochs, no hyperparameter comparisons or statistically significant/generalization claims.','Point-report labels only; nonpoint data retained outside this run. Missing covariates retained; no molar conversion or unknown purity assumptions.']}
(P/'RESULT_AUDIT.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({'status':report['status'],'checks':len(checks),'failed':[c['name'] for c in checks if not c['pass']],'units':report['units']}))
assert report['status']=='PASS'
