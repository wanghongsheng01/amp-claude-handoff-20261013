"""Read-only semantic and stored-state verification; no optimizer/model import."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[k]='1'
from pathlib import Path
import json,math,hashlib,collections,datetime
import numpy as np
P=Path(__file__).resolve().parent;R=P.parent.parent/'censored_likelihood_v1';O=R/'run'
read=lambda p:json.loads(p.read_text());rows=lambda p:[json.loads(s)for s in p.read_text().splitlines()];sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
checks=[]
def check(n,c):checks.append({'name':n,'pass':bool(c)})
def wgt(rs):
 g=collections.defaultdict(lambda:collections.Counter())
 for r in rs:g[r['source_family_group']][r['peptide_id']]+=1
 return np.array([1/len(g)/len(g[r['source_family_group']])/g[r['source_family_group']][r['peptide_id']]for r in rs])
AA='ACDEFGHIKLMNPQRSTVWY'
def feat(rs):return np.asarray([[r['sequence'].count(a)/len(r['sequence'])for a in AA]+[math.log(len(r['sequence']))]for r in rs])
f=read(R/'FREEZE.json');data=rows(R/'DATA.jsonl');lookup={r['activity_id']:r for r in data};pred=rows(O/'PREDICTIONS.jsonl');folds=read(R/'FOLDS.json');cfg=read(R/'PROTOCOL.json')
check('all_frozen_local_unchanged',all(sha(R/n)==h for n,h in f['local'].items()))
check('all_frozen_sources_unchanged',all(sha(R.parent/n)==h for n,h in f['sources'].items()))
check('raw_unit_group_correspondence',all(r['native_unit_group']=={'µg/ml':'ug_ml_native','µM':'uM_native'}[r['unit_raw']]for r in data))
check('log10_original_point_or_bound',all(abs(r['target_log10']-math.log10(r['upper_raw_unit']if r['label_kind']=='left'else r['lower_raw_unit']))<1e-14 for r in data))
check('training_censor_semantic_directions',all(r['label_kind']=='point'or r['label_kind']=='right'and r['operator_or_semantics']in['>','>=']or r['label_kind']=='left'and r['operator_or_semantics']in['<','<=']for r in data))
check('no_censored_evaluation',all(lookup[r['activity_id']]['label_kind']=='point'and r['target_log10']==lookup[r['activity_id']]['target_log10']for r in pred))
check('canonical_exact_encoding',all(set(r['sequence'])<=set(AA)and 1<=len(r['sequence'])<100 for r in data))
check('feature_fractions_sum1',np.allclose(feat(data)[:,:20].sum(axis=1),1,atol=1e-14,rtol=0))
check('point_only_scaling_difference_disclosed','all-training-X scaling' in cfg['old_baseline_comparability'])
max_prediction_error=0.;feature_support=[]
for unit,fs in folds.items():
 for fold in fs:
  tag=unit+'__'+fold['group'];state=read(O/(tag+'.json'));tr=[lookup[i]for i in fold['train']];tp=[r for r in tr if r['label_kind']=='point'];va=[lookup[i]for i in fold['validation']];w=wgt(tr);wp=wgt(tp)
  check(tag+'_unit_isolated',all(r['native_unit_group']==unit for r in tr+va))
  for k in ['peptide_id','sequence','annotation_fingerprint','sequence_family','source_family_group']:
   check(tag+'_disjoint_'+k,not({r[k]for r in tr}&{r[k]for r in va}))
  check(tag+'_source_disjoint',not({s for r in tr for s in r['source_keys']}&{s for r in va for s in r['source_keys']}))
  xx=feat(tr);mu=np.sum(xx*w[:,None],axis=0);sd=np.sqrt(np.sum((xx-mu)**2*w[:,None],axis=0));sd=np.where(sd>1e-12,sd,1.)
  check(tag+'_shared_scaler_train_only',np.allclose(mu,state['mean'],atol=1e-12,rtol=0)and np.allclose(sd,state['scale'],atol=1e-12,rtol=0))
  check(tag+'_weights_both_subsets',np.allclose(w,state['train_weights_all'],atol=1e-14,rtol=0)and np.allclose(wp,state['train_weights_point'],atol=1e-14,rtol=0))
  center=sum(wi*r['target_log10']for wi,r in zip(wp,tp));zv=np.column_stack([np.ones(len(va)),(feat(va)-mu)/sd]);errors=[]
  for arm in ['point_mean','point_ridge','endpoint_ridge','tail_ridge']:
   expected=np.full(len(va),center)if arm=='point_mean'else zv@np.asarray(state['states'][arm]);ps={r['activity_id']:r['prediction_log10']for r in pred if r['native_unit_group']==unit and r['source_family_group']==fold['group']and r['model']==arm}
   er=max(abs(ps[r['activity_id']]-y)for r,y in zip(va,expected));errors.append(er);max_prediction_error=max(max_prediction_error,er)
  check(tag+'_predictions_untransformed_log10_no_missing_inverse',max(errors)<1e-10)
report={'status':'PASS_NO_RECOVERABLE_IMPLEMENTATION_BUG_IDENTIFIED'if all(c['pass']for c in checks)else'FAIL_INVESTIGATE','utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'reviewer':'/root/native_baseline_engineering_audit','checks':checks,'max_saved_prediction_reconstruction_error':max_prediction_error,'semantics':['Targets use log10 of cached native concentrations; natural log is used only for sequence length feature. These are different intentional transformations.','Regressions directly predict log10 MIC with a fitted intercept. No response centering or exponentiation is required before log10-domain scoring; point_mean is a log10 training mean.','All-trained-X standardization is shared by three regression arms, using no validation X or labels; point-only weights and labels are separately normalized. This differs from the earlier baseline scaler and is disclosed.','Censored rows enter training, never point validation. Endpoint control deliberately treats bounds as endpoints; tail arm uses survival likelihood. This is a control design, not an accidental label conversion.','Same source/family groups are absent from each heldout training split. Remaining source heterogeneity and biological dependence are scientific limitations, not an identified partition implementation error.','Sequence composition cannot encode ordering, assay conditions or some chemistry distinctions. The representation restriction is known; it is not evidence that sequence features have no predictive signal.'],'boundaries':['No implementation bug recovered by this review; no reason to rerun or repair fitted models on current evidence.','Fixed linear-model negative results cannot establish absence of feature signal or predictability under another model.','Group drift, feature support, or covariance diagnostics are descriptive decompositions, not causal attribution.','Point selection and fixed Gaussian scale preclude interpreting observed point prediction errors as full latent MIC population risk.'],'sha256':{'freeze':sha(R/'FREEZE.json'),'predictions':sha(O/'PREDICTIONS.jsonl'),'model':sha(R/'model.py'),'runner':sha(R/'run.py'),'script':sha(P/'recheck_implementation.py')}}
(P/'IMPLEMENTATION_RECHECK.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({'status':report['status'],'checks':len(checks),'failed':[c['name']for c in checks if not c['pass']],'max_prediction_error':max_prediction_error}));assert all(c['pass']for c in checks)
