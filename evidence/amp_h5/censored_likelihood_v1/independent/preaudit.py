import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[k]='1'
from pathlib import Path
import json,hashlib,collections,math,sys,datetime,re
import numpy as np
from scipy.stats import norm
P=Path(__file__).resolve().parent;R=P.parent;H=R.parent
sys.path.insert(0,str(R));from model import objective
read=lambda p:json.loads(p.read_text());rows=lambda p:[json.loads(s)for s in p.read_text().splitlines()];sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
checks=[]
def check(n,c):checks.append({'name':n,'pass':bool(c)})
f=read(R/'FREEZE.json');cfg=read(R/'PROTOCOL.json');data=rows(R/'DATA.jsonl');source=rows(H/'dbaasp_coverage_v5/analysis/NATIVE_ROWS.jsonl');folds=read(R/'FOLDS.json');inventory=read(R/'INVENTORY.json')
check('fixed_freeze',sha(R/'FREEZE.json')=='8b723893076b8c2faab939905d5cf055537188329cee65b738c0b8d6cc93cc67')
check('local_hashes_match',all(sha(R/n)==h for n,h in f['local'].items()))
check('source_hashes_match',all(sha(H/n)==h for n,h in f['sources'].items()))
expected=[r for r in source if r['activity_id']!=12605 and(r['plain_Witten_primary_point_eligible']or r['plain_Witten_censor_eligible'])]
check('selected_exact_original_rows',len(expected)==len(data)==173 and all({k:v for k,v in d.items()if k not in ['target_log10','label_kind']}==s for d,s in zip(data,expected)))
check('activity_ids_unique',len({r['activity_id']for r in data})==173)
check('labels_and_bounds_semantics',all(r['label_kind']=={'point':'point','right_bound':'right','left_bound':'left'}[r['format']] and math.isclose(r['target_log10'],math.log10(r['upper_raw_unit']if r['label_kind']=='left'else r['lower_raw_unit']),abs_tol=1e-14) and (r['label_kind']=='point'and r['lower_raw_unit']==r['upper_raw_unit'] or r['label_kind']=='right'and r['operator_or_semantics']in ['>','>=']or r['label_kind']=='left'and r['operator_or_semantics']in ['<','<='])for r in data))
check('original_canonical_uppercase_not_transformed',all(re.fullmatch('[ACDEFGHIKLMNPQRSTVWY]{1,99}',r['sequence'])for r in data))
for key in ['peptide_id','sequence','annotation_fingerprint','sequence_family']:
 d=collections.defaultdict(set)
 for r in data:d[r[key]].add(r['source_family_group'])
 check(key+'_one_group',all(len(v)==1 for v in d.values()))
sources=collections.defaultdict(set)
for r in data:
 for s in r['source_keys']:sources[s].add(r['source_family_group'])
check('all_source_keys_one_group',all(len(v)==1 for v in sources.values()))
counts={}
for unit,(npnt,ncens,nfold)in [('ug_ml_native',(79,14,21)),('uM_native',(68,12,13))]:
 rr=[r for r in data if r['native_unit_group']==unit];p=[r for r in rr if r['label_kind']=='point'];c=[r for r in rr if r['label_kind']!='point'];lookup={r['activity_id']:r for r in rr};held=[]
 check(unit+'_counts',len(p)==npnt and len(c)==ncens and len(folds[unit])==nfold and all(r['label_kind']=='right'for r in c))
 check(unit+'_screen_gate',len({r['source_family_group']for r in c})>=3 and len({r['source_family_group']for r in p})>=5 and inventory[unit]['screening_gate'])
 for fold in folds[unit]:
  g=fold['group'];tr=[r for r in rr if r['source_family_group']!=g];va=[r for r in p if r['source_family_group']==g]
  check(unit+'_'+g+'_exact_partition',set(fold['train'])=={r['activity_id']for r in tr}and set(fold['validation'])=={r['activity_id']for r in va})
  check(unit+'_'+g+'_training_group_gate',len({r['source_family_group']for r in tr if r['label_kind']=='point'})>=3 and len({r['source_family_group']for r in tr if r['label_kind']!='point'})>=2)
  for key in ['peptide_id','sequence','annotation_fingerprint','sequence_family','source_family_group']:
   check(unit+'_'+g+'_disjoint_'+key,not({r[key]for r in tr}&{r[key]for r in va}))
  check(unit+'_'+g+'_disjoint_sources',not({s for r in tr for s in r['source_keys']}&{s for r in va for s in r['source_keys']}))
  held+=fold['validation']
 check(unit+'_all_points_once',collections.Counter(held)==collections.Counter(r['activity_id']for r in p))
 counts[unit]={'rows':len(rr),'points':len(p),'censors':len(c),'point_peptides':len({r['peptide_id']for r in p}),'point_families':len({r['sequence_family']for r in p}),'point_groups':nfold,'censor_groups':len({r['source_family_group']for r in c}),'censor_only_groups':sorted({r['source_family_group']for r in c}-{r['source_family_group']for r in p})}
# Independent synthetic likelihood, scipy.stats survival/CDF formulations, no optimizer/training.
z=np.array([[1.,-.8,.2],[1.,.4,-.3],[1.,.6,.9],[1.,-.1,.3]]);y=np.array([.1,1.2,-.4,.7]);w=np.array([.1,.2,.3,.4]);k=np.array(['point','right','left','right']);theta=np.array([.2,-.3,.5])
def independent(t):
 mu=z@t;loss=[];grad=[]
 for m,b,kind in zip(mu,y,k):
  if kind=='point':loss.append(.5*(m-b)**2);grad.append(m-b)
  elif kind=='right':loss.append(-norm.logsf(b-m));grad.append(-math.exp(norm.logpdf(b-m)-norm.logsf(b-m)))
  else:loss.append(-norm.logcdf(b-m));grad.append(math.exp(norm.logpdf(b-m)-norm.logcdf(b-m)))
 return float(w@loss+.5*np.dot(t[1:],t[1:])),z.T@(w*np.array(grad))+np.array([0,t[1],t[2]])
a,ag=objective(theta,z,y,w,k,True);b,bg=independent(theta)
check('independent_right_survival_left_CDF_objective_gradient',abs(a-b)<1e-12 and np.max(abs(ag-bg))<1e-12)
eps=1e-6;num=np.array([(independent(theta+eps*np.eye(3)[i])[0]-independent(theta-eps*np.eye(3)[i])[0])/(2*eps)for i in range(3)])
check('independent_numeric_gradient',np.max(abs(num-ag))<1e-8)
for bound in [-100.,0.,100.]:
 for kind,sgn in [('right',-1),('left',1)]:
  value,gradient=objective(np.zeros(3),np.array([[1.,0.,0.]]),np.array([bound]),np.array([1.]),np.array([kind]),True)
  check(str(bound)+'_'+kind+'_tail_finite_correct_gradient_direction',math.isfinite(value)and np.isfinite(gradient).all()and gradient[0]*sgn>=0)
check('same_fixed_optimizer_seed_budget',cfg['optimizer']['seed']==1729 and cfg['optimizer']['maxiter']==500 and cfg['optimizer']['maxfun']==10000 and cfg['limits']['maximum_regression_fits']==102 and cfg['limits']['wall_seconds_total']==600 and cfg['limits']['cpu_threads']==cfg['limits']['BLAS_threads']==1)
check('natural_run_not_started',not(R/'run').exists())
report={'status':'PASS'if all(c['pass']for c in checks)else'FAIL','reviewer':'/root/native_baseline_engineering_audit','utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'freeze_sha256':sha(R/'FREEZE.json'),'checks':checks,'counts':counts,'independent_gradient_max_error':float(np.max(abs(num-ag))),'static_review':['Right loss -log P(Y>bound)=-logPhi(mu-bound), derivative negative; left loss -logPhi(bound-mu), derivative positive; log_ndtr and logpdf ratio numerically stable at tested +/-100.','Point loss and penalty half factors preserve alpha1 ridge normal equations; intercept unpenalized; exactly22 coefficients.','Tail versus endpoint share rows,weights,scaling,capacity,initialization,optimizer and limits; only bound observation loss differs. Points-only arms independently renormalize point training subset.','Scaler uses training all-admitted X only; validation features enter only prediction. Convergence depends training objective only; no heldout score used.','All34folds validated with >=2training censor groups and >=3training point groups.3molar censor-only groups never counted as validation.','Runner verifies freeze and two independent signatures, serial102fits, CPU/BLAS env1 before numerical imports, deadline600s and RSS1.5GB; failure preserves partial and forbids automatic retry.'],'limitations':['Engineering approval only; independent scientific signature also required.','No original-paper semantics reread; labels and comparisons are cached reported annotations.','Runtime single-thread proof relies on BLAS env; no neural model execution or natural optimization by this auditor.','Point-only heldout metrics cannot evaluate latent censored outcomes or certify causal benefit.'],'reviewed_sha256':{n:sha(R/n)for n in f['local']}}
(P/'ENGINEERING_PREAUDIT.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({'status':report['status'],'checks':len(checks),'failed':[c['name']for c in checks if not c['pass']],'counts':counts,'gradient_error':report['independent_gradient_max_error']}));assert report['status']=='PASS'
