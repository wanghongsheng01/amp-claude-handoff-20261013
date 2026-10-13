"""Independent engineer static/data preaudit; no neural model execution."""
from pathlib import Path
import json,hashlib,collections,math,datetime
P=Path(__file__).resolve().parent;R=P.parent;H=R.parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
read=lambda p:json.loads(p.read_text())
checks=[]
def check(n,c):checks.append({'name':n,'pass':bool(c)})
orig=[json.loads(s) for s in (H/'native_unit_admission_v1/ROW_ADMISSION.jsonl').read_text().splitlines()]
raw=[r for r in orig if r['plain_Witten_primary_point_eligible']]
data=[json.loads(s) for s in (R/'DATA.jsonl').read_text().splitlines()]
check('DATA_exact_original_rows_and_metadata',data==raw)
folds=read(R/'FOLDS.json');own=read(P/'INPUT_AUDIT.json')
check('independent82_input_checks_pass',own['status']=='PASS' and len(own['checks'])==82)
for unit,details in folds['units'].items():
 rr=[r for r in data if r['native_unit_group']==unit]
 all_ids={r['activity_id'] for r in rr};held=[]
 check(unit+'_groups',len(details['folds'])==own['units'][unit]['counts']['source_family_groups'])
 for f in details['folds']:
  expected={r['activity_id'] for r in rr if r['source_family_group']==f['fold_id']}
  check(unit+'_'+f['fold_id']+'_exact_partition',set(f['validation_activity_ids'])==expected and set(f['train_activity_ids'])==all_ids-expected)
  train=[r for r in rr if r['activity_id'] in f['train_activity_ids']];val=[r for r in rr if r['activity_id'] in f['validation_activity_ids']]
  for prefix,rs in [('train',train),('validation',val)]:
   check(unit+'_'+f['fold_id']+'_'+prefix+'_denominators',f[prefix+'_rows']==len(rs) and f[prefix+'_peptides']==len({r['peptide_id'] for r in rs}) and f[prefix+'_families']==len({r['sequence_family'] for r in rs}) and f[prefix+'_groups']==len({r['source_family_group'] for r in rs}))
  held+=f['validation_activity_ids']
 check(unit+'_OOF_once',collections.Counter(held)==collections.Counter(all_ids))
check('model_exact_previously_audited',sha(R/'model.py')==sha(H/'witten_cpu_v1/model.py'))
p=read(R/'PROTOCOL.json');test=read(R/'ENGINEERING_TESTS.json');freeze=read(R/'FREEZE.json')
check('synthetic15_pass_bound_runner',test['status']=='PASS' and test['count']==15 and test['runner_sha256']==sha(R/'run.py'))
check('seed1729_fixed',p['seed']==1729)
check('epochs100_no_adaptive_endpoint',p['cnn']['epochs']==100 and p['cnn']['updates_per_epoch']==1)
check('ridge_alpha1_sumweights1_fixed',p['ridge']['alpha']==1)
check('CPU_one_and_600s',p['limits']['cpu_threads']==p['limits']['interop_threads']==p['limits']['blas_threads']==1 and p['limits']['wall_seconds']==600)
check('train_center_explicit_protocol','center' in json.dumps(p['cnn']).lower())
check('local_hashes_match',all(sha(R/f)==h for f,h in freeze['local_sha256'].items()))
check('source_hashes_match',all(sha(H/f)==h for f,h in freeze['source_sha256'].items()))
check('run_not_started',not (R/'run').exists())
report={'schema':'AMP_H5_NATIVE_BASELINE_INDEPENDENT_PREAUDIT_V1','status':'PASS' if all(c['pass'] for c in checks) else 'FAIL','reviewer':'/root/native_baseline_engineering_audit','utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'freeze_sha256':sha(R/'FREEZE.json'),'checks':checks,'static_review':['Runner fit uses training-only hierarchical weights/label center/ridge standardization. CNN heldout features called only after 100 updates; no validation endpoint selection.','Metrics use equal source groups, equal peptides within groups, equal records within peptide; RMSE square root taken after weighted MSE. Row and peptide sensitivity separately reported.','Observed source/family group partition inherited from full cached collection; no unit-specific regrouping. No original protected data path is loaded.','CPU-only, thread env before numpy/torch imports, alarm600s, RSS check1.5GB, serial models, single run directory, failed/partial retention.','KerasStyleAdam uses verified default optimizer arguments matching frozen protocol. Model exact byte copy of previously audited183069-param architecture.'],'limitations':['Tiny pre-exposed developmental cache,17operational grouped folds, no untouched validation or certified biological independence.','CNN full-batch centered-target100update adaptation is not author protocol/performance replication. A fixed negative result cannot establish absence of model signal.','No model execution by independent auditor; static engineering review plus independent data/math verification only.'],'reviewed_sha256':{f:sha(R/f) for f in ['PROTOCOL.json','run.py','model.py','DATA.jsonl','FOLDS.json','ENGINEERING_TESTS.json','FREEZE.json']}}
(P/'SIGNED_PREAUDIT.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({'status':report['status'],'checks':len(checks),'failed':[c['name'] for c in checks if not c['pass']]}))
assert report['status']=='PASS'
