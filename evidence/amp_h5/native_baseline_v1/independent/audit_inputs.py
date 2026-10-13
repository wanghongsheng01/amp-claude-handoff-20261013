"""Independent read-only checks on native-unit admitted rows, no model import."""
from pathlib import Path
import json, hashlib, collections, math, re
HERE=Path(__file__).resolve().parent
H=HERE.parent.parent
raw=H/'native_unit_admission_v1/ROW_ADMISSION.jsonl'
rows=[json.loads(s) for s in raw.read_text().splitlines()]
checks=[]
def check(name,ok):
 checks.append({'name':name,'pass':bool(ok)})
check('142_original_ecoli_rows_retained',len(rows)==142)
check('activity_id_unique',len({r['activity_id'] for r in rows})==len(rows))
for key in ['peptide_id','sequence','sequence_family']:
 mapping=collections.defaultdict(set)
 for r in rows:mapping[r[key]].add(r['source_family_group'])
 check(key+'_one_source_family_group',all(len(gs)==1 for gs in mapping.values()))
source_mapping=collections.defaultdict(set)
for r in rows:
 for src in r['source_keys']:source_mapping[src].add(r['source_family_group'])
check('all_observed_source_keys_one_group',all(len(gs)==1 for gs in source_mapping.values()))
units={}
for unit,expected in [('ug_ml_native',(46,33,17,10)),('uM_native',(39,13,8,7))]:
 rr=[r for r in rows if r['native_unit_group']==unit and r['plain_Witten_primary_point_eligible']]
 count=(len(rr),len({r['peptide_id'] for r in rr}),len({r['sequence_family'] for r in rr}),len({r['source_family_group'] for r in rr}))
 check(unit+'_four_counts',count==expected)
 check(unit+'_points_only',all(r['format']=='point' and r['lower_raw_unit']==r['upper_raw_unit'] and math.isfinite(r['lower_raw_unit']) and r['lower_raw_unit']>0 for r in rr))
 check(unit+'_raw_canonical_domain',all(re.fullmatch('[ACDEFGHIKLMNPQRSTVWY]{1,99}',r['sequence']) for r in rr))
 check(unit+'_source_attribution_and_no_pending',all(r['reference_index_valid'] and not r['attribution_pending'] and r['core_reported_label_eligible'] for r in rr))
 groups=collections.defaultdict(list)
 for r in rr:groups[r['source_family_group']].append(r)
 folds=[]
 for g,held in sorted(groups.items()):
  tr=[r for r in rr if r['source_family_group']!=g]
  for key in ['peptide_id','sequence','sequence_family']:
   check(unit+'_'+g+'_disjoint_'+key,not({r[key] for r in held}&{r[key] for r in tr}))
  check(unit+'_'+g+'_disjoint_sources',not({s for r in held for s in r['source_keys']}&{s for r in tr for s in r['source_keys']}))
  folds.append({'holdout_group':g,'held_rows':len(held),'held_peptides':len({r['peptide_id'] for r in held}),'held_families':len({r['sequence_family'] for r in held}),'train_rows':len(tr),'train_peptides':len({r['peptide_id'] for r in tr}),'train_groups':len(groups)-1})
 units[unit]={'counts':dict(zip(['rows','peptides','families','source_family_groups'],count)),'activity_ids':sorted(r['activity_id'] for r in rr),'folds':folds}
report={'status':'PASS' if all(c['pass'] for c in checks) else 'FAIL','sha256':hashlib.sha256(raw.read_bytes()).hexdigest(),'checks':checks,'units':units,'scope':'cached developmental material; operational groups are not certified independent biology; protected test untouched'}
(HERE/'INPUT_AUDIT.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({'status':report['status'],'checks':len(checks),'counts':{k:v['counts'] for k,v in units.items()}}))
assert report['status']=='PASS'
