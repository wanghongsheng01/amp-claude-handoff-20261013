"""Independent outcome accounting and unit arithmetic, never executes author script."""
import json,hashlib,collections,sys,math
from pathlib import Path
root=Path(__file__).resolve().parents[1]; version=sys.argv[1]if len(sys.argv)>1 else 'chemistry_units_v1'; package=root/version; results=package/'results'
read=lambda p:[json.loads(s)for s in p.read_text().splitlines()]
old=read(root/'dbaasp_bounded_v3/prepared/mic_measurements.jsonl'); chemistry=read(results/'chemical_identity_audit.jsonl'); rows=read(results/'mass_MIC_conversion_draft.jsonl');summary=json.loads((results/'SUMMARY.json').read_text())
rawinv=json.loads((root/'chem_cnn_audit_v1/RAW_CHEMISTRY_INDEPENDENT.json').read_text());rawby={r['peptide_id']:r for r in rawinv['rows']};byid={r['peptide_id']:r for r in chemistry};oldmap={(r['peptide_id'],r['activity_id']):r for r in old}
checks=[]
def check(name,v):
 assert v,name
 checks.append(name)
def stats(rs):return {'rows':len(rs),'peptide_ids':len({r['peptide_id']for r in rs}),'backbones':len({r['sequence'].upper()for r in rs}),'sequence_families':len({r['sequence_family']for r in rs}),'source_family_groups':len({r['source_family_group']for r in rs})}
check('all75source_ids',set(byid)==set(rawby) and len(chemistry)==75)
for p in (root/'dbaasp_bounded_v3/raw').glob('detail_*.json'):
 raw=json.loads(p.read_text());check('raw_sha:'+str(raw['id']),hashlib.sha256(p.read_bytes()).hexdigest()==rawby[raw['id']]['raw_sha256'])
for c in chemistry:
 if c['structure_status']=='PASS_REPRESENTED_STRUCTURE':
  iv=rawby[c['peptide_id']]['smiles'];check('mass:'+str(c['peptide_id']),all(abs(c['verified_represented_parent_MW_g_mol']-v['average_mass'])<1e-6 for v in iv));check('formula:'+str(c['peptide_id']),all(c['formula']==v['formula']for v in iv));check('no_ambiguous_basis:'+str(c['peptide_id']),c['assay_material_mass_basis']=='NOT_ESTABLISHED' and c['assay_molarity_verified']is False)
for i,r in enumerate(rows):
 original=oldmap[(r['peptide_id'],r['activity_id'])];check('rowraw:'+str(i),all(r[k]==v for k,v in original.items()));check('no_admit:'+str(i),r['conversion_status']!='PASS' and not r['assay_molarity_verified'] and not r['training_admitted'])
 if r['conditional_factor_uM_per_ug_ml']is not None:
  mw=rawby[r['peptide_id']]['smiles'][0]['average_mass'];factor=1000/mw;check('factor:'+str(i),abs(r['conditional_factor_uM_per_ug_ml']-factor)<1e-12)
  for side in ['lower','upper']:
   v=original[side+'_raw_unit'];got=r['conditional_parent_equivalent_'+side+'_uM'];check('scaled:'+str(i)+side,got is None if v is None else math.isclose(got,v*factor,abs_tol=1e-10,rel_tol=1e-12))
 else:check('no_pending_mass_scaling:'+str(i),r['conditional_parent_equivalent_lower_uM']is None and r['conditional_parent_equivalent_upper_uM']is None)
check('input86',len(rows)==86 and stats(rows)==summary['mass_MIC_rows'])
check('structurecounts',dict(collections.Counter(r['structure_status']for r in chemistry))==summary['structure_status_counts'])
for status in ['PASS','REFUSE','PENDING']:check('conversion_counts:'+status,stats([r for r in rows if r['conversion_status']==status])==summary['assay_conversion_status'][status])
conditional=[r for r in rows if r['conditional_factor_uM_per_ug_ml']is not None];check('conditionalcounts',stats(conditional)==summary['conditional_parent_equivalent_computable'])
candidates=[r for r in conditional if set(r['screen_exclusions'])=={'not_micromolar_no_conversion'}];check('candidatecounts',stats(candidates)==summary['conditional_screen_candidates_before_massbasis_chemistry_isolation'])
prior=[r for r in old if r['ecoli_target']and r['label_sequence_screen_pass']]
for field,key in [('peptide_id','potential_new_candidate_ids_relative_to_old_screen'),('sequence_family','potential_new_candidate_families_relative_to_old_screen'),('source_family_group','potential_new_candidate_source_groups_relative_to_old_screen')]:check(key,sorted({r[field]for r in candidates}-{r[field]for r in prior})==summary[key])
res={'status':'PASS_ACCOUNTING_UNIT_ARITHMETIC_NO_TRAINING_ADMISSION','package':version,'checks_passed':len(checks),'new_ids':summary['potential_new_candidate_ids_relative_to_old_screen'],'new_families':summary['potential_new_candidate_families_relative_to_old_screen'],'new_sourcegroups':summary['potential_new_candidate_source_groups_relative_to_old_screen'],'assay_pass_rows':summary['assay_conversion_status']['PASS']['rows'],'results_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest()for p in results.iterdir()if p.is_file()},'limitations':['Structuralgraph audit not sourcepaper assayed material verification','Conditional unit-scaled values do not resolve censor/range observation semantics','FamilyIDs andsourcegroups are inherited provisionalgroups, not independence guarantee or completed contamination isolation','StandardsequenceCNNcannotrepresentDresidues/cyclization/terminalchemistry; potentialdata IDs are not automaticallybaselineadmissible']}
(root/'chem_cnn_audit_v1'/('ACTUAL_'+version.upper()+'.json')).write_text(json.dumps(res,indent=2)+'\n');print(json.dumps(res))
