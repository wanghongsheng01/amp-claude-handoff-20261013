import json,hashlib,datetime
from pathlib import Path
r=Path(__file__).resolve().parent;h=r.parent
now=lambda:datetime.datetime.now(datetime.timezone.utc).isoformat()
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())
def write(p,d):p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
# Validate all pre-existing frozen package manifests before creating final index.
checks=[]
for package in ['dbaasp_bounded_v3','witten_cpu_v1','chem_cnn_audit_v1']:
 name='FINAL_MANIFEST.json' if package=='dbaasp_bounded_v3' else 'MANIFEST.json'
 d=read(h/package/name)
 for rel,entry in d['files'].items():
  expected=entry if isinstance(entry,str) else entry['sha256']
  assert sha(h/package/rel)==expected,(package,rel)
  checks.append([package,rel])
for package in ['chemistry_units_v1','chemistry_units_v2']:
 for rel,expected in read(h/package/'FREEZE.json')['local_sha256'].items():assert sha(h/package/rel)==expected,(package,rel)
 files={str(p.relative_to(h/package)):{'sha256':sha(p),'bytes':p.stat().st_size} for p in sorted((h/package).rglob('*')) if p.is_file() and '__pycache__' not in p.parts and p.name!='MANIFEST.json'}
 write(h/package/'MANIFEST.json',{'created_utc':now(),'files':files,'scope':'Code/config/source bindings/results/logs; bytecode cache excluded; generated after logs closed'})
rows=[json.loads(s) for s in (h/'chemistry_units_v2/results/mass_MIC_conversion_draft.jsonl').read_text().splitlines()]
rep=read(h/'chemistry_units_v2/results/CNN_REPRESENTATION_AUDIT.json');ids={p['peptide_id'] for p in rep if p['plain_CNN_representation_compatible']}
queue=[]
for i in sorted(ids):
 rs=[x for x in rows if x['peptide_id']==i and x['conditional_factor_uM_per_ug_ml'] is not None and all(y=='not_micromolar_no_conversion' for y in x['screen_exclusions'])]
 queue.append({'peptide_id':i,'sequence_family':rs[0]['sequence_family'],'source_family_group':rs[0]['source_family_group'],'activities':[{k:x[k] for k in ['activity_id','concentration_raw','unit_raw','operator_or_semantics','target_raw','medium','cfu','ph','reference_raw','source_records','raw_sha256','represented_formula','represented_parent_MW_g_mol','conditional_parent_equivalent_lower_uM','conditional_parent_equivalent_upper_uM']} for x in rs],'status':'PENDING_NOT_TRAINING_ADMITTED','required_evidence':['Exact activity-to-source linkage and original MIC observation semantics','Assayed material basis: covalent entity, counterion/salt/solvate/hydrate/purity or net-peptide convention sufficient for stated molarity','Null terminal annotation resolution','Opaque independent contamination-isolation receipt without opening test37','Grouped data sufficiency and predeclared natural protocol'],'source_citations_are_not_read_paper_evidence':True})
write(r/'CANDIDATE_EVIDENCE_QUEUE.json',{'created_utc':now(),'candidates':queue,'no_new_requests':True,'all_statuses_pending':True})
s=read(h/'chemistry_units_v2/results/SUMMARY.json');sign=read(h/'chem_cnn_audit_v1/SIGNOFF.json');cnn=read(h/'witten_cpu_v1/RESULT.json')
write(r/'DELIVERY.json',{'schema':'AMP_H5_CHEMISTRY_AND_CNN_DELIVERY_V1','created_utc':now(),'status':sign['status'],'subject':'Antimicrobial peptide AMP H5 only','chemistry':s,'cnn':sign['cnn'],'natural_training':'NOT_RUN_NOT_ADMITTED','model_slots':'Implementation and independent audit agents released their serial CPU slots','negative_results_preserved':['chemistry_units_v1:41generic-tautomer enumeration cap outcomes','chemistry_units_v2:24structural PENDING and86assay PENDING','witten_cpu_v1/attempt1:JSON serialization failure'],'independent_signoff_sha256':sha(h/'chem_cnn_audit_v1/SIGNOFF.json'),'package_manifest_sha256':{p:sha(h/p/('FINAL_MANIFEST.json' if p=='dbaasp_bounded_v3' else 'MANIFEST.json')) for p in ['dbaasp_bounded_v3','chemistry_units_v1','chemistry_units_v2','witten_cpu_v1','chem_cnn_audit_v1']},'prior_delivery_status_note':'witten_cpu_v1/DELIVERY.json was frozen before independent audit and is retained. SIGNOFF.json and this index provide subsequent reviewed status.','protection':{'new_HTTP_requests':0,'natural_model_training':False,'train_dev_test_opened_this_package':False,'test37_opened':False,'H0_H4_or_manuscript_edits':False,'uploads':False,'credential_use':False,'new_dependencies':False,'pretrained_weights':False},'backup':'BACKUP_BLOCKED'})
write(r/'VERIFICATION.json',{'status':'PASS','created_utc':now(),'previous_manifest_entries_verified':len(checks),'frozen_chemistry_bindings':'PASS_BOTH_VERSIONS','independent_actual_checks_v2':605,'readiness':'Data/model preparation audited; natural training not admitted'})
files={str(p.relative_to(h)):{'sha256':sha(p),'bytes':p.stat().st_size} for package in ['chemistry_units_v1','chemistry_units_v2','witten_cpu_v1','chem_cnn_audit_v1','chemistry_cnn_delivery_v1'] for p in sorted((h/package).rglob('*')) if p.is_file() and '__pycache__' not in p.parts and p!=r/'MANIFEST.json'}
write(r/'MANIFEST.json',{'created_utc':now(),'root':str(h),'files':files,'exclusions':['__pycache__','this self-referential manifest']})
for rel,e in read(r/'MANIFEST.json')['files'].items():assert sha(h/rel)==e['sha256'],rel
print(json.dumps({'status':'PASS','delivery_files':len(files),'prior_manifest_entries_verified':len(checks),'manifest_sha256':sha(r/'MANIFEST.json')}))
