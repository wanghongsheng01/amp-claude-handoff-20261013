import hashlib,json,datetime
from pathlib import Path
r=Path(__file__).resolve().parent;h=r.parent;sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
files=['prepare.py','PROTOCOL.json','DATA.jsonl','FOLDS.json','model.py','run.py','validate_engineering.py','ENGINEERING_TESTS.json','PREFLIGHT.json','freeze.py']
sources=['native_unit_admission_v1/ROW_ADMISSION.jsonl','native_unit_admission_v1/independent/SIGNOFF.json','witten_cpu_v1/model.py','chem_cnn_audit_v1/SIGNOFF.json','dbaasp_bounded_v3/prepared/peptides.jsonl']
assert sha(r/'model.py')==sha(h/'witten_cpu_v1/model.py')
f={'schema':'AMP_H5_NATIVE_BASELINE_FREEZE_V1','created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'local_sha256':{n:sha(r/n)for n in files},'source_sha256':{n:sha(h/n)for n in sources},'execution':'Await independent engineering PASS binding this freeze; no natural baseline has run','limits':{'wall_seconds':600,'CPU_threads':1,'total_CNN_updates':1700,'peak_RSS_bytes':1500000000},'seeds':[1729]}
(r/'FREEZE.json').write_text(json.dumps(f,indent=2)+'\n')
print(sha(r/'FREEZE.json'))
