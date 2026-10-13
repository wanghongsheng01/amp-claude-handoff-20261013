import datetime,hashlib,json
from pathlib import Path
r=Path(__file__).resolve().parent
d={'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'local_sha256':{n:hashlib.sha256((r/n).read_bytes()).hexdigest() for n in ['PLAN.json','audit_convert.py','validate_synthetic.py','SYNTHETIC_CHECKS.json']},'mode':'offline chemistry audit, not model training','limits':{'records':75,'no_generic_tautomer_enumeration':True,'no_new_requests':True},'source_and_measurement_semantics':'No material-mass-basis overrides','independent_preaudit_sha256':hashlib.sha256((r.parent/'chem_cnn_audit_v1/V2_NORMALIZATION_PREAUDIT.json').read_bytes()).hexdigest()}
(r/'FREEZE.json').write_text(json.dumps(d,indent=2)+'\n')
