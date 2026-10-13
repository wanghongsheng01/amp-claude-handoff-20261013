#!/usr/bin/env python3
"""Only stdlib exception-injection and AST equivalence; no natural/model work."""
import sys
sys.dont_write_bytecode=True
import ast,contextlib,hashlib,importlib.util,io,json,tempfile
from pathlib import Path
R=Path(__file__).resolve().parents[2];old=R/'implementation/ce_source_replacement_v1.py';new=R/'implementation/ce_source_replacement_v2.py'
spec=importlib.util.spec_from_file_location('v2_exception_test',new);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
checks={}
def check(key,value):
 checks[key]=bool(value)
 if not value:raise AssertionError(key)
def functions(p):return {x.name:x for x in ast.parse(p.read_text()).body if isinstance(x,ast.FunctionDef)}
f1,f2=functions(old),functions(new)
check('no_existing_function_added_or_removed',set(f1)==set(f2))
for name in sorted(set(f1)-{'execute','main'}):check('unchanged_function_'+name,ast.dump(f1[name],include_attributes=False)==ast.dump(f2[name],include_attributes=False))
check('execute_body_after_admission_unchanged',ast.dump(ast.Module(body=f1['execute'].body[2:],type_ignores=[]),include_attributes=False)==ast.dump(ast.Module(body=f2['execute'].body[2:],type_ignores=[]),include_attributes=False))
newtry=f2['execute'].body[1]
check('AdmissionFailure_wrap_contains_only_validate_request',isinstance(newtry,ast.Try) and len(newtry.body)==1 and isinstance(newtry.body[0],ast.Assign) and isinstance(newtry.body[0].value,ast.Call) and newtry.body[0].value.func.id=='validate_request')
original_execute=m.execute;original_validate=m.validate_request;original_output=m.OUTPUT;original_argv=sys.argv[:]
sys.argv=[str(new),'--contract',str(m.CONTRACT),'--plan',str(m.PLAN),'--receipt',str(m.RECEIPT),'--output',str(m.OUTPUT)]
def raise_error(exc):raise exc
def call_main():
 stream=io.StringIO()
 with contextlib.redirect_stderr(stream):code=m.main()
 return code,json.loads(stream.getvalue())
try:
 m.validate_request=lambda args:raise_error(ValueError('synthetic admission denied'))
 code,result=call_main()
 check('admission_declares_zero_calls_only_before_execution',code==2 and result['state']=='admission_failed_no_model' and result['model_calls']==0 and result['requested']==96 and result['not_executed']==96 and len(result['all_requested_slots'])==96)
 with tempfile.TemporaryDirectory(prefix='amp_source_exception_toy_',dir='/tmp') as td:
  m.OUTPUT=Path(td);status=m.OUTPUT/'status.json'
  status.write_text(json.dumps({'state':'completed','phase':'fixed_decoder_calls','decoder_forward_attempts':2,'prior_inverse_attempts':2,'prior_roundtrip_forward_attempts':2,'completed_inputs':32}))
  m.execute=lambda args:raise_error(OSError('synthetic post-model finalization error'))
  code,result=call_main()
  check('finalization_failure_does_not_reset_calls_or_denominators',code==2 and result['state']=='execution_or_finalization_failure_phase_unknown' and result['requested']==96 and result['model_calls'] is None and result['completed'] is None and result['not_executed'] is None)
  check('preserves_trustworthy_last_record_but_marks_stale',result['last_persisted_status']['decoder_forward_attempts']==2 and result['last_persisted_status']['completed_inputs']==32 and result['persisted_counts_may_be_stale'] and result['persisted_completed_is_not_success'])
  check('last_record_hash_exact',result['last_persisted_status_sha256']==hashlib.sha256(status.read_bytes()).hexdigest())
  status.write_text('{broken')
  code,result=call_main()
  check('corrupt_status_keeps_actual_counts_unknown',code==2 and result['model_calls'] is None and 'last_persisted_status_read_error' in result and result['state']=='execution_or_finalization_failure_phase_unknown')
  status.unlink();code,result=call_main()
  check('missing_status_not_admission_failure',code==2 and result['model_calls'] is None and result['last_persisted_status_unavailable']=='not present; actual counts unknown')
  m.execute=lambda args:raise_error(m.DiagnosticInterrupt('synthetic post-model interrupt'))
  code,result=call_main()
  check('post_admission_DiagnosticInterrupt_not_zero_call_claim',code==2 and result['state']=='execution_or_finalization_failure_phase_unknown' and result['model_calls'] is None)
finally:m.execute=original_execute;m.validate_request=original_validate;m.OUTPUT=original_output;sys.argv=original_argv
check('old_v1_script_preserved',hashlib.sha256(old.read_bytes()).hexdigest()=='cb14e4aad3977e6dbd8733d60a38a427886f504bd7ed8c8660915cc76b2466e4')
oldslots=json.loads((R/'protocol/CE_SOURCE_REPLACEMENT_REQUEST_SLOTS_v1.json').read_text());newslots=json.loads(m.SLOTS.read_text())
check('96_row_values_identical_after_version_change',oldslots['rows']==newslots['rows'])
check('seed_file_bytes_unchanged',m.SEEDS.read_bytes()==(R/'protocol/CE_SOURCE_REPLACEMENT_SAMPLING_SEEDS_v1.json').read_bytes())
check('all8_scope_paths_changed_to_v2',all('v2' in str(getattr(m,k)) for k in ('CONTRACT','PLAN','RECEIPT','OUTPUT','SUPERVISOR_OUTPUT','SEEDS','SLOTS')) and m.SCOPE.endswith('_V2'))
check('new_run_not_created',not m.OUTPUT.exists() and not m.SUPERVISOR_OUTPUT.exists())
report={'status':'PASS','checks_count':len(checks),'checks':checks,'v1_sha256':hashlib.sha256(old.read_bytes()).hexdigest(),'v2_sha256':hashlib.sha256(new.read_bytes()).hexdigest(),'natural_data_or_latents_read':False,'new_natural_RNG':False,'model_imports':0,'model_calls':0,'synthetic_exception_injections':5,'old_v1_blocker':'R-S1 broad main exception mislabeled post-admission failure as zero model admission failure; never executed'}
p=Path(__file__).with_name('NARROW_REPAIR_VALIDATION.json');assert not p.exists();p.write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n');print(json.dumps({'status':'PASS','checks':len(checks),'v2_sha256':report['v2_sha256']}))
