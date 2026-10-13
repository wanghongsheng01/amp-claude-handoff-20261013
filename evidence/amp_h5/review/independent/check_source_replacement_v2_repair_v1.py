import ast,contextlib,hashlib,importlib.util,io,json,sys,tempfile
from pathlib import Path
from datetime import datetime,timezone
sys.dont_write_bytecode=True
R=Path('/Users/loctek/Documents/Codex/2026-10-11/task-4/amp_h5');O=R/'review/independent';A=R/'implementation/ce_source_replacement_v1.py';B=R/'implementation/ce_source_replacement_v2.py';OUT=O/'CE_SOURCE_REPLACEMENT_R_S1_REPAIR_REVIEW_v2.json';assert not OUT.exists()
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();checks=[]
def ck(n,x):checks.append({'name':n,'pass':bool(x)})
ck('v1_frozen_unchanged',sha(A)=='cb14e4aad3977e6dbd8733d60a38a427886f504bd7ed8c8660915cc76b2466e4');ck('v2_exact_script',sha(B)=='e915f23cbf6157b20bd1d9a94fe831d326b0ab6a20b4013824224921a2bbf444')
a,b=ast.parse(A.read_text()),ast.parse(B.read_text());fa={x.name:x for x in a.body if isinstance(x,ast.FunctionDef)};fb={x.name:x for x in b.body if isinstance(x,ast.FunctionDef)}
ck('unchanged_all_other_functions',set(fa)==set(fb) and all(ast.dump(fa[k])==ast.dump(fb[k]) for k in fa if k not in ('execute','main')))
ck('execute_all_post_admission_AST_unchanged',[ast.dump(x) for x in fa['execute'].body[2:]]==[ast.dump(x) for x in fb['execute'].body[2:]])
tr=fb['execute'].body[1];ck('only_validate_request_inside_admission_try',isinstance(tr,ast.Try) and len(tr.body)==1 and isinstance(tr.body[0],ast.Assign) and ast.unparse(tr.body[0].value)=='validate_request(args)' and len(tr.handlers)==1 and ast.unparse(tr.handlers[0].type)=='Exception')
sp=importlib.util.spec_from_file_location('independent_v2_exception_synthetic',B);m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m)
original_exec,original_validate=m.execute,m.validate_request;oldargv=sys.argv;sys.argv=['scope','--contract',str(m.CONTRACT),'--plan',str(m.PLAN),'--receipt',str(m.RECEIPT),'--output',str(m.OUTPUT)]
def invoke():
 buf=io.StringIO()
 with contextlib.redirect_stderr(buf):code=m.main()
 return code,json.loads(buf.getvalue())
def failadmission(_):raise ValueError('synthetic admission only')
m.validate_request=failadmission;code,r=invoke();ck('true_admission_zero_calls96pending',code==2 and r['state']=='admission_failed_no_model' and r['model_calls']==0 and r['not_executed']==96 and len(r['all_requested_slots'])==96);m.validate_request=original_validate
with tempfile.TemporaryDirectory(prefix='amp_h5_source_exception_toy_') as directory:
 original_output=m.OUTPUT;m.OUTPUT=Path(directory);status={'state':'completed','decoder_forward_attempts':2,'prior_inverse_attempts':2,'completed_inputs':32};(m.OUTPUT/'status.json').write_text(json.dumps(status))
 def finalfail(_):raise OSError('synthetic finalization after persisted work')
 m.execute=finalfail;code,r=invoke();ck('postwork_failure_not_zeroed',code==2 and r['state']=='execution_or_finalization_failure_phase_unknown' and all(r[k] is None for k in ('model_calls','completed','failed','not_executed')))
 ck('last_persisted_counts_preserved_but_not_final',r['last_persisted_status']==status and r['persisted_counts_may_be_stale'] is True and r['persisted_completed_is_not_success'] is True and r['last_persisted_status_sha256']==sha(m.OUTPUT/'status.json'))
 (m.OUTPUT/'status.json').write_text('synthetic broken JSON');code,r=invoke();ck('corrupt_status_unknown_failure',code==2 and r['model_calls'] is None and 'last_persisted_status_read_error' in r)
 (m.OUTPUT/'status.json').unlink();code,r=invoke();ck('missing_status_unknown_failure',code==2 and r['model_calls'] is None and 'last_persisted_status_unavailable' in r)
 def interrupt(_):raise m.DiagnosticInterrupt('synthetic signal in finalization')
 m.execute=interrupt;code,r=invoke();ck('signal_unknown_not_admission',code==2 and r['state']=='execution_or_finalization_failure_phase_unknown' and r['model_calls'] is None);m.OUTPUT=original_output
m.execute=original_exec;sys.argv=oldargv
ck('no_ML_imports_or_real_output','torch' not in sys.modules and 'numpy' not in sys.modules and not m.OUTPUT.exists())
report={'created_utc':datetime.now(timezone.utc).isoformat(),'status':'PASS_R_S1_REPAIR_EXACT_V2_SCRIPT_ONLY' if all(x['pass'] for x in checks) else 'HOLD','execution_authorized':False,'checks':checks,'failed_checks':[x for x in checks if not x['pass']],'finding_resolution':'Only validate_request errors now authorize zero-model admission accounting. All other escaped execution/finalization/signal errors retain unknown true counters and optional explicitly stale saved evidence; exit2 cannot be called final success.','unchanged_scientific_scope':'All scientific/input/tensor/decoder functions and execute body after admission are AST-identical; versions change paths/scope only.','pending':'Exact finalv2 contract, evaluation binding and root plan still require final hash review and new receipt.','source_bindings':[{'path':str(p),'sha256':sha(p)} for p in [A,B,O/'CE_SOURCE_REPLACEMENT_CANDIDATE_REVIEW_v1.json']],'reviewer_actions':{'stdlib_only':True,'synthetic_exception_cases':5,'natural_data_reads':0,'natural_RNG_or_indices':0,'model_imports':0,'model_calls':0,'test37_reads':0},'transport_recovery':'Prior cancelled tool call left neither this script nor report; existence checked before this first actual execution.'}
with OUT.open('x') as f:json.dump(report,f,ensure_ascii=False,indent=2);f.write('\n')
print(json.dumps({'path':str(OUT),'sha256':sha(OUT),'checks':len(checks),'failed_checks':report['failed_checks']}))
