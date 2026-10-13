#!/usr/bin/env python3
import ast, hashlib, json, struct
from pathlib import Path
from datetime import datetime, timezone
R=Path('/Users/loctek/Documents/Codex/2026-10-11/task-4/amp_h5')
out=R/'review/independent/CE_DECODER_INPUT_CONTROL_PREPARATION_REVIEW_v2.json'
assert not out.exists()
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
p1=R/'implementation/ce_decoder_input_control_v1.py';p2=R/'implementation/ce_decoder_input_control_v2.py'
c1=R/'protocol/CE_DECODER_INPUT_CONTROL_CONTRACT_v1.json';c2=R/'protocol/CE_DECODER_INPUT_CONTROL_CONTRACT_v2.json'
a,b=p1.read_text(),p2.read_text();j1,j2=json.loads(c1.read_text()),json.loads(c2.read_text())
checks=[]
def check(n,x):checks.append({'name':n,'pass':bool(x)})
check('old_candidate_preserved',sha(p1)=='c3fadf38b1a4f7fed9b53c65e4b18905d3f4ed3744350e1575af0cb724513324' and sha(c1)=='01687cab73e6dd90843002e5f8eb764fd3241e1fdcb8bad4506a9d34d6ab16e1')
check('new_candidate_exact_hash',sha(p2)=='30211ed157ffcda9bf2f96620599beca9e48def87631cb7e4f3885cb0292fd98' and sha(c2)=='32ed71685ec5c988c76b7aca15d281f68cb882f7d753511fecc06c81ba348b86')
expected=a.replace('CONTROL_V1','CONTROL_V2').replace('CONTROL_CONTRACT_v1','CONTROL_CONTRACT_v2').replace('CONTROL_PLAN_v1','CONTROL_PLAN_v2').replace('CONTROL_PREFLIGHT_RECEIPT_v1','CONTROL_PREFLIGHT_RECEIPT_v2').replace('ce_decoder_input_control_v1','ce_decoder_input_control_v2').replace('ce_decoder_input_control_execution_v1','ce_decoder_input_control_execution_v2').replace("('completed','failed','interrupted')","('completed','completed_with_failures','failed','interrupted')").replace('not torch.equal(mu+epsilon*sigma,saved_z)','not tensor_bitwise_equal(mu+epsilon*sigma,saved_z)').replace('\ndef prepare_inputs(', '\ndef tensor_bitwise_equal(left,right):\n    return left.dtype==right.dtype and tuple(left.shape)==tuple(right.shape) and digest_tensor(left)==digest_tensor(right)\n\ndef prepare_inputs(')
check('only_two_repairs_and_version_paths_ast',ast.dump(ast.parse(expected))==ast.dump(ast.parse(b)))
keys={'created_utc','output_directory','supervisor_output_directory','scope_id','script_sha256','revision'}
check('all_other_contract_fields_unchanged',{k:v for k,v in j1.items() if k not in keys}=={k:v for k,v in j2.items() if k not in keys})
check('110_input_bindings_unchanged',len(j2['input_bindings'])==110 and j1['input_bindings']==j2['input_bindings'])
tree=ast.parse(b);helper=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='tensor_bitwise_equal');ns={'digest_tensor':lambda t:hashlib.sha256(t.data).hexdigest()};exec(compile(ast.fix_missing_locations(ast.Module(body=[helper],type_ignores=[])),str(p2),'exec'),ns)
class TensorStub:
 def __init__(self,data,dtype='float32',shape=(1,)):self.data,self.dtype,self.shape=data,dtype,shape
f=ns['tensor_bitwise_equal'];p=TensorStub(struct.pack('<f',0.0));m=TensorStub(struct.pack('<f',-0.0))
check('signed_zero_bytes_distinct',not f(p,m));check('identical_bytes_accepted',f(p,TensorStub(p.data)))
check('dtype_distinct_rejected',not f(p,TensorStub(p.data,'int32')));check('shape_distinct_rejected',not f(p,TensorStub(p.data,shape=(1,1))))
validate=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='validate_request')
guard=next(n for n in validate.body if isinstance(n,ast.If) and 'scalar.get' in ast.unparse(n.test))
test=compile(ast.Expression(guard.test),str(p2),'eval')
for state in ['completed','completed_with_failures','failed','interrupted']:
 check('terminal_'+state,not eval(test,{'scalar':{'state':state,'jobs':[]}}))
check('running_rejected',eval(test,{'scalar':{'state':'running','jobs':[]}}));check('running_child_rejected',eval(test,{'scalar':{'state':'completed','jobs':[{'state':'running'}]}}))
check('future_plan_receipt_outputs_absent',all(not (R/x).exists() for x in ['protocol/CE_DECODER_INPUT_CONTROL_PLAN_v2.json','review/independent/CE_DECODER_INPUT_CONTROL_PREFLIGHT_RECEIPT_v2.json','runs/ce_decoder_input_control_v2']))
check('new_control_still_requires_scalar_review_receipt',"r.get('scalar_terminal_review_checked') is not True" in b and "for path in (CONTRACT,Path(__file__).resolve(),SCALAR_STATUS,SCALAR_REVIEW)" in b)
failed=[x['name'] for x in checks if not x['pass']]
report={'created_utc':datetime.now(timezone.utc).isoformat(),'status':'PASS_NARROW_STATIC_REPAIRS_ONLY' if not failed else 'HOLD','execution_authorized':False,'checks':checks,'failed_checks':failed,'resolved_findings':['R-C1 dtype/shape/raw-byte SHA equality rejects signed zero differences before model construction.','R-C2 actual completed_with_failures enum accepted only as terminal, independent scope receipt still required.'],'scope_limitations':['Old A/B64 candidate is not authorized: latest user scope requires completion and independent review of all four arms, then a separate unified source-replacement protocol.','A remains decoder-space standard normal, not native flow prior; B remains empirical train posterior reference.','No natural arrays or live scalar output read, no torch/numpy imported, no RNG/model/forward/training, no test37.'],'bindings':[{'path':str(p),'sha256':sha(p)} for p in [p1,p2,c1,c2,R/'review/independent/CE_DECODER_INPUT_CONTROL_PREPARATION_REVIEW_v1.json']],'reviewer_actions':{'stdlib_only':True,'natural_tensor_loads':0,'random_draws':0,'model_calls':0,'live_scalar_reads':0,'test37_reads':0}}
out.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'report':str(out),'sha256':sha(out),'checks':len(checks),'failed':failed}))
