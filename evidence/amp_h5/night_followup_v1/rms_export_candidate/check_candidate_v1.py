#!/usr/bin/env python3
"""Pure stdlib synthetic/AST preparation checks. No torch, natural arrays or model construction."""
import ast,contextlib,hashlib,importlib.util,io,json,sys,tempfile
from pathlib import Path
from types import SimpleNamespace
R=Path('/Users/loctek/Documents/Codex/2026-10-11/task-4/amp_h5')
P=R/'implementation/ce_rms_reconstruction_v1.py'
sys.dont_write_bytecode=True
spec=importlib.util.spec_from_file_location('rms_synthetic',P);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
checks=[]
def check(name,value):
 if not value:raise AssertionError(name)
 checks.append({'name':name,'pass':True})
slots=m.slot_intent();rows=m.allocated_rows()
check('2880_unique_fixed_slots',len(slots)==2880 and len({x['sample_id'] for x in slots})==2880)
check('320M_2560S_fixed8',sum(x['mode']=='M' for x in slots)==320 and {x['seed'] for x in slots if x['mode']=='S'}==set(range(1422,1430)))
check('split_and_source_identity',all(x['source_row_id']==x['row_id']+(279 if x['split']=='development' else 0) for x in slots))
check('all_pending_rows_retained',all(x['status']=='not_executed' and x['prediction_finite']=='false' and x['length']==0 and json.loads(x['prediction_tokens48'])==[] for x in rows))
check('exact_frozen_slot_ledger',json.loads(m.SLOTS.read_text())['slots']==slots)
c=json.loads(m.CONTRACT.read_text())
check('contract_binding_script',c['script_sha256']==hashlib.sha256(P.read_bytes()).hexdigest())
check('99decoder_2encoder_fixed',9*((279+31)//32+(41+31)//32)==99 and c['model_calls']['encoder_batches']==2)
check('all29_input_hashes',not m.verify(c['input_bindings']))
text=P.read_text();tree=ast.parse(text)
functions={n.name:n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))}
epsilon=ast.unparse(functions['epsilon_banks'])
check('private_full320_epsilon_streams',"Generator(device='cpu')" in epsilon and '(320, 16, 48)' in epsilon and 'generator=generator' in epsilon and 'get_rng_state' in epsilon)
execute=ast.unparse(functions['execute'])
check('frozen_encoder_route_once',execute.count('m.posterior_parameters(')==1 and 'train_values' in execute)
check('no_forbidden_call_names',not any(isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr in ('train_vae','train_ddpm','generate','prior_flow','fit','backward','step') for n in ast.walk(tree)))
check('no_teacherforcing_or_inverse_call',not any(isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr in ('inverse','inverse_flow') for n in ast.walk(tree)))
check('post_manifest_clock_gate',execute.index("dump(OUTPUT / 'manifest.json'") < execute.index('closing_elapsed = time.perf_counter() - started') and 'if closing_elapsed > 120' in execute)
class T:
 def __init__(self,value,shape):self.value=value;self.shape=shape
 def numpy(self):return self.value
 def tolist(self):return self.value
saved=[]
class N:
 @staticmethod
 def savez_compressed(path,**kwargs):saved.append((str(path),kwargs))
def fake_decode(model,z,arm,observations):
 tokens=[[0]+[20]*47,[20]*48,[1]*48];lengths=[1,0,48]
 raw=T('raw',(3,21,48));observations.update(sampled_z=z,decoder_raw=raw)
 return ['A','','R'*48],T(tokens,(3,48)),T(lengths,(3,)),raw,T('scores',(3,49))
f=SimpleNamespace(decode=fake_decode,np=N,finite=lambda *a:None)
oldout=m.OUTPUT
with tempfile.TemporaryDirectory(prefix='rms_toy_') as d:
 m.OUTPUT=Path(d)
 batch=[dict(x) for x in rows[:3]];state={'decoder_forward_attempts':0};obs={'z':T('z',(3,16,48))}
 m.decode_batch(f,None,obs['z'],batch,{},Path(d)/'toy.npz','S',state,obs)
 check('toy_empty_and_cap_retained',len(batch)==3 and [x['status'] for x in batch]==['ok','empty_sequence','no_eos_at_limit'] and [x['first_pad_index'] for x in batch]==[1,0,-1])
 check('toy_tokens_finite_output',all(x['prediction_finite']=='true' and len(json.loads(x['prediction_tokens48']))==48 for x in batch))
 check('failure_observations_shared_reference','decoder_raw' in obs and state['decoder_forward_attempts']==1)
 legacy={(x['split'],x['row_id']):{'prediction_sequence':x['prediction_sequence'],'prediction_tokens48':x['prediction_tokens48'],'raw_prediction_tokens48':x['prediction_tokens48']} for x in batch}
 batch2=[dict(x) for x in rows[:3]]
 m.decode_batch(f,None,obs['z'],batch2,legacy,Path(d)/'toyM.npz','M',state,{})
 check('synthetic_M_exact_parity',all(x['prediction_sequence']==y['prediction_sequence'] for x,y in zip(batch,batch2)))
 legacy[('train',1)]['prediction_sequence']='A'
 try:m.decode_batch(f,None,obs['z'],[dict(x) for x in rows[:3]],legacy,Path(d)/'badM.npz','M',state,{})
 except ValueError:failed=True
 else:failed=False
 check('synthetic_M_parity_mismatch_fails',failed)
 m.OUTPUT=oldout
orig_execute=m.execute;origargv=sys.argv
sys.argv=['candidate','--contract','unused','--plan','unused','--receipt','unused','--output','unused']
for exc,expected in [(m.AdmissionFailure('synthetic'),'admission_failed_no_model'),(RuntimeError('synthetic_finalization'),'execution_or_finalization_failure_phase_unknown'),(m.ExportInterrupt('synthetic_interrupt'),'execution_or_finalization_failure_phase_unknown')]:
 def fail(_args,exc=exc):raise exc
 m.execute=fail;stream=io.StringIO()
 with contextlib.redirect_stderr(stream):code=m.main()
 report=json.loads(stream.getvalue())
 check('exception_'+type(exc).__name__,code==2 and report['state']==expected and report['requested']==2880 and (report['model_calls']==0 if isinstance(exc,m.AdmissionFailure) else report['model_calls'] is None))
m.execute=orig_execute;sys.argv=origargv
out={'status':'PASS','checks':checks,'check_count':len(checks),'natural_model_calls':0,'torch_imported':False,'natural_RNG_or_arrays':False,'scope':'stdlib fake tensors and AST only; verifies no claim of scientific outcome'}
path=R/'night_followup_v1/rms_export_candidate/SYNTHETIC_CHECKS.json'
if path.exists():raise FileExistsError(path)
path.write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps({'status':'PASS','checks':len(checks),'output':str(path)}))
