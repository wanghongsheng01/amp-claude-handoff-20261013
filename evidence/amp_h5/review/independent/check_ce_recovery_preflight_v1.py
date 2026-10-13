"""Independent exact binding plus synthetic validation only; no model imports."""
from pathlib import Path
import ast,copy,datetime,hashlib,importlib.util,json,os,sys,tempfile,types
H=Path('/Users/loctek/Documents/Codex/2026-10-11/task-4/amp_h5');O=H/'review/independent'
script=H/'implementation/ce_checkpoint_recovery_v1.py';planp=H/'protocol/CE_CHECKPOINT_RECOVERY_PLAN_v1.json';contractp=H/'protocol/CE_CHECKPOINT_RECOVERY_CONTRACT_v1.json'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text())
checks=[]
def ck(name,ok,detail=None):
 checks.append({'name':name,'pass':bool(ok),**({'detail':detail} if detail is not None else {})})
 if not ok:raise AssertionError(name)
plan=read(planp);contract=read(contractp);tree=ast.parse(script.read_text())
ck('exact_announced_plan_script_contract_hash',sha(planp)=='3e66ffa415dc759e396d25d4ff9c445c7c855fb0e48aaf96b834d6da1b1ae1e4' and sha(script)=='4eb9bd3fa480fef6cc23b87925dea8e6b2ce771be0290e84d635c6e86ff0bc6c' and sha(contractp)=='bf610d590f06fa21e4d1b7fc2459ec62d3564bdd5bd32fdbe1fbdfb8e5a71a00')
ck('all22_plan_bindings_hash_valid',len(plan['bindings'])==22 and all(sha(b['path'])==b['sha256'] for b in plan['bindings']))
ck('all_core_contract_bindings_hash_valid',all(sha(b['path'])==b['sha256'] for b in contract['core_bindings']))
ck('all25_original_files_hash_and_size_valid',len(contract['original_25_file_snapshot'])==25 and all(sha(b['path'])==b['sha256'] and Path(b['path']).stat().st_size==b['bytes'] for b in contract['original_25_file_snapshot']))
actual={str(p) for d in ('original_run_directory','original_supervisor_directory') for p in Path(contract[d]).iterdir() if p.is_file()}
ck('exact_original25_file_membership',actual=={b['path'] for b in contract['original_25_file_snapshot']})
ck('new_output_and_supervisor_dirs_absent',not Path(contract['output_directory']).exists() and not Path(contract['supervisor_output_directory']).exists())
ck('one_job_exact_caps',len(plan['jobs'])==1 and plan['jobs'][0]['outer_timeout_seconds']==1980 and plan['budgets_seconds']==contract['budget_seconds']=={'training_updates':0,'generation':1800,'outer_total':1980})
ck('phase_split_provenance_and_no_original_pass',contract['phase_provenance']['training']['copied_or_linked'] is False and contract['phase_provenance']['new_training_updates']==0 and all(v=='NOT_EVALUABLE_INTERRUPTION' for k,v in contract['phase_provenance']['original_outcome'].items() if k in ('E','L','G')))
ck('checkpoint_digests_match_independent_forensics',all(contract['checkpoint_tensor_digests'][name]==read(O/'CE_V2_INTERRUPTION_FORENSIC_v2.json')['state_digests'][name] for name in ('vae_state','ddpm_ema_state')))
calls=[ast.unparse(n.func) for n in ast.walk(tree) if isinstance(n,ast.Call)]
ck('no_train_optimizer_forward_or_fit_calls',not any(x.endswith(('.train_vae','.train_ddpm','.backward','.step','.posterior_parameters','.forward','.fit')) or 'optim.' in x for x in calls))
ck('exact_frozen_generate_and_exports',calls.count('m.generate')==1 and calls.count('m.reconstruction_export')==2)
ck('two_weights_only_strict_checkpoint_loads',calls.count('m.torch.load')==2 and calls.count('m.FeatureScaler')==0 and calls.count('m.FeatureScaler.__new__')==1 and all(any(k.arg=='weights_only' and isinstance(k.value,ast.Constant) and k.value.value is True for k in n.keywords) for n in ast.walk(tree) if isinstance(n,ast.Call) and ast.unparse(n.func)=='m.torch.load'))
execute=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='execute')
ck('authorization_before_import_load_output',next(i for i,n in enumerate(execute.body) if isinstance(n,ast.Assign) and isinstance(n.value,ast.Call) and ast.unparse(n.value.func)=='validate_request') < next(i for i,n in enumerate(execute.body) if isinstance(n,ast.Expr) and isinstance(n.value,ast.Call) and ast.unparse(n.value.func)=='spec.loader.exec_module'))
# Import the wrapper only. It imports standard library and defines functions, no models.
spec=importlib.util.spec_from_file_location('independent_recovery_wrapper_validation',script);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
ck('outer_interruption_bypasses_Exception',issubclass(m.RecoveryInterrupt,BaseException) and not issubclass(m.RecoveryInterrupt,Exception))
try:m.interruption_signal(15,None)
except m.RecoveryInterrupt:ck('signal_handler_raises_expected_only',True)
else:ck('signal_handler_raises_expected_only',False)
with tempfile.TemporaryDirectory(prefix='h5_recovery_independent_',dir='/tmp') as td:
 root=Path(td).resolve()/'amp_h5';src=root/'runs/formal_v2/categorical_ce';supsrc=root/'runs/single_ce_execution_v2'
 src.mkdir(parents=True);supsrc.mkdir(parents=True);(root/'protocol').mkdir();(root/'implementation').mkdir();(root/'review/independent').mkdir(parents=True)
 for i in range(23):(src/f'file{i}.json').write_text(str(i))
 for i in range(2):(supsrc/f'file{i}.json').write_text(str(i))
 fake_sup=root/'implementation/supervise.py';fake_sup.write_text('# synthetic supervisor\n')
 fake_frozen=root/'implementation/vae_pipeline_formal_v2.py';fake_frozen.write_text('# synthetic frozen runner\n')
 for k,v in {'ROOT':root,'SOURCE':src,'SOURCE_SUPERVISOR':supsrc,'FROZEN_RUNNER':fake_frozen,'OUTPUT':root/'runs/ce_checkpoint_recovery_v1','CONTRACT':root/'protocol/CE_CHECKPOINT_RECOVERY_CONTRACT_v1.json','PLAN':root/'protocol/CE_CHECKPOINT_RECOVERY_PLAN_v1.json','RECEIPT':root/'review/independent/CE_CHECKPOINT_RECOVERY_PREFLIGHT_RECEIPT_v1.json'}.items():setattr(m,k,v)
 lock=root.parent/'H5_PROCESS.lock';lock.write_text(str(os.getpid()));m.os=types.SimpleNamespace(getppid=lambda:os.getpid())
 c=copy.deepcopy(contract)
 c.update(original_run_directory=str(src),original_supervisor_directory=str(supsrc),output_directory=str(m.OUTPUT),frozen_runner_path=str(fake_frozen),supervisor_output_directory=str(root/'runs/ce_checkpoint_recovery_execution_v1'),process_lock_path=str(lock))
 c['core_bindings']=[{'path':str(fake_frozen),'sha256':sha(fake_frozen)}]
 c['original_25_file_snapshot']=[{'path':str(p),'sha256':sha(p),'bytes':p.stat().st_size} for d in (src,supsrc) for p in d.iterdir()]
 pl={'scope_id':m.SCOPE,'root_execution_authorized':True,'workspace':str(root.parent),'contract_sha256':None,'supervisor_path':str(fake_sup),'output_root':c['supervisor_output_directory'],'bindings':[],'jobs':[{'id':'ce_checkpoint_recovery_v1','argv':[c['python_executable'],'-u','-B',str(script),'--contract',str(m.CONTRACT),'--plan',str(m.PLAN),'--receipt',str(m.RECEIPT),'--output',str(m.OUTPUT)],'outer_timeout_seconds':1980}]}
 rec={'authorized':True,'scope_id':m.SCOPE,'plan_sha256':None,'contract_sha256':None,'recovery_script_sha256':sha(script),'supervisor_sha256':sha(fake_sup)}
 def write_case(c2=None,p2=None,r2=None):
  cc=copy.deepcopy(c if c2 is None else c2);pp=copy.deepcopy(pl if p2 is None else p2);rr=copy.deepcopy(rec if r2 is None else r2)
  m.CONTRACT.write_text(json.dumps(cc));pp['contract_sha256']=sha(m.CONTRACT);pp['bindings']=[{'path':str(m.CONTRACT),'sha256':sha(m.CONTRACT)},{'path':str(script),'sha256':sha(script)}]
  m.PLAN.write_text(json.dumps(pp));rr['plan_sha256']=sha(m.PLAN);rr['contract_sha256']=sha(m.CONTRACT);m.RECEIPT.write_text(json.dumps(rr))
 def validate():return m.validate_request(str(m.CONTRACT),str(m.PLAN),str(m.RECEIPT),str(m.OUTPUT))
 def reject(name,fn):
  try:fn()
  except (ValueError,PermissionError,FileNotFoundError):ck(name,True)
  else:ck(name,False)
 write_case();validate();ck('synthetic_complete_contract_accepts_no_output',not m.OUTPUT.exists())
 for name,kind,key,value in [('unauthorized_receipt','receipt','authorized',False),('wrong_receipt_scope','receipt','scope_id','OTHER'),('no_root_authorization','plan','root_execution_authorized',False),('changed_budget','contract','budget_seconds',{'training_updates':0,'generation':1801,'outer_total':1980}),('changed_seeds','contract','sampling_seeds',[422,999]),('wrong_request_counts','contract','generation_requests',{'FREE':1,'AMD':1})]:
  obj=copy.deepcopy({'receipt':rec,'plan':pl,'contract':c}[kind]);obj[key]=value
  write_case(**{{'receipt':'r2','plan':'p2','contract':'c2'}[kind]:obj});reject(name,validate)
 write_case();rr=json.loads(m.RECEIPT.read_text());rr['plan_sha256']='0'*64;m.RECEIPT.write_text(json.dumps(rr));reject('stale_receipt_plan_hash',validate)
 write_case();lock.write_text('99999999');reject('wrong_parent_lock_identity',validate);lock.write_text(str(os.getpid()))
 write_case();m.OUTPUT.mkdir();reject('existing_empty_output',validate);m.OUTPUT.rmdir()
 write_case();extra=src/'extra.json';extra.write_text('x');reject('extra_original_file',validate);extra.unlink()
 write_case();one=src/'file0.json';old=one.read_text();one.write_text('changed');reject('original_hash_drift',validate);one.write_text(old)
 write_case();m.RECEIPT.unlink();reject('missing_receipt',validate)
 write_case();reject('noncanonical_relative_scope',lambda:m.validate_request('relative.json',str(m.PLAN),str(m.RECEIPT),str(m.OUTPUT)))
 write_case();path=root/'alias';path.symlink_to(m.CONTRACT);reject('symlink_contract_scope',lambda:m.validate_request(str(path),str(m.PLAN),str(m.RECEIPT),str(m.OUTPUT)))
 ck('no_synthetic_model_output',not m.OUTPUT.exists())
ck('no_torch_numpy_or_model_imports','torch' not in sys.modules and 'numpy' not in sys.modules)
report={'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'status':'PASS_STATIC_SYNTHETIC_AND_BINDING_REVIEW','checks':checks,'checks_passed':len(checks),'plan_sha256':sha(planp),'script_sha256':sha(script),'contract_sha256':sha(contractp),'reviewer_actions':{'natural_model_calls':0,'natural_weight_loads':0,'training_updates':0,'generation_requests':0,'source_CSV_reads':0,'test37_reads':0,'original_outputs_written':0,'synthetic_contract_fixtures':'temporary standard-library-only; deleted after validation'},'authorized':False,'note':'Execution permission requires separate exact scoped receipt; this review report is not itself a receipt.'}
out=O/'CE_RECOVERY_INDEPENDENT_PREFLIGHT_CHECKS_v1.json'
with out.open('x') as f:json.dump(report,f,ensure_ascii=False,indent=2);f.write('\n')
print(json.dumps({'status':report['status'],'checks':len(checks),'sha256':sha(out),'path':str(out)}))
