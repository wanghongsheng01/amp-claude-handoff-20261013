import copy,hashlib,importlib.util,json,os,sys
from pathlib import Path
from types import SimpleNamespace
from datetime import datetime,timezone
sys.dont_write_bytecode=True
R=Path('/Users/loctek/Documents/Codex/2026-10-11/task-4/amp_h5');O=R/'review/independent';P=R/'protocol/CE_SOURCE_REPLACEMENT_PLAN_v2.json';C=R/'protocol/CE_SOURCE_REPLACEMENT_CONTRACT_v2.json';S=R/'implementation/ce_source_replacement_v2.py';OUT=O/'CE_SOURCE_REPLACEMENT_FINAL_PREFLIGHT_CHECKS_v2.json';assert not OUT.exists()
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();read=lambda p:json.loads(Path(p).read_text());p,c=read(P),read(C);checks=[]
def ck(n,x):checks.append({'name':n,'pass':bool(x)})
ck('exact_plan_contract_script',sha(P)=='91e64d4ca473c1a56b2c66b285f0f5b8fd0ccba3884fe15ba465e574108bb46f' and sha(C)=='7fcc1352f1902ea2ae2d72d8803f9a94defee97452b0a40a969e34c081e22ec8' and sha(S)=='e915f23cbf6157b20bd1d9a94fe831d326b0ab6a20b4013824224921a2bbf444')
ck('all53_unique_plan_bindings',len(p['bindings'])==len({x['path'] for x in p['bindings']})==53 and all(sha(x['path'])==x['sha256'] for x in p['bindings']))
ck('all47_unique_contract_bindings',len(c['input_bindings'])==len({x['path'] for x in c['input_bindings']})==47 and all(sha(x['path'])==x['sha256'] and Path(x['path']).stat().st_size==x['bytes'] for x in c['input_bindings']))
bound={x['path']:x['sha256'] for x in p['bindings']};ck('all_contract_inputs_present_in_plan',all(bound.get(x['path'])==x['sha256'] for x in c['input_bindings']))
for rel in ['feasibility/ce_source_replacement_candidate_v2/MANIFEST.json','protocol/ce_source_replacement_evaluation_v2/PREPARATION_MANIFEST.json']:
 j=read(R/rel);ck(rel+'_members_hashes',all(sha(x['path'])==x['sha256'] and ('bytes' not in x or Path(x['path']).stat().st_size==x['bytes']) for x in j['files']))
repair=O/'CE_SOURCE_REPLACEMENT_R_S1_REPAIR_REVIEW_v2.json';j=read(repair);ck('independent_R_S1_resolved',sha(repair)=='15b3c5023d82dd905c0f518799ed17a19ab297fdbb6f0130b12748651750bba6' and j['status']=='PASS_R_S1_REPAIR_EXACT_V2_SCRIPT_ONLY' and not j['failed_checks'])
sp=importlib.util.spec_from_file_location('independent_final_candidate',S);m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m)
ck('96_slots_match_exact_manifest',read(m.SLOTS)['rows']==m.allocated_rows() and len(m.allocated_rows())==96)
seeds=read(m.SEEDS);ck('four_method_epsilon_schedule',seeds=={key:{'FREE':422,'AMD':423} for key in [m.REUSED,*m.METHODS]})
ck('candidate_paths_no_existing_runs',not m.OUTPUT.exists() and not m.SUPERVISOR_OUTPUT.exists() and not m.RECEIPT.exists() and not (R/'runs/ce_source_replacement_v1').exists())
ck('single_exact180_job',len(p['jobs'])==1 and p['jobs'][0]['outer_timeout_seconds']==180 and p['jobs'][0]['id']=='ce_source_replacement_v2')
ck('frozen_caps_callcounts',c['budgets_seconds']=={'total_inner':120,'outer':180} and c['slots']=={'paired_slots':32,'per_terminal':16,'new_decoder_inputs':96,'decoder_batches':6,'D_inverse_batches':2,'D_roundtrip_forward_batches':2,'reused_DDPM_rows':32})
ck('root_scope_D_mandatory',p['scope_id']==c['scope_id']=='AMP_H5_CE_SOURCE_REPLACEMENT_V2' and p['root_execution_authorized'] is True and p['D_enabled'] is True and c['D_enabled'] is True)
# Mock only prospective receipt/parent PID for admission unit checks. No run/receipt/lock is written.
r={'authorized':True,'scope_id':m.SCOPE,'four_arm_scientific_review_checked':True,'scalar_terminal_review_checked':True,'plan_sha256':sha(P),'contract_sha256':sha(C),'control_script_sha256':sha(S),'supervisor_sha256':sha(p['supervisor_path']),'launcher_sha256':sha(p['launcher_path']),'scientific_review_sha256':c['scientific_review']['sha256']}
original_read=m.read;original_pid=os.getppid;lock_pid=int((R.parent/'H5_PROCESS.lock').read_text().strip());os.getppid=lambda:lock_pid
memory={m.CONTRACT:c,m.PLAN:p,m.RECEIPT:r};m.read=lambda path:copy.deepcopy(memory[Path(path)]) if Path(path) in memory else original_read(path)
args=SimpleNamespace(contract=str(m.CONTRACT),plan=str(m.PLAN),receipt=str(m.RECEIPT),output=str(m.OUTPUT))
try:
 m.validate_request(args);ck('synthetic_valid_exact_admission',True)
 def reject(name,path,key,value):
  old=memory[path][key];memory[path][key]=value
  try:m.validate_request(args);v=False
  except (ValueError,PermissionError):v=True
  finally:memory[path][key]=old
  ck(name,v)
 reject('unapproved_receipt_refused',m.RECEIPT,'authorized',False)
 reject('unchecked_science_refused',m.RECEIPT,'four_arm_scientific_review_checked',False)
 reject('unchecked_scalar_refused',m.RECEIPT,'scalar_terminal_review_checked',False)
 reject('stale_plan_hash_refused',m.RECEIPT,'plan_sha256','0'*64)
 reject('runtime_D_disabled_refused',m.CONTRACT,'D_enabled',False)
 reject('changed_scope_refused',m.PLAN,'scope_id','OLD_SCOPE')
 reject('changed_inner_budget_refused',m.CONTRACT,'budgets_seconds',{'total_inner':121,'outer':180})
 reject('changed_roundtrip_tolerance_refused',m.CONTRACT,'roundtrip_tolerance',{'atol':1e-4,'rtol':1e-5})
 reject('changed_permutation_seed_refused',m.CONTRACT,'permutation_seeds',[625,627])
 reject('changed_command_refused',m.PLAN,'jobs',[dict(p['jobs'][0],outer_timeout_seconds=181)])
 os.getppid=lambda:lock_pid+1
 try:m.validate_request(args);v=False
 except PermissionError:v=True
 ck('wrong_parent_lock_refused',v)
finally:m.read=original_read;os.getppid=original_pid
E=read(R/'protocol/ce_source_replacement_evaluation_v2/EVALUATION_PREPARATION.json')
ck('evaluation_current_v2_roots',E['future_run_root']==str(m.OUTPUT) and E['sampling_seed_metadata']['path']==str(m.SEEDS) and E['slots']['total_observations']==128)
ck('evaluation_no_original_gate_or_ranking','NOT_APPLICABLE_SOURCE_DIAGNOSTIC' in E['legacy_gate_artifact_interpretation'] and E['PROMOTE']=='HOLD' and c['interpretation']['new_sources_ranking_eligible'] is False)
ck('receipt_test_no_natural_effects','torch' not in sys.modules and 'numpy' not in sys.modules and not m.OUTPUT.exists() and not m.RECEIPT.exists())
result={'created_utc':datetime.now(timezone.utc).isoformat(),'status':'PASS_EXACT_V2_PREFLIGHT' if all(x['pass'] for x in checks) else 'HOLD','checks':checks,'failed_checks':[x for x in checks if not x['pass']],'scope_id':m.SCOPE,'plan_sha256':sha(P),'contract_sha256':sha(C),'script_sha256':sha(S),'reviewer_actions':{'stdlib_only':True,'synthetic_admission_checks_only':True,'natural_array_reads':0,'natural_RNG_or_indices':0,'model_calls':0,'test37_reads':0},'source_bindings':[{'path':str(x),'sha256':sha(x)} for x in [P,C,S,repair]],'requires_fresh_root_process_load_check_before_launch':True}
with OUT.open('x') as h:json.dump(result,h,ensure_ascii=False,indent=2);h.write('\n')
print(json.dumps({'path':str(OUT),'sha256':sha(OUT),'checks':len(checks),'failed_checks':result['failed_checks'],'status':result['status']}))
