import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):os.environ[k]='1'
import ast,contextlib,hashlib,importlib.util,io,json,sys
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
sys.dont_write_bytecode=True
R=Path('/Users/loctek/Documents/Codex/2026-10-11/task-4/amp_h5');O=R/'review/independent';S=R/'implementation/ce_source_replacement_v1.py';C=R/'protocol/CE_SOURCE_REPLACEMENT_CONTRACT_v1.json';P=R/'protocol/CE_SOURCE_REPLACEMENT_PLAN_v1.json';OUT=O/'CE_SOURCE_REPLACEMENT_CANDIDATE_REVIEW_v1.json';assert not OUT.exists()
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();c=json.loads(C.read_text());p=json.loads(P.read_text());checks=[]
def ck(n,x):checks.append({'name':n,'pass':bool(x)})
ck('frozen_candidate_contract_plan',sha(S)=='cb14e4aad3977e6dbd8733d60a38a427886f504bd7ed8c8660915cc76b2466e4' and sha(C)=='7226435783bd6bc84f3bfb13f1fe96359b82325b2b8640a554205b09c7d917eb' and sha(P)=='c659f89641cc32b4003180b3655432213613b59f6db821f55a3ca6efa2f8d892')
ck('42_contract_bindings_48plan_bindings',len(c['input_bindings'])==42 and len(p['bindings'])==48 and all(sha(x['path'])==x['sha256'] for x in c['input_bindings']+p['bindings']))
sp=importlib.util.spec_from_file_location('candidate_synthetic_only',S);m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m)
ck('96_preallocated_slots_length0',len(m.allocated_rows())==96 and all(r['length']==0 and r['sequence']=='' and r['status']=='not_executed' and r['ranking_eligible']=='false' for r in m.allocated_rows()))
# Deterministic toy values and hand-built permutations; no natural RNG calls.
b=np.arange(16*16*48,dtype=np.float32).reshape(16,16,48);b[0,0,0]=np.float32(-0.0);b[1,0,0]=np.float32(0.0)
perm=np.empty((16,48,16),dtype=np.int64);z=np.empty_like(b)
for ch in range(16):
 for pos in range(48):perm[ch,pos]=np.roll(np.arange(16),ch+pos);z[:,ch,pos]=b[perm[ch,pos],ch,pos]
m.verify_permutation(np,b,z,perm);ck('toy_full768_coordinate_bit_margins',True)
bad=z.copy();bad[0,0,0]=np.float32(+0.0)
try:m.verify_permutation(np,b,bad,perm);rejected=False
except ValueError:rejected=True
ck('signed_zero_mapping_corruption_rejected',rejected)
badperm=perm.copy();badperm[0,0,0]=badperm[0,0,1]
try:m.verify_permutation(np,b,z,badperm);rejected=False
except ValueError:rejected=True
ck('nonbijective_permutation_rejected',rejected)
# Reproduce R-S1 without any model: simulate execute raising a post-forward/finalization exception.
original=m.execute
class SyntheticFinalizationError(Exception):pass
def fail(_):raise SyntheticFinalizationError('synthetic error representing finalization after prior/decoder work')
m.execute=fail;oldargv=sys.argv;sys.argv=['candidate','--contract',str(C),'--plan',str(P),'--receipt',str(m.RECEIPT),'--output',str(m.OUTPUT)]
buf=io.StringIO()
try:
 with contextlib.redirect_stderr(buf):exitcode=m.main()
finally:sys.argv=oldargv;m.execute=original
record=json.loads(buf.getvalue());ck('R_S1_false_zero_model_classification_reproduced',exitcode==2 and record['state']=='admission_failed_no_model' and record['model_calls']==0 and record['not_executed']==96)
text=S.read_text();tree=ast.parse(text);execute=next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=='execute');prep=next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=='prepare_inputs')
ck('correct_B_C_and_reference_g_routes_static','zb=qmu+epsilon*qsigma' in text and "permute_columns(np,zb.numpy(),c['permutation_seeds'][label])" in text and 'scaler=m.FeatureScaler.__new__' in text and 'expected[i,ch,pos]=np.float32((float(value)-st[\'mean\'][ch])/st[\'scale\'][ch])' in text)
ck('base_preparation_precedes_model_constructor',text.index('tape,metadata=prepare_inputs')<text.index("model=m.make_model"))
ck('D_final_seal_before_any_decoder',text.index("dump(out/'DECODER_INPUTS_SEAL.json'")<text.index('seqs,tokens,lengths,raw,scores=m.decode'))
ck('one_inverse_one_forward_one_decode_callsites',sum(isinstance(n,ast.Call) and ast.unparse(n.func)=='model.vae.prior_flow' for n in ast.walk(execute))==2 and sum(isinstance(n,ast.Call) and ast.unparse(n.func)=='m.decode' for n in ast.walk(execute))==1)
ck('exact_fixed_call_count_guard',"!=(6,2,2):state['state']='failed'" in text)
ck('D_no_fallback_and_tolerance_fail',"raise FloatingPointError('D roundtrip tolerance failed; no threshold relaxation')" in text and c['D_enabled'] is True and c['no_runtime_D_omission_or_A_fallback'] is True)
ck('source_manifest_intrinsic_validation',"if digest != entry[\"corrected_sha256\"]" in (R/'implementation/vae_pipeline_formal_v2.py').read_text())
ck('closing_manifest_and_exit_consistency_static',"state['inner_budget_exceeded_during_finalization']=True" in text and "summary['state']='failed'" in text and "return 0 if state['state']=='completed' else 2" in text)
ck('current_candidate_not_executed',not m.OUTPUT.exists() and not m.RECEIPT.exists())
report={'created_utc':datetime.now(timezone.utc).isoformat(),'status':'HOLD_R_S1_EXCEPTION_CLASSIFICATION','execution_authorized':False,'checks':checks,'failed_checks':[x for x in checks if not x['pass']],'blocking_findings':[{'id':'R-S1','location':'main() broad Exception around entire execute()','issue':'Any uncaught exception after real inverse/decoder work, especially finally I/O/hash export failures, is falsely labeled admission_failed_no_model and 0completed/96not_executed.','synthetic_reproduction':{'exit_code':exitcode,'reported_state':record['state'],'reported_model_calls':record['model_calls'],'reported_not_executed':record['not_executed'],'test_used_no_models':True},'required_minimal_repair':'Confine zero-model admission evidence to validate_request failure. Other uncaught execution/finalization exceptions must preserve existing evidence and mark counts unknown or explicitly last-known, not claim zero. Preserve frozen v1; new version and new plan/receipt required.'}],'nonblocking_manual_review':['B paired posterior parameters and epsilon; C actual B z column permutation with signed-zero-safe bit multiset; D same train row g with frozen scaler/positive-zero PAD and independent scalar-loop bytes.','Full global epsilon provenance and base seal precede model; D2inverse+2roundtrip precede final96decoder-input seal;6fixed decoder calls preserve all slots.','Exact binding/source-runtime validation, weights_only load, before/after state identities, no refit/encoder/DDPM, legacy gatesNA and no ranking.','120-before-final and aftermanifest success checks plus closingledger/outer180 require all final artifacts+exit to agree.','Contract/plan binds mandatoryD, scientific closeout, selector/permutation/epsilon seeds and prospective96requestledger.'],'remaining_final_requirements':['R-S1 repaired in preserved new version and exact finalplan re-bound.','Evaluator sidecar should explicitly map actual reference_train_row_id to previously named selected_train_row_id; raw metadata is retained.','No further natural input/RNG preparation or model execution before a new successful exact receipt.'],'source_bindings':[{'path':str(x),'sha256':sha(x)} for x in [S,C,P,R/'feasibility/ce_source_replacement_candidate_v1/MANIFEST.json']],'reviewer_actions':{'natural_array_reads':0,'natural_RNG_or_indices':0,'toy_random_draws':0,'model_calls':0,'numpy_deterministic_toy_only':True,'test37_reads':0}}
with OUT.open('x') as h:json.dump(report,h,ensure_ascii=False,indent=2);h.write('\n')
print(json.dumps({'path':str(OUT),'sha256':sha(OUT),'checks':len(checks),'failed_checks':report['failed_checks'],'status':report['status']}))
