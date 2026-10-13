#!/usr/bin/env python3
"""Toy arrays, exact byte guards, frozen CSV reader; no natural arrays/model imports."""
import sys
sys.dont_write_bytecode=True
import ast,csv,hashlib,importlib.util,json,tempfile
from pathlib import Path
import numpy as np
R=Path(__file__).resolve().parents[2]
S=R/'implementation/ce_source_replacement_v2.py'
spec=importlib.util.spec_from_file_location('candidate_under_test',S);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
checks={}
def check(key,value):
 checks[key]=bool(value)
 if not value:raise AssertionError(key)
rows=m.allocated_rows()
check('96_fixed_unique_slots',len(rows)==96 and len({r['sample_id'] for r in rows})==96)
check('three_methods_two_terminals_16_each',all(sum(r['method']==method and r['terminal']==t for r in rows)==16 for method in m.METHODS for t in m.TERMINALS))
check('preallocation_empty_length_zero_all',all(r['sequence']=='' and r['length']==0 and r['status']=='not_executed' for r in rows))
check('D_32_extra_condition_no_ranking',sum(r['method']==m.METHODS[2] and r['condition_scope']=='reference_g_extra_residue_and_length' and r['ranking_eligible']=='false' for r in rows)==32)
check('prelocked_slot_manifest_exact',m.read(m.SLOTS)['rows']==rows)
check('sampling_seeds_epsilon_only',m.read(m.SEEDS)=={method:{'FREE':422,'AMD':423} for method in (m.REUSED,*m.METHODS)})
# Pure synthetic coordinate bank. It contains signed zeros, no biological records.
zb=np.arange(16*16*48,dtype=np.float32).reshape(16,16,48)
zb[0,0,0]=np.float32(0.0);zb[1,0,0]=np.float32(-0.0)
zc,p,before,after=m.permute_columns(np,zb,12345)
check('permutation_shape_and_dtype',p.shape==(16,48,16) and p.dtype==np.int64)
check('all768_coordinate_multisets_and_raw_mapping',m.verify_permutation(np,zb,zc,p) is None)
zc2,p2,before2,after2=m.permute_columns(np,zb,12345)
check('toy_PCG64_replay_exact',zc.tobytes()==zc2.tobytes() and p.tobytes()==p2.tobytes() and before==before2 and after==after2)
try:
 bad=p.copy();bad[0,0,1]=bad[0,0,0];m.verify_permutation(np,zb,zc,bad);caught=False
except ValueError:caught=True
check('reject_nonbijective_permutation',caught)
try:
 bad=zc.copy();bad[0,1,1]+=1;m.verify_permutation(np,zb,bad,p);caught=False
except ValueError:caught=True
check('reject_marginal_or_mapping_mutation',caught)
class FakeTensor:
 def __init__(self,a):self.a=a;self.dtype=a.dtype;self.shape=a.shape
 def detach(self):return self
 def cpu(self):return self
 def contiguous(self):return self
 def numpy(self):return self.a
check('bitwise_guard_rejects_signed_zero',not m.tensor_bitwise_equal(FakeTensor(np.array([0.],np.float32)),FakeTensor(np.array([-0.],np.float32))))
check('bitwise_guard_rejects_dtype',not m.tensor_bitwise_equal(FakeTensor(np.array([1],np.int32)),FakeTensor(np.array([1.],np.float32))))
check('bitwise_guard_rejects_shape',not m.tensor_bitwise_equal(FakeTensor(np.array([1.],np.float32)),FakeTensor(np.array([[1.]],np.float32))))
# Exercise the actual frozen evaluator reader with all128 slots and failed/pending rows.
espec=importlib.util.spec_from_file_location('frozen_reader',R/'implementation/evaluate_h5.py');e=importlib.util.module_from_spec(espec);espec.loader.exec_module(e)
with tempfile.TemporaryDirectory(prefix='amp_h5_source_synthetic_',dir='/tmp') as td:
 td=Path(td);control=[dict(r) for r in rows];reuse=[]
 for label,t in enumerate(m.TERMINALS):
  for i in range(16):reuse.append({'sample_id':f'{m.REUSED}_{t}_{i:03d}','method':m.REUSED,'terminal':t,'terminal_sample_id':i,'seed':422+label,'sequence':'ACD','status':'ok','length':3})
 statuses=('not_executed','failed','empty_sequence','ok','no_eos_at_limit')
 for i,r in enumerate(control):
  r['status']=statuses[i%len(statuses)]
  r['sequence']='ACD' if r['status']=='ok' else 'A'*48 if r['status']=='no_eos_at_limit' else ''
  r['length']=len(r['sequence'])
 m.csv_write(td/'generated.csv',control,m.FIELDS);m.csv_write(td/'reused.csv',reuse,list(reuse[0]))
 groups=e.sample_groups([td/'generated.csv',td/'reused.csv'],[m.REUSED,*m.METHODS],16,m.read(m.SEEDS))
 check('frozen_reader_all128_slots_with_fail_pending_empty',len(groups)==8 and sum(map(len,groups.values()))==128)
 check('frozen_reader_retains_all_noncompleted',sum(r['status'] in ('failed','not_executed') for g in groups.values() for r in g)==sum(r['status'] in ('failed','not_executed') for r in control))
source=S.read_text();tree=ast.parse(source)
check('module_top_level_stdlib_only',all(n.module in (None,'__future__','pathlib') for n in tree.body if isinstance(n,ast.ImportFrom)) and all(n.names[0].name in ('time','argparse','csv','hashlib','importlib.util','json','os','signal','sys','traceback') for n in tree.body if isinstance(n,ast.Import)))
check('flow_call_sites_exact_inverse_and_forward',source.count('model.vae.prior_flow(')==2 and 'reverse=True' in source and 'reverse=False' in source)
check('decoder_single_call_site_with_fixed_double_loop',source.count('m.decode(model,')==1 and 'for method in METHODS:' in source and 'for terminal in TERMINALS:' in source)
check('no_training_encoder_or_scaler_fit_call',not any(isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr in ('encoder','fit','train_vae','train_ddpm','backward','step') for n in ast.walk(tree)))
check('guard_all_source_inputs_before_model_construction',source.index('tape,metadata=prepare_inputs(')<source.index("model=m.make_model('categorical_ce'"))
check('D_roundtrip_and_final_seal_before_decode',source.index("dump(out/'DECODER_INPUTS_SEAL.json'")<source.index('seqs,tokens,lengths,raw,scores=m.decode('))
check('root_all_D_on_and_scientific_receipt_required',"c['D_enabled'] is not True" in source and "four_arm_scientific_review_checked" in source and "scientific_review_sha256" in source)
check('admission_failure_preserves96_and_zero_models',"'requested':96,'completed':0,'failed':0,'not_executed':96" in source and "'all_requested_slots':allocated_rows(),'model_calls':0" in source)
check('closing_overrun_forces_failed_and_exit2',"if closed_elapsed>120:" in source and "state['state']='failed';state['inner_budget_exceeded_during_finalization']=True" in source and "return 0 if state['state']=='completed' else 2" in source)
check('no_natural_output_created',not m.OUTPUT.exists() and not m.SUPERVISOR_OUTPUT.exists())
report={'status':'PASS','checks':checks,'checks_count':len(checks),'candidate_script_sha256':hashlib.sha256(S.read_bytes()).hexdigest(),
        'numpy_version':np.__version__,'natural_arrays_loaded':0,'new_natural_epsilon_draws':0,'synthetic_permutation_seed':12345,'model_imports':0,'model_calls':0,'test_read':False,
        'frozen_reader_rows':128,'note':'Author synthetic/static verification only; no execution authorization.'}
p=Path(__file__).with_name('AUTHOR_SYNTHETIC_VALIDATION.json')
if p.exists():raise RuntimeError('append-only result already exists')
p.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n');print(json.dumps({'status':report['status'],'checks':len(checks),'script_sha256':report['candidate_script_sha256']}))
