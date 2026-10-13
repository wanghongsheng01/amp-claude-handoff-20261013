import copy,hashlib,importlib.util,json,sys
from pathlib import Path
from datetime import datetime,timezone
sys.dont_write_bytecode=True
R=Path('/Users/loctek/Documents/Codex/2026-10-11/task-4/amp_h5');O=R/'review/independent'
D=R/'protocol/CE_SOURCE_REPLACEMENT_DIAGNOSTIC_DRAFT_v1.json';E=R/'protocol/ce_source_replacement_evaluation_v1/EVALUATION_PREPARATION.json';F=R/'review/literature/FOUR_ARM_FINAL_SCIENTIFIC_REVIEW_v1.json';OUT=O/'CE_SOURCE_REPLACEMENT_DRAFT_EVALUATION_PREAUDIT_v1.json';assert not OUT.exists()
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();read=lambda p:json.loads(Path(p).read_text())
d,e,f=read(D),read(E),read(F);checks=[]
def ck(n,v):checks.append({'name':n,'pass':bool(v)})
ck('draft_hash',sha(D)=='a5f12518b8b677c44f1b48a7da0b6ebe8a7626d9f7a6c4769c4ce12e3f7ce700');ck('evaluation_preparation_hash',sha(E)=='ac3229a71ac86a7c988b2324b0b6713af34c3c8e93209c81cc8603820b29efe3')
ck('four_arm_actual_scientific_closeout',sha(F)=='4fa10c677f8165152698243a99371e7ca22e4173529f9de64915bcb61b84e42a' and f['review_status']=='COMPLETED_SCIENTIFIC_CLOSEOUT' and f['four_arm_closeout_complete'] is True and f['blocking_scientific_mismatches']==[] and f['new_scope_execution_authorized_by_this_review'] is False)
ck('all_draft_bindings_hash_bytes',all(sha(x['path'])==x['sha256'] and Path(x['path']).stat().st_size==x['bytes'] for x in d['bindings']))
ck('frozen_evaluator_and_metrics',all(sha(x['path'])==x['sha256'] for x in e['frozen_sources']))
ck('128_rows96_new32_reused',e['slots']['total_observations']==128 and e['slots']['new_decode_observations']==96 and e['slots']['reused_DDPM']==32)
ck('epsilon_selection_permutation_separate',d['epsilon']['seeds']=={'FREE':422,'AMD':423} and d['B_selection']['seeds']=={'FREE':523,'AMD':524} and d['C_permutation']['seeds']=={'FREE':625,'AMD':626})
ck('roundtrip_tolerance_fixed',d['D_reference_g']['roundtrip']['tolerance_locked_proposal']['atol']==1e-5 and d['D_reference_g']['roundtrip']['tolerance_locked_proposal']['rtol']==1e-5)
ck('diagnostic_gates_not_applicable','NOT_APPLICABLE_SOURCE_DIAGNOSTIC' in e['legacy_gate_artifact_interpretation'] and e['PROMOTE']=='HOLD')
ck('fixed_old_train_dev_only',e['cohort_reference_bindings_not_opened_in_preparation'][0]['sha256']=='a232401fb46bb9419cf8d690f21476447b853862fd23622e81247b88add0b660' and e['cohort_reference_bindings_not_opened_in_preparation'][1]['sha256']=='65fdc7d402240235698d30aa769de3ea45c577cbcb28931970571ee808098ec1')
sp=importlib.util.spec_from_file_location('independent_frozen_eval',R/'implementation/evaluate_h5.py');m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m)
methods=e['method_ids'];seeds=e['sampling_seed_metadata']['expected']
rows=[{'sample_id':f'{method}_{t}_{i:03d}','method':method,'terminal':t,'sequence':'A','length':1,'status':'ok','seed':str(422+label),'terminal_sample_id':i} for method in methods for label,t in enumerate(('FREE','AMD')) for i in range(16)]
def run(rows):
 m.read_rows=lambda _:copy.deepcopy(rows);return m.sample_groups(['synthetic_only'],methods,16,seeds)
def fails(rows):
 try:run(rows);return False
 except (ValueError,KeyError):return True
ck('all128_exact_slots_accepted',sum(len(x) for x in run(rows).values())==128)
r=copy.deepcopy(rows);r[0].update(sequence='',length=0,status='not_executed');r[1].update(sequence='',length=0,status='failed');r[2].update(sequence='A',length=1,status='failed')
g=run(r);ck('empty_failed_pending_and_legal_failed_retained',sum(len(x) for x in g.values())==128 and list(g.values())[0] is not None)
r=copy.deepcopy(rows);r[0]['seed']='523';ck('selection_seed_not_epsilon_seed_rejected',fails(r))
r=copy.deepcopy(rows);r[0]['method']='CEdiag_existing_DDPM';ck('historical_alias_rejected',fails(r))
r=copy.deepcopy(rows);r[0].update(sequence='',length='',status='not_executed');ck('blank_length_failure_reproduced_for_final_contract',fails(r))
r=copy.deepcopy(rows);r.pop();ck('missing_slot_rejected',fails(r))
r=copy.deepcopy(rows);r[0]['terminal_sample_id']=1;ck('slot_index_disagreement_rejected',fails(r))
legacy=m.exploratory_gates({'groups':[]},[]);ck('legacy_gate_missing_operand_false_confirmed',legacy['generation_ce_vs_terminal_ar']['pass'] is False)
ck('no_torch_numpy_import','torch' not in sys.modules and 'numpy' not in sys.modules)
report={'created_utc':datetime.now(timezone.utc).isoformat(),'status':'PASS_DESIGN_AND_EVALUATION_PREPARATION_WITH_FINAL_CONTRACT_REQUIREMENTS' if all(x['pass'] for x in checks) else 'HOLD','execution_authorized':False,'checks':checks,'failed_checks':[x for x in checks if not x['pass']],'required_before_execution':['Final new contract must override historical draft optional/max/pending D language: exactly mandatoryB/C/D96, six decoder and two inverse plus two forward flow batches, no runtime downgrade.','Use final four method IDs from evaluation preparation; preserve mapping/source hashes to oldCE32 rather than redecoding.','Preallocate all96 rows with sequence empty and length0, plus valid integer terminal_sample_id. Blank length is rejected by frozen reader even for failed/pending rows.','Every complete attempt requires exact new implementation/contract/hashbound supervisor plan and independent execution receipt; finalized sampler-seed file must be bound.','Base seal contains epsilon/indices/Bz/Cz/g/perms before model construction; D z follows inverse and all decoder inputs are sealed before any decoder call.','Engineering terminal review must precede natural diagnostic evaluation. Partial failed raw metrics require incomplete-source interpretation; legacy gates NOT_APPLICABLE, never newFAIL/winner.'],'inference_boundaries':['C has no single paired training truth after per-coordinate cross-slot mixing; no reconstruction score against an arbitrary B row.','B/D share reference row index and epsilon but D uses residue/length g whereas B posterior carries additional sequence/global information; not a pure flow-effect causal comparison.','B/C/D are outside c-only candidate ranking and PROMOTE. Tail and readability differences are posthoc source diagnostics only.','No diagnostic result triggers more seeds, retries, scope additions, changed tolerance or adaptive metric selection.'],'source_bindings':[{'path':str(p),'sha256':sha(p)} for p in [D,E,F]],'reviewer_actions':{'stdlib_static_synthetic_only':True,'natural_tensor_loads':0,'cohort_content_reads':0,'random_draws':0,'model_calls':0,'test37_reads':0}}
with OUT.open('x') as h:json.dump(report,h,ensure_ascii=False,indent=2);h.write('\n')
print(json.dumps({'path':str(OUT),'sha256':sha(OUT),'checks':len(checks),'failed':report['failed_checks']}))
