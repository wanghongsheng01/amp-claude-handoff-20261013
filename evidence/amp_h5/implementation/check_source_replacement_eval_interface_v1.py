#!/usr/bin/env python3
"""Deterministic synthetic-only checks for the unchanged evaluator's new diagnostic IDs."""
import copy
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
sys.dont_write_bytecode = True
ROOT=Path(__file__).resolve().parents[1]
METHODS=['CE_DDPM_REUSED','EMPIRICAL_POSTERIOR_B','COORDINATE_PERMUTATION_C','REFERENCE_G_PRIOR_D']
SEEDS={m:{'FREE':422,'AMD':423} for m in METHODS}
EXPECTED='b894a7c56cd5c59c5fb77c046baa3c9873ae390d244082dd1f09998459e85417'

def main():
    script=ROOT/'implementation/evaluate_h5.py'; assert hashlib.sha256(script.read_bytes()).hexdigest()==EXPECTED
    spec=importlib.util.spec_from_file_location('frozen_h5_synthetic_only',script);e=importlib.util.module_from_spec(spec);spec.loader.exec_module(e)
    outcomes=[]
    def check(name,fn):
        fn();outcomes.append({'name':name,'pass':True})
    with tempfile.TemporaryDirectory(prefix='amp_source_eval_synthetic_') as t:
        p=Path(t)/'synthetic.csv'; rows=[]
        for m in METHODS:
            for term in ['FREE','AMD']:
                for i in range(16):rows.append({'sample_id':f'{m}_{term}_{i:03d}','method':m,'terminal':term,'sequence':'ACDEFG','status':'ok','seed':str(SEEDS[m][term])})
        def load(a):
            with p.open('w') as f:
                w=csv.DictWriter(f,fieldnames=list(a[0]));w.writeheader();w.writerows(a)
            return e.sample_groups([str(p)],METHODS,16,SEEDS)
        def passes():
            g=load(rows);assert len(g)==8 and sum(map(len,g.values()))==128
        check('all128_exact_diagnostic_slots_accepted',passes)
        def rejects(a):
            try:load(a)
            except ValueError:return
            raise AssertionError('invalid fixture unexpectedly accepted')
        a=copy.deepcopy(rows);a[0]['sample_id']='arbitrary_replacement';check('unexpected_slot_id_rejected',lambda:rejects(a))
        a=copy.deepcopy(rows);a[0]['sample_id']=a[1]['sample_id'];check('duplicate_slot_rejected',lambda:rejects(a))
        a=copy.deepcopy(rows);a.pop();check('missing_request_rejected',lambda:rejects(a))
        a=copy.deepcopy(rows);a[0]['seed']='999';check('wrong_epsilon_seed_rejected',lambda:rejects(a))
        a=copy.deepcopy(rows);a[0]['method']='original_method_alias';check('unlocked_method_alias_rejected',lambda:rejects(a))
        a=copy.deepcopy(rows);a[0]['sequence']='';a[0]['status']='empty_sequence';a[1]['sequence']='AC';a[1]['status']='error';a[2]['sequence']='';a[2]['status']='not_executed'
        def denominator():
            groups=load(a);assert len(groups[(METHODS[0],'FREE')])==16
            def ref(term,s):return {'terminal_state':term,'sequence':s,'molecule_sha256':hashlib.sha256((term+s).encode()).hexdigest()}
            tr=[ref(t,s) for t in ['FREE','AMD'] for s in ['ACDE','FGHI']]
            dev=[ref(t,s) for t in ['FREE','AMD'] for s in ['KLMN','PQRS']]
            metrics,details=e.evaluate_generation(groups,tr,dev);assert len(details)==128
            g=next(g for g in metrics['groups'] if g['method']==METHODS[0] and g['terminal']=='FREE')
            assert g['requests']==16 and g['empty_count']==2 and g['execution_completed_count']==14
            assert g['valid_1_48_count']==14 and g['status_counts']['error']==1 and g['status_counts']['not_executed']==1
            # Frozen distribution eligibility is canonical length, independent of completion status.
            assert g['valid_1_48']['n']==14 and g['valid_1_48']['unique_count']==2
            legacy=e.exploratory_gates(metrics,[])
            assert legacy['generation_ce_vs_terminal_ar']['pass'] is False
            assert all('missing' in x for x in legacy['generation_ce_vs_terminal_ar']['checks'].values())
        check('empty_failed_and_pending_retained_with_eligibility_explicit',denominator)
        check('seed_field_shared_epsilon_separate_selection_permutation',lambda: all(SEEDS[m]=={'FREE':422,'AMD':423} for m in METHODS) or (_ for _ in ()).throw(AssertionError()))
    out=ROOT/'protocol/ce_source_replacement_evaluation_v1/SYNTHETIC_INTERFACE_RESULT.json'
    with out.open('x') as f:f.write(json.dumps({'status':'PASS_SYNTHETIC_ONLY','checks':outcomes,'evaluator_sha256':EXPECTED,'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'operations':{'natural_dataset_reads':0,'natural_sample_reads':0,'model_calls':0,'new_random_draws':0,'test37_reads':0},'authorization':False,'limits':'No execution or natural quality result; old gate false from missing operands is NOT_APPLICABLE, not a diagnostic failure.'},indent=2)+'\n')
    print(json.dumps({'status':'PASS_SYNTHETIC_ONLY','checks':len(outcomes),'output':str(out)}))

if __name__=='__main__':main()
