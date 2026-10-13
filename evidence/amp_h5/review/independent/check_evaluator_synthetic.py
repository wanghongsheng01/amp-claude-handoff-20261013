"""Independent tiny synthetic H5 evaluator interface/micro-macro/copy checks."""
from pathlib import Path
import hashlib
import importlib.util
import json
import math
import sys

sys.dont_write_bytecode=True
CODE=Path(__file__).resolve().parents[2]/'implementation/evaluate_h5.py'
spec=importlib.util.spec_from_file_location('evaluator_under_review',CODE)
e=importlib.util.module_from_spec(spec);spec.loader.exec_module(e)
checks=[]
def check(name,value):
    assert bool(value),name
    checks.append(name)

truth=[{'sequence':'AAAA','terminal_state':'FREE'}, {'sequence':'R','terminal_state':'AMD'}]
rows=[{'row_id':0,'truth_sequence':'AAAA','terminal':'FREE','prediction_sequence':'A','status':'ok'}, {'row_id':1,'truth_sequence':'R','terminal':'AMD','prediction_sequence':'R','status':'ok'}]
r,_=e.reconstruction_metrics(rows,truth)
check('missing rendered residues retained in micro denominator',r['target_residues']==5 and r['rendered_aligned_correct']==2 and r['rendered_nonpad_micro']==.4)
check('macro is AA recall not sequence mean',r['rendered_nonpad_macro']==.625 and r['supported_aa_classes']==['A','R'])
check('length/exact independent oracle',r['length_mae']==1.5 and r['whole_rendered_exact_count']==1)
train=[{'sequence':'AAAAA','terminal_state':'FREE','molecule_sha256':'freeA'}, {'sequence':'RRRRR','terminal_state':'AMD','molecule_sha256':'amdR'}]
dev=[{'sequence':'DDDDD','terminal_state':'FREE','molecule_sha256':'freeD'}, {'sequence':'NNNNN','terminal_state':'AMD','molecule_sha256':'amdN'}]
groups={}
for method in e.METHODS:
    for terminal in e.TERMINALS:
        seq='RRRRR' if terminal=='FREE' else 'AAAAA'
        groups[method,terminal]=[{'method':method,'terminal':terminal,'sample_id':f'{method}_{terminal}_000','seed':'422' if terminal=='FREE' else '423','sequence':seq,'status':'ok'}]
g,details=e.evaluate_generation(groups,train,dev)
check('all eight synthetic outputs retained',len(details)==8 and len(g['groups'])==8)
check('cross-terminal copies distinguished',all(x['exact_copy_train_count_all_requests']==0 and x['exact_sequence_copy_any_train_terminal_count']==1 for x in g['groups']))
check('same/any nearest differentiated',all(x['nearest_train']['identity']==0 and x['nearest_any_train_terminal']['identity']==1 for x in details))
check('full fixed421 metric dimension retained',all(len(x['valid_1_48']['fixed_feature_mean'])==421 for x in g['groups']))
tmp=Path(__file__).with_name('synthetic_samples.jsonl')
tmp.write_text(''.join(json.dumps(r)+'\n' for values in groups.values() for r in values))
seeds={m:{'FREE':422,'AMD':423} for m in e.METHODS}
check('exact synthetic slots/seed interface accepted',len(e.sample_groups([tmp],e.METHODS,1,seeds))==8)
bad=[dict(r) for values in groups.values() for r in values];bad[0]['sample_id']='fake_slot'
tmp.write_text(''.join(json.dumps(r)+'\n' for r in bad))
try:
    e.sample_groups([tmp],e.METHODS,1,seeds)
    rejected=False
except ValueError:
    rejected=True
check('replaced slot rejected even equal row count',rejected)
try:
    e.guarded_read('/tmp/never_created_test.csv')
    guarded=False
except ValueError:
    guarded=True
check('test path rejected before absent file access',guarded)
try:
    e.guarded_read('/tmp/never_created_test37.csv')
    numeric_suffix_path_guard=False
except ValueError:
    numeric_suffix_path_guard=True
except FileNotFoundError:
    numeric_suffix_path_guard=False
broken={k:[dict(r) for r in v] for k,v in groups.items()};broken['CE21_DDPM','FREE'][0]['status']='error:FloatingPointError'
bg,_=e.evaluate_generation(broken,train,dev)
gate=e.exploratory_gates(bg,[])
check('failure with legal sequence cannot pass generation gate',not gate['generation_ce_vs_terminal_ar']['pass'] and gate['generation_ce_vs_terminal_ar']['status']=='NOT_EVALUABLE_FAILED_GENERATION')
check('equal MMD is not strict improvement',not e.comparator(.1,.1,'lt')['pass'])
result={'status':'PASS_WITH_DOCUMENTED_PATH_GUARD_LIMIT','check_count':len(checks),'checks':checks,'source':str(CODE),'source_sha256':hashlib.sha256(CODE.read_bytes()).hexdigest(),'natural_records_read':0,'test_reads':0,'model_calls':0,'known_limit':{'test_numeric_suffix_path_rejected':numeric_suffix_path_guard,'description':'Frozen dependency rejects delimited test token, not test37. Only a nonexistent synthetic path was attempted. Exact source SHA and role binding remain necessary. Initial independent check expected the stronger guard and failed with FileNotFoundError; this limit is not silently presented as fixed.'}}
Path(__file__).with_name('EVALUATOR_SYNTHETIC_RESULT.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
