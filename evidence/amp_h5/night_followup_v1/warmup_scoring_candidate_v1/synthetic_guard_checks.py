"""Deterministic integer-statistic fixtures only; no natural files or RNG."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import time

START = time.monotonic()
HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('guard', HERE / 'paired_guard_v1.py')
guard = importlib.util.module_from_spec(spec)
spec.loader.exec_module(guard)
checks = []

def fixture(m=14, s=12, length=0):
    rows = []
    for split, n in guard.SPLITS.items():
        for i in range(n):
            for mode, draw in [('M', -1)] + [('S', j) for j in range(8)]:
                rows.append({'split': split, 'row_id': i, 'mode': mode, 'draw_index': draw,
                    'seed': None if mode == 'M' else 1422 + draw,
                    'source_row_id': i + (279 if split == 'development' else 0),
                    'row_key': split + '_' + str(i), 'terminal': 'FREE' if i % 2 else 'AMD',
                    'prediction_complete_finite': True, 'status': 'ok',
                    'R_to_prediction': {'target_residues': 20, 'correct_target_positions': m if mode == 'M' else s,
                        'length_absolute': length if mode == 'M' else 0}})
    support = {a: (820 if a == 'A' else 0) for a in guard.AA}
    correct = {a: (s * 41 * 8 if a == 'A' else 0) for a in guard.AA}
    summary = [{'split': 'development', 'terminal': 'ALL', 'mode_group': 'S8',
        'all_predictions_complete_finite': True,
        'supported_AA_macro': {'support_per_original_sequence_set': support, 'correct_across_draws': correct,
            'supported_AA': ['A'], 'exposure_multiplier': 8}}]
    return rows, summary

def check(name, predicate):
    assert predicate, name
    checks.append({'name': name, 'status': 'PASS'})

def reject(name, b, c, bs, cs):
    try:
        guard.calculate(b, c, bs, cs)
    except ValueError:
        check(name, True)
    else:
        raise AssertionError('did not reject ' + name)

b, bs = fixture()
c, cs = fixture(s=13)
x = guard.calculate(b, c, bs, cs)
check('exact 1/20 boundary passes', x['resource_success'] and x['S8_gain_exact']['numerator'] == 1 and x['S8_gain_exact']['denominator'] == 20)
c2, cs2 = fixture(m=16, s=13)
x2 = guard.calculate(b, c2, bs, cs2)
check('wider gap is descriptive and not fifth guard', x2['resource_success'] and not x2['gap_shrunk_descriptive_only'])
c2, cs2 = fixture(m=13, s=13)
check('M decline fails', not guard.calculate(b, c2, bs, cs2)['resource_success'])
c2, cs2 = fixture(s=13, length=1)
check('M length1 fails baselinezero guard', not guard.calculate(b, c2, bs, cs2)['resource_success'])
c2, cs2 = fixture(s=12)
check('gap shrink with no S gain fails', not guard.calculate(b, c2, bs, cs2)['resource_success'])
c2 = copy.deepcopy(c); cs2 = copy.deepcopy(cs)
next(r for r in c2 if r['split']=='development' and r['mode']=='S')['R_to_prediction']['correct_target_positions'] -= 1
cs2[0]['supported_AA_macro']['correct_across_draws']['A'] -= 1
check('one count below exact boundary fails without tolerance', not guard.calculate(b, c2, bs, cs2)['resource_success'])
for name, mutate in [
    ('missing train slot invalidates all', lambda rows: rows.pop()),
    ('duplicate slot invalidates', lambda rows: rows.append(copy.deepcopy(rows[0]))),
    ('nonfinite train slot invalidates', lambda rows: rows[0].update(prediction_complete_finite=False)),
    ('failed status invalidates', lambda rows: rows[0].update(status='failed')),
    ('seed mismatch invalidates', lambda rows: rows[1].update(seed=999)),
    ('row key mismatch invalidates', lambda rows: rows[0].update(row_key='different')),
    ('terminal mismatch invalidates', lambda rows: rows[0].update(terminal='FREE')),
    ('target length mismatch invalidates', lambda rows: rows[0]['R_to_prediction'].update(target_residues=21)),
    ('float count invalidates', lambda rows: rows[0]['R_to_prediction'].update(correct_target_positions=14.0)),
    ('boolean count invalidates', lambda rows: rows[0]['R_to_prediction'].update(correct_target_positions=True)),
]:
    c2=copy.deepcopy(c); mutate(c2); reject(name,b,c2,bs,cs)
cs2 = copy.deepcopy(cs); cs2[0]['supported_AA_macro']['exposure_multiplier']=7
reject('macro exposure mismatch',b,c,bs,cs2)
cs2 = copy.deepcopy(cs); cs2[0]['supported_AA_macro']['correct_across_draws']['A'] -= 1
reject('macro counts must sum to row counts',b,c,bs,cs2)
bs2=copy.deepcopy(bs); cs2=copy.deepcopy(cs)
for summary in [bs2,cs2]:
    item=summary[0]['supported_AA_macro'];item['support_per_original_sequence_set'].update(A=779,C=41);item['supported_AA']=['A','C']
bs2[0]['supported_AA_macro']['correct_across_draws'].update(A=3608,C=328)
check('AA macro can fail despite sequence accuracy gain', not guard.calculate(b,c,bs2,cs2)['resource_success'])
b2,bs2=fixture(length=1)
reject('fixed baseline zero length must hold',b2,c,bs2,cs)
c2=copy.deepcopy(c)
c2[0].update(status='empty_sequence')
c2[0]['R_to_prediction'].update(correct_target_positions=0,length_absolute=20)
check('completed empty retained as data not incomplete', guard.validate_rows(c2)[('train',0,'M',-1)]['R_to_prediction']['correct_target_positions']==0)
original=(HERE.parent/'rms_scoring_candidate/score_rms_v2.py').read_text()
adapted=(HERE/'score_warmup_rms_v1.py').read_text()
check('scorer exactly identical except scope constant',adapted.replace("SCOPE = 'AMP_H5_WARMUP_RMS_SCORING_V1'", "SCOPE = 'AMP_H5_CE_RMS_SCORING_V2'")==original)
print(json.dumps({'status':'PASS','checks':checks,'check_count':len(checks),'elapsed_seconds':time.monotonic()-START,
    'scope':'pure deterministic synthetic statistics and code identity; no natural scoring',
    'model_calls':0,'natural_data_reads':0,'random_draws':0,
    'guard_sha256':hashlib.sha256((HERE/'paired_guard_v1.py').read_bytes()).hexdigest()}))
