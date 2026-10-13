#!/usr/bin/env python3
"""Read frozen RMS sufficient statistics; four exact resource guards, no model."""
import argparse
from collections import Counter
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import re
import signal
import sys
import time
sys.dont_write_bytecode = True

SCOPE = 'AMP_H5_WARMUP_PAIRED_RESOURCE_GUARD_V1'
AA = 'ACDEFGHIKLMNPQRSTVWY'
SPLITS = {'train': 279, 'development': 41}
EXPECTED = {(s, i, mode, draw) for s, n in SPLITS.items() for i in range(n)
            for mode, draw in [('M', -1)] + [('S', j) for j in range(8)]}
COMPLETE = {'ok', 'empty_sequence', 'no_eos_at_limit'}

def path(p):
    p = Path(p).resolve()
    if re.search(r'(^|[/_.\-])test($|[/_.\-])', str(p).lower()):
        raise ValueError('test path prohibited')
    return p

def sha(p):
    return hashlib.sha256(path(p).read_bytes()).hexdigest()

def load(p):
    return json.loads(path(p).read_text())

def dump(p, d):
    path(p).write_text(json.dumps(d, indent=2, ensure_ascii=False, allow_nan=False) + '\n')

def bindings(obj):
    found = {}
    def walk(value):
        if isinstance(value, dict):
            if isinstance(value.get('path'), str) and isinstance(value.get('sha256'), str):
                p = path(value['path'])
                if not Path(value['path']).is_absolute():
                    raise ValueError('receipt bindings must be absolute')
                if str(p) in found and found[str(p)] != value['sha256']:
                    raise ValueError('conflicting hash bindings')
                found[str(p)] = value['sha256']
            for child in value.values():
                walk(child)
        elif isinstance(value, list):
            for child in value:
                walk(child)
    walk(obj)
    return found

def exact(value):
    return {'numerator': value.numerator, 'denominator': value.denominator, 'value': float(value)}

def int_value(value, lo=0, hi=None):
    if type(value) is not int or value < lo or (hi is not None and value > hi):
        raise ValueError('invalid integer sufficient statistic')
    return value

def validate_rows(rows):
    result = {}
    for row in rows:
        key = (row['split'], row['row_id'], row['mode'], row['draw_index'])
        if key not in EXPECTED or key in result:
            raise ValueError('duplicate or unexpected fixed RMS slot')
        if row['prediction_complete_finite'] is not True or row['status'] not in COMPLETE:
            raise ValueError('fixed RMS slot incomplete/nonfinite')
        if row['seed'] != (None if key[2] == 'M' else 1422 + key[3]):
            raise ValueError('RMS seed identity changed')
        if row['source_row_id'] != key[1] + (279 if key[0] == 'development' else 0):
            raise ValueError('source row identity changed')
        if row['terminal'] not in ('FREE', 'AMD') or not row['row_key']:
            raise ValueError('reference identity missing')
        core = row['R_to_prediction']
        length = int_value(core['target_residues'], 1, 48)
        int_value(core['correct_target_positions'], 0, length)
        int_value(core['length_absolute'], 0, 48)
        result[key] = row
    if set(result) != EXPECTED:
        raise ValueError('fixed 320x9 RMS rows incomplete')
    return result

def macro_exact(summary, rows, mode):
    found = [x for x in summary if x['split'] == 'development' and x['terminal'] == 'ALL'
             and x['mode_group'] == mode]
    if len(found) != 1 or found[0]['all_predictions_complete_finite'] is not True:
        raise ValueError('unique complete macro summary missing')
    item = found[0]['supported_AA_macro']
    exposure = 1 if mode == 'M' else 8
    if item['exposure_multiplier'] != exposure:
        raise ValueError('macro exposure differs')
    support, correct = item['support_per_original_sequence_set'], item['correct_across_draws']
    if set(support) != set(AA) or set(correct) != set(AA):
        raise ValueError('AA sufficient statistics incomplete')
    for a in AA:
        int_value(support[a]); int_value(correct[a], 0, exposure * support[a])
    supported = [a for a in AA if support[a] > 0]
    if item['supported_AA'] != supported or not supported:
        raise ValueError('supported AA identity differs')
    total_target = sum(rows[('development', i, 'M', -1)]['R_to_prediction']['target_residues'] for i in range(41))
    keys = [('development', i, 'M', -1) for i in range(41)] if mode == 'M' else [('development', i, 'S', j) for i in range(41) for j in range(8)]
    if sum(support.values()) != total_target or sum(correct.values()) != sum(rows[k]['R_to_prediction']['correct_target_positions'] for k in keys):
        raise ValueError('macro sufficient counts disagree with complete row totals')
    return sum((Fraction(correct[a], exposure * support[a]) for a in supported), Fraction(0)) / len(supported), support

def calculate(baseline_rows, candidate_rows, baseline_summary, candidate_summary):
    """Pure toy-testable exact arithmetic; uses saved integer statistics only."""
    baseline = validate_rows(baseline_rows)
    candidate = validate_rows(candidate_rows)
    for key in EXPECTED:
        b, c = baseline[key], candidate[key]
        for field in ['row_key', 'source_row_id', 'terminal', 'seed']:
            if b[field] != c[field]:
                raise ValueError('paired identity differs: ' + field)
        if b['R_to_prediction']['target_residues'] != c['R_to_prediction']['target_residues']:
            raise ValueError('paired target lengths differ')
    outputs = {}
    supports = []
    for label, rows, summary in [('baseline', baseline, baseline_summary), ('candidate', candidate, candidate_summary)]:
        m = sum((Fraction(rows[('development', i, 'M', -1)]['R_to_prediction']['correct_target_positions'], rows[('development', i, 'M', -1)]['R_to_prediction']['target_residues']) for i in range(41)), Fraction(0)) / 41
        s = sum((Fraction(rows[('development', i, 'S', j)]['R_to_prediction']['correct_target_positions'], rows[('development', i, 'S', j)]['R_to_prediction']['target_residues']) for i in range(41) for j in range(8)), Fraction(0)) / (41 * 8)
        length = sum((Fraction(rows[('development', i, 'M', -1)]['R_to_prediction']['length_absolute']) for i in range(41)), Fraction(0)) / 41
        macro, support = macro_exact(summary, rows, 'S8'); supports.append(support)
        outputs[label] = {'M': m, 'S8': s, 'M_length_MAE': length, 'S8_supported_AA_macro': macro, 'M_minus_S8': m - s}
    if supports[0] != supports[1]:
        raise ValueError('paired supported AA counts differ')
    b, c = outputs['baseline'], outputs['candidate']
    if b['M_length_MAE'] != 0:
        raise ValueError('frozen baseline length MAE=0 violated')
    gain = c['S8'] - b['S8']
    guards = {'S8_sequence_equal_gain_at_least_1_over_20': gain >= Fraction(1, 20),
              'M_sequence_equal_nondecrease': c['M'] >= b['M'],
              'M_length_MAE_no_greater_than_baseline_zero': c['M_length_MAE'] <= b['M_length_MAE'],
              'S8_supported_AA_macro_nondecrease': c['S8_supported_AA_macro'] >= b['S8_supported_AA_macro']}
    return {'status': 'EVALUABLE', 'resource_success': all(guards.values()), 'guards': guards,
            'dev_sequences': 41, 'posterior_draws': 8, 'fixed_slots_per_model': 2880,
            'exact_values': {label: {k: exact(v) for k, v in values.items()} for label, values in outputs.items()},
            'S8_gain_exact': exact(gain), 'M_change_exact': exact(c['M'] - b['M']),
            'S8_macro_change_exact': exact(c['S8_supported_AA_macro'] - b['S8_supported_AA_macro']),
            'gap_change_exact': exact(c['M_minus_S8'] - b['M_minus_S8']),
            'gap_shrunk_descriptive_only': c['M_minus_S8'] < b['M_minus_S8'],
            'gap_is_not_a_fifth_guard': True,
            'decision_precedence': 'ROOT_PRELOCK four guards supersede the literature-v2 separate gap-shrink criterion. Gap and all side effects remain descriptive; no extra success condition is introduced.',
            'meaning': 'Resource success only; not independent replication, causal identification, biological activity, generation improvement, novelty, submission readiness or further execution authorization.'}

def read_run(spec, receipt_path, expected_scope):
    root = path(spec['root']); receipt = load(receipt_path)
    if receipt['status'] != 'PASS' or receipt.get('scope_id', receipt.get('scope')) != spec['receipt_scope_id']:
        raise ValueError('scoring postflight not exact PASS')
    if spec.get('receipt_section') and receipt[spec['receipt_section']]['status'] != 'PASS':
        raise ValueError('joint scoring postflight section is not PASS')
    receipt_bindings = bindings(receipt)
    required = [root / 'MANIFEST.json', root / 'CLOSING_LEDGER.json', path(spec['execution_result'])]
    for p in required:
        if receipt_bindings.get(str(p)) != sha(p):
            raise ValueError('scoring receipt does not bind exact terminal evidence: ' + str(p))
    manifest = load(root / 'MANIFEST.json'); bound = bindings(manifest)
    for name in ['ROW_METRICS.jsonl', 'STRATIFIED_CORE_SUMMARIES.json', 'STATUS.json']:
        p = root / name
        if bound.get(str(p)) != sha(p):
            raise ValueError('saved scoring statistic is not manifest-bound')
    closing = load(root / 'CLOSING_LEDGER.json'); execution = load(spec['execution_result']); status = load(root / 'STATUS.json')
    if closing['state'] != 'completed' or closing['scope_id'] != expected_scope or closing['manifest_sha256'] != sha(root / 'MANIFEST.json'):
        raise ValueError('scoring closing is not completed and bound')
    if closing['cap_seconds'] != 300 or closing['wall_seconds_after_manifest_before_closing'] >= 300:
        raise ValueError('scoring inner cap not satisfied')
    if execution['state'] != 'completed' or execution['exit_code'] != 0 or execution['wall_seconds'] > 360:
        raise ValueError('scoring outer execution incomplete/over budget')
    if execution['scope_id'] != expected_scope or execution['closing_ledger_sha256'] != sha(root / 'CLOSING_LEDGER.json'):
        raise ValueError('scoring execution scope or closing binding differs')
    if not status['all_inputs_unchanged'] or not status['all_sources_unchanged'] or status['scored_core_rows'] != 2880:
        raise ValueError('scoring inputs or complete rows invalid')
    rows = [json.loads(line) for line in (root / 'ROW_METRICS.jsonl').open()]
    return rows, load(root / 'STRATIFIED_CORE_SUMMARIES.json'), status['input_sha256']['reference_rows']

def run(config_path, approval_path, baseline_postflight, candidate_postflight, out):
    started = time.monotonic(); cfg = load(config_path); approval = load(approval_path)
    if cfg['scope_id'] != SCOPE or approval['scope_id'] != SCOPE or approval['status'] != 'PASS':
        raise ValueError('paired guard config/preaudit mismatch')
    if approval['paired_guard_config_sha256'] != sha(config_path) or approval['paired_guard_script_sha256'] != sha(__file__):
        raise ValueError('paired guard exact preaudit differs')
    for record in cfg['immutable_bindings']:
        if sha(record['path']) != record['sha256']:
            raise ValueError('frozen paired guard source changed')
    used = {path(config_path), path(approval_path), path(baseline_postflight), path(candidate_postflight), path(__file__)}
    used.update(path(record['path']) for record in cfg['immutable_bindings'])
    for spec in [cfg['baseline'], cfg['candidate']]:
        used.add(path(spec['execution_result']))
        used.update(path(spec['root']) / name for name in ['ROW_METRICS.jsonl', 'STRATIFIED_CORE_SUMMARIES.json', 'STATUS.json', 'MANIFEST.json', 'CLOSING_LEDGER.json'])
    initial_hashes = {str(p): sha(p) for p in used}
    out = path(out)
    if out != path(cfg['output_root']) or out.exists():
        raise ValueError('single execution output differs or already exists')
    out.mkdir(parents=True)
    result = {'status': 'NOT_EVALUABLE', 'resource_success': None}
    def budget_stop(*_):
        raise TimeoutError('30 second guard budget exceeded')
    old_handler = signal.signal(signal.SIGALRM, budget_stop)
    signal.setitimer(signal.ITIMER_REAL, max(.001, 30 - (time.monotonic() - started)))
    try:
        b, bs, br = read_run(cfg['baseline'], baseline_postflight, cfg['baseline']['scope_id'])
        c, cs, cr = read_run(cfg['candidate'], candidate_postflight, cfg['candidate']['scope_id'])
        if br != cr or br != cfg['fixed_reference_csv_sha256']:
            raise ValueError('paired reference CSV identity mismatch')
        result = calculate(b, c, bs, cs)
        if any(sha(p) != h for p, h in initial_hashes.items()):
            raise ValueError('paired guard input changed during read')
    except Exception as exc:
        result.update(status='NOT_EVALUABLE', resource_success=None, error=type(exc).__name__ + ': ' + str(exc))
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old_handler)
    elapsed = time.monotonic() - started
    if elapsed >= 30:
        result.update(status='NOT_EVALUABLE', resource_success=None, error='30 second guard budget exceeded')
    result.update(scope_id=SCOPE, created_utc=datetime.now(timezone.utc).isoformat(), elapsed_before_export_seconds=elapsed,
                  model_calls=0, training_updates=0, random_draws=0, descriptors_recomputed=0, test_access=False,
                  config_sha256=sha(config_path), approval_sha256=sha(approval_path),
                  baseline_postflight_sha256=sha(baseline_postflight), candidate_postflight_sha256=sha(candidate_postflight),
                  input_bindings=[{'path': p, 'sha256': h} for p, h in sorted(initial_hashes.items())])
    dump(out / 'PAIRED_RESOURCE_GUARD.json', result)
    files = [{'path': str(p), 'sha256': sha(p), 'bytes': p.stat().st_size} for p in sorted(out.iterdir()) if p.is_file()]
    dump(out / 'MANIFEST.json', {'scope_id': SCOPE, 'files': files})
    elapsed = time.monotonic() - started
    complete = result['status'] == 'EVALUABLE' and elapsed < 30
    dump(out / 'CLOSING_LEDGER.json', {'scope_id': SCOPE, 'state': 'completed' if complete else 'failed',
          'wall_seconds_after_manifest_before_closing': elapsed, 'cap_seconds': 30,
          'manifest_sha256': sha(out / 'MANIFEST.json'), 'model_calls': 0, 'outer_watchdog_required_seconds': 60})
    if time.monotonic() - started >= 30:
        complete = False
        dump(out / 'CLOSING_LEDGER.json', {'scope_id': SCOPE, 'state': 'capped', 'cap_seconds': 30,
              'wall_seconds_observed_after_closing': time.monotonic() - started,
              'manifest_sha256': sha(out / 'MANIFEST.json'), 'model_calls': 0, 'outer_watchdog_required_seconds': 60})
    print(json.dumps({'state': 'completed' if complete else 'failed', 'guard_status': result['status'],
                      'resource_success': result['resource_success'], 'wall_seconds': time.monotonic() - started}))
    return 0 if complete else 2

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    for name in ['config', 'approval', 'baseline-postflight', 'candidate-postflight', 'out']:
        ap.add_argument('--' + name, required=True)
    a = ap.parse_args()
    raise SystemExit(run(a.config, a.approval, a.baseline_postflight, a.candidate_postflight, a.out))
