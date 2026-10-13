#!/usr/bin/env python3
"""Stdlib-only slot/AST/latebinding synthetic checks; no model/data/RNG work."""
import ast
import copy
import hashlib
import json
from pathlib import Path
import sys

import export_postwarmup_rms_v1 as m


def main():
    count = 0

    def check(value, label):
        nonlocal count
        if not value:
            raise AssertionError(label)
        count += 1

    def rejects(fn, label):
        try:
            fn()
        except (ValueError, PermissionError, TypeError):
            check(True, label)
        else:
            check(False, label)

    check(not {'torch', 'numpy', 'rdkit'} & set(sys.modules), 'no model import')
    rows = m.slot_intent()
    check(len(rows) == 2880, 'fixed slots2880')
    check(len({r['sample_id'] for r in rows}) == 2880, 'unique slot ids')
    check(sum(r['mode'] == 'M' for r in rows) == 320, 'M320')
    check(sum(r['mode'] == 'S' for r in rows) == 2560, 'S2560')
    for mode, draw, seed in [('M', -1, ''), *[('S', i, s) for i, s in enumerate(range(1422, 1430))]]:
        group = [r for r in rows if r['mode'] == mode and r['seed'] == seed]
        check(len(group) == 320 and [r['source_row_id'] for r in group] == list(range(320)), 'identity/order per bank')
        check(all(r['draw_index'] == draw for r in group), 'draw provenance')
    check(sum((n + 31) // 32 for _, n, _ in m.SPLITS) == 11, 'new encoder9+2')
    check(9 * sum((n + 31) // 32 for _, n, _ in m.SPLITS) == 99, 'decoder99')
    check([min(32, n - i) for _, n, _ in m.SPLITS for i in range(0, n, 32)] == [32] * 8 + [23, 32, 9], 'batch sizes')
    check(all(r['status'] == 'not_executed' for r in m.allocated_rows()), 'all pending retained')
    tree = ast.parse(Path(m.__file__).read_text())
    functions = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
    bank_calls = [n.func.attr for n in ast.walk(functions['epsilon_banks']) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)]
    check(not {'Generator', 'manual_seed', 'randn', 'randn_like', 'seed'} & set(bank_calls), 'zero bank redraw')
    check('load' in bank_calls and 'from_numpy' in bank_calls, 'bank loading retained')
    all_calls = [n.func.attr for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)]
    check(all_calls.count('posterior_parameters') == 2, 'train and dev encoder routine calls')
    check(not {'train_vae', 'train_ddpm', 'generate', 'prior_flow', 'init_runtime'} & set(all_calls), 'no unrelated model route')
    check('train_posterior_parameters.npz' not in Path(m.__file__).read_text(), 'no historical trainq cache')
    check('legacy[' not in Path(m.__file__).read_text(), 'no legacy M prediction parity')
    # Synthetic late evidence: no files are accessed, no actual checkpoint exists.
    keys = ('checkpoint', 'checkpoint_metadata', 'training_summary', 'training_status',
            'training_supervisor', 'training_independent_postflight', 'training_execution_contract')
    late = {'scope_id': m.SCOPE, 'status': 'BOUND_SUCCESSFUL_WARMUP1000',
            'checkpoint_tensor_digest': 'a' * 64, 'independent_paired_trace_pass': True,
            'root_confirms_single_beta_schedule_only': True}
    for key in keys:
        late[key] = {'path': str(m.ROOT / 'synthetic_metadata' / (key + '.json')), 'sha256': 'synthetic'}
    objects = {
        late['checkpoint_metadata']['path']: {'updates': 1000, 'fixed_endpoint_reached': True,
            'sha256': 'synthetic', 'tensor_state_sha256': 'a' * 64},
        late['training_summary']['path']: {'updates': 1000, 'fixed_endpoint_reached': True, 'elapsed_seconds': 1500},
        late['training_status']['path']: {'state': 'completed_vae_only_not_scientific_success', 'vae_updates': 1000},
        late['training_supervisor']['path']: {'state': 'completed', 'jobs': [{'state': 'completed', 'exit_code': 0, 'seconds': 1510}]},
        late['training_independent_postflight']['path']: {'status': 'PASS', 'blocked': False,
            'bindings': [late[k] for k in keys if k != 'training_independent_postflight']},
    }
    old_sha, old_read, old_canonical = m.sha, m.read, m.canonical
    m.sha = lambda path: 'synthetic'
    m.read = lambda path: objects[str(path)]
    m.canonical = lambda value, expected: None
    try:
        check(m.validate_late(late) is late, 'synthetic late success')
        for key, field, value in (
            ('checkpoint_metadata', 'updates', 999),
            ('checkpoint_metadata', 'fixed_endpoint_reached', False),
            ('checkpoint_metadata', 'tensor_state_sha256', 'b' * 64),
            ('training_summary', 'elapsed_seconds', 1920.00001),
            ('training_status', 'state', 'NOT_EVALUABLE'),
            ('training_independent_postflight', 'status', 'HOLD'),
            ('training_independent_postflight', 'blocked', True),
            ('training_independent_postflight', 'bindings', []),
        ):
            target = objects[late[key]['path']]; old = target[field]; target[field] = value
            rejects(lambda: m.validate_late(late), 'bad late evidence:' + key + '/' + field)
            target[field] = old
        job = objects[late['training_supervisor']['path']]['jobs'][0]
        for field, value in (('seconds', 2100.0001), ('exit_code', 124), ('outer_timeout', True)):
            old = dict(job); job[field] = value
            rejects(lambda: m.validate_late(late), 'outer actual wall/completion gate')
            job.clear(); job.update(old)
        for field, value in (('status', 'UNBOUND_NOT_EXECUTABLE'), ('independent_paired_trace_pass', False),
                             ('root_confirms_single_beta_schedule_only', False)):
            bad = copy.deepcopy(late); bad[field] = value
            rejects(lambda: m.validate_late(bad), 'unbound or unreviewed late rejected')
    finally:
        m.sha, m.read, m.canonical = old_sha, old_read, old_canonical
    check(not {'torch', 'numpy', 'rdkit'} & set(sys.modules), 'no runtime import at finish')
    result = {'status': 'PASS', 'checks': count, 'scope': 'stdlib synthetic slots, AST and metadata only',
              'model_imports': 0, 'model_forwards': 0, 'cohort_reads': 0, 'epsilon_reads': 0,
              'test_reads': 0, 'random_draws': 0,
              'script_sha256': hashlib.sha256(Path(m.__file__).read_bytes()).hexdigest()}
    print(json.dumps(result, indent=2))
    return result


if __name__ == '__main__':
    main()
