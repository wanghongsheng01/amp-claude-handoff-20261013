#!/usr/bin/env python3
"""Single-use outer watchdog for the frozen, approved RMS scorer; no metric math."""
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time
from datetime import datetime, timezone

ROOT = Path('/Users/loctek/Documents/Codex/2026-10-11/task-4/amp_h5')
PYTHON = '/Users/loctek/Documents/Codex/2026-10-09/task-8/amp_commsbio_20261011/rebuild_experiments_20261011/env/bin/python'
SCORER = ROOT / 'night_followup_v1/warmup_scoring_candidate_v1/paired_guard_v1.py'
CONFIG = ROOT / 'night_followup_v1/warmup_scoring_candidate_v3/PAIRED_GUARD_CONTRACT.json'
APPROVAL = ROOT / 'review/independent/CE_WARMUP_PAIRED_RESOURCE_GUARD_PREFLIGHT_v2.json'
POSTFLIGHT = ROOT / 'review/independent/CE_WARMUP_RMS_SCORING_POSTFLIGHT_v1.json'
OUT = ROOT / 'night_followup_v1/warmup_paired_guard_run_v1'
EXECUTION = ROOT / 'night_followup_v1/warmup_paired_guard_execution_v1'
BASELINE = ROOT / 'night_followup_v1/review/NIGHT_FOLLOWUP_JOINT_POSTFLIGHT_v1.json'
BINDINGS = {
    str(BASELINE): '67254873fcff4dfb660bca8f729798dfdd0873b7f9d6012f0f36b0320dacf036',
    str(SCORER): '3bb4d3a3cd2b8247de8540ce5e5a8cabeb6992d51c0dd6b011f61ac6b396c2e2',
    str(CONFIG): '8b1e0740da6691381e88cee1d0462ec924d60faee3f4aec6e180340ff31e6b6c',
    str(APPROVAL): '7cac89b64a000dad19aae9ff2b3f8726076c11b4a81701f9b081bb0d953f9a39',
    str(POSTFLIGHT): 'b2bde7132a6879982313e50f662fe683a649fedf169d5b7a19f360c703b276cf',
}

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def write(path, data):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + '\n')

def main():
    for path, expected in BINDINGS.items():
        if sha(path) != expected:
            raise RuntimeError('Input hash mismatch: ' + path)
    pf = json.loads(POSTFLIGHT.read_text())
    if pf['status'] != 'PASS' or pf['scope_id'] != 'AMP_H5_WARMUP_RMS_SCORING_V1':
        raise RuntimeError('Model postflight is not the required PASS')
    if OUT.exists():
        raise FileExistsError('Scoring output already exists; no retry allowed')
    EXECUTION.mkdir(exist_ok=False)
    command = [PYTHON, '-B', str(SCORER), '--config', str(CONFIG), '--approval', str(APPROVAL),
               '--baseline-postflight', str(BASELINE), '--candidate-postflight', str(POSTFLIGHT), '--out', str(OUT)]
    started = datetime.now(timezone.utc).isoformat()
    base = {'scope_id': 'AMP_H5_WARMUP_PAIRED_RESOURCE_GUARD_V1', 'command': command,
            'root_authorization': 'Root direct single-use authorization after model postflight PASS',
            'started_utc': started, 'outer_cap_seconds': 60, 'inner_cap_seconds': 30,
            'inputs': BINDINGS, 'watchdog_sha256': sha(__file__), 'retry_allowed': False,
            'model_calls': 0, 'training_updates': 0, 'random_draws': 0, 'test_access': False}
    write(EXECUTION / 'COMMAND.json', base)
    start = time.monotonic()
    exit_code = None
    timeout = False
    error = None
    with (EXECUTION / 'stdout.log').open('w') as stdout, (EXECUTION / 'stderr.log').open('w') as stderr:
        try:
            process = subprocess.Popen(command, cwd=ROOT, stdout=stdout, stderr=stderr, start_new_session=True,
                                       env={**os.environ, 'PYTHONDONTWRITEBYTECODE': '1'})
            write(EXECUTION / 'RUNNING.json', {**base, 'pid': process.pid, 'state': 'running'})
            try:
                exit_code = process.wait(timeout=max(0, 60 - (time.monotonic() - start)))
            except subprocess.TimeoutExpired:
                timeout = True
                os.killpg(process.pid, signal.SIGKILL)
                exit_code = process.wait()
        except Exception as exc:
            error = type(exc).__name__ + ': ' + str(exc)
    wall = time.monotonic() - start
    closing_path = OUT / 'CLOSING_LEDGER.json'
    closing = json.loads(closing_path.read_text()) if closing_path.exists() else None
    unchanged = all(sha(p) == h for p, h in BINDINGS.items())
    completed = exit_code == 0 and not timeout and not error and unchanged and closing is not None and closing.get('state') == 'completed' and wall <= 60
    result = {**base, 'state': 'completed' if completed else ('timed_out' if timeout else 'failed'),
              'finished_utc': datetime.now(timezone.utc).isoformat(), 'exit_code': exit_code,
              'wall_seconds': wall, 'outer_overrun_seconds': max(0, wall - 60), 'timed_out': timeout,
              'error': error, 'all_inputs_unchanged': unchanged, 'closing_ledger': closing,
              'closing_ledger_sha256': sha(closing_path) if closing_path.exists() else None,
              'investment_gate_usable_pending_independent_review': completed,
              'writing_and_interactive_audit_wall_seconds': None,
              'wall_scope': 'Scoring subprocess launch through exit; includes process startup and shutdown, excludes wrapper result writing'}
    write(EXECUTION / 'RESULT.json', result)
    files = [{'path': str(p), 'sha256': sha(p), 'bytes': p.stat().st_size} for p in sorted(EXECUTION.iterdir()) if p.is_file()]
    write(EXECUTION / 'MANIFEST.json', {'scope_id': base['scope_id'], 'files': files})
    print(json.dumps({'state': result['state'], 'wall_seconds': wall, 'exit_code': exit_code,
                      'result': str(EXECUTION / 'RESULT.json'), 'result_sha256': sha(EXECUTION / 'RESULT.json')}))
    return 0 if completed else 2

if __name__ == '__main__':
    raise SystemExit(main())
