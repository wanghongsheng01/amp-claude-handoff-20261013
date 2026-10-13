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
SCORER = ROOT / 'night_followup_v1/rms_scoring_candidate/score_rms_v2.py'
CONFIG = ROOT / 'protocol/CE_RMS_SCORING_CONTRACT_v2.json'
APPROVAL = ROOT / 'review/independent/CE_RMS_SCORING_PREFLIGHT_v2.json'
POSTFLIGHT = ROOT / 'review/independent/CE_RMS_RECONSTRUCTION_POSTFLIGHT_v2.json'
OUT = ROOT / 'night_followup_v1/rms_scoring_run_v2'
EXECUTION = ROOT / 'night_followup_v1/rms_scoring_execution_v2'
BINDINGS = {
    str(SCORER): 'e2bae3222e5a13f0de4bc92f658ace2eaf336e17386eb42e416283b9986a2e75',
    str(CONFIG): 'ca3db6d0f70d9205d90f80f8c0ac000f88db4dd18800f41af964b0fcc4d0e3d1',
    str(APPROVAL): '20c85e6e9afeec4991eb208c6a7260339d1e4a87853877d95d576e4a621fafac',
    str(POSTFLIGHT): '9786d0be1c9d1fa345f598c626cb6d9b284f63aa41e60b8a706bc6772d5e5666',
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
    if pf['status'] != 'PASS' or pf['scope_id'] != 'AMP_H5_CE_RMS_RECONSTRUCTION_V2':
        raise RuntimeError('Model postflight is not the required PASS')
    if OUT.exists():
        raise FileExistsError('Scoring output already exists; no retry allowed')
    EXECUTION.mkdir(exist_ok=False)
    command = [PYTHON, '-B', str(SCORER), '--config', str(CONFIG), '--approval', str(APPROVAL),
               '--model-postflight', str(POSTFLIGHT), '--out', str(OUT)]
    started = datetime.now(timezone.utc).isoformat()
    base = {'scope_id': 'AMP_H5_CE_RMS_SCORING_V2', 'command': command,
            'root_authorization': 'Root direct single-use authorization after model postflight PASS',
            'started_utc': started, 'outer_cap_seconds': 360, 'inner_cap_seconds': 300,
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
                exit_code = process.wait(timeout=max(0, 360 - (time.monotonic() - start)))
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
    completed = exit_code == 0 and not timeout and not error and unchanged and closing is not None and closing.get('state') == 'completed' and wall <= 360
    result = {**base, 'state': 'completed' if completed else ('timed_out' if timeout else 'failed'),
              'finished_utc': datetime.now(timezone.utc).isoformat(), 'exit_code': exit_code,
              'wall_seconds': wall, 'outer_overrun_seconds': max(0, wall - 360), 'timed_out': timeout,
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
