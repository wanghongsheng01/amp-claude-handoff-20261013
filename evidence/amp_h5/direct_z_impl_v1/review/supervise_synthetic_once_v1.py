#!/usr/bin/env python3
"""Independent 180s synthetic-only guard; inert on import, no model imports.

Execution needs an exact root-issued window JSON. That document binds this
script, its worker, every candidate/input dependency and scoped job clearance.
This guard does not inspect processes or authorize any natural computation.
"""
import argparse
import datetime
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

H5 = Path('/Users/loctek/Documents/Codex/2026-10-11/task-4/amp_h5')
REVIEW = H5 / 'direct_z_impl_v1/review'
CAP = 180


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, obj):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    tmp.replace(path)


def inside(path, parent):
    resolved = Path(path).resolve()
    if parent not in resolved.parents:
        raise ValueError('Outside approved output scope: ' + str(path))
    return resolved


def main():
    start = time.monotonic()
    ap = argparse.ArgumentParser()
    ap.add_argument('--window', required=True)
    ap.add_argument('--window-sha256', required=True)
    a = ap.parse_args()
    window_path = inside(a.window, H5)
    if sha(window_path) != a.window_sha256:
        raise ValueError('Root window SHA mismatch')
    w = json.loads(window_path.read_text())
    if not (w['status'] == 'AUTHORIZED_SINGLE_SYNTHETIC_WINDOW'
            and w['outer_seconds'] == CAP
            and w['forward_max'] == 132
            and w['natural_execution_authorized'] is False
            and w['supervisor_sha256'] == sha(__file__)):
        raise ValueError('Explicit exact synthetic-only root window required')
    clearance_path = inside(w['managed_job_clearance']['path'], H5)
    if sha(clearance_path) != w['managed_job_clearance']['sha256']:
        raise ValueError('Managed-job clearance SHA mismatch')
    clearance = json.loads(clearance_path.read_text())
    if not (clearance['managed_h5_jobs_clear'] is True
            and clearance['host_process_scan_completed'] is False
            and clearance['atomic_lock_still_required'] is True
            and clearance['unresolved'] == []):
        raise ValueError('Unclosed managed H5 job or unsupported clearance claim')
    bindings = list(w['bindings']) + list(clearance['bindings'])
    bindings.extend([{'path': str(clearance_path),
                      'sha256': w['managed_job_clearance']['sha256']},
                     {'path': str(window_path), 'sha256': a.window_sha256}])
    for b in bindings:
        if sha(b['path']) != b['sha256']:
            raise ValueError('Binding drift: ' + b['path'])
    worker = inside(w['worker'], REVIEW)
    if not any(Path(b['path']).resolve() == worker for b in bindings):
        raise ValueError('Worker missing exact binding')
    output = inside(w['output'], REVIEW)
    if output != REVIEW / 'synthetic_window_v1':
        raise ValueError('Only the reserved v1 synthetic output is allowed')
    if output.exists():
        raise FileExistsError('No overwrite, retry or resume')
    # Existing shared lock; do not alter stale PID text or inspect any process.
    lock = (H5.parent / 'H5_PROCESS.lock').open('rb')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    collector_path = H5 / 'direct_z_impl_v1/ops/build_managed_job_record.py'
    if not any(Path(b['path']).resolve() == collector_path for b in bindings):
        raise ValueError('Managed inventory collector requires an exact window binding')
    spec = importlib.util.spec_from_file_location('bound_managed_job_collector', collector_path)
    collector = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(collector)
    if sorted(str(p) for p in collector.collect()) != sorted(clearance['inventory_paths']):
        raise ValueError('Managed launch inventory changed before lock admission')
    for b in clearance['bindings']:
        if sha(b['path']) != b['sha256']:
            raise ValueError('Managed job binding changed: ' + b['path'])
    reserve = REVIEW / 'SYNTHETIC_ATTEMPT_v1.json'
    with reserve.open('x') as f:
        json.dump({'window_sha256': a.window_sha256,
                   'started_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
                   'retries': 0, 'scope': 'synthetic_only'}, f)
        f.write('\n')
    output.mkdir(exist_ok=False)
    env = dict(os.environ)
    env.update({k: '1' for k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS',
                               'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS',
                               'NUMEXPR_NUM_THREADS')})
    env['PYTHONDONTWRITEBYTECODE'] = '1'
    env['H5_LOCK_FD'] = str(lock.fileno())
    env['H5_SUPERVISOR_PID'] = str(os.getpid())
    argv = [sys.executable, str(worker), '--window', str(window_path),
            '--window-sha256', a.window_sha256, '--out', str(output)]
    base = {'window_sha256': a.window_sha256, 'outer_seconds': CAP,
            'forward_max': 132, 'timeout_grace_seconds': 0,
            'natural_execution_authorized': False, 'argv': argv,
            'managed_clearance_sha256': w['managed_job_clearance']['sha256']}
    dump(output / 'COMMAND.json', base)
    process = None
    error = None
    timed_out = False
    rc = None
    try:
        with (output / 'console.log').open('x') as log:
            remaining = CAP - (time.monotonic() - start)
            if remaining <= 0:
                raise TimeoutError('Admission consumed the whole180s window')
            process = subprocess.Popen(argv, cwd=H5.parent, env=env,
                                       stdin=subprocess.DEVNULL, stdout=log,
                                       stderr=subprocess.STDOUT, start_new_session=True,
                                       pass_fds=(lock.fileno(),))
            try:
                rc = process.wait(timeout=max(0.0, CAP - (time.monotonic() - start)))
            except subprocess.TimeoutExpired:
                timed_out = True
                os.killpg(process.pid, signal.SIGKILL)
                rc = process.wait()
    except Exception as exc:
        error = type(exc).__name__ + ': ' + str(exc)
        if process is not None and process.poll() is None:
            os.killpg(process.pid, signal.SIGKILL)
            rc = process.wait()
    child_wall = time.monotonic() - start
    drift = []
    for b in bindings:
        try:
            if sha(b['path']) != b['sha256']:
                drift.append({'path': b['path'], 'error': 'sha256 mismatch'})
        except Exception as exc:
            drift.append({'path': b['path'], 'error': type(exc).__name__ + ': ' + str(exc)})
    worker_status = None
    try:
        worker_result = json.loads((output / 'RESULT.json').read_text())
        worker_status = worker_result['status']
        if worker_result['actual_model_forward'] > 132:
            raise ValueError('Worker forward cap exceeded')
        if worker_result['natural_forward'] != 0:
            raise ValueError('Worker natural calls are forbidden')
    except Exception as exc:
        error = (error + '; ' if error else '') + type(exc).__name__ + ': ' + str(exc)
    wall = time.monotonic() - start
    ok = (rc == 0 and worker_status == 'PASS' and not timed_out
          and error is None and not drift and wall <= CAP)
    dump(output / 'PROCESS_RESULT.json', {
        **base, 'status': 'PASS' if ok else 'BLOCKED', 'exit_code': rc,
        'timed_out': timed_out, 'error': error, 'binding_errors': drift,
        'worker_status': worker_status, 'wall_seconds': wall,
        'wall_through_child_exit_seconds': child_wall,
        'overrun_seconds': max(0.0, wall - CAP),
        'wall_scope': 'main entry through child exit and final bindings/report checks; final result serialization excluded; no compute grace',
        'natural_admission': False})
    # Keep the same open file description and exclusive lock until all work ends.
    fcntl.flock(lock, fcntl.LOCK_UN)
    lock.close()
    return 0 if ok else 2


if __name__ == '__main__':
    raise SystemExit(main())
