#!/usr/bin/env python3
"""One hash-approved AMP H5 process; hard660s, no timeout grace or retries."""
import argparse
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time

H = Path('/Users/loctek/Documents/Codex/2026-10-11/task-4/amp_h5')
W = H.parent

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def dump(p, obj):
    q = p.with_suffix(p.suffix + '.tmp')
    q.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    q.replace(p)

def main():
    started = time.monotonic()
    started_utc = datetime.datetime.now(datetime.timezone.utc).isoformat()
    ap = argparse.ArgumentParser()
    ap.add_argument('--plan', required=True)
    ap.add_argument('--receipt', required=True)
    a = ap.parse_args()
    planpath, receiptpath = Path(a.plan).resolve(), Path(a.receipt).resolve()
    for p in (planpath, receiptpath):
        if H not in p.parents:
            raise ValueError('H5 paths required')
    plan = json.loads(planpath.read_text())
    receipt = json.loads(receiptpath.read_text())
    if (receipt.get('status') != 'PASS' or receipt.get('authorized') is not True
            or receipt.get('blocked') is not False
            or receipt.get('plan_sha256') != sha(planpath)
            or receipt.get('supervisor_sha256') != sha(__file__)):
        raise ValueError('Exact independent preflight required')
    if plan['outer_timeout_seconds'] != 660 or plan['workspace'] != str(W):
        raise ValueError('Frozen660s/workspace mismatch')
    for b in plan['bindings']:
        if sha(b['path']) != b['sha256']:
            raise ValueError('Binding drift: ' + b['path'])
    execution = Path(plan['execution_root']).resolve()
    output = Path(plan['natural_output']).resolve()
    if H not in execution.parents or H not in output.parents:
        raise ValueError('Outputs must be in H5')
    if execution.exists() or output.exists():
        raise FileExistsError('No overwrite, retry or implicit resume')
    # Reuse the existing cross-H5 lock used by all prior supervisors.
    lock = (W / 'H5_PROCESS.lock').open('r+')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    lock.seek(0); lock.truncate(); lock.write(str(os.getpid())); lock.flush()
    execution.mkdir(parents=True, exist_ok=False)
    argv = plan['argv_prefix'] + ['--receipt', str(receiptpath), '--receipt-sha256', sha(receiptpath)]
    base = {'scope_id': plan['scope_id'], 'state': 'starting', 'supervisor_pid': os.getpid(),
            'started_utc': started_utc,
            'argv': argv, 'plan_sha256': sha(planpath), 'receipt_sha256': sha(receiptpath),
            'outer_timeout_seconds': 660, 'retry_allowed': False, 'timeout_grace_seconds': 0}
    dump(execution / 'COMMAND.json', base)
    dump(execution / 'STATUS.json', base)
    env = dict(os.environ)
    env.update({k: '1' for k in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
                               'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS']})
    env['PYTHONDONTWRITEBYTECODE'] = '1'
    rc, error, timed_out, pid = None, None, False, None
    with (execution / 'console.log').open('x') as log:
        try:
            if time.monotonic() - started >= 660:
                timed_out = True
                raise TimeoutError('Admission exhausted660s; child not launched')
            process = subprocess.Popen(argv, cwd=W, env=env, stdin=subprocess.DEVNULL,
                                       stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            pid = process.pid
            dump(execution / 'STATUS.json', {**base, 'state': 'running', 'child_pid': pid})
            try:
                rc = process.wait(timeout=max(0.0, 660 - (time.monotonic() - started)))
            except subprocess.TimeoutExpired:
                timed_out = True
                os.killpg(pid, signal.SIGKILL)
                rc = process.wait()
        except Exception as exc:
            error = type(exc).__name__ + ': ' + str(exc)
            if pid is not None and process.poll() is None:
                os.killpg(pid, signal.SIGKILL)
                rc = process.wait()
    wall = time.monotonic() - started
    binding_errors = []
    for b in plan['bindings']:
        try:
            if sha(b['path']) != b['sha256']:
                binding_errors.append({'path': b['path'], 'error': 'sha256 mismatch'})
        except Exception as exc:
            binding_errors.append({'path': b['path'], 'error': type(exc).__name__ + ': ' + str(exc)})
    unchanged = not binding_errors
    wall_after_bindings = time.monotonic() - started
    status = 'completed' if rc == 0 and not timed_out and error is None and unchanged and wall_after_bindings <= 660 else 'failed'
    result = {**base, 'state': status, 'child_pid': pid, 'exit_code': rc, 'timed_out': timed_out,
              'error': error, 'wall_seconds': wall_after_bindings,
              'wall_seconds_through_child_exit': wall,
              'overrun_seconds': max(0.0, wall_after_bindings-660),
              'all_bindings_unchanged': unchanged,
              'binding_errors': binding_errors,
              'finished_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'wall_scope': 'Supervisor main entry through child exit and final binding check, including admission hashing and initial bookkeeping. Final result writes excluded; no additional compute grace.',
              'output_requires_independent_postflight': True}
    dump(execution / 'RESULT.json', result)
    dump(execution / 'STATUS.json', result)
    print(json.dumps({'state': status, 'wall_seconds': wall_after_bindings, 'exit_code': rc, 'timed_out': timed_out}), flush=True)
    return 0 if status == 'completed' else 2

if __name__ == '__main__':
    raise SystemExit(main())
