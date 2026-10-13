#!/usr/bin/env python3
"""Detach exactly one approved H5 supervisor, retaining its finite lifetime."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess

H = Path('/Users/loctek/Documents/Codex/2026-10-11/task-4/amp_h5')
PY = '/Users/loctek/Documents/Codex/2026-10-09/task-8/amp_commsbio_20261011/rebuild_experiments_20261011/env/bin/python'

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--plan', required=True)
    ap.add_argument('--receipt', required=True)
    ap.add_argument('--record', required=True)
    a = ap.parse_args()
    planpath, receiptpath, record = [Path(x).resolve() for x in (a.plan, a.receipt, a.record)]
    for p in (planpath, receiptpath, record):
        if H not in p.parents:
            raise ValueError('H5 scope required')
    plan, receipt = json.loads(planpath.read_text()), json.loads(receiptpath.read_text())
    sup = Path(__file__).with_name('supervise_once.py').resolve()
    if (receipt.get('status') != 'PASS' or receipt.get('authorized') is not True or receipt.get('blocked') is not False
            or receipt.get('plan_sha256') != sha(planpath) or receipt.get('supervisor_sha256') != sha(sup)):
        raise ValueError('Independent exact-plan authorization required')
    for b in plan['bindings']:
        if sha(b['path']) != b['sha256']:
            raise ValueError('Binding drift')
    if not any(Path(b['path']).resolve() == Path(__file__).resolve() and b['sha256'] == sha(__file__) for b in plan['bindings']):
        raise ValueError('Launcher not bound')
    if Path(plan['execution_root']).exists() or Path(plan['natural_output']).exists():
        raise FileExistsError('Natural run already exists')
    record.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive record reservation happens before Popen, closing the duplicate-launch race.
    with record.open('x') as reservation:
        json.dump({'state': 'reserved', 'reserved_utc': datetime.datetime.now(datetime.timezone.utc).isoformat()}, reservation)
    argv = [PY, '-u', '-B', str(sup), '--plan', str(planpath), '--receipt', str(receiptpath)]
    with record.with_suffix('.stdout.log').open('x') as log:
        process = subprocess.Popen(argv, cwd=H.parent, env={**os.environ, 'PYTHONDONTWRITEBYTECODE': '1'},
                                   stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                   start_new_session=True, close_fds=True)
    try:
        keeper = subprocess.Popen(['/usr/bin/caffeinate', '-i', '-w', str(process.pid)],
                                  stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        keeper_pid = keeper.pid
    except OSError:
        keeper_pid = None
    data = {'state': 'launched', 'launched_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'supervisor_pid': process.pid, 'argv': argv, 'plan_sha256': sha(planpath),
            'receipt_sha256': sha(receiptpath), 'temporary_keep_awake_pid': keeper_pid,
            'retry_allowed': False, 'system_settings_changed': False}
    record.write_text(json.dumps(data, indent=2) + '\n')
    print(json.dumps(data), flush=True)

if __name__ == '__main__':
    main()
