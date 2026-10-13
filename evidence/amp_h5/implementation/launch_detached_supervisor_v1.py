#!/usr/bin/env python3
"""Launch one hash-reviewed bounded H5 supervisor in a separate process session."""
import argparse, datetime, hashlib, json, os, pathlib, subprocess

W = pathlib.Path('/Users/loctek/Documents/Codex/2026-10-11/task-4')
H = W / 'amp_h5'
SUP = H / 'implementation/supervise.py'
PY = '/Users/loctek/Documents/Codex/2026-10-09/task-8/amp_commsbio_20261011/rebuild_experiments_20261011/env/bin/python'

def digest(p):
    return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--plan', required=True)
    p.add_argument('--receipt', required=True)
    p.add_argument('--launch-record', required=True)
    a = p.parse_args()
    plan_path, receipt_path, record = map(lambda x: pathlib.Path(x).resolve(), (a.plan, a.receipt, a.launch_record))
    for path in (plan_path, receipt_path, record):
        if H not in path.parents:
            raise ValueError('H5 workspace scope required')
    plan, receipt = json.loads(plan_path.read_text()), json.loads(receipt_path.read_text())
    if not receipt.get('authorized') or receipt['plan_sha256'] != digest(plan_path) or receipt['supervisor_sha256'] != digest(SUP):
        raise ValueError('Independent bounded-plan authorization missing or changed')
    if plan['workspace'] != str(W) or pathlib.Path(plan['output_root']).exists():
        raise ValueError('Unexpected workspace or already existing supervisor output')
    if not any(pathlib.Path(x['path']).resolve() == pathlib.Path(__file__).resolve() and x['sha256'] == digest(__file__) for x in plan['bindings']):
        raise ValueError('Launcher must be bound in the independently reviewed plan')
    for binding in plan['bindings']:
        if digest(binding['path']) != binding['sha256']:
            raise ValueError('Plan binding drift')
    if record.exists():
        raise FileExistsError('Never relaunch an existing launch record')
    record.parent.mkdir(parents=True, exist_ok=True)
    stdout = record.with_suffix('.stdout.log')
    env = os.environ.copy()
    env.update({k:'1' for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS')})
    env['PYTHONDONTWRITEBYTECODE'] = '1'
    argv = [PY, '-u', '-B', str(SUP), '--plan', str(plan_path), '--receipt', str(receipt_path)]
    with stdout.open('x') as log:
        proc = subprocess.Popen(argv, cwd=W, env=env, stdin=subprocess.DEVNULL,
                                stdout=log, stderr=subprocess.STDOUT, start_new_session=True, close_fds=True)
    try:
        keeper = subprocess.Popen(['/usr/bin/caffeinate', '-i', '-w', str(proc.pid)],
                                  stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                                  stderr=subprocess.DEVNULL, start_new_session=True, close_fds=True)
        keep_awake = {'pid':keeper.pid,'until_supervisor_pid_exits':proc.pid}
    except OSError as exc:
        keep_awake = {'error':repr(exc),'supervisor_already_launched':True}
    data = {'launched_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'supervisor_pid':proc.pid,'argv':argv,'plan_sha256':digest(plan_path),
            'receipt_sha256':digest(receipt_path),'launcher_sha256':digest(__file__),
            'separate_process_session':True,'stdout':str(stdout),
            'lifetime_bound':'supervise.py per-job outer timeout and global H5_PROCESS.lock',
            'system_settings_changed':False,'temporary_keep_awake':keep_awake}
    with record.open('x') as f:
        json.dump(data, f, indent=2)
    print(json.dumps(data), flush=True)
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
