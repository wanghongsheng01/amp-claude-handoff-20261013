#!/usr/bin/env python3
"""Serial finite runner for independently reviewed, hash-bound AMP H5 jobs."""
import argparse,datetime,fcntl,hashlib,json,os,pathlib,subprocess,sys,time
sys.dont_write_bytecode=True

def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def dump(p,x):
    p=pathlib.Path(p);t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n');t.replace(p)
def main():
    p=argparse.ArgumentParser();p.add_argument('--plan',required=True);p.add_argument('--receipt',required=True);a=p.parse_args()
    plan=json.loads(pathlib.Path(a.plan).read_text());receipt=json.loads(pathlib.Path(a.receipt).read_text())
    if receipt['plan_sha256']!=sha(a.plan) or receipt['supervisor_sha256']!=sha(__file__) or not receipt['authorized']: raise ValueError('unbound START')
    for item in plan['bindings']:
        if sha(item['path'])!=item['sha256']: raise ValueError('prelocked file drift: '+item['path'])
    root=pathlib.Path(plan['output_root']);root.mkdir(parents=True,exist_ok=False)
    lock=open(pathlib.Path(plan['workspace'])/'H5_PROCESS.lock','a+')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    lock.seek(0);lock.truncate();lock.write(str(os.getpid()));lock.flush()
    state={'pid':os.getpid(),'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'state':'running','plan_sha256':sha(a.plan),'jobs':[]}
    dump(root/'SUPERVISOR_STATUS.json',state)
    env=os.environ.copy();env.update({k:'1' for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS')});env['PYTHONDONTWRITEBYTECODE']='1'
    for job in plan['jobs']:
        begun=time.perf_counter();row={'id':job['id'],'argv':job['argv'],'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'state':'running'};state['jobs'].append(row);dump(root/'SUPERVISOR_STATUS.json',state)
        try:
            with (root/(job['id']+'.console.log')).open('x') as log:
                process=subprocess.Popen(job['argv'],cwd=plan['workspace'],env=env,stdout=log,stderr=subprocess.STDOUT)
                row['pid']=process.pid;dump(root/'SUPERVISOR_STATUS.json',state)
                try: row['exit_code']=process.wait(timeout=job['outer_timeout_seconds'])
                except subprocess.TimeoutExpired:
                    process.terminate()
                    try: process.wait(timeout=30)
                    except subprocess.TimeoutExpired: process.kill();process.wait()
                    row['exit_code']=124;row['outer_timeout']=True
            row['state']='completed' if row['exit_code']==0 else 'failed'
        except Exception as e:row.update(state='launcher_failed',error=repr(e))
        row['seconds']=time.perf_counter()-begun;dump(root/'SUPERVISOR_STATUS.json',state)
        print(json.dumps(row),flush=True)
    state['state']='completed' if all(j['state']=='completed' for j in state['jobs']) else 'completed_with_failures';state['completed_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat();dump(root/'SUPERVISOR_STATUS.json',state)
    return 0 if state['state']=='completed' else 2
if __name__=='__main__':raise SystemExit(main())
