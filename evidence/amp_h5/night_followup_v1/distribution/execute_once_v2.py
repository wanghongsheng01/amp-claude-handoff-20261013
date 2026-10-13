#!/usr/bin/env python3
"""One-shot read-only analysis launcher; 180s outer watchdog, no retry."""
import argparse,hashlib,json,subprocess,time
from pathlib import Path
HERE=Path(__file__).resolve().parent
PYTHON='/Users/loctek/Documents/Codex/2026-10-09/task-8/amp_commsbio_20261011/rebuild_experiments_20261011/env/bin/python'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def record(p):return {'path':str(p),'bytes':p.stat().st_size,'sha256':sha(p)}
def dump(p,d):
 with p.open('x') as f:json.dump(d,f,ensure_ascii=False,indent=2);f.write('\n')
ap=argparse.ArgumentParser();ap.add_argument('--preaudit',required=True);ap.add_argument('--preaudit-sha256',required=True);a=ap.parse_args()
protocol=HERE/'PROTOCOL_v2.json';p=json.loads(protocol.read_text());receipt=Path(a.preaudit)
assert sha(receipt)==a.preaudit_sha256
r=json.loads(receipt.read_text());assert r['status']=='PASS' and r['scope_id']=='AMP_H5_NIGHT_DISTRIBUTION_V2' and r['protocol_sha256']==sha(protocol) and r['implementation_sha256']==p['implementation_sha256'] and r['natural_statistics_authorized'] is True
assert sha(__file__)==p['execution_wrapper_sha256'] and not Path(p['output_directory']).exists()
out=HERE/'execution_v2';out.mkdir(exist_ok=False)
argv=[PYTHON,'-B',p['implementation_path'],'--protocol',str(protocol),'--preaudit',str(receipt),'--preaudit-sha256',a.preaudit_sha256]
dump(out/'COMMAND.json',{'argv':argv,'cwd':str(HERE),'timeout_seconds':180,'retry':False,'protocol':record(protocol),'preaudit':record(receipt),'execution_wrapper':record(Path(__file__))})
started=time.perf_counter();timed_out=False;error=None
with (out/'stdout.log').open('x') as stdout,(out/'stderr.log').open('x') as stderr:
 try:completed=subprocess.run(argv,cwd=HERE,stdout=stdout,stderr=stderr,timeout=180,check=False);code=completed.returncode
 except subprocess.TimeoutExpired as e:timed_out=True;code=124;error=str(e)
wall=time.perf_counter()-started
closing_path=Path(p['output_directory'])/'CLOSING_LEDGER.json'
closing=json.loads(closing_path.read_text()) if closing_path.exists() else None
result={'exit_code':code,'outer_timeout':timed_out,'wall_seconds':wall,'error':error,'analysis_closing_state':None if closing is None else closing['state'],'model_calls':0,'test_reads':0,'no_retry':True,'PROMOTE':'HOLD','protocol':record(protocol),'preaudit':record(receipt)}
dump(out/'RESULT.json',result)
dump(out/'MANIFEST.json',{'files':[record(f) for f in sorted(out.iterdir()) if f.is_file()],'natural_attempts':1,'model_calls':0})
print(json.dumps(result))
raise SystemExit(code if code else (0 if closing and closing['state']=='completed' else 1))
