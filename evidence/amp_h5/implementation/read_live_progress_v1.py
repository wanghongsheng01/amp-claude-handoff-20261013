#!/usr/bin/env python3
"""Read-only compact progress snapshot; never infer exit from stale files."""
import argparse, datetime, json, pathlib
p=argparse.ArgumentParser()
p.add_argument('--execution',required=True)
p.add_argument('--arm',required=True)
p.add_argument('--run')
a=p.parse_args()
r=pathlib.Path(a.execution)
out={'observed_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
s=r/'SUPERVISOR_STATUS.json'
if s.exists():
    state=json.loads(s.read_text())
    out['supervisor']={k:state[k] for k in ('pid','state','created_utc','completed_utc') if k in state}
    out['supervisor']['jobs']=[{k:j[k] for k in ('id','pid','state','exit_code','seconds','outer_timeout','error') if k in j} for j in state.get('jobs',[])]
log=r/(a.arm+'.console.log')
if log.exists():
    rows=log.read_text().splitlines()
    out['console_line_count']=len(rows)
    selected=[]
    for row in reversed(rows):
        try: x=json.loads(row)
        except (ValueError,TypeError): continue
        if isinstance(x,dict):
            wanted=('step','stage','phase','terminal','t','elapsed_seconds','elapsed_stage_seconds','vae_updates','ddpm_updates','state','error','loss','reconstruction','kl','lr')
            selected.append({k:x[k] for k in wanted if k in x} or x)
            if len(selected)==2:break
    out['last_json_rows']=list(reversed(selected))
    if rows and not selected:out['last_line']=rows[-1]
if a.run:
    run=pathlib.Path(a.run)
    for name in ('generation_progress.jsonl','generation_summary.json','status.json'):
        f=run/name
        if not f.exists():continue
        try:
            lines=f.read_text().splitlines()
            out[name]=json.loads(lines[-1]) if name.endswith('jsonl') else json.loads(f.read_text())
        except (ValueError,IndexError) as e:out[name]={'read_error':str(e)}
print(json.dumps(out,ensure_ascii=False,indent=2))
