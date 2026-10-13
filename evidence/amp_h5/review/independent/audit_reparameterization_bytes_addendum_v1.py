#!/usr/bin/env python3
"""Narrow dtype/shape/byte provenance addendum, run only after scalar terminal.
Replays original RNG consumption; no model, forward or new generation request.
"""
from pathlib import Path
import argparse,datetime,hashlib,json,os
W=Path('/Users/loctek/Documents/Codex/2026-10-11/task-4');H=W/'amp_h5';O=H/'review/independent'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def main():
 p=argparse.ArgumentParser();p.add_argument('--scalar-terminal-confirmed',action='store_true',required=True);p.parse_args()
 out=O/'REPARAMETERIZATION_BYTE_LEVEL_ADDENDUM_v1.json'
 if out.exists():raise FileExistsError('Preserve prior addendum')
 scalar=read(H/'runs/single_scalar_legacy_execution_v2/SUPERVISOR_STATUS.json')
 if scalar.get('state')=='running' or any(x.get('state')=='running' for x in scalar.get('jobs',[])):raise RuntimeError('Do not inspect live scalar')
 for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
 import numpy as np
 import torch
 torch.set_num_threads(1)
 if torch.__version__!='2.2.2' or np.__version__!='1.26.4':raise RuntimeError('Original runtime required')
 groups=[('CE_recovery',H/'runs/ce_checkpoint_recovery_v1','audit_ce_checkpoint_recovery_terminal_v1.py','CE_CHECKPOINT_RECOVERY_TERMINAL_AUDIT_v1.json'),('MSE_v2',H/'runs/formal_v2/raw_onehot_mse','audit_mse_v2_terminal_v1.py','MSE_V2_TERMINAL_AUDIT_v1.json'),('SCALAR_v2',H/'runs/formal_v2/scalar_legacy','audit_scalar_v2_terminal_v1.py','SCALAR_V2_TERMINAL_AUDIT_v1.json')]
 facts=[];bindings=[];rng_before=torch.get_rng_state().clone()
 for method,root,script_name,report_name in groups:
  script=O/script_name;report=O/report_name
  if not report.exists():raise RuntimeError('Complete normal terminal or separate failure audit before byte addendum: '+method)
  bindings += [{'path':str(f),'sha256':sha(f)} for f in (script,report,root/'status.json')]
  if 'z.numpy().tobytes()==data[\'z\'].tobytes()' not in script.read_text():raise RuntimeError('Prior audit byte-expression changed')
  for label,t in enumerate(('FREE','AMD')):
   f=root/f'generated_{t}.npz'
   if not f.exists():facts.append({'method':method,'terminal':t,'status':'NO_SAVED_SUCCESS_ARRAY','byte_identity_claim':False});continue
   before=sha(f);bindings.append({'path':str(f),'sha256':before})
   with np.load(f,allow_pickle=False) as z:post=z['posterior_parameters'].copy();saved=z['z'].copy()
   if post.dtype!=np.float32 or saved.dtype!=np.float32 or post.shape!=(16,2,16,48) or saved.shape!=(16,16,48):raise RuntimeError('Unexpected original dtype/shape')
   xpost=torch.from_numpy(post);mu,logs=xpost[:,0],xpost[:,1]
   with torch.random.fork_rng(devices=[]),torch.no_grad():
    torch.manual_seed(422+label);initial=torch.randn(16,2,16,48)
    for unused in reversed(range(2,999)):torch.randn_like(initial)
    epsilon=torch.randn_like(mu);reconstructed=(mu+epsilon*logs.exp()).numpy()
   equal=reconstructed.dtype==saved.dtype and reconstructed.shape==saved.shape and reconstructed.tobytes()==saved.tobytes()
   facts.append({'method':method,'terminal':t,'status':'BYTE_IDENTICAL' if equal else 'BYTE_MISMATCH','dtype_saved':str(saved.dtype),'dtype_reconstructed':str(reconstructed.dtype),'shape_saved':list(saved.shape),'shape_reconstructed':list(reconstructed.shape),'saved_z_bytes_sha256':hashlib.sha256(saved.tobytes()).hexdigest(),'reconstructed_z_bytes_sha256':hashlib.sha256(reconstructed.tobytes()).hexdigest(),'epsilon_bytes_sha256':hashlib.sha256(epsilon.numpy().tobytes()).hexdigest(),'source_npz_sha256':before,'source_unchanged':sha(f)==before,'byte_identity_claim':equal})
 restored=torch.get_rng_state().numpy().tobytes()==rng_before.numpy().tobytes()
 failed=[r for r in facts if r['status']=='BYTE_MISMATCH' or r.get('source_unchanged') is False]
 report={'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'status':'PASS_ALL_AVAILABLE_SAVED_Z_BYTES' if not failed and restored else 'BYTE_IDENTITY_FINDINGS_HOLD','facts':facts,'failures':failed,'global_rng_restored_bytes':restored,'source_bindings':bindings,'audit_script_sha256':sha(__file__),'old_audit_clarification':'CE/MSE and prelocked scalar independent audits already compared numpy raw bytes, not torch.equal. This addendum explicitly records dtype, shape and byte digests using the original global RNG call forms. It does not change old reports.','reviewer_actions':{'model_constructions':0,'model_forwards':0,'new_generation_requests':0,'training_updates':0,'test37_reads':0,'RNG_replay':'Original422/423, one initial and997 reverse noises, then finalepsilon; no alternate seed/alignment or selected draws.'},'original_CE_E_L_G':'NOT_EVALUABLE_INTERRUPTION','PROMOTE':'HOLD','new_execution_authorized':False}
 if not all(sha(b['path'])==b['sha256'] for b in bindings):raise RuntimeError('Input changed during addendum')
 with out.open('x') as f:json.dump(report,f,ensure_ascii=False,indent=2);f.write('\n')
 print(json.dumps({'status':report['status'],'file':str(out),'sha256':sha(out),'facts':len(facts)}))
if __name__=='__main__':main()
