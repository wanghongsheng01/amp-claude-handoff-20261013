"""Saved latent numeric summaries only; no model, no biological/quality metric."""
from pathlib import Path
import os,json,hashlib,datetime
os.environ['OPENBLAS_NUM_THREADS']='1'
import numpy as np
H=Path('/Users/loctek/Documents/Codex/2026-10-11/task-4/amp_h5');B=H/'runs/ce_checkpoint_recovery_v1';T=H/'runs/formal_v2/categorical_ce';O=H/'review/independent'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
audit=json.loads((O/'CE_CHECKPOINT_RECOVERY_TERMINAL_AUDIT_v1.json').read_text());assert not audit['failed_checks']
with np.load(T/'train_posterior_parameters.npz',allow_pickle=False) as a:v=a['values'].astype(np.float64);labels=a['labels'].copy()
records={}
for label,t in enumerate(('FREE','AMD')):
 target=v[labels==label]; tr_mu=target[:,0];tr_logs=target[:,1];tr_sigma=np.exp(tr_logs)
 train_per_record=np.sqrt(np.mean(tr_mu**2+tr_sigma**2,axis=(1,2)))
 with np.load(B/f'generated_{t}.npz',allow_pickle=False) as a:g=a['posterior_parameters'].astype(np.float64);z=a['z'].astype(np.float64)
 mu,logs=g[:,0],g[:,1];sig=np.exp(logs)
 requests=[]
 for i in range(16):
  requests.append({'request_index':i,'mu_min':float(mu[i].min()),'mu_max':float(mu[i].max()),'logsigma_min':float(logs[i].min()),'logsigma_max':float(logs[i].max()),
     'sigma_min':float(sig[i].min()),'sigma_max':float(sig[i].max()),'actual_z_rms':float(np.sqrt(np.mean(z[i]**2))),
     'generated_posterior_analytic_qz_rms':float(np.sqrt(np.mean(mu[i]**2+sig[i]**2)))})
 records[t]={'train_per_record_analytic_qz_rms':{'min':float(train_per_record.min()),'max':float(train_per_record.max()),'q05_q50_q95':np.quantile(train_per_record,[.05,.5,.95]).tolist()},
  'generated_mu_coordinates_outside_observed_terminal_train_range':int(((mu<tr_mu.min())|(mu>tr_mu.max())).sum()),
  'generated_logs_coordinates_outside_observed_terminal_train_range':int(((logs<tr_logs.min())|(logs>tr_logs.max())).sum()),
  'coordinate_count':int(mu.size),'generated_logs_std_divided_by_terminal_train':float(logs.std()/tr_logs.std()),
  'generated_mu_mean_minus_terminal_train_mean':float(mu.mean()-tr_mu.mean()),'requests':requests}
report={'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'scope':'Saved latent scale description; no quality gate or adaptive threshold','precision':'Float64 descriptive exp and summaries; strict production float32 positivity and finite checks were independently established in the bound terminal audit.',
 'terminals':records,'bound_terminal_audit_sha256':sha(O/'CE_CHECKPOINT_RECOVERY_TERMINAL_AUDIT_v1.json'),
 'input_bindings':[{'path':str(p),'sha256':sha(p)} for p in [T/'train_posterior_parameters.npz',B/'generated_FREE.npz',B/'generated_AMD.npz']],
 'limitations':['Coordinate counts do not define independent sample size. There are16 generated requests per terminal from one fitted model/seed configuration.','Observed train extrema and per-record scales are descriptive, not preregistered pass thresholds, confidence bounds or goodness-of-fit tests.','A recovered model may show tail or mean mismatch while remaining numerically feasible; those mismatches must not be silently converted into quality-based control-arm selection.'],
 'actions':{'model_calls':0,'training':0,'new_generation':0,'natural_CSV_reads':0,'test37_reads':0}}
p=O/'CE_CHECKPOINT_RECOVERY_SCALE_DETAIL_v1.json'
with p.open('x') as f:json.dump(report,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
print(json.dumps({'input_count':len(report['input_bindings']),'summary':{t:{k:value for k,value in r.items() if k!='requests'} for t,r in records.items()},'path':str(p),'sha256':sha(p)},ensure_ascii=False))
