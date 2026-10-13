#!/usr/bin/env python3
"""Read only saved H5 smoke/H4 tensors; no models, datasets, RNG, or generation."""
from pathlib import Path
import json, hashlib, sys
sys.dont_write_bytecode = True
import numpy as np
import torch
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
R=Path('/Users/loctek/Documents/Codex/2026-10-09/task-8/amp_commsbio_20261011/rebuild_experiments_20261011')
W=Path('/Users/loctek/Documents/Codex/2026-10-11/task-4/amp_h5')
O=W/'feasibility/coordinate_audit'
inputs={}
def bind(p):
    p=Path(p); assert 'test' not in p.name.lower()
    inputs[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
    return p
def readj(p): return json.loads(bind(p).read_text())
def arrays(p,keys):
    with np.load(bind(p),allow_pickle=False) as z: return {k:z[k].copy() for k in keys}
def stats(x):
    x=np.asarray(x); y=x[np.isfinite(x)].astype(np.float64)
    out={'shape':list(x.shape),'dtype':str(x.dtype),'n':int(x.size),'finite':int(y.size)}
    if y.size: out.update(min=float(y.min()),max=float(y.max()),mean=float(y.mean()),std=float(y.std()),q01=float(np.quantile(y,.01)),median=float(np.median(y)),q99=float(np.quantile(y,.99)))
    return out
def exponent_stats(logs):
    # Pointwise algebra of stored values, not sampling or model execution.
    scale=torch.from_numpy(logs).exp().numpy()
    half=torch.from_numpy(logs).mul(.5).exp().numpy()
    return {'exp_logsigma':stats(scale),'exp_logsigma_positive_infinity':int(np.isposinf(scale).sum()),'exp_logsigma_zero':int((scale==0).sum()),
            'hypothetical_wrong_logvariance_half_exp_positive_infinity':int(np.isposinf(half).sum()),
            'hypothetical_wrong_logvariance_half_exp_zero':int((half==0).sum())}
def channels(x): return {'mu':stats(x[:,0]),'logsigma':stats(x[:,1]),**exponent_stats(x[:,1])}
def exact(a,b): return a.dtype==b.dtype and a.shape==b.shape and a.tobytes()==b.tobytes()
static=readj(R/'diagnostics_h4_h2_generation_bridge_v1/STATIC_INPUTS.json')
mu=[]; logs=[]
for entry in static['cache']:
    p=R/entry['path']; z=arrays(p,['mu','logsigma'])
    assert inputs[str(p)]==entry['sha256']
    mu.append(z['mu']);logs.append(z['logsigma'])
oldtrain=np.stack((np.concatenate(mu),np.concatenate(logs)),axis=1)
oldscaler=readj(R/'runner/dbaasp_v4_cpu_pilot_001/train_feature_scaler.json')
res={'identity':'AMP_H5_COORDINATE_SAVED_NUMERIC_AUDIT','no_model_loads':True,'no_generation':True,'no_training':True,'no_test_reads':True,
     'float32_exp_overflow_log_threshold_float64':float(np.log(np.finfo(np.float32).max)),
     'h4_train_cached':channels(oldtrain),'h5_smoke':{},'h4_saved_generation':{}}
for arm in ['categorical_ce','raw_onehot_mse','scalar_legacy']:
    d=W/'runs/smoke_v1'/arm
    train=arrays(d/'train_posterior_parameters.npz',['values'])['values']
    scaler=readj(d/'train_feature_scaler.json')
    row={'train':channels(train),'scaler_numeric_equal_to_h4':all(scaler[k]==oldscaler[k] for k in ['global','residue','fit_record_ids']),'generated':{}}
    for terminal in ['FREE','AMD']:
        x=arrays(d/f'generated_posterior_{terminal}.npz',['values'])['values']
        a=channels(x)
        for k,index in [('mu',0),('logsigma',1)]:
            a[k]['outside_own_train_global_minmax_count']=int(((x[:,index]<train[:,index].min())|(x[:,index]>train[:,index].max())).sum())
        row['generated'][terminal]=a
    res['h5_smoke'][arm]=row
schedule=arrays(R/'diagnostics_h4_h2_generation_bridge_v1/run/SCHEDULE.npz',['beta','alpha','alpha_hat'])
for terminal in ['FREE','AMD']:
    d=arrays(R/f'diagnostics_h4_h2_generation_bridge_v1/run/replay__{terminal}__posterior_parameter_ddpm.npz',
             ['ddpm_final_parameters','posterior_scale','epsilon','latent','hidden','raw_replay','raw_original','ddpm_last_input','ddpm_last_prediction','ddpm_last_noise','ddpm_last_state'])
    x=d['ddpm_final_parameters']; a=channels(x)
    scale=torch.from_numpy(x[:,1]).exp().numpy()
    z=(torch.from_numpy(x[:,0])+torch.from_numpy(d['epsilon'])*torch.from_numpy(scale)).numpy()
    before=torch.from_numpy(d['ddpm_last_input']);pred=torch.from_numpy(d['ddpm_last_prediction']);noise=torch.from_numpy(d['ddpm_last_noise'])
    alpha=torch.from_numpy(schedule['alpha'])[1];abar=torch.from_numpy(schedule['alpha_hat'])[1];beta=torch.from_numpy(schedule['beta'])[1]
    nxt=((before-(1-alpha)/(1-abar).sqrt()*pred)/alpha.sqrt()+beta.sqrt()*noise).numpy()
    a.update(latent=stats(d['latent']),hidden=stats(d['hidden']),raw=stats(d['raw_replay']),
             exp_matches_saved_bitwise=exact(scale,d['posterior_scale']),reparameterization_matches_saved_bitwise=exact(z,d['latent']),
             final_reverse_update_matches_saved_bitwise=exact(nxt,d['ddpm_last_state']),
             saved_last_equals_final_bitwise=exact(d['ddpm_last_state'],x),raw_replay_equals_original_bitwise=exact(d['raw_replay'],d['raw_original']))
    res['h4_saved_generation'][terminal]=a
res['input_sha256']=inputs
(O/'SAVED_NUMERIC_AUDIT.json').write_text(json.dumps(res,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
print(json.dumps({'h4_train':res['h4_train_cached'],'h5_smoke':res['h5_smoke'],'h4_generation':res['h4_saved_generation']},ensure_ascii=False,indent=2))
