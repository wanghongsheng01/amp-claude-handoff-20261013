"""Group-intact same-n saved-latent sensitivity; NumPy/stdlib, no model calls."""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[key]='1'
import argparse
import hashlib
import json
import time
from collections import defaultdict,Counter
from datetime import datetime,timezone
from pathlib import Path
import numpy as np

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,v):Path(p).write_text(json.dumps(v,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
def corr(a,b):
    a=np.asarray(a,dtype=np.float64).ravel();b=np.asarray(b,dtype=np.float64).ravel()
    a=a-a.mean();b=b-b.mean();den=np.sqrt(np.sum(a*a)*np.sum(b*b))
    return None if den==0 else float(np.sum(a*b)/den)

def suffix_counts(weights,target):
    f,a=target;n=len(weights)
    dp=[[[0 for _ in range(a+1)] for _ in range(f+1)] for _ in range(n+1)]
    dp[n][0][0]=1
    for i in reversed(range(n)):
        wf,wa=weights[i]
        for j in range(f+1):
            for k in range(a+1):
                dp[i][j][k]=dp[i+1][j][k]+(dp[i+1][j-wf][k-wa] if j>=wf and k>=wa else 0)
    return dp

def randbelow_raw(bitgen,n):
    """Uniform integer using PCG64 raw bits and rejection; never float ratios."""
    if n<=0:raise ValueError('positive bound required')
    if n==1:return 0
    bits=(n-1).bit_length();words=(bits+63)//64;mask=(1<<bits)-1
    while True:
        raw=bitgen.random_raw(words);value=0
        for x in raw:value=(value<<64)|int(x)
        value &= mask
        if value<n:return value

def choose_subset(weights,dp,target,bitgen):
    f,a=target;selected=[]
    for i,(wf,wa) in enumerate(weights):
        if f==0 and a==0:break
        take=dp[i+1][f-wf][a-wa] if f>=wf and a>=wa else 0
        skip=dp[i+1][f][a]
        if take+skip!=dp[i][f][a] or not take+skip:raise ValueError('DP count inconsistency')
        if take and (not skip or randbelow_raw(bitgen,take+skip)<take):
            selected.append(i);f-=wf;a-=wa
    if (f,a)!=(0,0):raise ValueError('exact target not achieved')
    return selected

def reference_stats(ref):
    return {'mean':float(ref.mean()),'std':float(ref.std()),'RMS':float(np.sqrt(np.mean(ref*ref))),
            'min':float(ref.min()),'max':float(ref.max()),'coord_min':ref.min(axis=0),'coord_max':ref.max(axis=0),
            'coord_q':np.quantile(ref,[.01,.99],axis=0),
            'profiles':{name:ref.mean(axis=axes) for name,axes in [('channel',(0,2)),('position',(0,1)),('map',0)]}}

def field_metrics(x,ref):
    mean=float(x.mean());std=float(x.std());rms=float(np.sqrt(np.mean(x*x)))
    if ref['std']<=0 or ref['RMS']<=0 or std<=0 or rms<=0:raise ValueError('undefined ratio')
    result={'signed_mean_shift':mean-ref['mean'],'absolute_mean_shift_in_ref_std':abs(mean-ref['mean'])/ref['std'],
            'std_ratio':std/ref['std'],'absolute_log_std_ratio':abs(float(np.log(std/ref['std']))),
            'RMS_ratio':rms/ref['RMS'],'absolute_log_RMS_ratio':abs(float(np.log(rms/ref['RMS'])))}
    for name,axes in [('channel',(0,2)),('position',(0,1)),('map',0)]:
        profile=x.mean(axis=axes);c=corr(profile,ref['profiles'][name])
        result[name+'_mean_correlation']=c
        result[name+'_mean_correlation_distance']=None if c is None else 1-c
        result[name+'_mean_RMSE_in_ref_RMS']=float(np.sqrt(np.mean((profile-ref['profiles'][name])**2))/ref['RMS'])
    for name,lo,hi in [('pooled_minmax',ref['min'],ref['max']),('coordinate_minmax',ref['coord_min'],ref['coord_max']),('coordinate_q01_q99',ref['coord_q'][0],ref['coord_q'][1])]:
        flags=(x<lo)|(x>hi)
        result['outside_'+name]=float(flags.mean())
        result['outside_'+name+'_per_peptide']=flags.mean(axis=(1,2)).tolist()
    return result

def dependency(mu,logs):
    return {'pooled_aligned':corr(mu,logs),
            'within_peptide_centered':corr(mu-mu.mean(axis=(1,2),keepdims=True),logs-logs.mean(axis=(1,2),keepdims=True)),
            'position_mean':corr(mu.mean(axis=(0,1)),logs.mean(axis=(0,1))),
            'channel_mean':corr(mu.mean(axis=(0,2)),logs.mean(axis=(0,2)))}

def metric_bundle(posterior,ref_fields,ref_dep,ref_q):
    mu,logs=posterior[:,0],posterior[:,1];sigma=np.exp(logs)
    result={}
    for name,x in [('mu',mu),('log_sigma',logs),('sigma',sigma)]:
        for k,v in field_metrics(x,ref_fields[name]).items():result[name+'/'+k]=v
    dep=dependency(mu,logs)
    for name,value in dep.items():
        result['mu_logsigma/'+name+'_correlation']=value
        result['mu_logsigma/'+name+'_absolute_difference_from_reference']=None if value is None or ref_dep[name] is None else abs(value-ref_dep[name])
    qmean=float(mu.mean());qrms=float(np.sqrt(np.mean(mu*mu+sigma*sigma)))
    result['analytic_q/mean_shift']=qmean-ref_q['mean']
    result['analytic_q/absolute_mean_shift_in_ref_RMS']=abs(qmean-ref_q['mean'])/ref_q['RMS']
    result['analytic_q/RMS_ratio']=qrms/ref_q['RMS']
    result['analytic_q/absolute_log_RMS_ratio']=abs(float(np.log(qrms/ref_q['RMS'])))
    return result

def summary(values):
    valid=[x for x in values if x is not None]
    if not valid:return {'total_repetitions':len(values),'defined':0,'undefined':len(values)}
    x=np.array(valid,dtype=np.float64)
    if not np.isfinite(x).all():raise ValueError('nonfinite metric')
    return {'total_repetitions':len(values),'defined':len(valid),'undefined':len(values)-len(valid),
        'mean':float(x.mean()),'std_ddof0':float(x.std()),'min':float(x.min()),'max':float(x.max()),
        'quantiles':dict(zip(['.025','.25','.5','.75','.975'],map(float,np.quantile(x,[.025,.25,.5,.75,.975]))))}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--protocol',required=True);a=ap.parse_args();start=time.perf_counter()
    path=Path(a.protocol);p=json.loads(path.read_text())
    if p['scope_id']!='AMP_H5_LATENT_SAME_N_GROUP_REFERENCE_V1' or np.__version__!='1.26.4':raise ValueError('scope/runtime differs')
    if sha(__file__)!=p['implementation_sha256']:raise ValueError('script differs')
    for b in p['input_bindings']:
        if sha(b['path'])!=b['sha256']:raise ValueError('changed input: '+b['path'])
    out=Path(p['output_directory']);assert not out.exists();out.mkdir()
    binding=json.loads(Path(p['row_binding_path']).read_text());rows=binding['splits']['train']['rows']
    assert [r['model_row_id'] for r in rows]==list(range(279))
    components=defaultdict(list)
    for r in rows:components[r['component_id']].append(r['model_row_id'])
    names=sorted(components);groups=[sorted(components[n]) for n in names]
    labels=np.array([r['class'] for r in rows]);assert np.array_equal(np.bincount(labels),[126,153]) and len(groups)==224
    weights=[(sum(labels[j]==0 for j in g),sum(labels[j]==1 for j in g)) for g in groups]
    weights=[tuple(map(int,w)) for w in weights];dp=suffix_counts(weights,(16,16));count=dp[0][16][16]
    assert str(count)==p['sampler']['feasible_component_subset_count']
    bitgen=np.random.PCG64(p['sampler']['seed']);initial_state=bitgen.state
    indices=[];inclusion=Counter()
    # All indices are sealed before any latent metric is computed or natural arrays loaded.
    for repeat in range(p['sampler']['repetitions']):
        state_before=bitgen.state;selected=choose_subset(weights,dp,(16,16),bitgen)
        chosen=sorted(j for i in selected for j in groups[i]);remain=sorted(set(range(279))-set(chosen))
        assert len(chosen)==32 and len(remain)==247
        held={t:[j for j in chosen if labels[j]==label] for label,t in enumerate(('FREE','AMD'))}
        refs={t:[j for j in remain if labels[j]==label] for label,t in enumerate(('FREE','AMD'))}
        assert all(len(v)==16 for v in held.values()) and [len(refs[t]) for t in ('FREE','AMD')]==[110,137]
        assert not {rows[j]['component_id'] for j in chosen}&{rows[j]['component_id'] for j in remain}
        ids=[names[i] for i in selected];inclusion.update(ids)
        indices.append({'repetition':repeat,'components':ids,'holdout_row_ids':held,'reference_row_ids':refs,
            'selection_probability_exact':'1/'+str(count),'RNG_state_before':state_before,'RNG_state_after':bitgen.state})
    signatures=[tuple(x['components']) for x in indices]
    seal={'protocol_sha256':sha(path),'script_sha256':sha(__file__),'initial_state':initial_state,'final_state':bitgen.state,
          'feasible_subset_count':str(count),'repetitions':len(indices),'duplicate_subset_repetitions':len(indices)-len(set(signatures)),
          'observed_component_inclusion_counts':{name:int(inclusion[name]) for name in names},'all_components':names,
          'all_indices':indices,'sealed_before_loading_natural_latent_arrays':True}
    dump(out/'INDEX_SEAL.json',seal)
    records=[]
    for method in p['methods']:
        with np.load(method['train_posterior'],allow_pickle=False) as z:
            train=z['values'].astype(np.float64)
            assert np.array_equal(z['row_ids'],np.arange(279)) and np.array_equal(z['labels'],labels)
        assert train.shape==(279,2,16,48) and np.isfinite(train).all()
        generated={}
        for terminal in ('FREE','AMD'):
            with np.load(method['generated_posterior'][terminal],allow_pickle=False) as z:generated[terminal]=z['values'].astype(np.float64)
            assert generated[terminal].shape==(16,2,16,48) and np.isfinite(generated[terminal]).all()
        for item in indices:
            for terminal in ('FREE','AMD'):
                ref=train[item['reference_row_ids'][terminal]];held=train[item['holdout_row_ids'][terminal]]
                mu,logs=ref[:,0],ref[:,1];sigma=np.exp(logs)
                rs={n:reference_stats(x) for n,x in [('mu',mu),('log_sigma',logs),('sigma',sigma)]}
                dep=dependency(mu,logs);q={'mean':float(mu.mean()),'RMS':float(np.sqrt(np.mean(mu*mu+sigma*sigma)))}
                gm=metric_bundle(generated[terminal],rs,dep,q);hm=metric_bundle(held,rs,dep,q)
                delta={k:(None if gm[k] is None or hm[k] is None else gm[k]-hm[k]) for k in gm if not isinstance(gm[k],list)}
                records.append({'method':method['name'],'terminal':terminal,'repetition':item['repetition'],
                    'generated_peptides':16,'holdout_peptides':16,'reference_peptides':len(ref),
                    'generated_vs_reference':gm,'holdout_vs_reference':hm,'paired_delta_generated_minus_holdout':delta})
    summaries={}
    for method in p['methods']:
        summaries[method['name']]={}
        for terminal in ('FREE','AMD'):
            rec=[x for x in records if x['method']==method['name'] and x['terminal']==terminal];s={}
            for metric in rec[0]['paired_delta_generated_minus_holdout']:
                g=[x['generated_vs_reference'][metric] for x in rec];h=[x['holdout_vs_reference'][metric] for x in rec]
                d=[x['paired_delta_generated_minus_holdout'][metric] for x in rec]
                valid=[v for v in d if v is not None]
                s[metric]={'generated_vs_reference':summary(g),'holdout_vs_reference':summary(h),'paired_delta':summary(d),
                    'fraction_delta_gt_zero_descriptive_only':sum(v>0 for v in valid)/len(valid) if valid else None}
            summaries[method['name']][terminal]=s
    for b in p['input_bindings']:assert sha(b['path'])==b['sha256']
    result={'scope_id':p['scope_id'],'created_utc':datetime.now(timezone.utc).isoformat(),'protocol_sha256':sha(path),'script_sha256':sha(__file__),
        'index_seal_sha256':sha(out/'INDEX_SEAL.json'),'repetitions_per_method_terminal':256,'records':records,'summaries':summaries,
        'covariance_rank_note':'16 peptide rows in1536(mu,logs) dimensions have centered sample covariance rank<=15. No covariance, whitening, W2, shrinkage, projection or coordinate-independence inference is performed.',
        'posthoc_exploratory':True,'dev_latent':'MISSING_NOT_FORWARDED','dev_sequence_outside_training_interval_reference':'MISSING_NOT_EVALUABLE_FROM_MEAN_RECONSTRUCTION',
        'reference_is_not_null_distribution_or_pvalue':True,'replicates_do_not_add_independent_biological_samples':True,
        'group_rule_boundary':'Frozen70%-alignment-identity connected components, not proved biological families; residual relatedness remains.',
        'actions':{'model_calls':0,'natural_forward':0,'new_latent_draws':0,'PCG64_component_subsets':256,'test_reads':0,'live_scalar_reads':0},
        'all_input_hashes_unchanged':True,'analysis_wall_seconds':time.perf_counter()-start,'original_CE_E_L_G':'NOT_EVALUABLE_INTERRUPTION','PROMOTE':'HOLD'}
    dump(out/'SAME_N_READOUT.json',result)
    dump(out/'COMPUTE_MANIFEST.json',{'files':[{'path':str(q),'sha256':sha(q)} for q in [out/'INDEX_SEAL.json',out/'SAME_N_READOUT.json']],
        'protocol_sha256':sha(path),'script_sha256':sha(__file__),'analysis_wall_seconds':result['analysis_wall_seconds'],'model_calls':0})
    print(json.dumps({'output':str(out),'readout_sha256':sha(out/'SAME_N_READOUT.json'),'analysis_wall_seconds':result['analysis_wall_seconds']}))

if __name__=='__main__':main()
