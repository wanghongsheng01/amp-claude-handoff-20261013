"""Read-only NumPy/stdlib diagnostic: no model imports, checkpoints, RNG or draws."""
import os
for _key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_key] = '1'
import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import numpy as np

Q = [0,.01,.05,.25,.5,.75,.95,.99,1]

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def scalar_summary(a):
    a = np.asarray(a, dtype=np.float64).ravel()
    assert a.size and np.isfinite(a).all()
    return {'count':int(a.size),'finite':int(np.isfinite(a).sum()),'mean':float(a.mean()),
            'std_ddof0':float(a.std()),'RMS':float(np.sqrt(np.mean(a*a))),
            'min':float(a.min()),'max':float(a.max()),
            'quantiles':dict(zip(map(str,Q),map(float,np.quantile(a,Q))))}

def corr(x,y):
    x=np.asarray(x,dtype=np.float64).ravel();y=np.asarray(y,dtype=np.float64).ravel()
    x=x-x.mean();y=y-y.mean()
    den=float(np.sqrt(np.sum(x*x)*np.sum(y*y)))
    return None if den==0 else float(np.sum(x*y)/den)

def comparison(x,y,scale):
    x=np.asarray(x);y=np.asarray(y)
    e=float(np.sqrt(np.mean((x-y)**2)))
    return {'correlation':corr(x,y),'RMSE':e,'RMSE_over_train_pooled_RMS':None if scale==0 else e/scale}

def structure(a):
    result={}
    for name,axes in [('channel',(0,2)),('position',(0,1))]:
        result[name]={'mean':a.mean(axis=axes).tolist(),'std_ddof0':a.std(axis=axes).tolist(),
                      'RMS':np.sqrt(np.mean(a*a,axis=axes)).tolist()}
    result['mean_map_16x48']=a.mean(axis=0).tolist()
    result['std_map_16x48']=a.std(axis=0).tolist()
    result['adjacent_position_correlation']=corr(a[:,:,:-1],a[:,:,1:])
    result['adjacent_channel_correlation']=corr(a[:,:-1,:],a[:,1:,:])
    return result

def describe(a,ids):
    a=np.asarray(a,dtype=np.float64)
    assert a.shape==(len(ids),16,48) and np.isfinite(a).all()
    rows=[]
    for row_id,x in zip(ids,a):
        rows.append({'row_id':int(row_id),'mean':float(x.mean()),'std_ddof0':float(x.std()),
                     'RMS':float(np.sqrt(np.mean(x*x))),'min':float(x.min()),'max':float(x.max())})
    return {'peptide_count':len(ids),'coordinate_count':int(a.size),
            'pooled_coordinate_summary_descriptive_only':scalar_summary(a),
            'per_peptide':rows,'per_peptide_summary':{k:scalar_summary([r[k] for r in rows]) for k in ('mean','std_ddof0','RMS','min','max')},
            'structure':structure(a)}

def mu_log_dependency(mu,logs):
    centered_mu=mu-mu.mean(axis=(1,2),keepdims=True)
    centered_logs=logs-logs.mean(axis=(1,2),keepdims=True)
    return {'pooled_aligned_coordinate_pearson':corr(mu,logs),
            'within_peptide_centered_pearson':corr(centered_mu,centered_logs),
            'per_peptide_pearson':[corr(x,y) for x,y in zip(mu,logs)],
            'channel_mean_pearson':corr(mu.mean(axis=(0,2)),logs.mean(axis=(0,2))),
            'position_mean_pearson':corr(mu.mean(axis=(0,1)),logs.mean(axis=(0,1)))}

def tail_match(train,generated):
    result={}
    for mode in ('pooled','matched_coordinate'):
        axis=None if mode=='pooled' else 0
        for interval,qs in [('minmax',[0,1]),('q01_q99',[.01,.99])]:
            lo,hi=np.quantile(train,qs,axis=axis)
            flag=(generated<lo)|(generated>hi)
            result[mode+'_'+interval]={'outside_coordinates':int(flag.sum()),'coordinate_denominator':int(flag.size),
                'fraction':float(flag.mean()),'per_peptide_fraction':flag.mean(axis=(1,2)).tolist(),
                'peptide_denominator':int(len(generated))}
    return result

def compare_fields(train,generated):
    scale=float(np.sqrt(np.mean(train*train)))
    result={'pooled_mean_difference':float(generated.mean()-train.mean()),
            'pooled_std_ratio':None if train.std()==0 else float(generated.std()/train.std()),
            'pooled_RMS_ratio':None if scale==0 else float(np.sqrt(np.mean(generated*generated))/scale),
            'tail_coverage':tail_match(train,generated)}
    for name,axes in [('channel',(0,2)),('position',(0,1)),('coordinate_map',0)]:
        result[name+'_mean_profile']=comparison(train.mean(axis=axes),generated.mean(axis=axes),scale)
    return result

def analytic_train_qz(mu,sigma,ids):
    second=mu*mu+sigma*sigma
    mean=float(mu.mean());second_moment=float(second.mean())
    result={'definition':'Empirical equal-peptide mixture of diagonal q(z|x); analytic moments, zero draws',
            'peptide_count':len(ids),'mean':mean,'std':float(np.sqrt(second_moment-mean*mean)),
            'RMS':float(np.sqrt(second_moment)),'quantiles':'UNAVAILABLE_WITHOUT_DRAW_OR_CDF_COMPUTATION',
            'per_peptide':[],'structure':{}}
    for i,m,s in zip(ids,mu,second):
        result['per_peptide'].append({'row_id':int(i),'mean':float(m.mean()),'std':float(np.sqrt(s.mean()-m.mean()**2)),
                                     'RMS':float(np.sqrt(s.mean()))})
    for name,axes in [('channel',(0,2)),('position',(0,1)),('coordinate_map',0)]:
        mn=mu.mean(axis=axes);sq=second.mean(axis=axes)
        result['structure'][name]={'mean':mn.tolist(),'std':np.sqrt(sq-mn*mn).tolist(),'RMS':np.sqrt(sq).tolist()}
    return result

def z_reference_comparison(mu,sigma,z):
    sq=mu*mu+sigma*sigma
    scale=float(np.sqrt(sq.mean()))
    result={'reference':'analytic train posterior mixture, not saved empirical train z',
            'mean_difference':float(z.mean()-mu.mean()),'RMS_ratio':float(np.sqrt(np.mean(z*z))/scale)}
    for name,axes in [('channel',(0,2)),('position',(0,1)),('coordinate_map',0)]:
        result[name+'_mean']=comparison(mu.mean(axis=axes),z.mean(axis=axes),scale)
        result[name+'_RMS']=comparison(np.sqrt(sq.mean(axis=axes)),np.sqrt(np.mean(z*z,axis=axes)),scale)
    return result

def reconstruction_reference(path,expected_rows):
    with Path(path).open() as f:rows=list(csv.DictReader(f))
    assert len(rows)==expected_rows and [int(x['row_id']) for x in rows]==list(range(expected_rows))
    assert all(x['status']=='ok' and x['terminal'] in ('FREE','AMD') for x in rows)
    result={'path':str(path),'decoding':'saved deterministic z=mu export; no posterior-sample reconstruction exists here',
            'sample_reconstruction':'NOT_EXPORTED','not_generated_peptide_success':True,'groups':{}}
    for name in ('ALL','FREE','AMD'):
        sub=[r for r in rows if name=='ALL' or r['terminal']==name]
        correct=[];lengths=[];exact=[];denom=[]
        for r in sub:
            t,p=r['truth_sequence'],r['prediction_sequence'];assert len(t)>0
            correct.append(sum(i<len(p) and a==p[i] for i,a in enumerate(t)))
            denom.append(len(t));lengths.append(abs(len(t)-len(p)));exact.append(t==p)
        result['groups'][name]={'rows':len(sub),'real_token_correct':sum(correct),'real_token_denominator':sum(denom),
            'real_token_micro_accuracy':sum(correct)/sum(denom),'real_token_macro_accuracy':float(np.mean(np.array(correct)/denom)),
            'length_MAE':float(np.mean(lengths)),'exact_count':sum(exact),'exact_fraction':sum(exact)/len(sub)}
    return result

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--protocol',required=True);args=ap.parse_args()
    proto_path=Path(args.protocol).resolve();protocol=json.loads(proto_path.read_text())
    assert protocol['scope_id']=='AMP_H5_SAVED_LATENT_MATCH_V1'
    for b in protocol['input_bindings']:assert sha(b['path'])==b['sha256'],b['path']
    out=Path(protocol['outputs']);assert not out.exists();out.mkdir()
    result={'created_utc':datetime.now(timezone.utc).isoformat(),'protocol_path':str(proto_path),'protocol_sha256':sha(proto_path),
            'script_sha256':sha(__file__),'scope':'READ_ONLY_EXPLORATORY_SAVED_LATENT_LOCALIZATION','methods':{},
            'denominator_warning':'Coordinates and sequence positions are dependent; 16 generated peptides per terminal share one trained model and one fixed sampling batch. No inferential tests or independent-coordinate sample claims.',
            'prohibited_actions_performed':{'model_imports':0,'checkpoint_loads':0,'forwards':0,'training':0,'random_draws':0,'live_scalar_reads':0,'test_reads':0}}
    residuals={}
    for method in protocol['methods']:
        name=method['method'];trainroot=Path(method['training_root']);genroot=Path(method['generation_root'])
        with np.load(trainroot/'train_posterior_parameters.npz',allow_pickle=False) as a:
            train=a['values'].astype(np.float64);labels=a['labels'];row_ids=a['row_ids']
        assert train.shape==(279,2,16,48) and np.isfinite(train).all()
        assert np.array_equal(row_ids,np.arange(279)) and set(labels.tolist())=={0,1}
        item={'training_root':str(trainroot),'generation_root':str(genroot),'terminals':{},'reconstruction_reference':{}}
        residuals[name]={}
        for label,term,count in [(0,'FREE',126),(1,'AMD',153)]:
            mask=labels==label;assert int(mask.sum())==count
            with np.load(genroot/f'generated_posterior_{term}.npz',allow_pickle=False) as a:posterior=a['values']
            with np.load(genroot/f'generated_{term}.npz',allow_pickle=False) as a:
                assert np.array_equal(posterior,a['posterior_parameters'])
                z=a['z'].astype(np.float64);generated_keys=list(a.files)
            assert posterior.shape==(16,2,16,48) and z.shape==(16,16,48)
            assert np.isfinite(posterior).all() and np.isfinite(z).all()
            tm,tl=train[mask,0],train[mask,1];ts=np.exp(tl)
            gm,gl=posterior[:,0].astype(np.float64),posterior[:,1].astype(np.float64);gs=np.exp(gl)
            assert np.isfinite(ts).all() and np.isfinite(gs).all() and (ts>0).all() and (gs>0).all()
            eps=(z-gm)/gs;residuals[name][term]=eps
            fields={'mu':(tm,gm),'log_sigma':(tl,gl),'sigma':(ts,gs)}
            t={'train_peptides':count,'generated_peptides':16,'train_row_ids':row_ids[mask].tolist(),
                'generation_request_indices':list(range(16)),'generated_npz_keys':generated_keys,
                'fields':{field:{'train':describe(a,row_ids[mask]),'generated':describe(g,list(range(16))),
                                 'comparison':compare_fields(a,g)} for field,(a,g) in fields.items()},
                'train_mu_logsigma_dependency':mu_log_dependency(tm,tl),'generated_mu_logsigma_dependency':mu_log_dependency(gm,gl),
                'analytic_train_qz':analytic_train_qz(tm,ts,row_ids[mask]),'saved_generated_z':describe(z,list(range(16))),
                'z_vs_train_qz':z_reference_comparison(tm,ts,z),
                'derived_epsilon_residual':{'source':'(saved z-mu)/exp(saved logsigma); deterministic approximate inverse, NOT exact stored epsilon',
                    'summary':describe(eps,list(range(16))),'correlation_with_mu':corr(eps,gm),'correlation_with_logsigma':corr(eps,gl),
                    'independence_claim':False}}
            item['terminals'][term]=t
        for split,n in [('train',279),('dev',41)]:item['reconstruction_reference'][split]=reconstruction_reference(genroot/f'{split}_reconstruction.csv',n)
        result['methods'][name]=item
    result['paired_random_provenance']={'seeds':{'FREE':422,'AMD':423},'generated_batch_shape':[16,2,16,48],
        'final_reparameterization_shape':[16,16,48],'code_and_independent_audit_binding':'See frozen protocol input bindings; full-chain RNG not rerun by this script',
        'exact_epsilon_key_in_generated_npz':False,'derived_residual_is_not_exact_saved_random_draw':True,
        'common_random_numbers_are_not_independent_replicates':True,'CE_MSE_residual_comparison':{}}
    if len(residuals)==2:
        for term in ('FREE','AMD'):
            x=residuals['categorical_ce'][term];y=residuals['raw_onehot_mse'][term]
            result['paired_random_provenance']['CE_MSE_residual_comparison'][term]={'correlation':corr(x,y),'max_absolute_difference':float(np.max(np.abs(x-y))),'RMS_difference':float(np.sqrt(np.mean((x-y)**2))),'exact_epsilon_identity_proven_by_this_inverse':False}
    for b in protocol['input_bindings']:assert sha(b['path'])==b['sha256'],b['path']
    result['input_hashes_unchanged_after_analysis']=True
    with (out/'LATENT_MATCH_READOUT.json').open('x') as f:json.dump(result,f,indent=2,ensure_ascii=False,allow_nan=False);f.write('\n')
    manifest={'created_utc':datetime.now(timezone.utc).isoformat(),'protocol_sha256':sha(proto_path),'script_sha256':sha(__file__),
              'files':[{'path':str(out/'LATENT_MATCH_READOUT.json'),'sha256':sha(out/'LATENT_MATCH_READOUT.json')}]}
    with (out/'COMPUTE_MANIFEST.json').open('x') as f:json.dump(manifest,f,indent=2);f.write('\n')
    print(json.dumps({'output':str(out),'readout_sha256':sha(out/'LATENT_MATCH_READOUT.json'),'input_bindings_verified':len(protocol['input_bindings']),'model_calls':0,'random_draws':0}))

if __name__=='__main__':main()
