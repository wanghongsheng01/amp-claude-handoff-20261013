#!/usr/bin/env python3
"""Independent saved-array audit; NumPy/stdlib only, no RNG/model/new outputs."""
from pathlib import Path
import ast,csv,datetime,hashlib,json,math,os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
import numpy as np
W=Path('/Users/loctek/Documents/Codex/2026-10-11/task-4');H=W/'amp_h5';O=H/'review/independent';D=H/'feasibility/latent_match_diagnostic_v1';R=H/'review/latent_route_static_v1'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def close(x,y):return x is None and y is None if x is None or y is None else math.isclose(float(x),float(y),rel_tol=1e-11,abs_tol=1e-12)
def correlation(x,y):
 x=np.asarray(x).ravel();y=np.asarray(y).ravel()
 return None if np.std(x)==0 or np.std(y)==0 else float(np.corrcoef(x,y)[0,1])
def main():
 out=O/'LATENT_MATCH_DIAGNOSTIC_INDEPENDENT_REVIEW_v1.json'
 if out.exists():raise FileExistsError('Never overwrite existing review')
 protocol_path=H/'protocol/LATENT_MATCH_DIAGNOSTIC_v1.json';p=read(protocol_path);j=read(D/'LATENT_MATCH_READOUT.json');delivery=read(D/'DELIVERY_MANIFEST.json');route=read(R/'LATENT_ROUTE_STATIC_AUDIT.json');checks=[]
 def ck(n,x,d=None):checks.append({'name':n,'pass':bool(x),**({'detail':d} if d is not None else {})})
 ck('exact_author_delivery_hash',sha(D/'DELIVERY_MANIFEST.json')=='9507ad6f6ad04f4bb94fa4f55f9a85b91984a73327c5aa7268474cb92b11459c')
 ck('delivery_all_members_hash_bytes_unique',len({b['path'] for b in delivery['files']})==len(delivery['files']) and all(sha(b['path'])==b['sha256'] and Path(b['path']).stat().st_size==b['bytes'] for b in delivery['files']))
 ck('delivery_external_binding_hashes',all(sha(b['path'])==b['sha256'] for b in delivery['external_bindings']))
 ck('all24_input_bindings_unchanged',len(p['input_bindings'])==24 and len({b['path'] for b in p['input_bindings']})==24 and all(sha(b['path'])==b['sha256'] for b in p['input_bindings']))
 ck('protocol_script_and_readout_binding',j['protocol_sha256']==sha(protocol_path) and j['script_sha256']==sha(H/'implementation/latent_match_saved_arrays_v1.py') and j['input_hashes_unchanged_after_analysis'])
 man=read(R/'MANIFEST.json');src=read(R/'SOURCE_BINDINGS.json')['files']
 ck('route_manifest_and24_source_bindings',all(sha(b['path'])==b['sha256'] and Path(b['path']).stat().st_size==b['bytes'] for b in man['files']) and len(src)==24 and all(sha(b['path'])==b['sha256'] for b in src))
 excerpts=[]
 for finding in route['findings']:
  for e in finding['evidence']:
   source=Path(e['path']).read_text().splitlines();wanted='\n'.join(f'{i}: {source[i-1]}' for i in range(e['first_line'],e['last_line']+1))
   excerpts.append(sha(e['path'])==e['sha256'] and wanted==e['excerpt'])
 ck('all_static_route_line_excerpts_exact',all(excerpts),{'excerpts_checked':len(excerpts)})
 ck('posthoc_noncausal_scope_explicit',read(H/'protocol/LATENT_MATCH_DIAGNOSTIC_v1_POSTHOC_ADDENDUM.json')['status']=='POSTHOC_EXPLORATORY_ONLY' and route['causal_claims_authorized'] is False and route['absence_of_found_bug_is_not_proof_of_absence'] is True and delivery['PROMOTE']=='HOLD')
 script=ast.parse((H/'implementation/latent_match_saved_arrays_v1.py').read_text());calls=[ast.unparse(n.func) for n in ast.walk(script) if isinstance(n,ast.Call)];imports=[n.module for n in ast.walk(script) if isinstance(n,ast.ImportFrom)]+[a.name for n in ast.walk(script) if isinstance(n,ast.Import) for a in n.names]
 ck('no_torch_checkpoint_or_RNG_in_compute_source',not any('torch' in (n or '') or 'random' in (n or '') for n in imports+calls) and not any('pickle' in n for n in calls))
 facts={};ncheck=0
 for m in p['methods']:
  name=m['method'];tr=Path(m['training_root']);gr=Path(m['generation_root']);item=j['methods'][name]
  with np.load(tr/'train_posterior_parameters.npz',allow_pickle=False) as a:v=a['values'].astype('float64');labels=a['labels'].copy();ids=a['row_ids'].copy()
  ck(name+'_train_schema279',v.shape==(279,2,16,48) and np.array_equal(ids,np.arange(279)) and np.isfinite(v).all())
  facts[name]={}
  for label,t,n in ((0,'FREE',126),(1,'AMD',153)):
   mask=labels==label;q=item['terminals'][t]
   with np.load(gr/f'generated_posterior_{t}.npz',allow_pickle=False) as a:gv=a['values'].astype('float64')
   with np.load(gr/f'generated_{t}.npz',allow_pickle=False) as a:z=a['z'].astype('float64');keys=list(a.files);saved=a['posterior_parameters'].astype('float64')
   ck(name+'_'+t+'_full_denominators_shapes',int(mask.sum())==q['train_peptides']==n and q['generated_peptides']==16 and q['train_row_ids']==ids[mask].tolist() and q['generation_request_indices']==list(range(16)) and gv.shape==(16,2,16,48) and z.shape==(16,16,48) and np.array_equal(saved,gv) and 'epsilon' not in keys)
   tm,tl=v[mask,0],v[mask,1];gm,gl=gv[:,0],gv[:,1];ts=np.exp(tl);gs=np.exp(gl)
   for field,ta,ga in (('mu',tm,gm),('log_sigma',tl,gl),('sigma',ts,gs)):
    data=q['fields'][field];good=[]
    for which,a,nn in (('train',ta,n),('generated',ga,16)):
     d=data[which];r=d['pooled_coordinate_summary_descriptive_only']
     good += [d['peptide_count']==nn,d['coordinate_count']==nn*768,r['count']==r['finite']==nn*768,len(d['per_peptide'])==nn]
     expected={'mean':a.mean(),'std_ddof0':a.std(ddof=0),'RMS':np.linalg.norm(a.ravel())/math.sqrt(a.size),'min':a.min(),'max':a.max()}
     good += [close(r[k],val) for k,val in expected.items()]
     good += [close(val,np.percentile(a,float(k)*100)) for k,val in r['quantiles'].items()]
     good += [np.allclose(d['structure']['channel']['mean'],a.mean(axis=2).mean(axis=0),rtol=1e-11,atol=1e-12),np.allclose(d['structure']['position']['mean'],a.mean(axis=1).mean(axis=0),rtol=1e-11,atol=1e-12),np.allclose(d['structure']['mean_map_16x48'],a.mean(axis=0),rtol=1e-11,atol=1e-12)]
     good += [close(d['structure']['adjacent_position_correlation'],correlation(a[:,:,:-1],a[:,:,1:])),close(d['structure']['adjacent_channel_correlation'],correlation(a[:,:-1,:],a[:,1:,:]))]
    cmp=data['comparison'];good += [close(cmp['pooled_std_ratio'],ga.std()/ta.std()),close(cmp['pooled_mean_difference'],ga.mean()-ta.mean())]
    for mode in ('pooled','matched_coordinate'):
     axis=None if mode=='pooled' else 0
     for suffix,ps in (('minmax',[0,100]),('q01_q99',[1,99])):
      lo,hi=np.percentile(ta,ps,axis=axis);bad=np.logical_or(ga<lo,ga>hi);r=cmp['tail_coverage'][mode+'_'+suffix]
      good += [r['coordinate_denominator']==12288,r['peptide_denominator']==16,r['outside_coordinates']==int(np.count_nonzero(bad)),close(r['fraction'],np.count_nonzero(bad)/12288),np.array_equal(np.asarray(r['per_peptide_fraction']),bad.reshape(16,-1).sum(axis=1)/768)]
    for axisname,axis in (('channel',(0,2)),('position',(0,1)),('coordinate_map',0)):
     x=ta.mean(axis=axis);y=ga.mean(axis=axis);r=cmp[axisname+'_mean_profile'];rmse=np.linalg.norm((x-y).ravel())/math.sqrt(x.size);scale=np.linalg.norm(ta.ravel())/math.sqrt(ta.size)
     good += [close(r['correlation'],correlation(x,y)),close(r['RMSE'],rmse),close(r['RMSE_over_train_pooled_RMS'],rmse/scale)]
    ck(name+'_'+t+'_'+field+'_independent_marginal_tail_structure_recompute',all(good),{'scalar_and_array_comparisons':len(good)});ncheck+=len(good)
   good=[]
   for which,mu,logs in (('train',tm,tl),('generated',gm,gl)):
    d=q[which+'_mu_logsigma_dependency'];good += [close(d['pooled_aligned_coordinate_pearson'],correlation(mu,logs)),close(d['within_peptide_centered_pearson'],correlation(mu-mu.mean(axis=(1,2),keepdims=True),logs-logs.mean(axis=(1,2),keepdims=True))),close(d['channel_mean_pearson'],correlation(mu.mean(axis=(0,2)),logs.mean(axis=(0,2)))),close(d['position_mean_pearson'],correlation(mu.mean(axis=(0,1)),logs.mean(axis=(0,1)))),all(close(x,correlation(a,b)) for x,a,b in zip(d['per_peptide_pearson'],mu,logs))]
   ck(name+'_'+t+'_dependence_definitions_recomputed',all(good));second=(tm*tm+ts*ts).mean();ref=q['analytic_train_qz']
   ck(name+'_'+t+'_analytic_q_and_saved_z_reference',close(ref['mean'],tm.mean()) and close(ref['RMS'],math.sqrt(second)) and close(ref['std'],math.sqrt(second-tm.mean()**2)) and ref['quantiles']=='UNAVAILABLE_WITHOUT_DRAW_OR_CDF_COMPUTATION' and close(q['z_vs_train_qz']['RMS_ratio'],math.sqrt(np.mean(z*z)/second)) and q['derived_epsilon_residual']['independence_claim'] is False)
   facts[name][t]={'records':n,'requests':16,'matched_coordinate_mu_outside_count':q['fields']['mu']['comparison']['tail_coverage']['matched_coordinate_minmax']['outside_coordinates'],'matched_coordinate_logsigma_outside_count':q['fields']['log_sigma']['comparison']['tail_coverage']['matched_coordinate_minmax']['outside_coordinates'],'coordinate_denominator':12288,'logsigma_std_ratio':q['fields']['log_sigma']['comparison']['pooled_std_ratio'],'z_RMS_ratio':q['z_vs_train_qz']['RMS_ratio']}
  for split,n in (('train',279),('dev',41)):
   rows=list(csv.DictReader((gr/f'{split}_reconstruction.csv').open(newline='')));rep=item['reconstruction_reference'][split];good=[len(rows)==n,[int(r['row_id']) for r in rows]==list(range(n)),rep['sample_reconstruction']=='NOT_EXPORTED']
   for group in ('ALL','FREE','AMD'):
    rs=[r for r in rows if group=='ALL' or r['terminal']==group];correct=[];den=[];le=[];ex=[]
    for r in rs:
     t,s=r['truth_sequence'],r['prediction_sequence'];correct.append(sum(a==b for a,b in zip(t,s)));den.append(len(t));le.append(abs(len(t)-len(s)));ex.append(t==s)
    r=rep['groups'][group];good += [r['rows']==len(rs),r['real_token_correct']==sum(correct),r['real_token_denominator']==sum(den),close(r['real_token_micro_accuracy'],sum(correct)/sum(den)),close(r['real_token_macro_accuracy'],sum(c/d for c,d in zip(correct,den))/len(rs)),close(r['length_MAE'],sum(le)/len(rs)),r['exact_count']==sum(ex)]
   ck(name+'_'+split+'_mean_reconstruction_counts_recomputed',all(good))
 ck('input24_still_unchanged_after_recompute',all(sha(b['path'])==b['sha256'] for b in p['input_bindings']))
 report={'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'status':'PASS_READONLY_EXPLORATORY_DIAGNOSTIC_REVIEW' if all(x['pass'] for x in checks) else 'HOLD','checks':checks,'failed_checks':[x for x in checks if not x['pass']],'recomputed_comparison_count_marginals_tails_profiles':ncheck,'facts':facts,'source_bindings':[{'path':str(x),'sha256':sha(x)} for x in (D/'DELIVERY_MANIFEST.json',D/'LATENT_MATCH_READOUT.json',R/'MANIFEST.json',R/'LATENT_ROUTE_STATIC_AUDIT.json',protocol_path,H/'implementation/latent_match_saved_arrays_v1.py')],'audit_script_sha256':sha(__file__),'interpretation':['Posthoc exploratory evidence only; prelocking this computation does not make original outcomes confirmatory.','Matched-coordinate ranges use126/153 training records at each coordinate; pooled ranges use all coordinates. Neither is legal latent support or a failure-rate denominator.','Correlations flatten coordinates or average channels/positions as explicitly named, not cross-peptide independence or causal covariance estimates.','Macro real-token accuracy here means equal-peptide average, not supported-AA macro recall used by the frozen main evaluator. Keep names/definitions separate.','Existing train/dev exports decode onlymu. No saved sampled reconstruction exists; raw/visible token columns are two representations of one decode.','No active-route logvariance/inverse-standardization error found; no proof that all code is correct. Native flow prior lives in zp and requires legal g inverse; direct z normal control is not native prior.'], 'future_control_design_status':'THEORETICALLY_SCOPABLE_BUT_NOT_EXECUTION_AUTHORIZED','future_control_requirements':['32 paired slots x2 branches=64 total decoder inputs, CE only and4 fixedbatch16 calls, original J; all outputs/failures retained.','Final executable version must use root-requested global CPU torch RNG under fork_rng, reconstruct exact original call sequence and bitwise savedz before anydecoder; draft localGenerator wording is historical, not final contract.','Save exactepsilon/index tapes and hashes before forward; B selectorPCG64 seeds523/524 on sorted terminal rowIDs; preserve duplicates.','A standard-normal z control has no terminal conditioning; group labels only. B is empirical train posterior reference and not novel generation.','Wait scalar true terminal and independent review, then exact new scope/plan/receipt; this report authorizes no forward/RNG/model work.'], 'reviewer_actions':{'numpy_stdlib_only':True,'model_imports':0,'checkpoint_loads':0,'forwards':0,'training':0,'RNG_calls':0,'new_generation':0,'test_reads':0,'live_scalar_reads':0},'original_CE_E_L_G':'NOT_EVALUABLE_INTERRUPTION','PROMOTE':'HOLD'}
 with out.open('x') as f:json.dump(report,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
 print(json.dumps({'status':report['status'],'checks':len(checks),'failed_checks':report['failed_checks'],'comparison_count':ncheck,'path':str(out),'sha256':sha(out)},ensure_ascii=False))
if __name__=='__main__':main()
