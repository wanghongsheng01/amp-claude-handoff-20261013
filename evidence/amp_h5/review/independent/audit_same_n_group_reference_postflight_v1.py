import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):os.environ[k]='1'
import hashlib,json,math,time
from collections import Counter,defaultdict
from functools import lru_cache
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
R=Path('/Users/loctek/Documents/Codex/2026-10-11/task-4/amp_h5');O=R/'review/independent';D=R/'feasibility/latent_same_n_group_reference_v1'
OUT=O/'LATENT_SAME_N_GROUP_REFERENCE_POSTFLIGHT_v1.json';assert not OUT.exists();start=time.perf_counter()
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();read=lambda p:json.loads(Path(p).read_text())
p=read(R/'protocol/LATENT_SAME_N_GROUP_REFERENCE_v1.json');r=read(D/'SAME_N_READOUT.json');seal=read(D/'INDEX_SEAL.json');manifest=read(D/'DELIVERY_MANIFEST.json')
checks=[];errors=[];numeric=0
def ck(k,v):checks.append({'name':k,'pass':bool(v)})
def equal(a,b,path):
 global numeric
 if isinstance(a,dict):
  if set(a)!=set(b):errors.append(path+':keys');return
  for k in a:equal(a[k],b[k],path+'/'+k)
 elif a is None or b is None:
  if a is not None or b is not None:errors.append(path+':null')
 elif isinstance(a,(list,tuple)):
  numeric+=len(a)
  if not np.allclose(a,b,atol=3e-12,rtol=3e-11):errors.append(path+':vector')
 elif isinstance(a,(float,int,np.floating,np.integer)):
  numeric+=1
  if not math.isclose(float(a),float(b),abs_tol=3e-12,rel_tol=3e-11):errors.append(path+':number')
 elif a!=b:errors.append(path+':value')
ck('runtime_numpy1_26_4',np.__version__=='1.26.4')
ck('exact_frozen_numerical_and_index_artifacts',sha(D/'SAME_N_READOUT.json')=='bded28a9a808f367dcfba11bc63d6bf53246e633b3f7af294012441a342bde22' and sha(D/'INDEX_SEAL.json')=='9882ca817a5c813dfc89d756080a15e34522dbe61b8fa4ed28e41661687b77c1')
ck('manifest_members_hashes_bytes_unique',len({x['path'] for x in manifest['files']})==6 and all(sha(x['path'])==x['sha256'] and Path(x['path']).stat().st_size==x['bytes'] and Path(x['path']).parent==D for x in manifest['files']))
allbind=p['input_bindings']+manifest['files']+[{'path':str(D/'DELIVERY_MANIFEST.json'),'sha256':sha(D/'DELIVERY_MANIFEST.json')}]
ck('all12_inputs_before',len(p['input_bindings'])==12 and all(sha(b['path'])==b['sha256'] for b in p['input_bindings']))
ck('protocol_script_preflight_bindings',r['protocol_sha256']==seal['protocol_sha256']==manifest['protocol_sha256']==sha(R/'protocol/LATENT_SAME_N_GROUP_REFERENCE_v1.json') and r['script_sha256']==seal['script_sha256']==manifest['script_sha256']==sha(p['implementation_path']) and sha(manifest['preflight_binding']['path'])==manifest['preflight_binding']['sha256'])
rows=read(p['row_binding_path'])['splits']['train']['rows'];labels=np.array([x['class'] for x in rows]);groups=defaultdict(list)
for i,x in enumerate(rows):assert x['model_row_id']==i;groups[x['component_id']].append(i)
names=sorted(groups);weights=[(sum(labels[groups[n]]==0),sum(labels[groups[n]]==1)) for n in names];weights=[tuple(map(int,x)) for x in weights]
@lru_cache(None)
def ways(i,f,a):
 if f<0 or a<0:return 0
 if i==len(names):return int(f==0 and a==0)
 wf,wa=weights[i];return ways(i+1,f,a)+ways(i+1,f-wf,a-wa)
count=ways(0,16,16);ck('independent_recursive_DP_count',str(count)==seal['feasible_subset_count']==p['sampler']['feasible_component_subset_count'])
bit=np.random.PCG64(20261011);ck('initial_RNG_state',bit.state==seal['initial_state'])
def integer_draw(n):
 if n==1:return 0
 k=(n-1).bit_length();mask=(1<<k)-1
 while True:
  raw=bit.random_raw((k+63)//64);u=sum(int(v)<<(64*j) for j,v in enumerate(raw[::-1]))&mask
  if u<n:return u
selected_check=[];counts=Counter();signatures=[]
for index,item in enumerate(seal['all_indices']):
 assert index==item['repetition'] and bit.state==item['RNG_state_before'];f=a=16;chosen=[]
 for i,(wf,wa) in enumerate(weights):
  if (f,a)==(0,0):break
  yes=ways(i+1,f-wf,a-wa);no=ways(i+1,f,a)
  if yes and (not no or integer_draw(yes+no)<yes):chosen.append(names[i]);f-=wf;a-=wa
 selected_check.append(chosen==item['components'] and (f,a)==(0,0) and bit.state==item['RNG_state_after'])
 hs=set(j for n in chosen for j in groups[n]);rs=set(range(279))-hs
 for label,t in enumerate(('FREE','AMD')):
  assert item['holdout_row_ids'][t]==sorted(j for j in hs if labels[j]==label)
  assert item['reference_row_ids'][t]==sorted(j for j in rs if labels[j]==label)
  assert len(item['holdout_row_ids'][t])==16 and len(item['reference_row_ids'][t])==[110,137][label]
 assert not {rows[i]['component_id'] for i in hs}&{rows[i]['component_id'] for i in rs}
 assert item['selection_probability_exact']=='1/'+str(count)
 counts.update(chosen);signatures.append(tuple(chosen))
ck('all256_index_and_RNG_replays_exact',len(selected_check)==256 and all(selected_check) and bit.state==seal['final_state'])
ck('complete_inclusion_and_duplicate_accounting',{n:counts[n] for n in names}==seal['observed_component_inclusion_counts'] and len(signatures)-len(set(signatures))==seal['duplicate_subset_repetitions']==0)

def pc(a,b):
 a=np.asarray(a).ravel();b=np.asarray(b).ravel()
 if np.var(a)==0 or np.var(b)==0:return None
 return float(np.corrcoef(a,b)[0,1])
def dep(a):
 m,l=a[:,0],a[:,1]
 return {'pooled_aligned':pc(m,l),'within_peptide_centered':pc(m-m.mean((1,2),keepdims=True),l-l.mean((1,2),keepdims=True)),'position_mean':pc(m.mean((0,1)),l.mean((0,1))),'channel_mean':pc(m.mean((0,2)),l.mean((0,2)))}
def compare(x,ref):
 z={}
 for name,y,b in [('mu',x[:,0],ref[:,0]),('log_sigma',x[:,1],ref[:,1]),('sigma',np.exp(x[:,1]),np.exp(ref[:,1]))]:
  prefix=name+'/';ym,ys,yr=float(y.mean()),float(y.std()),float(np.sqrt(np.square(y).mean()));bm,bs,br=float(b.mean()),float(b.std()),float(np.sqrt(np.square(b).mean()))
  z.update({prefix+'signed_mean_shift':ym-bm,prefix+'absolute_mean_shift_in_ref_std':abs(ym-bm)/bs,prefix+'std_ratio':ys/bs,prefix+'absolute_log_std_ratio':abs(math.log(ys/bs)),prefix+'RMS_ratio':yr/br,prefix+'absolute_log_RMS_ratio':abs(math.log(yr/br))})
  for name2,axes in [('channel',(0,2)),('position',(0,1)),('map',0)]:
   yp,bp=y.mean(axes),b.mean(axes);c=pc(yp,bp)
   z[prefix+name2+'_mean_correlation']=c;z[prefix+name2+'_mean_correlation_distance']=None if c is None else 1-c;z[prefix+name2+'_mean_RMSE_in_ref_RMS']=float(np.sqrt(np.square(yp-bp).mean())/br)
  q=np.percentile(b,[1,99],axis=0)
  for n,lo,hi in [('pooled_minmax',b.min(),b.max()),('coordinate_minmax',b.min(0),b.max(0)),('coordinate_q01_q99',q[0],q[1])]:
   flags=np.logical_or(y<lo,y>hi);z[prefix+'outside_'+n]=np.count_nonzero(flags)/flags.size;z[prefix+'outside_'+n+'_per_peptide']=(np.count_nonzero(flags,axis=(1,2))/(16*48)).tolist()
 dx,dr=dep(x),dep(ref)
 for k in dx:z['mu_logsigma/'+k+'_correlation']=dx[k];z['mu_logsigma/'+k+'_absolute_difference_from_reference']=None if dx[k] is None or dr[k] is None else abs(dx[k]-dr[k])
 xm,rm=float(x[:,0].mean()),float(ref[:,0].mean());xr=float(np.sqrt(np.mean(x[:,0]**2+np.exp(x[:,1])**2)));rr=float(np.sqrt(np.mean(ref[:,0]**2+np.exp(ref[:,1])**2)))
 z.update({'analytic_q/mean_shift':xm-rm,'analytic_q/absolute_mean_shift_in_ref_RMS':abs(xm-rm)/rr,'analytic_q/RMS_ratio':xr/rr,'analytic_q/absolute_log_RMS_ratio':abs(math.log(xr/rr))})
 return z
index={(x['method'],x['terminal'],x['repetition']):x for x in r['records']};ck('exact1024_unique_records',len(index)==len(r['records'])==1024)
for method in p['methods']:
 with np.load(method['train_posterior'],allow_pickle=False) as a:
  train=a['values'].astype('float64');assert np.array_equal(a['labels'],labels) and np.array_equal(a['row_ids'],np.arange(279))
 generated={}
 for t in ('FREE','AMD'):
  with np.load(method['generated_posterior'][t],allow_pickle=False) as a:generated[t]=a['values'].astype('float64')
 for item in seal['all_indices']:
  for t in ('FREE','AMD'):
   actual=index[(method['name'],t,item['repetition'])];ref=train[item['reference_row_ids'][t]];held=train[item['holdout_row_ids'][t]]
   assert actual['generated_peptides']==actual['holdout_peptides']==16 and actual['reference_peptides']==len(ref)
   gm=compare(generated[t],ref);hm=compare(held,ref);d={k:None if gm[k] is None or hm[k] is None else gm[k]-hm[k] for k in gm if not isinstance(gm[k],list)}
   equal(gm,actual['generated_vs_reference'],'recordG');equal(hm,actual['holdout_vs_reference'],'recordH');equal(d,actual['paired_delta_generated_minus_holdout'],'recordD')
ck('all1024_numeric_records_independently_recomputed',not errors)

def stat(v):
 defined=np.array([x for x in v if x is not None],dtype=float);s={'total_repetitions':len(v),'defined':len(defined),'undefined':len(v)-len(defined)}
 if len(defined):s.update(mean=float(defined.mean()),std_ddof0=float(defined.std()),min=float(defined.min()),max=float(defined.max()),quantiles=dict(zip(['.025','.25','.5','.75','.975'],map(float,np.percentile(defined,[2.5,25,50,75,97.5])))))
 return s
for method,ts in r['summaries'].items():
 for t,ms in ts.items():
  recs=[index[(method,t,i)] for i in range(256)]
  for metric,saved in ms.items():
   g=[v['generated_vs_reference'][metric] for v in recs];h=[v['holdout_vs_reference'][metric] for v in recs];d=[v['paired_delta_generated_minus_holdout'][metric] for v in recs];valid=[x for x in d if x is not None]
   equal({'generated_vs_reference':stat(g),'holdout_vs_reference':stat(h),'paired_delta':stat(d),'fraction_delta_gt_zero_descriptive_only':sum(x>0 for x in valid)/len(valid) if valid else None},saved,'summary')
ck('all_summaries_independently_recomputed',not errors)
ck('original_NE_dev_missing_no_inferential_claim',r['original_CE_E_L_G']=='NOT_EVALUABLE_INTERRUPTION' and r['PROMOTE']=='HOLD' and r['dev_latent']=='MISSING_NOT_FORWARDED' and r['reference_is_not_null_distribution_or_pvalue'] is True)
ck('all_input_and_output_bindings_stable',all(sha(b['path'])==b['sha256'] for b in allbind))
selected_facts={}
for method,ts in r['summaries'].items():
 selected_facts[method]={t:{k:ms[k]['paired_delta']['quantiles'] for k in ['mu/outside_coordinate_minmax','log_sigma/outside_coordinate_minmax','log_sigma/map_mean_correlation_distance','mu_logsigma/position_mean_absolute_difference_from_reference']} for t,ms in ts.items()}
report={'created_utc':datetime.now(timezone.utc).isoformat(),'status':'PASS_SAVED_ARRAY_SAME_N_DESCRIPTIVE_ANALYSIS' if all(c['pass'] for c in checks) and not errors else 'HOLD','scope_id':p['scope_id'],'checks':checks,'failed_checks':[c for c in checks if not c['pass']],'numeric_comparisons':numeric,'numerical_disagreements':errors[:50],'facts':{'repetitions':256,'records':1024,'components':224,'exact_subset_count':str(count),'component_inclusion_range':[min(counts[n] for n in names),max(counts[n] for n in names)],'components_observed_zero_inclusion':sum(counts[n]==0 for n in names),'duplicate_subsets':0,'reference_counts':{'FREE':110,'AMD':137},'generated_and_holdout_counts_each':16,'selected_paired_delta_empirical_quantiles':selected_facts},'source_bindings':allbind,'reviewer_actions':{'model_calls':0,'new_latent_draws':0,'new_subsets_selected':0,'prelocked_subset_RNG_replayed':256,'saved_arrays_recomputed':True,'test37_reads':0,'live_scalar_reads':0},'scope_limitations':['Posthoc descriptive finite-training-pool reference, not confidence intervals, pvalues, model-unseen validation or independent biological experiments.','Joint whole-component holdout prevents split of frozen similarity components only; uniform feasible subsets cause unequal inclusion probabilities. Eight groups were not selected by this256-replay stream but none was prefiltered.','Both operands use16 rows and identical110/137 reference; this addresses that count mismatch only, not all dependence, model bias or causal identification.','Statistics are low-order marginal/mean-profile/dependency summaries, not a verified estimate of full1536-dimensional covariance.','Dev latent remains missing; no forward was added. Existing tail proportions cannot prove DDPM implementation failure.','No new model work or source-substitution control authorized; original CE E/L/G remainsNE and PROMOTE HOLD.'],'original_CE_E_L_G':'NOT_EVALUABLE_INTERRUPTION','PROMOTE':'HOLD','audit_seconds':time.perf_counter()-start}
with OUT.open('x') as f:json.dump(report,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
print(json.dumps({'path':str(OUT),'sha256':sha(OUT),'status':report['status'],'checks':len(checks),'numeric_comparisons':numeric,'errors':errors[:10],'seconds':report['audit_seconds']}))
