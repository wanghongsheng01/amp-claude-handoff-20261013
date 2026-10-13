#!/usr/bin/env python3
"""Frozen 421-feature finite-sample diagnostic; no model, test, latent or new metric."""
import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
import time
START=time.perf_counter()
import argparse
from collections import Counter,defaultdict
import csv
import hashlib
import importlib.util
import itertools
import json
from pathlib import Path
import sys
import traceback
import numpy as np
sys.dont_write_bytecode=True
ROOT=Path('/Users/loctek/Documents/Codex/2026-10-11/task-4/amp_h5')
SCOPE='AMP_H5_NIGHT_DISTRIBUTION_V1'
METHODS=('CE21_DDPM','MSE21_DDPM','scalar_DDPM','terminal_LSTM')
BOOT_METHODS=('CE21_DDPM','terminal_LSTM')
TERMINALS=('FREE','AMD')
BLOCKS={'length':(0,1),'AA':(1,21),'dipeptide':(21,421)}
METRIC_SHA='f65eec16391d98a382d415e7b879f99555a33b33398f8be704661d1c7c027480'
DP_SHA='4f8e18e287b1f81a9103db1140f3ffb2d30ae4f182173fa26b63b156abc1861a'
ACTIVE_OUTPUT=None

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def record(p):
 p=Path(p);return {'path':str(p),'bytes':p.stat().st_size,'sha256':sha(p)}
def dump(p,d):
 with Path(p).open('x') as f:json.dump(d,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def load_module(path,name,expected):
 assert sha(path)==expected
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

def mmd_blocks(x,y):
 """Exactly squared difference of fixed feature means; no fitted kernel."""
 if len(x)==0 or len(y)==0:return {k:None for k in ('total',*BLOCKS)}
 delta=np.mean(x,axis=0)-np.mean(y,axis=0)
 r={k:float(np.sum(delta[a:b]**2)) for k,(a,b) in BLOCKS.items()}
 r['total']=float(np.sum(delta**2));assert abs(r['total']-sum(r[k] for k in BLOCKS))<1e-12
 return r

def summary(values):
 good=[v for v in values if v is not None]
 if not good:return {'repetitions':len(values),'defined':0,'undefined':len(values),'quantiles':None}
 a=np.asarray(good,dtype=np.float64);assert np.isfinite(a).all()
 return {'repetitions':len(values),'defined':len(good),'undefined':len(values)-len(good),'mean':float(a.mean()),'std_ddof0':float(a.std()),'min':float(a.min()),'max':float(a.max()),'quantiles':dict(zip(('q025','q50','q975'),map(float,np.quantile(a,[.025,.5,.975],method='linear'))))}

def empirical_position(observed,reference):
 vals=[v for v in reference if v is not None]
 if observed is None or not vals:return {'defined':False,'reference_n':len(vals),'position':None}
 below=sum(v<observed for v in vals);equal=sum(v==observed for v in vals);n=len(vals)
 return {'defined':True,'reference_n':n,'below':below,'exact_ties':equal,'above':n-below-equal,'midrank_position':(below+.5*equal)/n,'tie_position_interval':[below/n,(below+equal)/n],'interpretation':'empirical position in finite train-resampling reference; NOT pvalue, null or pass threshold'}

def analytic_srs(x,y,n):
 N=len(x);assert 1<=n<=N and N>1
 base=mmd_blocks(x,y);center=x-x.mean(axis=0);factor=(N-n)/(n*(N-1))
 variance={k:float(np.mean(np.sum(center[:,a:b]**2,axis=1))) for k,(a,b) in BLOCKS.items()};variance['total']=float(np.mean(np.sum(center**2,axis=1)))
 return {'N':N,'n':n,'finite_population_factor':factor,'population_mean_squared_deviation_denominator':'N','population_mean_squared_deviation':variance,'full_train_vs_fixed_dev':base,'exact_expected_sample_mean_mmd':{k:base[k]+factor*variance[k] for k in base},'scope':'uniform sequence sampling without replacement only; not whole-component conditional sampling'}

def component_rows(groups,draw):return [i for g in draw for i in groups[int(g)]]
def partition(ids,labels):return {t:[int(i) for i in ids if int(labels[int(i)])==c] for c,t in enumerate(TERMINALS)}
def selected_features(features,indices):
 vals=[features[int(i)] for i in indices if features[int(i)] is not None]
 return np.asarray(vals,dtype=np.float64).reshape((-1,421))

def bootstrap_panel(gen_features,dev_features,dev_labels,gen_indices,dev_ids):
 ids=partition(dev_ids,dev_labels);result={'dev_n':{t:len(v) for t,v in ids.items()},'methods':{},'CE_minus_AR':{}}
 for m in BOOT_METHODS:
  result['methods'][m]={}
  for t in TERMINALS:
   x=selected_features(gen_features[m][t],gen_indices[m][t]);y=dev_features[ids[t]]
   result['methods'][m][t]={'MMD':mmd_blocks(x,y),'generation_requests':16,'generation_valid_n':len(x),'dev_n':len(y)}
 for t in TERMINALS:
  left=result['methods']['CE21_DDPM'][t]['MMD'];right=result['methods']['terminal_LSTM'][t]['MMD']
  result['CE_minus_AR'][t]={k:None if left[k] is None or right[k] is None else left[k]-right[k] for k in left}
 result['CE_minus_AR_equal_terminal_mean']={k:None if any(result['CE_minus_AR'][t][k] is None for t in TERMINALS) else sum(result['CE_minus_AR'][t][k] for t in TERMINALS)/2 for k in ('total',*BLOCKS)}
 return result

def component_schema(rows):
 d=defaultdict(list)
 for i,r in enumerate(rows):d[r['component_id']].append(i)
 names=sorted(d);groups=[d[n] for n in names];weights=[tuple(sum(int(rows[i]['class'])==c for i in g) for c in range(2)) for g in groups]
 return names,groups,weights

def generate_indices(train_rows,dev_rows,dpmod,repetitions,seed):
 labels=np.array([int(r['class']) for r in train_rows]);dlabels=np.array([int(r['class']) for r in dev_rows])
 tn,tg,weights=component_schema(train_rows);dn,dg,dweights=component_schema(dev_rows);dp=dpmod.suffix_counts(weights,(16,16));count=dp[0][16][16];assert count>0
 children=np.random.SeedSequence(seed).spawn(4)
 seqrng=np.random.Generator(np.random.PCG64(children[0]));compbit=np.random.PCG64(children[1]);genrng=np.random.Generator(np.random.PCG64(children[2]));devrng=np.random.Generator(np.random.PCG64(children[3]))
 rngs=[seqrng.bit_generator,compbit,genrng.bit_generator,devrng.bit_generator];before=[r.state for r in rngs]
 pool={t:np.where(labels==c)[0] for c,t in enumerate(TERMINALS)};records=[]
 for rep in range(repetitions):
  seq={t:sorted(map(int,seqrng.choice(pool[t],size=16,replace=False))) for t in TERMINALS}
  comps=dpmod.choose_subset(weights,dp,(16,16),compbit);chosen=partition(component_rows(tg,comps),labels);assert all(len(chosen[t])==16 for t in TERMINALS)
  gen={m:{t:list(map(int,genrng.integers(0,16,size=16))) for t in TERMINALS} for m in BOOT_METHODS}
  devdraw=list(map(int,devrng.integers(0,len(dg),size=len(dg))));devrows=component_rows(dg,devdraw)
  records.append({'repetition':rep,'train_sequence_without_replacement_row_indices':seq,'train_whole_component_indices':comps,'train_component_row_indices':chosen,'generation_slot_indices':gen,'dev_component_indices_with_replacement':devdraw,'dev_row_indices_with_multiplicity':devrows,'dev_n':{t:len(v) for t,v in partition(devrows,dlabels).items()}})
 return {'root_seed':seed,'repetitions':repetitions,'streams':['sequence_SRSWOR','train_exact_component_subset','generation_bootstrap_CE_AR','dev_component_bootstrap'],'SeedSequence_spawn_keys':[list(c.spawn_key) for c in children],'initial_PCG64_states':before,'final_PCG64_states':[r.state for r in rngs],'train_component_names':tn,'train_component_rows':tg,'train_component_weights':weights,'dev_component_names':dn,'dev_component_rows':dg,'dev_component_weights':dweights,'uniform_feasible_train_component_subset_count':str(count),'indices':records}

def lengths_and_features(rows,features):
 lengths=[len(r['sequence']) for r in rows];a=np.asarray(features,dtype=np.float64);names,groups,w=component_schema(rows)
 return {'rows':len(rows),'length_counts':dict(sorted(Counter(lengths).items())),'length_quantiles_linear':dict(zip(('q0','q10','q25','q50','q75','q90','q100'),map(float,np.quantile(lengths,[0,.1,.25,.5,.75,.9,1],method='linear')))),'component_count':len(names),'component_size_counts':dict(sorted(Counter(map(len,groups)).items())),'fixed_feature_mean':a.mean(axis=0).tolist(),'fixed_feature_population_std':a.std(axis=0).tolist(),'fixed_feature_min':a.min(axis=0).tolist(),'fixed_feature_max':a.max(axis=0).tolist()}

def check_protocol(p):
 assert p['scope_id']==SCOPE and p['repetitions']==2048 and p['root_seed']==20571 and p['generation_bootstrap_methods']==list(BOOT_METHODS)
 assert p['point_estimate_methods']==list(METHODS) and p['sampling_n_per_terminal']==16 and p['numpy_version']=='1.26.4'
 assert np.__version__=='1.26.4' and sha(__file__)==p['implementation_sha256']
 required=[p[k] for k in ('train_path','development_path','row_binding_path','frozen_metrics_path','DP_source_path')]+[b['path'] for b in p['generated_inputs']]
 assert len(required)==len(set(required))==9
 assert [b['method'] for b in p['generated_inputs']]==list(METHODS)
 assert len(p['input_bindings'])==len(required) and {b['path'] for b in p['input_bindings']}==set(required)
 for b in p['input_bindings']:
  path=Path(b['path']);assert path.suffix in ('.json','.csv','.py') and 'test37' not in str(path).lower()
  assert path.is_absolute() and path.stat().st_size==b['bytes'] and sha(path)==b['sha256'],str(path)

def run(protocol_path,preaudit_path,preaudit_sha):
 global ACTIVE_OUTPUT
 assert sys.flags.optimize==0,'Do not disable input validation with Python optimization'
 p=json.loads(Path(protocol_path).read_text());check_protocol(p)
 protocol_sha_before=sha(protocol_path)
 assert sha(preaudit_path)==preaudit_sha
 pre=json.loads(Path(preaudit_path).read_text());assert pre['status']=='PASS' and pre['scope_id']==SCOPE and pre['protocol_sha256']==sha(protocol_path) and pre['implementation_sha256']==sha(__file__) and pre['natural_statistics_authorized'] is True
 out=Path(p['output_directory']);assert not out.exists();assert out==ROOT/'night_followup_v1/distribution/results_v1'
 base=load_module(p['frozen_metrics_path'],'frozen_421_metrics',METRIC_SHA);dpmod=load_module(p['DP_source_path'],'reviewed_DP_helpers',DP_SHA)
 def rows(path):
  with Path(path).open(newline='') as f:return list(csv.DictReader(f))
 tr,dev=rows(p['train_path']),rows(p['development_path']);assert len(tr)==279 and len(dev)==41
 binding=json.loads(Path(p['row_binding_path']).read_text())
 for role,cohort in [('train',tr),('val',dev)]:
  bound=binding['splits'][role]['rows'];assert len(bound)==len(cohort)
  for x,y in zip(cohort,bound):
   assert all(str(x[k])==str(y[k]) for k in ('sequence','sequence_sha256','molecule_sha256','terminal_state','class','component_id'))
   assert 1<=len(x['sequence'])<=48 and set(x['sequence'])<=base.AA_SET
   assert hashlib.sha256(x['sequence'].encode('ascii')).hexdigest()==x['sequence_sha256']
 assert not {r['component_id'] for r in tr}&{r['component_id'] for r in dev}
 assert not {r['sequence'] for r in tr}&{r['sequence'] for r in dev}
 assert Counter(r['terminal_state'] for r in tr)=={'FREE':126,'AMD':153} and Counter(r['terminal_state'] for r in dev)=={'FREE':18,'AMD':23}
 for cohort,expected_n,expected_cross in [(tr,224,7),(dev,33,2)]:
  _,groups,weights=component_schema(cohort);assert len(groups)==expected_n and sum(f>0 and a>0 for f,a in weights)==expected_cross
  assert all(r['terminal_state']==TERMINALS[int(r['class'])] for r in cohort)
 genrows={m:{t:[] for t in TERMINALS} for m in METHODS}
 for item in p['generated_inputs']:
  for r in rows(item['path']):
   assert r['method']==item['method'] and r['terminal'] in TERMINALS;genrows[r['method']][r['terminal']].append(r)
 for m in METHODS:
  for t in TERMINALS:
   a=sorted(genrows[m][t],key=lambda r:r['sample_id']);assert len(a)==16 and [r['sample_id'] for r in a]==[f'{m}_{t}_{i:03d}' for i in range(16)];genrows[m][t]=a
   assert all(int(r['length'])==len(r['sequence']) and int(r['seed'])=={'FREE':422,'AMD':423}[t] for r in a)
 out.mkdir(parents=True);ACTIVE_OUTPUT=out
 seal=generate_indices(tr,dev,dpmod,2048,20571);seal['sealed_before_feature_statistics']=True;dump(out/'INDEX_SEAL.json',seal)
 trf=np.asarray([base.feature_vector(r['sequence']) for r in tr],dtype=np.float64);df=np.asarray([base.feature_vector(r['sequence']) for r in dev],dtype=np.float64)
 tl=np.array([int(r['class']) for r in tr]);dl=np.array([int(r['class']) for r in dev]);trids=partition(range(len(tr)),tl);devids=partition(range(len(dev)),dl)
 gf={m:{t:[base.feature_vector(r['sequence']) if 1<=len(r['sequence'])<=48 and set(r['sequence'])<=base.AA_SET else None for r in genrows[m][t]] for t in TERMINALS} for m in METHODS}
 observed={};cohorts={};analytic={}
 for t in TERMINALS:
  x,y=trf[trids[t]],df[devids[t]]
  cohorts[t]={'train':lengths_and_features([tr[i] for i in trids[t]],x),'development':lengths_and_features([dev[i] for i in devids[t]],y),'train_minus_development_feature_mean':(x.mean(axis=0)-y.mean(axis=0)).tolist(),'train_development_MMD':mmd_blocks(x,y)}
  analytic[t]=analytic_srs(x,y,16)
 for m in METHODS:
  observed[m]={}
  for t in TERMINALS:
   x=selected_features(gf[m][t],range(16));a=genrows[m][t];seq=[r['sequence'] for i,r in enumerate(a) if gf[m][t][i] is not None]
   calculated=mmd_blocks(x,df[devids[t]])
   frozen=base.compare_summaries(base.sequence_summary(seq),base.sequence_summary([dev[i]['sequence'] for i in devids[t]]))['mmd2_linear_biased_fixed_features']
   assert (frozen is None and calculated['total'] is None) or abs(frozen-calculated['total'])<1e-12
   observed[m][t]={'requests':16,'status_counts':dict(Counter(r['status'] for r in a)),'valid_1_48_n':len(x),'valid_5_48_n':sum(5<=len(s)<=48 for s in seq),'unique_valid':len(set(seq)),'vs_train':mmd_blocks(x,trf[trids[t]]),'vs_development':calculated,'frozen_summary':base.sequence_summary(seq),'sample_ids':[r['sample_id'] for r in a]}
 refs=[];boots=[];fixedgen={m:{t:list(range(16)) for t in TERMINALS} for m in BOOT_METHODS};fulldev=list(range(len(dev)))
 for item in seal['indices']:
  if item['repetition']%64==0 and time.perf_counter()-START>p['wall_cap_seconds']:raise TimeoutError('fixed analysis budget; no automatic retry or more repetitions')
  rep={'repetition':item['repetition'],'sequence_without_replacement':{},'whole_component_exact_16_16':{}}
  for t in TERMINALS:
   for label,key in [('sequence_without_replacement','train_sequence_without_replacement_row_indices'),('whole_component_exact_16_16','train_component_row_indices')]:
    ix=item[key][t];rep[label][t]={'MMD':mmd_blocks(trf[ix],df[devids[t]]),'train_reference_n':16,'fixed_development_n':len(devids[t])}
  refs.append(rep)
  panels={'generation_only':bootstrap_panel(gf,df,dl,item['generation_slot_indices'],fulldev),'development_components_only':bootstrap_panel(gf,df,dl,fixedgen,item['dev_row_indices_with_multiplicity']),'joint':bootstrap_panel(gf,df,dl,item['generation_slot_indices'],item['dev_row_indices_with_multiplicity'])}
  boots.append({'repetition':item['repetition'],**panels})
 reference_summaries={}
 for label in ['sequence_without_replacement','whole_component_exact_16_16']:
  reference_summaries[label]={}
  for t in TERMINALS:
   values={k:[r[label][t]['MMD'][k] for r in refs] for k in ('total',*BLOCKS)}
   reference_summaries[label][t]={'MMD_summary':{k:summary(v) for k,v in values.items()},'generated_empirical_positions':{m:empirical_position(observed[m][t]['vs_development']['total'],values['total']) for m in METHODS}}
 for t in TERMINALS:analytic[t]['Monte_Carlo_mean_minus_exact_expectation']={k:reference_summaries['sequence_without_replacement'][t]['MMD_summary'][k]['mean']-v for k,v in analytic[t]['exact_expected_sample_mean_mmd'].items()}
 boot_summaries={}
 for panel in ['generation_only','development_components_only','joint']:
  boot_summaries[panel]={'methods':{},'CE_minus_AR':{},'dev_n_histogram':{t:dict(sorted(Counter(r[panel]['dev_n'][t] for r in boots).items())) for t in TERMINALS},'zero_terminal_repetitions':{t:sum(r[panel]['dev_n'][t]==0 for r in boots) for t in TERMINALS}}
  for m in BOOT_METHODS:
   boot_summaries[panel]['methods'][m]={t:{k:summary([r[panel]['methods'][m][t]['MMD'][k] for r in boots]) for k in ('total',*BLOCKS)} for t in TERMINALS}
  boot_summaries[panel]['CE_minus_AR']={t:{k:summary([r[panel]['CE_minus_AR'][t][k] for r in boots]) for k in ('total',*BLOCKS)} for t in TERMINALS}
  boot_summaries[panel]['CE_minus_AR_equal_terminal_mean']={k:summary([r[panel]['CE_minus_AR_equal_terminal_mean'][k] for r in boots]) for k in ('total',*BLOCKS)}
 for b in p['input_bindings']:assert sha(b['path'])==b['sha256'] and Path(b['path']).stat().st_size==b['bytes']
 assert sha(protocol_path)==protocol_sha_before and sha(preaudit_path)==preaudit_sha
 dump(out/'REFERENCE_REPLICATES.json',refs);dump(out/'BOOTSTRAP_REPLICATES.json',boots)
 result={'scope_id':SCOPE,'protocol':record(protocol_path),'preaudit':record(preaudit_path),'source':record(__file__),'index_seal':record(out/'INDEX_SEAL.json'),'cohort_descriptions':cohorts,'joint_component_schema':{k:seal[k] for k in ['train_component_names','train_component_weights','dev_component_names','dev_component_weights','uniform_feasible_train_component_subset_count']},'observed_four_arm':observed,'train16_fixed_dev_reference':reference_summaries,'sequence_SRSWOR_analytic_check':analytic,'fixed_fit_bootstrap_sensitivity':boot_summaries,'interpretation':p['interpretation'],'operations':{'model_calls':0,'test_reads':0,'training_updates':0,'encoder_calls':0,'new_generation_calls':0,'latent_array_reads':0,'fixed_repetitions_per_scheme':2048},'all_input_hashes_unchanged':True,'analysis_wall_seconds_before_summary_io':time.perf_counter()-START,'original_CE_E_L_G':'NOT_EVALUABLE_PERMANENTLY','PROMOTE':'HOLD'}
 dump(out/'SUMMARY.json',result)
 dump(out/'MANIFEST.json',{'files':[record(q) for q in sorted(out.iterdir()) if q.is_file()],'inputs':p['input_bindings'],'protocol':record(protocol_path),'source':record(__file__),'preaudit':record(preaudit_path),'all_input_hashes_unchanged':True,'model_calls':0,'test_reads':0})
 dump(out/'CLOSING_LEDGER.json',{'state':'completed','manifest_sha256':sha(out/'MANIFEST.json'),'process_seconds_after_manifest_before_closing':time.perf_counter()-START,'original_CE_E_L_G':'NOT_EVALUABLE_PERMANENTLY','PROMOTE':'HOLD'})
 print(json.dumps({'output':str(out),'manifest_sha256':sha(out/'MANIFEST.json'),'process_seconds':time.perf_counter()-START,'model_calls':0}))

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--protocol',required=True);ap.add_argument('--preaudit',required=True);ap.add_argument('--preaudit-sha256',required=True);args=ap.parse_args()
 try:run(args.protocol,args.preaudit,args.preaudit_sha256)
 except BaseException as e:
  if ACTIVE_OUTPUT is not None:
   dump(ACTIVE_OUTPUT/'FAILURE_LEDGER.json',{'state':'analysis_failed','error_type':type(e).__name__,'error':str(e),'traceback':traceback.format_exc(),'elapsed_seconds':time.perf_counter()-START,'model_calls':0,'test_reads':0,'no_automatic_retry':True,'PROMOTE':'HOLD'})
  raise
