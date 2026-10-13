#!/usr/bin/env python3
"""Tiny deterministic fixtures only. Never open natural CSV/bindings/arrays."""
import sys
sys.dont_write_bytecode=True
import importlib.util
import itertools
import json
from fractions import Fraction
from pathlib import Path
import time
START=time.perf_counter()
HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('distribution_candidate',HERE/'distribution_diagnostic_v2.py')
d=importlib.util.module_from_spec(spec);spec.loader.exec_module(d)
np=d.np
dp=d.load_module(d.ROOT/'implementation/latent_same_n_reference_v1.py','reviewed_dp_fixture',d.DP_SHA)
base=d.load_module(d.ROOT/'implementation/frozen_metrics_v1.py','frozen_metric_fixture',d.METRIC_SHA)
checks=[]
def check(name,condition):
 assert condition,name
 checks.append({'check':name,'passed':True})

# 1. SRSWOR formula checked by exact enumeration, including all-population n=N.
x=np.zeros((4,421));x[:,0]=[0.,1.,3.,4.];x[:,1]=[2.,0.,4.,1.];x[:,21]=[1.,3.,2.,4.]
y=np.zeros((3,421));y[:,0]=[1.,2.,1.];y[:,1]=[3.,1.,0.];y[:,21]=[2.,1.,0.]
exact=d.analytic_srs(x,y,2)
enumerated=[d.mmd_blocks(x[list(ix)],y) for ix in itertools.combinations(range(4),2)]
check('SRSWOR_expectation_all_6_subsets_3_blocks',all(abs(np.mean([r[k] for r in enumerated])-v)<1e-12 for k,v in exact['exact_expected_sample_mean_mmd'].items()))
allpop=d.analytic_srs(x,y,4)
check('SRSWOR_n_equals_N_variance_term_zero',allpop['finite_population_factor']==0 and allpop['exact_expected_sample_mean_mmd']==allpop['full_train_vs_fixed_dev'])
a=['ACD','AAA','W'];b=['FGH','AC','WW']
ab=d.mmd_blocks(np.array([base.feature_vector(s) for s in a]),np.array([base.feature_vector(s) for s in b]))
old=base.compare_summaries(base.sequence_summary(a),base.sequence_summary(b))['mmd2_linear_biased_fixed_features']
check('421_frozen_metric_and_additive_blocks_identical',abs(ab['total']-old)<1e-12 and abs(ab['total']-sum(ab[k] for k in d.BLOCKS))<1e-12)

# 2. Exact component subset counts and rational path probabilities; no frequency test.
weights=[(1,0),(0,1),(1,1),(1,0),(0,1)];target=(2,2);table=dp.suffix_counts(weights,target)
feasible=[tuple(i for i in range(len(weights)) if mask&(1<<i)) for mask in range(1<<len(weights)) if tuple(sum(weights[i][c] for i in range(len(weights)) if mask&(1<<i)) for c in range(2))==target]
probabilities={}
def traverse(i,f,a,chosen,prob):
 if f==0 and a==0:
  probabilities[tuple(chosen)]=prob;return
 take=table[i+1][f-weights[i][0]][a-weights[i][1]] if f>=weights[i][0] and a>=weights[i][1] else 0
 skip=table[i+1][f][a];assert take+skip==table[i][f][a]
 if take:traverse(i+1,f-weights[i][0],a-weights[i][1],chosen+[i],prob*Fraction(take,take+skip))
 if skip:traverse(i+1,f,a,chosen,prob*Fraction(skip,take+skip))
traverse(0,*target,[],Fraction(1,1))
check('component_DP_enumeration_and_exact_uniform_conditional_probability',set(probabilities)==set(feasible) and table[0][2][2]==len(feasible) and set(probabilities.values())=={Fraction(1,len(feasible))})
rng=np.random.PCG64(20571)
check('reviewed_choose_subset_returns_only_complete_feasible_groups',all(tuple(dp.choose_subset(weights,table,target,rng)) in feasible for _ in range(16)))

# 3. Cross-terminal multiplicities preserved; empty terminals remain undefined.
groups=[[0,1],[2],[3]];labels=np.array([0,1,0,1]);expanded=d.component_rows(groups,[0,0,2]);parts=d.partition(expanded,labels)
check('whole_dev_cross_terminal_group_shared_multiplicity',expanded==[0,1,0,1,3] and parts=={'FREE':[0,0],'AMD':[1,1,3]})
gen={m:{t:[np.zeros(421) for _ in range(16)] for t in d.TERMINALS} for m in d.BOOT_METHODS}
indices={m:{t:list(range(16)) for t in d.TERMINALS} for m in d.BOOT_METHODS}
features=np.zeros((4,421));features[:,0]=[1.,2.,3.,4.]
empty=d.bootstrap_panel(gen,features,labels,indices,d.component_rows(groups,[1,1,1]))
check('empty_terminal_null_without_fill_reject_or_macro',empty['dev_n']=={'FREE':3,'AMD':0} and all(v is None for v in empty['CE_minus_AR']['AMD'].values()) and all(v is None for v in empty['CE_minus_AR_equal_terminal_mean'].values()))
paired=d.bootstrap_panel(gen,features,labels,indices,expanded)
check('shared_dev_draw_CE_AR_difference_zero_when_generated_features_equal',all(v==0 for t in d.TERMINALS for v in paired['CE_minus_AR'][t].values()))
gen['CE21_DDPM']['FREE'][0]=None
keep=d.bootstrap_panel(gen,features,labels,indices,expanded)
check('slot_is_sampling_unit_invalid_slot_not_erased_from_denominator',keep['methods']['CE21_DDPM']['FREE']['generation_requests']==16 and keep['methods']['CE21_DDPM']['FREE']['generation_valid_n']==15)

# Tiny synthetic RNG/index replay; 8 repetitions, no natural rows or statistics.
train=[{'component_id':f'g{i:02d}','class':str(c)} for i in range(20) for c in (0,1)]
dev=[{'component_id':'cross','class':'0'},{'component_id':'cross','class':'1'},{'component_id':'free','class':'0'},{'component_id':'amide','class':'1'}]
s1=d.generate_indices(train,dev,dp,8,20571);s2=d.generate_indices(train,dev,dp,8,20571)
check('fixed_streams_replay_indices_exactly',s1==s2 and all(all(len(set(r['train_sequence_without_replacement_row_indices'][t]))==16 and len(r['train_component_row_indices'][t])==16 for t in d.TERMINALS) for r in s1['indices']))
check('exact_tie_position_is_not_tail_probability',d.empirical_position(1.,[1.,1.])['tie_position_interval']==[0.,1.] and d.empirical_position(1.,[1.,1.])['midrank_position']==.5)
json.dumps({'empty':empty,'summary':d.summary([None,None]),'seal':s1},allow_nan=False)
out=HERE/'SYNTHETIC_RESULT_v2.json'
d.dump(out,{'status':'PASS_SYNTHETIC_ONLY','implementation':d.record(HERE/'distribution_diagnostic_v2.py'),'fixture':d.record(__file__),'checks':checks,'check_count':len(checks),'natural_data_reads':0,'model_calls':0,'test_reads':0,'elapsed_seconds_before_result_io':time.perf_counter()-START,'numpy_version':np.__version__})
print(json.dumps({'status':'PASS_SYNTHETIC_ONLY','checks':len(checks),'result':str(out),'sha256':d.sha(out)}))
