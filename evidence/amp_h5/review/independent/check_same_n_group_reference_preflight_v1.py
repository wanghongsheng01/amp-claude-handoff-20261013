#!/usr/bin/env python3
import ast,hashlib,itertools,json
from fractions import Fraction
from pathlib import Path
from datetime import datetime,timezone
R=Path('/Users/loctek/Documents/Codex/2026-10-11/task-4/amp_h5')
P=R/'protocol/LATENT_SAME_N_GROUP_REFERENCE_v1.json';S=R/'implementation/latent_same_n_reference_v1.py'
OUT=R/'review/independent/LATENT_SAME_N_GROUP_REFERENCE_PREFLIGHT_v1.json'
assert not OUT.exists()
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
p=json.loads(P.read_text());s=S.read_text();tree=ast.parse(s);checks=[]
def check(name,value):checks.append({'name':name,'pass':bool(value)})
check('exact_protocol_and_script_hashes',sha(P)=='69996b7e20e0126878f2847c51e8a06d982924cc76d0b60e2d6d4e62c238d0b3' and sha(S)=='4f8e18e287b1f81a9103db1140f3ffb2d30ae4f182173fa26b63b156abc1861a' and p['implementation_sha256']==sha(S))
check('all12_frozen_input_hashes',len(p['input_bindings'])==12 and len({b['path'] for b in p['input_bindings']})==12 and all(sha(b['path'])==b['sha256'] for b in p['input_bindings']))
check('fixed256_shared_seed_and_counts',p['sampler']['repetitions']==256 and p['sampler']['seed']==20261011 and p['sampler']['target']=={'FREE':16,'AMD':16} and p['sampler']['reference']=={'FREE':110,'AMD':137})
check('only_CE_MSE_saved_inputs', [m['name'] for m in p['methods']]==['categorical_ce','raw_onehot_mse'] and all('scalar' not in b['path'] and 'test' not in b['path'] for b in p['input_bindings']))
nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ('suffix_counts','randbelow_raw','choose_subset')]
ns={};exec(compile(ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[])),str(S),'exec'),ns)
# Brute force labeled subsets and exact rational path probabilities, no RNG.
cases=[([(1,0),(0,1),(1,1)],(1,1)), ([(2,0),(0,2),(1,1),(1,0),(0,1)],(2,2)), ([(1,2),(2,1),(1,0),(0,1),(1,1)],(3,3)), ([(2,2),(1,0)],(0,0)), ([(2,2)],(1,1)), ([(1,0),(1,0),(0,1),(0,1),(1,1)],(2,2))]
path_counts=[];original_randbelow=ns['randbelow_raw']
for ci,(weights,target) in enumerate(cases):
 dp=ns['suffix_counts'](weights,target)
 valid=[tuple(i for i,bit in enumerate(bits) if bit) for bits in itertools.product([0,1],repeat=len(weights)) if tuple(sum(weights[i][j] for i,bit in enumerate(bits) if bit) for j in (0,1))==target]
 check('toy'+str(ci)+'_DP_matches_exhaustive',dp[0][target[0]][target[1]]==len(valid))
 masses=[]
 for subset in valid:
  f,a=target;answers=[];prob=Fraction(1)
  for i,(wf,wa) in enumerate(weights):
   if (f,a)==(0,0):break
   take=dp[i+1][f-wf][a-wa] if f>=wf and a>=wa else 0;skip=dp[i+1][f][a]
   if take and skip:answers.append((take+skip,0 if i in subset else take))
   prob*=Fraction(take if i in subset else skip,take+skip)
   if i in subset:f-=wf;a-=wa
  it=iter(answers)
  def draw(_,bound):
   expected,answer=next(it);assert expected==bound;return answer
  ns['randbelow_raw']=draw
  actual=tuple(ns['choose_subset'](weights,dp,target,None))
  assert actual==subset
  try:next(it);raise AssertionError('unused decision')
  except StopIteration:pass
  masses.append(prob)
 check('toy'+str(ci)+'_all_paths_uniform',not valid or all(v==Fraction(1,len(valid)) for v in masses) and sum(masses)==1)
 path_counts.append(len(valid))
ns['randbelow_raw']=original_randbelow
class FakeBits:
 def __init__(self,blocks):self.blocks=list(blocks);self.calls=[]
 def random_raw(self,words):
  self.calls.append(words);block=self.blocks.pop(0);assert len(block)==words;return block
f=FakeBits([[3],[2]])
check('integer_rejection_path',original_randbelow(f,3)==2 and f.calls==[1,1])
f=FakeBits([[(1<<64)-1,(1<<64)-1],[1,7]])
check('arbitrary_integer_gt64bit',original_randbelow(f,(1<<70)+123)==(1<<64)+7 and f.calls==[2,2])
f=FakeBits([]);check('deterministic_n1_no_draw',original_randbelow(f,1)==0 and not f.calls)
main=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main')
loads=[n.lineno for n in ast.walk(main) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and ast.unparse(n.func)=='np.load']
seal=[n.lineno for n in ast.walk(main) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='dump' and n.args and 'INDEX_SEAL.json' in ast.unparse(n.args[0])]
check('all_indices_sealed_before_any_array_loading',len(seal)==1 and len(loads)==2 and max(seal)<min(loads))
check('same_indices_and_reference_for_both_operands',"for method in p['methods']" in s and "for item in indices" in s and "gm=metric_bundle(generated[terminal],rs,dep,q);hm=metric_bundle(held,rs,dep,q)" in s)
check('row_and_terminal_and_component_checks_present',"np.array_equal(z['row_ids'],np.arange(279))" in s and "np.array_equal(z['labels'],labels)" in s and "assert not {rows[j]['component_id'] for j in chosen}&{rows[j]['component_id'] for j in remain}" in s)
check('component_order_and_inclusion_recorded',"names=sorted(components)" in s and "observed_component_inclusion_counts" in s and "RNG_state_before" in s and "duplicate_subset_repetitions" in s)
check('no_model_import_calls',not any(isinstance(n,ast.Import) and any(x.name.startswith('torch') for x in n.names) or isinstance(n,ast.ImportFrom) and (n.module or '').startswith('torch') for n in ast.walk(tree)) and not any(isinstance(n,ast.Call) and any(k in ast.unparse(n.func) for k in ('make_model','forward','decode','torch')) for n in ast.walk(tree)))
check('undefined_correlations_retained',"return None if den==0" in s and "'defined':len(valid),'undefined':len(values)-len(valid)" in s)
check('bounds_all279_seen_and_no_inferential_claims', 'All279' in p['comparison_design']['holdout_meaning'] and 'not confidence intervals' in p['frozen_statistics']['dependencies'] and 'NOT_EVALUABLE' in p['dev_latent']['consequence'] and p['original_CE_E_L_G']=='NOT_EVALUABLE_INTERRUPTION')
check('target_output_not_yet_created',not Path(p['output_directory']).exists())
failed=[c['name'] for c in checks if not c['pass']]
report={'created_utc':datetime.now(timezone.utc).isoformat(),'status':'PASS_STATIC_SYNTHETIC_SAVED_ARRAY_ANALYSIS_ONLY' if not failed else 'HOLD','scope_id':p['scope_id'],'model_execution_authorized':False,'pure_numpy_saved_array_analysis_eligible':not failed,'checks':checks,'failed_checks':failed,'exact_DP_bruteforce_subset_counts':path_counts,'manual_formula_review':['Joint whole-component subset DP includes mixed-terminal components once; exact integer branch sampling is uniform over feasible labeled subsets, not uniform peptide inclusion.','Gen16 and statistical-holdout16 compare against identical remaining110/137; all279 were model training examples.','Mean/std/RMS use pooled field entries; profile correlations use fixed16channel/48position/768map means, not independent biological observations.','Coordinate tails use per-coordinate extrema/linear1%-99% quantiles of only remainingreference, denominators16x16x48 and per-peptide16x48.','Analytic q RMS=sqrt(mean(mu^2+sigma^2)); no latent samples fabricated. Paired metric deltas and empirical quantiles are descriptive, not confidence intervals/null/pvalues.','All256 draws reused for CE/MSE; no dev arrays, scalar data, covariance/PCA/W2 or new model route.'],'limitations':['Posthoc after generation/latent findings; freeze prevents further selection but does not create confirmatory preregistration.','Component closure is frozen sequence-similarity grouping, not biological-family independence; empirical inclusion counts are not IPW correction.','These profile statistics do not estimate full1536-dimensional joint covariance; same n removes this sample-size mismatch only, not estimator/model dependence.','No terminal-independent new model control is approved; oldA/B64 remains paused for four-arm completion and new unified scope.'],'source_bindings':[{'path':str(P),'sha256':sha(P)},{'path':str(S),'sha256':sha(S)}]+p['input_bindings'],'reviewer_actions':{'stdlib_only':True,'natural_array_loads':0,'natural_subsets_selected':0,'random_draws':0,'model_calls':0,'live_scalar_reads':0,'test37_reads':0}}
OUT.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n');print(json.dumps({'report':str(OUT),'sha256':sha(OUT),'checks':len(checks),'failed':failed}))
