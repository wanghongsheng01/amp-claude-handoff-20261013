"""Single independent joint audit of frozen night diagnostics. No model/RNG/RDKit."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):os.environ[k]='1'
import time
START=time.perf_counter()
import csv,hashlib,json,math,traceback
from collections import Counter
from fractions import Fraction
from functools import lru_cache
from pathlib import Path
import numpy as np
R=Path('/Users/loctek/Documents/Codex/2026-10-11/task-4/amp_h5');N=R/'night_followup_v1'
SC=N/'rms_scoring_run_v2';M=R/'runs/ce_rms_reconstruction_v2';D=N/'distribution/results_v2'
OUT=N/'review/NIGHT_FOLLOWUP_JOINT_POSTFLIGHT_v1.json';CHECKS=[];BOUND={};NUMERIC=0
AA='ACDEFGHIKLMNPQRSTVWY';TS=('FREE','AMD');BK={'length':(0,1),'AA':(1,21),'dipeptide':(21,421)}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def rec(p):
 p=Path(p);return {'path':str(p),'sha256':sha(p),'bytes':p.stat().st_size}
def load(p):
 p=Path(p);BOUND[str(p)]=rec(p);return json.loads(p.read_text())
def rows(p):
 p=Path(p);BOUND[str(p)]=rec(p)
 with p.open(newline='') as f:return list(csv.DictReader(f))
def jsonl(p):
 p=Path(p);BOUND[str(p)]=rec(p);return [json.loads(x) for x in p.read_text().splitlines()]
def manifest(p):
 p=Path(p);d=load(p)
 for b in d['files']:
  q=Path(b['path']);q=q if q.is_absolute() else p.parent/q
  assert sha(q)==b['sha256'] and q.stat().st_size==b['bytes'],str(q)
 return d
def check(n,v=True):
 CHECKS.append({'name':n,'passed':bool(v)})
 if not v:raise AssertionError(n)
def eq(a,b):
 global NUMERIC
 if isinstance(a,dict):
  assert set(a)<=set(b)
  for k in a:eq(a[k],b[k])
 elif isinstance(a,(list,tuple)):
  assert len(a)==len(b)
  for x,y in zip(a,b):eq(x,y)
 elif isinstance(a,(int,float,np.number)) and not isinstance(a,bool):
  NUMERIC+=1;assert b is not None and math.isfinite(float(b)) and abs(float(a)-float(b))<=1e-11*max(1,abs(float(a))), (a,b)
 else:assert a==b,(a,b)
def core(a,b):
 prev=list(range(len(b)+1))
 for i,x in enumerate(a,1):
  now=[i]
  for j,y in enumerate(b,1):now.append(min(now[-1]+1,prev[j]+1,prev[j-1]+(x!=y)))
  prev=now
 c=sum(x==y for x,y in zip(a,b));den=max(len(a),len(b))
 return {'target_residues':len(a),'correct_target_positions':c,'nonpad_accuracy':c/len(a) if a else None,'whole_exact':a==b,'levenshtein_raw':prev[-1],'levenshtein_normalized':prev[-1]/den if den else 0.,'length_signed':len(b)-len(a),'length_absolute':abs(len(b)-len(a))}
@lru_cache(None)
def feature(s):
 if not s or len(s)>48 or not set(s)<=set(AA):return None
 c=Counter(s);p=Counter(s[i:i+2] for i in range(len(s)-1));z=math.sqrt(3)
 return np.array([len(s)/48/z]+[c[a]/len(s)/z for a in AA]+[p[a+b]/max(1,len(s)-1)/z for a in AA for b in AA])
def mmd(x,y):
 if not len(x) or not len(y):return {k:None for k in (*BK,'total')}
 delta=np.mean(x,axis=0)-np.mean(y,axis=0)
 return {**{k:float(np.sum(delta[a:b]**2)) for k,(a,b) in BK.items()},'total':float(np.sum(delta**2))}
def summary(values):
 good=[x for x in values if x is not None];a=np.array(good)
 if not good:return {'repetitions':len(values),'defined':0,'undefined':len(values),'quantiles':None}
 return {'repetitions':len(values),'defined':len(good),'undefined':len(values)-len(good),'mean':float(a.mean()),'std_ddof0':float(a.std()),'min':float(a.min()),'max':float(a.max()),'quantiles':dict(zip(('q025','q50','q975'),map(float,np.quantile(a,[.025,.5,.975],method='linear'))))}
def score_audit():
 execution=load(N/'rms_scoring_execution_v2/RESULT.json');closing=load(SC/'CLOSING_LEDGER.json');manifest(SC/'MANIFEST.json');status=load(SC/'STATUS.json')
 check('score_terminal_exit_and_caps',execution['state']=='completed' and execution['exit_code']==0 and not execution['timed_out'] and execution['wall_seconds']<=360 and closing['state']=='completed' and closing['wall_seconds_after_manifest_before_closing']<=300 and closing['manifest_sha256']==sha(SC/'MANIFEST.json'))
 check('score_authority_and_sources',all(sha(p)==h for p,h in execution['inputs'].items()) and status['all_inputs_unchanged'] and status['all_sources_unchanged'] and load(R/'review/independent/CE_RMS_RECONSTRUCTION_POSTFLIGHT_v2.json')['status']=='PASS')
 refs={(x['split'],int(x['row_id'])):x for x in rows(M/'reference_rows.csv')};pred={(x['split'],int(x['row_id']),x['mode'],int(x['draw_index'])):x for x in rows(M/'reconstructions.csv')}
 metrics={(x['split'],x['row_id'],x['mode'],x['draw_index']):x for x in jsonl(SC/'ROW_METRICS.jsonl')}
 check('score320refs2880predictions_complete',len(refs)==320 and len(pred)==len(metrics)==2880 and set(pred)==set(metrics) and all(x['status'] in ('ok','empty_sequence','no_eos_at_limit') and x['prediction_finite']=='true' for x in pred.values()))
 cache={(x['sequence'],x['terminal']):x for x in jsonl(SC/'DESCRIPTOR_CACHE.jsonl')};computed={}
 for key,row in pred.items():
  truth=refs[key[:2]]['truth_sequence'];text=row['prediction_sequence'];mean=pred[key[:2]+('M',-1)]['prediction_sequence'];actual=metrics[key]
  c=core(truth,text);eq(c,actual['R_to_prediction']);computed[key]=c
  if key[2]=='S':eq(core(mean,text),actual['M_to_S'])
  for tag,a,b in [('R_to_prediction',truth,text)]+([('M_to_S',mean,text)] if key[2]=='S' else []):
   fa,fb=feature(a),feature(b);saved=actual['feature_'+tag]
   if fa is None or fb is None:assert saved['valid'] is False and saved['delta421'] is None
   else:
    delta=fb-fa;eq(delta.tolist(),saved['delta421']);assert saved['valid']
    for name,(start,stop) in zip(('length1','aa20','dipeptide400'),BK.values()):
     v=delta[start:stop];eq({'signed_sum':math.fsum(v),'L1':math.fsum(abs(v)),'squared_L2':math.fsum(v*v)},saved['blocks'][name])
   left,right=cache[(a,row['terminal'])],cache[(b,row['terminal'])];desc=actual['descriptor_'+tag]
   if left['status']==right['status']=='ok':eq([y-x for x,y in zip(left['global_raw'],right['global_raw'])],desc['raw_delta']);assert desc['status']=='ok'
   else:assert desc['status']=='missing' and desc['raw_delta'] is None
 check('all2880_sequence_edit_length_and_saved_property_deltas',True)
 def acc(key):return Fraction(computed[key]['correct_target_positions'],computed[key]['target_residues'])
 ma=sum((acc(('development',i,'M',-1)) for i in range(41)),Fraction())/41
 sa=sum((sum((acc(('development',i,'S',j)) for j in range(8)),Fraction())/8 for i in range(41)),Fraction())/41
 gap=ma-sa;gate=load(SC/'INVESTMENT_SCREEN.json')
 eq({'M_sequence_equal_accuracy':float(ma),'S8_sequence_equal_accuracy':float(sa),'M_minus_S8':float(gap)},gate)
 check('exact_fraction_primary_gate_full41x8',gate['status']=='EVALUABLE' and gate['all320x9_complete_finite'] is True and gate['difference_exact_fraction']=={'numerator':gap.numerator,'denominator':gap.denominator} and gate['crosses_investment_threshold']==(gap>=Fraction(1,20)))
 strat=load(SC/'STRATIFIED_CORE_SUMMARIES.json');baseline={}
 for group in strat:
  split,terminal,mode=group['split'],group['terminal'],group['mode_group'];ids=[i for s,i in refs if s==split and (terminal=='ALL' or refs[(s,i)]['terminal']==terminal)]
  draws=group['draw_indices'];which='M' if mode=='M' else 'S';keys=[(split,i,which,j) for i in ids for j in draws];support=Counter(a for i in ids for a in refs[(split,i)]['truth_sequence']);correct=Counter()
  for k in keys:
   for a,b in zip(refs[k[:2]]['truth_sequence'],pred[k]['prediction_sequence']):
    if a==b:correct[a]+=1
  recall={a:correct[a]/(len(draws)*support[a]) if support[a] else None for a in AA};macro=sum(recall[a] for a in AA if support[a])/sum(support[a]>0 for a in AA)
  eq({'value':macro,'support_per_original_sequence_set':{a:support[a] for a in AA},'correct_across_draws':{a:correct[a] for a in AA},'recall_by_AA':recall},group['supported_AA_macro'])
  for field in ('nonpad_accuracy','whole_exact','levenshtein_raw','levenshtein_normalized','length_signed','length_absolute'):
   value=sum(sum(computed[(split,i,which,j)][field] for j in draws)/len(draws) for i in ids)/len(ids)
   eq(value,group['per_sequence_then_equal_sequence_means'][field]['mean'])
  if split=='development' and terminal=='ALL':baseline[mode]=group
 check('all60_stratified_groups_macro_support_counts_and_core_means',len(strat)==60)
 return {'status':'PASS','primary_gate':gate,'baseline_success_guards':{'M_equal_sequence_accuracy':float(ma),'S8_equal_sequence_accuracy':float(sa),'M_length_MAE':baseline['M']['per_sequence_then_equal_sequence_means']['length_absolute']['mean'],'S8_supported_AA_macro':baseline['S8']['supported_AA_macro']['value']},'scoring_outer_seconds':execution['wall_seconds'],'RMS_descriptor_chemistry_review':'Original function/version bound and saved raw deltas checked; no independent additional RDKit calls.'}
def distribution_audit():
 manifest(N/'distribution/DELIVERY_INDEX_v2.json');manifest(D/'MANIFEST.json');execution=load(N/'distribution/execution_v2/RESULT.json');closing=load(D/'CLOSING_LEDGER.json')
 check('distribution_completed_within_caps',execution['exit_code']==0 and not execution['outer_timeout'] and execution['wall_seconds']<=180 and closing['state']=='completed' and closing['process_seconds_after_manifest_before_closing']<=120 and closing['manifest_sha256']==sha(D/'MANIFEST.json'))
 p=load(N/'distribution/PROTOCOL_v2.json');s=load(D/'SUMMARY.json');seal=load(D/'INDEX_SEAL.json');rr=load(D/'REFERENCE_REPLICATES.json');br=load(D/'BOOTSTRAP_REPLICATES.json')
 for b in p['input_bindings']:assert sha(b['path'])==b['sha256']
 tr=rows(p['train_path']);dv=rows(p['development_path']);tf=np.array([feature(x['sequence']) for x in tr]);df=np.array([feature(x['sequence']) for x in dv]);di={t:[i for i,x in enumerate(dv) if x['terminal_state']==t] for t in TS}
 gf={}
 for item in p['generated_inputs']:
  rs=rows(item['path']);gf[item['method']]={t:np.array([feature(x['sequence']) for x in sorted(rs,key=lambda r:r['sample_id']) if x['terminal']==t]) for t in TS}
  for t in TS:eq(mmd(gf[item['method']][t],df[di[t]]),s['observed_four_arm'][item['method']][t]['vs_development'])
 check('frozen421_point_estimates_fourarms_independently_recomputed',True)
 tg,dg=seal['train_component_rows'],seal['dev_component_rows'];count=0
 assert len(seal['indices'])==len(rr)==len(br)==2048
 for idx,ref,boot in zip(seal['indices'],rr,br):
  assert idx['repetition']==ref['repetition']==boot['repetition']==count
  picked=[i for g in idx['train_whole_component_indices'] for i in tg[g]];drows=[i for g in idx['dev_component_indices_with_replacement'] for i in dg[g]]
  assert len(set(idx['train_whole_component_indices']))==len(idx['train_whole_component_indices']) and len(idx['dev_component_indices_with_replacement'])==33 and drows==idx['dev_row_indices_with_multiplicity']
  dd={t:[i for i in drows if dv[i]['terminal_state']==t] for t in TS}
  for t in TS:
   assert idx['train_component_row_indices'][t]==[i for i in picked if tr[i]['terminal_state']==t]
   assert len(idx['train_component_row_indices'][t])==16
   seqids=idx['train_sequence_without_replacement_row_indices'][t]
   assert len(seqids)==len(set(seqids))==16 and all(tr[i]['terminal_state']==t for i in seqids)
   for scheme,key in [('sequence_without_replacement','train_sequence_without_replacement_row_indices'),('whole_component_exact_16_16','train_component_row_indices')]:eq(mmd(tf[idx[key][t]],df[di[t]]),ref[scheme][t]['MMD'])
  for panel in ('generation_only','development_components_only','joint'):
   devindex=di if panel=='generation_only' else dd
   vals={}
   for method in ('CE21_DDPM','terminal_LSTM'):
    vals[method]={}
    for t in TS:
     genindex=list(range(16)) if panel=='development_components_only' else idx['generation_slot_indices'][method][t]
     assert len(genindex)==16 and all(0<=i<16 for i in genindex)
     vals[method][t]=mmd(gf[method][t][genindex],df[devindex[t]])
     eq(vals[method][t],boot[panel]['methods'][method][t]['MMD']);assert boot[panel]['methods'][method][t]['generation_requests']==16 and boot[panel]['dev_n'][t]==len(devindex[t])
   for k in (*BK,'total'):
    differences=[]
    for t in TS:
     x,y=vals['CE21_DDPM'][t][k],vals['terminal_LSTM'][t][k];v=None if x is None or y is None else x-y;eq(v,boot[panel]['CE_minus_AR'][t][k]);differences.append(v)
    eq(None if any(x is None for x in differences) else sum(differences)/2,boot[panel]['CE_minus_AR_equal_terminal_mean'][k])
  count+=1
 check('all2048_wholecomponent_indices_and_all_reference_bootstrap_MMDs_recomputed',count==2048)
 for scheme in ('sequence_without_replacement','whole_component_exact_16_16'):
  for t in TS:
   for k in (*BK,'total'):eq(summary([x[scheme][t]['MMD'][k] for x in rr]),s['train16_fixed_dev_reference'][scheme][t]['MMD_summary'][k])
   values=[x[scheme][t]['MMD']['total'] for x in rr]
   for method in gf:
    obs=s['observed_four_arm'][method][t]['vs_development']['total'];pos=s['train16_fixed_dev_reference'][scheme][t]['generated_empirical_positions'][method]
    assert pos['below']==sum(x<obs for x in values) and pos['exact_ties']==sum(x==obs for x in values) and pos['above']==sum(x>obs for x in values)
 for t in TS:
  x=tf[[i for i,v in enumerate(tr) if v['terminal_state']==t]];a=s['sequence_SRSWOR_analytic_check'][t];fac=(len(x)-16)/(16*(len(x)-1));center=x-x.mean(0);base=mmd(x,df[di[t]])
  for k,(lo,hi) in {**BK,'total':(0,421)}.items():eq(base[k]+fac*np.mean(np.sum(center[:,lo:hi]**2,axis=1)),a['exact_expected_sample_mean_mmd'][k])
 check('reference_quantiles_positions_and_SRS_population_denominator',True)
 check('distribution_scope_limits_preserved',s['original_CE_E_L_G']=='NOT_EVALUABLE_PERMANENTLY' and s['PROMOTE']=='HOLD')
 return {'status':'PASS','repetitions_checked':count,'new_RNG_calls':0,'outer_seconds':execution['wall_seconds'],'conclusion':'Fixed-fit empirical sensitivity only; ranges are not CI/p-values. No new method gate. Single training seed, repeatedly used dev and train-seen references remain.'}
def documents_audit():
 manifest(N/'posterior_path/MANIFEST.json');cached=load(N/'posterior_path/RESULTS.json')
 for row in cached['paired_raw_rows']:
  a=core(row['truth_sequence'],row['mean_prediction_sequence']);b=core(row['truth_sequence'],row['B_prediction_sequence'])
  eq(a['correct_target_positions'],row['mean_correct_residues']);eq(b['correct_target_positions'],row['B_correct_residues']);eq(a['nonpad_accuracy']-b['nonpad_accuracy'],row['mean_minus_B_accuracy'])
 check('B_cached32_slots31donors_exact_persequence_recount',len(cached['paired_raw_rows'])==32 and len({x['donor_train_row_id'] for x in cached['paired_raw_rows']})==31)
 manifest(N/'manuscript_candidate/MANIFEST.json');emap=load(N/'manuscript_candidate/EVIDENCE_MAP.json')
 for b in emap['sources'].values():assert sha(b['path'])==b['sha256'] and Path(b['path']).stat().st_size==b['bytes']
 docs={k:load(v['path']) for k,v in emap['sources'].items() if Path(v['path']).suffix=='.json'}
 for filename in ('TABLE_DEV_MEAN.csv','TABLE_FOUR_ARM_GENERATION.csv','TABLE_POSTHOC_SOURCE.csv'):
  for row in rows(N/'manuscript_candidate'/filename):
   source=docs[row['source_id']]
   for part in row['json_pointer'].split('/')[1:]:source=source[int(part)] if isinstance(source,list) else source[part]
   for k,v in row.items():
    if k in ('source_id','json_pointer') or k not in source:continue
    if isinstance(source[k],(int,float)):eq(float(v),source[k])
    elif isinstance(source[k],str):assert v==source[k]
 check('candidate_manifest_12sources_and_three_table_pointers_bound',len(emap['sources'])==12)
 methods=(N/'manuscript_candidate/METHODS_CANDIDATE.zh.md').read_text();results=(N/'manuscript_candidate/RESULTS_CANDIDATE.zh.md').read_text()
 check('candidate_explicit_scope_and_inference_boundaries',all(t in methods+results for t in ['NOT_EVALUABLE','HOLD','NOT_ESTABLISHED','逐坐标','B decoder-z','额外','9999.154718747828206','53.70622158146245758']) and emap['new_night_diagnostics_included'] is False and emap['original_manuscript_edited'] is False)
 return {'status':'PASS_WITH_MINOR_WORDING_NOTE','scope':'Sealed four-arm and source diagnostic candidate only, not the new night measurements; original manuscript unedited.','manual_review':'Read full Methods/Results; correct 701/41, eight zero-recall classes, 4x2x16 denominators, original CE NE versus recovery, AR limitations, B/C/D 96new+32reuse, C=B sampled-z, D extra reference-g, historical cost lower-bound and unknown wall. No superiority, novelty or activity claim.','minor_nonblocking_note':'In a future candidate version, say VAE logs per-update training loss and every100-step dev mean reconstruction, while AR records train/dev NLL; current generic training/development monitoring wording is less precise. Do not overwrite this sealed candidate.'}
if __name__=='__main__':
 if OUT.exists():raise FileExistsError('preserve prior audit; use explicit new version')
 result={'status':'FAIL','scope_id':'AMP_H5_NIGHT_FOLLOWUP_JOINT_REVIEW_V1','checks':CHECKS,'model_calls':0,'new_RNG_calls':0,'new_RDKit_calls':0,'test_reads':0,'original_CE_E_L_G':'NOT_EVALUABLE_PERMANENTLY','PROMOTE':'HOLD','training_authorized':False}
 try:
  result['scoring_postflight']=score_audit();result['distribution_postflight']=distribution_audit();result['candidate_documents_review']=documents_audit()
  result['status']='PASS';result['warmup_investment_judgment']={'eligible_for_exact_preflight':result['scoring_postflight']['primary_gate']['crosses_investment_threshold'],'judgment':'One bounded known-method beta-warmup VAE-only1000 falsification is reasonable under the prelocked four success guards. It is not a unique KL/sigma causal diagnosis or generation advance. No launch without exact plan and independent receipt.','root_guard_sha256':sha(N/'next_experiment/WARMUP_EXECUTION_GUARDS_ROOT_PRELOCK_v1.json')}
  result['candidate_warmup_static_review']={'status':'MATHEMATICS_READY_EXACT_EXECUTION_PREAUDIT_PENDING','manifest':rec(N/'warmup_candidate/MANIFEST.json'),'scope':'Read wrapper/objective/loop hooks/code-only bootstrap; no model or repeat synthetic run.','conditions':['Exact root receipt binds complete score closing/outer result and this exact rational gate; do not replace with a handwritten rounded float gap.','Final budget success requires external supervisor wall <=2100; wrapper completed and its pre-manifest timer are insufficient.','No posttraining RMS, DDPM or generation included automatically; separate scoped evaluation needed.'],'execution_authorized':False}
 except Exception as exc:result.update(error=repr(exc),traceback=traceback.format_exc())
 result['source_bindings']=list(BOUND.values())+[rec(Path(__file__))];result['check_count']=len(CHECKS);result['numeric_values_compared']=NUMERIC;result['audit_seconds_before_receipt_io']=time.perf_counter()-START
 with OUT.open('x') as f:json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
 print(json.dumps({'status':result['status'],'path':str(OUT),'sha256':sha(OUT),'checks':len(CHECKS),'numeric_values_compared':NUMERIC,'audit_seconds':result['audit_seconds_before_receipt_io'],'error':result.get('error')}))
 raise SystemExit(0 if result['status']=='PASS' else 2)
