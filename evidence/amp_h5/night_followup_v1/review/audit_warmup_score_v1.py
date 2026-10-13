"""Independent saved warmup scoring audit only. No model/RNG/RDKit."""
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
SC=N/'warmup_scoring_run_v1';M=N/'warmup_posttraining_rms_admission_fix_v2/run_v1'
OUT=R/'review/independent/CE_WARMUP_RMS_SCORING_POSTFLIGHT_v1.json';CHECKS=[];BOUND={};NUMERIC=0
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
 execution=load(N/'warmup_scoring_execution_v1/RESULT.json');closing=load(SC/'CLOSING_LEDGER.json');manifest(SC/'MANIFEST.json');status=load(SC/'STATUS.json')
 check('score_terminal_exit_and_caps',execution['state']=='completed' and execution['exit_code']==0 and not execution['timed_out'] and execution['wall_seconds']<=360 and closing['state']=='completed' and closing['wall_seconds_after_manifest_before_closing']<=300 and closing['manifest_sha256']==sha(SC/'MANIFEST.json'))
 check('score_authority_and_sources',all(sha(p)==h for p,h in execution['inputs'].items()) and status['all_inputs_unchanged'] and status['all_sources_unchanged'] and load(R/'review/independent/CE_WARMUP_RMS_RECONSTRUCTION_POSTFLIGHT_v1.json')['status']=='PASS')
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
 return {'status':'PASS','primary_gate':gate,'candidate_metrics':{'M_equal_sequence_accuracy':float(ma),'S8_equal_sequence_accuracy':float(sa),'M_length_MAE':baseline['M']['per_sequence_then_equal_sequence_means']['length_absolute']['mean'],'S8_supported_AA_macro':baseline['S8']['supported_AA_macro']['value']},'scoring_outer_seconds':execution['wall_seconds'],'M_dev_micro':{'correct':sum(computed[('development',i,'M',-1)]['correct_target_positions'] for i in range(41)),'target':sum(computed[('development',i,'M',-1)]['target_residues'] for i in range(41))},'RMS_descriptor_chemistry_review':'Original function/version bound and saved raw deltas checked; no independent additional RDKit calls.'}

if __name__=='__main__':
 if OUT.exists():raise FileExistsError('preserve prior audit')
 report={'status':'FAIL','blocked':True,'scope_id':'AMP_H5_WARMUP_RMS_SCORING_V1','checks':CHECKS,'model_calls':0,'new_RNG_calls':0,'new_RDKit_calls':0,'test_reads':0,'original_CE_E_L_G':'NOT_EVALUABLE','PROMOTE':'HOLD'}
 try:
  report.update(score_audit(),blocked=False)
  cfg=load(N/'warmup_scoring_candidate_v3/SCORING_CONTRACT.json');state=load(SC/'STATUS.json');ex=load(N/'warmup_scoring_execution_v1/RESULT.json');close=load(SC/'CLOSING_LEDGER.json')
  check('exact_output_scope_closing_and_authority',ex['scope_id']==close['scope_id']==state['scope_id']==report['scope_id'] and ex['closing_ledger_sha256']==sha(SC/'CLOSING_LEDGER.json'))
  modelpost=load(R/'review/independent/CE_WARMUP_RMS_RECONSTRUCTION_POSTFLIGHT_v1.json');bound={x['path']:x['sha256'] for x in modelpost['source_bindings']}
  check('model_actual_pass_scope_contract_manifest',modelpost['status']=='PASS' and modelpost['blocked'] is False and modelpost['scope_id']==cfg['model_scope_id'] and bound[cfg['model_contract']['path']]==sha(cfg['model_contract']['path'])==cfg['model_contract']['sha256'] and bound[str(M/'manifest.json')]==sha(M/'manifest.json'))
  check('fixed_source_bindings_and_model_csv_identity',all(sha(v['path'])==v['sha256'] for v in cfg['sources'].values()) and state['input_sha256']['reference_rows']==sha(M/'reference_rows.csv') and state['input_sha256']['reconstructions']==sha(M/'reconstructions.csv') and state['input_sha256']['model_manifest']==sha(M/'manifest.json'))
  assert all(sha(p)==v['sha256'] for p,v in BOUND.items())
  check('all_consumed_inputs_unchanged',True)
 except Exception as exc:report.update(status='FAIL',blocked=True,error=repr(exc),traceback=traceback.format_exc())
 report['source_bindings']=list(BOUND.values())+[rec(Path(__file__))]
 report['interpretation']='Scoring engineering verification only. The emitted M-minus-S investment screen is descriptive for this completed warmup candidate and is not its success gate. The separately frozen four paired guards decide resource success; no further training, DDPM or generation authorized.'
 report['descriptor_limit']='Saved original raw RDKit values and their deltas/missingness are checked; no independent new chemistry computation.'
 report['check_count']=len(CHECKS);report['numeric_values_compared']=NUMERIC;report['audit_seconds_before_receipt_io']=time.perf_counter()-START
 OUT.write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
 print(json.dumps({'status':report['status'],'path':str(OUT),'sha256':sha(OUT),'checks':len(CHECKS),'numeric':NUMERIC,'audit_seconds':report['audit_seconds_before_receipt_io'],'candidate_metrics':report.get('candidate_metrics'),'M_dev_micro':report.get('M_dev_micro')}))
 raise SystemExit(0 if report['status']=='PASS' else 2)
