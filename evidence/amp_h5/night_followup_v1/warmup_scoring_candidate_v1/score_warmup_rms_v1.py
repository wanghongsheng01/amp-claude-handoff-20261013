#!/usr/bin/env python3
"""Read-only R/M/S scoring. Natural data requires both preaudit and model postflight.

No torch/model imports, random draws, source-cohort access or fitted feature scales.
"""
import time
PROCESS_START = time.monotonic()
import argparse
import csv
import hashlib
import importlib.util
import importlib.metadata
import json
import math
import re
import signal
import sys
sys.dont_write_bytecode = True
from collections import Counter
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path

SCOPE = 'AMP_H5_WARMUP_RMS_SCORING_V1'
AA = 'ACDEFGHIKLMNPQRSTVWY'
TOKEN_AA = 'ARNDCEQGHILKMFPSTWYV'
COMPLETED = {'ok','empty_sequence','no_eos_at_limit'}
GLOBAL_NAMES = ['RingCount','NumAromaticRings','NumSaturatedRings','FractionCSP3',
                'HallKierAlpha','LabuteASA','MaxPartialCharge','MinPartialCharge','TPSA']
SPLITS = {'train':279,'development':41}
SEEDS = list(range(1422,1430))

class BudgetStop(BaseException): pass
def stop_budget(*_): raise BudgetStop('300 second scoring limit')
def guard_path(path):
 p=Path(path).resolve()
 if re.search(r'(^|[/_.\-])test($|[/_.\-])',str(p).lower()): raise ValueError('test path prohibited')
 return p
def sha(path): return hashlib.sha256(guard_path(path).read_bytes()).hexdigest()
def load(path): return json.loads(guard_path(path).read_bytes())
def dump(path,obj): Path(path).write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def load_module(name,path,expected):
 if sha(path)!=expected: raise ValueError('module hash mismatch: '+name)
 spec=importlib.util.spec_from_file_location(name,guard_path(path));m=importlib.util.module_from_spec(spec)
 sys.modules[name]=m;spec.loader.exec_module(m);return m
def all_bindings(obj,relative_to=None):
 found={}
 def walk(x):
  if isinstance(x,dict):
   if isinstance(x.get('path'),str) and isinstance(x.get('sha256'),str):
    path=Path(x['path'])
    if not path.is_absolute():
     if relative_to is None:raise ValueError('relative binding without a declared base')
     path=Path(relative_to)/path
    p=str(guard_path(path));h=x['sha256']
    if p in found and found[p]!=h: raise ValueError('conflicting binding')
    found[p]=h
   for v in x.values(): walk(v)
  elif isinstance(x,list):
   for v in x:walk(v)
 walk(obj);return found
def read_csv(path):
 with guard_path(path).open(newline='') as f:
  reader=csv.DictReader(f)
  if not reader.fieldnames or len(set(reader.fieldnames))!=len(reader.fieldnames): raise ValueError('invalid CSV headers')
  rows=list(reader)
 if any(None in r or any(v is None for v in r.values()) for r in rows): raise ValueError('ragged CSV')
 return rows
def flag(value):
 if value in ('true','1',True):return True
 if value in ('false','0',False):return False
 raise ValueError('invalid bool: '+repr(value))
def canonical(seq): return isinstance(seq,str) and 1<=len(seq)<=48 and set(seq)<=set(AA)

def validate_rows(references,predictions):
 """Bind all 320 R and 2880 M/S rows; no inner join or row filtering."""
 ref={}
 for r in references:
  split=r['split'];i=int(r['row_id']);key=(split,i)
  if split not in SPLITS or not 0<=i<SPLITS[split] or key in ref:raise ValueError('reference identity')
  if int(r['source_row_id'])!=i+(279 if split=='development' else 0):raise ValueError('source row map')
  if r['terminal'] not in ('FREE','AMD') or int(r['class'])!=('FREE','AMD').index(r['terminal']):raise ValueError('terminal/class map')
  if not canonical(r['truth_sequence']) or int(r['target_length'])!=len(r['truth_sequence']):raise ValueError('truth invalid')
  if not r['row_key']:raise ValueError('row_key missing')
  ref[key]=r
 expected_ref={(s,i) for s,n in SPLITS.items() for i in range(n)}
 if set(ref)!=expected_ref or len({r['row_key'] for r in ref.values()})!=320:raise ValueError('320 reference set incomplete')
 pred={};ids=set()
 for r in predictions:
  key=(r['split'],int(r['row_id']));truth=ref.get(key)
  if truth is None:raise ValueError('prediction outside truth')
  for field in ['row_key','source_row_id','terminal','class','truth_sequence']:
   if r[field]!=truth[field]:raise ValueError('prediction truth binding mismatch: '+field)
  mode=r['mode'];draw=int(r['draw_index'])
  if mode=='M':
   if draw!=-1 or r['seed']!='':raise ValueError('M draw/seed')
  elif mode=='S':
   if draw not in range(8) or int(r['seed'])!=SEEDS[draw]:raise ValueError('S draw/seed')
  else:raise ValueError('mode')
  pkey=key+(mode,draw)
  if pkey in pred or not r['sample_id'] or r['sample_id'] in ids:raise ValueError('duplicate prediction')
  ids.add(r['sample_id']);pred[pkey]=r
  finite=flag(r['prediction_finite']);seq=r['prediction_sequence'];tokens=json.loads(r['prediction_tokens48'])
  if r['status'] in COMPLETED:
   if not finite or not isinstance(seq,str) or len(seq)>48 or set(seq)-set(AA):raise ValueError('completed prediction invalid/nonfinite')
   if len(tokens)!=48 or any(type(t)!=int or not 0<=t<=20 for t in tokens):raise ValueError('tokens48 invalid')
   length=int(r['selected_length']);first=int(r['first_pad_index']);cap=flag(r['length48_cap'])
   if int(r['length'])!=len(seq) or length!=len(seq) or not 0<=length<=48:raise ValueError('J length disagrees')
   if tokens!=[TOKEN_AA.index(a) for a in seq]+[20]*(48-length):raise ValueError('tokens/J rendered prediction differs')
   if first!=(length if length<48 else -1) or cap!=(length==48):raise ValueError('PAD/cap differs')
   if r['status']=='empty_sequence' and length!=0:raise ValueError('empty status differs')
   if r['status']=='no_eos_at_limit' and length!=48:raise ValueError('cap status differs')
  elif r['status'] not in ('failed','not_executed','error','partial'):raise ValueError('unknown incomplete status')
 expected={k+(mode,draw) for k in expected_ref for mode,draw in [('M',-1)]+[('S',i) for i in range(8)]}
 if set(pred)!=expected:raise ValueError('2880 prediction rows incomplete')
 return ref,pred

def descriptor_delta(left,right):
 if left['status']!='ok' or right['status']!='ok':
  return {'status':'missing','raw_delta':None,'left_status':left['status'],'right_status':right['status']}
 return {'status':'ok','raw_delta':[b-a for a,b in zip(left['global_raw'],right['global_raw'])]}

def summarize_core(core,ref,pred,scored):
 summaries=[]
 for split,n in SPLITS.items():
  for mode,draw in [('M',-1)]+[('S',j) for j in range(8)]:
   group=[scored.get((split,i,mode,draw)) for i in range(n)]
   stats={}
   for field in ['nonpad_accuracy','whole_exact','levenshtein_raw','levenshtein_normalized','length_signed','length_absolute']:
    values=[g['R_to_prediction'][field] if g and g['R_to_prediction'] is not None else None for g in group]
    if field=='whole_exact':values=[int(v) if v is not None else None for v in values]
    stats[field]=core.summarize_numeric(values,n)
   correct=[g['R_to_prediction']['correct_target_positions'] if g and g['R_to_prediction'] is not None else None for g in group]
   stats['micro_accuracy']={'value':sum(correct)/sum(len(ref[(split,i)]['truth_sequence']) for i in range(n)) if all(x is not None for x in correct) else None,
                            'target_residues':sum(len(ref[(split,i)]['truth_sequence']) for i in range(n)),
                            'correct_target_positions':sum(correct) if all(x is not None for x in correct) else None}
   summaries.append({'split':split,'mode':mode,'draw_index':draw,'seed':None if draw<0 else SEEDS[draw],
                     'expected_rows':n,'computed_rows':sum(g is not None for g in group),
                     'prediction_status_counts':dict(Counter(pred[(split,i,mode,draw)]['status'] for i in range(n))),
                     'length48_cap_count':sum(flag(pred[(split,i,mode,draw)]['length48_cap']) for i in range(n)),
                     'core':stats})
 per_sequence=[]
 for split,n in SPLITS.items():
  for i in range(n):
   m=scored.get((split,i,'M',-1));samples=[scored.get((split,i,'S',j)) for j in range(8)]
   valid=(m is not None and m['prediction_complete_finite'] and all(s is not None and s['prediction_complete_finite'] for s in samples))
   ma=m['R_to_prediction']['nonpad_accuracy'] if m and m['R_to_prediction'] else None
   sa=[s['R_to_prediction']['nonpad_accuracy'] if s and s['R_to_prediction'] else None for s in samples]
   mean_s=sum(sa)/8 if valid and all(v is not None for v in sa) else None
   per_sequence.append({'split':split,'row_id':i,'row_key':ref[(split,i)]['row_key'],'M_accuracy':ma,
    'S8_accuracies':sa,'S8_mean_accuracy':mean_s,'M_minus_S8':ma-mean_s if valid and ma is not None else None,
    'complete_finite_all9':valid})
 dev=[r for r in per_sequence if r['split']=='development']
 # A failure anywhere in the fixed 320x9 forward contract invalidates the gate;
 # descriptor failures never enter this predicate.
 all_complete=all(r['complete_finite_all9'] for r in per_sequence)
 gate={'status':'NOT_EVALUABLE','threshold':0.05,'n_dev_sequences':41,'posterior_draws':8,
       'M_sequence_equal_accuracy':None,'S8_sequence_equal_accuracy':None,'M_minus_S8':None,
       'crosses_investment_threshold':None,'all320x9_complete_finite':all_complete,
       'meaning':'Prespecified investment screen only; not significance, causal proof, independent repeats or training authorization.',
       'descriptor_missing_affects_this_gate':False}
 if all_complete and all(r['M_minus_S8'] is not None for r in dev):
  ma_exact=sum((Fraction(scored[('development',i,'M',-1)]['R_to_prediction']['correct_target_positions'],len(ref[('development',i)]['truth_sequence'])) for i in range(41)),Fraction(0))/41
  sa_exact=sum((sum((Fraction(scored[('development',i,'S',j)]['R_to_prediction']['correct_target_positions'],len(ref[('development',i)]['truth_sequence'])) for j in range(8)),Fraction(0))/8 for i in range(41)),Fraction(0))/41
  delta_exact=ma_exact-sa_exact
  gate.update(status='EVALUABLE',M_sequence_equal_accuracy=float(ma_exact),S8_sequence_equal_accuracy=float(sa_exact),
   M_minus_S8=float(delta_exact),crosses_investment_threshold=delta_exact>=Fraction(1,20),
   difference_exact_fraction={'numerator':delta_exact.numerator,'denominator':delta_exact.denominator},
   comparison_rule='Exact rational counts/target lengths; compare against1/20, no floating tolerance.')
 s8=[]
 for split,n in SPLITS.items():
  metrics={}
  for field in ['nonpad_accuracy','whole_exact','levenshtein_raw','levenshtein_normalized','length_signed','length_absolute']:
   means=[]
   for i in range(n):
    samples=[scored.get((split,i,'S',j)) for j in range(8)]
    values=[s['R_to_prediction'][field] if s and s['prediction_complete_finite'] and s['R_to_prediction'] else None for s in samples]
    means.append(sum(values)/8 if all(v is not None for v in values) else None)
   metrics[field]=core.summarize_numeric(means,n)
  samples=[scored.get((split,i,'S',j)) for i in range(n) for j in range(8)]
  correct=[s['R_to_prediction']['correct_target_positions'] if s and s['prediction_complete_finite'] and s['R_to_prediction'] else None for s in samples]
  denominator=8*sum(len(ref[(split,i)]['truth_sequence']) for i in range(n))
  metrics['micro_accuracy']={'value':sum(correct)/denominator if all(v is not None for v in correct) else None,
                            'target_residues_across8':denominator,'all8_required_per_sequence':True}
  s8.append({'split':split,'expected_sequences':n,'draws':8,'aggregation':'For each original sequence average all8 draws, then equal-weight sequences; missing draw gives a missing per-sequence value, never average only surviving seeds.',
             'metrics':metrics})
 return summaries,per_sequence,gate,s8

def stratified_core(core,ref,pred,scored):
 """All/FREE/AMD summaries; supported-AA macro is distinct from seq-equal accuracy."""
 result=[]
 for split,n in SPLITS.items():
  for terminal in ['ALL','FREE','AMD']:
   ids=[i for i in range(n) if terminal=='ALL' or ref[(split,i)]['terminal']==terminal]
   support=Counter(a for i in ids for a in ref[(split,i)]['truth_sequence'])
   supported=[a for a in AA if support[a]]
   for label,mode,draws in [('M','M',[-1])]+[(f'S{j}','S',[j]) for j in range(8)]+[('S8','S',list(range(8)))]:
    keys=[(split,i,mode,j) for i in ids for j in draws]
    complete=all(k in scored and scored[k]['prediction_complete_finite'] and scored[k]['R_to_prediction'] is not None for k in keys)
    correct=Counter()
    if complete:
     for k in keys:
      for a,b in zip(ref[k[:2]]['truth_sequence'],pred[k]['prediction_sequence']):
       if a==b:correct[a]+=1
    exposure=len(draws)
    recalls={a:correct[a]/(exposure*support[a]) if complete and support[a] else None for a in AA}
    macro=sum(recalls[a] for a in supported)/len(supported) if complete and supported else None
    per_sequence_metrics={}
    for field in ['nonpad_accuracy','whole_exact','levenshtein_raw','levenshtein_normalized','length_signed','length_absolute']:
     values=[]
     for i in ids:
      samples=[scored.get((split,i,mode,j)) for j in draws]
      vals=[s['R_to_prediction'][field] if s and s['prediction_complete_finite'] and s['R_to_prediction'] is not None else None for s in samples]
      values.append(sum(vals)/exposure if all(v is not None for v in vals) else None)
     per_sequence_metrics[field]=core.summarize_numeric(values,len(ids))
     if field=='nonpad_accuracy':per_sequence_metrics['nonpad_error']=core.summarize_numeric([1-v if v is not None else None for v in values],len(ids))
    result.append({'split':split,'terminal':terminal,'mode_group':label,'draw_indices':draws,
     'expected_sequences':len(ids),'expected_predictions':len(keys),'all_predictions_complete_finite':complete,
     'per_sequence_then_equal_sequence_means':per_sequence_metrics,
     'supported_AA_macro':{'value':macro,'definition':'Equal mean recall over target-supported AA classes; S8 pools correct across8 draws and divides each class by8 times original target support. Not per-sequence accuracy.',
      'support_per_original_sequence_set':{a:support[a] for a in AA},'exposure_multiplier':exposure,
      'correct_across_draws':{a:correct[a] for a in AA} if complete else None,'recall_by_AA':recalls,'supported_AA':supported},
     'micro_accuracy':sum(correct.values())/(exposure*sum(support.values())) if complete and support else None,
     'prediction_status_counts':dict(Counter(pred[k]['status'] for k in keys)),
     'native_J48_cap_count':sum(flag(pred[k]['length48_cap']) for k in keys),
     'PAD_terminated_count':sum(pred[k]['status'] in COMPLETED and int(pred[k]['first_pad_index'])>=0 for k in keys)})
 return result

def run(config_path,approval_path,postflight_path,out):
 cfg=load(config_path);approval=load(approval_path);post=load(postflight_path)
 if cfg['scope_id']!=SCOPE or cfg['cap_seconds']!=300 or cfg['max_unique_descriptors']!=3200:raise ValueError('fixed scoring contract')
 if cfg['seeds']!=SEEDS or cfg['splits']!=SPLITS:raise ValueError('fixed seeds/splits')
 if approval['scope_id']!=SCOPE or approval['status']!='PASS':raise ValueError('scoring preaudit not PASS')
 if approval['scoring_config_sha256']!=sha(config_path) or approval['scoring_script_sha256']!=sha(__file__):raise ValueError('preaudit exact code/config differs')
 if post['status']!='PASS' or post['scope_id']!=cfg['model_scope_id']:raise ValueError('model postflight not PASS/exact scope')
 if sha(cfg['model_contract']['path'])!=cfg['model_contract']['sha256']:raise ValueError('model contract changed')
 root=guard_path(cfg['input_root']);out=guard_path(out)
 if out.exists():raise ValueError('single-execution output already exists; no overwrite/retry')
 if out!=guard_path(cfg['output_root']):raise ValueError('fixed output root')
 # Only the approved model output manifest may supply sequence-file bindings.
 manifest_path=root/'manifest.json';post_bind=all_bindings(post)
 if post_bind.get(str(guard_path(cfg['model_contract']['path'])))!=cfg['model_contract']['sha256']:raise ValueError('model contract not bound by postflight')
 if post_bind.get(str(manifest_path))!=sha(manifest_path):raise ValueError('model output manifest not bound by postflight')
 manifest=load(manifest_path);bound=all_bindings(manifest,root)
 reference_path=root/'reference_rows.csv';prediction_path=root/'reconstructions.csv'
 for p in [reference_path,prediction_path]:
  if bound.get(str(p))!=sha(p):raise ValueError('model output CSV not manifest-bound: '+str(p))
 source_records={}
 for name,source in cfg['sources'].items():
  if sha(source['path'])!=source['sha256']:raise ValueError('source binding: '+name)
  source_records[name]=source
 if importlib.metadata.version('rdkit')!=cfg['rdkit_distribution_version']:raise ValueError('RDKit version differs')
 core=load_module('rms_metric_core',**{'path':cfg['sources']['metric_core']['path'],'expected':cfg['sources']['metric_core']['sha256']})
 metrics=load_module('rms_frozen_metrics',cfg['sources']['frozen_metrics']['path'],cfg['sources']['frozen_metrics']['sha256'])
 molecular=load_module('rms_molecular_representation',cfg['sources']['molecular_representation']['path'],cfg['sources']['molecular_representation']['sha256'])
 if molecular.GLOBAL_NAMES!=GLOBAL_NAMES:raise ValueError('original 9global field order differs')
 references=read_csv(reference_path);predictions=read_csv(prediction_path)
 ref,pred=validate_rows(references,predictions)
 out.mkdir(parents=True);started=datetime.now(timezone.utc).isoformat();scored={};cache={};phase='core';capped=False;error=None
 initial={'config':sha(config_path),'approval':sha(approval_path),'postflight':sha(postflight_path),
          'reference_rows':sha(reference_path),'reconstructions':sha(prediction_path),'model_manifest':sha(manifest_path)}
 deadline=PROCESS_START+300
 def check():
  if time.monotonic()>=deadline:raise BudgetStop('300 second limit')
 oldhandler=signal.signal(signal.SIGALRM,stop_budget)
 signal.setitimer(signal.ITIMER_REAL,max(0.001,deadline-time.monotonic()))
 try:
  # Accuracy/edit/length first, preserving the full fixed denominator independently of descriptors.
  for split,n in SPLITS.items():
   for i in range(n):
    truth=ref[(split,i)]['truth_sequence'];mr=pred[(split,i,'M',-1)]
    for mode,draw in [('M',-1)]+[('S',j) for j in range(8)]:
     check();key=(split,i,mode,draw);r=pred[key]
     complete=r['status'] in COMPLETED and flag(r['prediction_finite'])
     value=core.pair_core(truth,r['prediction_sequence']) if complete else None
     ms=core.pair_core(mr['prediction_sequence'],r['prediction_sequence']) if mode=='S' and complete and mr['status'] in COMPLETED and flag(mr['prediction_finite']) else None
     scored[key]={'sample_id':r['sample_id'],'row_key':r['row_key'],'split':split,'row_id':i,
      'source_row_id':int(r['source_row_id']),'terminal':r['terminal'],'mode':mode,'draw_index':draw,
      'seed':None if draw<0 else SEEDS[draw],'status':r['status'],'prediction_complete_finite':complete,
      'R_to_prediction':value,'M_to_S':ms,'selected_length':int(r['selected_length']) if complete else None,
      'first_pad_index':int(r['first_pad_index']) if complete else None,'length48_cap':flag(r['length48_cap']),
      'PAD_terminated':complete and int(r['first_pad_index'])>=0,
      'feature_R_to_prediction':None,'feature_M_to_S':None,'descriptor_R_to_prediction':None,'descriptor_M_to_S':None}
  phase='fixed421'
  for key,row in scored.items():
   check();split,i,mode,draw=key;r=pred[key];mr=pred[(split,i,'M',-1)]
   row['feature_R_to_prediction']=core.feature_delta(ref[(split,i)]['truth_sequence'],r['prediction_sequence'],metrics.feature_vector) if row['prediction_complete_finite'] else {'valid':False,'status':'prediction_incomplete','delta421':None,'blocks':None}
   if mode=='S': row['feature_M_to_S']=core.feature_delta(mr['prediction_sequence'],r['prediction_sequence'],metrics.feature_vector) if row['prediction_complete_finite'] and mr['status'] in COMPLETED and flag(mr['prediction_finite']) else {'valid':False,'status':'prediction_incomplete','delta421':None,'blocks':None}
  phase='descriptor_cache'
  # Prelocked deterministic order: R, M, then S0..S7; train then development, row ID ascending.
  keys=[];seen=set()
  for mode,draw in [('R',-1),('M',-1)]+[('S',j) for j in range(8)]:
   for split,n in SPLITS.items():
    for i in range(n):
     r=ref[(split,i)] if mode=='R' else pred[(split,i,mode,draw)]
     sequence=r['truth_sequence'] if mode=='R' else r['prediction_sequence'];key=(sequence,r['terminal'])
     if key not in seen:seen.add(key);keys.append(key)
  if len(keys)>3200:raise ValueError('theoretical unique descriptor bound exceeded')
  for seq,terminal in keys:
   check();key=(seq,terminal)
   if not canonical(seq):
    cache[key]={'status':'not_applicable_empty_or_invalid','global_raw':None,'error':None};continue
   try:
    values=molecular.representation(seq,terminal)['global_raw']
    if len(values)!=9 or not all(math.isfinite(v) for v in values):raise ValueError('invalid 9global return')
    cache[key]={'status':'ok','global_raw':values,'error':None}
   except Exception as exc:
    cache[key]={'status':'rdkit_or_representation_failed','global_raw':None,'error':type(exc).__name__+': '+str(exc)}
  phase='descriptor_deltas'
  for key,row in scored.items():
   check();split,i,mode,draw=key;r=pred[key];terminal=r['terminal'];mr=pred[(split,i,'M',-1)]
   row['descriptor_R_to_prediction']=descriptor_delta(cache[(ref[(split,i)]['truth_sequence'],terminal)],cache[(r['prediction_sequence'],terminal)])
   if mode=='S':row['descriptor_M_to_S']=descriptor_delta(cache[(mr['prediction_sequence'],terminal)],cache[(r['prediction_sequence'],terminal)])
  phase='completed'
 except BudgetStop as exc:capped=True;error=str(exc)
 except Exception as exc:error=type(exc).__name__+': '+str(exc)
 finally:
  signal.setitimer(signal.ITIMER_REAL,0);signal.signal(signal.SIGALRM,oldhandler)
 # Export bookkeeping after stopping compute. Never resume skipped property calculations.
 groups,per_sequence,gate,s8=summarize_core(core,ref,pred,scored)
 with (out/'ROW_METRICS.jsonl').open('w') as f:
  for key in sorted(pred):
   row=scored.get(key,{'sample_id':pred[key]['sample_id'],'split':key[0],'row_id':key[1],'mode':key[2],
    'draw_index':key[3],'status':'scoring_not_executed','prediction_complete_finite':False})
   f.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')
 with (out/'DESCRIPTOR_CACHE.jsonl').open('w') as f:
  for (seq,terminal),value in cache.items():
   f.write(json.dumps({'sequence_sha256':hashlib.sha256(seq.encode()).hexdigest(),'sequence':seq,'terminal':terminal,**value},ensure_ascii=False,allow_nan=False)+'\n')
 dump(out/'PER_SEQUENCE_S8.json',per_sequence);dump(out/'CORE_SUMMARIES.json',groups);dump(out/'S8_AGGREGATE_SUMMARIES.json',s8);dump(out/'INVESTMENT_SCREEN.json',gate)
 dump(out/'STRATIFIED_CORE_SUMMARIES.json',stratified_core(core,ref,pred,scored))
 coverage=[]
 for split,n in SPLITS.items():
  for terminal in ['ALL','FREE','AMD']:
   ids=[i for i in range(n) if terminal=='ALL' or ref[(split,i)]['terminal']==terminal];den=len(ids)
   for mode,draw in [('M',-1)]+[('S',j) for j in range(8)]:
    for pair in ['R_to_prediction']+(['M_to_S'] if mode=='S' else []):
     rows=[scored.get((split,i,mode,draw),{}) for i in ids]
     for kind in ['feature','descriptor']:
      vals=[r.get(kind+'_'+pair) for r in rows]
      statuses=Counter(v.get('status','unknown') if v else 'not_computed' for v in vals)
      entry={'split':split,'terminal':terminal,'mode':mode,'draw_index':draw,'pair':pair,'kind':kind,'total_rows':den,'status_counts':dict(statuses)}
      if kind=='descriptor':
       entry['raw_delta_summaries']={name:core.summarize_numeric([v['raw_delta'][j] if v and v.get('status')=='ok' else None for v in vals],den) for j,name in enumerate(GLOBAL_NAMES)}
      else:
       entry['block_summaries']={block:{metric:core.summarize_numeric([v['blocks'][block][metric] if v and v.get('valid') and v.get('blocks') else None for v in vals],den) for metric in ['signed_sum','L1','squared_L2']} for block in ['length1','aa20','dipeptide400']}
      coverage.append(entry)
 dump(out/'PROPERTY_COVERAGE_AND_SUMMARIES.json',{'global_names':GLOBAL_NAMES,
  'units':'Raw RDKit 2023.9.5 Descriptors outputs with no scale/rounding. Source does not declare a unit table; no new units are assigned here.',
  'no_activity_prediction':True,'null_not_zero':True,'accuracy_denominator_never_filtered_by_property':True,
  'rows':coverage,'cache_status_counts':dict(Counter(v['status'] for v in cache.values()))})
 unchanged=initial=={'config':sha(config_path),'approval':sha(approval_path),'postflight':sha(postflight_path),
                    'reference_rows':sha(reference_path),'reconstructions':sha(prediction_path),'model_manifest':sha(manifest_path)}
 sources_unchanged=all(sha(v['path'])==v['sha256'] for v in cfg['sources'].values())
 elapsed=time.monotonic()-PROCESS_START
 status={'scope_id':SCOPE,'state':'capped' if capped else ('failed' if error else 'completed'),
  'phase_at_stop':phase,'error':error,'started_utc':started,'finished_utc':datetime.now(timezone.utc).isoformat(),
  'wall_seconds_before_final_exports':elapsed,'cap_seconds':300,'overrun_seconds':max(0,elapsed-300),
  'post_compute_bookkeeping_allowed':True,'rdkit_distribution_version':importlib.metadata.version('rdkit'),
  'reference_rows':320,'requested_prediction_rows':2880,'scored_core_rows':len(scored),
  'unique_descriptor_keys_computed_or_classified':len(cache),'sources':source_records,
  'input_sha256':initial,'all_inputs_unchanged':unchanged,'all_sources_unchanged':sources_unchanged,'model_calls':0,'training_updates':0,
  'random_draws':0,'test_access':False,'raw_cohort_access':False,'new_metric_scope':'paired R/M/S descriptive sequence metrics only',
  'descriptor_missing_can_coexist_with_valid_accuracy_screen':True}
 # STATUS is explicitly provisional until manifest/result IO and the post-write
 # budget check below. The outer watchdog covers final bookkeeping/exit.
 status['computation_state']=status['state']
 if status['state']=='completed':status['state']='finalizing'
 status['terminal_state_source']='CLOSING_LEDGER.json'
 dump(out/'STATUS.json',status)
 files=[]
 for p in sorted(out.iterdir()):
  if p.is_file():files.append({'path':str(p),'bytes':p.stat().st_size,'sha256':sha(p)})
 dump(out/'MANIFEST.json',{'scope_id':SCOPE,'files':files,'status_sha256':sha(out/'STATUS.json')})
 if not unchanged or not sources_unchanged:raise ValueError('inputs/source changed during scoring')
 after_manifest=time.monotonic()-PROCESS_START
 terminal='capped' if capped or after_manifest>=300 else ('failed' if error else 'completed')
 closing={'scope_id':SCOPE,'state':terminal,'manifest_sha256':sha(out/'MANIFEST.json'),
  'wall_seconds_after_manifest_before_closing':after_manifest,'cap_seconds':300,
  'budget_applies_through_manifest_and_closing_write':True,
  'model_calls':0,'primary_gate_status':gate['status'],'descriptor_or_feature_missing_does_not_change_core_gate':True,
  'not_self_included_in_manifest':True,'outer_watchdog_required_seconds':360}
 dump(out/'CLOSING_LEDGER.json',closing)
 after_closing=time.monotonic()-PROCESS_START
 if after_closing>=300:
  closing.update(state='capped',wall_seconds_observed_after_initial_closing_write=after_closing,
                 final_bookkeeping_after_cap=True)
  dump(out/'CLOSING_LEDGER.json',closing);terminal='capped'
 print(json.dumps({'state':terminal,'wall_seconds_after_closing_write':after_closing,
  'gate_status':gate['status'],'output_root':str(out)}))
 return 0 if terminal=='completed' else 2

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--config',required=True);ap.add_argument('--approval',required=True)
 ap.add_argument('--model-postflight',required=True);ap.add_argument('--out',required=True)
 a=ap.parse_args();sys.exit(run(a.config,a.approval,a.model_postflight,a.out))
