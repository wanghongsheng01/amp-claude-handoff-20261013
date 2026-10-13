"""Synthetic rows only; no model, RDKit molecule, cohort or natural predictions."""
import time
start=time.monotonic()
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
HERE=Path(__file__).resolve().parent
def module(name,path):
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
sc=module('score_rms_synthetic',HERE/'score_rms_v2.py')
core=module('core_synthetic',HERE/'rms_metric_core.py')
checks=[]
def check(name,condition):
 assert condition,name;checks.append(name)
def rejected(name,fn):
 try:fn()
 except ValueError:checks.append(name);return
 raise AssertionError(name)
refs=[];rows=[]
for split,n in sc.SPLITS.items():
 for i in range(n):
  terminal='FREE' if i<(126 if split=='train' else 18) else 'AMD'
  ref={'split':split,'row_id':str(i),'source_row_id':str(i+(279 if split=='development' else 0)),
       'terminal':terminal,'class':str(('FREE','AMD').index(terminal)),
       'truth_sequence':'ARND','target_length':'4','row_key':f'{split}:{i}'}
  refs.append(ref)
  for mode,draw in [('M',-1)]+[('S',j) for j in range(8)]:
   rows.append({**ref,'sample_id':f'{split}:{i}:{mode}:{draw}','mode':mode,'draw_index':str(draw),
    'seed':'' if mode=='M' else str(sc.SEEDS[draw]),'prediction_sequence':'ARND','length':'4',
    'status':'ok','selected_length':'4','first_pad_index':'4','length48_cap':'false',
    'prediction_tokens48':json.dumps([0,1,2,3]+[20]*44),'prediction_finite':'true'})
ref,pred=sc.validate_rows(refs,rows)
check('320 reference / 2880 exact slots',len(ref)==320 and len(pred)==2880)
def scored(p):
 return {k:{'R_to_prediction':core.pair_core(ref[k[:2]]['truth_sequence'],r['prediction_sequence']) if r['status'] in sc.COMPLETED else None,
            'prediction_complete_finite':r['status'] in sc.COMPLETED and sc.flag(r['prediction_finite'])} for k,r in p.items()}
groups,seq,gate,s8=sc.summarize_core(core,ref,pred,scored(pred))
check('perfect M and S zero screen',gate['M_minus_S8']==0 and gate['crosses_investment_threshold'] is False)
check('whole_exact numeric counts',all(g['core']['whole_exact']['mean']==1 for g in groups))
check('S8 seq equal / micro retained',s8[1]['metrics']['nonpad_accuracy']['mean']==1 and s8[1]['metrics']['micro_accuracy']['value']==1)
bad=copy.deepcopy(rows);bad[0]['prediction_tokens48']=json.dumps([0,14,11,2]+[20]*44)
rejected('wrong sorted token order rejected',lambda:sc.validate_rows(refs,bad))
rejected('missing S not silently dropped',lambda:sc.validate_rows(refs,rows[:-1]))
rejected('duplicate slot rejected',lambda:sc.validate_rows(refs,rows+[rows[0]]))
bad=copy.deepcopy(rows);bad[0]['truth_sequence']='AAAA'
rejected('truth mismatch rejected',lambda:sc.validate_rows(refs,bad))
bad=copy.deepcopy(rows);bad[1]['seed']='9999'
rejected('changed S seed rejected',lambda:sc.validate_rows(refs,bad))
bad=copy.deepcopy(rows);bad[0]['prediction_finite']='false'
rejected('complete nonfinite prediction rejected',lambda:sc.validate_rows(refs,bad))
empty=copy.deepcopy(rows)
empty[1].update(prediction_sequence='',length='0',selected_length='0',first_pad_index='0',status='empty_sequence',prediction_tokens48=json.dumps([20]*48))
_,ep=sc.validate_rows(refs,empty);_,_,eg,_=sc.summarize_core(core,ref,ep,scored(ep))
check('complete empty remains eligible denominator',eg['status']=='EVALUABLE' and ep[('train',0,'S',0)]['prediction_sequence']=='')
cap=copy.deepcopy(rows)
cap[1].update(prediction_sequence='A'*48,length='48',selected_length='48',first_pad_index='-1',length48_cap='true',status='no_eos_at_limit',prediction_tokens48=json.dumps([0]*48))
_,cp=sc.validate_rows(refs,cap);_,_,cg,_=sc.summarize_core(core,ref,cp,scored(cp))
check('complete J48 remains eligible',cg['status']=='EVALUABLE')
failed=copy.deepcopy(rows)
failed[-1].update(status='failed',prediction_finite='false',prediction_sequence='',prediction_tokens48='[]')
_,fp=sc.validate_rows(refs,failed);_,fs,fg,f8=sc.summarize_core(core,ref,fp,scored(fp))
check('failed draw NE not survivor average',fg['status']=='NOT_EVALUABLE' and fs[-1]['S8_mean_accuracy'] is None)
check('failure retains dev41 denominator',f8[1]['metrics']['nonpad_accuracy']['total']==41 and f8[1]['metrics']['nonpad_accuracy']['n_missing']==1)
changed=copy.deepcopy(rows)
for r in changed:
 if r['mode']=='S':r.update(prediction_sequence='AAAA',prediction_tokens48=json.dumps([0]*4+[20]*44))
_,sp=sc.validate_rows(refs,changed);_,_,sg,_=sc.summarize_core(core,ref,sp,scored(sp))
check('frozen difference threshold direction',sg['M_minus_S8']==.75 and sg['crosses_investment_threshold'] is True)
check('descriptor missing independent of core screen',sc.descriptor_delta({'status':'failed','global_raw':None},{'status':'ok','global_raw':[1]*9})['raw_delta'] is None and sg['status']=='EVALUABLE')
check('descriptor delta direction',sc.descriptor_delta({'status':'ok','global_raw':[1]*9},{'status':'ok','global_raw':[3]*9})['raw_delta']==[2]*9)
boundary_refs=copy.deepcopy(refs);boundary_rows=copy.deepcopy(rows)
for r in boundary_refs:r.update(truth_sequence='A'*20,target_length='20')
for r in boundary_rows:
 sequence='A'*20 if r['mode']=='M' else 'A'*19+'R'
 r.update(truth_sequence='A'*20,target_length='20',prediction_sequence=sequence,length='20',selected_length='20',first_pad_index='20',prediction_tokens48=json.dumps([sc.TOKEN_AA.index(a) for a in sequence]+[20]*28))
br,bp=sc.validate_rows(boundary_refs,boundary_rows)
bs={k:{'R_to_prediction':core.pair_core(br[k[:2]]['truth_sequence'],r['prediction_sequence']),'prediction_complete_finite':True} for k,r in bp.items()}
_,_,bg,_=sc.summarize_core(core,br,bp,bs)
check('exact5pp inclusive boundary',bg['crosses_investment_threshold'] is True and bg['difference_exact_fraction']=={'numerator':1,'denominator':20})
strata=sc.stratified_core(core,ref,pred,scored(pred))
check('all split terminal mode strata',len(strata)==60)
devfree=next(g for g in strata if g['split']=='development' and g['terminal']=='FREE' and g['mode_group']=='S8')
check('terminal18 S8 denominator and exposure',devfree['expected_sequences']==18 and devfree['expected_predictions']==144 and devfree['supported_AA_macro']['exposure_multiplier']==8)
check('perfect supportedAA macro',devfree['supported_AA_macro']['value']==1 and devfree['per_sequence_then_equal_sequence_means']['nonpad_error']['mean']==0)
badstrata=sc.stratified_core(core,ref,fp,scored(fp))
devamd=next(g for g in badstrata if g['split']=='development' and g['terminal']=='AMD' and g['mode_group']=='S8')
check('faileddraw macro missing not survivor macro',devamd['supported_AA_macro']['value'] is None and devamd['expected_sequences']==23)
print(json.dumps({'status':'PASS','check_count':len(checks),'checks':checks,'wall_seconds':time.monotonic()-start,
 'model_calls':0,'natural_data_reads':0,'RDKit_calls':0,'random_draws':0,
 'script_sha256':hashlib.sha256((HERE/'score_rms_v2.py').read_bytes()).hexdigest()},indent=2))
