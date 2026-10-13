"""Independent descriptive arithmetic only; no model/optimizer imported."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[k]='1'
from pathlib import Path
import json,hashlib,collections,math,datetime
import numpy as np
P=Path(__file__).resolve().parent;R=P.parent;S=R.parent/'censored_likelihood_v1'
read=lambda p:json.loads(p.read_text());rows=lambda p:[json.loads(s)for s in p.read_text().splitlines()];sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
checks=[]
def check(n,c):checks.append({'name':n,'pass':bool(c)})
def close(a,b):return abs(float(a)-float(b))<1e-9*max(1.,abs(float(a)),abs(float(b)))
def compare(name,own,reported):check(name,all(k in reported and(close(v,reported[k])if isinstance(v,(float,int))else v==reported[k])for k,v in own.items()))
def weights(rs):
 gs=collections.defaultdict(lambda:collections.Counter())
 for r in rs:gs[r['source_family_group']][r['peptide_id']]+=1
 return np.asarray([1/len(gs)/len(gs[r['source_family_group']])/gs[r['source_family_group']][r['peptide_id']]for r in rs])
AA='ACDEFGHIKLMNPQRSTVWY'
def feature(rs):return np.asarray([[r['sequence'].count(a)/len(r['sequence'])for a in AA]+[math.log(len(r['sequence']))]for r in rs])
data=rows(S/'DATA.jsonl');lookup={r['activity_id']:r for r in data};pred={(r['model'],r['activity_id']):r['prediction_log10']for r in rows(S/'run/PREDICTIONS.jsonl')};summary=read(R/'SUMMARY.json');gr={(r['unit'],r['group'],r['model']):r for r in read(R/'GROUP_ERRORS.json')};sup={r['activity_id']:r for r in read(R/'FEATURE_SUPPORT.json')};peps={(r['unit'],r['peptide_id']):r for r in read(R/'PEPTIDE_REPEATS.json')};selection=read(R/'POINT_CENSOR_SELECTION.json');conditions={(r['unit'],r['field']):r for r in read(R/'OBSERVED_CONDITIONS.json')};arms=['point_mean','point_ridge','endpoint_ridge','tail_ridge'];ownresults={}
for unit in ['ug_ml_native','uM_native']:
 rs=[r for r in data if r['native_unit_group']==unit and r['label_kind']=='point'];w=weights(rs);y=np.array([r['target_log10']for r in rs]);groups=sorted({r['source_family_group']for r in rs});means=[];variances=[];ownbygroup={};c=np.array([pred['point_mean',r['activity_id']]for r in rs]);outside=[];maxz=[]
 for g in groups:
  rr=[r for r in rs if r['source_family_group']==g];ww=weights(rr);yy=np.array([r['target_log10']for r in rr]);gm=ww@yy;gv=ww@((yy-gm)**2);means.append(gm);variances.append(gv);ownbygroup[g]={}
  receipt=read(S/'run'/(unit+'__'+g+'.json'));train=[lookup[i]for i in receipt['fold']['train']];x=feature(train);xv=feature(rr);mu=np.asarray(receipt['mean']);sd=np.asarray(receipt['scale']);mask=(xv<x.min(axis=0)-1e-12)|(xv>x.max(axis=0)+1e-12);z=(xv-mu)/sd
  for i,r in enumerate(rr):
   names=[AA[j]if j<20 else'loglength'for j in np.flatnonzero(mask[i])];out={'outside_train_feature_count':len(names),'outside_feature_names':names,'max_abs_standardized_feature':float(np.abs(z[i]).max())};compare(unit+'_'+str(r['activity_id'])+'_support',out,sup[r['activity_id']]);outside.append((r,len(names)));maxz.append(out['max_abs_standardized_feature'])
  for arm in arms:
   p=np.array([pred[arm,r['activity_id']]for r in rr]);e=p-yy;bias=ww@e;centered=ww@(e-bias)**2;mse=ww@(e*e)
   own={'point_rows':len(rr),'peptide_IDs':len({r['peptide_id']for r in rr}),'exact_sequences':len({r['sequence']for r in rr}),'families':len({r['sequence_family']for r in rr}),'label_mean_log10':float(gm),'label_min_log10':float(yy.min()),'label_max_log10':float(yy.max()),'label_within_group_variance':float(gv),'training_point_mean_log10':receipt['point_mean'],'heldout_minus_training_point_mean':float(gm-receipt['point_mean']),'prediction_mean_log10':float(ww@p),'bias':float(bias),'bias_squared':float(bias*bias),'centered_error_MSE':float(centered),'MSE':float(mse),'MAE':float(ww@np.abs(e)),'any_point_outside_train_feature_box':bool(mask.any())}
   compare(unit+'_'+g+'_'+arm+'_all_group_diagnostics',own,gr[unit,g,arm]);check(unit+'_'+g+'_'+arm+'_bias_identity',close(mse,bias*bias+centered));ownbygroup[g][arm]=own
 grand=float(w@y);between=float(np.mean((np.array(means)-grand)**2));within=float(np.mean(variances));total=float(w@(y-grand)**2)
 ownvar={'total_label_variance':total,'between_source_group_mean_variance':between,'within_source_group_variance':within,'between_fraction':between/total};compare(unit+'_label_variance',ownvar,summary[unit]['variance_decomposition']);check(unit+'_label_variance_identity',close(total,between+within))
 ownmodels={}
 for arm in arms:
  p=np.array([pred[arm,r['activity_id']]for r in rs]);mse=float(w@(p-y)**2);bias=float(np.mean([ownbygroup[g][arm]['bias_squared']for g in groups]));centered=float(np.mean([ownbygroup[g][arm]['centered_error_MSE']for g in groups]));a=p-c;energy=float(w@(a*a));align=float(w@(a*(y-c)));delta=float(w@((p-y)**2-(c-y)**2))
  md={'MSE':mse,'MAE':float(w@np.abs(p-y)),'mean_group_bias_squared':bias,'mean_centered_error_MSE':centered,'bias_squared_fraction_of_MSE':bias/mse,'versus_mean_adjustment_energy':energy,'versus_mean_adjustment_label_alignment':align,'versus_mean_delta_MSE':delta};compare(unit+'_'+arm+'_model_decomposition',md,summary[unit]['model_error_decomposition'][arm]);check(unit+'_'+arm+'_adjustment_identity',close(delta,energy-2*align));ownmodels[arm]=md
 for pid in sorted({r['peptide_id']for r in rs}):
  rr=[r for r in rs if r['peptide_id']==pid];ys=[r['target_log10']for r in rr];cc={tuple(str(r.get(k))for k in ['target_raw','medium','cfu','ph','ionic_strength','salt','note'])for r in rr};compare(unit+'_'+str(pid)+'_repeat',{'point_rows':len(rr),'distinct_reported_point_values':len(set(ys)),'log10_range':max(ys)-min(ys),'ratio_max_min':10**(max(ys)-min(ys)),'distinct_observed_condition_tuples':len(cc),'group':rr[0]['source_family_group'],'activity_ids':[r['activity_id']for r in rr]},peps[unit,pid])
 compare(unit+'_counts_and_span',{'point_rows':len(rs),'peptide_IDs':len({r['peptide_id']for r in rs}),'exact_sequences':len({r['sequence']for r in rs}),'families':len({r['sequence_family']for r in rs}),'source_groups':len(groups),'single_point_row_groups':sum(sum(r['source_family_group']==g for r in rs)==1 for g in groups),'one_peptide_groups':sum(len({r['peptide_id']for r in rs if r['source_family_group']==g})==1 for g in groups),'group_mean_min':min(means),'group_mean_max':max(means),'group_mean_log10_span':max(means)-min(means),'group_mean_geometric_ratio':10**(max(means)-min(means)),'grand_group_weighted_mean':grand},summary[unit])
 compare(unit+'_support_summary',{'rows_outside_any_training_feature_range':sum(n>0 for r,n in outside),'IDs_outside':len({r['peptide_id']for r,n in outside if n>0}),'groups_outside':len({r['source_family_group']for r,n in outside if n>0}),'max_abs_z':max(maxz)},summary[unit]['feature_box_support'])
 for field in ['target_raw','medium','cfu','ph','ionic_strength','salt']:
  missing=sum(r.get(field)is None or isinstance(r.get(field),str)and not r[field].strip()for r in rs);check(unit+'_'+field+'_normalized_missingness',conditions[unit,field]['missing_rows']==missing)
 ownresults[unit]={'variance_decomposition':ownvar,'model_error_decomposition':ownmodels,'groups':ownbygroup}
for r in selection:
 rr=[d for d in data if d['native_unit_group']==r['unit']and d['source_family_group']==r['group']and d['label_kind']==r['kind']]
 compare(r['unit']+'_'+r['group']+'_'+r['kind']+'_selection',{'rows':len(rr),'IDs':len({d['peptide_id']for d in rr}),'families':len({d['sequence_family']for d in rr}),'raw_threshold_or_point_min':min(d['lower_raw_unit']for d in rr),'raw_threshold_or_point_max':max(d['lower_raw_unit']for d in rr)},r)
bind=read(R/'SOURCE_BINDINGS.json');check('source_bindings_unchanged',all(sha(R.parent/n)==h for n,h in bind['source_sha256'].items()));check('diagnostic_code_bound',bind['diagnostic_code_sha256']==sha(R/'diagnose.py'))
(P/'RECOMPUTED_DIAGNOSTICS.json').write_text(json.dumps(ownresults,indent=2)+'\n')
report={'status':'PASS'if all(c['pass']for c in checks)else'FAIL','checks':checks,'scope':'Independent vectorized arithmetic from original frozen labels,predictions and states; no fit or model import. Includes source-group bias+centered MSE and adjustment energy/alignment identities, counts, every repeat/support/selection row and normalized missingness.','sha256':{n:sha(R/n)for n in ['SUMMARY.json','GROUP_ERRORS.json','PEPTIDE_REPEATS.json','FEATURE_SUPPORT.json','POINT_CENSOR_SELECTION.json','OBSERVED_CONDITIONS.json','SOURCE_BINDINGS.json','diagnose.py']},'limitations':['Descriptive weighted identities do not establish causal sources of failure.','Repeated labels may reflect conditions or annotation differences, not proven biological noise.','Coordinatewise feature bounds are not joint support tests or evidence that extrapolation caused error.']}
(P/'DIAGNOSTIC_AUDIT.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({'status':report['status'],'checks':len(checks),'failed':[c['name']for c in checks if not c['pass']]}));assert report['status']=='PASS'
