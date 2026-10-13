"""Read-only arithmetic diagnostics of frozen predictions. No optimization/model import."""
from pathlib import Path
import collections,csv,json,math,hashlib,datetime
R=Path(__file__).resolve().parent;S=R.parent/'censored_likelihood_v1'
def read(p):return json.loads(p.read_text())
def jl(p):return [json.loads(l)for l in p.read_text().splitlines()]
def dump(n,d):(R/n).write_text(json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def weights(rs):
 c=collections.defaultdict(collections.Counter)
 for r in rs:c[r['source_family_group']][r['peptide_id']]+=1
 return [1/(len(c)*len(c[r['source_family_group']])*c[r['source_family_group']][r['peptide_id']])for r in rs]
def avg(x,w):return sum(a*b for a,b in zip(x,w))
def feat(r):return [r['sequence'].count(a)/len(r['sequence'])for a in 'ACDEFGHIKLMNPQRSTVWY']+[math.log(len(r['sequence']))]
manifest=read(S/'FINAL_MANIFEST.json');assert all(sha(S/x['path'])==x['sha256']for x in manifest['files'])
data=jl(S/'DATA.jsonl');lookup={r['activity_id']:r for r in data};preds=jl(S/'run/PREDICTIONS.jsonl');pred={(p['model'],p['activity_id']):p['prediction_log10']for p in preds}
summary={};groups=[];peptides=[];support=[];selection=[];metarecords=[];arms=['point_mean','point_ridge','endpoint_ridge','tail_ridge'];sources=[S/'FINAL_MANIFEST.json',S/'DATA.jsonl',S/'run/PREDICTIONS.jsonl',S/'run/RESULT.json',S/'PROTOCOL.json',S/'model.py',S/'run.py']
for unit in ['ug_ml_native','uM_native']:
 allrs=[r for r in data if r['native_unit_group']==unit];rs=[r for r in allrs if r['label_kind']=='point'];w=weights(rs);y=[r['target_log10']for r in rs];grand=avg(y,w);gs=sorted({r['source_family_group']for r in rs});groupmeans={};groupvars={}
 for g in gs:
  rr=[r for r in rs if r['source_family_group']==g];ww=weights(rr);yy=[r['target_log10']for r in rr];gm=avg(yy,ww);gv=avg([(v-gm)**2 for v in yy],ww);groupmeans[g]=gm;groupvars[g]=gv
  receipt=S/'run'/(unit+'__'+g+'.json');d=read(receipt);sources.append(receipt);tr=[lookup[i]for i in d['fold']['train']];xtr=[feat(r)for r in tr];xmin=[min(x[j]for x in xtr)for j in range(21)];xmax=[max(x[j]for x in xtr)for j in range(21)];sup=[]
  for r in rr:
   x=feat(r);outside=[j for j,v in enumerate(x)if v<xmin[j]-1e-12 or v>xmax[j]+1e-12];z=[(v-d['mean'][j])/d['scale'][j]for j,v in enumerate(x)];item={'unit':unit,'group':g,'activity_id':r['activity_id'],'peptide_id':r['peptide_id'],'outside_train_feature_count':len(outside),'outside_feature_names':[list('ACDEFGHIKLMNPQRSTVWY') [j] if j<20 else 'loglength'for j in outside],'max_abs_standardized_feature':max(map(abs,z))};support.append(item);sup.append(item)
  for arm in arms:
   pp=[pred[(arm,r['activity_id'])]for r in rr];err=[p-v for p,v in zip(pp,yy)];bias=avg(err,ww);centered=avg([(e-bias)**2 for e in err],ww);mse=avg([e*e for e in err],ww);assert abs(mse-bias*bias-centered)<1e-10
   groups.append({'unit':unit,'group':g,'model':arm,'point_rows':len(rr),'peptide_IDs':len({r['peptide_id']for r in rr}),'exact_sequences':len({r['sequence']for r in rr}),'families':len({r['sequence_family']for r in rr}),'label_mean_log10':gm,'label_min_log10':min(yy),'label_max_log10':max(yy),'label_within_group_variance':gv,'training_point_mean_log10':d['point_mean'],'heldout_minus_training_point_mean':gm-d['point_mean'],'prediction_mean_log10':avg(pp,ww),'bias':bias,'bias_squared':bias*bias,'centered_error_MSE':centered,'MSE':mse,'MAE':avg(list(map(abs,err)),ww),'any_point_outside_train_feature_box':any(s['outside_train_feature_count']for s in sup)})
 model_decomp={}
 for arm in arms:
  pp=[pred[(arm,r['activity_id'])]for r in rs];gg=[g for g in groups if g['unit']==unit and g['model']==arm];mse=avg([(p-v)**2 for p,v in zip(pp,y)],w);bias=sum(g['bias_squared']for g in gg)/len(gs);centered=sum(g['centered_error_MSE']for g in gg)/len(gs);c=[pred[('point_mean',r['activity_id'])]for r in rs];delta=[p-b for p,b in zip(pp,c)];energy=avg([v*v for v in delta],w);align=avg([v*(t-b)for v,t,b in zip(delta,y,c)],w);meanmse=avg([(b-v)**2 for b,v in zip(c,y)],w)
  assert abs(mse-bias-centered)<1e-10 and abs(mse-meanmse-energy+2*align)<1e-10
  model_decomp[arm]={'MSE':mse,'MAE':avg([abs(p-v)for p,v in zip(pp,y)],w),'mean_group_bias_squared':bias,'mean_centered_error_MSE':centered,'bias_squared_fraction_of_MSE':bias/mse,'versus_mean_adjustment_energy':energy,'versus_mean_adjustment_label_alignment':align,'versus_mean_delta_MSE':mse-meanmse,'identity':'delta_MSE = E[(prediction-constant)^2] - 2 E[(prediction-constant)*(label-constant)]'}
 between=sum((v-grand)**2 for v in groupmeans.values())/len(gs);within=sum(groupvars.values())/len(gs);total=avg([(v-grand)**2 for v in y],w);assert abs(total-between-within)<1e-10
 for pid in sorted({r['peptide_id']for r in rs}):
  rr=[r for r in rs if r['peptide_id']==pid];yy=[r['target_log10']for r in rr];cond={tuple(str(r.get(k))for k in ['target_raw','medium','cfu','ph','ionic_strength','salt','note'])for r in rr}
  peptides.append({'unit':unit,'peptide_id':pid,'point_rows':len(rr),'distinct_reported_point_values':len(set(yy)),'log10_range':max(yy)-min(yy),'ratio_max_min':10**(max(yy)-min(yy)),'distinct_observed_condition_tuples':len(cond),'group':rr[0]['source_family_group'],'activity_ids':[r['activity_id']for r in rr]})
 pps=[p for p in peptides if p['unit']==unit];ss=[s for s in support if s['unit']==unit]
 summary[unit]={'point_rows':len(rs),'peptide_IDs':len({r['peptide_id']for r in rs}),'exact_sequences':len({r['sequence']for r in rs}),'families':len({r['sequence_family']for r in rs}),'source_groups':len(gs),'single_point_row_groups':sum(sum(r['source_family_group']==g for r in rs)==1 for g in gs),'one_peptide_groups':sum(len({r['peptide_id']for r in rs if r['source_family_group']==g})==1 for g in gs),'group_mean_min':min(groupmeans.values()),'group_mean_max':max(groupmeans.values()),'group_mean_log10_span':max(groupmeans.values())-min(groupmeans.values()),'group_mean_geometric_ratio':10**(max(groupmeans.values())-min(groupmeans.values())),'group_means':groupmeans,'grand_group_weighted_mean':grand,'variance_decomposition':{'total_label_variance':total,'between_source_group_mean_variance':between,'within_source_group_variance':within,'between_fraction':between/total},'model_error_decomposition':model_decomp,'repeated_peptides':{'multiple_point_records':sum(p['point_rows']>1 for p in pps),'multiple_distinct_point_values':sum(p['distinct_reported_point_values']>1 for p in pps),'more_than_twofold_point_range':sum(p['log10_range']>math.log10(2)+1e-12 for p in pps),'maximum_observed_ratio':max(p['ratio_max_min']for p in pps)},'feature_box_support':{'rows_outside_any_training_feature_range':sum(p['outside_train_feature_count']>0 for p in ss),'IDs_outside':len({p['peptide_id']for p in ss if p['outside_train_feature_count']>0}),'groups_outside':len({p['group']for p in ss if p['outside_train_feature_count']>0}),'max_abs_z':max(p['max_abs_standardized_feature']for p in ss),'warning':'Coordinatewise range is a descriptive support diagnostic, not a joint support/power test or proof of extrapolation causality'}}
 for kind in ['point','right','left']:
  rr=[r for r in allrs if r['label_kind']==kind]
  for g in sorted({r['source_family_group']for r in rr}):
   z=[r for r in rr if r['source_family_group']==g];selection.append({'unit':unit,'group':g,'kind':kind,'rows':len(z),'IDs':len({r['peptide_id']for r in z}),'families':len({r['sequence_family']for r in z}),'raw_threshold_or_point_min':min(r['lower_raw_unit']for r in z),'raw_threshold_or_point_max':max(r['lower_raw_unit']for r in z)})
 for key in ['target_raw','medium','cfu','ph','ionic_strength','salt']:
  counts=collections.Counter(str(r.get(key) or '')for r in rs);metarecords.append({'unit':unit,'field':key,'missing_rows':sum(n for v,n in counts.items() if not v.strip()),'distinct_nonempty_values':len({v.strip() for v in counts if v.strip()}),'observed_counts':dict(counts),'missingness_rule':'None, empty or whitespace-only is missing; raw strings preserved'})
for n,d in [('SUMMARY.json',summary),('GROUP_ERRORS.json',groups),('PEPTIDE_REPEATS.json',peptides),('FEATURE_SUPPORT.json',support),('POINT_CENSOR_SELECTION.json',selection),('OBSERVED_CONDITIONS.json',metarecords)]:dump(n,d)
for n,d in [('GROUP_ERRORS.csv',groups),('PEPTIDE_REPEATS.csv',peptides),('FEATURE_SUPPORT.csv',support),('POINT_CENSOR_SELECTION.csv',selection)]:
 with (R/n).open('w',newline='')as f:
  wr=csv.DictWriter(f,fieldnames=list(d[0]));wr.writeheader();wr.writerows(d)
dump('SOURCE_BINDINGS.json',{'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'source_manifest_verified_files':len(manifest['files']),'source_sha256':{str(p.relative_to(R.parent)):sha(p)for p in sources},'diagnostic_code_sha256':sha(Path(__file__)),'no_training':True,'no_network':True,'no_test37':True})
print(json.dumps(summary,ensure_ascii=False))
