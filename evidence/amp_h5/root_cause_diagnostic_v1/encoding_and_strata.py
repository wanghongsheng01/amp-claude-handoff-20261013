from pathlib import Path
import json,collections
R=Path(__file__).resolve().parent;S=R.parent/'censored_likelihood_v1'
data=[json.loads(l)for l in (S/'DATA.jsonl').read_text().splitlines()];pred=[json.loads(l)for l in (S/'run/PREDICTIONS.jsonl').read_text().splitlines()];out={};strata=[]
def weights(rs):
 c=collections.defaultdict(collections.Counter)
 for r in rs:c[r['source_family_group']][r['peptide_id']]+=1
 return [1/(len(c)*len(c[r['source_family_group']])*c[r['source_family_group']][r['peptide_id']])for r in rs]
def dump(n,d):(R/n).write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
for u in ['ug_ml_native','uM_native']:
 rs=[r for r in data if r['native_unit_group']==u and r['label_kind']=='point'];w=weights(rs);out[u]={}
 for label,keyfunc in [('peptide_ID',lambda r:r['peptide_id']),('exact_sequence',lambda r:r['sequence']),('composition_and_length',lambda r:tuple(r['sequence'].count(a)for a in 'ACDEFGHIKLMNPQRSTVWY')+(len(r['sequence']),))]:
  bins=collections.defaultdict(list)
  for r,ww in zip(rs,w):bins[(r['source_family_group'],keyfunc(r))].append((r,ww))
  floor=0.;details=[]
  for key,rr in bins.items():
   mass=sum(ww for r,ww in rr);mean=sum(r['target_log10']*ww for r,ww in rr)/mass;term=sum(ww*(r['target_log10']-mean)**2 for r,ww in rr);floor+=term
   if len(rr)>1:details.append({'group':key[0],'rows':len(rr),'peptide_IDs':sorted({r['peptide_id']for r,ww in rr}),'sequences':sorted({r['sequence']for r,ww in rr}),'activity_ids':[r['activity_id']for r,ww in rr],'unit_primary_weight_mass':mass,'weighted_within_bucket_variance_contribution':term,'label_log10_span':max(r['target_log10']for r,ww in rr)-min(r['target_log10']for r,ww in rr)})
   for arm in ['point_mean','point_ridge','endpoint_ridge','tail_ridge']:
    pp=[p['prediction_log10']for p in pred if p['model']==arm and p['activity_id'] in {r['activity_id']for r,ww in rr}]
    assert max(pp)-min(pp)<1e-10
  out[u][label]={'within_fold_buckets':len(bins),'same_input_empirical_MSE_floor_component':floor,'multiple_ID_buckets':sum(len(d['peptide_IDs'])>1 for d in details),'multiple_sequence_buckets':sum(len(d['sequences'])>1 for d in details),'details':details,'interpretation':'Weighted label dispersion for identical deterministic inputs within the same fitted fold; algebraic floor on this observed point subset, not estimated biological noise or out-of-sample lower bound. Bucket label means are diagnostic arithmetic, not trained predictors.'}
 allrs=[r for r in data if r['native_unit_group']==u];keys=sorted({(r['target_raw'],str(r.get('medium')or '').strip())for r in allrs if r['explicit_strain_heuristic'] and str(r.get('medium')or '').strip()})
 for target,medium in keys:
  rr=[r for r in allrs if r['target_raw']==target and str(r.get('medium')or '').strip()==medium]
  z={'unit':u,'target_raw':target,'medium_raw_trimmed':medium,'definition':'Exact cached target text + recorded medium; explicit_strain is prior heuristic, not independently verified strain identity; other assay conditions not harmonized'}
  for kind in ['point','right']:
   kk=[r for r in rr if r['label_kind']==kind];z[kind]={'rows':len(kk),'peptide_IDs':len({r['peptide_id']for r in kk}),'families':len({r['sequence_family']for r in kk}),'source_groups':len({r['source_family_group']for r in kk}),'groups':sorted({r['source_family_group']for r in kk}),'activity_ids':[r['activity_id']for r in kk]}
  strata.append(z)
dump('ENCODING_DISPERSION.json',out);dump('OBSERVED_ASSAY_STRATA.json',strata)
print(json.dumps({'encoding':{u:{k:{a:b for a,b in d.items()if a!='details'}for k,d in v.items()}for u,v in out.items()},'top_strata':{u:sorted([s for s in strata if s['unit']==u],key=lambda s:(s['point']['source_groups'],s['point']['rows']),reverse=True)[:3]for u in out}},ensure_ascii=False))
