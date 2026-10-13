from pathlib import Path
import json,math,hashlib,collections
P=Path(__file__).resolve().parent;R=P.parent;S=R.parent/'censored_likelihood_v1'
read=lambda p:json.loads(p.read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
data=[json.loads(s)for s in(S/'DATA.jsonl').read_text().splitlines()];pred={(r['model'],r['activity_id']):r['prediction_log10']for r in[json.loads(s)for s in(S/'run/PREDICTIONS.jsonl').read_text().splitlines()]};enc=read(R/'ENCODING_DISPERSION.json');strata=read(R/'OBSERVED_ASSAY_STRATA.json');checks=[];own={}
def check(n,c):checks.append({'name':n,'pass':bool(c)})
def close(a,b):return abs(a-b)<1e-10
for u,dd in enc.items():
 rs=[r for r in data if r['native_unit_group']==u and r['label_kind']=='point'];groups=collections.defaultdict(lambda:collections.Counter())
 for r in rs:groups[r['source_family_group']][r['peptide_id']]+=1
 weights={r['activity_id']:1/len(groups)/len(groups[r['source_family_group']])/groups[r['source_family_group']][r['peptide_id']]for r in rs};own[u]={}
 for encoding,key in [('peptide_ID',lambda r:r['peptide_id']),('exact_sequence',lambda r:r['sequence']),('composition_and_length',lambda r:''.join(sorted(r['sequence'])))]:
  buckets=collections.defaultdict(list)
  for r in rs:buckets[r['source_family_group'],key(r)].append(r)
  floor=0.;multiid=0;multiseq=0;rep=dd[encoding];details={tuple(sorted(d['activity_ids'])):d for d in rep['details']}
  for g,rr in buckets.items():
   ws=[weights[r['activity_id']]for r in rr];ys=[r['target_log10']for r in rr];mass=sum(ws);mu=sum(w*y for w,y in zip(ws,ys))/mass;variance=sum(w*(y-mu)**2 for w,y in zip(ws,ys));floor+=variance
   ids=sorted({r['peptide_id']for r in rr});seqs=sorted({r['sequence']for r in rr});multiid+=len(ids)>1;multiseq+=len(seqs)>1
   if len(rr)>1:
    d=details[tuple(sorted(r['activity_id']for r in rr))];weight_key='unit_primary_weight_mass'if'unit_primary_weight_mass'in d else'group_normalized_mass'
    check(u+'_'+encoding+'_'+str(min(r['activity_id']for r in rr))+'_bucket',d['rows']==len(rr)and d['peptide_IDs']==ids and d['sequences']==seqs and close(d[weight_key],mass)and close(d['weighted_within_bucket_variance_contribution'],variance)and close(d['label_log10_span'],max(ys)-min(ys)))
   check(u+'_'+encoding+'_'+str(min(r['activity_id']for r in rr))+'_same_predictions',all(max(pred[a,r['activity_id']]for r in rr)-min(pred[a,r['activity_id']]for r in rr)<1e-10 for a in ['point_mean','point_ridge','endpoint_ridge','tail_ridge']))
  check(u+'_'+encoding+'_summary',rep['within_fold_buckets']==len(buckets)and close(rep['same_input_empirical_MSE_floor_component'],floor)and rep['multiple_ID_buckets']==multiid and rep['multiple_sequence_buckets']==multiseq and len(details)==sum(len(rr)>1 for rr in buckets.values()))
  own[u][encoding]={'buckets':len(buckets),'empirical_MSE_floor_component':floor,'multiple_ID_buckets':multiid,'multiple_sequence_buckets':multiseq}
for s in strata:
 rr=[r for r in data if r['native_unit_group']==s['unit']and r['target_raw']==s['target_raw']and str(r.get('medium')or'').strip()==s['medium_raw_trimmed']]
 for kind in ['point','right']:
  rs=[r for r in rr if r['label_kind']==kind];d=s[kind];check(s['unit']+'_'+s['target_raw']+'_'+s['medium_raw_trimmed']+'_'+kind,set(d['activity_ids'])=={r['activity_id']for r in rs}and d['rows']==len(rs)and d['peptide_IDs']==len({r['peptide_id']for r in rs})and d['families']==len({r['sequence_family']for r in rs})and d['source_groups']==len({r['source_family_group']for r in rs})and d['groups']==sorted({r['source_family_group']for r in rs}))
expected={(r['native_unit_group'],r['target_raw'],str(r.get('medium')or'').strip())for r in data if r['explicit_strain_heuristic']and str(r.get('medium')or'').strip()}
check('all_observed_strata_exact',expected=={(s['unit'],s['target_raw'],s['medium_raw_trimmed'])for s in strata}and len(expected)==len(strata))
report={'status':'PASS'if all(c['pass']for c in checks)else'FAIL','checks':checks,'independent_encoding_summary':own,'sha256':{n:sha(R/n)for n in['ENCODING_DISPERSION.json','OBSERVED_ASSAY_STRATA.json','encoding_and_strata.py']},'interpretation':'Composition input equality checked independently by sorted amino-acid multiset; this implies equal residue counts and length. Bucket floor is algebraic weighted minimum on these same observed rows for a deterministic equal-input predictor, not fitted prediction, biological noise or generalization bound. Assay strata use cached target text and recorded medium only, not harmonized assay conditions.'}
(P/'ENCODING_STRATA_AUDIT.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({'status':report['status'],'checks':len(checks),'failed':[c['name']for c in checks if not c['pass']],'own':own}));assert report['status']=='PASS'
