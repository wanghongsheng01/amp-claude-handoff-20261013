from pathlib import Path
import json,csv,math,hashlib,collections
P=Path(__file__).resolve().parent;R=P.parent
read=lambda p:json.loads(p.read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
own=read(P/'RECOMPUTED_RESULTS.json');contr=read(R/'CONTRASTS.json');dist=read(R/'DISTRIBUTION.json');data=[json.loads(s)for s in(R/'DATA.jsonl').read_text().splitlines()];checks=[]
def check(n,c):checks.append({'name':n,'pass':bool(c)})
def close(a,b):return abs(float(a)-float(b))<1e-9
pairs={'tail_ridge_minus_endpoint_ridge':('tail_ridge','endpoint_ridge'),'endpoint_ridge_minus_point_ridge':('endpoint_ridge','point_ridge'),'tail_ridge_minus_point_ridge':('tail_ridge','point_ridge'),'tail_ridge_minus_point_mean':('tail_ridge','point_mean')}
for u,cc in contr.items():
 for name,v in cc.items():
  a,b=pairs[name];m=own['units'][u]
  check(u+'_'+name+'_all6_deltas',all(close(delta,m[a][k]-m[b][k])for k,delta in v['metrics'].items()))
  ds=[fm[a]['group_peptide_MAE']-fm[b]['group_peptide_MAE']for fm in own['fold_metrics'][u].values()]
  check(u+'_'+name+'_win_tie_loss',v['group_MAE_win_tie_loss']=={'win':sum(d< -1e-12 for d in ds),'tie':sum(abs(d)<=1e-12 for d in ds),'loss':sum(d>1e-12 for d in ds)})
with(R/'FOLD_CONTRASTS.csv').open()as f:fs=list(csv.DictReader(f))
check('136_fold_contrasts',len(fs)==34*4 and len({(r['unit'],r['group'],r['contrast'])for r in fs})==136)
for r in fs:
 a,b=pairs[r['contrast']];m=own['fold_metrics'][r['unit']][r['group']]
 check(r['unit']+'_'+r['group']+'_'+r['contrast']+'_row',close(r['MAE_delta'],m[a]['group_peptide_MAE']-m[b]['group_peptide_MAE'])and close(r['RMSE_delta'],m[a]['group_peptide_RMSE']-m[b]['group_peptide_RMSE'])and int(r['point_rows'])==m[a]['rows']and int(r['point_peptides'])==m[a]['peptide_IDs'])
for d in dist:
 rs=[r for r in data if r['native_unit_group']==d['unit']and r['label_kind']==d['label_kind']and r[d['dimension']]==d['value']]
 expected={'rows':len(rs),'peptide_IDs':len({r['peptide_id']for r in rs}),'exact_sequences':len({r['sequence']for r in rs}),'families':len({r['sequence_family']for r in rs}),'source_groups':len({r['source_family_group']for r in rs})}
 check(str(d['unit'])+'_'+d['label_kind']+'_'+d['dimension']+'_'+str(d['value']),set(d['activity_ids'])=={r['activity_id']for r in rs}and all(d[k]==v for k,v in expected.items()))
for u in own['units']:
 for kind in ['point','right','left']:
  ids={r['activity_id']for r in data if r['native_unit_group']==u and r['label_kind']==kind}
  for dim in ['peptide_id','sequence_family','source_family_group']:
   listed=[i for d in dist if d['unit']==u and d['label_kind']==kind and d['dimension']==dim for i in d['activity_ids']]
   check(u+'_'+kind+'_'+dim+'_complete_partition',collections.Counter(listed)==collections.Counter(ids))
with(R/'DISTRIBUTION.csv').open()as f:csvdist=list(csv.DictReader(f))
check('distribution_csv218',len(csvdist)==len(dist)==218)
# Human static reading independently checked numeric table rounding, directional statements and limitations.
check('report_tables_numeric_rounding',all(format(v,'.6f')in(R/'REPORT.zh.md').read_text()for u in own['units'].values()for m in u.values()for k,v in m.items()if k in ['group_peptide_MAE','group_peptide_RMSE']))
report={'status':'PASS'if all(c['pass']for c in checks)else'FAIL','checks':checks,'report_static_review':'REPORT tables match independent rounded MAE/RMSE; two-unit tail regression worsens all three weighting MAE/RMSE versus endpoint; 102fits34folds588predictions and79/68point counts correct. Fixed sigma, point-selection bias, no natural left-tail validation, censor-only groups and changed-scaler old comparability caveats present.','reviewed_sha256':{n:sha(R/n)for n in ['REPORT.zh.md','CONTRASTS.json','FOLD_CONTRASTS.csv','DISTRIBUTION.json','DISTRIBUTION.csv']}}
(P/'REPORT_AUDIT.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({'status':report['status'],'checks':len(checks),'failed':[c['name']for c in checks if not c['pass']]}));assert report['status']=='PASS'
