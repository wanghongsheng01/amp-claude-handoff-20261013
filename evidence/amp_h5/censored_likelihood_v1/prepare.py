from pathlib import Path
import json,collections,hashlib,math
R=Path(__file__).resolve().parent
S=R.parent/'dbaasp_coverage_v5/analysis/NATIVE_ROWS.jsonl'
rows=[json.loads(l) for l in S.read_text().splitlines()]
def dump(p,x): (R/p).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def stats(rs):
 return {'rows':len(rs),'peptide_IDs':len({r['peptide_id'] for r in rs}),'exact_sequences':len({r['sequence'] for r in rs}),'families':len({r['sequence_family'] for r in rs}),'source_groups':len({r['source_family_group'] for r in rs}),'by_group':{g:{'rows':len(z),'peptide_IDs':len({r['peptide_id']for r in z}),'families':len({r['sequence_family']for r in z}),'activity_ids':[r['activity_id']for r in z]}for g in sorted({r['source_family_group']for r in rs}) for z in [[r for r in rs if r['source_family_group']==g]]}}
selected=[];excluded=[]
for r in rows:
 reason=None
 if r['activity_id']==12605: reason='Multiple explicitly named clinical isolates share one bound; per-isolate attribution unresolved; conservative exclusion'
 elif not (r['plain_Witten_primary_point_eligible'] or r['plain_Witten_censor_eligible']):reason='Existing native chemistry/encoding/label admission not satisfied'
 if reason:excluded.append({'activity_id':r['activity_id'],'peptide_id':r['peptide_id'],'format':r['format'],'concentration_raw':r['concentration_raw'],'reason':reason,'prior_core_reasons':r['core_reasons']});continue
 kind={'point':'point','right_bound':'right','left_bound':'left'}[r['format']]
 bound=r['lower_raw_unit'] if kind!='left' else r['upper_raw_unit']
 assert math.isfinite(bound) and bound>0
 if kind!='point': assert r['operator_or_semantics'] in (['>','>='] if kind=='right' else ['<','<='])
 selected.append(dict(r,label_kind=kind,target_log10=math.log10(bound)))
(R/'DATA.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n' for r in selected))
dump('EXCLUSIONS.json',excluded)
out={};folds={}
for u in ['ug_ml_native','uM_native']:
 rs=[r for r in selected if r['native_unit_group']==u];points=[r for r in rs if r['label_kind']=='point'];cs=[r for r in rs if r['label_kind']!='point']
 out[u]={'all':stats(rs),'point':stats(points),'right':stats([r for r in cs if r['label_kind']=='right']),'left':stats([r for r in cs if r['label_kind']=='left']),'censor_only_groups':sorted({r['source_family_group']for r in cs}-{r['source_family_group']for r in points})}
 folds[u]=[]
 for g in sorted({r['source_family_group']for r in points}):
  tr=[r for r in rs if r['source_family_group']!=g];val=[r for r in points if r['source_family_group']==g];tp=[r for r in tr if r['label_kind']=='point'];tc=[r for r in tr if r['label_kind']!='point']
  for k in ['peptide_id','sequence','annotation_fingerprint','sequence_family','source_family_group']:
   assert not ({r[k]for r in tr}&{r[k]for r in val}),(u,g,k)
  assert not ({s for r in tr for s in r['source_keys']}&{s for r in val for s in r['source_keys']})
  folds[u].append({'group':g,'train':[r['activity_id']for r in tr],'validation':[r['activity_id']for r in val],'train_points':stats(tp),'train_censors':stats(tc),'validation_points':stats(val)})
 out[u]['minimum_training_censor_groups']=min(f['train_censors']['source_groups']for f in folds[u]);out[u]['minimum_training_point_groups']=min(f['train_points']['source_groups']for f in folds[u])
 out[u]['screening_gate']=out[u]['right']['source_groups']+out[u]['left']['source_groups']>=3 and out[u]['minimum_training_censor_groups']>=2 and out[u]['point']['source_groups']>=5 and out[u]['minimum_training_point_groups']>=3
 out[u]['point_selection_bias']='Validation observes exact reported points only, selected by assay ranges/reporting. Censored-only groups have no point validation; no estimate of latent censored outcomes or full population generalization.'
dump('INVENTORY.json',out);dump('FOLDS.json',folds)
print(json.dumps({u:{k:(v if not isinstance(v,dict) else {a:b for a,b in v.items()if a!='by_group'})for k,v in d.items()}for u,d in out.items()},ensure_ascii=False))
