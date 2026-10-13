"""Same reported-native-label semantics; offline grouping relations, no fitting."""
import collections,hashlib,json,math,re,datetime
from pathlib import Path
import chemistry_engine as chemistry
R=Path(__file__).resolve().parent;H=R.parent;OUT=R/'analysis'
def readl(p):return [json.loads(s)for s in p.read_text().splitlines()]
def write(n,x):(OUT/n).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def canon(x):return json.dumps(x,sort_keys=True,ensure_ascii=False,separators=(',',':'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
OUT.mkdir(exist_ok=False)
ps=readl(R/'prepared/peptides.jsonl');ms=readl(R/'prepared/mic_measurements.jsonl');op=readl(H/'dbaasp_checkpoint_v6/prepared/peptides.jsonl');om=readl(H/'dbaasp_checkpoint_v6/prepared/mic_measurements.jsonl');oldbaseline=[r for r in readl(H/'dbaasp_checkpoint_v6/analysis/NATIVE_ROWS.jsonl') if r['plain_Witten_primary_point_eligible']];oldnative=readl(H/'dbaasp_checkpoint_v6/analysis/NATIVE_ROWS.jsonl')
P={p['peptide_id']:p for p in ps};OP={p['peptide_id']:p for p in op};oldids=set(OP);newids=set(P)-oldids
# The chemistry algorithm is identical to frozen v2; unknown structures stay unknown.
audits=[chemistry.audit_peptide(p)for p in ps];A={a['peptide_id']:a for a in audits}
(OUT/'CHEMISTRY.jsonl').write_text(''.join(canon(a)+'\n'for a in audits))
attr={}
for p in ps:
 text=str(p['note_reference']or'')
 if re.search(r'part\s+of.{0,80}(dimer|two[\s-]*peptide|multimer|complex)',text,re.I):attr[p['peptide_id']]='PART_OF_MULTICOMPONENT_ENTITY_ATTRIBUTION_PENDING'
rows=[];AA=set('ACDEFGHIKLMNPQRSTVWY')
for m in ms:
 if not m['ecoli_target']:continue
 p=P[m['peptide_id']];a=A[m['peptide_id']];c=p['chemistry'];s=p['sequence'];why=[]
 unit='ug_ml_native'if m['unit_raw']=='µg/ml'else('uM_native'if m['unit_raw']in['µM','μM','uM']else'unsupported')
 if unit=='unsupported':why.append('UNSUPPORTED_NATIVE_UNIT')
 if m['format']not in['point','right_bound','left_bound']:why.append('UNRESOLVED_RANGE_PLUSMINUS_OR_PARSE')
 if not m['reference_index_valid']:why.append('SOURCE_ACTIVITY_LINK_UNRESOLVED')
 if not s:why.append('EMPTY_SEQUENCE')
 if a['structure_status']=='REFUSE':why.append('EXPLICIT_CHEMISTRY_CONFLICT_OR_UNSUPPORTED_COMPONENTS')
 if c.get('interchainBonds')or c.get('coordinationBonds'):why.append('EXPLICIT_MULTICOMPONENT_OR_COORDINATION_ATTRIBUTION_PENDING')
 if p['peptide_id']in attr:why.append(attr[p['peptide_id']])
 plain=bool(s)and set(s)<=AA and 1<=len(s)<100
 row=dict(m);row.update(native_unit_group=unit,core_reasons=why,core_reported_label_eligible=not why,plain_Witten_encoding_compatible=plain,plain_Witten_primary_point_eligible=not why and plain and m['format']=='point',plain_Witten_censor_eligible=not why and plain and m['format']in['right_bound','left_bound'],structure_status=a['structure_status'],note_reference=p['note_reference'],attribution_pending=p['peptide_id']in attr,origin_batch=p['raw_provenance']['batch'],scientific_scope='EXPOSED_DEVELOPMENT_MATERIAL_ONLY',execution_admitted=False)
 rows.append(row)
(OUT/'NATIVE_ROWS.jsonl').write_text(''.join(canon(r)+'\n'for r in rows))
def stats(rs):
 return {'rows':len(rs),'peptide_IDs':len({r['peptide_id']for r in rs}),'exact_sequences':len({r['sequence']for r in rs}),'annotation_fingerprints':len({r['annotation_fingerprint']for r in rs}),'families':len({r['sequence_family']for r in rs}),'source_family_components':len({r['source_family_group']for r in rs}),'IDs':sorted({r['peptide_id']for r in rs}),'family_ids':sorted({r['sequence_family']for r in rs}),'source_group_ids':sorted({r['source_family_group']for r in rs})}
def group_map(field):
 groups=collections.defaultdict(list)
 for p in ps:groups[p[field]].append(p)
 return [{'expanded_group':g,'members':[p['peptide_id']for p in members],'old_IDs':[p['peptide_id']for p in members if p['peptide_id']in oldids],'new_IDs':[p['peptide_id']for p in members if p['peptide_id']in newids],'old_groups':sorted({OP[p['peptide_id']][field]for p in members if p['peptide_id']in oldids}),'new_only_component':not any(p['peptide_id']in oldids for p in members),'bridges_old_components':len({OP[p['peptide_id']][field]for p in members if p['peptide_id']in oldids})>1}for g,members in sorted(groups.items())]
relations={'family_components':group_map('sequence_family'),'source_family_components':group_map('source_family_group'),'old_to_expanded':[{'peptide_id':i,'old_family':OP[i]['sequence_family'],'expanded_family':P[i]['sequence_family'],'old_source_group':OP[i]['source_family_group'],'expanded_source_group':P[i]['source_family_group']}for i in sorted(oldids)],'baseline_units':{}}
summary={'schema':'AMP_H5_CHECKPOINT_NATIVE_ANALYSIS_V7','created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'all_peptides':stats(ps),'old_peptides_in_expanded_groups':stats([p for p in ps if p['peptide_id']in oldids]),'new_peptides':stats([p for p in ps if p['peptide_id']in newids]),'all_MIC':stats(ms),'Ecoli_MIC':stats(rows),'native_units':{},'chemical_statuses':dict(collections.Counter(a['structure_status']for a in audits)),'attribution_pending':attr,'source_input_warning':'Native means cached reporting units; paper semantics not newly independently reverified','no_training':True,'all_development_not_new_test':True}
for unit in ['ug_ml_native','uM_native']:
 rs=[r for r in rows if r['native_unit_group']==unit];core=[r for r in rs if r['core_reported_label_eligible']and r['plain_Witten_encoding_compatible']];pts=[r for r in core if r['format']=='point'];b=[r for r in oldbaseline if r['native_unit_group']==unit];beforeacts={r['activity_id']:r for r in b};afteracts={r['activity_id']:r for r in pts};bgroups=collections.defaultdict(list)
 for r in b:bgroups[P[r['peptide_id']]['source_family_group']].append(r)
 oldpointids={r['peptide_id']for r in b};expanded_oldpoint_groups={P[i]['source_family_group']for i in oldpointids};expanded_oldpoint_families={P[i]['sequence_family']for i in oldpointids}
 newpoints=[r for r in pts if r['peptide_id']in newids]
 group_sizes=[]
 for g in sorted({r['source_family_group']for r in pts}):
  val=[r for r in pts if r['source_family_group']==g];train=[r for r in pts if r['source_family_group']!=g]
  group_sizes.append({'held_out_group':g,'validation':stats(val),'training':stats(train),'train_distinct_point_values':len({r['lower_raw_unit']for r in train}),'structurally_executable':len({r['source_family_group']for r in train})>=2 and bool(val)})
 summary['native_units'][unit]={'all':stats(rs),'point_plus_bound':stats(core),'point':stats(pts),'bounds':stats([r for r in core if r['format']!='point']),'new_points':stats(newpoints),'new_points_in_previously_uncovered_sources':stats([r for r in newpoints if r['source_family_group']not in expanded_oldpoint_groups]),'new_points_in_previously_uncovered_families':stats([r for r in newpoints if r['sequence_family']not in expanded_oldpoint_families]),'folds':group_sizes,'missing_point_covariates':{k:sum(r[k] is None or (isinstance(r[k],str) and not r[k].strip()) or r[k] is False for r in pts)for k in['medium','cfu','ph','ionic_strength','salt','both_termini_observed']},'excluded_reason_counts':dict(collections.Counter(e for r in rs for e in r['core_reasons']))}
 relations['baseline_units'][unit]={'prior_point_rows':len(b),'prior_point_IDs':len(oldpointids),'prior_group_count':len({r['source_family_group']for r in b}),'old_point_group_count_after_expansion':len(bgroups),'old_folds_connected_by_new_records':[{'expanded_group':g,'old_fold_ids':sorted({r['source_family_group']for r in members}),'old_activity_ids':[r['activity_id']for r in members]}for g,members in bgroups.items()if len({r['source_family_group']for r in members})>1],'prior_activity_eligibility_changed':[{'activity_id':i,'peptide_id':rr['peptide_id'],'new_reasons':next((r['core_reasons']for r in rows if r['activity_id']==i),['MISSING'])}for i,rr in beforeacts.items()if i not in afteracts],'newly_eligible_activity_ids':sorted(set(afteracts)-set(beforeacts)),'historical_results':'Oldmodel/folds/scores notmodified; expandedbridge interpretation appended only'}
# Prove that old raw input records are retained unchanged except recomputed family/source grouping.
OM={(r['peptide_id'],r['activity_id']):r for r in om};MM={(r['peptide_id'],r['activity_id']):r for r in ms}
for key,old in OM.items():
 assert key in MM
 for k,v in old.items():
  if k not in ['sequence_family','source_family_group']:assert MM[key][k]==v,(key,k)
for old in oldnative:
 now=next(r for r in rows if r['activity_id']==old['activity_id']and r['peptide_id']==old['peptide_id'])
 for k in ['core_reasons','core_reported_label_eligible','plain_Witten_encoding_compatible','plain_Witten_primary_point_eligible','plain_Witten_censor_eligible']:assert now[k]==old[k],(old['activity_id'],k)
summary['old_all1278_MIC_fields_unchanged_except_recomputed_groups']=True;summary['old240_native_eligibility_unchanged']=True
write('COUNTS.json',summary);write('GROUP_RELATIONSHIPS.json',relations);write('NEW_RECORDS.json',[{'peptide':p,'chemistry_audit':A[p['peptide_id']],'MIC_rows':[r for r in rows if r['peptide_id']==p['peptide_id']]}for p in ps if p['peptide_id']in newids])
print(json.dumps({'records':len(ps),'new_ids':sorted(newids),'units':{u:{k:v[k]for k in ['point','new_points','new_points_in_previously_uncovered_sources']}for u,v in summary['native_units'].items()},'chemistry':summary['chemical_statuses']},ensure_ascii=False))
