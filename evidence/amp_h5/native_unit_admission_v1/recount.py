"""Offline native-unit admission accounting; no model/split/protected-file access."""
import collections,datetime,hashlib,json,math
from pathlib import Path
R=Path(__file__).resolve().parent;H=R.parent
AA=set('ACDEFGHIKLMNPQRSTVWY')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def canonical(x):return json.dumps(x,sort_keys=True,ensure_ascii=False,separators=(',',':'))
def readl(p):return [json.loads(s) for s in p.read_text().splitlines()]
def write(n,d):(R/n).write_text(json.dumps(d,indent=2,ensure_ascii=False)+'\n')
def meaningful(x):
    if x is None or x=='':return False
    if isinstance(x,dict):return any(meaningful(v) for k,v in x.items() if k not in ('id','peptideId'))
    if isinstance(x,list):return any(meaningful(v) for v in x)
    return True
freeze=json.loads((R/'FREEZE.json').read_text())
for rel,h in freeze['inputs_sha256'].items():assert sha(H/rel)==h,rel
for rel,h in freeze['local_sha256'].items():assert sha(R/rel)==h,rel
ps=readl(H/'dbaasp_bounded_v3/prepared/peptides.jsonl');ms=readl(H/'dbaasp_bounded_v3/prepared/mic_measurements.jsonl');aa=readl(H/'chemistry_units_v2/results/chemical_identity_audit.jsonl')
P={p['peptide_id']:p for p in ps};A={p['peptide_id']:p for p in aa}
assert len(P)==75
rules=json.loads((R/'COUNTING_RULES.json').read_text());attr=set(rules['attribution_pending_ids'])
assert attr=={16,36,1783}
rows=[]
for m in ms:
    if not m['ecoli_target']:continue
    p=P[m['peptide_id']];a=A[m['peptide_id']];c=p['chemistry'];s=p['sequence'];why=[]
    unit='ug_ml_native' if m['unit_raw']=='µg/ml' else ('uM_native' if m['unit_raw'] in ['µM','μM','uM'] else 'unsupported')
    if unit=='unsupported':why.append('UNSUPPORTED_NATIVE_UNIT')
    if m['format'] not in ['point','left_bound','right_bound']:why.append('UNRESOLVED_RANGE_PLUSMINUS_OR_PARSE')
    if not m['reference_index_valid']:why.append('SOURCE_ACTIVITY_LINK_UNRESOLVED')
    if not s:why.append('EMPTY_SEQUENCE')
    if a['structure_status']=='REFUSE':why.append('EXPLICIT_CHEMISTRY_CONFLICT_OR_UNSUPPORTED_COMPONENTS')
    if c.get('interchainBonds') or c.get('coordinationBonds'):why.append('EXPLICIT_MULTICOMPONENT_OR_COORDINATION_ATTRIBUTION_PENDING')
    if p['peptide_id'] in attr:why.append('PART_OF_MULTICOMPONENT_ENTITY_ATTRIBUTION_PENDING')
    plain=bool(s) and set(s)<=AA and 1<=len(s)<100
    rr=dict(m)
    rr.update(native_unit_group=unit,core_reasons=why,core_reported_label_eligible=not why,plain_Witten_encoding_compatible=plain,plain_Witten_primary_point_eligible=not why and plain and m['format']=='point',plain_Witten_censor_eligible=not why and plain and m['format'] in ['left_bound','right_bound'],modification_annotated=any(bool(c.get(k)) for k in ['nTerminus','cTerminus','unusualAminoAcids','intrachainBonds','interchainBonds','coordinationBonds']),structure_status=a['structure_status'],note_reference=p['note_reference'],attribution_pending=p['peptide_id'] in attr,cfu_value_present=meaningful(m['cfu']),cfu_any_metadata_present=meaningful(m['cfu']) or meaningful(m['cfu_group']),scientific_eligibility_scope='reported-native-label exploratory task only',execution_admitted=False)
    keys=['peptide_id','annotation_fingerprint','target_raw','medium','cfu','cfu_group','ph','ionic_strength','salt','source_keys','note','unit_raw']
    sig={k:m[k] for k in keys}
    rr['recorded_condition_source_signature']=hashlib.sha256(canonical(sig).encode()).hexdigest()
    rr['potential_same_observation_signature']=hashlib.sha256(canonical(dict(sig,concentration_raw=m['concentration_raw'])).encode()).hexdigest()
    rows.append(rr)
def stats(rs):
    protocol=collections.Counter(r['recorded_condition_source_signature'] for r in rs);observations=collections.Counter(r['potential_same_observation_signature'] for r in rs)
    return {'rows':len(rs),'peptide_IDs':len({r['peptide_id'] for r in rs}),'exact_sequences':len({r['sequence'] for r in rs}),'chemical_annotation_fingerprints':len({r['annotation_fingerprint'] for r in rs}),'sequence_families':len({r['sequence_family'] for r in rs}),'source_family_components':len({r['source_family_group'] for r in rs}),'point_rows':sum(r['format']=='point' for r in rs),'right_bound_rows':sum(r['format']=='right_bound' for r in rs),'left_bound_rows':sum(r['format']=='left_bound' for r in rs),'recorded_condition_source_signatures':len(protocol),'repeated_condition_source_signatures':sum(n>1 for n in protocol.values()),'potential_same_observation_signatures':len(observations),'potential_duplicate_extra_rows':sum(n-1 for n in observations.values()),'IDs':sorted({r['peptide_id'] for r in rs}),'families':sorted({r['sequence_family'] for r in rs}),'source_components':sorted({r['source_family_group'] for r in rs})}
def eligible(rs):return [r for r in rs if r['core_reported_label_eligible'] and r['plain_Witten_encoding_compatible']]
summary={'schema':'AMP_H5_NATIVE_UNIT_COUNTS_V1','created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'records':75,'all_Ecoli':stats(rows),'by_native_unit':{},'row_hard_or_pending_reason_counts':dict(collections.Counter(e for r in rows for e in r['core_reasons'])),'attribution_pending_IDs':sorted(attr),'independence_warning':'Unique IDs and operational family/source components are not proven statistically independent samples; no isolation claim','execution':'NOT_RUN_NOT_ADMITTED_THIS_TURN','test37':'NOT_OPENED'}
for u in ['ug_ml_native','uM_native']:
    rs=[r for r in rows if r['native_unit_group']==u];es=eligible(rs)
    summary['by_native_unit'][u]={'all_rows':stats(rs),'core_labels_all_representations':stats([r for r in rs if r['core_reported_label_eligible']]),'plain_Witten_point_plus_bound':stats(es),'plain_Witten_primary_points':stats([r for r in es if r['format']=='point']),'plain_Witten_censored_reserve':stats([r for r in es if r['format'] in ['right_bound','left_bound']]),'attribution_pending_retained':stats([r for r in rs if r['attribution_pending']]),'strict_old_metadata_screen_except_unit':stats([r for r in es if r['explicit_strain_heuristic'] and r['medium']]),'without_attribution_exclusion_sensitivity':stats([r for r in rs if r['plain_Witten_encoding_compatible'] and all(w=='PART_OF_MULTICOMPONENT_ENTITY_ATTRIBUTION_PENDING' for w in r['core_reasons'])]),'point_feature_domains':{}}
    pts=[r for r in es if r['format']=='point']
    for field in ['explicit_strain_heuristic','medium','cfu_value_present','cfu_any_metadata_present','ph','ionic_strength','salt','both_termini_observed','modification_annotated']:
        summary['by_native_unit'][u]['point_feature_domains'][field]={'present':stats([r for r in pts if meaningful(r[field]) and r[field] is not False]),'absent_or_unknown':stats([r for r in pts if not meaningful(r[field]) or r[field] is False])}
    summary['by_native_unit'][u]['point_feature_domains']['represented_structure_pass']={'pass':stats([r for r in pts if r['structure_status']=='PASS_REPRESENTED_STRUCTURE']),'pending':stats([r for r in pts if r['structure_status']!='PASS_REPRESENTED_STRUCTURE'])}
(R/'ROW_ADMISSION.jsonl').write_text(''.join(canonical(r)+'\n' for r in rows))
write('COUNTS.json',summary)
write('PENDING_ATTRIBUTION.json',[{'peptide_id':i,'note_reference':P[i]['note_reference'],'rows':stats([r for r in rows if r['peptide_id']==i]),'decision':'Retain; source needs to identify whether single chain or mixture measured; not declared confirmed mixture'} for i in sorted(attr)])
# Keep repeat groups as records, not a deduplicated fabricated sample.
repeat=collections.defaultdict(list)
for r in rows:repeat[r['potential_same_observation_signature']].append({'peptide_id':r['peptide_id'],'activity_id':r['activity_id'],'native_unit_group':r['native_unit_group'],'concentration_raw':r['concentration_raw']})
write('POTENTIAL_DUPLICATE_LINKS.json',{'groups':[{'signature':k,'records':v,'decision':'Keep all; not assumed duplicate or independent replicate'} for k,v in repeat.items() if len(v)>1]})
print(json.dumps({u:{k:{x:v[x] for x in ['rows','peptide_IDs','sequence_families','source_family_components','point_rows','right_bound_rows']} for k,v in d.items() if k in ['all_rows','core_labels_all_representations','plain_Witten_point_plus_bound','plain_Witten_primary_points','plain_Witten_censored_reserve','strict_old_metadata_screen_except_unit','without_attribution_exclusion_sensitivity']} for u,d in summary['by_native_unit'].items()}))
