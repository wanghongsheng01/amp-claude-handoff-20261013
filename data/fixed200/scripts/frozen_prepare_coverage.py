"""Offline, deterministic supervised-table draft; does not access train/dev/test files."""
import csv, hashlib, json, re, collections, datetime, math
from pathlib import Path
R = Path(__file__).resolve().parent
OUT = R / 'prepared'
OUT.mkdir(exist_ok=True)
def sha(b): return hashlib.sha256(b).hexdigest()
def canonical(x): return json.dumps(x, sort_keys=True, ensure_ascii=False, separators=(',',':'))
def put(name,x): (OUT/name).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def named(x): return str((x or {}).get('name') or '') if isinstance(x,(dict,type(None))) else str(x)
def write_csv(name,rows,fields=None):
    fields = fields or (list(rows[0]) if rows else [])
    with (OUT/name).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for row in rows:w.writerow({k:canonical(v) if isinstance(v,(list,dict)) else v for k,v in row.items()})
def remove_ids(x):
    if isinstance(x,dict):return {k:remove_ids(v) for k,v in x.items() if k not in ('id','peptideId','reference')}
    if isinstance(x,list):return [remove_ids(v) for v in x]
    return x
NUM=r'(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?'
def parse(s):
    s=str(s or '').strip().replace('−','-').replace('–','-').replace('≥','>=').replace('≤','<=')
    if re.fullmatch(NUM,s):
        v=float(s);return ('point',v,v, 'point_as_reported') if math.isfinite(v) and v>0 else ('unresolved',None,None,'nonpositive_or_nonfinite')
    m=re.fullmatch(r'(>=|<=|>|<)\s*('+NUM+')',s)
    if m and math.isfinite(float(m[2])) and float(m[2])>0:
        op,v=m[1],float(m[2]);return ('right_bound',v,None,op) if op.startswith('>') else ('left_bound',None,v,op)
    m=re.fullmatch('('+NUM+')\s*-\s*('+NUM+')',s)
    if m and math.isfinite(float(m[2])) and 0<float(m[1])<=float(m[2]):return ('range_unresolved',float(m[1]),float(m[2]),'semantic_not_verified')
    m=re.fullmatch('('+NUM+')\s*(?:±|\+/-)\s*('+NUM+')',s)
    if m and all(math.isfinite(float(v)) for v in m.groups()):return ('plusminus_unresolved',None,None,'SD_SE_CI_n_unknown')
    return ('unresolved',None,None,'unparsed')
def target_strain(t):
    if not is_ecoli(t):return False
    rest=t[len('Escherichia coli'):].strip()
    tokens=re.findall(r'[A-Z0-9]+',rest.upper())
    resistance={'MR','MDR','XDR','PDR','ESBL','ESBLS','PRODUCING','PRODUCER','RESISTANT','RESISTANCE','MULTIDRUG','MULTI','DRUG','CARBAPENEM','COLISTIN','SENSITIVE','SUSCEPTIBLE','STRAIN','STRAINS','ISOLATE','ISOLATES','CLINICAL'}
    return bool(tokens) and not set(tokens)<=resistance
def is_ecoli(t):return bool(re.match(r'^Escherichia coli(?:\s|$)',t,re.I))
def source_key(a):
    pm=(a.get('pubmed') or {}).get('pubmedId')
    title=re.sub(r'\W+',' ',(a.get('title') or '').casefold()).strip()
    return 'PMID:'+str(pm) if pm else ('TITLE:'+str(a.get('year'))+':'+title if title and a.get('year') else None)
result=json.loads((R/'ACQUISITION_RESULT.json').read_text())
result['query_total_count']=2198
previous_raw=json.loads((R/'PREVIOUS_RAW_PROVENANCE.json').read_text())
raw_provenance={}
records=[]
for pid in result['completed_ids']:
    source_root=R.parent/previous_raw[str(pid)]['batch'] if str(pid) in previous_raw else R
    p=source_root/'raw'/f'detail_{pid}.json';d=json.loads(p.read_text())
    raw_provenance[int(pid)]={'batch':source_root.name,'raw_path':str(p.relative_to(R.parent)),'sha256':sha(p.read_bytes())}
    if str(d['id'])!=pid:raise ValueError('ID mismatch')
    records.append((d,sha(p.read_bytes())))
peptides=[];measurements=[]
for d,raw_hash in records:
    seq=d.get('sequence') or ''
    chem={k:d.get(k) for k in ['sequence','nTerminus','cTerminus','unusualAminoAcids','intrachainBonds','interchainBonds','coordinationBonds']}
    fingerprint=sha(canonical(remove_ids(chem)).encode())
    p={'peptide_id':d['id'],'name':d.get('name'),'sequence':seq,'length':len(seq),'uppercase_backbone':seq.upper(),'canonical_after_uppercase':bool(seq) and set(seq.upper())<=set('ACDEFGHIKLMNPQRSTVWY'),'has_lowercase':any(c.islower() for c in seq),'n_terminus':d.get('nTerminus'),'c_terminus':d.get('cTerminus'),'both_termini_observed':bool(d.get('nTerminus')) and bool(d.get('cTerminus')),'annotation_fingerprint':fingerprint,'chemistry':chem,'smiles':d.get('smiles'),'synthesis_type':d.get('synthesisType'),'note_reference':d.get('noteReference'),'articles':(d.get('articles') or []),'source_keys':[source_key(a) for a in (d.get('articles') or [])],'raw_sha256':raw_hash,'chemical_identity_verified':False,'isolation_status':'NOT_IMPLEMENTED'}
    p['raw_provenance']=raw_provenance[d['id']]
    peptides.append(p)
    for a in (d.get('targetActivities') or []):
        if named(a.get('activityMeasureGroup')).upper()!='MIC':continue
        t=named(a.get('targetSpecies'));unit=named(a.get('unit'));medium=named(a.get('medium'))
        kind,lo,hi,op=parse(a.get('concentration'));ref=str(a.get('reference') or '')
        inds=[int(x) for x in re.findall(r'\d+',ref)] if re.fullmatch(r'\s*\d+(?:\s*,\s*\d+)*\s*',ref) else []
        ref_ok=bool(inds) and all(1<=i<=len(p['articles']) for i in inds)
        mapped=[p['articles'][i-1] for i in inds] if ref_ok else []
        reasons=[]
        if not is_ecoli(t):reasons.append('other_target')
        if not p['canonical_after_uppercase']:reasons.append('noncanonical_sequence')
        if kind not in ('point','left_bound','right_bound'):reasons.append(kind)
        if unit not in ('µM','μM','uM'):reasons.append('not_micromolar_no_conversion')
        if not target_strain(t):reasons.append('strain_unresolved')
        if not medium:reasons.append('medium_missing')
        if not ref_ok:reasons.append('reference_unresolved')
        measurements.append({'peptide_id':d['id'],'activity_id':a.get('id'),'sequence':seq,'annotation_fingerprint':fingerprint,'target_raw':t,'ecoli_target':is_ecoli(t),'explicit_strain_heuristic':target_strain(t),'concentration_raw':a.get('concentration'),'unit_raw':unit,'format':kind,'lower_raw_unit':lo,'upper_raw_unit':hi,'operator_or_semantics':op,'medium':medium,'cfu':a.get('cfu'),'cfu_group':a.get('cfuGroup'),'raw_activity':a,'ph':a.get('ph'),'ionic_strength':a.get('ionicStrength'),'salt':a.get('saltType'),'note':a.get('note'),'reference_raw':ref,'reference_index_valid':ref_ok,'source_keys':[source_key(x) for x in mapped],'source_records':mapped,'raw_sha256':raw_hash,'both_termini_observed':p['both_termini_observed'],'label_sequence_screen_pass':not reasons,'screen_exclusions':reasons,'chemical_identity_verified':False,'isolation_status':'NOT_IMPLEMENTED','training_admitted':False})
class Union:
    def __init__(self,n):self.p=list(range(n))
    def find(self,a):
        while self.p[a]!=a:self.p[a]=self.p[self.p[a]];a=self.p[a]
        return a
    def join(self,a,b):
        a,b=self.find(a),self.find(b)
        if a!=b:self.p[max(a,b)]=min(a,b)
def distance(a,b):
    prev=list(range(len(b)+1))
    for i,x in enumerate(a,1):
        cur=[i]
        for j,y in enumerate(b,1):cur.append(min(cur[-1]+1,prev[j]+1,prev[j-1]+(x!=y)))
        prev=cur
    return prev[-1]
fam=Union(len(peptides));edges=[]
for i,p in enumerate(peptides):
    a=p['uppercase_backbone']
    for j in range(i):
        b=peptides[j]['uppercase_backbone'];length=max(len(a),len(b))
        if not a or not b or 5*abs(len(a)-len(b))>length:continue
        dist=distance(a,b)
        if 5*dist<=length:
            fam.join(i,j);edges.append({'id1':p['peptide_id'],'id2':peptides[j]['peptide_id'],'edit_distance':dist,'identity':1-dist/length})
groups=Union(len(peptides));src={}
for i,p in enumerate(peptides):
    groups.join(i,fam.find(i))
    for key in p['source_keys']:
        if key is None:continue
        if key in src:groups.join(i,src[key])
        else:src[key]=i
ids={p['peptide_id']:i for i,p in enumerate(peptides)}
assign=[]
for i,p in enumerate(peptides):
    p.update(sequence_family='F'+str(peptides[fam.find(i)]['peptide_id']),source_family_group='G'+str(peptides[groups.find(i)]['peptide_id']))
    assign.append({k:p[k] for k in ['peptide_id','annotation_fingerprint','sequence_family','source_family_group']})
for m in measurements:
    p=peptides[ids[m['peptide_id']]];m.update(sequence_family=p['sequence_family'],source_family_group=p['source_family_group'])
coverage=collections.defaultdict(list)
for m in measurements:
    if m['ecoli_target']:coverage[(m['target_raw'],m['medium'])].append(m)
cov=[]
for (target,medium),ms in sorted(coverage.items()):
    eligible=[m for m in ms if m['label_sequence_screen_pass']]
    cov.append({'target':target,'medium':medium,'rows':len(ms),'peptide_ids':sorted({m['peptide_id'] for m in ms}),'unique_peptides':len({m['peptide_id'] for m in ms}),'unique_backbones':len({m['sequence'].upper() for m in ms}),'sequence_families':len({m['sequence_family'] for m in ms}),'source_family_groups':len({m['source_family_group'] for m in ms}),'screen_pass_rows':len(eligible),'screen_pass_peptides':len({m['peptide_id'] for m in eligible}),'screen_pass_families':len({m['sequence_family'] for m in eligible}),'screen_pass_groups':len({m['source_family_group'] for m in eligible})})
ec=[m for m in measurements if m['ecoli_target']];eligible=[m for m in ec if m['label_sequence_screen_pass']]
eligible_cells=collections.defaultdict(set)
for m in eligible:eligible_cells[(m['target_raw'],m['medium'])].add(m['peptide_id'])
crossings=[]
cell_keys=sorted(eligible_cells)
for i,a in enumerate(cell_keys):
    for b in cell_keys[:i]:
        shared=eligible_cells[a]&eligible_cells[b]
        if not shared:continue
        crossings.append({'target_a':a[0],'medium_a':a[1],'target_b':b[0],'medium_b':b[1],'same_target_different_medium':a[0]==b[0] and a[1]!=b[1],'same_medium_different_target':a[1]==b[1] and a[0]!=b[0],'shared_peptide_ids':sorted(shared),'shared_peptides':len(shared),'shared_families':len({peptides[ids[k]]['sequence_family'] for k in shared}),'shared_source_family_groups':len({peptides[ids[k]]['source_family_group'] for k in shared})})
repeat=collections.Counter((m['peptide_id'],m['target_raw']) for m in ec)
full=collections.Counter((m['peptide_id'],m['target_raw'],m['medium'],m['cfu'],canonical(m['cfu_group']),m['ph'],m['ionic_strength'],m['salt'],m['reference_raw'],m['note']) for m in ec)
def stats(ms):
    return {'rows':len(ms),'unique_peptide_ids':len({m['peptide_id'] for m in ms}),'unique_exact_case_sequences':len({m['sequence'] for m in ms}),'unique_backbones':len({m['sequence'].upper() for m in ms}),'annotation_fingerprints':len({m['annotation_fingerprint'] for m in ms}),'sequence_families':len({m['sequence_family'] for m in ms}),'source_family_groups':len({m['source_family_group'] for m in ms})}
summary={'schema':'AMP_H5_CHECKPOINT_PASS_V7','created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'acquisition_status':result['status'],'query_total_at_snapshot':result['query_total_count'],'completed_records':len(peptides),'all_MIC':stats(measurements),'E_coli_MIC':stats(ec),'draft_label_sequence_screen':stats(eligible),'draft_plus_both_termini_observed':stats([m for m in eligible if m['both_termini_observed']]),'formal_training_admitted':0,'formal_admission_reason':'Source semantics, chemical identity and contamination isolation not yet verified; this is draft preparation only.','E_coli_format_counts':dict(collections.Counter(m['format'] for m in ec)),'E_coli_units':dict(collections.Counter(m['unit_raw'] for m in ec)),'E_coli_missing':{k:sum(not m[k] for m in ec) for k in ['medium','cfu','ph','ionic_strength','salt']},'E_coli_exclusion_counts':dict(collections.Counter(s for m in ec for s in m['screen_exclusions'])),'strain_medium_cells':len(cov),'cells_with_multiple_peptides':sum(x['unique_peptides']>=2 for x in cov),'screen_cells_with_multiple_families':sum(x['screen_pass_families']>=2 for x in cov),'screen_cells_with_multiple_source_family_groups':sum(x['screen_pass_groups']>=2 for x in cov),'largest_screen_cell':max(cov,key=lambda x:x['screen_pass_peptides']) if cov else None,'repeated_peptide_target_groups':sum(n>1 for n in repeat.values()),'repeated_same_recorded_condition_reference_groups':sum(n>1 for n in full.values()),'interpretation':'Counts describe only this fixed-position API sample; groups operational, not proven independent biology. Exact same conditions/source does not certify technical replicate. Do not infer database prevalence or efficacy.'}
summary['condition_crossings']={'pairs_sharing_any_screened_peptide':len(crossings),'same_target_different_medium_pairs':sum(x['same_target_different_medium'] for x in crossings),'same_medium_different_target_pairs':sum(x['same_medium_different_target'] for x in crossings),'pairs_sharing_multiple_families':sum(x['shared_families']>=2 for x in crossings),'interpretation':'Same peptide across cells provides observed support only; other assay covariates and source may differ. No causal condition effect or sufficient statistical power claim.'}
write_csv('condition_crossings.csv',crossings,['target_a','medium_a','target_b','medium_b','same_target_different_medium','same_medium_different_target','shared_peptide_ids','shared_peptides','shared_families','shared_source_family_groups'])
for name,rows in [('peptides.jsonl',peptides),('mic_measurements.jsonl',measurements)]:
    (OUT/name).write_text(''.join(canonical(x)+'\n' for x in rows))
write_csv('mic_measurements.csv',measurements)
write_csv('family_assignments.csv',assign)
write_csv('condition_coverage.csv',cov)
put('FAMILY_EDGES.json',edges);put('SUMMARY.json',summary)
put('ISOLATION_PLAN.json',{'status':'NOT_IMPLEMENTED','test37':'Remains sealed. This script never opens any local train/dev/test data. Independent custodian may later return opaque exclusion decisions only, without sequences/labels; no such receipt exists.','requirements':['Freeze full chemical identities and reconcile duplicate aliases, D/L, termini and bonds.','Inspect analogue-family and original study relationships in addition to provisional edit-distance groups. Choose deployment-relevant similarity threshold before training; current80% is a descriptive screen.','Check overlaps against authorized train279/dev41 and every generator/scorer/pretraining corpus with available provenance; previous dev remains used, no reset.','Have independent custodian screen sealed test37 with agreed identity/homology/chemical rules and provide opaque exclusions plus hashes.','Freeze grouped train/development and independent evaluation protocol only after source/chemistry and contamination gates; no split assignments produced now.'],'external_baseline_contamination':'QMAP/BERT/LLAMP/Witten corpus and pretrained-weight overlap unresolved; no repository datasets or test partitions accessed.'})
put('RAW_PROVENANCE.json',raw_provenance)
print(json.dumps(summary,ensure_ascii=False,indent=2))
