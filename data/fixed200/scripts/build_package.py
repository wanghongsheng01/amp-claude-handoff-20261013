import collections,csv,datetime,hashlib,importlib.util,json,shutil
from pathlib import Path
R=Path(__file__).resolve().parent;H=R.parent;V=H/'dbaasp_checkpoint_v7';P=R/'package'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def readl(p):return [json.loads(l) for l in p.read_text().splitlines()]
def put(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n')
def jsonl(p,rs):p.write_text(''.join(json.dumps(r,ensure_ascii=False,separators=(',',':'))+'\n' for r in rs))
def csvout(p,rows):
 with p.open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader()
  for r in rows:w.writerow({k:json.dumps(v,ensure_ascii=False) if isinstance(v,(dict,list)) else v for k,v in r.items()})
ids=json.loads((H/'dbaasp_bounded_v3/FROZEN_IDS.json').read_text())['ids'];rawmap=json.loads((V/'prepared/RAW_PROVENANCE.json').read_text());ps=readl(V/'prepared/peptides.jsonl');ms=readl(V/'prepared/mic_measurements.jsonl');ns=readl(V/'analysis/NATIVE_ROWS.jsonl');aud=readl(V/'analysis/CHEMISTRY.jsonl');A={a['peptide_id']:a for a in aud}
rawrows=[]
for pid in ids:
 obj=rawmap[pid];path=H/obj['raw_path'];raw=path.read_bytes();d=json.loads(raw)
 assert str(d['id'])==pid and d['complexity']['name'].lower()=='monomer' and sha(path)==obj['sha256']
 assert isinstance(d['targetActivities'],list) and isinstance(d['articles'],list)
 rawrows.append({'peptide_id':int(pid),'original_raw_path':obj['raw_path'],'sha256':obj['sha256'],'bytes':len(raw),'url':'https://dbaasp.org/peptides/'+pid,'id_verified':True,'monomer_verified':True,'parse_complete':True,'article_count':len(d['articles']),'activity_count':len(d['targetActivities'])})
assert len(rawrows)==200 and {p['peptide_id'] for p in ps}=={int(i) for i in ids}
csvout(P/'evidence/RAW_200_MANIFEST.csv',rawrows)
pp=[]
for p in ps:
 pp.append({k:p[k] for k in ['peptide_id','sequence','annotation_fingerprint','chemistry','note_reference','articles','raw_sha256']}|{'frozen_structure_status':A[p['peptide_id']]['structure_status']})
fields=['peptide_id','activity_id','target_raw','concentration_raw','unit_raw','medium','cfu','cfu_group','ph','ionic_strength','salt','note','reference_raw','raw_sha256']
mi=[{k:m[k] for k in fields} for m in ms];jsonl(P/'data/peptide_inputs.jsonl',pp);jsonl(P/'data/measurement_inputs.jsonl',mi)
for src,dst in [('SOURCE_INDEX.csv','data/SOURCE_INDEX.csv'),('MEASUREMENT_SOURCE_INDEX.csv','data/MEASUREMENT_SOURCE_INDEX.csv'),('ALL_200_COVERAGE.csv','evidence/ALL_200_COVERAGE.csv'),('analysis/CHEMISTRY.jsonl','evidence/FROZEN_CHEMISTRY_AUDIT.jsonl'),('analysis/COUNTS.json','evidence/FINAL_COUNTS.json'),('independent/POSTAUDIT.json','evidence/CHECKPOINT_INDEPENDENT_AUDIT.json')]:shutil.copyfile(V/src,P/dst)
for n in ['prepare_coverage.py','analyze_native.py','chemistry_engine.py']:shutil.copyfile(V/n,P/'scripts'/('frozen_'+n))
spec=importlib.util.spec_from_file_location('clean_data',P/'scripts/clean_data.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);pps,rebuild=mod.build();N={(r['peptide_id'],r['activity_id']):r for r in ns}
assert len(rebuild)==len(ms)==1788
for m,r in zip(ms,rebuild):
 for k in ['peptide_id','activity_id','format','lower_raw_unit','upper_raw_unit','operator_or_semantics','reference_index_valid','source_keys','sequence_family','source_family_group']:assert m[k]==r[k],(m['activity_id'],k,m[k],r[k])
 if r['ecoli_target']:
  n=N[(r['peptide_id'],r['activity_id'])]
  for k in ['native_unit_group','core_reasons','plain_Witten_encoding_compatible','plain_Witten_primary_point_eligible','plain_Witten_censor_eligible']:assert n[k]==r[k],(m['activity_id'],k)
mod.emit(P/'data/all_MIC.csv',rebuild);mod.emit(P/'data/Ecoli_native.csv',[r for r in rebuild if r['ecoli_target']]);mod.emit(P/'data/peptide_groups.csv',[{k:p[k] for k in ['peptide_id','sequence','sequence_family','source_family_group']} for p in pps])
counts={}
for size,folder in [(136,'dbaasp_coverage_v5'),(146,'dbaasp_checkpoint_v6'),(200,'dbaasp_checkpoint_v7')]:counts[size]=json.loads((H/folder/'analysis/COUNTS.json').read_text())
metrics=[]
for group in ['all_peptides','all_MIC','Ecoli_MIC']:
 for key in ['rows','peptide_IDs','exact_sequences','families','source_family_components']:
  vs=[counts[n][group][key] for n in [136,146,200]];metrics.append(dict(metric=group+'.'+key,snapshot136=vs[0],snapshot146=vs[1],final200=vs[2],delta_vs136=vs[2]-vs[0],delta_vs146=vs[2]-vs[1]))
for u in ['ug_ml_native','uM_native']:
 for fmt in ['point','bounds','point_plus_bound']:
  for key in ['rows','peptide_IDs','families','source_family_components']:
   vs=[counts[n]['native_units'][u][fmt][key] for n in [136,146,200]];metrics.append(dict(metric=u+'.'+fmt+'.'+key,snapshot136=vs[0],snapshot146=vs[1],final200=vs[2],delta_vs136=vs[2]-vs[0],delta_vs146=vs[2]-vs[1]))
csvout(P/'data/SNAPSHOT_DELTAS.csv',metrics)
sem={'all_MIC_formats':dict(collections.Counter(m['format'] for m in ms)),'Ecoli_formats':dict(collections.Counter(n['format'] for n in ns)),'Ecoli_raw_units_formats':{u:dict(collections.Counter(n['format'] for n in ns if n['unit_raw']==u)) for u in sorted({n['unit_raw'] for n in ns})},'eligible_native_unit_formats':{u:{fmt:dict(collections.Counter(n['format'] for n in ns if n['native_unit_group']==u and n[flag])) for fmt,flag in [('point','plain_Witten_primary_point_eligible'),('censor','plain_Witten_censor_eligible')]} for u in ['ug_ml_native','uM_native']},'one_sided_operators':dict(collections.Counter(m['operator_or_semantics'] for m in ms if m['format'] in ['right_bound','left_bound'])),'native_point_missing_covariates':{u:d['missing_point_covariates'] for u,d in counts[200]['native_units'].items()},'interpretation':'Keep raw strings/operators; no range midpoint, interval-censor inference, plus/minus SD/SE guess, log transformation, or mass-to-molar conversion.'}
put(P/'evidence/SEMANTICS_AUDIT.json',sem)
unresolved=[]
for m in ms:
 if not m['reference_index_valid']:
  p=next(p for p in ps if p['peptide_id']==m['peptide_id'])
  unresolved.append({'peptide_id':m['peptide_id'],'activity_id':m['activity_id'],'target':m['target_raw'],'concentration_raw':m['concentration_raw'],'unit_raw':m['unit_raw'],'reference_raw':m['reference_raw'],'format':m['format'],'candidate_article_metadata':p['articles'],'decision':'REMAIN_UNRESOLVED_NO_IMPUTATION','reason':'Empty measurement reference; one peptide-level article alone does not establish measurement-level provenance. No cached original paper evidence found by independent check. Range remains unresolved; not converted to point or verified interval censor.'})
put(P/'evidence/UNRESOLVED_REFERENCES.json',unresolved)
put(P/'evidence/VALIDATION.json',{'raw200_integrity':'PASS','raw_sha256_verified':200,'raw_complete_JSON_ID_monomer_verified':200,'all_MIC_rows':1788,'native_rows':318,'standalone_cleaner_matches_frozen_preparation_and_native_eligibility':True,'training':0,'new_data_requests':0,'test37_reads':0,'NO_GO':'UNCHANGED','original27_origin_and_selection':'AUTHOR_RESPONSE_PENDING_EVIDENCE_GRADE_UNCHANGED','source_hashes':{str(p.relative_to(H)):sha(p) for p in [V/'prepared/peptides.jsonl',V/'prepared/mic_measurements.jsonl',V/'analysis/NATIVE_ROWS.jsonl',V/'analysis/CHEMISTRY.jsonl',V/'FINAL_MANIFEST.json',H/'dbaasp_coverage_v5/analysis/COUNTS.json',H/'dbaasp_checkpoint_v6/analysis/COUNTS.json',H/'dbaasp_feasibility_v1/OFFICIAL_DOCS_TOOL_RESPONSE.txt']}})
license_text='''DBAASP DATA ATTRIBUTION / 数据来源与使用范围
Data obtained from DBAASP, https://dbaasp.org, public API detail snapshots acquired 2026-10-12. Public database records were minimally transformed for this private AMP H5 data audit.
The locally cached official Data Access Policy (https://dbaasp.org/terms-and-conditions, checked 2026-10-12) permits access, download, copying, use, adaptation and redistribution, with acknowledgement of DBAASP. This is the database's explicit policy; we do not assign an invented Creative Commons license.
DBAASP is supported by IBCEB (Tbilisi) and NIAID OCICB (Bethesda); its team collects/submits the database records. Suggested database citation: Pirtskhalava M et al. DBAASP v3. Nucleic Acids Research 49(D1), D288-D297 (2021). https://doi.org/10.1093/nar/gkaa991
This package contains public database-derived minimal analysis tables and citation metadata, not third-party full-text PDFs. Rights to linked publications are separate. Original response bytes remain in the local read-only provenance chain; this package provides their SHA256 inventory, not another raw-data scrape.
Private Library delivery only. No public upload, production use or permission to relabel these data as biological validation. Original cached policy evidence SHA256 is recorded in evidence/VALIDATION.json.
'''
(P/'DBAASP_ATTRIBUTION.txt').write_text(license_text)
readme='''AMP H5 — FINAL FIXED200 DATA ADMISSION RECEIPT (2026-10-12)
Read APPENDIX_1PAGE.pdf first, then data/SNAPSHOT_DELTAS.csv and DATA_DICTIONARY.csv.
Decision: descriptive-data handoff complete; formal model admission HOLD and scientific NO_GO unchanged. More data alone do not rescue failed EB-context results. Original27 generation origin/selection awaits the author's response; no evidence upgrade.
The fixed positional200 is not a random/representative whole-database sample. There are200 peptide IDs,192 exact-case sequences,141 descriptive80% edit-distance families and68 source-family components; these components are not certified independent biology.
All1788 MIC rows are repeated observations (165 peptide IDs). Ecoli318 rows cover158 IDs. Native unit groups remain separate; NEVER convert mass to molarity, pool units, midpoint ranges, guess SD/SE, or treat MIC rows as independent peptides.
Data inputs preserve case, missing values, raw concentration/reference strings and reported conditions. Two empty references stay unresolved; one is Ecoli. Absence of terminal annotations means unknown, not Free/Amide. No missing-field imputation.
Reproduce tables with Python3 stdlib only, from this directory:
  python3 scripts/clean_data.py --output rebuilt
This recomputes string parsing, label eligibility and original grouping rules; frozen molecular audit status is an explicit upstream input (evidence/FROZEN_CHEMISTRY_AUDIT.jsonl). No molecular audit or model run is concealed. frozen_*.py are unchanged upstream author provenance copies and retain original workspace layout/dependencies; standalone reproduction uses clean_data.py.
The package excludes third-party PDFs, credentials, model weights, train/dev/test sequences, and acquisition executable scripts. Existing200 public database sequences are authorized public data; they are not a new leakage-screened split. test37 was never opened. All earlier failed/negative experiments and acquisition failures remain in their original local files.
See DBAASP_ATTRIBUTION.txt for source policy and citation. Private Library attachment only; no public release. HASHES.sha256 covers packaged files except itself.
'''
(P/'README.txt').write_text(readme)
desc={'peptide_id':'DBAASP peptide record identifier; not a certified independent biological replicate','activity_id':'DBAASP target activity identifier; pair with peptide_id','sequence':'Exact reported case preserved; lowercase not silently converted for primary eligibility','annotation_fingerprint':'Frozen SHA256 of identity-related chemical annotations','chemistry':'Reported termini/unusual residues/bonds; absent means unknown','note_reference':'Peptide-level annotation; not an activity-level source mapping','articles':'Ordered reported citation metadata; reference1 maps to first article only when explicitly present','raw_sha256':'SHA256 of exact locally cached raw detail response','frozen_structure_status':'Frozen represented-chemistry audit verdict; PENDING is not an efficacy or molarity claim','target_raw':'Reported target species/strain text','concentration_raw':'Exact reported concentration string','unit_raw':'Exact reported unit; no mass/molar conversion','reference_raw':'Reported 1-based comma-separated citation indices; blank stays unresolved','format':'point/right_bound/left_bound/range_unresolved/plusminus_unresolved/unresolved','lower_raw_unit':'Parsed lower endpoint in original unit; range endpoint is not validated interval censoring','upper_raw_unit':'Parsed upper endpoint in original unit','operator_or_semantics':'Original strict/non-strict operator retained; range/plusminus uncertainty explicit','native_unit_group':'ug_ml_native only µg/ml; uM_native only µM/μM/uM; otherwise unsupported','ecoli_target':'Regex ^Escherichia coli followed by whitespace/end','reference_index_valid':'Explicit nonempty indices all within this peptide article array','source_keys':'PMID or year+normalized title for explicitly mapped references; no guessed source','sequence_family':'Connected component of >=80% normalized edit identity on uppercased backbones','source_family_group':'Family component further linked by peptide-level publication identities','structure_status':'Copied frozen molecular-audit status','core_reasons':'Reasons that bar native label eligibility, in frozen rule order','plain_Witten_encoding_compatible':'Exact uppercase canonical20AA and length1..99, not biological validity','plain_Witten_primary_point_eligible':'Ecoli + recognized unit + source/chemistry/core rules + compatible point; not formal training admission','plain_Witten_censor_eligible':'Same rules for one-sided bounds; bounds not exact labels','formal_training_admitted':'Always false: no new split/leakage/biological validation established','measurement_row_id':'Local row index; never an independent peptide count','medium':'Raw culture medium; whitespace-only means missing','cfu':'Raw inoculum value, no inferred unit or conversion','cfu_group':'Reported inoculum category object','ph':'Reported pH; blank means missing','ionic_strength':'Raw ionic strength; blank means missing','salt':'Raw salt annotation; blank means missing','note':'Reported activity note; no inference beyond frozen admission rules'}
dictrows=[]
for table,rows in [('peptide_inputs.jsonl',pp),('measurement_inputs.jsonl',mi),('all_MIC.csv',rebuild)]:
 for k in rows[0]:
  types=sorted({type(r[k]).__name__ for r in rows});dictrows.append({'table':table,'field':k,'json_value_types':','.join(types),'meaning':desc[k],'missing_rule':'Preserve raw null/blank; whitespace-only is missing for covariate counts; never impute'})
# Source indexes retain one row per measurement vs one row per normalized publication identity.
for table in ['SOURCE_INDEX.csv','MEASUREMENT_SOURCE_INDEX.csv']:
 with (P/'data'/table).open() as f:keys=next(csv.reader(f))
 for k in keys:dictrows.append({'table':table,'field':k,'json_value_types':'CSV scalar or JSON-array/object cell','meaning':{'source_index_key':'Unique source-row key used by measurement index','source_index_keys':'Zero or more source-row keys; empty unresolved references remain empty','independent_peptide_observation':'Always False; repeated MIC rows are not independent peptides','raw_path':'Relative local raw-cache path, bytes not bundled','dbaasp_detail_url':'Public record URL; no request is performed by this package','MIC_rows':'Count of measurement rows associated with source; not independent peptide count','unique_peptide_ids':'Distinct DBAASP record count associated with source'}.get(k,desc.get(k,'Reported source metadata or explicit list/count of linked IDs; see source and measurement index join keys')),'missing_rule':'Blank/null preserved; not guessed'})
csvout(P/'DATA_DICTIONARY.csv',dictrows)
print(json.dumps({'raw_verified':200,'rows':len(rebuild),'unresolved_references':len(unresolved),'cleaner_matches':True,'package_files':sum(p.is_file() for p in P.rglob('*'))}))
