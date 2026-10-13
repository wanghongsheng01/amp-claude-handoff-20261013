"""Read-only join of authorized original AMP materials, not a model or MIC reanalysis."""
from pathlib import Path
import csv,json,hashlib,re,zipfile,xml.etree.ElementTree as ET,collections,datetime
R=Path(__file__).resolve().parent
S=Path('/Users/loctek/Documents/Codex/2026-10-09/task-8/amp_commsbio_20261011')
N={'m':'http://schemas.openxmlformats.org/spreadsheetml/2006/main','r':'http://schemas.openxmlformats.org/officeDocument/2006/relationships'}
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
source_index={}
def source(p,role):
 p=Path(p);assert p.is_relative_to(S),p
 key=str(p.relative_to(S));rec={'relative_path':key,'absolute_path':str(p),'sha256':sha(p),'bytes':p.stat().st_size,'role':role};source_index[key]=rec;return key

def read(n,role='prior_audit_reused'):
 p=S/n;source(p,role);return json.loads(p.read_text())
def dump(n,d):(R/n).write_text(json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
linkkey=source(S/'feishu_history/ARCHIVE_SAMPLE_LINKAGE.csv','audited_sample_supplier_assay_linkage')
link=list(csv.DictReader((S/linkkey).open()));assert len(link)==27
mt=read('editorial_corrections/v6r2_substantive/mic_full_table/mic_full_table.json');micro={r['sample_id']:r for r in mt['rows']}
raw=read('feishu_history/mic_new_export_audit/results/new_recovered_tables.json')
visual=read('evidence_followup_v5/mic/visual_followup.json');hist={r['sample_id_normalized']:r for r in visual['mic_screenshot']['rows']}
archive=read('feishu_history/supplier_archive/archive_manifest.json');arcpdf={r['extracted_path']:r for r in archive['records']if r['kind']=='pdf'}
quality_audit=read('feishu_history/supplier_archive/HPLC_MS_TEXT_AUDIT.json')
for n in ['evidence_followup_v5/mic/MIC_EVIDENCE_FOLLOWUP.zh.md','evidence_followup_v5/run/SEARCH_CONCLUSION.zh.md','feishu_history/process_parsed/synthesis_report/fulltext.txt','feishu_history/process_parsed/preparation_process/fulltext.txt','feishu_history/docx_parsed/amp9.10_improved.fulltext.txt','feishu_history/docx_parsed/original_33p.fulltext.txt','evidence_followup_v5/mic/historical_20240818_docx_text.txt','editorial_corrections/candidate_v6r3/source/main.tex','editorial_corrections/v6r2_substantive/mic_full_table/semantic_contract.json']:
 source(S/n,'context_and_claim_locator')
for n in ['feishu_history/downloads/synthesis_COA__HONzbADeSod325xK88hcuYkXnWd.zip','feishu_history/downloads/synthesis_report__TZaqdNdH6okIykxilPQcWdYLnSf.docx','feishu_history/downloads/preparation_process__LcP1dyOTfoKqF3xRdZuckAYtnuc.docx','feishu_history/downloads/amp9.10_improved__KV0RdOukToDutyx7FLicsnQdnCh.docx','feishu_history/downloads/20240812-AMPModel__XITwdYoxuoXWn5xycBEcFkRmnGx.docx']:
 source(S/n,'original_downloaded_container')
wb=S/'feishu_history/downloads/results_MIC_1__MfiVbdsnlatNlysGQLrcoLuznYe.xlsx';wbkey=source(wb,'original_downloaded_MIC_workbook')
assert sha(wb)==next(x['sha256']for x in mt['sources']if x['alias']=='feishu_export_xlsx')
sheets={}
with zipfile.ZipFile(wb) as z:
 shared=[]
 if 'xl/sharedStrings.xml' in z.namelist():shared=[''.join(t.itertext())for t in ET.fromstring(z.read('xl/sharedStrings.xml')).findall('m:si',N)]
 rel={x.attrib['Id']:x.attrib['Target']for x in ET.fromstring(z.read('xl/_rels/workbook.xml.rels'))}
 for sh in ET.fromstring(z.read('xl/workbook.xml')).findall('m:sheets/m:sheet',N):
  target=rel[sh.attrib['{'+N['r']+'}id']];member=target.lstrip('/') if target.startswith('/')else 'xl/'+target
  cells={}
  for c in ET.fromstring(z.read(member)).findall('.//m:sheetData/m:row/m:c',N):
   v=c.find('m:v',N);rv=v.text if v is not None else None;t=c.attrib.get('t','n')
   if t=='s':decoded=shared[int(rv)]
   elif t=='inlineStr':decoded=''.join(c.find('m:is',N).itertext())
   else:decoded=rv or ''
   cells[c.attrib['r']]={'decoded':decoded,'storage_type':t,'raw_v':rv,'formula':None if c.find('m:f',N)is None else c.find('m:f',N).text}
  sheets[sh.attrib['name']]={'member':member,'sha256':hashlib.sha256(z.read(member)).hexdigest(),'cells':cells}
def cell(ref):
 sheet,address=ref.rsplit('!',1);v=sheets[sheet]['cells'][address]
 return dict(v,workbook=wbkey,sheet=sheet,address=address,worksheet_member=sheets[sheet]['member'],worksheet_sha256=sheets[sheet]['sha256'])
# Validate every prior recovered reading against actual XLSX cells, preserving strings.
rawchecks=0
for sheet,d in raw['organisms'].items():
 for rr in d['readings']:
  for c in rr['readings'].values():assert cell(c['ref'])['decoded']==c['value'];rawchecks+=1
  assert cell(sheet+'!P'+str(rr['row']))['decoded']==rr['MIC_P'];rawchecks+=1
 for a in d['aggregates']:assert cell(a['ref'])['decoded']==a['aggregate'];rawchecks+=1
for rr in raw['summary']:
 for org,value in rr['MIC'].items():assert cell(rr['refs'][org])['decoded']==value;rawchecks+=1
# Direct DOCX XML locator, preserve all original table values and captions without changing manuscript.
docxkey='feishu_history/downloads/amp9.10_improved__KV0RdOukToDutyx7FLicsnQdnCh.docx'
W={'w':'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
with zipfile.ZipFile(S/docxkey)as z:
 pars=[''.join(p.itertext()) for p in []] # text extraction below includes w:t only
 root=ET.fromstring(z.read('word/document.xml'));pars=[''.join(t.text or '' for t in p.findall('.//w:t',W))for p in root.findall('.//w:p',W)]
# Exact paragraph ordinals are independently extracted; cached transcript is retained separately.
docclaims=[{'paragraph_ordinal':i,'text':t}for i,t in enumerate(pars,1)if ('FMPVVLTKLFGVLHFRKIY'in t or 'LKFTFIKGK'in t or '6 AMPs'in t or '1000 samples'in t or 'Table 3'in t or '128 mg/mL'in t or '32 mg/mL'in t)]
dump('ORIGINAL_MANUSCRIPT_LOCATORS.json',{'source':docxkey,'part':'word/document.xml','paragraphs':docclaims,'note':'Ordinal counts all w:p including table cells; no revised manuscript was treated as original evidence'})
codecheck=read('evidence_followup_v5/run/code_hash_recheck.json');codes=[]
for f in codecheck['files']:
 p=Path(f['path'])
 if f.get('content_scanned') and p.suffix in {'.py','.yaml','.yml','.sh','.md'}:
  assert sha(p)==f['sha256'];codes.append(source(p,'fixed_repository_static_text_verified_no_execution'))
source(S/'code_source/SOURCE_RECEIPT.json','original_repository_receipt')
repo={'repository':'https://github.com/3DHumanRehab/AMP','fixed_commit':'e7486fb714cbf5c663cc01052e42fb63d4ba3a40','verified_static_text_files':codes,'historical_run_binding':'NOT_RECOVERED_IN_AUTHORIZED_SCOPED_MATERIALS','basis':'Prior bounded run/selection search reused; code defaults and save-path strings are not an executed run. No weight loading, training, test access, or fresh repository scan.', 'historical_claimed_pool':1000,'claimed_selection_rules':['positive charge pH7.4','AMP Scanner>0.9','instability10-40','reported training similarity<=70%'],'missing_links':['run/checkpoint/config/seed to sequence pool','stable generated IDs and condition labels','per-candidate stage scores/rejections','30 selected IDs to actual generation run','lot-to-assay-well custody'],'new_H0_H5_outputs_are_historical_wetlab_provenance':False}
dump('GENERATION_CHAIN.json',repo)
revised=(S/'editorial_corrections/candidate_v6r3/source/main.tex').read_text().splitlines()
orgmap={'E.coli ATCC25922':'ecoli_atcc_25922','P.aeruginosa ATCC27853':'paeruginosa_atcc_27853','S.aureus ATCC29213':'saureus_atcc_29213','E.faecalis ATCC29212':'efaecalis_atcc_29212'}
rows=[];qc_checks=[];raw_rows=[];flat=[]
for idx,l in enumerate(link):
 sid=l['sample_id'];m=micro[sid];h=hist[sid];qc={};seq=l['sequence']
 for kind,pathfield,hashfield in [('COA','coa_path','coa_sha256'),('HPLC','hplc_path','hplc_sha256'),('MS','ms_path','ms_sha256'),('solubility','solubility_path','solubility_sha256')]:
  p=Path(l[pathfield]);a=arcpdf[str(p)];assert sha(p)==l[hashfield]==a['sha256'];pk=source(p,'supplier_original_'+kind);tp=Path(a['text_path']);tk=source(tp,'supplier_prior_text_extraction_reused');txt=tp.read_text();compact=re.sub(r'\s+','',txt)
  if kind in ['COA','solubility']:assert seq in compact
  if kind=='COA':assert re.sub(r'\s+','',l['lot_no'])in compact and 'Modification' in txt and 'N/A'in txt
  qc[kind]={'source':pk,'sha256':a['sha256'],'archive_member':a['decoded_archive_member'],'text_source':tk,'review_scope':'File hash rechecked and relevant printed identifiers/sequence checked; existing audit reused; spectra not newly independently chemically interpreted'}
  qc_checks.append({'sample_id':sid,'type':kind,'hash_match':True,'sequence_direct_text_checked':kind in ['COA','solubility']})
 measurements={}
 for sheet,d in raw['organisms'].items():
  org=d['organism'];key=orgmap[org];a=next(x for x in d['aggregates']if x['id']==sid);rr=[x for x in d['readings']if x['sample_id_A']==sid];assert len(rr)==3
  originalcell=m['source_cells']['original_xlsx'][key];exportcell=m['source_cells']['feishu_export_xlsx'][key];c=cell('all!'+exportcell['address']);assert c['decoded']==m['values'][key]==a['aggregate']
  obs=[]
  for r in rr:
   rec={'sample_id':sid,'organism':org,'sheet':sheet,'row':r['row'],'sample_id_A':r['sample_id_A'],'sample_id_O':r['sample_id_O'],'endpoint_MIC_raw':r['MIC_P'],'endpoint_cell':cell(sheet+'!P'+str(r['row'])),'dilution_and_control_readings':{col:dict(v,source_cell=cell(v['ref']))for col,v in r['readings'].items()},'repeat_identity':'UNKNOWN_TECHNICAL_VS_BIOLOGICAL; three worksheet rows are not three verified independent assays'};obs.append(rec);raw_rows.append(rec)
  measurements[key]={'strain':org,'summary_raw':c['decoded'],'unit_normalized_for_display':'µg/mL','source_unit_original_raw':'ug/ml','source_unit_export_raw':sheets['all']['cells']['A1']['decoded'],'summary_current_source':c,'summary_original_source_prior_audit':dict(originalcell,source=mt['sources'][0]),'within_workbook_aggregate':{'raw':a['aggregate'],'cell':cell(a['ref'])},'reported_row_endpoints':a['row_MIC_values'],'reported_row_endpoint_refs':a['row_MIC_refs'],'worksheet_rows':len(obs),'raw_reading_file':'MIC_RAW_ROWS.jsonl','raw_reading_key':{'sample_id':sid,'organism':org},'repeat_type':d['replicate_type'],'aggregation_rule':'UNRESOLVED_NOT_RECOMPUTED','N_semantics':'UNDEFINED_NOT_NEGATIVE_NOT_RIGHT_CENSOR_NOT_IMPUTED','same_workbook_export_not_independent_repeat':True}
 conflict=[]
 if h['current_comparison']!='same_text':conflict.append({'type':'HISTORICAL_SCREENSHOT_VS_WORKBOOK','screenshot_raw':h['visible_ecoli_mic_raw'],'workbook_raw':m['values']['ecoli_atcc_25922'],'resolution':'UNRESOLVED; do not overwrite or count versions as repeats'})
 if h.get('sequence_uncertain_residues') or h.get('sequence_equals_coa') is not True:conflict.append({'type':'SCREENSHOT_SEQUENCE_TRANSCRIPTION_UNCERTAINTY','detail':h})
 if sid=='pp1-1':conflict.append({'type':'MIC_AGGREGATION_AND_BLANK','Sa_rows':['250','125','125'],'Sa_summary':'250','Ef_rows':['250','125','125'],'Ef_summary':'125','blank_anomaly_ref':'ecoli_atcc_25922_total!M6','blank_raw':cell('ecoli_atcc_25922_total!M6')['decoded'],'resolution':'No aggregation/blank-correction rule recovered; preserve all'})
 if sid=='pp1-7':conflict.append({'type':'PURITY_EXCEPTION_NOT_AUTOMATIC_EXCLUSION','COA_purity_percent':80.3,'order_spec':'>=80.0%','meets_order':True,'cannot_claim':'all27>=95%; cannot reinterpret N based on purity'})
 manuscript={'original_downloaded_docx':docxkey,'all27_group_level_mentions':['P0133:30 screened/27 synthesized','P0139:6 and22%, label-group rates; unresolved historical claims','P0141-0163:Table1 cross-model success-rate comparison; not matched experimental control'],'individual_original_reference':None,'historical_candidate_screenshot':{'source_audit':'evidence_followup_v5/mic/visual_followup.json','image_sha256':visual['mic_screenshot']['image_sha256'],'docx_anchor':'20240818 historical DOCX word/media/image9.png; P175/body1/p94','row':h},'revised_v6r3_reference':{'source':'editorial_corrections/candidate_v6r3/source/main.tex','lines':[i for i,t in enumerate(revised,1)if t.startswith(sid+' &')],'table_label':'tab:complete_mic_summary','status':'later provenance-preserving rewrite, not original historical assay evidence'}}
 if sid=='pp1-5':manuscript['individual_original_reference']={'basis':'exact sequence after joining original linewrap; representative Broad-spectrum AMP','paragraphs':['P0167','P0178 Fig.5','P0180 Table2','P0249-0253 Table3'],'MIC_table3_raw':['31','125','8','125'],'correspondence':'Numerically compatible with rounded workbook31.25/125/7.8125/125; no documented rounding rule; keep original values','structure_binding':'Adjacent sequence/caption association only; no per-run PDB/sample validation inferred'}
 if sid=='pp2-8':manuscript['individual_original_reference']={'basis':'exact sequence FMPVVLTKLFGVLHFRKIY','paragraphs':['P0167','P0174 Fig.4','P0180 Table2','P0244-0248 Table3'],'MIC_table3_raw':['/','/','16','63'],'correspondence':'Sa/Ef approximately match15.625/62.5; / versus workbookN both undefined, not proof of Gram-positive specificity','structure_binding':'Adjacent sequence/caption association only; predicted structures are not experimental structures'}
 thresholds={str(t):[k for k,v in m['values'].items()if v!='N' and float(v)<t]for t in [32,128,200]}
 row={'schema':'AMP_ORIGINAL27_EVIDENCE_CHAIN_V1','sample_id':sid,'source_linkage_csv_row':idx+2,'identity':{'supplier_product_name':l['supplier_product_name'],'sequence_reported':seq,'length':len(seq),'supplier_length_raw':l['length_aa'],'modification_raw':l['modification'],'N_terminus':None,'C_terminus':None,'terminal_status':'NOT_EXPLICITLY_CONFIRMED; N/A modification is not a recovered Free/Amide generation condition','salt_raw':l['coa_comment'],'historical_sequence_match_status':h.get('sequence_equals_coa'),'printed_sample_ID_only_is_not_custody':True},'generation_selection':{'status':'NOT_BOUND_TO_HISTORICAL_RUN','shared_reference':'GENERATION_CHAIN.json','historical_screenshot_ID_sequence_available':True,'historical_model_condition':None,'historical_seed':None,'historical_checkpoint':None,'per_stage_screening_scores':None,'candidate30_membership_evidence':'historical screenshot only','three_extra_candidates_final_state':'outside27; no inference of synthesis failure'},'synthesis':{'order_ID':l['coa_order_id'],'lot_no':l['lot_no'],'supplier_date_raw':l['coa_date_raw'],'date_scope':l['date_type'],'gross_weight_raw':l['gross_weight_spec_and_result'],'purity_percent_reported':l['hplc_purity_percent'],'purity_spec_raw':l['hplc_purity_spec'],'HPLC_largest_peak_area_percent_reported':l['hplc_largest_peak_area_percent_reported'],'theoretical_mw_reported':l['theoretical_mw'],'MS_observed_mw_reported':l['ms_observed_mw_reported'],'COA_mw_conclusion_raw':l['coa_mw_conclusion'],'quality_evidence':qc,'purity_mass_basis':'HPLC area% is not independently certified peptide content; TFA salt/gross mass not molar dose basis','assay_lot_and_well_binding':'NOT_RECOVERED; only exact printed sample ID links separate MIC workbook'},'solubility':{k:l[k]for k in ['coa_recommended_solvent','solubility_water','solubility_dpbs_pH_7_1','solubility_dmso','solubility_date_raw']},'MIC':measurements,'manuscript':manuscript,'unresolved_conflicts':conflict,'descriptive_numeric_below_threshold':thresholds,'claim_status':{'numeric_records':'CAN_REPORT_AS_SUPPLIED_RECORDED_MIC_WITH_PROVENANCE_AND_LIMITS','activity_success_rate':'NOT_ESTABLISHED; N and rule/denominator unresolved','controlled_generation_or_model_superiority':'NOT_SUPPORTED_BY_THIS_ID_LINKAGE','independent_biological_repeats':'NOT_ESTABLISHED','new_H5_run_link':'PROHIBITED_RETROACTIVE_BINDING'}}
 rows.append(row);flat.append({'sample_id':sid,'sequence':seq,'modification_raw':l['modification'],'terminal_status':row['identity']['terminal_status'],'order_id':l['coa_order_id'],'lot':l['lot_no'],'purity_percent':l['hplc_purity_percent'],'MS_observed_mw':l['ms_observed_mw_reported'],**m['values'],'reported_worksheet_rows_per_organism':'3 worksheet rows; independence unknown','historical_run_binding':'NOT_RECOVERED','assay_lot_binding':'ID_ONLY','original_representative':sid in ['pp1-5','pp2-8'],'conflicts':json.dumps(conflict,ensure_ascii=False)})
(R/'EVIDENCE_MAP_27.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n'for r in rows));dump('EVIDENCE_MAP_27.json',rows)
(R/'MIC_RAW_ROWS.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n'for r in raw_rows))
with (R/'EVIDENCE_MAP_27.csv').open('w',newline='')as f:
 wr=csv.DictWriter(f,fieldnames=list(flat[0]));wr.writeheader();wr.writerows(flat)
dump('OUTSIDE_27_HISTORICAL_CANDIDATES.json',[v for k,v in hist.items()if k not in micro])
counts={'samples':len(rows),'MIC_summary_cells':108,'numeric_cells':sum(v!='N'for r in mt['rows']for v in r['values'].values()),'N_cells':sum(v=='N'for r in mt['rows']for v in r['values'].values()),'all_N_samples':[r['sample_id']for r in mt['rows']if all(v=='N'for v in r['values'].values())],'supplier_PDFs_verified':len(qc_checks),'raw_MIC_worksheet_rows':len(raw_rows),'XLSX_cells_checked_against_prior_audit':rawchecks,'historical_run_bound_samples':0,'assay_lot_bound_samples':0,'explicit_terminal_confirmed_samples':0,'original_sequence_identified_representatives':['pp1-5','pp2-8'],'thresholds':{str(t):{'any_strain_IDs':[r['sample_id']for r in rows if r['descriptive_numeric_below_threshold'][str(t)]],'Sa_IDs':[r['sample_id']for r in rows if 'saureus_atcc_29213'in r['descriptive_numeric_below_threshold'][str(t)]]}for t in [32,128,200]},'N_and_blank_unchanged':True,'no_training':True,'no_test37':True,'scope':'Evidence present and partial links verified; absent bindings mean not recovered here, never proof of no wet experiment'}
dump('COUNTS.json',counts);dump('QC_CHECKS.json',qc_checks);dump('SOURCE_INDEX.json',source_index)
dump('SOURCE_PRESERVATION.json',{'source_files':len(source_index),'all_hashes_unchanged':all(sha(Path(v['absolute_path']))==v['sha256']for v in source_index.values()),'no_source_writes':True,'no_other_personal_file_search':True})
print(json.dumps(counts,ensure_ascii=False))
