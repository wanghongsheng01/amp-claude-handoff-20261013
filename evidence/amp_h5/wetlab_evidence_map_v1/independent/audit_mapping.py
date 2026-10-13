"""Independent output-to-source verification; no models, network or source writes."""
from pathlib import Path
import collections
import csv
import hashlib
import json
import posixpath
import re
import zipfile
import xml.etree.ElementTree as ET

S = Path('/Users/loctek/Documents/Codex/2026-10-09/task-8/amp_commsbio_20261011')
O = Path(__file__).resolve().parent
R = O.parent
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
maps = [json.loads(s) for s in (R / 'EVIDENCE_MAP_27.jsonl').read_text().splitlines()]
raw = [json.loads(s) for s in (R / 'MIC_RAW_ROWS.jsonl').read_text().splitlines()]
own = json.loads((O / 'EXISTING_SOURCE_AUDIT.json').read_text())
source_linkage = {r['sample_id']: r for r in csv.DictReader((S / 'feishu_history/ARCHIVE_SAMPLE_LINKAGE.csv').open())}
checks = []
def check(name, test): checks.append({'name': name, 'pass': bool(test)})
check('JSON_and_JSONL_identical', json.loads((R / 'EVIDENCE_MAP_27.json').read_text()) == maps)
check('27_ordered_IDs_equal_direct_workbook', [r['sample_id'] for r in maps] == [r['sample_id'] for r in own['rows']])
check('324_rows_and_27x4x3_unique_keys', len(raw) == len({(r['sample_id'], r['organism'], r['row']) for r in raw}) == 324)

ns = {'s': 'http://schemas.openxmlformats.org/spreadsheetml/2006/main',
      'r': 'http://schemas.openxmlformats.org/officeDocument/2006/relationships'}
workbook = S / 'feishu_history/downloads/results_MIC_1__MfiVbdsnlatNlysGQLrcoLuznYe.xlsx'
sheet_cells = {}
sheet_hashes = {}
with zipfile.ZipFile(workbook) as z:
    wb = ET.fromstring(z.read('xl/workbook.xml'))
    rels = {x.attrib['Id']: x.attrib['Target'] for x in ET.fromstring(z.read('xl/_rels/workbook.xml.rels'))}
    strings = [''.join(x.itertext()) for x in ET.fromstring(z.read('xl/sharedStrings.xml'))] if 'xl/sharedStrings.xml' in z.namelist() else []
    for sheet in wb.findall('s:sheets/s:sheet', ns):
        target = rels[sheet.attrib['{' + ns['r'] + '}id']]
        member = target.lstrip('/') if target.startswith('/') else posixpath.normpath('xl/' + target)
        xml = z.read(member)
        sheet_hashes[sheet.attrib['name']] = hashlib.sha256(xml).hexdigest()
        cells = {}
        for c in ET.fromstring(xml).findall('.//s:sheetData/s:row/s:c', ns):
            v = c.find('s:v', ns)
            if c.attrib.get('t') == 'inlineStr': value = ''.join(c.find('s:is', ns).itertext())
            elif c.attrib.get('t') == 's': value = strings[int(v.text)]
            else: value = v.text if v is not None else None
            cells[c.attrib['r']] = value
        sheet_cells[sheet.attrib['name']] = cells

cell_assertions = 0
def visit(value):
    global cell_assertions
    if isinstance(value, dict):
        if all(k in value for k in ['workbook', 'sheet', 'address', 'decoded', 'worksheet_sha256']):
            expected = sheet_cells[value['sheet']].get(value['address'])
            check('source_cell:' + value['sheet'] + '!' + value['address'], value['decoded'] == expected)
            check('sheet_hash:' + value['sheet'], value['worksheet_sha256'] == sheet_hashes[value['sheet']])
            cell_assertions += 1
        for v in value.values(): visit(v)
    elif isinstance(value, list):
        for v in value: visit(v)
visit(maps)
visit(raw)

columns = dict(zip(['ecoli_atcc_25922', 'paeruginosa_atcc_27853', 'saureus_atcc_29213', 'efaecalis_atcc_29212'], 'BCDE'))
byid = {r['sample_id']: r for r in maps}
raw_by_pair = collections.defaultdict(list)
for r in raw:
    raw_by_pair[(r['sample_id'], r['organism'])].append(r)
    check('raw_endpoint:' + r['sheet'] + str(r['row']), r['endpoint_MIC_raw'] == sheet_cells[r['sheet']].get('P' + str(r['row'])))
    for col in ['A', 'O']:
        check('raw_ID_cell:' + r['sheet'] + col + str(r['row']), r['sample_id_' + col] == sheet_cells[r['sheet']].get(col + str(r['row'])))

for r in maps:
    sid = r['sample_id']; old = source_linkage[sid]
    check(sid + ':sequence_and_lot', r['identity']['sequence_reported'] == old['sequence'] and r['synthesis']['lot_no'] == old['lot_no'])
    check(sid + ':no_inferred_termini', r['identity']['N_terminus'] is None and r['identity']['C_terminus'] is None and r['identity']['modification_raw'] == old['modification'])
    check(sid + ':salt_preserved', r['identity']['salt_raw'] == old['coa_comment'])
    check(sid + ':no_historical_run_imputation', r['generation_selection']['historical_checkpoint'] is None and r['generation_selection']['historical_seed'] is None and r['generation_selection']['historical_model_condition'] is None)
    check(sid + ':four_supplier_sources', set(r['synthesis']['quality_evidence']) == {'COA', 'HPLC', 'MS', 'solubility'})
    for kind, ev in r['synthesis']['quality_evidence'].items():
        check(sid + ':' + kind + ':actual_PDF_hash', sha(S / ev['source']) == ev['sha256'])
    for key, cell in r['MIC'].items():
        source_row = next(x for x in own['rows'] if x['sample_id'] == sid)
        check(sid + ':' + key + ':summary_matches_independent_read', cell['summary_raw'] == source_row['values'][columns[key]])
        linked = sorted(raw_by_pair[(sid, cell['strain'])], key=lambda x: x['row'])
        check(sid + ':' + key + ':exact_three_reported_endpoints', len(linked) == 3 and cell['reported_row_endpoints'] == [x['endpoint_MIC_raw'] for x in linked])
        check(sid + ':' + key + ':unknown_repeat_type', cell['repeat_type'].startswith('UNKNOWN'))
        check(sid + ':' + key + ':unknown_N', cell['N_semantics'] == 'UNDEFINED_NOT_NEGATIVE_NOT_RIGHT_CENSOR_NOT_IMPUTED')

counts = json.loads((R / 'COUNTS.json').read_text())
check('aggregate_counts', [counts[k] for k in ['samples', 'MIC_summary_cells', 'numeric_cells', 'N_cells']] == [27,108,30,78])
check('all_N_IDs_unchanged', counts['all_N_samples'] == own['summary']['all_N_sample_ids'])
for t in ['32', '128', '200']:
    check('threshold:' + t, counts['thresholds'][t]['any_strain_IDs'] == own['summary']['strict_thresholds'][t]['any_strain_IDs'] and counts['thresholds'][t]['Sa_IDs'] == own['summary']['strict_thresholds'][t]['S_aureus_IDs'])
check('pp2_3_version_conflict_retained', any(x.get('screenshot_raw') == 'N' and x.get('workbook_raw') == '250' for x in byid['pp2-3']['unresolved_conflicts']))
check('pp3_1_sequence_stays_uncertain', byid['pp3-1']['identity']['historical_sequence_match_status'] is None)
check('pp1_7_purity_not_auto_failure', any(x.get('meets_order') is True for x in byid['pp1-7']['unresolved_conflicts']))

docx = S / 'feishu_history/downloads/amp9.10_improved__KV0RdOukToDutyx7FLicsnQdnCh.docx'
w = {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
with zipfile.ZipFile(docx) as z:
    ps = [''.join(t.text or '' for t in p.findall('.//w:t', w)) for p in ET.fromstring(z.read('word/document.xml')).findall('.//w:p', w)]
loc = json.loads((R / 'ORIGINAL_MANUSCRIPT_LOCATORS.json').read_text())
check('original_docx_locator_text_exact', all(p['text'] == ps[p['paragraph_ordinal']-1] for p in loc['paragraphs']))
joined = re.sub(r'[\s-]', '', ps[166])
check('two_representative_sequences_in_original', all(byid[s]['identity']['sequence_reported'] in joined for s in ['pp1-5','pp2-8']))
check('original_Table3_pp1_5_kept_unrounded', byid['pp1-5']['manuscript']['individual_original_reference']['MIC_table3_raw'] == ps[249:253] == ['31','125','8','125'])
check('original_Table3_pp2_8_symbols_retained', byid['pp2-8']['manuscript']['individual_original_reference']['MIC_table3_raw'] == ps[244:248] == ['/','/','16','63'])
check('old_zero_percent_claim_not_deleted', '0%' in ps[138])
csv_rows = list(csv.DictReader((R / 'EVIDENCE_MAP_27.csv').open()))
check('CSV_all_IDs_and_values', len(csv_rows) == 27 and all(c['sample_id'] == m['sample_id'] and c['sequence'] == m['identity']['sequence_reported'] and all(c[k] == m['MIC'][k]['summary_raw'] for k in columns) for c,m in zip(csv_rows,maps)))
check('CSV_ambiguous_four_rows_header_corrected', 'four_MIC_rows_per_organism' not in csv_rows[0])

result = {'status': 'PASS' if all(x['pass'] for x in checks) else 'FAIL',
          'checks_passed': sum(x['pass'] for x in checks), 'checks_total': len(checks),
          'failed_checks': [x for x in checks if not x['pass']], 'checks': checks,
          'mapped_source_cells_verified': cell_assertions,
          'review_scope': 'Source-value, provenance, original manuscript anchors and scientific boundary; no independent MIC or chemical remeasurement',
          'source_unchanged_after_read': sha(workbook) == own['source_hashes']['feishu_history/downloads/results_MIC_1__MfiVbdsnlatNlysGQLrcoLuznYe.xlsx'],
          'output_sha256': {p.name: sha(p) for p in [R/'EVIDENCE_MAP_27.json', R/'EVIDENCE_MAP_27.jsonl', R/'EVIDENCE_MAP_27.csv', R/'MIC_RAW_ROWS.jsonl', R/'COUNTS.json', R/'GENERATION_CHAIN.json', R/'ORIGINAL_MANUSCRIPT_LOCATORS.json']}}
(O / 'MAPPING_AUDIT.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
print(json.dumps({k: result[k] for k in ['status','checks_passed','checks_total','mapped_source_cells_verified','failed_checks']}, ensure_ascii=False))
assert result['status'] == 'PASS'
