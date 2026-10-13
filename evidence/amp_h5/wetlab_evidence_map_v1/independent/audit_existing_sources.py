"""Bounded independent wet-lab record check; read-only source, stdlib only."""
from pathlib import Path
from decimal import Decimal
import csv
import hashlib
import json
import posixpath
import zipfile
import xml.etree.ElementTree as ET

S = Path('/Users/loctek/Documents/Codex/2026-10-09/task-8/amp_commsbio_20261011')
OUT = Path(__file__).resolve().parent
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
workbook = S / 'feishu_history/downloads/results_MIC_1__MfiVbdsnlatNlysGQLrcoLuznYe.xlsx'
before = sha(workbook)
ns = {'s': 'http://schemas.openxmlformats.org/spreadsheetml/2006/main',
      'r': 'http://schemas.openxmlformats.org/officeDocument/2006/relationships'}
with zipfile.ZipFile(workbook) as z:
    wb = ET.fromstring(z.read('xl/workbook.xml'))
    relations = {x.attrib['Id']: x.attrib['Target'] for x in ET.fromstring(z.read('xl/_rels/workbook.xml.rels'))}
    shared = []
    if 'xl/sharedStrings.xml' in z.namelist():
        shared = [''.join(x.itertext()) for x in ET.fromstring(z.read('xl/sharedStrings.xml'))]
    sheet = next(x for x in wb.findall('s:sheets/s:sheet', ns) if x.attrib['name'] == 'all')
    target = relations[sheet.attrib['{' + ns['r'] + '}id']]
    member = target.lstrip('/') if target.startswith('/') else posixpath.normpath('xl/' + target)
    raw_sheet = z.read(member)
    cells = {}
    for cell in ET.fromstring(raw_sheet).findall('.//s:sheetData/s:row/s:c', ns):
        typ = cell.attrib.get('t')
        v = cell.find('s:v', ns)
        if typ == 'inlineStr':
            value = ''.join(cell.find('s:is', ns).itertext())
        elif typ == 's':
            value = shared[int(v.text)]
        else:
            value = v.text if v is not None else ''
        cells[cell.attrib['r']] = value

rows = [{'sample_id': cells['A' + str(i)], 'worksheet_row': i,
         'values': {c: cells[c + str(i)] for c in 'BCDE'},
         'source_cells': {c: 'all!' + c + str(i) for c in 'ABCDE'}} for i in range(5, 32)]
existing_table = list(csv.DictReader((S / 'editorial_corrections/v6r2_substantive/mic_full_table/mic_full_table.csv').open()))
linkage = list(csv.DictReader((S / 'feishu_history/ARCHIVE_SAMPLE_LINKAGE.csv').open()))
link = {r['sample_id']: r for r in linkage}
names = ['ecoli_atcc_25922', 'paeruginosa_atcc_27853', 'saureus_atcc_29213', 'efaecalis_atcc_29212']
checks = []
def check(name, test): checks.append({'name': name, 'pass': bool(test)})
check('27_unique_IDs', len(rows) == len({r['sample_id'] for r in rows}) == 27)
check('existing_frozen_CSV_exact_transcription', all(a['sample_id'] == b['sample_id'] and
      all(a['values'][c] == b[n] for c, n in zip('BCDE', names)) for a, b in zip(rows, existing_table)))
check('27_supplier_ID_set_matches', {r['sample_id'] for r in rows} == set(link) and len(linkage) == 27)
check('all_modification_fields_NA_not_Free', all(r['modification'] == 'N/A' for r in linkage))
check('all_supplier_comments_TFA_salt', all(r['coa_comment'] == 'TFA salt' for r in linkage))
check('source_unchanged', before == sha(workbook))
summary = {'candidate_rows': len(rows), 'numeric_cells': sum(v != 'N' for r in rows for v in r['values'].values()),
           'undefined_N_cells': sum(v == 'N' for r in rows for v in r['values'].values()),
           'all_N_sample_ids': [r['sample_id'] for r in rows if all(v == 'N' for v in r['values'].values())],
           'numeric_by_column': {c: sum(r['values'][c] != 'N' for r in rows) for c in 'BCDE'},
           'strict_thresholds': {}}
for threshold in ['32', '128', '200']:
    selected = [r['sample_id'] for r in rows if any(v != 'N' and Decimal(v) < Decimal(threshold) for v in r['values'].values())]
    sa = [r['sample_id'] for r in rows if r['values']['D'] != 'N' and Decimal(r['values']['D']) < Decimal(threshold)]
    summary['strict_thresholds'][threshold] = {'any_strain_IDs': selected, 'S_aureus_IDs': sa}
check('numeric30_N78_allN16', summary['numeric_cells'] == 30 and summary['undefined_N_cells'] == 78 and len(summary['all_N_sample_ids']) == 16)
check('threshold_6_7_7_any_and6_sa', [len(summary['strict_thresholds'][t]['any_strain_IDs']) for t in ['32', '128', '200']] == [6, 7, 7] and
      all(len(summary['strict_thresholds'][t]['S_aureus_IDs']) == 6 for t in ['32', '128', '200']))

sources = [
 'feishu_history/downloads/results_MIC_1__MfiVbdsnlatNlysGQLrcoLuznYe.xlsx',
 'editorial_corrections/v6r2_substantive/mic_full_table/mic_full_table.csv',
 'editorial_corrections/v6r2_substantive/mic_full_table/independent_review/REVIEW.zh.md',
 'feishu_history/ARCHIVE_SAMPLE_LINKAGE.csv', 'feishu_history/ARCHIVE_EVIDENCE_AUDIT.zh.md',
 'feishu_history/PROCESS_RECORD_AUDIT.zh.md', 'feishu_history/MIC_NEW_EXPORT_AUDIT.zh.md',
 'feishu_history/supplier_archive/HPLC_MS_TEXT_AUDIT.zh.md',
 'evidence_followup_v5/mic/VISUAL_FOLLOWUP.zh.md',
 'evidence_followup_v5/mic/source_images/20240818-image9.png',
 'evidence_followup_v5/run/SEARCH_CONCLUSION.zh.md',
 'evidence_followup_v5/run/CURATED_EVIDENCE.zh.md',
 'evidence_followup_v5/mic/historical_20240818_docx_text.txt',
 'feishu_history/docx_parsed/original_33p.fulltext.txt',
 'review_checkpoint_v6r3_20261011/CHECKPOINT.zh.md',
]
result = {'status': 'PASS' if all(c['pass'] for c in checks) else 'FAIL', 'checks': checks,
          'source_root': str(S), 'source_hashes': {p: sha(S / p) for p in sources},
          'direct_source_parse': {'workbook': str(workbook), 'member': member,
                                  'worksheet_sha256': hashlib.sha256(raw_sheet).hexdigest(),
                                  'header_verbatim': {c: cells[c + '1'] for c in 'ABCDE'}},
          'rows': rows, 'summary': summary,
          'scope': 'Direct read of one supplied workbook all!A5:E31 plus prior source audits and linkage; no new wet experiment, model or full re-audit of108PDFs',
          'image_read_this_review': {'path': 'evidence_followup_v5/mic/source_images/20240818-image9.png',
                                     'observed_pp2_3_ecoli': 'N', 'does_not_resolve_version_conflict': True}}
(OUT / 'EXISTING_SOURCE_AUDIT.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
print(json.dumps({'status': result['status'], 'checks': len(checks), 'summary': summary}, ensure_ascii=False))
assert result['status'] == 'PASS'
