"""Small, offline, saved-artifact verification. No training or data acquisition."""
from pathlib import Path
import json,csv
R=Path(__file__).resolve().parent
e=R/'evidence'
for label,correct in [('CE',497),('MSE',88),('scalar',111)]:
    rows=json.loads((e/f'dev_reconstruction_{label}.json').read_text())
    assert len(rows)==41
    assert sum(r['true_length'] for r in rows)==701
    assert sum(r['correct_residues'] for r in rows)==correct
gen=list(csv.DictReader((e/'generation_all_distances.csv').open()))
lookup={(x['method'],x['terminal'],x['reference'],x['metric']):float(x['value']) for x in gen}
contrasts=0
for k,v in lookup.items():
    if k[0]!='CE21_DDPM':continue
    assert lookup[('MSE21_DDPM',)+k[1:]]>v
    assert lookup[('terminal_LSTM',)+k[1:]]<v
    contrasts+=1
assert contrasts==24
hist=json.loads((e/'historical_27_transcription.json').read_text())
assert len(hist)==27
assert sum(v=='N' for x in hist for v in x['MIC'].values())==78
assert sum(v!='N' for x in hist for v in x['MIC'].values())==30
assert [sum(any(v!='N' and float(v)<t for v in x['MIC'].values()) for x in hist) for t in [32,128,200]]==[6,7,7]
print('PASS: saved reconstruction counts, 24 matched distance orderings, and complete historical MIC counts. No independent experimental confirmation implied.')
