"""Offline author cleaning script. Python stdlib only; no HTTP, models, or unit conversion.
Rebuilds measurement semantics/grouping from minimal public DBAASP input tables.
Molecular audit status is a frozen upstream input, not recomputed here.
Usage: python3 scripts/clean_data.py --output rebuilt
"""
import argparse,collections,csv,json,math,re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
NUM=r'(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?'
def parse(value):
 s=str(value or '').strip().replace('−','-').replace('–','-').replace('≥','>=').replace('≤','<=')
 if re.fullmatch(NUM,s):
  v=float(s);return ('point',v,v,'point_as_reported') if math.isfinite(v) and v>0 else ('unresolved',None,None,'nonpositive_or_nonfinite')
 m=re.fullmatch(r'(>=|<=|>|<)\s*('+NUM+')',s)
 if m and math.isfinite(float(m[2])) and float(m[2])>0:
  op,v=m[1],float(m[2]);return ('right_bound',v,None,op) if op.startswith('>') else ('left_bound',None,v,op)
 m=re.fullmatch('('+NUM+r')\s*-\s*('+NUM+')',s)
 if m and math.isfinite(float(m[2])) and 0<float(m[1])<=float(m[2]):return ('range_unresolved',float(m[1]),float(m[2]),'semantic_not_verified')
 m=re.fullmatch('('+NUM+r')\s*(?:±|\+/-)\s*('+NUM+')',s)
 if m and all(math.isfinite(float(v)) for v in m.groups()):return ('plusminus_unresolved',None,None,'SD_SE_CI_n_unknown')
 return ('unresolved',None,None,'unparsed')
def distance(a,b):
 prev=list(range(len(b)+1))
 for i,x in enumerate(a,1):
  cur=[i]
  for j,y in enumerate(b,1):cur.append(min(cur[-1]+1,prev[j]+1,prev[j-1]+(x!=y)))
  prev=cur
 return prev[-1]
class Union:
 def __init__(self,n):self.p=list(range(n))
 def find(self,i):
  while self.p[i]!=i:self.p[i]=self.p[self.p[i]];i=self.p[i]
  return i
 def join(self,a,b):
  a,b=self.find(a),self.find(b)
  if a!=b:self.p[max(a,b)]=min(a,b)
def named(x):return str((x or {}).get('name') or '')
def source_key(a):
 pm=(a.get('pubmed') or {}).get('pubmedId');title=re.sub(r'\W+',' ',(a.get('title') or '').casefold()).strip()
 return 'PMID:'+str(pm) if pm else ('TITLE:'+str(a.get('year'))+':'+title if title and a.get('year') else None)
def emit(path,rows):
 with path.open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader()
  for r in rows:w.writerow({k:json.dumps(v,ensure_ascii=False,separators=(',',':')) if isinstance(v,(list,dict)) else v for k,v in r.items()})
def build():
 ps=[json.loads(l) for l in (ROOT/'data/peptide_inputs.jsonl').read_text().splitlines()]
 inputs=[json.loads(l) for l in (ROOT/'data/measurement_inputs.jsonl').read_text().splitlines()]
 f=Union(len(ps));g=Union(len(ps))
 for i,p in enumerate(ps):
  a=p['sequence'].upper()
  for j,q in enumerate(ps[:i]):
   b=q['sequence'].upper();length=max(len(a),len(b))
   if a and b and 5*abs(len(a)-len(b))<=length and 5*distance(a,b)<=length:f.join(i,j)
 src={}
 for i,p in enumerate(ps):
  g.join(i,f.find(i))
  for a in p['articles']:
   k=source_key(a)
   if k is not None:
    if k in src:g.join(i,src[k])
    else:src[k]=i
 for i,p in enumerate(ps):p.update(sequence_family='F'+str(ps[f.find(i)]['peptide_id']),source_family_group='G'+str(ps[g.find(i)]['peptide_id']))
 P={p['peptide_id']:p for p in ps};rows=[]
 for idx,m in enumerate(inputs,1):
  p=P[m['peptide_id']];kind,lo,hi,op=parse(m['concentration_raw']);ref=m['reference_raw'];inds=[int(x) for x in re.findall(r'\d+',ref)] if re.fullmatch(r'\s*\d+(?:\s*,\s*\d+)*\s*',ref) else []
  valid=bool(inds) and all(1<=i<=len(p['articles']) for i in inds)
  ecoli=bool(re.match(r'^Escherichia coli(?:\s|$)',m['target_raw'],re.I))
  unit='ug_ml_native' if m['unit_raw']=='µg/ml' else 'uM_native' if m['unit_raw'] in ['µM','μM','uM'] else 'unsupported'
  why=[]
  if unit=='unsupported':why.append('UNSUPPORTED_NATIVE_UNIT')
  if kind not in ['point','right_bound','left_bound']:why.append('UNRESOLVED_RANGE_PLUSMINUS_OR_PARSE')
  if not valid:why.append('SOURCE_ACTIVITY_LINK_UNRESOLVED')
  if not p['sequence']:why.append('EMPTY_SEQUENCE')
  if p['frozen_structure_status']=='REFUSE':why.append('EXPLICIT_CHEMISTRY_CONFLICT_OR_UNSUPPORTED_COMPONENTS')
  c=p['chemistry']
  if c.get('interchainBonds') or c.get('coordinationBonds'):why.append('EXPLICIT_MULTICOMPONENT_OR_COORDINATION_ATTRIBUTION_PENDING')
  if re.search(r'part\s+of.{0,80}(dimer|two[\s-]*peptide|multimer|complex)',str(p['note_reference'] or ''),re.I):why.append('PART_OF_MULTICOMPONENT_ENTITY_ATTRIBUTION_PENDING')
  plain=bool(p['sequence']) and set(p['sequence'])<=set('ACDEFGHIKLMNPQRSTVWY') and 1<=len(p['sequence'])<100
  rows.append(dict(m,measurement_row_id=idx,sequence=p['sequence'],format=kind,lower_raw_unit=lo,upper_raw_unit=hi,operator_or_semantics=op,ecoli_target=ecoli,native_unit_group=unit,reference_index_valid=valid,source_keys=[source_key(p['articles'][i-1]) for i in inds] if valid else [],sequence_family=p['sequence_family'],source_family_group=p['source_family_group'],structure_status=p['frozen_structure_status'],core_reasons=why,plain_Witten_encoding_compatible=plain,plain_Witten_primary_point_eligible=ecoli and not why and plain and kind=='point',plain_Witten_censor_eligible=ecoli and not why and plain and kind in ['right_bound','left_bound'],formal_training_admitted=False))
 return ps,rows
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--output',required=True);args=ap.parse_args();out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
 ps,rows=build();emit(out/'all_MIC.csv',rows);emit(out/'Ecoli_native.csv',[r for r in rows if r['ecoli_target']]);emit(out/'peptide_groups.csv',[{k:p[k] for k in ['peptide_id','sequence','sequence_family','source_family_group']} for p in ps])
 print(json.dumps({'peptide_ids':len(ps),'MIC_rows':len(rows),'Ecoli_rows':sum(r['ecoli_target'] for r in rows),'training':0}))
