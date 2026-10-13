"""Independent cached-record formula inventory; no natural-model execution or network."""
import json,re,hashlib,collections
from pathlib import Path
from rdkit import Chem,rdBase
from rdkit.Chem import Descriptors,rdMolDescriptors
ROOT=Path(__file__).resolve().parents[1]
AA={'A':'C3H7NO2','R':'C6H14N4O2','N':'C4H8N2O3','D':'C4H7NO4','C':'C3H7NO2S','E':'C5H9NO4','Q':'C5H10N2O3','G':'C2H5NO2','H':'C6H9N3O2','I':'C6H13NO2','L':'C6H13NO2','K':'C6H14N2O2','M':'C5H11NO2S','F':'C9H11NO2','P':'C5H9NO2','S':'C3H7NO3','T':'C4H9NO3','W':'C11H12N2O2','Y':'C9H11NO3','V':'C5H11NO2'}
def counts(f):return collections.Counter({e:int(n or 1) for e,n in re.findall(r'([A-Z][a-z]?)(\d*)',f)})
def formula(c):return ''.join(e+(str(c[e])if c[e]!=1 else '')for e in sorted(c)if c[e])
def seqformula(d):
 s=d['sequence']; notes=[]
 if d.get('unusualAminoAcids') or set(s.upper())-set(AA):return None,['unusual_residues_not_independently_reconstructed']
 if d.get('nTerminus'):return None,['nterminal_modification_unhandled']
 if (d.get('cTerminus')or{}).get('name')not in (None,'AMD'):return None,['cterminal_modification_unhandled']
 c=collections.Counter()
 for a in s.upper():c.update(counts(AA[a]))
 c['H']-=2*(len(s)-1);c['O']-=len(s)-1
 if d.get('cTerminus'):c['N']+=1;c['H']+=1;c['O']-=1
 for b in d.get('intrachainBonds')or[]:
  if b['type']['name']=='DSB':c['H']-=2
  elif b['type']['name']=='AMD' and b['cycleType']['name']=='NCB':c['H']-=2;c['O']-=1
  else:return None,['bond_type_unhandled']
 notes.append('null_termini_evaluated_as_structure_hypothesis_not_observed_free')
 return formula(c),notes
rows=[]
for p in sorted((ROOT/'dbaasp_bounded_v3/raw').glob('detail_*.json')):
 d=json.loads(p.read_text()); expected,notes=seqformula(d);out=[]
 for x in d.get('smiles')or[]:
  m=Chem.MolFromSmiles(x['smiles']); row={'smiles_id':x['id'],'parsed':m is not None}
  if m:
   row.update(formula=rdMolDescriptors.CalcMolFormula(m),average_mass=Descriptors.MolWt(m),fragments=len(Chem.GetMolFrags(m)),formal_charge=Chem.GetFormalCharge(m),undefined_atom_stereocenters=sum(v=='?'for _,v in Chem.FindMolChiralCenters(m,includeUnassigned=True)),formula_matches_sequence_hypothesis=expected==rdMolDescriptors.CalcMolFormula(m) if expected else None)
  out.append(row)
 rows.append({'peptide_id':d['id'],'raw_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'sequence_formula_hypothesis':expected,'caveats':notes,'smiles':out})
r={'rdkit_version':rdBase.rdkitVersion,'records':len(rows),'raw_smiles':sum(len(x['smiles'])for x in rows),'rows':rows,'interpretation':'Formula agreement is not chemical identity or assayed mass-basis verification; all source raw unchanged.'}
path=ROOT/'chem_cnn_audit_v1/RAW_CHEMISTRY_INDEPENDENT.json';path.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps({k:v for k,v in r.items()if k!='rows'}));print('mismatches',[(x['peptide_id'],x['sequence_formula_hypothesis'],list({s.get('formula')for s in x['smiles']}))for x in rows if any(s.get('formula_matches_sequence_hypothesis')is False for s in x['smiles'])])
