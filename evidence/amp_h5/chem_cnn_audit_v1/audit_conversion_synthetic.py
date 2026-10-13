"""Independent guard/chemistry fixtures, using zero acquired sequences."""
import json,sys,hashlib,copy,math
from pathlib import Path
from rdkit import Chem
root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root/'chemistry_units_v1'))
import audit_convert as a
checks=[]
def test(name,condition):
 assert condition,name
 checks.append(name)
def fixture(seq='AG',smiles=None,chem=None):
 c={'sequence':seq,'nTerminus':None,'cTerminus':None,'unusualAminoAcids':[],'intrachainBonds':[],'interchainBonds':[],'coordinationBonds':[]};c.update(chem or {})
 return {'peptide_id':-1,'sequence':seq,'sequence_family':'SYNTHETIC','source_family_group':'SYNTHETIC','chemistry':c,'smiles':[{'smiles':smiles or Chem.MolToSmiles(Chem.MolFromSequence(seq.upper())),'manuallyEdited':False}]}
p=fixture();x=a.audit_peptide(p);test('standard_formula',x['formula']=='C5H10N2O3');test('standard_mass',abs(x['verified_represented_parent_MW_g_mol']-146.146)<.001)
# Same formula, wrong D stereochemistry must not pass.
x=a.audit_peptide(fixture(seq='aG'));test('stereo_mismatch_not_pass',x['structure_status']!='PASS_REPRESENTED_STRUCTURE')
for tag,smiles in [('salt','NCC(=O)O.[Na+]'),('isotope','[13CH3][C@@H](N)C(=O)O'),('undefined','CC(N)C(=O)O'),('invalid','invalid')]:
 x=a.audit_peptide(fixture(smiles=smiles));test(tag+'_not_pass',x['structure_status']!='PASS_REPRESENTED_STRUCTURE' and x['verified_represented_parent_MW_g_mol'] is None)
# Explicit annotation conflict even though raw formula describes free parent.
x=a.audit_peptide(fixture(chem={'cTerminus':{'name':'AMD'}}));test('amide_formula_mismatch_refuse',x['structure_status']=='REFUSE')
# Parent graph agreement never upgrades unknown assay material basis.
x=a.audit_peptide(p)
for kind,operator,lo,hi in [('right_bound','>',64.,None),('left_bound','<=',None,8.),('range_unresolved','semantic_not_verified',1.,4.),('plusminus_unresolved','semantic_not_verified',2.,.4)]:
 row={'format':kind,'operator_or_semantics':operator,'lower_raw_unit':lo,'upper_raw_unit':hi,'unit_raw':'µg/ml','concentration_raw':'untouched'}
 c=a.convert_row(row,x);factor=1000/x['verified_represented_parent_MW_g_mol']
 test(kind+'_raw_and_semantics_retained',all(c[k]==v for k,v in row.items()))
 test(kind+'_no_admission',c['conversion_status']=='PENDING' and not c['assay_molarity_verified'] and not c['training_admitted'])
 test(kind+'_positive_linear_scale',factor>0 and all(c['conditional_parent_equivalent_'+side+'_uM']==(None if v is None else v*factor)for side,v in [('lower',lo),('upper',hi)]))
# Full D-Ile side-chain inversion exercised against complete enantiomer graph.
p=fixture(seq='iG');d=Chem.MolFromSequence('IG')
for atom in d.GetAtoms():
 if atom.GetMonomerInfo().GetResidueNumber()==1 and atom.GetChiralTag()!=Chem.ChiralType.CHI_UNSPECIFIED:atom.InvertChirality()
p['smiles'][0]['smiles']=Chem.MolToSmiles(d,isomericSmiles=True);x=a.audit_peptide(p);test('D_Ile_all_centers_inverted',x['structure_status']=='PASS_REPRESENTED_STRUCTURE')
out={'status':'PASS','checks':checks,'count':len(checks),'author_code_sha256':hashlib.sha256((root/'chemistry_units_v1/audit_convert.py').read_bytes()).hexdigest(),'limitation':'Synthetic guard tests only; author functions invoked, fixtures/assertions independently designed. No dataset rows added.'}
(root/'chem_cnn_audit_v1/CHEMISTRY_INDEPENDENT_SYNTHETIC.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
