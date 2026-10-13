import json,hashlib
from pathlib import Path
from rdkit import Chem
import audit_convert as a
root=Path(__file__).resolve().parent;checks=[]
def check(name,x):
    assert x,name
    checks.append(name)
x=Chem.MolFromSmiles('CNC(N)=N');y=Chem.MolFromSmiles('CN=C(N)N')
check('neutral_guanidine_tautomers_equal',a.canonical_arg_guanidine(x)==a.canonical_arg_guanidine(y))
check('normalization_idempotent',a.canonical_arg_guanidine(Chem.MolFromSmiles(a.canonical_arg_guanidine(y)))==a.canonical_arg_guanidine(y))
for name,s in [('non_guanidine','CC(=O)N'),('stereochemistry','N[C@@H](C)C(=O)O'),('histidine_unchanged','N[C@@H](Cc1c[nH]cn1)C(=O)O')]:
    m=Chem.MolFromSmiles(s);check(name,Chem.MolToSmiles(m,isomericSmiles=True)==a.canonical_arg_guanidine(m))
for name,s in [('isotope','CN=C([15NH2])N'),('charged','C[NH+]=C(N)N')]:
    try:a.canonical_arg_guanidine(Chem.MolFromSmiles(s));ok=False
    except ValueError:ok=True
    check(name+'_motif_rejected',ok)
def fixture(seq,chem=None):
    return {'peptide_id':-1,'sequence':seq,'sequence_family':'synthetic','source_family_group':'synthetic','chemistry':dict({'nTerminus':None,'cTerminus':None,'unusualAminoAcids':[],'intrachainBonds':[],'interchainBonds':[],'coordinationBonds':[]},**(chem or {}))}
for name,seq,chem in [('standard','ARG',{}),('amide','ARG',{'cTerminus':{'name':'AMD'}}),('D_isoleucine','iRG',{}),('disulfide','CRC',{'intrachainBonds':[{'type':{'name':'DSB'},'chainParticipating':{'name':'SSB'},'position1':1,'position2':3}]}),('cyclic','ARG',{'intrachainBonds':[{'type':{'name':'AMD'},'chainParticipating':{'name':'MMB'},'cycleType':{'name':'NCB'},'position1':1,'position2':3}]})]:
    p=fixture(seq,chem);m,formula,mods=a.expected_graph(p);p['smiles']=[{'smiles':Chem.MolToSmiles(m,isomericSmiles=True),'manuallyEdited':False}];r=a.audit_peptide(p)
    check(name+'_graph_formula_match',r['structure_status']=='PASS_REPRESENTED_STRUCTURE')
p=fixture('ARG');p['smiles']=[{'smiles':Chem.MolToSmiles(Chem.MolFromSequence('ARG'))+'.[Na+]'}]
check('mixture_refused',a.audit_peptide(p)['structure_status']=='REFUSE')
p=fixture('aRG');p['smiles']=[{'smiles':Chem.MolToSmiles(Chem.MolFromSequence('ARG'))}]
check('D_mismatch_not_erased',a.audit_peptide(p)['structure_status']=='PENDING')
p=fixture('ARG');p['smiles']=[{'smiles':Chem.MolToSmiles(Chem.MolFromSequence('ARG'))}];r=a.audit_peptide(p)
c=a.convert_row({'lower_raw_unit':32,'upper_raw_unit':None,'operator_or_semantics':'>','unit_raw':'µg/ml'},r)
check('right_censor_preserved',c['operator_or_semantics']=='>' and c['conditional_parent_equivalent_upper_uM'] is None)
check('unknown_material_not_admitted',c['conversion_status']=='PENDING' and not c['training_admitted'])
(root/'SYNTHETIC_CHECKS.json').write_text(json.dumps({'status':'PASS','checks':checks,'count':len(checks),'code_sha256':a.sha(root/'audit_convert.py'),'synthetic_only':True},indent=2)+'\n')
print(json.dumps({'status':'PASS','count':len(checks)}))
