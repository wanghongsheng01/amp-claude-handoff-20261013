"""Offline structural-mass verification and conditional MIC unit conversion.
Never infers assay salt/purity/mass basis from an automatically generated SMILES.
"""
import collections, csv, datetime, hashlib, json, math, re, time
from pathlib import Path
from rdkit import Chem, rdBase
from rdkit.Chem import Descriptors, rdMolDescriptors
from rdkit.Chem.MolStandardize import rdMolStandardize
ROOT=Path(__file__).resolve().parent
SOURCE=ROOT.parent/'dbaasp_bounded_v3'
OUT=ROOT/'results'
AA_FORMULAS={'A':'C3H7NO2','R':'C6H14N4O2','N':'C4H8N2O3','D':'C4H7NO4','C':'C3H7NO2S','E':'C5H9NO4','Q':'C5H10N2O3','G':'C2H5NO2','H':'C6H9N3O2','I':'C6H13NO2','L':'C6H13NO2','K':'C6H14N2O2','M':'C5H11NO2S','F':'C9H11NO2','P':'C5H9NO2','S':'C3H7NO3','T':'C4H9NO3','W':'C11H12N2O2','Y':'C9H11NO3','V':'C5H11NO2'}
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def name(x):return str((x or {}).get('name') or '')
def composition(s):
    d=collections.Counter()
    for e,n in re.findall(r'([A-Z][a-z]?)(\d*)',s):d[e]+=int(n or 1)
    return +d
def formula_sum(seq,c_amide,dsb_count,head_tail):
    d=collections.Counter()
    for aa in seq.upper():d.update(composition(AA_FORMULAS[aa]))
    waters=len(seq)-1+int(head_tail)
    d['H']-=2*waters+2*dsb_count;d['O']-=waters
    if c_amide:d['O']-=1;d['N']+=1;d['H']+=1
    return +d
def formula_text(d):return ''.join(e+(str(d[e]) if d[e]!=1 else '') for e in ['C','H']+sorted(k for k in d if k not in ('C','H')) if d[e])
def expected_graph(p):
    c=p['chemistry'];seq=p['sequence'];s=seq.upper()
    if not s or not set(s)<=set(AA_FORMULAS):raise ValueError('PENDING_NONCANONICAL_SEQUENCE')
    if c.get('unusualAminoAcids'):raise ValueError('PENDING_UNUSUAL_RESIDUE_ANNOTATION')
    if c.get('interchainBonds') or c.get('coordinationBonds'):raise ValueError('REFUSE_INTERCHAIN_OR_COORDINATION_STOICHIOMETRY')
    if c.get('nTerminus'):raise ValueError('PENDING_UNSUPPORTED_N_TERMINAL_MODIFICATION')
    ct=name(c.get('cTerminus'))
    if ct not in ('','AMD'):raise ValueError('PENDING_UNSUPPORTED_C_TERMINAL_MODIFICATION')
    mol=Chem.MolFromSequence(s)
    if mol is None:raise ValueError('PENDING_SEQUENCE_GRAPH_FAILURE')
    rw=Chem.RWMol(mol)
    for a in rw.GetAtoms():
        info=a.GetMonomerInfo()
        if info and seq[info.GetResidueNumber()-1].islower() and a.GetChiralTag()!=Chem.ChiralType.CHI_UNSPECIFIED:a.InvertChirality()
    def atom(residue,atom_name):
        found=[a.GetIdx() for a in rw.GetAtoms() if a.GetMonomerInfo() and a.GetMonomerInfo().GetResidueNumber()==residue and a.GetMonomerInfo().GetName().strip()==atom_name]
        if len(found)!=1:raise ValueError('PENDING_RESIDUE_ATOM_MAPPING')
        return found[0]
    head=False;disulfides=0;used_s=set()
    for bond in (c.get('intrachainBonds') or []):
        typ=name(bond.get('type'));chain=name(bond.get('chainParticipating'));cycle=name(bond.get('cycleType'))
        i,j=bond.get('position1'),bond.get('position2')
        if not isinstance(i,int) or not isinstance(j,int) or not 1<=i<=len(seq) or not 1<=j<=len(seq):raise ValueError('PENDING_BOND_POSITION')
        if typ=='DSB' and chain=='SSB':
            if s[i-1]!='C' or s[j-1]!='C' or i==j or i in used_s or j in used_s:raise ValueError('REFUSE_INVALID_DISULFIDE_ANNOTATION')
            rw.AddBond(atom(i,'SG'),atom(j,'SG'),Chem.BondType.SINGLE);used_s.update([i,j]);disulfides+=1
        elif typ=='AMD' and chain=='MMB' and cycle=='NCB' and {i,j}=={1,len(seq)}:
            if head or ct:raise ValueError('REFUSE_CONFLICTING_TERMINAL_CLOSURE')
            rw.AddBond(atom(1,'N'),atom(len(seq),'C'),Chem.BondType.SINGLE)
            rw.RemoveAtom(atom(len(seq),'OXT'));head=True
        else:raise ValueError('PENDING_UNSUPPORTED_BOND_ANNOTATION')
    if ct=='AMD':
        a=rw.GetAtomWithIdx(atom(len(seq),'OXT'));a.SetAtomicNum(7);a.SetNoImplicit(False);a.SetNumExplicitHs(0)
    mol=rw.GetMol();Chem.SanitizeMol(mol);Chem.AssignStereochemistry(mol,cleanIt=True,force=True)
    return mol,formula_sum(seq,ct=='AMD',disulfides,head),{'C_amidated':ct=='AMD','disulfide_count':disulfides,'head_tail_cyclic':head,'lowercase_D_residue_count':sum(x.islower() for x in seq),'null_termini_inferred_only_in_represented_graph':True}
def canonical_tautomer(m):
    t=rdMolStandardize.TautomerEnumerator();t.SetRemoveSp3Stereo(False);t.SetRemoveBondStereo(False);t.SetReassignStereo(True);t.SetMaxTautomers(256);t.SetMaxTransforms(1000)
    result=t.Enumerate(m)
    if result.status!=rdMolStandardize.TautomerEnumeratorStatus.Completed:raise ValueError('PENDING_TAUTOMER_ENUMERATION_LIMIT')
    chosen=t.PickCanonical(result)
    if len(Chem.FindMolChiralCenters(chosen,includeUnassigned=True))!=len(Chem.FindMolChiralCenters(m,includeUnassigned=True)):raise ValueError('PENDING_TAUTOMER_STEREOCENTER_COUNT_CHANGED')
    return Chem.MolToSmiles(chosen,isomericSmiles=True)
def audit_peptide(p):
    result={'peptide_id':p['peptide_id'],'sequence':p['sequence'],'sequence_family':p['sequence_family'],'source_family_group':p['source_family_group'],'structure_status':'PENDING','reasons':[],'formula':None,'verified_represented_parent_MW_g_mol':None,'assay_material_mass_basis':'NOT_ESTABLISHED','assay_molarity_verified':False,'source_SMILES_auto_generated':[],'variants':[]}
    mols=[]
    for k,entry in enumerate(p.get('smiles') or []):
        raw=entry.get('smiles') or '';v={'variant_index':k,'SMILES_sha256':hashlib.sha256(raw.encode()).hexdigest(),'metadata':{x:y for x,y in entry.items() if x!='smiles'}}
        result['source_SMILES_auto_generated'].append(entry.get('manuallyEdited') is False)
        try:
            m=Chem.MolFromSmiles(raw)
            if m is None:raise ValueError('REFUSE_INVALID_SMILES')
            v.update(fragments=len(Chem.GetMolFrags(m)),formal_charge=Chem.GetFormalCharge(m),formula=rdMolDescriptors.CalcMolFormula(m),average_MW=Descriptors.MolWt(m),exact_isotopic_mass=rdMolDescriptors.CalcExactMolWt(m),unspecified_chiral_centers=sum(tag=='?' for _,tag in Chem.FindMolChiralCenters(m,includeUnassigned=True)),isotope_atoms=sum(a.GetIsotope()!=0 for a in m.GetAtoms()))
            if v['fragments']!=1:raise ValueError('REFUSE_MULTIFRAGMENT_MIXTURE_OR_SALT')
            if v['formal_charge']!=0:raise ValueError('REFUSE_UNRESOLVED_COUNTERION_OR_CHARGE')
            if v['isotope_atoms']:raise ValueError('REFUSE_ISOTOPE_STOICHIOMETRY')
            if v['unspecified_chiral_centers']:raise ValueError('PENDING_UNSPECIFIED_STEREOCHEMISTRY')
            if not math.isfinite(v['average_MW']) or v['average_MW']<=0:raise ValueError('REFUSE_INVALID_MASS')
            mols.append(m);v['status']='BASIC_GRAPH_PASS'
        except Exception as exc:v['status']='REFUSE' if 'REFUSE_' in str(exc) else 'PENDING';v['reason']=str(exc);result['reasons'].append(str(exc))
        result['variants'].append(v)
    if not result['variants']:result['reasons'].append('PENDING_NO_SMILES')
    if result['reasons']:
        result['structure_status']='REFUSE' if any('REFUSE_' in s for s in result['reasons']) else 'PENDING';return result
    forms={v['formula'] for v in result['variants']}
    if len(forms)!=1:result.update(structure_status='REFUSE',reasons=['REFUSE_SMILES_FORMULA_CONFLICT']);return result
    try:
        expected,formula,mod=expected_graph(p)
        if any(composition(v['formula'])!=formula for v in result['variants']):raise ValueError('REFUSE_SEQUENCE_SMILES_FORMULA_MISMATCH')
        exact=Chem.MolToSmiles(expected,isomericSmiles=True);expected_taut=None
        for v,m in zip(result['variants'],mols):
            if Chem.MolToSmiles(m,isomericSmiles=True)==exact:v['identity_match']='EXACT_STEREO_GRAPH'
            else:
                if expected_taut is None:expected_taut=canonical_tautomer(expected)
                if canonical_tautomer(m)!=expected_taut:raise ValueError('PENDING_SEQUENCE_SMILES_STEREO_OR_GRAPH_MISMATCH')
                v['identity_match']='STEREO_PRESERVING_TAUTOMER_CANONICAL'
        mw=result['variants'][0]['average_MW'];pt=Chem.GetPeriodicTable();independent_mass=sum(pt.GetAtomicWeight(e)*n for e,n in formula.items())
        if max(abs(v['average_MW']-mw) for v in result['variants'])>1e-6 or abs(mw-independent_mass)>1e-6:raise ValueError('REFUSE_INDEPENDENT_MASS_MISMATCH')
        result.update(structure_status='PASS_REPRESENTED_STRUCTURE',reasons=['ASSAY_MATERIAL_MASS_BASIS_NOT_ESTABLISHED'],formula=formula_text(formula),verified_represented_parent_MW_g_mol=mw,formula_derived_mass_g_mol=independent_mass,supported_modifications=mod)
    except Exception as exc:
        result['structure_status']='REFUSE' if 'REFUSE_' in str(exc) else 'PENDING';result['reasons'].append(str(exc))
    return result
def convert_row(m,a):
    r=dict(m);r.update(structure_status=a['structure_status'],represented_formula=a['formula'],represented_parent_MW_g_mol=a['verified_represented_parent_MW_g_mol'],conversion_status='PENDING',conversion_reasons=list(a['reasons']),conditional_parent_equivalent_lower_uM=None,conditional_parent_equivalent_upper_uM=None,conditional_factor_uM_per_ug_ml=None,assay_molarity_verified=False,training_admitted=False)
    if a['structure_status']=='REFUSE':r['conversion_status']='REFUSE';return r
    mw=a['verified_represented_parent_MW_g_mol']
    if mw is not None:
        factor=1000/mw;r['conditional_factor_uM_per_ug_ml']=factor
        for side in ['lower','upper']:
            v=m.get(side+'_raw_unit')
            if v is not None:r['conditional_parent_equivalent_'+side+'_uM']=v*factor
        r['conversion_formula']='conditional parent-equivalent uM = raw ug/ml * 1000 / represented-parent average MW(g/mol)'
        r['conversion_reasons']=['ASSAY_MATERIAL_MASS_BASIS_NOT_ESTABLISHED']
        r['conditional_numeric_semantics']='Unit-scaled endpoints only. Preserve comparator; ranges/plusminus remain semantically unresolved. Not an admitted assay label.'
    return r
def stats(rows):
    return {'rows':len(rows),'peptide_ids':len({r['peptide_id'] for r in rows}),'backbones':len({r['sequence'].upper() for r in rows}),'sequence_families':len({r['sequence_family'] for r in rows}),'source_family_groups':len({r['source_family_group'] for r in rows})}
def main():
    started=time.monotonic();OUT.mkdir(exist_ok=False)
    plan=json.loads((ROOT/'PLAN.json').read_text());bindings=json.loads((ROOT/'FREEZE.json').read_text())
    for file,h in bindings['local_sha256'].items():
        if sha(ROOT/file)!=h:raise RuntimeError('Frozen local input mismatch '+file)
    for file,key in [('FINAL_MANIFEST.json','batch_final_manifest_sha256'),('prepared/peptides.jsonl','peptides_jsonl_sha256'),('prepared/mic_measurements.jsonl','MIC_jsonl_sha256')]:
        if sha(SOURCE/file)!=plan['inputs'][key]:raise RuntimeError('Source hash mismatch '+file)
    ps=[json.loads(s) for s in (SOURCE/'prepared/peptides.jsonl').read_text().splitlines()]
    ms=[json.loads(s) for s in (SOURCE/'prepared/mic_measurements.jsonl').read_text().splitlines()]
    audits=[audit_peptide(p) for p in ps];byid={p['peptide_id']:p for p in audits}
    rows=[convert_row(m,byid[m['peptide_id']]) for m in ms if m['ecoli_target'] and m['unit_raw']=='µg/ml']
    for file,data in [('chemical_identity_audit.jsonl',audits),('mass_MIC_conversion_draft.jsonl',rows)]:
        (OUT/file).write_text(''.join(json.dumps(x,ensure_ascii=False,allow_nan=False)+'\n' for x in data))
    with (OUT/'mass_MIC_conversion_draft.csv').open('w',newline='') as f:
        fields=list(dict.fromkeys(k for row in rows for k in row));w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for r in rows:w.writerow({k:json.dumps(v,ensure_ascii=False) if isinstance(v,(dict,list)) else v for k,v in r.items()})
    old=[m for m in ms if m['label_sequence_screen_pass']];oldids={m['peptide_id'] for m in old};oldfam={m['sequence_family'] for m in old};oldgroups={m['source_family_group'] for m in old}
    candidate=[r for r in rows if r['conditional_factor_uM_per_ug_ml'] is not None and all(x=='not_micromolar_no_conversion' for x in r['screen_exclusions'])]
    summary={'schema':'AMP_H5_CHEMICAL_UNIT_AUDIT_V1','rdkit_version':rdBase.rdkitVersion,'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'records':len(audits),'SMILES_variants':sum(len(p['variants']) for p in audits),'structure_status_counts':dict(collections.Counter(p['structure_status'] for p in audits)),'structure_reasons':dict(collections.Counter(reason for p in audits for reason in p['reasons'])),'mass_MIC_rows':stats(rows),'assay_conversion_status':{status:stats([r for r in rows if r['conversion_status']==status]) for status in ['PASS','REFUSE','PENDING']},'conditional_parent_equivalent_computable':stats([r for r in rows if r['conditional_factor_uM_per_ug_ml'] is not None]),'conditional_screen_candidates_before_massbasis_chemistry_isolation':stats(candidate),'potential_new_candidate_ids_relative_to_old_screen':sorted({r['peptide_id'] for r in candidate}-oldids),'potential_new_candidate_families_relative_to_old_screen':sorted({r['sequence_family'] for r in candidate}-oldfam),'potential_new_candidate_source_groups_relative_to_old_screen':sorted({r['source_family_group'] for r in candidate}-oldgroups),'actual_new_assay_verified_peptides':0,'actual_new_assay_verified_families':0,'actual_new_training_admitted_peptides':0,'isolation_status':'NOT_IMPLEMENTED; no train/dev/test files opened','elapsed_seconds':time.monotonic()-started,'decision':'Structural conversion feasibility assessed; retain conditional values as audit fields, not verified assay labels. Resolve material mass basis before any admission. No natural training.'}
    (OUT/'SUMMARY.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n');print(json.dumps(summary,ensure_ascii=False))
if __name__=='__main__':main()
