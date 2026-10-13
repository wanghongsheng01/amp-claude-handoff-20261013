import json,datetime
from pathlib import Path
root=Path(__file__).resolve().parent
old=root.parent/'chemistry_units_v1'
s=(old/'audit_convert.py').read_text().replace('from rdkit.Chem.MolStandardize import rdMolStandardize\n','')
start=s.index('def canonical_tautomer(');end=s.index('def audit_peptide(',start)
new='''def canonical_arg_guanidine(m):
    """Normalize only unlabelled, neutral arginine-like guanidine tautomers.
    Preserve formula, heavy connectivity, charge, stereo and all outside atoms/bonds.
    No generic tautomer enumeration or fragment/charge correction.
    """
    before=Chem.Mol(m);rw=Chem.RWMol(m);edited_atoms=set();edited_edges=set()
    def atom_state(a):
        return (a.GetAtomicNum(),a.GetIsotope(),a.GetFormalCharge(),str(a.GetChiralTag()),a.GetIsAromatic(),a.GetNumExplicitHs(),a.GetNoImplicit())
    def bond_state(b):
        return (str(b.GetBondType()),str(b.GetStereo()),tuple(b.GetStereoAtoms()),str(b.GetBondDir()),b.GetIsAromatic())
    edges_before={(min(b.GetBeginAtomIdx(),b.GetEndAtomIdx()),max(b.GetBeginAtomIdx(),b.GetEndAtomIdx())):bond_state(b) for b in before.GetBonds()}
    for center in list(rw.GetAtoms()):
        if center.GetAtomicNum()!=6 or center.GetIsAromatic() or center.GetDegree()!=3:continue
        ns=list(center.GetNeighbors());bs=list(center.GetBonds())
        if any(n.GetAtomicNum()!=7 for n in ns) or sorted(b.GetBondTypeAsDouble() for b in bs)!=[1.,1.,2.]:continue
        terminals=[n for n in ns if n.GetDegree()==1];chain=[n for n in ns if n.GetDegree()==2]
        if len(terminals)!=2 or len(chain)!=1:continue
        motif=[center]+ns
        if any(a.GetFormalCharge()!=0 or a.GetIsotope()!=0 or a.GetIsAromatic() or a.GetChiralTag()!=Chem.ChiralType.CHI_UNSPECIFIED for a in motif):raise ValueError('PENDING_GUANIDINE_MOTIF_UNSUPPORTED_ATOM_STATE')
        if any(b.GetStereo()!=Chem.BondStereo.STEREONONE or b.GetBondDir()!=Chem.BondDir.NONE for a in motif for b in a.GetBonds()):raise ValueError('PENDING_GUANIDINE_MOTIF_SPECIFIED_BOND_STEREO')
        double_idx=min(n.GetIdx() for n in terminals)
        for n in ns:
            b=rw.GetBondBetweenAtoms(center.GetIdx(),n.GetIdx());b.SetBondType(Chem.BondType.DOUBLE if n.GetIdx()==double_idx else Chem.BondType.SINGLE)
            edited_edges.add(tuple(sorted((center.GetIdx(),n.GetIdx()))));edited_atoms.add(n.GetIdx())
            n.SetNumExplicitHs(0);n.SetNoImplicit(False)
    after=rw.GetMol();Chem.SanitizeMol(after)
    edges_after={(min(b.GetBeginAtomIdx(),b.GetEndAtomIdx()),max(b.GetBeginAtomIdx(),b.GetEndAtomIdx())):bond_state(b) for b in after.GetBonds()}
    if before.GetNumAtoms()!=after.GetNumAtoms() or edges_before.keys()!=edges_after.keys():raise ValueError('REFUSE_NORMALIZATION_CONNECTIVITY_CHANGED')
    if rdMolDescriptors.CalcMolFormula(before)!=rdMolDescriptors.CalcMolFormula(after) or Chem.GetFormalCharge(before)!=Chem.GetFormalCharge(after):raise ValueError('REFUSE_NORMALIZATION_FORMULA_OR_CHARGE_CHANGED')
    for a,b in zip(before.GetAtoms(),after.GetAtoms()):
        x,y=atom_state(a),atom_state(b)
        if x[:5]!=y[:5] or (a.GetIdx() not in edited_atoms and x!=y):raise ValueError('REFUSE_NORMALIZATION_ATOM_STATE_CHANGED')
    if any(edges_before[e]!=edges_after[e] for e in edges_before if e not in edited_edges):raise ValueError('REFUSE_NORMALIZATION_OUTSIDE_BOND_CHANGED')
    if len(Chem.FindMolChiralCenters(before,includeUnassigned=True))!=len(Chem.FindMolChiralCenters(after,includeUnassigned=True)):raise ValueError('REFUSE_NORMALIZATION_STEREOCENTER_COUNT_CHANGED')
    for i in edited_atoms:
        a=after.GetAtomWithIdx(i);expected_h=1 if a.GetDegree()==2 or any(b.GetBondType()==Chem.BondType.DOUBLE for b in a.GetBonds()) else 2
        if a.GetTotalNumHs()!=expected_h:raise ValueError('REFUSE_NORMALIZATION_GUANIDINE_H_COUNT')
    return Chem.MolToSmiles(after,isomericSmiles=True)
'''
s=s[:start]+new+s[end:]
s=s.replace('canonical_tautomer(', 'canonical_arg_guanidine(').replace('STEREO_PRESERVING_TAUTOMER_CANONICAL','STEREO_PRESERVING_NEUTRAL_ARG_GUANIDINE_ONLY')
s=s.replace("'AMP_H5_CHEMICAL_UNIT_AUDIT_V1'","'AMP_H5_CHEMICAL_UNIT_AUDIT_V2'")
# Add representation audit without mistaking model encoding for chemistry/isolation admission.
s=s.replace("summary={'schema':", """def representation(r):
        p=next(p for p in ps if p['peptide_id']==r['peptide_id']);c=p['chemistry'];reasons=[]
        if not set(p['sequence'])<=set(AA_FORMULAS):reasons.append('D_OR_NONCANONICAL_RESIDUE_NOT_ENCODED')
        if len(p['sequence'])>100:reasons.append('EXCEEDS_WITTEN_100_AA_INPUT')
        if c.get('intrachainBonds') or c.get('interchainBonds') or c.get('coordinationBonds'):reasons.append('BOND_OR_CYCLE_NOT_ENCODED')
        if c.get('unusualAminoAcids'):reasons.append('UNUSUAL_RESIDUE_NOT_ENCODED')
        if c.get('nTerminus') or c.get('cTerminus'):reasons.append('TERMINAL_MODIFICATION_NOT_ENCODED')
        return {'peptide_id':p['peptide_id'],'sequence_family':p['sequence_family'],'source_family_group':p['source_family_group'],'plain_CNN_representation_compatible':not reasons,'unencoded_features':reasons,'null_terminal_annotation_still_unknown':not c.get('nTerminus') or not c.get('cTerminus')}
    rep={r['peptide_id']:representation(r) for r in candidate}
    compatible=[r for r in candidate if rep[r['peptide_id']]['plain_CNN_representation_compatible']]
    (OUT/'CNN_REPRESENTATION_AUDIT.json').write_text(json.dumps(list(rep.values()),indent=2)+'\\n')
    summary={'schema':""")
s=s.replace("'actual_new_assay_verified_peptides':0", "'plain_CNN_compatible_conditional_candidates':stats(compatible),'plain_CNN_compatible_potential_new_ids':sorted({r['peptide_id'] for r in compatible}-oldids),'plain_CNN_compatible_potential_new_families':sorted({r['sequence_family'] for r in compatible}-oldfam),'actual_new_assay_verified_peptides':0")
(root/'audit_convert.py').write_text(s)
p=json.loads((old/'PLAN.json').read_text());p['schema']='AMP_H5_CHEMISTRY_UNIT_PLAN_V2';p['created_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()
p['identity_checks'][-1]='Require exact stereo graph agreement or only neutral arginine-like guanidine tautomer normalization. Normalize C bonded 3 N, bond orders 1/1/2, 2 terminal N degree1 and 1 chain N degree2; reject labelled/charged/stereospecified motifs. Set chain N single and one terminal N double, reset H only on these N. Require invariant formula/charge/heavy edges/chiral tags, unchanged atoms and bonds outside motif, and 1/2/1 local N hydrogen counts. No generic tautomer search. All source variants must match independently reconstructed graph. Independent elemental formula and average molar mass still required.'
p['predecessor']='v1 preserved unchanged; 41 generic tautomer enumeration cap PENDING outcomes motivate this narrow, independently preaudited algorithm revision. No statistical target optimization.'
p['representation']='Report plain 20-AA CNN compatibility separately; D/noncanonical residues, bonds/cycles, terminal modifications and >100 length are not silently discarded. Null terminus stays unknown. Compatibility is not admission.'
p['outputs'].append('CNN_REPRESENTATION_AUDIT.json')
(root/'PLAN.json').write_text(json.dumps(p,indent=2)+'\n')
