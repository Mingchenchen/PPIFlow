from Bio.PDB import StructureBuilder
from Bio.PDB.PDBIO import PDBIO
from utils.constants import num_to_letter, one_to_three_letter, restype_to_heavyatom_names
import warnings
import torch

def create_structure_from_crds(aa,crds,atom_mask,chain_id=None,outPath="test.pdb", resseq=None, icode=None, save_traj=False):
    warnings.filterwarnings("ignore", ".*Used element.*")
    structure_builder = StructureBuilder.StructureBuilder()
    structure_builder.init_structure(0)
    if save_traj:
        assert len(crds.shape) == 4
        for model_idx in range(crds.shape[0]):
            structure_builder.init_model(model_idx)
            structure_builder.init_chain("A")
            structure_builder.init_seg(' ')

            for res_idx, res in enumerate(aa):
                aa_str = one_to_three_letter[res]
                if resseq is not None and icode is not None:
                    structure_builder.init_residue(aa_str, " ", resseq[res_idx], icode[res_idx])
                else:
                    structure_builder.init_residue(aa_str, " ", res_idx + 1, " ")
                for i, atom_name in enumerate(restype_to_heavyatom_names[one_to_three_letter[res]]):
                    if atom_name == '': continue
                    if not atom_mask[res_idx, i]: continue
                    if len(atom_name) == 1:
                        fullname = f' {atom_name}  '
                    elif len(atom_name) == 2:
                        fullname = f' {atom_name} '
                    elif len(atom_name) == 3:
                        fullname = f' {atom_name}'
                    else:
                        fullname = atom_name  # len == 4
                    structure_builder.init_atom(name=atom_name, coord=crds[model_idx, res_idx, i], b_factor=res_idx + 1.0,
                                                occupancy=1.0, altloc=" ", fullname=fullname)
    else:
        structure_builder.init_model(0)
        if chain_id is not None:
            unique_chains = np.unique(chain_id).tolist()
            if len(unique_chains) == 1 and unique_chains[0] == '':
                unique_chains = ['A']
                chain_id = np.array(['A'] * len(aa))
        else:
            chain_id = np.array(['A'] * len(aa))
            unique_chains = ['A']

        for i in unique_chains:
            structure_builder.init_chain(i)
            structure_builder.init_seg(' ')

            aa_chain = np.array(list(aa))[chain_id==i]
            crds_chain = crds[chain_id==i]
            atom_mask_chain = atom_mask[chain_id==i]
            res_id_chain = resseq[chain_id==i]
            icode_chain = icode[chain_id==i]

            for res_idx, res in enumerate(aa_chain):
                aa_str = one_to_three_letter[res]
                if resseq is not None and icode is not None:
                    icode_res = " " if icode_chain[res_idx] == '' else icode_chain[res_idx]
                    structure_builder.init_residue(aa_str, " ", int(res_id_chain[res_idx]), icode_res)
                else:
                    structure_builder.init_residue(aa_str," ",res_idx+1," ")
                for i,atom_name in enumerate(restype_to_heavyatom_names[one_to_three_letter[res]]):
                    if atom_name == '': continue
                    if not atom_mask_chain[res_idx, i]: continue
                    if len(atom_name) == 1:
                        fullname = f' {atom_name}  '
                    elif len(atom_name) == 2:
                        fullname = f' {atom_name} '
                    elif len(atom_name) == 3:
                        fullname = f' {atom_name}'
                    else:
                        fullname = atom_name  # len == 4
                    structure_builder.init_atom(name=atom_name,coord=crds_chain[res_idx,i],b_factor=res_idx+1.0,occupancy=1.0,altloc=" ",fullname=fullname)

    st = structure_builder.get_structure()
    io = PDBIO()
    io.set_structure(st)
    io.save(outPath)

from Bio import PDB
import numpy as np

# Atomic radii for various atom types.
# You can comment out the ones you don't care about or add new ones
atom_radii = {
#    "H": 1.20,  # Who cares about hydrogen??
    "C": 1.70,
    "N": 1.55,
    "O": 1.52,
    "S": 1.80,
    # "F": 1.47,
    # "P": 1.80,
    # "CL": 1.75,
    # "MG": 1.73,
}

def count_clashes(path, clash_cutoff=0.4):
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure(0, path)

    # Set what we count as a clash for each pair of atoms
    clash_cutoffs = {i + "_" + j: (clash_cutoff * (atom_radii[i] + atom_radii[j])) for i in atom_radii for j in atom_radii}
    # Extract atoms for which we have a radii
    atoms = [x for x in structure.get_atoms() if x.element in atom_radii]
    coords = np.array([a.coord for a in atoms], dtype="d")
    # Build a KDTree (speedy!!!)
    kdt = PDB.kdtrees.KDTree(coords)
    # Initialize a list to hold clashes
    clashes = []
    # Iterate through all atoms
    for atom_1 in atoms:
        # Find atoms that could be clashing
        kdt_search = kdt.search(np.array(atom_1.coord, dtype="d"), max(clash_cutoffs.values()))
        # Get index and distance of potential clashes
        potential_clash = [(a.index, a.radius) for a in kdt_search]
        for ix, atom_distance in potential_clash:
            atom_2 = atoms[ix]
            # Exclude clashes from atoms in the same residue
            if atom_1.parent.id == atom_2.parent.id:
                continue
            # Exclude clashes from peptide bonds
            elif (atom_2.name == "C" and atom_1.name == "N") or (atom_2.name == "N" and atom_1.name == "C"):
                continue
            # Exclude clashes from disulphide bridges
            elif (atom_2.name == "SG" and atom_1.name == "SG") and atom_distance > 1.88:
                continue
            if atom_distance < clash_cutoffs[atom_2.element + "_" + atom_1.element]:
                clashes.append((atom_1, atom_2))
    return len(clashes) // 2