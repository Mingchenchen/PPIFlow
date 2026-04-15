"""Metrics."""

import mdtraj as md
import numpy as np
from core.np import residue_constants
from tmtools import tm_align
from Bio.PDB import PDBParser
import os
import freesasa
import contextlib


@contextlib.contextmanager
def suppress_stderr():
    devnull = os.open(os.devnull, os.O_WRONLY)
    old_stderr = os.dup(2)
    os.dup2(devnull, 2)
    try:
        yield
    finally:
        os.dup2(old_stderr, 2)
        os.close(devnull)


def get_chain_sasa(pdb):
    params = freesasa.Parameters()
    params.setProbeRadius(2.4)
    try:
        # Read PDB file with "separate-chains" option to process each chain individually
        structureArray = freesasa.structureArray(
            pdb, options={"separate-chains": True}
        )

        # Iterate over each structure (each structure corresponds to one chain)
        sasa_A_B = 0
        for structure in structureArray:
            result = freesasa.calc(structure, params)
            sasa_tmp = result.totalArea()
            sasa_A_B += sasa_tmp
            # print(structure.chainLabel(1), sasa_tmp)

        # Process the entire pdb
        sasa_AB = freesasa.calc(
            freesasa.Structure(pdb), params
        ).totalArea()

        dsasa = sasa_A_B - sasa_AB

        return dsasa / 2
    except Exception as e:
        return None


def calculate_rog(pdb_file, chain_id):
    # Use PDBParser to parse the PDB file
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("protein", pdb_file)

    # Get atom coordinates of the specified chain
    atoms = [atom for atom in structure[0][chain_id].get_atoms()]

    # Calculate center of mass
    coords = np.array([atom.coord for atom in atoms])
    center_of_mass = np.mean(coords, axis=0)

    # Calculate ROG
    rog = np.sqrt(np.sum((coords - center_of_mass) ** 2) / len(atoms))

    return rog


def calc_tm_score(pos_1, pos_2, seq_1, seq_2):
    tm_results = tm_align(pos_1, pos_2, seq_1, seq_2)
    return tm_results.tm_norm_chain1, tm_results.tm_norm_chain2


def calc_mdtraj_metrics(pdb_path, use_chain_id="A"):  # mono-only
    try:
        traj = md.load(pdb_path)

        # # Correct way: get chain ID via residue.chain.chain_id
        a_chain_atoms = [
            atom.index
            for atom in traj.topology.atoms
            if atom.residue.chain.chain_id == use_chain_id
        ]  # log-fix

        assert len(a_chain_atoms) > 0

        # Slice to keep only chain A atoms
        traj = traj.atom_slice(a_chain_atoms)

        # Check if atoms were successfully read
        if traj.n_atoms == 0:
            raise ValueError(f"No atoms found in chain {use_chain_id}")

        # Calculate various metrics
        pdb_ss = md.compute_dssp(traj, simplified=True)
        pdb_coil_percent = np.mean(pdb_ss == "C")
        pdb_helix_percent = np.mean(pdb_ss == "H")
        pdb_strand_percent = np.mean(pdb_ss == "E")
        pdb_ss_percent = pdb_helix_percent + pdb_strand_percent
        # pdb_rg = md.compute_rg(traj)[0]
        seq_len = pdb_ss.shape[1]
        pdb_rg = calculate_rog(pdb_path, use_chain_id)  # log-fix
        pdb_rg_by_len = pdb_rg / seq_len  # log-fix

    except Exception as e:
        print(f"Error in calc_mdtraj_metrics for {pdb_path}: {e}")
        pdb_ss_percent = 0.0
        pdb_coil_percent = 1.0
        pdb_helix_percent = 0.0
        pdb_strand_percent = 0.0
        pdb_rg = 0.0
        pdb_rg_by_len = 0.0
        seq_len = 0.0
    return {
        "non_coil_percent": pdb_ss_percent,
        "coil_percent": pdb_coil_percent,
        "helix_percent": pdb_helix_percent,
        "strand_percent": pdb_strand_percent,
        "radius_of_gyration": pdb_rg,
        "rog_div_by_len": pdb_rg_by_len,
        "seq_length": seq_len,  # log-fix
    }


def calc_ca_ca_metrics(ca_pos, bond_tol=0.1, clash_tol=1.0):
    ca_bond_dists = np.linalg.norm(
        ca_pos - np.roll(ca_pos, 1, axis=0), axis=-1
    )[
        1:
    ]  # ca_pos:(467, 3) #ca_bond_dists: (466,)
    ca_ca_dev = np.mean(
        np.abs(ca_bond_dists - residue_constants.ca_ca)
    )  # Calculate the mean deviation between all C-alpha atom pairs - Root Mean Square Deviation (RMSD)
    ca_ca_valid = np.mean(
        ca_bond_dists < (residue_constants.ca_ca + bond_tol)
    )

    ca_ca_dists2d = np.linalg.norm(
        ca_pos[:, None, :] - ca_pos[None, :, :], axis=-1
    )  # ca_pos[:, None, :]:(467, 1, 3), ca_pos[None, :, :]:(1, 467, 3), ca_ca_dists2d:(467, 467)
    inter_dists = ca_ca_dists2d[
        np.where(np.triu(ca_ca_dists2d, k=0) > 0)
    ]  # (108811,)
    clashes = inter_dists < clash_tol

    return {
        "ca_ca_deviation": ca_ca_dev,
        "ca_ca_valid_percent": ca_ca_valid,
        "num_ca_ca_clashes": np.sum(clashes),
        "has_clash": max(clashes) > 0,  # *
    }


def calc_distance_metrics(pdb_path):
    _, hotspot_cover_rate = calculate_hotspot_proportion(pdb_path)
    avg_distance = calculate_avg_distance_CA(pdb_path)
    hotspot_to_binder_center_distance = hotspot_to_binder_center(pdb_path)
    hotspot_to_binder_interface_distance = hotspot_to_binder_interface(
        pdb_path
    )
    return {
        "interchain_avg_ca_distance": avg_distance,
        "hotspot_to_binder_center_distance": hotspot_to_binder_center_distance,
        "hotspot_to_binder_interface_distance": hotspot_to_binder_interface_distance,
        "hotspot_cover_rate": hotspot_cover_rate,
    }


# 1. hotspot cover rate
def calculate_hotspot_proportion(
    pdb_file, bfactor_type=2, chain_A="B", chain_B="A", threshold=10.0
):
    """
    Evaluates whether the predicted binder correctly targets critical binding hotspots on the target protein,
    by calculating how many input hotspots have at least one c-alpha atom on binder that is within 10A
    """

    # Parse PDB file
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("protein", pdb_file)

    # Extract chain A and chain B
    chain_A = structure[0][chain_A]
    chain_B = structure[0][chain_B]

    # Extract C-alpha atom coordinates of hotspot residues in chain A
    hotspot_coords_A = []
    for residue in chain_A:
        if (
            "CB" in residue and residue["CB"].get_bfactor() == bfactor_type
        ):  # Hotspot with B-factor of 2
            hotspot_coords_A.append(residue["CB"].get_coord())
        elif (
            "CA" in residue and residue["CA"].get_bfactor() == bfactor_type
        ):  # Hotspot with B-factor of 2
            hotspot_coords_A.append(residue["CA"].get_coord())

    # If no hotspot found, return 0
    if len(hotspot_coords_A) == 0:
        return 0.0

    # Extract C-alpha atom coordinates of all residues in chain B
    coords_B = [
        residue["CB"].get_coord() for residue in chain_B if "CB" in residue
    ]

    # Calculate distances and check if any hotspot meets the criteria
    hotspot_in_contact = 0
    for coord_A in hotspot_coords_A:
        distances = np.linalg.norm(coords_B - coord_A, axis=1)  # Calculate distance
        if np.any(
            distances <= threshold
        ):  # Check if any chain B residue is within threshold distance
            hotspot_in_contact += 1

    # Return the proportion of hotspots that meet the criteria
    # print(f'accuracy: {hotspot_in_contact} out of {len(hotspot_coords_A)}')
    # assert len(hotspot_coords_A) == 3
    return hotspot_in_contact, hotspot_in_contact / len(hotspot_coords_A)


# 3. average distance CA
def calculate_avg_distance_CA(
    pdb_file, target_chain="B", binder_chain="A"
):
    # Parse PDB file
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("protein", pdb_file)

    # Extract chain A and chain B
    chain_A = structure[0][binder_chain]
    chain_B = structure[0][target_chain]

    # Extract all alpha carbon atoms from chain A and chain B
    alpha_carbons_A = [res["CA"] for res in chain_A if "CA" in res]
    alpha_carbons_B = [res["CA"] for res in chain_B if "CA" in res]

    distances = []
    # Calculate distances from all chain B CA atoms to chain A CA atoms
    for ca_B in alpha_carbons_B:
        for ca_A in alpha_carbons_A:
            dist = ca_B - ca_A  # Calculate Euclidean distance between two points
            distances.append(dist)

    # Calculate the mean of all distances
    avg_distance = np.mean(distances)
    # pdbname = os.path.splitext(os.path.basename(pdb_file))[0]

    return avg_distance


# 4. hotspot to binder center
def hotspot_to_binder_center(
    pdb_file, bfactor_type=2, chain_A="B", chain_B="A"
):
    # Parse PDB file
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("protein", pdb_file)

    # Extract chain A and chain B
    chain_A = structure[0][chain_A]
    chain_B = structure[0][chain_B]

    # Extract C-alpha atom coordinates of hotspot residues in chain A
    hotspot_coords_A = []
    for residue in chain_A:
        if (
            "CA" in residue and residue["CA"].get_bfactor() == bfactor_type
        ):  # Hotspot with B-factor of 2
            hotspot_coords_A.append(residue["CA"].get_coord())

    # If no hotspot found, return 0
    if len(hotspot_coords_A) == 0:
        return 0.0

    # Extract C-alpha atom coordinates of all residues in chain B
    coords_B = [
        residue["CA"].get_coord() for residue in chain_B if "CA" in residue
    ]

    coords_B = np.array(coords_B)
    # print(coords_B.shape)
    binder_center = np.mean(coords_B, axis=0)

    hotspot_binder_dists = []
    for coord_A in hotspot_coords_A:
        distance = np.linalg.norm(coord_A - binder_center)  # Calculate distance
        # print(distance)
        hotspot_binder_dists.append(distance)

    return np.array(hotspot_binder_dists).mean()


# 5. hotspot to binder interface
def hotspot_to_binder_interface(
    pdb_file, bfactor_type=2, chain_A="B", chain_B="A"
):
    # Parse PDB file
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("protein", pdb_file)

    # Extract chain A and chain B
    chain_A = structure[0][chain_A]
    chain_B = structure[0][chain_B]

    # Extract C-alpha atom coordinates of hotspot residues in chain A
    hotspot_coords_A = []
    for residue in chain_A:
        if (
            "CB" in residue and residue["CB"].get_bfactor() == bfactor_type
        ):  # Hotspot with B-factor of 2
            hotspot_coords_A.append(residue["CB"].get_coord())
        elif (
            "CA" in residue and residue["CA"].get_bfactor() == bfactor_type
        ):  # Hotspot with B-factor of 2
            hotspot_coords_A.append(residue["CA"].get_coord())

    # If no hotspot found, return 0
    if len(hotspot_coords_A) == 0:
        return 0.0

    # Extract C-alpha atom coordinates of all residues in chain B
    coords_B = [
        residue["CA"].get_coord() for residue in chain_B if "CA" in residue
    ]

    # Calculate distances and check if any hotspot meets the criteria
    hotspot_distances = []
    for coord_A in hotspot_coords_A:
        distances = np.linalg.norm(coords_B - coord_A, axis=1)  # Calculate distance
        hotspot_distances.append(np.array(distances).min())

    return np.array(hotspot_distances).mean()


def calc_other_metrics(pdb_path, chain_id="A"):
    dsasa = get_chain_sasa(pdb_path)
    return {"dsasa": dsasa}
