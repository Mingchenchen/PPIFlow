import os
from Bio.PDB import PDBParser, PDBIO, Select
from Bio.PDB import Superimposer
import numpy as np
import mdtraj as md
import freesasa
from collections import defaultdict

from typing import Tuple, Optional, Set
from Bio.PDB import NeighborSearch
from Bio.PDB.Structure import Structure
from Bio.PDB.Atom import Atom


def _group_consecutive_numbers(numbers):
    """
    Helper function that groups consecutive numbers from a sorted list into sublists.

    Example: [25, 26, 27, 50, 51, 95, 96, 97] -> [[25, 26, 27], [50, 51], [95, 96, 97]]

    Args:
        numbers: A sorted list of integers.

    Returns:
        A list of lists, where each sublist contains consecutive numbers.
    """
    if not numbers:
        return []

    groups = []
    current_group = [numbers[0]]

    for i in range(1, len(numbers)):
        # Check if current number is consecutive to the previous one
        if numbers[i] == numbers[i - 1] + 1:
            current_group.append(numbers[i])
        else:
            # Not consecutive, finalize current group and start a new one
            groups.append(current_group)
            current_group = [numbers[i]]

    # Add the last group to the result list
    groups.append(current_group)
    return groups


def parse_pdb_and_get_cdr(pdb_path, heavy_chain, light_chain):
    """
    Parse a single PDB file and identify CDR regions based on B-factor=2.0.
    Returns the lengths of the 6 CDR regions (H1-H3 and L1-L3).

    Uses a two-step approach: first extract all CDR residue numbers, then group them.

    Args:
        pdb_path: Full path to the PDB file.
        heavy_chain: Chain identifier for the heavy chain.
        light_chain: Chain identifier for the light chain.

    Returns:
        A dictionary containing CDR lengths keyed by names like 'cdr_H1', 'cdr_H2', etc.
        Returns None if an error occurs.
    """

    # Step 1: Extract all CDR residue numbers
    # Using a set handles duplicate PDB ATOM records for the same residue
    cdr_residues_by_chain = defaultdict(set)

    try:
        with open(pdb_path, "r") as f:
            for line in f:
                if not line.startswith("ATOM"):
                    continue

                chain_id = line[21].strip()
                if chain_id not in [heavy_chain, light_chain]:
                    continue

                try:
                    b_factor = float(line[60:66])
                    if b_factor == 2.0:
                        res_seq = int(line[22:26])
                        cdr_residues_by_chain[chain_id].add(res_seq)
                except (ValueError, IndexError):
                    continue  # Skip malformed lines

        # Step 2: Group the extracted residue numbers
        # Heavy chain
        heavy_res_sorted = sorted(
            list(cdr_residues_by_chain.get(heavy_chain, set()))
        )
        heavy_cdrs = _group_consecutive_numbers(heavy_res_sorted)

        # Light chain
        if light_chain in cdr_residues_by_chain:
            light_res_sorted = sorted(
                list(cdr_residues_by_chain.get(light_chain, set()))
            )
            light_cdrs = _group_consecutive_numbers(light_res_sorted)

        # Step 3: Format and return results
        lengths = {}

        # Populate heavy chain CDR lengths
        for i in range(3):
            key = f"cdr_H{i+1}"
            lengths[key] = str(heavy_cdrs[i])

        # Populate light chain CDR lengths
        if light_chain in cdr_residues_by_chain:
            for i in range(3):
                key = f"cdr_L{i+1}"
                lengths[key] = str(light_cdrs[i])

        return lengths

    except FileNotFoundError:
        print(f"Error: File not found {pdb_path}")
        return None
    except Exception as e:
        print(f"Error parsing file {pdb_path}: {e}")
        return None


class ChainSelect(Select):
    """
    Helper class for selecting specific chains when saving PDB files.

    Inherits from Bio.PDB.Select to provide custom chain filtering.
    """

    def __init__(self, chain_ids):
        self.chain_ids = chain_ids

    def accept_chain(self, chain):
        """Accept only chains whose IDs are in the allowed list."""
        return chain.id in self.chain_ids


def extract_chains(pdb_path, out_path, chain_ids):
    """
    Extract specified chains from a PDB file using BioPython and save to a new file.

    Args:
        pdb_path: Path to the input PDB file.
        out_path: Path for the output PDB file.
        chain_ids: List of chain identifiers to extract.
    """
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("complex", pdb_path)
    io = PDBIO()
    io.set_structure(structure)
    io.save(out_path, ChainSelect(chain_ids))


def get_chain_sasa(pdb_path, antigen_chain, heavy_chain, light_chain):
    """
    Calculate the buried surface area (BSA) between antibody and antigen chains.

    Uses FreeSASA to compute solvent accessible surface areas and calculates
    the interface buried area using the formula:
    BSA = (sum of individual chain SASA - complex SASA) / 2

    Args:
        pdb_path: Path to the PDB file.
        antigen_chain: Chain identifier for the antigen.
        heavy_chain: Chain identifier for the heavy chain.
        light_chain: Chain identifier for the light chain.

    Returns:
        The buried surface area in square angstroms, or None if an error occurs.
    """
    params = freesasa.Parameters()
    params.setProbeRadius(2.4)
    try:
        # Read structure and calculate SASA per chain
        structureArray = freesasa.structureArray(
            pdb_path, options={"separate-chains": True}
        )

        # Calculate SASA for each chain
        chain_sasa = {}
        for structure in structureArray:
            chain_id = structure.chainLabel(1)
            result = freesasa.calc(structure, params)
            chain_sasa[chain_id] = result.totalArea()

        # Total SASA of all individual chains (A+B+C)
        sasa_individual_sum = sum(chain_sasa.values())

        # SASA of the entire ABC complex
        sasa_ABC = freesasa.calc(
            freesasa.Structure(pdb_path), params
        ).totalArea()

        if len(chain_sasa) == 2:
            return (sasa_individual_sum - sasa_ABC) / 2
        elif len(chain_sasa) == 3:
            # Extract A and B chains to a temporary PDB file using BioPython
            ab_pdb = os.path.join(os.path.dirname(pdb_path), "AB_tmp.pdb")
            ab_pdb = "AB_tmp.pdb"
            extract_chains(pdb_path, ab_pdb, [heavy_chain, light_chain])

            # Calculate SASA of the AB complex
            sasa_AB = freesasa.calc(
                freesasa.Structure(ab_pdb), params
            ).totalArea()

            # Sum of individual SASA for A and B chains
            sasa_A_B_sum = chain_sasa.get(heavy_chain, 0) + chain_sasa.get(
                light_chain, 0
            )

            # Calculate final buried surface area
            term1 = (sasa_individual_sum - sasa_ABC) / 2
            term2 = (sasa_A_B_sum - sasa_AB) / 2
            result = term1 - term2

            # Clean up temporary file
            if os.path.exists(ab_pdb):
                os.remove(ab_pdb)

            return result

    except Exception as e:
        print(f"Error: {e}")
        return None


def calculate_rog(pdb_file, chain_id):
    """
    Calculate the Radius of Gyration (ROG) for a specific chain in a PDB file.

    Args:
        pdb_file: Path to the PDB file.
        chain_id: Chain identifier to calculate ROG for.

    Returns:
        The radius of gyration value.
    """
    # Parse PDB file using PDBParser
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("protein", pdb_file)

    # Get atom coordinates for the specified chain
    atoms = [atom for atom in structure[0][chain_id].get_atoms()]

    # Calculate center of mass (centroid)
    coords = np.array([atom.coord for atom in atoms])
    center_of_mass = np.mean(coords, axis=0)

    # Calculate ROG
    rog = np.sqrt(np.sum((coords - center_of_mass) ** 2) / len(atoms))

    return rog


def calc_mdtraj_metrics_single_chain(pdb_path, use_chain_id="A"):
    """Calculate mdtraj metrics for a single chain in a PDB file."""
    try:
        traj = md.load(pdb_path)

        # Get atom indices for the specified chain
        a_chain_atoms = [
            atom.index
            for atom in traj.topology.atoms
            if atom.residue.chain.chain_id == use_chain_id
        ]

        assert len(a_chain_atoms) > 0

        # Slice trajectory to keep only atoms from the specified chain
        traj = traj.atom_slice(a_chain_atoms)

        # Check if atoms were successfully retrieved
        if traj.n_atoms == 0:
            raise ValueError(f"No atoms found in chain {use_chain_id}")

        # Calculate various metrics
        pdb_ss = md.compute_dssp(traj, simplified=True)
        pdb_coil_percent = np.mean(pdb_ss == "C")
        pdb_helix_percent = np.mean(pdb_ss == "H")
        pdb_strand_percent = np.mean(pdb_ss == "E")
        pdb_ss_percent = pdb_helix_percent + pdb_strand_percent
        seq_len = pdb_ss.shape[1]
        pdb_rg = calculate_rog(pdb_path, use_chain_id)
        pdb_rg_by_len = pdb_rg / seq_len

    except Exception as e:
        print(
            f"Error in calc_mdtraj_metrics_single_chain for {pdb_path} chain {use_chain_id}: {e}"
        )
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
        "seq_length": seq_len,
    }


def calc_mdtraj_metrics(pdb_path, use_chain_id=None):
    """
    Calculate mdtraj metrics for all chains or a specified chain in a PDB file.

    Args:
        pdb_path: Path to the PDB file.
        use_chain_id: Specific chain ID to calculate metrics for.
            If None, calculates metrics for all chains.

    Returns:
        If use_chain_id is specified, returns a dictionary of metrics for that chain.
        If use_chain_id is None, returns a dictionary mapping chain IDs to their metrics.
    """
    if use_chain_id is not None:
        # Single chain mode - maintain backward compatibility
        return calc_mdtraj_metrics_single_chain(pdb_path, use_chain_id)

    try:
        # Get all chain IDs from the PDB file
        traj = md.load(pdb_path)
        chain_ids = set()
        for atom in traj.topology.atoms:
            chain_ids.add(atom.residue.chain.chain_id)

        chain_ids = sorted(list(chain_ids))  # Sort for consistent ordering
        print(f"Found chains in {pdb_path}: {chain_ids}")

        # Calculate metrics for each chain
        all_chain_metrics = {}
        for chain_id in chain_ids:
            print(f"Calculating metrics for chain {chain_id}")
            chain_metrics = calc_mdtraj_metrics_single_chain(
                pdb_path, chain_id
            )
            all_chain_metrics[chain_id] = chain_metrics

        return all_chain_metrics

    except Exception as e:
        print(f"Error in calc_mdtraj_metrics for {pdb_path}: {e}")
        return {}


def calc_distance(atom1, atom2):
    """Calculate the Euclidean distance between two atoms."""
    return np.linalg.norm(atom1.coord - atom2.coord)


def get_CA_or_CB(residue):
    """
    Return the CB atom from a residue, or CA if CB is not present.

    Args:
        residue: A Bio.PDB Residue object.

    Returns:
        The CB atom if present, otherwise CA atom, or None if neither exists.
    """
    if "CB" in residue:
        return residue["CB"]
    elif "CA" in residue:
        return residue["CA"]
    else:
        return None


def calc_hotspot_coverage(
    pdb_path, antigen_chain, heavy_chain, light_chain, cutoff=10.0
):
    """
    Calculate the coverage ratio of antigen hotspot residues by antibody CDR regions.

    Hotspot residues are identified by B-factor = 1.0, and CDR residues by B-factor = 2.0.
    A hotspot is considered covered if any CDR residue is within the cutoff distance.

    Args:
        pdb_path: Path to the PDB file.
        antigen_chain: Chain identifier for the antigen.
        heavy_chain: Chain identifier for the heavy chain.
        light_chain: Chain identifier for the light chain.
        cutoff: Distance threshold in angstroms (default: 10.0).

    Returns:
        coverage_ratio: The fraction of hotspots covered by CDR regions.
        covered_hotspots: Sorted list of covered hotspot residue numbers.
    """
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("complex", pdb_path)
    num_chains = sum(1 for _ in structure.get_chains())
    if num_chains == 3:
        # Get antigen (C chain) and antibody (A, B chains)
        antigen = structure[0][antigen_chain]
        antibody_chains = [
            structure[0][heavy_chain],
            structure[0][light_chain],
        ]
    elif num_chains == 2:
        # Get antigen (C chain) and antibody (A chain)
        antigen = structure[0][antigen_chain]
        antibody_chains = [structure[0][heavy_chain]]
    else:
        raise ValueError("Invalid number of chains")

    # Hotspot residues (antigen residues with B-factor = 1)
    hotspots = [
        res
        for res in antigen
        if any(atom.bfactor == 1 for atom in res.get_atoms())
    ]

    # Antibody CDR residues (B-factor = 2)
    cdr_residues = []
    for chain in antibody_chains:
        for res in chain:
            if any(atom.bfactor == 2 for atom in res.get_atoms()):
                cdr_residues.append(res)

    covered_hotspots = set()

    for hres in hotspots:
        hatom = get_CA_or_CB(hres)
        if hatom is None:
            continue

        for cres in cdr_residues:
            catom = get_CA_or_CB(cres)
            if catom is None:
                continue

            dist = calc_distance(hatom, catom)
            if dist < cutoff:
                covered_hotspots.add(hres.id[1])
                break  # No need to check further if this hotspot is already covered

    coverage_ratio = (
        len(covered_hotspots) / len(hotspots) if hotspots else 0.0
    )

    return coverage_ratio, sorted(list(covered_hotspots))


def get_interface_residues(
    pdb_path: str, cutoff: float, heavy_chain_id: str, light_chain_id: str
) -> Tuple[str, Optional[str], float]:
    """
    Identify interface residues in heavy and light chains based on distance
    to other chains, and calculate the ratio of residues with B-factor ~2.0.

    Args:
        pdb_path: Path to the PDB file.
        cutoff: Distance cutoff in angstroms for interface detection.
        heavy_chain_id: Chain identifier for the heavy chain.
        light_chain_id: Chain identifier for the light chain.

    Returns:
        A tuple containing:
        - Comma-separated list of interface residue identifiers (e.g., "H123,H124")
        - Comma-separated list of interface residues with B-factor near 2.0, or None
        - Ratio of B-factor ~2.0 residues to total interface residues
    """
    pdb_name = os.path.basename(pdb_path)
    parser = PDBParser(QUIET=True)

    try:
        structure: Structure = parser.get_structure("complex", pdb_path)
        model = structure[0]
    except Exception:
        return (pdb_name, None, 0.0)

    # Validate presence of query chains
    query_chain_ids = [
        cid
        for cid in [heavy_chain_id, light_chain_id]
        if cid and cid in model
    ]

    if not query_chain_ids:
        return (pdb_name, "Chains not found", 0.0)

    # Define target chains (antigens/other) and collect target atoms
    # Optimization: Use NeighborSearch (KD-Tree) for spatial queries
    target_atoms = [
        atom
        for ch in model
        if ch.id not in query_chain_ids
        for atom in ch.get_atoms()
        if atom.get_id() == "CA"
    ]

    if not target_atoms:
        return ("", "", 0.0)

    ns = NeighborSearch(target_atoms)

    interface_residues: Set[str] = set()
    bfactor_2_residues: Set[str] = set()

    for chain_id in query_chain_ids:
        chain = model[chain_id]
        for residue in chain:
            # Use CA atom as the representative point for distance check
            if "CA" not in residue:
                continue

            ca_atom: Atom = residue["CA"]

            # Spatial search: Find any target atom within 'cutoff' distance
            # Returns list of atoms; we only care if the list is not empty
            nearby_targets = ns.search(ca_atom.coord, cutoff, level="A")

            if nearby_targets:
                # Format: "H123" (ChainID + Residue Number)
                res_key = f"{chain_id}{residue.id[1]}"
                interface_residues.add(res_key)

                # Check B-factor condition: absolute difference < 0.001
                if any(abs(atom.bfactor - 2.0) < 1e-3 for atom in residue):
                    bfactor_2_residues.add(res_key)

    # Calculate metrics
    ratio = 0.0
    if interface_residues:
        ratio = round(len(bfactor_2_residues) / len(interface_residues), 3)

    return (
        ",".join(sorted(interface_residues)),
        ",".join(sorted(bfactor_2_residues)),
        ratio,
    )


def detect_backbone_clash(pdb_path, min_clash_distance=2.0):
    """
    Detect backbone clashes in a PDB structure by checking CA atom distances.

    Args:
        pdb_path: Path to the PDB file.
        min_clash_distance: Minimum distance threshold for detecting clashes (default: 2.0 Å).

    Returns:
        True if clashes are detected, False otherwise.
    """
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("structure", pdb_path)

    all_coords = []

    for chain in structure.get_chains():
        for residue in chain:
            if "CA" not in residue:
                continue
            coord = residue["CA"].get_coord()
            all_coords.append(coord)

    n = len(all_coords)
    if n < 2:
        return False

    coords = np.array(all_coords)

    # Compute pairwise distance matrix
    diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
    dist_matrix = np.linalg.norm(diff, axis=-1)

    # Get upper triangle indices (excluding diagonal)
    i_idx, j_idx = np.triu_indices(n, k=1)

    has_clash = np.any(dist_matrix[i_idx, j_idx] < min_clash_distance)

    return bool(has_clash)


class BFactorSelect(Select):
    """
    Selector class for filtering residues based on B-factor threshold.

    Selects residues that contain at least one atom with B-factor rounded to 4.0.
    """

    def accept_residue(self, residue):
        """Return True if any atom in the residue has B-factor ~4.0."""
        for atom in residue:
            if round(atom.get_bfactor(), 2) == 4.0:
                return True
        return False


def filter_pdb_by_bfactor(input_file, output_file):
    """
    Filter a PDB file to include only residues with B-factor >= 4.0.

    Args:
        input_file: Path to the input PDB file.
        output_file: Path for the filtered output PDB file.
    """
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("struct", input_file)
    io = PDBIO()
    io.set_structure(structure)
    io.save(output_file, select=BFactorSelect())


def calc_rmsd(pdb1, pdb2):
    """
    Calculate the Root Mean Square Deviation (RMSD) between two PDB structures.

    Uses CA atoms for superposition and RMSD calculation.

    Args:
        pdb1: Path to the first PDB file.
        pdb2: Path to the second PDB file.

    Returns:
        The RMSD value between the two structures.

    Raises:
        ValueError: If the structures have different numbers of CA atoms.
    """
    parser = PDBParser(QUIET=True)
    s1 = parser.get_structure("s1", pdb1)
    s2 = parser.get_structure("s2", pdb2)

    atoms1 = [a for a in s1.get_atoms() if a.get_id() == "CA"]
    atoms2 = [a for a in s2.get_atoms() if a.get_id() == "CA"]

    if len(atoms1) != len(atoms2):
        raise ValueError(
            f"atom lengths not match: {pdb1} ({len(atoms1)}) vs {pdb2} ({len(atoms2)})"
        )

    sup = Superimposer()
    sup.set_atoms(atoms1, atoms2)
    return sup.rms
