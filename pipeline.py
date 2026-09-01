"""
pipeline.py — PPIFlow Pipeline Scheduler

Usage:
    python pipeline.py --task task.yaml --steps steps.yaml
    python pipeline.py --task task.yaml --steps steps.yaml --stage 1
    python pipeline.py --task task.yaml --steps steps.yaml --stage 2

Directory layout (under output_base_dir):
    stage1/
        ppiflow_output/       <- PPIFlowStep output
        mpnn_pdbs/            <- symlinks to PPIFlow PDB files
        mpnn_output/          <- MPNNStep output   (binder)
        abmpnn_output/        <- AbMPNNStep output (antibody/nanobody)
        flowpacker_output/    <- FlowpackerStep output
        af3score_output/      <- AF3scoreStep output
        filtered_iptm07/      <- FilterStep(iptm>0.7) symlinks
    stage2/
        rosetta_fix_output/   <- RosettaFixStep output
        fixed_positions.csv   <- converted for PartialStep
        before_partial_pdbs/  <- symlinks selected for PartialStep
        partial_output/       <- PartialStep output
        mpnn_pdbs/            <- symlinks to Partial PDB files + mpnn_seqs.csv
        mpnn_output/          <- per-target MPNNStep output (binder)
        abmpnn_output/        <- AbMPNNStep output (antibody/nanobody)
        flowpacker_output/    <- FlowpackerStep output
        af3score_output/      <- AF3scoreStep output
        filtered_iptm08/      <- FilterStep(iptm>0.8) symlinks
        refold_output/        <- ReFoldStep output
        dockq_output/         <- DockQStep output
        rosetta_relax_output/ <- RosettaRelaxStep output
"""

import argparse
import ast
import csv
import glob
import logging
import os
import re
import sys
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple

import yaml
from tqdm import tqdm
from typing import Any

_VALID_FILTER_OPS = (">", ">=", "<", "<=", "==", "!=")

_AA1_TO_AA3 = {
    "A": "ALA",
    "C": "CYS",
    "D": "ASP",
    "E": "GLU",
    "F": "PHE",
    "G": "GLY",
    "H": "HIS",
    "I": "ILE",
    "K": "LYS",
    "L": "LEU",
    "M": "MET",
    "N": "ASN",
    "P": "PRO",
    "Q": "GLN",
    "R": "ARG",
    "S": "SER",
    "T": "THR",
    "V": "VAL",
    "W": "TRP",
    "Y": "TYR",
    "X": "UNK",
}

def _parse_filter_string(expr: str) -> Dict[str, float]:
    """
    Parse filter expression string into dict form.

    Examples
    --------
    "> 0.7" -> {">": 0.7}
    ">= 0.3, < 0.8" -> {">=": 0.3, "<": 0.8}
    """
    if not isinstance(expr, str) or not expr.strip():
        raise ValueError(f"Invalid filter expression: {expr!r}")

    result: Dict[str, float] = {}

    parts = [p.strip() for p in expr.split(",") if p.strip()]
    if not parts:
        raise ValueError(f"Invalid filter expression: {expr!r}")

    pattern = re.compile(r"^(>=|<=|==|!=|>|<)\s*(-?\d+(?:\.\d+)?)$")

    for part in parts:
        m = pattern.match(part)
        if not m:
            raise ValueError(
                f"Invalid filter clause: {part!r}. "
                f"Expected format like '> 0.7' or '<= 10'"
            )
        op, value = m.groups()
        if op not in _VALID_FILTER_OPS:
            raise ValueError(f"Invalid operator: {op}")
        result[op] = float(value)

    return result


def _parse_filters_cfg(filters_cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert YAML filter config into FilterStep-compatible dict.

    Supported forms
    ---------------
    {
        "iptm": "> 0.7",
        "pae": "<= 10, > 1",
        "score": 0.5,
        "metric": {">": 0.7}
    }
    """
    if not isinstance(filters_cfg, dict) or not filters_cfg:
        raise ValueError("'filters' config must be a non-empty dict.")

    parsed: Dict[str, Any] = {}

    for metric, cond in filters_cfg.items():
        if isinstance(cond, str):
            parsed[metric] = _parse_filter_string(cond)
        elif isinstance(cond, (int, float)):
            parsed[metric] = cond
        elif isinstance(cond, dict):
            parsed[metric] = cond
        else:
            raise ValueError(
                f"Unsupported filter config for metric {metric!r}: {cond!r}"
            )

    return parsed

# ---------------------------------------------------------------------------
# Resolve package root so that `steps` can be imported regardless of CWD
# ---------------------------------------------------------------------------
_PIPELINE_DIR = os.path.dirname(os.path.abspath(__file__))
if _PIPELINE_DIR not in sys.path:
    sys.path.insert(0, _PIPELINE_DIR)

from steps import (  # noqa: E402
    PPIFlowStep,
    MPNNStep,
    AbMPNNStep,
    FlowpackerStep,
    AF3scoreStep,
    FilterStep,
    RosettaFixStep,
    PartialStep,
    ReFoldStep,
    DockQStep,
    RosettaRelaxStep,
    RankStep,
    ReportStep,
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("pipeline")


# ===========================================================================
# Helper: read FASTA (Biopython-free fallback for lightweight use)
# ===========================================================================


def _read_fasta(file_path: str) -> Tuple[List[str], List[int]]:
    """Return (sequences, seq_indices) from a FASTA file."""
    seqs: List[str] = []
    current_seq_parts: List[str] = []

    with open(file_path, "r") as fh:
        for line in fh:
            line = line.rstrip()
            if line.startswith(">"):
                if current_seq_parts:
                    seqs.append("".join(current_seq_parts))
                    current_seq_parts = []
            else:
                current_seq_parts.append(line)
        if current_seq_parts:
            seqs.append("".join(current_seq_parts))

    return seqs, list(range(len(seqs)))


def _process_one_fasta(
    fasta_path: str,
    prefix: Optional[str] = None,
) -> List[Tuple[str, str, int]]:
    """
    Parse one FASTA file and return rows for mpnn_seqs.csv.
    Rows: (link_name, sequence_dict_str, seq_idx)

    If prefix is provided, output link_name becomes:
        {prefix}_{pdb_filename}
    otherwise:
        {pdb_filename}
    """
    results: List[Tuple[str, str, int]] = []
    fasta_name = os.path.basename(fasta_path)
    pdb_filename = fasta_name[:-1] + "pdb"  # replace trailing 'a' → 'pdb'
    if fasta_name.endswith(".fa"):
        pdb_filename = fasta_name[:-3] + ".pdb"
    elif fasta_name.endswith(".fasta"):
        pdb_filename = fasta_name[:-6] + ".pdb"
    else:
        pdb_filename = os.path.splitext(fasta_name)[0] + ".pdb"

    seqs, seq_idxs = _read_fasta(fasta_path)

    if prefix:
        link_name = f"{prefix}_{pdb_filename}".lower()
    else:
        link_name = f"{pdb_filename}".lower()

    for seq, seq_idx in zip(seqs[1:], seq_idxs[1:]):
        parts = seq.split("/")
        if len(parts) >= 2:
            seq_str = str({"A": parts[0], "B": parts[1]})
        else:
            seq_str = str({"A": parts[0]})
        results.append((link_name, seq_str, seq_idx))

    return results


# ===========================================================================
# _collect_pdb_and_seqs
# ===========================================================================

def _convert_cif_to_pdb(cif_path: str, pdb_path: str) -> None:
    """
    Convert one mmCIF file to PDB using Biopython.
    """
    try:
        from Bio.PDB import MMCIFParser, PDBIO
    except ImportError as exc:
        raise ImportError(
            "Biopython is required for CIF->PDB conversion. "
            "Please install it: pip install biopython"
        ) from exc

    parser = MMCIFParser(QUIET=True)
    structure_id = os.path.splitext(os.path.basename(cif_path))[0]
    structure = parser.get_structure(structure_id, cif_path)

    os.makedirs(os.path.dirname(os.path.abspath(pdb_path)), exist_ok=True)

    io = PDBIO()
    io.set_structure(structure)
    io.save(pdb_path)


def _prepare_dockq_model_dir(refold_out: str, dockq_model_dir: str) -> str:
    """
    Reorganize ReFold output into the directory layout expected by DockQStep.

    Input layout (example)
    ----------------------
    refold_out/
        out/
            batch1_sample1024_fi_sample2_fi/
                seed-1020_sample-0/
                    model.cif
                seed-11_sample-1/
                    model.cif

    Output layout
    -------------
    dockq_model_dir/
        batch1_sample1024_fi_sample2_fi/
            seed-1020_sample-0.pdb
            seed-11_sample-1.pdb
    """
    refold_inner_out = os.path.join(refold_out, "out")
    if not os.path.isdir(refold_inner_out):
        raise FileNotFoundError(
            f"Expected ReFold output directory not found: {refold_inner_out}"
        )

    os.makedirs(dockq_model_dir, exist_ok=True)

    converted_count = 0
    skipped_count = 0

    for target_name in sorted(os.listdir(refold_inner_out)):
        target_dir = os.path.join(refold_inner_out, target_name)
        if not os.path.isdir(target_dir):
            continue

        dockq_target_dir = os.path.join(dockq_model_dir, target_name)
        os.makedirs(dockq_target_dir, exist_ok=True)

        for seed_name in sorted(os.listdir(target_dir)):
            seed_dir = os.path.join(target_dir, seed_name)
            if not os.path.isdir(seed_dir):
                continue

            cif_path = os.path.join(seed_dir, "model.cif")
            if not os.path.isfile(cif_path):
                log.warning("No model.cif found under %s, skipping", seed_dir)
                skipped_count += 1
                continue

            pdb_filename = f"{seed_name}.pdb"
            pdb_path = os.path.join(dockq_target_dir, pdb_filename)

            try:
                _convert_cif_to_pdb(cif_path, pdb_path)
                converted_count += 1
            except Exception as exc:
                log.error(
                    "Failed CIF->PDB conversion for %s -> %s: %s",
                    cif_path,
                    pdb_path,
                    exc,
                )
                skipped_count += 1

    log.info(
        "_prepare_dockq_model_dir: converted %d CIF files, skipped %d, output_dir=%s",
        converted_count,
        skipped_count,
        dockq_model_dir,
    )
    return dockq_model_dir

def _collect_pdb_and_seqs(
    mpnn_out_dir: str,
    stage_dir: str,
    max_workers: int = 8,
) -> str:
    """
    Gather MPNN sequences and write ``stage_dir/mpnn_pdbs/mpnn_seqs.csv``.
    """

    seqs_dir = os.path.join(mpnn_out_dir, "seqs")
    if not os.path.isdir(seqs_dir):
        seqs_dir = mpnn_out_dir

    fasta_files = glob.glob(os.path.join(seqs_dir, "*.fa")) + glob.glob(
        os.path.join(seqs_dir, "*.fasta")
    )

    log.info(
        "_collect_pdb_and_seqs: found %d FASTA files in %s",
        len(fasta_files),
        seqs_dir,
    )

    tasks = fasta_files
    all_results: List[Tuple[str, str, int]] = []

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_process_one_fasta, fasta_path): fasta_path
            for fasta_path in tasks
        }

        for future in tqdm(
            as_completed(futures),
            total=len(futures),
            desc="Collecting FASTA sequences",
        ):
            try:
                all_results.extend(future.result())
            except Exception as exc:
                log.error("Error processing %s: %s", futures[future], exc)

    output_csv = os.path.join(stage_dir, "mpnn_pdbs", "mpnn_seqs.csv")
    with open(output_csv, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["link_name", "sequence_dict", "seq_idx"])
        writer.writerows(all_results)

    log.info(
        "_collect_pdb_and_seqs: wrote %d sequence rows to %s",
        len(all_results),
        output_csv,
    )

    return output_csv


def _split_structure_and_seq_idx(pdbname: str) -> Tuple[str, Optional[int]]:
    """
    Split names like ``cd3d_cd3d_3_8`` into (``cd3d_cd3d_3``, 8).
    """
    name = os.path.splitext(os.path.basename(pdbname))[0]
    parts = name.split("_")
    if len(parts) > 2:
        try:
            return "_".join(parts[:-1]), int(parts[-1])
        except ValueError:
            pass
    return name, None


def _load_mpnn_sequence_templates(
    mpnn_seqs_csv: str,
) -> Tuple[
    Dict[str, Dict[int, Dict[str, str]]],
    Dict[str, Dict[str, str]],
]:
    """
    Read mpnn_seqs.csv and return per-seq records plus first template per structure.
    """
    if not os.path.isfile(mpnn_seqs_csv):
        raise FileNotFoundError(f"mpnn_seqs_csv not found: {mpnn_seqs_csv}")

    sequence_rows: Dict[str, Dict[int, Dict[str, str]]] = {}
    first_templates: Dict[str, Dict[str, str]] = {}

    with open(mpnn_seqs_csv, newline="") as fh:
        reader = csv.DictReader(fh)
        required = {"link_name", "sequence_dict", "seq_idx"}
        if not required.issubset(set(reader.fieldnames or [])):
            raise ValueError(
                f"{mpnn_seqs_csv} must contain columns {required}, "
                f"got {reader.fieldnames}"
            )

        for row in reader:
            link_name = (row.get("link_name") or "").strip()
            seq_idx_str = (row.get("seq_idx") or "").strip()
            sequence_dict_str = (row.get("sequence_dict") or "").strip()
            if not link_name or not seq_idx_str or not sequence_dict_str:
                continue

            try:
                seq_idx = int(seq_idx_str)
                raw_sequence_dict = ast.literal_eval(sequence_dict_str)
            except (ValueError, SyntaxError):
                log.warning(
                    "Could not parse sequence row in %s: link_name=%r seq_idx=%r",
                    mpnn_seqs_csv,
                    link_name,
                    seq_idx_str,
                )
                continue

            if not isinstance(raw_sequence_dict, dict):
                log.warning(
                    "sequence_dict is not a dict in %s: link_name=%r seq_idx=%r",
                    mpnn_seqs_csv,
                    link_name,
                    seq_idx_str,
                )
                continue

            structure_name = os.path.splitext(os.path.basename(link_name))[0].lower()
            sequence_dict = {
                str(chain): str(sequence)
                for chain, sequence in raw_sequence_dict.items()
            }

            sequence_rows.setdefault(structure_name, {})[seq_idx] = sequence_dict
            if structure_name not in first_templates:
                first_templates[structure_name] = dict(sequence_dict)

    return sequence_rows, first_templates


def _sequence_residue(
    sequence_rows: Dict[str, Dict[int, Dict[str, str]]],
    structure_name: str,
    seq_idx: Optional[int],
    chain: str,
    resseq: int,
) -> Optional[str]:
    """Return the one-letter amino acid at a 1-based residue position."""
    if seq_idx is None:
        return None

    structure_sequences = sequence_rows.get(structure_name.lower())
    if not structure_sequences:
        return None

    sequence_dict = structure_sequences.get(seq_idx)
    if not sequence_dict:
        return None

    sequence = sequence_dict.get(chain)
    if not sequence or resseq < 1 or resseq > len(sequence):
        return None

    return sequence[resseq - 1]


def _parse_merge_seq(merge_seq: str) -> Dict[str, str]:
    if not merge_seq or not merge_seq.strip():
        return {}

    try:
        raw_merge_seq = ast.literal_eval(merge_seq)
    except (ValueError, SyntaxError):
        log.warning("Could not parse merge_seq: %r", merge_seq)
        return {}

    if not isinstance(raw_merge_seq, dict):
        log.warning("merge_seq is not a dict: %r", merge_seq)
        return {}

    return {
        str(chain): str(sequence)
        for chain, sequence in raw_merge_seq.items()
        if sequence is not None
    }


def _rewrite_pdb_chains_with_merge_seq(
    src_pdb: str,
    dst_pdb: str,
    merge_seq: Dict[str, str],
) -> None:
    """
    Write a PDB copy whose ATOM residue names match merge_seq by chain order.
    """
    residue_index_by_chain: Dict[str, Dict[Tuple[str, str], int]] = {}
    short_chains = set()
    unknown_chains = set()

    with open(src_pdb, "r") as fh:
        lines = fh.readlines()

    output_lines: List[str] = []
    for line in lines:
        if not line.startswith("ATOM") or len(line) < 27:
            output_lines.append(line)
            continue

        chain = line[21].strip()
        sequence = merge_seq.get(chain)
        if not sequence:
            output_lines.append(line)
            continue

        residue_key = (line[22:26], line[26])
        chain_indices = residue_index_by_chain.setdefault(chain, {})
        if residue_key not in chain_indices:
            chain_indices[residue_key] = len(chain_indices)

        seq_pos = chain_indices[residue_key]
        if seq_pos >= len(sequence):
            if chain not in short_chains:
                log.warning(
                    "merge_seq for chain %s in %s is shorter than the PDB chain; "
                    "leaving remaining residues unchanged",
                    chain,
                    src_pdb,
                )
                short_chains.add(chain)
            output_lines.append(line)
            continue

        aa1 = sequence[seq_pos].upper()
        aa3 = _AA1_TO_AA3.get(aa1)
        if not aa3:
            if chain not in unknown_chains:
                log.warning(
                    "Unknown amino acid %r in merge_seq chain %s for %s; "
                    "leaving affected residues unchanged",
                    aa1,
                    chain,
                    src_pdb,
                )
                unknown_chains.add(chain)
            output_lines.append(line)
            continue

        output_lines.append(f"{line[:17]}{aa3:>3}{line[20:]}")

    with open(dst_pdb, "w") as fh:
        fh.writelines(output_lines)


# ===========================================================================
# _build_fixed_positions_csv
# ===========================================================================


def _build_fixed_positions_csv(
    rosetta_fix_output_dir: str,
    output_csv: str,
    gentype: str,
    mpnn_seqs_csv: str,
    energy_threshold: float = -5,
) -> str:
    """
    Convert RosettaFixStep's ``residue_energy.csv`` into the
    ``fixed_positions.csv`` format expected by PartialStep.

    Processing rules
    ----------------
    binder:
        read interface_energy_A_B/residue_energy.csv
        binder_chain = A

    nanobody:
        read interface_energy_A_C/residue_energy.csv
        binder_chain = A

    antibody:
        read interface_energy_A_C/residue_energy.csv and
             interface_energy_B_C/residue_energy.csv
        binder_chains = A, B

    Aggregation rules
    -----------------
    1. Group rows by structure name instead of pdbname.
       Example:
           cd3d_cd3d_3_8 -> cd3d_cd3d_3

    2. Merge binder_energy within the same structure.
       If the same residue position appears multiple times,
       keep the minimum energy.

    3. Select residues with energy < energy_threshold.

    4. If no residue passes the threshold, write fixed_positions = "NONE".

    Output columns
    --------------
    filename, fixed_positions, merge_seq

    filename uses structure name with .pdb suffix, e.g.:
        cd3d_cd3d_3.pdb
    """

    if gentype == "binder":
        energy_specs = [("interface_energy_A_B", "A")]
    elif gentype == "nanobody":
        energy_specs = [("interface_energy_A_C", "A")]
    elif gentype == "antibody":
        energy_specs = [
            ("interface_energy_A_C", "A"),
            ("interface_energy_B_C", "B"),
        ]
    else:
        raise ValueError(f"Unsupported gentype: {gentype}")

    sequence_rows, first_templates = _load_mpnn_sequence_templates(mpnn_seqs_csv)

    # structure_name -> chain -> resseq(int) -> min_energy(float)
    merged_energy: Dict[str, Dict[str, Dict[int, float]]] = {}
    # structure_name -> chain -> resseq(int) -> lowest-energy residue aa
    merged_fixed_aas: Dict[str, Dict[str, Dict[int, Tuple[float, str]]]] = {}

    for subdir_name, binder_chain in energy_specs:
        energy_csv = os.path.join(
            rosetta_fix_output_dir,
            subdir_name,
            "residue_energy.csv",
        )
        if not os.path.isfile(energy_csv):
            raise FileNotFoundError(
                f"Expected residue_energy.csv not found: {energy_csv}"
            )

        log.info(
            "_build_fixed_positions_csv: reading %s for gentype=%s, binder_chain=%s",
            energy_csv,
            gentype,
            binder_chain,
        )

        with open(energy_csv, newline="") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                pdbname = row.get("pdbname", "")
                if not pdbname:
                    pdbpath = row.get("pdbpath", "")
                    pdbname = os.path.splitext(os.path.basename(pdbpath))[0]

                structure_name, seq_idx = _split_structure_and_seq_idx(pdbname)
                structure_key = structure_name.lower()

                binder_energy_str = row.get("binder_energy", "{}")
                try:
                    binder_energy: Dict = ast.literal_eval(binder_energy_str)
                except (ValueError, SyntaxError):
                    log.warning(
                        "Could not parse binder_energy for %s in %s; skip this row",
                        pdbname,
                        energy_csv,
                    )
                    binder_energy = {}

                if structure_name not in merged_energy:
                    merged_energy[structure_name] = {}
                if binder_chain not in merged_energy[structure_name]:
                    merged_energy[structure_name][binder_chain] = {}
                if structure_name not in merged_fixed_aas:
                    merged_fixed_aas[structure_name] = {}
                if binder_chain not in merged_fixed_aas[structure_name]:
                    merged_fixed_aas[structure_name][binder_chain] = {}

                chain_energy = merged_energy[structure_name][binder_chain]
                chain_fixed_aas = merged_fixed_aas[structure_name][binder_chain]

                for resseq, energy in binder_energy.items():
                    try:
                        resseq_int = int(resseq)
                        energy_val = float(energy)
                    except (TypeError, ValueError):
                        log.warning(
                            "Invalid residue/energy pair in %s: resseq=%r energy=%r",
                            pdbname,
                            resseq,
                            energy,
                        )
                        continue

                    if resseq_int in chain_energy:
                        chain_energy[resseq_int] = min(
                            chain_energy[resseq_int], energy_val
                        )
                    else:
                        chain_energy[resseq_int] = energy_val

                    if energy_val >= energy_threshold:
                        continue

                    aa = _sequence_residue(
                        sequence_rows=sequence_rows,
                        structure_name=structure_key,
                        seq_idx=seq_idx,
                        chain=binder_chain,
                        resseq=resseq_int,
                    )
                    if aa is None:
                        log.warning(
                            "No sequence residue found for %s seq_idx=%r chain=%s resseq=%s",
                            structure_name,
                            seq_idx,
                            binder_chain,
                            resseq_int,
                        )
                        continue

                    existing = chain_fixed_aas.get(resseq_int)
                    if existing is None or energy_val < existing[0]:
                        chain_fixed_aas[resseq_int] = (energy_val, aa)

    output_rows: List[Tuple[str, str, str]] = []

    for structure_name in sorted(merged_energy.keys()):
        positions: List[str] = []
        structure_key = structure_name.lower()
        template_seq = {
            chain: list(sequence)
            for chain, sequence in first_templates.get(structure_key, {}).items()
        }

        for binder_chain in sorted(merged_energy[structure_name].keys()):
            chain_energy = merged_energy[structure_name][binder_chain]

            key_resseqs = sorted(
                resseq
                for resseq, energy in chain_energy.items()
                if energy < energy_threshold
            )

            positions.extend(f"{binder_chain}{resseq}" for resseq in key_resseqs)

            if binder_chain not in template_seq:
                continue

            for resseq in key_resseqs:
                fixed_aa = (
                    merged_fixed_aas.get(structure_name, {})
                    .get(binder_chain, {})
                    .get(resseq)
                )
                if fixed_aa is None:
                    continue

                seq_pos = resseq - 1
                if seq_pos < 0 or seq_pos >= len(template_seq[binder_chain]):
                    log.warning(
                        "Cannot merge fixed amino acid for %s chain=%s resseq=%s; "
                        "template sequence length=%d",
                        structure_name,
                        binder_chain,
                        resseq,
                        len(template_seq[binder_chain]),
                    )
                    continue

                template_seq[binder_chain][seq_pos] = fixed_aa[1]

        fixed_positions = ",".join(positions) if positions else "NONE"
        filename = f"{structure_name}.pdb"
        merge_seq = str(
            {chain: "".join(sequence) for chain, sequence in template_seq.items()}
        )
        output_rows.append((filename, fixed_positions, merge_seq))

    with open(output_csv, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["filename", "fixed_positions", "merge_seq"])
        writer.writerows(output_rows)

    log.info(
        "_build_fixed_positions_csv: wrote %d rows to %s",
        len(output_rows),
        output_csv,
    )
    return output_csv


def _link_before_partial_pdbs_from_fixed_positions(
    fixed_positions_csv: str,
    mpnn_pdbs_dir: str,
    before_partial_pdbs_dir: str,
) -> None:
    """
    Read ``fixed_positions.csv`` and prepare PDBs in ``before_partial_pdbs_dir``.

    Rows with merge_seq are copied with binder-chain residue names rewritten.
    Older rows without merge_seq keep the previous symlink behavior.
    """
    if not os.path.isfile(fixed_positions_csv):
        raise FileNotFoundError(f"fixed_positions_csv not found: {fixed_positions_csv}")

    if not os.path.isdir(mpnn_pdbs_dir):
        raise FileNotFoundError(f"mpnn_pdbs_dir not found: {mpnn_pdbs_dir}")

    os.makedirs(before_partial_pdbs_dir, exist_ok=True)

    linked_count = 0
    written_count = 0
    skipped_count = 0

    with open(fixed_positions_csv, newline="") as fh:
        reader = csv.DictReader(fh)
        if "filename" not in reader.fieldnames:
            raise ValueError(
                f"'filename' column not found in {fixed_positions_csv}. "
                f"Available columns: {reader.fieldnames}"
            )

        for row in reader:
            fname = row["filename"].strip()
            if not fname:
                skipped_count += 1
                continue

            src = os.path.join(mpnn_pdbs_dir, fname)
            if not os.path.lexists(src):
                log.warning("Source file not found for symlink creation: %s", src)
                skipped_count += 1
                continue

            real_src = os.path.realpath(src)
            dst = os.path.join(before_partial_pdbs_dir, fname)

            try:
                if os.path.lexists(dst):
                    os.remove(dst)
                merge_seq = _parse_merge_seq(row.get("merge_seq", ""))
                if merge_seq:
                    _rewrite_pdb_chains_with_merge_seq(real_src, dst, merge_seq)
                    written_count += 1
                else:
                    os.symlink(real_src, dst)
                    linked_count += 1
            except Exception as exc:
                log.error("Error preparing PDB: %s -> %s (%s)", real_src, dst, exc)
                skipped_count += 1

    log.info(
        "_link_before_partial_pdbs_from_fixed_positions: linked %d files, wrote %d files, skipped %d, output_dir=%s",
        linked_count,
        written_count,
        skipped_count,
        before_partial_pdbs_dir,
    )


# ===========================================================================
# Partial-stage MPNN helpers
# ===========================================================================


def _normalize_fixed_positions_for_mpnn(fixed_positions: str) -> Optional[str]:
    """
    Convert CSV fixed_positions like:
        "A6,A9,A16,A41,A42"
    into ProteinMPNN format:
        "6 9 16 41 42"

    Returns None if value is empty or NONE.
    """
    if fixed_positions is None:
        return None

    text = fixed_positions.strip()
    if not text or text.upper() == "NONE":
        return None

    positions: List[str] = []
    for token in text.split(","):
        token = token.strip()
        if not token:
            continue

        m = re.fullmatch(r"[A-Za-z]?(\d+)", token)
        if not m:
            raise ValueError(f"Invalid fixed position token: {token!r}")

        positions.append(m.group(1))

    return " ".join(positions) if positions else None

def _load_raw_fixed_positions_map(
    fixed_positions_csv: str,
) -> Dict[str, Optional[str]]:
    """
    Read fixed_positions.csv and return raw fixed_positions strings:
        {
            "cd3d_cd3d_0": "A6,A9,A16,A41,A42",
            "cd3d_cd3d_1": None,
            ...
        }
    """
    if not os.path.isfile(fixed_positions_csv):
        raise FileNotFoundError(f"fixed_positions_csv not found: {fixed_positions_csv}")

    result: Dict[str, Optional[str]] = {}

    with open(fixed_positions_csv, newline="") as fh:
        reader = csv.DictReader(fh)
        required = {"filename", "fixed_positions"}
        if not required.issubset(set(reader.fieldnames or [])):
            raise ValueError(
                f"{fixed_positions_csv} must contain columns {required}, "
                f"got {reader.fieldnames}"
            )

        for row in reader:
            filename = (row.get("filename") or "").strip()
            fixed_positions = (row.get("fixed_positions") or "").strip()

            if not filename:
                continue

            structure_name = os.path.splitext(os.path.basename(filename))[0].lower()

            if not fixed_positions or fixed_positions.upper() == "NONE":
                result[structure_name] = None
            else:
                result[structure_name] = fixed_positions

    return result

def _load_fixed_positions_map(fixed_positions_csv: str) -> Dict[str, Optional[str]]:
    """
    Read fixed_positions.csv and return:
        {
            "cd3d_cd3d_0": "6 9 16 41 42",
            "cd3d_cd3d_1": None,
            ...
        }
    """
    if not os.path.isfile(fixed_positions_csv):
        raise FileNotFoundError(f"fixed_positions_csv not found: {fixed_positions_csv}")

    result: Dict[str, Optional[str]] = {}

    with open(fixed_positions_csv, newline="") as fh:
        reader = csv.DictReader(fh)
        required = {"filename", "fixed_positions"}
        if not required.issubset(set(reader.fieldnames or [])):
            raise ValueError(
                f"{fixed_positions_csv} must contain columns {required}, "
                f"got {reader.fieldnames}"
            )

        for row in reader:
            filename = (row.get("filename") or "").strip()
            fixed_positions = (row.get("fixed_positions") or "").strip()

            if not filename:
                continue

            structure_name = os.path.splitext(os.path.basename(filename))[0].lower()
            result[structure_name] = _normalize_fixed_positions_for_mpnn(
                fixed_positions
            )

    return result


def _find_partial_sample_pdb_dirs(partial_out: str) -> List[Tuple[str, str]]:
    """
    Find per-target pdb directories under partial_out.

    Example return:
        [
            ("cd3d_cd3d_0", ".../partial_output/cd3d_cd3d_0"),
            ("cd3d_cd3d_1", ".../partial_output/cd3d_cd3d_1"),
        ]
    """
    if not os.path.isdir(partial_out):
        raise FileNotFoundError(f"partial_out not found: {partial_out}")

    results: List[Tuple[str, str]] = []

    for structure_name in sorted(os.listdir(partial_out)):
        structure_dir = os.path.join(partial_out, structure_name)
        if not os.path.isdir(structure_dir):
            continue

        # 直接在 structure_dir 下查找 pdb 文件
        pdb_files = glob.glob(os.path.join(structure_dir, "*.pdb"))

        if not pdb_files:
            log.warning("No pdb files found under %s", structure_dir)
            continue

        results.append((structure_name, structure_dir))

    return results


def _collect_partial_mpnn_pdb_and_seqs(
    mpnn_jobs: List[Tuple[str, str, str]],
    stage_dir: str,
    max_workers: int = 8,
) -> str:
    """
    Collect stage2 partial-MPNN outputs into:
        stage_dir/mpnn_pdbs/
        stage_dir/mpnn_pdbs/mpnn_seqs.csv

    Parameters
    ----------
    mpnn_jobs : list of (structure_name, sample_pdb_dir, mpnn_out_dir)
    """
    mpnn_pdbs_dir = os.path.join(stage_dir, "mpnn_pdbs")
    os.makedirs(mpnn_pdbs_dir, exist_ok=True)

    fasta_tasks: List[Tuple[str, str]] = []

    for structure_name, sample_pdb_dir, mpnn_out_dir in mpnn_jobs:
        # 1) collect pdb symlinks with structure prefix
        _collect_pdb_symlinks(
            src_dir=sample_pdb_dir,
            des_dir=mpnn_pdbs_dir,
            prefix=structure_name,
        )

        # 2) collect fasta files
        seqs_dir = os.path.join(mpnn_out_dir, "seqs")
        if not os.path.isdir(seqs_dir):
            seqs_dir = mpnn_out_dir

        fasta_files = glob.glob(os.path.join(seqs_dir, "*.fa")) + glob.glob(
            os.path.join(seqs_dir, "*.fasta")
        )

        for fasta_path in fasta_files:
            fasta_tasks.append((fasta_path, structure_name))

    log.info(
        "_collect_partial_mpnn_pdb_and_seqs: found %d FASTA files from %d MPNN jobs",
        len(fasta_tasks),
        len(mpnn_jobs),
    )

    all_results: List[Tuple[str, str, int]] = []

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_process_one_fasta, fasta_path, prefix): (
                fasta_path,
                prefix,
            )
            for fasta_path, prefix in fasta_tasks
        }

        for future in tqdm(
            as_completed(futures),
            total=len(futures),
            desc="Collecting partial MPNN FASTA sequences",
        ):
            fasta_path, prefix = futures[future]
            try:
                all_results.extend(future.result())
            except Exception as exc:
                log.error(
                    "Error processing FASTA %s (prefix=%s): %s",
                    fasta_path,
                    prefix,
                    exc,
                )

    output_csv = os.path.join(mpnn_pdbs_dir, "mpnn_seqs.csv")
    with open(output_csv, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["link_name", "sequence_dict", "seq_idx"])
        writer.writerows(all_results)

    log.info(
        "_collect_partial_mpnn_pdb_and_seqs: wrote %d sequence rows to %s",
        len(all_results),
        output_csv,
    )

    return output_csv


# ===========================================================================
# Pipeline class
# ===========================================================================


class Pipeline:
    """Main pipeline scheduler.

    Parameters
    ----------
    task_yaml : str
        Path to task.yaml (task-level config).
    steps_yaml : str
        Path to steps.yaml (step-level config).
    """

    def __init__(self, task_yaml: str, steps_yaml: str) -> None:
        with open(task_yaml, "r") as fh:
            task_cfg = yaml.safe_load(fh)
        with open(steps_yaml, "r") as fh:
            steps_cfg = yaml.safe_load(fh)

        self.task: Dict = task_cfg["task"]
        self.enabled: Dict[str, bool] = task_cfg.get("steps", {})
        self.steps_cfg: Dict = steps_cfg

        # Convenience shortcuts
        self.name: str = self.task["name"]
        self.gentype: str = self.task["gentype"]
        self.output_base_dir: str = os.path.abspath(self.task["output_base_dir"])

        # Unified Python interpreter for all steps; falls back to sys.executable
        self.python: str = (
            str(Path(steps_cfg["python"]).expanduser().resolve())
            if steps_cfg.get("python")
            else sys.executable
        )

        self.stage1_dir = os.path.join(self.output_base_dir, "stage1")
        self.stage2_dir = os.path.join(self.output_base_dir, "stage2")

        os.makedirs(self.stage1_dir, exist_ok=True)
        os.makedirs(self.stage2_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def _enabled(self, step_name: str) -> bool:
        return self.enabled.get(step_name, False)

    def _cfg(self, step_name: str) -> Dict:
        """Return steps.yaml config for a step (empty dict if not found)."""
        return self.steps_cfg.get(step_name, {})

    def _log_skip(self, step_name: str) -> None:
        log.info("SKIP %s (disabled in task.yaml)", step_name)

    def _log_start(self, step_name: str) -> None:
        log.info("=" * 60)
        log.info("START %s", step_name)

    def _log_done(self, step_name: str) -> None:
        log.info("DONE  %s", step_name)
        log.info("=" * 60)

    # ------------------------------------------------------------------
    # Stage 1
    # ------------------------------------------------------------------

    def run_stage1(self) -> None:
        log.info(">>>>>> STAGE 1 START <<<<<<")

        # ----------------------------------------------------------------
        # PPIFlowStep
        # ----------------------------------------------------------------
        ppiflow_out = os.path.join(self.stage1_dir, "ppiflow_output")

        if self._enabled("PPIFlowStep"):
            self._log_start("PPIFlowStep")
            cfg = self._cfg("PPIFlowStep")

            ppiflow_kwargs: Dict = dict(
                gentype=self.gentype,
                output_dir=ppiflow_out,
                model_weights=cfg["model_weights"],
                name=self.name,
                python=self.python,
                config=cfg.get("config"),
                samples_per_target=self.task.get("samples_per_target", 100),
                specified_hotspots=self.task.get("specified_hotspots"),
            )
            if self.gentype == "binder":
                ppiflow_kwargs.update(
                    input_pdb=self.task.get("input_pdb"),
                    input_csv=self.task.get("input_csv"),
                    target_chain=self.task.get("target_chain", "B"),
                    binder_chain=self.task.get("binder_chain", None),
                    sample_hotspot_rate_min=self.task.get(
                        "sample_hotspot_rate_min", 0.2
                    ),
                    sample_hotspot_rate_max=self.task.get(
                        "sample_hotspot_rate_max", 0.5
                    ),
                    samples_min_length=self.task.get("samples_min_length", 50),
                    samples_max_length=self.task.get("samples_max_length", 100),
                )
            else:  # antibody / nanobody
                ppiflow_kwargs.update(
                    antigen_pdb=self.task.get("antigen_pdb"),
                    antigen_chain=self.task.get("antigen_chain"),
                    framework_pdb=self.task.get("framework_pdb"),
                    heavy_chain=self.task.get("heavy_chain", "A"),
                    light_chain=self.task.get("light_chain"),
                    cdr_length=self.task.get("cdr_length"),
                )
            PPIFlowStep(**ppiflow_kwargs).run()
            self._log_done("PPIFlowStep")
        else:
            self._log_skip("PPIFlowStep")

        # ----------------------------------------------------------------
        # MPNNStep_stage1 / AbMPNNStep_stage1
        # ----------------------------------------------------------------
        mpnn_pdbs_dir = os.path.join(self.stage1_dir, "mpnn_pdbs")
        os.makedirs(mpnn_pdbs_dir, exist_ok=True)
        mpnn_seqs_csv = os.path.join(mpnn_pdbs_dir, "mpnn_seqs.csv")

        if self._enabled("MPNNStep_stage1") or self._enabled("AbMPNNStep_stage1"):
            # Create PDB symlinks from PPIFlow output first (no seqs yet)
            _collect_pdb_symlinks(ppiflow_out, mpnn_pdbs_dir, prefix=self.name)

        if self.gentype == "binder" and self._enabled("MPNNStep_stage1"):
            self._log_start("MPNNStep_stage1")
            cfg = self._cfg("MPNNStep_stage1")
            mpnn_out = os.path.join(self.stage1_dir, "mpnn_output")
            MPNNStep(
                folder_with_pdbs=mpnn_pdbs_dir,
                output_dir=mpnn_out,
                python=self.python,
                **{k: v for k, v in cfg.items() if k != "python" and v is not None},
            ).run()
            self._log_done("MPNNStep_stage1")
        elif self.gentype == "binder":
            self._log_skip("MPNNStep_stage1")
            mpnn_out = os.path.join(self.stage1_dir, "mpnn_output")

        if self.gentype in ("antibody", "nanobody") and self._enabled(
            "AbMPNNStep_stage1"
        ):
            self._log_start("AbMPNNStep_stage1")
            cfg = self._cfg("AbMPNNStep_stage1")
            mpnn_out = os.path.join(self.stage1_dir, "abmpnn_output")
            AbMPNNStep(
                folder_with_pdbs=mpnn_pdbs_dir,
                output_dir=mpnn_out,
                python=self.python,
                **{k: v for k, v in cfg.items() if k != "python" and v is not None},
            ).run()
            self._log_done("AbMPNNStep_stage1")
        elif self.gentype in ("antibody", "nanobody"):
            self._log_skip("AbMPNNStep_stage1")
            mpnn_out = os.path.join(self.stage1_dir, "abmpnn_output")

        # Collect sequences → mpnn_seqs.csv
        if self._enabled("MPNNStep_stage1") or self._enabled("AbMPNNStep_stage1"):
            mpnn_seqs_csv = _collect_pdb_and_seqs(
                mpnn_out_dir=mpnn_out,
                stage_dir=self.stage1_dir,
            )

        # ----------------------------------------------------------------
        # FlowpackerStep_stage1
        # ----------------------------------------------------------------
        flowpacker_out = os.path.join(self.stage1_dir, "flowpacker_output")
        flowpacker_pdbs_dir = os.path.join(flowpacker_out, "run_1")

        if self._enabled("FlowpackerStep_stage1"):
            self._log_start("FlowpackerStep_stage1")
            cfg = self._cfg("FlowpackerStep_stage1")
            FlowpackerStep(
                csv_path=mpnn_seqs_csv,
                pdb_dir=mpnn_pdbs_dir,
                output_dir=flowpacker_out,
                python_executable=self.python,
                **{
                    k: v
                    for k, v in cfg.items()
                    if k != "python_executable" and v is not None
                },
            ).run()
            self._log_done("FlowpackerStep_stage1")
        else:
            self._log_skip("FlowpackerStep_stage1")

        # ----------------------------------------------------------------
        # AF3scoreStep_stage1
        # ----------------------------------------------------------------
        af3score_out = os.path.join(self.stage1_dir, "af3score_output")

        if self._enabled("AF3scoreStep_stage1"):
            self._log_start("AF3scoreStep_stage1")
            cfg = self._cfg("AF3scoreStep_stage1")
            AF3scoreStep(
                input_pdb_dir=flowpacker_pdbs_dir,
                output_dir=af3score_out,
                python=self.python,
                **{k: v for k, v in cfg.items() if k != "python" and v is not None},
            ).run()
            self._log_done("AF3scoreStep_stage1")
        else:
            self._log_skip("AF3scoreStep_stage1")

        # ----------------------------------------------------------------
        # FilterStep_stage1  (iptm > 0.7)
        # ----------------------------------------------------------------
        filtered_out = os.path.join(self.stage1_dir, "filtered_iptm07")

        if self._enabled("FilterStep_stage1"):
            self._log_start("FilterStep_stage1")
            cfg = self._cfg("FilterStep_stage1")
            metrics_csv = os.path.join(af3score_out, "af3score_metrics.csv")
            FilterStep(
                pdb_dir=flowpacker_pdbs_dir,
                csv_file=metrics_csv,
                outdir=filtered_out,
                filters=_parse_filters_cfg(cfg["filters"]),
                output_csv=os.path.join(filtered_out, "filtered_iptm07.csv"),
                filename_col="description",
            ).run()
            self._log_done("FilterStep_stage1")
        else:
            self._log_skip("FilterStep_stage1")

        log.info(">>>>>> STAGE 1 DONE  <<<<<<")

    # ------------------------------------------------------------------
    # Stage 2
    # ------------------------------------------------------------------

    def run_stage2(self) -> None:
        log.info(">>>>>> STAGE 2 START <<<<<<")

        filtered_stage1 = os.path.join(self.stage1_dir, "filtered_iptm07")

        # ----------------------------------------------------------------
        # RosettaFixStep
        # ----------------------------------------------------------------
        rosetta_fix_out = os.path.join(self.stage2_dir, "rosetta_fix_output")

        if self._enabled("RosettaFixStep"):
            self._log_start("RosettaFixStep")
            cfg = self._cfg("RosettaFixStep")
            RosettaFixStep(
                gentype=self.gentype,
                output_dir=rosetta_fix_out,
                python=self.python,
                **{k: v for k, v in cfg.items() if k != "python" and v is not None},
            ).run(input_pdbdir=filtered_stage1)
            self._log_done("RosettaFixStep")
        else:
            self._log_skip("RosettaFixStep")

        # ----------------------------------------------------------------
        # Convert residue_energy.csv → fixed_positions.csv for PartialStep
        # ----------------------------------------------------------------
        fixed_positions_csv = os.path.join(self.stage2_dir, "fixed_positions.csv")
        mpnn_seqs_csv_stage1 = os.path.join(
            self.stage1_dir,
            "mpnn_pdbs",
            "mpnn_seqs.csv",
        )

        if self._enabled("PartialStep"):
            cfg = self._cfg("PartialStep")
            _build_fixed_positions_csv(
                rosetta_fix_output_dir=rosetta_fix_out,
                output_csv=fixed_positions_csv,
                gentype=self.gentype,
                mpnn_seqs_csv=mpnn_seqs_csv_stage1,
                energy_threshold=cfg.get("energy_threshold", -5),
            )

        # ----------------------------------------------------------------
        # Link stage1 mpnn_pdbs -> stage2 before_partial_pdbs
        # based on fixed_positions.csv
        # ----------------------------------------------------------------
        mpnn_pdbs_dir_stage1 = os.path.join(self.stage1_dir, "mpnn_pdbs")
        before_partial_pdbs_dir = os.path.join(self.stage2_dir, "before_partial_pdbs")

        if self._enabled("PartialStep"):
            _link_before_partial_pdbs_from_fixed_positions(
                fixed_positions_csv=fixed_positions_csv,
                mpnn_pdbs_dir=mpnn_pdbs_dir_stage1,
                before_partial_pdbs_dir=before_partial_pdbs_dir,
            )

        # ----------------------------------------------------------------
        # PartialStep
        # ----------------------------------------------------------------
        partial_out = os.path.join(self.stage2_dir, "partial_output")

        if self._enabled("PartialStep"):
            self._log_start("PartialStep")
            cfg = self._cfg("PartialStep")
            partial_kwargs: Dict = dict(
                gentype=self.gentype,
                pdb_dir=before_partial_pdbs_dir,
                fixed_positions_csv=fixed_positions_csv,
                output_dir=partial_out,
                model_weights=cfg["model_weights"],
                python=self.python,
                config=cfg.get("config"),
                samples_per_target=cfg.get("samples_per_target", 10),
                start_t=cfg.get("start_t"),
            )
            if self.gentype == "binder":
                partial_kwargs.update(
                    target_chain=cfg.get("target_chain", "B"),
                    binder_chain=cfg.get("binder_chain", "A"),
                    sample_hotspot_rate_min=cfg.get("sample_hotspot_rate_min", 0.2),
                    sample_hotspot_rate_max=cfg.get("sample_hotspot_rate_max", 0.5),
                )
            else:
                partial_kwargs.update(
                    antigen_chain=cfg.get("antigen_chain", "C"),
                    heavy_chain=cfg.get("heavy_chain", "A"),
                    light_chain=cfg.get("light_chain"),
                    retry_Limit=cfg.get("retry_Limit", 10),
                )
            PartialStep(**partial_kwargs).run()
            self._log_done("PartialStep")
        else:
            self._log_skip("PartialStep")

        # ----------------------------------------------------------------
        # MPNNStep_stage2 / AbMPNNStep_stage2
        # ----------------------------------------------------------------
        mpnn_pdbs_dir = os.path.join(self.stage2_dir, "mpnn_pdbs")
        os.makedirs(mpnn_pdbs_dir, exist_ok=True)

        if self.gentype == "binder" and self._enabled("MPNNStep_stage2"):
            self._log_start("MPNNStep_stage2")
            cfg = self._cfg("MPNNStep_stage2")
            mpnn_out = os.path.join(self.stage2_dir, "mpnn_output")
            os.makedirs(mpnn_out, exist_ok=True)

            fixed_positions_map = _load_fixed_positions_map(fixed_positions_csv)
            partial_sample_dirs = _find_partial_sample_pdb_dirs(partial_out)

            if not partial_sample_dirs:
                raise RuntimeError(
                    f"No partial sample pdb directories found under {partial_out}"
                )

            mpnn_jobs: List[Tuple[str, str, str]] = []

            for structure_name, sample_pdb_dir in partial_sample_dirs:
                structure_key = structure_name.lower()
                normalized_fixed_positions = fixed_positions_map.get(structure_key)

                protocol = "fix" if normalized_fixed_positions else "base"
                per_target_out = os.path.join(mpnn_out, structure_name)

                log.info(
                    "MPNNStep_stage2: structure=%s protocol=%s fixed_positions=%s input_dir=%s output_dir=%s",
                    structure_name,
                    protocol,
                    normalized_fixed_positions,
                    sample_pdb_dir,
                    per_target_out,
                )

                mpnn_kwargs = {
                    k: v
                    for k, v in cfg.items()
                    if k not in {"python", "protocol", "fixed_positions"}
                    and v is not None
                }

                MPNNStep(
                    protocol=protocol,
                    folder_with_pdbs=sample_pdb_dir,
                    output_dir=per_target_out,
                    python=self.python,
                    fixed_positions=normalized_fixed_positions,
                    **mpnn_kwargs,
                ).run()

                mpnn_jobs.append((structure_name, sample_pdb_dir, per_target_out))

            mpnn_seqs_csv = _collect_partial_mpnn_pdb_and_seqs(
                mpnn_jobs=mpnn_jobs,
                stage_dir=self.stage2_dir,
            )
            self._log_done("MPNNStep_stage2")

        elif self.gentype == "binder":
            self._log_skip("MPNNStep_stage2")
            mpnn_out = os.path.join(self.stage2_dir, "mpnn_output")
            mpnn_seqs_csv = os.path.join(self.stage2_dir, "mpnn_pdbs", "mpnn_seqs.csv")

        if self.gentype in ("antibody", "nanobody") and self._enabled(
            "AbMPNNStep_stage2"
        ):
            self._log_start("AbMPNNStep_stage2")
            cfg = self._cfg("AbMPNNStep_stage2")
            mpnn_out = os.path.join(self.stage2_dir, "abmpnn_output")
            os.makedirs(mpnn_out, exist_ok=True)

            raw_fixed_positions_map = _load_raw_fixed_positions_map(fixed_positions_csv)
            partial_sample_dirs = _find_partial_sample_pdb_dirs(partial_out)

            if not partial_sample_dirs:
                raise RuntimeError(
                    f"No partial sample pdb directories found under {partial_out}"
                )

            abmpnn_jobs: List[Tuple[str, str, str]] = []

            for structure_name, sample_pdb_dir in partial_sample_dirs:
                structure_key = structure_name.lower()
                per_target_fix_positions = raw_fixed_positions_map.get(structure_key)
                per_target_out = os.path.join(mpnn_out, structure_name)

                log.info(
                    "AbMPNNStep_stage2: structure=%s fix_positions=%s input_dir=%s output_dir=%s",
                    structure_name,
                    per_target_fix_positions,
                    sample_pdb_dir,
                    per_target_out,
                )

                abmpnn_kwargs = {
                    k: v
                    for k, v in cfg.items()
                    if k not in {"python", "fix_positions"} and v is not None
                }
                log.info("AbMPNNStep_stage2 kwargs: %s", abmpnn_kwargs)

                AbMPNNStep(
                    folder_with_pdbs=sample_pdb_dir,
                    output_dir=per_target_out,
                    python=self.python,
                    fix_positions=per_target_fix_positions,
                    **abmpnn_kwargs,
                ).run()

                abmpnn_jobs.append((structure_name, sample_pdb_dir, per_target_out))

            mpnn_seqs_csv = _collect_partial_mpnn_pdb_and_seqs(
                mpnn_jobs=abmpnn_jobs,
                stage_dir=self.stage2_dir,
            )
            self._log_done("AbMPNNStep_stage2")

        elif self.gentype in ("antibody", "nanobody"):
            self._log_skip("AbMPNNStep_stage2")
            mpnn_out = os.path.join(self.stage2_dir, "abmpnn_output")
            mpnn_seqs_csv = os.path.join(self.stage2_dir, "mpnn_pdbs", "mpnn_seqs.csv")

        # ----------------------------------------------------------------
        # FlowpackerStep_stage2
        # ----------------------------------------------------------------
        flowpacker_out = os.path.join(self.stage2_dir, "flowpacker_output")
        flowpacker_pdbs_dir = os.path.join(flowpacker_out, "run_1")

        if self._enabled("FlowpackerStep_stage2"):
            self._log_start("FlowpackerStep_stage2")
            cfg = self._cfg("FlowpackerStep_stage2")
            FlowpackerStep(
                csv_path=mpnn_seqs_csv,
                pdb_dir=mpnn_pdbs_dir,
                output_dir=flowpacker_out,
                python_executable=self.python,
                **{
                    k: v
                    for k, v in cfg.items()
                    if k != "python_executable" and v is not None
                },
            ).run()
            self._log_done("FlowpackerStep_stage2")
        else:
            self._log_skip("FlowpackerStep_stage2")

        # ----------------------------------------------------------------
        # AF3scoreStep_stage2
        # ----------------------------------------------------------------
        af3score_out = os.path.join(self.stage2_dir, "af3score_output")

        if self._enabled("AF3scoreStep_stage2"):
            self._log_start("AF3scoreStep_stage2")
            cfg = self._cfg("AF3scoreStep_stage2")
            AF3scoreStep(
                input_pdb_dir=flowpacker_pdbs_dir,
                output_dir=af3score_out,
                python=self.python,
                **{k: v for k, v in cfg.items() if k != "python" and v is not None},
            ).run()
            self._log_done("AF3scoreStep_stage2")
        else:
            self._log_skip("AF3scoreStep_stage2")

        # ----------------------------------------------------------------
        # FilterStep_stage2  (iptm > 0.8)
        # ----------------------------------------------------------------
        filtered_out = os.path.join(self.stage2_dir, "filtered_iptm08")

        if self._enabled("FilterStep_stage2"):
            self._log_start("FilterStep_stage2")
            cfg = self._cfg("FilterStep_stage2")
            metrics_csv = os.path.join(af3score_out, "af3score_metrics.csv")
            FilterStep(
                pdb_dir=flowpacker_pdbs_dir,
                csv_file=metrics_csv,
                outdir=filtered_out,
                filters=_parse_filters_cfg(cfg["filters"]),
                output_csv=os.path.join(filtered_out, "filtered_iptm08.csv"),
                filename_col="description",
            ).run()
            self._log_done("FilterStep_stage2")
        else:
            self._log_skip("FilterStep_stage2")

        # ----------------------------------------------------------------
        # ReFoldStep
        # ----------------------------------------------------------------
        refold_out = os.path.join(self.stage2_dir, "refold_output")

        if self._enabled("ReFoldStep"):
            self._log_start("ReFoldStep")
            cfg = self._cfg("ReFoldStep")
            ReFoldStep(
                pdb_dir=filtered_out,
                output_dir=refold_out,
                af3_python=self.python,
                db_dir=cfg["db_dir"],
                model_dir=cfg["model_dir"],
                hmmer_path=cfg["hmmer_path"],
                python=self.python,
                run_MSA=cfg.get("run_MSA", True),
                remove_template=cfg.get("remove_template", False),
                run_AF3_inference=cfg.get("run_AF3_inference", True),
                seed_num=cfg.get("seed_num", "5"),
                cuda_bin=cfg.get("cuda_bin", "/usr/local/cuda-12.6/bin"),
                cuda_lib=cfg.get("cuda_lib", "/usr/local/cuda-12.6/lib64"),
                xla_client_mem_fraction=cfg.get("xla_client_mem_fraction", "0.95"),
            ).run()
            self._log_done("ReFoldStep")
        else:
            self._log_skip("ReFoldStep")

        # ----------------------------------------------------------------
        # DockQStep
        # ----------------------------------------------------------------
        dockq_output_dir = os.path.join(self.stage2_dir, "dockq_output")
        dockq_out_csv = os.path.join(dockq_output_dir, "dockq_results.csv")
        dockq_model_dir = os.path.join(dockq_output_dir, "models")
        os.makedirs(dockq_output_dir, exist_ok=True)

        if self._enabled("DockQStep"):
            self._log_start("DockQStep")
            cfg = self._cfg("DockQStep")

            prepared_model_dir = _prepare_dockq_model_dir(
                refold_out=refold_out,
                dockq_model_dir=dockq_model_dir,
            )

            if cfg["dockq_exe"]:
                dockq_exe = str(Path(cfg["dockq_exe"]).expanduser().resolve())
            else:
                dockq_exe = "DockQ"

            DockQStep(
                reference_dir=filtered_out,
                model_dir=prepared_model_dir,
                output_csv=dockq_out_csv,
                dockq_exe=dockq_exe,
                dockq_args=cfg.get("dockq_args", ["--short"]),
                recursive=cfg.get("recursive", False),
            ).run()
            self._log_done("DockQStep")

        # ----------------------------------------------------------------
        # RosettaRelaxStep
        # ----------------------------------------------------------------
        relax_out = os.path.join(self.stage2_dir, "rosetta_relax_output")

        if self._enabled("RosettaRelaxStep"):
            self._log_start("RosettaRelaxStep")
            cfg = self._cfg("RosettaRelaxStep")
            RosettaRelaxStep(
                pdb_dir=filtered_out,
                output_dir=relax_out,
                batch_idx=cfg.get("batch_idx", "0"),
                dump_pdb=cfg.get("dump_pdb", False),
                python=self.python,
            ).run()
            self._log_done("RosettaRelaxStep")
        else:
            self._log_skip("RosettaRelaxStep")

        # ----------------------------------------------------------------
        # RankStep
        # ----------------------------------------------------------------
        design_output_dir = os.path.join(self.output_base_dir, "design_output")

        if self._enabled("RankStep"):
            self._log_start("RankStep")

            rosetta_csv = os.path.join(
                relax_out,
                "rosetta_complex_0.csv",
            )
            af3_csv = os.path.join(
                refold_out,
                "af3_iptm@1.csv",
            )
            filtered_out = os.path.join(self.stage2_dir, "filtered_iptm08")

            cfg = self._cfg("RankStep")

            RankStep(
                gentype=self.gentype,
                dockq_csv=dockq_out_csv,
                rosetta_csv=rosetta_csv,
                af3_csv=af3_csv,
                filtered_pdb_dir=filtered_out,
                output_dir=design_output_dir,
                dockq_threshold=cfg.get("dockq_threshold", 0.49),
                output_csv_name=cfg.get("output_csv_name", "ranked_designs.csv"),
            ).run()

            self._log_done("RankStep")
        else:
            self._log_skip("RankStep")

        # ----------------------------------------------------------------
        # ReportStep
        # ----------------------------------------------------------------
        if self._enabled("ReportStep"):
            self._log_start("ReportStep")
            cfg = self._cfg("RankStep")

            af3score_stage1_out = os.path.join(self.stage1_dir, "af3score_output")
            af3score_stage2_out = os.path.join(self.stage2_dir, "af3score_output")
            rank_cfg = self._cfg("RankStep")
            output_csv_name = rank_cfg.get("output_csv_name", "ranked_designs.csv")

            ReportStep(
                gentype=self.gentype,
                name=self.name,
                output_dir=design_output_dir,
                stage1_dir=self.stage1_dir,
                stage2_dir=self.stage2_dir,
                af3score_stage1_csv=os.path.join(af3score_stage1_out, "af3score_metrics.csv"),
                af3score_stage2_csv=os.path.join(af3score_stage2_out, "af3score_metrics.csv"),
                af3_refold_csv=os.path.join(refold_out, "af3_iptm@1.csv"),
                ranked_csv=os.path.join(design_output_dir, output_csv_name),
                output_csv_name=output_csv_name,
                report_filename=cfg.get("report_filename", "design_report.html"),
            ).run()

            self._log_done("ReportStep")
        else:
            self._log_skip("ReportStep")

        log.info(">>>>>> STAGE 2 DONE  <<<<<<")

    # ------------------------------------------------------------------
    # Entry point
    # ------------------------------------------------------------------

    def run(self, stage: Optional[int] = None) -> None:
        """Run the full pipeline or a specific stage.

        Parameters
        ----------
        stage : int, optional
            1 = stage 1 only, 2 = stage 2 only, None = both stages.
        """
        if stage is None or stage == 1:
            self.run_stage1()
        if stage is None or stage == 2:
            self.run_stage2()


# ===========================================================================
# Standalone helper (not exposed as a class method to keep functions pure)
# ===========================================================================


def _collect_pdb_symlinks(src_dir: str, des_dir: str, prefix: str) -> None:
    """Create symlinks for all *.pdb files in *src_dir* into *des_dir*."""
    if not os.path.isdir(src_dir):
        log.warning("_collect_pdb_symlinks: src_dir does not exist: %s", src_dir)
        return

    os.makedirs(des_dir, exist_ok=True)

    pdb_files = glob.glob(os.path.join(src_dir, "*.pdb"))
    for pdb_file in pdb_files:
        pdb_filename = os.path.basename(pdb_file)
        target_name = f"{prefix}_{pdb_filename}".lower()
        target_path = os.path.join(des_dir, target_name)
        try:
            if os.path.lexists(target_path):
                os.remove(target_path)
            os.symlink(os.path.abspath(pdb_file), target_path)
        except Exception as exc:
            log.error("Error creating symlink for %s: %s", pdb_file, exc)


# ===========================================================================
# CLI
# ===========================================================================


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="PPIFlow pipeline scheduler",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--task",
        default=os.path.join(_PIPELINE_DIR, "task.yaml"),
        help="Path to task.yaml",
    )
    parser.add_argument(
        "--steps",
        default=os.path.join(_PIPELINE_DIR, "steps.yaml"),
        help="Path to steps.yaml",
    )
    parser.add_argument(
        "--stage",
        type=int,
        choices=[1, 2],
        default=None,
        help="Run only stage 1 or stage 2 (omit to run both)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    pipeline = Pipeline(task_yaml=args.task, steps_yaml=args.steps)
    pipeline.run(stage=args.stage)
