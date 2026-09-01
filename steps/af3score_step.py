"""
AF3scoreStep - Pipeline step for AF3Score evaluation.

This step executes the AF3Score workflow without batch splitting:

  1. Prepare single-chain CIF files and AF3 JSON inputs
  2. Convert input PDBs to JAX/H5 inputs
  3. Run AF3Score inference
  4. Extract and verify metrics

Implementation style intentionally follows flowpacker_step.py.
"""

import csv
import glob
import json
import multiprocessing as mp
import os
import subprocess
import sys
import warnings
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import h5py
import jax
import jax.numpy as jnp
import numpy as np
from Bio import PDB
from Bio.PDB import MMCIFIO, PDBParser
from Bio.PDB.PDBExceptions import PDBConstructionWarning
from Bio.PDB.Structure import Structure


# Suppress warnings from Bio.PDB during parsing
warnings.filterwarnings("ignore", category=PDBConstructionWarning)


# Resolve the tools directory relative to this file
_TOOLS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "tools",
    "AF3Score",
)

_RUN_AF3SCORE_SCRIPT = os.path.join(_TOOLS_DIR, "run_af3score.py")

protein_letters_3to1 = {
    "ALA": "A",
    "CYS": "C",
    "ASP": "D",
    "GLU": "E",
    "PHE": "F",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LYS": "K",
    "LEU": "L",
    "MET": "M",
    "ASN": "N",
    "PRO": "P",
    "GLN": "Q",
    "ARG": "R",
    "SER": "S",
    "THR": "T",
    "VAL": "V",
    "TRP": "W",
    "TYR": "Y",
    "MSE": "M",
}

ATOM14: Dict[str, Tuple[str, ...]] = {
    "ALA": ("N", "CA", "C", "O", "CB", "OXT"),
    "ARG": (
        "N", "CA", "C", "O", "CB", "CG", "CD", "NE",
        "CZ", "NH1", "NH2", "OXT",
    ),
    "ASN": ("N", "CA", "C", "O", "CB", "CG", "OD1", "ND2", "OXT"),
    "ASP": ("N", "CA", "C", "O", "CB", "CG", "OD1", "OD2", "OXT"),
    "CYS": ("N", "CA", "C", "O", "CB", "SG", "OXT"),
    "GLN": ("N", "CA", "C", "O", "CB", "CG", "CD", "OE1", "NE2", "OXT"),
    "GLU": ("N", "CA", "C", "O", "CB", "CG", "CD", "OE1", "OE2", "OXT"),
    "GLY": ("N", "CA", "C", "O", "OXT"),
    "HIS": (
        "N", "CA", "C", "O", "CB", "CG", "ND1",
        "CD2", "CE1", "NE2", "OXT",
    ),
    "ILE": ("N", "CA", "C", "O", "CB", "CG1", "CG2", "CD1", "OXT"),
    "LEU": ("N", "CA", "C", "O", "CB", "CG", "CD1", "CD2", "OXT"),
    "LYS": ("N", "CA", "C", "O", "CB", "CG", "CD", "CE", "NZ", "OXT"),
    "MET": ("N", "CA", "C", "O", "CB", "CG", "SD", "CE", "OXT"),
    "PHE": (
        "N", "CA", "C", "O", "CB", "CG", "CD1",
        "CD2", "CE1", "CE2", "CZ", "OXT",
    ),
    "PRO": ("N", "CA", "C", "O", "CB", "CG", "CD", "OXT"),
    "SER": ("N", "CA", "C", "O", "CB", "OG", "OXT"),
    "THR": ("N", "CA", "C", "O", "CB", "OG1", "CG2", "OXT"),
    "TRP": (
        "N", "CA", "C", "O", "CB", "CG", "CD1", "CD2",
        "NE1", "CE2", "CE3", "CZ2", "CZ3", "CH2", "OXT",
    ),
    "TYR": (
        "N", "CA", "C", "O", "CB", "CG", "CD1", "CD2",
        "CE1", "CE2", "CZ", "OH", "OXT",
    ),
    "VAL": ("N", "CA", "C", "O", "CB", "CG1", "CG2", "OXT"),
    "UNK": (),
}


class AF3scoreStep:
    """Pipeline step that wraps the AF3Score workflow.

    Parameters
    ----------
    input_pdb_dir : str
        Directory containing input PDB files.
    output_dir : str
        Directory where outputs will be written.
    pipeline_script_dir : str, optional
        Directory containing AF3Score scripts. If not provided,
        scripts are resolved relative to this file.
    python : str, optional
        Python interpreter to use (default: current interpreter).
    db_dir : str, optional
        AlphaFold3 database directory.
    model_dir : str, optional
        AlphaFold3 model parameter directory.
    num_workers : int, optional
        Number of worker processes used in preparation and metrics steps.
        Default is ``max(1, cpu_count() - 4)``.
    """

    def __init__(
        self,
        input_pdb_dir: str,
        output_dir: str,
        pipeline_script_dir: Optional[str] = None,
        python: Optional[str] = None,
        db_dir: str = None,
        model_dir: str = None,
        num_workers: Optional[int] = None,
    ):
        if input_pdb_dir is None:
            raise ValueError("`input_pdb_dir` is required.")
        if output_dir is None:
            raise ValueError("`output_dir` is required.")

        self.input_pdb_dir = os.path.abspath(input_pdb_dir)
        self.output_dir = os.path.abspath(output_dir)
        self.pipeline_script_dir = pipeline_script_dir
        self.python = python or sys.executable
        self.db_dir = db_dir
        self.model_dir = model_dir
        self.num_workers = (
            int(num_workers)
            if num_workers is not None
            else max(1, mp.cpu_count() - 4)
        )

        self._bucket_size: Optional[int] = None
        self._init_output_paths()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _init_output_paths(self) -> None:
        """Initialize all output paths used by the AF3Score workflow."""
        self.af3_input_dir = os.path.join(self.output_dir, "af3_input")
        self.output_dir_cif = os.path.join(self.output_dir, "single_chain_cif")
        self.save_csv = os.path.join(self.output_dir, "single_seq.csv")
        self.output_dir_json = os.path.join(self.output_dir, "json")
        self.output_dir_jax = os.path.join(self.output_dir, "jax")
        self.output_dir_af3score = os.path.join(
            self.output_dir, "af3score_outputs"
        )
        self.metric_csv = os.path.join(self.output_dir, "af3score_metrics.csv")

    def _resolve_run_af3score_script(self) -> str:
        """Resolve run_af3score.py path."""
        if self.pipeline_script_dir is not None:
            script = os.path.join(self.pipeline_script_dir, "run_af3score.py")
        else:
            script = _RUN_AF3SCORE_SCRIPT

        if not os.path.exists(script):
            raise FileNotFoundError(f"run_af3score.py not found: {script}")

        return script

    def _ensure_input_dir(self) -> None:
        """Validate input PDB directory."""
        if not os.path.isdir(self.input_pdb_dir):
            raise FileNotFoundError(
                f"Input PDB directory not found: {self.input_pdb_dir}"
            )

        pdb_files = list(Path(self.input_pdb_dir).glob("*.pdb"))
        if not pdb_files:
            raise FileNotFoundError(
                f"No PDB files found in input directory: {self.input_pdb_dir}"
            )

    def _prepare_output_dirs(self) -> None:
        """Create all required output directories."""
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.af3_input_dir, exist_ok=True)
        os.makedirs(self.output_dir_cif, exist_ok=True)
        os.makedirs(self.output_dir_json, exist_ok=True)
        os.makedirs(self.output_dir_jax, exist_ok=True)
        os.makedirs(self.output_dir_af3score, exist_ok=True)

    def _run_subprocess(
        self,
        cmd: List[str],
        check: bool = True,
        env: Optional[Dict[str, str]] = None,
    ) -> subprocess.CompletedProcess:
        """Execute a subprocess command."""
        print(f"[AF3scoreStep] Running: {' '.join(cmd)}")
        return subprocess.run(cmd, check=check, env=env)

    # ------------------------------------------------------------------
    # Sequence / PDB helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _get_sequence_from_chain(chain) -> str:
        """Convert a Biopython chain object to a single-letter sequence."""
        sequence = ""
        for residue in chain:
            if residue.id[0] == " ":
                resname = residue.get_resname().upper()
                sequence += protein_letters_3to1.get(resname, "X")
        return sequence

    @classmethod
    def _process_single_pdb_for_seq(
        cls, args: Tuple[str, str]
    ) -> Tuple[Optional[str], Optional[Dict[str, str]], Optional[int]]:
        """Extract chain sequences and write per-chain CIF files."""
        input_pdb, output_dir_cif = args
        try:
            parser = PDB.PDBParser(QUIET=True)
            structure = parser.get_structure("structure", input_pdb)
            base_name = os.path.splitext(os.path.basename(input_pdb))[0]

            chain_sequences = {}
            merged_sequence = ""

            for chain in structure[0]:
                chain_id = chain.id
                sequence = cls._get_sequence_from_chain(chain)
                chain_sequences[chain_id] = sequence
                merged_sequence += sequence

                new_structure = PDB.Structure.Structure("new_structure")
                new_model = PDB.Model.Model(0)
                new_structure.add(new_model)
                new_model.add(chain.copy())

                cif_io = MMCIFIO()
                cif_io.set_structure(new_structure)
                cif_output = os.path.join(
                    output_dir_cif, f"{base_name}_chain_{chain_id}.cif"
                )
                cif_io.save(cif_output)

            return base_name, chain_sequences, len(merged_sequence)

        except Exception as e:
            print(f"[AF3scoreStep] Error processing {input_pdb}: {e}")
            return None, None, None

    @staticmethod
    def _format_msa_sequence(sequence: str) -> str:
        """Format a raw sequence into a basic MSA query string."""
        return f">query\n{sequence}\n"

    @staticmethod
    def _get_chain_sequences_from_row(
        row: Dict[str, object]
    ) -> List[Tuple[str, str]]:
        """Extract all non-empty chain sequences from a row dict."""
        chain_sequences = []
        chain_columns = [
            col
            for col in row.keys()
            if col.startswith("chain_") and col.endswith("_seq")
        ]
        for col in chain_columns:
            value = row.get(col, "")
            if value is not None and value != "":
                chain_id = col.split("_")[1]
                chain_sequences.append((chain_id, str(value)))
        return chain_sequences

    @classmethod
    def _generate_json_file(
        cls, task: Tuple[Dict[str, object], str, str]
    ) -> Optional[str]:
        """Generate one AF3 JSON input file."""
        row, cif_dir, output_dir = task
        complex_name = str(row["complex"])
        chain_sequences = cls._get_chain_sequences_from_row(row)

        if not chain_sequences:
            print(f"[AF3scoreStep] Warning: no valid chain sequences for {complex_name}")
            return None

        sequences = []
        for chain_id, sequence in chain_sequences:
            cif_filename = f"{complex_name}_chain_{chain_id}.cif"
            cif_path = os.path.join(cif_dir, cif_filename)

            if not os.path.exists(cif_path):
                print(f"[AF3scoreStep] Warning: {cif_filename} not found")
                continue

            sequences.append(
                {
                    "protein": {
                        "id": chain_id,
                        "sequence": sequence,
                        "modifications": [],
                        "unpairedMsa": cls._format_msa_sequence(sequence),
                        "pairedMsa": cls._format_msa_sequence(sequence),
                        "templates": [
                            {
                                "mmcifPath": cif_path,
                                "queryIndices": list(range(len(sequence))),
                                "templateIndices": list(range(len(sequence))),
                            }
                        ],
                    }
                }
            )

        if not sequences:
            print(f"[AF3scoreStep] Warning: no valid sequence data for {complex_name}")
            return None

        json_data = {
            "dialect": "alphafold3",
            "version": 1,
            "name": complex_name,
            "sequences": sequences,
            "modelSeeds": [10],
            "bondedAtomPairs": None,
            "userCCD": None,
        }

        output_filename = f"{complex_name}.json"
        output_path = os.path.join(output_dir, output_filename)
        with open(output_path, "w") as fw:
            json.dump(json_data, fw, indent=2)

        return output_filename

    def _compute_bucket_size(self) -> int:
        """Compute bucket size from the longest total sequence length in input PDBs."""
        pdb_files = list(Path(self.input_pdb_dir).glob("*.pdb"))
        if not pdb_files:
            raise FileNotFoundError(
                f"No PDB files found in input directory: {self.input_pdb_dir}"
            )

        process_args = [(str(f), self.output_dir_cif) for f in pdb_files]
        with mp.Pool(processes=self.num_workers) as pool:
            results = pool.map(self._process_single_pdb_for_seq, process_args)

        valid_lengths = [length for _, _, length in results if length is not None]
        if not valid_lengths:
            raise RuntimeError("Failed to compute bucket size from input PDBs.")

        self._bucket_size = max(valid_lengths)
        print(f"[AF3scoreStep] Auto-detected bucket_size={self._bucket_size}")
        return self._bucket_size

    def _prepare_af3_inputs(self) -> None:
        """Prepare CIF files, sequence CSV, and AF3 JSON inputs."""
        pdb_files = list(Path(self.input_pdb_dir).glob("*.pdb"))
        process_args = [(str(f), self.output_dir_cif) for f in pdb_files]

        print(f"[AF3scoreStep] Preparing AF3 inputs from {len(pdb_files)} PDB files")

        with mp.Pool(processes=self.num_workers) as pool:
            results = pool.map(self._process_single_pdb_for_seq, process_args)

        sequences_dict = {}
        for base_name, chain_sequences, length in results:
            if base_name is not None:
                sequences_dict[base_name] = {
                    "sequences": chain_sequences,
                    "length": length,
                }

        if not sequences_dict:
            raise RuntimeError("No valid sequence information was extracted.")

        all_chain_ids = set()
        for entry in sequences_dict.values():
            all_chain_ids.update(entry["sequences"].keys())
        all_chain_ids = sorted(all_chain_ids, key=str)

        rows = []
        for complex_name, entry in sequences_dict.items():
            row = {"complex": complex_name, "total_length": entry["length"]}
            for chain_id in all_chain_ids:
                row[f"chain_{chain_id}_seq"] = entry["sequences"].get(chain_id, "")
            rows.append(row)

        fieldnames = ["complex", "total_length"] + [
            f"chain_{chain_id}_seq" for chain_id in all_chain_ids
        ]

        with open(self.save_csv, "w", newline="") as fw:
            writer = csv.DictWriter(fw, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

        print(f"[AF3scoreStep] Sequence CSV written to: {self.save_csv}")

        if self._bucket_size is None:
            lengths = [int(row["total_length"]) for row in rows]
            self._bucket_size = max(lengths)

        json_tasks = [
            (row, self.output_dir_cif, self.output_dir_json) for row in rows
        ]

        with mp.Pool(processes=self.num_workers) as pool:
            json_results = pool.map(self._generate_json_file, json_tasks)

        success_count = sum(1 for r in json_results if r)
        print(
            f"[AF3scoreStep] AF3 JSON generation complete: {success_count} files created"
        )

        if success_count == 0:
            raise RuntimeError("No AF3 JSON files were created.")

    # ------------------------------------------------------------------
    # JAX / H5 helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _load_structure(file_path: str) -> Structure:
        """Load a PDB file and return a Biopython Structure object."""
        structure_id = os.path.basename(file_path).split(".")[0]
        parser = PDB.PDBParser(QUIET=True)
        return parser.get_structure(structure_id, file_path)

    @staticmethod
    def _get_sequence_length(
        structure: Structure, chain_ids: Optional[List[str]] = None
    ) -> int:
        """Calculate total amino acid sequence length."""
        model = structure[0]
        total_length = 0

        if chain_ids is None:
            chain_ids = [chain.id for chain in model]

        for chain_id in chain_ids:
            if chain_id not in model:
                print(f"[AF3scoreStep] Warning: Chain {chain_id} not found")
                continue
            chain = model[chain_id]
            total_length += len([res for res in chain if PDB.is_aa(res)])

        return total_length

    @staticmethod
    def _structure_to_array(
        structure: Structure, chain_ids: Optional[List[str]] = None
    ) -> np.ndarray:
        """Convert structure to [N_res, 24, 3] coordinate array."""
        model = structure[0]
        all_coords_list = []

        if chain_ids is None:
            chain_ids = [chain.id for chain in model]

        for chain_id in chain_ids:
            if chain_id not in model:
                print(f"[AF3scoreStep] Warning: Chain {chain_id} not found")
                continue

            chain = model[chain_id]
            chain_coords_list = []

            for res in chain:
                if not PDB.is_aa(res):
                    continue

                resname = res.get_resname()
                if resname not in ATOM14:
                    print(f"[AF3scoreStep] Warning: Residue {resname} not recognized")
                    continue

                res_coords_24 = np.zeros((24, 3))
                atom_order = ATOM14[resname]

                atom_index = 0
                for atom_name in atom_order:
                    if atom_name in res:
                        coord = res[atom_name].get_coord()
                        if atom_index < 24:
                            res_coords_24[atom_index] = coord
                            atom_index += 1
                        else:
                            break

                chain_coords_list.append(res_coords_24)

            if chain_coords_list:
                chain_coords = np.stack(chain_coords_list, axis=0)
                all_coords_list.append(chain_coords)

        if not all_coords_list:
            raise ValueError("No valid coordinates found in any chain")

        return np.concatenate(all_coords_list, axis=0)

    @staticmethod
    def _save_traced_array(
        traced_array: jax.Array,
        seq_length: int,
        save_path: str,
        metadata: Optional[Dict[str, object]] = None,
    ) -> None:
        """Save JAX array to HDF5."""
        if not save_path.endswith(".h5"):
            save_path = save_path + ".h5"

        numpy_array = np.array(jax.device_get(traced_array))

        with h5py.File(save_path, "w") as f:
            f.create_dataset("coordinates", data=numpy_array)
            f.create_dataset("seq_length", data=seq_length)
            f.create_dataset("shape", data=numpy_array.shape)

            if metadata:
                metadata_grp = f.create_group("metadata")
                for key, value in metadata.items():
                    metadata_grp.attrs[key] = value

    def _pdb_to_traced_array(
        self,
        pdb_path: str,
        chain_ids: Optional[List[str]] = None,
        num_copies: int = 1,
        save_path: Optional[str] = None,
    ):
        """Parse a PDB file, pad to global bucket, repeat, and convert to JAX."""
        if self._bucket_size is None:
            raise RuntimeError("Bucket size has not been initialized.")

        structure = self._load_structure(pdb_path)

        if chain_ids is None:
            chain_ids = [chain.id for chain in structure[0]]

        seq_length = self._get_sequence_length(structure, chain_ids)
        if seq_length > self._bucket_size:
            print(
                f"[AF3scoreStep] Warning: {os.path.basename(pdb_path)} "
                f"sequence length {seq_length} exceeds bucket {self._bucket_size}"
            )
            return None, seq_length

        coords = self._structure_to_array(structure, chain_ids)

        if seq_length < self._bucket_size:
            padding = np.zeros((self._bucket_size - seq_length, 24, 3))
            coords = np.concatenate([coords, padding], axis=0)

        coords_repeated = np.stack([coords] * num_copies)
        jax_array = jnp.array(coords_repeated)

        @jax.jit
        def get_traced_array(x):
            return x

        traced_array = get_traced_array(jax_array)

        if save_path:
            metadata = {
                "pdb_file": os.path.basename(pdb_path),
                "chain_ids": ",".join(chain_ids),
                "num_copies": num_copies,
                "original_length": seq_length,
                "padded_length": self._bucket_size,
            }
            self._save_traced_array(traced_array, seq_length, save_path, metadata)

        return traced_array, seq_length

    def _process_single_pdb_to_h5(
        self, args: Tuple[str, str, Optional[List[str]], int]
    ) -> Tuple[bool, str]:
        """Wrapper for multiprocessing H5 conversion."""
        input_path, output_path, chain_ids, num_copies = args
        try:
            result = self._pdb_to_traced_array(
                pdb_path=input_path,
                chain_ids=chain_ids,
                num_copies=num_copies,
                save_path=output_path,
            )
            if result[0] is None:
                return False, input_path
            return True, input_path
        except Exception as e:
            print(
                f"[AF3scoreStep] Error processing {os.path.basename(input_path)}: {e}"
            )
            return False, input_path

    def _prepare_jax_inputs(self) -> None:
        """Convert all PDBs to H5/JAX inputs without batch splitting."""
        os.makedirs(self.output_dir_jax, exist_ok=True)

        processing_args = []
        for filename in os.listdir(self.input_pdb_dir):
            if filename.endswith(".pdb"):
                input_path = os.path.join(self.input_pdb_dir, filename)
                output_path = os.path.join(
                    self.output_dir_jax, f"{os.path.splitext(filename)[0]}.h5"
                )
                processing_args.append((input_path, output_path, None, 1))

        if not processing_args:
            raise RuntimeError("No valid PDB files to process.")

        print(
            f"[AF3scoreStep] Preparing JAX/H5 inputs for {len(processing_args)} files "
            f"(bucket_size={self._bucket_size}, num_workers={self.num_workers})"
        )

        with mp.Pool(processes=self.num_workers) as pool:
            results = pool.map(self._process_single_pdb_to_h5, processing_args)

        success_paths = [path for success, path in results if success]
        failed_paths = [path for success, path in results if not success]

        print(f"[AF3scoreStep] Successfully processed: {len(success_paths)} files")
        print(f"[AF3scoreStep] Failed: {len(failed_paths)} files")

        if failed_paths:
            preview = ", ".join(os.path.basename(p) for p in failed_paths[:10])
            raise RuntimeError(
                f"Failed to generate H5 for {len(failed_paths)} files. "
                f"Examples: {preview}"
            )

    def _verify_h5_generation(self) -> None:
        """Verify that all input PDB files were converted to H5."""
        pdb_names = {p.stem for p in Path(self.input_pdb_dir).glob("*.pdb")}
        h5_names = {p.stem for p in Path(self.output_dir_jax).glob("*.h5")}

        missing = sorted(pdb_names - h5_names)
        if missing:
            preview = ", ".join(missing[:10])
            raise RuntimeError(
                f"Missing H5 files for {len(missing)} PDBs. Examples: {preview}"
            )

        print(f"[AF3scoreStep] Verified H5 generation: {len(h5_names)} files")

    # ------------------------------------------------------------------
    # AF3Score command helpers
    # ------------------------------------------------------------------

    def _af3score_env(self) -> Dict[str, str]:
        """Build environment variables required by AF3Score inference."""
        env = os.environ.copy()

        env["PATH"] = f"/usr/local/cuda-12.6/bin:{env.get('PATH', '')}"
        env["LD_LIBRARY_PATH"] = (
            f"/usr/local/cuda-12.6/lib64:{env.get('LD_LIBRARY_PATH', '')}"
        )
        env["XLA_FLAGS"] = "--xla_gpu_enable_triton_gemm=true"
        env["XLA_PYTHON_CLIENT_PREALLOCATE"] = "true"
        env["XLA_CLIENT_MEM_FRACTION"] = "0.95"

        cuda_nvcc_path = None
        try:
            completed = subprocess.run(
                [
                    self.python,
                    "-c",
                    "import site; print(site.getsitepackages()[0] + '/nvidia/cuda_nvcc/bin')",
                ],
                capture_output=True,
                text=True,
                check=True,
            )
            cuda_nvcc_path = completed.stdout.strip()
        except Exception:
            cuda_nvcc_path = None

        if cuda_nvcc_path:
            env["PATH"] = f"{cuda_nvcc_path}:{env.get('PATH', '')}"

        return env

    def _build_af3score_cmd(self) -> List[str]:
        """Build AF3Score inference command."""
        script = self._resolve_run_af3score_script()

        if self._bucket_size is None:
            raise RuntimeError("Bucket size has not been initialized.")

        return [
            self.python,
            script,
            f"--db_dir={self.db_dir}",
            f"--model_dir={self.model_dir}",
            f"--batch_json_dir={self.output_dir_json}",
            f"--batch_h5_dir={self.output_dir_jax}",
            f"--output_dir={self.output_dir_af3score}",
            "--run_data_pipeline=False",
            "--run_inference=true",
            "--init_guess=true",
            "--num_samples=1",
            f"--buckets={self._bucket_size}",
            "--write_cif_model=False",
            "--write_summary_confidences=true",
            "--write_full_confidences=true",
            "--write_best_model_root=false",
            "--write_ranking_scores_csv=false",
            "--write_terms_of_use_file=false",
            "--write_fold_input_json_file=false",
        ]

    # ------------------------------------------------------------------
    # Metrics helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _get_chains_from_pdb(pdb_path: str) -> List[str]:
        """Extract all unique chain IDs from a PDB file."""
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure("structure", pdb_path)
        model = structure[0]
        chains = [chain.id for chain in model.get_chains()]
        return sorted(set(chains))

    @staticmethod
    def _get_interface_res_from_pdb(
        pdb_file: str,
        chain1: str = "A",
        chain2: str = "B",
        dist_cutoff: int = 10,
    ) -> Tuple[List[int], List[int]]:
        """Identify interface residues between two chains using CA distances."""
        chain_coords = defaultdict(dict)

        with open(pdb_file, "r") as f:
            for line in f:
                if line.startswith("ATOM"):
                    atom_name = line[12:16].strip()
                    chain_id = line[21].strip()
                    residue_id = int(line[22:26])
                    x = float(line[30:38])
                    y = float(line[38:46])
                    z = float(line[46:54])

                    if atom_name == "CA":
                        chain_coords[chain_id][residue_id] = np.array([x, y, z])

        chain_1_res = sorted(chain_coords[chain1].keys())
        chain_2_res = sorted(chain_coords[chain2].keys())

        chain_1_coords = np.array([chain_coords[chain1][res] for res in chain_1_res])
        chain_2_coords = np.array([chain_coords[chain2][res] for res in chain_2_res])

        dist = np.sqrt(
            np.sum(
                (chain_1_coords[:, None, :] - chain_2_coords[None, :, :]) ** 2,
                axis=2,
            )
        )
        interface_residues = np.where(dist < dist_cutoff)

        interface_1 = sorted(set(chain_1_res[i] for i in interface_residues[0]))
        interface_2 = sorted(set(chain_2_res[i] for i in interface_residues[1]))

        return interface_1, interface_2

    @staticmethod
    def _extract_token_chain_and_res_ids(
        pdb_file: str,
    ) -> Tuple[List[str], List[int]]:
        """Extract token-level chain IDs and residue IDs from a PDB file."""
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure("structure", pdb_file)
        model = structure[0]

        token_chain_ids = []
        token_res_ids = []

        for chain in model:
            for residue in chain:
                if "CA" in residue:
                    token_chain_ids.append(chain.id)
                    token_res_ids.append(residue.id[1])

        return token_chain_ids, token_res_ids

    @classmethod
    def _parse_confidences_json(
        cls, conf_path: str, pdb_path: str
    ) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, float]]:
        """Parse confidences.json and calculate PAE-derived metrics."""
        with open(conf_path) as f:
            conf = json.load(f)

        chains = cls._get_chains_from_pdb(pdb_path)
        pae = np.array(conf["pae"])
        token_chain_ids, token_res_ids = cls._extract_token_chain_and_res_ids(pdb_path)

        chain_indices = {chain: [] for chain in chains}
        for i, chain in enumerate(token_chain_ids):
            chain_indices[chain].append(i)

        chain_pae = {
            chain: float(np.mean(pae[np.ix_(idxs, idxs)]))
            for chain, idxs in chain_indices.items()
        }

        ipae = {}
        pae_interaction = {}

        for ch1, ch2 in combinations(chains, 2):
            try:
                idx1_res, idx2_res = cls._get_interface_res_from_pdb(
                    pdb_path, chain1=ch1, chain2=ch2
                )
                idx1 = [
                    i
                    for i, (res_id, chain) in enumerate(zip(token_res_ids, token_chain_ids))
                    if chain == ch1 and res_id in idx1_res
                ]
                idx2 = [
                    i
                    for i, (res_id, chain) in enumerate(zip(token_res_ids, token_chain_ids))
                    if chain == ch2 and res_id in idx2_res
                ]

                pair_key = f"{ch1}_{ch2}"

                if idx1 and idx2:
                    ipae[pair_key] = float(
                        np.mean(
                            [
                                np.mean(pae[np.ix_(idx1, idx2)]),
                                np.mean(pae[np.ix_(idx2, idx1)]),
                            ]
                        )
                    )

                chain_1_indices = [
                    i for i, chain in enumerate(token_chain_ids) if chain == ch1
                ]
                chain_2_indices = [
                    i for i, chain in enumerate(token_chain_ids) if chain == ch2
                ]

                pae_interaction[pair_key] = float(
                    np.mean(
                        [
                            np.mean(pae[np.ix_(chain_1_indices, chain_2_indices)]),
                            np.mean(pae[np.ix_(chain_2_indices, chain_1_indices)]),
                        ]
                    )
                )

            except Exception as e:
                print(f"[AF3scoreStep] Warning: failed to process pair ({ch1}, {ch2}): {e}")

        return chain_pae, ipae, pae_interaction

    @staticmethod
    def _process_single_description(
        args: Tuple[str, str, str]
    ) -> Tuple[Optional[Dict[str, object]], Optional[str]]:
        """Worker function to process all metrics for a single prediction directory."""
        # from ipsae_calculator import load_af3_pae_and_chains, calculate_ipsae

        description, input_pdb_dir, base_dir = args
        try:
            base_path = Path(base_dir) / description / "seed-10_sample-0"
            summary_path = base_path / "summary_confidences.json"
            conf_path = base_path / "confidences.json"
            pdb_path = Path(input_pdb_dir) / f"{description}.pdb"

            if not summary_path.exists():
                return None, f"{description}: missing summary file"
            if not pdb_path.exists():
                return None, f"{description}: missing pdb file"
            if not conf_path.exists():
                return None, f"{description}: missing conf file"

            # ipsae_metrics = {}
            # pae_matrix, chain_ids, residue_types = load_af3_pae_and_chains(
            #     conf_path, pdb_path
            # )
            # ipsae_dict = calculate_ipsae(
            #     pae_matrix, chain_ids, residue_types, pae_cutoff=10
            # )
            # for k, v in ipsae_dict.items():
            #     ipsae_metrics[f"ipsae_{k}"] = v

            summary = json.loads(summary_path.read_text())
            conf = json.loads(conf_path.read_text())
            chains = AF3scoreStep._get_chains_from_pdb(str(pdb_path))

            iptm = dict(zip(chains, summary.get("chain_iptm", [])))
            ptm = dict(zip(chains, summary.get("chain_ptm", [])))

            iptm_matrix = summary["chain_pair_iptm"]
            interchain_iptm_dict = {}
            num_chains = len(chains)
            for i in range(num_chains):
                for j in range(i + 1, num_chains):
                    interchain_iptm_dict[f"iptm_{chains[i]}_{chains[j]}"] = iptm_matrix[i][j]

            atom_plddts = conf["atom_plddts"]
            atom_chain_ids = conf["atom_chain_ids"]

            chain_plddt = {
                ch: float(
                    np.mean(
                        [pl for pl, cid in zip(atom_plddts, atom_chain_ids) if cid == ch]
                    )
                )
                for ch in chains
            }

            chain_pae, ipae, inter_pae = AF3scoreStep._parse_confidences_json(
                str(conf_path), str(pdb_path)
            )

            result = {
                "description": f"{description}.pdb",
                "ptm": summary.get("ptm", 0.0),
                "iptm": summary.get("iptm", 0.0),
            }

            for ch in chains:
                result[f"chain_{ch}_plddt"] = chain_plddt.get(ch, np.nan)
                result[f"chain_{ch}_pae"] = chain_pae.get(ch, np.nan)
                result[f"chain_{ch}_ptm"] = ptm.get(ch, np.nan)
                result[f"chain_{ch}_iptm"] = iptm.get(ch, np.nan)

            # result.update(ipsae_metrics)
            result.update(interchain_iptm_dict)
            result.update({f"ipae_{k}": v for k, v in ipae.items()})
            result.update({f"inter_pae_{k}": v for k, v in inter_pae.items()})

            return result, None

        except Exception as e:
            return None, f"{description}: {e}"

    def _extract_all_metrics_parallel(
        self,
    ) -> Tuple[List[Dict[str, object]], List[str]]:
        """Extract AF3 metrics from all AF3Score output subdirectories."""
        descriptions = [
            d
            for d in os.listdir(self.output_dir_af3score)
            if (Path(self.output_dir_af3score) / d).is_dir()
        ]
        args_list = [
            (d, self.input_pdb_dir, self.output_dir_af3score) for d in descriptions
        ]

        results = []
        failed = []

        with mp.Pool(processes=self.num_workers) as pool:
            for res, err in pool.map(self._process_single_description, args_list):
                if err:
                    failed.append(err)
                elif res is not None:
                    results.append(res)

        return results, failed

    def _write_metrics_csv(self, rows: List[Dict[str, object]]) -> None:
        """Write extracted metric rows to CSV."""
        if not rows:
            raise RuntimeError("No metrics were extracted.")

        fieldnames = {k for row in rows for k in row.keys()}
        with open(self.metric_csv, "w", newline="") as fw:
            writer = csv.DictWriter(fw, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    def _extract_metrics(self) -> None:
        """Extract metrics and save them to CSV."""
        rows, failed = self._extract_all_metrics_parallel()
        self._write_metrics_csv(rows)

        failed_log_path = Path(self.output_dir_af3score) / "failed_records.txt"
        with open(failed_log_path, "w") as fw:
            fw.write("\n".join(failed))

        print(
            f"[AF3scoreStep] Successfully processed {len(rows)} items, "
            f"Failed: {len(failed)}"
        )
        print(f"[AF3scoreStep] Failure logs written to {failed_log_path}")

    def _verify_metrics_csv(self) -> None:
        """Verify final metrics CSV row count against input PDB count."""
        if not os.path.exists(self.metric_csv):
            raise FileNotFoundError(f"Metrics CSV not found: {self.metric_csv}")

        expected_count = len(list(Path(self.input_pdb_dir).glob("*.pdb")))
        with open(self.metric_csv, "r", newline="") as fr:
            row_count = max(sum(1 for _ in fr) - 1, 0)

        if row_count != expected_count:
            raise RuntimeError(
                f"Metrics verification failed. "
                f"Expected {expected_count} rows, got {row_count}."
            )

        print(f"[AF3scoreStep] Metrics verification successful: {row_count} rows")

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def build_cmd(self) -> List[str]:
        """Return the AF3Score inference command that would be executed by :meth:`run`.

        Returns
        -------
        list[str]
            The AF3Score inference command and its arguments as a list.
        """
        if self._bucket_size is None:
            return ["<bucket_size unresolved until input scan>"]
        return self._build_af3score_cmd()

    def run(self, check: bool = True) -> Dict[str, subprocess.CompletedProcess]:
        """Execute the AF3Score workflow.

        Parameters
        ----------
        check : bool
            If *True* (default), raise :class:`subprocess.CalledProcessError`
            when the inference script exits with a non-zero return code.

        Returns
        -------
        dict[str, subprocess.CompletedProcess]
            Result objects from subprocess-executed stages.
        """
        self._ensure_input_dir()
        self._prepare_output_dirs()

        print("[AF3scoreStep] Step 1/4: computing bucket size")
        self._compute_bucket_size()

        print("[AF3scoreStep] Step 2/4: preparing AF3 inputs")
        self._prepare_af3_inputs()

        print("[AF3scoreStep] Step 3/4: preparing JAX/H5 inputs")
        self._prepare_jax_inputs()
        self._verify_h5_generation()

        print("[AF3scoreStep] Step 4/4: running AF3Score inference")
        af3score_result = self._run_subprocess(
            self._build_af3score_cmd(),
            check=check,
            env=self._af3score_env(),
        )

        print("[AF3scoreStep] Step 5/5: extracting metrics")
        self._extract_metrics()
        self._verify_metrics_csv()

        return {
            "af3score": af3score_result,
        }