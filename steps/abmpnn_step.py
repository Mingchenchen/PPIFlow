"""
AbMPNNStep - Pipeline step for AbMPNN sequence design.

Execution order:
  1. Run make_fix_csv.py to generate fixed_positions.csv
  2. Optionally merge user-provided fixed positions into fixed_positions.csv
  3. Run protein_mpnn_run.py for AbMPNN design
"""

import csv
import os
import re
import subprocess
import sys
from typing import Dict, List, Optional


# Resolve the tools directory relative to this file
_TOOLS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "tools",
    "AbMPNN",
)

_FIX_CSV_SCRIPT = os.path.join(_TOOLS_DIR, "make_fix_csv.py")
_ABMPNN_SCRIPT = os.path.join(_TOOLS_DIR, "protein_mpnn_run.py")


class AbMPNNStep:
    """Pipeline step that wraps AbMPNN preprocessing and sequence design.

    Parameters
    ----------
    folder_with_pdbs : str
        Directory containing input PDB files.
    output_dir : str
        Directory where AbMPNN outputs will be written.
    pipeline_script_dir : str, optional
        Base directory containing ``make_fix_csv.py`` and
        ``protein_mpnn_run.py``. If not provided, scripts are resolved
        relative to this file.
    python : str, optional
        Python interpreter to use (default: current interpreter).
    checkpoint_dir : str, optional
        Path to ProteinMPNN model weights directory.
    model_name : str, optional
        Model name to pass to ``protein_mpnn_run.py`` (default: ``"abmpnn"``).
    chain_list : str, optional
        Chains to design (default: ``"A"``).
    num_seq_per_target : int, optional
        Number of sequences to sample per target (default: 8).
    sampling_temp : float, optional
        Sampling temperature (default: 0.5).
    seed : int, optional
        Random seed (default: 37).
    batch_size : int, optional
        Batch size for ProteinMPNN (default: same as ``num_seq_per_target``).
    omit_AAs : str, optional
        Amino acids to omit during design (default: ``"C"``).
    fix_positions : str, optional
        Extra fixed positions to merge into every row of ``fixed_positions.csv``.
        Example: ``"A6,A16,A41,A42,B3"``.
        This is compatible with nanobody (A only) and antibody (A/B).
    """

    def __init__(
        self,
        folder_with_pdbs: str,
        output_dir: str,
        pipeline_script_dir: Optional[str] = None,
        python: Optional[str] = None,
        checkpoint_dir: Optional[str] = None,
        model_name: str = "abmpnn",
        chain_list: str = "A",
        num_seq_per_target: int = 8,
        sampling_temp: float = 0.1,
        seed: int = 37,
        batch_size: Optional[int] = None,
        omit_AAs: str = "C",
        fix_positions: Optional[str] = None,
    ):
        self.folder_with_pdbs = folder_with_pdbs
        self.output_dir = output_dir
        self.pipeline_script_dir = pipeline_script_dir
        self.python = python or sys.executable
        self.checkpoint_dir = checkpoint_dir or os.path.join(
            _TOOLS_DIR, "model_weights"
        )
        self.model_name = model_name
        self.chain_list = chain_list
        self.num_seq_per_target = num_seq_per_target
        self.sampling_temp = sampling_temp
        self.seed = seed
        self.batch_size = batch_size or num_seq_per_target
        self.omit_AAs = omit_AAs
        self.fix_positions = fix_positions

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve_fix_csv_script_path(self) -> str:
        """Resolve make_fix_csv.py path."""
        if self.pipeline_script_dir is not None:
            script = os.path.join(
                self.pipeline_script_dir,
                "demo_scripts",
                "make_fix_csv.py",
            )
        else:
            script = _FIX_CSV_SCRIPT

        if not os.path.exists(script):
            raise FileNotFoundError(f"make_fix_csv script not found: {script}")

        return script

    def _resolve_abmpnn_script_path(self) -> str:
        """Resolve protein_mpnn_run.py path."""
        if self.pipeline_script_dir is not None:
            script = os.path.join(
                self.pipeline_script_dir,
                "protein_mpnn_run.py",
            )
        else:
            script = _ABMPNN_SCRIPT

        if not os.path.exists(script):
            raise FileNotFoundError(f"AbMPNN script not found: {script}")

        return script

    def _fixed_positions_csv_path(self) -> str:
        """Return the generated fixed_positions.csv path."""
        return os.path.join(self.folder_with_pdbs, "fixed_positions.csv")

    def _build_fix_csv_cmd(self):
        """Build the command list for fixed-position CSV generation."""
        if self.folder_with_pdbs is None:
            raise ValueError("`folder_with_pdbs` is required.")

        script = self._resolve_fix_csv_script_path()

        cmd = [
            self.python,
            script,
            self.folder_with_pdbs,
        ]

        return cmd

    def _build_abmpnn_cmd(self):
        """Build the command list for AbMPNN sequence design."""
        if self.folder_with_pdbs is None:
            raise ValueError("`folder_with_pdbs` is required.")
        if self.output_dir is None:
            raise ValueError("`output_dir` is required.")
        if self.checkpoint_dir is None:
            raise ValueError("`checkpoint_dir` is required.")

        script = self._resolve_abmpnn_script_path()
        fixed_positions_csv = self._fixed_positions_csv_path()

        if not os.path.exists(fixed_positions_csv):
            raise FileNotFoundError(
                f"Fixed positions CSV not found: {fixed_positions_csv}"
            )

        cmd = [
            self.python,
            script,
            "--path_to_model_weights",
            self.checkpoint_dir,
            "--model_name",
            self.model_name,
            "--folder_with_pdbs",
            self.folder_with_pdbs,
            "--out_folder",
            self.output_dir,
            "--chain_list",
            self.chain_list,
            "--position_list",
            fixed_positions_csv,
            "--num_seq_per_target",
            str(self.num_seq_per_target),
            "--sampling_temp",
            str(self.sampling_temp),
            "--seed",
            str(self.seed),
            "--batch_size",
            str(self.batch_size),
            "--omit_AAs",
            self.omit_AAs,
        ]

        return cmd

    def _parse_fix_positions(self, text: str) -> Dict[str, List[int]]:
        """
        Parse strings like:
            A6,A16,A41,A42,B3
        into:
            {"A": [6, 16, 41, 42], "B": [3]}
        """
        result: Dict[str, set] = {"A": set(), "B": set()}

        if text is None:
            return {"A": [], "B": []}

        for token in text.split(","):
            token = token.strip()
            if not token:
                continue

            m = re.fullmatch(r"([A-Za-z])(\d+)", token)
            if not m:
                raise ValueError(f"Invalid fixed position token: {token!r}")

            chain_id = m.group(1).upper()
            residue_num = int(m.group(2))

            if chain_id not in result:
                raise ValueError(
                    f"Unsupported chain in fix_positions: {chain_id!r}. "
                    f"Only A/B are supported."
                )

            result[chain_id].add(residue_num)

        return {
            "A": sorted(result["A"]),
            "B": sorted(result["B"]),
        }

    def _parse_motif_index(self, motif_index: str) -> Dict[str, List[int]]:
        """
        Parse AbMPNN motif_index string:
            "1 2 3-4 5 6"
        into:
            {"A": [1, 2, 3], "B": [4, 5, 6]}

        Compatible with empty left/right segments.
        """
        motif_index = (motif_index or "").strip()
        if "-" in motif_index:
            left, right = motif_index.split("-", 1)
        else:
            left, right = motif_index, ""

        def _parse_part(part: str) -> List[int]:
            items = []
            for x in part.strip().split():
                if not x:
                    continue
                items.append(int(x))
            return sorted(set(items))

        return {
            "A": _parse_part(left),
            "B": _parse_part(right),
        }

    def _format_motif_index(self, chain_to_positions: Dict[str, List[int]]) -> str:
        """
        Format:
            {"A": [1,2], "B": [3]}
        into:
            "1 2-3"
        """
        a_str = " ".join(str(x) for x in sorted(set(chain_to_positions.get("A", []))))
        b_str = " ".join(str(x) for x in sorted(set(chain_to_positions.get("B", []))))
        return f"{a_str}-{b_str}"

    def _merge_fix_positions_into_csv(self) -> None:
        """
        Merge self.fix_positions into every row of fixed_positions.csv.

        CSV schema:
            pdb_name,motif_index

        Examples
        --------
        original:
            sample_0,1 2 3-5 6 7
        fix_positions:
            A6,A16,A41,A42,B3

        result:
            sample_0,1 2 3 6 16 41 42-3 5 6 7
        """
        if not self.fix_positions:
            return

        csv_path = self._fixed_positions_csv_path()
        if not os.path.isfile(csv_path):
            raise FileNotFoundError(f"fixed_positions.csv not found: {csv_path}")

        extra_positions = self._parse_fix_positions(self.fix_positions)

        rows = []
        with open(csv_path, newline="") as fh:
            reader = csv.DictReader(fh)
            required = {"pdb_name", "motif_index"}
            if not required.issubset(set(reader.fieldnames or [])):
                raise ValueError(
                    f"{csv_path} must contain columns {required}, "
                    f"got {reader.fieldnames}"
                )

            for row in reader:
                chain_to_positions = self._parse_motif_index(row.get("motif_index", ""))

                merged = {
                    "A": sorted(
                        set(chain_to_positions["A"]) | set(extra_positions["A"])
                    ),
                    "B": sorted(
                        set(chain_to_positions["B"]) | set(extra_positions["B"])
                    ),
                }

                row["motif_index"] = self._format_motif_index(merged)
                rows.append(row)

        with open(csv_path, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=["pdb_name", "motif_index"])
            writer.writeheader()
            writer.writerows(rows)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def build_cmd(self):
        """Return the command lists that would be executed by :meth:`run`.

        Returns
        -------
        dict[str, list[str]]
            Command lists for preprocessing and AbMPNN execution.
        """
        return {
            "make_fix_csv": self._build_fix_csv_cmd(),
            "abmpnn": self._build_abmpnn_cmd()
            if os.path.exists(self._fixed_positions_csv_path())
            else None,
        }

    def run(self, check: bool = True):
        """Execute preprocessing and AbMPNN as subprocesses.

        Parameters
        ----------
        check : bool
            If *True* (default), raise :class:`subprocess.CalledProcessError`
            when a command exits with a non-zero return code.

        Returns
        -------
        dict[str, subprocess.CompletedProcess]
            Result objects from :func:`subprocess.run` for each stage.
        """
        os.makedirs(self.output_dir, exist_ok=True)

        fix_csv_cmd = self._build_fix_csv_cmd()
        print(f"[AbMPNNStep] Running make_fix_csv: {' '.join(fix_csv_cmd)}")
        fix_csv_result = subprocess.run(fix_csv_cmd, check=check)

        if self.fix_positions:
            print(
                f"[AbMPNNStep] Merging extra fix_positions into fixed_positions.csv: "
                f"{self.fix_positions}"
            )
            self._merge_fix_positions_into_csv()

        abmpnn_cmd = self._build_abmpnn_cmd()
        print(f"[AbMPNNStep] Running AbMPNN: {' '.join(abmpnn_cmd)}")
        abmpnn_result = subprocess.run(abmpnn_cmd, check=check)

        return {
            "make_fix_csv": fix_csv_result,
            "abmpnn": abmpnn_result,
        }
