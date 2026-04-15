"""
DockQStep - Pipeline step for DockQ evaluation of modeled complexes.

Scans a model directory where each subdirectory corresponds to one
reference PDB in ``reference_dir``.  For every model structure found in a
subdirectory, runs DockQ against the matching reference structure and
extracts pairwise DockQ values for all reported two-chain interfaces.

CSV output contains one row per reference/model PDB pair.
"""

import csv
import os
import re
import subprocess
from typing import Dict, List, Optional


VALID_PDB_EXTENSIONS = (".pdb", ".ent")


class DockQStep:
    """Pipeline step that wraps DockQ evaluation.

    Parameters
    ----------
    reference_dir : str
        Directory containing reference PDB files.
    model_dir : str
        Directory containing model subdirectories.  Each subdirectory name
        must match a reference PDB basename in ``reference_dir``.
    output_csv : str
        Path to the output CSV file.
    dockq_exe : str, optional
        DockQ executable name or path (default: ``"DockQ"``).
    dockq_args : list[str], optional
        Extra command-line arguments passed to DockQ before the input files.
        When *None*, defaults to ``["--short"]``.
    recursive : bool, optional
        If *True*, search model subdirectories recursively for PDB files.
        Otherwise only scan the first directory level (default: ``False``).
    """

    def __init__(
        self,
        reference_dir: str,
        model_dir: str,
        output_csv: str,
        dockq_exe: str = "DockQ",
        dockq_args: Optional[List[str]] = None,
        recursive: bool = False,
    ):
        self.reference_dir = reference_dir
        self.model_dir = model_dir
        self.output_csv = output_csv
        self.dockq_exe = dockq_exe
        self.dockq_args = dockq_args or ["--short"]
        self.recursive = recursive

    # ------------------------------------------------------------------
    # Discovery helpers
    # ------------------------------------------------------------------

    def _find_reference_pdb(self, name: str) -> str:
        """Return the matching reference PDB path for a target name."""
        for ext in VALID_PDB_EXTENSIONS:
            candidate = os.path.join(self.reference_dir, f"{name}{ext}")
            if os.path.isfile(candidate):
                return candidate
        raise FileNotFoundError(
            f"No reference PDB found for '{name}' in '{self.reference_dir}'."
        )

    def _collect_model_pdbs(self, target_dir: str) -> List[str]:
        """Collect model PDB files from a target directory."""
        model_pdbs = []

        if self.recursive:
            for root, _, files in os.walk(target_dir):
                for filename in sorted(files):
                    if filename.lower().endswith(VALID_PDB_EXTENSIONS):
                        model_pdbs.append(os.path.join(root, filename))
        else:
            for filename in sorted(os.listdir(target_dir)):
                path = os.path.join(target_dir, filename)
                if os.path.isfile(path) and filename.lower().endswith(
                    VALID_PDB_EXTENSIONS
                ):
                    model_pdbs.append(path)

        return model_pdbs

    def _iter_reference_model_pairs(self):
        """Yield ``(target_name, reference_pdb, model_pdb)`` tuples."""
        if not os.path.isdir(self.reference_dir):
            raise NotADirectoryError(
                f"reference_dir does not exist or is not a directory: "
                f"'{self.reference_dir}'"
            )
        if not os.path.isdir(self.model_dir):
            raise NotADirectoryError(
                f"model_dir does not exist or is not a directory: "
                f"'{self.model_dir}'"
            )

        for entry in sorted(os.listdir(self.model_dir)):
            target_dir = os.path.join(self.model_dir, entry)
            if not os.path.isdir(target_dir):
                continue

            reference_pdb = self._find_reference_pdb(entry)
            model_pdbs = self._collect_model_pdbs(target_dir)

            for model_pdb in model_pdbs:
                yield entry, reference_pdb, model_pdb

    # ------------------------------------------------------------------
    # Command builders / parsers
    # ------------------------------------------------------------------

    def _build_cmd(self, model_pdb: str, reference_pdb: str) -> List[str]:
        """Build the command list for DockQ evaluation."""
        cmd = [self.dockq_exe]
        cmd += list(self.dockq_args)
        cmd += [model_pdb, reference_pdb]
        return cmd

    def _normalize_chain_pair(self, pair_text: str) -> str:
        """Normalize a two-chain interface label.

        Examples
        --------
        ``"BA"`` -> ``"A_B"``
        ``"AC"`` -> ``"A_C"``
        """
        pair_text = pair_text.strip()
        if len(pair_text) != 2:
            return pair_text
        chains = sorted(pair_text)
        return f"{chains[0]}_{chains[1]}"

    def _parse_short_output(self, stdout: str) -> Dict[str, float]:
        """Parse DockQ ``--short`` output into pairwise DockQ values.

        Returns
        -------
        dict[str, float]
            Mapping from normalized chain-pair label (for example ``"A_B"``)
            to DockQ value.
        """
        results: Dict[str, float] = {}

        for line in stdout.splitlines():
            line = line.strip()
            if not line.startswith("DockQ "):
                continue

            dockq_match = re.search(r"\bDockQ\s+([0-9]*\.?[0-9]+)\b", line)
            mapping_match = re.search(r"\bmapping\s+([A-Za-z0-9]+):([A-Za-z0-9]+)\b", line)

            if dockq_match is None or mapping_match is None:
                continue

            dockq_value = float(dockq_match.group(1))
            model_pair = mapping_match.group(1)
            reference_pair = mapping_match.group(2)

            # 优先使用 reference pair 作为最终链对标签；若缺失则退回 model pair
            if len(reference_pair) == 2:
                pair_key = self._normalize_chain_pair(reference_pair)
            else:
                pair_key = self._normalize_chain_pair(model_pair)

            results[pair_key] = dockq_value

        return results

    # ------------------------------------------------------------------
    # Single evaluation
    # ------------------------------------------------------------------

    def evaluate_pair(self, reference_pdb: str, model_pdb: str) -> Dict[str, object]:
        """Run DockQ for one reference/model pair and parse results.

        Returns
        -------
        dict
            Result dictionary containing metadata and pairwise DockQ values.
        """
        cmd = self._build_cmd(model_pdb=model_pdb, reference_pdb=reference_pdb)
        print(f"[DockQStep] Running: {' '.join(cmd)}")

        completed = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True,
        )

        pair_scores = self._parse_short_output(completed.stdout)

        result: Dict[str, object] = {
            "reference_pdb": reference_pdb,
            "model_pdb": model_pdb,
        }
        result.update(pair_scores)
        return result

    # ------------------------------------------------------------------
    # CSV helpers
    # ------------------------------------------------------------------

    def _write_csv(self, rows: List[Dict[str, object]]) -> None:
        """Write evaluation rows to CSV."""
        os.makedirs(os.path.dirname(os.path.abspath(self.output_csv)), exist_ok=True)

        fieldnames = ["target_name", "reference_pdb", "model_pdb"]
        pair_columns = sorted(
            {
                key
                for row in rows
                for key in row.keys()
                if key not in ("target_name", "reference_pdb", "model_pdb")
            }
        )
        fieldnames += pair_columns

        with open(self.output_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def run(self) -> List[Dict[str, object]]:
        """Evaluate all reference/model pairs and save results to CSV.

        Returns
        -------
        list[dict]
            All collected result rows.
        """
        rows: List[Dict[str, object]] = []

        for target_name, reference_pdb, model_pdb in self._iter_reference_model_pairs():
            try:
                row = self.evaluate_pair(
                    reference_pdb=reference_pdb,
                    model_pdb=model_pdb,
                )
                row["target_name"] = target_name
                # 调整列顺序时希望 target_name 更靠前
                row = {
                    "target_name": row["target_name"],
                    "reference_pdb": row["reference_pdb"],
                    "model_pdb": row["model_pdb"],
                    **{
                        k: v
                        for k, v in row.items()
                        if k not in ("target_name", "reference_pdb", "model_pdb")
                    },
                }
                rows.append(row)
            except subprocess.CalledProcessError as exc:
                print(
                    f"[DockQStep] DockQ failed for model '{model_pdb}' "
                    f"vs reference '{reference_pdb}': {exc}"
                )

        if rows:
            self._write_csv(rows)
        else:
            # 即使没有结果，也输出空表头
            os.makedirs(
                os.path.dirname(os.path.abspath(self.output_csv)),
                exist_ok=True,
            )
            with open(self.output_csv, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=["target_name", "reference_pdb", "model_pdb"],
                )
                writer.writeheader()

        return rows