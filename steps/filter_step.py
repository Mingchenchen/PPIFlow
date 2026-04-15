"""
FilterStep - Pipeline step for filtering PDB files by metrics from a CSV file.

The CSV file must contain:
  - filename: PDB filename
  - one or more metric columns

This step filters rows according to the provided metric thresholds, then creates
symbolic links in ``outdir`` pointing to the original files in ``pdb_dir``,
and optionally writes the filtered rows into a new CSV file.
"""

import csv
import os
from typing import Dict, List, Optional, Union


_VALID_OPS = (">", ">=", "<", "<=", "==", "!=")


class FilterStep:
    """Pipeline step that filters PDB files using metrics from a CSV file."""

    def __init__(
        self,
        pdb_dir: str,
        csv_file: str,
        outdir: str,
        filters: Dict[str, Union[float, int, Dict[str, Union[float, int]]]],
        output_csv: Optional[str] = None,
        filename_col: str = "filename",
    ):
        self.pdb_dir = pdb_dir
        self.csv_file = csv_file
        self.outdir = outdir
        self.filters = filters
        self.output_csv = output_csv
        self.filename_col = filename_col

        self._validate_filters()

    # ------------------------------------------------------------------
    # Validation helpers
    # ------------------------------------------------------------------

    def _validate_filters(self):
        if not isinstance(self.filters, dict) or not self.filters:
            raise ValueError("'filters' must be a non-empty dict.")

        for metric, condition in self.filters.items():
            if isinstance(condition, dict):
                for op, value in condition.items():
                    if op not in _VALID_OPS:
                        raise ValueError(f"Invalid operator '{op}'")
                    if not isinstance(value, (int, float)):
                        raise ValueError("Threshold must be numeric.")
            elif not isinstance(condition, (int, float)):
                raise ValueError("Filter must be numeric or dict.")

    # ------------------------------------------------------------------
    # CSV helpers
    # ------------------------------------------------------------------

    def _read_csv_rows(self) -> List[dict]:
        if not os.path.isfile(self.csv_file):
            raise FileNotFoundError(self.csv_file)

        with open(self.csv_file, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            fieldnames = reader.fieldnames or []

        if self.filename_col not in fieldnames:
            raise ValueError(f"Missing column: {self.filename_col}")

        for metric in self.filters:
            if metric not in fieldnames:
                raise ValueError(f"Missing metric column: {metric}")

        return rows

    def _to_float(self, value, metric, filename):
        try:
            return float(value)
        except Exception:
            raise ValueError(
                f"Invalid value for metric '{metric}' in '{filename}': {value}"
            )

    def _write_filtered_csv(self, rows: List[dict]):
        if self.output_csv is None:
            return

        os.makedirs(os.path.dirname(os.path.abspath(self.output_csv)), exist_ok=True)

        with open(self.csv_file, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames or []

        with open(self.output_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    # ------------------------------------------------------------------
    # Filter logic
    # ------------------------------------------------------------------

    def _compare(self, actual, op, threshold):
        if op == ">":
            return actual > threshold
        if op == ">=":
            return actual >= threshold
        if op == "<":
            return actual < threshold
        if op == "<=":
            return actual <= threshold
        if op == "==":
            return actual == threshold
        if op == "!=":
            return actual != threshold
        raise ValueError(op)

    def _row_matches(self, row):
        filename = row.get(self.filename_col, "")

        for metric, condition in self.filters.items():
            val = self._to_float(row.get(metric), metric, filename)

            if isinstance(condition, dict):
                for op, th in condition.items():
                    if not self._compare(val, op, th):
                        return False
            else:
                if val < condition:
                    return False

        return True

    def _get_filtered_rows(self):
        rows = self._read_csv_rows()
        return [r for r in rows if self._row_matches(r)]

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def run(self) -> List[str]:
        if not os.path.isdir(self.pdb_dir):
            raise NotADirectoryError(self.pdb_dir)

        os.makedirs(self.outdir, exist_ok=True)

        filtered_rows = self._get_filtered_rows()
        filenames = [
            r[self.filename_col] for r in filtered_rows if r.get(self.filename_col)
        ]

        linked_files = []

        for fname in filenames:
            src = os.path.abspath(os.path.join(self.pdb_dir, fname))
            dst = os.path.join(self.outdir, fname)

            if not os.path.isfile(src):
                continue

            # 如果已存在，先删除
            if os.path.lexists(dst):
                os.remove(dst)

            os.symlink(src, dst)
            linked_files.append(dst)

        self._write_filtered_csv(filtered_rows)

        print(
            f"[FilterStep] Filtered {len(filtered_rows)} entries, "
            f"linked {len(linked_files)} files to: {self.outdir}"
        )

        if self.output_csv:
            print(f"[FilterStep] Filtered CSV written to: {self.output_csv}")

        return linked_files
