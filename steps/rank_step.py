import csv
import os
import shutil
import sys
from typing import Dict, List, Optional

VALID_GENTYPES = frozenset({"binder", "nanobody", "antibody"})

# 三字母到单字母氨基酸映射常量 (放入模块级别，避免重复分配)
THREE_TO_ONE = {
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
}


class RankStep:
    """Pipeline step that ranks final designs and exports selected outputs."""

    def __init__(
        self,
        gentype: str,
        dockq_csv: str,
        rosetta_csv: str,
        af3_csv: str,
        filtered_pdb_dir: str,
        output_dir: str,
        dockq_threshold: float = 0.49,
        output_csv_name: str = "ranked_designs.csv",
    ):
        if gentype not in VALID_GENTYPES:
            raise ValueError(
                f"Invalid gentype '{gentype}'. Must be one of: {tuple(VALID_GENTYPES)}"
            )

        self.gentype = gentype
        self.dockq_csv = dockq_csv
        self.rosetta_csv = rosetta_csv
        self.af3_csv = af3_csv
        self.filtered_pdb_dir = filtered_pdb_dir
        self.output_dir = output_dir
        self.dockq_threshold = float(dockq_threshold)
        self.output_csv_name = output_csv_name

    # ------------------------------------------------------------------
    # CSV helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _read_csv(csv_path: str) -> List[Dict[str, str]]:
        if not os.path.isfile(csv_path):
            raise FileNotFoundError(f"CSV file not found: {csv_path}")

        with open(csv_path, newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            if reader.fieldnames is None:
                raise ValueError(f"CSV file has no header: {csv_path}")
            return list(reader)

    @staticmethod
    def _write_csv(csv_path: str, rows: List[Dict[str, object]]) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(csv_path)), exist_ok=True)

        if not rows:
            open(csv_path, "w", newline="", encoding="utf-8").close()
            return

        # 获取所有去重且保持顺序的列名
        fieldnames = list(dict.fromkeys(key for row in rows for key in row))

        with open(csv_path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    @staticmethod
    def _safe_float(value: object) -> Optional[float]:
        if value is None or value == "":
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    # ------------------------------------------------------------------
    # Sequence Extraction
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_chains_from_pdb(pdb_file_path: str) -> Dict[str, str]:
        """
        Extract amino acid sequences per chain from a PDB file.
        Returns a dict mapping chain ID to its one-letter sequence.
        """
        if not os.path.isfile(pdb_file_path):
            return {}

        sequences: Dict[str, List[str]] = {}
        try:
            with open(pdb_file_path, "r", encoding="utf-8") as file:
                for line in file:
                    if not line.startswith("ATOM"):
                        continue

                    # 标准PDB格式切片: 12-15 为原子名, 17-19 为残基名, 21 为链ID
                    if line[12:16] == " CA ":
                        chain_id = line[21]
                        res_name = line[17:20].strip()

                        aa = THREE_TO_ONE.get(res_name)
                        if aa:
                            sequences.setdefault(chain_id, []).append(aa)

            # 合并每条链的氨基酸列表为字符串
            return {chain: "".join(seq) for chain, seq in sequences.items()}

        except (FileNotFoundError, IOError) as e:
            print(f"[Warning] Error reading {pdb_file_path}: {e}", file=sys.stderr)
            return {}

    # ------------------------------------------------------------------
    # DockQ processing
    # ------------------------------------------------------------------

    def _build_best_dockq_rows(self) -> List[Dict[str, object]]:
        rows = self._read_csv(self.dockq_csv)
        if not rows:
            return []

        required_cols = {"target_name", "reference_pdb", "model_pdb"}
        if not required_cols.issubset(rows[0].keys()):
            missing = required_cols - set(rows[0].keys())
            raise ValueError(
                f"dockq_csv is missing required columns: {sorted(missing)}"
            )

        best_rows: Dict[str, Dict[str, object]] = {}

        for row in rows:
            target_name = row.get("target_name", "").strip()
            if not target_name:
                continue

            numeric_values: List[float] = []
            for key, value in row.items():
                if key not in required_cols:
                    value_f = self._safe_float(value)
                    if value_f is not None:
                        numeric_values.append(value_f)

            if not numeric_values:
                continue

            dockq_mean = sum(numeric_values) / len(numeric_values)

            if dockq_mean <= self.dockq_threshold:
                continue

            existing_best = best_rows.get(target_name)
            if existing_best is None or dockq_mean > existing_best["dockq_mean"]:
                best_rows[target_name] = {
                    "target_name": target_name,
                    "reference_pdb": row["reference_pdb"],
                    "model_pdb": row["model_pdb"],
                    "dockq_mean": dockq_mean,
                }

        return list(best_rows.values())

    # ------------------------------------------------------------------
    # Merge helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _index_by_key(
        rows: List[Dict[str, str]], key_name: str
    ) -> Dict[str, Dict[str, str]]:
        return {
            row[key_name].strip(): row
            for row in rows
            if row.get(key_name) and row[key_name].strip()
        }

    def _get_rank_metric_col(self, af3_rows: List[Dict[str, str]]) -> str:
        if not af3_rows:
            raise ValueError("AF3 CSV is empty.")

        af3_cols = set(af3_rows[0].keys())

        if self.gentype == "binder":
            if "iptm" not in af3_cols:
                raise ValueError("AF3 CSV must contain column 'iptm' for binder mode.")
            return "iptm"

        for candidate in ("iptm_A_C", "chain_A_iptm"):
            if candidate in af3_cols:
                return candidate

        raise ValueError(
            f"AF3 CSV does not contain a valid antibody/nanobody rank column. "
            f"Expected one of: ['iptm_A_C', 'chain_A_iptm']"
        )

    def _merge_rows(self) -> List[Dict[str, object]]:
        dockq_rows = self._build_best_dockq_rows()
        if not dockq_rows:
            return []

        rosetta_rows = self._read_csv(self.rosetta_csv)
        af3_rows = self._read_csv(self.af3_csv)

        if not rosetta_rows or not af3_rows:
            raise ValueError("Rosetta or AF3 CSV is empty.")

        rank_metric_col = self._get_rank_metric_col(af3_rows)
        rosetta_index = self._index_by_key(rosetta_rows, "pdb_name")
        af3_index = self._index_by_key(af3_rows, "pdb_name")

        merged_rows: List[Dict[str, object]] = []

        for dockq_row in dockq_rows:
            target_name = str(dockq_row["target_name"])
            rosetta_row = rosetta_index.get(target_name)
            af3_row = af3_index.get(target_name)

            if not rosetta_row or not af3_row:
                continue

            interface_score = self._safe_float(rosetta_row.get("interface_score"))
            rank_metric = self._safe_float(af3_row.get(rank_metric_col))

            if interface_score is None or rank_metric is None:
                continue

            # 构建基础合并行
            merged = dict(af3_row)
            merged.update(
                {
                    "target_name": target_name,
                    "reference_pdb": dockq_row["reference_pdb"],
                    "model_pdb": dockq_row["model_pdb"],
                    "dockq_mean": dockq_row["dockq_mean"],
                    "pdb_name": rosetta_row["pdb_name"],
                    "interface_score": interface_score,
                    "rank_score": interface_score + 100.0 * rank_metric,
                }
            )

            # ======= 新增：提取序列信息并写入指定列 =======
            pdb_path = os.path.join(self.filtered_pdb_dir, f"{target_name}.pdb")
            chain_seqs = self._extract_chains_from_pdb(pdb_path)

            if self.gentype in ("binder", "nanobody"):
                merged["binder_seq"] = chain_seqs.get("A", "")
            elif self.gentype == "antibody":
                merged["heavy_seq"] = chain_seqs.get("A", "")
                merged["light_seq"] = chain_seqs.get("B", "")
            # ============================================

            merged_rows.append(merged)

        merged_rows.sort(key=lambda x: x["rank_score"], reverse=True)
        return merged_rows

    # ------------------------------------------------------------------
    # PDB export
    # ------------------------------------------------------------------

    def _copy_selected_pdbs(self, rows: List[Dict[str, object]]) -> None:
        if not rows:
            return

        pdb_output_dir = os.path.join(self.output_dir, "pdbs")
        os.makedirs(pdb_output_dir, exist_ok=True)

        for row in rows:
            target_name = str(row.get("target_name", "")).strip()
            if not target_name:
                continue

            src_pdb = os.path.join(self.filtered_pdb_dir, f"{target_name}.pdb")
            if os.path.isfile(src_pdb):
                dst_pdb = os.path.join(pdb_output_dir, f"{target_name}.pdb")
                shutil.copy2(src_pdb, dst_pdb)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def run(self) -> str:
        os.makedirs(self.output_dir, exist_ok=True)

        merged_rows = self._merge_rows()

        output_csv = os.path.join(self.output_dir, self.output_csv_name)
        self._write_csv(output_csv, merged_rows)

        self._copy_selected_pdbs(merged_rows)

        print(
            f"[RankStep] Done ({self.gentype}): rows={len(merged_rows)}, output_csv={output_csv}"
        )
        return output_csv
