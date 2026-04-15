"""
ReFoldStep - Pipeline step for direct AF3-based structure refolding.

This step follows the integrated AF3 workflow but executes everything
directly on the current compute node, without Slurm job submission.

Workflow:
  1. Convert input PDBs to FASTA via pdb2fa.py
  2. Generate AF3 JSON inputs via gen_json.py
  3. Run local MSA/data-pipeline for each JSON folder via run_alphafold.py
  4. Optionally remove templates from generated JSON files
  5. Run local AF3 inference for each prepared JSON folder
  6. Extract AF3 metrics via get_metric.py and save CSV to output_dir

The class is implemented in a style aligned with ``ppiflow_step.py``.
"""

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, List, Optional


class ReFoldStep:
    """Pipeline step that performs direct AF3 refolding from a PDB directory.

    Parameters
    ----------
    pdb_dir : str
        Directory containing input ``.pdb`` files.
    output_dir : str
        Root directory where all outputs will be written.
    af3_python : str
        Python interpreter for AlphaFold3-related scripts.
    db_dir : str
        AF3 database directory.
    model_dir : str
        AF3 model parameter directory.
    hmmer_path : str
        HMMER binary directory to prepend to ``PATH``.
    python : str, optional
        Python interpreter used for auxiliary scripts; default is current
        interpreter.

    Pipeline control parameters
    ---------------------------
    run_MSA : bool, optional
        Whether to run local MSA/data-pipeline generation.
    remove_template : bool, optional
        Whether to remove ``templates`` fields from JSON files in ``json2``.
    run_AF3_inference : bool, optional
        Whether to run AF3 inference from ``json2``.

    MSA-related parameters
    ----------------------
    seed_num : str, optional
        Seed number passed to ``gen_json.py``.

    Environment parameters
    ----------------------
    cuda_bin : str, optional
        CUDA bin directory prepended to ``PATH`` during inference.
    cuda_lib : str, optional
        CUDA lib64 directory prepended to ``LD_LIBRARY_PATH`` during inference.
    xla_client_mem_fraction : str, optional
        Value assigned to ``XLA_CLIENT_MEM_FRACTION``.
    """

    def __init__(
        self,
        pdb_dir: str,
        output_dir: str,
        af3_python: str,
        db_dir: str,
        model_dir: str,
        hmmer_path: str,
        python: Optional[str] = None,
        # ---- pipeline control ----
        run_MSA: bool = True,
        remove_template: bool = False,
        run_AF3_inference: bool = True,
        # ---- MSA params ----
        seed_num: str = "5",
        # ---- env params ----
        cuda_bin: str = "/usr/local/cuda-12.6/bin",
        cuda_lib: str = "/usr/local/cuda-12.6/lib64",
        xla_client_mem_fraction: str = "0.95",
    ):
        self.pdb_dir = os.path.abspath(pdb_dir)
        self.output_dir = os.path.abspath(output_dir)
        self.af3_python = af3_python
        self.db_dir = db_dir
        self.model_dir = model_dir
        self.hmmer_path = hmmer_path
        self.python = python or sys.executable

        self.run_MSA = run_MSA
        self.remove_template = remove_template
        self.run_AF3_inference = run_AF3_inference

        self.seed_num = str(seed_num)

        self.cuda_bin = cuda_bin
        self.cuda_lib = cuda_lib
        self.xla_client_mem_fraction = xla_client_mem_fraction

        self._pipeline_script_dir = os.path.dirname(os.path.abspath(__file__))

        self.fasta_file = os.path.join(self.output_dir, "fasta_all.fa")
        self.json_root = os.path.join(self.output_dir, "json")
        self.json2_root = os.path.join(self.output_dir, "json2")
        self.out_root = os.path.join(self.output_dir, "out")
        self.metric_csv = os.path.join(self.output_dir, "af3_metrics.csv")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _safe_run(
        self,
        cmd: List[str],
        env: Optional[dict] = None,
        check: bool = True,
    ) -> subprocess.CompletedProcess:
        """Run a command directly on the current node."""
        print(f"[ReFoldStep] Running: {' '.join(cmd)}")
        return subprocess.run(cmd, check=check, env=env)

    def _get_script(self, script_name: str) -> str:
        """Resolve a helper script relative to this file."""
        script_path = os.path.join(self._pipeline_script_dir, "AF3_inference", script_name)
        if not os.path.exists(script_path):
            raise FileNotFoundError(f"Missing script: {script_path}")
        return script_path

    def _prepare_dirs(self):
        """Create required output directories."""
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.json_root, exist_ok=True)
        os.makedirs(self.json2_root, exist_ok=True)
        os.makedirs(self.out_root, exist_ok=True)

    def _iter_subdirs(self, root_dir: str) -> List[str]:
        """Return sorted immediate subdirectories under a root."""
        if not os.path.isdir(root_dir):
            return []
        return sorted(
            os.path.join(root_dir, name)
            for name in os.listdir(root_dir)
            if os.path.isdir(os.path.join(root_dir, name))
        )

    def _build_env_for_inference(self) -> dict:
        """Build environment variables for AF3 inference."""
        env = os.environ.copy()

        env["PATH"] = f"{self.cuda_bin}:{env.get('PATH', '')}"
        env["LD_LIBRARY_PATH"] = f"{self.cuda_lib}:{env.get('LD_LIBRARY_PATH', '')}"
        env["PATH"] = f"{self.hmmer_path}:{env.get('PATH', '')}"

        env["XLA_FLAGS"] = "--xla_gpu_enable_triton_gemm=false"
        env["XLA_PYTHON_CLIENT_PREALLOCATE"] = "true"
        env["XLA_CLIENT_MEM_FRACTION"] = self.xla_client_mem_fraction

        return env

    # ------------------------------------------------------------------
    # Template patcher
    # ------------------------------------------------------------------

    @staticmethod
    def _set_templates_empty(node: Any) -> bool:
        """Recursively replace all non-empty ``templates`` with empty lists."""
        modified = False

        if isinstance(node, dict):
            if "templates" in node and node["templates"] != []:
                node["templates"] = []
                modified = True
            for value in node.values():
                if ReFoldStep._set_templates_empty(value):
                    modified = True

        elif isinstance(node, list):
            for item in node:
                if ReFoldStep._set_templates_empty(item):
                    modified = True

        return modified

    def _patch_json_file(self, json_path: str) -> bool:
        """Patch one JSON file in-place. Return True if modified."""
        path = Path(json_path)

        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        if not self._set_templates_empty(data):
            return False

        with tempfile.NamedTemporaryFile(
            "w",
            delete=False,
            dir=path.parent,
            encoding="utf-8",
            suffix=".tmp",
        ) as tmp:
            json.dump(data, tmp, ensure_ascii=False, indent=2)
            tmp.write("\n")
            tmp_path = Path(tmp.name)

        tmp_path.replace(path)
        return True

    def _remove_templates(self):
        """Remove templates from all JSON files under json2."""
        root = Path(self.json2_root)
        if not root.exists():
            print(f"[ReFoldStep] json2 directory not found, skip: {self.json2_root}")
            return

        json_files = sorted(root.rglob("*.json"))
        if not json_files:
            print(f"[ReFoldStep] No JSON files found under: {self.json2_root}")
            return

        modified_count = 0
        for json_file in json_files:
            if self._patch_json_file(str(json_file)):
                modified_count += 1

        print(
            f"[ReFoldStep] Template patching completed. "
            f"Modified {modified_count} files."
        )

    # ------------------------------------------------------------------
    # Command builders
    # ------------------------------------------------------------------

    def _build_pdb2fa_cmd(self) -> List[str]:
        """Build command for pdb2fa.py."""
        if not os.path.isdir(self.pdb_dir):
            raise ValueError(f"`pdb_dir` does not exist: {self.pdb_dir}")

        script = self._get_script("pdb2fa.py")
        return [
            self.af3_python,
            script,
            self.pdb_dir,
            self.output_dir,
        ]

    def _build_gen_json_cmd(self) -> List[str]:
        """Build command for gen_json.py."""
        script = self._get_script("gen_json.py")
        return [
            self.af3_python,
            script,
            self.fasta_file,
            self.json_root,
            self.seed_num,
        ]

    def _build_local_msa_cmd(self, input_dir: str) -> List[str]:
        """Build command for local AF3 data-pipeline/MSA generation."""
        script = self._get_script("run_alphafold.py")
        return [
            self.af3_python,
            script,
            "--db_dir",
            self.db_dir,
            "--model_dir",
            self.model_dir,
            "--input_dir",
            input_dir,
            "--output_dir",
            self.json2_root,
            "--run_data_pipeline=true",
            "--run_inference=False",
        ]

    def _build_local_inference_cmd(self, input_dir: str) -> List[str]:
        """Build command for local AF3 inference."""
        script = self._get_script("run_alphafold.py")
        return [
            self.af3_python,
            script,
            "--db_dir",
            self.db_dir,
            "--model_dir",
            self.model_dir,
            "--input_dir",
            input_dir,
            "--output_dir",
            self.out_root,
            "--run_data_pipeline=False",
            "--run_inference=true",
        ]

    def _build_get_metric_cmd(self) -> List[str]:
        """Build command for AF3 metric extraction."""
        script = self._get_script("get_metric.py")
        return [
            self.af3_python,
            script,
            "--af3_output_dir",
            self.out_root,
            "--pdb_dir",
            self.pdb_dir,
            "--save_metric_csv",
            self.metric_csv,
        ]

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def build_cmd(self):
        """Return a high-level description of the workflow commands.

        Returns
        -------
        dict
            A structured preview of the commands involved in the run.
        """
        preview = {
            "pdb2fa": self._build_pdb2fa_cmd(),
            "gen_json": self._build_gen_json_cmd(),
            "run_MSA": self.run_MSA,
            "remove_template": self.remove_template,
            "run_AF3_inference": self.run_AF3_inference,
            "get_metric": self._build_get_metric_cmd(),
        }
        return preview

    def run(self, check: bool = True) -> None:
        """Execute the full direct AF3 refolding workflow.

        Parameters
        ----------
        check : bool
            If *True* (default), raise :class:`subprocess.CalledProcessError`
            when any subprocess exits with a non-zero return code.
        """
        self._prepare_dirs()

        print("[ReFoldStep] Starting direct AF3 refolding workflow")
        print(f"[ReFoldStep] Input PDB dir: {self.pdb_dir}")
        print(f"[ReFoldStep] Output dir:    {self.output_dir}")

        # Step 1. pdb2fa
        self._safe_run(self._build_pdb2fa_cmd(), check=check)

        # Step 2. gen_json
        if not os.path.exists(self.fasta_file):
            raise FileNotFoundError(f"Missing FASTA file: {self.fasta_file}")
        self._safe_run(self._build_gen_json_cmd(), check=check)

        # Step 3. local MSA / data pipeline
        if self.run_MSA:
            input_dirs = self._iter_subdirs(self.json_root)
            if not input_dirs:
                raise RuntimeError(
                    f"No input JSON directories found in: {self.json_root}"
                )

            print(
                f"[ReFoldStep] Running local MSA/data pipeline for {len(input_dirs)} inputs"
            )
            for input_dir in input_dirs:
                self._safe_run(
                    self._build_local_msa_cmd(input_dir),
                    check=check,
                )
        else:
            print("[ReFoldStep] Skip local MSA/data pipeline")

        # Step 4. template removal
        if self.remove_template:
            print("[ReFoldStep] Removing templates from json2")
            self._remove_templates()
        else:
            print("[ReFoldStep] Keeping templates")

        # Step 5. AF3 inference
        if self.run_AF3_inference:
            input_dirs = self._iter_subdirs(self.json2_root)
            if not input_dirs:
                raise RuntimeError(
                    f"No prepared JSON directories found in: {self.json2_root}"
                )

            env = self._build_env_for_inference()
            print(f"[ReFoldStep] Running AF3 inference for {len(input_dirs)} inputs")
            for input_dir in input_dirs:
                self._safe_run(
                    self._build_local_inference_cmd(input_dir),
                    env=env,
                    check=check,
                )
        else:
            print("[ReFoldStep] Skip AF3 inference")

        # Step 6. metric extraction
        print(f"[ReFoldStep] Extracting AF3 metrics to: {self.metric_csv}")
        self._safe_run(self._build_get_metric_cmd(), check=check)

        print("[ReFoldStep] Workflow completed successfully")