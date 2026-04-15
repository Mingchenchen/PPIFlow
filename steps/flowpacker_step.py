"""
FlowpackerStep - Pipeline step for Flowpacker structure sampling.

This step wraps the Flowpacker antibody sampling script
(`sampler_pdb_antibody.py`) and executes it as a subprocess.
"""

import os
import subprocess
import sys
from typing import Optional


# Resolve the Flowpacker tools directory relative to this file.
_FLOWPACKER_TOOLS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "tools",
    "flowpacker",
)

_DEFAULT_SAMPLER_SCRIPT = os.path.join(
    _FLOWPACKER_TOOLS_DIR, "sampler_pdb_antibody.py"
)


class FlowpackerStep:
    """Pipeline step that wraps the Flowpacker antibody sampling script.

    Parameters
    ----------
    config_path : str
        Path to the Flowpacker YAML configuration file.
    output_dir : str
        Directory where sampling outputs will be written.
    csv_path : str
        Path to the input CSV file.
    sampler_script_dir : str, optional
        Directory containing ``sampler_pdb_antibody.py``. If not provided,
        the script path is resolved relative to this file.
    python_executable : str, optional
        Python interpreter used to run the sampling script.
        Defaults to the current interpreter.
    use_gt_masks : bool, optional
        Whether to pass ``--use_gt_masks`` to the sampler script.
        Defaults to ``True``.
    """

    def __init__(
        self,
        config_path: str,
        output_dir: str,
        csv_path: str,
        pdb_dir: str,
        sampler_script_dir: Optional[str] = None,
        python_executable: Optional[str] = None,
        use_gt_masks: bool = True,
    ):
        self.config_path = config_path
        self.output_dir = output_dir
        self.csv_path = csv_path
        self.pdb_dir = pdb_dir
        self.sampler_script_dir = sampler_script_dir
        self.python_executable = python_executable or sys.executable
        self.use_gt_masks = use_gt_masks

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve_sampler_script_path(self) -> str:
        """Resolve the antibody sampler script path."""
        if self.sampler_script_dir is not None:
            sampler_script_path = os.path.join(
                self.sampler_script_dir, "sampler_pdb_antibody.py"
            )
        else:
            sampler_script_path = _DEFAULT_SAMPLER_SCRIPT

        if not os.path.exists(sampler_script_path):
            raise FileNotFoundError(
                f"Sampler script not found: {sampler_script_path}"
            )

        return sampler_script_path

    def _build_command(self) -> list[str]:
        """Build the subprocess command for Flowpacker sampling."""
        if self.config_path is None:
            raise ValueError("`config_path` is required.")
        if self.output_dir is None:
            raise ValueError("`output_dir` is required.")
        if self.csv_path is None:
            raise ValueError("`csv_path` is required.")

        sampler_script_path = self._resolve_sampler_script_path()

        command = [
            self.python_executable,
            sampler_script_path,
            self.config_path,
            "--save_dir",
            self.output_dir,
            "--use_gt_masks",
            str(self.use_gt_masks),
            "--csv_file",
            self.csv_path,
            "--pdb_dir",
            self.pdb_dir,
            "--ckpt",
            f"{_FLOWPACKER_TOOLS_DIR}/checkpoints/cluster.pth",
        ]
        return command

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def build_cmd(self) -> list[str]:
        """Return the command that would be executed by :meth:`run`.

        Returns
        -------
        list[str]
            Command and arguments as a list of strings.
        """
        return self._build_command()

    def run(self, check: bool = True) -> subprocess.CompletedProcess:
        """Execute the Flowpacker sampler script as a subprocess.

        Parameters
        ----------
        check : bool, optional
            If ``True``, raise :class:`subprocess.CalledProcessError`
            when the subprocess exits with a non-zero status code.
            Defaults to ``True``.

        Returns
        -------
        subprocess.CompletedProcess
            Result returned by :func:`subprocess.run`.
        """
        os.makedirs(self.output_dir, exist_ok=True)
        command = self.build_cmd()
        print(f"[FlowpackerStep] Running: {' '.join(command)}")

        env = os.environ.copy()
        current_path = env.get("PATH", "")
        env["PATH"] = f"{_FLOWPACKER_TOOLS_DIR}{os.pathsep}{current_path}"

        result = subprocess.run(command, check=check, env=env)
        return result