"""
MPNNStep - Pipeline step for ProteinMPNN sequence design.

Supports three protocol types (protocol):
  - base:  Standard ProteinMPNN design
  - bias:  ProteinMPNN design with amino-acid bias
  - fix:   ProteinMPNN design with fixed positions
"""

import os
import subprocess
import sys
from typing import List, Optional


# Resolve the ProteinMPNN code directory relative to this file
_CODE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "tools",
    "ProteinMPNN",
)

_PARSE_SCRIPT = os.path.join(_CODE_DIR, "helper_scripts", "parse_multiple_chains.py")
_ASSIGN_SCRIPT = os.path.join(_CODE_DIR, "helper_scripts", "assign_fixed_chains.py")
_BIAS_SCRIPT = os.path.join(_CODE_DIR, "helper_scripts", "make_bias_AA.py")
_FIXED_SCRIPT = os.path.join(
    _CODE_DIR, "helper_scripts", "make_fixed_positions_dict.py"
)
_MPNN_MAIN = os.path.join(_CODE_DIR, "protein_mpnn_run.py")

VALID_PROTOCOLS = ("base", "bias", "fix")


class MPNNStep:
    """Pipeline step that wraps ProteinMPNN preprocessing and design scripts.

    Parameters
    ----------
    protocol : str
        Protocol type: one of ``"base"``, ``"bias"``, or ``"fix"``.
    folder_with_pdbs : str
        Directory containing input PDB files.
    output_dir : str
        Directory where outputs will be written.
    num_seq_per_target : int
        Number of sequences to generate for each target.
    sampling_temp : float | str
        Sampling temperature passed to ProteinMPNN.
    batch_size : int
        Batch size passed to ProteinMPNN.
    code_dir : str, optional
        Root directory containing ProteinMPNN scripts. If not provided,
        scripts are resolved relative to this file.
    python : str, optional
        Python interpreter to use (default: current interpreter).
    chains_to_design : str, optional
        Chain list to design (default: ``"A"``).
    omit_AAs : str, optional
        Amino acids to omit during design (default: ``"C"``).
    use_soluble_model : bool, optional
        Whether to pass ``--use_soluble_model`` to ProteinMPNN main script.
        This is only applied for the ``"base"`` protocol by default behavior
        matching the provided shell script.
    bias_AA_list : str, optional
        Space-separated amino-acid list for the ``"bias"`` protocol.
        Default: ``"A E K R D F Y W"``.
    bias_list : str, optional
        Space-separated bias values for the ``"bias"`` protocol.
        Default: ``"-0.5 -1 -1 0.2 0.2 0.5 0.2 0.2"``.
    fixed_positions : str, optional
        Space-separated fixed positions for the ``"fix"`` protocol.
        Required when ``protocol="fix"``.
    """

    def __init__(
        self,
        protocol: str,
        folder_with_pdbs: str,
        output_dir: str,
        num_seq_per_target: int,
        sampling_temp,
        batch_size: int,
        code_dir: Optional[str] = None,
        python: Optional[str] = None,
        chains_to_design: str = "A",
        omit_AAs: str = "C",
        use_soluble_model: bool = True,
        bias_AA_list: str = "A E K R D F Y W",
        bias_list: str = "-0.5 -1 -1 0.2 0.2 0.5 0.2 0.2",
        fixed_positions: Optional[str] = None,
    ):
        if protocol not in VALID_PROTOCOLS:
            raise ValueError(
                f"Invalid protocol '{protocol}'. Must be one of: {VALID_PROTOCOLS}"
            )

        self.protocol = protocol
        self.folder_with_pdbs = folder_with_pdbs
        self.output_dir = output_dir
        self.num_seq_per_target = num_seq_per_target
        self.sampling_temp = sampling_temp
        self.batch_size = batch_size
        self.code_dir = code_dir or _CODE_DIR
        self.python = python or sys.executable
        self.chains_to_design = chains_to_design
        self.omit_AAs = omit_AAs
        self.use_soluble_model = use_soluble_model
        self.bias_AA_list = bias_AA_list
        self.bias_list = bias_list
        self.fixed_positions = fixed_positions

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _script(self, relative_path: str) -> str:
        """Resolve a script path under the ProteinMPNN code directory."""
        script = os.path.join(self.code_dir, relative_path)
        if not os.path.exists(script):
            raise FileNotFoundError(f"Script not found: {script}")
        return script

    def _validate_inputs(self) -> None:
        """Validate required inputs."""
        if self.folder_with_pdbs is None:
            raise ValueError("`folder_with_pdbs` is required.")
        if self.output_dir is None:
            raise ValueError("`output_dir` is required.")
        if self.num_seq_per_target is None:
            raise ValueError("`num_seq_per_target` is required.")
        if self.sampling_temp is None:
            raise ValueError("`sampling_temp` is required.")
        if self.batch_size is None:
            raise ValueError("`batch_size` is required.")

        if self.protocol == "fix" and self.fixed_positions is None:
            raise ValueError("`fixed_positions` is required when protocol is 'fix'.")

    def _path_for_parsed_chains(self) -> str:
        """Return parsed chains JSONL path."""
        return os.path.join(self.output_dir, "parsed_pdbs.jsonl")

    def _path_for_assigned_chains(self) -> str:
        """Return assigned chains JSONL path."""
        return os.path.join(self.output_dir, "assigned_pdbs.jsonl")

    def _path_for_bias(self) -> str:
        """Return amino-acid bias JSONL path."""
        return os.path.join(self.output_dir, "bias_pdbs.jsonl")

    def _path_for_fixed_positions(self) -> str:
        """Return fixed positions JSONL path."""
        return os.path.join(self.output_dir, "fixed_pdbs.jsonl")

    def _build_parse_cmd(self) -> List[str]:
        """Build command for parsing PDB chains."""
        return [
            self.python,
            self._script(os.path.join("helper_scripts", "parse_multiple_chains.py")),
            "--input_path",
            self.folder_with_pdbs,
            "--output_path",
            self._path_for_parsed_chains(),
        ]

    def _build_assign_cmd(self) -> List[str]:
        """Build command for assigning design chains."""
        return [
            self.python,
            self._script(os.path.join("helper_scripts", "assign_fixed_chains.py")),
            "--input_path",
            self._path_for_parsed_chains(),
            "--output_path",
            self._path_for_assigned_chains(),
            "--chain_list",
            self.chains_to_design,
        ]

    def _build_bias_cmd(self) -> List[str]:
        """Build command for generating amino-acid bias JSONL."""
        return [
            self.python,
            self._script(os.path.join("helper_scripts", "make_bias_AA.py")),
            "--output_path",
            self._path_for_bias(),
            "--AA_list",
            self.bias_AA_list,
            "--bias_list",
            self.bias_list,
        ]

    def _build_fixed_positions_cmd(self) -> List[str]:
        """Build command for generating fixed-position JSONL."""
        return [
            self.python,
            self._script(
                os.path.join("helper_scripts", "make_fixed_positions_dict.py")
            ),
            "--input_path",
            self._path_for_parsed_chains(),
            "--output_path",
            self._path_for_fixed_positions(),
            "--chain_list",
            self.chains_to_design,
            "--position_list",
            self.fixed_positions,
        ]

    def _build_mpnn_cmd(self) -> List[str]:
        """Build command for ProteinMPNN main run."""
        cmd = [
            self.python,
            self._script("protein_mpnn_run.py"),
            "--jsonl_path",
            self._path_for_parsed_chains(),
            "--chain_id_jsonl",
            self._path_for_assigned_chains(),
            "--out_folder",
            self.output_dir,
            "--num_seq_per_target",
            str(self.num_seq_per_target),
            "--sampling_temp",
            str(self.sampling_temp),
            "--batch_size",
            str(self.batch_size),
            "--omit_AAs",
            self.omit_AAs,
        ]

        if self.protocol == "bias":
            cmd.extend(["--bias_AA_jsonl", self._path_for_bias()])

        if self.protocol == "fix":
            cmd.extend(["--fixed_positions_jsonl", self._path_for_fixed_positions()])

        if self.protocol == "base" and self.use_soluble_model:
            cmd.append("--use_soluble_model")

        return cmd

    def _build_cmds(self) -> List[List[str]]:
        """Build the command list for the selected ProteinMPNN protocol."""
        self._validate_inputs()

        cmds: List[List[str]] = []

        if self.protocol == "bias":
            cmds.append(self._build_bias_cmd())

        cmds.append(self._build_parse_cmd())
        cmds.append(self._build_assign_cmd())

        if self.protocol == "fix":
            cmds.append(self._build_fixed_positions_cmd())

        cmds.append(self._build_mpnn_cmd())

        return cmds

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def build_cmd(self) -> List[List[str]]:
        """Return the command lists that would be executed by :meth:`run`.

        Returns
        -------
        list[list[str]]
            The sequence of commands and their arguments as a list.
        """
        return self._build_cmds()

    def run(self, check: bool = True) -> List[subprocess.CompletedProcess]:
        """Execute the ProteinMPNN pipeline as a sequence of subprocess calls.

        Parameters
        ----------
        check : bool
            If *True* (default), raise :class:`subprocess.CalledProcessError`
            when any command exits with a non-zero return code.

        Returns
        -------
        list[subprocess.CompletedProcess]
            The result objects from :func:`subprocess.run`.
        """
        os.makedirs(self.output_dir, exist_ok=True)
        cmds = self.build_cmd()

        results = []
        for cmd in cmds:
            print(f"[MPNNStep] Running ({self.protocol}): {' '.join(cmd)}")
            result = subprocess.run(cmd, check=check, cwd=self.code_dir)
            results.append(result)

        return results
