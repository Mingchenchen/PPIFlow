"""
PPIFlowStep - Pipeline step for PPIFlow structure sampling.

Supports three generation types (gentype):
  - binder:    Binder design via sample_binder.py
  - nanobody:  Nanobody design (heavy chain only) via sample_antibody_nanobody.py
  - antibody:  Antibody design (heavy + light chain) via sample_antibody_nanobody.py
"""

import os
import subprocess
import sys
from typing import Optional


# Resolve the tools directory relative to this file
_TOOLS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "tools",
    "PPIFlow",
)

_BINDER_SCRIPT = os.path.join(_TOOLS_DIR, "sample_binder.py")
_ANTIBODY_SCRIPT = os.path.join(_TOOLS_DIR, "sample_antibody_nanobody.py")

VALID_GENTYPES = ("binder", "nanobody", "antibody")


class PPIFlowStep:
    """Pipeline step that wraps PPIFlow sampling scripts.

    Parameters
    ----------
    gentype : str
        Generation type: one of ``"binder"``, ``"nanobody"``, or
        ``"antibody"``.
    output_dir : str
        Directory where outputs will be written.
    model_weights : str
        Path to the model checkpoint file.
    name : str, optional
        Identifier for this run (default: ``"test_target"``).
    config : str, optional
        Path to the YAML config file.  When *None* the scripts' own
        defaults are used.
    python : str, optional
        Python interpreter to use (default: current interpreter).

    Binder-specific parameters
    --------------------------
    input_pdb : str, optional
        Input PDB file (mutually exclusive with ``input_csv``).
    input_csv : str, optional
        Pre-processed input CSV (mutually exclusive with ``input_pdb``).
    target_chain : str, optional
        Chain ID of the target protein (default: ``"R"``).
    binder_chain : str, optional
        Chain ID of the existing binder (used for hotspot auto-detection).
    specified_hotspots : str, optional
        Comma-separated hotspot residues, e.g. ``"R162,R165"``.
    sample_hotspot_rate_min : float, optional
        Minimum hotspot sampling rate (default: ``0.2``).
    sample_hotspot_rate_max : float, optional
        Maximum hotspot sampling rate (default: ``0.5``).
    samples_min_length : int, optional
        Minimum residue count per sample (default: ``50``).
    samples_max_length : int, optional
        Maximum residue count per sample (default: ``100``).
    samples_per_target : int, optional
        Number of samples to generate (default: ``100``).

    Antibody/Nanobody-specific parameters
    --------------------------------------
    antigen_pdb : str, optional
        Path to antigen PDB file.
    antigen_chain : str, optional
        Chain ID(s) of the antigen.
    specified_hotspots : str, optional
        Hotspot residues on the antigen, e.g. ``"C56,C58"``.
    framework_pdb : str, optional
        Path to the antibody/nanobody framework PDB file.
    heavy_chain : str, optional
        Chain ID of the heavy chain.
    light_chain : str, optional
        Chain ID of the light chain.  Required for ``gentype="antibody"``,
        omit for ``gentype="nanobody"``.
    cdr_length : str, optional
        CDR length specification string (see ``sample_antibody_nanobody.py``
        for format).
    samples_per_target : int, optional
        Number of samples to generate (default: ``100``).
    """

    def __init__(
        self,
        gentype: str,
        output_dir: str,
        model_weights: str,
        name: str = "test_target",
        config: Optional[str] = None,
        python: Optional[str] = None,
        specified_hotspots: Optional[str] = None,
        samples_per_target: int = 100,
        # ---- binder params ----
        input_pdb: Optional[str] = None,
        input_csv: Optional[str] = None,
        target_chain: str = "B",
        binder_chain: Optional[str] = None,
        sample_hotspot_rate_min: float = 0.2,
        sample_hotspot_rate_max: float = 0.5,
        samples_min_length: int = 50,
        samples_max_length: int = 100,
        # ---- antibody / nanobody params ----
        antigen_pdb: Optional[str] = None,
        antigen_chain: Optional[str] = None,
        framework_pdb: Optional[str] = None,
        heavy_chain: Optional[str] = "A",
        light_chain: Optional[str] = "B",
        cdr_length: Optional[str] = None,
    ):
        if gentype not in VALID_GENTYPES:
            raise ValueError(
                f"Invalid gentype '{gentype}'. Must be one of: {VALID_GENTYPES}"
            )

        self.gentype = gentype
        self.output_dir = output_dir
        self.model_weights = model_weights
        self.name = name
        self.config = config
        self.python = python or sys.executable

        # Binder params
        self.input_pdb = input_pdb
        self.input_csv = input_csv
        self.target_chain = target_chain
        self.binder_chain = binder_chain
        self.specified_hotspots = specified_hotspots
        self.sample_hotspot_rate_min = sample_hotspot_rate_min
        self.sample_hotspot_rate_max = sample_hotspot_rate_max
        self.samples_min_length = samples_min_length
        self.samples_max_length = samples_max_length
        self.samples_per_target = samples_per_target

        # Antibody / nanobody params
        self.antigen_pdb = antigen_pdb
        self.antigen_chain = antigen_chain
        self.framework_pdb = framework_pdb
        self.heavy_chain = heavy_chain
        self.light_chain = light_chain
        self.cdr_length = cdr_length

    # ------------------------------------------------------------------
    # Command builders
    # ------------------------------------------------------------------

    def _build_binder_cmd(self):
        """Build the command list for binder sampling."""
        if self.input_pdb is None and self.input_csv is None:
            raise ValueError(
                "gentype='binder' requires either 'input_pdb' or 'input_csv'."
            )

        cmd = [self.python, _BINDER_SCRIPT]

        if self.input_pdb is not None:
            cmd += ["--input_pdb", self.input_pdb]
        else:
            cmd += ["--input_csv", self.input_csv]

        cmd += ["--target_chain", self.target_chain]

        if self.binder_chain is not None:
            cmd += ["--binder_chain", self.binder_chain]

        if self.specified_hotspots is not None:
            cmd += ["--specified_hotspots", self.specified_hotspots]

        cmd += [
            "--sample_hotspot_rate_min", str(self.sample_hotspot_rate_min),
            "--sample_hotspot_rate_max", str(self.sample_hotspot_rate_max),
            "--samples_min_length", str(self.samples_min_length),
            "--samples_max_length", str(self.samples_max_length),
            "--samples_per_target", str(self.samples_per_target),
            "--model_weights", self.model_weights,
            "--output_dir", self.output_dir,
            "--name", self.name,
        ]

        if self.config is not None:
            cmd += ["--config", self.config]

        return cmd

    def _build_antibody_nanobody_cmd(self):
        """Build the command list for antibody/nanobody sampling."""
        missing = []
        if self.antigen_pdb is None:
            missing.append("antigen_pdb")
        if self.antigen_chain is None:
            missing.append("antigen_chain")
        if self.framework_pdb is None:
            missing.append("framework_pdb")
        if self.heavy_chain is None:
            missing.append("heavy_chain")
        if self.gentype == "antibody" and self.light_chain is None:
            missing.append("light_chain")
        if missing:
            raise ValueError(
                f"gentype='{self.gentype}' requires: {missing}"
            )

        cmd = [self.python, _ANTIBODY_SCRIPT]

        cmd += [
            "--antigen_pdb", self.antigen_pdb,
            "--antigen_chain", self.antigen_chain,
            "--framework_pdb", self.framework_pdb,
            "--heavy_chain", self.heavy_chain,
        ]

        if self.light_chain is not None:
            cmd += ["--light_chain", self.light_chain]

        if self.specified_hotspots is not None:
            cmd += ["--specified_hotspots", self.specified_hotspots]

        if self.cdr_length is not None:
            cmd += ["--cdr_length", self.cdr_length]

        cmd += [
            "--samples_per_target", str(self.samples_per_target),
            "--model_weights", self.model_weights,
            "--output_dir", self.output_dir,
            "--name", self.name,
        ]

        if self.config is not None:
            cmd += ["--config", self.config]

        return cmd

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def build_cmd(self):
        """Return the command list that would be executed by :meth:`run`.

        Returns
        -------
        list[str]
            The command and its arguments as a list.
        """
        if self.gentype == "binder":
            return self._build_binder_cmd()
        else:
            return self._build_antibody_nanobody_cmd()

    def run(self, check: bool = True) -> subprocess.CompletedProcess:
        """Execute the sampling script as a subprocess.

        Parameters
        ----------
        check : bool
            If *True* (default), raise :class:`subprocess.CalledProcessError`
            when the script exits with a non-zero return code.

        Returns
        -------
        subprocess.CompletedProcess
            The result object from :func:`subprocess.run`.
        """
        os.makedirs(self.output_dir, exist_ok=True)
        cmd = self.build_cmd()
        print(f"[PPIFlowStep] Running ({self.gentype}): {' '.join(cmd)}")
        result = subprocess.run(cmd, check=check)
        return result
