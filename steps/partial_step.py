"""
PartialStep - Pipeline step for PPIFlow partial structure sampling.

Supports three generation types (gentype):
  - binder:    Binder partial design via sample_binder_partial.py
  - nanobody:  Nanobody partial design via sample_antibody_nanobody_partial.py
  - antibody:  Antibody partial design via sample_antibody_nanobody_partial.py
"""

import csv
import os
import subprocess
import sys
from typing import Dict, List, Optional

from Bio import PDB


# Resolve the tools directory relative to this file
_TOOLS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "tools",
    "PPIFlow",
)

_BINDER_SCRIPT = os.path.join(_TOOLS_DIR, "sample_binder_partial.py")
_ANTIBODY_SCRIPT = os.path.join(_TOOLS_DIR, "sample_antibody_nanobody_partial.py")

VALID_GENTYPES = ("binder", "nanobody", "antibody")


class PartialStep:
    """Pipeline step that wraps PPIFlow partial sampling scripts.

    Parameters
    ----------
    gentype : str
        Generation type: one of ``"binder"``, ``"nanobody"``, or
        ``"antibody"``.
    pdb_dir : str
        Directory containing input PDB files.
    fixed_positions_csv : str
        CSV file containing ``fixed_positions`` for each PDB in ``pdb_dir``.
    output_dir : str
        Directory where outputs will be written.
    model_weights : str
        Path to the model checkpoint file.
    name : str, optional
        Optional run prefix. If omitted, each PDB basename is used.
    config : str, optional
        Path to the YAML config file. When *None* the scripts' own
        defaults are used.
    python : str, optional
        Python interpreter to use (default: current interpreter).

    Binder-specific parameters
    --------------------------
    target_chain : str, optional
        Chain ID of the target protein (default: ``"B"``).
    binder_chain : str, optional
        Chain ID of the binder/motif chain (default: ``"L"``).
    sample_hotspot_rate_min : float, optional
        Minimum hotspot sampling rate (default: ``0.2``).
    sample_hotspot_rate_max : float, optional
        Maximum hotspot sampling rate (default: ``0.5``).
    samples_per_target : int, optional
        Number of samples to generate (default: ``100``).
    start_t : float, optional
        Starting t value for sampling (default: ``0.15``).

    Antibody/Nanobody-specific parameters
    -------------------------------------
    antigen_chain : str, optional
        Chain ID(s) of the antigen (default: ``"C"``).
    heavy_chain : str, optional
        Chain ID of the heavy chain (default: ``"A"``).
    light_chain : str, optional
        Chain ID of the light chain. Required for ``gentype="antibody"``,
        omit for ``gentype="nanobody"``.
    samples_per_target : int, optional
        Number of samples to generate (default: ``100``).
    start_t : float, optional
        Starting t value for sampling (default: ``0.8``).
    retry_Limit : int, optional
        Maximum retry attempts if sampling fails (default: ``10``).
    """

    def __init__(
        self,
        gentype: str,
        pdb_dir: str,
        fixed_positions_csv: str,
        output_dir: str,
        model_weights: str,
        name: Optional[str] = None,
        config: Optional[str] = None,
        python: Optional[str] = None,
        # ---- binder params ----
        target_chain: str = "B",
        binder_chain: str = "A",
        sample_hotspot_rate_min: float = 0.2,
        sample_hotspot_rate_max: float = 0.5,
        samples_per_target: int = 100,
        start_t: Optional[float] = None,
        # ---- antibody / nanobody params ----
        antigen_chain: str = "C",
        heavy_chain: str = "A",
        light_chain: Optional[str] = None,
        retry_Limit: int = 10,
    ):
        if gentype not in VALID_GENTYPES:
            raise ValueError(
                f"Invalid gentype '{gentype}'. Must be one of: {VALID_GENTYPES}"
            )

        self.gentype = gentype
        self.pdb_dir = pdb_dir
        self.fixed_positions_csv = fixed_positions_csv
        self.output_dir = output_dir
        self.model_weights = model_weights
        self.name = name
        self.config = config
        self.python = python or sys.executable

        # Binder params
        self.target_chain = target_chain
        self.binder_chain = binder_chain
        self.sample_hotspot_rate_min = sample_hotspot_rate_min
        self.sample_hotspot_rate_max = sample_hotspot_rate_max
        self.samples_per_target = samples_per_target
        self.start_t = start_t

        # Antibody / nanobody params
        self.antigen_chain = antigen_chain
        self.heavy_chain = heavy_chain
        self.light_chain = light_chain
        self.retry_Limit = retry_Limit

        self._fixed_positions_map = self._load_fixed_positions_map()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_fixed_positions_map(self) -> Dict[str, str]:
        """Load fixed_positions mapping from CSV.

        The CSV must contain a ``fixed_positions`` column and one of:
        ``pdb``, ``pdb_name``, ``filename``, or ``file``.
        """
        if not os.path.exists(self.fixed_positions_csv):
            raise FileNotFoundError(
                f"fixed_positions CSV not found: {self.fixed_positions_csv}"
            )

        with open(self.fixed_positions_csv, "r", newline="") as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames or []

            pdb_col = None
            for candidate in ("pdb", "pdb_name", "filename", "file"):
                if candidate in fieldnames:
                    pdb_col = candidate
                    break

            if pdb_col is None:
                raise ValueError(
                    "fixed_positions CSV must contain one of columns: "
                    "'pdb', 'pdb_name', 'filename', 'file'."
                )

            if "fixed_positions" not in fieldnames:
                raise ValueError(
                    "fixed_positions CSV must contain column 'fixed_positions'."
                )

            mapping = {}
            for row in reader:
                pdb_key = os.path.basename(row[pdb_col]).strip()
                fixed_positions = (row["fixed_positions"] or "").strip()
                if pdb_key:
                    mapping[pdb_key] = fixed_positions if (fixed_positions != "NONE") else None

        return mapping

    def _get_pdb_files(self) -> List[str]:
        """Return sorted PDB files under ``pdb_dir``."""
        if not os.path.isdir(self.pdb_dir):
            raise NotADirectoryError(f"pdb_dir is not a directory: {self.pdb_dir}")

        pdb_files = []
        for fname in sorted(os.listdir(self.pdb_dir)):
            if fname.lower().endswith(".pdb"):
                pdb_files.append(os.path.join(self.pdb_dir, fname))

        if not pdb_files:
            raise ValueError(f"No PDB files found in directory: {self.pdb_dir}")

        return pdb_files

    def _get_fixed_positions(self, pdb_path: str) -> str:
        """Return fixed_positions for a given PDB file."""
        pdb_name = os.path.basename(pdb_path)
        if pdb_name not in self._fixed_positions_map:
            raise ValueError(
                f"No fixed_positions entry found in CSV for PDB: {pdb_name}"
            )

        fixed_positions = self._fixed_positions_map[pdb_name]

        return fixed_positions

    def _residue_has_bfactor(self, residue, target_bfactor: float) -> bool:
        """Return whether a residue contains any atom with the given B-factor."""
        for atom in residue.get_atoms():
            if abs(float(atom.get_bfactor()) - float(target_bfactor)) < 1e-6:
                return True
        return False

    def _format_residue_id(self, chain_id: str, residue) -> str:
        """Format residue identifier as ChainID + resseq [+ insertion code]."""
        hetflag, resseq, icode = residue.get_id()
        icode = icode.strip()
        return f"{chain_id}{resseq}{icode}"

    def _extract_bfactor_residues(
        self,
        pdb_path: str,
        bfactor: float = 2.0,
        chain_id: Optional[str] = None,
    ) -> str:
        """Extract residues with the given B-factor from a PDB file.

        Parameters
        ----------
        pdb_path : str
            PDB file path.
        bfactor : float, optional
            Target B-factor value (default: ``2.0``).
        chain_id : str, optional
            Restrict extraction to a specific chain when provided.

        Returns
        -------
        str
            Comma-separated residue specification string, e.g.
            ``"B12,B15,B18"``.
        """
        parser = PDB.PDBParser(QUIET=True)
        structure = parser.get_structure("partial_step", pdb_path)

        residues = []
        for chain in structure.get_chains():
            if chain_id is not None and chain.id != chain_id:
                continue

            for residue in chain:
                if not PDB.is_aa(residue, standard=True):
                    continue
                if self._residue_has_bfactor(residue, bfactor):
                    residues.append(self._format_residue_id(chain.id, residue))

        return ",".join(residues)

    def _get_run_name(self, pdb_path: str) -> str:
        """Return run name for a PDB file."""
        pdb_stem = os.path.splitext(os.path.basename(pdb_path))[0]
        if self.name:
            return f"{self.name}_{pdb_stem}"
        return pdb_stem

    def _get_output_subdir(self, pdb_path: str) -> str:
        """Return output subdirectory for a PDB file."""
        pdb_stem = os.path.splitext(os.path.basename(pdb_path))[0]
        return os.path.join(self.output_dir, pdb_stem)

    # ------------------------------------------------------------------
    # Command builders
    # ------------------------------------------------------------------

    def _build_binder_cmd(self, pdb_path: str):
        """Build the command list for binder partial sampling."""
        fixed_positions = self._get_fixed_positions(pdb_path)

        specified_hotspots = self._extract_bfactor_residues(
            pdb_path,
            bfactor=2.0,
            chain_id="B",
        )

        if not specified_hotspots:
            raise ValueError(
                f"No binder specified_hotspots found from B-chain bfactor=2 "
                f"for PDB: {pdb_path}"
            )

        cmd = [self.python, _BINDER_SCRIPT]

        cmd += [
            "--input_pdb",
            pdb_path,
            "--target_chain",
            self.target_chain,
            "--binder_chain",
            self.binder_chain,
            "--specified_hotspots",
            specified_hotspots,
            "--sample_hotspot_rate_min",
            str(self.sample_hotspot_rate_min),
            "--sample_hotspot_rate_max",
            str(self.sample_hotspot_rate_max),
            "--samples_per_target",
            str(self.samples_per_target),
            "--model_weights",
            self.model_weights,
            "--output_dir",
            self._get_output_subdir(pdb_path),
            "--name",
            self._get_run_name(pdb_path),
            "--start_t",
            str(self.start_t if self.start_t is not None else 0.7),
        ]

        if fixed_positions:
            cmd += ["--fixed_positions", fixed_positions]

        if self.config is not None:
            cmd += ["--config", self.config]

        return cmd

    def _build_antibody_nanobody_cmd(self, pdb_path: str):
        """Build the command list for antibody/nanobody partial sampling."""
        missing = []
        fixed_positions = self._get_fixed_positions(pdb_path)

        specified_hotspots = self._extract_bfactor_residues(
            pdb_path,
            bfactor=1.0,
            chain_id="C",
        )
        cdr_position = self._extract_bfactor_residues(
            pdb_path,
            bfactor=2.0,
            chain_id=None,
        )

        if not self.antigen_chain:
            missing.append("antigen_chain")
        if not self.heavy_chain:
            missing.append("heavy_chain")
        if self.gentype == "antibody" and self.light_chain is None:
            missing.append("light_chain")
        if not specified_hotspots:
            missing.append("specified_hotspots(from C-chain bfactor=2)")
        if not cdr_position:
            missing.append("cdr_position(from bfactor=2 residues)")

        if missing:
            raise ValueError(
                f"gentype='{self.gentype}' requires: {missing} for PDB '{pdb_path}'"
            )

        cmd = [self.python, _ANTIBODY_SCRIPT]

        cmd += [
            "--complex_pdb",
            pdb_path,
            "--cdr_position",
            cdr_position,
            "--antigen_chain",
            self.antigen_chain,
            "--heavy_chain",
            self.heavy_chain,
            "--specified_hotspots",
            specified_hotspots,
            "--start_t",
            str(self.start_t if self.start_t is not None else 0.8),
            "--samples_per_target",
            str(self.samples_per_target),
            "--output_dir",
            self._get_output_subdir(pdb_path),
            "--retry_Limit",
            str(self.retry_Limit),
            "--model_weights",
            self.model_weights,
            "--name",
            self._get_run_name(pdb_path),
        ]
        if fixed_positions:
            cmd += ["--fixed_positions", fixed_positions]

        if self.light_chain is not None:
            cmd += ["--light_chain", self.light_chain]

        if self.config is not None:
            cmd += ["--config", self.config]

        return cmd

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def build_cmd(self, pdb_path: str):
        """Return the command list for a single PDB file.

        Parameters
        ----------
        pdb_path : str
            Input PDB file path.

        Returns
        -------
        list[str]
            The command and its arguments as a list.
        """
        if self.gentype == "binder":
            return self._build_binder_cmd(pdb_path)
        else:
            return self._build_antibody_nanobody_cmd(pdb_path)

    def run(self, check: bool = True) -> List[subprocess.CompletedProcess]:
        """Execute partial sampling scripts for all PDBs in ``pdb_dir``.

        Parameters
        ----------
        check : bool
            If *True* (default), raise :class:`subprocess.CalledProcessError`
            when a script exits with a non-zero return code.

        Returns
        -------
        list[subprocess.CompletedProcess]
            Result objects from :func:`subprocess.run` for all PDB files.
        """
        os.makedirs(self.output_dir, exist_ok=True)

        results = []
        for pdb_path in self._get_pdb_files():
            output_subdir = self._get_output_subdir(pdb_path)
            os.makedirs(output_subdir, exist_ok=True)

            cmd = self.build_cmd(pdb_path)
            print(
                f"[PartialStep] Running ({self.gentype}) for "
                f"{os.path.basename(pdb_path)}: {' '.join(cmd)}"
            )
            result = subprocess.run(cmd, check=check)
            results.append(result)

        return results
