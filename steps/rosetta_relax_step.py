"""
RosettaRelaxStep - Pipeline step for Rosetta complex relaxation.

Wraps ``relax_complex.py`` and executes Rosetta relax on all ``.pdb`` files
in the given input directory.
"""

import os
import subprocess
import sys
from typing import Optional


# Resolve the relax script path relative to this file
_RELAX_SCRIPT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "rosetta",
    "relax_complex.py",
)


class RosettaRelaxStep:
    """Pipeline step that wraps Rosetta relax script.

    Parameters
    ----------
    pdb_dir : str
        Directory containing input PDB files.
    output_dir : str
        Directory where relaxed structures and score CSV will be written.
    batch_idx : str
        Batch index used by ``relax_complex.py`` for naming output CSV.
    dump_pdb : bool, optional
        Whether to dump relaxed PDB files (default: ``False``).
    python : str, optional
        Python interpreter to use (default: current interpreter).
    """

    def __init__(
        self,
        pdb_dir: str,
        output_dir: str,
        batch_idx: str,
        dump_pdb: bool = False,
        python: Optional[str] = None,
    ):
        self.pdb_dir = pdb_dir
        self.output_dir = output_dir
        self.batch_idx = batch_idx
        self.dump_pdb = dump_pdb
        self.python = python or sys.executable

    # ------------------------------------------------------------------
    # Command builder
    # ------------------------------------------------------------------

    def build_cmd(self):
        """Return the command list that would be executed by :meth:`run`.

        Returns
        -------
        list[str]
            The command and its arguments as a list.
        """
        cmd = [
            self.python,
            _RELAX_SCRIPT,
            "--pdb_dir",
            self.pdb_dir,
            "--output_dir",
            self.output_dir,
            "--dump_pdb",
            str(self.dump_pdb),
            "--batch_idx",
            self.batch_idx,
        ]
        return cmd

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def run(self, check: bool = True) -> subprocess.CompletedProcess:
        """Execute the Rosetta relax script as a subprocess.

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
        print(f"[RosettaRelaxStep] Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, check=check)
        return result
