"""
RosettaFixStep - Pipeline step for batched Rosetta relax/fix and interface energy extraction.

Supports three generation types (gentype):
  - binder:    run Rosetta for all PDBs, then extract interface energy for chains A/B
  - nanobody:  run Rosetta for all PDBs, then extract interface energy for chains A/C
  - antibody:  run Rosetta for all PDBs, then extract interface energy twice for chains A/C and B/C
"""

import glob
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple


_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROSETTA_STEP_DIR = os.path.join(_THIS_DIR, "rosetta")

_NATIVE_XML = os.path.join(_ROSETTA_STEP_DIR, "native.xml")
_INTERFACE_ENERGY_SCRIPT = os.path.join(_ROSETTA_STEP_DIR, "1get_interface_energy.py")

VALID_GENTYPES = ("binder", "nanobody", "antibody")


class RosettaFixStep:
    """Pipeline step that wraps batched Rosetta relax/fix and interface-energy extraction.

    Parameters
    ----------
    gentype : str
        One of ``"binder"``, ``"nanobody"``, or ``"antibody"``.
    output_dir : str
        Output root directory for this step.
    rosetta_bin : str
        Path to Rosetta executable.
    native_xml : str, optional
        XML template path.
    python : str, optional
        Python interpreter to use.
    interface_dist : float, optional
        Distance threshold for interface residue detection.
    plot_toggle : bool, optional
        Whether to dump interface-energy plots.
    overwrite : bool, optional
        Whether to remove an existing per-PDB work directory before rerun.
    """

    def __init__(
        self,
        gentype: str,
        output_dir: str,
        rosetta_bin: str,
        native_xml: Optional[str] = None,
        python: Optional[str] = None,
        interface_dist: float = 12.0,
        plot_toggle: bool = False,
        overwrite: bool = False,
    ):
        if gentype not in VALID_GENTYPES:
            raise ValueError(
                f"Invalid gentype '{gentype}'. Must be one of: {VALID_GENTYPES}"
            )

        self.gentype = gentype
        self.output_dir = os.path.abspath(output_dir)
        self.rosetta_bin = rosetta_bin
        self.native_xml = os.path.abspath(native_xml or _NATIVE_XML)
        self.python = python or sys.executable
        self.interface_dist = interface_dist
        self.plot_toggle = plot_toggle
        self.overwrite = overwrite

    # ------------------------------------------------------------------
    # Helper methods
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_pdb_chains(pdb_file: str) -> Dict[str, Tuple[int, int]]:
        """Extract per-chain residue start/end indices from a PDB file."""
        chains: Dict[str, List[int]] = {}

        with open(pdb_file, "r") as f:
            for line in f:
                if not line.startswith("ATOM"):
                    continue

                chain_id = line[21].strip()
                if not chain_id:
                    continue

                try:
                    res_seq = int(line[22:26].strip())
                except ValueError:
                    continue

                if chain_id not in chains:
                    chains[chain_id] = [res_seq, res_seq]
                else:
                    chains[chain_id][1] = res_seq

        return {k: (v[0], v[1]) for k, v in chains.items()}

    @staticmethod
    def _build_resnums(chains: Dict[str, Tuple[int, int]]) -> str:
        """Build Rosetta residue-range string like '1A-100A,1B-120B'."""
        return ",".join(
            f"{start}{chain}-{end}{chain}" for chain, (start, end) in chains.items()
        )

    @staticmethod
    def _replace_resnums(xml_text: str, resnums: str) -> str:
        """Replace the resnums attribute in XML."""
        return re.sub(r'resnums="[^"]+"', f'resnums="{resnums}"', xml_text)

    @staticmethod
    def _replace_antibody_movemap(xml_text: str) -> str:
        """Replace FastRelax MoveMap for antibody mode."""
        new_movemap = """<MoveMap name="backbone">
                <Chain number="1" chi="1" bb="1"/>
                <Chain number="2" chi="1" bb="1"/>
                <Chain number="3" chi="1" bb="0"/>
            </MoveMap>"""

        pattern = r"<MoveMap name=\"backbone\">.*?</MoveMap>"
        updated = re.sub(pattern, new_movemap, xml_text, flags=re.DOTALL)

        if updated == xml_text:
            raise ValueError('Failed to locate <MoveMap name="backbone"> in native.xml')

        return updated

    def _get_chain_pairs(self) -> List[Tuple[str, str]]:
        """Return interface-energy chain pairs based on gentype."""
        if self.gentype == "binder":
            return [("A", "B")]
        if self.gentype == "nanobody":
            return [("A", "C")]
        if self.gentype == "antibody":
            return [("A", "C"), ("B", "C")]
        raise ValueError(f"Unsupported gentype: {self.gentype}")

    def _build_paths(self, input_pdb: str) -> Dict[str, str]:
        """Build per-PDB working paths."""
        pdb_name = Path(input_pdb).stem
        work_dir = os.path.join(self.output_dir, pdb_name)
        out_dir = os.path.join(work_dir, "out")
        plot_dir = os.path.join(work_dir, "plots")
        local_pdb = os.path.join(work_dir, f"{pdb_name}.pdb")
        update_xml_path = os.path.join(work_dir, "update.xml")
        rosetta_out_path = os.path.join(out_dir, f"{pdb_name}.out")

        return {
            "pdb_name": pdb_name,
            "work_dir": work_dir,
            "out_dir": out_dir,
            "plot_dir": plot_dir,
            "local_pdb": local_pdb,
            "update_xml": update_xml_path,
            "rosetta_out": rosetta_out_path,
        }

    def _prepare_workdir(self, input_pdb: str, paths: Dict[str, str]) -> None:
        """Create/reset work directory and copy input PDB."""
        if not os.path.isfile(input_pdb):
            raise FileNotFoundError(f"Input PDB not found: {input_pdb}")

        if not os.path.isfile(self.native_xml):
            raise FileNotFoundError(f"native.xml not found: {self.native_xml}")

        if self.overwrite and os.path.isdir(paths["work_dir"]):
            shutil.rmtree(paths["work_dir"])

        os.makedirs(paths["work_dir"], exist_ok=True)
        os.makedirs(paths["out_dir"], exist_ok=True)
        os.makedirs(paths["plot_dir"], exist_ok=True)

        shutil.copy2(input_pdb, paths["local_pdb"])

    def _generate_update_xml(self, local_pdb: str, update_xml_path: str) -> str:
        """Generate update.xml directly in Python, without CLI call."""
        chains = self._parse_pdb_chains(local_pdb)
        if not chains:
            raise ValueError(f"No valid ATOM records/chains parsed from: {local_pdb}")

        with open(self.native_xml, "r") as f:
            xml_text = f.read()

        resnums = self._build_resnums(chains)
        xml_text = self._replace_resnums(xml_text, resnums)

        if self.gentype == "antibody":
            xml_text = self._replace_antibody_movemap(xml_text)

        with open(update_xml_path, "w") as f:
            f.write(xml_text)

        return update_xml_path

    def _build_rosetta_cmd(self, local_pdb: str) -> List[str]:
        """Build the Rosetta command list."""
        return [
            self.rosetta_bin,
            "-parser:protocol",
            "update.xml",
            "-s",
            os.path.basename(local_pdb),
            "-overwrite",
            "-ignore_zero_occupancy",
            "false",
        ]

    def _run_rosetta(
        self,
        work_dir: str,
        local_pdb: str,
        rosetta_out_path: str,
    ) -> subprocess.CompletedProcess:
        """Run Rosetta in the working directory and redirect stdout to output log."""
        cmd = self._build_rosetta_cmd(local_pdb)
        print(f"[RosettaFixStep] Running Rosetta: {' '.join(cmd)}")

        with open(rosetta_out_path, "w") as fout:
            result = subprocess.run(
                cmd,
                cwd=work_dir,
                stdout=fout,
                stderr=subprocess.STDOUT,
                check=True,
            )
        return result

    def _prepare_one(self, input_pdb: str) -> Dict[str, object]:
        """Prepare one PDB and run Rosetta only."""
        input_pdb = os.path.abspath(input_pdb)
        paths = self._build_paths(input_pdb)

        print(f"[RosettaFixStep] Start Rosetta ({self.gentype}): {input_pdb}")

        self._prepare_workdir(input_pdb, paths)
        self._generate_update_xml(paths["local_pdb"], paths["update_xml"])
        self._run_rosetta(
            work_dir=paths["work_dir"],
            local_pdb=paths["local_pdb"],
            rosetta_out_path=paths["rosetta_out"],
        )

        print(f"[RosettaFixStep] Done Rosetta ({self.gentype}): {input_pdb}")

        return {
            "gentype": self.gentype,
            "input_pdb": input_pdb,
            "pdb_name": paths["pdb_name"],
            "work_dir": paths["work_dir"],
            "local_pdb": paths["local_pdb"],
            "update_xml": paths["update_xml"],
            "rosetta_out": paths["rosetta_out"],
        }

    def _run_interface_energy_batch(
        self,
        input_pdbdir: str,
    ) -> Dict[str, str]:
        """Run interface-energy extraction after all Rosetta jobs are finished."""
        if not os.path.isfile(_INTERFACE_ENERGY_SCRIPT):
            raise FileNotFoundError(
                f"Interface-energy script not found: {_INTERFACE_ENERGY_SCRIPT}"
            )

        interface_outputs: Dict[str, str] = {}

        for binder_id, target_id in self._get_chain_pairs():
            pair_name = f"{binder_id}_{target_id}"
            pair_output_dir = os.path.join(
                self.output_dir, f"interface_energy_{pair_name}"
            )
            os.makedirs(pair_output_dir, exist_ok=True)

            cmd = [
                self.python,
                _INTERFACE_ENERGY_SCRIPT,
                "--input_pdbdir",
                input_pdbdir,
                "--rosetta_dir",
                self.output_dir,
                "--binder_id",
                binder_id,
                "--target_id",
                target_id,
                "--output_dir",
                pair_output_dir,
                "--interface_dist",
                str(self.interface_dist),
                "--plot_toggle",
                str(self.plot_toggle),
            ]

            print(
                f"[RosettaFixStep] Running interface-energy batch: "
                f"{pair_name} -> {' '.join(cmd)}"
            )

            subprocess.run(cmd, check=True)
            interface_outputs[pair_name] = pair_output_dir

        return interface_outputs

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def run(self, input_pdbdir: str) -> Dict[str, object]:
        """Execute the full RosettaFix pipeline for all PDB files in a directory.

        Execution order
        ---------------
        1. Collect all PDBs in ``input_pdbdir``
        2. For each PDB: prepare work dir, generate update.xml, run Rosetta
        3. After all PDBs finish Rosetta, run interface-energy extraction in batch

        Parameters
        ----------
        input_pdbdir : str
            Directory containing input ``.pdb`` files.

        Returns
        -------
        dict
            Batch summary.
        """
        input_pdbdir = os.path.abspath(input_pdbdir)

        if not os.path.isdir(input_pdbdir):
            raise FileNotFoundError(f"Input PDB directory not found: {input_pdbdir}")

        input_pdbs = sorted(glob.glob(os.path.join(input_pdbdir, "*.pdb")))
        if not input_pdbs:
            raise ValueError(f"No .pdb files found in directory: {input_pdbdir}")

        os.makedirs(self.output_dir, exist_ok=True)

        total = len(input_pdbs)
        rosetta_results: List[Dict[str, object]] = []

        print(
            f"[RosettaFixStep] Found {total} PDB files in {input_pdbdir}. "
            f"Start Rosetta batch..."
        )

        # phase 1: all Rosetta first
        for idx, input_pdb in enumerate(input_pdbs, start=1):
            print(f"[RosettaFixStep] Rosetta progress: {idx}/{total}")
            result = self._prepare_one(input_pdb)
            rosetta_results.append(result)

        print(
            "[RosettaFixStep] All Rosetta runs finished. Start interface-energy batch..."
        )

        # phase 2: interface energy after all Rosetta done
        interface_outputs = self._run_interface_energy_batch(input_pdbdir)

        return {
            "gentype": self.gentype,
            "input_pdbdir": input_pdbdir,
            "num_pdbs": total,
            "rosetta_results": rosetta_results,
            "interface_outputs": interface_outputs,
        }
