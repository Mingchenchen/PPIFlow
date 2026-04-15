import pandas as pd
import numpy as np
import os,sys
import glob
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial

PDB_COLSPECS = [
    (0, 6),  # record_type
    (6, 11),  # atom_num
    (12, 16),  # atom_name
    (17, 20),  # residue_name
    (21, 22),  # chain_id
    (22, 26),  # residue_num
    (30, 38),  # x
    (38, 46),  # y
    (46, 54),  # z
    (60, 66),  # beta_factor
]
PDB_COLNAMES = ["record_type", "atom_num", "atom_name", "residue_name", "chain_id", "residue_num", "x", "y", "z", "beta_factor"]


def read_pdb_optimized(pdb_file):

    with open(pdb_file, "r") as f:
        atom_lines = [line for line in f if line.startswith(("ATOM", "HETATM"))]

    if not atom_lines:
        return pd.DataFrame(columns=PDB_COLNAMES)
    from io import StringIO

    atom_text = "".join(atom_lines)

    df = pd.read_fwf(StringIO(atom_text), colspecs=PDB_COLSPECS, names=PDB_COLNAMES, header=None)
    df["atom_num"] = pd.to_numeric(df["atom_num"], errors="coerce").astype("Int64")
    df["residue_num"] = pd.to_numeric(df["residue_num"], errors="coerce").astype("Int64")
    df["beta_factor"] = pd.to_numeric(df["beta_factor"], errors="coerce").fillna(0.0)

    return df

def process_single_pdb(pdb_file, target_beta=4.0):

    try:
        df = read_pdb_optimized(pdb_file)
        if df.empty:
            return None
        chain_A_data = df[df["chain_id"] == "A"]
        max_A_residue = chain_A_data["residue_num"].max() if not chain_A_data.empty else 0
        filtered_df = df[np.isclose(df["beta_factor"], target_beta, atol=0.001)]
        if filtered_df.empty:
            return None

        unique_residues = filtered_df.drop_duplicates(subset=["chain_id", "residue_num"])

        results = {}
        for chain_id, group in unique_residues.groupby("chain_id"):
            if chain_id == "B":
                residue_indices = group["residue_num"]
                results[chain_id] = sorted(residue_indices.tolist())
            else:
                residue_indices = group["residue_num"]
                results[chain_id] = sorted(residue_indices.tolist())

        pdb_name = os.path.basename(pdb_file).replace(".pdb", "")

        a_indices_str = " ".join(map(str, results.get("A", [])))
        b_indices_str = " ".join(map(str, results.get("B", [])))

        motif_str = f"{a_indices_str}-{b_indices_str}"

        return [pdb_name, motif_str]

    except Exception as e:
        print(f"Error processing file {pdb_file}: {e}")
        return None
def main(input_dir, sub_dirs=None):

    if sub_dirs is None:
        folders = [input_dir]
    else:
        folders = [os.path.join(input_dir, folder) for folder in sub_dirs]

    for folder in tqdm(folders, desc="Processing folders"):
        if not os.path.exists(folder) or not os.path.isdir(folder):
            print(f"{folder} is not existing or not a directory. Skipping.")
            continue
        all_pdb_files = glob.glob(os.path.join(folder, "*.pdb"))
        if not all_pdb_files:
            continue

        fixed_positions_data = []
        with ProcessPoolExecutor(max_workers=32) as executor:
            futures = {executor.submit(process_single_pdb, pdb, 4.0): pdb for pdb in all_pdb_files}

            for future in as_completed(futures):
                result = future.result()
                if result:
                    fixed_positions_data.append(result)

        if fixed_positions_data:

            fixed_positions_data.sort(key=lambda x: x[0])

            fixed_positions_df = pd.DataFrame(fixed_positions_data, columns=["pdb_name", "motif_index"])
            output_csv_path = os.path.join(folder, "fixed_positions.csv")
            fixed_positions_df.to_csv(output_csv_path, index=False)
            print(f"Saved to {output_csv_path}")

if __name__ == "__main__":
    main(input_dir=sys.argv[1])



