import numpy as np
import os
import re
from data import protein
from core.utils import rigid_utils


Rigid = rigid_utils.Rigid


def swap_index(idx, array_list):
    id, count = np.unique(idx, return_counts=True)
    if id.size == 2:
        target_len, binder_len = count[id == 0].item(), count[id == 1].item()
        new_array_list = []
        for arr in array_list:
            new_arr = np.zeros_like(arr)
            new_arr[:binder_len] = arr[-binder_len:]
            new_arr[-target_len:] = arr[:target_len]
            new_array_list.append(new_arr)
        return new_array_list
    else:
        return array_list


def create_full_prot(atom37: np.ndarray, atom37_mask: np.ndarray, aatype=None, b_factors=None, chain_index=None, binder=False):
    assert atom37.ndim == 3
    assert atom37.shape[-1] == 3
    assert atom37.shape[-2] == 37
    n = atom37.shape[0]

    chain_index = np.zeros(n) if chain_index is None else chain_index.squeeze().cpu().numpy()
    aatype = np.zeros(n, dtype=int) if aatype is None else aatype.squeeze().cpu().numpy()
    
    if b_factors is None:
        b_factors = np.zeros([n, 37])
    else:
        b_factors = np.zeros([n, 37]) + b_factors.unsqueeze(1).cpu().numpy()

    if binder:
        chain_index, atom37, atom37_mask, aatype, b_factors = swap_index(chain_index, [chain_index, atom37, atom37_mask, aatype, b_factors])
        chain_index = np.where(chain_index, 0, 1)
        residue_index = np.concatenate([np.arange(1, sum(chain_index==0)+1), np.arange(1, sum(chain_index==1)+1),])
    else:
        if chain_index is None:
            residue_index = np.arange(1, n + 1)
        else:
            residue_index = np.zeros(n, dtype=int)
            for chain_id in np.unique(chain_index):
                mask = chain_index == chain_id
                residue_index[mask] = np.arange(1, np.sum(mask) + 1)
    aatype = aatype.astype(int)

    
    return protein.Protein(
        atom_positions=atom37, atom_mask=atom37_mask, aatype=aatype, residue_index=residue_index, chain_index=chain_index, b_factors=b_factors
    )


def write_prot_to_pdb(
    prot_pos: np.ndarray,
    file_path: str,
    aatype=None,
    overwrite=False,
    no_indexing=False,
    b_factors=None,
    chain_index=None,
    binder=False,
):
    if overwrite:
        max_existing_idx = 0
    else:
        file_dir = os.path.dirname(file_path)
        file_name = os.path.basename(file_path).strip(".pdb")
        existing_files = [x for x in os.listdir(file_dir) if file_name in x]
        max_existing_idx = max(
            [int(re.findall(r"_(\d+).pdb", x)[0]) for x in existing_files if re.findall(r"_(\d+).pdb", x) if re.findall(r"_(\d+).pdb", x)] + [0]
        )
    if not no_indexing:
        save_path = file_path.replace(".pdb", "") + f"_{max_existing_idx+1}.pdb"
    else:
        save_path = file_path
    with open(save_path, "w") as f:
        if prot_pos.ndim == 4:
            for t, pos37 in enumerate(prot_pos):
                atom37_mask = np.sum(np.abs(pos37), axis=-1) > 1e-7
                prot = create_full_prot(pos37, atom37_mask, aatype=aatype, b_factors=b_factors, chain_index=chain_index, binder=binder)
                pdb_prot = protein.to_pdb(prot, model=t + 1, add_end=False)
                f.write(pdb_prot)
        elif prot_pos.ndim == 3:
            atom37_mask = np.sum(np.abs(prot_pos), axis=-1) > 1e-7
            prot = create_full_prot(prot_pos, atom37_mask, aatype=aatype, b_factors=b_factors, chain_index=chain_index, binder=binder)
            pdb_prot = protein.to_pdb(prot, model=1, add_end=False)
            f.write(pdb_prot)
        else:
            raise ValueError(f"Invalid positions shape {prot_pos.shape}")
        f.write("END")
    return save_path
