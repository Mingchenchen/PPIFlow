"""Script for preprocessing PDB files."""
import sys
sys.path.append('/Users/linjie/projects/protein-frame-flow_gpuold')
import argparse
import ast
import dataclasses
import functools as fn
import pandas as pd
import os
import multiprocessing as mp
import time
from Bio import PDB
import numpy as np
import mdtraj as md
from tqdm import tqdm

from data import utils as du
from data import parsers
from data import errors
import warnings
warnings.filterwarnings("ignore")


# Define the parser
parser = argparse.ArgumentParser(
    description='PDB processing script.')
parser.add_argument(
    '--csv_file',
    help='Path to directory with PDB files.',
    type=str,
    default='/Users/linjie/projects/protein-frame-flow_gpuold/cd3_ucht1_RL/output_file.csv')
   
parser.add_argument(
    '--datatype',
    help='train or val',
    type=str,
    default='test'
)
parser.add_argument(
    '--write_dir',
    help='Path to write results to.',
    type=str,
    default='/Users/linjie/projects/protein-frame-flow_gpuold/cd3_ucht1_RL/pkl')

parser.add_argument(
    '--num_processes',
    help='Number of processes.',
    type=int,
    default=1)
parser.add_argument(
    '--debug',
    help='Turn on for debugging.',
    action='store_true')
parser.add_argument(
    '--verbose',
    help='Whether to log everything.',
    action='store_true')


def generate_virtual_binder_feats(feats):
    '''this virtual binder is not used in model, just for code simplicity'''
    # with open(processed_path, 'rb')as f:
    #     feats = pickle.load(f)
    unique_chain_idx, counts = np.unique(feats['chain_index'], return_counts=True)
    if len(unique_chain_idx)==1:
        target_seq_len = feats['chain_index'].shape[0]
        # binder_seq_len = np.random.randint(50, target_seq_len)
        binder_seq_len = target_seq_len
        new_feat = {}
        for k, v in feats.items():
            if k == 'chain_index':
                new_feat[k] = np.concatenate([v, [37]*binder_seq_len], axis=0)
            elif k == 'modeled_idx':
                new_feat[k] = np.concatenate([v,  list(range(v[-1]+1, v[-1]+1+binder_seq_len))], axis=0)
            else:
                new_feat[k] = np.concatenate([v, v[:binder_seq_len]], axis=0)

        new_feat['target_interface_residues'] = feats['target_interface_residues']
        # with open(processed_path, 'wb')as f:
        #     pickle.dump(new_feat, f)
        return new_feat
    else:
        return feats


def load_rotamers(pdbfile, chain_id='A'):
    """
    读取多MODEL PDB文件并拼接氨基酸特征

    参数:
        pdb_file (str): PDB文件路径

    返回:
        list: 包含所有MODEL氨基酸特征的列表
    """
    # 创建PDB解析器
    parser = PDB.PDBParser()
    # 解析PDB文件
    structure = parser.get_structure("multi_model", pdbfile)
    # 存储所有氨基酸特征的列表
    struct_feats = []
    # 遍历所有MODEL
    for model in structure:
        # # 获取A链
        # chain = model[chain_id]
        #
        # # 获取氨基酸残基（每个MODEL应该只有一个）
        # residues = list(chain.get_residues())
        # if len(residues) != 1:
        #     print(f"警告: MODEL {model.id} 的A链不包含1个氨基酸残基, {len(residues)}")
        #     continue

        chain = model[chain_id]
        residues_to_remove = []  # 存储待删除的残基

        # 遍历所有残基，标记非1号氨基酸
        for residue in chain:
            if residue.id[1] != 1:
                residues_to_remove.append(residue.id)  # 记录要删除的残基ID

        # 删除非1号氨基酸
        for res_id in residues_to_remove:
            chain.detach_child(res_id)

        # 检查是否剩下1号氨基酸
        if len(list(chain.get_residues())) == 0:
            print(f"警告: MODEL {model.id} 的{chain_id}链没有序号为1的氨基酸")

        chain_prot = parsers.process_chain(chain, chain_id)
        chain_dict = dataclasses.asdict(chain_prot)
        chain_dict = du.parse_chain_feats(chain_dict, normalize_positions=False)
        struct_feats.append(chain_dict)

    all_struct_feats = du.concat_np_features(struct_feats, False)
    all_rotamer_feats = {
        'rotamer_aatype': all_struct_feats['aatype'],
        'rotamer_atom_positions': all_struct_feats['atom_positions'],   # *
        'rotamer_atom_mask': all_struct_feats['atom_mask']
    }

    # import pdb; pdb.set_trace()

    return all_rotamer_feats


def process_file(row, write_dir, process_rotamers=False):
    """Processes protein file into usable, smaller pickles.

    Args:
        row: Dict or pd.Series containing input information.
        write_dir: Directory to write pickles to.

    Returns:
        Saves processed protein to pickle and returns metadata.

    Raises:
        DataError if a known filtering rule is hit.
        All other errors are unexpected and are propogated.
    """
    file_path = row['pdbfile']
    metadata = {}
    # pdb_name = f"{row['PDBID']}_t{row['chain1_id']}_b{row['chain2_id']}"
    pdb_name = f"{row['PDBID']}_{row['chain1_id']}"
    metadata['id'] = row['PDBID']
    metadata['pdb_name'] = pdb_name
    processed_path = os.path.join(write_dir, f'{pdb_name}.pkl')
    metadata.update({'target_id': row['chain1_id'], 'binder_id': row['chain2_id'] if 'chain2_id' in row else 'L',
                     'processed_path': processed_path, 'raw_path': file_path, 'data_level': 1})

    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure(pdb_name, file_path)

    # Extract all chains
    struct_chains = {
        chain.id.upper(): chain
        for chain in structure.get_chains()}
    metadata['num_chains'] = len(struct_chains)
    metadata['target_seq_len'] = len(struct_chains[row['chain1_id']])
    metadata['binder_seq_len'] = len(struct_chains[row['chain2_id']])
    print('Chains in input pdb: ', metadata['num_chains'])

    # Extract features
    struct_feats = []
    all_seqs = set()
    for chain_id, chain in struct_chains.items():
        # Convert chain id into int
        chain_id = du.chain_str_to_int(chain_id)
        chain_prot = parsers.process_chain(chain, chain_id)
        chain_dict = dataclasses.asdict(chain_prot)
        chain_dict = du.parse_chain_feats(chain_dict, normalize_positions=False)
        all_seqs.add(tuple(chain_dict['aatype']))
        struct_feats.append(chain_dict)
    # if len(all_seqs) == 1:
    #     metadata['quaternary_category'] = 'homomer'
    # else:
    #     metadata['quaternary_category'] = 'heteromer'
    complex_feats = du.concat_np_features(struct_feats, False)

    # Process geometry features
    complex_aatype = complex_feats['aatype']
    metadata['seq_len'] = len(complex_aatype)
    modeled_idx = np.where(complex_aatype != 20)[0]
    if np.sum(complex_aatype != 20) == 0:
        raise errors.LengthError('No modeled residues')
    # min_modeled_idx = np.min(modeled_idx)
    # max_modeled_idx = np.max(modeled_idx)
    # metadata['modeled_seq_len'] = max_modeled_idx - min_modeled_idx + 1
    metadata['modeled_seq_len'] = len(modeled_idx)
    complex_feats['modeled_idx'] = modeled_idx

    # complex_feats['target_interface_residues'] = ast.literal_eval(row['chain1_residues'])

    complex_feats = generate_virtual_binder_feats(complex_feats)    # only used in binder design task

    complex_feats['target_interface_residues'] = row['chain1_residues']
    if 'chain2_residues' in row.keys():
        complex_feats['binder_interface_residues'] = row['chain2_residues']
    else:
        complex_feats['binder_interface_residues'] = []
    if 'contact_pairs10A' in row.keys():
        complex_feats['contact_pairs'] = [(row['chain1_id'], row['chain2_id'], pair[0], pair[1]) for pair in
                                          row['contact_pairs10A']]      # fix-bug
    else:
        complex_feats['contact_pairs'] = []

    if 'binder_motif' in row.keys():
        complex_feats['binder_motif'] = row['binder_motif']

    if process_rotamers:
        rotamer_file_path = row['rotamer_pdbfile']
        rotamer_feats = load_rotamers(rotamer_file_path)
        complex_feats.update(rotamer_feats)

    # Write features to pickles.
    # if not os.path.exists(processed_path):
    #     du.write_pkl(processed_path, complex_feats)
    du.write_pkl(processed_path, complex_feats)

    # Return metadata
    return metadata


def process_serially(df, write_dir):
    all_metadata = []
    for i, row in tqdm(enumerate(df), total=len(df)):
        try:
            start_time = time.time()
            metadata = process_file(
                row,
                write_dir)
            elapsed_time = time.time() - start_time
            # print(f"'Finished {row['SrNo']}_{row['PDBID']} in {elapsed_time:2.2f}s'")
            all_metadata.append(metadata)
        except errors.DataError as e:
            print(f'Failed {row}: {e}')
    return all_metadata


def process_fn(
        file_path,
        verbose=None,
        write_dir=None):
    try:
        start_time = time.time()
        metadata = process_file(
            file_path,
            write_dir)
        elapsed_time = time.time() - start_time
        if verbose:
            print(f'Finished {file_path} in {elapsed_time:2.2f}s')
        return metadata
    except errors.DataError as e:
        if verbose:
            print(f'Failed {file_path}: {e}')


def main(args):
    df = pd.read_csv(args.csv_file)
    total_num_paths = len(df)
    write_dir = args.write_dir
    if not os.path.exists(write_dir):
        os.makedirs(write_dir)
    if args.debug:
        metadata_file_name = 'metadata_debug.csv'
    else:
        metadata_file_name = args.datatype+'_metadata.csv'
    metadata_path = os.path.join(write_dir, '../', metadata_file_name)
    print(f'Files will be written to {write_dir}')

    # Process each mmcif file
    if args.num_processes == 1 or args.debug:
        all_metadata = process_serially(
            [row for idx, row in df.iterrows()],
            write_dir)
    else:
        _process_fn = fn.partial(
            process_fn,
            verbose=args.verbose,
            write_dir=write_dir)
        with mp.Pool(processes=args.num_processes) as pool:
            all_metadata = list(tqdm(pool.imap(_process_fn, [row for idx, row in df.iterrows()]), total=total_num_paths))
        all_metadata = [x for x in all_metadata if x is not None]
    metadata_df = pd.DataFrame(all_metadata)
    metadata_df.to_csv(metadata_path, index=False, float_format='%.4f')
    succeeded = len(all_metadata)
    print(
        f'Finished processing {succeeded}/{total_num_paths} files')


if __name__ == "__main__":
    # Don't use GPU
    args = parser.parse_args()
    main(args)



