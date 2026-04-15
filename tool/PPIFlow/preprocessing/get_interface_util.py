import os
from time import time
from tqdm import tqdm
from glob import glob
from Bio import PDB
import numpy as np
import pandas as pd
import argparse
from Bio.PDB import PDBParser
# from multiprocessing import Pool,cpu_count


# 提取残基中 CB 或 CA 原子的位置
def get_cb_or_ca(residue):
    if "CB" in residue:
        return residue["CB"].coord
    elif "CA" in residue:
        return residue["CA"].coord
    return None

def get_residue_pairs_within_distance(pdb_file, chain1_id, chain2_id, distance_threshold=10.0):
    """
    获取PDB文件中两个链的残基对，其CB原子距离小于指定阈值，如果没有CB原子，则用CA原子。

    :param pdb_file: PDB 文件路径
    :param chain1_id: 第一个链的 ID
    :param chain2_id: 第二个链的 ID
    :param distance_threshold: 距离阈值 (默认10.0 Å)
    :return: 符合条件的残基对列表
    """
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("protein", pdb_file)
    model = structure[0]  # 使用第一个模型
    chain1 = model[chain1_id]
    chain2 = model[chain2_id]

    selected_pairs = set()  # 存储满足条件的残基对索引
    selected_target = set()
    selected_binder = set()
    for res1 in chain1:
        coord1 = get_cb_or_ca(res1)
        for res2 in chain2:
            coord2 = get_cb_or_ca(res2)
            if coord1 is not None and coord2 is not None:
                distance = np.linalg.norm(coord1 - coord2)
                if distance <= distance_threshold:
                    selected_pairs.add((res1.id[1], res2.id[1]))  # 保存残基索引对
                    selected_target.add(res1.id[1])
                    selected_binder.add(res2.id[1])
    print(selected_pairs)
    return [list(selected_pairs), sorted(list(selected_target)), sorted(list(selected_binder))]



def main():
    parser = argparse.ArgumentParser(description='Analyze complex complex interfaces')
    parser.add_argument('--input', type=str, help='Input cluster csv file', required=True)
    parser.add_argument('--output', type=str, help='Output csv file with contact_pairs residues', required=True)
    parser.add_argument('--chain1_id', type=str, help='chain1_id', default='R')
    parser.add_argument('--chain2_id', type=str, help='chain2_id', default='L')
    parser.add_argument('--cutoff', type=float, help='interface cutoff', default=10)
    args = parser.parse_args()
    print(args)
    start_time = time()

    pdbdir = "/Users/linjie/projects/protein-frame-flow_gpuold/apdb/pdb"
    df = pd.read_csv(args.input)
    df['pdbfile'] = df['PDBID'].apply(lambda x: f"{pdbdir}/{x}.pdb")
    pair_list = []
    target_list = []
    binder_list = []
    for pdbfile in tqdm(df['pdbfile'], total=len(df)):
        result = get_residue_pairs_within_distance(pdbfile, args.chain1_id, args.chain2_id, args.cutoff)
        pair_list.append(result[0])
        target_list.append(result[1])
        binder_list.append(result[2])
    df['contact_pairs10A'] = pair_list
    df['chain1_residues'] = target_list
    df['chain2_residues'] = binder_list
    df['chain1_id'] = args.chain1_id
    df['chain2_id'] = args.chain2_id
    df.to_csv(f"{args.output}", index=False)
    print(f"finished in {time()-start_time} seconds")


if __name__ == '__main__':
    main()