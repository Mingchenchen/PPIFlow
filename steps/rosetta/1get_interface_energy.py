import os
from glob import glob
from functools import partial
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from Bio import PDB
import ast
import argparse
import warnings
warnings.filterwarnings("ignore")

def get_rosetta_result(logfile):
    # output_file = logfile.replace(".out", ".csv")
    outlist = []
    with open(logfile, 'r') as infile:
        # 逐行读取输入文件
        for line in infile:
            # 检查行是否以 "ResResE" 开头
            if line.rstrip('\n').startswith("ResResE"):
                # 将符合条件的行写入输出文件
                outlist.append(line.split())
    if len(outlist) == 0:
        print(logfile)
        print("No logfile Error")
        return None
    outdf = pd.DataFrame(outlist[2:], columns=outlist[0])
    return outdf

def get_interchain_score(rosetta_path):
    #获取原始rosetta的csv
    raw_score_df = get_rosetta_result(rosetta_path)
    # raw_score_df = pd.read_csv(raw_score_path, sep="\s+")[1:]
    if raw_score_df is None:
        return None

    # 获取interchain的csv
    # 创建映射字典，根据链ID选择对应的数字
    # chain_map = {"76": "L", "82": "R", "65": "A", "66": "B"}
    chain_map = {str(i) : chr(i) for i in range(65, 91)}
    raw_score_df['binder_id'] = raw_score_df['Res1'].str.split("_", expand=True)[1].str[:2].map(chain_map)
    raw_score_df['target_id'] = raw_score_df['Res2'].str.split("_", expand=True)[1].str[:2].map(chain_map)
    raw_score_df['binder_res'] = raw_score_df['Res1'].str.split("_", expand=True)[1].str[2:]
    raw_score_df['target_res'] = raw_score_df['Res2'].str.split("_", expand=True)[1].str[2:]

    interchain_score_df = raw_score_df.loc[raw_score_df['binder_id']!=raw_score_df['target_id'],
            ['binder_id', 'target_id', 'binder_res', 'target_res', 'total']]
    interchain_score_df['total'] = interchain_score_df['total'].astype(float)
    interchain_score_df = interchain_score_df.loc[interchain_score_df['total']<0]
    interchain_score_df.reset_index(drop=True, inplace=True)
    # interchain_score_path = f'{os.path.dirname(rosetta_path)}/interchain_score.csv'
    # interchain_score_df.to_csv(interchain_score_path, index=False)
    return interchain_score_df


#########interface residues#########
# 提取残基中 CB 或 CA 原子的位置
def get_cb_or_ca(residue):
    if "CB" in residue:
        return residue["CB"].coord
    elif "CA" in residue:
        return residue["CA"].coord
    return None

def get_residue_pairs_within_distance(pdb_file, binder_id, target_id, distance_threshold=10.0):
    """
    获取PDB文件中两个链的残基对，其CB原子距离小于指定阈值，如果没有CB原子，则用CA原子。

    :param pdb_file: PDB 文件路径
    :param binder_id: 第一个链的 ID
    :param target_id: 第二个链的 ID
    :param distance_threshold: 距离阈值 (默认10.0 Å)
    :return: 符合条件的残基对列表
    """
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure("protein", pdb_file)
    model = structure[0]
    binder_chain = model[binder_id]
    target_chain = model[target_id]

    selected_pairs = set()  # 存储满足条件的残基对索引
    # selected_target = set()
    # selected_binder = set()
    for res1 in binder_chain:
        coord1 = get_cb_or_ca(res1)
        for res2 in target_chain:
            coord2 = get_cb_or_ca(res2)
            if coord1 is not None and coord2 is not None:
                distance = np.linalg.norm(coord1 - coord2)
                if distance <= distance_threshold:
                    selected_pairs.add((res1.id[1], res2.id[1]))  # 保存残基索引对
                    # selected_target.add(res1.id[1])
                    # selected_binder.add(res2.id[1])

    return selected_pairs

#####################################

def plot_score(df, plot_path):
    # 生成透视表，构造二维矩阵
    heatmap_data = df.pivot(index="target_res", columns="binder_res", values="total")

    # 画热力图
    plt.figure(figsize=(6, 5))
    sns.heatmap(heatmap_data, cmap="coolwarm", fmt=".1f", vmin=-5, vmax=1)

    plt.title("Score Heatmap")
    plt.xlabel("binder")
    plt.ylabel("target")
    plt.tight_layout()
    plt.savefig(plot_path, dpi=100)

def get_interface_energy(interchain_score_df, interface_pair, binder_id, plot_path, plot_toggle):
    # interchain_score_df = pd.read_csv(interchain_score_path)
    #('L', 'R', l_res, r_res): (pair_idx1, pair_idx2, l_res, r_res)

    interchain_score_df['in_interface'] = interchain_score_df.apply(lambda row: (int(row['binder_res']), int(row['target_res'])) in interface_pair, axis=1)
    interface_score_df = interchain_score_df.loc[interchain_score_df['in_interface']==True]
    summed_df = interface_score_df.groupby('binder_res')['total'].sum().reset_index()
    summed_dict = summed_df.set_index('binder_res')['total'].to_dict()
    print(interface_score_df.head(2))
    print(interface_score_df.columns)
    if plot_toggle:
        plot_score(interchain_score_df, plot_path)

    return summed_dict

def get_input_df(args):
    input_csv = args.input_csv
    input_pdbdir = args.input_pdbdir
    rosetta_dir = f"{args.rosetta_dir}"

    if (input_csv is None) and (input_pdbdir is None):
        raise ValueError("input_csv and input_pdbdir cannot be None at the same time")
    elif (input_csv is not None) and (input_pdbdir is not None):
        raise ValueError("Only one input_csv and input_pdbdir needs to be provided.")
    if input_csv:
        df = pd.read_csv(input_csv)

    else:
        pdbfiles = glob(f"{input_pdbdir}/*.pdb")
        print(f"Found pdb: {len(pdbfiles)}")
        df = pd.DataFrame({'pdbpath': pdbfiles})
    df["pdbname"] = df["pdbpath"].apply(lambda x: os.path.splitext(os.path.basename(x))[0])
    df['rosetta_path'] = df['pdbpath'].apply(lambda
                                                  x: f"{rosetta_dir}/{os.path.basename(x).split('.pdb')[0]}/out/{os.path.basename(x).split('.pdb')[0]}.out")

    df["target_id"] = args.target_id
    df['binder_id'] = args.binder_id

    return df

def main(row, output_dir="", distance_threshold=10, plot_toggle=True):
    logfile = row['rosetta_path']
    pdb_file = row['pdbpath']
    binder_id = row['binder_id']
    target_id = row['target_id']
    # pklfile = row['processed_path']
    try:
        """step1: get interchain energy file"""
        if not os.path.exists(logfile):
            print(row)
            print('logfile does not exist')
        interchain_score_path = get_interchain_score(logfile)
        """step2: get interface pair"""
        interface_pair = get_residue_pairs_within_distance(pdb_file, binder_id, target_id, distance_threshold=distance_threshold)
        # interface_pair = get_residue_pairs_within_distance(pklfile)

        """step3: get_interface_energy"""
        print(os.path.join(output_dir, os.path.basename(pdb_file).split('.pdb')[0]+".png"))
        summed_dict = get_interface_energy(interchain_score_path, interface_pair, binder_id, os.path.join(output_dir, os.path.basename(pdb_file).split('.pdb')[0]+".png"), plot_toggle)
    except Exception as e:
        print(e)
        return {}

    return summed_dict

def plot_binder_score(df_path, title, fontsize=15, savepath=None):
    df = pd.read_csv(df_path)
    # df = df.loc[df['binder_energy'] !="{}"]
    df['binder_energy'] = df['binder_energy'].apply(lambda x: ast.literal_eval(x))
    binder_energy = [value for dictionary in df['binder_energy'] for value in dictionary.values()]
    print("Extracted values:", len(binder_energy))
    # 绘制分布图
    plt.figure(figsize=(10, 6))
    # sns.histplot(binder_energy, kde=True)
    sns.kdeplot(binder_energy, fill=True)

    plt.title(title, fontsize=15)
    # 调整 x 轴和 y 轴刻度的字体大小
    plt.tick_params(axis='x', labelsize=fontsize)  # 调整 x 轴刻度字体大小
    plt.tick_params(axis='y', labelsize=fontsize)  # 调整 y 轴刻度字体大小
    plt.xlabel('energy', fontsize=fontsize)
    plt.ylabel('density', fontsize=fontsize)

    # plt.ylabel('Frequency', fontsize=fontsize)
    plt.xlim((-10, 0))
    plt.tight_layout()
    if savepath is not None:
        plt.savefig(savepath)

    plt.show()

def bool_type(bool_str: str):
    bool_str_lower = bool_str.lower()
    if bool_str_lower in ('false', 'f', 'no', 'n', '0'):
        return False
    elif bool_str_lower in ('true', 't', 'yes', 'y', '1'):
        return True
    else:
        raise ValueError(f'Cannot interpret {bool_str} as bool')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_csv', type=str, help='csv file contains pdb pdbpath. When the input_pdbdir is not provided, input_csv must be provided!')
    parser.add_argument('--input_pdbdir', type=str, help='The pdb folder contains pdb files. When the input_csv is not provided, input_pdbdir must be provided!')
    parser.add_argument('--rosetta_dir', type=str, help='rosetta results directory of all pdb results are saved', required=True)
    parser.add_argument('--binder_id', type=str, help='binder chain id', required=True)
    parser.add_argument('--target_id', type=str, help='target chain id', required=True)
    parser.add_argument('--output_dir', type=str, help='output csv file of all pdb results', required=True)
    parser.add_argument("--interface_dist", type=float, default=12.0, help="interface distance between target and binder")
    parser.add_argument("--plot_toggle", default=True, type=bool_type, help="dump plot or not")
    args = parser.parse_args()

    plot_toggle = args.plot_toggle
    output_dir = args.output_dir
    interface_dist = args.interface_dist
    os.makedirs(os.path.dirname(output_dir), exist_ok=True)

    #step1：获取输入的df
    df = get_input_df(args)

    #step2：获取rosetta result
    print("start Pool")
    # func = partial(main, output_dir=output_dir, distance_threshold=interface_dist)
    # with Pool(processes=cpu_count())as pool:
    #     results = list(tqdm(pool.imap(main, [row for _, row in df.iterrows()]), total=len(df)))

    # partial with the extra arguments
    func = partial(main, output_dir=output_dir, distance_threshold=interface_dist, plot_toggle=plot_toggle)
    # correctly use `func` in imap
    with Pool(processes=32) as pool:
        results = list(tqdm(pool.imap(func, [row for _, row in df.iterrows()]), total=len(df)))

    df['binder_energy'] = results

    # df.drop(columns=['pdbpath', 'pdbname'])
    df.to_csv(os.path.join(output_dir, "residue_energy.csv"), index=False)
    print(output_dir)

    #step3: 画图
    title = "interface_binder_residues_energy_sum"
    savepath = os.path.join(output_dir, 'residue_energy_interface_binder_residues_energy_sum.png')
    plot_binder_score(os.path.join(output_dir, "residue_energy.csv"), title=title, savepath=savepath)
