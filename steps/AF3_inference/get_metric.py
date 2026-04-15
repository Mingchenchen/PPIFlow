import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
import multiprocessing as mp
from tqdm import tqdm
import glob
import argparse
import functools


from ipsae_calculator import load_af3_pae_and_chains, calculate_ipsae


def parse_confidences_json(conf_path):
    """
    从 confidences.json 文件中解析 pLDDT 和 PAE 相关指标。
    """
    with open(conf_path) as f:
        conf = json.load(f)

    # 从 token_chain_ids 中动态获取所有唯一的链ID
    token_chain_ids = conf["token_chain_ids"]
    chain_ids = sorted(list(set(token_chain_ids)))

    pae = np.array(conf["pae"])

    # 按链划分索引
    chain_indices = {chain: [] for chain in chain_ids}
    for i, chain_id in enumerate(token_chain_ids):
        chain_indices[chain_id].append(i)

    # 计算链内PAE (intra-chain PAE)
    chain_pae = {chain: float(np.mean(pae[np.ix_(idxs, idxs)])) for chain, idxs in chain_indices.items()}

    return chain_pae, chain_ids


def process_single_description(args):
    """
    处理单个seed目录，提取所有相关的质量评估指标。
    args: (base_path_str, pdb_dir_str)
    """
    base_path_str, pdb_dir_str = args

    try:
        base_path = Path(base_path_str)
        seed = base_path.name
        pdb_name = base_path.parent.name  # 假设目录结构是 .../pdb_name/seed_...

        pdb_path = Path(pdb_dir_str) / f"{pdb_name}.pdb"
        if not pdb_path.exists():
            return None, f"{pdb_name}/{seed}: Missing PDB file {pdb_path}."

        summary_path = base_path / "summary_confidences.json"
        conf_path = base_path / "confidences.json"

        if not summary_path.exists() or not conf_path.exists():
            return None, f"{pdb_name}/{seed}: Missing confidences files."

        summary = json.loads(summary_path.read_text())
        conf = json.loads(conf_path.read_text())

        # --- ipSAE 计算 ---
        ipsae_metrics = {}
        pae_matrix, pdb_chain_ids, residue_types = load_af3_pae_and_chains(conf_path, pdb_path)
        ipsae_dict = calculate_ipsae(pae_matrix, pdb_chain_ids, residue_types, pae_cutoff=10)

        for k, v in ipsae_dict.items():
            ipsae_metrics[f"ipsae_{k}"] = v

        # --- 基础指标提取 ---

        # 统一使用 chain_ids
        chain_ids = pd.unique(pdb_chain_ids).tolist()

        # chain_iptm / chain_ptm
        iptm_dict = dict(zip(chain_ids, summary.get("chain_iptm", [0.0] * len(chain_ids))))
        ptm_dict = dict(zip(chain_ids, summary.get("chain_ptm", [0.0] * len(chain_ids))))

        # interchain_iptm
        iptm_matrix = summary["chain_pair_iptm"]
        interchain_iptm_dict = {}
        num_chains = len(chain_ids)
        for i in range(num_chains):
            for j in range(i + 1, num_chains):  # 只取上三角，避免重复
                chain_i = chain_ids[i]
                chain_j = chain_ids[j]
                iptm_score = iptm_matrix[i][j]
                interchain_iptm_dict[f"iptm_{chain_i}_{chain_j}"] = iptm_score

        # pLDDT
        atom_plddts = conf["atom_plddts"]
        atom_chain_ids_list = conf["atom_chain_ids"]
        chain_plddt = {}
        for ch in chain_ids:
            plddts = [pl for pl, cid in zip(atom_plddts, atom_chain_ids_list) if cid == ch]
            chain_plddt[ch] = float(np.mean(plddts)) if plddts else 0.0
        # 原始顺序的链 key
        original_keys = list(chain_plddt.keys())
        # 构建新字典，用 chain_ids 替换 key
        new_chain_plddt = {new_key: chain_plddt[old_key] for new_key, old_key in zip(chain_ids, original_keys)}

        # PAE相关指标 (Intra-chain)
        chain_pae_dict, _ = parse_confidences_json(conf_path)
        original_keys = list(chain_pae_dict.keys())
        chain_pae = {new_key: chain_pae_dict[old_key] for new_key, old_key in zip(chain_ids, original_keys)}

        # --- 结果整合 ---
        result = {
            "pdb_name": pdb_name,
            "seed": seed,
            "ptm": summary.get("ptm", 0.0),
            "iptm": summary.get("iptm", 0.0),
        }

        # 添加每条链的指标 (动态添加，支持多链)
        for ch in chain_ids:
            result[f"chain_{ch}_plddt"] = new_chain_plddt.get(ch, np.nan)
            result[f"chain_{ch}_pae"] = chain_pae.get(ch, np.nan)
            result[f"chain_{ch}_ptm"] = ptm_dict.get(ch, np.nan)
            result[f"chain_{ch}_iptm"] = iptm_dict.get(ch, np.nan)

        # 添加 ipSAE 指标
        result.update(ipsae_metrics)
        result.update(interchain_iptm_dict)

        return result, None

    except Exception as e:
        pdb_name = Path(base_path_str).parent.name
        seed = Path(base_path_str).name
        return None, f"Error processing {pdb_name}/{seed}: {str(e)}"


def extract_all_metrics_parallel(base_dir, pdb_dir, num_workers=None):
    """
    使用glob查找所有目标目录，并并行处理它们。
    """
    pattern = os.path.join(base_dir, "*", "seed*")
    all_seed_dirs = glob.glob(pattern)

    if not all_seed_dirs:
        print(f"Warning: No 'seed_*' directories found matching the pattern in '{base_dir}'.")
        print(f"Expected structure: {base_dir}/<name>/<name>/seed*")
        return pd.DataFrame(), []

    print(f"Found {len(all_seed_dirs)} 'seed_*' directories to process.")

    results = []
    failed = []

    num_procs = num_workers or 24

    # 准备参数列表，将 pdb_dir 传递给每个任务
    task_args = [(d, pdb_dir) for d in all_seed_dirs]

    with mp.Pool(processes=num_procs) as pool:
        # 注意：imap_unordered 只能传一个参数，所以我们将路径和配置打包成元组 task_args
        for res, err in tqdm(pool.imap_unordered(process_single_description, task_args), total=len(all_seed_dirs), desc="Processing AF3 outputs"):
            if err:
                failed.append(err)
            elif res:
                results.append(res)

    df = pd.DataFrame(results)
    return df, failed


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract metrics from AlphaFold3 output directories.")
    parser.add_argument(
        "--af3_output_dir",
        required=True,  # 建议设为 required 或保留 default
        help="Base directory containing the AF3 outputs.",
    )
    parser.add_argument(
        "--pdb_dir",
        required=True,
        help="Directory containing the ground truth PDB files for ipSAE calculation.",
    )
    parser.add_argument(
        "--save_metric_csv",
        default="./af3_metrics.csv",
        help="Path to save the resulting metrics CSV file.",
    )
    parser.add_argument("--num_workers", type=int, default=32, help="Number of parallel workers to use.")
    args = parser.parse_args()

    # 路径处理
    output_dir = Path(args.af3_output_dir)
    pdb_dir = args.pdb_dir
    save_path = Path(args.save_metric_csv)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    df, failed = extract_all_metrics_parallel(args.af3_output_dir, pdb_dir, args.num_workers)

    if not df.empty:
        # 将列排序，让通用指标在前，链特异指标在后
        cols = list(df.columns)
        # base_cols = ["pdb_name", "seed", "complex_iptm", "complex_ptm"]
        # other_cols = [c for c in cols if c not in base_cols]
        # df = df[base_cols + sorted(other_cols)]

        df.to_csv(save_path, index=False)

        # 提取最佳模型 (基于 complex_iptm)
        if "iptm" in df.columns:
            # 处理 NaN 值，防止 idxmax 报错
            temp_df = df.dropna(subset=["iptm"])
            if not temp_df.empty:
                df_iptmmax = temp_df.loc[temp_df.groupby("pdb_name")["iptm"].idxmax()].reset_index(drop=True)
                df_iptmmax_csv = save_path.parent / "af3_iptm@1.csv"
                df_iptmmax.to_csv(df_iptmmax_csv, index=False)

        print(f"✅ Successfully processed {len(df)} items.")
        print(f"Metrics saved to {save_path}")
    else:
        print("⚠️ No items were successfully processed.")

    if failed:
        print(f"❌ Encountered {len(failed)} failures.")
        failed_records_path = save_path.parent / "failed_records.txt"
        with open(failed_records_path, "w") as fw:
            fw.write("\n".join(failed))
        print(f"Failure logs written to {failed_records_path}")
