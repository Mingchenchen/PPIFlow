import json
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, List, Optional, Union, Set

# --- 常量定义 ---
# 标准氨基酸集合 (3字母代码)
PROTEIN_RESIDUES: Set[str] = {
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL"
}

# 核酸残基集合 (包括 DNA 和 RNA)
NUCLEIC_ACIDS: Set[str] = {
    "DA", "DC", "DT", "DG", "A", "C", "U", "G"
}

# 合并集合用于 Token 判断
VALID_RESIDUES: Set[str] = PROTEIN_RESIDUES | NUCLEIC_ACIDS


def parse_pdb_atom_line(line: str) -> Optional[Dict[str, Union[int, str]]]:
    """
    解析 PDB ATOM/HETATM 行。
    如果行格式不符合标准长度，返回 None。
    """
    if len(line) < 54:
        return None

    try:
        return {
            "atom_num": int(line[6:11]),
            "atom_name": line[12:16].strip(),
            "residue_name": line[17:20].strip(),
            "chain_id": line[21].strip(),
            "residue_seq_num": int(line[22:26]),
        }
    except ValueError:
        return None


def load_af3_pae_and_chains(json_path: Union[str, Path], pdb_path: Union[str, Path]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    从 AF3 输出文件中提取 PAE 矩阵、链 ID 和残基类型。
    
    逻辑：
    1. 解析 PDB 确定哪些残基对应 AF3 的 Token (通常是 CA 或 C1)。
    2. 读取 JSON 获取原始 PAE。
    3. 利用 PDB 生成的 mask 对 PAE 进行切片。
    """
    json_path = Path(json_path)
    pdb_path = Path(pdb_path)

    # 1. 解析 PDB 构建 mask
    token_mask = []
    chains = []
    residue_types = []

    if not pdb_path.exists():
        raise FileNotFoundError(f"PDB 文件未找到: {pdb_path}")

    with open(pdb_path, "r") as f:
        for line in f:
            if not (line.startswith("ATOM") or line.startswith("HETATM")):
                continue

            atom = parse_pdb_atom_line(line)
            if atom is None:
                continue

            atom_name = atom["atom_name"]
            res_name = atom["residue_name"]

            # Token 逻辑:
            # 1. 蛋白 CA 或 核酸 C1 -> Token=1 (保留)
            if atom_name == "CA" or (res_name in NUCLEIC_ACIDS and "C1" in atom_name):
                token_mask.append(1)
                chains.append(atom["chain_id"])
                residue_types.append(res_name)
            
            # 2. 非骨架原子 且 非标准残基 (如配体、修饰) -> Token=0 (在 PAE 中存在但需被 Mask)
            # 标准残基的其他原子不占 Token，所以忽略
            # elif (atom_name != "CA" and 
            #       "C1" not in atom_name and 
            #       res_name not in VALID_RESIDUES):
            #     token_mask.append(0)

    token_array = np.array(token_mask, dtype=bool) # 直接转为 bool 以便索引
    chain_ids = np.array(chains)
    res_types = np.array(residue_types)

    # 2. 读取 JSON
    if not json_path.exists():
        raise FileNotFoundError(f"JSON 文件未找到: {json_path}")

    with open(json_path, "r") as f:
        data = json.load(f)

    # 兼容不同的 JSON 键名
    if "pae" in data:
        raw_pae = np.array(data["pae"])
    elif "predicted_aligned_error" in data:
        raw_pae = np.array(data["predicted_aligned_error"])
    else:
        # 尝试查找列表中的第一个元素（有时 AF3 输出是 list）
        if isinstance(data, list) and len(data) > 0 and "pae" in data[0]:
            raw_pae = np.array(data[0]["pae"])
        else:
            raise ValueError(f"JSON 文件中未找到 'pae' 数据: {json_path}")

    # 3. 验证与切片
    n_tokens = len(token_mask)
    n_pae = raw_pae.shape[0]

    if n_tokens != n_pae:
        print(f"[警告] PDB 解析 Token 数 ({n_tokens}) 与 PAE 维度 ({n_pae}) 不一致。")
        print("这可能导致切片错误。请确认 PDB 是否包含所有原子以及是否为对应的模型文件。")
        # 尝试只取最小长度以防止 Crash，但数据可能不对齐
        min_dim = min(n_tokens, n_pae)
        token_array = token_array[:min_dim]
        raw_pae = raw_pae[:min_dim, :min_dim]

    # 使用布尔索引进行双向切片 (只保留有效残基行/列)
    filtered_pae = raw_pae[np.ix_(token_array, token_array)]

    return filtered_pae, chain_ids, res_types


def _calc_d0_array(L_array: np.ndarray, pair_type: str = "protein") -> np.ndarray:
    """计算 d0 归一化因子 (向量化版本)。"""
    # L 最小取 27
    L = np.maximum(27.0, L_array.astype(float))
    
    min_value = 2.0 if pair_type == "nucleic_acid" else 1.0
    
    # 公式: d0 = 1.24 * (L-15)^(1/3) - 1.8
    d0 = 1.24 * np.cbrt(L - 15.0) - 1.8
    return np.maximum(min_value, d0)


def _classify_chain_type(residue_types_subset: np.ndarray) -> str:
    """判断链类型：只要包含核酸即视为核酸链。"""
    if np.isin(residue_types_subset, list(NUCLEIC_ACIDS)).any():
        return "nucleic_acid"
    return "protein"


def calculate_ipsae(
    pae_matrix: np.ndarray, 
    chain_ids: np.ndarray, 
    residue_types: Optional[np.ndarray] = None, 
    pae_cutoff: float = 10.0
) -> Dict[Tuple[str, str], float]:
    """
    计算 ipSAE 分数 (优化版)。
    
    优化：完全向量化计算 Mean PTM，移除 Python 循环。
    """
    unique_chains = np.unique(chain_ids)
    scores = {}

    # 预判链类型
    chain_type_map = {}
    if residue_types is not None:
        for chain in unique_chains:
            mask = chain_ids == chain
            chain_type_map[chain] = _classify_chain_type(residue_types[mask])
    else:
        for chain in unique_chains:
            chain_type_map[chain] = "protein"

    # 遍历链对
    for chain1 in unique_chains:
        for chain2 in unique_chains:
            if chain1 == chain2:
                continue

            # 确定相互作用类型
            c1_type = chain_type_map[chain1]
            c2_type = chain_type_map[chain2]
            pair_type = "nucleic_acid" if "nucleic_acid" in (c1_type, c2_type) else "protein"

            # 提取子矩阵
            mask_c1 = chain_ids == chain1
            mask_c2 = chain_ids == chain2
            
            # sub_pae shape: (N_res_c1, N_res_c2)
            sub_pae = pae_matrix[np.ix_(mask_c1, mask_c2)]
            
            if sub_pae.size == 0:
                scores[(chain1, chain2)] = 0.0
                continue

            # 1. 识别有效相互作用
            valid_mask = sub_pae < pae_cutoff # Boolean matrix

            # 2. 计算 n0res (每个残基的有效接触数)
            # axis=1 表示对 Chain1 的每个残基，统计它与 Chain2 的接触数
            n0res_per_residue = np.sum(valid_mask, axis=1)

            # 3. 计算 d0
            d0_per_residue = _calc_d0_array(n0res_per_residue, pair_type)

            # 4. 计算 PTM 矩阵
            # d0_per_residue[:, None] 用于广播: (N, 1) 对 (N, M)
            ptm_matrix = 1.0 / (1.0 + (sub_pae / d0_per_residue[:, np.newaxis]) ** 2.0)

            # 5. 计算 ipSAE (向量化平均)
            # 我们只关心 valid_mask 为 True 的位置的 PTM 平均值
            
            # 将无效位置的 PTM 设为 0，方便求和
            masked_ptm_sum = np.sum(ptm_matrix * valid_mask, axis=1)
            
            # 防止除以 0 (对于没有有效接触的残基)
            with np.errstate(divide='ignore', invalid='ignore'):
                ipsae_per_residue = masked_ptm_sum / n0res_per_residue
            
            # 将 NaN (0/0 的情况) 替换为 0.0
            ipsae_per_residue = np.nan_to_num(ipsae_per_residue, nan=0.0)

            # 6. 取最大值作为最终得分
            final_score = np.max(ipsae_per_residue) if ipsae_per_residue.size > 0 else 0.0
            
            # scores[(chain1, chain2)] = float(final_score)
            scores[f"{chain1}_{chain2}"] = float(final_score)

    return scores


if __name__ == "__main__":
    pdb_file = Path("/Users/wanghongzhun/Documents/Code/AF3score/ipsae_test/7a0w_ef_b.pdb")
    json_file = Path("/Users/wanghongzhun/Documents/Code/AF3score/ipsae_test/7a0w_ef_b/seed-10_sample-0/confidences.json")

    if pdb_file.exists() and json_file.exists():
        try:
            print(f"正在处理: {pdb_file} ...")
            pae, chains, res_types = load_af3_pae_and_chains(json_file, pdb_file)
            
            # 打印形状以供调试
            print(f"PAE shape: {pae.shape}, Chains shape: {chains.shape}")
            
            results = calculate_ipsae(pae, chains, res_types, pae_cutoff=10)
            
            print("\nipSAE Scores (Directional A -> B):")
            for pair, score in results.items():
                print(f"  {pair[0]} -> {pair[1]}: {score:.4f}")
                
        except Exception as e:
            print(f"处理出错: {e}")
    else:
        print("未找到示例文件 (1AKI.pdb / 1AKI.json)，请检查路径。")