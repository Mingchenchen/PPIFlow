import numpy as np
from Bio.PDB import PDBParser

def detect_backbone_clash(pdb_path, min_clash_distance=2.0):
    """
    检测 PDB 文件中是否存在骨架 CA 原子之间的冲突（clash）
    只返回 True / False
    
    参数：
        pdb_path: str, PDB 文件路径
        min_clash_distance: float, 冲突阈值 (Å)
    
    返回：
        bool — 是否存在冲突
    """
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("structure", pdb_path)
    
    all_coords = []

    # 提取所有链的 CA 原子坐标
    for chain in structure.get_chains():
        for residue in chain:
            if "CA" not in residue:
                continue
            coord = residue["CA"].get_coord()
            all_coords.append(coord)

    n = len(all_coords)
    if n < 2:
        return False  # 少于两个原子，不可能有冲突

    coords = np.array(all_coords)

    # 计算距离矩阵
    diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
    dist_matrix = np.linalg.norm(diff, axis=-1)

    # 只取上三角部分（避免重复与自身）
    i_idx, j_idx = np.triu_indices(n, k=1)

    # 判断是否存在距离小于阈值的原子对
    has_clash = np.any(dist_matrix[i_idx, j_idx] < min_clash_distance)

    return bool(has_clash)

def calculate_distance(atom1, atom2):
    """计算两个原子之间的欧几里得距离"""
    return np.linalg.norm(atom1.get_coord() - atom2.get_coord())


def get_backbone_atoms(chain):
    """获取链的主链原子（N, CA, C）"""
    backbone_atoms = []

    for residue in chain:
        # 跳过异质原子（如水分子、辅因子等）
        if residue.get_id()[0] != " ":
            continue

        atoms = {}
        for atom in residue:
            if atom.get_name() in ["N", "CA", "C"]:
                atoms[atom.get_name()] = atom

        # 确保三个主链原子都存在
        if "N" in atoms and "CA" in atoms and "C" in atoms:
            backbone_atoms.append(
                {
                    "residue": residue,
                    "N": atoms["N"],
                    "CA": atoms["CA"],
                    "C": atoms["C"],
                    "residue_id": residue.get_id()[1],
                }
            )

    return backbone_atoms


def detect_breaks_in_chain(chain, max_peptide_bond_length=1.5):
    """
    检测链中的断裂
    利用向量化计算相邻残基的C-N距离
    """
    backbone_atoms = get_backbone_atoms(chain)
    n = len(backbone_atoms)

    if n < 2:
        return 0, []

    # 提取相邻残基的 C 和 N 坐标
    c_coords = np.array([res["C"].get_coord() for res in backbone_atoms[:-1]])
    n_coords = np.array([res["N"].get_coord() for res in backbone_atoms[1:]])

    distances = np.linalg.norm(c_coords - n_coords, axis=1)
    breaks_mask = distances > max_peptide_bond_length

    break_info = [
        {
            "residue1": backbone_atoms[i]["residue_id"],
            "residue2": backbone_atoms[i + 1]["residue_id"],
            "distance": float(distances[i]),
            "chain_id": chain.get_id(),
        }
        for i in np.where(breaks_mask)[0]
    ]

    return len(break_info), break_info


def detect_breaks_in_structure(pdb_file, max_peptide_bond_length=1.5):
    """
    检测PDB结构中的所有断裂

    Args:
        pdb_file: PDB文件路径
        max_peptide_bond_length: 最大肽键长度（Å）

    Returns:
        total_breaks: 总断裂数量
        all_break_info: 所有断裂信息
    """
    try:
        # 解析PDB文件
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure("protein", pdb_file)
    except Exception as e:
        print(f"错误：无法解析PDB文件 {pdb_file}")
        print(f"错误信息：{e}")
        return 0, []

    total_breaks = 0
    all_break_info = []

    # 遍历所有链
    for model in structure:
        for chain in model:
            chain_breaks, chain_break_info = detect_breaks_in_chain(
                chain, max_peptide_bond_length
            )
            total_breaks += chain_breaks
            all_break_info.extend(chain_break_info)

    return total_breaks, all_break_info