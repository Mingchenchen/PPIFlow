import argparse
import json
import numpy as np
import os

def make_fixed_positions_dict(parsed_chain_dict_list, chain_list, position_list, specify_non_fixed=False):
    if not position_list.endswith(".csv"):
        raise ValueError("Expected a CSV file as --position_list")

    # 构造 {PDB_ID: [res_idx, ...]} 映射表
    fixed_dict = {}
    with open(position_list, "r") as f:
        lines = f.readlines()
    for line in lines[1:]:  # 跳过表头
        line = line.strip()
        if not line:
            continue
        parts = line.split(",", 1)
        if len(parts) != 2:
            raise ValueError(f"Invalid CSV line: {line}")
        pdb_id = parts[0].strip()
        chains_pos_str = parts[1].split("-")
        fixed_dict[pdb_id] = []
        for pos_str in chains_pos_str:
            fixed_dict[pdb_id].append([int(x) for x in pos_str.strip().split()])
            # fixed_dict[pdb_id] = [int(x) for x in pos_str.split()]   if pos_str else []

    global_designed_chain_list = chain_list.split()
    my_dict = {}

    for result in parsed_chain_dict_list:
        pdb_id = result['name']
        all_chain_list = [item[-1:] for item in list(result) if item.startswith('seq_chain')]
        fixed_position_dict = {}
        fixed_residues = fixed_dict.get(pdb_id, [])

        for idx, chain in enumerate(all_chain_list):
            seq_length = len(result[f'seq_chain_{chain}'])
            all_residues = list(range(1, seq_length + 1))
            if not specify_non_fixed:
                fixed_position_dict[chain] = fixed_residues[idx] if chain in global_designed_chain_list else []
            else:
                if chain not in global_designed_chain_list:
                    fixed_position_dict[chain] = all_residues
                else:
                    fixed_position_dict[chain] = list(set(all_residues) - set(fixed_residues))

        my_dict[pdb_id] = fixed_position_dict
    # print(my_dict)
    return my_dict

def main(args):
    with open(args.input_path, 'r') as f:
        json_list = [json.loads(line) for line in f if line.strip()]

    fixed_dict = make_fixed_positions_dict(
        parsed_chain_dict_list=json_list,
        chain_list=args.chain_list,
        position_list=args.position_list,
        specify_non_fixed=args.specify_non_fixed
    )

    with open(args.output_path, 'w') as f:
        json.dump(fixed_dict, f, indent=2)

if __name__ == "__main__":
    argparser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    argparser.add_argument("--input_path", type=str, help="Path to the parsed PDBs")
    argparser.add_argument("--output_path", type=str, help="Path to the output dictionary")
    argparser.add_argument("--chain_list", type=str, default='', help="List of the chains that need to be fixed")
    argparser.add_argument("--position_list", type=str, default='', help="CSV file with columns PDB_ID,fixed_positions")
    argparser.add_argument("--specify_non_fixed", action="store_true", default=False, help="Allows specifying just residues that need to be designed (default: false)")

    args = argparser.parse_args()
    main(args)