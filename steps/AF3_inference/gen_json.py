import json
import random
import os
from Bio import SeqIO

# 允许使用的链ID，从A开始
chain_ids = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

def generate_json_from_fasta(fasta_file, output_dir="jsons", random_seeds_count=10):
    """
    读取 FASTA 文件，并为每条序列生成 JSON 文件，支持多链，存放在指定文件夹中。
    参数:
    - fasta_file: str, 输入 FASTA 文件路径
    - output_dir: str, 输出 JSON 存放的文件夹 (默认 "json_outputs")
    - random_seeds_count: int, 使用的随机种子数量
    """

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    fixed_seeds = [1, 10, 3090, 999, 42, 101, 1024, 1012, 4090, 1020]
    extended_seeds = fixed_seeds + [i for i in range(1, 201) if i not in fixed_seeds]

    
    for record in SeqIO.parse(fasta_file, "fasta"):
        seq_name = record.id
        sequence = str(record.seq)

        # 拆分多链序列
        chains = sequence.split(":")

        sequences_block = []
        for idx, chain_seq in enumerate(chains):
            if idx >= len(chain_ids):
                raise ValueError(f"链数量超过支持范围({len(chain_ids)})!")
            chain_id = chain_ids[idx]

            sequences_block.append({
                "protein": {
                    "id": chain_id,
                    "sequence": chain_seq,
                    "unpairedMsa": None,
                    "pairedMsa": None,
                    "templates": None
                }
            })

        random_seeds_count = min(random_seeds_count, len(extended_seeds))
        model_seeds = extended_seeds[:random_seeds_count]

        json_data = {
            "name": seq_name,
            "sequences": sequences_block,
            "modelSeeds": model_seeds,
            "dialect": "alphafold3",
            "version": 1
        }

        os.makedirs(os.path.join(output_dir, seq_name), exist_ok=True)
        json_filename = os.path.join(output_dir, seq_name, f"{seq_name}.json")

        with open(json_filename, "w", encoding="utf-8") as json_file:
            json.dump(json_data, json_file, indent=2)

    print(f"✅ JSON 文件已生成: {output_dir}/")

import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert FASTA to JSON with multi-chain support")
    parser.add_argument("fasta_file", help="Input FASTA file")
    parser.add_argument("output_dir", help="Output directory for JSON files")
    parser.add_argument("random_seeds_count", type=int, help="Number of model seeds")

    args = parser.parse_args()
    generate_json_from_fasta(args.fasta_file, args.output_dir, args.random_seeds_count)