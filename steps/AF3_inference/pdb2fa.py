#!/usr/bin/env python3
"""
PDB to FASTA/CSV converter

Extract amino acid sequences from PDB files and convert to FASTA or CSV format.
"""

import os
import sys
import csv
import argparse
from pathlib import Path
from multiprocessing import Pool, cpu_count
from typing import Dict, List, Tuple, Optional

# 三字母到单字母氨基酸映射
THREE_TO_ONE = {
    "ALA": "A",
    "CYS": "C",
    "ASP": "D",
    "GLU": "E",
    "PHE": "F",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LYS": "K",
    "LEU": "L",
    "MET": "M",
    "ASN": "N",
    "PRO": "P",
    "GLN": "Q",
    "ARG": "R",
    "SER": "S",
    "THR": "T",
    "VAL": "V",
    "TRP": "W",
    "TYR": "Y",
}


def extract_sequence_from_pdb(pdb_file_path: str) -> Tuple[str, str]:
    """
    Extract amino acid sequences from a PDB file.

    Args:
        pdb_file_path: Path to the PDB file

    Returns:
        Tuple of (filename, sequence_string)
    """
    sequences: Dict[str, List[str]] = {}

    try:
        with open(pdb_file_path, "r", encoding="utf-8") as file:
            for line in file:
                # 检查行长度和格式
                if len(line) < 22 or not line.startswith("ATOM"):
                    continue

                # 提取CA原子
                if line[13:15].strip() == "CA":
                    chain_id = line[21].strip()
                    res_name = line[17:20].strip()

                    if res_name in THREE_TO_ONE:
                        if chain_id not in sequences:
                            sequences[chain_id] = []
                        sequences[chain_id].append(THREE_TO_ONE[res_name])

        # 连接所有链的序列
        sequence_str = ":".join("".join(seq) for seq in sequences.values())
        filename = os.path.basename(pdb_file_path)

        return filename, sequence_str

    except (FileNotFoundError, IOError) as e:
        print(f"Error reading {pdb_file_path}: {e}", file=sys.stderr)
        return os.path.basename(pdb_file_path), ""


def write_fasta(results: List[Tuple[str, str]], output_path: str) -> None:
    """
    Write results to FASTA format.

    Args:
        results: List of (filename, sequence) tuples
        output_path: Path to the output file
    """
    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as fasta_file:
        for pdb_name, sequence in results:
            if sequence:
                seq_name = Path(pdb_name).stem
                fasta_file.write(f">{seq_name}\n{sequence}\n")


def write_csv(results: List[Tuple[str, str]], output_path: str) -> None:
    """
    Write results to CSV format.

    Args:
        results: List of (filename, sequence) tuples
        output_path: Path to the output file
    """
    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["pdb_name", "sequence"])

        for pdb_name, sequence in results:
            if sequence:
                writer.writerow([pdb_name, sequence])


def get_pdb_files(input_dir: str) -> List[str]:
    """
    Get list of PDB files from input directory.

    Args:
        input_dir: Directory containing PDB files

    Returns:
        List of PDB file paths

    Raises:
        FileNotFoundError: If directory doesn't exist
        ValueError: If no PDB files found
    """
    if not os.path.exists(input_dir):
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    if not os.path.isdir(input_dir):
        raise ValueError(f"Input path is not a directory: {input_dir}")

    pdb_files = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith(".pdb")]

    if not pdb_files:
        raise ValueError(f"No PDB files found in directory: {input_dir}")

    return pdb_files


def get_optimal_process_count(user_processes: Optional[int] = None) -> int:
    """
    Determine optimal number of processes.

    Args:
        user_processes: User-specified number of processes

    Returns:
        Optimal number of processes
    """
    if user_processes is not None and user_processes > 0:
        return min(user_processes, cpu_count())

    # 默认使用所有核心，但不超过8个防止资源耗尽
    return min(cpu_count(), 24)


def process_pdb_files(input_dir: str, output_dir: str, output_format: str = "fa", num_processes: Optional[int] = None) -> str:
    """
    Process all PDB files in input directory and generate output file.

    Args:
        input_dir: Directory containing PDB files
        output_dir: Directory for output files
        output_format: Output format ('fa' or 'csv')
        num_processes: Number of processes to use

    Returns:
        Path to the generated output file
    """
    # 验证输出格式
    if output_format not in ["fa", "csv"]:
        raise ValueError("Output format must be 'fa' or 'csv'")

    # 获取PDB文件列表
    pdb_files = get_pdb_files(input_dir)
    print(f"Found {len(pdb_files)} PDB files")

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 生成输出文件路径
    output_file = os.path.join(output_dir, f"fasta_all.{output_format}")

    # 确定进程数
    processes = get_optimal_process_count(num_processes)
    print(f"Using {processes} processes")

    # 并行处理PDB文件
    with Pool(processes=processes) as pool:
        results = pool.map(extract_sequence_from_pdb, pdb_files)

    # 过滤有效结果
    valid_results = [(name, seq) for name, seq in results if seq]
    print(f"Successfully extracted sequences from {len(valid_results)} files")

    # 写入输出文件
    if output_format == "fa":
        write_fasta(valid_results, output_file)
    else:
        write_csv(valid_results, output_file)

    return output_file


def parse_arguments() -> argparse.Namespace:
    """
    Parse command line arguments.

    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="Convert PDB files to FASTA or CSV format",
    )

    parser.add_argument("input_dir", type=str, help="Directory containing PDB files to process")
    parser.add_argument("output_dir", type=str, help="Directory where output files will be saved")
    parser.add_argument("--format", "-f", choices=["fa", "csv"], default="fa", help="Output format (fa for FASTA, csv for CSV) [default: fa]")
    parser.add_argument("--processes", "-p", type=int, default=None, help="Number of processes to use [default: optimal based on system]")

    return parser.parse_args()


def main() -> None:
    """Main function."""
    try:
        args = parse_arguments()

        # 处理PDB文件
        output_file = process_pdb_files(input_dir=args.input_dir, output_dir=args.output_dir, output_format=args.format, num_processes=args.processes)

        print(f"Successfully generated {args.format.upper()} file: {output_file}")

    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
