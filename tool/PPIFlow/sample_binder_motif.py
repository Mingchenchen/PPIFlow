"""
Backbone Structure Sampling Pipeline - Binder Design
"""

import os
import shutil
import pandas as pd
import time
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
import argparse
from omegaconf import DictConfig, OmegaConf

from experiments.inference_binder import Experiment

from preprocessing.get_interface_util import get_residue_pairs_within_distance
from preprocessing.process_pdb_for_inputs import process_file

"""
Config Manager: 读取、修改和保存YAML配置文件
"""


class ConfigManager:
    def __init__(self, default_config_path: str):
        self.default_config = self._load_config(default_config_path)
        self.current_config = None

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """加载YAML配置文件"""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)

    def update_config(self, updates: Dict[str, Any]) -> Dict[str, Any]:
        """
        深度更新配置字典
        示例: updates = {'model': {'dropout': 0.2}, 'input': {'pdb_path': 'new_path.pdb'}}
        """

        def deep_update(original, update):
            for key, value in update.items():
                if isinstance(value, dict) and key in original:
                    deep_update(original[key], value)
                else:
                    original[key] = value
            return original

        self.current_config = deep_update(self.default_config.copy(), updates)
        return self.current_config

    def save_config(self, save_path: str):
        """保存配置到YAML文件"""
        config_to_save = self.current_config
        with open(save_path, 'w') as f:
            yaml.dump(config_to_save, f, sort_keys=False, default_flow_style=False)
        print(f"Config saved to: {save_path}")



# --------------------------
# 预处理模块
# --------------------------
def preprocess_csv_and_pkl(pdb_path, output_dir, args) -> str:
    """
    处理PDB文件生成pkl和metadata
    返回: csv_path
    """
    print('preprocessing data: ', pdb_path)
    input_info = {'pdbfile': pdb_path, 'PDBID': args.name}
    if args.specified_hotspots is not None:
        print('use specified hotspots: ', args.specified_hotspots)
        hotspot_residues = [int(x.strip(args.target_chain)) for x in args.specified_hotspots.split(',')]
        input_info['chain1_residues'] = hotspot_residues
        input_info['chain1_id'] = args.target_chain
        pass
    else:
        print('hotspots not specified, generate hotspots according to known binder interface...')
        assert args.binder_chain is not None, 'must provide binder chain if no hotspots are given'
        interface_data = get_residue_pairs_within_distance(pdb_path, args.target_chain, args.binder_chain,
                                                           distance_threshold=10.0)

        input_info['contact_pairs10A'] = interface_data[0]
        input_info['chain1_residues'] = interface_data[1]
        input_info['chain2_residues'] = interface_data[2]
        input_info['chain1_id'] = args.target_chain
        input_info['chain2_id'] = args.binder_chain

    metadata = process_file(input_info, write_dir=output_dir)
    metadata['num_chains'] = 2
    metadata['contig'] = args.motif_contig
    metadata['sample_binder_len'] = args.sample_binder_len

    metadata_df = pd.DataFrame([metadata])    # one item only
    csv_path = os.path.join(output_dir, f'{args.name}_input.csv')
    metadata_df.to_csv(csv_path, index=False)

    return csv_path



# --------------------------
# 主流程
# --------------------------
def run_pipeline(args):
    # 1. 创建输出目录
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # 2. 预处理
    if args.input_pdb is not None:
        # print(f"\nPreprocessing {args.input_pdb} for metadata...")
        assert args.target_chain is not None, 'must provide target chain id'
        input_data_dir = os.path.join(output_dir, 'input')
        os.makedirs(input_data_dir, exist_ok=True)
        processed_csv_path = preprocess_csv_and_pkl(
            pdb_path=args.input_pdb,
            output_dir=input_data_dir,
            args=args
        )
    else:
        assert args.input_csv is not None, 'must provide either input csv or pdb'
        processed_csv_path = args.input_csv
        shutil.copy(processed_csv_path, os.path.join(output_dir, f'{args.name}_input.csv'))

    # 3. 加载并更新配置
    conf = ConfigManager(args.config)    # load default config
    update_configs = {
        'ppi_dataset':
            {'test_csv_path': processed_csv_path,
             'samples_per_target': args.samples_per_target,
             'define_hotspots': True if (args.specified_hotspots is not None) else False,
             'min_hotspot_ratio': args.sample_hotspot_rate_min,
             'max_hotspot_ratio': args.sample_hotspot_rate_max},
        'experiment':
            {'testing_model': {'ckpt_path': args.model_weights, 'save_dir': args.output_dir},
             'checkpointer': {'dirpath': args.output_dir}},
    }
    conf.update_config(update_configs)
    # import pdb; pdb.set_trace()
    config_save_path = os.path.join(output_dir, 'binder_sample_config.yml')
    conf.save_config(config_save_path)

    with open(config_save_path, "r") as f:
        cfg_dict = yaml.safe_load(f)
    cfg = OmegaConf.create(cfg_dict)

    # 4. 初始化模型
    print("\nInitializing model...")
    t1 = time.time()
    exp = Experiment(cfg=cfg)

    # 5. 执行推理
    print("\nRunning inference...")
    exp.test()

    t2 = time.time()
    print(f'sample finished in {round(t2 - t1, 2)} seconds')

    return


# --------------------------
# 命令行接口
# --------------------------
def get_parser():
    parser = argparse.ArgumentParser(
        description="Protein Structure Prediction Pipeline")

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--input_pdb", type=str, help="Input protein PDB file path")
    group.add_argument("--input_csv", type=str, help="Input CSV file path")

    parser.add_argument("--target_chain", type=str, default='R',
                        help="Chain id of the target (only single chain target is supported), e.g., B")
    parser.add_argument("--binder_chain", type=str, default='L',
                        help="Chain id of the binder (motif), e.g., L")

    # model configs
    parser.add_argument("--config", type=str, default="./configs/test_ppi_complex_motif.yaml",
                        help="default configuration file")

    # hotspots
    parser.add_argument("--specified_hotspots", type=str,
                        help="Specify hotspot residues, e.g., 'R162,R165,R210'")
    parser.add_argument('--sample_hotspot_rate_min', type=float, default=0.2,
                        help='Minimum sampling rate (default: 0.2)')
    parser.add_argument('--sample_hotspot_rate_max', type=float, default=0.5,
                        help='Maximum sampling rate (default: 0.5)')

    # motif
    parser.add_argument('--motif_contig', type=str, required=True,
                        help="Motif contig, e.g., '10-40,L19-27,10-40'")
    parser.add_argument('--sample_binder_len', type=str, required=True,
                        help="Sample binder length or length range, e.g., 60, or 60-90")

    # samples
    parser.add_argument('--samples_per_target', type=int, default=100, help='number of samples')

    # checkpoint
    parser.add_argument("--model_weights", type=str,
                        default='/lustre/grp/cmclab/share/guoly/ppiflow/ckpt/20250407_motif_v45_monomer_rcsb_from_mono_ep131/epoch=88-step=80367.ckpt',
                        help="Model weights file path")

    # output dir
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory")
    parser.add_argument("--name", type=str, default="test_target",
                        help="Test target name")
    return parser


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()

    # 验证输入文件
    assert os.path.exists(args.input_pdb), f"PDB file not found: {args.input_pdb}"
    assert os.path.exists(args.config), f"Config file not found: {args.config}"
    assert os.path.exists(args.model_weights), f"Model weights file not found: {args.model_weights}"

    if args.specified_hotspots:
        if args.sample_hotspot_rate_min != 0.2 or args.sample_hotspot_rate_max != 0.5:
            parser.error("--specified_hotspots cannot be used together with sampling rate args")

    os.makedirs(args.output_dir, exist_ok=True)

    run_pipeline(args)
