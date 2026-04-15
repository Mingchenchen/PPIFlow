"""
Monomer Structure Sampling Pipeline
"""

import os
import argparse
import yaml
import ast
from typing import Dict, Any, Optional
from omegaconf import OmegaConf

from experiments.inference_monomer import EvalRunner


class ConfigManager:
    """Manages YAML configuration loading, updating, and saving."""

    def __init__(self, default_config_path: str):
        self.default_config = self._load_config(default_config_path)
        self.current_config: Optional[Dict[str, Any]] = None

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load YAML configuration file."""
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")
        with open(config_path, "r") as f:
            return yaml.safe_load(f)

    def update_config(self, updates: Dict[str, Any]) -> Dict[str, Any]:
        """Deep update configuration dictionary.

        Args:
            updates: Nested dictionary with updates to apply.

        Returns:
            Updated configuration dictionary.
        """

        def deep_update(original: Dict, update: Dict) -> Dict:
            for key, value in update.items():
                if isinstance(value, dict) and key in original:
                    deep_update(original[key], value)
                else:
                    original[key] = value
            return original

        # Create a copy to avoid modifying the original default config
        import copy
        self.current_config = deep_update(
            copy.deepcopy(self.default_config), updates
        )
        return self.current_config

    def save_config(self, save_path: str) -> None:
        """Save current configuration to YAML file."""
        with open(save_path, "w") as f:
            yaml.dump(
                self.current_config,
                f,
                sort_keys=False,
                default_flow_style=False,
            )
        print(f"Config saved to: {save_path}")


def parse_list_arg(arg_value: str):
    """Safely parse string representation of list to python list."""
    if arg_value is None:
        return None
    try:
        return ast.literal_eval(arg_value)
    except (ValueError, SyntaxError):
        # Fallback: if user provided comma separated string without brackets
        return [x.strip() for x in arg_value.split(',')]


def run_pipeline(args) -> None:
    """Execute the complete sampling pipeline.

    Args:
        args: Parsed command-line arguments.
    """
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # Load configuration
    config_manager = ConfigManager(args.config)
    
    # Base inference config
    update_configs = {
        "inference": {
            "ckpt_path": args.model_weights,
            "predict_dir": output_dir,
            "samples": {}  # Will be populated based on mode
        }
    }

    # Mode Selection Logic
    if args.length_subset is not None:
        # === Mode 1: Unconditional Monomer Sampling ===
        print(f"Mode: Unconditional Sampling (Lengths: {args.length_subset})")
        
        length_list = parse_list_arg(args.length_subset)
        
        update_configs["inference"]["samples"].update({
            "samples_per_length": args.samples_num,
            "length_subset": length_list
        })

    elif args.motif_csv is not None:
        # === Mode 2: Monomer Motif Scaffolding ===
        print(f"Mode: Motif Scaffolding (CSV: {args.motif_csv})")
        
        update_configs["inference"]["samples"].update({
            "csv_path": args.motif_csv,
            "samples_per_target": args.samples_num
        })
        
        if args.motif_names is not None:
            target_list = parse_list_arg(args.motif_names)
            update_configs["inference"]["samples"]["target_subset"] = target_list
            print(f"Target Subset: {target_list}")

    # Apply updates
    config_manager.update_config(update_configs)

    # Save config for reproducibility
    config_save_path = os.path.join(output_dir, "sample_config.yml")
    config_manager.save_config(config_save_path)

    # Convert to OmegaConf and run
    # Note: OmegaConf.create handles standard python dicts created by yaml.safe_load
    cfg = OmegaConf.create(config_manager.current_config)

    print("\nInitializing model and running inference...")
    runner = EvalRunner(cfg)
    runner.run_sampling()
    print(f"Sampling finished. Results saved to {output_dir}")


def get_parser() -> argparse.ArgumentParser:
    """Create and return the argument parser."""
    parser = argparse.ArgumentParser(
        description="Protein Structure Prediction Pipeline - Monomer Sampling"
    )

    # Model configuration
    parser.add_argument(
        "--config",
        type=str,
        default="./configs/inference_unconditional.yaml",
        help="Base configuration file path (yaml)",
    )

    # Checkpoint and output (required)
    parser.add_argument(
        "--model_weights",
        type=str,
        required=True,
        help="Model weights checkpoint file path (.ckpt)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for results",
    )

    # Sampling configuration
    # Mode 1: Unconditional
    parser.add_argument(
        "--length_subset",
        type=str,
        default=None,
        help="[Mode 1] List of sequence lengths for unconditional sampling. E.g., '[50, 100]'",
    )

    # Mode 2: Motif/Scaffolding
    parser.add_argument(
        "--motif_csv",
        type=str,
        default=None,
        help="[Mode 2] Path to CSV file containing motif definitions.",
    )

    parser.add_argument(
        "--motif_names",
        type=str,
        default=None,
        help="[Mode 2] Subset of target names from the CSV to sample. E.g., \"['01_1LDB']\"",
    )

    # Common
    parser.add_argument(
        "--samples_num",
        type=int,
        default=100,
        help="Number of samples to generate (per length or per target). Default: 100",
    )

    return parser


def validate_inputs(args) -> None:
    """Validate that required input files exist and arguments are consistent."""
    assert os.path.exists(
        args.config
    ), f"Config file not found: {args.config}"
    assert os.path.exists(
        args.model_weights
    ), f"Model weights not found: {args.model_weights}"

    # Mutual exclusion check for modes
    has_length = args.length_subset is not None
    has_motif = args.motif_csv is not None

    if has_length and has_motif:
        raise ValueError(
            "Conflict: Both '--length_subset' and '--motif_csv' provided. "
            "Please choose either Unconditional Sampling OR Motif Scaffolding."
        )
    
    if not has_length and not has_motif:
        raise ValueError(
            "Missing arguments: Must provide either '--length_subset' (for unconditional) "
            "or '--motif_csv' (for motif scaffolding)."
        )

    if args.motif_csv and not os.path.exists(args.motif_csv):
        raise FileNotFoundError(f"Motif CSV file not found: {args.motif_csv}")


def main() -> None:
    """Main entry point for the monomer sampling pipeline."""
    parser = get_parser()
    args = parser.parse_args()

    try:
        validate_inputs(args)
        run_pipeline(args)
    except Exception as e:
        print(f"Error: {e}")
        exit(1)


if __name__ == "__main__":
    main()