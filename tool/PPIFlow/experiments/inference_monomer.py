"""Script for running inference and evaluation."""
import sys
import os
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

import time
import numpy as np
import hydra
import torch
import GPUtil
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from omegaconf import DictConfig, OmegaConf
from experiments import utils as eu
from models.flow_module_monomer import FlowModule


torch.set_float32_matmul_precision('high')
log = eu.get_pylogger(__name__)


class EvalRunner:

    def __init__(self, cfg: DictConfig):
        """Initialize sampler.

        Args:
            cfg: inference config.
        """
        ckpt_path = cfg.inference.ckpt_path

        self._cfg = cfg
        self._exp_cfg = cfg.experiment
        self._infer_cfg = cfg.inference
        self._samples_cfg = self._infer_cfg.samples
        self._rng = np.random.default_rng(self._infer_cfg.seed)

        # Set-up output directory only on rank 0
        local_rank = os.environ.get('LOCAL_RANK', 0)
        if local_rank == 0:
            inference_dir = self.setup_inference_dir(ckpt_path)
            self._exp_cfg.inference_dir = inference_dir
            config_path = os.path.join(inference_dir, 'config.yaml')
            with open(config_path, 'w') as f:
                OmegaConf.save(config=self._cfg, f=f)
            log.info(f'Saving inference config to {config_path}')

        # Read checkpoint and initialize module.
        if torch.cuda.is_available():
            map_location = None
        else:
            map_location = "cpu"

        print(f'loading checkpoint from {ckpt_path}')
        # self._flow_module = FlowModule.load_from_checkpoint(
        #     checkpoint_path=ckpt_path,
        #     cfg=self._cfg,
        #     map_location=map_location
        # )
        self._flow_module = FlowModule(self._cfg)
        self._flow_module.load_state_dict(
            torch.load(ckpt_path, map_location=map_location, weights_only=False)["state_dict"]
        )

        log.info(pl.utilities.model_summary.ModelSummary(self._flow_module))
        self._flow_module.eval()
        self._flow_module._infer_cfg = self._infer_cfg
        self._flow_module._samples_cfg = self._samples_cfg

    @property
    def inference_dir(self):
        return self._flow_module.inference_dir

    def setup_inference_dir(self, ckpt_path):
        output_dir = self._infer_cfg.predict_dir
        os.makedirs(output_dir, exist_ok=True)
        log.info(f'Saving results to {output_dir}')
        return output_dir

    def run_sampling(self):
        if torch.cuda.is_available():
            devices = GPUtil.getAvailable(
                order='memory', limit = 8)[:self._infer_cfg.num_gpus]
        else:
            devices = "cpu"
        log.info(f"Using devices: {devices}")
        log.info(f'Evaluating {self._infer_cfg.task}')
        if self._infer_cfg.task == 'unconditional':
            eval_dataset = eu.LengthDataset(self._samples_cfg)
        elif self._infer_cfg.task == 'scaffolding':
            eval_dataset = eu.ScaffoldingDataset(self._samples_cfg)
        else:
            raise ValueError(f'Unknown task {self._infer_cfg.task}')
        dataloader = torch.utils.data.DataLoader(
            eval_dataset, batch_size=1, shuffle=False, drop_last=False)
        if torch.cuda.is_available():
            trainer = Trainer(
                accelerator="gpu",
                strategy="ddp",
                devices=devices,
            )
        else:
            trainer = Trainer(
                accelerator="cpu",
            )
        trainer.predict(self._flow_module, dataloaders=dataloader)


@hydra.main(version_base=None, config_path="../configs", config_name="inference_unconditional.yaml")
def run(cfg: DictConfig) -> None:

    # Read model checkpoint.
    log.info(f'Starting inference with {cfg.inference.num_gpus} GPUs')
    start_time = time.time()
    sampler = EvalRunner(cfg)
    sampler.run_sampling()
    elapsed_time = time.time() - start_time
    log.info(f'Finished in {elapsed_time:.2f}s')

if __name__ == '__main__':
    run()

