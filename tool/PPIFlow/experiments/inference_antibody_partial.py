import os

from data.datasets_antibody import AntibodyPartialDataset

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import torch
import hydra
from omegaconf import DictConfig, OmegaConf
from lightning import LightningModule
from lightning.pytorch import Trainer
from torch.utils.data import DataLoader
from models.flow_module_antibody_partial import FlowModule
from experiments import utils as eu
import warnings

log = eu.get_pylogger(__name__)
torch.set_float32_matmul_precision('high')
warnings.filterwarnings('ignore')

class Experiment:

    def __init__(self, *, cfg: DictConfig):
        self._cfg = cfg
        self._data_cfg = cfg.data
        self._exp_cfg = cfg.experiment
        self.dataset_cfg = cfg.ppi_dataset
        self.test_cfg = self._exp_cfg.testing_model if "testing_model" in self._exp_cfg else None
        self.batch_size = self.dataset_cfg.samples_batch_size if self.dataset_cfg.sample_original_binder_len else 1
        self._task = self._data_cfg.task
        self._test_dataset = AntibodyPartialDataset(dataset_cfg=self.dataset_cfg, task=self._task)
        if torch.cuda.is_available():
            if torch.cuda.device_count() > 1:
                self._train_device_ids = [0]
            else:
                self._train_device_ids = [0]

        else:
            self._train_device_ids = "cpu"  # debug mode
        log.info(f"Test with devices: {self._train_device_ids}")
        log.info(f"Test with devices: {self._exp_cfg.num_devices}")
        log.info(f"Test with devices: {eu.get_available_device(self._exp_cfg.num_devices)}")
        self._module: LightningModule = FlowModule(self._cfg)
        if self.test_cfg.ckpt_path is None:
            raise "Not model ckpt_path to test."

    def test(self):
        if self._exp_cfg.debug:
            log.info("Debug mode.")
            logger = None
            self._train_device_ids = [self._train_device_ids[0]]
            self._data_cfg.loader.num_workers = 0

        testloader = DataLoader(
            self._test_dataset,
            shuffle=False,
            batch_size=self.batch_size
        )

        if torch.cuda.is_available():
            trainer = Trainer(
                logger=False,
                use_distributed_sampler=False,
                enable_model_summary=True,
                devices=self._train_device_ids,
            )
        else:
            self._exp_cfg.trainer.update({'accelerator': 'cpu'})
            self._exp_cfg.trainer = {k: v for k, v in self._exp_cfg.trainer.items() if
                                     k != "strategy"}
            trainer = Trainer(
                logger=False,
                use_distributed_sampler=False,
                enable_model_summary=True,
            )

        self._module.load_state_dict(torch.load(self.test_cfg.ckpt_path, map_location="cpu", weights_only=False)["state_dict"])
        trainer.test(
            model=self._module,
            ckpt_path=None,
            dataloaders=testloader,
        )
        # trainer.test(
        #     model=self._module,
        #     ckpt_path=self.test_cfg.ckpt_path,
        #     dataloaders=testloader)


@hydra.main(version_base=None, config_path="../configs", config_name="test_antibody.yaml")
def main(cfg: DictConfig):

    if cfg.experiment.warm_start is not None and cfg.experiment.warm_start_cfg_override:
        # Loads warm start config.
        warm_start_cfg_path = os.path.join(
            os.path.dirname(cfg.experiment.warm_start), 'config.yaml')
        warm_start_cfg = OmegaConf.load(warm_start_cfg_path)

        OmegaConf.set_struct(cfg.model, False)
        OmegaConf.set_struct(warm_start_cfg.model, False)
        cfg.model = OmegaConf.merge(cfg.model, warm_start_cfg.model)
        OmegaConf.set_struct(cfg.model, True)
        log.info(f'Loaded warm start config from {warm_start_cfg_path}')
    exp = Experiment(cfg=cfg)
    exp.test()

if __name__ == "__main__":
    main()
