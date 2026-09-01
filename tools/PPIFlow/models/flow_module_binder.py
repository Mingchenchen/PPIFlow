from typing import Any
import torch
import os
import numpy as np
import pandas as pd
import logging
import torch.distributed as dist
from lightning import LightningModule

from analysis import metrics
from analysis import utils as au
from models.flow_model_binder import FlowModel
from models import utils as mu
from data.interpolant_binder import Interpolant
from data import utils as du
from data import all_atom
from data import so3_utils
from data import residue_constants
from omegaconf import OmegaConf

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FlowModule(LightningModule):

    def __init__(self, cfg):
        super().__init__()
        self._print_logger = logging.getLogger(__name__)
        self._exp_cfg = cfg.experiment
        self._model_cfg = cfg.model
        self._data_cfg = cfg.data
        self._interpolant_cfg = cfg.interpolant

        # Set-up vector field prediction model
        OmegaConf.set_struct(cfg, False)
        cfg.model["task"] = self._data_cfg.task
        self.model = FlowModel(cfg.model)

        # Set-up interpolant
        self.interpolant = Interpolant(cfg.interpolant)

        self.test_epoch_metrics = []
        self.save_hyperparameters()

        self._checkpoint_dir = None
        self._inference_dir = None

    @property
    def checkpoint_dir(self):
        if self._checkpoint_dir is None:
            if dist.is_initialized():
                if dist.get_rank() == 0:
                    checkpoint_dir = [self._exp_cfg.checkpointer.dirpath]
                else:
                    checkpoint_dir = [None]
                dist.broadcast_object_list(checkpoint_dir, src=0)
                checkpoint_dir = checkpoint_dir[0]
            else:
                checkpoint_dir = self._exp_cfg.checkpointer.dirpath
            self._checkpoint_dir = checkpoint_dir
            os.makedirs(self._checkpoint_dir, exist_ok=True)
        return self._checkpoint_dir

    @property
    def inference_dir(self):
        if self._inference_dir is None:
            if dist.is_initialized():
                if dist.get_rank() == 0:
                    inference_dir = [self._exp_cfg.inference_dir]
                else:
                    inference_dir = [None]
                dist.broadcast_object_list(inference_dir, src=0)
                inference_dir = inference_dir[0]
            else:
                inference_dir = self._exp_cfg.inference_dir
            self._inference_dir = inference_dir
            os.makedirs(self._inference_dir, exist_ok=True)
        return self._inference_dir

    def model_step(self, noisy_batch: Any):
        # Model output predictions.
        model_output = self.model(noisy_batch)
        pred_trans_1 = model_output["pred_trans"]  # torch.Size([1, N, 3])
        pred_rotmats_1 = model_output[
            "pred_rotmats"
        ]  # torch.Size([1, N, 3, 3])
        pred_rots_vf = so3_utils.calc_rot_vf(
            noisy_batch["rotmats_t"], pred_rotmats_1
        )  # torch.Size([1, N, 3])
        if torch.any(torch.isnan(pred_rots_vf)):
            raise ValueError("NaN encountered in pred_rots_vf")
        pred_batch = {
            "pred_trans": pred_trans_1,
            "pred_rotmats": pred_rotmats_1,
            "pred_rots_vf": pred_rots_vf,
        }
        if "pred_aatype" in model_output:
            pred_batch["pred_aatype"] = model_output["pred_aatype"]
        out_loss = self.get_loss(noisy_batch, pred_batch, istraining=True)
        return out_loss, [(pred_trans_1, pred_rotmats_1)]

    def _log_scalar(
        self,
        key,
        value,
        on_step=False,
        on_epoch=True,
        prog_bar=True,
        batch_size=None,
        sync_dist=False,
        rank_zero_only=True,
    ):
        if sync_dist and rank_zero_only:
            raise ValueError(
                "Unable to sync dist when rank_zero_only=True"
            )
        self.log(
            key,
            value,
            on_step=on_step,
            on_epoch=on_epoch,
            prog_bar=prog_bar,
            batch_size=batch_size,
            sync_dist=sync_dist,
            rank_zero_only=rank_zero_only,
        )

    def on_test_start(self):
        self.pdb_lists = []

    def test_step(self, batch: Any, batch_idx: int):
        test_metric = dict()
        res_mask = batch["res_mask"]
        for k in [
            "hotspot_mask",
            "aatype_rc",
            "trans_rc",
            "rotmats_rc",
            "rc_node_mask",
        ]:
            batch[k] = batch[k] if k in batch else None
        self.interpolant.set_device(res_mask.device)
        num_batch, num_res = res_mask.shape
        diffuse_mask = batch["diffuse_mask"]
        sample_ids = batch["sample_ids"]

        atom37_traj, out_batch = self.interpolant.sample(
            num_batch,
            num_res,
            self.model,
            trans_1=batch["trans_1"],
            rotmats_1=batch["rotmats_1"],
            diffuse_mask=diffuse_mask,
            hotspot_mask=batch["hotspot_mask"],
            aatype=batch["aatype"],
            chain_idx=batch["chain_idx"],
        )  # [torch.Size([1, N, 37, 3]), ...num_timesteps..., torch.Size([1, N, 37, 3])]

        write_aatype = batch["aatype"] * (1 - batch["diffuse_mask"])

        check_traj_timesteps = [-1]
        for traj_t in check_traj_timesteps:
            samples = atom37_traj[traj_t].numpy()  # (1, N, 37, 3)
            for i in range(num_batch):
                if self._data_cfg.task == "binder":
                    b_factors = (
                        batch["hotspot_mask"][i]
                        + batch["target_interface_mask"][i]
                    )
                elif self._data_cfg.task == "binder_motif":
                    b_factors = (
                        batch["hotspot_mask"][i]
                        + batch["target_interface_mask"][i]
                        + batch["binder_motif_mask"][i]
                    )
                else:
                    b_factors = None

                save_dir = self._exp_cfg.testing_model.save_dir
                os.makedirs(save_dir, exist_ok=True)

                # Write out sample to PDB file
                target_name = batch["pdb_name"][i]
                final_pos = samples[i]
                saved_path = au.write_prot_to_pdb(
                    final_pos,
                    os.path.join(
                        save_dir,
                        f"{target_name}_{sample_ids[i].item()}.pdb",
                    ),
                    no_indexing=True,
                    chain_index=batch["chain_idx"][i],
                    aatype=write_aatype[i],
                    b_factors=b_factors,
                    binder=True,
                    # hotspot,target_interface mask. 1:hotspot, 2:target_interface
                )

                # Write sample motif location
                if (self._data_cfg.task == "binder_motif") and (
                    traj_t == -1
                ):
                    chain_mask = batch["chain_idx"][i] == 1
                    chain_indices = torch.where(chain_mask)[0]
                    cropped_mask = batch["binder_motif_mask"][i][
                        chain_indices
                    ]
                    one_positions = torch.where(cropped_mask == 1)[0]
                    motif_residues = one_positions + 1
                    np.savetxt(
                        os.path.join(
                            save_dir,
                            f"sample{sample_ids[i].item()}_motif_residues.txt",
                        ),
                        motif_residues.cpu().numpy(),
                        fmt="%d",
                        delimiter="\n",
                    )

                if traj_t == -1:
                    mdtraj_metrics = metrics.calc_mdtraj_metrics(
                        saved_path
                    )
                    ca_idx = residue_constants.atom_order["CA"]
                    ca_ca_metrics = metrics.calc_ca_ca_metrics(
                        final_pos[:, ca_idx]
                    )
                    other_metrics = metrics.calc_other_metrics(saved_path)

                    test_metric["pdb_name"] = f"{target_name}_{sample_ids[i].item()}.pdb"
                    test_metric.update(mdtraj_metrics)
                    test_metric.update(ca_ca_metrics)
                    test_metric.update(other_metrics)

                    self.test_epoch_metrics.append(test_metric)


    def on_test_end(self):
        if len(self.test_epoch_metrics) > 0:
            test_metrics_df = pd.DataFrame(self.test_epoch_metrics)
            test_metrics_df.to_csv(
                os.path.join(
                    self._exp_cfg.testing_model.save_dir,
                    f"sample_metrics.csv",
                ),
                float_format="%.4f",
                index=False,
            )
        self.test_epoch_metrics.clear()
