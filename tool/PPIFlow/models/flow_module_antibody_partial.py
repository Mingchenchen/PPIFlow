from typing import Any
import torch
import os
import logging
import torch.distributed as dist
import copy
from lightning import LightningModule


from analysis import pdb_clash
from analysis import utils as au
from models.flow_model_antibody import FlowModel
from data.interpolant_antibody_partial import Interpolant
from data import so3_utils
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
        pred_trans_1 = model_output["pred_trans"]  # torch.Size([1, 596, 3])
        pred_rotmats_1 = model_output["pred_rotmats"]  # torch.Size([1, 596, 3, 3])
        pred_rots_vf = so3_utils.calc_rot_vf(noisy_batch["rotmats_t"], pred_rotmats_1)  # torch.Size([1, 596, 3])
        if torch.any(torch.isnan(pred_rots_vf)):
            raise ValueError("NaN encountered in pred_rots_vf")
        pred_batch = {"pred_trans": pred_trans_1, "pred_rotmats": pred_rotmats_1, "pred_rots_vf": pred_rots_vf}
        if "pred_aatype" in model_output:
            pred_batch["pred_aatype"] = model_output["pred_aatype"]
        out_loss = self.get_loss(noisy_batch, pred_batch, istraining=True)
        return out_loss, [(pred_trans_1, pred_rotmats_1)]

    def test_step(self, batch: Any, batch_idx: int):

        test_metric = dict()
        res_mask = batch["res_mask"]
        if "motif_groups_mask" not in batch:  # framework_pair_mask
            batch["motif_groups_mask"] = batch["framework_mask"][:, None, :] * batch["framework_mask"][:, :, None]

        for k in ["hotspot_mask", "framework_mask", "motif_groups_mask", "pos_fixed_mask"]:
            batch[k] = batch[k] if k in batch else None

        self.interpolant.set_device(res_mask.device)
        num_batch, num_res = res_mask.shape
        diffuse_mask = batch["diffuse_mask"]

        # set initial noise offset (at hotspots center)    # *
        if self._interpolant_cfg.starting_at_hotspot_center:
            hotspot_mask = batch["hotspot_mask"][0].bool()  # assume batch size is 1 at inference
            hotspot_trans = batch["trans_1"][0][hotspot_mask]
            hotspot_center = hotspot_trans.mean(dim=0)
        else:
            hotspot_center = None
        # Output path configuration
        save_dir = self._exp_cfg.testing_model.save_dir
        os.makedirs(save_dir, exist_ok=True)

        # Initialize batch for partial generation
        origin_batch = copy.deepcopy(batch)
        attempt_num = self._exp_cfg.retry_Limit
        for attempt in range(attempt_num):
            # Partial: add noise to start_t
            batch = self.interpolant.corrupt_batch(origin_batch, 0)
            batch["so3_t"] = batch["so3_t"][0].item()
            num_timesteps = int(self._interpolant_cfg.sampling.num_timesteps * (1 - batch["so3_t"]))

            atom37_traj, out_batch = self.interpolant.sample_antibody(
                num_batch,
                num_res,
                self.model,
                trans_1=batch["trans_1"],
                rotmats_1=batch["rotmats_1"],
                init_binder_offset=hotspot_center,
                diffuse_mask=diffuse_mask,
                hotspot_mask=batch["hotspot_mask"],
                aatype=batch["aatype"],
                chain_idx=batch["chain_idx"],
                chain_group_idx=batch["chain_group_idx"],
                pos_fixed_mask=batch["pos_fixed_mask"],
                binder_motif_mask=batch["binder_motif_mask"],
                trans_0=batch["trans_t"],
                rotmats_0=batch["rotmats_t"],
                num_timesteps=num_timesteps,
            )

            fix_sequence_mask = batch["fix_sequence_mask"]
            write_aatype = batch["aatype"] * fix_sequence_mask + 0 * (1 - fix_sequence_mask)

            samples = atom37_traj[-1].numpy()
            pdb_path = os.path.join(save_dir, f"sample{batch_idx}.pdb")


            for i in range(num_batch):
                # b_factors = 2 * batch["hotspot_mask"][i] + (1 - batch["diffuse_mask"][i]) + (batch["chain_group_idx"][i] - 1)
                b_factors = batch["hotspot_mask"][i] + 4 * batch["binder_motif_mask"][i]
                # In antigen: 1=hotspot, 0=others
                # In antibody: 1=structurally fixed, 0=others
                final_pos = samples[i]
                au.write_prot_to_pdb(
                    final_pos, pdb_path, no_indexing=True, chain_index=batch["chain_idx"][i], aatype=write_aatype[i], b_factors=b_factors
                )

            total_breaks, _ = pdb_clash.detect_breaks_in_structure(pdb_path, max_peptide_bond_length=1.6)
            clash = pdb_clash.detect_backbone_clash(pdb_path)

            if (total_breaks == 0) and not clash:
                break
            else:
                print(f"Attempt {attempt}: Break={total_breaks}, Clash={clash}")
