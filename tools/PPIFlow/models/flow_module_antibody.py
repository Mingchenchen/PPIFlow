from typing import Any
import torch
import torch.nn.functional as F
import os
import logging
import torch.distributed as dist
import pandas as pd

from lightning import LightningModule
from omegaconf import OmegaConf

from analysis.antibody_metric import (
    filter_pdb_by_bfactor,
    calc_rmsd,
    detect_backbone_clash,
    get_interface_residues,
    calc_hotspot_coverage,
    calc_mdtraj_metrics,
    get_chain_sasa,
    parse_pdb_and_get_cdr,
)
from analysis import utils as au

from models.flow_model_antibody import FlowModel
from data.interpolant_antibody import Interpolant
from data import so3_utils


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

    def test_step(self, batch: Any, batch_idx: int):

        test_metric = dict()
        res_mask = batch["res_mask"]
        if "motif_groups_mask" not in batch:  # framework_pair_mask
            batch["motif_groups_mask"] = (
                batch["framework_mask"][:, None, :]
                * batch["framework_mask"][:, :, None]
            )

        for k in [
            "hotspot_mask",
            "framework_mask",
            "motif_groups_mask",
            "pos_fixed_mask",
        ]:
            batch[k] = batch[k] if k in batch else None

        self.interpolant.set_device(res_mask.device)
        num_batch, num_res = res_mask.shape
        diffuse_mask = batch["diffuse_mask"]

        save_dir = self._exp_cfg.testing_model.save_dir
        os.makedirs(save_dir, exist_ok=True)

        antigen_chain = self._exp_cfg.testing_model.antigen_chain
        heavy_chain = self._exp_cfg.testing_model.heavy_chain
        light_chain = self._exp_cfg.testing_model.light_chain

        for attempt in range(20):
            atom37_traj, out_batch = self.interpolant.sample_antibody(
                num_batch,
                num_res,
                self.model,
                trans_1=batch["trans_1"],
                rotmats_1=batch["rotmats_1"],
                diffuse_mask=diffuse_mask,
                hotspot_mask=batch["hotspot_mask"],
                aatype=batch["aatype"],
                chain_idx=batch["chain_idx"],
                chain_group_idx=batch["chain_group_idx"],
                pos_fixed_mask=batch["pos_fixed_mask"],
                binder_motif_mask=batch["binder_motif_mask"],
            )

            write_aatype = batch["aatype"] * (
                1 - batch["pos_fixed_mask"] + batch["binder_motif_mask"]
            )
            samples = atom37_traj[-1].numpy()

            for i in range(num_batch):
                pdb_name = batch["pdb_name"][i]
                pdb_path = os.path.join(save_dir, f"{pdb_name}.pdb")

                b_factors = (
                    batch["hotspot_mask"][i]
                    + 2 * batch["diffuse_mask"][i]
                    + 2 * batch["binder_motif_mask"][i]
                )
                final_pos = samples[i]
                au.write_prot_to_pdb(
                    final_pos,
                    pdb_path,
                    no_indexing=True,
                    chain_index=batch["chain_idx"][i],
                    aatype=write_aatype[i],
                    b_factors=b_factors,
                )
            reference_pdb = self._exp_cfg.framework_pdb
            output_pdb = os.path.join(
                save_dir, f"sample{batch_idx}_filtered.pdb"
            )

            filter_pdb_by_bfactor(pdb_path, output_pdb)
            rmsd = calc_rmsd(output_pdb, reference_pdb)
            has_clash = detect_backbone_clash(pdb_path)

            if rmsd < 1 and not has_clash:
                test_metric["sample"] = os.path.basename(pdb_path)
                test_metric["pdb_path"] = pdb_path

                (
                    interface_residues,
                    cdr_interface_residues,
                    cdr_interface_ratio,
                ) = get_interface_residues(pdb_path, 10, heavy_chain, light_chain)
                test_metric["interface_residues"] = interface_residues
                test_metric["cdr_interface_residues"] = (
                    cdr_interface_residues
                )
                test_metric["cdr_interface_ratio"] = cdr_interface_ratio

                hotspot_coverage, hotspot_list = calc_hotspot_coverage(
                    pdb_path, antigen_chain, heavy_chain, light_chain
                )
                test_metric["hotspot_coverage"] = hotspot_coverage
                test_metric["coverage_hotspot_list"] = hotspot_list

                cdr_dict = parse_pdb_and_get_cdr(pdb_path, heavy_chain, light_chain)
                test_metric.update(cdr_dict)

                mdtraj_metrics_all_chains = calc_mdtraj_metrics(pdb_path)
                # 处理多链mdtraj指标：为每条链的指标添加链ID前缀
                mdtraj_metrics = {}
                if isinstance(mdtraj_metrics_all_chains, dict):
                    for (
                        chain_id,
                        chain_metrics,
                    ) in mdtraj_metrics_all_chains.items():
                        for (
                            metric_name,
                            metric_value,
                        ) in chain_metrics.items():
                            mdtraj_metrics[
                                f"chain_{chain_id}_{metric_name}"
                            ] = metric_value
                test_metric.update(mdtraj_metrics)

                sasa = get_chain_sasa(pdb_path, antigen_chain, heavy_chain, light_chain)
                test_metric["dsasa"] = sasa

                test_metric["rmsd_framework"] = rmsd
                test_metric["has_clash"] = has_clash

                self.test_epoch_metrics.append(test_metric)

                if os.path.exists(output_pdb):
                    os.remove(output_pdb)
                break
            else:
                print(
                    f"Attempt {attempt}: RMSD={rmsd:.3f}, Clash={has_clash}"
                )
                if os.path.exists(pdb_path):
                    os.remove(pdb_path)
                if os.path.exists(output_pdb):
                    os.remove(output_pdb)

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
