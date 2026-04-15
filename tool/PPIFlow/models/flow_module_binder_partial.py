from typing import Any
import torch
import os
import numpy as np
import pandas as pd
import logging
import torch.distributed as dist
from pytorch_lightning import LightningModule
from analysis import metrics 
from analysis import utils as au
from models.flow_model_binder import FlowModelPart
from models import utils as mu
from data.interpolant_binder_partial import Interpolant 
from data import utils as du
from data import all_atom
from data import so3_utils
from data import residue_constants
from experiments import utils as eu
from omegaconf import OmegaConf

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logging.getLogger('urllib3.connectionpool').setLevel(logging.ERROR)

class FlowModule(LightningModule):

    def __init__(self, cfg):
        super().__init__()
        self._print_logger = logging.getLogger(__name__)
        self._exp_cfg = cfg.experiment
        self._model_cfg = cfg.model
        self._data_cfg = cfg.data
        self._interpolant_cfg = cfg.interpolant
        self.task = self._data_cfg.task
        # Set-up vector field prediction model
        OmegaConf.set_struct(cfg, False)
        cfg.model["task"] = self._data_cfg.task
        self.model = FlowModelPart(cfg.model)
        # self.model._set_static_graph()
        # self.model = self.model.half()

        # Set-up interpolant
        self.interpolant = Interpolant(cfg.interpolant)

        self.validation_epoch_metrics = []
        self.validation_epoch_samples = []
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
        pred_trans_1 = model_output['pred_trans']#torch.Size([1, 596, 3])
        pred_rotmats_1 = model_output['pred_rotmats']#torch.Size([1, 596, 3, 3])
        pred_rots_vf = so3_utils.calc_rot_vf(noisy_batch['rotmats_t'], pred_rotmats_1)#torch.Size([1, 596, 3])
        if torch.any(torch.isnan(pred_rots_vf)):
            raise ValueError('NaN encountered in pred_rots_vf')
        pred_batch = {'pred_trans': pred_trans_1, 'pred_rotmats': pred_rotmats_1, 'pred_rots_vf': pred_rots_vf}
        if 'pred_aatype' in model_output:
            pred_batch['pred_aatype'] = model_output['pred_aatype']
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
            sync_dist=False, # modify
            rank_zero_only=True
        ):
        if sync_dist and rank_zero_only:
            raise ValueError('Unable to sync dist when rank_zero_only=True')
        self.log(
            key,
            value,
            on_step=on_step,
            on_epoch=on_epoch,
            prog_bar=prog_bar,
            batch_size=batch_size,
            sync_dist=sync_dist,
            rank_zero_only=rank_zero_only
        )


    def configure_optimizers(self):
        return torch.optim.AdamW(
            params=self.model.parameters(),
            **self._exp_cfg.optimizer
        )

    def predict_step(self, batch, batch_idx):
        del batch_idx # Unused
        if torch.cuda.is_available():
            device = f'cuda:{torch.cuda.current_device()}'
        else:
            device = 'cpu'#debug mode
        interpolant = Interpolant(self._infer_cfg.interpolant)
        interpolant.set_device(device)

        sample_ids = batch['sample_id'].squeeze().tolist()
        sample_ids = [sample_ids] if isinstance(sample_ids, int) else sample_ids
        num_batch = len(sample_ids)
        if 'diffuse_mask' in batch: # motif-scaffolding
            batch['hotspot_mask'] = torch.zeros_like(batch['aatype'], device=batch['aatype'].device, dtype=torch.int)
            target = batch['target'][0]
            trans_1 = batch['trans_1']
            rotmats_1 = batch['rotmats_1']
            diffuse_mask = batch['diffuse_mask']
            true_bb_pos = all_atom.atom37_from_trans_rot(trans_1, rotmats_1, 1 - diffuse_mask)
            true_bb_pos = true_bb_pos[..., :3, :].reshape(-1, 3).cpu().numpy()
            _, sample_length, _ = trans_1.shape
            sample_dirs = os.path.join(self.inference_dir, target)
        else: # unconditional
            batch['diffuse_mask'] = torch.ones_like(batch['aatype'], device=batch['aatype'].device, dtype=torch.int)
            batch['hotspot_mask'] = torch.zeros_like(batch['aatype'], device=batch['aatype'].device, dtype=torch.int)
            sample_length = batch['num_res'].item()
            true_bb_pos = None
            sample_dirs = os.path.join(
                self.inference_dir, f'length_{sample_length}')
            trans_1 = rotmats_1 = None

        # Sample batch
        atom37_traj, out_batch = interpolant.sample(
            num_batch, sample_length, self.model,
            trans_1=trans_1, rotmats_1=rotmats_1, diffuse_mask=batch['diffuse_mask'],
            aatype=batch['aatype'], hotspot_mask=batch['hotspot_mask'], chain_idx=batch['chain_idx'],
        )

        batch['rotmats_t'] = out_batch['rotmats_t']
        pred_aatype = out_batch['pred_aatype']
        if self._model_cfg.seq_decoder.use:
            write_aatype = torch.argmax(pred_aatype, dim=2)
            write_aatype = write_aatype*batch['diffuse_mask'] + batch['aatype']*(1-batch['diffuse_mask'])
        else:
            write_aatype = batch['aatype']*(1-batch['diffuse_mask'])
            ###motif redesign aatype mask
            redesign_aatype_mask = (write_aatype==21)*(1-batch['diffuse_mask'])
            write_aatype = write_aatype*(write_aatype!=21)*(1-batch['diffuse_mask'])
        samples = atom37_traj[-1].numpy()  # (1, 248, 37, 3)
        os.makedirs(sample_dirs, exist_ok=True)
        for i in range(num_batch):
            # Write out sample to PDB file
            final_pos = samples[i]
            # if "binder" in self._data_cfg.task:
            #     b_factors = batch['hotspot_mask'][i]+batch['target_interface_mask'][i]
            if self._infer_cfg.task=="scaffolding":
                b_factors = (1-batch['diffuse_mask'][i])+redesign_aatype_mask[i]
            else:
                b_factors = None
            saved_path = au.write_prot_to_pdb(
                final_pos,
                os.path.join(sample_dirs, f"sample{sample_ids[i]}.pdb"),
                no_indexing=True,
                aatype=write_aatype[i],
                b_factors=b_factors
                # hotspot,target_interface mask. 1:hotspot, 2:target_interface
            )

    def on_test_start(self):
        self.pdb_lists = []

    def test_step(self, batch: Any, batch_idx: int):
        # print(f"validation batch_num:{batch['res_mask'].shape[0]}")
        res_mask = batch['res_mask']
        for k in ['hotspot_mask', 'aatype_rc', 'trans_rc', 'rotmats_rc', 'rc_node_mask', 'motif_groups_mask']:
            batch[k] = batch[k] if k in batch else None
        self.interpolant.set_device(res_mask.device)
        num_batch, num_res = res_mask.shape
        diffuse_mask = batch['diffuse_mask']
        sample_ids = batch['sample_ids']

        # set initial noise offset (at hotspots center)    # *
        if self._interpolant_cfg.starting_at_hotspot_center:
            hotspot_mask = batch['hotspot_mask'][0].bool()   # assume batch size is 1 at inference
            hotspot_trans = batch['trans_1'][0][hotspot_mask]
            hotspot_center = hotspot_trans.mean(dim=0)
        else:
            hotspot_center = None
        if self.task == 'binder_motif_partial': #partial
            batch = self.interpolant.corrupt_batch(batch)
            batch['so3_t'] = batch['so3_t'][0].item()
            num_timesteps = int(100*(1-batch['so3_t']))
        else:
            batch['trans_t'] = None
            batch['rotmats_t'] = None
            batch['so3_t'] = None
            num_timesteps = None

        atom37_traj, out_batch = self.interpolant.sample(
            num_batch,
            num_res,
            self.model,
            trans_1=batch['trans_1'],
            rotmats_1=batch['rotmats_1'],
            init_binder_offset=hotspot_center,     # *
            diffuse_mask=diffuse_mask,
            hotspot_mask=batch['hotspot_mask'],
            aatype=batch['aatype'],
            aatype_rc=batch['aatype_rc'],
            trans_rc=batch['trans_rc'],
            rotmats_rc=batch['rotmats_rc'],
            rc_node_mask=batch['rc_node_mask'],
            chain_idx=batch['chain_idx'],
            trans_0=batch['trans_t'],
            rotmats_0=batch['rotmats_t'],
            min_t=batch['so3_t'],
            num_timesteps=num_timesteps,
            motif_groups_mask=batch['motif_groups_mask']
        )#[torch.Size([1, 248, 37, 3]), ...num_timesteps..., torch.Size([1, 248, 37, 3])]

        pred_aatype = out_batch['pred_aatype']
        if self._model_cfg.seq_decoder.use:
            write_aatype = torch.argmax(pred_aatype, dim=2)
            write_aatype = write_aatype*batch['diffuse_mask'] + batch['aatype']*(1-batch['diffuse_mask'])
        else:
            write_aatype = batch['aatype']*(1-batch['diffuse_mask'])


        check_traj_timesteps = [-1]
        # for debug intermediate timesteps
        # check_traj_timesteps = list(range(self._interpolant_cfg.sampling.num_timesteps))+[-1]
        for traj_t in check_traj_timesteps:
            samples = atom37_traj[traj_t].numpy()#(1, 248, 37, 3)
            batch_metrics = []
            for i in range(num_batch):
                if self._data_cfg.task == 'binder':
                    b_factors = batch['hotspot_mask'][i] + batch['target_interface_mask'][i]
                elif self._data_cfg.task == 'binder_motif':
                    b_factors = batch['hotspot_mask'][i] + batch['target_interface_mask'][i] + \
                                batch['binder_motif_mask'][i]
                elif self._data_cfg.task == "inpainting":
                    b_factors = batch['diffuse_mask'][i]
                elif self._data_cfg.task == "binder_motif_partial":
                    b_factors = batch['hotspot_mask'][i] + batch['target_interface_mask'][i] + \
                                batch['binder_motif_mask'][i]
                else:
                    b_factors = None

                # save_dir = os.path.join(self._exp_cfg.testing_model.save_dir, f"sample_{batch['pdb_name'][i]}")
                save_dir = self._exp_cfg.testing_model.save_dir
                os.makedirs(save_dir, exist_ok=True)
                # Write out sample to PDB file
                final_pos = samples[i]
                saved_path = au.write_prot_to_pdb(
                    final_pos,
                    os.path.join(save_dir, f"sample{sample_ids[i].item()}_{traj_t}.pdb"),
                    no_indexing=True,
                    chain_index=batch['chain_idx'][i],
                    aatype=write_aatype[i],
                    b_factors=b_factors,
                    binder=True,
                )
