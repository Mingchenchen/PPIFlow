from typing import Any
import torch
import os
import logging
import torch.distributed as dist
from pytorch_lightning import LightningModule
from omegaconf import OmegaConf

from analysis import utils as au
from models.flow_model_monomer import FlowModel
from data.interpolant_monomer import Interpolant 
from data import all_atom
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
        pred_trans_1 = model_output['pred_trans']
        pred_rotmats_1 = model_output['pred_rotmats']
        if torch.any(torch.isnan(pred_rotmats_1)):
            print('pred_rotmats_1',noisy_batch['pdb_name'])
            raise ValueError('NaN encountered in pred_rotmats_1')
        pred_rots_vf = so3_utils.calc_rot_vf(noisy_batch['rotmats_t'], pred_rotmats_1)
        pred_batch = {'pred_trans': pred_trans_1, 'pred_rotmats': pred_rotmats_1, 'pred_rots_vf': pred_rots_vf}
        if 'pred_aatype' in model_output:
            pred_batch['pred_aatype'] = model_output['pred_aatype']
        out_loss = self.get_loss(noisy_batch, pred_batch, istraining=True)
        return out_loss, [(pred_trans_1, pred_rotmats_1)]

    def configure_optimizers(self):
        return torch.optim.AdamW(
            params=self.model.parameters(),
            **self._exp_cfg.optimizer
        )

    def predict_step(self, batch, batch_idx):
        del batch_idx
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
            sample_length = batch['num_res'].item()
            true_bb_pos = None
            sample_dirs = os.path.join(
                self.inference_dir, f'length_{sample_length}')
            trans_1 = rotmats_1 = None

        # Sample batch
        atom37_traj, out_batch = interpolant.sample(
            num_batch, sample_length, self.model,
            trans_1=trans_1, rotmats_1=rotmats_1, diffuse_mask=batch['diffuse_mask'],
            aatype=batch['aatype'], hotspot_mask=None, chain_idx=batch['chain_idx']
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
        samples = atom37_traj[-1].numpy()  # (1, N, 37, 3)
        os.makedirs(sample_dirs, exist_ok=True)
        for i in range(num_batch):
            # Write out sample to PDB file
            final_pos = samples[i]
            if "binder" in self._data_cfg.task:
                b_factors = batch['hotspot_mask'][i]+batch['target_interface_mask'][i]
            elif self._data_cfg.task=="inpainting":
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
