from collections import defaultdict
import torch
from data import so3_utils
from data import utils as du
from scipy.spatial.transform import Rotation
from data import all_atom
import copy


def _centered_gaussian(num_batch, num_res, device):
    noise = torch.randn(num_batch, num_res, 3, device=device)
    return noise - torch.mean(noise, dim=-2, keepdims=True)

def _uniform_so3(num_batch, num_res, device):
    return torch.tensor(
        Rotation.random(num_batch*num_res).as_matrix(),
        device=device,
        dtype=torch.float32,
    ).reshape(num_batch, num_res, 3, 3)

def _trans_diffuse_mask(trans_t, trans_1, diffuse_mask):
    return trans_t * diffuse_mask[..., None] + trans_1 * (1 - diffuse_mask[..., None])

def _rots_diffuse_mask(rotmats_t, rotmats_1, diffuse_mask):
    return (
        rotmats_t * diffuse_mask[..., None, None]
        + rotmats_1 * (1 - diffuse_mask[..., None, None])
    )


class Interpolant:

    def __init__(self, cfg):
        self._cfg = cfg
        self._rots_cfg = cfg.rots
        self._trans_cfg = cfg.trans
        self._sample_cfg = cfg.sampling
        self._igso3 = None

    @property
    def igso3(self):
        if self._igso3 is None:
            sigma_grid = torch.linspace(0.1, 1.5, 1000)
            self._igso3 = so3_utils.SampleIGSO3(
                1000, sigma_grid, cache_dir='.cache')
        return self._igso3

    def set_device(self, device):
        self._device = device

    def sample_t(self, num_batch):
        t = torch.rand(num_batch, device=self._device)
        return t * (1 - 2*self._cfg.min_t) + self._cfg.min_t

    def sample_t_beta(self, num_batch, alpha=3.0, beta=5.0, min_t=0.01):
        t = torch.distributions.Beta(alpha, beta).sample((num_batch,))
        t = t.to(self._device)
        return t * (1 - 2 * min_t) + min_t

    def sample_t_piecewise(self, num_batch, cur_epoch):
        intervals = torch.tensor(self._cfg.piecewise_t_method.intervals, dtype=torch.float32)
        probs = torch.tensor(self._cfg.piecewise_t_method.probs, dtype=torch.float32)
        start_epochs = self._cfg.piecewise_t_method.start_epochs
        assert cur_epoch >= start_epochs[0]
        if cur_epoch < start_epochs[1]:
            # Choose intervals based on probabilities
            chosen_intervals = torch.multinomial(torch.tensor(probs[0]), num_batch, replacement=True)
        else:
            chosen_intervals = torch.multinomial(torch.tensor(probs[1]), num_batch, replacement=True)

        # Sample uniformly within selected intervals
        samples = torch.rand(num_batch)
        t = intervals[chosen_intervals, 0] + samples * (intervals[chosen_intervals, 1] - intervals[chosen_intervals, 0])
        t = t.to(self._device)
        return t * (1 - 2 * self._cfg.min_t) + self._cfg.min_t

    def _corrupt_trans(self, trans_1, t, res_mask, diffuse_mask, trans_scale=1):
        trans_nm_0 = _centered_gaussian(*res_mask.shape, self._device)
        trans_0 = trans_nm_0 * trans_scale
        trans_t = (1 - t[..., None]) * trans_0 + t[..., None] * trans_1
        trans_t = _trans_diffuse_mask(trans_t, trans_1, diffuse_mask)
        return trans_t * res_mask[..., None]
    
    def _corrupt_rotmats(self, rotmats_1, t, res_mask, diffuse_mask):
        num_batch, num_res = res_mask.shape
        noisy_rotmats = self.igso3.sample(
            torch.tensor([1.5]),
            num_batch*num_res
        ).to(self._device)
        noisy_rotmats = noisy_rotmats.reshape(num_batch, num_res, 3, 3)
        rotmats_0 = torch.einsum(
            "...ij,...jk->...ik", rotmats_1, noisy_rotmats)
        rotmats_t = so3_utils.geodesic_t(t[..., None], rotmats_1, rotmats_0)
        identity = torch.eye(3, device=self._device)
        rotmats_t = (
            rotmats_t * res_mask[..., None, None]
            + identity[None, None] * (1 - res_mask[..., None, None])
        )
        return _rots_diffuse_mask(rotmats_t, rotmats_1, diffuse_mask)

    def corrupt_batch(self, batch, cur_epoch):
        noisy_batch = copy.deepcopy(batch)

        # [B, N, 3]
        trans_1 = batch['trans_1']

        # [B, N, 3, 3]
        rotmats_1 = batch['rotmats_1']

        # [B, N]
        res_mask = batch['res_mask']
        diffuse_mask = batch['diffuse_mask']
        num_batch, _ = diffuse_mask.shape

        # [B, 1]
        if self._cfg.t_method == 'uniform':
            t = self.sample_t(num_batch)[:, None]
        elif self._cfg.t_method == 'beta':
            t = self.sample_t_beta(num_batch, alpha=self._cfg.alpha_t)[:, None]
        elif self._cfg.t_method == 'piecewise':
            if cur_epoch >= self._cfg.piecewise_t_method.start_epochs[0]:
                t = self.sample_t_piecewise(num_batch, cur_epoch=cur_epoch)[:, None]
            else:
                t = self.sample_t(num_batch)[:, None]
        else:
            raise ValueError(f'Unknown noise level sampling method: {self._cfg.t_method}')
        so3_t = t
        r3_t = t
        noisy_batch['so3_t'] = so3_t
        noisy_batch['r3_t'] = r3_t

        # Apply corruptions
        if self._trans_cfg.corrupt:
            trans_t = self._corrupt_trans(
                trans_1, r3_t, res_mask, diffuse_mask, trans_scale=self._trans_cfg.var_scale)
        else:
            trans_t = trans_1
        if torch.any(torch.isnan(trans_t)):
            raise ValueError('NaN in trans_t during corruption')
        noisy_batch['trans_t'] = trans_t

        if self._rots_cfg.corrupt:
            rotmats_t = self._corrupt_rotmats(
                rotmats_1, so3_t, res_mask, diffuse_mask)
        else:
            rotmats_t = rotmats_1
        if torch.any(torch.isnan(rotmats_t)):
            raise ValueError('NaN in rotmats_t during corruption')
        noisy_batch['rotmats_t'] = rotmats_t
        return noisy_batch
    
    def rot_sample_kappa(self, t):
        if self._rots_cfg.sample_schedule == 'exp':
            return 1 - torch.exp(-t*self._rots_cfg.exp_rate)
        elif self._rots_cfg.sample_schedule == 'linear':
            return t
        else:
            raise ValueError(
                f'Invalid schedule: {self._rots_cfg.sample_schedule}')

    def _trans_vector_field(self, t, trans_1, trans_t):
        return (trans_1 - trans_t) / (1 - t)

    def _trans_euler_step(self, d_t, t, trans_1, trans_t):
        assert d_t > 0
        trans_vf = self._trans_vector_field(t, trans_1, trans_t)
        return trans_t + trans_vf * d_t

    def _rots_euler_step(self, d_t, t, rotmats_1, rotmats_t):
        if self._rots_cfg.sample_schedule == 'linear':
            scaling = 1 / (1 - t)
        elif self._rots_cfg.sample_schedule == 'exp':
            scaling = self._rots_cfg.exp_rate
        else:
            raise ValueError(
                f'Unknown sample schedule {self._rots_cfg.sample_schedule}')
        return so3_utils.geodesic_t(
            scaling * d_t, rotmats_1, rotmats_t)

    def sample(
            self,
            num_batch,
            num_res,
            model,
            num_timesteps=None,
            trans_0=None,
            rotmats_0=None,
            trans_1=None,
            rotmats_1=None,
            diffuse_mask=None,
            hotspot_mask=None,
            chain_idx=None,
            res_idx=None,
            aatype=None,
        ):
        res_mask = torch.ones(num_batch, num_res, device=self._device)

        # Set-up initial prior samples
        if trans_0 is None:
            trans_0 = _centered_gaussian(
                num_batch, num_res, self._device) * self._trans_cfg.var_scale
        if rotmats_0 is None:
            rotmats_0 = _uniform_so3(num_batch, num_res, self._device)
        if res_idx is None:
            res_idx = torch.arange(
                num_res,
                device=self._device,
                dtype=torch.float32)[None].repeat(num_batch, 1)
        batch = {
            'res_mask': res_mask,
            'diffuse_mask': diffuse_mask,
            'res_idx': res_idx,
            'chain_idx': chain_idx,
        }

        if hotspot_mask is not None:
            batch['hotspot_mask'] = hotspot_mask
        if aatype is not None:
            batch['aatype'] = aatype

        diffuse_mask = diffuse_mask.expand(num_batch, -1)
        batch['diffuse_mask'] = diffuse_mask
        rotmats_0 = _rots_diffuse_mask(rotmats_0, rotmats_1, diffuse_mask)
        trans_0 = _trans_diffuse_mask(trans_0, trans_1, diffuse_mask)
        if torch.isnan(trans_0).any():
            raise ValueError('NaN detected in trans_0')

        # Set-up time
        if num_timesteps is None:
            num_timesteps = self._sample_cfg.num_timesteps
        ts = torch.linspace(self._cfg.min_t, 1.0, num_timesteps)
        t_1 = ts[0]

        prot_traj = [(trans_0, rotmats_0)]
        clean_traj = []
        for i, t_2 in enumerate(ts[1:]):
            # Run model.
            trans_t_1, rotmats_t_1 = prot_traj[-1]
            if self._trans_cfg.corrupt:
                batch['trans_t'] = trans_t_1
            else:
                if trans_1 is None:
                    raise ValueError('Must provide trans_1 if not corrupting.')
                batch['trans_t'] = trans_1
            if self._rots_cfg.corrupt:
                batch['rotmats_t'] = rotmats_t_1
            else:
                if rotmats_1 is None:
                    raise ValueError('Must provide rotmats_1 if not corrupting.')
                batch['rotmats_t'] = rotmats_1
            batch['t'] = torch.ones((num_batch, 1), device=self._device) * t_1
            batch['so3_t'] = batch['t']
            batch['r3_t'] = batch['t']
            d_t = t_2 - t_1

            with torch.no_grad():
                model_out = model(batch)

            # Process model output.
            pred_trans_1 = model_out['pred_trans']
            pred_rotmats_1 = model_out['pred_rotmats']
            clean_traj.append(
                (pred_trans_1.detach().cpu(), pred_rotmats_1.detach().cpu())
            )
            if self._cfg.self_condition:
                batch['trans_sc'] = (
                    pred_trans_1 * diffuse_mask[..., None]
                    + trans_1 * (1 - diffuse_mask[..., None])
                )

            # Take reverse step
            trans_t_2 = self._trans_euler_step(
                d_t, t_1, pred_trans_1, trans_t_1)
            rotmats_t_2 = self._rots_euler_step(
                d_t, t_1, pred_rotmats_1, rotmats_t_1)

            trans_t_2 = _trans_diffuse_mask(trans_t_2, trans_1, diffuse_mask)
            rotmats_t_2 = _rots_diffuse_mask(rotmats_t_2, rotmats_1, diffuse_mask)

            prot_traj.append((trans_t_2, rotmats_t_2))
            t_1 = t_2

        # We only integrated to min_t, so need to make a final step
        t_1 = ts[-1]
        trans_t_1, rotmats_t_1 = prot_traj[-1]
        if self._trans_cfg.corrupt:
            batch['trans_t'] = trans_t_1
        else:
            if trans_1 is None:
                raise ValueError('Must provide trans_1 if not corrupting.')
            batch['trans_t'] = trans_1
        if self._rots_cfg.corrupt:
            batch['rotmats_t'] = rotmats_t_1
        else:
            if rotmats_1 is None:
                raise ValueError('Must provide rotmats_1 if not corrupting.')
            batch['rotmats_t'] = rotmats_1
        batch['t'] = torch.ones((num_batch, 1), device=self._device) * t_1
        with torch.no_grad():
            model_out = model(batch)
        pred_trans_1 = model_out['pred_trans']
        pred_rotmats_1 = model_out['pred_rotmats']
        clean_traj.append(
            (pred_trans_1.detach().cpu(), pred_rotmats_1.detach().cpu())
        )
        prot_traj.append((pred_trans_1, pred_rotmats_1))

        # Convert trajectories to atom37.
        atom37_traj = all_atom.transrot_to_atom37(prot_traj, res_mask)
        # clean_atom37_traj = all_atom.transrot_to_atom37(clean_traj, res_mask)

        out_batch = {'pred_trans': pred_trans_1, 'pred_rotmats': pred_rotmats_1, 'rotmats_t': batch['rotmats_t'], 'pred_aatype': model_out['pred_aatype']}

        return atom37_traj, out_batch    # *

