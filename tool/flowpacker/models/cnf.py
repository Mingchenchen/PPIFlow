import torch
import torch.nn as nn
from torch import Tensor
from torch.distributions import Uniform
import math
from dataset_cluster import get_edge_features
from utils.sidechain_utils import Idealizer
from utils.so2_utils import exp
from torch_cluster import radius_graph, knn_graph
import numpy as np
import copy
from loss import CFMLoss
from utils.so2_utils import log, exp
import random
from torch.autograd.functional import jvp

class CNF(nn.Module):
    def __init__(self, model, config, eps=1e-3, coeff=8.0, stepsize=100, method='euler', mode='vf'):
        super().__init__()
        self.model = model
        self.config = config
        self.eps = eps
        self.coeff = coeff
        self.stepsize = stepsize
        self.method = method
        self.mode = mode
        self.idealizer = Idealizer(use_native_bb_coords=True)

    def forward(self, *args, **kwargs) -> Tensor:
        return self.model(*args, **kwargs)

    def get_vf(self, t, xt, batch):
        dihedral_sin_cos = torch.cat([batch.bb_dihedral.sin(), batch.bb_dihedral.cos()], dim=-1)
        node_feat = torch.cat([batch.aa_onehot, dihedral_sin_cos], dim=-1)

        x_scaled = (xt - math.pi) * batch.chi_mask
        noised_crds = self.idealizer(batch.aa_num, batch.pos[:, :4], x_scaled)  # rescale back to [-pi,pi]
        noised_crds = noised_crds * batch.atom_mask.unsqueeze(-1)

        # edge index
        ca = batch.pos[:, 1]
        if self.config.model.use_virtual_cb:
            b = batch.pos[:, 1, :] - batch.pos[:, 0, :]
            c = batch.pos[:, 2, :] - batch.pos[:, 1, :]
            a = torch.cross(b, c, dim=-1)
            node_crds = -0.58273431 * a + 0.56802827 * b - 0.54067466 * c + batch.pos[:, 1, :]
        else:
            node_crds = ca

        if self.config.data.edge_type == 'radius':
            edge_index = radius_graph(node_crds, r=self.config.data.max_radius,
                                      max_num_neighbors=self.config.data.max_neighbors, batch=batch.batch)
        elif self.config.data.edge_type == 'knn':
            edge_index = knn_graph(node_crds, k=self.config.data.max_neighbors, batch=batch.batch)
        else:
            raise NotImplementedError('wrong edge type')

        edge_feat = get_edge_features(noised_crds, edge_index, batch.atom_mask,
                                      all_atoms=self.config.model.add_dist_to_edge)

        pred_chi = self(t, xt, node_feat, node_crds, edge_index, edge_feat, batch.chi_mask, batch.batch, atom_mask=batch.atom_mask,
                        self_cond=None)

        return pred_chi


    def decode(self, batch, return_traj=False, inpaint='') -> Tensor:
        ca_crds, chi, chi_alt, batch_id, chi_mask, atom_mask = batch.pos[:,1], batch.chi, batch.chi_alt, batch.batch, batch.chi_mask, batch.atom_mask
        chi = (chi + math.pi) * chi_mask
        x0 = torch.rand_like(chi) * 2*math.pi # / 10 # uniform prior [0,2pi]
        x0[batch.chi_alt_mask==1] = x0[batch.chi_alt_mask==1] % math.pi

        x0 = x0 * chi_mask
        t_range = torch.linspace(0, 1.0-self.eps, self.stepsize).to(chi)

        traj = [x0]
        t1 = t_range[0]

        if inpaint != '':
            cond_mask = torch.zeros(x0.shape[0], 4, device=x0.device)
            cond_str = inpaint.split('/')
            for s in cond_str:
                if '_' not in s:
                    assert s in batch.chain_id[0], 'wrong chain id found'
                    chain_mask = batch.chain_id[0] == s
                    cond_mask[chain_mask] = 1
                else:
                    chain, res = s.split('_')
                    assert chain in batch.chain_id[0], 'wrong chain id found'
                    unique, counts = np.unique(batch.chain_id[0], return_counts=True)
                    index = np.where(unique == chain)[0].item()
                    add_res_counter = counts[:index].sum()
                    if '-' in res:
                        start, end = res.split('-')
                        start, end = int(start) + add_res_counter - 1, int(end) + add_res_counter
                        cond_mask[start:end] = 1
                    else:
                        cond_mask[int(res)+add_res_counter-1] = 1
        else:
            cond_mask = torch.ones(x0.shape[0], 4, device=x0.device)

        for t2 in t_range[1:]:
            dt = t2 - t1
            t = torch.ones(x0.shape[0], 1, device=x0.device) * t2
            t1 = t2
            for r in range(1): # multiround sampling
                xt = traj[-1]
                xt[cond_mask==0] = chi[cond_mask==0]
                pred_chi_vf = self.get_vf(t, xt, batch)

                if self.mode == 'vf':
                    pred_chi_vf = self.coeff * pred_chi_vf * (1 - t) # exp schedule
                    xt_next = self.euler_step(dt, xt, pred_chi_vf, batch.chi_alt_mask)

                elif self.mode == 'cfm':
                    xt_next = xt + (dt * pred_chi_vf)
                    xt_next = torch.remainder(xt_next, 2*math.pi)
                else: raise NotImplementedError('wrong mode')
                traj.append(xt_next)

        if return_traj:
            return traj
        else:
            return traj[-1] # just return last step

    def euler_step(self, dt, xt, vf, alt_mask):
        return exp(xt, vf * dt, alt_mask=alt_mask)

    # Hutchinson trace estimation
    def log_prob(self, batch) -> Tensor:
        chi, mask = batch.chi, batch.chi_mask
        t_range = torch.linspace(1.0 - self.eps, 0, self.stepsize).to(chi)
        xt = chi
        t1 = t_range[0]
        p1 = torch.zeros_like(chi)
        for t2 in t_range[1:]:
            eps = torch.randn_like(chi)
            dt = t2 - t1
            t = torch.ones(chi.shape[0], 1, device=chi.device) * t2
            t1 = t2
            with torch.enable_grad():
                xt = xt.requires_grad_()
                pred_chi_vf = self.get_vf(t, xt, batch)

            jacobian = torch.autograd.grad(pred_chi_vf, xt, eps)[0]
            ft = jacobian * eps
            p1 = p1 + (ft * dt) # euler likelihood
            pred_chi_vf_exp = self.coeff * pred_chi_vf * (1 - t)
            xt = self.euler_step(dt, xt, -pred_chi_vf_exp, batch.chi_alt_mask) # take euler step in reverse direction

        # p(x0) for uniform distribution is ln(2*pi)
        p0 = torch.ones_like(xt) * math.log(2*torch.pi)
        likelihood = (p0 - p1) * mask

        return likelihood

    def get_noise_func(self, x0, x1, alt_mask=None):
        logmap = log(x0, x1)

        def path(t):
            xt = exp(x0, t * logmap, alt_mask=alt_mask)
            return xt

        return path