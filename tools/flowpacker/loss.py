import torch
import torch.nn as nn
from torch import Tensor
import math
from utils.so2_utils import log, exp
from utils.constants import van_der_waals_radius, atom37_to_14_mask
from utils.sidechain_utils import Idealizer
from dataset_cluster import get_edge_features
from torch.autograd.functional import jvp
import random
from torch_cluster import radius_graph, knn_graph

def get_noise_func(x0, x1, alt_mask=None):
    logmap = log(x0,x1)
    def path(t):
        xt = exp(x0, t*logmap, alt_mask=alt_mask)
        return xt
    return path

class CFMLoss(nn.Module):
    def __init__(self, model: nn.Module, config, eps=1e-3):
        super().__init__()
        self.model = model
        self.config = config
        self.eps = eps
        self.idealizer = Idealizer(use_native_bb_coords=True)

    def forward(self, batch) -> Tensor:
        ca_crds, chi, chi_alt, batch_id, aa_mask, atom_mask, chi_mask = batch.pos[:, 1], batch.chi, batch.chi_alt, \
                                                                        batch.batch, batch.aa_mask, batch.atom_mask, batch.chi_mask

        batch_size = batch_id.max() + 1
        t = torch.rand([batch_size,1], device=ca_crds.device) * (1-self.eps)
        t = t[batch_id]

        chi = (chi + math.pi) * chi_mask

        dihedral_sin_cos = torch.cat([batch.bb_dihedral.sin(), batch.bb_dihedral.cos()],dim=-1)
        node_feat = torch.cat([batch.aa_onehot, dihedral_sin_cos], dim=-1)

        def cond_u(x0, x1, t, alt_mask=None):
            path = get_noise_func(x0, x1, alt_mask)
            x_t, u_t = jvp(path, (t,), (torch.ones_like(t).to(t),))
            return x_t, u_t

        x0 = torch.rand_like(chi) * (2*math.pi) # uniform prior [0,2pi)
        x0[batch.chi_alt_mask == 1] = x0[batch.chi_alt_mask == 1] % math.pi

        xt, ut = cond_u(x0, chi, t, batch.chi_alt_mask)
        xt = xt * chi_mask

        x_scaled = (xt-math.pi) * chi_mask
        noised_crds = self.idealizer(batch.aa_num, batch.pos[:,:4], x_scaled) # rescale back to [-pi,pi]
        noised_crds = noised_crds * atom_mask.unsqueeze(-1)

        # edge index
        ca = batch.pos[:,1]
        if self.config.model.use_virtual_cb:
            b = batch.pos[:,1,:] - batch.pos[:,0,:]
            c = batch.pos[:,2,:] - batch.pos[:,1,:]
            a = torch.cross(b, c, dim=-1)
            node_crds = -0.58273431*a + 0.56802827*b - 0.54067466*c + batch.pos[:,1,:]
        else:
            node_crds = ca

        if self.config.data.edge_type == 'radius':
            edge_index = radius_graph(node_crds, r=self.config.data.max_radius, max_num_neighbors=self.config.data.max_neighbors, batch=batch.batch)
        elif self.config.data.edge_type == 'knn':
            edge_index = knn_graph(node_crds, k=self.config.data.max_neighbors, batch=batch.batch)
        else: raise NotImplementedError('wrong edge type')

        edge_feat = get_edge_features(noised_crds, edge_index, atom_mask, all_atoms=self.config.model.add_dist_to_edge)

        # self-conditioning
        if self.config.train.self_condition:
            with torch.no_grad():
                chi_sc = torch.zeros_like(chi)
                if random.random() > 0.5:
                    chi_sc = self.model(t, xt, node_feat, node_crds, edge_index, edge_feat,
                                        chi_mask, batch.batch, atom_mask=atom_mask, self_cond=chi_sc)
                    if self.config.train.loss_type == 'x0':
                        chi_sc = torch.remainder(chi_sc, 2*math.pi)
        else:
            chi_sc = None

        pred_chi = self.model(t, xt, node_feat, node_crds, edge_index, edge_feat, chi_mask,
                              batch.batch, atom_mask=atom_mask, self_cond=chi_sc)

        weight = 1 / ((1 - torch.clamp(t, max=0.9).square()))
        if self.config.train.loss_type == 'vf':
            diff = (pred_chi - ut).square() * weight
        elif self.config.train.loss_type == 'cfm':
            diff = (pred_chi - chi + x0).square() * weight
        elif self.config.train.loss_type == 'x0':
            diff = log(pred_chi, chi).square() * weight
        else: raise NotImplementedError('wrong loss type')
        chi_loss = (diff * chi_mask).sum() / chi_mask.sum().clamp(min=1)

        return chi_loss