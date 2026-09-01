import torch
import torch.nn as nn
from torch import Tensor
from utils.sidechain_utils import Idealizer
from dataset_cluster import get_edge_features
from torch_cluster import radius_graph, knn_graph

class ConfLoss(nn.Module):
    def __init__(self, model: nn.Module, config, eps=1e-3):
        super().__init__()
        self.model = model
        self.config = config
        self.eps = eps
        self.idealizer = Idealizer(use_native_bb_coords=True)

    def forward(self, batch) -> Tensor:
        ca_crds, chi_pred, chi_gt, batch_id, aa_mask, atom_mask, chi_mask = batch.pos[:, 1], batch.sample_chi, batch.chi, \
                                                                        batch.batch, batch.aa_mask, batch.atom_mask, batch.chi_mask
        sc_mask = (atom_mask[:,4:].sum(-1) > 0).unsqueeze(-1)

        batch_size = batch_id.max() + 1
        t = torch.ones([batch_size,1], device=ca_crds.device) # t [0.8,1.0]
        t = t[batch_id]

        dihedral_sin_cos = torch.cat([batch.bb_dihedral.sin(), batch.bb_dihedral.cos()],dim=-1)
        node_feat = torch.cat([batch.aa_onehot, dihedral_sin_cos], dim=-1)

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

        pred_crds = self.idealizer(batch.aa_num, batch.pos[:, :4], chi_pred) * atom_mask.unsqueeze(-1)
        edge_feat = get_edge_features(pred_crds, edge_index, atom_mask, all_atoms=self.config.model.add_dist_to_edge)

        pred_rmsd = self.model(t, chi_pred, node_feat, node_crds, batch.edge_index, edge_feat, sc_mask,
                              batch.batch, atom_mask=atom_mask)

        # rmsd loss
        gt_crds = self.virtual_to_all_atom(batch.aa_num, batch.pos[:,:4], chi_gt) * atom_mask.unsqueeze(-1)
        gt_rmsd = (((pred_crds[:,4:] - gt_crds[:,4:]).square()).sum(-1).sum(-1) / atom_mask[:,4:].sum(-1).clamp(min=1)).sqrt()

        loss = ((pred_rmsd - gt_rmsd[...,None]).square() * sc_mask).sum() / sc_mask.sum()

        return loss
