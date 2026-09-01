import torch
from torch import nn

from models.utils import get_index_embedding, calc_distogram, calc_inter_distogram

class EdgeMotifNet(nn.Module):

    def __init__(self, module_cfg):
        super(EdgeMotifNet, self).__init__()
        self._cfg = module_cfg

        self.c_s = self._cfg.c_s
        self.c_p = self._cfg.c_p
        self.feat_dim = self._cfg.feat_dim

        if self._cfg.pairwise_direction:
            total_edge_feats = self._cfg.num_bins + 2 + 1 + 3
        else:
            total_edge_feats = self._cfg.num_bins + 2 + 1

        self.edge_embedder = nn.Sequential(
            nn.Linear(total_edge_feats, self.c_p),
            nn.ReLU(),
            nn.Linear(self.c_p, self.c_p),
            nn.ReLU(),
            nn.Linear(self.c_p, self.c_p),
            nn.LayerNorm(self.c_p),
        )

    def _cross_concat_inter(self, feats_1d_res, feats_1d_pnt, num_batch, num_res, num_pnt):
        return torch.cat([
            torch.tile(feats_1d_res[:, :, None, :], (1, 1, num_pnt, 1)),
            torch.tile(feats_1d_pnt[:, None, :, :], (1, num_res, 1, 1)),
        ], dim=-1).float().reshape([num_batch, num_res, num_pnt, -1])


    def _cross_concat(self, feats_1d, num_batch, num_res):
        return torch.cat([
            torch.tile(feats_1d[:, :, None, :], (1, 1, num_res, 1)),
            torch.tile(feats_1d[:, None, :, :], (1, num_res, 1, 1)),
        ], dim=-1).float().reshape([num_batch, num_res, num_res, -1])

    def _pairwise_directions(self, coords, eps=1e-8):
        """
        coords: [B, N, 3]
        return: [B, N, N, 3]
        """
        # coords[..., None, :] => [B, N, 1, 3]
        # coords[:, None, ...] => [B, 1, N, 3]
        diff = coords[:, None, :, :] - coords[:, :, None, :]  # [B, N, N, 3]
        dist = torch.norm(diff, dim=-1, keepdim=True)  # [B, N, N, 1]
        direction = diff / (dist + eps)
        return direction

    def forward(self, t, rc_t, p_mask, chain_id=None):
        # Input: [b, n_res, c_s]
        num_batch = p_mask.shape[0]
        num_res = p_mask.shape[1]

        dist_feats = calc_inter_distogram(
            t, rc_t, min_bin=1e-3, max_bin=20.0, num_bins=self._cfg.num_bins)
        dist_feats *= p_mask.unsqueeze(-1) # n,n,20

        all_edge_feats = [dist_feats]
        all_edge_feats.extend([p_mask.unsqueeze(-1), p_mask.unsqueeze(-1)])  # n,n,2

        if self._cfg.embed_chain:
            chain_feat = (chain_id.unsqueeze(2) == chain_id.unsqueeze(1)).float().unsqueeze(-1)
            all_edge_feats.append(chain_feat)

        if self._cfg.pairwise_direction:
            pairwise_directions_feats = self._pairwise_directions(rc_t) # n,n,3
            pairwise_directions_feats *= p_mask.unsqueeze(-1)
            all_edge_feats.append(pairwise_directions_feats)

        edge_feats = torch.concat(all_edge_feats, dim=-1)*p_mask.unsqueeze(-1)
        edge_feats = self.edge_embedder(edge_feats)
        return edge_feats
