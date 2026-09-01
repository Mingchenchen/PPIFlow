import torch
from torch import nn

from models.utils import get_index_embedding, calc_distogram, calc_inter_distogram

class EdgeFeatureNet(nn.Module):

    def __init__(self, module_cfg):
        super(EdgeFeatureNet, self).__init__()
        self._cfg = module_cfg

        self.c_s = self._cfg.c_s
        self.c_p = self._cfg.c_p
        self.feat_dim = self._cfg.feat_dim

        self.linear_s_p = nn.Linear(self.c_s, self.feat_dim)
        self.linear_relpos = nn.Linear(self.feat_dim, self.feat_dim)

        total_edge_feats = self.feat_dim * 3 + self._cfg.num_bins * 2
        if self._cfg.embed_diffuse_mask:
            total_edge_feats += 2
        if (self._cfg.task in ["binder", "binder_motif"]) and self._cfg.embed_hotspot_mask:
            total_edge_feats += 2
        if self._cfg.embed_chain:
            total_edge_feats +=2

        self.edge_embedder = nn.Sequential(
            nn.Linear(total_edge_feats, self.c_p),
            nn.ReLU(),
            nn.Linear(self.c_p, self.c_p),
            nn.ReLU(),
            nn.Linear(self.c_p, self.c_p),
            nn.LayerNorm(self.c_p),
        )

    def embed_relpos(self, r):
        # AlphaFold 2 Algorithm 4 & 5
        # Based on OpenFold utils/tensor_utils.py
        # Input: [b, n_res]
        # [b, n_res, n_res]
        d = r[:, :, None] - r[:, None, :]
        pos_emb = get_index_embedding(d, self._cfg.feat_dim, max_len=2056)
        return self.linear_relpos(pos_emb)

    def _cross_concat(self, feats_1d, num_batch, num_res):
        return torch.cat([
            torch.tile(feats_1d[:, :, None, :], (1, 1, num_res, 1)),
            torch.tile(feats_1d[:, None, :, :], (1, num_res, 1, 1)),
        ], dim=-1).float().reshape([num_batch, num_res, num_res, -1])

    def forward(self, s, t, sc_t, p_mask, diffuse_mask, hotspot_mask=None, chain_id=None):
        # Input: [b, n_res, c_s]
        num_batch, num_res, _ = s.shape

        # [b, n_res, c_p]
        p_i = self.linear_s_p(s)
        cross_node_feats = self._cross_concat(p_i, num_batch, num_res)

        # [b, n_res]
        r = torch.arange(
            num_res, device=s.device).unsqueeze(0).repeat(num_batch, 1)
        relpos_feats = self.embed_relpos(r)

        dist_feats = calc_distogram(
            t, min_bin=1e-3, max_bin=20.0, num_bins=self._cfg.num_bins)
        sc_feats = calc_distogram(
            sc_t, min_bin=1e-3, max_bin=20.0, num_bins=self._cfg.num_bins)

        all_edge_feats = [cross_node_feats, relpos_feats, dist_feats, sc_feats]
        if self._cfg.embed_diffuse_mask:
            diff_feat = self._cross_concat(diffuse_mask[..., None], num_batch, num_res)
            all_edge_feats.append(diff_feat)
        if (self._cfg.task in ["binder", "binder_motif"]) and self._cfg.embed_hotspot_mask:
            hotspot_feat = self._cross_concat(hotspot_mask[..., None], num_batch, num_res)
            all_edge_feats.append(hotspot_feat)
        if self._cfg.embed_chain:
            chain_feat = self._cross_concat(chain_id[..., None], num_batch, num_res)
            all_edge_feats.append(chain_feat)

        edge_feats = self.edge_embedder(torch.concat(all_edge_feats, dim=-1))
        edge_feats *= p_mask.unsqueeze(-1)
        return edge_feats

class AbEdgeFeatureNet(nn.Module):
    def __init__(self, module_cfg):
        super(AbEdgeFeatureNet, self).__init__()
        self._cfg = module_cfg

        self.c_s = self._cfg.c_s
        self.c_p = self._cfg.c_p
        self.feat_dim = self._cfg.feat_dim

        self.linear_s_p = nn.Linear(self.c_s, self.feat_dim)
        self.linear_relpos = nn.Linear(self.feat_dim, self.feat_dim)

        total_edge_feats = self.feat_dim * 3 + self._cfg.num_bins * 2
        if self._cfg.embed_diffuse_mask:
            total_edge_feats += 2
        if (self._cfg.task in ["binder", "binder_motif"]) and self._cfg.embed_hotspot_mask:
            total_edge_feats += 2
        if self._cfg.embed_chain:
            total_edge_feats +=2

        self.edge_embedder = nn.Sequential(
            nn.Linear(total_edge_feats, self.c_p),
            nn.ReLU(),
            nn.Linear(self.c_p, self.c_p),
            nn.ReLU(),
            nn.Linear(self.c_p, self.c_p),
            nn.LayerNorm(self.c_p),
        )

    def embed_relpos(self, r):
        # AlphaFold 2 Algorithm 4 & 5
        # Based on OpenFold utils/tensor_utils.py
        # Input: [b, n_res]
        # [b, n_res, n_res]
        d = r[:, :, None] - r[:, None, :]
        pos_emb = get_index_embedding(d, self._cfg.feat_dim, max_len=2056)#torch.Size([1, 596, 596, 64])
        return self.linear_relpos(pos_emb)

    def _cross_concat(self, feats_1d, num_batch, num_res):
        return torch.cat([
            torch.tile(feats_1d[:, :, None, :], (1, 1, num_res, 1)),#torch.Size([1, 596, 64])-->torch.Size([1, 596, 596, 64])
            torch.tile(feats_1d[:, None, :, :], (1, num_res, 1, 1)),
        ], dim=-1).float().reshape([num_batch, num_res, num_res, -1])#torch.Size([1, 596, 596, 128])-->torch.Size([1, 596, 596, 128])

    def forward(self, s, t, sc_t, p_mask, diffuse_mask, hotspot_mask=None, chain_id=None, chain_group_id=None):  #
        # Input: [b, n_res, c_s]
        num_batch, num_res, _ = s.shape

        # [b, n_res, c_p]
        p_i = self.linear_s_p(s)
        cross_node_feats = self._cross_concat(p_i, num_batch, num_res)

        # [b, n_res]
        r = torch.arange(
            num_res, device=s.device).unsqueeze(0).repeat(num_batch, 1)
        relpos_feats = self.embed_relpos(r)

        dist_feats = calc_distogram(
            t, min_bin=1e-3, max_bin=20.0, num_bins=self._cfg.num_bins)
        sc_feats = calc_distogram(
            sc_t, min_bin=1e-3, max_bin=20.0, num_bins=self._cfg.num_bins)

        all_edge_feats = [cross_node_feats, relpos_feats, dist_feats, sc_feats]
        if self._cfg.embed_diffuse_mask:
            if diffuse_mask.dim()==2:
                diff_feat = self._cross_concat(diffuse_mask[..., None], num_batch, num_res)
            elif diffuse_mask.dim()==3:
                diff_feat = torch.cat([diffuse_mask.unsqueeze(-1), diffuse_mask.unsqueeze(-1)], dim=-1)
            all_edge_feats.append(diff_feat)
        if (self._cfg.task in ["binder", "binder_motif"]) and self._cfg.embed_hotspot_mask:
            hotspot_feat = self._cross_concat(hotspot_mask[..., None], num_batch, num_res)
            all_edge_feats.append(hotspot_feat)
        if self._cfg.embed_chain:
            cross_chain_feats = (chain_id.unsqueeze(2) == chain_id.unsqueeze(1)).float().unsqueeze(-1)
            cross_chain_group_feats = (chain_group_id.unsqueeze(2) == chain_group_id.unsqueeze(1)).float().unsqueeze(-1)
            all_edge_feats.append(cross_chain_feats)
            all_edge_feats.append(cross_chain_group_feats)

        edge_feats = self.edge_embedder(torch.concat(all_edge_feats, dim=-1))
        edge_feats *= p_mask.unsqueeze(-1)
        return edge_feats

class RcFeatureNet(nn.Module):
    def __init__(self, module_cfg):
        super(RcFeatureNet, self).__init__()
        self._cfg = module_cfg

        self.c_s = self._cfg.c_s
        self.c_p = self._cfg.c_p
        self.feat_dim = self._cfg.feat_dim

        self.linear_s_p = nn.Linear(self.c_s, self.feat_dim)

        total_edge_feats = self.feat_dim * 2 + self._cfg.num_bins + 2
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
            torch.tile(feats_1d_res[:, :, None, :], (1, 1, num_pnt, 1)),#torch.Size([1, 596, 64]) -->torch.Size([1, 596, 596, 64])
            torch.tile(feats_1d_pnt[:, None, :, :], (1, num_res, 1, 1)),#torch.Size([1, 16, 64]) -->torch.Size([1, 16, 16, 64])
        ], dim=-1).float().reshape([num_batch, num_res, num_pnt, -1])#torch.Size([1, 596, 16, 128])-->torch.Size([1, 596, 16, 128])

    def forward(self, aatype_feats, aatype_rc_feats, t, rc_t, p_mask, diffuse_mask, hotspot_mask):
        # num_batch, num_res, _ = s.shape
        num_batch = aatype_feats.shape[0]
        num_res = aatype_feats.shape[1]
        num_pnt = aatype_rc_feats.shape[1]

        assert self._cfg.embed_diffuse_mask
        assert self._cfg.embed_hotspot_mask
        # assert self._cfg.task == "binder_rotamer"

        # [b, n_res, c_p]
        # p_i = self.linear_s_p(s)  # torch.Size([1, 70, 256])-->torch.Size([1, 70, 64])
        # cross_node_feats = self._cross_concat(p_i, num_batch, num_res)  # torch.Size([1, 70, 70, 128])

        # [b, n_res]
        # r = torch.arange(
        #     num_res, device=s.device).unsqueeze(0).repeat(num_batch, 1)  # torch.Size([1, 596])
        # relpos_feats = self.embed_relpos(r)  # torch.Size([1, 70, 70, 64])


        aatype_feats_p = self.linear_s_p(aatype_feats)      # torch.Size([1, N_res, 256])-->torch.Size([1, N_res, 64])
        aatype_rc_feats_p = self.linear_s_p(aatype_rc_feats)     # torch.Size([1, N_pnt, 256])-->torch.Size([1, N_pnt, 64])
        cross_node_feats = self._cross_concat_inter(aatype_feats_p, aatype_rc_feats_p, num_batch, num_res, num_pnt)    # torch.Size([1, N_res, N_pnt, 128])

        dist_feats = calc_inter_distogram(
            t, rc_t, min_bin=1e-3, max_bin=20.0, num_bins=self._cfg.num_bins)
        # todo: orientation feats
        # sc_feats = calc_distogram(
        #     sc_t, min_bin=1e-3, max_bin=20.0, num_bins=self._cfg.num_bins)

        all_edge_feats = [cross_node_feats, dist_feats]

        # diff_feat = self._cross_concat(diffuse_mask[..., None], num_batch, num_res)  # torch.Size([1, 596, 596, 2])
        diff_feat = torch.tile(diffuse_mask[:, :, None, None], (1, 1, num_pnt, 1))
        all_edge_feats.append(diff_feat)

        # hotspot_feat = self._cross_concat(hotspot_mask[..., None], num_batch, num_res)
        hotspot_feat = torch.tile(hotspot_mask[:, :, None, None], (1, 1, num_pnt, 1))
        all_edge_feats.append(hotspot_feat)

        edge_feats = self.edge_embedder(torch.concat(all_edge_feats,
                                                     dim=-1))  # torch.Size([1, 70, 16, 152])-->torch.Size([1, 70, 16, 128])
        edge_feats *= p_mask.unsqueeze(-1)


        return edge_feats
