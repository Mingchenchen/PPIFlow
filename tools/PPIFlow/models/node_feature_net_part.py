import torch
from torch import nn
from models.utils import get_index_embedding, get_time_embedding


class NodeFeatureNet(nn.Module):

    def __init__(self, module_cfg):
        super(NodeFeatureNet, self).__init__()
        self._cfg = module_cfg
        self.c_s = self._cfg.c_s
        self.c_pos_emb = self._cfg.c_pos_emb
        self.c_timestep_emb = self._cfg.c_timestep_emb
        embed_size = self._cfg.c_pos_emb + self._cfg.c_timestep_emb * 2 + 1
        if self._cfg.task in ["binder", "binder_motif", "binder_rotamer", "binder_motif_partial"]:#partial
            embed_size += 1
        if self._cfg.embed_chain:
            embed_size += 1

        if self._cfg.embed_aatype:
            embed_size += self._cfg.c_aatype_emb
            # self.aatype_layer = nn.Linear(22, self._cfg.c_aatype_emb)
            self.aatype_emb = nn.Embedding(self._cfg.n_aatype, self._cfg.c_aatype_emb)    # n_aatype: 22, c_aatype_emb: 128
        self.linear = nn.Linear(embed_size, self.c_s)

    def embed_t(self, timesteps, mask):
        timestep_emb = get_time_embedding(
            timesteps[:, 0],
            self.c_timestep_emb,
            max_positions=2056
        )[:, None, :].repeat(1, mask.shape[1], 1)
        return timestep_emb * mask.unsqueeze(-1)#torch.Size([1, 596, 128])

    def embed_aatype(self, aatype, mask):
        aatype_emb = self.aatype_emb(aatype)
        return aatype_emb * mask.unsqueeze(-1)

    def forward(self, so3_t, r3_t, res_mask, diffuse_mask, pos, hotspot_mask=None, chain_id=None, aatype=None):   #
        # s: [b]

        b, num_res, device = res_mask.shape[0], res_mask.shape[1], res_mask.device

        # [b, n_res, c_pos_emb]
        # pos = torch.arange(num_res, dtype=torch.float32).to(device)[None]
        if pos is None:
            pos_emb = torch.zeros(b, num_res, self.c_pos_emb, device=device)
        else:
            pos_emb = get_index_embedding(pos, self.c_pos_emb, max_len=2056)#torch.Size([1, 70, 128])
        pos_emb = pos_emb * res_mask.unsqueeze(-1)

        # [b, n_res, c_timestep_emb]
        input_feats = [
            pos_emb,#torch.Size([1, 70, 128])
            diffuse_mask[..., None],#torch.Size([1, 70,1])
            self.embed_t(so3_t, res_mask),#torch.Size([1, 70, 128])
            self.embed_t(r3_t, res_mask),#torch.Size([1, 70, 128])
        ]
        if hotspot_mask is not None:
            assert self._cfg.task in ["binder", "binder_motif", "binder_rotamer", "binder_motif_partial"]#partial
            input_feats.append(hotspot_mask[..., None])##torch.Size([1, 70])
        if self._cfg.embed_chain:
            input_feats.append(chain_id[..., None])

        if self._cfg.embed_aatype:
            aa_emb = self.embed_aatype(aatype, res_mask)
            input_feats.append(aa_emb)  ##torch.Size([1, 70, 128])
        # xk
        linear_input = torch.cat(input_feats, dim=-1)
        out = self.linear(linear_input)#old:torch.Size([1, 92, 385])-->torch.Size([1, 92, 256]) #new: torch.Size([1, 92, 386])-->torch.Size([1, 92, 256]) # add aatype: torch.Size([1, 92, 508])-->torch.Size([1, 92, 256])
        return out

