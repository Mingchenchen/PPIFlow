
import torch
from torch import nn

from models.node_feature_net import NodeFeatureNet
from models.edge_feature_net import EdgeFeatureNet
from models.pairformer import PairformerStack
from models.denoising import IPAStack
from data import utils as du
# from analysis.utils import get_gpu_memory_usage
from core.utils.checkpointing import get_checkpoint_fn


class FlowModel(nn.Module):

    def __init__(self, model_conf):
        super(FlowModel, self).__init__()
        self._model_conf = model_conf
        self._ipa_conf = model_conf.ipa
        self.rigids_ang_to_nm = lambda x: x.apply_trans_fn(lambda x: x * du.ANG_TO_NM_SCALE)
        self.rigids_nm_to_ang = lambda x: x.apply_trans_fn(lambda x: x * du.NM_TO_ANG_SCALE)
        self.task = self._model_conf['task']
        model_conf.node_features['task'] = self.task
        model_conf.edge_features['task'] = self.task
        self.node_feature_net = NodeFeatureNet(model_conf.node_features)
        self.edge_feature_net = EdgeFeatureNet(model_conf.edge_features)

        if model_conf.pairformer.use:
            self.conditioning_pair_net = PairformerStack(
                n_blocks=self._model_conf.pairformer.num_blocks,
                n_heads=self._model_conf.pairformer.no_heads,
                c_hidden_pair_att=self._model_conf.pairformer.c_hidden_pair_att,   # *
                no_heads_pair=self._model_conf.pairformer.no_heads_pair,
                blocks_per_ckpt=1
            )

        # Attention trunk
        self.ipa_net = IPAStack(self._model_conf, blocks_per_ckpt=None)

    def forward(self, input_feats):
        node_mask = input_feats['res_mask']    #torch.Size([1, N_res)
        edge_mask = node_mask[:, None] * node_mask[:, :, None]    #torch.Size([1, N_res, N_res])
        diffuse_mask = input_feats['diffuse_mask']  # torch.Size([1, N_res])

        res_index = input_feats['res_idx']       #torch.Size([1, N_res])
        aatype = torch.where(diffuse_mask == 1, torch.tensor(21), input_feats['aatype'])    #torch.Size([1, N_res])
        so3_t = input_feats['so3_t']#torch.Size([1, 1])
        r3_t = input_feats['r3_t']#torch.Size([1, 1])
        trans_t = input_feats['trans_t']
        rotmats_t = input_feats['rotmats_t']
        hotspot_mask = input_feats['hotspot_mask'] if 'hotspot_mask' in input_feats else None

        # Initialize node and edge embeddings
        init_node_embed = self.node_feature_net(
            so3_t,
            r3_t,
            node_mask,
            diffuse_mask,
            res_index,
            hotspot_mask,
            input_feats['chain_idx'],
            aatype
        )

        if 'trans_sc' not in input_feats:
            trans_sc = torch.zeros_like(trans_t)
        else:
            trans_sc = input_feats['trans_sc']
        init_edge_embed = self.edge_feature_net(
            init_node_embed,
            trans_t,
            trans_sc,
            edge_mask,
            diffuse_mask,
            hotspot_mask,
            input_feats['chain_idx']
        )

        # Initial rigids
        curr_rigids = du.create_rigid(rotmats_t, trans_t)

        # Main trunk
        curr_rigids = self.rigids_ang_to_nm(curr_rigids)
        node_embed = init_node_embed * node_mask[..., None]
        edge_embed = init_edge_embed * edge_mask[..., None]

        # pairformer stack (for conditions embedding)
        if self._model_conf.pairformer.use:
            node_embed, edge_embed = self.conditioning_pair_net(node_embed, edge_embed, None)

        pred_trans, pred_rotmats = self.ipa_net(node_embed, edge_embed, curr_rigids, node_mask, diffuse_mask, edge_mask)

        return {
            'pred_trans': pred_trans,#torch.Size([1, N_res, 3])
            'pred_rotmats': pred_rotmats,#torch.Size([1, N_res, 3, 3]),
            'pred_aatype': None
        }
