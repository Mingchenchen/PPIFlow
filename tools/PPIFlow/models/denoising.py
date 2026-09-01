import torch
from torch import nn
from typing import Optional
from functools import partial
import torch.nn.functional as F
from models import ipa_pytorch
from data import utils as du
from core.utils.checkpointing import checkpoint_blocks


class IPABlock(nn.Module):
    def __init__(self, model_conf, last_block=False):
        super(IPABlock, self).__init__()
        self._model_conf = model_conf
        self._ipa_conf = self._model_conf.ipa
        self._last_block = last_block
        tfmr_in = self._ipa_conf.c_s
        self.trunk = nn.ModuleDict()
        self.trunk[f'ipa'] = ipa_pytorch.InvariantPointAttention(self._ipa_conf)
        self.trunk[f'ipa_ln'] = nn.LayerNorm(self._ipa_conf.c_s)
        self.trunk[f'seq_tfmr'] = torch.nn.TransformerEncoder(
            torch.nn.TransformerEncoderLayer(
                d_model=tfmr_in,
                nhead=self._ipa_conf.seq_tfmr_num_heads,
                dim_feedforward=self._ipa_conf.c_s,
                batch_first=True,
                dropout=0.0,
                norm_first=False
            ),
            self._ipa_conf.seq_tfmr_num_layers,
            enable_nested_tensor=False)
        self.trunk[f'post_tfmr'] = ipa_pytorch.Linear(
            tfmr_in, self._ipa_conf.c_s, init="final")
        self.trunk[f'node_transition'] = ipa_pytorch.StructureModuleTransition(
            c=self._ipa_conf.c_s)
        self.trunk[f'bb_update'] = ipa_pytorch.BackboneUpdate(
            self._ipa_conf.c_s, use_rot_updates=True)

        if not self._last_block:
            # No edge update on the last block.
            edge_in = self._model_conf.edge_embed_size
            self.trunk[f'edge_transition'] = ipa_pytorch.EdgeTransition(
                node_embed_size=self._ipa_conf.c_s,
                edge_embed_in=edge_in,
                edge_embed_out=self._model_conf.edge_embed_size,
                use_tri_update=False
            )
        
    def forward(self, node_embed, edge_embed, curr_rigids, node_mask, diffuse_mask, edge_mask):
        ipa_embed = self.trunk[f'ipa'](
            node_embed,
            edge_embed,
            curr_rigids,
            node_mask)
        ipa_embed *= node_mask[..., None]
        node_embed = self.trunk[f'ipa_ln'](node_embed + ipa_embed)
        seq_tfmr_out = self.trunk[f'seq_tfmr'](
            node_embed, src_key_padding_mask=(1 - node_mask).to(torch.bool))
        node_embed = node_embed + self.trunk[f'post_tfmr'](seq_tfmr_out)
        node_embed = self.trunk[f'node_transition'](node_embed)
        node_embed = node_embed * node_mask[..., None]
        rigid_update = self.trunk[f'bb_update'](
            node_embed * node_mask[..., None])
        curr_rigids = curr_rigids.compose_q_update_vec(
            rigid_update, (node_mask * diffuse_mask)[..., None])
        if not self._last_block:
            edge_embed = self.trunk[f'edge_transition'](
                node_embed, edge_embed)
            edge_embed *= edge_mask[..., None]

        return node_embed, edge_embed, curr_rigids


class IPAStack(nn.Module):
    def __init__(self, model_conf, blocks_per_ckpt: Optional[int] = None,) -> None:
        super(IPAStack, self).__init__()
        self.rigids_nm_to_ang = lambda x: x.apply_trans_fn(lambda x: x * du.NM_TO_ANG_SCALE)
        self._model_conf = model_conf
        self.n_blocks = self._model_conf.ipa.num_blocks
        self.blocks_per_ckpt = blocks_per_ckpt
        self.blocks = nn.ModuleList()
        for i in range(self.n_blocks):
            last_block = i==self.n_blocks-1
            block = IPABlock(model_conf=self._model_conf, last_block=last_block)
            self.blocks.append(block)

    def _prep_blocks(
        self,
        node_mask: Optional[torch.Tensor],
        diffuse_mask: Optional[torch.Tensor],
        edge_mask: Optional[torch.Tensor],
        clear_cache_between_blocks: bool = False,
    ):
        blocks = [
            partial(
                b,
                node_mask=node_mask,
                diffuse_mask=diffuse_mask,
                edge_mask=edge_mask
            )
            for b in self.blocks
        ]

        def clear_cache(b, *args, **kwargs):
            torch.cuda.empty_cache()
            return b(*args, **kwargs)

        if clear_cache_between_blocks:
            blocks = [partial(clear_cache, b) for b in blocks]
        return blocks

    def forward(
        self,
        node_embed: torch.Tensor,
        edge_embed: torch.Tensor,
        curr_rigids: torch.Tensor,
        node_mask: torch.Tensor,
        diffuse_mask: torch.Tensor,
        edge_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            use_memory_efficient_kernel (bool): Whether to use memory-efficient kernel. Defaults to False.
            use_deepspeed_evo_attention (bool): Whether to use DeepSpeed evolutionary attention. Defaults to False.
            use_lma (bool): Whether to use low-memory attention. Defaults to False.
            inplace_safe (bool): Whether it is safe to use inplace operations. Defaults to False.
            chunk_size (Optional[int]): Chunk size for memory-efficient operations. Defaults to None.
        """

        if node_embed.shape[-2] > 2000 and (not self.training):
            clear_cache_between_blocks = True
        else:
            clear_cache_between_blocks = False
        blocks = self._prep_blocks(
            node_mask=node_mask,
            diffuse_mask=diffuse_mask,
            edge_mask=edge_mask,
            clear_cache_between_blocks=clear_cache_between_blocks,
        )

        blocks_per_ckpt = self.blocks_per_ckpt
        if not torch.is_grad_enabled():
            blocks_per_ckpt = None

        node_embed, edge_embed, curr_rigids = checkpoint_blocks(
            blocks,
            args=(node_embed, edge_embed, curr_rigids),
            blocks_per_ckpt=blocks_per_ckpt,
        )

        curr_rigids = self.rigids_nm_to_ang(curr_rigids)
        pred_trans = curr_rigids.get_trans()
        pred_rotmats = curr_rigids.get_rots().get_rot_mats()
        return pred_trans, pred_rotmats
