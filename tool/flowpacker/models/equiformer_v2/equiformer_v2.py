import logging
import time
import math
import numpy as np
import torch
import torch.nn as nn

from .smearing import GaussianSmearing

try:
    from e3nn import o3
except ImportError:
    pass

from .gaussian_rbf import GaussianRadialBasisLayer
from torch.nn import Linear
from .edge_rot_mat import init_edge_rot_mat
from .so3 import (
    CoefficientMappingModule,
    SO3_Embedding,
    SO3_Grid,
    SO3_Rotation,
    SO3_LinearV2
)
from .module_list import ModuleListInfo
from .so2_ops import SO2_Convolution
from .radial_function import RadialFunction
from .layer_norm import (
    EquivariantLayerNormArray, 
    EquivariantLayerNormArraySphericalHarmonics, 
    EquivariantRMSNormArraySphericalHarmonics,
    EquivariantRMSNormArraySphericalHarmonicsV2,
    get_normalization_layer
)
from .transformer_block import (
    SO2EquivariantGraphAttention,
    FeedForwardNetwork,
    TransBlockV2, 
)
from torch_cluster import radius_graph, knn_graph

class PositionalEncodings(torch.nn.Module):
    def __init__(self, num_embeddings=64, period_range=[2,1000]):
        super(PositionalEncodings, self).__init__()
        self.num_embeddings = num_embeddings
        self.period_range = period_range

    def forward(self, d):
        frequency = torch.exp(
            torch.arange(0, self.num_embeddings, 2, dtype=torch.float32)
            * -(np.log(10000.0) / self.num_embeddings)
        ).to(d)
        angles = d * frequency.view((1,-1))
        E = torch.cat((torch.cos(angles), torch.sin(angles)), -1)
        return E

class EquiformerV2(nn.Module):
    """
    Equiformer with graph attention built upon SO(2) convolution and feedforward network built upon S2 activation

    Args:
        max_neighbors (int):    Maximum number of neighbors per atom
        max_radius (float):     Maximum distance between nieghboring atoms in Angstroms
        max_num_elements (int): Maximum atomic number

        num_layers (int):             Number of layers in the GNN
        sphere_channels (int):        Number of spherical channels (one set per resolution)
        attn_hidden_channels (int): Number of hidden channels used during SO(2) graph attention
        num_heads (int):            Number of attention heads
        attn_alpha_head (int):      Number of channels for alpha vector in each attention head
        attn_value_head (int):      Number of channels for value vector in each attention head
        ffn_hidden_channels (int):  Number of hidden channels used during feedforward network
        norm_type (str):            Type of normalization layer (['layer_norm', 'layer_norm_sh', 'rms_norm_sh'])

        lmax_list (int):              List of maximum degree of the spherical harmonics (1 to 10)
        mmax_list (int):              List of maximum order of the spherical harmonics (0 to lmax)
        grid_resolution (int):        Resolution of SO3_Grid
        
        num_sphere_samples (int):     Number of samples used to approximate the integration of the sphere in the output blocks
        
        edge_channels (int):                Number of channels for the edge invariant features
        use_atom_edge_embedding (bool):     Whether to use atomic embedding along with relative distance for edge scalar features
        share_atom_edge_embedding (bool):   Whether to share `atom_edge_embedding` across all blocks
        use_m_share_rad (bool):             Whether all m components within a type-L vector of one channel share radial function weights
        distance_function ("gaussian", "sigmoid", "linearsigmoid", "silu"):  Basis function used for distances
        
        attn_activation (str):      Type of activation function for SO(2) graph attention
        use_s2_act_attn (bool):     Whether to use attention after S2 activation. Otherwise, use the same attention as Equiformer
        use_attn_renorm (bool):     Whether to re-normalize attention weights
        ffn_activation (str):       Type of activation function for feedforward network
        use_gate_act (bool):        If `True`, use gate activation. Otherwise, use S2 activation
        use_grid_mlp (bool):        If `True`, use projecting to grids and performing MLPs for FFNs. 

        _act (bool):      If `True`, use separable S2 activation when `use_gate_act` is False.

        alpha_drop (float):         Dropout rate for attention weights
        drop_path_rate (float):     Drop path rate
        proj_drop (float):          Dropout rate for outputs of attention and FFN in Transformer blocks

        weight_init (str):          ['normal', 'uniform'] initialization of weights of linear layers except those in radial functions
    """
    def __init__(
        self,
        node_feature_in=21,
        edge_feature_in=128,

        num_layers=12,
        sphere_channels=128,
        attn_hidden_channels=128,
        num_heads=8,
        attn_alpha_channels=32,
        attn_value_channels=16,
        ffn_hidden_channels=512,
        
        norm_type='rms_norm_sh',
        
        lmax_list=[6],
        mmax_list=[2],
        grid_resolution=None, 

        num_sphere_samples=128,

        edge_channels=128,
        use_atom_edge_embedding=True, 
        share_atom_edge_embedding=False,
        use_m_share_rad=False,

        attn_activation='silu',
        use_s2_act_attn=False, 
        use_attn_renorm=True,
        ffn_activation='silu',
        use_gate_act=False,
        use_grid_mlp=True,
        use_sep_s2_act=True,

        alpha_drop=0.1,
        drop_path_rate=0.05, 
        proj_drop=0.0, 

        weight_init='normal',
        self_condition=False,
        use_virtual_cb=False,
        edge_type='knn',
        max_radius=12.0,
        max_neighbors=30,
        out_dim=4,
        **kwargs,
    ):
        super().__init__()

        self.node_feature_in = node_feature_in
        self.edge_feature_in = edge_feature_in

        self.num_layers = num_layers
        self.sphere_channels = sphere_channels
        self.attn_hidden_channels = attn_hidden_channels
        self.num_heads = num_heads
        self.attn_alpha_channels = attn_alpha_channels
        self.attn_value_channels = attn_value_channels
        self.ffn_hidden_channels = ffn_hidden_channels
        self.norm_type = norm_type
        
        self.lmax_list = lmax_list
        self.mmax_list = mmax_list
        self.grid_resolution = grid_resolution

        self.num_sphere_samples = num_sphere_samples

        self.edge_channels = edge_channels
        self.use_atom_edge_embedding = use_atom_edge_embedding 
        self.share_atom_edge_embedding = share_atom_edge_embedding
        if self.share_atom_edge_embedding:
            assert self.use_atom_edge_embedding
            self.block_use_atom_edge_embedding = False
        else:
            self.block_use_atom_edge_embedding = self.use_atom_edge_embedding
        self.use_m_share_rad = use_m_share_rad

        self.attn_activation = attn_activation
        self.use_s2_act_attn = use_s2_act_attn
        self.use_attn_renorm = use_attn_renorm
        self.ffn_activation = ffn_activation
        self.use_gate_act = use_gate_act
        self.use_grid_mlp = use_grid_mlp
        self.use_sep_s2_act = use_sep_s2_act
        
        self.alpha_drop = alpha_drop
        self.drop_path_rate = drop_path_rate
        self.proj_drop = proj_drop

        self.weight_init = weight_init
        assert self.weight_init in ['normal', 'uniform']

        self.device = 'cpu' #torch.cuda.current_device()

        self.grad_forces = False
        self.num_resolutions = len(self.lmax_list)
        self.sphere_channels_all = self.num_resolutions * self.sphere_channels
        self.self_condition = self_condition
        self.use_virtual_cb = use_virtual_cb
        self.edge_type = edge_type
        self.max_radius = max_radius
        self.max_neighbors = max_neighbors
        
        # Weights for message initialization
        if self_condition:
            self.node_feature_in = self.node_feature_in + 4

        self.sphere_embedding = nn.Linear(self.node_feature_in, self.sphere_channels_all)
        
        # Initialize the sizes of radial functions (input channels and 2 hidden channels)
        self.edge_channels_list = [self.edge_channels] * 3

        # Initialize atom edge embedding
        self.source_embedding = nn.Linear(self.node_feature_in, self.edge_channels_list[-1])
        self.target_embedding = nn.Linear(self.node_feature_in, self.edge_channels_list[-1])
        self.edge_feat_embedding = nn.Linear(self.edge_feature_in, self.edge_channels_list[-1])
        self.edge_channels_list[0] = 3 * self.edge_channels_list[-1]
        
        # Initialize the module that compute WignerD matrices and other values for spherical harmonic calculations
        self.SO3_rotation = nn.ModuleList()
        for i in range(self.num_resolutions):
            self.SO3_rotation.append(SO3_Rotation(self.lmax_list[i]))

        # Initialize conversion between degree l and order m layouts
        self.mappingReduced = CoefficientMappingModule(self.lmax_list, self.mmax_list)

        # Initialize the transformations between spherical and grid representations
        self.SO3_grid = ModuleListInfo('({}, {})'.format(max(self.lmax_list), max(self.lmax_list)))
        for l in range(max(self.lmax_list) + 1):
            SO3_m_grid = nn.ModuleList()
            for m in range(max(self.lmax_list) + 1):
                SO3_m_grid.append(
                    SO3_Grid(
                        l, 
                        m, 
                        resolution=self.grid_resolution, 
                        normalization='component'
                    )
                )
            self.SO3_grid.append(SO3_m_grid)

        # Initialize the blocks for each layer of EquiformerV2
        self.blocks = nn.ModuleList()
        for i in range(self.num_layers):
            block = TransBlockV2(
                self.sphere_channels,
                self.attn_hidden_channels,
                self.num_heads,
                self.attn_alpha_channels,
                self.attn_value_channels,
                self.ffn_hidden_channels,
                self.sphere_channels, 
                self.lmax_list,
                self.mmax_list,
                self.SO3_rotation,
                self.mappingReduced,
                self.SO3_grid,
                self.node_feature_in,
                self.edge_channels_list,
                self.block_use_atom_edge_embedding,
                self.use_m_share_rad,
                self.attn_activation,
                self.use_s2_act_attn,
                self.use_attn_renorm,
                self.ffn_activation,
                self.use_gate_act,
                self.use_grid_mlp,
                self.use_sep_s2_act,
                self.norm_type,
                self.alpha_drop, 
                self.drop_path_rate,
                self.proj_drop
            )
            self.blocks.append(block)
        
        # Output blocks for energy and forces
        self.norm = get_normalization_layer(self.norm_type, lmax=max(self.lmax_list), num_channels=self.sphere_channels)
        self.scalar_block = FeedForwardNetwork(
            self.sphere_channels,
            self.ffn_hidden_channels,
            out_dim,
            self.lmax_list,
            self.mmax_list,
            self.SO3_grid,
            self.ffn_activation,
            self.use_gate_act,
            self.use_grid_mlp,
            self.use_sep_s2_act
        )

        self.pos_emb = PositionalEncodings()
        # self.l1_linear = torch.nn.Linear(28, self.sphere_channels, bias=False)
            
        self.apply(self._init_weights)
        self.apply(self._uniform_init_rad_func_linear_weights)

    def generate_graph(self, pos, edge_index):
        j, i = edge_index
        distance_vec = pos[j] - pos[i]
        edge_dist = distance_vec.norm(dim=-1)

        return (
            edge_dist,
            distance_vec,
        )

    def forward(self, t, sc_torsions, node_feat, pos, edge_index, edge_feat, mask, batch, atom_mask=None, self_cond=None):
        self.batch_size = pos.shape[0]
        self.dtype = pos.dtype
        self.device = pos.device

        # # use virtual Cb
        # if self.use_virtual_cb:
        #     b = pos[:,1,:] - pos[:,0,:]
        #     c = pos[:,2,:] - pos[:,1,:]
        #     a = torch.cross(b, c, dim=-1)
        #     node_crds = -0.58273431*a + 0.56802827*b - 0.54067466*c + pos[:,1,:]
        # else:
        #     node_crds = pos[:,1]

        num_atoms = pos.shape[0]
        edge_distance, edge_distance_vec = self.generate_graph(pos, edge_index)

        ###############################################################
        # Initialize data structures
        ###############################################################

        # Compute 3x3 rotation matrix per edge
        edge_rot_mat = self._init_edge_rot_mat(edge_distance_vec)

        # Initialize the WignerD matrices and other values for spherical harmonic calculations
        for i in range(self.num_resolutions):
            self.SO3_rotation[i].set_wigner(edge_rot_mat)

        ###############################################################
        # Initialize node embeddings
        ###############################################################

        # Init per node representations using an atomic number based embedding
        x = SO3_Embedding(
            num_atoms,
            self.lmax_list,
            self.sphere_channels,
            self.device,
            self.dtype,
        )

        offset_res = 0
        offset = 0

        t_embed = self.pos_emb(t)
        node_feat_in = torch.cat([node_feat,t_embed,sc_torsions],dim=-1)
        if self_cond is not None:
            node_feat_in = torch.cat([node_feat_in,self_cond],dim=-1)

        # Initialize the l = 0, m = 0 coefficients for each resolution
        for i in range(self.num_resolutions):
            if self.num_resolutions == 1:
                x.embedding[:, offset_res, :] = self.sphere_embedding(node_feat_in)
            else:
                x.embedding[:, offset_res, :] = self.sphere_embedding(
                    node_feat_in
                    )[:, offset : offset + self.sphere_channels]
            offset = offset + self.sphere_channels
            offset_res = offset_res + int((self.lmax_list[i] + 1) ** 2)

        # # Add coordinates and relative positions to l=1 features
        # # Linear layers are equivariant for same degree features
        # if len(pos.shape) == 3:
        #     l1_in = []
        #     l1_in.append(pos.transpose(-1,-2))
        #     if atom_mask is not None:
        #         relpos_to_ca = (pos - pos[:,1].unsqueeze(-2)) * atom_mask.unsqueeze(-1)
        #         l1_in.append(relpos_to_ca.transpose(-1,-2))
        #     l1_in = torch.cat(l1_in, dim=-1)
        #     x.embedding[:,1:4] = self.l1_linear(l1_in)

        # Edge encoding (distance and atom edge)
        edge_feat_embedding = self.edge_feat_embedding(edge_feat)
        source_element = node_feat_in[edge_index[0]]  # Source node features
        target_element = node_feat_in[edge_index[1]]  # Target node features
        source_embedding = self.source_embedding(source_element)
        target_embedding = self.target_embedding(target_element)
        edge_feat = torch.cat((source_embedding, target_embedding, edge_feat_embedding), dim=1)

        ###############################################################
        # Update spherical node embeddings
        ###############################################################

        for i in range(self.num_layers):
            x = self.blocks[i](
                x,                  # SO3_Embedding
                node_feat_in,
                edge_feat,
                edge_index,
                batch=batch    # for GraphDropPath
            )

        # Final layer norm
        x.embedding = self.norm(x.embedding)
        pred_scalars = self.scalar_block(x)
        pred_scalars = pred_scalars.embedding.narrow(1, 0, 1).squeeze(1) # extract scalars

        return pred_scalars * mask

    # Initialize the edge rotation matrics
    def _init_edge_rot_mat(self, edge_distance_vec):
        return init_edge_rot_mat(edge_distance_vec)

    @property
    def num_params(self):
        return sum(p.numel() for p in self.parameters())


    def _init_weights(self, m):
        if (isinstance(m, torch.nn.Linear)
            or isinstance(m, SO3_LinearV2)
        ):
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)
            if self.weight_init == 'normal':
                std = 1 / math.sqrt(m.in_features)
                torch.nn.init.normal_(m.weight, 0, std)

        elif isinstance(m, torch.nn.LayerNorm):
            torch.nn.init.constant_(m.bias, 0)
            torch.nn.init.constant_(m.weight, 1.0)

    
    def _uniform_init_rad_func_linear_weights(self, m):
        if (isinstance(m, RadialFunction)):
            m.apply(self._uniform_init_linear_weights)


    def _uniform_init_linear_weights(self, m):
        if isinstance(m, torch.nn.Linear):
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)
            std = 1 / math.sqrt(m.in_features)
            torch.nn.init.uniform_(m.weight, -std, std)

    
    @torch.jit.ignore
    def no_weight_decay(self):
        no_wd_list = []
        named_parameters_list = [name for name, _ in self.named_parameters()]
        for module_name, module in self.named_modules():
            if (isinstance(module, torch.nn.Linear) 
                or isinstance(module, SO3_LinearV2)
                or isinstance(module, torch.nn.LayerNorm)
                or isinstance(module, EquivariantLayerNormArray)
                or isinstance(module, EquivariantLayerNormArraySphericalHarmonics)
                or isinstance(module, EquivariantRMSNormArraySphericalHarmonics)
                or isinstance(module, EquivariantRMSNormArraySphericalHarmonicsV2)
                or isinstance(module, GaussianRadialBasisLayer)):
                for parameter_name, _ in module.named_parameters():
                    if (isinstance(module, torch.nn.Linear)
                        or isinstance(module, SO3_LinearV2)
                    ):
                        if 'weight' in parameter_name:
                            continue
                    global_parameter_name = module_name + '.' + parameter_name
                    assert global_parameter_name in named_parameters_list
                    no_wd_list.append(global_parameter_name)
        return set(no_wd_list)