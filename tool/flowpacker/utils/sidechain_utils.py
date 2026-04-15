import torch
import torch.nn as nn

from openfold.np.residue_constants import (
    restype_rigid_group_default_frame,
    restype_atom14_to_rigid_group,
    restype_atom14_mask,
    restype_atom14_rigid_group_positions,
)
from openfold.utils.feats import (
    frames_and_literature_positions_to_atom14_pos,
    torsion_angles_to_frames,
)
from openfold.utils.rigid_utils import Rigid
from utils.constants import chi_true_indices, chi_mask
from biotite.structure import dihedral

class Idealizer(nn.Module):
    def __init__(self, use_native_bb_coords=True):
        super(Idealizer, self).__init__()
        self.use_native_bb_coords = use_native_bb_coords

    def forward(self, aa, bb_coords, sc_torsions):
        # Backbone frames
        n, ca, c, _ = torch.unbind(bb_coords,dim=-2)
        bb_rigids = Rigid.make_transform_from_reference(n,ca,c)

        # Coordinates -> torsion angles
        bb_dihedrals = get_bb_dihedral(n, ca, c)
        # [*, N, 7, 2]
        angles = torch.cat([bb_dihedrals,sc_torsions], dim=-1)
        angles = torch.stack([angles.sin(), angles.cos()], dim=-1)

        all_frames_to_global = self.torsion_angles_to_frames(
            bb_rigids,
            angles,
            aa,
        )

        pred_xyz = self.frames_and_literature_positions_to_atom14_pos(
            all_frames_to_global,
            aa,
        )

        if self.use_native_bb_coords:
            pred_xyz[:,:4] = bb_coords

        return pred_xyz

    def _init_residue_constants(self, float_dtype, device):
        if not hasattr(self, "default_frames"):
            self.register_buffer(
                "default_frames",
                torch.tensor(
                    restype_rigid_group_default_frame,
                    dtype=float_dtype,
                    device=device,
                    requires_grad=False,
                ),
                persistent=False,
            )
        if not hasattr(self, "group_idx"):
            self.register_buffer(
                "group_idx",
                torch.tensor(
                    restype_atom14_to_rigid_group,
                    device=device,
                    requires_grad=False,
                ),
                persistent=False,
            )
        if not hasattr(self, "atom_mask"):
            self.register_buffer(
                "atom_mask",
                torch.tensor(
                    restype_atom14_mask,
                    dtype=float_dtype,
                    device=device,
                    requires_grad=False,
                ),
                persistent=False,
            )
        if not hasattr(self, "lit_positions"):
            self.register_buffer(
                "lit_positions",
                torch.tensor(
                    restype_atom14_rigid_group_positions,
                    dtype=float_dtype,
                    device=device,
                    requires_grad=False,
                ),
                persistent=False,
            )

    def torsion_angles_to_frames(self, r, alpha, f):
        # Lazily initialize the residue constants on the correct device
        self._init_residue_constants(alpha.dtype, alpha.device)
        # Separated purely to make testing less annoying
        return torsion_angles_to_frames(r, alpha, f, self.default_frames)

    def frames_and_literature_positions_to_atom14_pos(
            self, r, f  # [*, N, 8]  # [*, N]
    ):
        # Lazily initialize the residue constants on the correct device
        self._init_residue_constants(r.get_rots().dtype, r.get_rots().device)
        return frames_and_literature_positions_to_atom14_pos(
            r,
            f,
            self.default_frames,
            self.group_idx,
            self.atom_mask,
            self.lit_positions,
        )

from torch import Tensor
from typing import Tuple

def dihedral(p):
    p0 = p[0]
    p1 = p[1]
    p2 = p[2]
    p3 = p[3]

    b0 = -1.0*(p1 - p0)
    b1 = p2 - p1
    b2 = p3 - p2

    b0xb1 = torch.cross(b0, b1, dim=-1)
    b1xb2 = torch.cross(b2, b1, dim=-1)

    b0xb1_x_b1xb2 = torch.cross(b0xb1, b1xb2, -1)

    y = torch.sum(b0xb1_x_b1xb2 * b1, dim=-1) *(1.0/torch.linalg.norm(b1, dim=-1))
    x = torch.sum(b0xb1 * b1xb2, dim=-1)

    return torch.nan_to_num(torch.arctan2(y, x))

def get_chi_angles(aa, pos, atom_mask):
    mask = chi_mask.to(aa.device)

    # If one-hot, convert to index representation
    if len(aa.shape) > 1:
        aa = torch.argmax(aa, dim=-1)

    mask = mask[aa]
    atom_indices = chi_true_indices.to(aa.device)
    chi_indices = atom_indices[aa].long()
    c1, c2, c3, c4 = torch.unbind(chi_indices,1)

    # TODO: find way to parallelize this
    chi1 = pos[c1==1].view(pos.shape[0],4,3)
    chi2 = pos[c2==1].view(pos.shape[0],4,3)
    chi3 = pos[c3==1].view(pos.shape[0],4,3)
    chi4 = pos[c4==1].view(pos.shape[0],4,3)

    # if any atomic coordinates are missing
    chi1_mask = atom_mask[c1==1].view(pos.shape[0],4).sum(-1) == 4
    chi2_mask = atom_mask[c2 == 1].view(pos.shape[0], 4).sum(-1) == 4
    chi3_mask = atom_mask[c3 == 1].view(pos.shape[0], 4).sum(-1) == 4
    chi4_mask = atom_mask[c4 == 1].view(pos.shape[0], 4).sum(-1) == 4

    atom_missing_mask = torch.stack([chi1_mask, chi2_mask, chi3_mask, chi4_mask],dim=-1)
    all_mask = atom_missing_mask * mask

    chi1 = dihedral(torch.unbind(chi1, 1))
    chi2 = dihedral(torch.unbind(chi2, 1))
    chi3 = dihedral(torch.unbind(chi3, 1))
    chi4 = dihedral(torch.unbind(chi4, 1))

    chi_angles = torch.stack([chi1, chi2, chi3, chi4],dim=-1) * all_mask

    return chi_angles, all_mask

def get_bb_dihedral(N: Tensor, CA: Tensor, C: Tensor) -> Tuple[Tensor, ...]:
    """
    Gets backbone dihedrals for
    :param N: (n,3) or (b,n,3) tensor of backbone Nitrogen coordinates
    :param CA: (n,3) or (b,n,3) tensor of backbone C-alpha coordinates
    :param C: (n,3) or (b,n,3) tensor of backbone Carbon coordinates
    :return: phi, psi, and omega dihedrals angles (each of shape (n,) or (b,n))
    """
    assert all([len(N.shape) == len(x.shape) for x in (CA, C)])
    phi, psi, omega = [torch.zeros(N.shape[0], device=N.device) for _ in range(3)]
    phi[1:] = dihedral([C[:-1], N[1:], CA[1:], C[1:]])
    psi[:-1] = dihedral([N[:-1], CA[:-1], C[:-1], N[1:]])
    omega[:-1] = dihedral([CA[:-1], C[:-1], N[1:], CA[1:]])

    return torch.stack([omega, phi, psi],dim=-1)