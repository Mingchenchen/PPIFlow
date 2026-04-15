import torch
import math
from utils.constants import heavyatom_to_label, vdw_tensor

def rmsd(pred, target, mask):
    return ((pred-target).square().sum() / mask.sum()).sqrt()

def angle_ae(pred, target):
    ae = torch.abs(pred - target)
    ae_alt = torch.abs(ae - 2*math.pi)
    ae_min = torch.minimum(ae, ae_alt)
    return ae_min

def angle_mae(pred, target, target_alt, mask, deg=True):
    ae = angle_ae(pred, target)
    ae_alt = angle_ae(pred, target_alt)
    ae_min = torch.minimum(ae, ae_alt)
    mae = ((ae_min*mask).sum() / mask.sum())
    if deg:
        return mae * 180 / math.pi
    return mae

def angle_acc(pred, target, target_alt, mask, threshold=20):
    ae = angle_ae(pred, target)
    ae_alt = angle_ae(pred, target_alt)
    ae_min = torch.minimum(ae, ae_alt)
    acc = torch.logical_and(ae_min <= (threshold * math.pi / 180), mask == 1).sum() / mask.sum()
    return acc

def metrics_per_chi(pred, target, target_alt, chi_mask, threshold=20, deg=True):
    mae_d, acc_d = {}, {}
    for i in range(4):
        mae = angle_mae(pred[..., i], target[...,i], target_alt[...,i], chi_mask[...,i], deg=deg)
        acc = angle_acc(pred[..., i], target[...,i], target_alt[...,i], chi_mask[...,i], threshold=threshold)
        mae_d[f'chi{i+1}'] = mae.item()
        acc_d[f'chi{i+1}'] = acc.item()
    return mae_d, acc_d

# average atom rmsd per residue
def atom_rmsd(pred, target, mask):
    return (((pred-target)*mask[...,None]).square().sum(-1).sum(-1) / mask.sum(-1).clamp(min=1)).sqrt().mean().item()

def count_clashes(crds, atom_type, mask, threshold=0.6):
    crds = crds.view(-1, 3)
    dist = torch.cdist(crds, crds) # pairwise distances
    atom_type = atom_type.view(-1)
    mask = mask.view(-1)
    vdw = vdw_tensor.to(crds.device)[atom_type]
    vdw_pair = (vdw.unsqueeze(-1) + vdw.unsqueeze(0)) * threshold

    # generate masks
    mask_pair = mask.unsqueeze(-1) * mask.unsqueeze(0)
    # for simplicity we just mask out clashes within each residue - may underestimate number of clashes
    # use block diagonal matrix
    block = torch.ones(14,14)
    mask_bonded =  ~torch.block_diag(*[block for _ in range(int(crds.shape[0]/14))]).to(device=crds.device, dtype=torch.bool)
    # mask peptide bonds - mask adjacent C-N, which occurs every 14 atoms (N: 14n, C = 2+14n) with n = num residues
    # we must mask mat[2,14], mat[16,28], mat[30,42]... mat[(2+14n), (14+14n)]
    x = torch.arange(2, crds.shape[0]-14, step=14, device=crds.device)
    y = torch.arange(14, crds.shape[0], step=14, device=crds.device)
    mask_bonded[x,y] = 0
    mask_bonded[y,x] = 0

    assert mask_bonded.shape == mask_pair.shape
    all_mask = mask_pair * mask_bonded

    clashes = (dist < vdw_pair) * all_mask
    clashes = torch.triu(clashes, diagonal=1).sum() # mask out redundancies

    return clashes.item()



