import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from pathlib import Path
import multiprocessing
import logging
import numpy as np
import random
from tqdm.contrib.concurrent import process_map

import warnings
import biotite.structure as struc
from biotite.structure.io.pdb import PDBFile
from biotite.structure.io.pdbx import PDBxFile, get_structure

from utils.constants import three_to_one_letter, letter_to_num, max_num_heavy_atoms, \
    restype_to_heavyatom_names, heavyatom_to_label, chi_alt_truths, num_to_letter, chi_true_indices, chi_mask, atom_types, atom_type_num

from utils.sidechain_utils import get_bb_dihedral, get_chi_angles

from torch_geometric.data import Data, DataLoader
from torch_cluster import radius_graph, knn_graph
import math
from scipy.spatial.distance import cdist
from tqdm.auto import tqdm
import pandas as pd
import shutil

class ProteinDataset(Dataset):
    def __init__(self, sample_path, gt_path, min_length=40, max_length=512, edge_type='radius', max_radius=8.0, max_num_neighbors=30,
                 scale_coords=1.0, filter_length=True, test=False, **kwargs):
        self.sample_path = Path(sample_path)
        self.gt_path = Path(gt_path)
        self.min_length = min_length
        self.max_length = max_length
        self.edge_type = edge_type
        self.max_radius = max_radius
        self.max_num_neighbors = max_num_neighbors
        self.scale_coords = scale_coords
        self.filter_length = filter_length

        # Ignore biotite warnings
        warnings.filterwarnings("ignore", ".*elements were guessed from atom_.*")

        # get gt pdbs
        pdbs = []
        for p in self.gt_path.iterdir():
            pdbs.append(p.stem)
        pdbs = set(pdbs)

        # get number of runs
        self.run_paths = [p for p in self.sample_path.iterdir() if p.is_dir()]

        for p in self.run_paths:
            sample_pdbs = set([i.stem for i in p.iterdir()])
            pdbs = pdbs.intersection(sample_pdbs)

        self.unique_pdbs = list(pdbs)
        # UNCOMMENT BELOW IF THERES SHAPE ISSUES WITH SAMPLES
        # self.unique_pdbs = list(process_map(self.filter_pdbs, pdbs, chunksize=10))
        # self.unique_pdbs = [i for i in self.unique_pdbs if i]

        print(f'Loaded {len(self.unique_pdbs)} pdbs containing {len(self.run_paths)} samples each...')

    def filter_pdbs(self, pdb):
        # check structures to make sure if compatible
        all_runs = True
        for p in self.run_paths:
            structure = self.to_tensor(self.get_features(self.gt_path.joinpath(f'{pdb}.cif')))
            sample = self.to_tensor(self.get_features(p.joinpath(f'{pdb}.pdb')))
            coords = structure['coord']
            aa_str = structure['aa']
            aa_num = torch.LongTensor([letter_to_num.get(i, 20) for i in aa_str])

            # There seems to be an issue with some pdbs where missing coordinates are just duplicated as previous residue's
            # coordinates - remove them since they dont work with equiformerv2 when computing edge_vec
            pairwise_dist = torch.cdist(coords[:, 1], coords[:, 1])
            x, y = torch.triu_indices(len(coords), len(coords))
            pairwise_dist[x, y] = 9999
            x, y = torch.nonzero(pairwise_dist < 0.01, as_tuple=True)
            if (x != y).sum() > 0:
                duplicate_mask = torch.ones(coords.shape[0], dtype=bool)
                duplicate_mask[x[x != y]] = 0
                aa_num = aa_num[duplicate_mask]
                aa_str = ''.join([num_to_letter[i.item()] for i in aa_num])

            if aa_str != sample['aa']:
                all_runs = False
                # shutil.move(p.joinpath(f'{pdb}.pdb'), p.parent.joinpath(f'{p.stem}_{pdb}.pdb'))

        if all_runs:
            return pdb
        else:
            print(pdb)
            return None

    def get_features(self, path):
        try:
            if path.suffix == ".cif":
                with open(path, "r") as f:
                    structure = PDBxFile.read(f)
                    structure = get_structure(structure)
            else:
                with open(path, "r") as f:
                    structure = PDBFile.read(f)
                    structure = structure.get_structure()
        except:
            return None

        # if struc.get_chain_count(structure) > 1: return None # only single chains

        _, aa = struc.get_residues(structure)
        # Replace nonstandard amino acids with X
        for idx, a in enumerate(aa):
            if a not in three_to_one_letter.keys():
                aa[idx] = 'UNK'

        aa_str = [three_to_one_letter.get(i,'X') for i in aa]
        aa_num = [letter_to_num[i] for i in aa_str]

        # if len(aa_str) > self.max_length or len(aa_str) < self.min_length:
        #     return None
        # if len(aa_str) < self.min_length: return None

        aa_mask = np.ones(len(aa))
        atom14_mask = np.zeros((len(aa), max_num_heavy_atoms))
        atom37_mask = np.zeros((len(aa), atom_type_num))
        # Iterate through all residues
        coords, coords37, atom_type = [], [], []
        for res_idx, res in enumerate(struc.residue_iter(structure)):
            res_coords = res.coord[0]
            res_name = aa[res_idx]

            if res_name == "UNK":
                aa_mask[res_idx] = 0

            # Append true coords
            res_crd14 = np.zeros((max_num_heavy_atoms, 3))
            res_crd37 = np.zeros((atom_type_num, 3))
            res_atom_type = []
            for atom14_idx, r in enumerate(restype_to_heavyatom_names[res_name]):
                if r == '':
                    res_atom_type.append(4)
                    continue
                atom37_idx = atom_types.index(r)
                res_atom_type.append(heavyatom_to_label[r[0]])
                i = np.where(res.atom_name == r)[0]
                if i.size == 0:
                    res_crd14[atom14_idx] = 0
                    res_crd37[atom37_idx] = 0

                else:
                    res_crd14[atom14_idx] = res_coords[i[0]]
                    atom14_mask[res_idx, atom14_idx] = 1
                    res_crd37[atom37_idx] = res_coords[i[0]]
                    atom37_mask[res_idx, atom37_idx] = 1
            coords.append(res_crd14)
            coords37.append(res_crd37)
            atom_type.append(res_atom_type)

        coords = np.array(coords)
        atom_type = np.array(atom_type)
        aa_num = np.array(aa_num)

        assert len(coords) == len(aa_num)

        return {
            "coord": coords,
            "atom_type": atom_type,
            "aa": aa_str,
            "mask": aa_mask,
            "atom_mask": atom14_mask,
        }

    def to_tensor(self, d, exclude=[]):
        feat_dtypes = {
            "coord": torch.float32,
            "atom_type": torch.long,
            "aa": None,
            "mask": torch.long,
            "atom_mask": torch.long,
        }

        for x in exclude:
            del d[x]

        for k,v in d.items():
            if feat_dtypes[k] is not None:
                d[k] = torch.tensor(v).to(dtype=feat_dtypes[k])

        return d


    def __getitem__(self, idx):
        random_pdb = random.choice(self.unique_pdbs)
        random_run = random.choice(self.run_paths)

        structure = self.to_tensor(self.get_features(self.gt_path.joinpath(f'{random_pdb}.cif')))
        sample = self.to_tensor(self.get_features(random_run.joinpath(f'{random_pdb}.pdb')))
        sample_crds = sample['coord']

        coords = structure['coord']
        aa_str = structure['aa']
        atom_mask = structure['atom_mask']
        aa_mask = structure['mask']
        aa_num = torch.LongTensor([letter_to_num.get(i, 20) for i in aa_str])
        atom_type = structure['atom_type']

        # There seems to be an issue with some pdbs where missing coordinates are just duplicated as previous residue's
        # coordinates - remove them since they dont work with equiformerv2 when computing edge_vec
        pairwise_dist = torch.cdist(coords[:,1], coords[:,1])
        x,y = torch.triu_indices(len(coords),len(coords))
        pairwise_dist[x,y] = 9999
        x, y = torch.nonzero(pairwise_dist < 0.01, as_tuple=True)
        if (x != y).sum() > 0:
            duplicate_mask = torch.ones(coords.shape[0], dtype=bool)
            duplicate_mask[x[x != y]] = 0
            coords = coords[duplicate_mask]
            aa_num = aa_num[duplicate_mask]
            atom_mask = atom_mask[duplicate_mask]
            aa_str = ''.join([num_to_letter[i.item()] for i in aa_num])
            aa_mask = aa_mask[duplicate_mask]
            atom_type = atom_type[duplicate_mask]
            # sample_crds = sample_crds[duplicate_mask]

        origin = coords[:,:4].reshape(-1, 3).mean(0) # CoM of backbone atoms
        coords = (coords - origin.unsqueeze(0)) * atom_mask.unsqueeze(-1)
        sample_crds = (sample_crds - origin.unsqueeze(0)) * atom_mask.unsqueeze(-1)

        # aa to tensor
        seq_onehot = F.one_hot(aa_num,num_classes=21).float()

        bb_dihedral = get_bb_dihedral(coords[:,0], coords[:,1], coords[:,2])
        chi_angles, chi_mask = get_chi_angles(aa_num, coords, atom_mask)
        chi_alt_mask = chi_alt_truths[aa_num] == 1
        chi_angles[chi_alt_mask] = ((chi_angles[chi_alt_mask] + math.pi) % math.pi) - math.pi
        chi_alt_angles = chi_angles.clone()
        # first move to [0, 2pi] and then add pi and then back to [-pi, pi] - this seems unnecessarily convoluted
        chi_alt_angles[chi_alt_mask] = ((chi_angles[chi_alt_mask] + (2*math.pi)) % (2*math.pi)) - math.pi

        sample_chi, _ = get_chi_angles(aa_num, sample_crds, atom_mask)
        sample_chi[chi_alt_mask] = ((sample_chi[chi_alt_mask] + math.pi) % math.pi) - math.pi

        # mask unknown residues
        chi_mask = chi_mask * aa_mask.unsqueeze(-1)
        chi_angles = chi_angles * chi_mask
        chi_alt_angles = chi_alt_angles * chi_mask
        sample_chi = sample_chi * chi_mask

        # edge index
        ca = coords[:,1]
        if self.edge_type == 'radius':
            edge_index = radius_graph(ca, r=self.max_radius, max_num_neighbors=self.max_num_neighbors)
        elif self.edge_type == 'knn':
            edge_index = knn_graph(ca, k=self.max_num_neighbors)
        else: raise NotImplementedError('wrong edge type')

        # edge features
        # edge_feat = get_edge_features(coords, edge_index)
        edge_feat = None

        data = Data(edge_index=edge_index, aa_str=aa_str, aa_num=aa_num, aa_onehot=seq_onehot, id=random_pdb,
                 pos=coords, edge_attr=edge_feat, aa_mask=aa_mask, bb_dihedral=bb_dihedral, chi=chi_angles,
                    chi_alt=chi_alt_angles, chi_mask=chi_mask, atom_mask=atom_mask, chi_alt_mask=chi_alt_mask,
                    atom_type=atom_type, sample_chi=sample_chi)

        return data

    def __len__(self):
        return len(self.unique_pdbs)

def _rbf(D, num_rbf=16):
    device = D.device
    D_min, D_max, D_count = 0., 16., num_rbf
    D_mu = torch.linspace(D_min, D_max, D_count, device=device)
    D_mu = D_mu.view([1, -1])
    D_sigma = (D_max - D_min) / D_count
    D_expand = torch.unsqueeze(D, -1)
    RBF = torch.exp(-((D_expand - D_mu) / D_sigma) ** 2)
    return RBF

def get_edge_features(X, edge_index, atom_mask=None, all_atoms=False, max_radius=16.0):
    edge_src, edge_dst = edge_index

    if all_atoms:
        atom_list = torch.unbind(X, 1)
    else:
        b = X[:, 1, :] - X[:, 0, :]
        c = X[:, 2, :] - X[:, 1, :]
        a = torch.cross(b, c, dim=-1)
        Cb = -0.58273431 * a + 0.56802827 * b - 0.54067466 * c + X[:, 1, :]
        Ca = X[:, 1, :]
        N = X[:, 0, :]
        C = X[:, 2, :]
        O = X[:, 3, :]

        atom_list = [N, Ca, C, O, Cb]

    rbf_all = []
    for idx1, i in enumerate(atom_list):
        for idx2, j in enumerate(atom_list):
            if idx1 == idx2: continue
            dist = torch.linalg.norm(i[edge_src] - j[edge_dst], dim=-1)
            rbf = _rbf(dist)

            # mask out non-existing atoms
            if atom_mask is not None:
                mask_src = atom_mask[edge_src, idx1]
                mask_dst = atom_mask[edge_dst, idx2]
                mask = mask_src * mask_dst # torch.logical_and
                rbf = rbf * mask.unsqueeze(-1)

            rbf_all.append(rbf)
    rbf_all = torch.cat(rbf_all, dim=-1)

    relpos = torch.clamp(edge_src - edge_dst, min=-32, max=32) + 32
    relpos = F.one_hot(relpos, num_classes=65).float()

    edge_feat = torch.cat([rbf_all, relpos], dim=-1)
    return edge_feat

def get_dataloader(config, sample=False, ddp=False):
    if not sample:
        train_ds = ProteinDataset(**config.data)

    batch_size = config.train.batch_size if not sample else config.sample.batch_size

    if ddp:
        from torch.utils.data.distributed import DistributedSampler
        train_sampler = DistributedSampler(train_ds)
        test_sampler = DistributedSampler(train_ds)
        train_dl = DataLoader(train_ds, batch_size=batch_size, sampler=train_sampler)
        test_dl = DataLoader(train_ds, batch_size=batch_size, sampler=test_sampler)
        return train_dl, test_dl, train_sampler, test_sampler
    else:
        if not sample:
            train_dl = DataLoader(train_ds, batch_size=batch_size, num_workers=0, shuffle=True)
        else:
            train_dl = None
        test_dl = DataLoader(train_ds, batch_size=batch_size, num_workers=0, shuffle=True)
        return train_dl, test_dl, None, None
