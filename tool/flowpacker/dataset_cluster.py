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
from biotite.structure.io.pdbx import CIFFile, get_structure

from utils.constants import three_to_one_letter, letter_to_num, max_num_heavy_atoms, \
    restype_to_heavyatom_names, heavyatom_to_label, chi_alt_truths, num_to_letter, chi_true_indices, chi_mask, atom_types, atom_type_num

from utils.sidechain_utils import get_bb_dihedral, get_chi_angles

from torch_geometric.data import Data, DataLoader
from torch_cluster import radius_graph, knn_graph
import math
from scipy.spatial.distance import cdist
from tqdm.auto import tqdm
import pandas as pd


class ProteinDataset(Dataset):
    def __init__(self, dataset_path, cluster_path, min_length=40, max_length=512, edge_type='radius', max_radius=8.0, max_num_neighbors=30,
                 scale_coords=1.0, filter_length=True, test=False, **kwargs):
        self.data_path = Path(dataset_path)
        self.cluster_path = cluster_path
        self.min_length = min_length
        self.max_length = max_length
        self.edge_type = edge_type
        self.max_radius = max_radius
        self.max_num_neighbors = max_num_neighbors
        self.scale_coords = scale_coords
        self.filter_length = filter_length

        # Ignore biotite warnings
        warnings.filterwarnings("ignore", ".*elements were guessed from atom_.*")

        if cluster_path is not None and not test:
            cluster_info = pd.read_csv(cluster_path, sep='\t', header=None).to_dict(orient='records')
            mem_to_rep = {}
            structures = []
            for i in cluster_info:
                cluster_rep = i[0]
                cluster_mem = i[1]
                mem_to_rep[cluster_mem] = cluster_rep
                structures.append(cluster_mem)

            unique_reps = list(np.unique(list(mem_to_rep.values())))
            self.clusters = {i:[] for i in unique_reps}
        else:
            structures = [i.stem for i in self.data_path.iterdir()]
            mem_to_rep = {i:i for i in structures}
            self.clusters = {i:[] for i in mem_to_rep.keys()}

        pdb_datapath = [i.stem for i in self.data_path.iterdir()]

        num_struct = 0
        for s in tqdm(structures):
            if s not in pdb_datapath: continue
            # # check structures
            # try:
            #     path = torch.load(self.data_path.joinpath(f'{s}.pth'))
            # except FileNotFoundError:
            #     try:
            #         if self.data_path.joinpath(f'{s}.pdb').exists():
            #             path = self.data_path.joinpath(f'{s}.pdb')
            #             with open(path, "r") as f:
            #                 structure = PDBFile.read(f)
            #                 structure = structure.get_structure()
            #         elif self.data_path.joinpath(f'{s}.cif').exists():
            #             path = self.data_path.joinpath(f'{s}.cif')
            #             with open(path, "r") as f:
            #                 structure = PDBxFile.read(f)
            #                 structure = get_structure(structure)
            #     except Exception:
            #         continue
            #
            #     if struc.get_chain_count(structure) > 1: continue
            #     _, aa = struc.get_residues(structure)
            #     if len(aa) < 40 or len(aa) > 512: continue
            cluster_rep = mem_to_rep[s]
            self.clusters[cluster_rep].append(s)
            num_struct += 1

        # remove clusters not found in structures - mainly for debugging
        cluster_copy = self.clusters.copy()
        for k,v in cluster_copy.items():
            if not v:
                self.clusters.pop(k)

        self.num_to_rep_id = {idx:i for idx,i in enumerate(self.clusters.keys())}

        print(f'Loaded {len(self.clusters)} clusters containing {num_struct} structures...')

    def parse_pdb(self, paths):
        logging.info(
            f"Computing full dataset of {len(paths)} with {multiprocessing.cpu_count()} threads"
        )
        data = list(process_map(self.get_features, paths, chunksize=100))

        return data

    def get_features(self, path):
        try:
            if path.suffix == ".cif":
                with open(path, "r") as f:
                    structure = CIFFile.read(f)
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
        coords, coords37, atom_type, chain_ids, res_id, icode = [], [], [], [], [], []
        for res_idx, res in enumerate(struc.residue_iter(structure)):
            res_coords = res.coord[0]
            res_name = aa[res_idx]
            chain_ids.append(res.chain_id[0])
            res_id.append(res.res_id[0])
            icode.append(res.ins_code[0])

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
        chain_ids = np.array(chain_ids)
        res_id = np.array(res_id)
        icode = np.array(icode)

        assert len(coords) == len(aa_num)

        return {
            "coord": coords,
            "atom_type": atom_type,
            "aa": aa_str,
            "mask": aa_mask,
            "atom_mask": atom14_mask,
            "chain_id": chain_ids,
            "res_id": res_id,
            "icode": icode
        }

    def to_tensor(self, d, exclude=[]):
        feat_dtypes = {
            "coord": torch.float32,
            "atom_type": torch.long,
            "aa": None,
            "mask": torch.long,
            "atom_mask": torch.long,
            "chain_id": None,
            "res_id": None,
            "icode": None
        }

        for x in exclude:
            del d[x]

        for k,v in d.items():
            if feat_dtypes[k] is not None:
                d[k] = torch.tensor(v).to(dtype=feat_dtypes[k])

        return d

    def __getitem__(self, idx):
        rep_id = self.num_to_rep_id[idx]
        pdb_id = random.choice(self.clusters[rep_id])
        try:
            structure = self.to_tensor(torch.load(self.data_path.joinpath(f'{pdb_id}.pth')))
        except FileNotFoundError:
            if self.data_path.joinpath(f'{pdb_id}.pdb').exists():
                structure = self.to_tensor(self.get_features(self.data_path.joinpath(f'{pdb_id}.pdb')))
            elif self.data_path.joinpath(f'{pdb_id}.cif').exists():
                structure = self.to_tensor(self.get_features(self.data_path.joinpath(f'{pdb_id}.cif')))
            else:
                raise FileNotFoundError(f'{pdb_id} pdb file not found')

        coords = structure['coord']
        aa_str = structure['aa']
        atom_mask = structure['atom_mask']
        aa_mask = structure['mask']
        aa_num = torch.LongTensor([letter_to_num.get(i, 20) for i in aa_str])
        atom_type = structure['atom_type']
        chain_id = structure['chain_id']
        res_id, icode = structure['res_id'], structure['icode']

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
            chain_id = chain_id[duplicate_mask]
            res_id = res_id[duplicate_mask]
            icode = icode[duplicate_mask]

        origin = coords[:,:4].reshape(-1, 3).mean(0) # CoM of backbone atoms
        coords = (coords - origin.unsqueeze(0)) * atom_mask.unsqueeze(-1)

        # aa to tensor
        seq_onehot = F.one_hot(aa_num,num_classes=21).float()

        if self.filter_length:
            if len(coords) > self.max_length:
                rand_start = random.randint(0, len(coords) - self.max_length)
                rand_end = rand_start + self.max_length
                coords = coords[rand_start:rand_end]
                aa_num = aa_num[rand_start:rand_end]
                aa_str = aa_str[rand_start:rand_end]
                atom_mask = atom_mask[rand_start:rand_end]
                aa_mask = aa_mask[rand_start:rand_end]
                seq_onehot = seq_onehot[rand_start:rand_end]
                atom_type = atom_type[rand_start:rand_end]
                chain_id = chain_id[rand_start:rand_end]
                res_id = res_id[rand_start:rand_end]
                icode = icode[rand_start:rand_end]

        bb_dihedral = get_bb_dihedral(coords[:,0], coords[:,1], coords[:,2])
        chi_angles, chi_mask = get_chi_angles(aa_num, coords, atom_mask)
        chi_alt_mask = chi_alt_truths[aa_num] == 1
        chi_angles[chi_alt_mask] = ((chi_angles[chi_alt_mask] + math.pi) % math.pi) - math.pi
        chi_alt_angles = chi_angles.clone()
        # first move to [0, 2pi] and then add pi and then back to [-pi, pi] - this seems unnecessarily convoluted
        chi_alt_angles[chi_alt_mask] = ((chi_angles[chi_alt_mask] + (2*math.pi)) % (2*math.pi)) - math.pi

        # mask unknown residues
        chi_mask = chi_mask * aa_mask.unsqueeze(-1)
        chi_angles = chi_angles * chi_mask
        chi_alt_angles = chi_alt_angles * chi_mask

        # edge index
        ca = coords[:,1]
        if self.edge_type == 'radius':
            edge_index = radius_graph(ca, r=self.max_radius, max_num_neighbors=self.max_num_neighbors)
        elif self.edge_type == 'knn':
            edge_index = knn_graph(ca, k=self.max_num_neighbors)
        else: raise NotImplementedError('wrong edge type')

        edge_feat = None

        data = Data(edge_index=edge_index, aa_str=aa_str, aa_num=aa_num, aa_onehot=seq_onehot, id=pdb_id,
                 pos=coords, edge_attr=edge_feat, aa_mask=aa_mask, bb_dihedral=bb_dihedral, chi=chi_angles,
                    chi_alt=chi_alt_angles, chi_mask=chi_mask, atom_mask=atom_mask, chi_alt_mask=chi_alt_mask,
                    atom_type=atom_type, chain_id=chain_id, res_id=res_id, icode=icode)

        return data

    def __len__(self):
        return len(list(self.clusters.keys()))

def get_edge_features(X, edge_index, atom_mask=None, all_atoms=False, chain_index=None):
    edge_src, edge_dst = edge_index
    edge_feat = []
    relpos = torch.clamp(edge_src - edge_dst, min=-32, max=32) + 32

    # if chain_index is not None:



    relpos = F.one_hot(relpos, num_classes=65).float()
    edge_feat.append(relpos)

    if all_atoms and atom_mask is not None:
        X_src = X[edge_src]
        X_dst = X[edge_dst]
        mask_src = atom_mask[edge_src]
        mask_dst = atom_mask[edge_dst]

        dist = torch.cdist(X_src, X_dst) * mask_src.unsqueeze(-1) * mask_dst.unsqueeze(-2)
        dist = dist.view(-1, 196).clamp(max=12.)

        edge_feat.append(dist)

    edge_feat = torch.cat(edge_feat, dim=-1)
    return edge_feat

def get_dataloader(config, sample=False, ddp=False):
    if not sample:
        train_ds = ProteinDataset(dataset_path=config.data.train_path, **config.data)
    test_ds = ProteinDataset(dataset_path=config.data.test_path, **config.data, filter_length=False, test=True)

    batch_size = config.train.batch_size if not sample else config.sample.batch_size

    if ddp:
        from torch.utils.data.distributed import DistributedSampler
        train_sampler = DistributedSampler(train_ds)
        test_sampler = DistributedSampler(test_ds)
        train_dl = DataLoader(train_ds, batch_size=batch_size, sampler=train_sampler)
        test_dl = DataLoader(test_ds, batch_size=batch_size, sampler=test_sampler)
        return train_dl, test_dl, train_sampler, test_sampler
    else:
        if not sample:
            train_dl = DataLoader(train_ds, batch_size=batch_size, num_workers=0, shuffle=True)
        else:
            train_dl = None
        test_dl = DataLoader(test_ds, batch_size=batch_size, num_workers=0, shuffle=True)
        return train_dl, test_dl, None, None
