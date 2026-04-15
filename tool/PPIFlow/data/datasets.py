import abc
import math

import numpy as np
import pandas as pd
import logging
import tree
import torch
import random

from torch.utils.data import Dataset
from data import utils as du
from core.data import data_transforms
from core.utils import rigid_utils
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import torch.nn.functional as F

from motif_scaffolding import save_motif_segments
from experiments.utils import get_sampled_mask

torch.manual_seed(0)

def _rog_filter(df, quantile):
    y_quant = pd.pivot_table(
        df,
        values='radius_gyration',
        index='modeled_seq_len',
        aggfunc=lambda x: np.quantile(x, quantile)
    )
    x_quant = y_quant.index.to_numpy()
    y_quant = y_quant.radius_gyration.to_numpy()

    # Fit polynomial regressor
    poly = PolynomialFeatures(degree=4, include_bias=True)
    poly_features = poly.fit_transform(x_quant[:, None])
    poly_reg_model = LinearRegression()
    poly_reg_model.fit(poly_features, y_quant)

    # Calculate cutoff for all sequence lengths
    max_len = df.modeled_seq_len.max()
    pred_poly_features = poly.fit_transform(np.arange(max_len)[:, None])
    # Add a little more.
    pred_y = poly_reg_model.predict(pred_poly_features) + 0.1

    row_rog_cutoffs = df.modeled_seq_len.map(lambda x: pred_y[x - 1])
    return df[df.radius_gyration < row_rog_cutoffs]


def _length_filter(data_csv, min_res, max_res):
    return data_csv[
        (data_csv.modeled_seq_len >= min_res)
        & (data_csv.modeled_seq_len <= max_res)
        ]


def _plddt_percent_filter(data_csv, min_plddt_percent):
    return data_csv[data_csv.num_confident_plddt > min_plddt_percent]


def _max_coil_filter(data_csv, max_coil_percent):
    return data_csv[data_csv.coil_percent <= max_coil_percent]


def _data_source_filter(data_csv):
    data_csv['source_is_ddi'] = data_csv['processed_path'].apply(lambda x: 'domain_domain' in x)
    return data_csv[data_csv.source_is_ddi == False]


def _crop_long_motif(input_list):
    if len(input_list) <= 20:
        return input_list, []
    n = random.randint(3, 20)
    max_start = len(input_list) - n
    start = random.randint(0, max_start)
    cropped = input_list[start: start + n]
    complement = input_list[:start] + input_list[start + n:]
    return cropped, complement


def _sample_complex_motif_mask(feats, motif_cfg):
    if motif_cfg.define_motif:
        #sample all motif
        sample_index = list(range(len(feats['binder_motif'])))
    else:
        #sample 1~n-1 -> 1-2
        if len(feats['binder_motif'])==1:
            binder_motif_num = 1
        else:
            binder_motif_num = random.sample(range(1, 3), 1)[0]   # *
        perm_indices = torch.randperm(len(feats['binder_motif']))   # *
        sample_index = []
        for perm_idx in perm_indices:
            if len(feats['binder_motif'][perm_idx]) < 3:
                continue
            sample_index.append(perm_idx)
            if len(sample_index) >= binder_motif_num:
                break

        if binder_motif_num == 1:
            sample_index = [sample_index]

    binder_motif = []
    no_binder_motif = []
    for i in range(len(feats['binder_motif'])):
        if i in sample_index:
            if len(feats['binder_motif'][i]) > 20:
                cropped, complement = _crop_long_motif(feats['binder_motif'][i])
                binder_motif.extend(cropped)
                no_binder_motif.extend(complement)
            else:
                binder_motif.extend(feats['binder_motif'][i])
        else:
            no_binder_motif.extend(feats['binder_motif'][i])

    binder_motif_mask = np.isin(feats['residue_index'], binder_motif)

    return binder_motif_mask, no_binder_motif


def _get_new_hotspot_interface(feats, no_binder_motif, binder_id):
    # get remaining target interface residues after selecting some motifs on binder interface
    b_pair_idx = 3 if binder_id=='L' else 2
    t_pair_idx = 2 if binder_id=='L' else 3
    new_contact_pairs = [pair for pair in feats['contact_pairs'] if pair[b_pair_idx] in no_binder_motif]
    hotspot_interface_residues = [pair[t_pair_idx] for pair in new_contact_pairs]
    hotspot_interface_mask = np.isin(feats['residue_index'], hotspot_interface_residues)

    return hotspot_interface_mask


def _sample_monomer_motif_mask(batch, monomer_motif_cfg):
    """
    monomer motif sample strategy    # * original function: def _sample_scaffold_mask
    monomer_motif_cfg:  self._dataset_cfg.motif
    hyperparams include:
    min_motif_percent:
        Minimum percentage of residues (out of the total sequence length
        of the input structure) to be defined as motif residues.
    max_motif_percent:
        Maximum percentage of residues (out of the total sequence length
        of the input structure) to be defined as motif residues.
    motif_min_n_seg:
        Minimum number of motif segments.
    motif_max_n_seg:
        Maximum number of motif segments.
    """
    # Sample number of motif residues
    num_res = batch['residue_index'].shape[0]
    motif_n_res = np.random.randint(
        np.floor(num_res * monomer_motif_cfg.min_motif_percent),
        np.ceil(num_res * monomer_motif_cfg.max_motif_percent)
    )
    motif_n_res = max(motif_n_res, 3)

    # Sample number of motif segments
    motif_n_seg = np.random.randint(
        monomer_motif_cfg.motif_min_n_seg,
        min(monomer_motif_cfg.motif_max_n_seg, motif_n_res) + 2
    )

    if motif_n_seg >= motif_n_res:
        motif_n_seg = motif_n_seg - 1

    # Sample motif segments
    indices = sorted(np.random.choice(motif_n_res - 1, motif_n_seg - 1, replace=False) + 1)
    indices = [0] + indices + [motif_n_res]
    motif_seg_lens = [indices[i + 1] - indices[i] for i in range(motif_n_seg)]

    # Generate motif mask
    segs = [''.join(['1'] * l) for l in motif_seg_lens]
    segs.extend(['0'] * (num_res - motif_n_res))
    random.shuffle(segs)
    motif_mask = torch.tensor([int(elt) for elt in ''.join(segs)], dtype=torch.int)
    scaffold_mask = 1 - motif_mask
    return scaffold_mask * batch['bb_mask']


def _process_csv_row(csv_row, motif_cfg=None):
    processed_file_path = csv_row['processed_path']
    processed_feats = du.read_pkl(processed_file_path)
    processed_feats = du.parse_chain_feats(processed_feats, normalize_positions=True)

    binder_interface_label = 'binder_interface_residues'
    target_interface_label = 'target_interface_residues'

    # binder motif mask
    if (csv_row['num_chains'] == 2) and ('binder_motif' in processed_feats) and motif_cfg:
        binder_id = csv_row['binder_id']
        binder_motif_mask, no_binder_motif = _sample_complex_motif_mask(processed_feats, motif_cfg)
        hotspot_interface_mask = _get_new_hotspot_interface(processed_feats, no_binder_motif, binder_id)
    else:
        binder_motif_mask = None
        hotspot_interface_mask = None

    # monomer motif mask
    if (csv_row['num_chains'] == 1) and motif_cfg:
        monomer_motif_mask = _sample_monomer_motif_mask(processed_feats, motif_cfg)
    else:
        monomer_motif_mask = None
    processed_feats['monomer_motif_mask'] = monomer_motif_mask

    # reset chain index as binder=1 target=0, or all=0 if it is a monomer
    new_chain_idx = np.zeros_like(processed_feats['residue_index'])
    if csv_row['num_chains'] == 2:
        if pd.isnull(csv_row['binder_id']):
            binder_chain_id = -1
        else:
            binder_chain_id = du.chain_str_to_int(csv_row['binder_id'])
        binder_chain_index = np.nonzero(processed_feats['chain_index'] == binder_chain_id)[0]
        new_chain_idx[binder_chain_index] = 1
    processed_feats['chain_index'] = new_chain_idx

    # set target_interface_residues mask
    if csv_row['num_chains'] == 2:
        target_interface_mask = np.isin(processed_feats['residue_index'], processed_feats[target_interface_label])
        binder_interface_mask = np.isin(processed_feats['residue_index'], processed_feats[binder_interface_label])
        target_interface_mask = target_interface_mask * (new_chain_idx == 0)
        binder_interface_mask = binder_interface_mask * (new_chain_idx == 1)
    else:
        target_interface_mask = np.zeros_like(processed_feats['residue_index'])
        binder_interface_mask = np.zeros_like(processed_feats['residue_index'])
    target_interface_mask = torch.tensor(target_interface_mask, dtype=int)
    binder_interface_mask = torch.tensor(binder_interface_mask, dtype=int)
    processed_feats['target_interface_mask'] = target_interface_mask
    processed_feats['binder_interface_mask'] = binder_interface_mask

    # set binder motif mask
    if (csv_row['num_chains'] == 2) and ('binder_motif' in processed_feats) and motif_cfg:
        binder_motif_mask = binder_motif_mask * (new_chain_idx == 1)
        binder_motif_mask = torch.tensor(binder_motif_mask, dtype=int)
        hotspot_interface_mask = hotspot_interface_mask * (new_chain_idx == 0)
        hotspot_interface_mask = torch.tensor(hotspot_interface_mask, dtype=int)
    processed_feats['hotspot_interface_mask'] = hotspot_interface_mask
    processed_feats['binder_motif_mask'] = binder_motif_mask

    # leave out non-tensor data
    processed_feats = {k: v for k, v in processed_feats.items() if
                       k not in ['target_interface_residues', 'binder_interface_residues', 'contact_pairs', 'contig', 'length', 'binder_motif']}

    # Only take modeled residues.
    modeled_idx = processed_feats.pop('modeled_idx')
    if len(modeled_idx) != len(processed_feats['bb_mask']):
        processed_feats = tree.map_structure(lambda x: x[modeled_idx] if x is not None else None, processed_feats)

    # Run through OpenFold data transforms.
    chain_feats = {
        'aatype': torch.tensor(processed_feats['aatype']).long(),  # values range from 0 to 19, 0 is ALA
        'all_atom_positions': torch.tensor(processed_feats['atom_positions']).double(),
        'all_atom_mask': torch.tensor(processed_feats['atom_mask']).double()
    }
    chain_feats = data_transforms.atom37_to_frames(chain_feats)
    rigids_1 = rigid_utils.Rigid.from_tensor_4x4(chain_feats['rigidgroups_gt_frames'])[:, 0]
    rotmats_1 = rigids_1.get_rots().get_rot_mats()  # torch.Size([256, 3, 3])
    trans_1 = rigids_1.get_trans()  # torch.Size([256, 3])
    res_plddt = processed_feats['b_factors'][:, 1]  # (256,)
    res_mask = torch.tensor(processed_feats['bb_mask']).int()  # torch.Size([256])

    # reset residue index
    # Re-number residue indices for each chain such that it starts from 1.
    # Randomize chain indices.
    if csv_row['num_chains'] == 1:
        chain_idx = processed_feats['chain_index']
        res_idx = processed_feats['residue_index']
        new_res_idx = np.zeros_like(res_idx)
        new_chain_idx = np.zeros_like(res_idx)
        all_chain_idx = np.unique(chain_idx).tolist()
        assert len(all_chain_idx) == 1
        shuffled_chain_idx = np.array(
            random.sample(all_chain_idx, len(all_chain_idx))) - np.min(all_chain_idx) + 1
        for i, chain_id in enumerate(all_chain_idx):
            chain_mask = (chain_idx == chain_id).astype(int)
            chain_min_idx = np.min(res_idx + (1 - chain_mask) * 1e3).astype(int)
            new_res_idx = new_res_idx + (res_idx - chain_min_idx + 1) * chain_mask
            replacement_chain_id = shuffled_chain_idx[i]
            new_chain_idx = new_chain_idx + replacement_chain_id * chain_mask
        processed_feats['chain_idx'] = new_chain_idx
    else:
        assert csv_row['num_chains'] == 2
        new_res_idx = np.zeros_like(processed_feats['residue_index'])
        for i, chain_id in enumerate([0, 1]):
            chain_mask = (processed_feats['chain_index'] == chain_id).astype(int)
            chain_min_idx = np.min(processed_feats['residue_index'] + (1 - chain_mask) * 1e3).astype(int)
            new_res_idx = new_res_idx + (processed_feats['residue_index'] - chain_min_idx + 1) * chain_mask
            if chain_id == 0:
                target_max_res_idx = np.max(new_res_idx * chain_mask)
            else:
                new_res_idx = new_res_idx + (target_max_res_idx * chain_mask)

    output_feats = {
        'res_plddt': res_plddt,
        'aatype': chain_feats['aatype'],
        'rotmats_1': rotmats_1,
        'trans_1': trans_1,
        'res_mask': res_mask,
        'chain_idx': torch.tensor(processed_feats['chain_index']),
        'res_idx': torch.tensor(new_res_idx),
        'original_res_idx': torch.tensor(processed_feats['residue_index']),
        'target_interface_mask': processed_feats['target_interface_mask'],
        'binder_interface_mask': processed_feats['binder_interface_mask'],
        'hotspot_interface_mask': processed_feats['hotspot_interface_mask'],
        'binder_motif_mask': processed_feats['binder_motif_mask'],
        'monomer_motif_mask': processed_feats['monomer_motif_mask'],
        'all_atom_positions': chain_feats['all_atom_positions'],
        'all_atom_mask': chain_feats['all_atom_mask'],
    }
    output_feats = {k: v for k, v in output_feats.items() if v is not None}

    if torch.isnan(trans_1).any() or torch.isnan(rotmats_1).any():
        raise ValueError(f'Found NaNs in {processed_file_path}')
    return output_feats


def _add_plddt_mask(feats, plddt_threshold):
    feats['plddt_mask'] = torch.tensor(
        feats['res_plddt'] > plddt_threshold).int()


def _read_clusters(cluster_path):
    pdb_to_cluster = {}
    with open(cluster_path, "r") as f:
        for i, line in enumerate(f):
            for chain in line.split(' '):
                pdb = chain.split('_')[0]
                pdb_to_cluster[pdb.upper()] = i
    return pdb_to_cluster


class BaseDataset(Dataset):
    def __init__(
            self,
            *,
            dataset_cfg,
            is_training,
            task,
    ):
        self._log = logging.getLogger(__name__)
        self._is_training = is_training
        self._dataset_cfg = dataset_cfg
        self.task = task
        self.raw_csv = pd.read_csv(self.dataset_cfg.csv_path)
        metadata_csv = self._filter_metadata(self.raw_csv)
        metadata_csv = metadata_csv.sort_values(
            'modeled_seq_len', ascending=False)
        self._create_split(metadata_csv)
        self._cache = {}
        self._rng = np.random.default_rng(seed=self._dataset_cfg.seed)

    @property
    def is_training(self):
        return self._is_training

    @property
    def dataset_cfg(self):
        return self._dataset_cfg

    def __len__(self):
        return len(self.csv)

    @abc.abstractmethod
    def _filter_metadata(self, raw_csv: pd.DataFrame) -> pd.DataFrame:
        pass

    def process_csv_row(self, csv_row):
        path = csv_row['processed_path']
        seq_len = csv_row['modeled_seq_len']
        # Large protein files are slow to read. Cache them.
        use_cache = seq_len > self._dataset_cfg.cache_num_res
        if use_cache and path in self._cache:
            return self._cache[path]
        processed_row = _process_csv_row(csv_row, motif_cfg=self.motif_cfg)
        if processed_row is None:
            return None
        processed_row['pdb_name'] = csv_row['pdb_name']
        # processed_row['original_index'] = f"{csv_row['original_index']}_{csv_row['target_seq_len']}_{csv_row['binder_seq_len']}"
        processed_row['original_index'] = f"{csv_row['original_index']}"
        if use_cache:
            self._cache[path] = processed_row
        return processed_row

    def _sample_hotspot_mask(self, feats):
        mask_label = 'target_interface_mask'
        if torch.sum(feats[mask_label] == 1).item()==0:
            hotspot_mask = torch.zeros_like(feats[mask_label])
        else:
            target_interface_index = torch.nonzero(feats[mask_label] == 1).reshape(-1)
            if self._dataset_cfg.define_hotspots:
                hotspot_index = torch.range(0, target_interface_index.shape[0] - 1, dtype=torch.long)
            else:
                try:
                    hotspot_num = self._rng.integers(
                        low=math.ceil(target_interface_index.shape[0] * self._dataset_cfg.min_hotspot_ratio),
                        high=math.ceil(target_interface_index.shape[0] * self.dataset_cfg.max_hotspot_ratio),
                        size=(1,)
                    )
                except:
                    hotspot_num = self._rng.integers(
                        low=math.ceil(target_interface_index.shape[0] * self._dataset_cfg.min_hotspot_ratio),
                        high=math.ceil(target_interface_index.shape[0] * self.dataset_cfg.max_hotspot_ratio)+1,
                        size=(1,)
                    )
                hotspot_num = max(hotspot_num.item(), self._dataset_cfg.samples_min_hotspots)
                hotspot_index = torch.randperm(target_interface_index.shape[0])[:hotspot_num]
            hotspot_mask = torch.zeros(feats[mask_label].shape)
            hotspot_mask[target_interface_index[hotspot_index]] = 1
            hotspot_mask = hotspot_mask.int()
        return hotspot_mask

    def setup_target_hotspots(self, feats):
        hotspot_mask = self._sample_hotspot_mask(feats)
        return hotspot_mask

    def setup_binder_mask(self, feats):
        '''
            target chain set as diffuse_mask=0, binder chain set as diffuse_mask=1
            hotspots are retrieved from or sampled among given residues
        '''
        target_chain_id = 0
        # diffuse_index = np.nonzero(feats['chain_idx'] != target_chain_id)[0] #**
        diffuse_index = torch.nonzero(feats['chain_idx'] != target_chain_id).reshape(-1)
        diffuse_mask = torch.zeros(feats['chain_idx'].shape, dtype=torch.int)
        diffuse_mask[diffuse_index] = 1  # 0:motif, 1:to diffuse
        if torch.sum(diffuse_mask) < 1:
            diffuse_mask = torch.ones_like(diffuse_mask)
        diffuse_mask = diffuse_mask.int()
        return diffuse_mask

    def setup_binder_mask_with_motif(self, feats):
        '''
            target chain as well as binder motif set as diffuse_mask=0, binder remaining parts set as diffuse_mask=1
        '''

        target_chain_id = 0
        diffuse_mask = torch.where(
            (feats['chain_idx'] != target_chain_id) & (feats['binder_motif_mask'] != 1),
            1, 0
        )
        if torch.sum(diffuse_mask) < 1:
            diffuse_mask = torch.ones_like(diffuse_mask)
        diffuse_mask = diffuse_mask.int()
        return diffuse_mask

    def setup_monomer_inpainting(self, feats):
        diffuse_mask = torch.where(
            feats['monomer_motif_mask'] != 1,
            1, 0
        )
        if torch.sum(diffuse_mask) < 1:
            diffuse_mask = torch.ones_like(diffuse_mask)
        return diffuse_mask

    def post_process_feats(self, feats):
        # Center based on motif locations
        motif_mask = 1 - feats['diffuse_mask']  # 1:motif 0:to diffuse
        trans_1 = feats['trans_1']
        motif_1 = trans_1 * motif_mask[:, None]
        motif_com = torch.sum(motif_1, dim=0) / (torch.sum(motif_mask) + 1)
        trans_1 -= motif_com[None, :]
        feats['trans_1'] = trans_1
        return feats

    def __getitem__(self, row_idx):
        while True:
            # Process data example.
            csv_row = self.csv.iloc[row_idx]
            feats = self.process_csv_row(csv_row)
            if feats is not None:
                if csv_row['num_chains'] == 1:   # selected item is monomer
                    if np.random.rand() < 0.5 and ('monomer_motif_mask' in feats):
                        feats['diffuse_mask'] = self.setup_monomer_inpainting(feats)
                    else:
                        diffuse_mask = torch.ones_like(feats['res_mask'])
                        feats['diffuse_mask'] = diffuse_mask.int()
                    hotspot_mask = torch.zeros_like(feats['res_mask'])
                    feats['hotspot_mask'] = hotspot_mask.int()
                else:
                    assert csv_row['num_chains'] == 2   # selected item is complex
                    if self.task == 'binder_motif' and (np.random.rand() < 0.33):
                        feats['diffuse_mask'] = self.setup_binder_mask_with_motif(feats)
                    else:
                        feats['diffuse_mask'] = self.setup_binder_mask(feats)

                    hotspot_mask = self.setup_target_hotspots(feats)
                    feats['hotspot_mask'] = hotspot_mask
                    feats = self.post_process_feats(feats)

                feats['csv_idx'] = torch.ones(1, dtype=torch.long) * row_idx
                return feats

            fail_row_idx = row_idx
            row_idx = np.random.randint(0, len(self.csv))
            print(f'warning: dataloader of row {fail_row_idx} failed, skip; use {row_idx} instead')


class PpiDataset(BaseDataset):
    def __init__(
            self,
            *,
            dataset_cfg,
            is_training,
            task,
            is_additional_val=False
    ):
        self._log = logging.getLogger(__name__)
        self._is_training = is_training
        self._dataset_cfg = dataset_cfg
        self.task = task
        self.motif_cfg = self._dataset_cfg.motif if self.task=='binder_motif' else None
        self._cache = {}
        self._rng = np.random.default_rng(seed=self._dataset_cfg.seed)
        if self._is_training == True:
            csv_path = self.dataset_cfg.train_csv_path
            datatype = "Train"
        else:
            if is_additional_val:
                csv_path = self.dataset_cfg.val_csv_path1
            else:
                csv_path = self.dataset_cfg.val_csv_path
            datatype = "Val"

        # Process clusters
        self.csv = pd.read_csv(csv_path)
        self.csv = self._filter_metadata(self.csv)
        self.csv.reset_index(drop=False, names=['original_index'], inplace=True)
        self.csv['index'] = list(range(len(self.csv)))
        self._log.info(f'{datatype} data num: {len(self.csv)}')

    def _filter_metadata(self, raw_csv):
        data_csv = raw_csv
        filter_cfg = self.dataset_cfg.filter
        if filter_cfg.activate:
            data_csv = data_csv[data_csv['data_level'] == 1]
            if not self._is_training:
                print(f"train filter_cfg.max_complex_len:{filter_cfg.max_complex_len_val}")
                data_csv = data_csv[data_csv['seq_len'] <= filter_cfg.max_complex_len_val]
            else:
                print(f"validation filter_cfg.max_complex_len:{filter_cfg.max_complex_len}")
                data_csv = data_csv[data_csv['seq_len'] <= filter_cfg.max_complex_len]
            data_csv = data_csv[data_csv['homo_dimer_rate'] <= filter_cfg.max_homo_dimer_rate]
            data_csv = _max_coil_filter(data_csv, filter_cfg.max_coil_percent)
            data_csv = data_csv[data_csv['binder_seq_len'] <= filter_cfg.max_binder_len]
            data_csv = data_csv[data_csv['target_seq_len'] >= filter_cfg.min_target_len]

            if not self._is_training:
                data_csv = data_csv[data_csv['binder_seq_len'] <= data_csv['target_seq_len']]

        set_type = 'train' if self._is_training else 'val'
        print(f"(ppi) raw {set_type} data {len(raw_csv)},after filter data num: {len(data_csv)}, cluster: {len(data_csv['cluster'].unique())}.")

        return data_csv


class Ppi_Monomer_Dataset(BaseDataset):

    def __init__(
            self,
            *,
            dataset_cfg,
            is_training,
            task,
            is_additional_val=False
    ):
        self._log = logging.getLogger(__name__)
        self._is_training = is_training
        self._dataset_cfg = dataset_cfg
        self.task = task
        self.motif_cfg = self._dataset_cfg.motif if self.task == 'binder_motif' else None

        self._cache = {}
        self._rng = np.random.default_rng(seed=self._dataset_cfg.seed)
        if self._is_training == True:
            csv_path = self.dataset_cfg.train_csv_path
            datatype = "Train"
        else:
            if is_additional_val:
                csv_path = self.dataset_cfg.val_csv_path1
            else:
                csv_path = self.dataset_cfg.val_csv_path
            datatype = "Val"

        # Process clusters
        self.csv = pd.read_csv(csv_path)
        self.csv = self._filter_metadata(self.csv)
        self.csv.reset_index(drop=False, names=['original_index'], inplace=True)
        self.csv['index'] = list(range(len(self.csv)))
        self._log.info(f'{datatype} data num: {len(self.csv)}')

    def _filter_metadata(self, raw_csv):
        ppi_csv = self._filter_metadata_ppi(raw_csv)
        monomer_csv = self._filter_metadata_monomer(raw_csv)
        result_csv = pd.concat([monomer_csv, ppi_csv])
        print(f"raw data {len(raw_csv)},after filter data num: {len(result_csv)}")
        return result_csv

    def _filter_metadata_ppi(self, raw_csv):
        """Filter metadata ppi."""
        data_csv = raw_csv[raw_csv['num_chains'] == 2]
        filter_cfg = self.dataset_cfg.filter
        if filter_cfg.activate:
            data_csv = data_csv[data_csv['data_level'] == 1]
            if not self._is_training:
                print(f"filter_cfg.max_complex_len:{filter_cfg.max_complex_len_val}")
                data_csv = data_csv[data_csv['seq_len'] <= filter_cfg.max_complex_len_val]
            else:
                print(f"filter_cfg.max_complex_len:{filter_cfg.max_complex_len}")
                data_csv = data_csv[data_csv['seq_len'] <= filter_cfg.max_complex_len]
            data_csv = data_csv[data_csv['homo_dimer_rate'] <= filter_cfg.max_homo_dimer_rate]
            data_csv = _max_coil_filter(data_csv, filter_cfg.max_coil_percent)
            data_csv = data_csv[data_csv['binder_seq_len'] <= filter_cfg.max_binder_len]
            data_csv = data_csv[data_csv['target_seq_len'] >= filter_cfg.min_target_len]

            if not self._is_training:
                data_csv = data_csv[data_csv['binder_seq_len'] <= data_csv['target_seq_len']]
        set_type = 'train' if self._is_training else 'val'
        print(f"(ppi_monomer) raw {set_type} ppi data {len(raw_csv)},after filter data num: {len(data_csv)}, cluster: {len(data_csv['cluster'].unique())}.")
        return data_csv

    def _filter_metadata_monomer(self, raw_csv):
        """Filter metadata monomer."""
        filter_cfg = self.dataset_cfg.filter
        data_csv = raw_csv[raw_csv['num_chains'] == 1]
        len_0 = len(data_csv)
        if len_0 > 0:
            data_csv = data_csv[data_csv['seq_len'] <= filter_cfg.max_monomer_len]
            data_csv = _max_coil_filter(data_csv, filter_cfg.max_coil_percent)
            set_type = 'train' if self._is_training else 'val'
            n_uniq_clusts = len(data_csv['cluster'].unique())
            print(f"raw {set_type} monomer data {len(raw_csv)},after filter data num: {len(data_csv)}, cluster: {n_uniq_clusts}.")
        n_uniq_clusts = len(data_csv['cluster'].unique())
        print(f'(ppi_monomer) use monomer data: {len(data_csv)}, cluster: {n_uniq_clusts} ')
        return data_csv


class PpiTestDataset(BaseDataset):
    def __init__(
            self,
            *,
            dataset_cfg,
            task,
    ):
        self._log = logging.getLogger(__name__)
        self._dataset_cfg = dataset_cfg
        self.task = task
        self._cache = {}
        self._rng = np.random.default_rng(seed=self._dataset_cfg.seed)
        csv_path = self.dataset_cfg.test_csv_path
        datatype = "Test"
        self.motif_cfg = self._dataset_cfg.motif if self.task == 'binder_motif' else None
        self.csv = pd.read_csv(csv_path)
        self.csv.reset_index(drop=False, names=['original_index'], inplace=True)
        self.csv['index'] = list(range(len(self.csv)))
        self._log.info(f'{datatype} data num: {len(self.csv)}')

        self.batch_size = self.dataset_cfg.samples_batch_size
        self.samples_per_target = dataset_cfg.samples_per_target
        assert self.samples_per_target % self.batch_size == 0
        self.n_samples = self.samples_per_target // self.batch_size

        self.binder_len = self._rng.integers(
            low=self._dataset_cfg.samples_min_length,
            high=self._dataset_cfg.samples_max_length,
            size=(self.n_samples,)
        )
        self.binder_len = np.repeat(self.binder_len, self.batch_size)

        all_sample_ids = []
        for row_id in range(len(self.csv)):
            target_row = self.csv.iloc[row_id]
            for sample_id, blen in zip(range(self.samples_per_target), self.binder_len):
                all_sample_ids.append((target_row, sample_id, blen))
        self.all_sample_ids = all_sample_ids

    def _filter_metadata(self, raw_csv):
        """Filter metadata."""
        filter_cfg = self.dataset_cfg.filter
        data_csv = raw_csv[raw_csv['target_seq_len'] <= filter_cfg.max_target_num_res]
        data_csv = data_csv[(data_csv['binder_seq_len'] <= filter_cfg.max_binder_num_res) & (
                    data_csv['binder_seq_len'] >= filter_cfg.min_binder_num_res)]
        print(f"raw data {len(raw_csv)}, after filter data num: {len(data_csv)}")
        return data_csv

    def __len__(self):
        return len(self.all_sample_ids)

    def __getitem__(self, row_idx):
        # Process data example.
        csv_row, sample_ids, binder_len = self.all_sample_ids[row_idx]
        feats = self.process_csv_row(csv_row)

        pdb_diffuse_mask = self.setup_binder_mask(feats)
        feats['diffuse_mask'] = pdb_diffuse_mask.int()
        hotspot_mask = self.setup_target_hotspots(feats)
        feats['hotspot_mask'] = hotspot_mask.int()

        feats = self.post_process_feats(feats)

        target_index = np.nonzero(pdb_diffuse_mask == 0)[:, 0]
        target_feats = {
            k: v[target_index] for k, v in feats.items() if k not in ['pdb_name', 'original_index']
        }
        if self._dataset_cfg.sample_original_binder_len == True:
            binder_len = int(sum(pdb_diffuse_mask).item())
        target_len = len(target_index)
        total_length = target_len + binder_len
        diffuse_mask = torch.ones(total_length, dtype=torch.int)
        hotspot_mask = torch.zeros(total_length, dtype=torch.int)
        trans_1 = torch.zeros(total_length, 3)  # torch.Size([total_length, 3])
        rotmats_1 = torch.eye(3)[None].repeat(total_length, 1, 1)  # torch.Size([total_length, 3, 3])
        aatype = torch.zeros(total_length, dtype=torch.int64)  # torch.Size([total_length])
        res_mask = torch.ones(total_length, dtype=torch.int)
        chain_idx = torch.ones(total_length, dtype=torch.int)
        target_interface_mask = torch.zeros(total_length)

        trans_1[:target_len] = target_feats['trans_1']
        rotmats_1[:target_len] = target_feats['rotmats_1']
        aatype[:target_len] = target_feats['aatype']
        aatype[target_len:] = 21    # 20
        diffuse_mask[:target_len] = 0
        hotspot_mask[:target_len] = target_feats['hotspot_mask']
        res_mask[:target_len] = target_feats['res_mask']
        chain_idx[:target_len] = target_feats['chain_idx']
        target_interface_mask[:target_len] = target_feats['target_interface_mask']

        output_feats = {
            'diffuse_mask': diffuse_mask,
            'hotspot_mask': hotspot_mask,
            'trans_1': trans_1,
            'rotmats_1': rotmats_1,
            'aatype': aatype.long(),
            'pdb_name': csv_row['pdb_name'],
            'original_index': csv_row['original_index'],
            'sample_ids': sample_ids,
            'res_mask': res_mask,
            'chain_idx': chain_idx,
            'target_interface_mask': target_interface_mask
        }
        return output_feats


class PpiScaffoldingTestDataset(BaseDataset):
    def __init__(self, dataset_cfg, task):

        self._dataset_cfg = dataset_cfg
        self._benchmark_df = pd.read_csv(self._dataset_cfg.test_csv_path)
        self.task = task
        self.motif_cfg = self._dataset_cfg.motif if self.task=='binder_motif' else None

        self._rng = np.random.default_rng(seed=self._dataset_cfg.seed)
        self._log = logging.getLogger(__name__)
        self._cache = {}
        if self.dataset_cfg.sample_pdbname is not None:
            self.dataset_cfg.sample_pdbname = [self.dataset_cfg.sample_pdbname] if isinstance(
                self.dataset_cfg.sample_pdbname, str) else self.dataset_cfg.sample_pdbname
            self._benchmark_df = self._benchmark_df.loc[self._benchmark_df['id'].isin(self.dataset_cfg.sample_pdbname)]
        self._benchmark_df.reset_index(drop=False, names=['original_index'], inplace=True)
        self._benchmark_df['index'] = list(range(len(self._benchmark_df)))

        self.batch_size = self._dataset_cfg.samples_batch_size
        self.samples_per_target = self._dataset_cfg.samples_per_target
        assert self.samples_per_target % self.batch_size == 0
        self.n_samples = self.samples_per_target // self.batch_size

        self.binder_len = self._rng.integers(
            low=self._dataset_cfg.samples_min_length,
            high=self._dataset_cfg.samples_max_length,
            size=(self.n_samples,)
        )
        self.binder_len = np.repeat(self.binder_len, self.batch_size)

        all_sample_ids = []
        for row_id in range(len(self._benchmark_df)):
            target_row = self._benchmark_df.iloc[row_id]
            for sample_id, blen in zip(range(self.samples_per_target), self.binder_len):
                all_sample_ids.append((target_row, sample_id, blen))
        self.all_sample_ids = all_sample_ids

    def _get_new_hotspot_interface(self, feats, source_segments):
        binder_motif = []
        for segment in source_segments:
            binder_motif.extend(list(range(segment[0], segment[1]+1)))

        b_pair_idx = 3
        t_pair_idx = 2
        new_contact_pairs = [pair for pair in feats['contact_pairs'] if pair[b_pair_idx] not in binder_motif]
        hotspot_interface_residues = [pair[t_pair_idx] for pair in new_contact_pairs]
        hotspot_interface_mask = torch.isin(feats['original_res_idx'], torch.tensor(hotspot_interface_residues))*(1-feats['diffuse_mask'])
        return hotspot_interface_mask

    def __len__(self):
        return len(self.all_sample_ids)

    def __getitem__(self, idx):
        # Process data example.
        csv_row, sample_ids, binder_len = self.all_sample_ids[idx]
        feats = self.process_csv_row(csv_row)

        pdb_diffuse_mask = self.setup_binder_mask(feats)
        feats['diffuse_mask'] = pdb_diffuse_mask.int()
        feats = self.post_process_feats(feats)

        #############motif mask###############
        if isinstance(csv_row.sample_binder_len, str):
            lengths = csv_row.sample_binder_len.split('-')
            if len(lengths) == 1:
                start_length = lengths[0]
                end_length = lengths[0]
            else:
                start_length, end_length = lengths
            sample_lengths = [int(start_length), int(end_length)+1]
        else:
            sample_lengths = None
        sample_contig, sampled_binder_length, _ = get_sampled_mask(csv_row.contig, sample_lengths)
        motif_locations, original_source_segments = save_motif_segments.motif_locations_from_contig(sample_contig[0])  #[(18, 37), (53, 72)]; [(29, 48), (106, 127)]
        source_segments = []
        for segments in original_source_segments:
            original_binder_motif = torch.tensor(list(range(segments[0], segments[-1]+1)))
            binder_motif = torch.nonzero(torch.isin(feats['original_res_idx']*feats['diffuse_mask'], original_binder_motif), as_tuple=True)[0]
            source_segments.append((binder_motif[0].item(), binder_motif[-1].item()))

        #######################################
        ################hotspot_mask###########
        feat_tmp = du.read_pkl(csv_row['processed_path'])
        feats['contact_pairs'] = feat_tmp['contact_pairs']
        del feat_tmp
        feats['hotspot_interface_mask'] = self._get_new_hotspot_interface(feats, original_source_segments)
        feats['hotspot_mask'] = self.setup_target_hotspots(feats)
        #######################################

        target_index = np.nonzero(pdb_diffuse_mask == 0)[:, 0]
        binder_index = np.nonzero(pdb_diffuse_mask == 1)[:, 0]
        target_feats = {
            k: v[target_index] for k, v in feats.items() if k not in ['pdb_name', 'original_index', 'contact_pairs']
        }

        target_len = len(target_index)
        total_length = target_len + sampled_binder_length

        diffuse_mask = torch.ones(total_length, dtype=torch.int)
        hotspot_mask = torch.zeros(total_length, dtype=torch.int)
        trans_1 = torch.zeros(total_length, 3)  # torch.Size([total_length, 3])
        rotmats_1 = torch.eye(3)[None].repeat(total_length, 1, 1)  # torch.Size([total_length, 3, 3])
        aatype = torch.zeros(total_length, dtype=torch.int64)  # torch.Size([total_length])
        res_mask = torch.ones(total_length, dtype=torch.int)
        chain_idx = torch.ones(total_length, dtype=torch.int)
        target_interface_mask = torch.zeros(total_length, dtype=torch.int)
        binder_motif_mask = torch.zeros(total_length, dtype=torch.int)

        # target locations
        diffuse_mask[:target_len] = 0
        hotspot_mask[:target_len] = target_feats['hotspot_mask']
        trans_1[:target_len] = target_feats['trans_1']
        rotmats_1[:target_len] = target_feats['rotmats_1']
        aatype[:target_len] = target_feats['aatype']
        aatype[target_len:] = 20   # *
        chain_idx[:target_len] = target_feats['chain_idx']
        target_interface_mask[:target_len] = target_feats['target_interface_mask']
        binder_motif_mask[:target_len] = 0

        # motif locations
        for generate_location, true_res_interval in zip(motif_locations, source_segments):
            start, end = generate_location
            j, k = true_res_interval
            start = target_len + start
            end = target_len + end
            diffuse_mask[start:end+1] = 0
            trans_1[start:end+1] = feats['trans_1'][j:k+1]
            rotmats_1[start:end+1] = feats['rotmats_1'][j:k+1]
            aatype[start:end+1] = feats['aatype'][j:k+1]
            binder_motif_mask[start:end+1] = 1

        output_feats = {
            'diffuse_mask': diffuse_mask,
            'hotspot_mask': hotspot_mask,
            'trans_1': trans_1,
            'rotmats_1': rotmats_1,
            'aatype': aatype,
            'pdb_name': csv_row['pdb_name'],
            'original_index': sample_ids,
            'sample_ids': sample_ids,
            'res_mask': res_mask,
            'chain_idx': chain_idx,
            'target_interface_mask': target_interface_mask,
            'binder_motif_mask' : binder_motif_mask
        }
        output_feats = self.post_process_feats(output_feats)
        return output_feats

