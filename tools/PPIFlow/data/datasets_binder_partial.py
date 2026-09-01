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
import itertools
import re
from collections import defaultdict

torch.manual_seed(0)


def _rog_filter(df, quantile):
    y_quant = pd.pivot_table(
        df,
        values="radius_gyration",
        index="modeled_seq_len",
        aggfunc=lambda x: np.quantile(x, quantile),
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
    data_csv["source_is_ddi"] = data_csv["processed_path"].apply(
        lambda x: "domain_domain" in x
    )
    return data_csv[data_csv.source_is_ddi == False]


def _crop_long_motif(input_list):
    """Crop motif if length > 20, otherwise return as-is."""
    if len(input_list) <= 20:
        return input_list, []
    n = random.randint(3, 20)
    max_start = len(input_list) - n
    start = random.randint(0, max_start)
    cropped = input_list[start : start + n]
    complement = input_list[:start] + input_list[start + n :]
    return cropped, complement


def _sample_complex_motif_mask(feats, motif_cfg):
    """Sample motif indices based on configuration."""
    if motif_cfg.define_motif:
        sample_index = list(range(len(feats["binder_motif"])))
    else:
        if len(feats["binder_motif"]) == 1:
            binder_motif_num = 1
        else:
            binder_motif_num = random.sample(range(1, 3), 1)[0]
        perm_indices = torch.randperm(len(feats["binder_motif"]))
        sample_index = []
        for perm_idx in perm_indices:
            if len(feats["binder_motif"][perm_idx]) < 3:
                continue
            sample_index.append(perm_idx)
            if len(sample_index) >= binder_motif_num:
                break

        if binder_motif_num == 1:
            sample_index = [sample_index]

    binder_motif = []
    for i in range(len(feats["binder_motif"])):
        if i in sample_index:
            if len(feats["binder_motif"][i]) > 20:
                cropped, complement = _crop_long_motif(
                    feats["binder_motif"][i]
                )
                binder_motif.extend(cropped)
            else:
                binder_motif.extend(feats["binder_motif"][i])

    no_binder_motif = list(
        set(feats["binder_interface_residues"]) - set(binder_motif)
    )
    binder_motif_mask = np.isin(feats["residue_index"], binder_motif)

    return binder_motif_mask, no_binder_motif


def _get_new_hotspot_interface(feats, no_binder_motif, binder_id):
    b_pair_idx = 3 if binder_id == "L" or binder_id == "A" else 2
    t_pair_idx = 2 if binder_id == "L" or binder_id == "A" else 3
    new_contact_pairs = [
        pair
        for pair in feats["contact_pairs"]
        if pair[b_pair_idx] in no_binder_motif
    ]
    hotspot_interface_residues = [
        pair[t_pair_idx] for pair in new_contact_pairs
    ]
    hotspot_interface_mask = np.isin(
        feats["residue_index"], hotspot_interface_residues
    )

    return torch.tensor(hotspot_interface_mask)


def _process_csv_row(csv_row, motif_cfg=None):
    """Process a single CSV row and extract features."""
    processed_file_path = csv_row["processed_path"]
    processed_feats = du.read_pkl(processed_file_path)
    processed_feats = du.parse_chain_feats(
        processed_feats, normalize_positions=True
    )

    binder_interface_label = "binder_interface_residues"
    target_interface_label = "target_interface_residues"
    binder_id = csv_row["binder_id"]
    original_binder_motif = (
        processed_feats["binder_motif"]
        if "binder_motif" in processed_feats
        else None
    )

    # Get target_interface_residues mask
    target_interface_mask = np.isin(
        processed_feats["residue_index"],
        processed_feats[target_interface_label],
    )
    binder_interface_mask = np.isin(
        processed_feats["residue_index"],
        processed_feats[binder_interface_label],
    )
    # Binder motif mask
    if ("binder_motif" in processed_feats) and motif_cfg:
        binder_motif_mask, no_binder_motif = _sample_complex_motif_mask(
            processed_feats, motif_cfg
        )
        hotspot_interface_mask = _get_new_hotspot_interface(
            processed_feats, no_binder_motif, binder_id
        )
    else:
        binder_motif_mask = None
        hotspot_interface_mask = None

    processed_feats = {
        k: v
        for k, v in processed_feats.items()
        if k
        not in [
            "target_interface_residues",
            "binder_interface_residues",
            "contact_pairs",
            "contig",
            "length",
            "binder_motif",
        ]
    }
    all_chain_idx, counts = np.unique(
        processed_feats["chain_index"], return_counts=True
    )
    new_chain_idx = np.zeros_like(processed_feats["residue_index"])

    # Reset chain index as binder=1 target=0
    if all_chain_idx.size == 2:
        if "binder_label" in csv_row.keys():
            binder_chain_id = du.chain_str_to_int(csv_row["binder_label"])
        else:
            binder_chain_id = du.chain_str_to_int(csv_row["binder_id"])
        binder_chain_index = np.nonzero(
            processed_feats["chain_index"] == binder_chain_id
        )[0]
        new_chain_idx[binder_chain_index] = 1

    target_interface_mask = target_interface_mask * (new_chain_idx == 0)
    binder_interface_mask = binder_interface_mask * (new_chain_idx == 1)
    target_interface_mask = torch.tensor(target_interface_mask, dtype=int)
    binder_interface_mask = torch.tensor(binder_interface_mask, dtype=int)

    if binder_motif_mask is not None and motif_cfg:
        binder_motif_mask = binder_motif_mask * (new_chain_idx == 1)
        binder_motif_mask = torch.tensor(binder_motif_mask, dtype=int)
        hotspot_interface_mask = hotspot_interface_mask * (
            new_chain_idx == 0
        )
        hotspot_interface_mask = torch.tensor(
            hotspot_interface_mask, dtype=int
        )
        processed_feats["binder_motif_mask"] = binder_motif_mask

    processed_feats["chain_index"] = new_chain_idx
    processed_feats["target_interface_mask"] = target_interface_mask
    processed_feats["binder_interface_mask"] = binder_interface_mask

    # Only take modeled residues.
    modeled_idx = processed_feats["modeled_idx"]
    min_idx = np.min(modeled_idx)
    max_idx = np.max(modeled_idx)
    del processed_feats["modeled_idx"]
    processed_feats = tree.map_structure(
        lambda x: x[min_idx : (max_idx + 1)], processed_feats
    )

    # Run through OpenFold data transforms.
    chain_feats = {
        "aatype": torch.tensor(
            processed_feats["aatype"]
        ).long(),
        "all_atom_positions": torch.tensor(
            processed_feats["atom_positions"]
        ).double(),
        "all_atom_mask": torch.tensor(
            processed_feats["atom_mask"]
        ).double(),
    }
    chain_feats = data_transforms.atom37_to_frames(chain_feats)
    rigids_1 = rigid_utils.Rigid.from_tensor_4x4(
        chain_feats["rigidgroups_gt_frames"]
    )[:, 0]
    rotmats_1 = (
        rigids_1.get_rots().get_rot_mats()
    )
    trans_1 = rigids_1.get_trans()
    res_plddt = processed_feats["b_factors"][:, 1]
    res_mask = torch.tensor(
        processed_feats["bb_mask"]
    ).int()

    # Reset residue index
    new_res_idx = np.zeros_like(processed_feats["residue_index"])
    for i, chain_id in enumerate([0, 1]):
        chain_mask = (processed_feats["chain_index"] == chain_id).astype(
            int
        )
        chain_min_idx = np.min(
            processed_feats["residue_index"] + (1 - chain_mask) * 1e3
        ).astype(int)
        new_res_idx = (
            new_res_idx
            + (processed_feats["residue_index"] - chain_min_idx + 1)
            * chain_mask
        )
        if chain_id == 0:
            target_max_res_idx = np.max(new_res_idx * chain_mask)
        else:
            new_res_idx = new_res_idx + (target_max_res_idx * chain_mask)

    output_feats = {
        "res_plddt": res_plddt,
        "aatype": chain_feats["aatype"],
        "rotmats_1": rotmats_1,
        "trans_1": trans_1,
        "res_mask": res_mask,
        "chain_idx": torch.tensor(new_chain_idx),
        "res_idx": torch.tensor(new_res_idx),
        "original_res_idx": torch.tensor(processed_feats["residue_index"]),
        "target_interface_mask": target_interface_mask,
        "binder_interface_mask": binder_interface_mask,
        "hotspot_interface_mask": hotspot_interface_mask,
        "binder_motif_mask": binder_motif_mask,
        "all_atom_positions": chain_feats["all_atom_positions"],
        "all_atom_mask": chain_feats["all_atom_mask"],
        "original_binder_motif": original_binder_motif,
    }
    output_feats = {k: v for k, v in output_feats.items() if v is not None}

    if torch.isnan(trans_1).any() or torch.isnan(rotmats_1).any():
        raise ValueError(f"Found NaNs in {processed_file_path}")
    return output_feats


def _add_plddt_mask(feats, plddt_threshold):
    feats["plddt_mask"] = torch.tensor(
        feats["res_plddt"] > plddt_threshold
    ).int()


def _read_clusters(cluster_path):
    pdb_to_cluster = {}
    with open(cluster_path, "r") as f:
        for i, line in enumerate(f):
            for chain in line.split(" "):
                pdb = chain.split("_")[0]
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
            "modeled_seq_len", ascending=False
        )
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

    def _create_split(self, data_csv):
        # Training or validation specific logic.
        if self.is_training:
            self.csv = data_csv
            self._log.info(f"Training: {len(self.csv)} examples")
        else:
            if self._dataset_cfg.max_eval_length is None:
                eval_lengths = data_csv.modeled_seq_len
            else:
                eval_lengths = data_csv.modeled_seq_len[
                    data_csv.modeled_seq_len
                    <= self._dataset_cfg.max_eval_length
                ]
            all_lengths = np.sort(eval_lengths.unique())
            length_indices = (len(all_lengths) - 1) * np.linspace(
                0.0, 1.0, self.dataset_cfg.num_eval_lengths
            )
            length_indices = length_indices.astype(int)
            eval_lengths = all_lengths[length_indices]
            eval_csv = data_csv[
                data_csv.modeled_seq_len.isin(eval_lengths)
            ]

            # Fix a random seed to get the same split each time.
            eval_csv = eval_csv.groupby("modeled_seq_len").sample(
                self.dataset_cfg.samples_per_eval_length,
                replace=True,
                random_state=123,
            )
            eval_csv = eval_csv.sort_values(
                "modeled_seq_len", ascending=False
            )
            self.csv = eval_csv
            self._log.info(
                f"Validation: {len(self.csv)} examples with lengths {eval_lengths}"
            )
        self.csv["index"] = list(range(len(self.csv)))

    def process_csv_row(self, csv_row):
        """Process a single CSV row with caching for large proteins."""
        path = csv_row["processed_path"]
        seq_len = csv_row["modeled_seq_len"]
        use_cache = seq_len > self._dataset_cfg.cache_num_res
        if use_cache and path in self._cache:
            return self._cache[path]
        processed_row = _process_csv_row(csv_row, motif_cfg=self.motif_cfg)
        if processed_row is None:
            return None
        processed_row["pdb_name"] = csv_row["pdb_name"]
        processed_row["original_index"] = f"{csv_row['original_index']}"
        if use_cache:
            self._cache[path] = processed_row
        return processed_row

    def _sample_hotspot_mask(self, feats):
        """Sample or define hotspot mask for training."""
        mask_label = (
            "hotspot_interface_mask"
            if "hotspot_interface_mask" in feats
            else "target_interface_mask"
        )
        mask_label = (
            "target_interface_mask"
            if self._dataset_cfg.define_hotspots == True
            else mask_label
        )

        if torch.sum(feats[mask_label] == 1).item() == 0:
            hotspot_mask = torch.zeros_like(feats[mask_label])
        else:
            target_interface_index = torch.nonzero(
                feats[mask_label] == 1
            ).reshape(-1)
            if self._dataset_cfg.define_hotspots:
                hotspot_index = torch.range(
                    0,
                    target_interface_index.shape[0] - 1,
                    dtype=torch.long,
                )
            else:
                try:
                    hotspot_num = self._rng.integers(
                        low=math.ceil(
                            target_interface_index.shape[0]
                            * self._dataset_cfg.min_hotspot_ratio
                        ),
                        high=math.ceil(
                            target_interface_index.shape[0]
                            * self.dataset_cfg.max_hotspot_ratio
                        ),
                        size=(1,),
                    )
                except:
                    hotspot_num = self._rng.integers(
                        low=math.ceil(
                            target_interface_index.shape[0]
                            * self._dataset_cfg.min_hotspot_ratio
                        ),
                        high=math.ceil(
                            target_interface_index.shape[0]
                            * self.dataset_cfg.max_hotspot_ratio
                        )
                        + 1,
                        size=(1,),
                    )
                hotspot_num = max(
                    hotspot_num.item(),
                    self._dataset_cfg.samples_min_hotspots,
                )
                hotspot_index = torch.randperm(
                    target_interface_index.shape[0]
                )[:hotspot_num]
            hotspot_mask = torch.zeros(feats[mask_label].shape)
            hotspot_mask[target_interface_index[hotspot_index]] = 1
            hotspot_mask = hotspot_mask * (1 - feats["diffuse_mask"])
            hotspot_mask = hotspot_mask.int()
        return hotspot_mask

    def setup_target_hotspots(self, feats):
        hotspot_mask = self._sample_hotspot_mask(feats)
        return hotspot_mask

    def setup_binder_mask(self, feats):
        """
        Set target chain as diffuse_mask=0, binder chain as diffuse_mask=1.
        Hotspots are retrieved from or sampled among given residues.
        """
        target_chain_id = 0
        diffuse_index = torch.nonzero(
            feats["chain_idx"] != target_chain_id
        ).reshape(-1)
        diffuse_mask = torch.zeros(
            feats["chain_idx"].shape, dtype=torch.int
        )
        diffuse_mask[diffuse_index] = 1
        if torch.sum(diffuse_mask) < 1:
            diffuse_mask = torch.ones_like(diffuse_mask)
        diffuse_mask = diffuse_mask.int()
        return diffuse_mask

    def setup_binder_mask_with_motif(self, feats):
        """
        Set target chain and binder motif as diffuse_mask=0,
        binder remaining parts as diffuse_mask=1.
        """
        target_chain_id = 0
        diffuse_mask = torch.where(
            (feats["chain_idx"] != target_chain_id)
            & (feats["binder_motif_mask"] != 1),
            1,
            0,
        )

        if torch.sum(diffuse_mask) < 1:
            diffuse_mask = torch.ones_like(diffuse_mask)
        diffuse_mask = diffuse_mask.int()
        return diffuse_mask

    def setup_rotamer(self, processed_feats, rotamer_cfg):
        """Setup rotamer features for training."""
        binder_interface_label = "binder_interface_residues"

        # Get binder_rotamer residues mask
        if rotamer_cfg.define_rotamers:
            sample_size = len(processed_feats[binder_interface_label])
        else:
            random_float = random.uniform(
                rotamer_cfg.min_rotamer_ratio,
                rotamer_cfg.max_rotamer_ratio,
            )
            sample_size = min(
                int(
                    len(processed_feats[binder_interface_label])
                    * random_float
                ),
                rotamer_cfg.samples_max_rotamer,
            )
        binder_r_points = random.sample(
            processed_feats[binder_interface_label], sample_size
        )
        binder_rotamer_mask = np.isin(
            processed_feats["residue_index"], binder_r_points
        )

        new_chain_idx = processed_feats["chain_idx"]
        binder_rotamer_mask = binder_rotamer_mask * (new_chain_idx == 1)

        if len(binder_rotamer_mask) != len(processed_feats["aatype"]):
            print(
                f"WARNING: binder_rotamer_mask and aatype features length not equal!  ",
                len(new_chain_idx),
                len(binder_rotamer_mask),
                len(processed_feats["aatype"]),
            )
            return None

        processed_feats["rotamer_aatype"] = processed_feats["aatype"][
            binder_rotamer_mask == 1
        ]
        processed_feats["rotamer_atom_positions"] = processed_feats[
            "atom_positions"
        ][binder_rotamer_mask == 1]
        processed_feats["rotamer_atom_mask"] = processed_feats[
            "atom_mask"
        ][binder_rotamer_mask == 1]

        rotamer_feats = {
            "aatype": torch.tensor(
                processed_feats["rotamer_aatype"]
            ).long(),
            "all_atom_positions": torch.tensor(
                processed_feats["rotamer_atom_positions"]
            ).double(),
            "all_atom_mask": torch.tensor(
                processed_feats["rotamer_atom_mask"]
            ).double(),
        }
        rotamer_feats = data_transforms.atom37_to_frames(rotamer_feats)
        rigids_rc = rigid_utils.Rigid.from_tensor_4x4(
            rotamer_feats["rigidgroups_gt_frames"]
        )[:, 0]
        rotmats_rc = (
            rigids_rc.get_rots().get_rot_mats()
        )
        trans_rc = rigids_rc.get_trans()
        aatype_rc = rotamer_feats["aatype"]
        rc_node_mask = torch.ones_like(aatype_rc, dtype=torch.float)
        pad_size = rotamer_cfg.samples_max_rotamer - len(
            aatype_rc
        )
        # Pad trans_rc to fixed size
        trans_rc = F.pad(trans_rc, (0, 0, 0, pad_size))
        # Pad rotmats_rc to fixed size
        rotmats_rc = F.pad(rotmats_rc, (0, 0, 0, 0, 0, pad_size))
        # Pad aatype_rc to fixed size
        aatype_rc = F.pad(aatype_rc, (0, pad_size))
        # Pad rc_node_mask to fixed size
        rc_node_mask = F.pad(rc_node_mask, (0, pad_size))
        rotamer_info = {
            "method": "gt_interface",
            "rotamers": binder_r_points,
            "rotamer_size": len(binder_r_points),
        }

        processed_feats.update(
            {
                "binder_rotamer_mask": binder_rotamer_mask,
                "aatype_rc": aatype_rc,
                "rotmats_rc": rotmats_rc,
                "trans_rc": trans_rc,
                "rc_node_mask": rc_node_mask,
                "rotamer_info": rotamer_info,
            }
        )

        return processed_feats

    def post_process_feats(self, feats):
        """Center coordinates based on motif locations."""
        motif_mask = 1 - feats["diffuse_mask"]
        trans_1 = feats["trans_1"]
        motif_1 = trans_1 * motif_mask[:, None]
        motif_com = torch.sum(motif_1, dim=0) / (
            torch.sum(motif_mask) + 1
        )
        trans_1 -= motif_com[None, :]
        feats["trans_1"] = trans_1

        return feats

    def __getitem__(self, row_idx):
        """Get a single data example."""
        while True:
            csv_row = self.csv.iloc[row_idx]
            feats = self.process_csv_row(csv_row)
            if feats is not None:
                if csv_row["num_chains"] == 1:
                    diffuse_mask = torch.ones_like(feats["res_mask"])
                    hotspot_mask = torch.zeros_like(feats["res_mask"])
                    feats["diffuse_mask"] = diffuse_mask.int()
                    feats["hotspot_mask"] = hotspot_mask.int()
                else:
                    if self.task == "binder_motif":
                        if np.random.rand() < 0.5:
                            feats["diffuse_mask"] = (
                                self.setup_binder_mask_with_motif(feats)
                            )
                        else:
                            feats["diffuse_mask"] = self.setup_binder_mask(
                                feats
                            )
                    else:
                        feats["diffuse_mask"] = self.setup_binder_mask(
                            feats
                        )

                    hotspot_mask = self.setup_target_hotspots(feats)
                    feats["hotspot_mask"] = hotspot_mask

                    feats = self.post_process_feats(feats)

                feats["csv_idx"] = (
                    torch.ones(1, dtype=torch.long) * row_idx
                )
                return feats

            fail_row_idx = row_idx
            row_idx = np.random.randint(0, len(self.csv))
            print(
                f"warning: dataloader of row {fail_row_idx} failed, skip; use {row_idx} instead"
            )


class PpiDataset(BaseDataset):
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
        self.motif_cfg = (
            self._dataset_cfg.motif
            if self.task == "binder_motif"
            else None
        )
        self._cache = {}
        self._rng = np.random.default_rng(seed=self._dataset_cfg.seed)
        if self.is_training == "test":  # partial
            csv_path = self.dataset_cfg.test_csv_path
            datatype = "test"
        elif self._is_training == True:
            csv_path = self.dataset_cfg.train_csv_path
            datatype = "Train"
        elif self._is_training == False:
            csv_path = self.dataset_cfg.val_csv_path
            datatype = "Val"

        # Process clusters
        self.csv = pd.read_csv(csv_path)
        self.csv = self._filter_metadata(self.csv)
        # self.csv = self.csv.sort_values(
        #     'modeled_seq_len', ascending=False)
        self.csv.reset_index(
            drop=False, names=["original_index"], inplace=True
        )
        self.csv["index"] = list(range(len(self.csv)))
        self._log.info(f"{datatype} data num: {len(self.csv)}")

    def _filter_metadata(self, raw_csv):
        """Filter metadata for PPI dataset."""
        data_csv = raw_csv
        filter_cfg = self.dataset_cfg.filter
        if filter_cfg.activate:
            data_csv = data_csv[data_csv["data_level"] == 1]
            print(
                f"filter_cfg.max_complex_len:{filter_cfg.max_complex_len}"
            )
            data_csv = data_csv[
                data_csv["seq_len"] <= filter_cfg.max_complex_len
            ]
            data_csv = data_csv[
                data_csv["seq_len"] == data_csv["modeled_seq_len"]
            ]
            data_csv = data_csv[
                data_csv["homo_dimer_rate"]
                <= filter_cfg.max_homo_dimer_rate
            ]
            print(
                f"raw data {len(raw_csv)},after filter data num: {len(data_csv)}, cluster: {len(data_csv['cluster'].unique())}."
            )
        return data_csv


class Ppi_Monomer_Dataset(BaseDataset):

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

        self._cache = {}
        self._rng = np.random.default_rng(seed=self._dataset_cfg.seed)
        if self._is_training == True:
            csv_path = self.dataset_cfg.train_csv_path
            datatype = "Train"
        else:
            csv_path = self.dataset_cfg.val_csv_path
            datatype = "Val"

        # Process clusters
        self.csv = pd.read_csv(csv_path)
        self.csv = self._filter_metadata(self.csv)
        self.csv = self.csv.sort_values("modeled_seq_len", ascending=False)
        self.csv.reset_index(
            drop=False, names=["original_index"], inplace=True
        )
        self.csv["index"] = list(range(len(self.csv)))
        self._log.info(f"{datatype} data num: {len(self.csv)}")

    def _filter_metadata(self, raw_csv):
        ppi_csv = self._filter_metadata_ppi(raw_csv)
        monomer_csv = self._filter_metadata_monomer(raw_csv)
        result_csv = pd.concat([monomer_csv, ppi_csv])
        print(
            f"raw data {len(raw_csv)},after filter data num: {len(result_csv)}"
        )
        return result_csv

    def _filter_metadata(self, raw_csv):
        """Filter metadata for PPI dataset."""
        filter_cfg = self.dataset_cfg.filter.ppi
        data_csv = raw_csv[raw_csv["num_chains"] == 2]
        len_0 = len(data_csv)
        if len_0 > 0:
            data_csv = data_csv[
                data_csv["target_seq_len"] <= filter_cfg.max_target_num_res
            ]
            len1 = len(data_csv)
            data_csv = data_csv[
                (
                    data_csv["binder_seq_len"]
                    <= filter_cfg.max_binder_num_res
                )
                & (
                    data_csv["binder_seq_len"]
                    >= filter_cfg.min_binder_num_res
                )
            ]
            len2 = len(data_csv)
            data_csv = data_csv[
                data_csv["binder_contact_rate"]
                <= filter_cfg.max_binder_contact_rate
            ]
            len3 = len(data_csv)
            data_csv = data_csv[
                data_csv["seq_len"] == data_csv["modeled_seq_len"]
            ]
            len4 = len(data_csv)
            data_csv = _max_coil_filter(
                data_csv, filter_cfg.max_coil_percent
            )
            data_csv = _rog_filter(data_csv, filter_cfg.rog_quantile)
            len5 = len(data_csv)
            print(
                f"ppi raw data {len_0},after filter data num: {len(data_csv)}. Filters: {len1}->{len2}->{len3}->{len4}->{len5}"
            )
        return data_csv

    def _filter_metadata_monomer(self, raw_csv):
        """Filter metadata for monomer dataset."""
        filter_cfg = self.dataset_cfg.filter.monomer
        data_csv = raw_csv[raw_csv["num_chains"] == 1]
        len_0 = len(data_csv)
        if len_0 > 0:
            data_csv = data_csv[
                data_csv["seq_len"] <= filter_cfg.max_num_res
            ]
            len1 = len(data_csv)
            data_csv = data_csv[
                data_csv["seq_len"] >= filter_cfg.min_num_res
            ]
            len2 = len(data_csv)
            data_csv = data_csv[
                data_csv["seq_len"] == data_csv["modeled_seq_len"]
            ]
            len4 = len(data_csv)
            data_csv = _max_coil_filter(
                data_csv, filter_cfg.max_coil_percent
            )
            data_csv = _rog_filter(data_csv, filter_cfg.rog_quantile)
            len5 = len(data_csv)
            print(
                f"monomer raw data {len_0},after filter data num: {len(data_csv)}. Filters: {len1}->{len2}->{len4}->{len5}"
            )
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
        self.motif_cfg = None
        self._cache = {}
        self._rng = np.random.default_rng(seed=self._dataset_cfg.seed)
        csv_path = self.dataset_cfg.test_csv_path
        datatype = "Test"
        self.csv = pd.read_csv(csv_path)
        if self.dataset_cfg.sample_pdbname is not None:
            self.dataset_cfg.sample_pdbname = (
                [self.dataset_cfg.sample_pdbname]
                if isinstance(self.dataset_cfg.sample_pdbname, str)
                else self.dataset_cfg.sample_pdbname
            )
            self.csv = self.csv.loc[
                self.csv["pdb_name"].isin(self.dataset_cfg.sample_pdbname)
            ]
        self.csv.reset_index(
            drop=False, names=["original_index"], inplace=True
        )
        self.csv["index"] = list(range(len(self.csv)))
        self._log.info(
            f"{datatype} data num: {len(self.csv)}, sample_pdbname:{self.dataset_cfg.sample_pdbname}"
        )

        # self.batch_size = self.dataset_cfg.samples_batch_size if self.dataset_cfg.sample_original_binder_len else 1
        self.batch_size = self.dataset_cfg.samples_batch_size
        self.samples_per_target = dataset_cfg.samples_per_target
        assert self.samples_per_target % self.batch_size == 0
        self.n_samples = self.samples_per_target // self.batch_size

        self.binder_len = self._rng.integers(
            low=self._dataset_cfg.samples_min_length,
            high=self._dataset_cfg.samples_max_length,
            size=(self.n_samples,),
        )
        self.binder_len = np.repeat(self.binder_len, self.batch_size)

        all_sample_ids = []
        for row_id in range(len(self.csv)):
            target_row = self.csv.iloc[row_id]
            for sample_id, blen in zip(
                range(self.samples_per_target), self.binder_len
            ):
                # sample_ids = torch.tensor([batch_size * sample_id + i for i in range(batch_size)])
                all_sample_ids.append((target_row, sample_id, blen))
        self.all_sample_ids = all_sample_ids

    def _filter_metadata(self, raw_csv):
        """Filter metadata for test dataset."""
        filter_cfg = self.dataset_cfg.filter
        data_csv = raw_csv[
            raw_csv["target_seq_len"] <= filter_cfg.max_target_num_res
        ]
        data_csv = data_csv[
            (data_csv["binder_seq_len"] <= filter_cfg.max_binder_num_res)
            & (data_csv["binder_seq_len"] >= filter_cfg.min_binder_num_res)
        ]
        print(
            f"raw data {len(raw_csv)}, after filter data num: {len(data_csv)}"
        )
        return data_csv

    def __len__(self):
        return len(self.all_sample_ids)

    def __getitem__(self, row_idx):
        """Get a single test data example."""
        csv_row, sample_ids, binder_len = self.all_sample_ids[row_idx]
        feats = self.process_csv_row(csv_row)

        pdb_diffuse_mask = self.setup_binder_mask(feats)
        feats["diffuse_mask"] = pdb_diffuse_mask.int()
        hotspot_mask = self.setup_target_hotspots(feats)
        feats["hotspot_mask"] = hotspot_mask.int()

        feats = self.post_process_feats(feats)

        target_index = np.nonzero(pdb_diffuse_mask == 0)[:, 0]
        target_feats = {
            k: v[target_index]
            for k, v in feats.items()
            if k not in ["pdb_name", "original_index"]
        }
        if self._dataset_cfg.sample_original_binder_len == True:
            binder_len = int(sum(pdb_diffuse_mask).item())
        target_len = len(target_index)
        total_length = target_len + binder_len
        diffuse_mask = torch.ones(total_length, dtype=torch.int)
        hotspot_mask = torch.zeros(total_length, dtype=torch.int)
        trans_1 = torch.zeros(
            total_length, 3
        )
        rotmats_1 = torch.eye(3)[None].repeat(
            total_length, 1, 1
        )
        aatype = torch.zeros(
            total_length, dtype=torch.int64
        )
        res_mask = torch.ones(total_length, dtype=torch.int)
        chain_idx = torch.ones(total_length, dtype=torch.int)
        target_interface_mask = torch.zeros(total_length)

        trans_1[:target_len] = target_feats["trans_1"]
        rotmats_1[:target_len] = target_feats["rotmats_1"]
        aatype[:target_len] = target_feats["aatype"]
        aatype[target_len:] = 21
        diffuse_mask[:target_len] = 0
        hotspot_mask[:target_len] = target_feats["hotspot_mask"]
        res_mask[:target_len] = target_feats["res_mask"]
        chain_idx[:target_len] = target_feats["chain_idx"]
        target_interface_mask[:target_len] = target_feats[
            "target_interface_mask"
        ]

        output_feats = {
            "diffuse_mask": diffuse_mask,
            "hotspot_mask": hotspot_mask,
            "trans_1": trans_1,
            "rotmats_1": rotmats_1,
            "aatype": aatype.long(),
            "pdb_name": csv_row["pdb_name"],
            "original_index": csv_row["original_index"],
            "sample_ids": sample_ids,
            "res_mask": res_mask,
            "chain_idx": chain_idx,
            "target_interface_mask": target_interface_mask,
        }
        return output_feats


class PpiScaffoldingTestDataset(BaseDataset):
    def __init__(self, dataset_cfg, task):

        self._dataset_cfg = dataset_cfg
        self._benchmark_df = pd.read_csv(self._dataset_cfg.test_csv_path)
        self.task = task
        self.motif_cfg = (
            self._dataset_cfg.motif
            if self.task == "binder_motif"
            else None
        )

        self._rng = np.random.default_rng(seed=self._dataset_cfg.seed)
        self._log = logging.getLogger(__name__)
        self._cache = {}
        if self.dataset_cfg.sample_pdbname is not None:
            self.dataset_cfg.sample_pdbname = (
                [self.dataset_cfg.sample_pdbname]
                if isinstance(self.dataset_cfg.sample_pdbname, str)
                else self.dataset_cfg.sample_pdbname
            )
            self._benchmark_df = self._benchmark_df.loc[
                self._benchmark_df["id"].isin(
                    self.dataset_cfg.sample_pdbname
                )
            ]
        self._benchmark_df.reset_index(
            drop=False, names=["original_index"], inplace=True
        )
        self._benchmark_df["index"] = list(range(len(self._benchmark_df)))
        # if self._samples_cfg.target_subset is not None:
        #     self._benchmark_df = self._benchmark_df[
        #         self._benchmark_df.target.isin(self._samples_cfg.target_subset)
        #     ]
        # if len(self._benchmark_df) == 0:
        #     raise ValueError('No targets found.')

        # contigs_by_test_case = save_motif_segments.load_contigs_by_test_case(
        #     self._benchmark_df)

        self.batch_size = self._dataset_cfg.samples_batch_size
        self.samples_per_target = self._dataset_cfg.samples_per_target
        assert self.samples_per_target % self.batch_size == 0
        self.n_samples = self.samples_per_target // self.batch_size

        self.binder_len = self._rng.integers(
            low=self._dataset_cfg.samples_min_length,
            high=self._dataset_cfg.samples_max_length,
            size=(self.n_samples,),
        )
        self.binder_len = np.repeat(self.binder_len, self.batch_size)

        all_sample_ids = []
        for row_id in range(len(self._benchmark_df)):
            target_row = self._benchmark_df.iloc[row_id]
            for sample_id, blen in zip(
                range(self.samples_per_target), self.binder_len
            ):
                # sample_ids = torch.tensor([batch_size * sample_id + i for i in range(batch_size)])
                all_sample_ids.append((target_row, sample_id, blen))
        self.all_sample_ids = all_sample_ids

    def _get_new_hotspot_interface(
        self, feats, source_segments, binder_id
    ):
        binder_motif = []
        for segment in source_segments:
            binder_motif.extend(list(range(segment[0], segment[1] + 1)))

        b_pair_idx = 3 if binder_id == "L" else 2
        t_pair_idx = 2 if binder_id == "L" else 3
        new_contact_pairs = [
            pair
            for pair in feats["contact_pairs"]
            if pair[b_pair_idx] not in binder_motif
        ]
        hotspot_interface_residues = [
            pair[t_pair_idx] for pair in new_contact_pairs
        ]
        hotspot_interface_mask = torch.isin(
            feats["original_res_idx"],
            torch.tensor(hotspot_interface_residues),
        ) * (1 - feats["diffuse_mask"])
        # binder_motif_mask = torch.isin(feats['original_res_idx'], torch.tensor(binder_motif))*(1-feats['diffuse_mask'])
        return hotspot_interface_mask

    def __len__(self):
        return len(self.all_sample_ids)

    def process_input_contig(self, input_contig):
        items = input_contig.split(",")
        cleaned_items = []
        chain_dict = defaultdict(list)

        for item in items:
            if ":" in item:
                chain_id, range_part = item.split(":", 1)
                cleaned_items.append(range_part)
                chain_dict[chain_id].append(range_part)
            elif re.match(
                r"^[A-Za-z]?\d+-\d+$", item
            ):  # 如 L171-187 / 171-187 / X5-20
                cleaned_items.append(item)
            else:
                # 是普通区间，如 10-25
                cleaned_items.append(item)

        return ",".join(cleaned_items), dict(chain_dict)

    def __getitem__(self, idx):
        # Process data example.
        csv_row, sample_ids, binder_len = self.all_sample_ids[idx]
        feats = self.process_csv_row(csv_row)

        pdb_diffuse_mask = self.setup_binder_mask(feats)
        feats["diffuse_mask"] = pdb_diffuse_mask.int()
        feats = self.post_process_feats(feats)

        #############motif mask###############
        if isinstance(csv_row.sample_binder_len, str):
            lengths = csv_row.sample_binder_len.split("-")
            if len(lengths) == 1:
                start_length = lengths[0]
                end_length = lengths[0]
            else:
                start_length, end_length = lengths
            sample_lengths = [int(start_length), int(end_length) + 1]
        else:
            sample_lengths = None
        # multimotif
        # contig, contig_groups = self.process_input_contig(csv_row.contig)
        sample_contig, sampled_binder_length, _, motif_groups = (
            get_sampled_mask(csv_row.contig, sample_lengths)
        )
        motif_locations, original_source_segments = (
            save_motif_segments.motif_locations_from_contig(
                sample_contig[0]
            )
        )  # [(18, 37), (53, 72)]; [(29, 48), (106, 127)]
        source_segments = []
        for segments in original_source_segments:
            original_binder_motif = torch.tensor(
                list(range(segments[0], segments[-1] + 1))
            )
            binder_motif = torch.nonzero(
                torch.isin(
                    feats["original_res_idx"] * feats["diffuse_mask"],
                    original_binder_motif,
                ),
                as_tuple=True,
            )[0]
            source_segments.append(
                (binder_motif[0].item(), binder_motif[-1].item())
            )

        #######################################
        ################hotspot_mask###########
        feat_tmp = du.read_pkl(csv_row["processed_path"])
        feats["contact_pairs"] = feat_tmp["contact_pairs"]
        del feat_tmp
        feats["hotspot_interface_mask"] = self._get_new_hotspot_interface(
            feats, original_source_segments, "L"
        )
        feats["hotspot_mask"] = self.setup_target_hotspots(
            feats
        )  # self._dataset_cfg.define_hotspots=False
        #######################################

        target_index = np.nonzero(pdb_diffuse_mask == 0)[:, 0]
        binder_index = np.nonzero(pdb_diffuse_mask == 1)[:, 0]
        target_feats = {
            k: v[target_index]
            for k, v in feats.items()
            if k not in ["pdb_name", "original_index", "contact_pairs"]
        }
        true_binder_feats = {
            k: v[binder_index]
            for k, v in feats.items()
            if k not in ["pdb_name", "original_index", "contact_pairs"]
        }
        # if self._dataset_cfg.sample_original_binder_len == True:
        #     binder_len = int(sum(pdb_diffuse_mask).item())
        target_len = len(target_index)
        total_length = target_len + sampled_binder_length

        diffuse_mask = torch.ones(total_length, dtype=torch.int)
        hotspot_mask = torch.zeros(total_length, dtype=torch.int)
        trans_1 = torch.zeros(
            total_length, 3
        )  # torch.Size([total_length, 3])
        rotmats_1 = torch.eye(3)[None].repeat(
            total_length, 1, 1
        )  # torch.Size([total_length, 3, 3])
        aatype = torch.zeros(
            total_length, dtype=torch.int64
        )  # torch.Size([total_length])
        res_mask = torch.ones(total_length, dtype=torch.int)
        chain_idx = torch.ones(total_length, dtype=torch.int)
        target_interface_mask = torch.zeros(total_length, dtype=torch.int)
        binder_motif_mask = torch.zeros(total_length, dtype=torch.int)

        # target locations
        diffuse_mask[:target_len] = 0
        hotspot_mask[:target_len] = target_feats["hotspot_mask"]
        trans_1[:target_len] = target_feats["trans_1"]
        rotmats_1[:target_len] = target_feats["rotmats_1"]
        aatype[:target_len] = target_feats["aatype"]
        aatype[target_len:] = 20  # *
        # trans_1[diffuse_mask == 0] = target_feats['trans_1']
        # rotmats_1[diffuse_mask == 0] = target_feats['rotmats_1']
        # aatype[diffuse_mask == 0] = target_feats['aatype']
        # aatype[diffuse_mask == 1] = 20  # *
        # res_mask[:target_len] = target_feats['res_mask']
        chain_idx[:target_len] = target_feats["chain_idx"]
        target_interface_mask[:target_len] = target_feats[
            "target_interface_mask"
        ]
        binder_motif_mask[:target_len] = 0

        # motif locations
        for generate_location, true_res_interval in zip(
            motif_locations, source_segments
        ):
            start, end = generate_location
            j, k = true_res_interval
            start = target_len + start
            end = target_len + end
            diffuse_mask[start : end + 1] = 0
            trans_1[start : end + 1] = true_binder_feats["trans_1"][
                j : k + 1
            ]
            rotmats_1[start : end + 1] = true_binder_feats["rotmats_1"][
                j : k + 1
            ]
            aatype[start : end + 1] = true_binder_feats["aatype"][
                j : k + 1
            ]
            binder_motif_mask[start : end + 1] = 1

        output_feats = {
            "diffuse_mask": diffuse_mask,
            "hotspot_mask": hotspot_mask,
            "trans_1": trans_1,
            "rotmats_1": rotmats_1,
            "aatype": aatype,
            "pdb_name": csv_row["pdb_name"],
            "original_index": sample_ids,
            "sample_ids": sample_ids,
            "res_mask": res_mask,
            "chain_idx": chain_idx,
            "target_interface_mask": target_interface_mask,
            "binder_motif_mask": binder_motif_mask,
        }
        output_feats = self.post_process_feats(output_feats)

        # multimotif
        # Create motif structure mask
        if np.max(motif_groups) >= 2:
            motif_groups_mask = np.zeros(
                (total_length, total_length), dtype=np.bool_
            )
            motif_groups.extend(
                [np.max(motif_groups) + 1] * target_len
            )  #:target chain as a group
            num_groups = np.max(motif_groups)
            for i in range(1, 1 + num_groups):
                motif_group_sequence_mask = np.equal(motif_groups, i)
                motif_groups_mask += (
                    motif_group_sequence_mask[:, np.newaxis]
                    * motif_group_sequence_mask[np.newaxis, :]
                )
            motif_groups_mask = torch.tensor(
                motif_groups_mask, dtype=torch.int
            )
            output_feats.update({"motif_groups_mask": motif_groups_mask})

        # add binder_motif_map info
        binder_motif_map = {}
        for (start_a, end_a), (start_b, end_b) in zip(
            original_source_segments, motif_locations
        ):
            # 生成范围内的数字列表
            range_a = list(range(start_a, end_a + 1))
            range_b = list(range(start_b + 1, end_b + 2))
            # 将数字列表转换为字符串
            key = ",".join(map(str, range_a))
            value = ",".join(map(str, range_b))
            binder_motif_map[key] = value
        output_feats.update({"binder_motif_map": str(binder_motif_map)})

        return output_feats


class PpiTestPartialDataset(BaseDataset):

    def __init__(
        self,
        *,
        dataset_cfg,
        task,
    ):
        self._log = logging.getLogger(__name__)
        self._dataset_cfg = dataset_cfg
        self.task = task
        self.motif_cfg = self.dataset_cfg.motif
        self._cache = {}
        self._rng = np.random.default_rng(seed=self._dataset_cfg.seed)
        csv_path = self.dataset_cfg.test_csv_path
        datatype = "Test"
        self.csv = pd.read_csv(csv_path)
        if self.dataset_cfg.sample_pdbname is not None:
            self.dataset_cfg.sample_pdbname = (
                [self.dataset_cfg.sample_pdbname]
                if isinstance(self.dataset_cfg.sample_pdbname, str)
                else self.dataset_cfg.sample_pdbname
            )
            self.csv = self.csv.loc[
                self.csv["pdb_name"].isin(self.dataset_cfg.sample_pdbname)
            ]
        self.csv.reset_index(
            drop=False, names=["original_index"], inplace=True
        )
        self.csv["index"] = list(range(len(self.csv)))
        self._log.info(
            f"{datatype} data num: {len(self.csv)}, sample_pdbname:{self.dataset_cfg.sample_pdbname}"
        )

        # self.batch_size = self.dataset_cfg.samples_batch_size if self.dataset_cfg.sample_original_binder_len else 1
        self.batch_size = self.dataset_cfg.samples_batch_size
        self.samples_per_target = dataset_cfg.samples_per_target
        assert self.samples_per_target % self.batch_size == 0
        self.n_samples = self.samples_per_target // self.batch_size
        self.binder_len = np.repeat(
            np.array(self.csv["binder_seq_len"]), self.n_samples
        )
        if self.dataset_cfg.scaffold_contig is not None:
            self.scaffold_contig, self.add_binder_len = (
                self.dataset_cfg.scaffold_contig.split(",")
            )
            self.add_binder_len = int(self.add_binder_len)
            self.binder_len += self.add_binder_len
            start_scaffold = int(self.scaffold_contig.split("-")[0][1:])
            end_scaffold = int(self.scaffold_contig.split("-")[1])
            self.scaffold_residues = torch.tensor(
                list(range(start_scaffold, end_scaffold + 1))
            )
            self.insert_scaffold_res_idx = (
                start_scaffold + (end_scaffold - start_scaffold + 1) // 2
            )
        self.binder_len = np.repeat(self.binder_len, self.batch_size)

        all_sample_ids = []
        for row_id in range(len(self.csv)):
            target_row = self.csv.iloc[row_id]
            for sample_id, blen in zip(
                range(self.samples_per_target), self.binder_len
            ):
                # sample_ids = torch.tensor([batch_size * sample_id + i for i in range(batch_size)])
                all_sample_ids.append((target_row, sample_id, blen))
        self.all_sample_ids = all_sample_ids

    def get_new_binder_index(
        self, binder_index, binder_res_idx, insert_scaffold_res_idx
    ):
        index_of_insert = (
            binder_res_idx == insert_scaffold_res_idx
        ).nonzero(as_tuple=True)[0]
        # 检查是否找到
        if index_of_insert.numel() > 0:
            index_of_insert = index_of_insert.item()  # 获取索引值
            # 在 99 的位置后面插入 10
            modified_binder_index = torch.cat(
                (
                    binder_index[: index_of_insert + 1],
                    torch.tensor([index_of_insert]),
                    binder_index[index_of_insert + 1 :],
                )
            )
        else:
            modified_binder_index = (
                binder_index  # 如果没有找到，则保持原张量不变
            )

        return modified_binder_index, index_of_insert + 1

    def __len__(self):
        return len(self.all_sample_ids)

    def __getitem__(self, row_idx):
        # Process data example.
        csv_row, sample_ids, binder_len = self.all_sample_ids[row_idx]
        feats = self.process_csv_row(csv_row)

        pdb_diffuse_mask = self.setup_binder_mask(feats)
        feats["diffuse_mask"] = pdb_diffuse_mask.int()
        hotspot_mask = self.setup_target_hotspots(feats)
        feats["hotspot_mask"] = hotspot_mask.int()
        if self._dataset_cfg.motif:
            feats["diffuse_mask"] = feats["diffuse_mask"] * (
                1 - feats["binder_motif_mask"]
            )

        feats = self.post_process_feats(feats)

        # target_index = np.nonzero(pdb_diffuse_mask == 0)[:, 0]
        # target_feats = {
        #     k: v[target_index] for k, v in feats.items() if k not in ['pdb_name', 'original_index']
        # }
        # binder_index = np.nonzero(pdb_diffuse_mask == 1)[:, 0]
        # binder_feats = {
        #     k: v[binder_index] for k, v in feats.items() if k not in ['pdb_name', 'original_index']
        # }
        feats["aatype"][feats["diffuse_mask"] == 1] = 21

        # target在前， binder在后
        target_index = np.nonzero(pdb_diffuse_mask == 0)[:, 0]
        binder_index = np.nonzero(pdb_diffuse_mask == 1)[:, 0]
        binder_res_idx = feats["original_res_idx"][binder_index]
        target_feats = {
            k: v[target_index]
            for k, v in feats.items()
            if k
            not in [
                "pdb_name",
                "original_index",
                "contact_pairs",
                "original_binder_motif",
            ]
        }
        binder_feats = {
            k: v[binder_index]
            for k, v in feats.items()
            if k
            not in [
                "pdb_name",
                "original_index",
                "contact_pairs",
                "original_binder_motif",
            ]
        }
        if self.dataset_cfg.scaffold_contig is not None:
            if self.add_binder_len < 0:
                binder_index = binder_index[
                    binder_index != self.insert_scaffold_res_idx
                ]
                binder_feats = {
                    k: v[binder_index]
                    for k, v in feats.items()
                    if k
                    not in ["pdb_name", "original_index", "contact_pairs"]
                }
            else:
                binder_index, index_of_insert = self.get_new_binder_index(
                    binder_index,
                    binder_res_idx,
                    self.insert_scaffold_res_idx,
                )
                indices = torch.nonzero(
                    torch.isin(
                        binder_feats["original_res_idx"],
                        self.scaffold_residues,
                    ),
                    as_tuple=True,
                )[0]
                scaffold_trans_1_mean = torch.mean(
                    binder_feats["trans_1"][indices], dim=0
                )
                scaffold_rotmats_1_mean = torch.mean(
                    binder_feats["rotmats_1"][indices], dim=0
                )
                binder_feats = {
                    k: v[binder_index]
                    for k, v in feats.items()
                    if k
                    not in [
                        "pdb_name",
                        "original_index",
                        "contact_pairs",
                        "original_binder_motif",
                    ]
                }
                binder_feats["trans_1"][
                    index_of_insert
                ] = scaffold_trans_1_mean
                binder_feats["rotmats_1"][
                    index_of_insert
                ] = scaffold_rotmats_1_mean

        target_len = len(target_index)
        total_length = len(target_index) + len(binder_index)

        diffuse_mask = torch.ones(total_length, dtype=torch.int)
        hotspot_mask = torch.zeros(total_length, dtype=torch.int)
        trans_1 = torch.zeros(
            total_length, 3
        )  # torch.Size([total_length, 3])
        rotmats_1 = torch.eye(3)[None].repeat(
            total_length, 1, 1
        )  # torch.Size([total_length, 3, 3])
        aatype = torch.zeros(
            total_length, dtype=torch.int64
        )  # torch.Size([total_length])
        res_mask = torch.ones(total_length, dtype=torch.int)
        chain_idx = torch.ones(total_length, dtype=torch.int)
        target_interface_mask = torch.zeros(total_length, dtype=torch.int)
        binder_motif_mask = torch.zeros(total_length, dtype=torch.int)

        # target locations
        diffuse_mask[:target_len] = target_feats["diffuse_mask"]
        hotspot_mask[:target_len] = target_feats["hotspot_mask"]
        trans_1[:target_len] = target_feats["trans_1"]
        rotmats_1[:target_len] = target_feats["rotmats_1"]
        aatype[:target_len] = target_feats["aatype"]
        res_mask[:target_len] = target_feats["res_mask"]
        chain_idx[:target_len] = target_feats["chain_idx"]
        target_interface_mask[:target_len] = target_feats[
            "target_interface_mask"
        ]

        # binder
        # target locations
        diffuse_mask[target_len:] = binder_feats["diffuse_mask"]
        hotspot_mask[target_len:] = binder_feats["hotspot_mask"]
        trans_1[target_len:] = binder_feats["trans_1"]
        rotmats_1[target_len:] = binder_feats["rotmats_1"]
        aatype[target_len:] = binder_feats["aatype"]
        res_mask[target_len:] = binder_feats["res_mask"]
        chain_idx[target_len:] = binder_feats["chain_idx"]
        target_interface_mask[target_len:] = binder_feats[
            "target_interface_mask"
        ]

        if self._dataset_cfg.motif:
            binder_motif_mask[:target_len] = target_feats[
                "binder_motif_mask"
            ]
            binder_motif_mask[target_len:] = binder_feats[
                "binder_motif_mask"
            ]

        output_feats = {
            "diffuse_mask": diffuse_mask,
            "hotspot_mask": hotspot_mask,
            "trans_1": trans_1,
            "rotmats_1": rotmats_1,
            "aatype": aatype,
            "pdb_name": csv_row["pdb_name"],
            "original_index": csv_row["original_index"],
            "sample_ids": sample_ids,
            "res_mask": res_mask,
            "chain_idx": chain_idx,
            "target_interface_mask": target_interface_mask,
            "binder_motif_mask": binder_motif_mask,
            # 'res_idx': res_idx
        }

        if "original_binder_motif" in feats:
            original_binder_motif = [
                [str(str_i) for str_i in i]
                for i in feats["original_binder_motif"]
            ]
            original_binder_motif = str(
                {",".join(i): (",".join(i)) for i in original_binder_motif}
            )
            output_feats.update(
                {"binder_motif_map": original_binder_motif}
            )
        return output_feats
