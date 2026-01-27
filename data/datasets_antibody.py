import numpy as np
import pandas as pd
import logging
import tree
import json
import torch

from torch.utils.data import Dataset
from data import utils as du
from core.data import data_transforms
from core.utils import rigid_utils

torch.manual_seed(0)


class AntibodyTestDataset(Dataset):
    def __init__(self, dataset_cfg, task):
        self._dataset_cfg = dataset_cfg
        # read csv
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
        self._benchmark_df.reset_index(
            drop=False, names=["original_index"], inplace=True
        )
        self._benchmark_df["index"] = list(range(len(self._benchmark_df)))

        self.batch_size = self._dataset_cfg.samples_batch_size
        self.samples_per_target = self._dataset_cfg.samples_per_target
        assert self.samples_per_target % self.batch_size == 0
        self.n_samples = self.samples_per_target // self.batch_size

        all_sample_ids = []
        for row_id in range(len(self._benchmark_df)):
            target_row = self._benchmark_df.iloc[row_id]
            for sample_id in range(self.samples_per_target):
                all_sample_ids.append((target_row, sample_id))
        self.all_sample_ids = all_sample_ids

    def __len__(self):
        return len(self.all_sample_ids)

    def process_csv_row(self, csv_row):
        processed_file_path = csv_row["processed_path"]
        processed_feats = du.read_pkl(processed_file_path)
        processed_feats = du.parse_chain_feats(
            processed_feats, normalize_positions=True
        )

        framework_mask = (
            1 - processed_feats["cdr_mask"]
        ) * processed_feats[
            "chain_groups"
        ]  # ab framework
        processed_feats["chain_group_index"] = processed_feats[
            "chain_groups"
        ]
        framework_mask = torch.tensor(framework_mask, dtype=int)
        processed_feats["framework_mask"] = framework_mask

        # leave out non-tensor data
        processed_feats = {
            k: v
            for k, v in processed_feats.items()
            if k
            not in [
                "target_interface_residues",
                "binder_interface_residues",
                "contact_pairs",
                "target_interface_chains",
                "binder_interface_chains",
                "contig",
                "length",
                "binder_motif",
                "binder_motif_chains",
            ]
        }

        # Only take modeled residues
        modeled_idx = processed_feats.pop("modeled_idx")
        if len(modeled_idx) != len(processed_feats["bb_mask"]):
            processed_feats = tree.map_structure(
                lambda x: x[modeled_idx] if x is not None else None,
                processed_feats,
            )

        # Run through OpenFold data transforms
        chain_feats = {
            "aatype": torch.tensor(
                processed_feats["aatype"]
            ).long(),  # values range from 0 to 19, 0 is ALA
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
        )  # torch.Size([256, 3, 3])
        trans_1 = rigids_1.get_trans()  # torch.Size([256, 3])
        res_plddt = processed_feats["b_factors"][:, 1]  # (256,)
        res_mask = torch.tensor(
            processed_feats["bb_mask"]
        ).int()  # torch.Size([256])

        # reset residue index
        # Re-number residue indices for each chain such that it starts from 1.
        # already starts from 1 in raw feats, skip
        new_res_idx = processed_feats["residue_index"]

        output_feats = {
            "res_plddt": res_plddt,
            "aatype": chain_feats["aatype"],
            "rotmats_1": rotmats_1,
            "trans_1": trans_1,
            "res_mask": res_mask,
            "chain_idx": torch.tensor(processed_feats["chain_index"]),
            "chain_group_idx": torch.tensor(
                processed_feats["chain_group_index"]
            ),
            "res_idx": torch.tensor(new_res_idx),
            "original_res_idx": torch.tensor(
                processed_feats["residue_index"]
            ),
            "framework_mask": processed_feats["framework_mask"],
            "all_atom_positions": chain_feats["all_atom_positions"],
            "all_atom_mask": chain_feats["all_atom_mask"],
            "cdr_mask": processed_feats["cdr_mask"],
            "hotspots": csv_row["hotspots"],
        }
        output_feats = {
            k: v for k, v in output_feats.items() if v is not None
        }

        if torch.isnan(trans_1).any() or torch.isnan(rotmats_1).any():
            raise ValueError(f"Found NaNs in {processed_file_path}")

        if output_feats is None:
            return None
        output_feats["pdb_name"] = csv_row["pdb_name"]
        output_feats["original_index"] = f"{csv_row['original_index']}"
        return output_feats

    def setup_antibody_mask(self, feats):
        """
        target chain set as diffuse_mask=0, binder chain set as diffuse_mask=1
        """
        target_chain_id = 0
        diffuse_index = torch.nonzero(
            feats["chain_group_idx"] != target_chain_id
        ).reshape(-1)
        diffuse_mask = torch.zeros(
            feats["chain_group_idx"].shape, dtype=torch.int
        )
        diffuse_mask[diffuse_index] = 1  # 0:motif, 1:to diffuse
        if torch.sum(diffuse_mask) < 1:  # Should only happen rarely.
            diffuse_mask = torch.ones_like(diffuse_mask)
        diffuse_mask = diffuse_mask.int()
        return diffuse_mask

    def post_process_feats(self, feats):
        motif_mask = 1 - feats["diffuse_mask"]  # 1:antigen, 0:to diffuse

        trans_1 = feats["trans_1"]
        motif_1 = trans_1 * motif_mask[:, None]
        motif_com = torch.sum(motif_1, dim=0) / (torch.sum(motif_mask) + 1)
        trans_1 -= motif_com[None, :]
        feats["trans_1"] = trans_1
        return feats

    def setup_target_hotspots(self, feats):
        hotspot_mask = torch.zeros_like(feats["diffuse_mask"])
        hotspot_index = json.loads(feats["hotspots"])
        hotspot_mask[hotspot_index] = 1
        return hotspot_mask

    def __getitem__(self, idx):
        csv_row, sample_ids = self.all_sample_ids[idx]
        feats = self.process_csv_row(csv_row)

        antibody_mask = self.setup_antibody_mask(feats)
        feats["diffuse_mask"] = antibody_mask.int()
        feats = self.post_process_feats(feats)

        feats["hotspot_mask"] = self.setup_target_hotspots(feats)

        target_index = np.nonzero(antibody_mask == 0)[:, 0]
        antibody_index = np.nonzero(antibody_mask == 1)[:, 0]
        target_feats = {
            k: v[target_index]
            for k, v in feats.items()
            if k not in ["pdb_name", "original_index", "hotspots"]
        }

        target_len = len(target_index)
        antibody_len = len(antibody_index)
        total_length = target_len + antibody_len

        cdr_mask = torch.tensor(feats["cdr_mask"], dtype=torch.int)
        framework_mask = feats["framework_mask"]
        hotspot_mask = feats["hotspot_mask"]
        aatype = feats["aatype"]

        res_mask = torch.ones(total_length, dtype=torch.int)
        chain_idx = feats["chain_idx"]
        chain_group_idx = feats["chain_group_idx"]
        aatype = aatype * (1 - cdr_mask) + 20 * cdr_mask
        pos_fixed_mask = cdr_mask + framework_mask  # 0: fixed, 1: no fixed
        diffuse_mask = pos_fixed_mask

        output_feats = {
            "diffuse_mask": diffuse_mask,
            "hotspot_mask": hotspot_mask,
            "trans_1": feats["trans_1"],
            "rotmats_1": feats["rotmats_1"],
            "aatype": aatype,
            "pdb_name": csv_row["pdb_name"],
            "original_index": sample_ids,
            "sample_ids": sample_ids,
            "res_mask": res_mask,
            "chain_idx": chain_idx,
            "chain_group_idx": chain_group_idx,
            "framework_mask": framework_mask,
            "pos_fixed_mask": pos_fixed_mask,
        }
        output_feats = self.post_process_feats(output_feats)
        output_feats.update({"binder_motif_mask": framework_mask})
        return output_feats


class AntibodyPartialDataset(Dataset):
    def __init__(self, dataset_cfg, task):
        self._dataset_cfg = dataset_cfg
        # read csv
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
        self._benchmark_df.reset_index(
            drop=False, names=["original_index"], inplace=True
        )
        self._benchmark_df["index"] = list(range(len(self._benchmark_df)))

        self.batch_size = self._dataset_cfg.samples_batch_size
        self.samples_per_target = self._dataset_cfg.samples_per_target
        assert self.samples_per_target % self.batch_size == 0
        self.n_samples = self.samples_per_target // self.batch_size

        # # TODO
        # self.binder_len = 0
        # self.binder_len = np.repeat(self.binder_len, self.batch_size)

        all_sample_ids = []
        for row_id in range(len(self._benchmark_df)):
            target_row = self._benchmark_df.iloc[row_id]
            # for sample_id, blen in zip(range(self.samples_per_target), self.binder_len):
            #     all_sample_ids.append((target_row, sample_id, blen))
            for sample_id in range(self.samples_per_target):
                all_sample_ids.append((target_row, sample_id))
        self.all_sample_ids = all_sample_ids

    def __len__(self):
        return len(self.all_sample_ids)

    def process_csv_row(self, csv_row):
        processed_file_path = csv_row["processed_path"]
        processed_feats = du.read_pkl(processed_file_path)
        processed_feats = du.parse_chain_feats(
            processed_feats, normalize_positions=True
        )

        framework_mask = (
            1 - processed_feats["cdr_mask"]
        ) * processed_feats[
            "chain_groups"
        ]  # ab framework
        framework_mask = torch.tensor(framework_mask, dtype=int)
        processed_feats["framework_mask"] = framework_mask
        processed_feats["chain_group_index"] = processed_feats[
            "chain_groups"
        ]  # v4

        # leave out non-tensor data
        processed_feats = {
            k: v
            for k, v in processed_feats.items()
            if k
            not in [
                "target_interface_residues",
                "binder_interface_residues",
                "contact_pairs",
                "target_interface_chains",
                "binder_interface_chains",
                "contig",
                "length",
                "binder_motif",
                "binder_motif_chains",
            ]
        }  # v4

        # Only take modeled residues
        modeled_idx = processed_feats.pop("modeled_idx")  # fix-modeled-res
        if len(modeled_idx) != len(
            processed_feats["bb_mask"]
        ):  # contains non canonical residues
            # import pdb; pdb.set_trace()
            processed_feats = tree.map_structure(
                lambda x: x[modeled_idx] if x is not None else None,
                processed_feats,
            )

        # Run through OpenFold data transforms
        chain_feats = {
            "aatype": torch.tensor(
                processed_feats["aatype"]
            ).long(),  # values range from 0 to 19, 0 is ALA
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
        )  # torch.Size([256, 3, 3])
        trans_1 = rigids_1.get_trans()  # torch.Size([256, 3])
        res_plddt = processed_feats["b_factors"][:, 1]  # (256,)
        res_mask = torch.tensor(
            processed_feats["bb_mask"]
        ).int()  # torch.Size([256])

        # reset residue index
        # Re-number residue indices for each chain such that it starts from 1.
        # v4: already starts from 1 in raw feats, skip
        new_res_idx = processed_feats["residue_index"]

        output_feats = {
            "res_plddt": res_plddt,
            "aatype": chain_feats["aatype"],
            "rotmats_1": rotmats_1,
            "trans_1": trans_1,
            "res_mask": res_mask,
            # 'chain_idx': torch.tensor(new_chain_idx),
            "chain_idx": torch.tensor(processed_feats["chain_index"]),
            "chain_group_idx": torch.tensor(
                processed_feats["chain_group_index"]
            ),
            "res_idx": torch.tensor(new_res_idx),
            "original_res_idx": torch.tensor(
                processed_feats["residue_index"]
            ),
            "framework_mask": processed_feats["framework_mask"],
            "all_atom_positions": chain_feats["all_atom_positions"],
            "all_atom_mask": chain_feats["all_atom_mask"],
            "cdr_mask": processed_feats["cdr_mask"],
            "hotspots": csv_row["hotspots"],
            "fix_structure_mask": torch.tensor(
                processed_feats["fix_structure_mask"]
            ),
        }
        output_feats = {
            k: v for k, v in output_feats.items() if v is not None
        }

        if torch.isnan(trans_1).any() or torch.isnan(rotmats_1).any():
            raise ValueError(f"Found NaNs in {processed_file_path}")

        if output_feats is None:
            return None
        output_feats["pdb_name"] = csv_row["pdb_name"]
        output_feats["original_index"] = f"{csv_row['original_index']}"
        return output_feats

    def setup_antibody_mask(self, feats):
        """
        target chain set as diffuse_mask=0, binder chain set as diffuse_mask=1
        """
        target_chain_id = 0
        diffuse_index = torch.nonzero(
            feats["chain_group_idx"] != target_chain_id
        ).reshape(-1)
        diffuse_mask = torch.zeros(
            feats["chain_group_idx"].shape, dtype=torch.int
        )
        diffuse_mask[diffuse_index] = 1  # 0:motif, 1:to diffuse
        if torch.sum(diffuse_mask) < 1:  # Should only happen rarely.
            diffuse_mask = torch.ones_like(diffuse_mask)
        diffuse_mask = diffuse_mask.int()
        return diffuse_mask

    def post_process_feats(self, feats):
        motif_mask = 1 - feats["diffuse_mask"]  # 1:antigen, 0:to diffuse

        trans_1 = feats["trans_1"]
        motif_1 = trans_1 * motif_mask[:, None]
        motif_com = torch.sum(motif_1, dim=0) / (torch.sum(motif_mask) + 1)
        trans_1 -= motif_com[None, :]
        feats["trans_1"] = trans_1
        return feats

    def setup_target_hotspots(self, feats):
        hotspot_mask = torch.zeros_like(feats["diffuse_mask"])
        hotspot_index = json.loads(feats["hotspots"])
        # hotspot_index = [int(part.strip()[1:]) for part in hotspots.split(',')]
        hotspot_mask[hotspot_index] = 1
        return hotspot_mask

    def __getitem__(self, idx):
        csv_row, sample_ids = self.all_sample_ids[idx]
        feats = self.process_csv_row(csv_row)

        antibody_mask = self.setup_antibody_mask(feats)
        feats["diffuse_mask"] = antibody_mask.int()
        feats = self.post_process_feats(feats)

        feats["hotspot_mask"] = self.setup_target_hotspots(feats)

        target_index = np.nonzero(antibody_mask == 0)[:, 0]
        antibody_index = np.nonzero(antibody_mask == 1)[:, 0]
        # target_feats = {k: v[target_index] for k, v in feats.items() if k not in ["pdb_name", "original_index", "hotspots"]}

        target_len = len(target_index)
        antibody_len = len(antibody_index)
        total_length = target_len + antibody_len

        framework_mask = feats["framework_mask"]
        hotspot_mask = feats["hotspot_mask"]
        aatype = feats["aatype"]

        res_mask = torch.ones(total_length, dtype=torch.int)
        chain_idx = feats["chain_idx"]
        chain_group_idx = feats["chain_group_idx"]
        fix_structure_mask = feats["fix_structure_mask"]
        fix_sequence_mask = feats["fix_structure_mask"] + framework_mask
        # Binarize it to 0 or 1
        fix_sequence_mask = (fix_sequence_mask > 0).long()  

        aatype = aatype * fix_sequence_mask + 20 * (1 - fix_sequence_mask)
        pos_fixed_mask = 1 - fix_structure_mask  # 0: fixed, 1: no fixed
        diffuse_mask = pos_fixed_mask

        output_feats = {
            "diffuse_mask": diffuse_mask,
            "hotspot_mask": hotspot_mask,
            "trans_1": feats["trans_1"],
            "rotmats_1": feats["rotmats_1"],
            "aatype": aatype,
            "pdb_name": csv_row["pdb_name"],
            "original_index": sample_ids,
            "sample_ids": sample_ids,
            "res_mask": res_mask,
            "chain_idx": chain_idx,
            "chain_group_idx": chain_group_idx,
            "pos_fixed_mask": pos_fixed_mask,
            "fix_sequence_mask": fix_sequence_mask,
            "framework_mask": framework_mask,
            "cdr_mask": feats["cdr_mask"],
        }
        output_feats = self.post_process_feats(output_feats)
        output_feats.update({"binder_motif_mask": framework_mask})
        return output_feats
