import os
import time
from tqdm import tqdm
import torch

from utils.loader import load_seed, load_device, load_ema, load_checkpoint, load_config
from utils.logger import Logger, set_log
from utils.train_utils import count_parameters
from pathlib import Path
from dataset_cluster import get_dataloader
from utils.structure_utils import create_structure_from_crds
from utils.sidechain_utils import Idealizer
from models.cnf import CNF
from models.confidence import Confidence
from models.equiformer_v2.equiformer_v2 import EquiformerV2
from utils.metrics import metrics_per_chi, atom_rmsd
import math
import shutil
from utils.constants import chi_mask as chi_mask_true
from utils.constants import atom14_mask as atom_mask_true

class Sampler(object):
    def __init__(self, config, use_gt_masks=False, ddp=False):
        super(Sampler, self).__init__()
        self.config = config
        self.use_gt_masks = use_gt_masks
        self.seed = load_seed(self.config.seed)
        self.device = load_device()
        self.train_loader, self.test_loader, _, _ = get_dataloader(self.config, ddp=ddp, sample=True)
        self.idealizer = Idealizer(use_native_bb_coords=True)

    def sample(self, ts, name='test', save_traj=False, inpaint=''):
        self.config.exp_name = ts
        self.ckpt = f'{ts}'

        print(f'{self.ckpt}')
        ckpt_dict = torch.load(self.config.ckpt)
        train_cfg = ckpt_dict['config']
        self.log_folder_name, self.log_dir, self.ckpt_dir = set_log(train_cfg)
        self.model = CNF(EquiformerV2(**train_cfg.model), train_cfg, coeff=self.config.sample.coeff,
                         stepsize=self.config.sample.num_steps, mode=self.config.mode).cuda()
        # self.model = torch.compile(self.model)
        print(f'Number of parameters: {count_parameters(self.model)}')
        self.ema = load_ema(self.model, decay=train_cfg.train.ema)
        self.model, self.ema = load_checkpoint(self.model, self.ema, ckpt_dict)
        self.model.eval()
        self.ema.copy_to(self.model.parameters())

        if self.config.conf_ckpt is not None:
            conf_ckpt = torch.load(self.config.conf_ckpt)
            self.conf_model = Confidence(EquiformerV2(**conf_ckpt['config'].model), conf_ckpt['config']).cuda()
            if 'module.' in list(conf_ckpt["state_dict"].keys())[0]:
                state_dict = {k[7:]: v for k, v in conf_ckpt["state_dict"].items()}
            self.conf_model.load_state_dict(state_dict)

        logger = Logger(str(os.path.join(self.log_dir, f'{self.ckpt}.log')), mode='a')
        logger.log(f'{self.ckpt}', verbose=False)

        save_path = Path(f'./samples/{name}')
        save_path.mkdir(exist_ok=True, parents=True)

        # sample_path = save_path.joinpath(ts)
        sample_path = save_path
        sample_path.mkdir(exist_ok=True, parents=True)

        output_dict = {}
        with torch.no_grad():
            for batch in tqdm(self.test_loader):
                batch = batch.to(f'cuda:{self.device[0]}')
                aa_str, aa_onehot, aa_num, coords, mask, atom_mask, batch_id, pdb_codes = batch.aa_str, batch.aa_onehot, batch.aa_num, \
                                                                                          batch.pos, batch.aa_mask, batch.atom_mask, batch.batch, batch.id
                chi, chi_alt, chi_mask = batch.chi, batch.chi_alt, batch.chi_mask
                bb_coords = coords[:, :4]

                if self.use_gt_masks:
                    chi_mask = chi_mask_true.to(aa_num)
                    chi_mask = chi_mask[aa_num]
                    batch.chi_mask = chi_mask

                    atom_mask = atom_mask_true.to(atom_mask)
                    atom_mask = atom_mask[aa_num]
                    batch.atom_mask = atom_mask

                chi = (chi + math.pi) * chi_mask
                chi_alt = (chi_alt + math.pi) * chi_mask

                batch_size = batch_id.max().item() + 1
                output_dict = {**output_dict, **{i:{} for i in pdb_codes}}

                with torch.no_grad():
                    best_pred_idx, best_pred_rmsd, best_gt_idx, best_gt_rmsd = 0, 999, 0, 999
                    for sample_idx in range(self.config.sample.n_samples):
                        # check if files exist
                        exists = True
                        for i in range(batch_size):
                            pdb_path = sample_path.joinpath(f"run_{sample_idx + 1}", f"{pdb_codes[i]}.pdb")
                            if not pdb_path.exists():
                                exists = False

                        if exists: continue

                        pred_sc = self.model.decode(batch, return_traj=save_traj, inpaint=inpaint)
                        pred_sc = (pred_sc - math.pi) * chi_mask  # shift torsions back to [-pi,pi]

                        if save_traj:
                            pred_sc_traj = pred_sc.clone()
                            pred_sc = pred_sc[-1]

                        all_atom_coords = self.idealizer(aa_num, bb_coords, pred_sc) * atom_mask.unsqueeze(-1)
                        gt_idealized = self.idealizer(aa_num, bb_coords, chi-math.pi) * atom_mask.unsqueeze(-1)
                        pred_sc = (pred_sc + math.pi) * chi_mask

                        for i in range(batch_size):
                            metrics = {}
                            chi_batch = chi[batch_id == i]
                            chi_alt_batch = chi_alt[batch_id == i]
                            chi_mask_batch = chi_mask[batch_id == i]
                            pred_batch = pred_sc[batch_id == i]
                            atom_mask_batch = atom_mask[batch_id == i]
                            crds_batch = coords[batch_id == i]
                            crds_batch_idealized = gt_idealized[batch_id == i]
                            pred_pos_batch = all_atom_coords[batch_id == i]
                            chain_id_batch = batch.chain_id[i]
                            res_id_batch = batch.res_id[i]
                            icode_batch = batch.icode[i]

                            # calculate core and surface residues
                            # core: 20 Cb within 10A, surface: at most 15 Cb in 10A
                            cb_dist = torch.cdist(crds_batch[:,4], crds_batch[:,4])
                            cb_dist_w10 = ((cb_dist < 10) * cb_dist != 0).sum(-1)
                            core = cb_dist_w10 >= 20
                            surface = cb_dist_w10 <= 15
                            mae, acc = metrics_per_chi(pred_batch, chi_batch, chi_alt_batch, chi_mask_batch)
                            core_mae, core_acc = metrics_per_chi(pred_batch[core], chi_batch[core], chi_alt_batch[core], chi_mask_batch[core])
                            surface_mae, surface_acc = metrics_per_chi(pred_batch[surface], chi_batch[surface], chi_alt_batch[surface],
                                                                 chi_mask_batch[surface])
                            rmsd = atom_rmsd(pred_pos_batch[:,4:], crds_batch[:,4:], atom_mask_batch[:,4:])
                            rmsd_idealized = atom_rmsd(pred_pos_batch[:,4:], crds_batch_idealized[:,4:], atom_mask_batch[:,4:])
                            core_rmsd = atom_rmsd(pred_pos_batch[core][:,4:], crds_batch[core][:,4:], atom_mask_batch[core][:,4:])
                            surface_rmsd = atom_rmsd(pred_pos_batch[surface][:,4:], crds_batch[surface][:,4:], atom_mask_batch[surface][:,4:])
                            # clash = count_clashes(pred_pos_batch, atom_type_batch, atom_mask_batch)
                            clash = 0

                            metrics['angle_mae'] = mae
                            metrics['angle_acc'] = acc
                            metrics['core_mae'] = core_mae
                            metrics['core_acc'] = core_acc
                            metrics['surf_mae'] = surface_mae
                            metrics['surf_acc'] = surface_acc
                            metrics['atom_rmsd'] = rmsd
                            metrics['atom_rmsd_ideal'] = rmsd_idealized
                            metrics['core_rmsd'] = core_rmsd
                            metrics['surface_rmsd'] = surface_rmsd
                            metrics['clash'] = clash

                            output_dict[pdb_codes[i]][f'run_{sample_idx+1}'] = metrics

                            # save structure
                            pdb_path = sample_path.joinpath(f"run_{sample_idx+1}",f"{pdb_codes[i]}.pdb")
                            pdb_path.parent.mkdir(exist_ok=True,parents=True)

                            if save_traj:
                                aa_batch = aa_num[batch_id==i]
                                crds_traj = []
                                for traj in pred_sc_traj:
                                    traj_batch = traj[batch_id==i]
                                    crds = self.idealizer(aa_batch, pred_pos_batch[:,:4], traj_batch) * atom_mask_batch.unsqueeze(-1)
                                    crds_traj.append(crds)
                                crds_traj = torch.stack(crds_traj)
                                create_structure_from_crds(aa_str[i], crds_traj.cpu(), atom_mask_batch.cpu(), chain_id_batch,
                                                           resseq=res_id_batch, icode=icode_batch, outPath=str(pdb_path), save_traj=True)
                            else:
                                create_structure_from_crds(aa_str[i], pred_pos_batch.cpu(), atom_mask_batch.cpu(), chain_id_batch,
                                                       resseq=res_id_batch, icode=icode_batch, outPath=str(pdb_path), save_traj=False)

                            if self.config.conf_ckpt is not None:
                                best_path = sample_path.joinpath('best_run', f"{pdb_codes[i]}.pdb")
                                best_path.parent.mkdir(exist_ok=True, parents=True)

                                pred_rmsd, gt_rmsd = self.conf_model.get_pred(pred_sc, batch)
                                pred_rmsd = pred_rmsd.mean().item()
                                gt_rmsd = gt_rmsd.mean().item()
                                if best_pred_rmsd > pred_rmsd:
                                    best_pred_idx = sample_idx
                                    best_pred_rmsd = pred_rmsd
                                    shutil.copy(sample_path.joinpath(f'run_{best_pred_idx+1}', f"{pdb_codes[i]}.pdb"), best_path)
                                    if best_gt_rmsd > gt_rmsd:
                                        best_gt_idx = sample_idx
                                        best_gt_rmsd = gt_rmsd

                                    output_dict[pdb_codes[i]]['best_pred_idx'] = best_pred_idx + 1
                                    output_dict[pdb_codes[i]]['best_gt_idx'] = best_gt_idx + 1
                                    output_dict[pdb_codes[i]]['best_pred_rmsd'] = best_pred_rmsd
                                    output_dict[pdb_codes[i]]['best_gt_rmsd'] = best_gt_rmsd

            torch.save(output_dict, sample_path.joinpath('output_dict.pth'))

        print(' ')
        return self.ckpt

if __name__ == '__main__':
    import argparse

    start_time = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str)
    parser.add_argument('name', type=str, default='test')
    parser.add_argument('--save_traj', type=bool, default=False)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--use_gt_masks', type=bool, default=False)
    parser.add_argument('--inpaint', type=str, default='')

    args = parser.parse_args()

    config = load_config(args.config, seed=args.seed, inference=True)
    sampler = Sampler(config, args.use_gt_masks)
    sampler.sample(time.strftime('%b%d-%H:%M:%S', time.gmtime()), name=args.name, save_traj=args.save_traj, inpaint=args.inpaint)
    print(f'Inference took a total of {time.time() - start_time} seconds')