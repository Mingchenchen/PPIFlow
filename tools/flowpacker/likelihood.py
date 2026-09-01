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

    def sample(self, ts, name='test'):
        self.config.exp_name = ts
        self.ckpt = f'{ts}'

        print(f'{self.ckpt}')
        ckpt_dict = torch.load(self.config.ckpt)
        train_cfg = ckpt_dict['config']
        self.log_folder_name, self.log_dir, self.ckpt_dir = set_log(train_cfg)
        print(self.config.sample)
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

        save_path = Path(f'./likelihood')
        save_path.mkdir(exist_ok=True, parents=True)

        out_dict = {}
        for batch in tqdm(self.test_loader):
            batch_id = batch.id[0]
            batch = batch.to(f'cuda:{self.device[0]}')
            mask = batch.chi_mask
            out_dict[batch_id] = {}

            logp_list = []
            for _ in range(self.config.sample.n_samples):
                with torch.no_grad():
                    likelihood = self.model.log_prob(batch).detach().cpu()
                    logp_list.append(likelihood)
            logp_all = torch.stack(logp_list,dim=0)
            logp_mean = torch.mean(logp_all,dim=0) * mask.detach().cpu()

            out_dict[batch_id]['logp_raw'] = logp_mean
            out_dict[batch_id]['logp_sum'] = logp_mean.sum().item()
            out_dict[batch_id]['logp_mean'] = (logp_mean.sum() / mask.sum()).item()

        torch.save(out_dict, save_path.joinpath(f'{name}.pth'))

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
    sampler.sample(time.strftime('%b%d-%H:%M:%S', time.gmtime()), name=args.name)
    print(f'Inference took a total of {time.time() - start_time} seconds')