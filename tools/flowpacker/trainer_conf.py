import os
import time
from tqdm import tqdm, trange
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from utils.loader import load_seed,  load_ema, load_config
from utils.logger import Logger, set_log, start_log
from utils.train_utils import count_parameters
from pathlib import Path
from dataset_conf import get_dataloader as get_dataloader
from utils.structure_utils import create_structure_from_crds
from loss_conf import ConfLoss
from models.cnf import CNF
from models.equiformer_v2.equiformer_v2 import EquiformerV2
import math
from utils.metrics import metrics_per_chi, atom_rmsd
from torch.nn.parallel import DistributedDataParallel as DDP

class Trainer(object):
    def __init__(self, config, ddp=False, device=None):
        super(Trainer, self).__init__()
        self.config = config
        self.ddp = ddp
        self.log_folder_name, self.log_dir, self.ckpt_dir = set_log(self.config)
        self.seed = load_seed(self.config.seed)
        self.device = device if device is not None else 'cuda'
        self.train_loader, self.test_loader, self.train_sampler, self.test_sampler = get_dataloader(self.config,ddp=ddp)

    def train(self, ts, resume=False):
        self.config.exp_name = ts
        self.ckpt = f'{ts}'

        # -------- Load models, optimizers, ema --------
        config.model.pop('name')
        self.model = CNF(EquiformerV2(**self.config.model).cuda(), self.config)

        if self.ddp:
            self.model = DDP(self.model, device_ids=[self.device], find_unused_parameters=False)

        # self.model = torch.compile(self.model)
        print(f'Number of parameters: {count_parameters(self.model)}')
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.config.train.lr,
                                    weight_decay=self.config.train.weight_decay)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=self.config.train.lr_decay)
        self.ema = load_ema(self.model, decay=self.config.train.ema)

        print(f'{self.ckpt}')

        if resume and self.config.ckpt is not None:
            ckpt_dict = torch.load(f'./checkpoints/{self.config.data.data}/{self.config.train.name}/{self.config.ckpt}.pth')
            self.model.load_state_dict(ckpt_dict['state_dict'])
            self.optimizer.load_state_dict(ckpt_dict['optimizer'])
            self.ema.load_state_dict(ckpt_dict['ema'])
            start_epoch = int(self.config.ckpt.split("_")[-1])
            print(f'Loaded checkpoint {self.config.ckpt}')
        else:
            start_epoch = 0

        logger = Logger(str(os.path.join(self.log_dir, f'{self.ckpt}.log')), mode='a')
        logger.log(f'{self.ckpt}', verbose=False)
        start_log(logger, self.config)

        writer = SummaryWriter(os.path.join(*['logs_train', 'tensorboard', self.config.data.data,
                                            self.config.train.name, self.config.exp_name]))

        save_path = Path(f'./checkpoints/{self.config.data.data}/{self.config.train.name}/')
        save_path.mkdir(exist_ok=True, parents=True)

        sample_path = save_path.joinpath('samples')
        sample_path.mkdir(exist_ok=True, parents=True)

        self.loss_fn = ConfLoss(self.model, self.config)
        num_iter = 0
        # -------- Training --------
        for epoch in trange(start_epoch, (self.config.train.num_epochs), desc = '[Epoch]', position = 1, leave=False):
            self.train_loss, self.train_chi, self.train_atom, self.train_dist = [],[],[],[]
            self.model.train()
            if self.ddp:
                self.train_sampler.set_epoch(epoch)
                self.test_sampler.set_epoch(epoch)
            start_time = time.time()

            for _, train_b in enumerate(self.train_loader):
                num_iter += 1
                train_b = train_b.to(self.device)
                self.optimizer.zero_grad()
                loss = self.loss_fn(train_b)
                loss.backward()

                if self.config.train.grad_norm > 0:
                    grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.train.grad_norm)

                self.optimizer.step()
                
                # -------- EMA update --------
                self.ema.update(self.model.parameters())
                self.train_loss.append(loss.item())

            if self.config.train.lr_schedule:
                self.scheduler.step()

            self.model.eval()
            test_loss_total, test_loss_chi, test_loss_atom, test_loss_dist = [], [], [], []
            with torch.no_grad():
                for _, test_b in enumerate(self.test_loader):
                    test_b = test_b.to(self.device)
                    loss = self.loss_fn(test_b)
                    test_loss_total.append(loss.item())

            if not self.ddp or dist.get_rank() == 0:
                mean_test_loss = np.mean(test_loss_total)
                mean_train_total = np.mean(self.train_loss)

                writer.add_scalar("train_loss", mean_train_total, epoch)
                writer.add_scalar("test_loss", mean_test_loss, epoch)
                writer.flush()

                # -------- Log losses --------
                logger.log(f'[EPOCH {epoch+1:04d}] | time: {time.time()-start_time:.2f} sec | '
                               f'train loss: {mean_train_total:.3e} | '
                               f'test loss: {mean_test_loss:.3e}', verbose=False)

                if epoch % self.config.train.print_interval == self.config.train.print_interval-1:
                    tqdm.write(f'[EPOCH {epoch+1:04d}] | time: {time.time()-start_time:.2f} sec | '
                               f'train loss: {mean_train_total:.3e} | '
                               f'test loss: {mean_test_loss:.3e}')

                # -------- Save checkpoints --------
                if epoch % self.config.train.save_interval == self.config.train.save_interval-1:
                    save_name = f'_{epoch+1}' if epoch < self.config.train.num_epochs - 1 else ''
                    torch.save({
                        'config': self.config,
                        'state_dict': self.model.state_dict(),
                        'ema': self.ema.state_dict(),
                        'optimizer': self.optimizer.state_dict()
                        }, save_path.joinpath(f'{self.ckpt + save_name}.pth'))

        print(' ')
        return self.ckpt


if __name__ == '__main__':
    import argparse
    import torch.distributed as dist
    import torch.utils.data.distributed

    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str)
    parser.add_argument('--resume', type=bool, default=False)
    parser.add_argument('--ddp', type=bool, default=False)
    parser.add_argument('--seed', type=int, default=42)

    parser.add_argument('--init_method', default='tcp://127.0.0.1:3456', type=str, help='')
    parser.add_argument('--dist-backend', default='nccl', type=str, help='')
    parser.add_argument('--world_size', default=1, type=int, help='')
    parser.add_argument('--distributed', action='store_true', help='')

    args = parser.parse_args()

    current_device = None
    if args.ddp:
        ngpus_per_node = torch.cuda.device_count()

        local_rank = int(os.environ.get("SLURM_LOCALID"))
        rank = int(os.environ.get("SLURM_NODEID")) * ngpus_per_node + local_rank

        current_device = local_rank

        torch.cuda.set_device(current_device)

        # init the process group
        dist.init_process_group(backend=args.dist_backend, init_method=args.init_method, world_size=args.world_size,
                                rank=rank)

    config = load_config(args.config, seed=args.seed)
    trainer = Trainer(config, ddp=args.ddp, device=current_device)
    trainer.train(time.strftime('%b%d-%H:%M:%S', time.gmtime()),resume=args.resume)
