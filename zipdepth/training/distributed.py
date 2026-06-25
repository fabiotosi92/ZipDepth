"""Distributed training utilities."""

import gc
import os
import random

import numpy as np
import torch
import torch.distributed as dist


def setup_distributed():
    """Initialise NCCL process group from torchrun environment variables.

    Returns:
        (rank, world_size, local_rank, is_distributed)
    """
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank       = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
        dist.init_process_group(backend='nccl', init_method='env://',
                                rank=rank, world_size=world_size)
        torch.cuda.set_device(local_rank)
        return rank, world_size, local_rank, True
    return 0, 1, 0, False


def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()


def barrier():
    if dist.is_initialized():
        dist.barrier()


def print_main(msg: str, rank: int):
    if rank == 0:
        print(msg)


def fix_state_dict_prefix(state_dict: dict, is_distributed: bool) -> dict:
    """Add or strip the 'module.' DDP prefix to match the model wrapper."""
    first_key = next(iter(state_dict))
    if is_distributed and not first_key.startswith('module.'):
        return {f'module.{k}': v for k, v in state_dict.items()}
    if not is_distributed and first_key.startswith('module.'):
        return {k[len('module.'):]: v for k, v in state_dict.items()}
    return state_dict


def worker_init_fn(worker_id: int):
    """Seed each DataLoader worker independently for reproducibility."""
    gc.set_threshold(100, 5, 5)
    gc.enable()
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


class WorkerDistributedSampler(torch.utils.data.Sampler):
    """Memory-efficient distributed sampler — each rank holds only its own indices."""

    def __init__(self, dataset, num_replicas=None, rank=None,
                 shuffle=True, seed=4321, drop_last=False):
        if num_replicas is None:
            num_replicas = dist.get_world_size()
        if rank is None:
            rank = dist.get_rank()

        self.dataset      = dataset
        self.num_replicas = num_replicas
        self.rank         = rank
        self.epoch        = 0
        self.drop_last    = drop_last
        self.shuffle      = shuffle
        self.seed         = seed

        n = len(self.dataset)
        if self.drop_last and n % self.num_replicas != 0:
            self.num_samples = n // self.num_replicas
        else:
            self.num_samples = (n + self.num_replicas - 1) // self.num_replicas

        self.total_size = self.num_samples * self.num_replicas
        print(f"[Rank {self.rank}] WorkerDistributedSampler: "
              f"{self.num_samples:,} indices "
              f"({self.num_samples * 28 / 1024 / 1024:.1f} MB)")

    def __iter__(self):
        n       = len(self.dataset)
        indices = list(range(self.rank, n, self.num_replicas))
        if self.shuffle:
            g    = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            perm = torch.randperm(len(indices), generator=g).tolist()
            indices = [indices[i] for i in perm]
        if len(indices) < self.num_samples:
            indices += indices[:self.num_samples - len(indices)]
        return iter(indices[:self.num_samples])

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch: int):
        self.epoch = epoch
