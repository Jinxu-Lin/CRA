"""
Seed management for full reproducibility.

Covers: torch, numpy, random, CUDA, cuDNN deterministic, DataLoader worker seeds.
"""

import random
import numpy as np
import torch


def set_seed(seed: int):
    """
    Set all random seeds for reproducibility.

    Covers torch, numpy, random, CUDA, cuDNN.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id: int):
    """
    DataLoader worker init function for reproducible data loading.

    Use as: DataLoader(..., worker_init_fn=seed_worker)
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_generator(seed: int) -> torch.Generator:
    """
    Create a torch Generator with a specific seed.

    Use as: DataLoader(..., generator=get_generator(seed))
    """
    g = torch.Generator()
    g.manual_seed(seed)
    return g
