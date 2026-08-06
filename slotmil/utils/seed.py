"""Reproducibility helpers. plan.md line 116 requires >=5 seeds with reported
variance, so seeding has to actually cover every source of randomness."""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def set_seed(seed: int, deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if deterministic:
        # Off by default: it costs throughput, and seed-to-seed variance is
        # itself a reported quantity rather than something to suppress.
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def seed_worker(worker_id: int) -> None:
    """DataLoader worker seeding -- without this, workers share a seed and
    stochastic slice subsampling correlates across them."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
