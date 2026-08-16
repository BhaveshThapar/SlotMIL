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
    """Reseed the *legacy global* RNGs inside each DataLoader worker.

    This covers ``np.random.*`` module-level calls and ``random``, and nothing
    else. It specifically does **not** reach a ``np.random.Generator`` held on a
    dataset object: those are forked with identical state and
    ``np.random.seed()`` cannot touch them. This docstring used to claim it
    stopped slice subsampling correlating across workers; it never could, and
    ``FeatureBagDataset`` now derives its subsample from ``(seed, epoch, index)``
    rather than from worker-forked Generator state. Keep this function for the
    globals -- just do not build new stochasticity on top of it.
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
