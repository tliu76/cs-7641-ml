"""
Seed control for reproducible, leakage-safe experiments.

Uncertainty in this assignment is operational: seed sensitivity, trajectory
variance, and stability bands. Every run must set and log its seed so that
stable effects can be separated from lucky runs.
"""

from __future__ import annotations

import os
import random

import numpy as np


def set_all_seeds(seed: int, deterministic: bool = True) -> None:
    """Seed Python, NumPy, and PyTorch (CPU + CUDA) for one run."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:  # torch not installed yet; NumPy/random still seeded.
        pass


def seed_worker(worker_id: int) -> None:
    """DataLoader worker init for reproducible shuffling."""
    worker_seed = (int(os.environ.get("PYTHONHASHSEED", "0")) + worker_id) % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
