"""Determinism.

Runs of the earlier pipeline spread between 0.52 and 0.76 on a single
configuration because no seed was fixed and cuDNN was left in its default,
non-deterministic mode. Every run must pass through this function.
"""
from __future__ import annotations

import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
