"""Project-wide utilities.

Call ``seed_everything`` once at the top of an entry script, before any
module that consumes randomness is constructed. Pass the same seed through
``config["seed"]`` so library code reads one source of truth.
"""

import os
import random

import numpy as np
import torch


def seed_everything(seed: int) -> int:
    """Seed python, numpy and torch (CPU + CUDA) for reproducibility.

    Also forces cuDNN into deterministic mode. Call this once at the entry
    point, before dataset construction and model init.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    return seed
