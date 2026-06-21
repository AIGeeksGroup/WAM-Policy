from __future__ import annotations

import random

import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np
    except ModuleNotFoundError:
        np = None

    if np is not None:
        np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
