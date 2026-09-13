"""logicNNが使用する乱数生成器へ共通seedを設定する。"""

from __future__ import annotations

import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Python、NumPy、PyTorch CPUおよび利用可能なCUDAへseedを設定する。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
