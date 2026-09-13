"""総epoch数を周期とする学習率schedulerを生成する。"""

from torch.optim import Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR

from utils.config.schema import SchedulerConfig

SCHEDULER_REGISTRY = {"cosine_annealing": CosineAnnealingLR}

def build_scheduler(config: SchedulerConfig, optimizer: Optimizer, epochs: int) -> CosineAnnealingLR:
    """総epoch数と設定済み最小学習率を指定したcosine schedulerを作る。"""
    return SCHEDULER_REGISTRY[config.name](optimizer, T_max=epochs, eta_min=config.minimum_learning_rate)
