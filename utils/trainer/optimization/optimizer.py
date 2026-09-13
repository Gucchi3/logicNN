"""登録されたoptimizerへ設定値とモデルのParameterを渡す。"""

from collections.abc import Iterable

from torch import nn
from torch.optim import AdamW

from utils.config.schema import OptimizerConfig

OPTIMIZER_REGISTRY = {"adamw": AdamW}

def build_optimizer(config: OptimizerConfig, parameters: Iterable[nn.Parameter]) -> AdamW:
    """lrとweight decayを設定し、betas・eps等はTorch既定値のAdamWを作る。"""
    return OPTIMIZER_REGISTRY[config.name](parameters, lr=config.learning_rate, weight_decay=config.weight_decay)
