"""学習用と評価用の交差エントロピーを明示的に生成する。"""

from torch import nn

from utils.config.schema import LossConfig

LOSS_REGISTRY = {"cross_entropy": nn.CrossEntropyLoss}

def build_training_loss(config: LossConfig) -> nn.CrossEntropyLoss:
    """学習用に、設定済みlabel smoothingを適用した損失関数を生成する。"""
    return LOSS_REGISTRY[config.name](label_smoothing=config.label_smoothing, reduction="mean")


def build_evaluation_loss(config: LossConfig) -> nn.CrossEntropyLoss:
    """検証・テスト用に、label smoothingを適用しない損失関数を生成する。"""
    return LOSS_REGISTRY[config.name](label_smoothing=0.0, reduction="mean")
