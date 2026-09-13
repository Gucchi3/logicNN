"""1 epochの学習・評価と、その集計結果の型を公開する。"""

from .epoch import EpochMetrics, evaluate, train_one_epoch

__all__ = ["EpochMetrics", "evaluate", "train_one_epoch"]
