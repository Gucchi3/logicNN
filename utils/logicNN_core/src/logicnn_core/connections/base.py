"""固定・学習可能接続の共通設定とinterfaceを定義する。"""

from abc import ABC, abstractmethod

from torch import Tensor, nn

from ..layer_settings import ConnectionConfig, _validate_positive_integer

__all__ = ["Connections"]


class Connections(nn.Module, ABC):
    """接続固有のTensorの所有を具象classへ委ね、不変設定と入力本数を管理する。"""

    def __init__(self, num_inputs: int, config: ConnectionConfig) -> None:
        """入力本数と設定型を検証し、生成時のconfigを保持する。"""
        super().__init__()
        _validate_positive_integer("num_inputs", num_inputs)
        if not isinstance(config, ConnectionConfig):
            raise TypeError("config はConnectionConfigで指定してください")
        self.num_inputs = num_inputs
        self._config    = config

    @property
    def config(self) -> ConnectionConfig:
        """生成時の不変設定を返し、別設定への差替えを禁止する。"""
        return self._config

    @abstractmethod
    def forward(self, inputs: Tensor) -> Tensor:
        """入力から各論理ゲートへ渡す値を選択する。"""
        raise NotImplementedError
