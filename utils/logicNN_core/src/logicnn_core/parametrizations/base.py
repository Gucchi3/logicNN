"""LUT表現の共通設定、型・shape検証と真理値表ID変換を定義する。"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from torch import Tensor, nn

from ..functional.sampling import gumbel_sigmoid, temperature_sigmoid
from ..layer_settings import LUTConfig, _validate_positive_number

__all__ = ["LUTParametrization"]

_FLOAT_DTYPES = (torch.float16, torch.bfloat16, torch.float32, torch.float64)


class LUTParametrization(nn.Module, ABC):
    """重みを所有せず、LUTの学習表現と実行時temperatureを管理する。"""

    _kind: str = ""

    def __init__(self, config: LUTConfig) -> None:
        """方式の一致を検証し、不変の生成時設定と実行用の値を分離する。"""
        super().__init__()
        if config.kind != self._kind:
            raise ValueError(f"{type(self).__name__} のconfig.kindは {self._kind!r} で指定してください")
        self._config          = config
        self.num_inputs       = config.num_inputs
        self.num_coefficients = 16 if config.kind == "raw" else 1 << self.num_inputs
        self.temperature      = float(config.temperature)

    @property
    def config(self) -> LUTConfig:
        """構築時の不変設定を返し、方式と実行属性の整合性を保つ。"""
        return self._config

    def set_temperature(self, temperature: float) -> None:
        """元configを変更せず、検証済みの正の有限temperatureへ更新する。"""
        _validate_positive_number("temperature", temperature)
        self.temperature = float(temperature)

    def _initialization_options(self, count: int, device: torch.device | str | None, dtype: torch.dtype | None) -> tuple[torch.device, torch.dtype]:
        """初期化の件数・dtypeを検証し、省略したdeviceとdtypeをPyTorch既定値で解決する。"""
        if type(count) is not int:
            raise TypeError("count はboolではないPython整数で指定してください")
        if count <= 0 or count * self.num_coefficients > torch.iinfo(torch.int64).max:
            raise ValueError("count は正の値とし、重みの要素数がint64上限を超えないよう指定してください")
        chosen_dtype = torch.get_default_dtype() if dtype is None else dtype
        if chosen_dtype not in _FLOAT_DTYPES:
            raise TypeError("dtype はfloat16／bfloat16／float32／float64で指定してください")
        chosen_device = torch.empty(0, dtype=chosen_dtype, device=device).device
        return chosen_device, chosen_dtype

    def _sample_binary(self, logits: Tensor, training: bool) -> Tensor:
        """sigmoid系の学習選択を行い、評価時は温度や乱数を使わず0を厳密に超えた値を選ぶ。"""
        if not training:
            return (logits > 0).to(dtype=logits.dtype)
        sampling = self.config.sampling
        hard     = sampling in ("hard", "gumbel_hard")
        if sampling.startswith("gumbel_"):
            return gumbel_sigmoid(logits, temperature=self.temperature, hard=hard)
        return temperature_sigmoid(logits, temperature=self.temperature, hard=hard)

    def truth_tables_with_ids(self, weights: Tensor) -> tuple[Tensor, Tensor | None]:
        """bool真理値表とrank 4以下のMSB-first整数IDを返し、高rankのIDはNoneにする。"""
        tables = self.truth_tables(weights)
        if self.num_inputs > 4:
            return tables, None
        shifts = torch.arange((1 << self.num_inputs) - 1, -1, -1, device=tables.device)
        ids    = (tables.to(dtype=torch.int64) << shifts).sum(dim=-1)
        return tables, ids

    @abstractmethod
    def initialize(self, count: int, *, device: torch.device | str | None = None, dtype: torch.dtype | None = None) -> Tensor:
        """方式別の初期重みを生成し、Parameter化と所有を呼出側へ委ねる。"""
        raise NotImplementedError

    @abstractmethod
    def forward(self, inputs: Tensor, weights: Tensor, *, training: bool, contraction: str) -> Tensor:
        """方式別の学習・評価計算を明示された縮約軸に沿って実行する。"""
        raise NotImplementedError

    @abstractmethod
    def truth_tables(self, weights: Tensor) -> Tensor:
        """方式別の重みから末尾に真理値を並べたbool Tensorを返す。"""
        raise NotImplementedError
