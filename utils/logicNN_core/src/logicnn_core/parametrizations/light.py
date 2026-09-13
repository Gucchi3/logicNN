"""真理値表の各出力をsigmoidで学習するLightパラメータ化を提供する。"""

# torchlogix/parametrization.pyのLightLUTParametrizationを基に再構成した。MIT License。
# Copyright (c) 2021-2023 Dr. Felix Petersen
# Copyright (c) 2024-present Dr. Lino Gerlach
# 許諾条件の全文と変更の由来は同梱LICENSEおよびTHIRD_PARTY_NOTICESを参照。

from __future__ import annotations

import torch
from torch import Tensor

from ..functional.walsh import weighted_light_basis_sum
from .base import LUTParametrization

__all__ = ["LightLUTParametrization"]


class LightLUTParametrization(LUTParametrization):
    """rank 2・4・6のLight係数を生成し、真理値表の多重線形補間を計算する。"""

    _kind = "light"

    def initialize(self, count: int, *, device: torch.device | str | None = None, dtype: torch.dtype | None = None) -> Tensor:
        """residualは正規乱数へ前半−3・後半＋3を加え、randomは一様乱数で初期化する。"""
        device, dtype = self._initialization_options(count, device, dtype)
        if self.config.weight_init == "random":
            return torch.rand(count, self.num_coefficients, device=device, dtype=dtype)
        weights = torch.randn(count, self.num_coefficients, device=device, dtype=dtype)
        weights[:, :self.num_coefficients // 2] -= 3
        weights[:, self.num_coefficients // 2:] += 3
        return weights

    def forward(self, inputs: Tensor, weights: Tensor, *, training: bool, contraction: str) -> Tensor:
        """truth table係数をsamplingしてLight基底と縮約し、入力自体は二値化しない。"""
        inputs       = inputs.to(dtype=weights.dtype)
        coefficients = self._sample_binary(weights, training)
        return weighted_light_basis_sum(inputs, coefficients, contraction, self.num_inputs, materialize_basis=self.config.materialize_basis)

    def truth_tables(self, weights: Tensor) -> Tensor:
        """係数が厳密に正の位置をTrueとして元のゲート・真理値表shapeで返す。"""
        return weights > 0
