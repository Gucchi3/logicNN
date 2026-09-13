"""16個の二入力論理関数をlogitで選択するRawパラメータ化を提供する。"""

# torchlogix/parametrization.pyのRaw方式を基に再構成した。MIT License。
# Copyright (c) 2021-2023 Dr. Felix Petersen
# Copyright (c) 2024-present Dr. Lino Gerlach
# 許諾条件の全文と変更の由来は同梱LICENSEおよびTHIRD_PARTY_NOTICESを参照。

from __future__ import annotations

import math
from string import ascii_letters

import torch
import torch.nn.functional as torch_functional
from torch import Tensor

from ..functional.combinatorics import truth_table_from_id
from ..functional.logic import all_binary_logic_outputs, apply_binary_lut
from ..functional.sampling import temperature_softmax
from .base import LUTParametrization

__all__ = ["RawLUTParametrization"]


class RawLUTParametrization(LUTParametrization):
    """各ゲートが持つ16 logitから論理関数の混合または離散選択を計算する。"""

    _kind = "raw"

    def initialize(self, count: int, *, device: torch.device | str | None = None, dtype: torch.dtype | None = None) -> Tensor:
        """旧残差式でID3を優先するか、標準正規分布で16 logitを初期化する。"""
        device, dtype = self._initialization_options(count, device, dtype)
        if self.config.weight_init == "random":
            return torch.randn(count, self.num_coefficients, device=device, dtype=dtype)
        probability = self.config.residual_probability
        value       = (math.log(15 * probability - 7) - math.log(1 - probability)) * self.temperature
        if not math.isfinite(value) or abs(value) > torch.finfo(dtype).max:
            raise ValueError("Raw残差初期値は指定dtypeで表現できる有限値である必要があります。")
        weights       = torch.zeros(count, self.num_coefficients, device=device, dtype=dtype)
        weights[:, 3] = value
        return weights

    def _sample_weights(self, weights: Tensor, training: bool) -> Tensor:
        """学習時は温度付き選択、評価時は先頭最大IDの決定的one-hotを返す。"""
        if not training:
            return torch_functional.one_hot(weights.argmax(-1), self.num_coefficients).to(dtype=weights.dtype)
        sampling = self.config.sampling
        hard     = sampling in ("hard", "gumbel_hard")
        if sampling.startswith("gumbel"):
            dtype  = torch.float64 if weights.dtype == torch.float64 else torch.float32
            values = weights.to(dtype=dtype)
            # PyTorch標準と同じ指数乱数由来のGumbelノイズを使い、温度除算はC2の安定経路へ渡す。
            noise  = -torch.empty_like(values, memory_format=torch.contiguous_format).exponential_().log()
            return temperature_softmax(values + noise, temperature=self.temperature, hard=hard).to(dtype=weights.dtype)
        return temperature_softmax(weights, temperature=self.temperature, hard=hard)

    def forward(self, inputs: Tensor, weights: Tensor, *, training: bool, contraction: str) -> Tensor:
        """二入力の連続論理基底を、明示された重み・基底のeinsum式で縮約する。"""
        inputs        = inputs.to(dtype=weights.dtype)
        sampled       = self._sample_weights(weights, training)
        first, second = inputs[:, 0], inputs[:, 1]
        if self.config.materialize_basis:
            operands, output       = contraction.split("->")
            weight_axes, basis_axes = operands.split(",")
            coefficient_axis       = next(label for label in ascii_letters if label not in contraction)
            full_contraction       = f"{weight_axes}{coefficient_axis},{basis_axes}{coefficient_axis}->{output}"
            return torch.einsum(full_contraction, sampled, all_binary_logic_outputs(first, second))
        result = torch.einsum(contraction, sampled[..., 0], apply_binary_lut(first, second, 0))
        for lut_id in range(1, self.num_coefficients):
            result = result + torch.einsum(contraction, sampled[..., lut_id], apply_binary_lut(first, second, lut_id))
        return result

    def truth_tables(self, weights: Tensor) -> Tensor:
        """末尾16 logitの先頭最大IDを選び、先行shapeを保つbool真理値表を返す。"""
        return truth_table_from_id(weights.argmax(-1), self.num_inputs)
