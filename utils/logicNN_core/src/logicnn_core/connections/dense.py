"""固定Dense接続と旧torchlogixの代替勾配を持つ学習可能接続を提供する。"""

# torchlogix/connections.pyのDense接続を基に再構成した。MIT License。
# Copyright (c) 2021-2023 Dr. Felix Petersen
# Copyright (c) 2024-present Dr. Lino Gerlach
# 許諾条件の全文と変更の由来は同梱LICENSEおよびTHIRD_PARTY_NOTICESを参照。

from __future__ import annotations

import math
from typing import Any

import torch
from torch import Tensor

from ..functional.sampling import temperature_softmax
from ..layer_settings import ConnectionConfig, _validate_positive_integer, _validate_positive_number
from .base import Connections

__all__ = ["FixedDenseConnections", "LearnableDenseConnections"]

_FLOAT_DTYPES = (torch.float16, torch.bfloat16, torch.float32, torch.float64)
_INT64_MAX    = torch.iinfo(torch.int64).max


def _validate_dense_config(config: ConnectionConfig, kind: str, in_features: int, out_features: int, num_inputs: int, candidates: int = 1) -> None:
    """Denseの方式・次元・適用できる設定とindex総数を検証する。"""
    if config.kind != kind:
        raise ValueError(f"このDense接続のconfig.kindは{kind!r}で指定してください。")
    if config.channel_group_size is not None:
        raise ValueError("channel_group_sizeはDense接続には指定できません。")
    _validate_positive_integer("in_features", in_features)
    _validate_positive_integer("out_features", out_features)
    if in_features > _INT64_MAX or out_features * num_inputs * candidates > _INT64_MAX:
        raise ValueError("Denseの特徴数またはindex要素数がint64上限を超えます。")
    if config.init == "random_unique" and candidates * num_inputs > in_features:
        raise ValueError("random_uniqueではin_features >= num_inputs * num_candidatesが必要です。")


def _unique_indices(in_features: int, selections: int, out_features: int, device: torch.device) -> Tensor:
    """各ゲート内だけで重複しない順序付き候補を選び、ゲート間の共有は許す。"""
    columns = [torch.randperm(in_features, device=device)[:selections].clone() for _ in range(out_features)]
    return torch.stack(columns, dim=-1)


class _LearnableConnectionFunction(torch.autograd.Function):
    """forwardは候補を1本選び、backwardだけ旧独自の重み・入力勾配を使用する。"""

    @staticmethod
    def forward(ctx: Any, inputs: Tensor, weights: Tensor, temperature: float, use_gumbel: bool, indices: Tensor) -> Tensor:
        """任意Gumbelノイズ付きargmaxで各端子の接続先を選択する。"""
        dtype  = torch.float64 if weights.dtype == torch.float64 else torch.float32
        values = weights.to(dtype=dtype)
        if use_gumbel:
            uniform = torch.rand_like(values)
            noise   = -torch.log(-torch.log(uniform + 1e-20) + 1e-20)
        else:
            noise = torch.zeros_like(values)
        choices  = (values + noise).argmax(dim=0, keepdim=True)
        selected = indices.gather(0, choices).squeeze(0)
        ctx.temperature = temperature
        ctx.save_for_backward(inputs, weights, noise, indices)
        return inputs[:, selected]

    @staticmethod
    def backward(ctx: Any, output_gradient: Tensor) -> tuple[Tensor | None, Tensor | None, None, None, None]:
        """旧代替勾配を広い演算dtypeで集計し、入力と重みそれぞれのdtypeへ戻す。"""
        inputs, weights, noise, indices = ctx.saved_tensors
        dtype          = torch.float64 if torch.float64 in (inputs.dtype, weights.dtype) else torch.float32
        values         = inputs.to(dtype=dtype)
        gradient       = output_gradient.to(dtype=dtype)
        input_gradient = weight_gradient = None
        if ctx.needs_input_grad[1]:
            candidate_values = 2 * values[:, indices] - 1
            weight_gradient  = torch.einsum("bkro,bro->kro", candidate_values, gradient).to(dtype=weights.dtype)
        if ctx.needs_input_grad[0]:
            logits        = weights.to(dtype=dtype) + noise.to(dtype=dtype)
            probabilities = temperature_softmax(logits, temperature=ctx.temperature, dim=0)
            contribution  = probabilities.unsqueeze(0) * gradient.unsqueeze(1)
            positions     = indices.reshape(1, -1).expand(values.shape[0], -1)
            input_gradient = torch.zeros_like(values)
            input_gradient.scatter_add_(1, positions, contribution.reshape(values.shape[0], indices.numel()))
            input_gradient = input_gradient.to(dtype=inputs.dtype)
        return input_gradient, weight_gradient, None, None, None


class FixedDenseConnections(Connections):
    """各LUTへ渡す入力位置をpersistent bufferとして固定する。"""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        num_inputs: int = 2,
        config: ConnectionConfig = ConnectionConfig(),
        device: torch.device | str | None = None,
    ) -> None:
        """固定接続の設定を検証し、各LUTの入力indexを初期化する。"""
        super().__init__(num_inputs, config)
        _validate_dense_config(config, "fixed", in_features, out_features, num_inputs)
        self.in_features  = in_features
        self.out_features = out_features
        chosen_device     = torch.empty(0, device=device).device
        if config.init == "random_unique":
            indices = _unique_indices(in_features, num_inputs, out_features, chosen_device)
        else:
            indices = torch.randint(in_features, (num_inputs, out_features), device=chosen_device)
        self.register_buffer("indices", indices)

    def get_indices(self) -> Tensor:
        """固定接続indexを[num_inputs, out_features]で返す。"""
        return self.indices

    def forward(self, inputs: Tensor) -> Tensor:
        """任意の先行軸をflattenし、入力dtypeを保って各LUTの入力を取り出す。"""
        values = inputs.reshape(math.prod(inputs.shape[:-1]), self.in_features)
        return values[:, self.indices]


class LearnableDenseConnections(Connections):
    """固定候補集合から各端子の接続を学習し、eval時は決定的に選択する。"""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        num_inputs: int = 2,
        config: ConnectionConfig = ConnectionConfig(kind="learnable"),
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        """候補bufferと[0,1)の接続logitを指定device・dtypeへ生成する。"""
        super().__init__(num_inputs, config)
        _validate_positive_integer("in_features", in_features)
        candidates = in_features if config.num_candidates is None else config.num_candidates
        _validate_dense_config(config, "learnable", in_features, out_features, num_inputs, candidates)
        chosen_dtype = torch.get_default_dtype() if dtype is None else dtype
        if chosen_dtype not in _FLOAT_DTYPES:
            raise TypeError("dtypeはfloat16／bfloat16／float32／float64で指定してください。")
        chosen_device       = torch.empty(0, device=device).device
        self.in_features    = in_features
        self.out_features   = out_features
        self.num_candidates = candidates
        self.temperature    = float(config.temperature)
        shape               = (candidates, num_inputs, out_features)
        if config.num_candidates is None:
            indices = torch.arange(in_features, device=chosen_device).view(in_features, 1, 1).expand(shape).contiguous()
        elif config.init == "random_unique":
            indices = _unique_indices(in_features, candidates * num_inputs, out_features, chosen_device).reshape(shape)
        else:
            indices = torch.randint(in_features, shape, device=chosen_device)
        self.register_buffer("indices", indices)
        self.weights = torch.nn.Parameter(torch.rand(shape, device=chosen_device, dtype=chosen_dtype))

    def set_temperature(self, temperature: float) -> None:
        """正の有限temperatureへ更新し、元の不変configを変更しない。"""
        _validate_positive_number("temperature", temperature)
        self.temperature = float(temperature)

    def get_indices(self) -> Tensor:
        """乱数なしの先頭最大候補を[num_inputs, out_features]で返す。"""
        choices = self.weights.argmax(dim=0, keepdim=True)
        return self.indices.gather(0, choices).squeeze(0)

    def forward(self, inputs: Tensor) -> Tensor:
        """学習時は旧独自勾配、評価時は通常gatherで入力値とdtypeを保持する。"""
        values = inputs.reshape(math.prod(inputs.shape[:-1]), self.in_features)
        if self.training:
            return _LearnableConnectionFunction.apply(values, self.weights, self.temperature, self.config.use_gumbel, self.indices)
        choices = self.weights.argmax(dim=0, keepdim=True)
        indices = self.indices.gather(0, choices).squeeze(0)
        return values[:, indices]
