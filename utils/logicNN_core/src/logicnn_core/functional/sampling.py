"""温度付き選択、Logisticノイズ二値化とstraight-through勾配を提供する。"""

# torchlogix/functional.pyのsamplingとGradFactorを基に再構成した。MIT License。
# Copyright (c) 2021-2023 Dr. Felix Petersen
# Copyright (c) 2024-present Dr. Lino Gerlach
# 許諾条件の全文と変更の由来は同梱LICENSEおよびTHIRD_PARTY_NOTICESを参照。

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

__all__ = ["gumbel_sigmoid", "scale_gradient", "temperature_sigmoid", "temperature_softmax"]


def _prepare_logits(logits: Tensor, temperature: float) -> tuple[Tensor, float]:
    """低precisionや極端な温度で必要な演算dtypeへ移す。"""
    temperature = float(temperature)
    bounds      = torch.finfo(torch.float32)
    dtype       = torch.float64 if logits.dtype == torch.float64 or not bounds.tiny <= temperature <= bounds.max else torch.float32
    return logits.to(dtype=dtype), temperature


def temperature_softmax(logits: Tensor, *, temperature: float = 1.0, dim: int = -1, hard: bool = False) -> Tensor:
    """温度付きsoftmaxを返し、hard時は先頭最大indexを選んでsoft勾配を保つ。"""
    values, temperature = _prepare_logits(logits, temperature)
    if values.numel() > 0:
        values = values - values.amax(dim=dim, keepdim=True)
    soft = torch.softmax(values / temperature, dim=dim).to(dtype=logits.dtype)
    if not hard:
        return soft
    if soft.numel() == 0:
        return soft
    index    = logits.argmax(dim=dim, keepdim=True)
    discrete = torch.zeros_like(soft).scatter_(dim, index, 1)
    return (discrete - soft).detach() + soft


def temperature_sigmoid(logits: Tensor, *, temperature: float = 1.0, hard: bool = False) -> Tensor:
    """温度付きsigmoidを返し、hard時は0.5を厳密に超えた要素を1にする。"""
    values, temperature = _prepare_logits(logits, temperature)
    soft                = torch.sigmoid(values / temperature).to(dtype=logits.dtype)
    if not hard:
        return soft
    discrete = (soft > 0.5).to(dtype=logits.dtype)
    return (discrete - soft).detach() + soft


def gumbel_sigmoid(
    logits: Tensor,
    *,
    temperature: float = 1.0,
    hard: bool = False,
    threshold: float = 0.5,
    generator: torch.Generator | None = None,
) -> Tensor:
    """一様乱数のlog oddsを加えたsigmoidと、任意閾値のstraight-through二値化を返す。"""
    values, temperature = _prepare_logits(logits, temperature)
    threshold           = float(threshold)
    uniform = torch.rand(logits.shape, dtype=logits.dtype, device=logits.device, generator=generator).to(dtype=values.dtype)
    bounds  = torch.finfo(values.dtype)
    uniform = uniform.clamp(min=bounds.tiny, max=1.0 - bounds.eps)
    noise   = uniform.log() - torch.log1p(-uniform)
    soft    = torch.sigmoid((values + noise) / temperature).to(dtype=logits.dtype)
    if not hard:
        return soft
    discrete = (soft > threshold).to(dtype=logits.dtype)
    return (discrete - soft).detach() + soft


class _ScaleGradient(torch.autograd.Function):
    """forwardの算術演算を避け、入力勾配だけに係数を乗じる。"""

    @staticmethod
    def forward(ctx: Any, inputs: Tensor, factor: float) -> Tensor:
        """入力をそのまま返し、逆伝播に使う係数だけを記録する。"""
        ctx.factor = factor
        return inputs

    @staticmethod
    def backward(ctx: Any, gradient: Tensor) -> tuple[Tensor, None]:
        """入力勾配を指定倍率にし、非Tensor係数の勾配は返さない。"""
        return gradient * ctx.factor, None


def scale_gradient(inputs: Tensor, factor: float) -> Tensor:
    """入力値・shape・dtypeを変更せず、逆伝播の勾配だけを係数倍する。"""
    return _ScaleGradient.apply(inputs, float(factor))
