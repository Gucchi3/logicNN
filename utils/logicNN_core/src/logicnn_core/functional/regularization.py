"""ゲート平均の正則化と、係数のin-place再スケールを提供する。"""

# torchlogix/functional.pyの正則化・再スケールを基に再構成した。MIT License。
# Copyright (c) 2021-2023 Dr. Felix Petersen
# Copyright (c) 2024-present Dr. Lino Gerlach
# 許諾条件の全文と変更の由来は同梱LICENSEおよびTHIRD_PARTY_NOTICESを参照。

from __future__ import annotations

import torch
from torch import Tensor

__all__ = ["regularization_loss", "rescale_weights_"]


def _validate_choice(name: str, value: str | None, choices: tuple[str, ...]) -> None:
    """Noneまたは対応する方式名だけを受理し、保留warpは理由を付けて拒否する。"""
    if value is not None and not isinstance(value, str):
        raise TypeError(f"{name} は文字列またはNoneで指定してください")
    if name == "kind" and value == "warp":
        raise ValueError("warp 正則化は実装保留です。Warpパラメータ化本体とは別の機能です")
    if value is not None and value not in choices:
        raise ValueError(f"未対応の{name}: {value!r}。対応方式は {choices} またはNoneです")


def regularization_loss(weights: Tensor, kind: str | None = None) -> Tensor:
    """旧L2／abs_sumのゲート別ペナルティを全ゲートで平均したscalarを返す。"""
    _validate_choice("kind", kind, ("L2", "abs_sum"))
    if kind is None:
        return weights.reshape(-1)[0] * 0
    values    = weights.to(dtype=torch.float64 if weights.dtype == torch.float64 else torch.float32)
    statistic = values.square().sum(dim=-1) if kind == "L2" else values.sum(dim=-1).abs()
    return (1 - statistic).square().mean().to(dtype=weights.dtype)


def rescale_weights_(weights: Tensor, method: str | None = None) -> None:
    """ゲートごとの係数をin-place更新し、Parameterと既存勾配を保持する。"""
    _validate_choice("method", method, ("clip", "L2", "abs_sum"))
    if method is None:
        return
    with torch.no_grad():
        values = weights.to(dtype=torch.float64 if weights.dtype == torch.float64 else torch.float32)
        if method == "clip":
            candidate = values.clamp(-1, 1)
        else:
            denominator = values.norm(p=2, dim=-1, keepdim=True) if method == "L2" else values.sum(dim=-1, keepdim=True).abs()
            candidate = values / denominator
        candidate = candidate.to(dtype=weights.dtype)
        weights.copy_(candidate)
