"""二入力の連続基底と、多入力でも追跡可能な離散LUT演算を提供する。"""

# Derived from torchlogix / difflogic; distributed under the MIT License.
# Copyright (c) 2021-2023 Dr. Felix Petersen
# Copyright (c) 2024-present Dr. Lino Gerlach
# See LICENSE and THIRD_PARTY_NOTICES.md in the distribution.

import torch
from torch import Tensor

__all__ = ["all_binary_logic_outputs", "apply_binary_lut", "apply_export_luts", "apply_export_truth_tables"]


def _broadcast_inputs(a: Tensor, b: Tensor) -> tuple[Tensor, Tensor]:
    """入力を共通dtypeへ揃え、PyTorchのbroadcastへ渡す。"""
    dtype = torch.promote_types(a.dtype, b.dtype)
    return torch.broadcast_tensors(a.to(dtype), b.to(dtype))


def _first_eight_outputs(a: Tensor, b: Tensor) -> tuple[Tensor, ...]:
    """共通計算を再利用してID 0〜7の論理値または連続基底を返す。"""
    if a.dtype == torch.bool:
        return torch.zeros_like(a), a & b, a & ~b, a, ~a & b, b, a ^ b, a | b
    ab     = a * b
    summed = a + b
    zero   = a * 0 + b * 0
    return zero, ab, a - ab, a, b - ab, b, summed - 2 * ab, summed - ab


def apply_binary_lut(a: Tensor, b: Tensor, lut_id: int) -> Tensor:
    """固定IDの二入力論理関数をbroadcastし、浮動小数点には連続式を適用する。"""
    a, b   = _broadcast_inputs(a, b)
    index  = lut_id if lut_id < 8 else 15 - lut_id
    result = _first_eight_outputs(a, b)[index]
    if lut_id < 8:
        return result
    return ~result if result.dtype == torch.bool else 1 - result


def all_binary_logic_outputs(a: Tensor, b: Tensor) -> Tensor:
    """broadcast後の末尾にID順で16論理関数の値を並べる。"""
    a, b  = _broadcast_inputs(a, b)
    first = _first_eight_outputs(a, b)
    last  = tuple(~value if a.dtype == torch.bool else 1 - value for value in reversed(first))
    return torch.stack(first + last, dim=-1)


def apply_export_luts(a: Tensor, b: Tensor, lut_ids: Tensor) -> Tensor:
    """bool入力と整数IDをbroadcastし、値のPython抽出なしでLUTを適用する。"""
    a, b, ids = torch.broadcast_tensors(a, b, lut_ids.to(torch.int64))
    bit00 = (ids & 8) != 0
    bit01 = (ids & 4) != 0
    bit10 = (ids & 2) != 0
    bit11 = (ids & 1) != 0
    return (~a & ~b & bit00) | (~a & b & bit01) | (a & ~b & bit10) | (a & b & bit11)


def apply_export_truth_tables(inputs: tuple[Tensor, ...], tables: Tensor) -> Tensor:
    """MSB-firstのbool真理値表をAND・OR・NOTへ分解し、入力依存の表引きなしで評価する。"""
    values = tables
    for signal in reversed(inputs):
        selected = signal.unsqueeze(-1)
        values   = (~selected & values[..., 0::2]) | (selected & values[..., 1::2])
    return values.squeeze(-1)
