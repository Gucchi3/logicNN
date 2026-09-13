"""Walsh変換・Light基底と実体化を省く重み付き縮約を提供する。"""

# torchlogix/functional.pyのWalsh・Light基底を基に再構成した。MIT License。
# Copyright (c) 2021-2023 Dr. Felix Petersen
# Copyright (c) 2024-present Dr. Lino Gerlach
# 許諾条件の全文と変更の由来は同梱LICENSEおよびTHIRD_PARTY_NOTICESを参照。

from __future__ import annotations

from collections.abc import Iterator
from string import ascii_letters

import torch
from torch import Tensor

__all__ = ["fast_walsh_hadamard", "light_basis", "walsh_basis", "weighted_light_basis_sum", "weighted_walsh_basis_sum"]


def _prepare_inputs(inputs: Tensor, rank: int, input_dim: int) -> Tensor:
    """入力軸の長さを確認して末尾へ移し、他の次元の順序を維持する。"""
    if inputs.shape[input_dim] != rank:
        raise ValueError(f"inputs のinput_dim軸はrank={rank}と一致する必要があります")
    return inputs.movedim(input_dim, -1)


def fast_walsh_hadamard(inputs: Tensor, rank: int) -> Tensor:
    """末尾2**rank要素を非正規化変換し、先行shape・dtype・deviceを維持する。"""
    entries = 1 << rank
    if inputs.ndim == 0 or inputs.shape[-1] != entries:
        raise ValueError(f"inputs の最終軸は2**rank={entries}要素で指定してください")
    values = inputs
    for stage in range(rank):
        half   = 1 << stage
        blocks = values.reshape(*inputs.shape[:-1], entries // (2 * half), 2 * half)
        first  = blocks[..., :half]
        second = blocks[..., half:]
        values = torch.cat((first + second, first - second), dim=-1).reshape(inputs.shape)
    return values


def _materialized_basis(values: Tensor, rank: int, light: bool) -> Tensor:
    """入力を先頭からKronecker積で展開し、元の真理値表順を保つ。"""
    basis = torch.ones_like(values[..., :1])
    for index in range(rank):
        variable = values[..., index:index + 1]
        absent   = basis * (1 - variable) if light else basis
        present  = basis * variable
        basis    = torch.stack((absent, present), dim=-1).flatten(-2)
    return basis


def walsh_basis(inputs: Tensor, rank: int, *, input_dim: int = 1) -> Tensor:
    """符号化済み入力のWalsh基底を返し、rank 2では末尾を[1,B,A,AB]にする。"""
    values = _prepare_inputs(inputs, rank, input_dim)
    return _materialized_basis(values, rank, light=False)


def light_basis(inputs: Tensor, rank: int, *, input_dim: int = 1) -> Tensor:
    """各入力のxまたは1-xを真理値表順に掛けたLight基底を末尾軸へ返す。"""
    values = _prepare_inputs(inputs, rank, input_dim)
    return _materialized_basis(values, rank, light=True)


def _basis_terms(values: Tensor, rank: int, light: bool) -> Iterator[Tensor]:
    """全基底Tensorを確保せず、真理値表順の基底項を一つずつ生成する。"""
    for mask in range(1 << rank):
        term = torch.ones_like(values[..., 0])
        for index in range(rank):
            if mask & (1 << (rank - index - 1)):
                term = term * values[..., index]
            elif light:
                term = term * (1 - values[..., index])
        yield term


def _weighted_basis_sum(inputs: Tensor, weights: Tensor, contraction: str, rank: int, input_dim: int, materialize_basis: bool, light: bool) -> Tensor:
    """係数軸を共有する重み付き基底計算を、指定した縮約順で実行する。"""
    values = _prepare_inputs(inputs, rank, input_dim)
    if weights.ndim == 0 or weights.shape[-1] != 1 << rank:
        raise ValueError(f"weights の最終軸は2**rank={1 << rank}要素で指定してください")
    dtype   = torch.promote_types(values.dtype, weights.dtype)
    values  = values.to(dtype=dtype)
    weights = weights.to(dtype=dtype)
    if materialize_basis:
        operands, output       = contraction.split("->")
        weight_axes, basis_axes = operands.split(",")
        coefficient_axis       = next(label for label in ascii_letters if label not in contraction)
        full_contraction       = f"{weight_axes}{coefficient_axis},{basis_axes}{coefficient_axis}->{output}"
        return torch.einsum(full_contraction, weights, _materialized_basis(values, rank, light))
    terms  = _basis_terms(values, rank, light)
    result = torch.einsum(contraction, weights[..., 0], next(terms))
    for index, term in enumerate(terms, start=1):
        result = result + torch.einsum(contraction, weights[..., index], term)
    return result


def weighted_walsh_basis_sum(
    inputs: Tensor,
    weights: Tensor,
    contraction: str,
    rank: int,
    *,
    input_dim: int = 1,
    materialize_basis: bool = False,
) -> Tensor:
    """Walsh係数を縮約し、既定では全基底を保持せず入力・係数の勾配を保つ。"""
    return _weighted_basis_sum(inputs, weights, contraction, rank, input_dim, materialize_basis, light=False)


def weighted_light_basis_sum(
    inputs: Tensor,
    weights: Tensor,
    contraction: str,
    rank: int,
    *,
    input_dim: int = 1,
    materialize_basis: bool = False,
) -> Tensor:
    """Light係数を縮約し、既定では全基底を保持せず入力・係数の勾配を保つ。"""
    return _weighted_basis_sum(inputs, weights, contraction, rank, input_dim, materialize_basis, light=True)
