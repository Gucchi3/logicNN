"""Walsh係数で論理関数を学習し、基底和の符号から真理値表を取り出す。"""

# torchlogix/parametrization.pyのWarpLUTParametrizationを基に再構成した。MIT License。
# Copyright (c) 2021-2023 Dr. Felix Petersen
# Copyright (c) 2024-present Dr. Lino Gerlach
# 許諾条件の全文と変更の由来は同梱LICENSEおよびTHIRD_PARTY_NOTICESを参照。

from __future__ import annotations

import math

import torch
from torch import Tensor

from ..functional.walsh import fast_walsh_hadamard, weighted_walsh_basis_sum
from .base import LUTParametrization

__all__ = ["WarpLUTParametrization"]


def _lookup_binary_outputs(inputs: Tensor, tables: Tensor, binary: Tensor, contraction: str) -> tuple[Tensor, Tensor]:
    """ゲート軸を保つ縮約に沿って表の行・入力bit列・二値maskを整列し、bool出力を引く。"""
    operands, output = contraction.replace(" ", "").split("->")
    labels           = set(operands.replace(",", "").replace(".", ""))
    if not labels <= set(output) or ("..." in operands and "..." not in output):
        raise ValueError("Warpの二値evalでは各ゲートの軸を出力へ残してください。複数ゲートを還元するcontractionは未対応です")

    entries = tables.shape[-1]
    rows    = torch.arange(tables.numel() // entries, device=tables.device).reshape(tables.shape[:-1])
    indices = torch.zeros_like(binary, dtype=torch.int64)
    for bit in (inputs == 1).unbind(dim=1):
        indices = 2 * indices + bit.to(dtype=torch.int64)
    row_ones   = torch.ones_like(rows)
    input_ones = torch.ones_like(indices)
    aligned_rows    = torch.einsum(contraction, rows, input_ones)
    aligned_indices = torch.einsum(contraction, row_ones, indices)
    aligned_binary  = torch.einsum(contraction, row_ones, binary.to(dtype=torch.int64)).bool()
    return tables.reshape(-1, entries)[aligned_rows, aligned_indices], aligned_binary


class WarpLUTParametrization(LUTParametrization):
    """rank 1・2・4・6のWalsh係数を保持せず、生成・計算・離散化を担当する。"""

    _kind = "warp"

    def initialize(self, count: int, *, device: torch.device | str | None = None, dtype: torch.dtype | None = None) -> Tensor:
        """通常residual・真理値表catalog・正規乱数から指定数のWalsh係数を生成する。"""
        device, dtype  = self._initialization_options(count, device, dtype)
        initialization = self.config.weight_init
        probability    = self.config.residual_probability
        if initialization == "random":
            return torch.randn(count, self.num_coefficients, device=device, dtype=dtype)
        if initialization == "residual":
            coefficient = self.temperature * (math.log(probability) - math.log1p(-probability))
            if not math.isfinite(coefficient) or abs(coefficient) > torch.finfo(dtype).max:
                raise ValueError("residualの初期係数が指定dtypeの有限値で表現できません")
            weights = torch.zeros(count, self.num_coefficients, device=device, dtype=dtype)
            weights[:, self.num_coefficients // 2] = coefficient
            return weights

        weights = torch.zeros(count, self.num_coefficients, device=device, dtype=dtype)
        weights[:, self.num_coefficients // 2] = 1
        resampled_count = round(count * (1 - probability))
        truth_tables    = 1 - 2 * torch.randint(0, 2, (resampled_count, self.num_coefficients), device=device)
        coefficients    = fast_walsh_hadamard(truth_tables, self.num_inputs).to(dtype=dtype) / self.num_coefficients
        indices         = torch.randperm(count, device=device)
        weights[indices[:resampled_count]] = coefficients
        return weights

    def forward(self, inputs: Tensor, weights: Tensor, *, training: bool, contraction: str) -> Tensor:
        """学習・連続evalはWalsh和を使い、二値evalは共通の高精度真理値表を参照する。"""
        original_inputs = inputs
        inputs          = inputs.to(dtype=weights.dtype)
        if not training:
            binary     = ((original_inputs == 0) | (original_inputs == 1)).all(dim=1)
            use_tables = binary.numel() == 0 or bool(binary.any())
            if use_tables:
                tables                = self.truth_tables(weights)
                discrete, output_mask = _lookup_binary_outputs(original_inputs, tables, binary, contraction)
                discrete              = discrete.to(dtype=weights.dtype)
                if bool(binary.all()):
                    return discrete

        signed_inputs   = 1 - 2 * inputs
        result = weighted_walsh_basis_sum(signed_inputs, weights, contraction, self.num_inputs, materialize_basis=self.config.materialize_basis)
        output = self._sample_binary(-result, training)
        return torch.where(output_mask, discrete, output) if not training and use_tables else output

    def truth_tables(self, weights: Tensor) -> Tensor:
        """CPU float64の固定順Walsh変換で負の要素を選び、独立したbool表を元deviceへ返す。"""
        snapshot = weights.detach().to(device="cpu", dtype=torch.float64, copy=True)
        if not bool(torch.isfinite(snapshot).all()):
            raise ValueError("Warpの真理値表は有限の重みから生成してください")
        transformed = fast_walsh_hadamard(snapshot, self.num_inputs)
        # FWHT途中のInf／NaNは和差の後も残るため、最終結果の検証で途中overflowも拒否できる。
        if not bool(torch.isfinite(transformed).all()):
            raise ValueError("Warpの真理値表生成で非有限値が発生しました。float64の変換範囲を超えています")
        return (transformed < 0).to(device=weights.device)
