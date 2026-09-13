"""任意の先行shapeを保つ、固定・学習可能接続の全結合論理層を提供する。"""

# torchlogix/layers/dense.pyを基に再構成した。MIT License。
# Copyright (c) 2021-2023 Dr. Felix Petersen
# Copyright (c) 2024-present Dr. Lino Gerlach
# 許諾条件の全文と変更の由来は同梱LICENSEおよびTHIRD_PARTY_NOTICESを参照。

from __future__ import annotations

import torch
from torch import Tensor, nn

from ..connections import build_dense_connections
from ..functional.logic import apply_export_luts, apply_export_truth_tables
from ..functional.regularization import regularization_loss, rescale_weights_
from ..functional.sampling import scale_gradient
from ..layer_settings import ConnectionConfig, LUTConfig, _validate_positive_integer
from .base import LogicLayer

__all__ = ["LogicDense"]


class LogicDense(LogicLayer):
    """各出力ゲートのLUTと入力接続を組み合わせ、末尾の特徴軸を変換する。"""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        lut: LUTConfig = LUTConfig(),
        connections: ConnectionConfig = ConnectionConfig(),
        gradient_scale: float = 1.0,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        """LUTのParameterと接続moduleを、指定した特徴数・device・dtypeで生成する。"""
        _validate_positive_integer("in_features", in_features)
        _validate_positive_integer("out_features", out_features)
        super().__init__(lut=lut, connections=connections, gradient_scale=gradient_scale)
        self.in_features  = in_features
        self.out_features = out_features
        self.weight       = nn.Parameter(self.parametrization.initialize(out_features, device=device, dtype=dtype))
        self.connections  = build_dense_connections(in_features, out_features, num_inputs=self.num_inputs, config=connections,
                                                    device=self.weight.device, dtype=self.weight.dtype)
        self._export_table_name   = "_export_lut_ids" if self.num_inputs == 2 else "_export_truth_tables"
        self._export_buffer_names = (self._export_table_name, "_export_indices")

    def forward(self, inputs: Tensor) -> Tensor:
        """元入力の精度を保って接続を選び、通常LUT計算または固定bool回路を適用する。"""
        if self.export_mode:
            if inputs.dtype != torch.bool:
                raise TypeError("export入力はbool Tensorで指定してください。必要な二値化を層の前で明示してください")
            selected = inputs[..., self._export_indices]
            if self.num_inputs == 2:
                return apply_export_luts(selected[..., 0, :], selected[..., 1, :], self._export_lut_ids)
            return apply_export_truth_tables(selected.unbind(-2), self._export_truth_tables)
        values   = scale_gradient(inputs, self.gradient_scale) if self.gradient_scale != 1.0 else inputs
        selected = self.connections(values)
        result   = self.parametrization(selected, self.weight, training=self.training, contraction="n,bn->bn")
        return result.reshape(*inputs.shape[:-1], self.out_features)

    def truth_tables(self) -> Tensor:
        """最新の重みから、出力ゲート順の独立したbool真理値表を生成する。"""
        return self.parametrization.truth_tables(self.weight)

    def truth_tables_with_ids(self) -> tuple[Tensor, Tensor | None]:
        """最新のbool表とMSB-first整数IDを返し、高rankではIDをNoneにする。"""
        return self.parametrization.truth_tables_with_ids(self.weight)

    def regularization_loss(self, kind: str | None = None) -> Tensor:
        """接続logitとは分けて、LUT重みのゲート平均正則化を返す。"""
        return regularization_loss(self.weight, kind)

    def rescale_weights_(self, method: str | None = None) -> None:
        """LUT重みのParameterを維持したまま再スケールし、接続logitは変更しない。"""
        rescale_weights_(self.weight, method)

    def _make_export_buffers(self) -> dict[str, Tensor]:
        """2入力は整数ID、それ以外はbool表として、決定的な接続と一緒に独立保存する。"""
        values  = self.truth_tables_with_ids()[1] if self.num_inputs == 2 else self.truth_tables()
        indices = self.connections.get_indices()
        return {self._export_table_name: values.detach().clone(), "_export_indices": indices.detach().clone()}

    def extra_repr(self) -> str:
        """特徴数・LUT方式・接続方式と回路出力状態を簡潔に表示する。"""
        return (f"in_features={self.in_features}, out_features={self.out_features}, num_inputs={self.num_inputs}, "
                f"lut={self.lut_config.kind}, connections={self.connection_config.kind}, export_mode={self.export_mode}")
