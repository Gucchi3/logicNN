"""固定受容野と多段論理木を組み合わせる、汎用2D・3D論理畳み込みを提供する。"""

# torchlogix/layers/conv.pyを基に再構成した。MIT License。
# Copyright (c) 2021-2023 Dr. Felix Petersen
# Copyright (c) 2024-present Dr. Lino Gerlach
# 許諾条件の全文と変更の由来は同梱LICENSEおよびTHIRD_PARTY_NOTICESを参照。

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from ..connections import build_conv_connections
from ..functional.logic import apply_export_luts, apply_export_truth_tables
from ..functional.regularization import regularization_loss, rescale_weights_
from ..functional.sampling import scale_gradient
from ..layer_settings import ConnectionConfig, LUTConfig
from .base import LogicLayer

__all__ = ["LogicConv2d", "LogicConv3d"]


class _LogicConvNd(LogicLayer):
    """各levelのnode・channel別LUTを所有し、接続のrank軸を共通式で集約する。"""

    def __init__(
        self,
        input_size: int | tuple[int, ...],
        in_channels: int,
        out_channels: int,
        tree_depth: int,
        kernel_size: int | tuple[int, ...],
        *,
        stride: int | tuple[int, ...] = 1,
        padding: int | tuple[int, ...] = 0,
        conv_dimension: int,
        lut: LUTConfig = LUTConfig(),
        connections: ConnectionConfig = ConnectionConfig(),
        gradient_scale: float = 1.0,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        """接続geometryを検証し、旧来のnode単位の抽選で各levelの重みを生成する。"""
        super().__init__(lut=lut, connections=connections, gradient_scale=gradient_scale)
        device, dtype    = self.parametrization._initialization_options(1, device, dtype)
        self.connections = build_conv_connections(input_size, in_channels, out_channels, tree_depth, kernel_size, num_inputs=self.num_inputs,
                                                   stride=stride, padding=padding, conv_dimension=conv_dimension, config=connections, device=device)
        self.input_size     = self.connections.input_size
        self.in_channels    = self.connections.in_channels
        self.out_channels   = self.connections.out_channels
        self.tree_depth     = self.connections.tree_depth
        self.kernel_size    = self.connections.kernel_size
        self.stride         = self.connections.stride
        self.padding        = self.connections.padding
        self.conv_dimension = self.connections.conv_dimension
        self.output_size    = self.connections.output_size
        self.node_counts    = self.connections.node_counts
        self.tree_weights   = nn.ParameterList()
        total_gates         = sum(self.node_counts) * self.out_channels
        self.parametrization._initialization_options(total_gates, device, dtype)
        for nodes in self.node_counts:
            weights = torch.stack([self.parametrization.initialize(out_channels, device=device, dtype=dtype) for _ in range(nodes)])
            self.tree_weights.append(nn.Parameter(weights))
        self._export_table_kind   = "lut_ids" if self.num_inputs == 2 else "truth_tables"
        self._export_buffer_names = tuple(f"_export_{kind}_{level}" for level in range(tree_depth) for kind in (self._export_table_kind, "indices"))

    def forward(self, inputs: Tensor) -> Tensor:
        """入力勾配倍率を一度だけ適用し、全levelを集約して元の空間次元へ戻す。"""
        if self.export_mode:
            if inputs.dtype != torch.bool:
                raise TypeError("export入力はbool Tensorで指定してください。必要な二値化を層の前で明示してください")
            return self._forward_export(inputs)
        values = scale_gradient(inputs, self.gradient_scale) if self.gradient_scale != 1.0 else inputs
        for level, weights in enumerate(self.tree_weights):
            selected = self.connections(values, level)
            values   = self.parametrization(selected, weights, training=self.training, contraction="fc,bcsf->bcsf")
        return values.reshape(inputs.shape[0], self.out_channels, *self.output_size)

    def truth_tables(self) -> list[Tensor]:
        """最新の重みから、各levelのnode・channel順を保つbool真理値表を返す。"""
        return [self.parametrization.truth_tables(weights) for weights in self.tree_weights]

    def truth_tables_with_ids(self) -> tuple[list[Tensor], list[Tensor | None]]:
        """levelごとの真理値表と整数IDを返し、rank6のIDはNoneとして扱う。"""
        pairs = [self.parametrization.truth_tables_with_ids(weights) for weights in self.tree_weights]
        return [table for table, _ in pairs], [ids for _, ids in pairs]

    def regularization_loss(self, kind: str | None = None) -> Tensor:
        """全levelのLUTをゲート軸へ連結し、level数ではなく全ゲート数で平均する。"""
        weights = torch.cat([values.reshape(-1, values.shape[-1]) for values in self.tree_weights])
        return regularization_loss(weights, kind)

    def rescale_weights_(self, method: str | None = None) -> None:
        """各levelの係数だけを再スケールし、Parameterと既存gradを保持する。"""
        for weights in self.tree_weights:
            rescale_weights_(weights, method)

    def _make_export_buffers(self) -> dict[str, Tensor]:
        """全levelの整数IDまたはbool表を、固定した接続と一緒に独立保存する。"""
        values  = self.truth_tables_with_ids()[1] if self.num_inputs == 2 else self.truth_tables()
        buffers = {}
        for level, (table, indices) in enumerate(zip(values, self.connections.indices)):
            buffers[f"_export_{self._export_table_kind}_{level}"] = table.detach().clone()
            buffers[f"_export_indices_{level}"] = indices.detach().clone()
        return buffers

    def _forward_export(self, inputs: Tensor) -> Tensor:
        """固定snapshotだけでpadding・入力選択・bool LUTを実行し、FXの値抽出を避ける。"""
        values = inputs
        if any(self.padding):
            padding = tuple(value for pad in reversed(self.padding) for value in (pad, pad))
            values  = F.pad(values, padding, mode="constant", value=False)
        for level in range(self.tree_depth):
            indices = getattr(self, f"_export_indices_{level}")
            if level == 0:
                coordinates = indices.unbind(-1)
                selected    = values[(slice(None), coordinates[-1], *coordinates[:-1])]
            else:
                selected = values[..., indices].movedim(-2, 1)
            if self.num_inputs == 2:
                ids    = getattr(self, f"_export_lut_ids_{level}").T.unsqueeze(1)
                values = apply_export_luts(selected[:, 0], selected[:, 1], ids)
            else:
                tables = getattr(self, f"_export_truth_tables_{level}").transpose(0, 1).unsqueeze(1)
                values = apply_export_truth_tables(selected.unbind(1), tables)
        return values.reshape(inputs.shape[0], self.out_channels, *self.output_size)

    def extra_repr(self) -> str:
        """入力geometry、論理木の深さ、LUT方式と回路出力状態を簡潔に表示する。"""
        return (f"input_size={self.input_size}, in_channels={self.in_channels}, out_channels={self.out_channels}, "
                f"tree_depth={self.tree_depth}, kernel_size={self.kernel_size}, stride={self.stride}, padding={self.padding}, "
                f"num_inputs={self.num_inputs}, lut={self.lut_config.kind}, export_mode={self.export_mode}")


class LogicConv2d(_LogicConvNd):
    """画像の各受容野を多段論理木で処理し、2Dのchannel出力を生成する。"""

    def __init__(
        self,
        input_size: int | tuple[int, int],
        in_channels: int,
        out_channels: int,
        tree_depth: int,
        kernel_size: int | tuple[int, int],
        *,
        stride: int | tuple[int, int] = 1,
        padding: int | tuple[int, int] = 0,
        lut: LUTConfig = LUTConfig(),
        connections: ConnectionConfig = ConnectionConfig(),
        gradient_scale: float = 1.0,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        """2Dの入力・kernel・stride・paddingを共通論理Convへ渡す。"""
        super().__init__(input_size, in_channels, out_channels, tree_depth, kernel_size, stride=stride, padding=padding, conv_dimension=2,
                         lut=lut, connections=connections, gradient_scale=gradient_scale, device=device, dtype=dtype)


class LogicConv3d(_LogicConvNd):
    """volumeの各受容野を多段論理木で処理し、3Dのchannel出力を生成する。"""

    def __init__(
        self,
        input_size: int | tuple[int, int, int],
        in_channels: int,
        out_channels: int,
        tree_depth: int,
        kernel_size: int | tuple[int, int, int],
        *,
        stride: int | tuple[int, int, int] = 1,
        padding: int | tuple[int, int, int] = 0,
        lut: LUTConfig = LUTConfig(),
        connections: ConnectionConfig = ConnectionConfig(),
        gradient_scale: float = 1.0,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        """3Dの入力・kernel・stride・paddingを共通論理Convへ渡す。"""
        super().__init__(input_size, in_channels, out_channels, tree_depth, kernel_size, stride=stride, padding=padding, conv_dimension=3,
                         lut=lut, connections=connections, gradient_scale=gradient_scale, device=device, dtype=dtype)
