"""DenseとConvの型付き引数から、設定に合う接続方式を生成する。"""

from __future__ import annotations

import torch

from ..layer_settings import ConnectionConfig
from .convolution import FixedConvConnections
from .dense import FixedDenseConnections, LearnableDenseConnections

__all__ = ["build_dense_connections", "build_conv_connections"]


def build_dense_connections(
    in_features: int,
    out_features: int,
    *,
    num_inputs: int = 2,
    config: ConnectionConfig = ConnectionConfig(),
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
) -> FixedDenseConnections | LearnableDenseConnections:
    """Denseの固定・学習可能接続を生成し、dtypeは接続logitの初期化だけに適用する。"""
    if not isinstance(config, ConnectionConfig):
        raise TypeError("config はConnectionConfigで指定してください")
    if dtype is not None and dtype not in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
        raise TypeError("dtype はfloat16／bfloat16／float32／float64で指定してください")
    if config.kind == "fixed":
        return FixedDenseConnections(in_features, out_features, num_inputs=num_inputs, config=config, device=device)
    return LearnableDenseConnections(in_features, out_features, num_inputs=num_inputs, config=config, device=device, dtype=dtype)


def build_conv_connections(
    input_size: int | tuple[int, ...],
    in_channels: int,
    out_channels: int,
    tree_depth: int,
    kernel_size: int | tuple[int, ...],
    *,
    num_inputs: int = 2,
    stride: int | tuple[int, ...] = 1,
    padding: int | tuple[int, ...] = 0,
    conv_dimension: int = 2,
    config: ConnectionConfig = ConnectionConfig(),
    device: torch.device | str | None = None,
) -> FixedConvConnections:
    """2D／3Dの固定Conv接続へ空間形状と方式設定を明示的に渡す。"""
    if not isinstance(config, ConnectionConfig):
        raise TypeError("config はConnectionConfigで指定してください")
    return FixedConvConnections(input_size, in_channels, out_channels, tree_depth, kernel_size, num_inputs=num_inputs, stride=stride,
                                padding=padding, conv_dimension=conv_dimension, config=config, device=device)
