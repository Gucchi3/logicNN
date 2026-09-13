"""MNIST・CIFAR-10の旧Conv構成を、新APIの汎用2Dモデル例として整理する。"""

# torchlogix/models/conv.pyのモデル構成を基に再構成した。MIT License。
# Copyright (c) 2021-2023 Dr. Felix Petersen
# Copyright (c) 2024-present Dr. Lino Gerlach
# 許諾条件と出典はcoreのLICENSE・THIRD_PARTY_NOTICES.mdおよび同階層README.mdを参照。

from __future__ import annotations

import math
from dataclasses import dataclass, replace

import torch
from torch import Tensor, nn

from logicnn_core.layer_settings import ConnectionConfig, LUTConfig
from logicnn_core.layers import GroupSum, LogicConv2d, LogicDense, OrPooling2d


@dataclass(frozen=True, slots=True)
class ConvolutionStage:
    """一つの論理Convと直後のpoolingを決める空間・接続設定を保持する。"""

    out_channels: int
    kernel_size: int = 3
    tree_depth: int = 3
    stride: int = 1
    padding: int = 0
    pool_size: int = 2
    pool_stride: int = 2
    pool_padding: int = 0
    channel_group_size: int | None = None


@dataclass(frozen=True, slots=True)
class ConvolutionPreset:
    """二値化後の[C,H,W]、Conv段列、Dense幅列とクラス集約を記録する。"""

    input_shape: tuple[int, int, int]
    stages: tuple[ConvolutionStage, ...]
    hidden_features: tuple[int, ...]
    num_classes: int
    tau: float
    source_names: tuple[str, ...] = ()


class ConvolutionReferenceModel(nn.Module):
    """明示した前処理と任意のConv/pooling段列を使う、データ取得を持たないモデル例。"""

    def __init__(
        self, preset: ConvolutionPreset, *, lut: LUTConfig, preprocessing: nn.Module | None = None,
        conv_connections: ConnectionConfig = ConnectionConfig(), dense_connections: ConnectionConfig = ConnectionConfig(),
        gradient_scale: float = 1.0, device: torch.device | str | None = None, dtype: torch.dtype | None = None,
    ) -> None:
        """各層の実geometryから次の入力shapeを求め、Conv列とDense列を組み立てる。"""
        super().__init__()
        output = GroupSum(preset.num_classes, tau=preset.tau)
        if len(preset.input_shape) != 3 or any(size <= 0 for size in preset.input_shape) or not preset.stages:
            raise ValueError("input_shapeは正の[C,H,W]、stagesは1段以上で指定してください")
        self.preset      = preset
        self.input_shape = preset.input_shape
        self.input_size  = math.prod(preset.input_shape)
        self.num_classes = preset.num_classes
        self.preprocessing = nn.Identity() if preprocessing is None else preprocessing
        self.preprocessing.to(device=device, dtype=dtype)
        channels, *spatial = preset.input_shape
        layers: list[nn.Module] = []
        for stage in preset.stages:
            connections = replace(conv_connections, channel_group_size=stage.channel_group_size)
            conv = LogicConv2d(tuple(spatial), channels, stage.out_channels, stage.tree_depth, stage.kernel_size, stride=stage.stride,
                               padding=stage.padding, lut=lut, connections=connections, gradient_scale=gradient_scale, device=device, dtype=dtype)
            pool = OrPooling2d(stage.pool_size, stride=stage.pool_stride, padding=stage.pool_padding)
            spatial = tuple((size + 2 * pad - width) // step + 1 for size, width, step, pad in
                            zip(conv.output_size, pool.kernel_size, pool.stride, pool.padding))
            if any(size <= 0 for size in spatial):
                raise ValueError("Conv/pooling後の空間サイズは正である必要があります")
            layers.extend((conv, pool))
            channels = stage.out_channels
        self.encoded_output_shape = (channels, *spatial)
        in_features = math.prod(self.encoded_output_shape)
        final_width = preset.hidden_features[-1] if preset.hidden_features else in_features
        if final_width % preset.num_classes:
            raise ValueError("最終特徴数はnum_classesで割り切れる必要があります")
        layers.append(nn.Flatten())
        for out_features in preset.hidden_features:
            layers.append(LogicDense(in_features, out_features, lut=lut, connections=dense_connections, gradient_scale=gradient_scale,
                                     device=device, dtype=dtype))
            in_features = out_features
        self.network = nn.Sequential(*layers, output)

    def forward(self, inputs: Tensor) -> Tensor:
        """前処理後の[C,H,W]を確認し、論理ConvとDenseを通したクラス集約値を返す。"""
        values = self.preprocessing(inputs)
        if values.shape[1:] != self.input_shape:
            raise ValueError(f"前処理後の入力は[batch, {self.input_shape}]で指定してください")
        return self.network(values)


def _build_presets() -> dict[str, ConvolutionPreset]:
    """旧Convモデルのchannel倍率・padding・tauを、重みを確保しないpresetへ写す。"""
    presets: dict[str, ConvolutionPreset] = {}
    for size, width, tau in (("base", 16, 1.0), ("tiny", 4, 1.0), ("small", 16, 6.5), ("medium", 64, 28.0), ("large", 1024, 35.0)):
        stages = (ConvolutionStage(width, kernel_size=5), ConvolutionStage(3 * width, pool_padding=1),
                  ConvolutionStage(9 * width, pool_padding=1))
        source = "ClgnMnist" + ("" if size == "base" else size.title())
        presets[f"mnist_{size}"] = ConvolutionPreset((1, 28, 28), stages, (1280 * width, 640 * width, 320 * width), 10, tau, (source,))
    for size, width, bits, tau, first_group in (("small", 32, 2, 20.0, 2), ("medium", 256, 2, 40.0, 2), ("large", 512, 5, 280.0, 2),
                                               ("small2", 32, 2, 20.0, 1), ("medium2", 256, 2, 40.0, 1)):
        stages = tuple(ConvolutionStage(factor * width, padding=1, channel_group_size=first_group if index == 0 else 2)
                       for index, factor in enumerate((1, 4, 16, 32)))
        presets[f"cifar10_{size}"] = ConvolutionPreset((3 * bits, 32, 32), stages, (1280 * width, 640 * width, 320 * width), 10, tau,
                                                      (f"ClgnCifar10{size.title()}",))
    return presets


CONVOLUTION_PRESETS = _build_presets()
