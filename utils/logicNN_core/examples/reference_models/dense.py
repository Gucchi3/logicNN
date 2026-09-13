"""旧データセット別Denseモデルの形状知識を、不変presetと共通組立てへ整理する。"""

# torchlogix/models/dense.pyのモデル構成を基に再構成した。MIT License。
# Copyright (c) 2021-2023 Dr. Felix Petersen
# Copyright (c) 2024-present Dr. Lino Gerlach
# 許諾条件と出典はcoreのLICENSE・THIRD_PARTY_NOTICES.mdおよび同階層README.mdを参照。

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn

from logicnn_core.layer_settings import ConnectionConfig, LUTConfig
from logicnn_core.layers import GroupSum, LogicDense


@dataclass(frozen=True, slots=True)
class DensePreset:
    """二値化後の入力shape、各層幅、集約と接続方式の構造情報だけを保持する。"""

    input_shape: tuple[int, ...]
    hidden_features: tuple[int, ...]
    num_classes: int
    tau: float
    num_inputs: int = 2
    learnable_layers: int = 0
    source_names: tuple[str, ...] = ()


class DenseReferenceModel(nn.Module):
    """明示した前処理、Flatten、論理Dense列、GroupSumを組み合わせる汎用例。"""

    def __init__(
        self, preset: DensePreset, *, lut: LUTConfig, preprocessing: nn.Module | None = None,
        connections: ConnectionConfig = ConnectionConfig(), learnable_connections: ConnectionConfig = ConnectionConfig(kind="learnable"),
        gradient_scale: float = 1.0, device: torch.device | str | None = None, dtype: torch.dtype | None = None,
    ) -> None:
        """presetの形状を確認し、先頭の指定段だけ学習可能接続を使って新APIの層を作る。"""
        super().__init__()
        output = GroupSum(preset.num_classes, tau=preset.tau)
        if not preset.input_shape or any(size <= 0 for size in preset.input_shape) or not preset.hidden_features:
            raise ValueError("input_shapeは正の空でないshape、hidden_featuresは1段以上で指定してください")
        if not 0 <= preset.learnable_layers <= len(preset.hidden_features):
            raise ValueError("learnable_layersは0以上Dense段数以下で指定してください")
        if preset.hidden_features[-1] % preset.num_classes:
            raise ValueError("最終Dense幅はnum_classesで割り切れる必要があります")
        if lut.num_inputs != preset.num_inputs:
            raise ValueError("LUTConfig.num_inputsはpresetのnum_inputsと一致させてください")
        if preset.learnable_layers and learnable_connections.kind != "learnable":
            raise ValueError("先頭の学習可能接続にはkind='learnable'を指定してください")
        self.preset      = preset
        self.input_shape = preset.input_shape
        self.input_size  = math.prod(preset.input_shape)
        self.num_classes = preset.num_classes
        self.preprocessing = nn.Identity() if preprocessing is None else preprocessing
        self.preprocessing.to(device=device, dtype=dtype)
        layers: list[nn.Module] = [nn.Flatten()]
        in_features = self.input_size
        for index, out_features in enumerate(preset.hidden_features):
            selected = learnable_connections if index < preset.learnable_layers else connections
            layers.append(LogicDense(in_features, out_features, lut=lut, connections=selected, gradient_scale=gradient_scale, device=device, dtype=dtype))
            in_features = out_features
        self.network = nn.Sequential(*layers, output)

    def forward(self, inputs: Tensor) -> Tensor:
        """前処理後のshapeを保護し、二値化済み入力からクラス別集約値を返す。"""
        values = self.preprocessing(inputs)
        if values.shape[1:] != self.input_shape:
            raise ValueError(f"前処理後の入力は[batch, {self.input_shape}]で指定してください")
        return self.network(values)


def _build_presets() -> dict[str, DensePreset]:
    """旧presetの数値をまとめ、重みやデータを生成せず構成一覧だけを作る。"""
    presets: dict[str, DensePreset] = {}

    def add(name: str, shape: tuple[int, ...], widths: tuple[int, ...], classes: int, tau: float, rank: int, learn: int, *sources: str) -> None:
        """解釈済みの一構成を旧名の出典とともに登録する。"""
        presets[name] = DensePreset(shape, widths, classes, tau, rank, learn, sources)

    for size, rank, width, tau in (("tiny", 2, 1000, 10.0), ("small", 2, 8000, 10.0), ("small", 4, 4000, 10.0),
                                   ("small", 6, 2330, 10.0), ("medium", 2, 64000, 1 / 0.03)):
        suffix = "" if rank == 2 else f"_rank{rank}"
        source = f"DlgnMnist{size.title()}" + ("" if rank == 2 else f"Rank{rank}")
        add(f"mnist_{size}{suffix}", (1, 28, 28), (width,) * 5, 10, tau, rank, 0, source)
        if size == "small":
            add(f"mnist_{size}{suffix}_learn1", (1, 28, 28), (width,) * 5, 10, tau, rank, 1, source + "Learn1")

    for rank, width in ((2, 12000), (4, 6000), (6, 3000)):
        suffix = "" if rank == 2 else f"_rank{rank}"
        source = "DlgnCifar10Small" + ("" if rank == 2 else f"Rank{rank}")
        add(f"cifar10_small{suffix}", (9, 32, 32), (width,) * 4, 10, 1 / 0.03, rank, 0, source)
    add("cifar10_medium", (9, 32, 32), (128000,) * 4, 10, 100.0, 2, 0, "DlgnCifar10Medium")
    for factor in (1, 2, 4):
        suffix = "" if factor == 1 else str(factor)
        add(f"cifar10_large{suffix}", (15, 32, 32), (256000 * factor,) * 5, 10, 100.0, 2, 0, f"DlgnCifar10Large{suffix}")
    for rank, width, tau in ((2, 12000, 1 / 0.03), (4, 6000, 100.0), (6, 4000, 100.0)):
        for factor in (1, 2, 3, 4, 5, 10, 20):
            depth  = "" if factor == 1 else str(factor)
            suffix = "" if rank == 2 else f"_rank{rank}"
            source = f"DlgnCifar10Deep{depth}" + ("" if rank == 2 else f"Rank{rank}")
            add(f"cifar10_deep{depth}{suffix}", (9, 32, 32), (width,) * (4 * factor), 10, tau, rank, 0, source)
    for rank, width in ((2, 128000), (4, 56000), (6, 42000)):
        for factor in (1, 3):
            depth  = "" if factor == 1 else str(factor)
            suffix = "" if rank == 2 else f"_rank{rank}"
            source = f"DlgnCifar10MediumDeep{depth}" + ("" if rank == 2 else f"Rank{rank}")
            add(f"cifar10_medium_deep{depth}{suffix}", (9, 32, 32), (width,) * (4 * factor), 10, 100.0, rank, 0, source)

    for size, depth, rank, width in (("small", 2, 2, 32000), ("small", 2, 4, 16000), ("medium", 4, 2, 128000),
                                     ("medium", 4, 4, 64000), ("medium", 4, 6, 42330)):
        for bits in (10, 20, 50, 100):
            suffix = "" if rank == 2 else f"_rank{rank}"
            source = f"DlgnJsc{size.title()}{bits}Bits" + ("" if rank == 2 else f"Rank{rank}")
            add(f"jsc_{size}_{bits}bit{suffix}", (16 * bits,), (width,) * depth, 5, 50.0, rank, 0, source)

    for dataset, bits, rank, widths, tau, learning in (
        ("cifar10", 10, 2, (24000, 24000), 1 / 0.03, (0, 2)), ("cifar10", 10, 6, (8000,), 1 / 0.03, (0,)),
        ("fashion_mnist", 7, 2, (8000, 8000), 1 / 0.061, (0, 2)), ("fashion_mnist", 7, 6, (2000, 2000), 1 / 0.122, (0, 2)),
    ):
        shape = (3 * bits, 32, 32) if dataset == "cifar10" else (1, 28, 28 * bits)
        base  = "DlgnCifar10Dwn" if dataset == "cifar10" else "DlgnFashionMnistDwn"
        for learn in learning:
            suffix = "" if learn == 0 else f"_learn{learn}"
            source = f"{base}Rank{rank}" + ("" if learn == 0 else f"Learn{learn}")
            add(f"{dataset}_dwn_rank{rank}{suffix}", shape, widths, 10, tau, rank, learn, source)

    for learn in (0, 1):
        add(f"jsc_dwn_tiny_rank6_learn{learn}", (3200,), (10,), 5, 1 / 0.7, 6, learn, f"DlgnJscDwnTinyRank6Learn{learn}")
    for bits in (1, 2, 5, 10, 20, 50, 100, 200):
        for learn in (0, 1):
            source  = f"DlgnJscDwnSmallRank6Bits{bits}" + ("Learn1" if learn else "")
            aliases = (f"DlgnJscDwnSmallRank6Learn{learn}",) if bits == 200 else ()
            add(f"jsc_dwn_small_{bits}bit_rank6_learn{learn}", (16 * bits,), (50,), 5, 1 / 0.3, 6, learn, source, *aliases)
    for rank, width in ((2, 1080), (4, 540), (6, 360)):
        for bits in (2, 5, 10, 20, 50, 100):
            add(f"jsc_dwn_medium_{bits}bit_rank{rank}", (16 * bits,), (width,), 5, 10.0, rank, 0, f"DlgnJscDwnMediumRank{rank}Bits{bits}")
    for learn in (0, 1):
        add(f"jsc_dwn_medium_200bit_rank6_learn{learn}", (3200,), (360,), 5, 10.0, 6, learn, f"DlgnJscDwnMediumRank6Learn{learn}")
    for bits in (2, 5, 10, 20, 50, 100, 200):
        aliases = ("DlgnJscDwnLargeRank6Learn0",) if bits == 200 else ()
        add(f"jsc_dwn_large_{bits}bit_rank6", (16 * bits,), (2400,), 5, 1 / 0.03, 6, 0, f"DlgnJscDwnLargeRank6Bits{bits}", *aliases)
    add("jsc_dwn_large_200bit_rank6_learn1", (3200,), (2400,), 5, 1 / 0.03, 6, 1, "DlgnJscDwnLargeRank6Learn1")
    return presets


DENSE_PRESETS = _build_presets()
