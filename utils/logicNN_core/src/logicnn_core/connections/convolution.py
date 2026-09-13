"""固定した受容野の接続と、2D・3Dの論理tree入力の取得を担当する。"""

# torchlogix/connections.pyのFixedConvConnectionsを基に再構成した。MIT License。
# Copyright (c) 2021-2023 Dr. Felix Petersen
# Copyright (c) 2024-present Dr. Lino Gerlach
# 許諾条件の全文と変更の由来は同梱LICENSEおよびTHIRD_PARTY_NOTICESを参照。

from __future__ import annotations

import torch
from torch import Tensor
from torch.nn import functional as F

from ..functional.combinatorics import _bounded_binomial, sample_unique_combinations
from ..layer_settings import ConnectionConfig, _validate_positive_integer
from .base import Connections

__all__ = ["FixedConvConnections"]

_INT64_MAX = torch.iinfo(torch.int64).max


def _positive_integer(name: str, value: int) -> None:
    """正のPython整数を要求し、indexのint64範囲を超える値を拒否する。"""
    _validate_positive_integer(name, value)
    if value > _INT64_MAX:
        raise ValueError(f"{name} はint64の上限以下で指定してください")


def _spatial_tuple(name: str, value: int | tuple[int, ...], dimensions: int, *, allow_zero: bool = False) -> tuple[int, ...]:
    """intまたは次元数と一致する整数tupleを検証して空間値へ正規化する。"""
    if type(value) is int:
        values = (value,) * dimensions
    elif isinstance(value, tuple):
        values = value
    else:
        raise TypeError(f"{name} はintまたは{dimensions}要素のtupleで指定してください")
    if len(values) != dimensions:
        raise ValueError(f"{name} は{dimensions}要素で指定してください")
    minimum = 0 if allow_zero else 1
    for size in values:
        if type(size) is not int:
            raise TypeError(f"{name} の各要素はboolではないPython整数で指定してください")
        if not minimum <= size <= _INT64_MAX:
            raise ValueError(f"{name} の各要素は{minimum}以上int64上限以下で指定してください")
    return values


def _bounded_product(name: str, *factors: int) -> int:
    """導出サイズの積がint64上限を超えないか、巨大Tensorの生成前に検証する。"""
    result = 1
    for factor in factors:
        if factor > 0 and result > _INT64_MAX // factor:
            raise ValueError(f"{name} の要素数がint64上限を超えます")
        result *= factor
    return result


def _bounded_power(base: int, exponent: int) -> int:
    """treeの巨大累乗を展開せず、int64内に収まるleaf数だけを計算する。"""
    if base == 1:
        return 1
    if exponent >= 63:
        raise ValueError("treeのleaf数がint64上限を超えます")
    result = 1
    for _ in range(exponent):
        result = _bounded_product("tree", result, base)
    return result


class FixedConvConnections(Connections):
    """固定局所接続を保存し、すべてのtree levelでrank軸をindex 1へ揃える。"""

    def __init__(
        self,
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
    ) -> None:
        """空間shapeと抽選条件を検証し、局所座標・開始位置・全levelのindexを保存する。"""
        super().__init__(num_inputs, config)
        if config.kind != "fixed":
            raise ValueError("FixedConvConnections はkind='fixed'だけに対応しています")
        if type(conv_dimension) is not int:
            raise TypeError("conv_dimension はboolではないPython整数で指定してください")
        if conv_dimension not in (2, 3):
            raise ValueError("conv_dimension は2または3で指定してください")
        for name, value in (("num_inputs", num_inputs), ("in_channels", in_channels), ("out_channels", out_channels), ("tree_depth", tree_depth)):
            _positive_integer(name, value)

        self.conv_dimension = conv_dimension
        self.in_channels    = in_channels
        self.out_channels   = out_channels
        self.tree_depth     = tree_depth
        self.input_size     = _spatial_tuple("input_size", input_size, conv_dimension)
        self.kernel_size    = _spatial_tuple("kernel_size", kernel_size, conv_dimension)
        self.stride         = _spatial_tuple("stride", stride, conv_dimension)
        self.padding        = _spatial_tuple("padding", padding, conv_dimension, allow_zero=True)
        if any(kernel > size for kernel, size in zip(self.kernel_size, self.input_size)):
            raise ValueError("kernel_size はpadding前のinput_size以下で指定してください")
        self.padded_size = tuple(size + 2 * pad for size, pad in zip(self.input_size, self.padding))
        self.output_size = tuple((size - kernel) // step + 1 for size, kernel, step in zip(self.padded_size, self.kernel_size, self.stride))
        if any(size > _INT64_MAX for size in self.padded_size) or any(size < 1 for size in self.output_size):
            raise ValueError("padding後の空間sizeはint64内、出力sizeは1以上である必要があります")
        self.num_positions  = _bounded_product("kernel_positions", *self.output_size)
        self._spatial_count = _bounded_product("受容野", *self.kernel_size)
        self._leaf_count    = _bounded_power(num_inputs, tree_depth)
        self._candidate_count = _bounded_product("受容野の候補", self._spatial_count, in_channels)
        _bounded_product("padding後の入力", in_channels, *self.padded_size)
        _bounded_product("indices_0", self._leaf_count, out_channels, self.num_positions, conv_dimension + 1)
        _bounded_product("kernel_positions", self.num_positions, conv_dimension)
        lower_elements = tree_depth - 1 if num_inputs == 1 else (self._leaf_count - num_inputs) // (num_inputs - 1)
        _bounded_product("後続tree indexのbytes", lower_elements, 8)
        self.node_counts = tuple(_bounded_power(num_inputs, tree_depth - level - 1) for level in range(tree_depth))
        self._validate_sampling_feasibility()

        chosen_device = torch.empty(0, dtype=torch.int64, device=device).device
        coordinates = self._sample_kernel_coordinates(chosen_device)
        positions   = self._make_kernel_positions(chosen_device)
        self.register_buffer("kernel_coordinates", coordinates)
        self.register_buffer("kernel_positions", positions)
        self.register_buffer("indices_0", self._make_first_indices(coordinates, positions))
        for level in range(1, tree_depth):
            indices = torch.arange(self.node_counts[level - 1], device=chosen_device).reshape(-1, num_inputs).T.contiguous()
            self.register_buffer(f"indices_{level}", indices)

    @property
    def indices(self) -> tuple[Tensor, ...]:
        """persistent bufferとして保存した各tree levelのindexを順序付きtupleで返す。"""
        return tuple(getattr(self, f"indices_{level}") for level in range(self.tree_depth))

    def _validate_sampling_feasibility(self) -> None:
        """channel群への均等配分と、重複なし抽選に必要な候補数を確認する。"""
        group = self.config.channel_group_size
        if group is not None:
            if group > self.in_channels:
                raise ValueError("channel_group_size はin_channels以下で指定してください")
            if self._leaf_count % group:
                raise ValueError("treeの全leaf数はchannel_group_sizeで割り切れる必要があります")
            if self.config.init == "random_unique" and self._leaf_count // group > self._spatial_count:
                raise ValueError("channelごとの重複なしleaf数が受容野の空間位置数を超えます")
        elif self.config.init == "random_unique":
            if self.num_inputs > self._candidate_count or _bounded_binomial(self._candidate_count, self.num_inputs) < self.node_counts[0]:
                raise ValueError("受容野に必要な数の重複しない入力tupleがありません")

    def _sample_kernel_coordinates(self, device: torch.device) -> Tensor:
        """全候補を列挙せず局所位置を抽選し、末尾をspatial座標・channelの順にする。"""
        group = self.config.channel_group_size
        if group is None:
            shape = (self.out_channels, self.node_counts[0], self.num_inputs)
            if self.config.init == "random":
                flat = torch.randint(self._candidate_count, shape, device=device)
            else:
                flat = sample_unique_combinations(self._candidate_count, self.num_inputs, self.node_counts[0], self.out_channels, device=device)
        else:
            per_channel = self._leaf_count // group
            selected    = []
            for kernel in range(self.out_channels):
                start = kernel % (self.in_channels - group + 1)
                if self.config.init == "random":
                    spatial = torch.randint(self._spatial_count, (group, per_channel), device=device)
                else:
                    spatial = sample_unique_combinations(self._spatial_count, 1, per_channel, group, device=device).squeeze(-1)
                channels = torch.arange(start, start + group, device=device).unsqueeze(-1)
                values   = (spatial * self.in_channels + channels).reshape(-1)
                order    = torch.randperm(self._leaf_count, device=device)
                selected.append(values[order].reshape(self.node_counts[0], self.num_inputs))
            flat = torch.stack(selected)
        parts   = [flat % self.in_channels]
        spatial = flat // self.in_channels
        for size in reversed(self.kernel_size):
            parts.insert(0, spatial % size)
            spatial = spatial // size
        return torch.stack(parts, dim=-1).permute(2, 0, 1, 3).contiguous()

    def _make_kernel_positions(self, device: torch.device) -> Tensor:
        """padding済み空間での滑走開始位置を、最終空間軸が最速で進む順に作る。"""
        axes = [torch.arange(size, device=device) * step for size, step in zip(self.output_size, self.stride)]
        grid = torch.meshgrid(*axes, indexing="ij")
        return torch.stack(grid, dim=-1).reshape(self.num_positions, self.conv_dimension)

    def _make_first_indices(self, coordinates: Tensor, positions: Tensor) -> Tensor:
        """局所座標へ滑走開始位置を加算し、level 0の絶対padding座標を作る。"""
        spatial  = coordinates[..., :-1].unsqueeze(2) + positions.reshape(1, 1, self.num_positions, 1, self.conv_dimension)
        channels = coordinates[..., -1:].unsqueeze(2).expand(self.num_inputs, self.out_channels, self.num_positions, self.node_counts[0], 1)
        return torch.cat((spatial, channels), dim=-1)

    def forward(self, inputs: Tensor, tree_level: int = 0) -> Tensor:
        """level 0は一度だけpaddingして局所入力を取得し、後段はrank軸をindex 1へ揃える。"""
        if tree_level == 0:
            if any(self.padding):
                padding = tuple(value for pad in reversed(self.padding) for value in (pad, pad))
                inputs  = F.pad(inputs, padding, mode="constant", value=0)
            coordinates = self.indices_0.unbind(-1)
            return inputs[(slice(None), coordinates[-1], *coordinates[:-1])]
        return inputs[..., self.indices[tree_level]].movedim(-2, 1)
