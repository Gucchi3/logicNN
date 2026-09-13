"""2つの論理Convと明示的なskip pathを、shape一致を確認してORで結合する。"""

# torchlogix/modules/resblock.pyを基に再構成した。MIT License。
# Copyright (c) 2021-2023 Dr. Felix Petersen
# Copyright (c) 2024-present Dr. Lino Gerlach
# 許諾条件の全文と変更の由来は同梱LICENSEおよびTHIRD_PARTY_NOTICESを参照。

from __future__ import annotations

import torch
from torch import Tensor, nn

from ..export_mode import set_export_mode
from ..layer_settings import ConnectionConfig, LUTConfig, _validate_boolean
from ..layers.convolution import LogicConv2d, LogicConv3d
from ..layers.pooling import OrPooling2d, OrPooling3d

__all__ = ["ResidualLogicBlock"]


def _output_shape(module: nn.Module, shape: tuple[int, ...], dimensions: int, ancestors: frozenset[int] = frozenset()) -> tuple[int, ...]:
    """既知moduleのmetadataだけでchannel・空間shapeを追跡し、dummy forwardを避ける。"""
    if id(module) in ancestors:
        raise ValueError("projectionに循環参照を含めることはできません")
    if type(module) is nn.Identity:
        return shape
    if type(module) is nn.Sequential:
        ancestors = ancestors | {id(module)}
        for child in module:
            shape = _output_shape(child, shape, dimensions, ancestors)
        return shape
    if type(module) in (LogicConv2d, LogicConv3d):
        if module.conv_dimension != dimensions or shape != (module.in_channels, *module.input_size):
            raise ValueError("projectionのLogicConv入力shape・空間次元が直前の出力と一致しません")
        return (module.out_channels, *module.output_size)
    if type(module) in (OrPooling2d, OrPooling3d):
        expected_dimensions = 2 if type(module) is OrPooling2d else 3
        if expected_dimensions != dimensions:
            raise ValueError("projectionのOR poolingの空間次元がResidualと一致しません")
        spatial = tuple((size + 2 * pad - width) // step + 1 for size, pad, width, step in zip(shape[1:], module.padding, module.kernel_size, module.stride))
        if any(size <= 0 for size in spatial):
            raise ValueError("projectionのOR poolingから得られる出力shapeは正の空間サイズが必要です")
        return (shape[0], *spatial)
    if type(module) is ResidualLogicBlock:
        if module.conv_dimension != dimensions or module.input_shape != shape:
            raise ValueError("projectionのResidual入力shape・空間次元が直前の出力と一致しません")
        return module.output_shape
    raise TypeError(f"projectionの静的shape検証で未対応のmoduleです: {type(module).__name__}。既知の論理Conv・OR pooling・Identity・Sequentialを使用してください")


class ResidualLogicBlock(nn.Module):
    """主経路の2Convと利用者指定のprojectionを、通常はrelaxed OR、回路ではbool ORで結ぶ。"""

    def __init__(
        self,
        input_size: int | tuple[int, ...],
        in_channels: int,
        out_channels: int,
        *,
        tree_depth: int = 3,
        kernel_size: int | tuple[int, ...] = 3,
        padding: int | tuple[int, ...] = 1,
        downsample: bool = False,
        conv_dimension: int = 2,
        projection: nn.Module | None = None,
        lut: LUTConfig = LUTConfig(),
        connections: ConnectionConfig = ConnectionConfig(),
        gradient_scale: float = 1.0,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        """実際のConv・pool公式で両経路のshapeを確定し、自動projectionなしで整合を検証する。"""
        super().__init__()
        _validate_boolean("downsample", downsample)
        if type(conv_dimension) is not int:
            raise TypeError("conv_dimension はboolではないPython整数で指定してください")
        if conv_dimension not in (2, 3):
            raise ValueError("conv_dimension は2または3で指定してください")
        if projection is not None and not isinstance(projection, nn.Module):
            raise TypeError("projection はnn.ModuleまたはNoneで指定してください")
        conv   = LogicConv2d if conv_dimension == 2 else LogicConv3d
        pool   = OrPooling2d if conv_dimension == 2 else OrPooling3d
        first  = conv(input_size, in_channels, out_channels, tree_depth, kernel_size, padding=padding, lut=lut, connections=connections,
                      gradient_scale=gradient_scale, device=device, dtype=dtype)
        pool_1 = pool(2, 2) if downsample else nn.Identity()
        middle = _output_shape(pool_1, (out_channels, *first.output_size), conv_dimension)
        second = conv(middle[1:], out_channels, out_channels, tree_depth, kernel_size, padding=padding, lut=lut, connections=connections,
                      gradient_scale=gradient_scale, device=device, dtype=dtype)
        pool_2 = pool(2, 2) if downsample else nn.Identity()
        self.main              = nn.Sequential(first, pool_1, second, pool_2)
        self.shortcut          = nn.Identity() if projection is None else projection
        self.input_size        = first.input_size
        self.input_shape       = (in_channels, *self.input_size)
        self.output_shape      = _output_shape(pool_2, (out_channels, *second.output_size), conv_dimension)
        self.output_size       = self.output_shape[1:]
        self.in_channels       = in_channels
        self.out_channels      = out_channels
        self.tree_depth        = tree_depth
        self.kernel_size       = first.kernel_size
        self.padding           = first.padding
        self.downsample        = downsample
        self.conv_dimension    = conv_dimension
        self.lut_config        = first.lut_config
        self.connection_config = first.connection_config
        self.gradient_scale    = first.gradient_scale
        self.export_mode       = False
        shortcut_shape         = _output_shape(self.shortcut, self.input_shape, conv_dimension)
        if shortcut_shape != self.output_shape:
            raise ValueError(f"mainの出力shape {self.output_shape} とskipのshape {shortcut_shape} が異なります。一致するprojectionを明示してください")

    def forward(self, inputs: Tensor) -> Tensor:
        """mainとshortcutを、broadcastせずrelaxed ORまたはbool ORで結合する。"""
        main     = self.main(inputs)
        shortcut = self.shortcut(inputs)

        if main.shape != shortcut.shape:
            raise ValueError("mainとshortcutのshapeは一致させてください。暗黙のbroadcastは行いません")
        if self.export_mode:
            if main.dtype != torch.bool or shortcut.dtype != torch.bool:
                raise TypeError("export中のmainとshortcutはbool Tensorを返す必要があります")
            return main | shortcut

        return main + shortcut - main * shortcut

    def set_export_mode(self, enabled: bool = True) -> None:
        """共通の単一走査へ委譲し、共有moduleを重複更新せず内部のmodeを伝播する。"""
        set_export_mode(self, enabled)

    def _set_export_mode_local(self, enabled: bool = True) -> None:
        """共通walkerから呼ばれたときに、再帰せず自分の回路出力flagだけを設定する。"""
        _validate_boolean("enabled", enabled)
        self.export_mode = enabled

    def train(self, mode: bool = True) -> ResidualLogicBlock:
        """回路出力中の学習切替を明示拒否し、それ以外は標準のmodule切替を使う。"""
        _validate_boolean("mode", mode)
        if mode and self.export_mode:
            raise RuntimeError("export中は学習できません。先にset_export_mode(False)で明示解除してからtrain()を呼んでください")
        return super().train(mode)

    def extra_repr(self) -> str:
        """入力・出力shapeと空間次元、縮小設定、回路出力状態を簡潔に表示する。"""
        return (f"input_shape={self.input_shape}, output_shape={self.output_shape}, conv_dimension={self.conv_dimension}, "
                f"tree_depth={self.tree_depth}, downsample={self.downsample}, export_mode={self.export_mode}")
