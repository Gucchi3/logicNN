"""通常のfloat32 maxと回路出力用bool ORを切り替えるpooling層。"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as torch_functional
from torch import Tensor, nn

from ..layer_settings import _validate_boolean

# torchlogix/layers/pool.pyのmax poolingとbool OR処理を基に再構成した。MIT License。
# Copyright (c) 2021-2023 Dr. Felix Petersen
# Copyright (c) 2024-present Dr. Lino Gerlach
# 許諾条件の全文と変更の由来は同梱LICENSEおよびTHIRD_PARTY_NOTICESを参照。

def _spatial_tuple(name: str, value: int | tuple[int, ...], dimensions: int, *, allow_zero: bool = False) -> tuple[int, ...]:
    """空間設定を次元数に合う整数tupleへそろえ、各値の下限を確認する。"""
    if type(value) is int:
        values = (value,) * dimensions
    elif isinstance(value, tuple):
        values = value
    else:
        raise TypeError(f"{name} はboolではない整数または整数tupleで指定してください")
    if len(values) != dimensions:
        raise ValueError(f"{name} のtupleには {dimensions} 個の値が必要です")
    if any(type(item) is not int for item in values):
        raise TypeError(f"{name} の各値はboolではない整数で指定してください")
    if any(item < (0 if allow_zero else 1) for item in values):
        raise ValueError(f"{name} の各値は {'0以上' if allow_zero else '0より大きい'}整数で指定してください")
    return values


class _OrPooling(nn.Module):
    """空間次元数だけが異なるOR poolingの検証と実行をまとめる。"""

    _dimensions: int

    def __init__(self, kernel_size: int | tuple[int, ...], stride: int | tuple[int, ...] | None = None, padding: int | tuple[int, ...] = 0) -> None:
        """窓・移動幅・paddingを検証し、通常モードのpooling層を作る。"""
        super().__init__()
        self.kernel_size = _spatial_tuple("kernel_size", kernel_size, self._dimensions)
        self.stride      = self.kernel_size if stride is None else _spatial_tuple("stride", stride, self._dimensions)
        self.padding     = _spatial_tuple("padding", padding, self._dimensions, allow_zero=True)
        self.export_mode = False
        if any(pad > width // 2 for pad, width in zip(self.padding, self.kernel_size)):
            raise ValueError("padding は各次元のkernel_size // 2以下で指定してください")

    def forward(self, inputs: Tensor) -> Tensor:
        """通常はfloat32 max、回路出力中はboolの窓内ORを計算する。"""
        if self.export_mode and inputs.dtype != torch.bool:
            raise TypeError("回路出力モードのpooling入力はbool Tensorで指定してください")
        if inputs.shape[0] == 0:
            output_shape = tuple((size + 2 * pad - width) // step + 1 for size, pad, width, step
                                 in zip(inputs.shape[2:], self.padding, self.kernel_size, self.stride))
            values = inputs if self.export_mode else inputs.float()
            return values.reshape(0, inputs.shape[1], *output_shape)
        if self.export_mode:
            return self._boolean_pool(inputs)
        pool = torch_functional.max_pool2d if self._dimensions == 2 else torch_functional.max_pool3d
        return pool(inputs.float(), self.kernel_size, self.stride, self.padding)

    def _boolean_pool(self, inputs: Tensor) -> Tensor:
        """Falseでpaddingした窓を展開し、bit単位ORだけで集約する。"""
        padding = tuple(size for width in reversed(self.padding) for size in (width, width))
        values  = torch_functional.pad(inputs, padding, value=False)
        for axis, (width, step) in enumerate(zip(self.kernel_size, self.stride), start=2):
            values = values.unfold(axis, width, step)
        windows = values.flatten(start_dim=self._dimensions + 2)
        result  = windows[..., 0]
        for index in range(1, math.prod(self.kernel_size)):
            result = result | windows[..., index]
        return result

    def set_export_mode(self, enabled: bool = True) -> None:
        """回路出力モードを切り替え、解除した場合も評価状態を維持する。"""
        _validate_boolean("enabled", enabled)
        self.eval()
        self.export_mode = enabled

    def train(self, mode: bool = True) -> _OrPooling:
        """回路出力モードの明示解除前に学習へ戻る操作を拒否する。"""
        if mode and self.export_mode:
            raise RuntimeError("回路出力中は学習できません。先にset_export_mode(False)で明示解除してからtrain()を呼んでください")
        return super().train(mode)


class OrPooling2d(_OrPooling):
    """2次元の空間窓を通常maxまたは回路用ORで集約する。"""

    _dimensions = 2


class OrPooling3d(_OrPooling):
    """3次元の空間窓を通常maxまたは回路用ORで集約する。"""

    _dimensions = 3
