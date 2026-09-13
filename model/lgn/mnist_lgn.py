"""モデル内の固定定数から、前処理済みMNIST用の論理ゲートNNを構築する。"""

from __future__ import annotations

import math
from dataclasses import asdict

from logicnn_core.layer_settings import ConnectionConfig, LUTConfig
from logicnn_core.layers import GroupSum, LogicConv2d, LogicDense, OrPooling2d
from torch import Tensor, nn

INPUT_SHAPE         = (1, 28, 28)
NUM_CLASSES         = 10
CONV_KERNELS        = 16
TREE_DEPTH          = 2
RECEPTIVE_FIELD     = 3
CONV_STRIDE         = 1
CONV_PADDING        = 0
POOL_KERNEL_SIZE    = 2
POOL_STRIDE         = 2
POOL_PADDING        = 0
HIDDEN_SIZE         = 4000
GROUP_TAU           = 8.0
GROUP_BIAS          = 0.0
FLATTEN_START_DIM   = 1
FLATTEN_END_DIM     = -1
GRADIENT_SCALE      = 1.0
LUT_SETTINGS        = LUTConfig()
CONNECTION_SETTINGS = ConnectionConfig()
CONV_OUTPUT_SIZE    = tuple((size + 2 * CONV_PADDING - RECEPTIVE_FIELD) // CONV_STRIDE + 1 for size in INPUT_SHAPE[1:])
POOL_OUTPUT_SIZE    = tuple((size + 2 * POOL_PADDING - POOL_KERNEL_SIZE) // POOL_STRIDE + 1 for size in CONV_OUTPUT_SIZE)
FLATTENED_SIZE      = CONV_KERNELS * math.prod(POOL_OUTPUT_SIZE)


def _logic_settings() -> dict[str, object]:
    """旧モデルから維持したLUT・接続の既定設定を独立した記録用辞書にする。"""
    return {"lut": asdict(LUT_SETTINGS), "connections": asdict(CONNECTION_SETTINGS), "gradient_scale": GRADIENT_SCALE}


class MNISTLGN(nn.Module):
    """論理畳み込み・OR pooling・2段DenseでMNISTの10クラススコアを返す。"""

    input_shape: tuple[int, ...] = INPUT_SHAPE
    num_classes: int            = NUM_CLASSES

    def __init__(self) -> None:
        """構造と学習方式をモデル定義内の定数だけから組み立てる。"""
        super().__init__()
        self.network = nn.Sequential(
            LogicConv2d(
                INPUT_SHAPE[1:], INPUT_SHAPE[0], CONV_KERNELS, TREE_DEPTH, RECEPTIVE_FIELD, stride=CONV_STRIDE, padding=CONV_PADDING,
                lut=LUT_SETTINGS, connections=CONNECTION_SETTINGS, gradient_scale=GRADIENT_SCALE,
            ),
            OrPooling2d(kernel_size=POOL_KERNEL_SIZE, stride=POOL_STRIDE, padding=POOL_PADDING),
            nn.Flatten(start_dim=FLATTEN_START_DIM, end_dim=FLATTEN_END_DIM),
            LogicDense(FLATTENED_SIZE, HIDDEN_SIZE, lut=LUT_SETTINGS, connections=CONNECTION_SETTINGS, gradient_scale=GRADIENT_SCALE),
            LogicDense(HIDDEN_SIZE, HIDDEN_SIZE, lut=LUT_SETTINGS, connections=CONNECTION_SETTINGS, gradient_scale=GRADIENT_SCALE),
            GroupSum(NUM_CLASSES, tau=GROUP_TAU, bias=GROUP_BIAS),
        )

    @property
    def input_size(self) -> int:
        """batchを含まない入力形状から、入力要素数を読み取り専用で導出する。"""
        return math.prod(self.input_shape)

    def forward(self, inputs: Tensor) -> Tensor:
        """前処理済みの[batch,1,28,28]を[batch,10]のクラススコアへ変換する。"""
        return self.network(inputs)

    def architecture_info(self) -> dict[str, object]:
        """層順序・全固定設定・導出shapeをJSON互換の独立した辞書として返す。"""
        return {
            "input_shape": list(self.input_shape), "input_size": self.input_size, "num_classes": self.num_classes,
            "layers": [
                {
                    "type": "LogicConv2d", "input_size": list(INPUT_SHAPE[1:]), "in_channels": INPUT_SHAPE[0], "out_channels": CONV_KERNELS,
                    "tree_depth": TREE_DEPTH, "kernel_size": [RECEPTIVE_FIELD] * 2, "stride": [CONV_STRIDE] * 2, "padding": [CONV_PADDING] * 2,
                    "output_size": list(CONV_OUTPUT_SIZE), **_logic_settings(),
                },
                {
                    "type": "OrPooling2d", "kernel_size": [POOL_KERNEL_SIZE] * 2, "stride": [POOL_STRIDE] * 2, "padding": [POOL_PADDING] * 2,
                    "output_size": list(POOL_OUTPUT_SIZE),
                },
                {"type": "Flatten", "start_dim": FLATTEN_START_DIM, "end_dim": FLATTEN_END_DIM, "out_features": FLATTENED_SIZE},
                {"type": "LogicDense", "in_features": FLATTENED_SIZE, "out_features": HIDDEN_SIZE, **_logic_settings()},
                {"type": "LogicDense", "in_features": HIDDEN_SIZE, "out_features": HIDDEN_SIZE, **_logic_settings()},
                {"type": "GroupSum", "groups": NUM_CLASSES, "tau": GROUP_TAU, "bias": GROUP_BIAS},
            ],
        }
