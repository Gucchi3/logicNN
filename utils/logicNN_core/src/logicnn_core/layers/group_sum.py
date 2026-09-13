"""最後の特徴軸を連続したgroupへ分け、biasと温度を適用する。"""

from __future__ import annotations

from torch import Tensor, nn

from ..layer_settings import _validate_boolean, _validate_finite_number, _validate_positive_integer, _validate_positive_number

# torchlogix/layers/groupsum.pyのgroup集約と条件付き演算を基に再構成した。MIT License。
# Copyright (c) 2021-2023 Dr. Felix Petersen
# Copyright (c) 2024-present Dr. Lino Gerlach
# 許諾条件の全文と変更の由来は同梱LICENSEおよびTHIRD_PARTY_NOTICESを参照。

class GroupSum(nn.Module):
    """連続する特徴群を合計する、学習可能な重みを持たない集約層。"""

    def __init__(self, groups: int, *, tau: float = 1.0, bias: float = 0.0) -> None:
        """group数と有限なbias・正のtauを検証して集約層を作る。"""
        super().__init__()
        _validate_positive_integer("groups", groups)
        _validate_positive_number("tau", tau)
        _validate_finite_number("bias", bias)
        self.groups      = groups
        self.tau         = float(tau)
        self.bias        = float(bias)
        self.export_mode = False

    def forward(self, inputs: Tensor) -> Tensor:
        """各groupの和に非既定biasとtauだけを適用し、Torchのdtype演算を保つ。"""
        shape  = (*inputs.shape[:-1], self.groups, inputs.shape[-1] // self.groups)
        result = inputs.reshape(shape).sum(-1)
        if self.bias != 0:
            result = result + self.bias
        if self.tau != 1:
            result = result / self.tau
        return result

    def set_export_mode(self, enabled: bool = True) -> None:
        """集約計算を変えずに回路出力状態を切り替え、評価状態へ移る。"""
        _validate_boolean("enabled", enabled)
        self.eval()
        self.export_mode = enabled

    def train(self, mode: bool = True) -> GroupSum:
        """回路出力モードの明示解除前に学習へ戻る操作を拒否する。"""
        if mode and self.export_mode:
            raise RuntimeError("回路出力中は学習できません。先にset_export_mode(False)で明示解除してからtrain()を呼んでください")
        return super().train(mode)
