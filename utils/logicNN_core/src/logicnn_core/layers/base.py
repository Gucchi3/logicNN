"""論理層の共通設定と、明示的な回路出力モードの切替を管理する。"""

# torchlogix/layers/base.pyの論理層interfaceを基に再構成した。MIT License。
# Copyright (c) 2021-2023 Dr. Felix Petersen
# Copyright (c) 2024-present Dr. Lino Gerlach
# 許諾条件の全文と変更の由来は同梱LICENSEおよびTHIRD_PARTY_NOTICESを参照。

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from torch import Tensor, nn

from ..layer_settings import ConnectionConfig, LUTConfig, _validate_boolean, _validate_finite_number
from ..parametrizations import build_parametrization

__all__ = ["LogicLayer"]


class LogicLayer(nn.Module, ABC):
    """計算方式の共通設定を保持し、層固有の重みと接続をsubclassへ委ねる。"""

    _export_buffer_names: tuple[str, ...] = ()

    def __init__(self, *, lut: LUTConfig = LUTConfig(), connections: ConnectionConfig = ConnectionConfig(), gradient_scale: float = 1.0) -> None:
        """不変の生成時設定と実行モードを分離し、状態復元後の再生成を登録する。"""
        super().__init__()
        _validate_finite_number("gradient_scale", gradient_scale)
        self.lut_config        = lut
        self.connection_config = connections
        self.num_inputs        = lut.num_inputs
        self.gradient_scale    = float(gradient_scale)
        self.parametrization   = build_parametrization(lut)
        self.export_mode       = False
        self.register_load_state_dict_post_hook(self._refresh_loaded_export_state)

    def train(self, mode: bool = True) -> LogicLayer:
        """回路出力中の学習切替を拒否し、それ以外は子moduleへ標準の切替を伝える。"""
        if mode and self.export_mode:
            raise RuntimeError("export中は学習できません。先にset_export_mode(False)で明示解除してからtrain()を呼んでください")
        return super().train(mode)

    def set_export_mode(self, enabled: bool = True) -> None:
        """検証済みの回路用snapshotへ切り替え、解除後は通常評価へ戻す。"""
        _validate_boolean("enabled", enabled)
        buffers = self._make_export_buffers() if enabled else {}
        self.eval()
        for name in self._export_buffer_names:
            if enabled:
                self.register_buffer(name, buffers[name], persistent=True)
            elif name in self._buffers:
                delattr(self, name)
        self.export_mode = enabled

    def _load_from_state_dict(
        self,
        state_dict: dict[str, Any],
        prefix: str,
        local_metadata: dict[str, Any],
        strict: bool,
        missing_keys: list[str],
        unexpected_keys: list[str],
        error_msgs: list[str],
    ) -> None:
        """既知の回路用派生bufferだけを無視し、重みと接続の標準復元を行う。"""
        derived_keys = {prefix + name for name in self._export_buffer_names}
        for key in derived_keys:
            state_dict.pop(key, None)
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)
        missing_keys[:] = [key for key in missing_keys if key not in derived_keys]

    def _refresh_loaded_export_state(self, module: nn.Module, incompatible_keys: Any) -> None:
        """子接続の読込完了後に、復元した重みと接続から回路用snapshotを作り直す。"""
        if self.export_mode:
            self.set_export_mode(True)

    @abstractmethod
    def _make_export_buffers(self) -> dict[str, Tensor]:
        """状態を変更する前に、層固有の回路用bufferを検証して独立コピーする。"""
        raise NotImplementedError

    @abstractmethod
    def forward(self, inputs: Tensor) -> Tensor:
        """層固有の入力shapeで論理ゲートを学習・評価・回路実行する。"""
        raise NotImplementedError

    @abstractmethod
    def truth_tables(self) -> Tensor | list[Tensor]:
        """最新のLUT重みから層内のbool真理値表を生成する。"""
        raise NotImplementedError

    @abstractmethod
    def truth_tables_with_ids(self) -> tuple[Tensor, Tensor | None] | tuple[list[Tensor], list[Tensor | None]]:
        """最新の真理値表と、入力本数が対応する場合の整数IDを返す。"""
        raise NotImplementedError

    @abstractmethod
    def regularization_loss(self, kind: str | None = None) -> Tensor:
        """接続logitを含めず、層内の全LUTゲートの平均正則化を返す。"""
        raise NotImplementedError

    @abstractmethod
    def rescale_weights_(self, method: str | None = None) -> None:
        """LUT重みだけを検証済みの再スケール方式で更新する。"""
        raise NotImplementedError
