"""固定・soft・学習可能な閾値で特徴をthermometer encodingへ変換する。"""

# torchlogix/layers/binarization.pyの閾値生成・差分学習・軸結合を基に再構成した。MIT License。
# Copyright (c) 2021-2023 Dr. Felix Petersen
# Copyright (c) 2024-present Dr. Lino Gerlach
# 許諾条件の全文と変更の由来は同梱LICENSEおよびTHIRD_PARTY_NOTICESを参照。

from __future__ import annotations

import weakref
from abc import ABC, abstractmethod
from typing import Any

import torch
import torch.nn.functional as torch_functional
from torch import Tensor, nn

from ..functional import gumbel_sigmoid, temperature_sigmoid
from ..layer_settings import _validate_boolean, _validate_choice, _validate_finite_number, _validate_positive_integer, _validate_positive_number


class Binarization(nn.Module, ABC):
    """閾値生成と回路出力用snapshotの管理を共有する二値化interface。"""

    def __init__(self, *, feature_dim: int = -2) -> None:
        """結合軸の型を検証し、回路出力状態とstate復元処理を準備する。"""
        super().__init__()
        if type(feature_dim) is not int:
            raise TypeError("feature_dim はboolではない整数で指定してください")
        self.feature_dim = feature_dim
        self.export_mode = False
        self.register_load_state_dict_post_hook(self._refresh_loaded_state)

    @staticmethod
    def initial_thresholds(dataset: Tensor, bits: int, scope: str, method: str = "uniform") -> Tensor:
        """データ範囲の等分点または旧floor順位の分位点から初期閾値を作る。"""
        _validate_positive_integer("bits", bits)
        _validate_choice("scope", scope, ("global", "feature", "channel"))
        _validate_choice("method", method, ("uniform", "quantile"))
        values = dataset.detach()
        if scope == "global":
            values = values.flatten()
        elif scope == "feature":
            values = values.movedim(0, -1)
        else:
            values = values.transpose(0, 1).reshape(values.shape[1], -1)
        if method == "uniform":
            minimum = values.amin(dim=-1, keepdim=True)
            maximum = values.amax(dim=-1, keepdim=True)
            steps   = torch.arange(1, bits + 1, device=values.device)
            result  = minimum + steps * ((maximum - minimum) / (bits + 1))
        else:
            count   = values.shape[-1]
            indices = torch.tensor([count * index // (bits + 1) for index in range(1, bits + 1)], device=values.device, dtype=torch.int64)
            result  = values.sort(dim=-1).values.index_select(-1, indices)
        return result

    def get_thresholds(self) -> Tensor | None:
        """通常計算に使う現在の閾値を返し、DummyではNoneを返す。"""
        return getattr(self, "thresholds", None)

    def _evaluation_thresholds(self) -> Tensor | None:
        """学習状態によらない評価用閾値を返す。"""
        return self.get_thresholds()

    def set_export_mode(self, enabled: bool = True) -> None:
        """評価用閾値の独立snapshotへ切り替え、解除後も評価状態を維持する。"""
        _validate_boolean("enabled", enabled)
        snapshot = self._evaluation_thresholds() if enabled else None
        if snapshot is not None:
            snapshot = snapshot.detach().clone()
        self.eval()
        if snapshot is not None:
            self.register_buffer("_export_thresholds", snapshot, persistent=True)
        elif "_export_thresholds" in self._buffers:
            delattr(self, "_export_thresholds")
        self.export_mode = enabled

    def train(self, mode: bool = True) -> Binarization:
        """回路出力を明示解除するまで学習状態への切替を拒否する。"""
        if mode and self.export_mode:
            raise RuntimeError("回路出力中は学習できません。先にset_export_mode(False)で明示解除してからtrain()を呼んでください")
        return super().train(mode)

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
        """派生snapshotを読み捨て、閾値の復元はPyTorchの標準処理へ渡す。"""
        derived_key = prefix + "_export_thresholds"
        state_dict.pop(derived_key, None)
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)
        missing_keys[:] = [key for key in missing_keys if key != derived_key]

    def _refresh_loaded_state(self, module: nn.Module, incompatible_keys: Any) -> None:
        """state復元後に現在の閾値から回路出力用snapshotを作り直す。"""
        if self.export_mode:
            self.set_export_mode()

    @abstractmethod
    def forward(self, inputs: Tensor) -> Tensor:
        """各二値化方式に応じた連続値または離散bitを返す。"""
        raise NotImplementedError


class FixedBinarization(Binarization):
    """学習可能な重みを持たず、厳密な閾値比較でbitを生成する。"""

    def __init__(self, thresholds: Tensor | list, *, feature_dim: int = -2) -> None:
        """指定された閾値を独立した永続bufferとして保持する。"""
        super().__init__(feature_dim=feature_dim)
        if isinstance(thresholds, list):
            thresholds = torch.tensor(thresholds, dtype=torch.float32)
        if thresholds.ndim < 1 or thresholds.numel() == 0:
            raise ValueError("thresholds には非空の末尾bit軸が必要です")
        self.register_buffer("thresholds", thresholds.detach().clone(), persistent=True)

    def _sample_training(self, inputs: Tensor, thresholds: Tensor) -> Tensor:
        """固定閾値の学習経路も評価時と同じfloat32二値比較を用いる。"""
        return (inputs > thresholds).to(dtype=torch.float32)

    def forward(self, inputs: Tensor) -> Tensor:
        """通常は方式別sampling、評価と回路出力では厳密な閾値比較を行う。"""
        thresholds = self._export_thresholds if self.export_mode else self.get_thresholds()
        if inputs.ndim >= 3 and thresholds.ndim == 2:
            thresholds = thresholds.reshape(1, thresholds.shape[0], *([1] * (inputs.ndim - 2)), thresholds.shape[-1])
        values = inputs.unsqueeze(-1)
        if self.export_mode:
            result = values > thresholds
        elif self.training:
            result = self._sample_training(values, thresholds)
        else:
            result = (values > thresholds).to(dtype=torch.float32)
        if not -result.ndim <= self.feature_dim < result.ndim - 1 or self.feature_dim == -1:
            raise ValueError("feature_dim は追加したbit軸以外の入力軸で指定してください")
        axis  = self.feature_dim % result.ndim
        order = [*range(axis + 1), result.ndim - 1, *range(axis + 1, result.ndim - 1)]
        shape = [*result.shape[:-1]]
        shape[axis] *= result.shape[-1]
        return result.permute(order).reshape(shape)


class DummyBinarization(Binarization):
    """通常はfloat32変換だけを行い、回路出力ではbool入力をそのまま渡す。"""

    def __init__(self) -> None:
        """閾値や学習Parameterを作らず共通のmode管理だけを用意する。"""
        super().__init__()

    def forward(self, inputs: Tensor) -> Tensor:
        """値とshapeを変えず、通常のfloat32変換またはbool通過を行う。"""
        if self.export_mode:
            if inputs.dtype != torch.bool:
                raise TypeError("DummyBinarizationの回路出力入力はbool Tensorで指定してください")
            return inputs
        return inputs.float()


class SoftBinarization(FixedBinarization):
    """固定閾値を学習中だけ温度付きsigmoidで連続化する。"""

    def __init__(self, thresholds: Tensor | list, *, temperature: float = 0.1, feature_dim: int = -2) -> None:
        """固定閾値と正有限のsampling温度を保持する。"""
        _validate_positive_number("temperature", temperature)
        super().__init__(thresholds, feature_dim=feature_dim)
        self.temperature = float(temperature)

    def _sample_training(self, inputs: Tensor, thresholds: Tensor) -> Tensor:
        """元入力と閾値の差のdtypeを保って温度付きsigmoidを計算する。"""
        return temperature_sigmoid(inputs - thresholds, temperature=self.temperature)


class LearnableBinarization(FixedBinarization):
    """旧式の差分Parameterを学習し、trainとevalの異なる閾値式を維持する。"""

    def __init__(
        self,
        thresholds: Tensor | list,
        *,
        feature_dim: int = -2,
        sampling_temperature: float = 0.1,
        ordering_temperature: float = 0.1,
        sampling: str = "soft",
        max_gradient_norm: float = 0.001,
    ) -> None:
        """元閾値の差分をParameterへ保存し、全勾配を制限するhookを登録する。"""
        _validate_positive_number("sampling_temperature", sampling_temperature)
        _validate_positive_number("ordering_temperature", ordering_temperature)
        _validate_finite_number("max_gradient_norm", max_gradient_norm)
        if max_gradient_norm < 0:
            raise ValueError("max_gradient_norm は0以上の値で指定してください")
        _validate_choice("sampling", sampling, ("soft", "hard", "gumbel_soft", "gumbel_hard"))
        super().__init__(thresholds, feature_dim=feature_dim)
        self.sampling_temperature = float(sampling_temperature)
        self.ordering_temperature = float(ordering_temperature)
        self.sampling             = sampling
        self.max_gradient_norm    = float(max_gradient_norm)
        leading = self.thresholds.new_zeros((*self.thresholds.shape[:-1], 1))
        diffs   = torch.diff(self.thresholds, prepend=leading, dim=-1)
        self.raw_diffs = nn.Parameter(diffs)
        self._ensure_gradient_hook()

    def _ensure_gradient_hook(self) -> None:
        """現在のParameterへ一度だけhookを付け、差し替えやfreeze解除にも対応する。"""
        reference = getattr(self, "_gradient_hook_parameter", None)
        if self.raw_diffs.requires_grad and (reference is None or reference() is not self.raw_diffs):
            self.raw_diffs.register_hook(self._clip_gradient)
            self._gradient_hook_parameter = weakref.ref(self.raw_diffs)

    def _clip_gradient(self, gradient: Tensor) -> Tensor:
        """全raw差分の勾配normへ旧上限と1e-6分母を適用する。"""
        norm = gradient.norm()
        if norm > self.max_gradient_norm:
            gradient = gradient * (self.max_gradient_norm / (norm + 1e-6))
        return gradient

    def get_thresholds(self) -> Tensor:
        """trainは正差分へ変換して累積し、evalはraw差分を直接累積する。"""
        self._ensure_gradient_hook()
        if not self.training:
            return self.raw_diffs.cumsum(-1)
        first = self.raw_diffs[..., :1]
        scale = self.ordering_temperature + 1e-6
        rest  = scale * torch_functional.softplus(self.raw_diffs[..., 1:] / scale)
        return torch.cat((first, rest), dim=-1).cumsum(-1)

    def _evaluation_thresholds(self) -> Tensor:
        """現在のmodeによらず、負差分も保持したraw cumsumを評価閾値とする。"""
        self._ensure_gradient_hook()
        return self.raw_diffs.cumsum(-1)

    def _sample_training(self, inputs: Tensor, thresholds: Tensor) -> Tensor:
        """指定したsigmoid系samplingを適用し、旧hard出力のdtype昇格を維持する。"""
        logits  = inputs - thresholds
        sampler = gumbel_sigmoid if self.sampling.startswith("gumbel") else temperature_sigmoid
        hard    = self.sampling.endswith("hard")
        result  = sampler(logits, temperature=self.sampling_temperature, hard=hard)
        return result.to(dtype=torch.promote_types(logits.dtype, torch.float32)) if hard else result

    def _refresh_loaded_state(self, module: nn.Module, incompatible_keys: Any) -> None:
        """assign復元後のParameterへhookを付け直し、回路snapshotを再生成する。"""
        self._ensure_gradient_hook()
        super()._refresh_loaded_state(module, incompatible_keys)

    def __getstate__(self) -> dict[str, Any]:
        """Parameterへのweakrefを保存せず、複製先でhookを再登録できる状態を返す。"""
        state = super().__getstate__()
        state.pop("_gradient_hook_parameter", None)
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        """複製・復元されたParameterに独立した勾配制限hookを登録する。"""
        super().__setstate__(state)
        self._ensure_gradient_hook()
