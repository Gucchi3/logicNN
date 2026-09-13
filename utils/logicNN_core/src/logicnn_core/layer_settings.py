"""モデル定義内で使う不変なLUT設定と接続設定を検証する。"""

from __future__ import annotations

import math
from dataclasses import dataclass

__all__ = ["ConnectionConfig", "LUTConfig"]

_LUT_INPUT_COUNTS = {"raw": (2,), "warp": (1, 2, 4, 6), "light": (2, 4, 6)}
_SAMPLING_MODES   = ("soft", "hard", "gumbel_soft", "gumbel_hard")
_INITIALIZATIONS  = ("residual", "residual_catalog", "random")


def _validate_choice(name: str, value: object, choices: tuple[str, ...]) -> None:
    """登録名の型と許可された文字列を確認する。"""
    if not isinstance(value, str):
        raise TypeError(f"{name} は文字列で指定してください")
    if value not in choices:
        raise ValueError(f"{name} は {choices} から指定してください: {value!r}")


def _validate_positive_integer(name: str, value: object) -> None:
    """boolや暗黙の型変換を許さず正の整数を確認する。"""
    if type(value) is not int:
        raise TypeError(f"{name} はboolではない整数で指定してください")
    if value <= 0:
        raise ValueError(f"{name} は0より大きい整数で指定してください")


def _validate_finite_number(name: str, value: object) -> None:
    """intまたはfloatで表現される有限の実数を確認する。"""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} はboolではないintまたはfloatで指定してください")
    try:
        finite = math.isfinite(value)
    except OverflowError:
        raise ValueError(f"{name} は浮動小数点で扱える有限値で指定してください") from None
    if not finite:
        raise ValueError(f"{name} は有限値で指定してください")


def _validate_positive_number(name: str, value: object) -> None:
    """温度などの設定値が正の有限数であることを確認する。"""
    _validate_finite_number(name, value)
    if value <= 0:
        raise ValueError(f"{name} は0より大きい値で指定してください")


def _validate_boolean(name: str, value: object) -> None:
    """真偽値を数値や文字列から暗黙変換せず確認する。"""
    if type(value) is not bool:
        raise TypeError(f"{name} はboolで指定してください")


@dataclass(frozen=True, slots=True)
class LUTConfig:
    """ゲートの入力本数、学習表現、初期化方法を指定する。"""

    kind: str                   = "raw"
    num_inputs: int              = 2
    sampling: str               = "soft"
    temperature: float          = 1.0
    weight_init: str            = "residual"
    residual_probability: float = 0.951
    materialize_basis: bool     = False

    def __post_init__(self) -> None:
        """LUT方式別の対応範囲と設定値の型・数値制約を確認する。"""
        _validate_choice("kind", self.kind, tuple(_LUT_INPUT_COUNTS))
        _validate_positive_integer("num_inputs", self.num_inputs)
        if self.num_inputs not in _LUT_INPUT_COUNTS[self.kind]:
            raise ValueError(f"{self.kind} のnum_inputsは {_LUT_INPUT_COUNTS[self.kind]} に対応しています")

        _validate_choice("sampling", self.sampling, _SAMPLING_MODES)
        _validate_choice("weight_init", self.weight_init, _INITIALIZATIONS)
        if self.weight_init == "residual_catalog" and self.kind != "warp":
            raise ValueError("weight_init='residual_catalog' はkind='warp'でのみ使用できます")

        _validate_positive_number("temperature", self.temperature)
        _validate_finite_number("residual_probability", self.residual_probability)
        if not 0 < self.residual_probability < 1:
            raise ValueError("residual_probability は0より大きく1より小さい値で指定してください")
        if self.kind == "raw" and self.weight_init == "residual" and self.residual_probability <= 7 / 15:
            raise ValueError("Rawのresidual初期化では 7/15 < residual_probability < 1 が必要です")
        _validate_boolean("materialize_basis", self.materialize_basis)


@dataclass(frozen=True, slots=True)
class ConnectionConfig:
    """固定・学習可能接続と、候補数やchannel群の制約を指定する。"""

    kind: str                      = "fixed"
    init: str                      = "random"
    temperature: float             = 1.0
    use_gumbel: bool                = False
    num_candidates: int | None     = None
    channel_group_size: int | None = None

    def __post_init__(self) -> None:
        """接続方式ごとの設定を検証し、shape依存の検証は接続構築へ委ねる。"""
        _validate_choice("kind", self.kind, ("fixed", "learnable"))
        _validate_choice("init", self.init, ("random", "random_unique"))
        _validate_positive_number("temperature", self.temperature)
        _validate_boolean("use_gumbel", self.use_gumbel)
        if self.num_candidates is not None:
            _validate_positive_integer("num_candidates", self.num_candidates)
        if self.channel_group_size is not None:
            _validate_positive_integer("channel_group_size", self.channel_group_size)

        if self.kind == "fixed":
            if self.temperature != 1.0 or self.use_gumbel or self.num_candidates is not None:
                raise ValueError("固定接続ではtemperature、use_gumbel、num_candidatesを既定値から変更できません")
        else:
            if self.channel_group_size is not None:
                raise ValueError("channel_group_size は固定Conv接続でのみ使用できます")
            if self.init == "random_unique" and self.num_candidates is None:
                raise ValueError("学習可能なrandom_unique接続ではnum_candidatesを正の整数で明示してください")
