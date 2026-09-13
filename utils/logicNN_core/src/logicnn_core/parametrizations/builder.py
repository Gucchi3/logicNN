"""LUT設定の登録名から、対応するパラメータ化を構築する。"""

from ..layer_settings import LUTConfig
from .base import LUTParametrization
from .light import LightLUTParametrization
from .raw import RawLUTParametrization
from .warp import WarpLUTParametrization

__all__ = ["build_parametrization"]

_PARAMETRIZATIONS: dict[str, type[LUTParametrization]] = {
    "raw": RawLUTParametrization,
    "warp": WarpLUTParametrization,
    "light": LightLUTParametrization,
}


def build_parametrization(config: LUTConfig) -> LUTParametrization:
    """検証済み設定のkindに対応する計算方式を作り、未知の登録名は明示拒否する。"""
    if not isinstance(config, LUTConfig):
        raise TypeError("config はLUTConfigで指定してください")
    constructor = _PARAMETRIZATIONS.get(config.kind)
    if constructor is None:
        raise ValueError(f"未対応のLUTパラメータ化です: {config.kind!r}")
    return constructor(config)
