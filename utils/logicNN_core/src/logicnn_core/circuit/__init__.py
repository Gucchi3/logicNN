"""論理回路と外部FXの出力信号参照を公開する。"""

from typing import TYPE_CHECKING, Any

from .types import FxSignalReference

if TYPE_CHECKING:
    from .circuit import Circuit

__all__ = ["Circuit", "FxSignalReference"]


def __getattr__(name: str) -> Any:
    """型だけのimportにPyTorchを要求せず、必要になった回路入口を返す。"""
    if name == "Circuit":
        from .circuit import Circuit

        globals()[name] = Circuit
        return Circuit
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
