"""汎用論理ゲートNNライブラリの実装済み公開APIを提供する。"""

from typing import TYPE_CHECKING, Any

from .exceptions import LogicNNCoreError

if TYPE_CHECKING:
    from .circuit import Circuit
    from .export_mode import set_export_mode

__version__ = "0.1.0"
__all__     = ["Circuit", "LogicNNCoreError", "set_export_mode"]


def __getattr__(name: str) -> Any:
    """回路と共通切替を遅延公開し、設定のみのimportを軽量に保つ。"""
    if name == "Circuit":
        from .circuit import Circuit

        globals()[name] = Circuit
        return Circuit
    if name == "set_export_mode":
        from .export_mode import set_export_mode

        globals()[name] = set_export_mode
        return set_export_mode
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
