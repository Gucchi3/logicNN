"""module tree内の回路出力状態を、共有objectの重複なしで切り替える。"""

from torch import nn

from .layer_settings import _validate_boolean

__all__ = ["set_export_mode"]


def set_export_mode(module: nn.Module, enabled: bool = True) -> None:
    """model全体を評価状態へ移し、対応する各moduleを一度だけ明示的に切り替える。"""
    if not isinstance(module, nn.Module):
        raise TypeError("module はtorch.nn.Moduleで指定してください")
    _validate_boolean("enabled", enabled)
    modules = tuple(module.modules())
    for child in modules:
        local  = getattr(child, "_set_export_mode_local", None)
        setter = local if callable(local) else getattr(child, "set_export_mode", None)
        if callable(setter):
            setter(enabled)
    module.eval()
