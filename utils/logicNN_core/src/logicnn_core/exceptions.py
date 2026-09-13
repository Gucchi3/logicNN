"""回路変換・実行の失敗に付随する診断情報を定義する。"""

from __future__ import annotations


class LogicNNCoreError(RuntimeError):
    """coreの処理失敗と、原因・対処方法を保持する。"""

    def __init__(self, message: str, *, detail: str | None = None, hint: str | None = None) -> None:
        """標準の例外メッセージと任意の診断情報を保存する。"""
        super().__init__(message)
        self.message = message
        self.detail  = detail
        self.hint    = hint
