"""logicNNで利用者へ通知する想定済みエラーを定義する。"""

from __future__ import annotations


class LogicNNError(Exception):
    """利用者が修正できる想定済みエラーを表す。"""

    def __init__(self, message: str, *, detail: str, hint: str) -> None:
        """失敗内容、詳細および修正方法を保持する。"""
        super().__init__(message)
        self.message = message
        self.detail  = detail
        self.hint    = hint

