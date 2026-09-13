"""生成済みの回路sourceを既存fileを保護して保存する。"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

from ...exceptions import LogicNNCoreError


def write_source(source: str, path: str | Path) -> None:
    """同じdirectoryの一時fileを置換し、失敗時は既存sourceを変更しない。"""
    temporary = None
    try:
        destination = Path(path)
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", newline="\n", dir=destination.parent, prefix=f".{destination.name}.", suffix=".tmp", delete=False,
        ) as file:
            temporary = file.name
            file.write(source)
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary, destination)
        temporary = None
    except (OSError, UnicodeError, ValueError, TypeError) as error:
        raise LogicNNCoreError(
            f"回路sourceの保存に失敗しました: {path}", detail=f"{type(error).__name__}: {error}",
            hint="親directoryと書込み権限を確認してください。既存fileは置換成功前には変更されません",
        ) from error
    finally:
        if temporary is not None:
            try:
                os.unlink(temporary)
            except OSError:
                pass
