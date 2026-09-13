"""通常学習と分離して登録済みデータセットを取得する。"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable, Sequence
from pathlib import Path

from rich.console import Console

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DownloadFunction = Callable[[Path], tuple[int, int, Path]]


def main(argv: Sequence[str] | None = None) -> int:
    """CLI引数で選択されたデータセットを取得し、保存先と件数をRichで表示する。"""
    arguments = _parse_arguments(argv)
    root      = arguments.root if arguments.root.is_absolute() else PROJECT_ROOT / arguments.root
    try:
        downloader = _downloaders()[arguments.dataset]
    except KeyError:
        Console(stderr=True).print(f"[red]未登録のデータセットです:[/red] {arguments.dataset}")
        return 1

    from utils.exceptions import LogicNNError

    try:
        train_size, test_size, destination = downloader(root.resolve())
    except LogicNNError as error:
        console = Console(stderr=True)
        console.print(f"[red]{error.message}[/red]")
        console.print(error.detail)
        console.print(f"対応: {error.hint}")
        return 1

    Console().print(f"[green]データセットの準備が完了しました[/green]\n保存先: {destination}\n学習用: {train_size:,} 件\nテスト用: {test_size:,} 件")
    return 0


def _parse_arguments(argv: Sequence[str] | None) -> argparse.Namespace:
    """データセット名とプロジェクト基準の保存先をCLIから読み取る。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="mnist", help="取得するデータセットの登録名。既定値はmnist。")
    parser.add_argument("--root", type=Path, default=Path("data"), help="データの保存先。既定値はプロジェクト内のdata。")
    return parser.parse_args(argv)


def _downloaders() -> dict[str, DownloadFunction]:
    """データセット登録名と取得処理の対応表を返す。"""
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from utils.data.mnist import download_mnist

    return {"mnist": download_mnist}


if __name__ == "__main__":
    raise SystemExit(main())
