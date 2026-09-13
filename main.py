"""設定JSONを選び、論理ゲートNNの学習workflowを開始するCLIを提供する。"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from utils.config.loader import load_config
from utils.console import show_error, show_interrupted
from utils.exceptions import LogicNNError
from utils.trainer.workflow import run_training


def main(argv: Sequence[str] | None = None) -> int:
    """CLI設定を読んで学習を一度実行し、既知エラーと利用者中断だけを通知する。"""
    parser = argparse.ArgumentParser(description="論理ゲートNNを学習し、設定に応じて回路を出力します。", allow_abbrev=False)
    parser.add_argument("--config", default="config/mnist_lgn.json", metavar="PATH", help="設定JSONのパス（既定: config/mnist_lgn.json）")
    args = parser.parse_args(argv)
    try:
        config_path = Path(args.config).expanduser().resolve()
        config      = load_config(config_path)
        run_training(config, config_path)
    except LogicNNError as error:
        show_error(error)
        return 1
    except KeyboardInterrupt:
        show_interrupted()
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
