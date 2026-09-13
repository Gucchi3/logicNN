"""学習状態を変更せず、実行条件・epoch指標・結果・エラーをRichで表示する。"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from rich.console import Console
from rich.table import Table
from rich.text import Text

if TYPE_CHECKING:
    import torch

    from utils.config.schema import AppConfig
    from utils.exceptions import LogicNNError
    from utils.results.metrics import EpochRecord
    from utils.trainer.workflow import TrainingResult

_console       = Console()
_error_console = Console(stderr=True)


def _summary(title: str, rows: list[tuple[str, str]]) -> None:
    """値をRich記法として解釈しない2列表を一度だけ表示する。"""
    table = Table(title=title, show_header=False, box=None, padding=(0, 1))
    table.add_column(style="cyan", no_wrap=True)
    table.add_column(overflow="fold")
    for label, value in rows:
        table.add_row(Text(label), Text(value))
    _console.print(table)


def show_start(config: AppConfig, run_dir: Path, device: torch.device) -> None:
    """現在の表示項目を、旧版と同じ区分付きの枠線表で学習開始時に表示する。"""
    checkpoint = str(config.run.initial_checkpoint_path) if config.run.initial_checkpoint_path is not None else "なし（新規重み）"
    table      = Table(title="学習開始", show_header=True, header_style="bold")
    table.add_column("Section", style="bold cyan", no_wrap=True)
    table.add_column("Key", style="green", no_wrap=True)
    table.add_column("Value", style="white", overflow="fold")
    sections = (
        ("Environment", [("device", str(device)), ("seed", str(config.run.seed))]),
        ("Model", [("モデル", config.model.name), ("初期重み", checkpoint)]),
        ("Data", [("データセット", config.data.name)]),
        ("Training", [
            ("batch size", str(config.data.batch_size)),
            ("epochs", str(config.train.epochs)),
            ("loss", f"{config.train.loss.name} (label_smoothing={config.train.loss.label_smoothing:g})"),
            ("optimizer", f"{config.train.optimizer.name} (lr={config.train.optimizer.learning_rate:g})"),
            ("scheduler", config.train.scheduler.name),
        ]),
        ("Output", [("回路出力", "有効" if config.circuit.enabled else "無効"), ("実行ディレクトリ", str(run_dir))]),
    )
    for section, rows in sections:
        for label, value in rows:
            table.add_row(Text(section), Text(label), Text(value))
        if section != "Output":
            table.add_row("", "", "")
    _console.print()
    _console.print(table)
    _console.print()


def show_epoch(record: EpochRecord, epochs: int) -> None:
    """学習率と損失を小数点以下3桁に揃え、旧版と同じ項目別の色で表示する。"""
    message = Text()
    message.append("Epoch", style="bold cyan")
    message.append(f" {record.epoch}/{epochs}", style="cyan")
    message.append(" | ")
    message.append("lr", style="bold magenta")
    message.append(f"={record.learning_rate:.3f}", style="magenta")
    message.append(" | ")
    message.append("train loss", style="bold yellow")
    message.append(f"={record.train_loss:.3f} ", style="yellow")
    message.append("accuracy", style="bold yellow")
    message.append(f"={record.train_accuracy:.2%}", style="yellow")
    message.append(" | ")
    message.append("validation loss", style="bold blue")
    message.append(f"={record.validation_loss:.3f} ", style="blue")
    message.append("accuracy", style="bold red")
    message.append(f"={record.validation_accuracy:.2%}", style="red")
    _console.print(message)


def show_complete(result: TrainingResult) -> None:
    """best epoch、検証精度、公式テスト指標と保存先を完了時にまとめて表示する。"""
    _summary("学習完了", [
        ("best epoch", str(result.best_epoch)),
        ("best validation accuracy", f"{result.best_validation_accuracy:.2%}"),
        ("test loss", f"{result.test_metrics.loss:.6g}"),
        ("test accuracy", f"{result.test_metrics.accuracy:.2%}"),
        ("test samples", str(result.test_metrics.samples)),
        ("保存先", str(result.run_dir)),
    ])


def show_error(error: LogicNNError) -> None:
    """想定済みエラーのmessage、detailとhintをtracebackなしで標準エラーへ表示する。"""
    _error_console.print(Text(f"[ERROR] {error.message}", style="bold red"))
    _error_console.print(Text(error.detail))
    _error_console.print(Text(f"対応: {error.hint}"))


def show_interrupted() -> None:
    """利用者による中断を一行で標準エラーへ通知する。"""
    _error_console.print(Text("[中断] 利用者の操作により実行を中断しました。", style="yellow"))
