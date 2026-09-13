"""epoch履歴、学習曲線およびbestモデルの最終テスト指標を保存する。"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

from utils.exceptions import LogicNNError
from utils.trainer.epoch import EpochMetrics

from .metadata import _atomic_write, _write_json


@dataclass(frozen=True)
class EpochRecord:
    """1 epochで使用した学習率と学習・検証の平均指標を保持する。"""

    epoch: int
    learning_rate: float
    train_loss: float
    train_accuracy: float
    validation_loss: float
    validation_accuracy: float


def append_epoch_record(run_dir: Path, record: EpochRecord) -> None:
    """1 epochをJSONLの1行として追記し、既存履歴を維持する。"""
    path    = Path(run_dir) / "metrics.jsonl"
    payload = json.dumps(asdict(record), ensure_ascii=False, allow_nan=False) + "\n"
    try:
        with path.open("a", encoding="utf-8", newline="\n") as file:
            file.write(payload)
            file.flush()
    except OSError as error:
        raise LogicNNError("epoch履歴を保存できません", detail=f"{path}: {error}", hint="保存先と書込権限を確認してください") from error


def save_curves(run_dir: Path, history: Sequence[EpochRecord]) -> None:
    """独立したAgg canvasで2列の学習曲線を保存し、プロセスのbackendを変更しない。"""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    epochs                = [record.epoch for record in history]
    train_losses          = [record.train_loss for record in history]
    validation_losses     = [record.validation_loss for record in history]
    validation_accuracies = [100 * record.validation_accuracy for record in history]
    train_label           = "train"
    validation_label      = "validation"
    accuracy_label        = "validation acc"
    if history:
        train_label      += f" (latest: {train_losses[-1]:.4f}, best: {min(train_losses):.4f})"
        validation_label += f" (latest: {validation_losses[-1]:.4f}, best: {min(validation_losses):.4f})"
        accuracy_label   += f" (latest: {validation_accuracies[-1]:.2f}%, best: {max(validation_accuracies):.2f}%)"
    figure = Figure(figsize=(12, 4), layout="tight")
    canvas = FigureCanvasAgg(figure)
    axes   = figure.subplots(1, 2)
    axes[0].plot(epochs, train_losses, label=train_label)
    axes[0].plot(epochs, validation_losses, label=validation_label)
    axes[0].set(xlabel="epoch", ylabel="loss")
    axes[1].plot(epochs, validation_accuracies, label=accuracy_label, color="red")
    axes[1].set(xlabel="epoch", ylabel="accuracy (%)")
    for axis in axes:
        axis.legend()
        axis.grid(True)

    try:
        _atomic_write(Path(run_dir) / "curves.png", canvas.print_png)
    finally:
        figure.clear()


def save_test_metrics(run_dir: Path, epoch: int, metrics: EpochMetrics) -> None:
    """bestチェックポイントのepochと最終テスト指標を固定ファイル名で保存する。"""
    _write_json(Path(run_dir) / "test_metrics.json", {"checkpoint": "model_best.pth", "epoch": epoch, **asdict(metrics)})
