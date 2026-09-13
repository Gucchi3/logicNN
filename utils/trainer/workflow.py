"""学習・検証・成果物・最良モデルの最終評価を、仕様の順序で調整する。"""

from __future__ import annotations

import json
import math
import platform
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import logicnn_core
import torch
import torchvision
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR

from model.builder import MODEL_REGISTRY, build_model
from utils.config.schema import AppConfig
from utils.console import show_complete, show_epoch, show_start
from utils.data.loader import DATASET_REGISTRY, build_data_bundle
from utils.data.types import DatasetMetadata
from utils.exceptions import LogicNNError
from utils.export import export_circuit
from utils.results import (
    EpochRecord,
    append_epoch_record,
    create_run_directory,
    load_checkpoint,
    save_checkpoint,
    save_curves,
    save_resolved_config,
    save_test_metrics,
    save_training_info,
)
from utils.runtime.device import select_device
from utils.runtime.seed import set_seed

from .epoch import EpochMetrics, evaluate, train_one_epoch
from .optimization.loss import LOSS_REGISTRY, build_evaluation_loss, build_training_loss
from .optimization.optimizer import OPTIMIZER_REGISTRY, build_optimizer
from .optimization.scheduler import SCHEDULER_REGISTRY, build_scheduler


@dataclass(frozen=True)
class TrainingResult:
    """今回の保存先と、保存済みbestによる最終評価結果を返す。"""

    run_dir: Path
    best_epoch: int
    best_validation_accuracy: float
    test_metrics: EpochMetrics


def _validate_registered_names(config: AppConfig) -> None:
    """データやmodelの生成・保存先作成より前に、全登録名を検証する。"""
    selections = (
        ("data.name", config.data.name, DATASET_REGISTRY), ("model.name", config.model.name, MODEL_REGISTRY),
        ("train.loss.name", config.train.loss.name, LOSS_REGISTRY), ("train.optimizer.name", config.train.optimizer.name, OPTIMIZER_REGISTRY),
        ("train.scheduler.name", config.train.scheduler.name, SCHEDULER_REGISTRY),
    )
    for field, name, registry in selections:
        if name not in registry:
            available = ", ".join(sorted(registry))
            raise LogicNNError("未登録の名前です", detail=f"{field}: {name!r}\n登録済み: {available}", hint="対応する登録名を設定してください")


def _model_information(model: nn.Module, name: str, dataset: DatasetMetadata) -> dict[str, object]:
    """モデル共通契約とデータ適合性を確認し、保存する構造情報を独立dictへ変換する。"""
    shape   = getattr(model, "input_shape", None)
    size    = getattr(model, "input_size", None)
    classes = getattr(model, "num_classes", None)
    describe = getattr(model, "architecture_info", None)
    if type(shape) is not tuple or not shape or any(type(value) is not int or value <= 0 for value in shape):
        raise LogicNNError("モデルの入力形状が不正です", detail=f"model={name}, input_shape={shape!r}", hint="モデル定義に正整数tupleのinput_shapeを定義してください")
    if type(size) is not int or size != math.prod(shape) or type(classes) is not int or classes <= 0 or not callable(describe):
        raise LogicNNError("モデル共通契約が不正です", detail=f"model={name}, input_size={size!r}, num_classes={classes!r}", hint="モデルの共通propertyと構造情報関数を確認してください")
    for field, expected, actual in (("input_shape", dataset.input_shape, shape), ("num_classes", dataset.num_classes, classes)):
        if actual != expected:
            raise LogicNNError(
                "モデルとデータセットが適合しません", detail=f"{field}: dataset={expected!r}, model={actual!r}",
                hint="入力形状とクラス数が一致するモデルとデータセットを選択してください",
            )
    architecture = describe()
    if type(architecture) is not dict:
        raise LogicNNError("モデル構造情報が不正です", detail=f"model={name}: architecture_infoはdictが必要です", hint="JSONで保存できる構造情報を返してください")
    try:
        architecture = json.loads(json.dumps(architecture, allow_nan=False))
    except (ValueError, TypeError, RecursionError) as error:
        raise LogicNNError("モデル構造情報を保存できません", detail=f"model={name}: {error}", hint="循環やTensorを含まない有限のJSON基本型を返してください") from error
    return {"name": name, "input_shape": list(shape), "input_size": size, "num_classes": classes, "architecture": architecture}


def _restore_model(model: nn.Module, path: Path, information: Mapping[str, object], dataset: DatasetMetadata) -> tuple[nn.Module, dict[str, object]]:
    """metadataを照合し、strict読込に成功した独立コピーだけを採用する。"""
    checkpoint = load_checkpoint(path)
    expected_dataset = json.loads(json.dumps(asdict(dataset)))
    comparisons = [(f"model.{field}", expected, checkpoint["model"][field]) for field, expected in information.items()]
    comparisons.extend(
        (f"dataset.{field}", expected_dataset[field], checkpoint["dataset"][field]) for field in ("name", "input_shape", "num_classes", "preprocessing")
    )
    for field, expected, actual in comparisons:
        if expected != actual:
            raise LogicNNError(
                "チェックポイントが実行対象と一致しません", detail=f"対象: {path}\n{field}\n期待値: {expected!r}\n実際値: {actual!r}",
                hint="同じモデル構造と前処理のlogicNNチェックポイントを指定してください",
            )
    candidate = deepcopy(model)
    try:
        candidate.load_state_dict(checkpoint["model_state_dict"], strict=True)
    except (RuntimeError, ValueError) as error:
        raise LogicNNError("チェックポイントの重みが適合しません", detail=f"対象: {path}\n{error}", hint="重みのキーと形状がモデル定義と一致するpthを指定してください") from error
    return candidate, checkpoint


def _training_information(config: AppConfig, loss: nn.CrossEntropyLoss, optimizer: Optimizer, scheduler: CosineAnnealingLR) -> dict[str, object]:
    """明示設定とライブラリ既定値を含む、実際の最適化条件だけを記録する。"""
    return {
        "epochs": config.train.epochs,
        "loss": {"name": config.train.loss.name, "label_smoothing": loss.label_smoothing, "reduction": loss.reduction, "ignore_index": loss.ignore_index},
        "optimizer": {"name": config.train.optimizer.name, **deepcopy(optimizer.defaults)},
        "scheduler": {"name": config.train.scheduler.name, "T_max": scheduler.T_max, "eta_min": scheduler.eta_min},
    }


def run_training(config: AppConfig, config_path: Path) -> TrainingResult:
    """新規学習をepoch 1から行い、保存したbestの公式テストと任意回路出力まで実行する。"""
    started = datetime.now().astimezone().isoformat()
    device  = select_device(config.run.device)
    _validate_registered_names(config)
    version = getattr(logicnn_core, "__version__", None)
    if type(version) is not str or not version.strip():
        raise LogicNNError("logicNN_coreの版情報がありません", detail="配布物の__version__を取得できません", hint="ローカルcore packageの導入状態を確認してください")
    set_seed(config.run.seed)
    run_dir = create_run_directory(config.run.log_dir)
    save_resolved_config(run_dir, config)
    bundle  = build_data_bundle(config.data, config.run.seed, device.type == "cuda")
    model   = build_model(config.model.name)
    model_information = _model_information(model, config.model.name, bundle.metadata)
    initialized = None
    if config.run.initial_checkpoint_path is not None:
        path = config.run.initial_checkpoint_path
        model, checkpoint = _restore_model(model, path, model_information, bundle.metadata)
        initialized = {
            "path": str(path.resolve()), "format_version": checkpoint["format_version"],
            "checkpoint_type": checkpoint["checkpoint_type"], "epoch": checkpoint["epoch"],
        }
    model      = model.to(device)
    train_loss = build_training_loss(config.train.loss).to(device)
    eval_loss  = build_evaluation_loss(config.train.loss).to(device)
    optimizer  = build_optimizer(config.train.optimizer, model.parameters())
    scheduler  = build_scheduler(config.train.scheduler, optimizer, config.train.epochs)
    training   = _training_information(config, train_loss, optimizer, scheduler)
    information = {
        "started_at": started, "config_path": str(Path(config_path).resolve()), "device": {"requested": config.run.device, "resolved": str(device)},
        "software": {
            "python": platform.python_version(), "pytorch": str(torch.__version__), "torchvision": str(torchvision.__version__), "logicnn_core": version,
        },
        "model": model_information, "dataset": asdict(bundle.metadata), "optimization": {name: training[name] for name in ("loss", "optimizer", "scheduler")},
        "evaluation_modes": {"train": "continuous", "validation": "discrete", "test": "discrete", "circuit": "discrete"},
        "initialized_from_checkpoint": initialized,
    }
    save_training_info(run_dir, information)
    show_start(config, run_dir, device)
    history: list[EpochRecord] = []
    best_validation_accuracy = -1.0
    best_epoch: int | None = None
    for epoch in range(1, config.train.epochs + 1):
        learning_rate = float(optimizer.param_groups[0]["lr"])
        train_metrics = train_one_epoch(model, bundle.train_loader, train_loss, optimizer, device)
        validation    = evaluate(model, bundle.validation_loader, eval_loss, device)
        if validation.accuracy > best_validation_accuracy:
            save_checkpoint(
                run_dir / "model_best.pth", model, checkpoint_type="best", epoch=epoch, model_metadata=model_information,
                dataset_metadata=bundle.metadata, training_metadata=training, validation_metrics=validation,
            )
            best_validation_accuracy = validation.accuracy
            best_epoch = epoch
        record = EpochRecord(epoch, learning_rate, train_metrics.loss, train_metrics.accuracy, validation.loss, validation.accuracy)
        append_epoch_record(run_dir, record)
        history.append(record)
        save_curves(run_dir, history)
        show_epoch(record, config.train.epochs)
        scheduler.step()
    save_checkpoint(
        run_dir / "model_final.pth", model, checkpoint_type="final", epoch=config.train.epochs, model_metadata=model_information,
        dataset_metadata=bundle.metadata, training_metadata=training, validation_metrics=validation,
    )
    assert best_epoch is not None
    model, _ = _restore_model(model, run_dir / "model_best.pth", model_information, bundle.metadata)
    test_metrics = evaluate(model, bundle.test_loader, eval_loss, device)
    save_test_metrics(run_dir, best_epoch, test_metrics)
    if config.circuit.enabled:
        export_circuit(model, bundle.metadata.input_shape, run_dir)
    result = TrainingResult(run_dir, best_epoch, best_validation_accuracy, test_metrics)
    show_complete(result)
    return result
