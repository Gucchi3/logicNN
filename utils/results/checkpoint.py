"""重みだけを次の学習の初期値として使う、logicNN形式のcheckpointを保存・読込する。"""

from __future__ import annotations

import pickle
from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path
from typing import BinaryIO, Literal

import torch
from torch import Tensor, nn

from utils.data.types import DatasetMetadata
from utils.exceptions import LogicNNError
from utils.trainer.epoch import EpochMetrics

from .metadata import _atomic_write, _dataset_metadata, _fail, _integer, _json_value, _metric_values, _model_metadata, _object, _training_metadata

_FIELDS = {"format_name", "format_version", "checkpoint_type", "epoch", "model_state_dict", "model", "dataset", "training", "validation_metrics"}


def _validate_checkpoint(value: object) -> dict[str, object]:
    """checkpointの全必須項目・基本型・Tensor領域を重み適用前に検証する。"""
    value = _object(value, _FIELDS, "checkpoint")
    if type(value["format_name"]) is not str or value["format_name"] != "logicNN_checkpoint":
        _fail("format_name", "logicNN_checkpoint形式だけを読み込めます")
    if type(value["format_version"]) is not int or value["format_version"] != 1:
        _fail("format_version", "対応する形式版は整数1です")
    if type(value["checkpoint_type"]) is not str or value["checkpoint_type"] not in ("best", "final"):
        _fail("checkpoint_type", "bestまたはfinalが必要です")
    _integer(value["epoch"], "epoch", 1)
    state = value["model_state_dict"]
    if not isinstance(state, Mapping) or any(type(key) is not str or not isinstance(item, Tensor) for key, item in state.items()):
        _fail("model_state_dict", "strからTensorへの辞書が必要です")
    if any(tensor.layout is not torch.strided or tensor.device.type != "cpu" for tensor in state.values()):
        _fail("model_state_dict", "CPU上のstrided Tensorが必要です")
    _model_metadata(value["model"])
    _dataset_metadata(value["dataset"])
    _training_metadata(value["training"])
    _metric_values(value["validation_metrics"], "validation_metrics")
    return value


def save_checkpoint(
    path: Path,
    model: nn.Module,
    *,
    checkpoint_type: Literal["best", "final"],
    epoch: int,
    model_metadata: Mapping[str, object],
    dataset_metadata: DatasetMetadata,
    training_metadata: Mapping[str, object],
    validation_metrics: EpochMetrics,
) -> None:
    """元modelを変更せずCPU重みのsnapshotとmetadataだけを原子的に保存する。"""
    path  = Path(path)
    state = model.state_dict()
    if any(not isinstance(value, Tensor) for value in state.values()):
        _fail("model_state_dict", "Tensor以外のextra stateは保存対象ではありません")
    snapshot = {key: value.detach().to(device="cpu", copy=True) for key, value in state.items()}
    payload = {
        "format_name": "logicNN_checkpoint", "format_version": 1, "checkpoint_type": checkpoint_type, "epoch": epoch,
        "model_state_dict": snapshot, "model": _json_value(model_metadata), "dataset": _json_value(asdict(dataset_metadata)),
        "training": _json_value(training_metadata), "validation_metrics": asdict(validation_metrics),
    }
    _integer(epoch, "epoch", 1)
    if checkpoint_type not in ("best", "final"):
        _fail("checkpoint_type", "bestまたはfinalが必要です")

    def write(file: BinaryIO) -> None:
        """検証済みpayloadをoptimizer等の実行状態を含めず保存する。"""
        torch.save(payload, file)

    _atomic_write(path, write)


def load_checkpoint(path: Path) -> dict[str, object]:
    """安全なweights-only読込と形式検証を行い、モデルへ未適用のpayloadを返す。"""
    path = Path(path)
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except (OSError, RuntimeError, ValueError, EOFError, pickle.UnpicklingError) as error:
        raise LogicNNError("チェックポイントを読み込めません", detail=f"{path}: {error}", hint="logicNN形式の破損していないpthファイルを指定してください") from error
    try:
        return _validate_checkpoint(payload)
    except LogicNNError as error:
        raise LogicNNError("チェックポイントの形式が不正です", detail=f"{path}: {error.detail}", hint=error.hint) from error
    except RecursionError as error:
        raise LogicNNError("チェックポイントの形式が不正です", detail=f"{path}: metadataに循環または深過ぎる入れ子があります",
                           hint="循環しないJSON互換metadataを持つcheckpointを指定してください") from error
