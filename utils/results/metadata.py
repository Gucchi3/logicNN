"""成果物用metadataの基本型検証と、安全なJSON保存をまとめる。"""

from __future__ import annotations

import json
import math
import os
import tempfile
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import BinaryIO, NoReturn

from utils.config.schema import AppConfig
from utils.exceptions import LogicNNError


def _fail(target: str, message: str) -> NoReturn:
    """保存内容の不正を対象fieldと修正方針を持つ共通例外にする。"""
    raise LogicNNError("成果物の形式が不正です", detail=f"{target}: {message}", hint="仕様に従うmetadataの型・必須項目・有限値を確認してください")


def _json_value(value: object, target: str = "metadata") -> object:
    """基本型だけを再帰的に複製し、tupleをJSON arrayへ正規化する。"""
    if value is None or type(value) in (str, bool, int):
        return value
    if type(value) is float:
        if not math.isfinite(value):
            _fail(target, "NaN・Infinityを保存できません")
        return value
    if isinstance(value, Mapping):
        if any(type(key) is not str for key in value):
            _fail(target, "辞書のkeyはstrが必要です")
        return {key: _json_value(item, f"{target}.{key}") for key, item in value.items()}
    if type(value) in (list, tuple):
        return [_json_value(item, f"{target}[{index}]") for index, item in enumerate(value)]
    _fail(target, f"基本Python型ではありません: {type(value).__name__}")


def _object(value: object, fields: set[str], target: str) -> dict[str, object]:
    """必須キーと未知キーを検証し、objectを辞書として返す。"""
    if type(value) is not dict:
        _fail(target, f"必要なキーは{sorted(fields)}です")
    if set(value) != fields:
        missing = sorted(fields - set(value))
        unknown = sorted(repr(key) for key in set(value) - fields)
        _fail(target, f"必須field欠落={missing}、未知field={unknown}")
    return value


def _text(value: object, target: str) -> None:
    """空でない通常文字列であることを確認する。"""
    if type(value) is not str or not value.strip():
        _fail(target, "空でないstrが必要です")


def _integer(value: object, target: str, minimum: int = 0) -> None:
    """boolを除外し、下限を満たすPython整数であることを確認する。"""
    if type(value) is not int or value < minimum:
        _fail(target, f"{minimum}以上のPython intが必要です")


def _float(value: object, target: str, *, probability: bool = False) -> None:
    """有限なPython floatと、必要なら0～1の範囲を確認する。"""
    if type(value) is not float or not math.isfinite(value) or (probability and not 0 <= value <= 1):
        _fail(target, "0～1の有限floatが必要です" if probability else "有限なPython floatが必要です")


def _shape(value: object, target: str) -> None:
    """batchを除く形状が非空の正整数listであることを確認する。"""
    if type(value) is not list or not value:
        _fail(target, "非空の正整数listが必要です")
    for index, size in enumerate(value):
        _integer(size, f"{target}[{index}]", 1)


def _model_metadata(value: object) -> dict[str, object]:
    """モデル名・形状・構造の保存形式を検証する。"""
    value = _object(value, {"name", "input_shape", "input_size", "num_classes", "architecture"}, "model")
    _text(value["name"], "model.name")
    _shape(value["input_shape"], "model.input_shape")
    _integer(value["input_size"], "model.input_size", 1)
    _integer(value["num_classes"], "model.num_classes", 1)
    if value["input_size"] != math.prod(value["input_shape"]):
        _fail("model.input_size", "input_shapeの積と一致しません")
    if type(value["architecture"]) is not dict:
        _fail("model.architecture", "構造情報のdictが必要です")
    _json_value(value, "model")
    return value


def _dataset_metadata(value: object) -> dict[str, object]:
    """特定データセットに依存せず、全DatasetMetadata fieldを検証する。"""
    fields = {"name", "input_shape", "num_classes", "class_names", "train_size", "validation_size", "test_size", "split_seed", "preprocessing"}
    value  = _object(value, fields, "dataset")
    _text(value["name"], "dataset.name")
    _shape(value["input_shape"], "dataset.input_shape")
    _integer(value["num_classes"], "dataset.num_classes", 1)
    names = value["class_names"]
    if type(names) is not list or len(names) != value["num_classes"]:
        _fail("dataset.class_names", "num_classesと同数のlistが必要です")
    for index, name in enumerate(names):
        _text(name, f"dataset.class_names[{index}]")
    for field in ("train_size", "validation_size", "test_size"):
        _integer(value[field], f"dataset.{field}")
    if value["split_seed"] is not None:
        _integer(value["split_seed"], "dataset.split_seed")
    preprocessing = _object(value["preprocessing"], {"name", "parameters", "output_dtype"}, "dataset.preprocessing")
    _text(preprocessing["name"], "dataset.preprocessing.name")
    _text(preprocessing["output_dtype"], "dataset.preprocessing.output_dtype")
    if type(preprocessing["parameters"]) is not dict:
        _fail("dataset.preprocessing.parameters", "前処理設定のdictが必要です")
    _json_value(preprocessing, "dataset.preprocessing")
    return value


def _training_metadata(value: object, target: str = "training") -> dict[str, object]:
    """学習回数と各最適化方式の名前・実設定を検証する。"""
    value = _object(value, {"epochs", "loss", "optimizer", "scheduler"}, target)
    _integer(value["epochs"], f"{target}.epochs", 1)
    for field in ("loss", "optimizer", "scheduler"):
        settings = value[field]
        if type(settings) is not dict or "name" not in settings:
            _fail(f"{target}.{field}", "nameと実際の設定値を含むdictが必要です")
        _text(settings["name"], f"{target}.{field}.name")
        _json_value(settings, f"{target}.{field}")
    return value


def _metric_values(value: object, target: str) -> dict[str, object]:
    """EpochMetricsの保存形式と有限値・正解率・件数を検証する。"""
    value = _object(value, {"loss", "accuracy", "samples"}, target)
    _float(value["loss"], f"{target}.loss")
    _float(value["accuracy"], f"{target}.accuracy", probability=True)
    _integer(value["samples"], f"{target}.samples", 1)
    return value


def _atomic_write(path: Path, write: Callable[[BinaryIO], None]) -> None:
    """同じdirectoryの一時fileへ完成させ、成功時だけ既存成果物を置換する。"""
    path      = Path(path)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile("w+b", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False) as file:
            temporary = file.name
            write(file)
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary, path)
        temporary = None
    except (OSError, ValueError, TypeError, RuntimeError) as error:
        raise LogicNNError("成果物を保存できません", detail=f"{path}: {error}", hint="保存先・空き容量・書込権限を確認してください") from error
    finally:
        if temporary is not None:
            try:
                os.unlink(temporary)
            except OSError:
                pass


def _write_json(path: Path, value: object) -> None:
    """JSON化を完了してからUTF-8で原子的に保存する。"""
    try:
        payload = (json.dumps(_json_value(value), ensure_ascii=False, allow_nan=False, indent=2) + "\n").encode("utf-8")
    except (UnicodeError, ValueError, RecursionError) as error:
        raise LogicNNError("成果物をJSONへ変換できません", detail=f"{path}: {error}", hint="有限値と循環しないJSON互換metadataを指定してください") from error

    def write(file: BinaryIO) -> None:
        """作成済みのJSON byte列を一時fileへ書く。"""
        file.write(payload)

    _atomic_write(path, write)


def save_resolved_config(run_dir: Path, config: AppConfig) -> None:
    """解決済みAppConfig全体をconfig.jsonへ保存する。"""
    _write_json(Path(run_dir) / "config.json", config.model_dump(mode="json"))


def save_training_info(run_dir: Path, information: Mapping[str, object]) -> None:
    """workflowが組み立てた実行情報を、JSON化とファイル保存だけ行う。"""
    _write_json(Path(run_dir) / "training_info.json", information)
