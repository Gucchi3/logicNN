"""元モデルを保護し、簡略化した回路JSON・C・Verilogを順に直接保存する。"""

from __future__ import annotations

import json
from collections.abc import Sequence
from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING

import logicnn_core

from utils.exceptions import LogicNNError

if TYPE_CHECKING:
    from torch import nn


def _export_error(stage: str, path: Path, error: Exception) -> LogicNNError:
    """失敗段階と保存先にcore固有の原因・対処方法を付けて境界の診断へ変換する。"""
    detail = f"段階: {stage}\n保存先: {path}\n原因: {type(error).__name__}: {error}"
    hint   = "モデルの対応演算と保存先・書込権限を確認してください。完了済みの学習・回路成果物は残しています。"
    if isinstance(error, logicnn_core.LogicNNCoreError):
        if error.detail:
            detail += f"\n{error.detail}"
        if error.hint:
            hint = f"{error.hint}\n完了済みの学習・回路成果物は残しています。"
    return LogicNNError("学習後の回路出力に失敗しました", detail=detail, hint=hint)


def export_circuit(model: nn.Module, input_shape: Sequence[int], run_dir: Path) -> None:
    """CPU評価コピーから3形式を順次保存し、失敗しても完成済み成果物を保持する。"""
    directory = Path(run_dir) / "circuit"
    target    = directory
    stage     = "モデルコピー"
    try:
        copied = deepcopy(model)
        stage  = "CPU転送"
        copied.cpu()
        stage = "評価モード切替"
        copied.eval()
        stage   = "回路変換"
        circuit = logicnn_core.Circuit.from_model(copied, input_shape=list(input_shape))
        stage   = "回路簡略化"
        circuit.simplify()
        stage = "保存ディレクトリ作成"
        directory.mkdir(exist_ok=True)

        stage   = "JSON生成・保存"
        target  = directory / "circuit_simplified.json"
        payload = circuit.to_dict()
        try:
            text = json.dumps(payload, ensure_ascii=False, allow_nan=False, indent=2) + "\n"
        except (TypeError, ValueError) as error:
            raise _export_error(stage, target, error) from error
        target.write_text(text, encoding="utf-8", newline="\n")

        stage  = "C生成・保存"
        target = directory / "circuit.c"
        target.write_text(circuit.c_source(), encoding="utf-8", newline="\n")
        stage  = "Verilog生成・保存"
        target = directory / "circuit.v"
        target.write_text(circuit.verilog_source(), encoding="utf-8", newline="\n")
    except (logicnn_core.LogicNNCoreError, OSError, UnicodeError) as error:
        raise _export_error(stage, target, error) from error
