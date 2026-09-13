"""回路IRを厳密なJSON schemaへ変換し、数値情報と元出力対応を保存する。"""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Mapping
from enum import Enum
from pathlib import Path
from typing import NoReturn, TypeVar

from ..exceptions import LogicNNCoreError
from .types import CircuitData, Gate, GateOp, LogicalOutput, ScalarConstant, ScalarOp, ScalarOperation, SumReduction, TensorDType
from .validation import validate_circuit

_FORMAT           = "logicnn_core.circuit"
_SCHEMA_VERSION   = 1
_ROOT_FIELDS      = {"format", "schema_version", "input_shape", "output_shape", "output_dtype", "gates", "reductions", "output_ids", "logical_outputs"}
_GATE_FIELDS      = {"output_id", "op", "input_ids"}
_REDUCTION_FIELDS  = {"output_id", "input_ids", "dtype", "operations"}
_OPERATION_FIELDS = {"op", "output_dtype", "operand", "scalar_first", "alpha", "input_ndim"}
_CONSTANT_FIELDS  = {"value", "tensor_dtype", "tensor_ndim"}
_LOGICAL_FIELDS   = {"node_ids"}
_EnumType         = TypeVar("_EnumType", bound=Enum)


def _fail(target: str, message: str, hint: str) -> NoReturn:
    """JSON変換の失敗箇所と対処方針を共通例外で通知する。"""
    raise LogicNNCoreError(f"{target}: {message}", detail=f"回路JSON schema version {_SCHEMA_VERSION}: {target}", hint=hint)


def _object(value: object, fields: set[str], target: str) -> Mapping[str, object]:
    """JSON objectの全必須fieldと未知fieldを、default補完せず検査する。"""
    if not isinstance(value, Mapping):
        _fail(target, "JSON objectが必要です", "schemaのfield名をkeyに持つMappingを指定してください")
    if any(type(key) is not str for key in value):
        _fail(target, "JSON objectのkeyは文字列が必要です", "全てのkeyにschemaで定めた文字列を使用してください")
    missing = fields - value.keys()
    unknown = value.keys() - fields
    if missing or unknown:
        _fail(target, f"必須field欠落={sorted(missing)}, 未知field={sorted(unknown)}", "schema version 1の全fieldを省略せず指定し、未知fieldを除いてください")
    return value


def _array(value: object, target: str) -> list[object]:
    """順序と重複を持つJSON arrayをPython listだけに限定する。"""
    if type(value) is not list:
        _fail(target, "JSON arrayが必要です", "tupleや文字列ではなく、順序を保持したlistを指定してください")
    return value


def _enum(value: object, enum_type: type[_EnumType], target: str) -> _EnumType:
    """正式なJSON文字列からだけenumを復元し、未知の演算やdtypeを拒否する。"""
    if type(value) is not str:
        _fail(target, "enum値はJSON文字列が必要です", "schemaで定めた演算名またはdtype名を通常の文字列として指定してください")
    try:
        return enum_type(value)
    except ValueError as error:
        choices = ", ".join(item.value for item in enum_type)
        raise LogicNNCoreError(f"{target}: 未対応の値です: {value!r}", detail=str(error), hint=f"次の正式な値を使用してください: {choices}") from error


def _constant_from_dict(value: object, target: str) -> ScalarConstant | None:
    """型付き定数の値・Tensor dtype・rankを変えずに読み取る。"""
    if value is None:
        return None
    fields = _object(value, _CONSTANT_FIELDS, target)
    dtype  = None if fields["tensor_dtype"] is None else _enum(fields["tensor_dtype"], TensorDType, f"{target}.tensor_dtype")
    return ScalarConstant(fields["value"], dtype, fields["tensor_ndim"])


def _operation_from_dict(value: object, target: str) -> ScalarOperation:
    """各演算の順序・左右・alpha・入出力数値情報を全て復元する。"""
    fields  = _object(value, _OPERATION_FIELDS, target)
    op      = _enum(fields["op"], ScalarOp, f"{target}.op")
    dtype   = _enum(fields["output_dtype"], TensorDType, f"{target}.output_dtype")
    operand = _constant_from_dict(fields["operand"], f"{target}.operand")
    return ScalarOperation(op, dtype, operand, fields["scalar_first"], fields["alpha"], fields["input_ndim"])


def _reduction_from_dict(value: object, target: str) -> SumReduction:
    """sumの入力とdtype、全ての数値演算をJSONに保存された順に復元する。"""
    fields     = _object(value, _REDUCTION_FIELDS, target)
    input_ids  = tuple(_array(fields["input_ids"], f"{target}.input_ids"))
    dtype      = _enum(fields["dtype"], TensorDType, f"{target}.dtype")
    operations = _array(fields["operations"], f"{target}.operations")
    steps      = tuple(_operation_from_dict(item, f"{target}.operations[{index}]") for index, item in enumerate(operations))
    return SumReduction(fields["output_id"], input_ids, dtype, steps)


def _gate_from_dict(value: object, target: str) -> Gate:
    """gateの演算と入力位置を変更せずにJSON objectから復元する。"""
    fields = _object(value, _GATE_FIELDS, target)
    op     = _enum(fields["op"], GateOp, f"{target}.op")
    inputs = tuple(_array(fields["input_ids"], f"{target}.input_ids"))
    return Gate(fields["output_id"], op, inputs)


def _logical_from_dict(value: object, target: str) -> LogicalOutput:
    """元の論理出力対応を重複や定数参照を含めそのまま復元する。"""
    fields = _object(value, _LOGICAL_FIELDS, target)
    return LogicalOutput(tuple(_array(fields["node_ids"], f"{target}.node_ids")))


def from_dict(value: Mapping[str, object]) -> CircuitData:
    """厳密なschema version 1のMappingを検証し、独立した回路IRを返す。"""
    fields = _object(value, _ROOT_FIELDS, "JSON")
    if type(fields["format"]) is not str or fields["format"] != _FORMAT:
        _fail("format", f"未対応の回路形式です: {fields['format']!r}", f"format={_FORMAT!r}の回路JSONを指定してください")
    if type(fields["schema_version"]) is not int or fields["schema_version"] != _SCHEMA_VERSION:
        _fail("schema_version", f"未対応のschema versionです: {fields['schema_version']!r}", "正式な整数1を指定してください。旧草案の互換読込は行いません")
    input_shape  = tuple(_array(fields["input_shape"], "input_shape"))
    output_shape = tuple(_array(fields["output_shape"], "output_shape"))
    output_dtype = _enum(fields["output_dtype"], TensorDType, "output_dtype")
    gates        = [_gate_from_dict(item, f"gates[{index}]") for index, item in enumerate(_array(fields["gates"], "gates"))]
    reductions   = [_reduction_from_dict(item, f"reductions[{index}]") for index, item in enumerate(_array(fields["reductions"], "reductions"))]
    output_ids   = list(_array(fields["output_ids"], "output_ids"))
    logical      = [_logical_from_dict(item, f"logical_outputs[{index}]") for index, item in enumerate(_array(fields["logical_outputs"], "logical_outputs"))]
    data         = CircuitData(input_shape, output_shape, gates, reductions, output_ids, logical, output_dtype)
    validate_circuit(data)
    return data


def _constant_to_dict(value: ScalarConstant | None) -> dict[str, object] | None:
    """Python数値の型と符号付き0を保って、型付き定数の全fieldを出力する。"""
    if value is None:
        return None
    return {"value": value.value, "tensor_dtype": value.tensor_dtype.value if value.tensor_dtype is not None else None, "tensor_ndim": value.tensor_ndim}


def _operation_to_dict(value: ScalarOperation) -> dict[str, object]:
    """数値演算を簡約せず、計算順序の再現に必要な全fieldを出力する。"""
    return {
        "op": value.op.value, "output_dtype": value.output_dtype.value, "operand": _constant_to_dict(value.operand),
        "scalar_first": value.scalar_first, "alpha": value.alpha, "input_ndim": value.input_ndim,
    }


def to_dict(data: CircuitData) -> dict[str, object]:
    """回路IRを検証し、共有されないcontainerだけからなるJSON表現を返す。"""
    validate_circuit(data)
    return {
        "format": _FORMAT, "schema_version": _SCHEMA_VERSION, "input_shape": list(data.input_shape), "output_shape": list(data.output_shape),
        "output_dtype": data.output_dtype.value,
        "gates": [{"output_id": gate.output_id, "op": gate.op.value, "input_ids": list(gate.input_ids)} for gate in data.gates],
        "reductions": [{
            "output_id": reduction.output_id, "input_ids": list(reduction.input_ids), "dtype": reduction.dtype.value,
            "operations": [_operation_to_dict(operation) for operation in reduction.operations],
        } for reduction in data.reductions],
        "output_ids": list(data.output_ids), "logical_outputs": [{"node_ids": list(output.node_ids)} for output in data.logical_outputs],
    }


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    """JSONの全階層で重複keyを検出し、後勝ちによる情報の消失を防ぐ。"""
    value: dict[str, object] = {}
    for key, item in pairs:
        if key in value:
            _fail("JSON", f"重複するkeyがあります: {key!r}", "同一object内では各fieldを一度だけ指定してください")
        value[key] = item
    return value


def _reject_constant(value: str) -> NoReturn:
    """標準JSONではないNaNとInfinityの拡張tokenを拒否する。"""
    _fail("JSON", f"非有限の数値tokenは使用できません: {value}", "NaNやInfinityを含まない有限の数値を使用してください")


def _file_error(path: object, action: str, error: Exception) -> LogicNNCoreError:
    """file操作の失敗を対象pathと原因を持つ共通例外へ変換する。"""
    hint = "既存の親directory、fileの権限、UTF-8 JSON形式を確認してください"
    return LogicNNCoreError(f"回路JSONの{action}に失敗しました: {path}", detail=f"{type(error).__name__}: {error}", hint=hint)


def read_json(path: str | os.PathLike[str]) -> CircuitData:
    """UTF-8 JSONを厳密に読み、元の数値情報を保った回路IRを復元する。"""
    try:
        with Path(path).open("r", encoding="utf-8") as file:
            value = json.load(file, object_pairs_hook=_unique_object, parse_constant=_reject_constant)
    except (OSError, UnicodeError, ValueError, TypeError, RecursionError) as error:
        raise _file_error(path, "読込", error) from error
    return from_dict(value)


def write_json(data: CircuitData, path: str | os.PathLike[str]) -> None:
    """検証済みIRを同じdirectoryの一時fileへ書き、既存fileを原子的に置換する。"""
    encoded   = to_dict(data)
    temporary = None
    try:
        payload     = json.dumps(encoded, ensure_ascii=False, allow_nan=False, indent=2) + "\n"
        destination = Path(path)
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", newline="\n", dir=destination.parent, prefix=f".{destination.name}.", suffix=".tmp", delete=False,
        ) as file:
            temporary = file.name
            file.write(payload)
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary, destination)
        temporary = None
    except (OSError, UnicodeError, ValueError, TypeError, RecursionError) as error:
        raise _file_error(path, "保存", error) from error
    finally:
        if temporary is not None:
            try:
                os.unlink(temporary)
            except OSError:
                pass
