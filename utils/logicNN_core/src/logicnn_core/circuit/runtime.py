"""保存済みの論理回路と数値演算列を、学習状態を持たずCPU上で評価する。"""

from __future__ import annotations

import math
from typing import NoReturn

import numpy as np
import torch
from torch import Tensor

from ..exceptions import LogicNNCoreError
from .types import CircuitData, Gate, GateOp, ScalarOp, ScalarOperation, TensorDType

_DTYPES = {dtype: getattr(torch, dtype.value) for dtype in TensorDType}
_ARITHMETIC = {ScalarOp.ADD: torch.add, ScalarOp.SUB: torch.sub, ScalarOp.MUL: torch.mul, ScalarOp.DIV: torch.div}


def _fail(stage: str, message: str) -> NoReturn:
    """失敗した回路処理と入力・IRの修正方針を共通例外で報告する。"""
    raise LogicNNCoreError(message, detail=f"回路実行: {stage}", hint="保存した演算のdtypeと、batch付きの厳密な0／1入力を確認してください")


def _apply_operation(source: Tensor, operation: ScalarOperation) -> Tensor:
    """1段の型・左右・alphaを保ち、既定dtype依存だけ明示型へ置き換えて計算する。"""
    dtype = _DTYPES[operation.output_dtype]
    if operation.op is ScalarOp.CAST:
        return source.to(dtype=dtype)
    constant = operation.operand
    assert constant is not None
    scalar: Tensor | int | float = constant.value
    if constant.tensor_dtype is not None:
        scalar = torch.tensor(scalar, dtype=_DTYPES[constant.tensor_dtype], device="cpu").reshape((1,) * constant.tensor_ndim)
    scalar_float = scalar.is_floating_point() if isinstance(scalar, Tensor) else isinstance(scalar, float)
    default_dependent = not source.is_floating_point() and (
        (constant.tensor_dtype is None and scalar_float) or (operation.op is ScalarOp.DIV and not scalar_float)
    )
    cast_result = False
    if default_dependent and dtype.is_floating_point:
        if dtype in (torch.float16, torch.bfloat16):
            if not operation.scalar_first and operation.op in (ScalarOp.MUL, ScalarOp.DIV):
                source      = source.to(dtype=dtype).to(dtype=torch.float32)
                cast_result = True
            else:
                scalar = torch.tensor(constant.value, dtype=dtype, device="cpu").reshape((1,) * constant.tensor_ndim)
        else:
            source = source.to(dtype=dtype)
    left, right = (scalar, source) if operation.scalar_first else (source, scalar)
    kwargs      = {"alpha": operation.alpha} if operation.op in (ScalarOp.ADD, ScalarOp.SUB) else {}
    result      = _ARITHMETIC[operation.op](left, right, **kwargs)
    if cast_result:
        result = result.to(dtype=dtype)
    return result


def validate_numeric_operations(data: CircuitData) -> None:
    """構造検証済みIRの各数値型が実行可能か、値を変えず代表Tensorで確認する。"""
    for reduction in data.reductions:
        dtype = _DTYPES[reduction.dtype]
        for index, operation in enumerate(reduction.operations):
            try:
                sample   = torch.ones((1,) * operation.input_ndim, dtype=dtype, device="cpu")
                result   = _apply_operation(sample, operation)
                expected = _DTYPES[operation.output_dtype]
                if result.dtype != expected:
                    _fail(operation.op.value, f"保存dtype={expected}と演算結果dtype={result.dtype}が一致しません")
            except LogicNNCoreError:
                raise
            except (RuntimeError, TypeError, ValueError, OverflowError) as error:
                _fail(f"node={reduction.output_id}, operation={index}", f"数値演算を実行できません: {error}")
            dtype = _DTYPES[operation.output_dtype]


def _binary_inputs(inputs: Tensor | np.ndarray, shape: tuple[int, ...]) -> Tensor:
    """入力shapeと厳密な二値性を確認して、元データから独立したCPU bool列へ変換する。"""
    if not isinstance(inputs, (Tensor, np.ndarray)):
        _fail("inputs", "入力はTensorまたはNumPy配列で指定してください")
    if inputs.ndim != len(shape) + 1 or tuple(inputs.shape[1:]) != shape:
        _fail("inputs.shape", f"入力shapeは[batch, {', '.join(map(str, shape))}]が必要です: {tuple(inputs.shape)}")
    if isinstance(inputs, Tensor):
        if inputs.layout is not torch.strided or inputs.device.type == "meta" or inputs.dtype not in _DTYPES.values():
            _fail("inputs", "入力は実データを持つstridedのbool・標準整数・浮動小数点Tensorが必要です")
        source = inputs.detach().to(device="cpu")
        if source.dtype is not torch.bool and not ((source == 0) | (source == 1)).all():
            _fail("inputs.values", "入力には厳密な0／1以外の値が含まれています")
        binary = source.to(dtype=torch.bool, copy=True)
    else:
        if inputs.dtype.kind not in "biuf":
            _fail("inputs.dtype", "NumPy入力はbool・整数・浮動小数点配列が必要です")
        if inputs.dtype.kind != "b" and not np.all((inputs == 0) | (inputs == 1)):
            _fail("inputs.values", "入力には厳密な0／1以外の値が含まれています")
        binary = torch.from_numpy(np.array(inputs, dtype=np.bool_, order="C", copy=True))
    return binary.reshape(inputs.shape[0], math.prod(shape))


def _gate_value(gate: Gate, values: dict[int, Tensor], batch: int) -> Tensor:
    """1個の論理gateをbool演算だけで全sampleへ適用する。"""
    op = gate.op
    if op in (GateOp.CONST_FALSE, GateOp.CONST_TRUE):
        return torch.full((batch,), op is GateOp.CONST_TRUE, dtype=torch.bool, device="cpu")
    a = values[gate.input_ids[0]]
    if op is GateOp.WIRE:
        return a
    if op is GateOp.NOT:
        return ~a
    b = values[gate.input_ids[1]]
    if op in (GateOp.AND, GateOp.NAND):
        return a & b if op is GateOp.AND else ~(a & b)
    if op in (GateOp.OR, GateOp.NOR):
        return a | b if op is GateOp.OR else ~(a | b)
    if op in (GateOp.XOR, GateOp.XNOR):
        return a ^ b if op is GateOp.XOR else ~(a ^ b)
    if op is GateOp.AND_NOT_B:
        return a & ~b
    if op is GateOp.AND_NOT_A:
        return ~a & b
    if op is GateOp.OR_NOT_B:
        return a | ~b
    return ~a | b


def _evaluate_reductions(data: CircuitData, values: dict[int, Tensor], batch: int) -> None:
    """論理bitを指定型で集約し、各数値演算の順序とrankを保って値表へ追加する。"""
    for reduction in data.reductions:
        dtype  = _DTYPES[reduction.dtype]
        result = torch.zeros((batch,), dtype=dtype, device="cpu")
        if reduction.input_ids:
            result = torch.stack([values[node_id] for node_id in reduction.input_ids], dim=1).sum(dim=1, dtype=dtype)
        for index, operation in enumerate(reduction.operations):
            try:
                if operation.input_ndim == 0:
                    result = torch.stack([_apply_operation(value.reshape(()), operation).reshape(()) for value in result])
                else:
                    result = _apply_operation(result.reshape(batch, *((1,) * (operation.input_ndim - 1))), operation).reshape(batch)
            except LogicNNCoreError:
                raise
            except (RuntimeError, TypeError, ValueError, OverflowError) as error:
                _fail(f"node={reduction.output_id}, operation={index}", f"数値演算を実行できません: {error}")
        values[reduction.output_id] = result


@torch.no_grad()
def evaluate(data: CircuitData, inputs: Tensor | np.ndarray, *, logical: bool = False) -> Tensor:
    """構造・数値型を検証済みのIRで二値入力を評価し、独立したCPU Tensorを返す。"""
    binary = _binary_inputs(inputs, data.input_shape)
    batch  = binary.shape[0]
    ids    = [node_id for group in data.logical_outputs for node_id in group.node_ids] if logical else data.output_ids
    shape  = (batch, len(ids)) if logical else (batch, *data.output_shape)
    dtype  = torch.bool if logical else _DTYPES[data.output_dtype]
    if batch == 0:
        return torch.empty(shape, dtype=dtype, device="cpu")
    values = {index: binary[:, index] for index in range(binary.shape[1])}
    for gate in data.gates:
        values[gate.output_id] = _gate_value(gate, values, batch)
    if not logical:
        _evaluate_reductions(data, values, batch)
    return torch.stack([values[node_id].to(dtype=dtype) for node_id in ids], dim=1).reshape(shape)
