"""FX Tensorの配線対応を保持したまま、純粋な二値演算と集約を回路IRへ変換する。"""

# torchlogix/circuit.pyのFX変換を基に、無変更のgraph解析と出力境界保護へ再構成した。MIT License。
# Copyright (c) 2021-2023 Dr. Felix Petersen
# Copyright (c) 2024-present Dr. Lino Gerlach
# 許諾条件の全文と変更の由来は同梱LICENSEおよびTHIRD_PARTY_NOTICESを参照。

from __future__ import annotations

import math
import operator
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, NoReturn

import torch
from torch import Tensor
from torch.fx import GraphModule, Node

from ..exceptions import LogicNNCoreError
from .types import CircuitData, FxSignalReference, Gate, GateOp, LogicalOutput, ScalarConstant, ScalarOp, ScalarOperation, SumReduction, TensorDType
from .validation import normalize_input_shape, validate_circuit

__all__ = ["build_from_fx_graph"]

_FLOATS = (torch.float16, torch.bfloat16, torch.float32, torch.float64)
_INTS   = (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64)
_EXACT_INTEGER_LIMIT = {torch.float16: 2**11, torch.bfloat16: 2**8, torch.float32: 2**24, torch.float64: 2**53}


def _aten(names: str) -> set[Callable[..., Any]]:
    """コード内で列挙したATen演算だけを、実際の登録済みcallableへ解決する。"""
    result = set()
    for name in names.split():
        operation, overload = name.split(".")
        result.add(getattr(getattr(torch.ops.aten, operation), overload))
    return result


_WIRING = _aten("view.default _unsafe_view.default reshape.default flatten.using_ints permute.default transpose.int t.default "
                "expand.default clone.default alias.default detach.default lift_fresh_copy.default unsqueeze.default squeeze.default "
                "squeeze.dim squeeze.dims flip.default slice.Tensor select.int index.Tensor index_select.default unfold.default unbind.int")
_JOIN   = _aten("cat.default stack.default")
_CAST   = _aten("_to_copy.default to.dtype to.device")
_PAD    = _aten("constant_pad_nd.default pad.default")
_SUM    = _aten("sum.dim_IntList sum.default")
_MATH   = _aten("add.Tensor add.Scalar sub.Tensor sub.Scalar rsub.Tensor rsub.Scalar mul.Tensor mul.Scalar div.Tensor div.Scalar")
_MATH  |= {operator.add, operator.sub, operator.mul, operator.truediv}
_BINARY = _aten("bitwise_and.Tensor bitwise_and.Scalar bitwise_or.Tensor bitwise_or.Scalar bitwise_xor.Tensor bitwise_xor.Scalar "
                "__and__.Tensor __or__.Tensor __xor__.Tensor logical_and.default logical_or.default logical_xor.default "
                "eq.Tensor eq.Scalar ne.Tensor ne.Scalar gt.Tensor gt.Scalar ge.Tensor ge.Scalar lt.Tensor lt.Scalar le.Tensor le.Scalar")
_BINARY |= {operator.and_, operator.or_, operator.xor, operator.eq, operator.ne, operator.gt, operator.ge, operator.lt, operator.le}
_UNARY  = _aten("bitwise_not.default logical_not.default") | {operator.invert}
_LIKE   = _aten("zeros_like.default ones_like.default full_like.default")
_CREATE = _aten("arange.default arange.start arange.start_step zeros.default ones.default full.default scalar_tensor.default")
_PURE   = _WIRING | _JOIN | _CAST | _PAD | _SUM | _MATH | _BINARY | _UNARY | _LIKE | _CREATE | {operator.getitem}
_TABLE_OPS = {1: GateOp.AND, 2: GateOp.AND_NOT_B, 4: GateOp.AND_NOT_A, 6: GateOp.XOR, 7: GateOp.OR,
              8: GateOp.NOR, 9: GateOp.XNOR, 11: GateOp.OR_NOT_B, 13: GateOp.OR_NOT_A, 14: GateOp.NAND}


@dataclass(frozen=True, slots=True)
class _Signals:
    """元Tensorのshapeで並んだ回路IDと、FX上の演算dtypeを分けて保持する。"""

    ids: Tensor
    dtype: torch.dtype


def _contains_signals(value: Any) -> bool:
    """引数のcontainer内に入力依存のTensor参照があるか確認する。"""
    if isinstance(value, _Signals):
        return True
    if isinstance(value, (tuple, list)):
        return any(_contains_signals(item) for item in value)
    if isinstance(value, dict):
        return any(_contains_signals(item) for item in value.values())
    return False


def _metadata(node: Node) -> Any:
    """make_fxまたはShapePropが保存したTensorのshape・dtype情報を取得する。"""
    value = node.meta.get("val")
    return value if hasattr(value, "shape") and hasattr(value, "dtype") else node.meta.get("tensor_meta")


class _Builder:
    """各FX nodeの値と論理信号を保存し、入力順を変えずにgate・SRを追加する。"""

    def __init__(self, graph_module: GraphModule, input_shape: Sequence[int]) -> None:
        """graph型とshapeを検証し、CPU専用の配線管理を初期化する。"""
        if not isinstance(graph_module, GraphModule):
            raise LogicNNCoreError("graph_moduleはtorch.fx.GraphModuleで指定してください", hint="make_fx等で得たGraphModuleを指定してください")
        self.graph       = graph_module
        self.input_shape = normalize_input_shape(input_shape)
        self.next_id     = math.prod(self.input_shape)
        self.gates: list[Gate] = []
        self.reductions: list[SumReduction] = []
        self.reduction_ids: dict[int, SumReduction] = {}
        self.constants: dict[bool, int] = {}
        self.values: dict[Node, Any] = {}
        self.by_name: dict[str, Any] = {}
        self.node: Node | None = None

    def _fail(self, reason: str, *, stage: str = "FX→IR変換") -> NoReturn:
        """対象node・target・shape・stageと対応方法を含む例外を送出する。"""
        metadata = _metadata(self.node) if self.node is not None else None
        detail   = f"node={getattr(self.node, 'name', None)}, target={getattr(self.node, 'target', None)}, "
        detail  += f"shape={getattr(metadata, 'shape', None)}, stage={stage}: {reason}"
        raise LogicNNCoreError(f"回路へ変換できません: {reason}", detail=detail,
                               hint="二値入力・対応済みの純粋演算・正しいFX metadataと元の論理出力対応情報を指定してください")

    def _resolve(self, value: Any) -> Any:
        """先行FX nodeだけを解決し、tuple・list・dictの引数構造を維持する。"""
        if isinstance(value, Node):
            if value not in self.values:
                self._fail(f"先行していないFX参照です: {value.name}")
            return self.values[value]
        if isinstance(value, (tuple, list)):
            return type(value)(self._resolve(item) for item in value)
        if isinstance(value, dict):
            return {key: self._resolve(item) for key, item in value.items()}
        if isinstance(value, slice):
            return slice(self._resolve(value.start), self._resolve(value.stop), self._resolve(value.step))
        return value

    def _attribute(self, target: str) -> Any:
        """登録buffer等を直接取得して独立CPUコピーし、propertyや任意関数を実行しない。"""
        value: Any = self.graph
        for part in target.split("."):
            if not isinstance(value, torch.nn.Module):
                self._fail(f"get_attrの途中参照はmoduleである必要があります: {target}")
            fields = {**vars(value), **value._parameters, **value._buffers, **value._modules}
            if part not in fields:
                self._fail(f"get_attrの値が存在しません: {target}")
            value = fields[part]
        if isinstance(value, Tensor):
            if value.layout != torch.strided or value.device.type == "meta":
                self._fail("get_attrは値を持つstrided Tensorである必要があります")
            return value.detach().to(device="cpu").clone()
        if type(value) in (bool, int, float, str, tuple, list) or value is None:
            return value
        self._fail(f"get_attrの定数型は未対応です: {type(value).__name__}")

    def _constant(self, value: bool) -> int:
        """必要な論理定数gateを一度だけ生成し、出力位置から参照可能にする。"""
        if value not in self.constants:
            output_id = self.next_id
            self.next_id += 1
            self.gates.append(Gate(output_id, GateOp.CONST_TRUE if value else GateOp.CONST_FALSE, ()))
            self.constants[value] = output_id
        return self.constants[value]

    def _signals(self, value: Any) -> _Signals:
        """入力依存Tensorを維持し、0／1の定数だけを論理gate参照へ変換する。"""
        if isinstance(value, _Signals):
            return value
        if not isinstance(value, Tensor):
            value = torch.as_tensor(value, device="cpu")
        if value.layout != torch.strided or value.dtype not in (torch.bool, *_INTS, *_FLOATS):
            self._fail("論理信号にできない定数dtype・layoutです")
        if not bool(((value == 0) | (value == 1)).all()):
            self._fail("論理信号の定数は0または1に限定されます")
        ids = [self._constant(bool(item)) for item in value.reshape(-1).tolist()]
        return _Signals(torch.tensor(ids, dtype=torch.int64, device="cpu").reshape(value.shape), value.dtype)

    def _require_bits(self, value: _Signals) -> None:
        """SRの数値結果を通常の論理bitとして再利用していないか検証する。"""
        if any(int(index) in self.reduction_ids for index in value.ids.reshape(-1).tolist()):
            self._fail("reduction結果を論理gate入力やsumの入力として扱うことは未対応です")

    def _wrap(self, value: Any, dtype: torch.dtype) -> Any:
        """配線操作のTensorまたはtuple結果へ、元の演算dtypeを戻す。"""
        if isinstance(value, Tensor):
            return _Signals(value, dtype)
        if isinstance(value, (tuple, list)):
            return type(value)(self._wrap(item, dtype) for item in value)
        return value

    def _check_metadata(self, value: Any) -> None:
        """利用できるFX shape・dtypeと解析結果を照合し、元traceの数値型を黙って変更しない。"""
        metadata = _metadata(self.node)
        tensor   = value.ids if isinstance(value, _Signals) else value
        if metadata is None or not isinstance(tensor, Tensor):
            return
        if tuple(metadata.shape) != tuple(tensor.shape) or metadata.dtype != value.dtype:
            self._fail(f"FX metadataのshape・dtypeと解析結果が一致しません: parsed_shape={tuple(tensor.shape)}, parsed_dtype={value.dtype}")

    def _append_operation(self, source: _Signals, operation: ScalarOperation, shape: torch.Size | tuple[int, ...]) -> _Signals:
        """各集約の不変の計算列へ1演算を追記し、先行branchと元のbit対応を保持する。"""
        ids = []
        for index in source.ids.expand(shape).reshape(-1).tolist():
            original = self.reduction_ids.get(index)
            if original is None:
                if operation.op is ScalarOp.CAST:
                    ids.append(index)
                    continue
                self._fail("論理bitとreductionが混在するTensorへの四則演算は未対応です")
            output_id = self.next_id
            self.next_id += 1
            reduction = SumReduction(output_id, original.input_ids, original.dtype, (*original.operations, operation))
            self.reductions.append(reduction)
            self.reduction_ids[output_id] = reduction
            ids.append(output_id)
        dtype = getattr(torch, operation.output_dtype.value)
        return _Signals(torch.tensor(ids, dtype=torch.int64, device="cpu").reshape(shape), dtype)

    def _cast_signals(self, source: _Signals, dtype: torch.dtype) -> _Signals:
        """数値集約のcastを演算列へ残し、0／1だけの配線では値の対応を維持する。"""
        if dtype not in (torch.bool, *_INTS, *_FLOATS):
            self._fail("このdtype変換は回路の対応範囲外です")
        operation = ScalarOperation(ScalarOp.CAST, TensorDType(str(dtype).removeprefix("torch.")), input_ndim=source.ids.ndim)
        return self._append_operation(source, operation, source.ids.shape)

    def _wire(self, target: Callable[..., Any], args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
        """登録済みの配線操作だけをID Tensorに適用し、値をTensorのshapeから推測しない。"""
        if target is operator.getitem and isinstance(args[0], (tuple, list)):
            if _contains_signals(args[1]):
                self._fail("入力依存のcontainer indexは未対応です")
            return operator.getitem(*args)
        if target in _JOIN:
            operands = [self._signals(value) for value in args[0]]
            dtype    = operands[0].dtype
            for value in operands[1:]:
                dtype = torch.promote_types(dtype, value.dtype)
            operands = [self._cast_signals(value, dtype) if value.dtype != dtype else value for value in operands]
            result = target([value.ids for value in operands], *args[1:], **kwargs)
            return _Signals(result, dtype)
        source = self._signals(args[0])
        if _contains_signals(args[1:]) or _contains_signals(kwargs):
            self._fail("入力依存のindexやshapeによる配線は未対応です")
        if target in _CAST:
            dtype = kwargs.get("dtype")
            if dtype is None:
                dtype = next((value for value in args[1:] if isinstance(value, torch.dtype)), source.dtype)
            return self._cast_signals(source, dtype)
        if target in _PAD:
            if str(target) == "aten.pad.default" and (len(args) > 2 and args[2] != "constant" or kwargs.get("mode", "constant") != "constant"):
                self._fail("constant以外のpaddingは未対応です")
            position = 3 if str(target) == "aten.pad.default" else 2
            fill     = kwargs.get("value", args[position] if len(args) > position else 0)
            fill     = 0 if fill is None else fill
            if fill not in (0, 1):
                self._fail("padding値は0または1で指定してください")
            result = torch.ops.aten.constant_pad_nd.default(source.ids, args[1], self._constant(bool(fill)))
            return _Signals(result, source.dtype)
        result = target(source.ids, *args[1:], **kwargs)
        return self._wrap(result, source.dtype)

    def _truth_gate(self, truth: int, first: int | None, second: int | None) -> int:
        """二入力真理値表を正式GateOpへ正規化し、wireと定数を直接参照する。"""
        if truth in (0, 15):
            return self._constant(truth == 15)
        if truth in (3, 5):
            return first if truth == 3 else second
        if truth in (10, 12):
            op       = GateOp.NOT
            operands = (second if truth == 10 else first,)
        else:
            op       = _TABLE_OPS[truth]
            operands = (first, second)
        output_id = self.next_id
        self.next_id += 1
        self.gates.append(Gate(output_id, op, operands))
        return output_id

    def _logic(self, target: Callable[..., Any], args: tuple[Any, ...], kwargs: dict[str, Any]) -> _Signals:
        """0と1で演算を評価して真理値表を作り、比較・論理演算を同じIRへ変換する。"""
        operands = args[:1] if target in _UNARY else args[:2]
        if len(operands) > 2 or any(not isinstance(value, (_Signals, Tensor, bool, int, float)) for value in operands):
            self._fail("論理演算の引数型は未対応です")
        for value in operands:
            if isinstance(value, _Signals):
                self._require_bits(value)
        shapes = [value.ids.shape if isinstance(value, _Signals) else torch.as_tensor(value, device="cpu").shape for value in operands]
        shape  = torch.broadcast_shapes(*shapes)
        truth  = torch.zeros(shape, dtype=torch.int64, device="cpu")
        dtype  = torch.bool
        for first, second in ((0, 0), (0, 1), (1, 0), (1, 1)):
            choices = (first, second)
            actual  = [torch.full(shape, choices[index], dtype=value.dtype, device="cpu") if isinstance(value, _Signals) else value
                       for index, value in enumerate(operands)]
            result = target(*actual, **kwargs)
            result = torch.as_tensor(result, device="cpu").expand(shape)
            if not bool(((result == 0) | (result == 1)).all()):
                self._fail("演算の結果が0／1にならず、論理gateとして表現できません")
            truth = truth * 2 + result.to(torch.int64)
            dtype = result.dtype
        references = [value.ids.expand(shape).reshape(-1).tolist() if isinstance(value, _Signals) else [None] * math.prod(shape) for value in operands]
        if len(references) == 1:
            references.append([None] * math.prod(shape))
        ids = [self._truth_gate(code, first, second) for code, first, second in zip(truth.reshape(-1).tolist(), *references)]
        return _Signals(torch.tensor(ids, dtype=torch.int64, device="cpu").reshape(shape), dtype)

    def _sum(self, args: tuple[Any, ...], kwargs: dict[str, Any]) -> _Signals:
        """元のrow-major bit順を保存し、定数も除かず各groupのSRを作る。"""
        source = self._signals(args[0])
        self._require_bits(source)
        dimensions = kwargs.get("dim", args[1] if len(args) > 1 else None)
        if dimensions is None or dimensions == [] or dimensions == ():
            dimensions = tuple(range(source.ids.ndim))
        elif type(dimensions) is int:
            dimensions = (dimensions,)
        if not isinstance(dimensions, (tuple, list)) or any(type(dim) is not int or not -source.ids.ndim <= dim < source.ids.ndim for dim in dimensions):
            self._fail("sumの軸指定が不正です")
        axes = tuple(sorted(dim % source.ids.ndim for dim in dimensions))
        if len(set(axes)) != len(axes):
            self._fail("sumの軸指定に重複があります")
        remaining = tuple(dim for dim in range(source.ids.ndim) if dim not in axes)
        width     = math.prod(source.ids.shape[dim] for dim in axes)
        if width == 0:
            self._fail("元の論理出力対応を持たない空groupのsumは未対応です")
        dtype = kwargs.get("dtype") or (torch.int64 if source.dtype in (torch.bool, *_INTS) else source.dtype)
        if dtype not in (*_INTS, *_FLOATS):
            self._fail("sumのdtypeは数値の合計を表現できる必要があります")
        limit = torch.iinfo(dtype).max if dtype in _INTS else _EXACT_INTEGER_LIMIT[dtype]
        if width > limit:
            self._fail(f"sumのdtype={dtype}は0〜{width}の全整数を正確に表現できません。dtype付きoverflow・丸めのIR化は未対応です")
        rows = source.ids.permute(*remaining, *axes).reshape(-1, width)
        ids  = []
        for row in rows:
            output_id = self.next_id
            self.next_id += 1
            reduction = SumReduction(output_id, tuple(row.tolist()), TensorDType(str(dtype).removeprefix("torch.")))
            self.reductions.append(reduction)
            self.reduction_ids[output_id] = reduction
            ids.append(output_id)
        keepdim = kwargs.get("keepdim", args[2] if len(args) > 2 else False)
        if type(keepdim) is not bool:
            self._fail("sumのkeepdimはboolで指定してください")
        shape = tuple(1 if dim in axes else size for dim, size in enumerate(source.ids.shape)) if keepdim else tuple(source.ids.shape[dim] for dim in remaining)
        return _Signals(torch.tensor(ids, dtype=torch.int64, device="cpu").reshape(shape), dtype)

    def _affine(self, target: Callable[..., Any], args: tuple[Any, ...], kwargs: dict[str, Any]) -> _Signals:
        """scalar四則演算の型・左右関係・順序を保存し、途中丸めを係数へ統合しない。"""
        positions = [index for index, value in enumerate(args[:2]) if isinstance(value, _Signals)]
        if len(positions) != 1:
            self._fail("reduction間の四則演算は未対応です")
        position = positions[0]
        source   = args[position]
        scalar   = args[1 - position]
        shape    = torch.broadcast_shapes(source.ids.shape, scalar.shape if isinstance(scalar, Tensor) else ())
        if isinstance(scalar, Tensor):
            if scalar.numel() != 1 or scalar.dtype not in (torch.bool, *_INTS, *_FLOATS):
                self._fail("reductionの定数は対応dtypeの1要素Tensorで指定してください")
            value   = int(scalar.item()) if scalar.dtype is torch.bool else scalar.item()
            operand = ScalarConstant(value, TensorDType(str(scalar.dtype).removeprefix("torch.")), scalar.ndim)
        else:
            value   = scalar
            operand = ScalarConstant(value)
        if type(value) not in (int, float) or type(value) is float and not math.isfinite(value):
            self._fail("reductionの定数は有限のintまたはfloatで指定してください")
        name  = str(target)
        alpha = kwargs.get("alpha", 1)
        if type(alpha) not in (int, float) or type(alpha) is float and not math.isfinite(alpha):
            self._fail("alphaは有限の数値で指定してください")
        if target is operator.add or "aten.add." in name:
            op = ScalarOp.ADD
        elif target is operator.sub or "aten.sub." in name or "aten.rsub." in name:
            op = ScalarOp.SUB
        elif target is operator.mul or "aten.mul." in name:
            op = ScalarOp.MUL
        else:
            op = ScalarOp.DIV
        sample = [torch.ones((1,) * value.ids.ndim, dtype=value.dtype, device="cpu") if isinstance(value, _Signals) else value for value in args]
        dtype  = target(*sample, **kwargs).dtype
        if dtype not in (torch.bool, *_INTS, *_FLOATS):
            self._fail("scalar演算結果のdtypeは回路の対応範囲外です")
        scalar_first = position == (0 if "aten.rsub." in name else 1)
        operation    = ScalarOperation(op, TensorDType(str(dtype).removeprefix("torch.")), operand, scalar_first, alpha, source.ids.ndim)
        return self._append_operation(source, operation, shape)

    def _constant_call(self, target: Callable[..., Any], args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
        """副作用のない登録済み演算だけを、元graphと独立したCPU値に適用する。"""
        options = dict(kwargs)
        if target in _CREATE | _LIKE or target in _CAST and "device" in options:
            options["device"] = torch.device("cpu")
        if target is torch.ops.aten.to.device and len(args) > 1:
            args = (args[0], torch.device("cpu"), *args[2:])
            options.pop("device", None)
        return target(*args, **options)

    def _call(self, target: Callable[..., Any], args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
        """対応演算を分類し、未知またはin-place処理を実行前に拒否する。"""
        if target not in _PURE or getattr(getattr(target, "_schema", None), "is_mutable", False):
            self._fail("未対応または副作用のある演算です")
        if target in _SUM:
            return self._sum(args, kwargs)
        if not _contains_signals((args, kwargs)):
            return self._constant_call(target, args, kwargs)
        if target in _WIRING | _JOIN | _CAST | _PAD or target is operator.getitem:
            return self._wire(target, args, kwargs)
        if target in _LIKE:
            source = self._signals(args[0])
            dummy  = torch.zeros(source.ids.shape, dtype=source.dtype, device="cpu")
            return self._constant_call(target, (dummy, *args[1:]), kwargs)
        if target in _BINARY | _UNARY:
            return self._logic(target, args, kwargs)
        if target in _MATH:
            return self._affine(target, args, kwargs)
        self._fail("この演算の入力依存引数は未対応です")

    def _logical_outputs(self, metadata: Sequence[Sequence[FxSignalReference | bool]] | None, outputs: _Signals) -> list[LogicalOutput]:
        """内部graphの未簡約SRか、外部から指定された元の対応情報を順序どおり保存する。"""
        if metadata is None:
            return [LogicalOutput(self.reduction_ids[index].input_ids if index in self.reduction_ids else (index,))
                    for index in outputs.ids.reshape(-1).tolist()]
        if isinstance(metadata, (str, bytes)) or not isinstance(metadata, Sequence) or len(metadata) != outputs.ids.numel():
            self._fail("logical_outputsのgroup数は最終出力数に一致する必要があります", stage="論理出力対応の解決")
        result = []
        for group in metadata:
            if isinstance(group, (str, bytes)) or not isinstance(group, Sequence) or not group:
                self._fail("logical_outputsの各groupは空でないSequenceで指定してください", stage="論理出力対応の解決")
            ids = []
            for reference in group:
                if type(reference) is bool:
                    ids.append(self._constant(reference))
                    continue
                if not isinstance(reference, FxSignalReference) or type(reference.node_name) is not str or type(reference.flat_index) is not int:
                    self._fail("各信号はboolまたは正しいFxSignalReferenceで指定してください", stage="論理出力対応の解決")
                if reference.node_name not in self.by_name:
                    self._fail(f"参照されたFX nodeがgraphに存在しません: {reference.node_name}", stage="論理出力対応の解決")
                value = self.by_name[reference.node_name]
                if not isinstance(value, (Tensor, _Signals)):
                    self._fail("FxSignalReferenceはTensorの1要素を参照する必要があります", stage="論理出力対応の解決")
                size = value.ids.numel() if isinstance(value, _Signals) else value.numel()
                if not 0 <= reference.flat_index < size:
                    self._fail("FxSignalReferenceのflat_indexが範囲外です", stage="論理出力対応の解決")
                if isinstance(value, _Signals):
                    index = int(value.ids.reshape(-1)[reference.flat_index])
                else:
                    selected = self._signals(value.reshape(-1)[reference.flat_index])
                    index    = int(selected.ids)
                if index in self.reduction_ids:
                    self._fail("logical_outputsはreduction結果ではなく元の論理bitを参照してください", stage="論理出力対応の解決")
                ids.append(index)
            result.append(LogicalOutput(tuple(ids)))
        return result

    def build(self, logical_outputs: Sequence[Sequence[FxSignalReference | bool]] | None) -> CircuitData:
        """FXを先行順に解析し、batchを除いた出力shapeと全論理対応を検証して返す。"""
        placeholders = [node for node in self.graph.graph.nodes if node.op == "placeholder"]
        if len(placeholders) != 1:
            self._fail("FX placeholderは1個だけ必要です", stage="入力検証")
        self.node = placeholders[0]
        metadata  = _metadata(self.node)
        if metadata is None or not hasattr(metadata, "shape") or not hasattr(metadata, "dtype"):
            self._fail("placeholderのshapeとdtype metadataが不足しています", stage="入力検証")
        if tuple(metadata.shape) != (1, *self.input_shape) or metadata.dtype not in (torch.bool, *_FLOATS):
            self._fail("placeholderはbatch=1とinput_shapeに一致するboolまたは0／1浮動小数点Tensorが必要です", stage="入力検証")
        outputs = None
        for node in self.graph.graph.nodes:
            self.node = node
            if outputs is not None:
                self._fail("outputより後のnodeは参照できません")
            try:
                if node.op == "placeholder":
                    ids   = torch.arange(math.prod(self.input_shape), dtype=torch.int64, device="cpu").reshape(1, *self.input_shape)
                    value = _Signals(ids, metadata.dtype)
                elif node.op == "get_attr":
                    value = self._attribute(node.target)
                elif node.op == "call_function":
                    value = self._call(node.target, self._resolve(node.args), self._resolve(node.kwargs))
                elif node.op == "output":
                    value = self._resolve(node.args[0])
                    if not isinstance(value, (_Signals, Tensor)):
                        self._fail("最終出力は単一のTensorである必要があります")
                    outputs = self._signals(value)
                    self._check_metadata(outputs)
                    continue
                else:
                    self._fail(f"未対応のFX node種別です: {node.op}")
                self._check_metadata(value)
                self.values[node]       = value
                self.by_name[node.name] = value
            except LogicNNCoreError:
                raise
            except torch.OutOfMemoryError:
                raise
            except (TypeError, ValueError, RuntimeError, IndexError, KeyError, OverflowError) as error:
                self._fail(f"登録演算の引数・shapeを処理できません: {error}")
        if outputs is None or outputs.ids.ndim < 2 or outputs.ids.shape[0] != 1 or any(size <= 0 for size in outputs.ids.shape[1:]):
            self._fail("最終出力はbatch=1と、非空の正整数output_shapeを持つTensorである必要があります", stage="出力検証")
        logical = self._logical_outputs(logical_outputs, outputs)
        dtype   = TensorDType(str(outputs.dtype).removeprefix("torch."))
        data    = CircuitData(self.input_shape, tuple(outputs.ids.shape[1:]), self.gates, self.reductions, outputs.ids.reshape(-1).tolist(), logical, dtype)
        validate_circuit(data)
        return data


def build_from_fx_graph(
    graph_module: GraphModule,
    input_shape: Sequence[int],
    *,
    logical_outputs: Sequence[Sequence[FxSignalReference | bool]],
) -> CircuitData:
    """外部FXを変更せずIRへ変換し、必須の元出力対応情報から論理出力境界を保存する。"""
    builder = _Builder(graph_module, input_shape)
    if logical_outputs is None:
        builder._fail("外部FXには元のlogical_outputsを明示してください", stage="論理出力対応の検証")
    return builder.build(logical_outputs)


def _build_traced_graph(graph_module: GraphModule, input_shape: Sequence[int]) -> CircuitData:
    """未加工の自前FXから、sumの定数吸収前の論理出力対応を自動記録する。"""
    return _Builder(graph_module, input_shape).build(None)
