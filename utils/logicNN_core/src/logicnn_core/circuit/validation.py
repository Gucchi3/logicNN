"""回路IRの型・依存順・論理出力対応を変更せずに検証する。"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import NoReturn

from ..exceptions import LogicNNCoreError
from .types import CircuitData, Gate, GateOp, LogicalOutput, ScalarConstant, ScalarOp, ScalarOperation, SumReduction, TensorDType

_ARITIES = {op: 2 for op in GateOp}
_ARITIES.update({GateOp.CONST_FALSE: 0, GateOp.CONST_TRUE: 0, GateOp.WIRE: 1, GateOp.NOT: 1})
_INTEGER_RANGES = {
    TensorDType.BOOL: (0, 1), TensorDType.UINT8: (0, 2**8 - 1), TensorDType.INT8: (-2**7, 2**7 - 1),
    TensorDType.INT16: (-2**15, 2**15 - 1), TensorDType.INT32: (-2**31, 2**31 - 1), TensorDType.INT64: (-2**63, 2**63 - 1),
}
_EXACT_COUNT_LIMITS = {dtype: bounds[1] for dtype, bounds in _INTEGER_RANGES.items() if dtype is not TensorDType.BOOL}
_EXACT_COUNT_LIMITS.update({TensorDType.FLOAT16: 2**11, TensorDType.BFLOAT16: 2**8, TensorDType.FLOAT32: 2**24, TensorDType.FLOAT64: 2**53})


def _fail(target: str, message: str, hint: str) -> NoReturn:
    """不正箇所と修正方針を含む回路検証用の共通例外を送出する。"""
    raise LogicNNCoreError(f"{target}: {message}", detail=f"回路IR検証対象: {target}", hint=hint)


def _validate_id(value: object, target: str) -> int:
    """node IDをboolではない0以上のPython整数に限定する。"""
    if type(value) is not int or value < 0:
        _fail(target, f"IDは0以上の整数が必要です: {value!r}", "boolやfloatへ変換せず、0以上のPython intを指定してください")
    return value


def _validate_shape(value: object, target: str) -> int:
    """batchを含まない正整数tupleを検証し、要素数の積を返す。"""
    if not isinstance(value, tuple) or not value:
        _fail(target, "shapeは非空tupleが必要です", "batchを除いた正整数のtupleを指定してください")
    for index, size in enumerate(value):
        if type(size) is not int or size <= 0:
            _fail(f"{target}[{index}]", f"各軸には正の整数が必要です: {size!r}", "boolや0を含めず、正のPython intでshapeを指定してください")
    return math.prod(value)


def normalize_input_shape(input_shape: Sequence[int]) -> tuple[int, ...]:
    """生成APIの正整数Sequenceを検証し、IRで使用する非空tupleへそろえる。"""
    if isinstance(input_shape, (str, bytes)) or not isinstance(input_shape, Sequence):
        _fail("input_shape", "shapeは文字列ではないSequenceが必要です", "batchを除いた正整数のlistやtupleを指定してください")
    shape = tuple(input_shape)
    _validate_shape(shape, "input_shape")
    return shape


def _validate_ids(values: object, target: str) -> tuple[int, ...]:
    """node内の参照一覧が整数IDだけを含むtupleであることを確認する。"""
    if not isinstance(values, tuple):
        _fail(target, "参照一覧はtupleが必要です", "順序と重複を保持したtupleでnode IDを指定してください")
    for index, value in enumerate(values):
        _validate_id(value, f"{target}[{index}]")
    return values


def _validate_new_id(value: object, target: str, input_count: int, defined: set[int]) -> int:
    """新しいoutput IDが入力wireや既定nodeと衝突しないことを確認する。"""
    output_id = _validate_id(value, target)
    if output_id < input_count or output_id in defined:
        _fail(target, f"ID {output_id} は既に定義されています", "入力wire・gate・reductionで重複しないoutput IDを割り当ててください")
    return output_id


def _validate_bit_references(values: tuple[int, ...], target: str, input_count: int, gate_ids: set[int]) -> None:
    """各参照を入力wireまたは既に定義済みの論理gateへ限定する。"""
    for index, value in enumerate(values):
        if value >= input_count and value not in gate_ids:
            _fail(
                f"{target}[{index}]", f"ID {value} は参照可能な論理bitではありません",
                "入力wireか先行gateを参照してください。未定義・前方・循環参照やreduction結果は指定できません",
            )


def _validate_gate(gate: Gate, index: int, input_count: int, gate_ids: set[int]) -> None:
    """gateの型・演算・入力本数と依存順を検証し、出力IDを登録する。"""
    target = f"gates[{index}]"
    if not isinstance(gate, Gate):
        _fail(target, "nodeはGate型が必要です", "JSON等の外部データをGateへ変換してから検証してください")
    output_id = _validate_new_id(gate.output_id, f"{target}.output_id", input_count, gate_ids)
    if not isinstance(gate.op, GateOp):
        _fail(f"{target}.op", f"未対応の演算です: {gate.op!r}", "正式なGateOpを指定し、NOT_A/NOT_Bは入力位置を正規化してください")
    input_ids = _validate_ids(gate.input_ids, f"{target}.input_ids")
    arity     = _ARITIES[gate.op]
    if len(input_ids) != arity:
        _fail(f"{target}.input_ids", f"{gate.op.value} の入力本数は {arity} 本です: {len(input_ids)}", "演算に対応する本数の入力IDを指定してください")
    _validate_bit_references(input_ids, f"{target}.input_ids", input_count, gate_ids)
    gate_ids.add(output_id)


def _validate_number(value: object, target: str) -> None:
    """数値定数を有限なPython int/floatに限定し、型や符号付き0を変えない。"""
    if type(value) not in (int, float):
        _fail(target, f"有限の実数が必要です: {value!r}", "boolではない有限のintまたはfloatを指定してください")
    try:
        finite = math.isfinite(value)
    except OverflowError:
        finite = False
    if not finite:
        _fail(target, f"有限値が必要です: {value!r}", "有限のPython intまたはfloatを指定してください")


def _validate_dtype(value: object, target: str) -> None:
    """dtypeを文字列から暗黙変換せず正式なTensorDTypeとして検証する。"""
    if not isinstance(value, TensorDType):
        _fail(target, f"TensorDTypeが必要です: {value!r}", "対応するbool・整数・浮動小数点型をTensorDTypeで指定してください")


def _validate_rank(value: object, target: str) -> None:
    """型昇格に必要なTensor rankを0以上のPython整数として検証する。"""
    if type(value) is not int or value < 0:
        _fail(target, f"rankは0以上の整数が必要です: {value!r}", "boolやfloatではなく、元TensorのrankをPython intで指定してください")


def _validate_scalar(operand: object, target: str) -> None:
    """scalar定数の型・有限性・Tensor dtypeとrankの組合せを検証する。"""
    if not isinstance(operand, ScalarConstant):
        _fail(target, "operandはScalarConstant型が必要です", "Python数値か1要素Tensorの値と元dtype・rankを保存してください")
    _validate_number(operand.value, f"{target}.value")
    _validate_rank(operand.tensor_ndim, f"{target}.tensor_ndim")
    if operand.tensor_dtype is None:
        if operand.tensor_ndim != 0:
            _fail(f"{target}.tensor_ndim", "Python scalarのtensor_ndimは0が必要です", "Tensorの場合は元のtensor_dtypeも指定してください")
        return
    _validate_dtype(operand.tensor_dtype, f"{target}.tensor_dtype")
    if operand.tensor_dtype in _INTEGER_RANGES:
        lower, upper = _INTEGER_RANGES[operand.tensor_dtype]
        if type(operand.value) is not int or not lower <= operand.value <= upper:
            _fail(f"{target}.value", f"{operand.tensor_dtype.value}の整数範囲を満たしません", "元Tensorの値を、dtype範囲内のPython intとして保存してください")


def _validate_operation(operation: object, target: str) -> None:
    """1段の数値演算を検証し、左右関係や中間型を省略・統合しない。"""
    if not isinstance(operation, ScalarOperation):
        _fail(target, "演算はScalarOperation型が必要です", "元の演算順にScalarOperationを保存してください")
    if not isinstance(operation.op, ScalarOp):
        _fail(f"{target}.op", f"未対応のscalar演算です: {operation.op!r}", "正式なScalarOpでadd・sub・mul・div・castを指定してください")
    _validate_dtype(operation.output_dtype, f"{target}.output_dtype")
    _validate_rank(operation.input_ndim, f"{target}.input_ndim")
    _validate_number(operation.alpha, f"{target}.alpha")
    if type(operation.scalar_first) is not bool:
        _fail(f"{target}.scalar_first", "左右指定はboolが必要です", "元の演算でscalarが左辺ならTrue、右辺ならFalseを指定してください")
    if operation.op is ScalarOp.CAST:
        if operation.operand is not None:
            _fail(f"{target}.operand", "castはoperandを持ちません", "型変換だけをoperand=Noneで保存してください")
        if operation.scalar_first:
            _fail(f"{target}.scalar_first", "castにoperandの左右指定はありません", "castではscalar_first=Falseを指定してください")
    else:
        _validate_scalar(operation.operand, f"{target}.operand")
    if operation.op not in (ScalarOp.ADD, ScalarOp.SUB) and operation.alpha != 1:
        _fail(f"{target}.alpha", "add/sub以外のalphaは1が必要です", "mul・div・castではalpha=1を指定してください")


def _validate_reduction(reduction: SumReduction, index: int, input_count: int, gate_ids: set[int], defined: set[int]) -> None:
    """集約の論理入力・sum dtype・順序付き演算を検証し、出力IDを登録する。"""
    target = f"reductions[{index}]"
    if not isinstance(reduction, SumReduction):
        _fail(target, "nodeはSumReduction型が必要です", "集約nodeをSumReductionとして定義してください")
    output_id = _validate_new_id(reduction.output_id, f"{target}.output_id", input_count, defined)
    input_ids = _validate_ids(reduction.input_ids, f"{target}.input_ids")
    _validate_bit_references(input_ids, f"{target}.input_ids", input_count, gate_ids)
    _validate_dtype(reduction.dtype, f"{target}.dtype")
    if reduction.dtype is TensorDType.BOOL:
        _fail(f"{target}.dtype", "boolは0/1の個数を表すsum dtypeとして未対応です", "全入力の個数を正確に表現できる整数または浮動小数点型を指定してください")
    if len(input_ids) > _EXACT_COUNT_LIMITS[reduction.dtype]:
        _fail(f"{target}.input_ids", f"{reduction.dtype.value}では全入力の個数を正確に表現できません", "overflowや丸めなしで0から入力本数を表せるsum dtypeを使用してください")
    if not isinstance(reduction.operations, tuple):
        _fail(f"{target}.operations", "演算一覧はtupleが必要です", "元の実行順を保ったScalarOperationのtupleを指定してください")
    for operation_index, operation in enumerate(reduction.operations):
        _validate_operation(operation, f"{target}.operations[{operation_index}]")
    defined.add(output_id)


def validate_circuit(data: CircuitData) -> None:
    """IRの型・参照・出力対応を検証し、順序や重複を一切変更しない。"""
    if not isinstance(data, CircuitData):
        _fail("CircuitData", "検証対象はCircuitData型が必要です", "正式な回路IRへ変換してからvalidate_circuitを呼んでください")
    input_count  = _validate_shape(data.input_shape, "input_shape")
    output_count = _validate_shape(data.output_shape, "output_shape")
    _validate_dtype(data.output_dtype, "output_dtype")
    for name in ("gates", "reductions", "output_ids", "logical_outputs"):
        if not isinstance(getattr(data, name), list):
            _fail(name, "一覧はlistが必要です", "順序と重複を保持したlistで指定してください")
    for name in ("output_ids", "logical_outputs"):
        if len(getattr(data, name)) != output_count:
            _fail(name, f"件数はoutput_shapeの積 {output_count} と一致する必要があります", "各最終出力位置に出力IDと元の論理出力対応を1個ずつ保存してください")

    gate_ids: set[int] = set()
    for index, gate in enumerate(data.gates):
        _validate_gate(gate, index, input_count, gate_ids)
    defined = gate_ids.copy()
    for index, reduction in enumerate(data.reductions):
        _validate_reduction(reduction, index, input_count, gate_ids, defined)
    numeric_outputs = {reduction.output_id: reduction.operations[-1].output_dtype if reduction.operations else reduction.dtype for reduction in data.reductions}
    for index, value in enumerate(data.output_ids):
        output_id = _validate_id(value, f"output_ids[{index}]")
        if output_id >= input_count and output_id not in defined:
            _fail(f"output_ids[{index}]", f"ID {output_id} は未定義です", "入力wire・gate・reductionのいずれかの既定IDを指定してください")
        if output_id in numeric_outputs and numeric_outputs[output_id] is not data.output_dtype:
            _fail(f"output_ids[{index}]", "reductionの最終dtypeとoutput_dtypeが一致しません", "元の型変換をcast演算として保存し、最終Tensorのdtypeを一致させてください")
    for index, output in enumerate(data.logical_outputs):
        target = f"logical_outputs[{index}]"
        if not isinstance(output, LogicalOutput):
            _fail(target, "対応表の要素はLogicalOutput型が必要です", "元の最終出力ごとにLogicalOutputを作成してください")
        node_ids = _validate_ids(output.node_ids, f"{target}.node_ids")
        if not node_ids:
            _fail(f"{target}.node_ids", "元の論理出力対応は空にできません", "定数や重複を除去せず、集約前のbit対応を保存してください")
        _validate_bit_references(node_ids, f"{target}.node_ids", input_count, gate_ids)
