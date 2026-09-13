"""数値演算の境界と生出力の対応を保ち、論理回路を固定順で簡略化する。"""

from __future__ import annotations

from dataclasses import replace

from ..exceptions import LogicNNCoreError
from .types import CircuitData, Gate, GateOp, LogicalOutput, ScalarConstant, ScalarOp, ScalarOperation, TensorDType
from .validation import validate_circuit

_TRUTH = {
    GateOp.CONST_FALSE: 0b0000, GateOp.CONST_TRUE: 0b1111, GateOp.WIRE: 0b1100, GateOp.NOT: 0b0011,
    GateOp.AND: 0b1000, GateOp.OR: 0b1110, GateOp.XOR: 0b0110, GateOp.NAND: 0b0111, GateOp.NOR: 0b0001,
    GateOp.XNOR: 0b1001, GateOp.AND_NOT_B: 0b0100, GateOp.AND_NOT_A: 0b0010, GateOp.OR_NOT_B: 0b1101, GateOp.OR_NOT_A: 0b1011,
}
_BINARY_OPS = {truth: op for op, truth in _TRUTH.items() if op not in (GateOp.CONST_FALSE, GateOp.CONST_TRUE, GateOp.WIRE, GateOp.NOT)}
_COMMUTATIVE = frozenset((GateOp.AND, GateOp.OR, GateOp.XOR, GateOp.NAND, GateOp.NOR, GateOp.XNOR))
_INTEGER_DTYPES = frozenset((TensorDType.UINT8, TensorDType.INT8, TensorDType.INT16, TensorDType.INT32, TensorDType.INT64))
_CONSTANTS = {GateOp.CONST_FALSE: False, GateOp.CONST_TRUE: True}


def _bit(op: GateOp, left: bool, right: bool = False) -> bool:
    """00・01・10・11の真理値表から指定した入力組合せの出力を返す。"""
    return bool((_TRUTH[op] >> (2 * int(left) + int(right))) & 1)


def _unary(output_id: int, input_id: int, false_value: bool, true_value: bool) -> Gate:
    """1入力の真理値を定数・WIRE・NOTのいずれかへ正規化する。"""
    if false_value == true_value:
        return Gate(output_id, GateOp.CONST_TRUE if true_value else GateOp.CONST_FALSE, ())
    return Gate(output_id, GateOp.NOT if false_value else GateOp.WIRE, (input_id,))


def _from_truth(output_id: int, input_ids: tuple[int, int], truth: int) -> Gate:
    """2入力の真理値を入力位置の意味を失わず正式なGateOpへ変換する。"""
    if truth in (0, 15):
        return Gate(output_id, GateOp.CONST_TRUE if truth else GateOp.CONST_FALSE, ())
    if truth in (12, 3, 10, 5):
        input_id = input_ids[0 if truth in (12, 3) else 1]
        return Gate(output_id, GateOp.WIRE if truth in (12, 10) else GateOp.NOT, (input_id,))
    return Gate(output_id, _BINARY_OPS[truth], input_ids)


def _replace_references(data: CircuitData, replacements: dict[int, int]) -> int:
    """削除する同値gateの参照を、生出力を含むすべての使用箇所で置き換える。"""
    if not replacements:
        return 0
    gates = []
    for gate in data.gates:
        if gate.output_id in replacements:
            continue
        input_ids = tuple(replacements.get(value, value) for value in gate.input_ids)
        gates.append(replace(gate, input_ids=input_ids) if input_ids != gate.input_ids else gate)
    data.gates = gates
    data.reductions = [replace(item, input_ids=tuple(replacements.get(value, value) for value in item.input_ids)) for item in data.reductions]
    data.output_ids = [replacements.get(value, value) for value in data.output_ids]
    data.logical_outputs = [LogicalOutput(tuple(replacements.get(value, value) for value in item.node_ids)) for item in data.logical_outputs]
    return len(replacements)


def _fuse_not_inputs(data: CircuitData) -> int:
    """入力側NOTを後続gateの真理値へ融合し、必要な元NOT出力は残す。"""
    changed = 0
    known: dict[int, Gate] = {}
    for index, gate in enumerate(data.gates):
        new_gate = gate
        if gate.op is GateOp.NOT:
            previous = known.get(gate.input_ids[0])
            if previous is not None and previous.op is GateOp.NOT:
                new_gate = Gate(gate.output_id, GateOp.WIRE, previous.input_ids)
        elif len(gate.input_ids) == 2:
            left, right = gate.input_ids
            first       = known.get(left)
            second      = known.get(right)
            invert_left = first is not None and first.op is GateOp.NOT
            invert_right = second is not None and second.op is GateOp.NOT
            if invert_left or invert_right:
                left  = first.input_ids[0] if invert_left else left
                right = second.input_ids[0] if invert_right else right
                truth = sum(int(_bit(gate.op, bool(a) ^ invert_left, bool(b) ^ invert_right)) << (2 * a + b) for a in (0, 1) for b in (0, 1))
                new_gate = _from_truth(gate.output_id, (left, right), truth)
        if new_gate != gate:
            data.gates[index] = new_gate
            changed += 1
        known[new_gate.output_id] = new_gate
    return changed


def _remove_unused(data: CircuitData) -> int:
    """最終数値出力と保存済み生出力の両方から必要なnodeだけを残す。"""
    needed = set(data.output_ids)
    needed.update(value for output in data.logical_outputs for value in output.node_ids)
    reductions = []
    for reduction in data.reductions:
        if reduction.output_id in needed:
            reductions.append(reduction)
            needed.update(reduction.input_ids)
    gates = []
    for gate in reversed(data.gates):
        if gate.output_id in needed:
            gates.append(gate)
            needed.update(gate.input_ids)
    changed         = len(data.gates) - len(gates) + len(data.reductions) - len(reductions)
    data.gates      = list(reversed(gates))
    data.reductions = reductions
    return changed


def _fold_constants(data: CircuitData) -> int:
    """既知の定数入力と同一入力の論理式だけを定数・配線・NOTへ畳み込む。"""
    constants: dict[int, bool] = {}
    changed = 0
    for index, gate in enumerate(data.gates):
        new_gate = gate
        if gate.op in _CONSTANTS:
            constants[gate.output_id] = _CONSTANTS[gate.op]
            continue
        if len(gate.input_ids) == 1 and gate.input_ids[0] in constants:
            value    = _bit(gate.op, constants[gate.input_ids[0]])
            new_gate = Gate(gate.output_id, GateOp.CONST_TRUE if value else GateOp.CONST_FALSE, ())
        elif len(gate.input_ids) == 2:
            left, right = gate.input_ids
            if left in constants and right in constants:
                value    = _bit(gate.op, constants[left], constants[right])
                new_gate = Gate(gate.output_id, GateOp.CONST_TRUE if value else GateOp.CONST_FALSE, ())
            elif left in constants:
                new_gate = _unary(gate.output_id, right, _bit(gate.op, constants[left], False), _bit(gate.op, constants[left], True))
            elif right in constants:
                new_gate = _unary(gate.output_id, left, _bit(gate.op, False, constants[right]), _bit(gate.op, True, constants[right]))
            elif left == right:
                new_gate = _unary(gate.output_id, left, _bit(gate.op, False, False), _bit(gate.op, True, True))
        if new_gate != gate:
            data.gates[index] = new_gate
            changed += 1
        if new_gate.op in _CONSTANTS:
            constants[new_gate.output_id] = _CONSTANTS[new_gate.op]
    return changed


def _bypass_wires(data: CircuitData) -> int:
    """依存順のWIRE鎖を元の信号へ解決し、出力位置を維持して削除する。"""
    replacements = {}
    for gate in data.gates:
        if gate.op is GateOp.WIRE:
            replacements[gate.output_id] = replacements.get(gate.input_ids[0], gate.input_ids[0])
    return _replace_references(data, replacements)


def _remove_duplicates(data: CircuitData) -> int:
    """同じ演算と接続のgateを共有し、可換でない入力の左右は入れ替えない。"""
    canonical: dict[tuple[GateOp, tuple[int, ...]], int] = {}
    replacements = {}
    for gate in data.gates:
        input_ids = tuple(replacements.get(value, value) for value in gate.input_ids)
        key_ids   = tuple(sorted(input_ids)) if gate.op in _COMMUTATIVE else input_ids
        key       = (gate.op, key_ids)
        if key in canonical:
            replacements[gate.output_id] = canonical[key]
        else:
            canonical[key] = gate.output_id
    return _replace_references(data, replacements)


def _fold_sum_constants(data: CircuitData) -> int:
    """整数sum内の定数を同dtypeの先頭ADDへ移し、元の数値演算列を保つ。"""
    constants = {gate.output_id: _CONSTANTS[gate.op] for gate in data.gates if gate.op in _CONSTANTS}
    changed   = 0
    for index, reduction in enumerate(data.reductions):
        if reduction.dtype not in _INTEGER_DTYPES:
            continue
        input_ids = tuple(value for value in reduction.input_ids if value not in constants)
        if len(input_ids) == len(reduction.input_ids):
            continue
        true_count = sum(constants.get(value, False) for value in reduction.input_ids)
        operations = reduction.operations
        if true_count:
            addition   = ScalarOperation(ScalarOp.ADD, reduction.dtype, ScalarConstant(true_count))
            operations = (addition,) + operations
        data.reductions[index] = replace(reduction, input_ids=input_ids, operations=operations)
        changed += 1
    return changed


_PASSES = (_fuse_not_inputs, _remove_unused, _fold_constants, _bypass_wires, _remove_duplicates, _fold_sum_constants)


def simplify(data: CircuitData, *, max_passes: int = 1000) -> None:
    """候補IRの開始時と完了時だけ検証し、変更がなくなるまで固定順で簡略化する。"""
    if type(max_passes) is not int or max_passes <= 0:
        raise LogicNNCoreError("simplify: max_passesは正のPython intが必要です", detail=f"max_passes={max_passes!r}", hint="bool以外の正整数を指定してください")
    validate_circuit(data)
    for _ in range(max_passes):
        changed = 0
        for run_pass in _PASSES:
            changed += run_pass(data)
        if not changed:
            validate_circuit(data)
            return
    raise LogicNNCoreError(
        "simplify: max_passes以内に簡略化が収束しませんでした", detail=f"max_passes={max_passes}",
        hint="巡回上限を増やしてください。元のCircuitを保持したまま、簡略化処理を確認してください",
    )
