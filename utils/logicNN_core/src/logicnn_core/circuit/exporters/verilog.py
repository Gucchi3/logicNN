"""元の生出力境界だけを、状態や数値集約を持たないVerilogへ変換する。"""

from __future__ import annotations

import math
from collections import Counter

from ...exceptions import LogicNNCoreError
from ..types import CircuitData, Gate, GateOp
from ..validation import validate_circuit


def _needed_gates(data: CircuitData, outputs: list[int]) -> list[Gate]:
    """生出力だけをrootとして依存gateを収集し、元の依存順で返す。"""
    needed = set(outputs)
    gates  = []
    for gate in reversed(data.gates):
        if gate.output_id in needed:
            gates.append(gate)
            needed.update(gate.input_ids)
    return list(reversed(gates))


def _parts(gate: Gate) -> tuple[str | int, ...]:
    """1個のgateをbitwise演算記号と入力IDの短い列へ変換する。"""
    op = gate.op
    if op is GateOp.CONST_FALSE:
        return ("1'b0",)
    if op is GateOp.CONST_TRUE:
        return ("1'b1",)
    a = gate.input_ids[0]
    if op is GateOp.WIRE:
        return (a,)
    if op is GateOp.NOT:
        return "~(", a, ")"
    b = gate.input_ids[1]
    if op in (GateOp.AND, GateOp.OR, GateOp.XOR):
        operator = {GateOp.AND: " & ", GateOp.OR: " | ", GateOp.XOR: " ^ "}[op]
        return "(", a, operator, b, ")"
    if op in (GateOp.NAND, GateOp.NOR, GateOp.XNOR):
        operator = {GateOp.NAND: " & ", GateOp.NOR: " | ", GateOp.XNOR: " ^ "}[op]
        return "~(", a, operator, b, ")"
    if op is GateOp.AND_NOT_B:
        return "(", a, " & ~(", b, "))"
    if op is GateOp.AND_NOT_A:
        return "(~(", a, ") & ", b, ")"
    if op is GateOp.OR_NOT_B:
        return "(", a, " | ~(", b, "))"
    return "(~(", a, ") | ", b, ")"


def _expression(root: int, gates: dict[int, Gate], inline: set[int], input_count: int, *, definition: bool = False) -> str:
    """単一利用gateだけを明示stackで展開し、深い鎖でも再帰と中間文字列複製を避ける。"""
    stack: list[str | int] = list(reversed(_parts(gates[root]))) if definition else [root]
    text: list[str] = []
    while stack:
        part = stack.pop()
        if isinstance(part, str):
            text.append(part)
        elif part in inline:
            stack.extend(reversed(_parts(gates[part])))
        elif part < input_count:
            text.append(f"logic_input[{part}]")
        else:
            text.append(f"gate_{part}")
    return "".join(text)


def verilog_source(data: CircuitData, *, inline_single_use: bool = False) -> str:
    """検証済みの元出力対応から、位置と重複を維持した組合せVerilog sourceを返す。"""
    if type(inline_single_use) is not bool:
        raise LogicNNCoreError(
            "inline_single_useはboolで指定してください", detail=f"Verilog生成: inline_single_use={inline_single_use!r}",
            hint="単一利用gateの展開にはTrue、明示wireによる出力にはFalseを指定してください",
        )
    validate_circuit(data)
    input_count = math.prod(data.input_shape)
    outputs     = [node_id for group in data.logical_outputs for node_id in group.node_ids]
    gates       = _needed_gates(data, outputs)
    uses        = Counter(outputs)
    for gate in gates:
        uses.update(gate.input_ids)
    inline = {gate.output_id for gate in gates if inline_single_use and uses[gate.output_id] == 1}
    lookup = {gate.output_id: gate for gate in gates}
    wires  = [gate for gate in gates if gate.output_id not in inline]
    lines  = [
        "// logicNN_core: combinational logic with already-binarized inputs.",
        "// Input and raw output index i refer to row-major flattened position i.",
        f"// input_shape={data.input_shape}; raw_output_count={len(outputs)}",
        "module logicnn_circuit (", f"    input wire [{input_count - 1}:0] logic_input,",
        f"    output wire [{len(outputs) - 1}:0] logic_output", ");", "",
    ]
    lines.extend(f"    wire gate_{gate.output_id};" for gate in wires)
    if wires:
        lines.append("")
    for gate in wires:
        expression = _expression(gate.output_id, lookup, inline, input_count, definition=True)
        lines.append(f"    assign gate_{gate.output_id} = {expression};")
    if wires:
        lines.append("")
    offset = 0
    for index, group in enumerate(data.logical_outputs):
        lines.append(f"    // group {index}: offset={offset}, count={len(group.node_ids)}")
        for node_id in group.node_ids:
            expression = _expression(node_id, lookup, inline, input_count)
            lines.append(f"    assign logic_output[{offset}] = {expression};")
            offset += 1
    lines.extend(["endmodule", ""])
    return "\n".join(lines)
