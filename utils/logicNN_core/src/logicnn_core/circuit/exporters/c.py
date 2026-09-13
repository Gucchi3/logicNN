"""二値入力の回路を、型と数値演算順序を保つ独立したCソースへ変換する。"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Callable
from typing import NoReturn

from ...exceptions import LogicNNCoreError
from ..types import CircuitData, Gate, GateOp, ScalarOp, ScalarOperation, TensorDType
from ..validation import validate_circuit

_C_TYPES = {
    TensorDType.BOOL: "bool", TensorDType.UINT8: "uint8_t", TensorDType.INT8: "int8_t", TensorDType.INT16: "int16_t",
    TensorDType.INT32: "int32_t", TensorDType.INT64: "int64_t", TensorDType.FLOAT32: "float", TensorDType.FLOAT64: "double",
}
_FLOATS = (TensorDType.FLOAT32, TensorDType.FLOAT64)
_WIDTHS = {TensorDType.UINT8: 8, TensorDType.INT8: 8, TensorDType.INT16: 16, TensorDType.INT32: 32, TensorDType.INT64: 64}


def _fail(message: str) -> NoReturn:
    """再現未確認のC変換条件を、黙って別の数値型へ変更せず報告する。"""
    raise LogicNNCoreError(message, detail="C source lowering", hint="元のIRは維持されています。対応する型・演算を使用するかPython評価を利用してください")


def _ctype(dtype: TensorDType) -> str:
    """対応するC型名を返し、低精度型を暗黙にfloatへ置き換えない。"""
    if dtype not in _C_TYPES:
        _fail(f"C出力で数値型 {dtype.value} の再現は未対応です")
    return _C_TYPES[dtype]


def _literal(value: int | float, dtype: TensorDType) -> str:
    """有限の保存定数を型付きC式にし、整数の下位bitと浮動小数点の精度を維持する。"""
    name = _ctype(dtype)
    if dtype in _FLOATS:
        literal = repr(float(value))
        return f"(({name})({literal}))"
    if dtype is TensorDType.BOOL:
        return "true" if value else "false"
    if type(value) is not int:
        _fail("整数結果へ小数定数を暗黙に変換するC演算は未対応です")
    return _wrap_integer(f"UINT64_C({value % (1 << 64)})", dtype)


def _wrap_integer(expression: str, dtype: TensorDType) -> str:
    """unsigned算術の下位bitを、符号付きoverflowなしで保存整数型へ戻す。"""
    if dtype is TensorDType.BOOL:
        return f"((bool)({expression}))"
    if dtype is TensorDType.UINT8:
        return f"((uint8_t)({expression}))"
    return f"logicnn_int{_WIDTHS[dtype]}((uint64_t)({expression}))"


def _integer_helpers() -> list[str]:
    """unsignedからsignedへの変換を、範囲外のsigned castに依存せず定義する。"""
    lines = []
    for width in (8, 16, 32, 64):
        unsigned = f"uint{width}_t"
        signed   = f"int{width}_t"
        lines.extend([
            f"static {signed} logicnn_int{width}(uint64_t value) {{",
            f"    {unsigned} bits = ({unsigned})value;",
            f"    return bits <= INT{width}_MAX ? ({signed})bits : ({signed})(-1 - ({signed})(({unsigned})~bits));",
            "}",
        ])
    return lines


def _needed_gates(data: CircuitData) -> tuple[dict[int, Gate], Counter[int]]:
    """集約後の出力へ到達するgateと、その実際の利用回数を取得する。"""
    nodes = {gate.output_id: gate for gate in data.gates}
    roots = [*data.output_ids, *(node_id for reduction in data.reductions for node_id in reduction.input_ids)]
    uses  = Counter(roots)
    seen: set[int] = set()
    stack = list(roots)
    while stack:
        node_id = stack.pop()
        if node_id not in nodes or node_id in seen:
            continue
        seen.add(node_id)
        uses.update(nodes[node_id].input_ids)
        stack.extend(nodes[node_id].input_ids)
    return {node_id: gate for node_id, gate in nodes.items() if node_id in seen}, uses


def _gate_tokens(gate: Gate, packed: bool) -> tuple[str | int, ...]:
    """論理gateを子参照と記号の短いtoken列へ分解する。"""
    op = gate.op
    if op is GateOp.CONST_FALSE:
        return ("0" if packed else "false",)
    if op is GateOp.CONST_TRUE:
        return ("~((uint64_t)0)" if packed else "true",)
    a = gate.input_ids[0]
    if op is GateOp.WIRE:
        return (a,)
    inverse = "~" if packed else "!"
    if op is GateOp.NOT:
        return (f"({inverse}(", a, "))")
    b = gate.input_ids[1]
    if op in (GateOp.AND, GateOp.NAND, GateOp.OR, GateOp.NOR, GateOp.XOR, GateOp.XNOR):
        symbol = "&" if op in (GateOp.AND, GateOp.NAND) else "|" if op in (GateOp.OR, GateOp.NOR) else "^"
        prefix = inverse if op in (GateOp.NAND, GateOp.NOR, GateOp.XNOR) else ""
        return (f"({prefix}(", a, f" {symbol} ", b, "))")
    symbol = "&" if op in (GateOp.AND_NOT_A, GateOp.AND_NOT_B) else "|"
    if op in (GateOp.AND_NOT_A, GateOp.OR_NOT_A):
        return (f"(({inverse}(", a, f")) {symbol} ", b, ")")
    return ("(", a, f" {symbol} ({inverse}(", b, ")))")


def _expression(node_id: int, nodes: dict[int, Gate], inline: set[int], reference: Callable[[int], str], packed: bool) -> str:
    """inline対象だけを非再帰で展開し、長い単一利用chainの文字列複製を避ける。"""
    tokens: list[str] = []
    pending: list[str | int] = [node_id]
    while pending:
        token = pending.pop()
        if isinstance(token, str):
            tokens.append(token)
        elif token in inline:
            pending.extend(reversed(_gate_tokens(nodes[token], packed)))
        else:
            tokens.append(reference(token))
    return "".join(tokens)


def _operation_lines(source: str, source_dtype: TensorDType, operation: ScalarOperation, target: str) -> list[str]:
    """元の数値演算1段を独立したtyped変数へ生成し、式を他段とまとめない。"""
    dtype = operation.output_dtype
    ctype = _ctype(dtype)
    if operation.op is ScalarOp.CAST:
        if source_dtype in _FLOATS and dtype in _WIDTHS:
            _fail("浮動小数点から整数へのcastは、範囲外・NaNの再現が未確認のためCでは未対応です")
        expression = f"(({ctype})({source}))" if dtype in (*_FLOATS, TensorDType.BOOL) else _wrap_integer(source, dtype)
    else:
        constant = operation.operand
        assert constant is not None
        if constant.tensor_dtype is not None:
            _ctype(constant.tensor_dtype)
        if dtype in _FLOATS:
            scalar = _literal(constant.value, constant.tensor_dtype or dtype)
            left   = f"(({ctype})({source}))"
            right  = f"(({ctype})({scalar}))"
            alpha  = _literal(operation.alpha, dtype)
        else:
            scalar = _literal(constant.value, dtype)
            left   = f"((uint64_t)({source}))"
            right  = f"((uint64_t)({scalar}))"
            alpha  = _literal(operation.alpha, dtype)
        if operation.scalar_first:
            left, right = right, left
        if operation.op in (ScalarOp.ADD, ScalarOp.SUB):
            sign = "+" if operation.op is ScalarOp.ADD else "-"
            expression = f"({left} {sign} ({alpha} * {right}))"
        elif operation.op is ScalarOp.MUL:
            expression = f"({left} * {right})"
        else:
            if dtype not in _FLOATS:
                _fail("整数dtypeを結果に指定した除算はCでは未対応です")
            expression = f"({left} / {right})"
        if dtype not in _FLOATS:
            expression = _wrap_integer(expression, dtype)
    return [f"volatile {ctype} {target} = {expression};"]


def c_source(data: CircuitData, *, inline_single_use: bool = False, pack_bits: int | None = None) -> str:
    """boolまたはsample間packed入力と、非packedの集約後出力を持つC関数を生成する。"""
    validate_circuit(data)
    if type(inline_single_use) is not bool:
        _fail("inline_single_useはboolで指定してください")
    if pack_bits is not None and (type(pack_bits) is not int or pack_bits not in (8, 16, 32, 64)):
        _fail("pack_bitsはNoneまたは8／16／32／64で指定してください")
    output_type = _ctype(data.output_dtype)
    input_count = math.prod(data.input_shape)
    output_count = math.prod(data.output_shape)
    word_type   = "bool" if pack_bits is None else f"uint{pack_bits}_t"
    nodes, uses = _needed_gates(data)
    inline      = {node_id for node_id in nodes if inline_single_use and uses[node_id] == 1}

    def reference(node_id: int) -> str:
        """入力とgateを、生成関数の局所的で衝突しない識別子へ対応付ける。"""
        return f"inputs[input_offset + {node_id}]" if node_id < input_count else f"gate_{node_id}"

    def signal(node_id: int) -> str:
        """必要なinlineだけを展開した論理値式を返す。"""
        return _expression(node_id, nodes, inline, reference, pack_bits is not None)

    def bit(node_id: int) -> str:
        """論理wordから現在sampleの1bitを取り出す。"""
        expression = signal(node_id)
        return expression if pack_bits is None else f"(((uint64_t)({expression}) >> bit_index) & UINT64_C(1))"

    lines = [
        "/* Generated by logicNN-core. Input values must already be binary. */",
        f"/* Input features: {input_count}; output values per sample: {output_count}. */",
        "/* Outputs are unpacked, sample-major. Packed inputs are group-major then feature-major; bit 0 is the first sample. */",
        "#include <stdbool.h>", "#include <stddef.h>", "#include <stdint.h>",
        "#if defined(_WIN32)", "#define LOGICNN_API __declspec(dllexport)", "#else", "#define LOGICNN_API", "#endif",
        *_integer_helpers(),
        f"LOGICNN_API void logicnn_evaluate(const {word_type} *inputs, {output_type} *outputs, size_t samples) {{",
    ]
    if pack_bits is None:
        lines.extend(["    for (size_t sample = 0; sample < samples; ++sample) {", f"        size_t input_offset = sample * {input_count};"])
    else:
        lines.extend([
            f"    for (size_t group = 0; group < samples / {pack_bits} + (samples % {pack_bits} != 0); ++group) {{",
            f"        size_t input_offset = group * {input_count};",
        ])
    for node_id, gate in nodes.items():
        if node_id not in inline:
            expression = _expression(node_id, nodes, inline | {node_id}, reference, pack_bits is not None)
            lines.append(f"        {word_type} gate_{node_id} = ({word_type})({expression});")
    indent = "        "
    if pack_bits is not None:
        lines.extend([
            f"        for (size_t bit_index = 0; bit_index < {pack_bits}; ++bit_index) {{",
            f"            size_t sample = group * {pack_bits} + bit_index;",
            "            if (sample >= samples) break;",
        ])
        indent = "            "
    scores: dict[int, str] = {}
    for reduction in data.reductions:
        dtype = reduction.dtype
        ctype = _ctype(dtype)
        count = f"count_{reduction.output_id}"
        lines.append(f"{indent}uint64_t {count} = 0;")
        for node_id in reduction.input_ids:
            lines.append(f"{indent}{count} += (uint64_t)({bit(node_id)});")
        source = f"value_{reduction.output_id}_0"
        lines.append(f"{indent}{ctype} {source} = ({ctype}){count};")
        for index, operation in enumerate(reduction.operations, start=1):
            target = f"value_{reduction.output_id}_{index}"
            lines.extend(indent + line for line in _operation_lines(source, dtype, operation, target))
            source, dtype = target, operation.output_dtype
        scores[reduction.output_id] = source
    for index, node_id in enumerate(data.output_ids):
        expression = scores[node_id] if node_id in scores else bit(node_id)
        lines.append(f"{indent}outputs[sample * {output_count} + {index}] = ({output_type})({expression});")
    if pack_bits is not None:
        lines.append("        }")
    lines.extend(["    }", "}", ""])
    return "\n".join(lines)
