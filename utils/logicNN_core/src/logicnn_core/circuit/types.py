"""Torchに依存しない回路IRと外部FX信号参照の型を定義する。"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

__all__ = [
    "CircuitData", "FxSignalReference", "Gate", "GateOp", "LogicalOutput", "ScalarConstant", "ScalarOp", "ScalarOperation", "SumReduction", "TensorDType",
]


class GateOp(str, Enum):
    """入力位置を正規化した定数・配線・論理gateの演算名。"""

    CONST_FALSE = "const_false"
    CONST_TRUE  = "const_true"
    WIRE        = "wire"
    NOT         = "not"
    AND         = "and"
    OR          = "or"
    XOR         = "xor"
    NAND        = "nand"
    NOR         = "nor"
    XNOR        = "xnor"
    AND_NOT_B   = "and_not_b"
    AND_NOT_A   = "and_not_a"
    OR_NOT_B    = "or_not_b"
    OR_NOT_A    = "or_not_a"


class TensorDType(str, Enum):
    """回路のsum・各数値演算・最終Tensorで保持する数値型。"""

    BOOL     = "bool"
    UINT8    = "uint8"
    INT8     = "int8"
    INT16    = "int16"
    INT32    = "int32"
    INT64    = "int64"
    FLOAT16  = "float16"
    BFLOAT16 = "bfloat16"
    FLOAT32  = "float32"
    FLOAT64  = "float64"


class ScalarOp(str, Enum):
    """sum後に元の順序と数値型を維持して実行するscalar演算。"""

    ADD  = "add"
    SUB  = "sub"
    MUL  = "mul"
    DIV  = "div"
    CAST = "cast"


@dataclass(frozen=True, slots=True)
class Gate:
    """既定の論理bitを入力に取り、1本の論理bitを定義するgate。"""

    output_id: int
    op: GateOp
    input_ids: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class ScalarConstant:
    """Python数値と1要素Tensorの型・rankを区別して保持する定数。"""

    value: int | float
    tensor_dtype: TensorDType | None = None
    tensor_ndim: int = 0


@dataclass(frozen=True, slots=True)
class ScalarOperation:
    """元のoperand左右・alpha・入力rank・出力dtypeを保持する1段の数値演算。"""

    op: ScalarOp
    output_dtype: TensorDType
    operand: ScalarConstant | None = None
    scalar_first: bool = False
    alpha: int | float = 1
    input_ndim: int = 1


@dataclass(frozen=True, slots=True)
class SumReduction:
    """論理bitの和と、その後に順番どおり実行する数値演算列を保持する集約node。"""

    output_id: int
    input_ids: tuple[int, ...]
    dtype: TensorDType = TensorDType.INT64
    operations: tuple[ScalarOperation, ...] = ()


@dataclass(frozen=True, slots=True)
class LogicalOutput:
    """元の最終出力1個に対応する集約前bitの順序付き参照。"""

    node_ids: tuple[int, ...]


@dataclass(slots=True)
class CircuitData:
    """入出力shape、依存順のnode、集約前後の出力対応を保持する回路IR。"""

    input_shape: tuple[int, ...]
    output_shape: tuple[int, ...]
    gates: list[Gate]
    reductions: list[SumReduction]
    output_ids: list[int]
    logical_outputs: list[LogicalOutput]
    output_dtype: TensorDType


@dataclass(frozen=True, slots=True)
class FxSignalReference:
    """FX nodeのbatch 1を含むflatten位置で元の論理信号を指定する参照。"""

    node_name: str
    flat_index: int
