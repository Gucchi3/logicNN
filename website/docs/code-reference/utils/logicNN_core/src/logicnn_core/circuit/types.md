---
title: types.py
sidebar_label: types.py
slug: /code-reference/utils/logicNN_core/src/logicnn_core/circuit/types
toc_min_heading_level: 2
toc_max_heading_level: 2
pagination_prev: null
pagination_next: null
---

# types.py

`utils/logicNN_core/src/logicnn_core/circuit/types.py`

PyTorchに依存しない回路の中間表現（IR）を定義します。論理ゲート、加算による集約、その後の数値演算を分けて保持し、最終数値出力に加えて集約前の生ビットの対応も保存します。

{/* source-sha256: 3667ccf7d42f80e0b1d46b003c6a9268c5c5c97ff1d1e2ebe26bdbe4d898c57d */}

- このファイルには手書きの関数・メソッドはありません。dataclassの初期化・比較などはPythonが生成します。
- Gateなどの各レコードはfrozenですが、CircuitDataは簡略化処理で一覧を更新するため変更可能です。

## GateOp {/* #gateop-class */}

1ビット出力の論理演算を表す文字列Enumです。定数は0入力、WIREとNOTは1入力、その他は2入力です。入力側の否定は演算名と入力順で表します。

継承元：`str, Enum`

| メンバー | 値 | 説明 |
|---|---|---|
| `CONST_FALSE` | `'const_false'` | 常に0を出力。 |
| `CONST_TRUE` | `'const_true'` | 常に1を出力。 |
| `WIRE` | `'wire'` | 入力Aをそのまま出力。 |
| `NOT` | `'not'` | 入力Aを反転。 |
| `AND` | `'and'` | A AND B。 |
| `OR` | `'or'` | A OR B。 |
| `XOR` | `'xor'` | A XOR B。 |
| `NAND` | `'nand'` | ANDの反転。 |
| `NOR` | `'nor'` | ORの反転。 |
| `XNOR` | `'xnor'` | XORの反転。 |
| `AND_NOT_B` | `'and_not_b'` | A AND NOT B。 |
| `AND_NOT_A` | `'and_not_a'` | NOT A AND B。 |
| `OR_NOT_B` | `'or_not_b'` | A OR NOT B。 |
| `OR_NOT_A` | `'or_not_a'` | NOT A OR B。 |

このクラスには独自の関数実装はありません。

<details>
<summary>GateOp の定義を開く</summary>

```python
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
```

</details>

## TensorDType {/* #tensordtype-class */}

回路の集約、各数値演算、最終出力で使用する型を表す文字列Enumです。論理出力はboolですが、0/1を加算した結果やその後の演算には整数・浮動小数点型を使用します。

継承元：`str, Enum`

| メンバー | 値 | 説明 |
|---|---|---|
| `BOOL` | `'bool'` | 論理値。 |
| `UINT8` | `'uint8'` | 8ビット符号なし整数。 |
| `INT8` | `'int8'` | 8ビット符号付き整数。 |
| `INT16` | `'int16'` | 16ビット符号付き整数。 |
| `INT32` | `'int32'` | 32ビット符号付き整数。 |
| `INT64` | `'int64'` | 64ビット符号付き整数。 |
| `FLOAT16` | `'float16'` | 半精度浮動小数点。 |
| `BFLOAT16` | `'bfloat16'` | bfloat16形式。 |
| `FLOAT32` | `'float32'` | 単精度浮動小数点。 |
| `FLOAT64` | `'float64'` | 倍精度浮動小数点。 |

このクラスには独自の関数実装はありません。

<details>
<summary>TensorDType の定義を開く</summary>

```python
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
```

</details>

## ScalarOp {/* #scalarop-class */}

論理ビットの加算後に実行する演算を表す文字列Enumです。ADD・SUBではalphaも保持し、演算の左右と型変換の位置を再現します。

継承元：`str, Enum`

| メンバー | 値 | 説明 |
|---|---|---|
| `ADD` | `'add'` | 加算。 |
| `SUB` | `'sub'` | 減算。 |
| `MUL` | `'mul'` | 乗算。 |
| `DIV` | `'div'` | 除算。 |
| `CAST` | `'cast'` | 指定dtypeへの型変換。 |

このクラスには独自の関数実装はありません。

<details>
<summary>ScalarOp の定義を開く</summary>

```python
class ScalarOp(str, Enum):
    """sum後に元の順序と数値型を維持して実行するscalar演算。"""

    ADD  = "add"
    SUB  = "sub"
    MUL  = "mul"
    DIV  = "div"
    CAST = "cast"
```

</details>

## Gate {/* #gate-class */}

1本の論理ビットを生成する変更不可のレコードです。input_idsの順序は左右の入力の意味を持ち、非可換演算では入れ替えられません。

| 属性 | 型 | 既定値 | 説明 |
|---|---|---|---|
| `output_id` | `int` | `必須` | このゲートが定義する新しい信号ID。 |
| `op` | `GateOp` | `必須` | 実行するGateOp。 |
| `input_ids` | `tuple[int, ...]` | `必須` | 入力信号IDの順序付きtuple。演算に応じて0・1・2本。 |

`@dataclass` により、初期化などのメソッドが自動生成されます。表の属性は初期化時に指定します。

このクラスには独自の関数実装はありません。

<details>
<summary>Gate の定義を開く</summary>

```python
@dataclass(frozen=True, slots=True)
class Gate:
    """既定の論理bitを入力に取り、1本の論理bitを定義するgate。"""

    output_id: int
    op: GateOp
    input_ids: tuple[int, ...]
```

</details>

## ScalarConstant {/* #scalarconstant-class */}

数値演算に使う定数を保持します。Python数値と1要素TensorではPyTorchの型昇格が異なるため、値だけでなく元のTensorのdtypeと次元数も区別して保存します。

| 属性 | 型 | 既定値 | 説明 |
|---|---|---|---|
| `value` | `int \| float` | `必須` | 有限のPython intまたはfloatによる定数値。 |
| `tensor_dtype` | `TensorDType \| None` | `None` | 元がTensorならそのdtype。Python数値ならNone。 |
| `tensor_ndim` | `int` | `0` | 元Tensorの次元数。Python数値では0。 |

`@dataclass` により、初期化などのメソッドが自動生成されます。表の属性は初期化時に指定します。

このクラスには独自の関数実装はありません。

<details>
<summary>ScalarConstant の定義を開く</summary>

```python
@dataclass(frozen=True, slots=True)
class ScalarConstant:
    """Python数値と1要素Tensorの型・rankを区別して保持する定数。"""

    value: int | float
    tensor_dtype: TensorDType | None = None
    tensor_ndim: int = 0
```

</details>

## ScalarOperation {/* #scalaroperation-class */}

集約後の数値演算を1段ずつ記録します。演算の結合や定数の一括計算を行わず、元の左右関係、alpha、入力の次元数、結果のdtypeを保持します。

| 属性 | 型 | 既定値 | 説明 |
|---|---|---|---|
| `op` | `ScalarOp` | `必須` | ADD・SUB・MUL・DIV・CASTのいずれか。 |
| `output_dtype` | `TensorDType` | `必須` | この演算直後の値のdtype。 |
| `operand` | `ScalarConstant \| None` | `None` | 数値演算の定数側の値。CASTではNone。 |
| `scalar_first` | `bool` | `False` | 定数が元の演算の左側ならTrue。右側ならFalse。 |
| `alpha` | `int \| float` | `1` | ADD・SUBで右側の値へ適用する係数。それ以外は1。 |
| `input_ndim` | `int` | `1` | 型昇格を再現するための、演算直前のTensorの次元数。 |

`@dataclass` により、初期化などのメソッドが自動生成されます。表の属性は初期化時に指定します。

このクラスには独自の関数実装はありません。

<details>
<summary>ScalarOperation の定義を開く</summary>

```python
@dataclass(frozen=True, slots=True)
class ScalarOperation:
    """元のoperand左右・alpha・入力rank・出力dtypeを保持する1段の数値演算。"""

    op: ScalarOp
    output_dtype: TensorDType
    operand: ScalarConstant | None = None
    scalar_first: bool = False
    alpha: int | float = 1
    input_ndim: int = 1
```

</details>

## SumReduction {/* #sumreduction-class */}

論理ビットを指定dtypeで加算し、その結果へ数値演算を順番に適用するレコードです。同じビットIDの重複は加算回数の意味を持ちます。

| 属性 | 型 | 既定値 | 説明 |
|---|---|---|---|
| `output_id` | `int` | `必須` | 集約と数値演算を終えた結果のID。 |
| `input_ids` | `tuple[int, ...]` | `必須` | 加算対象の論理ビットID。順序と重複を保持。 |
| `dtype` | `TensorDType` | `TensorDType.INT64` | 加算そのものに使うdtype。 |
| `operations` | `tuple[ScalarOperation, ...]` | `()` | 加算後に実行するScalarOperationの順序付きtuple。 |

`@dataclass` により、初期化などのメソッドが自動生成されます。表の属性は初期化時に指定します。

このクラスには独自の関数実装はありません。

<details>
<summary>SumReduction の定義を開く</summary>

```python
@dataclass(frozen=True, slots=True)
class SumReduction:
    """論理bitの和と、その後に順番どおり実行する数値演算列を保持する集約node。"""

    output_id: int
    input_ids: tuple[int, ...]
    dtype: TensorDType = TensorDType.INT64
    operations: tuple[ScalarOperation, ...] = ()
```

</details>

## LogicalOutput {/* #logicaloutput-class */}

最終出力1個に対応する、集約前の生ビット列を保持します。簡略化で定数を加算へ吸収しても、この対応を残すことでVerilog出力の本数と順序を維持します。

| 属性 | 型 | 既定値 | 説明 |
|---|---|---|---|
| `node_ids` | `tuple[int, ...]` | `必須` | 元の生ビットを指す非空の順序付きtuple。定数参照と重複も保持。 |

`@dataclass` により、初期化などのメソッドが自動生成されます。表の属性は初期化時に指定します。

このクラスには独自の関数実装はありません。

<details>
<summary>LogicalOutput の定義を開く</summary>

```python
@dataclass(frozen=True, slots=True)
class LogicalOutput:
    """元の最終出力1個に対応する集約前bitの順序付き参照。"""

    node_ids: tuple[int, ...]
```

</details>

## CircuitData {/* #circuitdata-class */}

回路全体の入出力形状、依存順の論理ゲート、数値集約、2種類の出力対応をまとめる変更可能なデータ構造です。入力信号のIDはinput_shapeの要素を平坦化した位置に対応します。

| 属性 | 型 | 既定値 | 説明 |
|---|---|---|---|
| `input_shape` | `tuple[int, ...]` | `必須` | バッチ軸を除いた入力形状。非空の正整数tuple。 |
| `output_shape` | `tuple[int, ...]` | `必須` | バッチ軸を除いた最終数値出力形状。 |
| `gates` | `list[Gate]` | `必須` | 参照先が先に定義される順で並んだGateのlist。 |
| `reductions` | `list[SumReduction]` | `必須` | 論理ビットの和と数値演算を持つSumReductionのlist。 |
| `output_ids` | `list[int]` | `必須` | 最終出力の平坦化順に並べた信号・集約ID。 |
| `logical_outputs` | `list[LogicalOutput]` | `必須` | 各最終出力に対応する集約前ビット列。 |
| `output_dtype` | `TensorDType` | `必須` | 最終Tensorのdtype。 |

`@dataclass` により、初期化などのメソッドが自動生成されます。表の属性は初期化時に指定します。

このクラスには独自の関数実装はありません。

<details>
<summary>CircuitData の定義を開く</summary>

```python
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
```

</details>

## FxSignalReference {/* #fxsignalreference-class */}

FXグラフから、集約前の論理信号を外部指定するための参照です。ノード名と、そのノードのバッチ1の出力を平坦化した位置を組み合わせます。

| 属性 | 型 | 既定値 | 説明 |
|---|---|---|---|
| `node_name` | `str` | `必須` | 対象FXノードの名前。 |
| `flat_index` | `int` | `必須` | バッチサイズ1のノード出力を平坦化したときの位置。 |

`@dataclass` により、初期化などのメソッドが自動生成されます。表の属性は初期化時に指定します。

このクラスには独自の関数実装はありません。

<details>
<summary>FxSignalReference の定義を開く</summary>

```python
@dataclass(frozen=True, slots=True)
class FxSignalReference:
    """FX nodeのbatch 1を含むflatten位置で元の論理信号を指定する参照。"""

    node_name: str
    flat_index: int
```

</details>
