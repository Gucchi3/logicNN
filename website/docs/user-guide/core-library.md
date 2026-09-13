---
title: ライブラリの直接利用
description: logicnn_coreをPyTorchから直接使用し、小型モデルの学習・評価・回路保存・再読込を行います。
---

# ライブラリの直接利用

このページは、[logicNNライブラリ](./logicnn-library/index.md)の実践編です。初めて論理ゲートNNを扱う場合は、[基礎](./logicnn-library/basics.md)と[学習アルゴリズム](./logicnn-library/algorithms.md)を先に読むと、以下のコードで学習する値と評価時の動作を理解しやすくなります。

`logicNN_core` は、データセットを限定しない論理ゲートNN用ライブラリです。Pythonから読み込む際の名前は `logicnn_core` です。`main.py` を使わず、自分で用意したTensorやPyTorchの学習処理と組み合わせて使用できます。

[環境構築](../startup/setup.md)の手順で同梱ライブラリをインストールし、その仮想環境を有効化してから実行してください。以下の例ではMNISTのダウンロード、アプリの設定JSON、モデル登録は不要です。

## 1. アプリの実行との違い

| `main.py` から使用する場合 | ライブラリを直接使用する場合 |
|---|---|
| JSONでデータセットや学習条件を選ぶ | Pythonコードでモデルと学習処理を記述する |
| 登録済みモデルをbuilderが生成する | `nn.Sequential` や自作の `nn.Module` を使う |
| 学習・検証・best選択・保存を自動実行する | 評価のタイミングと保存方法を自分で指定する |
| 学習後に3形式をまとめて出力する | 必要な回路APIだけを呼び出す |

直接利用では、アプリ用の `input_shape` 属性や `architecture_info()` をモデルに追加する必要はありません。回路化するときの入力形状は、`Circuit.from_model()` の引数で指定します。

## 2. 小さなモデルで一連の処理を試す

4要素の二値入力から2クラスのスコアを求める例です。`LogicDense(4, 8)` で8個の論理ゲートを作り、`GroupSum(2)` で4個ずつの出力を加算します。

作業用のフォルダに `core_example.py` として保存してください。出力はこのスクリプトと同じフォルダ内の `core_example_output/` に保存され、アプリの `log/` は使用しません。

```python title="core_example.py"
"""4標本の学習・評価と、回路の保存・再読込を確認する。"""

from pathlib import Path

import torch
from torch import nn

from logicnn_core import Circuit
from logicnn_core.layer_settings import ConnectionConfig, LUTConfig
from logicnn_core.layers import GroupSum, LogicDense


def main() -> None:
    """小さな2クラスモデルを学習し、評価結果と回路の結果を比較する。"""
    torch.manual_seed(0)
    inputs  = torch.tensor([[0, 0, 0, 0], [0, 1, 1, 0], [1, 0, 0, 1], [1, 1, 1, 1]], dtype=torch.float32)
    targets = torch.tensor([0, 1, 1, 0], dtype=torch.int64)
    model   = nn.Sequential(
        LogicDense(4, 8, lut=LUTConfig(kind="raw", num_inputs=2), connections=ConnectionConfig(init="random_unique")),
        GroupSum(2),
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=0.0)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for _ in range(20):
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(inputs), targets)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        reference = model(inputs)

    circuit = Circuit.from_model(model, input_shape=(4,))
    circuit.simplify()
    scores   = circuit.evaluate(inputs)
    raw_bits = circuit.evaluate_logical_outputs(inputs)
    torch.testing.assert_close(scores.to(reference.dtype), reference)

    output = Path(__file__).resolve().parent / "core_example_output"
    output.mkdir(exist_ok=True)
    circuit.write_json(output / "circuit.json")
    circuit.write_c(output / "circuit.c")
    circuit.write_verilog(output / "circuit.v")

    restored = Circuit.from_json(output / "circuit.json")
    torch.testing.assert_close(restored.evaluate(inputs), scores)
    torch.testing.assert_close(restored.evaluate_logical_outputs(inputs), raw_bits)
    print("scores:", tuple(scores.shape))
    print("raw bits:", tuple(raw_bits.shape))
    print("モデルと回路の評価、およびJSON再読込後の評価が一致しました。")
    print("保存先:", output)


if __name__ == "__main__":
    main()
```

仮想環境を有効化した端末で、保存したスクリプトのあるフォルダへ移動し、次のように実行します。

```shell
python core_example.py
```

この例では、4標本をまとめて使って重みを20回更新します。処理の使い方を確認するための人工データであり、分類精度の評価やMNISTの学習例ではありません。

### 確認できる結果

正常に終了すると、次の形状と保存先が表示されます。

```text
scores: (4, 2)
raw bits: (4, 8)
モデルと回路の評価、およびJSON再読込後の評価が一致しました。
```

- `scores`：4標本それぞれについて、2グループの加算結果。
- `raw_bits`：4標本それぞれについて、加算前の8個の論理出力。
- `circuit.json`：読み直して評価できる回路情報。
- `circuit.c`：2個の集約後出力を返すCソース。
- `circuit.v`：8個の生ビットを返すVerilogソース。

この例の `GroupSum` はbiasとtauを既定値のまま使います。通常のモデル評価では浮動小数点の0／1を加算し、回路側ではboolを加算するため、スコアのdtypeはそれぞれfloat32とint64になります。比較時には数値の意味をそろえるため、回路側を参照結果のdtypeへ変換しています。

`torch.testing.assert_close()` は、この例の結果を比較するための確認処理です。学習する各バッチへ追加する必要はありません。CとVerilogはソース生成までで、コンパイルやシミュレーションはこの例に含みません。

同じスクリプトを再実行すると、`core_example_output/` の同名ファイルを置き換えます。過去の結果を残す場合は、実行前に出力フォルダを変更してください。

## 3. 入出力の形状を決める

### LogicDense

`LogicDense(in_features, out_features)` は入力の最終軸を変換します。入力が `[batch, in_features]` なら出力は `[batch, out_features]` です。最終軸より前の形状は保持します。

`out_features` はゲート数です。各ゲートへ何本の入力をつなぐかは、別の設定である `LUTConfig.num_inputs` で指定します。例えば `LogicDense(6, 12, lut=LUTConfig(kind="warp", num_inputs=4))` は、6個の入力特徴から4本ずつ接続するゲートを12個作ります。

### GroupSum

`GroupSum(groups)` は最終軸を連続する同じ大きさのグループへ分け、各グループを加算します。入力の最終軸の要素数は、`groups` で割り切れるようにしてください。

`GroupSum(2, bias=1.0, tau=2.0)` なら、各グループについて `(グループ内の和 + 1.0) / 2.0` を計算します。グループ数は正の整数、tauは正の有限値、biasは有限値を指定します。この層自身に学習可能な重みはありません。

### Circuit

`Circuit.from_model(model, input_shape=(4,))` の `input_shape` には、バッチ軸を含めません。生成後の `circuit.evaluate(inputs)` には、バッチ軸を含む `[batch, 4]` を渡します。1標本でも `[1, 4]` が必要です。

回路評価の入力はTensorまたはNumPy配列で、値はboolまたは厳密な0／1に限ります。例えば0.3を渡しても自動二値化しません。通常の学習入力は浮動小数点Tensorとし、回路と比較する際は、学習時と同じ前処理で二値化した入力を用意してください。

回路評価はCPUで実行し、結果もCPU Tensorを返します。`evaluate()` は集約後の形状、`evaluate_logical_outputs()` は `[batch, 生ビットの総数]` のbool Tensorを返します。

## 4. LUTと接続を変更する

LUTは、入力の0／1の組合せに対してどの値を出力するかを表す真理値表です。学習中の表現は、次の方式から選択できます。

| `LUTConfig.kind` | 対応する `num_inputs` | 学習する表現 |
|---|---|---|
| `"raw"` | 2 | 16種類の2入力論理関数を選ぶ重み |
| `"warp"` | 1、2、4、6 | Walsh基底に対する係数 |
| `"light"` | 2、4、6 | 真理値表の各出力を決める係数 |

4入力・6入力のゲートを使う場合は、`warp` または `light` を指定します。例えば、上の例のモデルを次の構成へ置き換えられます。入力データを1標本あたり6要素へ変更し、回路生成の行も `Circuit.from_model(model, input_shape=(6,))` に変更してください。この構成の集約後の出力は3個になります。

```python
model = nn.Sequential(
    LogicDense(6, 12, lut=LUTConfig(kind="warp", num_inputs=6), connections=ConnectionConfig(init="random_unique")),
    GroupSum(3),
)
```

回路化時には、多入力の真理値表をAND・OR・NOTなどの論理演算へ展開します。4入力・6入力だからCやVerilogへの変換を別APIで行う、という違いはありません。ただし、FPGAの専用LUTプリミティブへ直接割り当てる機能ではありません。

各ゲートの真理値表は `layer.truth_tables()` で取得できます。`LogicDense.truth_tables()` の戻り値は `[out_features, 2**num_inputs]` のbool Tensorです。`truth_tables_with_ids()` は表と整数IDを返しますが、6入力ではIDが `None` になり、表を使って表現します。これは回路出力が未対応という意味ではありません。

### 接続方式

`ConnectionConfig()` の既定値は固定接続です。生成時に選んだ入力位置をその後も使用します。

- `kind="fixed", init="random"`：各入力位置を独立に抽選します。同じゲート内で同じ入力を選ぶ場合があります。
- `kind="fixed", init="random_unique"`：Denseでは同じゲート内の重複を避けます。入力特徴数はゲートの入力本数以上にしてください。
- `kind="learnable"`：Denseの候補接続を重みで選択します。評価時は、重みが最大の候補を使用します。

学習可能接続の `num_candidates=None` は全入力特徴を候補にします。`init="random_unique"` を併用する場合は候補数を明示し、`in_features >= num_inputs * num_candidates` を満たす必要があります。

畳み込み用の `LogicConv2d`・`LogicConv3d` も公開されていますが、対応する接続方式は固定接続です。DenseとConvでは接続の抽選範囲や入力形状が異なるため、[論理畳み込み層](/code-reference/utils/logicNN_core/src/logicnn_core/layers/convolution)と[接続設定](/code-reference/utils/logicNN_core/src/logicnn_core/layer_settings)を参照してください。

## 5. 学習・評価・回路出力モードを使い分ける

| 操作 | 用途 |
|---|---|
| `model.train()` | 勾配を使ってゲートや接続を学習する |
| `model.eval()` | 学習済みの選択結果で評価する。勾配の無効化は別途 `torch.no_grad()` を使う |
| `set_export_mode(model)` | 対応する層を、固定したbool演算による回路出力モードへ切り替える |

`Circuit.from_model()` は、独立コピーに対してCPU移動・評価モード・回路出力モードへの切り替えを行います。通常は自分で `set_export_mode()` を呼ぶ必要がなく、元モデルのモードと重みも変更しません。

回路用のモデル動作そのものを確認したい場合だけ、明示的に切り替えます。次は、上の例で作成した `model` と `inputs` を使う場合の操作です。

```python
from logicnn_core import set_export_mode

set_export_mode(model)
with torch.no_grad():
    export_scores = model(inputs.bool())

set_export_mode(model, False)
model.train()
```

回路出力モードの `LogicDense` はbool入力を要求します。最後に `GroupSum` がある場合、モデル全体の出力はboolではなく加算後の数値になります。

回路出力モードのまま `model.train()` を呼ぶとエラーになります。先に明示解除してください。解除後も評価モードのため、学習を再開する場合は続けて `model.train()` を呼びます。

## 6. 保存した回路を使う

回路JSONは `Circuit.from_json(path)` で読み直せます。再評価にはモデルのPythonクラスや `.pth` は不要ですが、`logicnn_core` を導入した環境は必要です。生成時に保存した入力形状と同じ形状の二値データを渡します。

`Circuit.evaluate()` はPython側の回路評価であり、Cコンパイラを必要としません。Cを使って実行する場合は、別途コンパイルします。

```python
circuit.compile()
compiled_scores = circuit.evaluate(inputs, compiled=True)
```

この操作は、WindowsではMSVCの `cl`、それ以外では `cc`・`gcc`・`clang` のいずれかが利用できる環境で行います。コンパイラは自動インストールしません。Windowsでは `cl` が見つかる開発用端末を使い、Pythonの仮想環境も有効化してください。未導入やコンパイル失敗の場合、Python評価への自動切り替えは行いません。

`compile()` は内部の一時フォルダでCを生成・コンパイルして読み込みます。`write_c()` で保存したファイルを外部で編集しても、その編集内容を読み込む操作にはなりません。また、`simplify()` を呼ぶとコンパイル済み状態が解除されるため、その後のcompiled評価には再コンパイルが必要です。

C・Verilogの入出力、対応する数値型、生成物に含まれない処理は[回路出力](./circuit.md)を参照してください。任意のPyTorch演算が回路化できるわけではなく、学習に成功しても未対応演算が変換時に見つかる場合があります。

各引数と内部処理の詳細は、[LogicDense](/code-reference/utils/logicNN_core/src/logicnn_core/layers/dense)、[GroupSum](/code-reference/utils/logicNN_core/src/logicnn_core/layers/group_sum)、[Circuit](/code-reference/utils/logicNN_core/src/logicnn_core/circuit/circuit)で確認できます。
