---
title: モデルの処理をコードで追う
description: 小型DenseモデルとMNISTLGNを例に、入力から接続・LUT・集約・学習までを現行コードに沿って説明します。
---

# モデルの処理をコードで追う

このページでは、モデルが入力を受け取ってからクラススコアを返すまでを、現行コードと対応付けて読みます。まず小型モデルで `forward()` の基本を確認し、その後に実際の `MNISTLGN` を追います。

環境は[環境構築](../../startup/setup.md)の手順で用意した仮想環境を使用します。例は `python ファイル名.py` で実行できます。MNISTを取得したり、実際の学習ログを変更したりする必要はありません。

## 1. ライブラリの読み込みとアプリのモデル選択を分けて考える

| コードの場所 | 役割 |
|---|---|
| `logicnn_core.layers` | 汎用の論理層を提供するPythonパッケージ |
| `model/lgn/mnist_lgn.py` | 汎用の層を組み合わせて、MNIST用モデルを定義する |
| `model/builder.py` | 設定JSONの登録名から、モデルを生成する |
| `utils/trainer/epoch.py` | 1 epochの学習・評価を実行する |
| `utils/trainer/workflow.py` | データ・モデル・保存・評価・回路出力を順序立てて実行する |

`from logicnn_core.layers import LogicDense` は、層のクラスを読み込む処理です。`LogicDense(...)` を呼ぶと、実際の重みと接続を持つインスタンスが生成されます。importしただけでモデルの学習が始まるわけではありません。

アプリの `build_model("mnist_lgn")` は登録表から `MNISTLGN` を選び、引数なしで生成します。モデル構造はそのクラスのPythonコード内で決まります。ライブラリを直接使用する小型例では、この登録表を経由する必要はありません。

このほか、コア内部にも `build_parametrization()` や `build_dense_connections()` があります。これらはモデル名ではなく、`LUTConfig.kind` や `ConnectionConfig.kind` から内部の計算方式を選ぶ関数です。

参照：[モデルbuilder](/code-reference/model/builder)、[LUT builder](/code-reference/utils/logicNN_core/src/logicnn_core/parametrizations/builder)、[接続builder](/code-reference/utils/logicNN_core/src/logicnn_core/connections/builder)

## 2. 小型モデルで1回の学習を実行する

4個の入力 → 8個の論理ゲート → 2クラスのスコア、というモデルです。4標本のラベルは説明用に手で指定したもので、実用的な精度を得るためのデータではありません。

```python title="follow_dense.py"
"""小型論理Denseモデルで、中間形状と1回の学習を確認する。"""

import torch
from torch import nn

from logicnn_core.layer_settings import ConnectionConfig, LUTConfig
from logicnn_core.layers import GroupSum, LogicDense


def main() -> None:
    """接続・論理出力・スコアの形状を表示し、1バッチだけ重みを更新する。"""
    torch.manual_seed(0)
    inputs  = torch.tensor([[0, 0, 0, 0], [0, 1, 1, 0], [1, 0, 0, 1], [1, 1, 1, 1]], dtype=torch.float32)
    targets = torch.tensor([0, 1, 1, 0], dtype=torch.int64)
    layer   = LogicDense(4, 8, lut=LUTConfig(kind="raw"), connections=ConnectionConfig(init="random_unique"))
    model   = nn.Sequential(layer, GroupSum(2))
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=0.0)
    criterion = nn.CrossEntropyLoss()

    model.train()
    with torch.no_grad():
        selected = layer.connections(inputs)
        features = layer(inputs)
        print("接続で選んだ入力:", tuple(selected.shape))
        print("論理層の出力:", tuple(features.shape))
        print("LUT重み:", tuple(layer.weight.shape))

    optimizer.zero_grad(set_to_none=True)
    scores = model(inputs)
    loss   = criterion(scores, targets)
    loss.backward()
    optimizer.step()

    print("クラススコア:", tuple(scores.shape))
    print("LUT重みの勾配:", tuple(layer.weight.grad.shape))
    model.eval()
    with torch.no_grad():
        predictions = model(inputs).argmax(dim=1)
    print("評価時の予測:", predictions.tolist())
    print("離散化した真理値表:", tuple(layer.truth_tables().shape))


if __name__ == "__main__":
    main()
```

表示される形状は次のとおりです。予測クラスの値そのものは、1回の更新で正解することを期待するためのものではありません。

```text
接続で選んだ入力: (4, 2, 8)
論理層の出力: (4, 8)
LUT重み: (8, 16)
クラススコア: (4, 2)
LUT重みの勾配: (8, 16)
離散化した真理値表: (8, 4)
```

上の `no_grad()` による中間値の表示は、処理を観察するためだけのものです。通常の学習では、続く `scores = model(inputs)` から計算を始めれば十分です。

## 3. LogicDense.forward()の内側

通常モードでは、[LogicDense.forward()](/code-reference/utils/logicNN_core/src/logicnn_core/layers/dense#logicdense-forward) は次の順に計算します。

### 3.1 接続先から入力を取り出す

```python
selected = self.connections(values)
```

今回の固定接続は、`connections.indices` に `[2, 8]` の入力番号を保持します。あるゲートの番号が `[1, 3]` なら、そのゲートは入力ベクトルの1番目と3番目を参照します。Pythonの添字は0から始まるので、これは2個目と4個目の値です。

たとえば1標本が `[0, 1, 0, 1]` なら、そのゲートへ渡る値は `[1, 1]` です。8ゲート分を4標本について取り出すと、`selected` は `[4, 2, 8]` になります。

この段階ではANDやORはまだ計算していません。また、固定接続なので、学習が進んでも `indices` はoptimizerによって更新されません。

### 3.2 LUT表現がゲートの計算を行う

```python
result = self.parametrization(selected, self.weight, training=self.training, contraction="n,bn->bn")
```

Raw方式では各ゲートが16種類の論理関数に対するlogitを持ちます。学習時の既定設定では、logitからsoftmaxで混合比を求め、各論理関数の連続化した出力を混ぜます。

`weight` の `[8, 16]` と、選んだ入力の `[4, 2, 8]` を使い、結果の `[4, 8]` を求めます。`contraction="n,bn->bn"` は内部の掛け合わせと集計に使う軸指定です。ここでは `n` がゲート、`b` が標本を表し、各ゲートの係数を全標本へ適用します。利用者が普通に `LogicDense` を呼ぶ際に、この文字列を指定する必要はありません。

`RawLUTParametrization` は重みを自身に保存せず、引数 `self.weight` を受け取ります。重みの保存・optimizerへの登録は `LogicDense`、計算式は `RawLUTParametrization` という分担です。

`gradient_scale` が既定の1.0でなければ、この前に入力側へ返す勾配の倍率を設定します。forwardの入力値を単純にその倍率で変える機能ではありません。

### 3.3 層の出力形状へ戻す

```python
return result.reshape(*inputs.shape[:-1], self.out_features)
```

末尾の特徴軸を8に置き換え、それ以外の先行軸を元の形に戻します。今回の入力は `[4, 4]` なので、出力は `[4, 8]` です。

## 4. GroupSum・loss・backwardをつなげる

### GroupSumは8個のゲート出力を2個のスコアへまとめる

`nn.Sequential` は論理層の `[4, 8]` を次の `GroupSum(2)` へ渡します。各標本の先頭4個をクラス0用、末尾4個をクラス1用として足し、`[4, 2]` を返します。

どのゲートがどのクラスに属するかは並び順で決まります。ゲートの出力順を無計画に入れ替えると、集計されるクラスも変わります。

### CrossEntropyLossはスコアと正解クラスから損失を計算する

入力は `scores=[4, 2]` と `targets=[4]` です。正解ラベルは `[0, 1]` のような確率ベクトルではなく、各標本の正解クラス番号を1つずつ並べます。

モデルの出力にあらかじめsoftmaxを追加する必要はありません。`CrossEntropyLoss` に確率化前のスコアを渡します。`argmax()` は正解率や予測を求める処理であり、学習用lossに渡す出力ではありません。

### backwardは勾配を計算し、optimizer.stepが重みを更新する

| 処理 | 行うこと |
|---|---|
| `optimizer.zero_grad(...)` | 前回の勾配を消す |
| `scores = model(inputs)` | 現在の重みで出力を求める |
| `loss = criterion(scores, targets)` | 正解とのずれを1つの損失値へまとめる |
| `loss.backward()` | 損失から各Parameterへの勾配を計算する |
| `optimizer.step()` | 勾配とoptimizerの規則を使ってParameterを更新する |

この例の学習対象は `layer.weight` です。勾配の形状も `[8, 16]` になります。入力Tensor自体は学習パラメータではないため、この例では画像や標本を書き換えているわけではありません。

実際のアプリの [train_one_epoch()](/code-reference/utils/trainer/epoch#train-one-epoch) も、DataLoaderから取り出した各バッチに対してこの手順を繰り返します。保存・scheduler更新・検証のタイミングは [workflow.py](/code-reference/utils/trainer/workflow) で調整します。

## 5. eval()で何が変わるか

`model.eval()` は層を評価モードへ切り替えます。Raw方式では、混合に使っていた16候補から最大logitの1候補を選びます。今回の0/1入力と固定接続なら、論理層の出力も0/1になります。

`eval()` だけで勾配の記録が停止するわけではありません。評価の実行では `torch.no_grad()` も組み合わせます。反対に、`no_grad()` だけでは層は学習モードのままなので、最初の観察用コードは学習時の混合出力を表示しています。

`layer.truth_tables()` は各ゲートを離散化した真理値表を返します。2入力には4通りの組み合わせがあるので、8ゲートの表は `[8, 4]` です。これは学習重み `[8, 16]` とは別の表です。

また、評価モードと回路出力モードは同じではありません。bool演算による回路実行に切り替える方法と、解除して学習へ戻す手順は[ライブラリの直接利用](../core-library.md)を参照してください。

## 6. 現行MNISTLGNの入力から出力まで

[MNISTLGN](/code-reference/model/lgn/mnist_lgn) の `forward()` 自体は `self.network(inputs)` を返すだけです。実際の層順序は `__init__()` 内の `nn.Sequential` にあります。

### 入力はモデルに渡す前に二値化される

アプリの [mnist.py](/code-reference/utils/data/mnist) は、画像を `ToTensor()` でTensorへ変換し、[binarize_greater_than()](/code-reference/utils/data/preprocessing) で0より大きい画素を1.0、それ以外を0.0にします。MNISTの画像は `[B, 1, 28, 28]` のfloat32としてモデルへ入ります。

この二値化はモデル内部のLUT計算ではありません。入力の前処理と、ゲートの重みを離散化する処理を分けて考えてください。

### 層ごとの形状

| 順番 | 層 | 入力 → 出力 | 読み取る設定 |
|---|---|---|---|
| 1 | `LogicConv2d` | `[B, 1, 28, 28]` → `[B, 16, 26, 26]` | 受容野3×3、stride 1、padding 0、論理木2段 |
| 2 | `OrPooling2d` | `[B, 16, 26, 26]` → `[B, 16, 13, 13]` | 窓2×2、stride 2 |
| 3 | `nn.Flatten` | `[B, 16, 13, 13]` → `[B, 2704]` | バッチ以外の軸を平坦化 |
| 4 | `LogicDense` | `[B, 2704]` → `[B, 4000]` | 4000ゲート |
| 5 | `LogicDense` | `[B, 4000]` → `[B, 4000]` | 4000ゲート |
| 6 | `GroupSum` | `[B, 4000]` → `[B, 10]` | 10グループ、tau 8、bias 0 |

最後は400ゲートずつを足して8で割った10個のスコアです。`scores.argmax(dim=1)` が、数字0〜9の予測になります。現在のモデルはRaw・2入力・固定接続を使います。

### 論理Convの内部では、空間位置ごとに2段の木を計算する

3×3の受容野に対し、2入力ゲート2段の論理木は4本の末端入力を使います。既定のランダム固定接続は重複を許すため、必ず4画素すべてが異なるとは限りません。

1出力チャネルの論理木には、下段に2ゲート、上段に1ゲートがあります。16出力チャネルなので、Raw方式の重みは次の形状になります。

| 内部の値 | 形状 | 意味 |
|---|---|---|
| `tree_weights[0]` | `[2, 16, 16]` | 下段2ゲート × 16チャネル × 16候補 |
| `tree_weights[1]` | `[1, 16, 16]` | 上段1ゲート × 16チャネル × 16候補 |

26×26の各位置に別の学習パラメータを持つわけではありません。同じ論理木の重みと受容野内の接続位置を、全位置で共有します。

`_LogicConvNd.forward()` は `tree_weights` を順に取り出し、段ごとに「接続から入力を取り出す → LUTを計算する」を繰り返します。Denseで読んだ構造に、空間位置と木の段が追加されたものとして読むと整理できます。

## 7. ソースコードを読む順番

初めて読む場合は、次の順に進めてください。ファイル内の全関数を一度に読む必要はなく、表の対象を先に追います。

| 順番 | ファイル・説明ページ | 最初に読む内容 |
|---|---|---|
| 1 | [model/lgn/mnist_lgn.py](/code-reference/model/lgn/mnist_lgn) | 構造定数、`__init__()` の層順序、`forward()` |
| 2 | [layers/dense.py](/code-reference/utils/logicNN_core/src/logicnn_core/layers/dense) | `weight` と `connections` の生成、通常モードの `forward()` |
| 3 | [connections/dense.py](/code-reference/utils/logicNN_core/src/logicnn_core/connections/dense) | `FixedDenseConnections` の入力番号と取得処理 |
| 4 | [parametrizations/raw.py](/code-reference/utils/logicNN_core/src/logicnn_core/parametrizations/raw) | logitの選択、16候補の計算、真理値表への変換 |
| 5 | [functional/logic.py](/code-reference/utils/logicNN_core/src/logicnn_core/functional/logic) | AND・ORなどを連続値にも適用する式 |
| 6 | [layers/group_sum.py](/code-reference/utils/logicNN_core/src/logicnn_core/layers/group_sum) | 特徴の分割、和、bias、tauの順序 |
| 7 | [utils/trainer/epoch.py](/code-reference/utils/trainer/epoch) | forward・loss・backward・更新の1バッチ分 |
| 8 | [layers/convolution.py](/code-reference/utils/logicNN_core/src/logicnn_core/layers/convolution) | 論理木の重みと、段ごとのループ |
| 9 | [circuit/circuit.py](/code-reference/utils/logicNN_core/src/logicnn_core/circuit/circuit) | モデルとは独立した回路の生成・実行・出力 |

アプリ全体の起動や保存を調べる場合は、ここから [main.py](/code-reference/main) → [workflow.py](/code-reference/utils/trainer/workflow) へ進みます。ライブラリの数式をさらに調べる場合は `parametrizations/` と `functional/`、接続の学習方法を調べる場合は `LearnableDenseConnections` とそのbackwardへ進みます。

コードリファレンスには内部関数も含めた個別の説明とアクティビティ図があります。このページで処理全体の位置付けを確認し、細部が必要になった関数を参照する読み方を想定しています。
