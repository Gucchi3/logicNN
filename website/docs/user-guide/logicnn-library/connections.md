---
title: 入力接続と畳み込み
description: 固定接続、学習可能接続、Dense層、Conv層の論理木とTensorの形状を具体例で説明します。
---

# 入力接続と畳み込み

[ゲートを学習するアルゴリズム](algorithms.md)では、ゲートに渡された入力から出力を求める計算を説明しました。このページでは、その前段にある「どの入力を、どのゲートへ渡すか」を扱います。

1つの論理ゲートは、入力のすべてを使うとは限りません。例えば100個の入力特徴があっても、2入力ゲートが受け取るのは選ばれた2個だけです。ゲートのLUT重みを変えることと、入力の接続先を変えることは別の操作です。

## 1. 入力数・出力数・ゲートの入力数を区別する

```python
from logicnn_core.layer_settings import ConnectionConfig, LUTConfig
from logicnn_core.layers import LogicDense

layer = LogicDense(6, 3, lut=LUTConfig(kind="light", num_inputs=2), connections=ConnectionConfig(init="random_unique"))
```

この例の3つの数値は、それぞれ異なる意味を持ちます。

| 指定 | 意味 |
|---|---|
| `in_features=6` | 層へ渡す1標本あたりの特徴が6個 |
| `out_features=3` | 層の出力ゲートが3個 |
| `num_inputs=2` | 各出力ゲートが入力特徴を2個ずつ受け取る |

2入力ゲートが3個あるので、入力端子は合計6個です。ただし、複数のゲートが同じ入力特徴を使っても構いません。入力特徴を6個すべて使い切ることは保証されません。

## 2. 固定Dense接続

### 2.1 接続先を整数で記録する

固定接続は、層を作るときに接続先を決め、その後の学習では変更しません。例えば、次の接続が選ばれたとします。以下の値は説明用であり、上のコードを実行したときに必ずこの接続になるわけではありません。

```text
入力特徴 = [x0, x1, x2, x3, x4, x5]

ゲート0の入力 = (x0, x2)
ゲート1の入力 = (x2, x5)
ゲート2の入力 = (x4, x1)

indices = [[0, 2, 4],
           [2, 5, 1]]
```

`indices` の各列が1ゲート、各行がそのゲートの入力端子です。shapeは `[num_inputs, out_features]`、この例では `[2, 3]` です。

この整数Tensorは `Parameter` ではなく、永続的なbufferとして保存します。optimizerで更新されませんが、`state_dict()` に含まれ、重みと一緒に保存・復元されます。

### 2.2 randomとrandom_unique

| `ConnectionConfig.init` | 固定Dense接続の抽選 |
|---|---|
| `random` | 各入力端子の接続先を独立に選ぶ。同じゲートで同じ特徴を複数回選ぶ場合もある |
| `random_unique` | 1ゲート内で同じ特徴を重複して選ばない。別のゲートとの共有は許可する |

`random_unique` は、モデル全体で接続先を重複させないという意味ではありません。2入力なら入力特徴が最低2個、6入力なら最低6個必要です。

固定接続では、接続用の `temperature`・`use_gumbel`・`num_candidates` を変更しません。接続を学習しなくても、ゲートのLUT重みは通常どおり学習できます。

### 2.3 Tensorの形状を追う

batch sizeが4の場合、処理は次のshapeで進みます。

| 段階 | shape | 内容 |
|---|---|---|
| 層への入力 | `[4, 6]` | 4標本、それぞれ6特徴 |
| 接続先から取り出した値 | `[4, 2, 3]` | 4標本、2入力端子、3ゲート |
| 各ゲートのLUT計算後 | `[4, 3]` | 4標本、それぞれ3出力 |

`LogicDense` は最後の軸を特徴として扱います。例えば `[batch, time, 6]` なら、一度先行軸をまとめて処理し、最終的に `[batch, time, 3]` へ戻します。各時刻に別のLUTを作るわけではなく、同じ層の重みと接続を使います。

実装では [connections/dense.py](../../code-reference/utils/logicNN_core/src/logicnn_core/connections/dense.md)が入力を選び、[layers/dense.py](../../code-reference/utils/logicNN_core/src/logicnn_core/layers/dense.md)がLUTの計算と出力shapeへの復元を行います。

## 3. 学習可能なDense接続

### 3.1 候補の集合は固定し、その中から選ぶ

学習可能接続では、各ゲートの各入力端子に複数の候補を用意します。その候補からどれを使うかを、接続用のlogitで決めます。

```python
from logicnn_core.layer_settings import ConnectionConfig, LUTConfig
from logicnn_core.layers import LogicDense

connection_config = ConnectionConfig(kind="learnable", init="random_unique", num_candidates=2)
layer = LogicDense(6, 3, lut=LUTConfig(kind="light", num_inputs=2), connections=connection_config)
```

この設定では、3個のゲートそれぞれに2つの入力端子があり、各端子に2個の候補を用意します。

```text
あるゲートの第1入力：候補 [x0, x2] から1本を選ぶ
あるゲートの第2入力：候補 [x3, x5] から1本を選ぶ
```

候補を記録する `indices` と選択用の `weights` は、どちらも `[num_candidates, num_inputs, out_features]`、この例では `[2, 2, 3]` です。

候補の整数 `indices` は学習中に入れ替わりません。optimizerが更新するのは `weights` です。したがって、この例の第1入力が学習によって `x4` へつながることはありません。`x4` はその端子の候補に含まれていないためです。

### 3.2 候補数を省略した場合

`ConnectionConfig(kind="learnable")` のように `num_candidates` を省略すると、各入力端子が全入力特徴を候補として持ちます。入力が6特徴なら、各端子の候補は `[x0, x1, x2, x3, x4, x5]` です。

明示的に候補数を指定した場合は、次のように抽選します。

| 指定 | 候補集合の作り方 |
|---|---|
| `init="random"` | 候補を独立に抽選する。候補内に同じ入力特徴が現れる場合がある |
| `init="random_unique"` | 1ゲート内の全入力端子・全候補を通して、同じ入力特徴を重複させない |

学習可能な `random_unique` では、`num_candidates` の明示が必要です。また、`in_features >= num_inputs * num_candidates` を満たす必要があります。例えば2入力・候補2個なら、最低4特徴が必要です。

これは固定Denseの `random_unique` より強い条件です。「最終的に選ぶ2本だけが異なる」のではなく、「そのゲートの候補集合全体で重複させない」設定だからです。

### 3.3 順方向では、混合せず1本を選ぶ

現行の学習可能接続は、学習時でも最大logitの候補を1本選んで、その入力値をそのまま渡します。softmaxの比率で複数の入力を混ぜているわけではありません。

例えば候補が `x0=0.2`、`x1=0.8`、logitが `[0.1, 0.9]` なら、`x1` を選んで `0.8` を渡します。

`use_gumbel=True` の場合は、学習時だけlogitへGumbelノイズを加えてから選びます。候補を探索するために、同じ重み・入力でも別の接続が使われることがあります。通常評価と `get_indices()` はノイズなしの最大logitを選び、同値なら最初の候補を使います。

### 3.4 接続を選ぶ処理には代替勾配を使う

最大値の位置を選ぶ `argmax` は、そのままでは接続の学習に必要な勾配を提供できません。この実装では、順方向は1本選択のままにし、逆伝播だけ専用の計算へ置き換えます。これはtorchlogix由来の実装であり、通常のsoftmax混合の微分ではありません。

ここでは1ゲートの1入力端子だけを考えます。候補を `k`、標本を `b`、後段から届く勾配を `g[b]` とすると、接続logitへの勾配は次の式です。

```text
接続logitの勾配[k] = sum_b((2 * 候補入力[b, k] - 1) * g[b])
```

通常の勾配は「この値を少し増やしたとき、損失がどちらへ変わるか」を表します。ただし、ここでは離散的な接続選択の厳密な微分ではなく、接続重みの更新に使う代替値を定義しています。例えば候補入力が `[0.2, 0.8]`、後段の勾配が `0.5` の1標本では、logitへの代替勾配は `[-0.3, 0.3]` です。選ばれていない候補にもこの値が計算されます。

入力側へ戻す勾配は、次のように候補のsoftmax比率で分配します。

```text
p[k] = softmax((logit[k] + 学習時に使ったノイズ[k]) / temperature)
候補入力[k]へ加える勾配 = p[k] * g
```

同じ元入力を複数の候補・端子・ゲートが使う場合は、その寄与を加算します。

接続の `temperature` は、この逆伝播時の配分に使います。ノイズなしの順方向で、温度を変えるだけで最大logitの接続先が変わることはありません。ゲートのLUT用 `temperature` とは別に管理します。

詳細は [connections/dense.py の `_LearnableConnectionFunction`](../../code-reference/utils/logicNN_core/src/logicnn_core/connections/dense.md)を参照してください。初めて使用する場合、独自の逆伝播を書く必要はなく、`ConnectionConfig(kind="learnable")` を層へ渡せばこの処理が使われます。

## 4. Convは局所領域を論理木で処理する

### 4.1 通常の畳み込みと何が違うか

`LogicConv2d` は、画像の小さな領域を順に処理します。ただし、領域内の画素と実数のカーネル重みを掛けて足す通常の畳み込みではありません。領域内から入力を選び、複数段の論理ゲートでまとめます。

この「前段の出力を次段のゲートへ渡して、最後に1出力へまとめる構造」を論理木と呼びます。

```text
選ばれた4入力 → 2入力ゲートを2個 → 2入力ゲートを1個 → 1出力
```

`num_inputs=2`、`tree_depth=2` なら上の構造です。1つの出力位置・1つの出力channelあたり、入力端子は4個、論理ゲートは合計3個になります。

ここでいう4入力は論理木全体の入力です。各ゲートが4入力になるわけではありません。また、接続の抽選方法によっては、4つの入力端子が互いに異なる画素を参照するとは限りません。

### 4.2 木の深さと入力数

1ゲートの入力数を `r`、木の深さを `d` とすると、木の最初に必要な入力端子は `r ** d` 個です。各段のゲート数は、入力側から `r ** (d - 1)`、`r ** (d - 2)`、最後が1個になります。

| 設定 | 木の入力端子数 | 入力側からのゲート数 | ゲート合計 |
|---|---:|---|---:|
| 2入力・深さ1 | 2 | 1 | 1 |
| 2入力・深さ2 | 4 | 2 → 1 | 3 |
| 2入力・深さ3 | 8 | 4 → 2 → 1 | 7 |
| 4入力・深さ2 | 16 | 4 → 1 | 5 |
| 6入力・深さ2 | 36 | 6 → 1 | 7 |

これらのゲート数は、1出力channelの1つの木についての値です。同じ木のLUTと局所接続を、画像内の各出力位置で共有します。出力channelが異なれば、別のLUTと局所接続を持ちます。

### 4.3 小さな画像を処理する例

```python
import torch

from logicnn_core.layer_settings import ConnectionConfig, LUTConfig
from logicnn_core.layers import LogicConv2d

lut         = LUTConfig(kind="raw", num_inputs=2)
connections = ConnectionConfig(init="random_unique")
layer = LogicConv2d((8, 8), 1, 2, 2, 3, stride=1, padding=0, lut=lut, connections=connections)
inputs = torch.zeros(4, 1, 8, 8)
outputs = layer(inputs)
print(tuple(outputs.shape))  # (4, 2, 6, 6)
```

位置引数の意味は、`input_size=(8, 8)`、`in_channels=1`、`out_channels=2`、`tree_depth=2`、`kernel_size=3` です。

3×3の局所領域を、8×8画像の上で1画素ずつ動かします。縦横とも、出力サイズは次の式で6になります。

```text
出力サイズ = floor((入力サイズ + 2 * padding - kernel_size) / stride) + 1
           = floor((8 + 0 - 3) / 1) + 1
           = 6
```

ここで `floor` は、小数部分を切り捨てる操作です。`padding` は周囲へ0を追加し、`stride` は領域を動かす間隔を決めます。現行実装では、`kernel_size` はpadding前の入力サイズ以下で指定します。

### 4.4 Conv内部のTensor

上の例は、4標本・2出力channel・36出力位置・深さ2の論理木です。2入力ゲートなので、処理途中は次のようになります。

| 段階 | shape | 軸の意味 |
|---|---|---|
| 入力画像 | `[4, 1, 8, 8]` | 標本、channel、高さ、幅 |
| 0段目へ渡す入力 | `[4, 2, 2, 36, 2]` | 標本、ゲートの入力端子、出力channel、位置、段内のゲート |
| 0段目の出力 | `[4, 2, 36, 2]` | 標本、出力channel、位置、段内のゲート |
| 1段目へ渡す入力 | `[4, 2, 2, 36, 1]` | 標本、ゲートの入力端子、出力channel、位置、段内のゲート |
| 1段目の出力 | `[4, 2, 36, 1]` | 標本、出力channel、位置、最後のゲート |
| 層の出力 | `[4, 2, 6, 6]` | 標本、出力channel、高さ、幅 |

`tree_weights[level]` は `[その段のゲート数, out_channels, LUT係数数]` です。この例の `raw` なら、0段目が `[2, 2, 16]`、1段目が `[1, 2, 16]` です。位置36個分の重みを別々に持つわけではありません。

コード中の `contraction="fc,bcsf->bcsf"` は、この軸を対応付ける `einsum` の指定です。`f` は段内のゲート、`c` はchannel、`b` は標本、`s` は位置を表します。LUT内部では係数・基底の軸についても積と和を計算し、位置ごとに出力を求めます。

## 5. 固定Conv接続の抽選

現行の `LogicConv2d` と `LogicConv3d` は、固定接続のみ対応しています。Denseと異なり、`ConnectionConfig(kind="learnable")` は使えません。

局所領域内の空間位置と入力channelの組を接続候補とし、最初の段へ渡す入力を選びます。2段目以降は、前段の出力を順に `num_inputs` 個ずつまとめます。後段ごとに画像内の別の位置を再抽選するわけではありません。

### 5.1 channelの範囲を指定しない場合

| `init` | 最初の段の接続 |
|---|---|
| `random` | 空間位置とchannelの全候補から、それぞれ独立に抽選する |
| `random_unique` | 各ゲート内の入力を重複させず、同じ出力channelの最初の段で同一の入力組合せを繰り返さない |

Convの `random_unique` では、**異なるゲートが一部の入力を共有することはあります**。例えば `(x0, x1)` と `(x0, x2)` は異なる組合せなので選べます。木全体で各画素を一度しか使わない、という意味ではありません。

現在の組合せ生成では各組合せを昇順で並べます。ゲート内の入力の並びは真理値表の列に対応するため、接続を理解する際は組合せだけでなく、その順序も確認します。

### 5.2 channel_group_sizeを指定する場合

`ConnectionConfig(channel_group_size=2)` のように指定すると、各出力channelが参照する入力channelを、連続した2channelに限定します。木の入力端子を、その2channelへ同数ずつ割り当てます。

| 条件 | 理由 |
|---|---|
| `channel_group_size <= in_channels` | 存在する入力channelの範囲から選ぶため |
| 木の入力端子数が `channel_group_size` で割り切れる | channelごとに同数を割り当てるため |
| `random_unique` では、channelあたりの入力端子数が局所領域の位置数以下 | 同じchannel内で位置を重複させず選ぶため |

参照するchannel群の開始位置は、`出力channel番号 % (in_channels - channel_group_size + 1)` です。群は重なり得るため、一般的な畳み込みの `groups` と同じ設定ではありません。

この指定は固定Conv専用です。Dense接続や学習可能接続には指定しません。

### 5.3 3Dへの拡張

`LogicConv3d` は、奥行き・高さ・幅を持つ入力を同じ方法で処理します。入力shapeは `[batch, in_channels, depth, height, width]` です。論理木、固定接続、channelごとの重み共有という考え方は2Dと共通です。

コードの詳細は [connections/convolution.py](../../code-reference/utils/logicNN_core/src/logicnn_core/connections/convolution.md)と [layers/convolution.py](../../code-reference/utils/logicNN_core/src/logicnn_core/layers/convolution.md)にあります。

## 6. 固定と学習可能をどう使い分けるか

固定接続は、「配線は初期化時に決め、ゲートの論理関数を学習する」構成です。まずモデル全体の動作を理解する場合は、この2つを分離して考えられます。

学習可能接続は、「候補集合の中で配線も学習する」構成です。候補数、初期化、独自の代替勾配を含むため、固定接続とは学習条件が変わります。精度が必ず上がる機能としてではなく、比較するモデル構造の1つとして扱います。

どちらも、回路出力の時点では接続が確定します。回路内でlogitやsoftmaxを計算して配線先を選び直すわけではありません。LUTの真理値表と確定した接続を保存し、固定した回路として実行します。

次は[クラスとモデルの組み立て](classes.md)で層の役割を整理し、[モデルの処理をコードで追う](reading-code.md)でモデル全体の処理を確認してください。
