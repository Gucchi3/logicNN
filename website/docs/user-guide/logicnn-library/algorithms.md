---
title: ゲートを学習するアルゴリズム
description: raw・light・warpが何を学習するかを、真理値表、数値例、現行コードの計算式から説明します。
---

# ゲートを学習するアルゴリズム

論理ゲートNNでは、学習によって「このゲートをANDにするか、XORにするか」といった論理関数を決めます。ただし、ANDやXORという名前を直接足し引きすることはできません。そこで、論理関数の選択や出力を実数のパラメータで表し、損失が小さくなるように更新します。

このページでは、[論理ゲートNNの基礎](basics.md)で紹介する真理値表から出発し、現行 `logicnn_core` の `raw`・`light`・`warp` を説明します。以下の式は、このライブラリの実装を読み解くためのものです。同じ名称の研究手法に含まれる機能を、すべて実装しているという意味ではありません。

## 1. 「重み」は何を表すのか

一般的な全結合層の重みは、入力に掛ける係数です。一方、このライブラリの論理層では、**ゲートの計算方法を決める重み**と、**入力の接続先を決める重み**を区別します。

| 種類 | 値が表すもの | 主な保存先 |
|---|---|---|
| `raw` のLUT重み | 16種類の論理関数それぞれの選ばれやすさ | `LogicDense.weight` |
| `light` のLUT重み | 真理値表の各行を1にする度合い | `LogicDense.weight` |
| `warp` のLUT重み | 入力から出力を計算するWalsh基底の係数 | `LogicDense.weight` |
| 学習可能接続の重み | 各入力端子で、どの接続候補を選ぶか | `LogicDense.connections.weights` |

Conv層では、LUT重みを `tree_weights` に段ごとに保存します。LUTの計算方式を表す `parametrization` 自体は重みを所有せず、層から渡された重みで計算します。

接続の学習は[入力接続と畳み込み](connections.md)で説明します。まずは「各ゲートに入力が届いた後、何を計算するか」に注目してください。

## 2. 二値の計算を連続値へ拡張する

二値入力では、ANDは両方が1の場合だけ1を返します。この動作は、入力を `a`、`b` とすると `a * b` でも表せます。

| 論理関数 | 二値入力に対する意味 | この実装で用いる連続式 |
|---|---|---|
| AND | 両方が1なら1 | `a * b` |
| OR | どちらかが1なら1 | `a + b - a * b` |
| XOR | 入力が異なれば1 | `a + b - 2 * a * b` |
| NOT A | Aを反転する | `1 - a` |

`a=0.2`、`b=0.7` とすると、ANDは `0.14`、ORは `0.76`、XORは `0.62` です。これは整数のビット演算ではなく、学習のための連続値の計算です。入力が0か1なら元の論理演算と一致し、途中の値に対しても入力やパラメータを少し変えた影響を計算できます。

このような連続化によって論理ゲートの選択を勾配法で学習する考え方は、[Deep Differentiable Logic Gate Networks](https://arxiv.org/abs/2210.08277)で扱われています。本ライブラリの具体的な16個の式は、[functional/logic.py](../../code-reference/utils/logicNN_core/src/logicnn_core/functional/logic.md)で確認できます。

ここでの通常の入力範囲は0から1です。`LogicDense` や `LogicConv2d` が入力画像を自動的に正規化・二値化するわけではありません。

## 3. raw：16種類の論理関数から選ぶ

### 3.1 16個の重みが必要な理由

2入力の真理値表には、`00`、`01`、`10`、`11` の4行があります。各行の出力を0か1にできるため、論理関数は `2 ** 4 = 16` 種類です。

`raw` は、1ゲートにつき16個のlogitを持ちます。logitは選択用の実数であり、まだ確率ではありません。負の値でも構いません。softmaxによって、合計が1になる混合比率に変換します。

```text
z[i]：論理関数 i のlogit
T   ：temperature
f[i]：論理関数 i の連続式

p[i] = exp(z[i] / T) / sum_j(exp(z[j] / T))
y    = sum_i(p[i] * f[i](a, b))
```

`sum` は指定した項をすべて足す記号です。`exp` は指数関数です。最初は「大きいlogitほど大きい比率になる」と理解すれば十分です。実装では計算を安定させるため、指数関数の前に最大値を差し引きます。

### 3.2 数値で確かめる

ANDのlogitだけを `log(15)`、残り15個を0、温度を1とします。このときANDの比率は `15 / (15 + 15) = 0.5`、残りはそれぞれ `1 / 30` です。

`a=0.2`、`b=0.7` のとき、16個の論理関数の連続出力の合計は8です。したがって、学習時の出力は次のようになります。

```text
ANDの出力   = 0.2 * 0.7 = 0.14
学習時の出力 = 0.5 * 0.14 + (8 - 0.14) / 30 = 0.332
```

評価時は最大logitのANDだけを選ぶので、この連続入力に対する出力は `0.14` です。入力が `a=1`、`b=0` なら、評価時の出力は通常のANDと同じ0になります。

このように、**学習時は複数の関数を混ぜ、評価時は1つの関数を選ぶ**ため、同じ重み・入力でも学習時と評価時で結果が変わることがあります。

### 3.3 評価と真理値表

`model.eval()` では、各ゲートの最大logitを持つ関数を1つ選びます。同値の場合は最初のIDを選び、乱数や温度は使いません。`truth_tables()` も同じ選択結果から真理値表を作ります。

この実装では、ANDはID 1、入力Aの通過はID 3、XORはID 6、ORはID 7です。IDの2進表現を、真理値表の `00`、`01`、`10`、`11` に対応させています。

```text
XORのID = 6 = 0b0110
入力順  = 00, 01, 10, 11
出力    =  0,  1,  1,  0
```

`raw` は2入力専用です。詳しい処理は [parametrizations/raw.py](../../code-reference/utils/logicNN_core/src/logicnn_core/parametrizations/raw.md)にあります。

### 3.4 1個のゲートにANDを学習させる

次は、真理値表を直接設定せず、4組の入力と正解からANDを学習する例です。2入力・1出力の `LogicDense` を作り、出力と正解の差を二乗して平均する `MSELoss` を使います。1個のゲートの出力を学ぶため、分類用の `GroupSum` や `CrossEntropyLoss` は不要です。アプリ全体の損失設定を変更する例ではありません。

[環境構築](../../startup/setup.md)で用意した仮想環境を有効化し、以下を `learn_and.py` として保存して `python learn_and.py` で実行してください。CPUだけで動き、データセットのダウンロードやアプリへのモデル登録は不要です。

```python title="learn_and.py"
"""4組の入出力から、1個の論理ゲートにANDを学習させる。"""

import torch
from torch import nn

from logicnn_core.layer_settings import ConnectionConfig, LUTConfig
from logicnn_core.layers import LogicDense


def main() -> None:
    """200回の重み更新後に、確定したゲートの出力と真理値表を表示する。"""
    torch.manual_seed(0)
    inputs  = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float32)
    targets = torch.tensor([[0], [0], [0], [1]], dtype=torch.float32)
    lut     = LUTConfig(kind="raw", num_inputs=2, weight_init="random")
    layer   = LogicDense(2, 1, lut=lut, connections=ConnectionConfig(init="random_unique"))
    optimizer = torch.optim.Adam(layer.parameters(), lr=0.1)
    criterion = nn.MSELoss()

    layer.train()
    for step in range(200):
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(layer(inputs), targets)
        loss.backward()
        optimizer.step()
        if step in (0, 49, 99, 199):
            print(f"更新 {step + 1:3d} 回目: loss={loss.item():.4f}")

    layer.eval()
    with torch.no_grad():
        outputs = layer(inputs)
    for row, output in zip(inputs.int().tolist(), outputs.int().tolist()):
        print(f"入力 {row} → 出力 {output[0]}")
    print("真理値表:", layer.truth_tables().int().tolist())


if __name__ == "__main__":
    main()
```

現行のCPU環境でこの例を実行し、最終的に次の出力と真理値表になることを確認しています。これはこの小さな例での確認結果であり、任意の設定やデータに対して200回で収束する保証ではありません。

```text
入力 [0, 0] → 出力 0
入力 [0, 1] → 出力 0
入力 [1, 0] → 出力 0
入力 [1, 1] → 出力 1
真理値表: [[0, 0, 0, 1]]
```

学習ループでは16個の候補の混合比率を更新し、最後の `layer.eval()` で1つへ確定しています。今回はANDが選ばれました。固定接続は2入力の順序を入れ替える場合がありますが、ANDは入力を交換しても結果が変わりません。

## 4. light：真理値表の各行を学習する

### 4.1 関数の候補ではなく、表の出力を持つ

`light` は16種類の候補を列挙せず、真理値表の各行にlogitを1つずつ持ちます。2入力なら4個、4入力なら16個、6入力なら64個です。

学習時の標準設定 `sampling="soft"` では、各logitをsigmoidで0から1の値に変換します。

```text
sigmoid(z) = 1 / (1 + exp(-z))
q[row]     = sigmoid(weight[row] / T)
```

sigmoidは大きい正の値を1に近づけ、大きい負の値を0に近づけます。0を渡すと0.5になります。softmaxとは異なり、`q` 全体の合計を1にする処理ではありません。

### 4.2 入力が連続値なら、表の行を補間する

二値入力なら、真理値表の該当する1行を使えば出力が求まります。`a=0.2` のような連続入力では、0の側と1の側の両方を比率付きで使います。

2入力では、行ごとの比率と出力は次の式になります。

```text
phi00 = (1 - a) * (1 - b)
phi01 = (1 - a) * b
phi10 = a * (1 - b)
phi11 = a * b

y = q00 * phi00 + q01 * phi01 + q10 * phi10 + q11 * phi11
```

これを多重線形補間と呼びます。「真理値表の出力を、入力から決まる比率で混ぜる計算」です。各入力が0から1なら4つの比率は非負で、合計は1です。

`a=0.2`、`b=0.7` では、比率は `[0.24, 0.56, 0.06, 0.14]` になります。

| 入力の組合せ | 重み | 学習時の表の値 | 評価時の表の値 |
|---|---:|---:|---:|
| `00` | -2 | 約0.1192 | 0 |
| `01` | 2 | 約0.8808 | 1 |
| `10` | 2 | 約0.8808 | 1 |
| `11` | -2 | 約0.1192 | 0 |

温度1でこの重みを使うと、学習時の出力は約 `0.5914` です。評価時の表はXORなので、同じ連続入力では `0.56 + 0.06 = 0.62` になります。入力を `0, 1` にすると、評価時には該当する行の1が返ります。

### 4.3 評価時に二値化する対象

`light` の評価時は、**表の重みが0より大きいか**で真理値表を確定します。重みが0なら出力0です。入力そのものに `0.5` などの閾値を適用する処理ではありません。

そのため、確定した真理値表を使っていても、入力が連続値なら出力も連続値になり得ます。回路と比較するときは、モデル側と回路側へ同じ二値入力を渡してください。

詳しい処理は [parametrizations/light.py](../../code-reference/utils/logicNN_core/src/logicnn_core/parametrizations/light.md)と [functional/walsh.py](../../code-reference/utils/logicNN_core/src/logicnn_core/functional/walsh.md)にあります。

## 5. warp：入力の組合せを式で表す

### 5.1 係数は「式のどの項を強くするか」を表す

`warp` も1ゲートにつき `2 ** num_inputs` 個の係数を持ちます。ただし、各係数は真理値表の1行に直接対応しません。入力から作った項に掛ける値です。

まず入力を、0が1、1が-1になるように変換します。

```text
A = 1 - 2 * a
B = 1 - 2 * b
```

2入力の場合、この実装は `[1, B, A, A * B]` という順序で項を作ります。このような計算の基本となる項を「基底」と呼びます。

```text
s = c0 + c1 * B + c2 * A + c3 * A * B
y = sigmoid(-s / T)    # sampling="soft" の学習時
```

| 係数 | 掛ける項 | 変えたときに影響する部分 |
|---|---|---|
| `c0` | `1` | すべての入力に共通する値 |
| `c1` | `B` | 2番目の入力によって符号が変わる項 |
| `c2` | `A` | 1番目の入力によって符号が変わる項 |
| `c3` | `A * B` | 2つの入力が同じか異なるかで符号が変わる項 |

例えば、`c3=2` にすると `A * B` の影響が2倍になります。`c3=-1` にすると、その項の符号が反転します。これが「入力から出力を決める計算式の係数」という意味です。ANDを選ぶ確率のような値ではありません。

### 5.2 XORを表す例

`[c0, c1, c2, c3] = [0, 0, 0, 1]` とすると、`s = A * B` です。

| `a` | `b` | `A` | `B` | `s` | 学習時の出力・温度1 | 評価時の出力 |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0 | 1 | 1 | 1 | 約0.2689 | 0 |
| 0 | 1 | 1 | -1 | -1 | 約0.7311 | 1 |
| 1 | 0 | -1 | 1 | -1 | 約0.7311 | 1 |
| 1 | 1 | -1 | -1 | 1 | 約0.2689 | 0 |

評価時の出力はXORです。`c3` を正のまま大きくすると、二値入力に対する学習時の出力も0と1に近づきます。

`a=0.2`、`b=0.7` なら `A=0.6`、`B=-0.4`、`s=-0.24` です。学習時の出力は `sigmoid(0.24)`、約 `0.5597` になります。`light` と同じXORの真理値表を表せても、学習時の連続式が同じとは限りません。

### 5.3 真理値表へ変換する

真理値表を取り出すときは、全二値入力について `s` を求め、**`s` が0より小さい行を1**にします。`s=0` の行は0です。

これを係数全体からまとめて計算するのがWalsh–Hadamard変換です。現行実装は和と差を段階的に計算する `fast_walsh_hadamard()` を使います。2入力なら、係数から次の4つの値を作る計算に相当します。

```text
s00 = c0 + c1 + c2 + c3
s01 = c0 - c1 + c2 - c3
s10 = c0 + c1 - c2 - c3
s11 = c0 - c1 - c2 + c3
```

`truth_tables()` は係数をCPUのfloat64へコピーして変換し、boolの表を元のdeviceへ戻します。通常評価でも、二値入力に対してはこの表を参照します。連続入力に対してはWalshの式を計算して符号判定します。この違いは、二値入力を使った通常評価と回路出力で、同じ真理値表を使用するための実装上の工夫です。

Walsh表現を論理NNの学習に利用する研究については、[WARP Logic Neural Networks](https://arxiv.org/abs/2602.03527)を参照してください。本ライブラリの符号・係数順・評価経路は、[parametrizations/warp.py](../../code-reference/utils/logicNN_core/src/logicnn_core/parametrizations/warp.md)と [functional/walsh.py](../../code-reference/utils/logicNN_core/src/logicnn_core/functional/walsh.md)が基準です。

## 6. 方式と入力数を選ぶ

| `LUTConfig.kind` | 対応する入力数 | 1ゲートの学習パラメータ数 | 確定する方法 |
|---|---|---|---|
| `raw` | 2 | 16 | 最大logitの論理関数を選ぶ |
| `light` | 2、4、6 | 4、16、64 | 真理値表の各重みが正なら1 |
| `warp` | 1、2、4、6 | 2、4、16、64 | Walshの計算結果が負なら1 |

入力数が増えても、ゲートの出力は1個です。`num_inputs=4` は4クラスの分類を意味せず、各ゲートに4本の信号を接続するという意味です。

まず2入力の `raw` で「候補を混ぜてから1つに決める」処理を理解し、次に `light` と `warp` の表現を比べると読み進めやすくなります。方式を変えれば必ず精度が上がるという関係ではありません。実際の精度はモデル構造・初期化・学習条件と合わせて評価します。

```python
from logicnn_core.layer_settings import LUTConfig

raw_config   = LUTConfig(kind="raw", num_inputs=2)
light_config = LUTConfig(kind="light", num_inputs=4)
warp_config  = LUTConfig(kind="warp", num_inputs=6)
```

`LUTConfig(num_inputs=4)` だけでは、既定の `kind="raw"` と対応しないため使用できません。方式と入力数を一緒に指定してください。

## 7. 温度とsamplingは何を変えるのか

### 7.1 温度

`LUTConfig.temperature` は、softmaxやsigmoidの変化の鋭さを調整します。同じ重みなら、小さい正の温度ほど選択が明確になり、大きい温度ほど中間的な値になりやすくなります。学習率とは別の値です。

温度を小さくするだけで、学習が必ず改善するわけではありません。値が0や1に近づきすぎると、重みの小さな変化が出力に伝わりにくくなる場合があります。

既に作った層では `layer.parametrization.set_temperature(value)` で変更できます。アプリのcosine annealing schedulerはoptimizerの学習率を更新するもので、LUTの温度を自動更新する設定ではありません。

### 7.2 softとhard

| `sampling` | 学習時の計算 | 乱数 |
|---|---|---|
| `soft` | softmaxまたはsigmoidの連続値を使う | 使わない |
| `hard` | 選択を離散化し、逆伝播には連続式の勾配を使う | 使わない |
| `gumbel_soft` | ノイズを加えてから連続値を求める | 使う |
| `gumbel_hard` | ノイズを加えて離散選択し、逆伝播には連続式の勾配を使う | 使う |

`hard` で用いる「計算結果は離散値、勾配は連続式のもの」という処理をstraight-through推定と呼びます。離散化そのものの厳密な微分を計算しているわけではありません。

また、離散化する場所は方式で異なります。`raw` は論理関数の選択、`light` は表の各行、`warp` はWalsh和から求めた出力です。`raw` と `light` では、`hard` でも入力が連続値なら最終出力が連続値になる場合があります。

実装上、`raw` のGumbel系samplingはlogitへGumbelノイズを加えます。`light` と `warp` の二値選択では、一様乱数のlog oddsから作るLogisticノイズを使います。どちらも通常評価と真理値表の生成には乱数を使いません。

対応するコードは [functional/sampling.py](../../code-reference/utils/logicNN_core/src/logicnn_core/functional/sampling.md)です。

## 8. 初期化は学習の出発点を決める

`LUTConfig.weight_init` は、学習開始前のLUT重みを決めます。接続先の初期化とは別です。

| 指定 | `raw` | `light` | `warp` |
|---|---|---|---|
| `random` | 標準正規乱数 | 0以上1未満の一様乱数 | 標準正規乱数 |
| `residual` | 入力Aを通過させるID 3を優先 | 表の前半へ-3、後半へ+3を加えた正規乱数 | 入力Aに対応する係数を設定 |
| `residual_catalog` | 非対応 | 非対応 | 入力Aを通すゲートを基本とし、一部をランダムな真理値表の係数に置き換える |

残差初期化は、学習の出発点を入力の通過に近づけるものです。別の枝を追加する `ResidualLogicBlock` モジュールとは異なります。

`residual_probability` の意味は方式で共通ではありません。`raw` と `warp` の残差初期化では、標準のsoft計算で二値の入力Aに対する出力を、おおむね `1 - p` と `p` にするように設定します。`raw` の「ID 3を選ぶ比率そのもの」が `p` という意味ではありません。例えば `p=0.951` なら、ID 3の初期比率は約 `0.9081` です。

`light` の初期化は上表の固定した±3を使い、`residual_probability` を計算に使用しません。`warp` の `residual_catalog` では、おおむね `1 - p` の割合を別の真理値表に置き換えます。これらは現行コードの挙動として区別してください。

## 9. ゲートの出力から損失へつなぐ

一般的な分類モデルでは、各ゲートへ個別に正解の真理値表を与えるのではなく、モデル全体の出力と正解ラベルから損失を求め、その勾配を前の層まで伝えます。先ほどのANDの例は、1個のゲートの入出力を学ぶための小さな教材です。

```text
入力の前処理 → 接続先の選択 → LUTの計算 → 必要な層を繰り返す
                                      → GroupSum → 分類損失
```

分類モデルの最後では、複数のゲートの出力をクラスごとに加算します。例えば、6出力を `GroupSum(2, tau=2)` に渡すと、先頭3個と後半3個を別々に足して2で割ります。

```text
ゲート出力 = [1, 0, 1, 0, 1, 0]
クラス0    = (1 + 0 + 1) / 2 = 1.0
クラス1    = (0 + 1 + 0) / 2 = 0.5
予測クラス = 0
```

一般形は `score = (group内の合計 + bias) / tau` です。`GroupSum` は学習可能な重みを持ちません。`tau` は分類スコアの大きさを調整する値で、LUTの `temperature` とは別です。

学習時は、このスコアを `CrossEntropyLoss` などへ渡し、`loss.backward()` で勾配を求め、`optimizer.step()` で重みを更新します。`CrossEntropyLoss` へ渡す前に自分でsoftmaxを掛ける必要はありません。実行可能な一連の例は[ライブラリの直接利用](../core-library.md)にあります。

## 10. 入力の二値化とLUTの離散化を分ける

| 処理 | 何を決めるか | 例 |
|---|---|---|
| 入力の二値化 | 入力値を0か1へ変換する | 明るさが閾値より大きければ1 |
| LUTの離散化 | 学習したゲートの論理関数を確定する | 最大logitのANDを選ぶ |
| 接続の確定 | 学習可能接続の候補を1本へ決める | 入力特徴7を選ぶ |

MNIST用アプリでは、データ側の `binarize_greater_than()` で入力を二値化します。ライブラリを直接使う場合は、`FixedBinarization`、`SoftBinarization`、`LearnableBinarization` なども選べます。

固定二値化は閾値比較そのものであり、通常の連続的な入力勾配を提供しません。`SoftBinarization` は学習中の比較をsigmoidに置き換えます。`LearnableBinarization` は閾値を決めるパラメータも更新します。各クラスの使い方と注意点は[クラスとモデルの組み立て](classes.md)を参照してください。

回路出力では、モデルと同じ前処理で作った二値入力を使います。画像のUINT8値をそのまま0/1回路へ渡すのではありません。

## 11. 任意の補助処理

通常の学習ループだけで始められます。以下の機能はライブラリにありますが、呼び出さなければ実行されません。

| 機能 | 実際の意味 |
|---|---|
| `gradient_scale` | 層の入力へ戻す勾配を指定倍率にする。順方向の出力は変えない |
| `regularization_loss("L2")` | ゲートごとに `(1 - sum(weight ** 2)) ** 2` を計算して平均する |
| `regularization_loss("abs_sum")` | ゲートごとに `(1 - abs(sum(weight))) ** 2` を計算して平均する |
| `rescale_weights_("clip")` | LUT重みを-1から1へ制限する |
| `rescale_weights_("L2")` | LUT重みをゲートごとのL2ノルムで割る |
| `rescale_weights_("abs_sum")` | LUT重みをゲートごとの和の絶対値で割る |

ここでの `L2` 正則化は、一般的な「重みの二乗和を小さくするペナルティ」と同じではありません。また、`abs_sum` は絶対値の和ではなく、和の絶対値です。再スケールで割る値が0になる設定にも注意が必要です。

正則化を使う場合は返り値を分類損失へ明示的に加えます。再スケールは明示的に呼び出した時点で重みを変更します。いずれも接続のlogitを変更する処理ではありません。`regularization_loss("warp")` は未実装で、`kind="warp"` のLUT本体とは別の機能です。

詳しい処理は [functional/regularization.py](../../code-reference/utils/logicNN_core/src/logicnn_core/functional/regularization.md)にあります。

## 12. 次に読むページ

次は[入力接続と畳み込み](connections.md)で、各ゲートへ渡す入力がどのように選ばれるかを確認してください。層を組み合わせて動かす場合は、[クラスとモデルの組み立て](classes.md)と[ライブラリの直接利用](../core-library.md)へ進めます。
