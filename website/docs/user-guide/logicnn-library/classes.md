---
title: クラスとモデルの組み立て
description: PyTorchの基本から、logicnn_coreの設定・層・接続・LUT表現・回路クラスの関係を説明します。
---

# クラスとモデルの組み立て

論理ゲートNNのコードは、「入力の接続先」「ゲートの計算方法」「層の並び」を分けて読むと理解しやすくなります。このページでは、概念と実際のクラスを対応付け、小さなモデルを自分で組み立てるための知識を整理します。論理ゲートや真理値表がまだ分からない場合は、先に[論理ゲートNNの基礎](basics.md)を参照してください。

ここで扱うライブラリのPythonパッケージ名は `logicnn_core` です。実装は `utils/logicNN_core/src/logicnn_core/` にあります。学習アプリの設定JSONとは別に、層の設定はPythonコードで指定します。

## 1. 先に押さえるPyTorchの基本

### Tensor：複数の値を並べたデータ

`Tensor` は、数値の配列と、その形状・型・計算するデバイスを持つオブジェクトです。

| 表記 | 意味 | 例 |
|---|---|---|
| `shape` | 各軸の要素数 | 画像2枚なら `(2, 1, 28, 28)` |
| `dtype` | 要素の型 | `torch.float32`、`torch.bool` |
| `device` | データを保持し計算する場所 | `cpu`、`cuda` |
| `batch` | 一度に処理する標本の数 | 上の例では2枚 |

以降の表ではバッチサイズを `B` と書きます。`[B, 4]` は「各標本に4個の入力があり、それをB標本まとめた形状」です。`[B, C, H, W]` は「標本・チャネル・高さ・幅」の順です。

**値が0または1であることと、型が `bool` であることは別です。** 学習アプリは0.0・1.0を `float32` で渡します。回路出力モードの論理層には `bool` を渡します。

### nn.Module：計算と状態をひとまとめにするクラス

PyTorchの層やモデルは `nn.Module` を継承します。

| 記述 | 意味 |
|---|---|
| `__init__()` | 層を生成した時点で、重みや内部の層を用意する |
| `forward(inputs)` | 入力から出力を計算する手順を書く |
| `model(inputs)` | PyTorchの呼び出し処理を経て `forward()` を実行する |
| `nn.Sequential(...)` | 複数の層を記述順に呼び出す |
| `model.to(device)` | 登録されたパラメータとバッファを指定デバイスへ移す |

通常の実行では `model.forward(inputs)` ではなく、`model(inputs)` と書きます。自作クラスの `__init__()` では `super().__init__()` を呼んでから、内部の層を `self.network` などへ登録します。

### Parameterとbuffer：どちらもモデルの状態だが、用途が違う

| 種類 | 主な用途 | このライブラリの例 |
|---|---|---|
| `nn.Parameter` | 通常、勾配を使って学習する値 | `LogicDense.weight`、学習可能な接続の `weights` |
| buffer（バッファ） | モデルと一緒に保持・移動するが、通常のoptimizerでは更新しないTensor | 固定接続の `indices`、固定二値化の `thresholds` |
| 通常の属性 | 構造や動作を指定する値 | `in_features`、`num_inputs`、`GroupSum.tau` |

`model.parameters()` にはParameterが列挙され、これをoptimizerへ渡します。`state_dict()` にはParameterと永続バッファが入ります。たとえば固定接続の番号も保存されるため、同じゲート重みだけでなく、同じ配線を復元できます。

`state_dict()` はモデル構造のPythonコードそのものではありません。通常の属性がすべて自動保存されるわけでもないため、復元時には同じ構造のモデルを用意します。アプリは構造情報も別途記録・照合します。

## 2. 1つの論理層を構成する3つの要素

`LogicDense` は、内部に「接続を選ぶオブジェクト」と「LUTを計算するオブジェクト」を持ちます。

| 要素 | 決めること | 実際のコード |
|---|---|---|
| 層 | 入出力幅、重みの保持、処理の組み合わせ | `LogicDense`、`LogicConv2d`、`LogicConv3d` |
| 接続 | 各ゲートの入力をどこから取得するか | `layer.connections` |
| LUT表現 | 学習用の重みを、どのような論理計算へ変換するか | `layer.parametrization` |

処理は概念的に「入力 → 接続先の値を取り出す → LUTを計算する → 層の出力を返す」と進みます。接続は入力値を選び、LUTは選ばれた値から出力値を求めます。この2つは別の処理です。

### LUTConfigとConnectionConfig

次のコードは4個の入力特徴から8個の出力を作る層です。各出力は2入力ゲートで計算します。

```python title="inspect_layer.py"
"""論理Dense層の形状と、学習する値・固定する値を確認する。"""

import torch

from logicnn_core.layer_settings import ConnectionConfig, LUTConfig
from logicnn_core.layers import LogicDense

torch.manual_seed(0)
lut         = LUTConfig(kind="raw", num_inputs=2)
connections = ConnectionConfig(kind="fixed", init="random_unique")
layer       = LogicDense(4, 8, lut=lut, connections=connections)
inputs      = torch.tensor([[0, 1, 0, 1], [1, 0, 1, 0]], dtype=torch.float32)

print("出力:", tuple(layer(inputs).shape))
print("Parameter:", [(name, tuple(value.shape)) for name, value in layer.named_parameters()])
print("buffer:", [(name, tuple(value.shape)) for name, value in layer.named_buffers()])
```

出力例は次のとおりです。

```text
出力: (2, 8)
Parameter: [('weight', (8, 16))]
buffer: [('connections.indices', (2, 8))]
```

8個のゲートがあり、それぞれに16個のRaw方式の重みを持つため、`weight` は `[8, 16]` です。接続番号は「2本の入力端子 × 8個のゲート」なので `[2, 8]` です。入力特徴数4と、1ゲートの入力本数2を混同しないでください。

`LUTConfig` と `ConnectionConfig` は変更不可の設定オブジェクトです。作成済みの設定に代入するのではなく、新しい設定を作って層を構築します。実行中の温度を変更する必要がある場合は、対応する `set_temperature()` を使います。温度の変更は、作成時のconfig値を書き換える操作ではありません。

各設定値の詳細は[設定クラス](/code-reference/utils/logicNN_core/src/logicnn_core/layer_settings)を参照してください。

## 3. モデルを組み立てる層

### LogicDense：特徴の並びを論理ゲートで変換する

`LogicDense(in_features, out_features, ...)` は、末尾の特徴軸を変換します。通常の例では `[B, in_features]` から `[B, out_features]` になります。入力が `[B, T, in_features]` のように追加の軸を持つ場合も、先行する `[B, T]` を保ちます。

通常の全結合層 `nn.Linear` と異なり、各出力が全入力の重み付き和を計算するわけではありません。各ゲートは、接続設定で選ばれた `num_inputs` 本だけを使います。

`LogicDense(4, 8)` の「8」はクラス数とは限りません。次の層へ渡す特徴数です。分類なら、後段に `GroupSum` を置いてクラス数へまとめられます。

詳細：[LogicDense](/code-reference/utils/logicNN_core/src/logicnn_core/layers/dense)

### LogicConv2d・LogicConv3d：局所的な入力を論理木で処理する

画像や立体データでは、全体を平らにする前に、近くの値の組み合わせを調べることができます。論理Convは受容野と呼ぶ小さな領域から入力を選び、複数段の論理ゲートで計算します。

| 設定・属性 | 意味 |
|---|---|
| `input_size` | バッチ・チャネルを除く空間形状。2Dなら `(H, W)` |
| `in_channels` / `out_channels` | 入力と出力のチャネル数 |
| `kernel_size` | 入力を選ぶ受容野の大きさ |
| `stride` / `padding` | 受容野を動かす間隔と、周囲に追加する0の幅 |
| `tree_depth` | 1つの受容野を処理する論理木の段数 |
| `tree_weights` | 論理木の段ごとに保持する学習パラメータ |

2Dの入出力は `[B, in_channels, H, W]` と `[B, out_channels, H_out, W_out]`、3Dでは空間軸が3本になります。畳み込みのLUT重みと受容野内の相対的な接続位置は、空間位置を移動しても共有されます。

`tree_depth=2` はモデル全体が2層という意味ではありません。2入力ゲートなら、1つの論理木が「下段の2ゲート → 上段の1ゲート」という構造を持つという意味です。現行の論理Convが対応する接続は固定接続です。

詳細：[論理Conv](/code-reference/utils/logicNN_core/src/logicnn_core/layers/convolution)

### OrPooling2d・OrPooling3d：近くの値をまとめて空間を縮小する

通常モードでは、窓内の最大値を取ります。たとえば `OrPooling2d(2, 2)` は2×2の領域を1つの値へまとめ、窓を2ずつ動かします。回路出力モードでは、同じ領域をboolのORでまとめます。入力が0/1なら最大値とORは同じ結果になります。

学習可能な重みはありません。値が連続値の学習時に、残差結合の `a + b - a * b` を使う層ではない点に注意してください。

詳細：[OR pooling](/code-reference/utils/logicNN_core/src/logicnn_core/layers/pooling)

### GroupSum：複数の出力をクラススコアへまとめる

`GroupSum(groups, tau=1.0, bias=0.0)` は末尾の特徴を連続したグループへ等分します。出力幅は `groups` になります。入力幅は `groups` で割り切れる必要があります。

たとえば `GroupSum(2, tau=2.0)` に8特徴を渡すと、先頭4個と末尾4個を別々に足し、それぞれを2で割ります。計算順は次のとおりです。

```text
各グループのスコア = (グループ内の出力の和 + bias) / tau
```

この層は学習パラメータを持ちません。スコアは確率ではなく、最大スコアの位置を `argmax` で取ると予測クラス番号になります。Verilogでは、この加算より前の論理出力を取り出す設計です。

詳細：[GroupSum](/code-reference/utils/logicNN_core/src/logicnn_core/layers/group_sum)

### ResidualLogicBlock：主経路と別経路をORで結ぶ

`ResidualLogicBlock` は主経路の2つの論理Convと、入力を別経路から渡すshortcutを組み合わせます。通常モードでは `main + shortcut - main * shortcut`、回路出力モードではboolのORで結合します。一般的な残差ブロックの単純加算とは異なります。

両経路の出力形状は一致させます。省略時のshortcutは `nn.Identity()`、つまり入力をそのまま返す処理です。チャネル数や空間サイズを変える場合は、それに合った `projection` を利用者が指定します。自動的に補正用の層が追加されるわけではありません。

`downsample=True` では主経路の各Convの後にpoolingが入り、poolingは合計2回です。shortcutの形状も最終出力に合わせてください。

詳細：[ResidualLogicBlock](/code-reference/utils/logicNN_core/src/logicnn_core/modules/residual)

## 4. 二値化クラスとアプリの前処理

入力値をビットへ変える方法も、データやモデルに応じて選べます。

| クラス | 学習中 | 評価時 |
|---|---|---|
| `FixedBinarization` | 固定閾値との比較で0/1を返す | 同じ固定閾値で比較する |
| `SoftBinarization` | 固定閾値との差をsigmoidで連続値へ変える | 固定閾値で比較する |
| `LearnableBinarization` | 閾値を生成する差分パラメータも学習する | 学習済み差分から閾値を求めて比較する |
| `DummyBinarization` | 値をfloat32へ変換するだけ | 値をfloat32へ変換するだけ |

閾値比較は `入力 > 閾値` です。等しい場合は0になります。閾値を複数用意すると、1つの入力から複数ビットを作るthermometer encodingができます。これは整数の通常の2進表現とは異なり、各ビットが「それぞれの閾値を超えたか」を表します。

たとえば値0.7を閾値 `[0.25, 0.5, 0.75]` と比較すると、`[1, 1, 0]` になります。入力 `[B, F]` にこの3閾値を使い、既定の `feature_dim=-2` で特徴軸へ結合すると出力は `[B, 3 * F]` になります。画像ではチャネル軸へ結合したい場合など、`feature_dim` を明示して形状を合わせます。

`LearnableBinarization` の学習時と評価時の閾値生成式は同一ではありません。高度な使い方では、[二値化クラスのコードリファレンス](/code-reference/utils/logicNN_core/src/logicnn_core/layers/binarization)で式と軸の扱いを確認してください。

現在のMNISTモデルは、これらの層を内部に持ちません。アプリの [binarize_greater_than()](/code-reference/utils/data/preprocessing) であらかじめ二値化します。また、`Circuit` による変換は二値入力を前提とするため、画像の読み込みや任意の実数前処理まで自動的にC・Verilogへ移す機能ではありません。

## 5. 層の内側にある接続・LUT表現

### 接続クラス

| クラス | 実際に保持する値 | 学習で変わる部分 |
|---|---|---|
| `FixedDenseConnections` | 各端子が参照する入力番号 | なし |
| `LearnableDenseConnections` | 入力候補の番号と、候補を選ぶ重み | 候補を選ぶ重み |
| `FixedConvConnections` | 受容野内の位置、各空間位置、論理木の接続 | なし |

`random_unique` の固定Dense接続では、1つのゲート内の入力番号が重複しないように選びます。異なるゲート同士で同じ入力を使うことは禁止しません。

学習可能なDense接続も、forwardで候補の値を単純に平均する方式ではありません。各端子が1本を選び、学習には専用の代替勾配を使います。通常はこれらのクラスを直接生成せず、`ConnectionConfig` を層へ渡します。

考え方は[入力接続の解説](connections.md)、実装の詳細は[Dense接続](/code-reference/utils/logicNN_core/src/logicnn_core/connections/dense)・[Conv接続](/code-reference/utils/logicNN_core/src/logicnn_core/connections/convolution)を参照してください。

### LUTParametrizationとその実装

`Parametrization` は、ここでは「論理関数をどのような学習パラメータで表現するか」を意味します。

| クラス | 学習パラメータの意味 | 対応する入力本数 |
|---|---|---|
| `RawLUTParametrization` | 16種類の2入力論理関数の選択用logit | 2 |
| `LightLUTParametrization` | 真理値表の各行の出力を決めるlogit | 2・4・6 |
| `WarpLUTParametrization` | Walsh基底の係数 | 1・2・4・6 |

これらのクラスは計算方法を提供し、LUTの学習重みそのものは層の `weight` または `tree_weights` に置かれます。`layer.truth_tables()` は、現在の重みを離散化したboolの真理値表を返します。Denseでは全ゲート分をまとめた1つのTensor、Convでは論理木の段ごとのTensorを並べたリストになります。

数式と数値例は[ゲートを学習するアルゴリズム](algorithms.md)で説明します。初めてソースを読む場合は、16候補から選ぶRaw方式の [raw.py](/code-reference/utils/logicNN_core/src/logicnn_core/parametrizations/raw) を先に読むと、層・接続・LUTの分担を追いやすくなります。共通のinterfaceは [LUTParametrization](/code-reference/utils/logicNN_core/src/logicnn_core/parametrizations/base) にあります。

## 6. Circuit：学習済みモデルとは別の回路オブジェクト

`Circuit` は `nn.Module` ではありません。学習済みの論理計算を、配線・論理演算・加算などのデータとして保持します。optimizerで学習する対象でもありません。

| 操作 | 用途 |
|---|---|
| `Circuit.from_model(model, input_shape)` | モデルの独立コピーから回路を作る |
| `circuit.evaluate(inputs)` | 二値入力から集約後の出力をCPUで求める |
| `circuit.evaluate_logical_outputs(inputs)` | 加算前の論理出力を元の順番で得る |
| `circuit.simplify()` | 出力の意味を保って回路を簡略化する |
| `circuit.write_json(...)` | 再読み込み可能な回路データを保存する |
| `circuit.write_c(...)` / `write_verilog(...)` | C・Verilogのソースコードを保存する |

内部データの `CircuitData` は回路全体、`Gate` は1つの論理演算、`SumReduction` は論理出力の加算と後続の数値演算を表します。`LogicalOutput` は加算前の元のビット列との対応を保持します。通常の利用で、これらを手作業で構築する必要はありません。

モデルを学習し直しても、既に作った `Circuit` は自動更新されません。新しい重みを反映するには再度変換します。

詳細：[Circuit](/code-reference/utils/logicNN_core/src/logicnn_core/circuit/circuit)、[回路データの型](/code-reference/utils/logicNN_core/src/logicnn_core/circuit/types)

## 7. 次にコードを読むときの着眼点

クラスを開いたら、まず「どのTensorが重みか」「接続番号はどこにあるか」「forwardはどのオブジェクトを呼ぶか」を確認してください。最初から保存・復元や回路出力用の内部処理をすべて理解する必要はありません。

次の[モデルの処理をコードで追う](reading-code.md)では、小型Denseモデルと実際の `MNISTLGN` を使って、形状の変化と呼び出し順を確認します。実際に一連の処理を試す場合は[ライブラリの直接利用](../core-library.md)へ進んでください。
