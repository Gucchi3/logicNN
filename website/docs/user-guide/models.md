---
title: モデルの追加
description: モデルの共通インターフェースと、モデル定義・登録・設定変更の手順を説明します。
---

# モデルの追加

モデルを追加するときは、`model/` にクラスを定義し、モデルの登録表へ追加します。設定JSONでは登録名だけを選び、層数・層幅・クラス数・LUT方式などの構造はモデルのPythonファイル内で決めます。

ここで説明するのは、`main.py` から実行する**分類用の学習アプリケーション**への追加方法です。汎用ライブラリの `logicnn_core` を単独で使う場合は、アプリケーションの登録表や以下のモデル情報は必須ではありません。

## 1. モデルが提供する情報

[モデルbuilder](/code-reference/model/builder)は、登録されたクラスを**引数なし**で生成します。モデルは `nn.Module` を継承し、次の情報と処理を提供します。

| 項目 | 定義する内容 | 使用箇所 |
|---|---|---|
| `input_shape` | バッチを除く入力形状。正の整数からなるタプル | データとの形状照合、回路変換 |
| `input_size` | `input_shape` の要素数の積 | モデル情報の照合・保存 |
| `num_classes` | 出力クラス数。正の整数 | データとのクラス数照合 |
| `forward(inputs)` | `[batch, *input_shape]` を受け取り、`[batch, num_classes]` のスコアを返す | 学習・検証・最終テスト |
| `architecture_info()` | 実際の層構成と設定を記録したJSON互換の辞書 | 学習情報とチェックポイントの保存・照合 |

`input_shape` と `num_classes` は、既存モデルと同じくクラス属性で定義できます。`input_size` は形状から求める読み取り専用の `property` にすると、二重管理を避けられます。バッチサイズは形状に含めません。

現在の学習処理は入力を `float32`、正解ラベルを `int64` へ変換します。ラベルは `[batch]` のクラス番号、モデル出力は確率化前のスコアとします。モデル内で `argmax` を取ったり、CrossEntropyLossの前にsoftmaxを追加したりする必要はありません。

### 構造情報の書き方

`architecture_info()` には、層の順序、入出力幅、LUT・接続の設定、集約時の `tau` など、構造を理解し照合するための値を記録します。Tensor・モジュール・循環参照・NaN・Infは含めません。

学習開始時には公開属性とデータの形状・クラス数を照合し、構造情報がJSONへ保存できることを確認します。ただし、**構造情報の内容を実際の各層から自動で検証する処理ではありません**。層構成を変更したら、記録する値も同じ定数から更新してください。

初期重みを読み込む際は、モデル名と構造情報も照合します。構造を変更した別モデルへ、既存のチェックポイントをそのまま指定することはできません。詳しい照合処理は[学習ワークフロー](/code-reference/utils/trainer/workflow)を参照してください。

## 2. モデル定義を追加する

以下は `model/lgn/mnist_dense_small.py` に追加する例です。**説明用の未同梱・未登録モデル**であり、現在の `mnist_lgn` を置き換えるものではありません。精度比較用ではなく、追加手順を確認するために小さい構造にしています。

前処理済みMNISTの784要素を100個の論理ゲートへ接続し、10グループに分けてクラススコアを計算します。

```python
"""前処理済みMNISTを小型の論理Denseで分類する追加例。"""

import math
from dataclasses import asdict

from logicnn_core.layer_settings import ConnectionConfig, LUTConfig
from logicnn_core.layers import GroupSum, LogicDense
from torch import Tensor, nn

INPUT_SHAPE         = (1, 28, 28)
NUM_CLASSES         = 10
HIDDEN_FEATURES     = 100
GROUP_TAU           = 10.0
GROUP_BIAS          = 0.0
GRADIENT_SCALE      = 1.0
LUT_SETTINGS        = LUTConfig(kind="raw", num_inputs=2)
CONNECTION_SETTINGS = ConnectionConfig()


class SmallMNISTDense(nn.Module):
    """Flatten・論理Dense・GroupSumで10クラスのスコアを返す。"""

    input_shape: tuple[int, ...] = INPUT_SHAPE
    num_classes: int            = NUM_CLASSES

    def __init__(self) -> None:
        """ファイル内の定数からモデルを構築する。"""
        super().__init__()
        self.network = nn.Sequential(
            nn.Flatten(start_dim=1, end_dim=-1),
            LogicDense(self.input_size, HIDDEN_FEATURES, lut=LUT_SETTINGS, connections=CONNECTION_SETTINGS, gradient_scale=GRADIENT_SCALE),
            GroupSum(NUM_CLASSES, tau=GROUP_TAU, bias=GROUP_BIAS),
        )

    @property
    def input_size(self) -> int:
        """バッチを除く入力の要素数を返す。"""
        return math.prod(self.input_shape)

    def forward(self, inputs: Tensor) -> Tensor:
        """[batch, 1, 28, 28]から[batch, 10]のスコアを返す。"""
        return self.network(inputs)

    def architecture_info(self) -> dict[str, object]:
        """実際の層順序と設定をJSON互換の辞書で返す。"""
        return {
            "input_shape": list(self.input_shape), "input_size": self.input_size, "num_classes": self.num_classes,
            "layers": [
                {"type": "Flatten", "start_dim": 1, "end_dim": -1, "out_features": self.input_size},
                {
                    "type": "LogicDense", "in_features": self.input_size, "out_features": HIDDEN_FEATURES,
                    "lut": asdict(LUT_SETTINGS), "connections": asdict(CONNECTION_SETTINGS), "gradient_scale": GRADIENT_SCALE,
                },
                {"type": "GroupSum", "groups": NUM_CLASSES, "tau": GROUP_TAU, "bias": GROUP_BIAS},
            ],
        }
```

`GroupSum` の直前の特徴数は、グループ数で割り切れる必要があります。この例では `100 / 10 = 10` 個のゲート出力をクラスごとに集約します。

実際のConv・poolingを含む構成は[既存のMNISTモデル](/code-reference/model/lgn/mnist_lgn)を参照してください。

## 3. モデルを登録する

ファイルを作るだけでは、builderから選択できません。次の2か所へ追加します。

### `model/lgn/__init__.py`

新しいクラスをimportし、既存の公開名を残して `__all__` へ追加します。

```python
from .mnist_dense_small import SmallMNISTDense
from .mnist_lgn import MNISTLGN

__all__ = ["MNISTLGN", "SmallMNISTDense"]
```

### `model/builder.py`

`.lgn` からのimportと `MODEL_REGISTRY` を更新します。`build_model()` 自体を変更する必要はありません。

```python
from .lgn import MNISTLGN, SmallMNISTDense

MODEL_REGISTRY: dict[str, type[nn.Module]] = {
    "mnist_lgn": MNISTLGN,
    "mnist_dense_small": SmallMNISTDense,
}
```

別のサブフォルダへ置く場合も、builderへクラスをimportして登録する点は同じです。

## 4. 設定から選択する

既存の `config/mnist_lgn.json` を別名、例えば `config/mnist_dense_small.json` へコピーし、`model.name` を変更します。以下は変更する部分だけで、設定ファイル全体ではありません。

```json
{
  "model": {
    "name": "mnist_dense_small"
  }
}
```

この例では `data.name` は `mnist` のままです。新規モデルの初回実行は `run.initial_checkpoint_path` を `null` にします。

登録後、プロジェクトルートから通常の入口で実行します。

```powershell
python main.py --config config/mnist_dense_small.json
```

## 5. LUTの入力本数を変更する

変更箇所はモデルファイル内の `LUT_SETTINGS` です。入力本数だけを変え、対応しないLUT方式をそのまま使わないようにします。

| ゲートの入力本数 | 対応するLUT方式 |
|---|---|
| 2 | `raw`、`warp`、`light` |
| 4 | `warp`、`light` |
| 6 | `warp`、`light` |

例えば、4入力のWarpへ変更する場合は次の設定にします。

```python
LUT_SETTINGS = LUTConfig(kind="warp", num_inputs=4)
```

6入力なら `num_inputs=6` とします。`LUTConfig(num_inputs=4)` だけでは、既定の `kind="raw"` と組み合わせられないためエラーになります。Warpは1入力にも対応していますが、この追加例は2・4・6入力の構成を対象にしています。

入力本数はゲート内部の入力数であり、画像のチャネル数やバッチサイズではありません。Convの論理木や重複なし接続を使う場合には、受容野の候補数などの条件も確認してください。設定の詳細は[LUTConfig・ConnectionConfig](/code-reference/utils/logicNN_core/src/logicnn_core/layer_settings)にあります。

## 6. 追加後に確認すること

1. 引数なしで生成でき、異なるバッチサイズで `[batch, num_classes]` を返す。
2. `input_shape` と `num_classes` がデータセットのメタデータと一致する。
3. 構造情報が実際の層と一致し、JSONへ保存できる。
4. 小さいバッチで損失計算・逆伝播・評価が動く。
5. 回路出力を使う場合は、CPU上のbool入力で変換でき、C・Verilogを生成できる。

これらは追加時の開発確認です。通常の学習へ毎回の開発者用テストを組み込む必要はありません。

任意のPyTorch層がそのまま回路へ変換できるわけではありません。論理計算にはcoreの対応層を使い、通常の分類学習だけを先に確認するときは `circuit.enabled` を `false` にします。回路生成の内部処理は[Circuit](/code-reference/utils/logicNN_core/src/logicnn_core/circuit/circuit)を参照してください。
