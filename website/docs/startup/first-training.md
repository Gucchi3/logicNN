---
title: 最小構成での実行
description: MNISTをCPUで1 epoch学習し、評価と保存までの一連の処理を確認します。
pagination_prev: startup/dataset
pagination_next: startup/results
---

# 最小構成での実行

このページでは、CPUで1 epochの学習を実行します。初回の目的は、データの読み込みから学習、評価、結果の保存までが正常に動作することを確認することです。

ここでの「最小構成」は、学習回数を1 epochにし、回路出力を省く構成です。モデルを小型化したり、データを一部だけに制限したりするものではありません。1 epochとは、学習用データ全体を一巡して重みを更新する単位です。

## 1. 初回実行用の設定ファイルを用意する

エディタで `config/mnist_startup.json` を新規作成し、次の内容をUTF-8で保存してください。通常設定の `config/mnist_lgn.json` は変更しません。

```json title="config/mnist_startup.json"
{
  "run": {
    "seed": 0,
    "device": "cpu",
    "log_dir": "log",
    "initial_checkpoint_path": null
  },
  "data": {
    "name": "mnist",
    "root": "data",
    "batch_size": 128,
    "num_workers": 0
  },
  "model": {
    "name": "mnist_lgn"
  },
  "train": {
    "epochs": 1,
    "loss": {
      "name": "cross_entropy",
      "label_smoothing": 0.0
    },
    "optimizer": {
      "name": "adamw",
      "learning_rate": 0.01,
      "weight_decay": 0.0
    },
    "scheduler": {
      "name": "cosine_annealing",
      "minimum_learning_rate": 0.0001
    }
  },
  "circuit": {
    "enabled": false
  }
}
```

この例で通常設定から変更している項目は、次の3点です。

| 設定項目 | 初回実行の値 | 意味 |
|---|---|---|
| `run.device` | `"cpu"` | GPUを使用せず実行する |
| `train.epochs` | `1` | 学習用データを1回だけ一巡する |
| `circuit.enabled` | `false` | 学習終了後の回路出力を行わない |

`initial_checkpoint_path` は `null` のままとし、新しい重みから学習を開始します。設定JSONは全項目を含めて保存してください。上の3項目だけを記載する部分設定には対応していません。

## 2. 学習を実行する

```shell
python main.py --config config/mnist_startup.json
```

上記のコマンドは、MNISTのダウンロードが完了し、仮想環境が有効な状態で、プロジェクトルートから実行してください。`--config` は使用するJSONファイルを指定する引数です。

## 3. 実行中の表示を確認する

最初に「学習開始」の表が表示されます。データセットが `mnist`、モデルが `mnist_lgn`、deviceが `cpu`、epochsが `1`、回路出力が「無効」であることを確認します。

1 epochの学習と検証が終わると、次の形式で指標が表示されます。以下の数値は表示例であり、期待精度を示すものではありません。

```text
Epoch 1/1 | lr=0.010 | train loss=1.053 accuracy=69.39% | validation loss=1.136 accuracy=65.57%
```

バッチ単位の進捗表示はありません。この行が表示されるまでに、学習用50,000件と検証用10,000件を処理します。CPUでの所要時間は実行環境によって異なります。

その後、保存済みのbestモデルでテスト用10,000件を評価し、「学習完了」と保存先を表示します。「学習完了」が表示され、端末の入力待ちに戻れば正常終了です。Ctrl+Cで中断した場合は、正常終了した実行とは区別してください。

## 4. 通常設定での起動方法を確認する

初回の結果を確認した後、通常設定で学習する場合は次のコマンドを使用します。このページを進めるために、続けて実行する必要はありません。

```shell
python main.py
```

`--config` を省略すると `config/mnist_lgn.json` を使用します。配布時の設定は100 epoch、デバイスは `auto`、回路出力は有効です。初回実行用の1 epoch設定ではない点に注意してください。

学習回数やデバイスはJSON内で指定します。`--epochs` や `--device` というコマンドライン引数はありません。

次は [結果の確認](./results.md) に進みます。
