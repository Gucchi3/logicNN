---
title: 設定ファイル
description: 設定JSONの全項目、指定できる値、ファイルパスの解釈を説明します。
---

# 設定ファイル

学習条件はJSONファイルで指定します。通常は `config/mnist_lgn.json` を使用し、条件を変えて試す場合は、コピーを別名で保存して編集します。

## 設定ファイルを選択する

例えば `config/experiment.json` を用意した場合は、次のコマンドで実行します。

```shell
python main.py --config config/experiment.json
```

`--config` を省略すると `config/mnist_lgn.json` を読み込みます。設定は起動時に一度読み込まれるため、実行中にJSONを書き換えても、その実行の条件は変わりません。

設定は `run`、`data`、`model`、`train`、`circuit` の5区分です。以下の「配布時の値」は同梱JSONの値であり、省略時に自動補完される値ではありません。設定ファイルには全項目が必要です。

## 実行設定：run

| 項目 | 配布時の値 | 指定内容 |
|---|---|---|
| `run.seed` | `0` | 0～4,294,967,295の整数。乱数の初期化、MNISTの分割・学習順序に使用する |
| `run.device` | `"auto"` | `"auto"`、`"cpu"`、`"cuda"` のいずれか |
| `run.log_dir` | `"log"` | 実行ごとの保存フォルダを作成する親フォルダ。空文字列は指定不可 |
| `run.initial_checkpoint_path` | `null` | 初期重みに使うlogicNN形式の `.pth` のパス。新しい重みで開始する場合は `null` |

`auto` はCUDAを利用できる場合にGPU、それ以外の場合にCPUを使用します。`cuda` を明示して利用できなかった場合はエラーになります。設定値だけでCUDA対応のPyTorchを導入することはできません。

## データ設定：data

| 項目 | 配布時の値 | 指定内容 |
|---|---|---|
| `data.name` | `"mnist"` | データセットの登録名。現在の登録名は `mnist` |
| `data.root` | `"data"` | データセットの保存先となる親フォルダ。空文字列は指定不可 |
| `data.batch_size` | `128` | 1以上の整数。学習・検証・テストで1回に処理する標本数 |
| `data.num_workers` | `0` | 0以上の整数。データを読み込む子プロセス数。0では学習と同じプロセスで読み込む |

MNISTの保存先は `data.root` の下の `mnist/` です。`data.root` に `data/mnist` を指定すると、さらに `mnist` が追加されるため、通常は `data` を指定します。

MNISTでは学習50,000件、検証10,000件、テスト10,000件を使用します。件数を指定する `validation_size` はありません。別のデータセットの分割方法は、そのデータセットの実装で定義します。

## モデル設定：model

| 項目 | 配布時の値 | 指定内容 |
|---|---|---|
| `model.name` | `"mnist_lgn"` | モデルの登録名。現在の登録名は `mnist_lgn` |

層数、層の幅、ゲートの入力数、クラス数などの構造はモデル定義内で指定します。JSONに `parameters` や `input_shape` を追加して変更する方式ではありません。[モデルの追加](./models.md)を参照してください。

## 学習設定：train

| 項目 | 配布時の値 | 指定内容 |
|---|---|---|
| `train.epochs` | `100` | 1以上の整数。今回の実行で学習するepoch数 |
| `train.loss.name` | `"cross_entropy"` | 損失関数の登録名。現在は交差エントロピーのみ |
| `train.loss.label_smoothing` | `0.0` | 0～1の数値。学習時の正解ラベルを平滑化する強さ。0では無効 |
| `train.optimizer.name` | `"adamw"` | 最適化手法の登録名。現在はAdamWのみ |
| `train.optimizer.learning_rate` | `0.01` | 0より大きい数値。学習開始時の学習率 |
| `train.optimizer.weight_decay` | `0.0` | 0以上の数値。重みを小さくする正則化の強さ |
| `train.scheduler.name` | `"cosine_annealing"` | 学習率の更新方式。現在はCosine Annealingのみ |
| `train.scheduler.minimum_learning_rate` | `0.0001` | 0以上、初期学習率以下の数値。学習率更新の下限 |

検証とテストでは `label_smoothing=0.0` の損失関数を使用します。AdamWの `betas` や `eps` などはPyTorchの既定値を使い、JSONでは指定しません。実際に使われた値は `training_info.json` に保存します。

Cosine Annealingの `T_max` は `train.epochs` から設定されます。各epochの終了時に学習率を更新するため、表示・保存される学習率は、そのepochで使用した値です。最終epochの後にも更新するため、最後の履歴に記録された値が下限値そのものになるとは限りません。

未登録の名前へ変更するだけでは、新しい損失関数・最適化手法・学習率更新方式は使用できません。方式を追加する場合は、`utils/trainer/optimization/` の実装と登録表を変更し、追加の設定項目が必要なら設定型も更新します。方式によっては、共通の学習処理や実行条件の保存処理も更新する必要があります。

## 回路出力設定：circuit

| 項目 | 配布時の値 | 指定内容 |
|---|---|---|
| `circuit.enabled` | `true` | 学習終了後の回路簡略化とJSON・C・Verilog出力を一括で有効にする |

個別の形式だけを選ぶスイッチはありません。学習アプリとは別に形式を選択する場合は、[ライブラリの直接利用](./core-library.md)を参照してください。

## パスの基準

| 指定する場所 | 相対パスの基準 |
|---|---|
| コマンドの `--config` | コマンドを実行したフォルダ |
| JSONの `data.root`、`run.log_dir`、`run.initial_checkpoint_path` | `main.py` があるプロジェクトルート |
| データ取得ツールの `--root` | プロジェクトルート |

設定ファイルを別のフォルダへ移しても、JSON内の相対パスの基準は変わりません。絶対パスも使用できます。

WindowsのJSON内では、`"C:/work/logicNN/data"` のように `/` を使うと記述が簡潔になります。バックスラッシュを使う場合はJSONのエスケープが必要です。

## 編集時の注意

- UTF-8のJSONとして保存します。コメントや末尾の余分なカンマは記載できません。
- `true`、`false`、`null` を文字列として引用符で囲まないでください。
- 数値を `"100"` のような文字列にしないでください。整数の項目には整数を指定します。
- 未知の項目はエラーになります。省略による既定値の補完や、複数ファイルの設定の合成は行いません。
- 学習回数やデバイスを指定する `--epochs`、`--device` はありません。JSONを編集します。

設定型の詳細は[設定スキーマのコードリファレンス](../code-reference/utils/config/schema.md)を参照してください。
