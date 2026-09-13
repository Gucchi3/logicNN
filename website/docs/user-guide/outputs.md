---
title: ログと成果物
description: 実行ごとの保存先、各ファイルの用途、学習履歴と評価結果の読み方を説明します。
---

# ログと成果物

## 実行ごとの保存先

`run.log_dir` の下に、実行開始時のローカル日時を使った `YYYYMMDD_HHMMSS/` を作成します。同じ名前が既にある場合は `_01` などを追加します。新しい実行で既存の実行フォルダを上書きしません。

保存先は開始時の「実行ディレクトリ」と完了時の「保存先」に表示されます。条件の異なる実験では、`run.log_dir` を `log/experiment_a` などに分けることもできます。

## 通常の成果物

| ファイル | 内容・用途 |
|---|---|
| `config.json` | 今回使用した全設定。設定内のパスは絶対パスへ解決した状態 |
| `training_info.json` | 実行デバイス、Python・PyTorch等のバージョン、モデル構造、データ件数・前処理、実際の最適化設定、初期重みの由来 |
| `metrics.jsonl` | 1行に1 epoch分の学習・検証指標を保存する履歴 |
| `curves.png` | 保存済み履歴から作成した損失と検証精度のグラフ |
| `model_best.pth` | 最も高い検証精度を得たモデル状態と関連情報 |
| `model_final.pth` | 最後のepochのモデル状態と関連情報 |
| `test_metrics.json` | 保存済みbestでテスト用データを評価した結果 |

回路出力が有効な場合は `circuit/` が追加されます。詳細は[回路出力](./circuit.md)を参照してください。端末の表示内容をそのまま保存するテキストログは生成しません。

## epoch履歴を読む

`metrics.jsonl` はJSON Lines形式です。ファイル全体が一つのJSON配列になっているわけではありません。エディタで1行ずつ読むか、Pythonで各行をJSONとして読み込みます。

| 項目 | 内容 |
|---|---|
| `epoch` | epoch番号。1から開始 |
| `learning_rate` | そのepochで使用した学習率 |
| `train_loss` | 学習用データの平均損失 |
| `train_accuracy` | 学習用データの正解率 |
| `validation_loss` | 検証用データの平均損失 |
| `validation_accuracy` | 検証用データの正解率 |

損失は標本数で重み付けした平均です。小さい最終バッチも、各標本が同じ重みになるよう集計します。正解率は0～1の割合で、`0.9342` は93.42%を意味します。

epoch行の学習率と損失は小数点以下3桁、正解率は百分率の小数点以下2桁で表示します。JSONには表示用の丸めを適用しません。小さい学習率が端末で `0.000` と表示された場合も、記録値が0とは限りません。

### Pythonで履歴を読み込む例

以下はプロジェクトルートから実行するPythonコードの例です。`run_dir` を実際の保存先に置き換えてください。

```python
import json
from pathlib import Path

run_dir = Path("log/20260913_120000")
records = [json.loads(line) for line in (run_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
latest  = records[-1]
print(f"epoch={latest['epoch']}, validation accuracy={latest['validation_accuracy']:.2%}")
```

少なくとも1 epochの記録がある実行を対象にします。このコードはファイルを読み込むだけで、学習を再実行しません。

## 学習曲線を読む

`curves.png` の左側は学習損失と検証損失、右側は赤色の検証精度です。学習精度はグラフには描かず、`metrics.jsonl` に保存します。

凡例の `latest` は最新epochの値、`best` は履歴中の最小損失または最高精度です。損失の `best` が記録されたepochと、`model_best.pth` を保存したepochは一致するとは限りません。

点のマーカーを付けないため、1 epochだけでは線が見えない場合があります。凡例と履歴の数値を確認してください。

## 最終評価を読む

`test_metrics.json` には、使用した `checkpoint`、bestの `epoch`、平均損失 `loss`、正解率 `accuracy`、標本数 `samples` を保存します。これはfinalではなく、保存済みbestを読み直して評価した結果です。

`training_info.json` の `evaluation_modes` には、学習と評価の演算モードを記録します。指標の違いは[学習の実行](./training.md)を参照してください。

## 保存結果の扱い

`config.json` は実際に使用した条件の記録です。他のPCやフォルダで再利用する場合は、絶対パスになったデータ・出力先・初期重みのパスを確認してください。元の `config/` 内の設定ファイルの方が、相対パスを使って再利用しやすい場合があります。

エラーや中断が発生した場合も、一部の成果物は残ります。実行フォルダの存在だけで成功と判断せず、「学習完了」の表示と必要なファイルの保存を確認します。回路生成だけに失敗した場合は、学習結果と回路出力の成否を分けて確認してください。

モデルの再利用は[保存済み重みの利用](./checkpoints.md)、保存処理の詳細は[結果保存のコードリファレンス](../code-reference/utils/results/metrics.md)を参照してください。
