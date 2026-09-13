---
sidebar_position: 2
title: アプリのAPI
---

# アプリのAPI

これらは `logicNN/` 内からimportするアプリ側の入口です。通常の利用者は `main.py` を起点にし、以下は拡張や開発テストで使用します。

## 設定・データ・モデル

| import先・入口 | 返すもの／役割 |
|---|---|
| `utils.config.load_config(path)` | JSONを検証し、パスをプロジェクト基準で解決した`AppConfig` |
| `utils.data.build_data_bundle(config, seed, pin_memory)` | 登録されたデータセットbuilderから`DataBundle` |
| `model.build_model(name)` | モデル定義内の定数で作った新しい`nn.Module` |
| `utils.runtime.select_device(requested)` | `auto`／`cpu`／`cuda`から実deviceを決定 |
| `utils.runtime.set_seed(seed)` | 使用する乱数の初期化 |

`DataBundle` は `train_loader`、`validation_loader`、`test_loader`、`metadata` を持ちます。`DatasetMetadata` はbatchを除く `input_shape`、そこから導出する `input_size`、クラス情報、各分割の件数、分割seed、前処理情報を持ちます。詳細は [拡張方法](../user-guide/extension.md) を参照してください。

## 学習と評価

| import先・入口 | 役割 |
|---|---|
| `utils.trainer.workflow.run_training(config, config_path)` | 全体調整。学習・保存・best再読込・公式test・任意回路出力を実行 |
| `utils.trainer.epoch.train_one_epoch(model, loader, loss, optimizer, device)` | 1 epoch学習して`EpochMetrics`を返す |
| `utils.trainer.epoch.evaluate(model, loader, loss, device)` | 勾配追跡なしで全標本を評価し`EpochMetrics`を返す |

`EpochMetrics` は平均 `loss`、0〜1の `accuracy`、評価した `samples` を持ちます。`TrainingResult` は `run_dir`、`best_epoch`、`best_validation_accuracy`、`test_metrics` を持ちます。

loss／optimizer／schedulerのbuilderは `utils/trainer/optimization/` に分けています。設定名と実装を対応付ける表は各ファイル内で管理し、workflowへ方式別の条件分岐を増やしません。

## 成果物とエラー

`utils.results.load_checkpoint(path)` は検証済みのcheckpoint辞書を返します。読み込んだだけではモデルへ適用せず、workflowが構造・前処理を照合したうえで独立モデルへ適用します。

`utils.export.export_circuit(model, input_shape, run_dir)` は `None` を返し、`run_dir/circuit/` にJSON・C・Verilogを順に生成します。compileや実シミュレーションは行いません。

想定済みのアプリ側エラーは `utils.exceptions.LogicNNError` です。`message`、`detail`、`hint` をRichの表示へ渡します。coreの変換エラーはこの境界でアプリ側のエラーへ変換します。
