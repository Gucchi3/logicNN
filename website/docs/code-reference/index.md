---
title: コードリファレンス
slug: /code-reference
pagination_next: code-reference/main
---

# コードリファレンス

ソースファイルを選び、その中にある関数の機能と処理の流れを確認できます。

左のメニューは、logicNNのPythonソースと同じフォルダ・ファイル階層です。フォルダを展開して目的のファイルを探せます。

## 目的から探す

| 調べたい内容 | 入口 |
|---|---|
| プログラムの起動 | [main.py](./main.md#main) |
| 学習全体の処理順序 | [utils/trainer/workflow.py](./utils/trainer/workflow.md#run-training) |
| 1 epochの学習・評価 | [utils/trainer/epoch.py](./utils/trainer/epoch.md) |
| 設定ファイルの読み込み・型 | [utils/config/loader.py](./utils/config/loader.md)、[schema.py](./utils/config/schema.md) |
| データセットと二値化 | [utils/data/loader.py](./utils/data/loader.md)、[preprocessing.py](./utils/data/preprocessing.md) |
| モデルの選択と定義 | [model/builder.py](./model/builder.md)、[mnist_lgn.py](./model/lgn/mnist_lgn.md) |
| 重みと学習結果の保存 | [utils/results/checkpoint.py](./utils/results/checkpoint.md)、[metrics.py](./utils/results/metrics.md) |
| 学習済みモデルの回路出力 | [utils/export/circuit_exporter.py](./utils/export/circuit_exporter.md) |
| コアライブラリの入口 | [logicnn_core/\_\_init\_\_.py](./utils/logicNN_core/src/logicnn_core/__init__.md) |
| 論理層・接続・ゲートの学習方式 | [layers/base.py](./utils/logicNN_core/src/logicnn_core/layers/base.md)、[connections/builder.py](./utils/logicNN_core/src/logicnn_core/connections/builder.md)、[parametrizations/builder.py](./utils/logicNN_core/src/logicnn_core/parametrizations/builder.md) |
| 回路の変換・簡略化・実行・C/Verilog生成 | [circuit/circuit.py](./utils/logicNN_core/src/logicnn_core/circuit/circuit.md) |

## 読み方

1. **左のメニュー**でフォルダ・ファイルを選びます。
2. **ページ内の目次**で関数・クラスや、引数・処理図などの項目を選びます。
3. **処理図とソースコード**を見比べ、必要なら呼び出し先のファイルへ進みます。

対象は `main.py`、`model/`、`tools/`、`utils/` 内のPythonファイルです。同梱ライブラリも `utils/logicNN_core/` の実際の階層で表示します。環境・データ・実行ログ・文書サイトの生成物は、このソース一覧には含めません。

公開関数だけでなく、`_` で始まる内部用関数、クラスのメソッド、プロパティ、関数内で定義された関数も掲載しています。継承元の外部ライブラリが提供するメソッドや、dataclassが自動生成するメソッドは独自実装と区別します。

## 各関数の記載内容

関数定義、機能概要、引数、内部アクティビティ図、戻り値、折りたたみのソースコードを掲載します。必要な場合は、状態の変更・ファイル出力、例外・注意事項、呼び出し例、関連項目を追加します。

図は上から下へ読みます。ひし形は条件分岐、戻る矢印は繰り返し、二重枠は別の関数の呼び出しを表します。複数の式を一つの処理としてまとめる場合もあります。呼び出し先の詳細や、図に省略した例外の条件は本文と該当関数のページで説明します。

このリファレンスは現在の実装を説明します。設計上の方針や過去の検討経緯とは分けて扱います。コードを変更した場合は、該当する説明と図も更新します。

操作手順は [ユーザーガイド](../user-guide/index.md)、システム全体の構成と設計上のルールは [仕様書](../specifications/index.md) で扱います。既存の [APIメモ](../api/index.md) は、内容を整理するまで残しています。
