---
sidebar_position: 1
title: API
slug: /api
---

# API

:::info[コードリファレンスへ整理中]

ファイル階層から探す新しい [コードリファレンス](../code-reference/index.md) を作成しています。まず [main.py の main()](../code-reference/main.md#main) の説明と内部アクティビティ図を確認できます。このページ以下は、整理前のAPIメモとして保持しています。

:::

コードから利用する入口を説明します。学習コマンドの操作は [ユーザーガイド](../user-guide/index.md) を参照してください。

| 対象 | 内容 |
|---|---|
| [アプリのAPI](./application.md) | 設定、DataBundle、モデルbuilder、学習、成果物 |
| [coreの使い方](./core.md) | 小型の論理NNから回路生成までの実行例 |
| [coreの構成と設定](./core-components.md) | 層、LUT・接続設定、数値関数、モード |
| [回路API](./circuit.md) | 変換、実行、簡略化、保存、compileの契約 |

`utils/logicNN_core/` の配布名は `logicNN-core`、Pythonのimport名は `logicnn_core` です。アプリ側の `utils` と独立ライブラリのAPIは別物です。詳細な制約と対応範囲は [core詳細設計](../specifications/logicnn-core/detailed-design.md) が正本です。
