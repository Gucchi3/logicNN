---
title: スタートアップガイド
slug: /startup
displayed_sidebar: startupSidebar
pagination_prev: null
pagination_next: startup/setup
---

# スタートアップガイド

このガイドでは、Python環境の構築から、データセットの取得、初回の学習、保存結果の確認までを説明します。MNISTを例として、CPUで1 epochの学習を実行します。

| 順序 | ページ | 完了時の状態 |
|---|---|---|
| 1 | [環境構築](./setup.md) | 仮想環境内で `python main.py --help` を実行できる |
| 2 | [データセットのダウンロード](./dataset.md) | MNISTの学習用・テスト用データを取得できている |
| 3 | [最小構成での実行](./first-training.md) | 1 epochの学習と評価が正常に終了している |
| 4 | [結果の確認](./results.md) | 指標、学習曲線、保存済みモデルの意味を確認できる |

操作は `main.py` と `pyproject.toml` があるプロジェクトルート `logicNN/` で行います。uvで仮想環境を作成し、`uv pip install` で依存パッケージを導入した後、環境を有効化した端末で通常の `python` コマンドを使用します。

初回確認では回路出力を無効にします。通常の学習設定やモデル構造は変更しません。論理ゲートNNのライブラリ `logicNN_core` は同梱されているため、別途取得する必要はありません。
