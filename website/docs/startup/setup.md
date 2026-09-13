---
title: 環境構築
description: uvで仮想環境を作成し、logicNNの実行に必要なパッケージを導入します。
pagination_prev: null
pagination_next: startup/dataset
---

# 環境構築

このページでは、uvでPythonの仮想環境を作成し、CPUで学習を実行できる状態にします。以下はWindowsのPowerShellを使用する手順です。

## 1. 作業フォルダを確認する

配布されたプロジェクトを展開し、`main.py` と `pyproject.toml` があるフォルダを端末で開きます。本ガイドでは、このフォルダを「プロジェクトルート」と呼びます。

現在の構成で、親の `project/` フォルダから移動する場合は次のように実行します。

```shell
cd logicNN
```

```shell
uv --version
```

uvのバージョンが表示されることを確認します。未導入の場合は、[uvの公式インストール手順](https://docs.astral.sh/uv/getting-started/installation/)に従って導入してください。パッケージとデータセットの初回取得にはインターネット接続が必要です。

## 2. 仮想環境を作成して有効化する

```shell
uv venv --python 3.13
.\.venv\Scripts\Activate.ps1
```

プロジェクト内に `.venv/` が作成されます。この例ではPython 3.13を使用します。プロジェクトが定義するPythonの下限は3.10です。指定したPythonが見つからない場合、uvは必要なPythonを取得します。[uvの仮想環境の説明](https://docs.astral.sh/uv/pip/environments/)

既にこのプロジェクト用の環境がある場合は、作成コマンドを再実行せず、その環境を有効化してください。別のプロジェクトの環境が有効になっている場合は、先に `deactivate` で解除します。

<details>
<summary>Linux・macOSで仮想環境を有効化する場合</summary>

`uv venv --python 3.13` を実行した後、bashやzshでは次のコマンドを使用します。

```shell
source .venv/bin/activate
```

以降の `uv pip install` と `python` のコマンドは共通です。

</details>

PowerShellでスクリプトの実行が制限されている場合は、端末をコマンドプロンプトに切り替え、プロジェクトルートで `.venv\Scripts\activate.bat` を実行してください。

有効化後、実際に使用するPythonを確認します。

```shell
python -c "import sys; print(sys.executable)"
```

表示されたパスが、使用する仮想環境内のPythonを指していることを確認してください。新しい端末を開いた場合は、その都度環境の有効化が必要です。

## 3. 依存パッケージをインストールする

仮想環境を有効にした状態で、次のコマンドを実行します。

```shell
uv pip install -r pyproject.toml -e utils/logicNN_core --torch-backend cpu
```

- `-r pyproject.toml`：プロジェクトが定義する依存パッケージをインストールします。
- `-e utils/logicNN_core`：同梱の `logicNN_core` を編集可能な形式でインストールします。
- `--torch-backend cpu`：CPU用のPyTorchを選択します。

依存パッケージの指定方法は[uvのパッケージ管理](https://docs.astral.sh/uv/pip/packages/)、PyTorchの選択方法は[uvとPyTorchの連携](https://docs.astral.sh/uv/guides/integration/pytorch/#automatic-backend-selection)を参照してください。

本ガイドの手順では `uv.lock` に環境を同期せず、上記の指定から依存関係を解決します。開発用テストのパッケージ、Cコンパイラ、Verilogシミュレータ、文書サイト用のNode.jsは、初回の学習には不要です。

### 既存のGPU環境を使用する場合

既にCUDA対応のPyTorch環境で学習できる場合は、その環境を有効化し、CPU用のインストール手順は実行しないでください。新たにGPU環境を構築する場合は、利用するNVIDIA GPUとドライバに対応したPyTorchを導入します。

`run.device` の `auto` はCUDAを利用できればGPU、それ以外はCPUを選びます。`cuda` を指定しても、CPU用のPyTorchがCUDA対応に切り替わるわけではありません。本ガイドの最小実行では `cpu` を指定します。

## 4. 起動を確認する

```shell
python main.py --help
```

ヘルプに `--config PATH` が表示されれば、このページの作業は完了です。このコマンドでは学習やデータセットのダウンロードは行いません。

次は [データセットのダウンロード](./dataset.md) に進みます。
