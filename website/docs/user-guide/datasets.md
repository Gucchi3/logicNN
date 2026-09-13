---
title: データセットの追加
description: DataBundleの共通形式に従い、データの取得・前処理・読込と登録を追加する手順を説明します。
---

# データセットの追加

`main.py` の学習処理は、データセット名から専用のbuilderを選びます。新しい分類データセットでも、`DataBundle` の形でデータと情報を返せば、共通の学習・評価・保存処理を使えます。

現在同梱されている登録は `mnist` のみです。このページの `fashion_mnist` は**追加方法を説明するための未同梱・未登録の例**です。名前を設定JSONへ書くだけでは利用できません。

## 1. 取得・前処理・読込を分ける

| 追加するもの | 配置・登録先 | 内容 |
|---|---|---|
| データセット専用処理 | `utils/data/<dataset>.py` | 取得済みデータの読込、分割、DataLoader、メタデータの生成 |
| Tensorの前処理 | `utils/data/preprocessing.py` など | 二値化など、モデルから独立した変換 |
| 学習用builderの登録 | `utils/data/loader.py` | データセット名と `build_*_bundle` の対応付け |
| データ取得の実行入口 | `tools/download_dataset.py` | `_downloaders()` へ専用取得関数を登録 |
| 実行条件 | 設定JSONの `data` | 登録名、保存先、バッチサイズ、worker数 |

通常の学習用builderではダウンロードしません。既存MNISTと同じく、取得は事前にツールから明示的に実行します。取得処理の関数はデータセット専用ファイルへまとめられますが、その実行入口は `tools/` に分けます。

## 2. 学習処理へ渡す共通形式

### builderの引数と戻り値

追加する関数は、位置引数で次の3つを受け取り、`DataBundle` を返します。

```python
def build_fashion_mnist_bundle(config: DataConfig, seed: int, pin_memory: bool) -> DataBundle:
    """取得済みデータから学習・検証・テスト用のDataBundleを作る。"""
```

| 引数 | 内容 |
|---|---|
| `config` | `name`、`root`、`batch_size`、`num_workers` を持つ `DataConfig` |
| `seed` | 実行設定の `run.seed`。分割や学習時の並べ替えに使用 |
| `pin_memory` | 学習先がCUDAなら `True`、CPUなら `False` が渡される |

`config.root` はデータ保存先の親です。データセットごとにその下へ専用ディレクトリを作ります。設定の相対パスはプロジェクトルートを基準に解決されます。

`DataBundle` には次の4項目を設定します。

| 項目 | 内容 |
|---|---|
| `train_loader` | 学習用DataLoader。通常は `shuffle=True` |
| `validation_loader` | 学習中のモデル選択に使うDataLoader。通常は `shuffle=False` |
| `test_loader` | 学習後にbestモデルを評価するDataLoader。通常は `shuffle=False` |
| `metadata` | 形状・クラス・件数・前処理を記録する `DatasetMetadata` |

学習・検証・テストのいずれも、少なくとも1標本を供給するようにします。共通のepoch処理は空のDataLoaderをエラーにします。

### バッチの形状と値

現在の学習アプリケーションは、**単一ラベルの分類**を対象にしています。

| 値 | 形状 | 内容 |
|---|---|---|
| `inputs` | `[batch, *input_shape]` | 論理モデルへ渡す入力。既存MNISTと同じ運用では、前処理済みの `float32` の0/1 |
| `targets` | `[batch]` | 0から `num_classes - 1` までの整数クラス番号。`int64` |

クラス名の並びも番号に合わせます。例えば `class_names[3]` はラベル3のクラス名です。one-hotラベルや回帰値をそのまま渡す形式ではありません。

epoch処理は入力を `float32`、ラベルを `int64` へ変換しますが、入力の二値化やラベルの意味の検証までは行いません。小数ラベルが整数変換されることに頼らず、データセット側で正しいクラス番号を用意してください。

coreには連続入力の計算や二値化層もあります。ただし、現在のアプリケーションの回路出力は二値化済み入力を前提とします。新しいデータを追加するだけで、前処理を含む連続入力のC・Verilog回路へ自動拡張されるわけではありません。

### メタデータ

`DatasetMetadata` には次の情報を記録します。

- `name`：登録したデータセット名。
- `input_shape`：**前処理後**の、バッチを除く標本形状。
- `num_classes`、`class_names`：クラス数と番号順のクラス名。
- `train_size`、`validation_size`、`test_size`：各分割の実際の標本数。
- `split_seed`：分割に使ったseed。無作為分割をしない場合は `None` とできます。
- `preprocessing`：前処理名・JSON互換の設定辞書・出力dtype。

`input_size` は `input_shape` の積から求めるプロパティなので、コンストラクタには渡しません。`validation_size` は実際の件数を**記録するメタデータ**であり、設定JSONへ追加する項目ではありません。

型の詳細は[DataBundle・DatasetMetadata・PreprocessingMetadata](/code-reference/utils/data/types)を参照してください。

## 3. データセット専用ファイルを追加する

以下は `utils/data/fashion_mnist.py` に追加する例です。MNISTと同じ `[1, 28, 28]`・10クラスのFashion-MNISTを、学習用50,000件・検証用10,000件・公式テスト10,000件で使用します。

二値化は既存の `binarize_greater_than` を使います。この閾値設定は手順を示すための例であり、精度に最適な前処理であることを意味しません。

```python
"""Fashion-MNISTの取得・前処理・分割をまとめた追加例。"""

import random
from functools import partial
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import FashionMNIST
from torchvision.transforms import Compose, ToTensor

from utils.config import DataConfig
from utils.exceptions import LogicNNError

from .preprocessing import binarize_greater_than
from .types import DataBundle, DatasetMetadata, PreprocessingMetadata

INPUT_SHAPE     = (1, 28, 28)
VALIDATION_SIZE = 10_000


def _load_fashion_mnist(root: Path, *, download: bool) -> tuple[FashionMNIST, FashionMNIST]:
    """専用ディレクトリから公式データを読み、共通の前処理を設定する。"""
    destination = (root / "fashion_mnist").resolve()
    transform   = Compose([ToTensor(), partial(binarize_greater_than, threshold=0.0)])
    try:
        train = FashionMNIST(root=destination, train=True, download=download, transform=transform)
        test  = FashionMNIST(root=destination, train=False, download=download, transform=transform)
    except (OSError, RuntimeError) as error:
        raise LogicNNError(
            "Fashion-MNISTを準備できません", detail=f"保存先: {destination}\n原因: {error}",
            hint="取得ツールの実行状況、保存先の権限、取得時のネットワーク接続を確認してください",
        ) from error
    return train, test


def _seed_worker(worker_id: int) -> None:
    """workerに割り当てられたseedをPythonとNumPyにも反映する。"""
    del worker_id
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def build_fashion_mnist_bundle(config: DataConfig, seed: int, pin_memory: bool) -> DataBundle:
    """取得済みデータを分割し、3つのDataLoaderとメタデータを返す。"""
    official_train, official_test = _load_fashion_mnist(config.root, download=False)
    split_generator = torch.Generator().manual_seed(seed)
    train_generator = torch.Generator().manual_seed(seed)
    sizes           = (len(official_train) - VALIDATION_SIZE, VALIDATION_SIZE)
    train_data, validation_data = random_split(official_train, sizes, generator=split_generator)
    loader_options = {
        "batch_size": config.batch_size, "num_workers": config.num_workers, "pin_memory": pin_memory,
        "drop_last": False, "persistent_workers": config.num_workers > 0, "worker_init_fn": _seed_worker,
    }
    train_loader      = DataLoader(train_data, shuffle=True, generator=train_generator, **loader_options)
    validation_loader = DataLoader(validation_data, shuffle=False, **loader_options)
    test_loader       = DataLoader(official_test, shuffle=False, **loader_options)
    class_names       = tuple(official_train.classes)
    preprocessing     = PreprocessingMetadata("binary_threshold", {"threshold": 0.0, "comparison": "greater_than"}, "float32")
    metadata          = DatasetMetadata(
        name="fashion_mnist", input_shape=INPUT_SHAPE, num_classes=len(class_names), class_names=class_names,
        train_size=len(train_data), validation_size=len(validation_data), test_size=len(official_test),
        split_seed=seed, preprocessing=preprocessing,
    )
    return DataBundle(train_loader=train_loader, validation_loader=validation_loader, test_loader=test_loader, metadata=metadata)


def download_fashion_mnist(root: Path) -> tuple[int, int, Path]:
    """公式データを取得し、公式分割の件数と保存先を返す。"""
    train, test = _load_fashion_mnist(root, download=True)
    return len(train), len(test), (root / "fashion_mnist").resolve()
```

この例の学習用builderは、分割と並べ替えに別々のGeneratorを使っています。学習用の並べ替えが進んでも、検証用の分割を再抽選しません。分割比率や件数はデータセットごとに決め、MNISTの値をすべてのデータへ強制しないでください。

別の前処理が必要なら `utils/data/` の適切なファイルへ関数を追加し、実際の変換と `PreprocessingMetadata` を同時に更新します。Windowsで複数workerを使う場合は、前処理やworker初期化関数をモジュール直下に定義し、lambdaや関数内のローカル関数をDataLoaderへ渡さない構成にします。

## 4. 学習用builderを登録する

`utils/data/loader.py` に専用builderのimportを追加し、既存登録を残して `DATASET_REGISTRY` へ追記します。

```python
from .fashion_mnist import build_fashion_mnist_bundle
from .mnist import build_mnist_bundle

DATASET_REGISTRY: dict[str, DatasetBuilder] = {
    "mnist": build_mnist_bundle,
    "fashion_mnist": build_fashion_mnist_bundle,
}
```

共通の `build_data_bundle()` は変更不要です。学習処理は登録名から専用関数を選び、`(config, seed, pin_memory)` を渡します。[データbuilder](/code-reference/utils/data/loader)に内部の呼び出しを記載しています。

## 5. 取得ツールへ登録する

`tools/download_dataset.py` の `_downloaders()` 内で、既存のパス設定を残したまま取得関数のimportと対応表を追加します。

```python
def _downloaders() -> dict[str, DownloadFunction]:
    """データセット登録名と取得処理の対応表を返す。"""
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from utils.data.fashion_mnist import download_fashion_mnist
    from utils.data.mnist import download_mnist

    return {"mnist": download_mnist, "fashion_mnist": download_fashion_mnist}
```

取得関数の共通形式は、`Path` を受け取り、`(公式学習用の件数, テスト用の件数, 保存先Path)` を返す形です。取得ツールの表示上は学習用・テスト用の2件数であり、学習用から検証用を分ける処理はその後のbuilderで行います。

登録後、プロジェクトルートから取得します。

```powershell
python tools/download_dataset.py --dataset fashion_mnist --root data
```

取得ツールの内部は[download_dataset.py](/code-reference/tools/download_dataset)を参照してください。ローカルで用意する独自データならダウンロード機能は必須ではなく、専用builderが読む配置へ事前に保存する手順を用意します。

## 6. 対応モデルと一緒に設定を切り替える

既存の設定JSONを別名へコピーし、`data.name` を登録名へ変更します。以下は変更する部分だけです。

```json
{
  "data": {
    "name": "fashion_mnist",
    "root": "data",
    "batch_size": 128,
    "num_workers": 0
  }
}
```

モデルの `input_shape`・`num_classes` とデータのメタデータが一致する必要があります。この例はMNISTと同じ形状・クラス数なので、構造上は既存 `mnist_lgn` でも生成できます。ただし、衣服の分類にMNISTの学習済み重みを無条件に流用する意味ではありません。初回は `run.initial_checkpoint_path` を `null` にして学習します。

形状やクラス数が異なる場合は[モデルの追加](./models.md)も行い、`model.name` を切り替えてください。データセット名・形状・クラス数・前処理の情報は、チェックポイントの読込時にも照合されます。

## 7. 追加後に確認すること

1. 小さいデータで各DataLoaderを1バッチ読み、入力shape・dtype・0/1の値・ラベル範囲を確認する。
2. 学習・検証・テストが意図した分割になり、メタデータの件数と実データが一致することを確認する。
3. モデルの入力shapeとクラス数が一致し、損失計算と評価が動くことを確認する。
4. 取得済みデータで通常学習を実行し、学習開始時にダウンロードが走らないことを確認する。

開発時の確認には小さい人工データを使えます。通常運用の学習へ、開発者用テストや追加の毎バッチ検査を組み込む必要はありません。
