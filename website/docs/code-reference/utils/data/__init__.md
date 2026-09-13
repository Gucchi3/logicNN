---
title: __init__.py
sidebar_label: __init__.py
slug: /code-reference/utils/data/__init__
toc_min_heading_level: 2
toc_max_heading_level: 2
pagination_prev: null
pagination_next: null
---

# __init__.py

`utils/data/__init__.py`

データセット共通型、データ生成関数、二値化関数を公開します。

{/* source-sha256: cd4d3279802b9b5c219db2fc8e3e3e81db09ae9d654d87e52dff9a26e60a058e */}

このファイルには関数・クラスの定義はありません。パッケージの初期化や、別ファイルで定義した名前の公開に使用します。

<details>
<summary>ファイルの内容を開く</summary>

```python
"""データセット共通型、前処理および登録名からの読込を公開する。"""

from .loader import build_data_bundle
from .preprocessing import binarize_greater_than
from .types import DataBundle, DatasetMetadata, PreprocessingMetadata

__all__ = ("DataBundle", "DatasetMetadata", "PreprocessingMetadata", "binarize_greater_than", "build_data_bundle")
```

</details>

## 関連ファイル

- [utils/data/loader.py](/code-reference/utils/data/loader)
- [utils/data/preprocessing.py](/code-reference/utils/data/preprocessing)
- [utils/data/types.py](/code-reference/utils/data/types)
