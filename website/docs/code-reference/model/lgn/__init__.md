---
title: __init__.py
sidebar_label: __init__.py
slug: /code-reference/model/lgn/__init__
toc_min_heading_level: 2
toc_max_heading_level: 2
pagination_prev: null
pagination_next: null
---

# __init__.py

`model/lgn/__init__.py`

アプリケーション用のモデルクラス `MNISTLGN` を公開します。

{/* source-sha256: 2dd1fe277638bbadd08b6c5d53332501e86ba5aff8728ebcf5f2a10ea0f26b97 */}

このファイルには関数・クラスの定義はありません。パッケージの初期化や、別ファイルで定義した名前の公開に使用します。

<details>
<summary>ファイルの内容を開く</summary>

```python
"""アプリケーションで使用する論理ゲートNNモデルを公開する。"""

from .mnist_lgn import MNISTLGN

__all__ = ["MNISTLGN"]
```

</details>

## 関連ファイル

- [model/lgn/mnist_lgn.py](/code-reference/model/lgn/mnist_lgn)
