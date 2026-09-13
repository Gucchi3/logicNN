---
title: __init__.py
sidebar_label: __init__.py
slug: /code-reference/model/__init__
toc_min_heading_level: 2
toc_max_heading_level: 2
pagination_prev: null
pagination_next: null
---

# __init__.py

`model/__init__.py`

モデル登録表 `MODEL_REGISTRY` と生成関数 `build_model()` を公開します。

{/* source-sha256: 96174a0a2fa56b4e3af31247e2ea1c7ffc5154b0640c750cc72beb37da354bf3 */}

このファイルには関数・クラスの定義はありません。パッケージの初期化や、別ファイルで定義した名前の公開に使用します。

<details>
<summary>ファイルの内容を開く</summary>

```python
"""モデルの登録表と、名前だけを受け取る生成入口を公開する。"""

from .builder import MODEL_REGISTRY, build_model

__all__ = ["MODEL_REGISTRY", "build_model"]
```

</details>

## 関連ファイル

- [model/builder.py](/code-reference/model/builder)
