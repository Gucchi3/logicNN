---
title: __init__.py
sidebar_label: __init__.py
slug: /code-reference/utils/runtime/__init__
toc_min_heading_level: 2
toc_max_heading_level: 2
pagination_prev: null
pagination_next: null
---

# __init__.py

`utils/runtime/__init__.py`

実行デバイスの選択と乱数seedの設定関数を公開します。

{/* source-sha256: f38f2b8b87539f843e9e977074e5765d916020b7034c226a7a79e134114f9cd5 */}

このファイルには関数・クラスの定義はありません。パッケージの初期化や、別ファイルで定義した名前の公開に使用します。

<details>
<summary>ファイルの内容を開く</summary>

```python
"""実行デバイスの選択と乱数seed設定を公開する。"""

from .device import select_device
from .seed import set_seed

__all__ = ("select_device", "set_seed")
```

</details>

## 関連ファイル

- [utils/runtime/device.py](/code-reference/utils/runtime/device)
- [utils/runtime/seed.py](/code-reference/utils/runtime/seed)
