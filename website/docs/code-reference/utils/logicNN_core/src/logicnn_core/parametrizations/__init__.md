---
title: __init__.py
sidebar_label: __init__.py
slug: /code-reference/utils/logicNN_core/src/logicnn_core/parametrizations/__init__
toc_min_heading_level: 2
toc_max_heading_level: 2
pagination_prev: null
pagination_next: null
---

# __init__.py

`utils/logicNN_core/src/logicnn_core/parametrizations/__init__.py`

Raw・Warp・Lightのパラメータ化、共通基底、設定から構築するbuilderを再公開します。

{/* source-sha256: f539d6db26faf02f9e93fbe357889c62f6d2ba2b07231c1e4e2118e37ea92764 */}

このファイルには関数・クラスの定義はありません。パッケージの初期化や、別ファイルで定義した名前の公開に使用します。

<details>
<summary>ファイルの内容を開く</summary>

```python
"""実装済みのLUTパラメータ化と設定に基づくbuilderを公開する。"""

from .base import LUTParametrization
from .builder import build_parametrization
from .light import LightLUTParametrization
from .raw import RawLUTParametrization
from .warp import WarpLUTParametrization

__all__ = ["LUTParametrization", "RawLUTParametrization", "WarpLUTParametrization", "LightLUTParametrization", "build_parametrization"]
```

</details>

## 関連ファイル

- [utils/logicNN_core/src/logicnn_core/parametrizations/base.py](/code-reference/utils/logicNN_core/src/logicnn_core/parametrizations/base)
- [utils/logicNN_core/src/logicnn_core/parametrizations/builder.py](/code-reference/utils/logicNN_core/src/logicnn_core/parametrizations/builder)
- [utils/logicNN_core/src/logicnn_core/parametrizations/light.py](/code-reference/utils/logicNN_core/src/logicnn_core/parametrizations/light)
- [utils/logicNN_core/src/logicnn_core/parametrizations/raw.py](/code-reference/utils/logicNN_core/src/logicnn_core/parametrizations/raw)
- [utils/logicNN_core/src/logicnn_core/parametrizations/warp.py](/code-reference/utils/logicNN_core/src/logicnn_core/parametrizations/warp)
