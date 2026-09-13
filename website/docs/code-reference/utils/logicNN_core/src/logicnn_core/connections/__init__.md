---
title: __init__.py
sidebar_label: __init__.py
slug: /code-reference/utils/logicNN_core/src/logicnn_core/connections/__init__
toc_min_heading_level: 2
toc_max_heading_level: 2
pagination_prev: null
pagination_next: null
---

# __init__.py

`utils/logicNN_core/src/logicnn_core/connections/__init__.py`

接続クラスと生成関数の公開窓口です。固定接続、学習可能な全結合接続、畳み込み接続を、このパッケージからまとめて読み込めます。このファイル自身には関数定義はありません。

{/* source-sha256: 672c1853f1c8e29dfe01e76ed8e4c192171612b506a0bc708eabf7493430426d */}

このファイルには関数・クラスの定義はありません。パッケージの初期化や、別ファイルで定義した名前の公開に使用します。

<details>
<summary>ファイルの内容を開く</summary>

```python
"""固定・学習可能な接続と、構造別の型付きbuilderを公開する。"""

from .base import Connections
from .builder import build_conv_connections, build_dense_connections
from .convolution import FixedConvConnections
from .dense import FixedDenseConnections, LearnableDenseConnections

__all__ = [
    "Connections",
    "FixedDenseConnections",
    "LearnableDenseConnections",
    "FixedConvConnections",
    "build_dense_connections",
    "build_conv_connections",
]
```

</details>

## 関連ファイル

- [utils/logicNN_core/src/logicnn_core/connections/base.py](/code-reference/utils/logicNN_core/src/logicnn_core/connections/base)
- [utils/logicNN_core/src/logicnn_core/connections/builder.py](/code-reference/utils/logicNN_core/src/logicnn_core/connections/builder)
- [utils/logicNN_core/src/logicnn_core/connections/convolution.py](/code-reference/utils/logicNN_core/src/logicnn_core/connections/convolution)
- [utils/logicNN_core/src/logicnn_core/connections/dense.py](/code-reference/utils/logicNN_core/src/logicnn_core/connections/dense)
