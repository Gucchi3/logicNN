---
title: __init__.py
sidebar_label: __init__.py
slug: /code-reference/utils/logicNN_core/src/logicnn_core/modules/__init__
toc_min_heading_level: 2
toc_max_heading_level: 2
pagination_prev: null
pagination_next: null
---

# __init__.py

`utils/logicNN_core/src/logicnn_core/modules/__init__.py`

ResidualLogicBlockをmodulesパッケージの公開APIとして再公開します。

{/* source-sha256: a76f865b03c5370b354e18dfe71fef82759f5bbbe2a0d3c5c82c11fc1ad0026a */}

このファイルには関数・クラスの定義はありません。パッケージの初期化や、別ファイルで定義した名前の公開に使用します。

<details>
<summary>ファイルの内容を開く</summary>

```python
"""実装済みの複合論理moduleを公開する。"""

from .residual import ResidualLogicBlock

__all__ = ["ResidualLogicBlock"]
```

</details>

## 関連ファイル

- [utils/logicNN_core/src/logicnn_core/modules/residual.py](/code-reference/utils/logicNN_core/src/logicnn_core/modules/residual)
