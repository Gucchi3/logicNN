---
title: __init__.py
sidebar_label: __init__.py
slug: /code-reference/utils/logicNN_core/src/logicnn_core/layers/__init__
toc_min_heading_level: 2
toc_max_heading_level: 2
pagination_prev: null
pagination_next: null
---

# __init__.py

`utils/logicNN_core/src/logicnn_core/layers/__init__.py`

論理Dense・Conv・プーリング・GroupSum・二値化層と共通基底を、layersパッケージからimportできるように再公開します。

{/* source-sha256: 9268c5f045431df8784f2b4927de7d78d9e48a61589d7770d859fba2abcdb01d */}

このファイルには関数・クラスの定義はありません。パッケージの初期化や、別ファイルで定義した名前の公開に使用します。

<details>
<summary>ファイルの内容を開く</summary>

```python
"""実装済みの論理層と共通基底を公開する。"""

from .base import LogicLayer
from .binarization import Binarization, DummyBinarization, FixedBinarization, LearnableBinarization, SoftBinarization
from .convolution import LogicConv2d, LogicConv3d
from .dense import LogicDense
from .group_sum import GroupSum
from .pooling import OrPooling2d, OrPooling3d

__all__ = [
    "LogicLayer", "LogicDense", "LogicConv2d", "LogicConv3d", "OrPooling2d", "OrPooling3d", "GroupSum",
    "Binarization", "FixedBinarization", "DummyBinarization", "SoftBinarization", "LearnableBinarization",
]
```

</details>

## 関連ファイル

- [utils/logicNN_core/src/logicnn_core/layers/base.py](/code-reference/utils/logicNN_core/src/logicnn_core/layers/base)
- [utils/logicNN_core/src/logicnn_core/layers/binarization.py](/code-reference/utils/logicNN_core/src/logicnn_core/layers/binarization)
- [utils/logicNN_core/src/logicnn_core/layers/convolution.py](/code-reference/utils/logicNN_core/src/logicnn_core/layers/convolution)
- [utils/logicNN_core/src/logicnn_core/layers/dense.py](/code-reference/utils/logicNN_core/src/logicnn_core/layers/dense)
- [utils/logicNN_core/src/logicnn_core/layers/group_sum.py](/code-reference/utils/logicNN_core/src/logicnn_core/layers/group_sum)
- [utils/logicNN_core/src/logicnn_core/layers/pooling.py](/code-reference/utils/logicNN_core/src/logicnn_core/layers/pooling)
