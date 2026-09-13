---
title: __init__.py
sidebar_label: __init__.py
slug: /code-reference/utils/trainer/optimization/__init__
toc_min_heading_level: 2
toc_max_heading_level: 2
pagination_prev: null
pagination_next: null
---

# __init__.py

`utils/trainer/optimization/__init__.py`

学習用・評価用の損失関数、optimizer、schedulerの生成関数を公開します。

{/* source-sha256: d0d8988e1f624e02971bafa11e587fd6b78db1815e1498e3a4994590f9925580 */}

このファイルには関数・クラスの定義はありません。パッケージの初期化や、別ファイルで定義した名前の公開に使用します。

<details>
<summary>ファイルの内容を開く</summary>

```python
"""学習・評価の損失と、optimizer・schedulerの生成入口を公開する。"""

from .loss import build_evaluation_loss, build_training_loss
from .optimizer import build_optimizer
from .scheduler import build_scheduler

__all__ = ["build_evaluation_loss", "build_optimizer", "build_scheduler", "build_training_loss"]
```

</details>

## 関連ファイル

- [utils/trainer/optimization/loss.py](/code-reference/utils/trainer/optimization/loss)
- [utils/trainer/optimization/optimizer.py](/code-reference/utils/trainer/optimization/optimizer)
- [utils/trainer/optimization/scheduler.py](/code-reference/utils/trainer/optimization/scheduler)
