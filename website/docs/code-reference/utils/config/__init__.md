---
title: __init__.py
sidebar_label: __init__.py
slug: /code-reference/utils/config/__init__
toc_min_heading_level: 2
toc_max_heading_level: 2
pagination_prev: null
pagination_next: null
---

# __init__.py

`utils/config/__init__.py`

設定の読み込み関数と、各設定区分を表すPydanticモデルを公開します。

{/* source-sha256: 3d65ce93e68a11de1baba02d4b6161a34918baea12a1bb704aeb75260977b5bc */}

このファイルには関数・クラスの定義はありません。パッケージの初期化や、別ファイルで定義した名前の公開に使用します。

<details>
<summary>ファイルの内容を開く</summary>

```python
"""設定スキーマと設定ファイル読込を公開する。"""

from .loader import load_config
from .schema import AppConfig, CircuitConfig, DataConfig, LossConfig, ModelConfig, OptimizerConfig, RunConfig, SchedulerConfig, TrainConfig

__all__ = (
    "AppConfig",
    "CircuitConfig",
    "DataConfig",
    "LossConfig",
    "ModelConfig",
    "OptimizerConfig",
    "RunConfig",
    "SchedulerConfig",
    "TrainConfig",
    "load_config",
)
```

</details>

## 関連ファイル

- [utils/config/loader.py](/code-reference/utils/config/loader)
- [utils/config/schema.py](/code-reference/utils/config/schema)
