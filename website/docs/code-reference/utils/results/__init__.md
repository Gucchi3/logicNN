---
title: __init__.py
sidebar_label: __init__.py
slug: /code-reference/utils/results/__init__
toc_min_heading_level: 2
toc_max_heading_level: 2
pagination_prev: null
pagination_next: null
---

# __init__.py

`utils/results/__init__.py`

実行ディレクトリ、設定、学習情報、履歴、グラフ、チェックポイント、テスト結果を保存する入口を公開します。

{/* source-sha256: f9d5bf938366f0217fbaca84236d145956541892a4e8b341876238d3e0a36241 */}

このファイルには関数・クラスの定義はありません。パッケージの初期化や、別ファイルで定義した名前の公開に使用します。

<details>
<summary>ファイルの内容を開く</summary>

```python
"""実行成果物・履歴・チェックポイントの保存入口を公開する。"""

from .checkpoint import load_checkpoint, save_checkpoint
from .metadata import save_resolved_config, save_training_info
from .metrics import EpochRecord, append_epoch_record, save_curves, save_test_metrics
from .run_directory import create_run_directory

__all__ = [
    "EpochRecord", "append_epoch_record", "create_run_directory", "load_checkpoint", "save_checkpoint", "save_curves", "save_resolved_config",
    "save_test_metrics", "save_training_info",
]
```

</details>

## 関連ファイル

- [utils/results/checkpoint.py](/code-reference/utils/results/checkpoint)
- [utils/results/metadata.py](/code-reference/utils/results/metadata)
- [utils/results/metrics.py](/code-reference/utils/results/metrics)
- [utils/results/run_directory.py](/code-reference/utils/results/run_directory)
