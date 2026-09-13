---
title: __init__.py
sidebar_label: __init__.py
slug: /code-reference/utils/trainer/__init__
toc_min_heading_level: 2
toc_max_heading_level: 2
pagination_prev: null
pagination_next: null
---

# __init__.py

`utils/trainer/__init__.py`

1 epochの学習・評価と、それらの集計結果を表す `EpochMetrics` を公開します。

{/* source-sha256: 3e8ab248d1f7133f1650a0629500ffedfe537c9f0a30a72b7d88027cde8b3eb6 */}

このファイルには関数・クラスの定義はありません。パッケージの初期化や、別ファイルで定義した名前の公開に使用します。

<details>
<summary>ファイルの内容を開く</summary>

```python
"""1 epochの学習・評価と、その集計結果の型を公開する。"""

from .epoch import EpochMetrics, evaluate, train_one_epoch

__all__ = ["EpochMetrics", "evaluate", "train_one_epoch"]
```

</details>

## 関連ファイル

- [utils/trainer/epoch.py](/code-reference/utils/trainer/epoch)
