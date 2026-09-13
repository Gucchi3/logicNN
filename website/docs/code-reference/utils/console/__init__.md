---
title: __init__.py
sidebar_label: __init__.py
slug: /code-reference/utils/console/__init__
toc_min_heading_level: 2
toc_max_heading_level: 2
pagination_prev: null
pagination_next: null
---

# __init__.py

`utils/console/__init__.py`

学習開始、epoch進捗、完了、エラー、中断を表示する関数を公開します。

{/* source-sha256: 38c66543823bb9348989b8422a7e50735c0fb176f9ec04eed500b4789d48643b */}

このファイルには関数・クラスの定義はありません。パッケージの初期化や、別ファイルで定義した名前の公開に使用します。

<details>
<summary>ファイルの内容を開く</summary>

```python
"""通常実行の開始・進捗・完了と利用者向け診断のRich表示を公開する。"""

from .rich_display import show_complete, show_epoch, show_error, show_interrupted, show_start

__all__ = ["show_start", "show_epoch", "show_complete", "show_error", "show_interrupted"]
```

</details>

## 関連ファイル

- [utils/console/rich_display.py](/code-reference/utils/console/rich_display)
