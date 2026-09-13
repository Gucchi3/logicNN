---
title: __init__.py
sidebar_label: __init__.py
slug: /code-reference/utils/export/__init__
toc_min_heading_level: 2
toc_max_heading_level: 2
pagination_prev: null
pagination_next: null
---

# __init__.py

`utils/export/__init__.py`

学習済みモデルから回路成果物を生成・保存する `export_circuit()` を公開します。

{/* source-sha256: 720bebc995e88a08a522e83927f222cc3fa1f9d1905e50adeacb6427c9788450 */}

このファイルには関数・クラスの定義はありません。パッケージの初期化や、別ファイルで定義した名前の公開に使用します。

<details>
<summary>ファイルの内容を開く</summary>

```python
"""学習済みモデルから回路成果物を順次保存する入口を公開する。"""

from .circuit_exporter import export_circuit

__all__ = ["export_circuit"]
```

</details>

## 関連ファイル

- [utils/export/circuit_exporter.py](/code-reference/utils/export/circuit_exporter)
