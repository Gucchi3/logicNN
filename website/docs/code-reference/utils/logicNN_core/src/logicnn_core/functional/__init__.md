---
title: __init__.py
sidebar_label: __init__.py
slug: /code-reference/utils/logicNN_core/src/logicnn_core/functional/__init__
toc_min_heading_level: 2
toc_max_heading_level: 2
pagination_prev: null
pagination_next: null
---

# __init__.py

`utils/logicNN_core/src/logicnn_core/functional/__init__.py`

論理演算、組合せ抽選、温度付き選択、正則化、Walsh・Light基底の公開関数をまとめます。各関数の実装は同じフォルダ内の用途別ファイルにあります。このファイル自身には関数定義はありません。

{/* source-sha256: 25b5d58fe4fcb8a09e89bfcaabd6610cabbcd27bd56966cfc411c6bc77983f3d */}

このファイルには関数・クラスの定義はありません。パッケージの初期化や、別ファイルで定義した名前の公開に使用します。

<details>
<summary>ファイルの内容を開く</summary>

```python
"""論理ゲートNNの独立した数学関数を公開する。"""

from .combinatorics import build_binomial_table, sample_unique_combinations, truth_table_from_id, unrank_combinations
from .logic import all_binary_logic_outputs, apply_binary_lut, apply_export_luts, apply_export_truth_tables
from .regularization import regularization_loss, rescale_weights_
from .sampling import gumbel_sigmoid, scale_gradient, temperature_sigmoid, temperature_softmax
from .walsh import fast_walsh_hadamard, light_basis, walsh_basis, weighted_light_basis_sum, weighted_walsh_basis_sum

__all__ = [
    "all_binary_logic_outputs",
    "apply_binary_lut",
    "apply_export_luts",
    "apply_export_truth_tables",
    "build_binomial_table",
    "fast_walsh_hadamard",
    "gumbel_sigmoid",
    "light_basis",
    "regularization_loss",
    "rescale_weights_",
    "sample_unique_combinations",
    "scale_gradient",
    "temperature_sigmoid",
    "temperature_softmax",
    "truth_table_from_id",
    "unrank_combinations",
    "walsh_basis",
    "weighted_light_basis_sum",
    "weighted_walsh_basis_sum",
]
```

</details>

## 関連ファイル

- [utils/logicNN_core/src/logicnn_core/functional/combinatorics.py](/code-reference/utils/logicNN_core/src/logicnn_core/functional/combinatorics)
- [utils/logicNN_core/src/logicnn_core/functional/logic.py](/code-reference/utils/logicNN_core/src/logicnn_core/functional/logic)
- [utils/logicNN_core/src/logicnn_core/functional/regularization.py](/code-reference/utils/logicNN_core/src/logicnn_core/functional/regularization)
- [utils/logicNN_core/src/logicnn_core/functional/sampling.py](/code-reference/utils/logicNN_core/src/logicnn_core/functional/sampling)
- [utils/logicNN_core/src/logicnn_core/functional/walsh.py](/code-reference/utils/logicNN_core/src/logicnn_core/functional/walsh)
