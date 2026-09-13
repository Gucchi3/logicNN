---
sidebar_position: 3
title: coreの使い方
---

# coreの使い方

次は画像やMNISTに依存しない、4入力・8個の論理ゲート・2出力の最小例です。coreを導入したPython環境で実行できます。

```python
import torch
from torch import nn

from logicnn_core import Circuit
from logicnn_core.layer_settings import LUTConfig
from logicnn_core.layers import GroupSum, LogicDense

torch.manual_seed(0)
model = nn.Sequential(LogicDense(4, 8, lut=LUTConfig()), GroupSum(2, tau=4)).eval()
inputs = torch.tensor([[0, 0, 1, 1], [1, 0, 1, 0]], dtype=torch.bool)

with torch.no_grad():
    reference = model(inputs.float())

circuit = Circuit.from_model(model, input_shape=(4,))
circuit.simplify()
scores = circuit.evaluate(inputs)
raw_bits = circuit.evaluate_logical_outputs(inputs)

torch.testing.assert_close(scores, reference, rtol=0, atol=0)
assert scores.shape == (2, 2)
assert raw_bits.shape == (2, 8)

circuit.write_json("example.json")
circuit.write_c("example.c")
circuit.write_verilog("example.v")
restored = Circuit.from_json("example.json")
assert torch.equal(restored.evaluate(inputs), scores)
```

生成される `example.c` は集約後の2スコア、`example.v` は集約前の8 bitを出力します。上の例はCをcompileしないため、コンパイラは不要です。

学習するときは通常のPyTorchと同様にoptimizerを用意し、浮動小数点入力を渡して `model.train()`、lossのbackward、optimizerのstepを行います。回路生成は学習後のモデルを独立コピーするため、元モデルのdeviceや重みを変更しません。

## native Cで実行する

対応コンパイラがある環境では、上の例に続けて実行できます。

```python
circuit.compile(optimization_level=1, pack_bits=32)
native_scores = circuit.evaluate(inputs, compiled=True)
torch.testing.assert_close(native_scores, scores, rtol=0, atol=0)
```

`pack_bits=32` は32標本ずつ入力bitをまとめる内部形式です。画像の画素値や重みの量子化bit数ではありません。入力は通常のbatch付き0／1 Tensorのままで、packingは実行APIが行います。Pythonへ返る値はCPU Tensorです。

未対応compilerや演算を別の経路で成功したことにするfallbackはありません。対応範囲は [回路API](./circuit.md) を確認してください。
