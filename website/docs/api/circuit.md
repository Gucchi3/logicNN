---
sidebar_position: 5
title: 回路API
---

# 回路API

`from logicnn_core import Circuit` で利用します。通常のPyTorchモデルによる学習や `model(inputs)` と、保存した回路の実行は別の入口です。

## 生成と保存

| API | 契約 |
|---|---|
| `Circuit.from_model(model, input_shape)` | 元モデルを独立コピーし、CPU・評価・回路出力モードで追跡。input_shapeにbatchは含めない |
| `Circuit.from_fx_graph(graph_module, input_shape, logical_outputs=...)` | 外部FXから生成。元の生出力対応が必須で、欠落時に推測しない |
| `Circuit.from_dict(data)`／`Circuit.from_json(path)` | 型・構造・出力対応を検証して独立回路を生成 |
| `to_dict()`／`write_json(path)` | 数値型、演算列、元出力対応を含めて保存。native状態は保存しない |
| `c_source(...)`／`verilog_source(...)` | ソース文字列を返す |
| `write_c(path, ...)`／`write_verilog(path, ...)` | 生成成功後に原子的に保存する |

外部FXの参照には `logicnn_core.circuit.FxSignalReference(node_name, flat_index)` またはbool定数を使います。通常の `from_model` では元の出力対応を自動記録します。

## 実行と簡略化

| API | 結果 |
|---|---|
| `evaluate(inputs)` | `[batch,*output_shape]`の集約後CPU Tensor |
| `evaluate_logical_outputs(inputs)` | `[batch,元の生出力数]`のCPU bool Tensor |
| `logical_output_ids()` | 元group順・bit順の出力ID tuple |
| `simplify(max_passes=1000)` | 候補上で簡略化し、成功したときだけ置換。返り値はNone |
| `validate()` | 保持するIRの構造と数値型を明示的に検証 |

入力はTensorまたはNumPy配列で、batch付きの厳密な0／1です。単一標本もbatch=1にします。空batchにも対応します。画像の正規化や二値化、学習勾配は生成しません。

bitと整数は一致を守り、小数は元の演算順序・型を保持した微小な丸め差を許容します。数値契約を満たせない変換を、別precisionや大きな許容誤差で隠しません。

## compile

`compile(optimization_level=1, pack_bits=None)` が成功した後、`evaluate(inputs, compiled=True)` を使用できます。最適化水準は0〜3、packingはNoneまたは8／16／32／64です。入力をsample方向にまとめても返り値は通常のCPU Tensorです。

簡略化成功後は再compileが必要です。compile失敗時は以前の成功状態を保持し、Python実行へ自動fallbackしません。compilerはWindowsではnative MSVC、その他ではcc／gcc／clangから選びます。

Cの現対応型はbool、uint8、int8／16／32／64、float32／64です。float16／bfloat16の演算や浮動小数点から整数へのcastは未対応として拒否します。これはIR／Python側の対応を削除したものではありません。

## エラーと未完の範囲

変換・IR・native実行の問題は `LogicNNCoreError` の詳細・対処案を確認します。通常のTensor演算ではPyTorchの標準エラーもそのまま返ります。Alkaid連携などの保留は [実装・検証記録](../specifications/logicnn-core/verification.md) と区別して扱います。
