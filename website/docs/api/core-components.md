---
sidebar_position: 4
title: coreの構成と設定
---

# coreの構成と設定

## 層の入口

層は `logicnn_core.layers` からimportします。`ResidualLogicBlock` は `logicnn_core.modules` からimportします。

| クラス | 役割 |
|---|---|
| `LogicDense` | 最後の特徴軸に対する論理ゲート層。batchなどの先行軸は維持 |
| `LogicConv2d`／`LogicConv3d` | 固定の空間接続と論理treeによる畳み込み |
| `OrPooling2d`／`OrPooling3d` | 通常計算ではmax pooling、回路出力モードでは空間領域のbool OR |
| `GroupSum` | 最終特徴を連続するgroupに分けて合計し、必要なbias・tau演算を適用 |
| `FixedBinarization` | 固定閾値による二値化 |
| `DummyBinarization` | 通常はfloat32への変換、回路出力モードではbool入力をそのまま返す。閾値比較はしない |
| `SoftBinarization` | 学習時は滑らかな閾値計算、評価時は離散化 |
| `LearnableBinarization` | 学習可能な閾値差分を所有する二値化 |
| `ResidualLogicBlock` | 主経路と明示的shortcutをOR結合。2D／3D対応 |

`GroupSum(groups, tau=..., bias=...)` では、最後の特徴数がgroupsで割り切れる必要があります。余りを捨てません。Residualの両経路も同じshapeが必要で、暗黙のbroadcastは使いません。

## LUTと接続の設定

`logicnn_core.layer_settings` の不変設定型をモデル内で使います。通常のアプリ設定JSONに同じ項目を追加するものではありません。

- `LUTConfig(kind="raw", num_inputs=2)`：Rawの二入力16論理関数選択。
- `kind="warp"`：Walsh基底で論理関数を表す方式。
- `kind="light"`：真理値表係数による方式。
- `ConnectionConfig(kind="fixed")`：固定接続。
- `ConnectionConfig(kind="learnable")`：Denseの学習可能接続。候補選択・独自勾配は旧手法を維持。

方式ごとのrank・初期化・sampling・温度などの条件は [詳細設計](../specifications/logicnn-core/detailed-design.md) を参照してください。**学習対応と回路出力対応は同じ範囲ではありません。** LUTの回路出力は現在rank 2のみです。rank 1／4／6は明示的に拒否し、別rankへ置換しません。これらの回路対応は段階実装の未完部分で、学習機能を削除したものではありません。

`warp`パラメータ化本体は実装済みです。一方、補助正則化の名前 `warp` は別の機能で、承認済みの保留として指定時に明示エラーとなります。

## 数学関数と下位モジュール

`logicnn_core.functional` は論理関数、sampling、Walsh／Light計算、組合せ、正則化を公開します。`parametrizations/` は外側の層が所有する重みを計算へ変換し、`connections/` は入力の接続を管理します。

一般的なモデル作成では層と設定型を使えばよく、下位の接続indexや内部IRを直接書き換える必要はありません。

## モード切替え

`logicnn_core.set_export_mode(model, enabled=True)` は共有moduleを重複更新せず回路出力状態を伝播します。回路出力中の `train(True)` はエラーです。戻すときは `set_export_mode(model, False)` を明示してから `model.train()` を呼びます。

`Circuit.from_model()` はこの処理をコピー上で行うため、通常の回路生成で元モデルを手動切替えする必要はありません。
