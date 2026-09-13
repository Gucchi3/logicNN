# 参照モデル

旧torchlogixのデータセット別モデルに書かれていた層幅、段数、入力bit数、tau、接続の指定を、
`DensePreset` / `ConvolutionPreset` と共通のモデル組立てへ整理したソースコード例です。
学習済み重み、データ取得、学習ループ、旧クラス名の互換APIは含みません。
`logicnn_core` のトップレベルからはimportせず、配布wheelにも含めません。

## 1. どこを見るか

| ファイル | 内容 |
| --- | --- |
| `dense.py` | 不変の `DensePreset`、`DENSE_PRESETS`、汎用 `DenseReferenceModel` |
| `convolution.py` | 不変の `ConvolutionStage` / `ConvolutionPreset`、`CONVOLUTION_PRESETS`、汎用 `ConvolutionReferenceModel` |
| このREADME | 入力の意味、出典との対応、推測せず保留した構成 |

旧名は各presetの `source_names` に出典文字列として残します。旧クラスを継承したり、別名として公開したりはしません。
数値は同梱されていた `torchlogix/models/dense.py` と `torchlogix/models/conv.py` に由来します。
論文の精度を再現したという意味ではなく、モデルコードで確認できる構造を移したものです。
ライセンスは [LICENSE](../../LICENSE) と [THIRD_PARTY_NOTICES.md](../../THIRD_PARTY_NOTICES.md) を参照してください。

## 2. 入力とLUTの指定

`input_shape` は、batch軸を除いた **前処理後・bit展開後のshape** です。
MNISTの1 bit画像は `(1, 28, 28)`、3 bitのCIFAR-10は `(9, 32, 32)`、
16特徴を20 bitずつ展開したJSCは `(320,)` として記録します。
Fashion-MNIST DWNは旧既定の `feature_dim=-2` を維持し、7 bitを画像の幅へ結合した `(1, 28, 196)` です。
学習可能接続を使うときは、0/1値も浮動小数点Tensorで入力してください。

前処理は `preprocessing=` へ明示的に `FixedBinarization` などのmoduleを渡します。
省略した場合は `nn.Identity` で、画像の正規化や二値化を自動で追加しません。
CIFAR-10は旧constructorに従い `feature_dim=1` とし、bitをchannelへまとめます。
Fashion-MNIST DWNはその上書きをせず、`FixedBinarization(thresholds)` の既定 `feature_dim=-2` を使います。
入力 `[B,1,28,28]` へ末尾bit軸を追加してから幅と結合するため、flatten後は各pixelの直後にその7 bitが並びます。
channelへbitを展開するCIFAR-10方式に変更すると、同じ総要素数でもbit順序が変わります。
閾値の値や算出に使うデータは呼出側が決め、presetだけから推測しません。

Denseの旧定義が直接規定していたのは主にflatten後の要素数ですが、旧constructorと既定のbit結合軸も合わせて保持します。
旧呼出側が別の軸へ明示変更していた実験を再現する場合は、その前処理と出力shapeを `replace(preset, input_shape=...)` で指定してください。
ここでは未指定の閾値やsampling条件まで含む旧実験の精度再現を保証しません。

LUT表現も `lut=LUTConfig(...)` として明示指定します。旧モデルの `Rank4` / `Rank6` という名前は
幅の選択に使われていましたが、実際のLUT方式は外部引数に委ねられていました。
したがってWarpまたはLightを勝手に選びません。Denseではpresetの `num_inputs` と指定したrankの一致を確認します。
Conv構成は外部LUT指定をそのまま使います。

## 3. 使い方

coreを導入した環境で、カレントディレクトリをこのフォルダの親 `examples/` にしてください。

```python
import torch
from logicnn_core.layer_settings import LUTConfig
from reference_models.dense import DENSE_PRESETS, DenseReferenceModel

preset = DENSE_PRESETS["mnist_tiny"]
model = DenseReferenceModel(preset, lut=LUTConfig(kind="raw", num_inputs=2))
inputs = torch.randint(0, 2, (1, *preset.input_shape)).float()
scores = model(inputs)  # [1, 10]
```

自分のデータセットに合わせた小型構成も、同じ組立てを使えます。

```python
from logicnn_core.layer_settings import LUTConfig
from reference_models.dense import DensePreset, DenseReferenceModel

preset = DensePreset(input_shape=(12,), hidden_features=(24, 12), num_classes=3, tau=2.0)
model = DenseReferenceModel(preset, lut=LUTConfig(kind="light", num_inputs=2))
```

`DenseReferenceModel` は `preprocessing → Flatten → LogicDense列 → GroupSum` です。
`learnable_layers` で指定した先頭の段だけ `learnable_connections` を使い、残りは `connections` を使います。
既定はそれぞれ新APIの学習可能接続と固定ランダム接続で、候補数やGumbel等は明示変更できます。

`ConvolutionReferenceModel` は `preprocessing → (LogicConv2d → OrPooling2d)列 → Flatten → LogicDense列 → GroupSum` です。
各段の出力shapeから次の入力shapeを計算します。`channel_group_size` はpresetの各段が決め、
それ以外の接続条件は `conv_connections`、Denseの接続は `dense_connections` が担当します。

既存presetの変更には `dataclasses.replace(preset, ...)` を使えます。変更後は独自構成であり、元の参照構成そのものではありません。
大きなpresetは大量のメモリを使います。import時は形状情報だけを作り、実モデルは明示的に構築した場合だけ確保します。
特に学習可能接続で候補数を省略すると、前層の全特徴が候補となります。

## 4. 保持したDense構成

同一幅の繰返しは「幅 × 段数」と表記します。JSCは5クラス、それ以外は10クラスです。
ここでのbit数は入力特徴あたりの二値化出力本数で、重みの量子化bit数ではありません。

| 旧モデル群 | bit数・構造 | tau・接続 |
| --- | --- | --- |
| MNIST Tiny / Small / Medium | 1 bit、1000 / 8000 / 64000 × 5段 | 10 / 10 / `1/0.03` |
| MNIST SmallRank4 / SmallRank6 | 1 bit、4000 / 2330 × 5段 | 10。Small各rankのLearn1も保持 |
| CIFAR-10 Small（rank 2/4/6） | 3 bit、12000 / 6000 / 3000 × 4段 | `1/0.03` |
| CIFAR-10 Medium / Large / Large2 / Large4 | 3 bit・128000 × 4段、5 bit・256000 / 512000 / 1024000 × 5段 | 100 |
| CIFAR-10 Deep（rank 2/4/6） | 3 bit、12000 / 6000 / 4000、段数 `4 × {1,2,3,4,5,10,20}` | rank 2は `1/0.03`、rank 4/6は100 |
| CIFAR-10 MediumDeep（rank 2/4/6） | 3 bit、128000 / 56000 / 42000、4段または12段 | 100 |
| JSC Small（rank 2/4） | 10/20/50/100 bit、32000 / 16000 × 2段 | 50 |
| JSC Medium（rank 2/4/6） | 10/20/50/100 bit、128000 / 64000 / 42330 × 4段 | 50 |
| CIFAR-10 DWN（rank 2/6） | 10 bit、`(24000,24000)` / `(8000,)` | `1/0.03`。rank 2のLearn2も保持 |
| Fashion-MNIST DWN（rank 2/6） | 7 bit、入力shape `(1,28,196)`、`(8000,8000)` / `(2000,2000)` | `1/0.061` / `1/0.122`。両rankのLearn2も保持 |
| JSC DWN TinyRank6 | 200 bit、`(10,)` | `1/0.7`。Learn0/1 |
| JSC DWN SmallRank6 | 1/2/5/10/20/50/100/200 bit、`(50,)` | `1/0.3`。各bit数のLearn0/1 |
| JSC DWN Medium（rank 2/4/6） | 2/5/10/20/50/100 bit、`(1080,)` / `(540,)` / `(360,)` | 10。rank 6は200 bitのLearn0/1も保持 |
| JSC DWN LargeRank6 | 2/5/10/20/50/100/200 bit、`(2400,)` | `1/0.03`。200 bitのLearn1も保持 |

旧JSCの `n_input_bits=None` の基底クラスはbit数を呼出側が決める構成です。
ここでは旧コードで具体的に指定されたbit数を登録し、その他のbit数は新しい `DensePreset` または `replace` で指定します。
旧DWNの `n_learnable_layers=None` の中間クラスも、Learn0/1/2を決めるための基底であり、曖昧なまま実行presetにはしません。

## 5. 保持したConv構成

| preset群 | 入力とConv構造 | Dense幅・tau |
| --- | --- | --- |
| MNIST base / tiny / small / medium / large | `(1,28,28)`、channel幅 `k,3k,9k`、tree depth 3、kernel `5,3,3`、Conv padding 0。各段で2×2/stride 2 pooling、pool padding `0,1,1` | flatten `81k`、Dense `1280k,640k,320k`。`(k,tau)=(16,1),(4,1),(16,6.5),(64,28),(1024,35)` |
| CIFAR-10 small / medium / large | 入力bit数 `2,2,5`、channel幅 `k,4k,16k,32k`、tree depth 3、kernel 3/padding 1。各段で2×2/stride 2 pooling | flatten `128k`、Dense `1280k,640k,320k`。`(k,tau)=(32,20),(256,40),(512,280)` |
| CIFAR-10 small2 / medium2 | small / mediumと同じ形状。最初のConvだけchannel group size 1、後続は2 | 元のsmall / mediumと同じ |

通常のCIFAR-10 Convは全段のchannel group sizeが2です。
旧MNIST Convの `dummy` 前処理は閾値比較をせず、入力をfloat32へ変換します。
その変換も再現するときは `preprocessing=DummyBinarization()` を明示してください。
参照組立ての既定 `nn.Identity` は、用意済み入力のdtypeを前処理段では変更しません。
旧 `CNN` はMNIST baseと同じ幾何形状の汎用入口で、クラス数とtauは外部指定でした。
必要なら `mnist_base` の `num_classes` / `tau` を明示的に変更します。

## 6. 推測せず保留した構成

| 対象 | 旧コードにある知識 | 保留理由 |
| --- | --- | --- |
| Fashion-MNISTの通常DLGN（DWN以外） | 3 bit、5段、rank 2/4/6の幅8000/4000/2330、tau 10、各rankのLearn1 | `binarization="uniform"` が選択される一方、旧 `setup_binarization` のmodule登録にこの名前がない。閾値作成法のuniformとmodule方式を同一視して修正しない |
| MNIST DWN LargeRank2 | 3 bit、`(6000,6000)`、tau `1/0.071`、Learn0/1/2 | 未登録の `binarization="distributive"` を使用 |
| MNIST DWN SmallRank6 | 1 bit、`(1000,500)`、tau `1/0.245`、Learn0/1/2 | 同じ未登録方式を使用 |
| MNIST DWN LargeRank6 | 3 bit、`(2000,1000)`、tau `1/0.173`、Learn0/1/2 | 同じ未登録方式を使用 |
| JSC SmallRank6と10/20/50/100 bit派生 | 2段、幅10666、5クラス、tau 50 | 10666は5で割り切れずGroupSumの構造が成立しない。10665等へ丸めたり、クラス数を変えたりしない |

以上は情報の削除ではなく、構造と未決事項をこの一覧へ保存した保留です。
未登録の方式をどの新API構成で表すか、非整除の出力幅をどう定義するかが決まるまでは、実行用presetへ追加しません。

## 7. 検証範囲

開発用 `tests/core/test_reference_models.py` で、全presetの形状情報、出典の代表値、
小型の実Dense/Convによるforward/backward、前処理の差替え、最終クラス数を確認します。
Fashion-MNIST DWNのpixel→bit順とCIFAR-10のchannel→bit展開は、軸結合関数を使わない独立した期待値と比較します。
大規模presetの全実体化、実データ取得、論文精度の再学習、学習済み重みの互換性はこの試験に含みません。
