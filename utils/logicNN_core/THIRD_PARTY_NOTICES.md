# Third-party notices

## torchlogix / difflogic

logicNN-coreは、torchlogixの機能とアルゴリズムを基礎資料として、汎用論理ゲートニューラルネットワークのライブラリを再設計するプロジェクトです。開発リポジトリ内の参照元は `logicnn/utils/torchlogix/` です。これは由来の記録であり、インストール後の実行に必要なパスではありません。

参照元のMITライセンスには次の著作権表示があります。

```text
Copyright (c) 2021-2023 Dr. Felix Petersen
Copyright (c) 2024-present Dr. Lino Gerlach
```

許諾条件および免責条項を含む全文は、同梱の `LICENSE` に保持しています。wheel、sdistの両方にこのnoticeとLICENSEを同梱します。

## 現段階の由来と変更範囲

実装単位C1では、独立パッケージの構成、例外の入口、層設定のデータ型と検証規則を新しい仕様に沿って作成しました。Raw／Warp／Lightの選択肢や初期化の既定値などの既存仕様を参照し、新しい設定契約として整理しています。

- パッケージ版をGit由来ではなく、静的な `0.1.0` としました。
- 設定名を `LUTConfig`、`ConnectionConfig`、ゲートの入力本数を `num_inputs` としました。
- Rawのresidual初期化では、既存の式を維持するため、確率の許容範囲を `7/15 < p < 1` として事前検証します。既定値 `0.951` は維持します。
- torchlogixを実行時依存としてimportしません。

### C2: 論理関数とsampling

- `src/logicnn_core/functional/logic.py`: 旧 `src/torchlogix/functional.py` の `ID_TO_OP`、`compute_all_logic_ops_vectorized`、bool演算表、`apply_luts_export_mode` を基に、責務と公開名を整理しました。真理値表ID・多重線形連続式を維持し、broadcastとbool入力を明示的に扱います。
- export演算ではデータ依存のmask indexingを使わず、真理値表の4項を組み合わせます。16要素表のindex選択でID範囲を検証し、Tensor値のPython抽出なしでFX追跡できるようにしました。
- `src/logicnn_core/functional/sampling.py`: 旧 `softmax`、`sigmoid`、`gumbel_sigmoid`、`GradFactor` を基に再構成しました。hardのforward離散化とsoft勾配、Logisticノイズ、forward不変の勾配倍率を維持します。
- 温度・閾値・倍率の型と有限値を明示検証し、出力dtypeを保ちます。低precision内部演算、最大値を引くsoftmax、乱数端点の制限により数値安定性を改善し、Gumbel用のPyTorch generatorを追加しました。
- hard softmaxは低precision・高温時の確率丸めで選択先が変わらないよう元logitのargmaxを使い、真の同点では先頭indexを選びます。
- 有限のPython整数を温度・閾値・倍率として受けた場合、検証後にfloatへ正規化してPyTorchの整数scalar変換によるoverflowを避けます。

### C3: Walsh／Light・組合せ・正則化

- `src/logicnn_core/functional/walsh.py`: 旧FWHT、Walsh／Light基底と重み付き縮約を責務別に整理しました。非正規化変換と真理値表順の基底を維持し、任意の先行shapeと入力軸、空入力、非連続Tensorへ対応しました。materialize経路のeinsum係数軸名の衝突を避けています。
- `src/logicnn_core/functional/combinatorics.py`: 旧二項係数表・combinadic unranking・接続sampling・ID変換を基に再構成しました。colex順を維持し、多次元rank入力のshape消失を修正しました。全候補の巨大な表を作らない二分探索と疎な部分Fisher-Yates samplingを用い、乱数をPython randomからPyTorch seed／generatorへ統一しました。真理値表は符号値でなく明示boolとして返します。
- `src/logicnn_core/functional/regularization.py`: 旧L2／abs_sumのゲート別の式を維持し、承認済み仕様に従って全ゲート平均のscalar Tensorへ統一しました。低precision集計、有限値検証、失敗時に部分更新を残さない候補結果の事前検証を追加しました。補助正則化warpは承認済み保留として明示拒否し、式を移植・置換していません。

### C4: Raw／Warp／Lightパラメータ化

- `src/logicnn_core/parametrizations/raw.py`、`warp.py`、`light.py`: 旧 `src/torchlogix/parametrization.py` の同名方式を基に、初期化・forward・真理値表抽出を方式別のファイルへ分割しました。Rawの16ゲート選択、WarpのWalsh表現、Lightの真理値表係数、および方式別の初期化式を維持しています。
- `parametrizations/base.py`、`builder.py`: 共通interfaceと方式の選択を整理し、不変の生成時設定、実行中temperature、外側の層が所有する学習weightを分離しました。入力検証とMSB-firstのID変換を共通化しています。
- RawのGumbel選択は標準的な指数乱数からのノイズとC2の安定化softmaxを組み合わせ、極端な有限temperatureでも不用意に非有限値を生まないようにしています。
- Warpの真理値表抽出とforwardの加算順の違いで、0付近のbitが食い違う例を確認しました。その後CORE-DES-007が承認され、CPU float64の固定順FWHTで生成するbool表を二値evalと抽出の共通基準にしました。元deviceの維持、重みコピーの独立性、非有限値・変換overflowの拒否を追加しています。
- 二値evalは整数indexから表を参照し、materializeによる判定の差をなくしました。旧forwardとは境界付近でbitが変わる場合がある承認済みの変更です。学習・連続evalの式は維持し、連続入力が混在しても二値の入力組だけ表を使います。現parametrizationはキャッシュを持たず、毎回最新の重みから表を生成します。

### C5: 固定・学習可能な接続

- `connections/dense.py`: 旧 `src/torchlogix/connections.py` の固定・学習可能Dense接続を基に再構成しました。CORE-DES-008に従い、学習時のargmaxによる1候補の選択と、入力値を使うlogit勾配・softmaxで分配する入力勾配を維持しています。通常のsoft mixingやhard softmaxの勾配へ置換していません。
- 学習時のGumbelノイズは旧の式を使い、forwardとbackwardで共有します。評価は承認済みの乱数なしargmaxです。低precisionでの内部集計、任意先行shape、空batch、入力dtypeの保持、保存index検証を追加しています。候補の重複規則は新仕様を適用し、旧の全入力coverage制約は追加していません。
- `connections/convolution.py`: 旧の固定畳み込み接続を基に、2D／3Dの座標、channel group、tree levelの入力選択を整理しました。3Dの座標軸数を修正し、paddingの担当とrank軸の位置を統一しています。巨大な全組合せ表を作らず、C3のsamplingを再利用します。
- `connections/base.py`、`builder.py`: 共通interfaceと構造別の型付き生成入口を設けました。接続index・座標はpersistent buffer、学習する接続logitはParameterとして保存し、再構築時にshapeと参照を検証します。

### C6: 共通論理層とDense

- `layers/base.py`、`layers/dense.py`: 旧の共通基底とDense層を基に、生成時設定・LUT Parameter・接続moduleの所有関係を整理しました。接続選択、LUT演算、入力勾配倍率の順番は維持し、任意先行shape・空batchに対応しています。
- rank 2のexportはbool LUTと離散接続の独立snapshotを使用します。export中の学習への暗黙復帰を禁止し、既知の派生bufferはstate復元後の重みと接続から再生成する規則を追加しました。
- 旧互換用の属性aliasや未実装APIは公開せず、層は `logicnn_core.layers` から利用する構成としました。

### C7: 多段畳み込み・pooling・集約

- `layers/convolution.py`: 旧 `layers/conv.py` のnodeごとの初期化、tree level順のLUT計算、座標に基づく入力選択を維持しました。通常paddingとrank軸整列はC5へ移管し、3Dやtuple形状にも同じ方式で対応しています。旧で未適用だったConvの勾配倍率は、共通契約に従って元入力へ1回だけ適用しています。
- 全levelの正則化は承認済みの全ゲート平均に統一しました。再スケールは候補の事前検証と全levelの更新を分け、失敗時の部分更新を防ぎます。exportの表ID・接続は独立コピーし、通常checkpointから再生成できます。
- `layers/pooling.py`: 旧 `layers/pool.py` の通常float32 maxとbool ORを維持し、2D／3Dの共通処理、形状検証、空batch、明示的なexport状態管理を整理しました。
- `layers/group_sum.py`: 旧 `layers/groupsum.py` の連続groupの合計、bias加算後のtau除算、既定値時の演算省略を維持しました。PyTorchのdtypeと数値範囲に従い、独自のepsilonや暗黙の高precision化は追加していません。

### C8: 二値化・Residual・共通export切替

- `layers/binarization.py`: 旧 `layers/binarization.py` のFixed／Dummy／Soft／Learnableと閾値生成を基に再構成しました。CORE-DES-009に従い、学習時だけ後続差分をsoftplusで正にし、評価時は生差分を累積する旧計算を維持しています。単純差分による初期化、sampling、元dtypeのnormを用いたParameter勾配clipも維持し、独自の閾値共通化や並べ替えは行っていません。
- 閾値と入力の比較は元dtypeで行います。channel閾値は画像以外の `[B, C, L]` にも対応し、export snapshot、shape・値検証、deepcopy／Parameter差替え後のhook再装着を追加しました。低precisionのhard出力のfloat32昇格は旧層に合わせ、既に承認したC2の安定化sampling自体は維持しています。
- `modules/residual.py`: 旧 `modules/resblock.py` の2段ConvとOR接合を基に、2D／3Dの共通処理へ整理しました。downsample時に各Convの後でpoolingする旧構成を維持し、shortcutはモデル側の明示projectionと事前の形状検証で扱います。未承認の自動projectionや暗黙のbroadcastは追加しません。
- `export_mode.py`: 共有・入れ子moduleを一度ずつ処理する共通入口を実装しました。Residual自身のflag変更と子の再帰処理を分離し、旧互換aliasやCircuitの未実装stubは公開していません。

### C9: 回路データ・FX変換・公開入口

- `circuit/types.py`、`validation.py`: 旧 `circuit.py` のGate／SumReductionを基に、状態と処理を分離しました。IR IDと演算入力本数、参照順序、shape、係数を明示検証し、`LogicalOutput` で集約前の定数位置・重複・順序を独立保存する契約を追加しました。
- `circuit/builder.py`: 旧のFX配線をID Tensorへ写す方式とbool gate・sumの抽出を基に再構成しました。未登録・副作用演算を読み飛ばさず拒否し、graph自体を変更しない変換、sumの算術分岐、外部FXの必須出力対応情報を扱います。
- CORE-DES-012により、旧の係数へまとめる集約IRを、sum dtypeと順序付き `ScalarOperation` へ置き換えました。元のscalar型・Tensor rank・alpha・左右関係・cast・各出力dtypeを保存し、途中丸めを消す式の統合を行いません。集約前の論理出力対応は独立して維持します。
- `circuit/circuit.py`: 旧 `from_model` の元モデル変更をやめ、独立コピーを元dtypeのままCPU・eval・exportへ移して追跡します。CPU sampleを明示し、元モデルとCPU乱数状態を保護します。C・Verilogの入力は承認済みの二値化済み0／1で、画像の二値化前処理は生成しません。
- 回路の簡略化・JSON・実行・C／Verilog出力本体は後続単位です。これらの未実装APIを成功するstubとして公開していません。

### C10: 保存・簡略化・CPU回路実行

- `circuit/serialization.py`: 新しいschema version 1に従い独立実装しました。全field・型・版・出力対応を厳密に保存復元し、旧の係数表現や互換aliasは読み込みません。保存失敗時の既存file保護を追加しました。
- `circuit/simplification.py`: 承認済みの6passを新IR向けに独立実装しました。元の論理出力対応を全参照置換・到達性判定に含め、整数sumだけを同dtypeの先頭ADDへ整理し、浮動小数点演算列は変更しません。
- `circuit/runtime.py`: 保存IRをCPUのbool gateと順序付きPyTorch演算として実行する処理を独立実装しました。CORE-DES-013／014の入力・返却型と微小丸め差の契約を使用し、旧のPython／compiledで異なる返却型は引き継ぎません。数値dtypeの検証と既定dtype依存箇所の局所的な明示型処理を追加し、元のCPU kernelを模倣する独自演算engineは導入していません。
- `circuit/circuit.py`: これらの実行・保存・復元を公開APIへ統合し、簡略化はコピー上の全処理が成功してから所有IRを置換します。C backend・Verilogの未実装APIを成功するstubとして公開しません。

### C11: C／Verilog生成・native実行

- `circuit/exporters/c.py`、`verilog.py`、`writing.py` は新IR向けに実装しました。旧の論理式生成・sample方向のpackingという構成を参照しつつ、保存型の演算列と元の生出力境界を使用します。Verilogへ旧の加算・集約回路は生成しません。
- `circuit/compiled_runtime.py` はnative compiler選択、子process環境、共有libraryの寿命、CPU Tensor返却を分離しました。compile成功時だけ置換し、簡略化成功時に無効化します。
- 実行時の重複検証は2026-09-13の承認に従い削減しました。生成／読込の検証と、意味が黙って変わる境界は維持します。独自の検査ON/OFF APIは追加していません。

### C12: 参照モデル

- `examples/reference_models/dense.py` と `convolution.py` は旧 `src/torchlogix/models/dense.py` と `conv.py` のモデル構成を、不変presetと汎用組立てへ再構成しました。MNISTだけでなくCIFAR-10、Fashion-MNIST、JSCの構造を扱います。
- 旧クラス名は出典文字列として保存し、互換aliasを公開しません。LUTと前処理は呼出側で明示し、旧側で未登録の二値化方式と非整除の出力幅は勝手に修正せず、同階層READMEへ保留情報を残しています。
- これらは利用者のモデル定義の参考となるexamplesで、coreのwheelへは含めません。旧学習済み重みや精度の再現を保証する移植ではありません。

今後、由来のある計算処理を追加・変更するときは、対象ファイルと変更内容をここへ追記し、元のライセンス表示を保持します。元の著作者がlogicNN-coreの再設計や変更を承認したことを意味するものではありません。
