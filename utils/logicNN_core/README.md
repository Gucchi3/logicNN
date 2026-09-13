# logicNN-core

PyTorch向けの汎用論理ゲートニューラルネットワークライブラリです。torchlogixを基礎資料として、責務・命名・公開APIを再設計しています。特定のデータセットや外側のlogicNNアプリケーションには依存しません。

## 現在の実装範囲

パッケージ基盤・設定、数学関数、Raw／Warp／Lightパラメータ化、固定・学習可能な接続、Dense・2D／3D畳み込み・OR pooling・GroupSum・二値化・Residualに加え、回路構築・簡略化・JSON保存・CPU実行・C／Verilog生成・native C実行を実装しています。版番号はGit情報を参照しない固定値 `0.1.0` です。詳細ドキュメントは今後整理します。

C4で見つかったWarpの二値出力の不一致は、承認済みCORE-DES-007に従って修正しました。CPU float64・固定順FWHTで生成するbool表を二値evalと抽出の共通基準にし、学習の式は維持しています。CORE-DES-008に従い、学習可能接続はtorchlogixのhard選択と独自勾配を維持しました。C8の学習可能二値化はCORE-DES-009に従い、trainでは後続差分をsoftplusで正にし、eval／exportでは生差分を累積する旧計算を維持します。evalの閾値を独自に並べ替えず、trainとevalの一致やeval時の昇順を保証しません。

- 配布名: `logicNN-core`
- Python import名: `logicnn_core`
- Python: 3.10以上
- 基本依存: PyTorch、NumPy
- 現在の入口: `logicnn_core.__version__`、`logicnn_core.LogicNNCoreError`、`logicnn_core.set_export_mode`、`logicnn_core.Circuit`
- モデル内の設定型: `logicnn_core.layer_settings.LUTConfig`、`ConnectionConfig`
- 数学関数: `logicnn_core.functional`。16論理演算、export用LUT適用、温度付き選択、Gumbel二値化、勾配倍率、Walsh／Light、組合せ、正則化・再スケール
- パラメータ化: `logicnn_core.parametrizations` の基底class、Raw／Warp／Light、`build_parametrization`
- 接続: `logicnn_core.connections` の固定／学習可能Dense、固定2D／3D Conv、共通基底と構造別builder
- 論理層: `logicnn_core.layers` の `LogicLayer`、`LogicDense`、`LogicConv2d`、`LogicConv3d`、`OrPooling2d`、`OrPooling3d`、`GroupSum`。層はrootから再exportせず、layersからimportします。
- 二値化: `logicnn_core.layers` の `Binarization`、`FixedBinarization`、`DummyBinarization`、`SoftBinarization`、`LearnableBinarization`
- 複合層: `logicnn_core.modules.ResidualLogicBlock`。2D／3Dに対応し、shortcutのprojectionはモデル側から明示的に与えます。

設定型はモデル定義ファイル内で使用するためのものです。アプリケーションのJSON設定を増やすものではありません。C1のimportと設定値検証自体はPyTorch・NumPyを読み込みません。

CPU／CUDA、独立配布、C実行およびVerilatorによる実シミュレーションを検証済みです。検証範囲と結果は外側アプリケーションの `log/accuracy_comparison_50epochs/` に保持しています。開発テスト本体は運用フォルダ外へ退避しています。

```python
from logicnn_core import __version__
from logicnn_core.layer_settings import ConnectionConfig, LUTConfig

lut         = LUTConfig(kind="raw", num_inputs=2)
connections = ConnectionConfig(kind="fixed", init="random")
```

DenseとConvは学習・評価・bool exportを利用できます。対応入力数はRawが2、Warpが1／2／4／6、Lightが2／4／6です。共通 `set_export_mode` は共有moduleを重複処理せず、モデル全体を切り替えます。二値化・Residual・pooling・GroupSumとの結合も検証しています。回路構築、`Circuit.evaluate()`／`evaluate_logical_outputs()`／`simplify()`、dict／JSON往復、`c_source()`／`verilog_source()`／`write_c()`／`write_verilog()`／`compile()` を利用できます。

C・Verilogとも二値化済み0／1入力を前提にし、画像の二値化処理は生成しません。外部FXは `logicnn_core.circuit.FxSignalReference` またはbool定数の列による `logical_outputs` を必須とし、通常のモデル変換は元の出力対応を自動記録します。

回路データは元の計算順序・数値型を保存します。sum後の演算は `ScalarOperation` の順序付き列とし、Python scalarとTensor定数の区別、rank、alpha、左右関係、castを保持します。Cは集約後スコア、Verilogは集約前の生bitを出力します。生bitの数・順序・定数・重複は簡略化後も維持します。

回路実行はCPU Tensorを返し、学習の勾配追跡を行いません。入力はbatch付きTensor／NumPy配列の厳密な0／1で、通常のモデル実行や学習のdeviceは変更しません。簡略化は独立コピー上で成功してから置換し、JSONも書込失敗時に既存fileを保護します。`compile()` 成功後は `evaluate(inputs, compiled=True)` でnative Cを実行できます。簡略化後は再compileが必要です。compilerがない場合にPythonへ自動で切り替えることはありません。

native compileにはWindowsではMSVC、その他ではcc／gcc／clangを使用します。現在の実機検証はWindowsのMSVCです。Cはbool、uint8、int8／16／32／64、float32／64に対応し、float16／bfloat16の演算と浮動小数点から整数へのcastは診断付きで拒否します。これはIR／Python側の機能削除ではなく、Cでの未対応範囲です。`pack_bits=8/16/32/64` は複数sampleの入力をbit詰めする設定で、画像の画素値ではありません。

CORE-DES-014に従い、bit・整数・出力の数と順序は厳密に維持し、小数は元の演算順序・型を保った微小な丸め差を許容します。CPU kernelの模倣や、本番での元モデルとの自動比較は追加しません。既存の学習処理や合格済みテストを一括で書き換えず、対応できる箇所へ限定して適用します。最新のテスト結果はDocusaurusの実装・検証記録を参照してください。

補助正則化の方式名 `warp` は、2026-09-12の承認により実装保留としました。`functional.regularization_loss(weights, kind="warp")` とDense／Conv層の同入口はValueErrorとし、黙ってゼロを返す処理や他方式への置換は行いません。`LUTConfig(kind="warp")` と実装済みのWarp方式本体は維持します。

Warpの通常evalはキャッシュせず、呼出しごとに最新の外部重みから表を生成します。層のexportは二入力では表ID、それ以外ではbool表と、離散接続の独立snapshotを保持し、再enable／state復元で更新します。

## 単体での導入と配布

以下はこのディレクトリをカレントディレクトリとするコマンドです。

```shell
python -m pip install -e .
```

ビルドにはsetuptoolsとwheelを使用し、Git履歴や親アプリケーションを必要としません。

wheelとsdistには `LICENSE` と `THIRD_PARTY_NOTICES.md` を同梱します。wheelには開発者用テストや参照モデルのexamplesを含めません。

## 参照モデル

[参照モデル](examples/reference_models/README.md) では、MNIST・CIFAR-10・Fashion-MNIST・JSCのDense／Conv構成を不変presetと汎用組立てへ整理しています。特定のデータセットにcoreを限定するものではありません。旧側で未登録の二値化方式や非整除の出力幅は、同READMEへ情報を残して保留しています。

## ライセンスと由来

MIT Licenseです。参照元の著作権表示を `LICENSE` に保持しています。再設計部分と移植部分の由来は `THIRD_PARTY_NOTICES.md` に記録します。
