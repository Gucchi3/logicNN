---
sidebar_position: 1
title: 要件定義
---

# logicNN_core 要件定義書

## 1. 文書情報

- 文書状態: 草案
- 作成日: 2026-09-12
- 対象: 汎用論理ゲートニューラルネットワークライブラリ `logicNN_core`
- 配置: `logicNN/utils/logicNN_core/`
- Python import名: `logicnn_core`
- 基礎資料: `logicnn_old/utils/torchlogix/`
- ライセンス方針: MITライセンスと原著作者表示を継承

## 2. 背景

logicNNは当初、同梱したtorchlogixをローカル依存として直接使用する設計だった。しかし、外部ライブラリの構成、版情報、巨大な単一ファイル、アプリケーション側の独自修正との境界が分かりにくく、logicNNの目的である理解しやすさと改善しやすさを損なう可能性がある。

そこで、torchlogixが提供する論理ゲートNNの知識と機能を基礎に、責務、命名、公開API、テスト可能性および配布境界を再設計する。単なるMNIST専用実装ではなく、Dense、2D／3D畳み込み、複数LUT表現、複数接続方式、二値化、回路IRおよびコード生成を組み合わせられる汎用ライブラリとする。

## 3. 目的

1. PyTorchの `nn.Module` として組み合わせられる論理ゲートNN層を提供する。
2. 学習時の連続緩和と、評価時の離散論理演算を同じモデルで扱えるようにする。
3. 学習済みモデルを明示的な回路IRへ変換し、簡略化、実行、保存およびコード生成を可能にする。
4. MNIST、CIFAR、表形式データ、音声特徴量など、特定データセットに依存しないAPIを提供する。
5. logicNN内で保守しやすく、将来は単独パッケージとして切り出せる構造にする。
6. torchlogixの有用な要素を不用意に削除せず、維持、再配置または段階実装のいずれかを明記する。

## 4. 用語

| 用語 | 定義 |
|---|---|
| LUT | 入力ビット列から1ビットを返す真理値表。入力本数は公開設定でnum_inputs、数学説明ではrankと表す |
| 連続緩和 | 勾配法で学習できるよう、論理選択を連続値で近似する計算 |
| 離散評価 | 学習済みパラメータから一つの真理値表を選び、0／1の論理演算として評価する計算 |
| export mode | モデルを回路へ追跡できる、純粋な論理・添字演算へ切り替えた状態 |
| 回路IR | ゲート、接続、定数および集約を表す、実行先に依存しない中間表現 |
| 参照モデル | ライブラリの層を組み合わせた例。汎用公開APIには含めない |

## 5. 対象範囲

### 5.1 正式対象

- 2入力の全16論理関数と、LUT rankに応じた真理値表処理。
- Raw、Warp、LightのLUTパラメータ化。
- soft、hard、gumbel-soft、gumbel-hardの学習時サンプリング。
- 固定Dense接続、学習可能Dense接続および固定畳み込み接続。
- `LogicDense`、`LogicConv2d`、`LogicConv3d`。
- `OrPooling2d`、`OrPooling3d`、`GroupSum`。
- 固定、無処理、soft、学習可能の二値化層。
- `ResidualLogicBlock`。
- モデル全体のexport mode切替。
- PyTorchモデルまたはFX graphからの回路IR構築。
- 回路の定数畳み込み、不要ゲート除去、wire bypass、NOT融合、重複除去およびsum reduction簡約。
- 回路IRのPython実行、任意のCコンパイルによる高速実行、辞書／JSONの保存と復元。
- 完全な回路を評価するCコードと、最終集約を除いた論理ゲート部分だけを表すVerilogコードの生成。
- Alkaid連携を任意機能として分離した状態で維持すること。
- torchlogixに含まれるデータセット別モデルを参照モデルとして整理して残すこと。

### 5.2 初回リリースの優先順位

正式対象を削除せず、実装を依存関係に従って段階化する。最初の縦方向の完成範囲は、共通論理演算、Raw rank 2、固定Dense接続、`LogicDense`、`LogicConv2d`、`OrPooling2d`、`GroupSum`、回路IR、完全なJSON／C、および最終集約を除いたVerilogとする。その後、Warp／Light、高rank、学習可能接続、3D、二値化、Residual、compile、Alkaid、参照モデルを追加する。

初期の実装順序は対応範囲の縮小を意味しない。各段階の完了条件を満たすまで、未実装機能を実装済みとして公開しない。

### 5.3 対象外

- データセットの取得、分割、augmentationおよびDataLoader生成。
- loss、optimizer、schedulerおよびepochループ。
- logicNNアプリケーションの設定JSON、Rich表示、チェックポイントおよびログ管理。
- 学習済み参照モデルの重み配布。
- FPGA配置配線、論理合成、消費電力解析およびタイミング収束の保証。
- Cコンパイラ、VerilatorおよびAlkaid本体の自動インストール。
- torchlogixのprivate API、内部ファイル名または不具合まで含めた完全互換。
- 分散学習制御。Tensor計算がPyTorchの対応デバイスで動作することとは区別する。

### 5.4 承認済みの実装保留

2026-09-12、補助正則化の方式名 `warp` だけを当面の実装範囲から外す方針が承認された。目的と計算式の根拠が未確定のためであり、機能が無意味だと判断したものではない。目的・数式・検証方法が定まった段階で別途再検討する。

Warpパラメータ化そのもの、temperature／sampling、残差初期化および回路出力は正式対象として維持する。L2／abs_sumの正則化と重み再スケールも本保留の対象ではない。保留中の正則化を指定した場合は、成功を装わず明示エラーにする。

## 6. 機能要件

### 6.1 パッケージと公開API

#### CORE-FR-PKG-001 独立境界

`utils/logicNN_core/` は独自の `pyproject.toml`、`README.md`、`LICENSE`、`src/logicnn_core/` を持ち、logicNNアプリケーションの親ディレクトリをimportしないこと。

#### CORE-FR-PKG-002 利用方法

logicNN開発中はルートプロジェクトからローカルパス依存として利用でき、将来はディレクトリ単体でwheelとsource distributionを作成・インストールできること。

#### CORE-FR-PKG-003 公開入口

主な公開入口を `logicnn_core.Circuit`、`logicnn_core.layers`、`logicnn_core.modules` および `logicnn_core.set_export_mode()` に限定すること。内部最適化関数を不用意にトップレベルへ公開しないこと。

#### CORE-FR-PKG-004 版情報

ライブラリはGitメタデータに依存しない静的なパッケージ版を持つこと。開発開始時は `0.1.0` とし、独立配布後はSemantic Versioningに従うこと。

### 6.2 論理関数とLUT

#### CORE-FR-FNC-001 二入力論理関数

二つの0／1入力に対する全16種類の論理関数を、IDと真理値表の対応が一意になる形式で計算できること。

#### CORE-FR-FNC-002 連続値入力

学習時は `[0, 1]` の連続値Tensorに対して微分可能な論理基底を計算し、評価・export時は離散値に対して決定的に計算すること。

#### CORE-FR-FNC-003 高rank基底

Walsh-Hadamard基底、Light基底、組合せindex、真理値表ID変換を責務別関数として提供し、サポートrankをパラメータ化方式ごとに検証すること。

#### CORE-FR-FNC-004 補助計算

temperature付きsigmoid／softmax、Gumbel sampling、勾配係数、正則化損失および重み再スケールを提供すること。

補助正則化 `warp` は5.4節の承認済み保留に従い実装しない。指定時は未対応であることを示す `ValueError` とし、ゼロを返す仮実装や別方式への暗黙置換を行わない。LUTパラメータ化の登録名 `warp` とは別の指定として扱う。

### 6.3 LUTパラメータ化

#### CORE-FR-PAR-001 共通契約

すべてのパラメータ化方式は、重み初期化、forward、LUT取得、LUTとID取得、temperature更新の共通契約を満たすこと。

#### CORE-FR-PAR-002 Raw

Raw方式はrank 2を正式対応し、16論理関数のlogitから学習時の選択と評価時のargmax選択を行うこと。

#### CORE-FR-PAR-003 Warp

Warp方式はrank 1、2、4、6を正式対応し、Walsh-Hadamard係数から連続学習と離散真理値表取得を行うこと。

#### CORE-FR-PAR-004 Light

Light方式はrank 2、4、6を正式対応し、軽量な基底係数から連続学習と離散真理値表取得を行うこと。

#### CORE-FR-PAR-005 選択と検証

登録名 `raw`、`warp`、`light` から方式を生成し、未知名、非対応rank、非正temperatureおよび不正なsampling／初期化方式を生成時に拒否すること。

### 6.4 接続

#### CORE-FR-CON-001 共通契約

接続方式は入力から各LUTへ渡す特徴を選択し、入力次元、出力次元、rank、indexの範囲およびshapeを検証すること。

#### CORE-FR-CON-002 固定Dense

固定Dense接続はランダム接続と重複抑制接続を生成し、state dictへindexを保存して再現できること。

#### CORE-FR-CON-003 学習可能Dense

学習可能Dense接続はtemperatureとGumbel選択を使用して接続候補を学習し、評価時には決定的な接続へ離散化できること。

#### CORE-FR-CON-004 固定畳み込み

固定畳み込み接続は2D／3Dの受容野、stride、padding、channel、kernelおよびtree levelに対応するindexを生成すること。

#### CORE-FR-CON-005 構築

構造種別と接続名の組合せから方式を生成し、未対応の組合せを明確なエラーにすること。

### 6.5 論理層

#### CORE-FR-LAY-001 共通層

論理層は `torch.nn.Module` として、train／eval、state dict、デバイス移動、LUT取得、正則化、重み再スケールおよびexport modeを一貫して扱えること。

#### CORE-FR-LAY-002 LogicDense

`LogicDense` は末尾次元を入力特徴として任意の先行次元を維持し、指定した出力特徴数へ変換すること。パラメータ化方式と接続方式を独立に選択できること。

#### CORE-FR-LAY-003 LogicConv

`LogicConv2d` と `LogicConv3d` は論理木で受容野を集約し、入力空間、channel、kernel、tree depth、受容野、stride、paddingをMNISTに依存せず指定できること。

#### CORE-FR-LAY-004 Pooling

OR poolingは2D／3D Tensorに対してkernel、stride、paddingを適用し、train／eval／exportで同じ論理的ORを表すこと。

#### CORE-FR-LAY-005 Binarization

固定閾値、無処理、soft閾値および学習可能閾値を提供し、特徴ごとまたは共有の閾値を扱えること。logicNNアプリケーションがデータ側で二値化する設計を妨げず、他の利用者が層として選択できること。

CORE-DES-009の承認に従い、学習可能閾値はtorchlogixのモード別計算を維持すること。trainでは後続差分のsoftplusと累積和、eval／exportでは生差分の累積和を用いる。評価時の昇順を追加保証したり、閾値を暗黙に並べ替えたりしないこと。

#### CORE-FR-LAY-006 GroupSum

`GroupSum` は論理出力を指定グループ数へ集約し、temperature scaleとbiasを適用できること。入力数がグループ数で割り切れない場合の規則を明示すること。

#### CORE-FR-LAY-007 Residual

`ResidualLogicBlock` は論理畳み込みとskip pathを組み合わせ、入出力shapeの制約を生成時に検証すること。

### 6.6 動作モード

#### CORE-FR-MODE-001 学習

通常modeの `module.train()` は連続緩和された計算を選び、学習対象パラメータへ有限な勾配を伝播できること。export mode有効中の `train(True)` は `RuntimeError` とし、暗黙にexport modeを解除しないこと。学習へ戻るには、先に `set_export_mode(module, False)` を呼び、その後 `module.train()` を呼ぶこと（2026-09-12承認済み）。

#### CORE-FR-MODE-002 評価

`module.eval()` は学習済み重みから決定的なLUTと接続を選択し、同じ入力とstateで同じ出力を返すこと。

Warpの0／1入力の評価と回路化は、CORE-DES-007で承認した共通のbool真理値表を基準とすること。表は高精度・固定順で生成し、重みや対応するモデル状態が変わった場合は再生成する。学習の式・勾配は維持し、連続入力を暗黙に二値化しないこと。生成済みの同じ表と接続による一致を要求し、異なる環境で再生成した表や数学的無限精度との一致を保証する規定ではない。

#### CORE-FR-MODE-003 export

`set_export_mode(module, True)` は再帰的に対応層を評価状態とexport可能な演算へ切り替えること。`False` で一時bufferを除去し、通常評価へ戻せること。

#### CORE-FR-MODE-004 状態保護

回路変換は呼出側のモデルを変更せず、複製したCPU・評価・export状態から行うこと。

### 6.7 回路IR

#### CORE-FR-CIR-001 構築

対応するPyTorchモデルまたはFX graphから、入力、定数、16種類の2入力ゲート、出力およびsum reductionを持つ回路IRを構築できること。

#### CORE-FR-CIR-002 検証

回路生成時とJSON読込時に、ID一意性、参照整合性、循環がないこと、入力数、出力数、gate opおよびreduction範囲を検証すること。

#### CORE-FR-CIR-003 簡略化

簡略化は意味を変えず、NOT入力融合、不要ゲート除去、定数畳み込み、wire bypass、重複除去およびsum reduction簡約を固定順で反復すること。集約後の出力だけでなく、CORE-FR-CIR-009で定める集約前の論理出力も保つこと。

#### CORE-FR-CIR-004 実行

回路IRをPythonで実行でき、任意でCへcompileした実装を呼び出せること。両者は対応入力に対して同じ出力を返すこと。

#### CORE-FR-CIR-005 永続化

版付きの辞書／JSON形式へ保存し、情報を失わず復元できること。未知の将来版は暗黙に受理しないこと。

#### CORE-FR-CIR-006 C

対応回路から自己完結したCコードを生成し、inlineとbit packingの既存オプションを維持すること。

Cも二値化済みの0／1を入力するモデルを対象とする（CORE-DES-010承認済み）。uint8の0〜255の画素や浮動小数点値から二値化する前処理関数・wrapperは今回生成しない。通常はbool配列、packed経路は複数sampleの0／1をwordへ詰めた入力とし、画素のuint8値とは区別する。GroupSumを含む集約後の出力をCに残す既決定は維持する。

#### CORE-FR-CIR-007 Verilog

対応回路から組合せ回路のVerilogを生成し、入力から最終論理層までの定数、NOTおよびgateを一意な信号名で表現すること。回路IRの出力がsum reductionの場合は加算回路を生成せず、簡略化前に保存した集約前の論理出力をVerilogの生出力端子へ接続すること。Verilog backendはsum、tauによる除算、bias加算、スコアscaleおよびクラス順位決定を担当しないこと。生出力のnode ID列を取得し、同じ列をPythonで評価できるAPIを提供すること。

Verilogへ渡す入力は、外側で二値化済みの0／1信号とする。uint8の0〜255の画素や浮動小数点値から二値化する前処理は、生成するVerilogの対象に含めない（CORE-DES-010のVerilog部分を承認）。二値化層の学習機能を削除する決定ではない。

#### CORE-FR-CIR-008 失敗

未対応のPyTorch演算、無効なshape、過大な展開またはコード生成不能を、処理段階と対象演算が分かるエラーとして報告すること。

#### CORE-FR-CIR-009 論理出力の保存

回路構築時に、最終集約前の出力対応表を簡略化より先に保存すること。簡略化、JSON保存・再読込およびVerilog生成を通して、各出力の数、順序、0／1値、定数の位置および同じ信号の重複出力を保つこと。集約のない出力はその論理値を保存すること。内部node IDの変更やgateの共有は許すが、出力位置を削除・並べ替えしないこと。簡略化後のsum reduction入力から対応表を作り直さないこと。

この方針は2026-09-12に承認済みである。対応表の型名・API名は詳細設計の草案として管理する。

CORE-DES-011の承認により、外部FXを直接受け取る場合は、簡略化前の出力対応情報も必須とする。欠落時は加工後の集約式から推測せず、必要な情報を示して拒否する。通常の `from_model()` ではライブラリが対応情報を自動記録し、利用者へ手書きの対応表を要求しない。

### 6.8 任意連携と参照モデル

#### CORE-FR-OPT-001 Alkaid

Alkaid連携はoptional dependencyと専用adapterへ隔離し、未導入でも基本ライブラリをimport・学習・回路出力できること。

#### CORE-FR-OPT-002 参照モデル

torchlogixのMNIST、Fashion-MNIST、CIFAR-10、JSC向けモデル知識を `examples/reference_models/` に整理して残すこと。これらはデータセットを自動取得せず、トップレベル公開APIにも含めないこと。

## 7. 非機能要件

### CORE-NFR-001 保守性

functional、parametrization、connection、layer、circuit、exporterおよびintegrationを分割し、一ファイルに異なる責務を集中させないこと。

2026-09-13の変更承認により、独自のチェック機構は最低限にする。生成時の設定、外部ファイル読込、回路生成・C実行の境界で必要な検証を行い、通常のforwardでは生成済み内部状態を信頼する。PyTorchが検出する型・layout・deviceのエラーを各階層で再検査しない。黙ったbroadcast、非整除、虚部の消失、論理bitの意味変更などを防ぐ条件は残す。検証ON/OFF設定や別の検証フレームワークは追加しない。

### CORE-NFR-002 コード規則

関数呼出しは行長内なら一行にし、関連する代入は可読性を損なわない範囲で `=` を揃え、すべての関数・メソッド定義直下に簡潔なdocstringを書くこと。公開APIは型注釈を持つこと。

### CORE-NFR-003 汎用性

パッケージコードにMNISTの件数、28×28、10クラス、特定batch size、特定lossまたは特定optimizerを埋め込まないこと。

### CORE-NFR-004 配布可能性

ルート外からのimport、wheel install、editable installおよびsource distribution buildが成功し、ビルド時にGit履歴を必要としないこと。

### CORE-NFR-005 依存最小化

基本依存はPyTorchとNumPyに限定することを目標とし、Rich、TorchVision、Alkaidおよび開発ツールを基本import経路へ含めないこと。

### CORE-NFR-006 再現性

固定接続index、学習可能接続のstate、LUT重みおよびexport bufferはstate dictで追跡可能にすること。乱数を使うAPIはPyTorchのseedまたは明示的generatorに従うこと。

### CORE-NFR-007 CPU／CUDA

学習・評価層はPyTorchが対応するCPUと単一CUDAデバイスで動作し、内部Tensorを固定デバイスへ無断で作成しないこと。回路IR構築とコード生成はCPUで行うこと。

CORE-DES-007のWarp共通表生成は、数値判定規則を統一するためCPU float64で行う。生成したbool表は元deviceへ戻し、評価入力のCPU転送を要求しない。学習の計算deviceは変更しない。

### CORE-NFR-008 性能

正しさを優先しつつ、基底全体の不要なmaterializeを避ける経路を維持すること。性能変更は固定benchmarkで測定し、著しい退行を記録なしに受け入れないこと。

### CORE-NFR-009 文書

概念、公開API、対応rank、Tensor shape、動作モード、回路JSONおよびエクスポート制約をDocusaurusに記録すること。

### CORE-NFR-010 帰属

torchlogixまたはdifflogic由来のコード・アルゴリズムを含むファイルは、MITライセンス、著作権表示および由来を保持すること。新規実装部分との境界を `THIRD_PARTY_NOTICES.md` に記録すること。

## 8. 機能維持表

| torchlogixの要素 | logicNN_coreでの扱い |
|---|---|
| `functional.py` | 機能別ファイルへ分割して維持。ただし補助正則化 `warp` は下記の承認済み保留 |
| 補助正則化 `warp` | 当面は実装保留。指定時は明示エラーとし、再検討事項として追跡 |
| Raw／Warp／Light parametrization | すべて維持 |
| fixed／learnable Dense connections | 維持 |
| fixed Conv connections | 2D／3Dとも維持 |
| Dense／Conv2d／Conv3d | 維持 |
| OR Pooling 2D／3D | 維持 |
| 4種類のBinarization | 維持 |
| GroupSum | 維持 |
| ResidualLogicBlock | 維持 |
| export mode | 維持し、状態遷移を明文化 |
| Circuit構築・簡略化・実行・compile | 責務別に分割して維持 |
| JSON／C | GroupSumを含む完全な回路として維持 |
| Verilog | 加算回路を除外し、最終論理層の生出力だけを生成 |
| Alkaid plugin | optional integrationとして維持 |
| データセット別モデル群 | 参照モデルへ移して維持 |
| experiments／notebook | 学習アプリには持ち込まず、再現可能なexamples／文書へ整理 |
| torchlogixのprivate互換alias | 原則維持しない。必要性を個別審査 |

## 9. 受入条件

### CORE-AC-001 独立インストール

`utils/logicNN_core/` 単体からwheelを作成し、空の仮想環境へインストールして公開APIをimportできること。

### CORE-AC-002 学習と離散評価

複数のrank・パラメータ化・接続方式でforward、backward、state保存・復元および決定的なevalが成功すること。

### CORE-AC-003 汎用shape

MNIST以外の小型Dense、2D画像、3D volumeの固定例でshape契約を満たすこと。

### CORE-AC-004 回路等価性

対応する小型モデルについて、評価モデル、回路IR、JSON再読込および生成Cが全入力組合せで同じ最終結果を返すこと。VerilogはGroupSum直前の論理値と全件一致し、加算・tau・biasを含まないこと。

### CORE-AC-005 logicNN統合

MNISTLGNが `logicnn_core` の公開APIだけで構築・学習・評価・回路出力でき、現行logicNNとの精度比較基準を満たすこと。

### CORE-AC-006 維持機能

機能維持表の各項目が、実装済み、参照例へ移行済み、または明示的に未実装として追跡されており、理由なく消失した項目がないこと。

## 10. 制約とリスク

- torchlogixの数値的不具合まで維持すると品質が下がるため、出力差が仕様修正か退行かをテスト単位で判断する必要がある。
- Raw方式はrank増加に対して関数数が二重指数的に増えるため、rank 2限定を維持する。
- 高rank LUTはID表現や回路展開が急増するため、学習対応と回路出力対応を分けて明示する。
- FX追跡はPyTorch版の影響を受けるため、対応演算と検証済みPyTorch版を記録する。
- C／Verilog等価性には外部コンパイラが必要だが、通常利用時には実行しない。
- 全機能の再設計は複数実装単位に及ぶため、縦方向の最小機能を完成させながら機能維持表を消化する。

## 11. 要確認事項

1. 承認済み: パッケージ表示名 `logicNN-core`、import名 `logicnn_core`、モデル定義内の `LUTConfig`／`ConnectionConfig`、ゲート入力本数 `num_inputs`。
2. 承認済み: export mode有効中の `train(True)` は明示エラーにし、明示解除してから学習へ戻す。
3. 2026-09-12に仕様に従った段階的な実装・検証・レビューの開始指示を受けた。最新の停止条件は仕様変更または不明点がある場合とする。テスト不備や仕様が明確な実装不具合は修正・再検証して継続する。未確定の契約を推測で補完しない。

参照モデルは配布wheelから除外してsource repositoryの `examples/` に残す。Alkaidはoptional dependencyとして最後に実装するが、参照モデルとともにcore全体の完成条件へ含める。旧torchlogix／logicnnチェックポイントの後方互換toolは、logicNN全体の既決定に従って対象外とする。
