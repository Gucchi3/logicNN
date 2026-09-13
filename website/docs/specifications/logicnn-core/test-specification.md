---
sidebar_position: 4
title: テスト仕様
---

# logicNN_core テスト仕様書

## 1. 文書情報

- 文書状態: 草案
- 作成日: 2026-09-12
- 対象: [logicNN_core要件](./requirements.md)と[詳細設計](./detailed-design.md)
- 開発時配置: `logicNN/tests/core/`

2026-09-12に本書に従った段階的な実装・検証・レビューの開始指示を受けた。承認済みの命名、論理出力保護およびexport中の学習切替規則を反映する。本書はテスト計画であり、合格結果は実際の実行後に別途記録する。テストコードの不備や仕様が明確な実装不具合は修正・再検証して継続し、仕様定義の変更や不明点がある場合に中断して報告する。

## 2. 目的

1. 数学的な論理結果と勾配が正しいことを確認する。
2. torchlogixから維持する機能が再設計時に欠落・退行しないことを確認する。
3. MNIST以外のshape、rank、接続方式でも動くことを確認する。
4. train／eval／exportの状態遷移が決定的で安全であることを確認する。
5. 回路IR、JSONおよびCが変換元モデルの最終出力と等価であり、VerilogがGroupSum直前の生の論理出力と等価であることを確認する。
6. logicNN本体に依存せず単独packageとしてbuild・installできることを確認する。

## 3. テストの境界

2026-09-13のチェック最小化承認を適用する。通常forwardの不正な型・device等について、独自の例外型・文言を一律には要求しない。内部bufferを手動破壊した状態を毎forwardで全検査するテストは、constructor／state_dict読込／回路抽出の境界テストへ整理する。正常な数値・勾配・shape・モード・生bitと、黙ったbroadcastを防ぐ検証は維持する。検証済み状態を繰り返し全走査しないことも確認する。

- pytestテストは開発中だけ `tests/core/` に置き、logicNN通常実行から呼び出さない。
- C compiler、VerilatorおよびAlkaidを必要とするテストはmarkerで分離する。
- 通常単体テストはネットワークへ接続しない。
- 旧torchlogixは比較oracleとして読取り専用で使用し、新coreからimportしない。
- 受入完了後、運用版からテストコードと開発依存を削除するが、本書と検証結果は残す。

## 4. marker

| marker | 用途 |
|---|---|
| `core_unit` | 外部tool不要の単体テスト |
| `core_parity` | torchlogixとの比較 |
| `core_cuda` | CUDAがある場合だけ実行 |
| `core_external` | C compiler／Verilatorが必要 |
| `core_optional` | Alkaidなどoptional dependencyが必要 |
| `core_package` | wheel／sdist buildと隔離install |
| `core_benchmark` | 性能記録。通常合否から分離 |

## 5. 基準データ

| fixture | 内容 |
|---|---|
| `binary_pairs` | 4通りの二入力bool値 |
| `all_lut_ids` | 0〜15のLUT ID |
| `tiny_dense_inputs` | 全組合せを列挙できる4特徴入力 |
| `tiny_image_inputs` | MNISTに依存しない1×5×7等の二値画像 |
| `tiny_volume_inputs` | 1×4×5×6等の二値volume |
| `fixed_lut_weights` | 各方式で既知LUTを表す重み |
| `fixed_connections` | 手計算可能な接続index |
| `tiny_logic_model` | Dense／Conv／Pooling／GroupSumを個別に組み合わせたmodel |
| `invalid_circuit_dicts` | schema違反を一つずつ持つJSON object |

乱数テストはseedを明示し、期待値を乱数系列そのものへ過度に固定せず、範囲、一意性、再現性およびshapeを検証する。

## 6. 品質テスト

| ID | 検証 | 合格条件 |
|---|---|---|
| CORE-QA-001 | Ruff | error 0件 |
| CORE-QA-002 | Python compile | 対応Pythonで全sourceがcompile可能 |
| CORE-QA-003 | docstring AST検査 | 全関数・メソッド直下に簡潔なdocstringがある |
| CORE-QA-004 | import依存検査 | coreからlogicNN本体、TorchVision、Richをimportしない |
| CORE-QA-005 | public API検査 | `__all__` と文書の公開名が一致し、`logicnn_core.layer_settings` から設定型をimportでき、`LUTConfig.num_inputs` を使用できる |
| CORE-QA-006 | 元コード帰属検査 | 派生fileにlicense／notice要件を満たす記録がある |
| CORE-QA-007 | source file責務 | circuitやfunctionalを再び巨大単一fileへ集約していない |

## 7. Functional単体テスト

### 7.0 設定型・例外（実装単位C1）

| ID | 対象 | 合格条件 |
|---|---|---|
| CORE-SET-001 | 既定値・不変性 | LUTConfig／ConnectionConfigの既定値が一致し、frozen／slotsとasdictに対応 |
| CORE-SET-002 | LUT方式・入力本数 | Raw2、Warp1／2／4／6、Light2／4／6を受理し、未対応の組合せと不正型を拒否 |
| CORE-SET-003 | sampling・初期化 | 4sampling、方式別初期化名を受理し、residual_catalogはWarpだけが使用できる |
| CORE-SET-004 | 数値・真偽値 | temperatureは正の有限値、確率は0と1の間、bool項目はboolだけを受理 |
| CORE-SET-005 | Raw residual確率 | 7/15以下と1以上を拒否し、下限直上・0.5・既定0.951を受理。他方式とrandom初期化へ追加制約を波及しない |
| CORE-SET-006 | 接続固有設定 | 不正な型、非正の候補数・channel群、方式に適用不能な非既定値、learnable uniqueの候補数未指定を拒否 |
| CORE-SET-007 | 例外 | LogicNNCoreErrorはRuntimeErrorであり、message/detail/hintと標準例外の文字列表現を保持 |

特徴数・channel数など実モデルのshapeに依存する検証はC5の接続構築時に行い、C1の設定型だけで架空のshapeを仮定しない。

### 7.1 数学基盤（実装単位C2・C3）

| ID | 対象 | 条件 | 合格条件 |
|---|---|---|---|
| CORE-FNC-001 | 16論理関数 | 4入力組×16 ID | 真理値表と全件一致 |
| CORE-FNC-002 | vectorized演算 | broadcastするTensor | scalar基準と一致しshape保持 |
| CORE-FNC-003 | export LUT | bool Tensorと全ID | 通常離散演算と完全一致 |
| CORE-FNC-004 | softmax | 複数temperature | 有限、総和1、gradient有限 |
| CORE-FNC-005 | hard sampling | 固定logit | forward one-hot、backward有限 |
| CORE-FNC-006 | Gumbel | seed固定 | 同じseedで同じ結果 |
| CORE-FNC-007 | gradient scale | factor 0.5／1／2 | forward同値、gradientだけ指定倍率 |
| CORE-FNC-008 | FWHT | rank 1／2／4／6 | 明示Hadamard行列との積に一致 |
| CORE-FNC-009 | Walsh basis | 対応rank | materialize経路と省memory経路が許容誤差内 |
| CORE-FNC-010 | Light basis | rank 2／4／6 | materialize経路と省memory経路が許容誤差内 |
| CORE-FNC-011 | 組合せunrank | 小さいn、k全rank | itertools基準と一致 |
| CORE-FNC-012 | unique sampling | seed固定 | 範囲内、tuple重複なし、再現可能 |
| CORE-FNC-013 | truth table ID | 対応rank | ID→表→IDの往復一致 |
| CORE-FNC-014 | 不正値 | ID範囲外、temperature 0等 | 明示的なValueError |

C2では001〜007・014に加え、真理値表の独立した多重線形補間、数値微分、hard／soft勾配の同値性、低precision、高温、巨大有限整数、空Tensor、混在dtype、FX graphの異なる入力・IDでの再実行を検査する。型違いはTypeError、値・shape違いはValueErrorとして区別する。CUDA試験は利用できる環境だけで行い、未実施を合格扱いにしない。

C3の正則化は、[詳細設計17.3節](./detailed-design.md#173-c3開始前の確認事項) の確認を経て、warpの保留に加え、層内全ゲート平均・scalar返却も承認された。C3ではfunctionalの数式と数値保護を、C6／C7では層内・Conv全levelの全ゲート平均を検証する。

### 7.2 warp正則化の保留契約（計画のみ）

[詳細設計17.4節](./detailed-design.md#174-warp正則化の実装保留承認済み) の承認済み方針を、各公開入口の実装時に検証する。ここにあるのはテスト仕様であり、テストコード実装・実行の完了記録ではない。

| ID | 対象・実装単位 | 合格条件 |
|---|---|---|
| CORE-REG-001 | functional正則化入口／C3 | 正則化方式 `warp` の指定でValueError。エラー文が対象方式と実装保留を示し、損失値を返さない |
| CORE-REG-002 | Dense／Conv公開入口／C6・C7 | 各層の `regularization_loss(kind="warp")` も同じ未対応エラーを呼出側へ伝える。ゼロ返却、警告だけの続行、別方式への置換をしない |
| CORE-REG-003 | エラー時の状態保護／C3・C6・C7 | 拒否前後で重み・既存の勾配が不変。層の試験ではtrain／eval状態も不変 |
| CORE-REG-004 | Warp本体との分離／C1・C4以降 | `LUTConfig(kind="warp")` は引き続き受理し、既存のCORE-PAR-006とCORE-GEN-002などのWarp本体の計画を削除しない |

保留中の旧exp式を計算するテストや、その式との数値一致は現行実装の完了条件に含めない。Warpパラメータ化、他の対応済み正則化および再スケールの検証は、保留を理由に省略しない。

### 7.3 正則化と再スケールの数値契約

| ID | 対象 | 合格条件 |
|---|---|---|
| CORE-REG-005 | L2／abs_sum | 手計算したゲートごとの旧式を全ゲートで平均したscalar Tensorと一致。複数先行軸でも同じ規則 |
| CORE-REG-006 | dtype／device／勾配 | 出力は入力と同じdtype／device。数値微分・解析勾配と一致し、Noneはscalar zeroとゼロ勾配 |
| CORE-REG-007 | 再スケール | Noneは無変更、clipは[-1,1]、L2はノルム、abs_sumは和の絶対値で末尾軸を正規化。既存の勾配を保持 |
| CORE-REG-008 | ゼロ分母と原子性 | ゼロ行や正負相殺を含む全分母を検証し、ValueError。正常行も含めて元Tensorを一切変更しない |
| CORE-REG-009 | 不正入力・数値範囲 | 不正型・方式名・空の重み・非有限値を明示拒否。再スケールの結果が表現不能な場合も書換え前に止める |
| CORE-REG-010 | 層内集約／C6・C7 | Dense全ゲート、およびゲート数の異なるConv全levelを通じた平均が手計算と一致。levelの単純平均に置換しない |

## 8. Parametrizationテスト

| ID | 対象 | 合格条件 |
|---|---|---|
| CORE-PAR-001 | builder登録名 | raw／warp／lightを生成し未知名を拒否 |
| CORE-PAR-002 | 方式別rank | 対応表どおり受理・拒否 |
| CORE-PAR-003 | 初期化 | shape、dtype、deviceが指定どおり |
| CORE-PAR-004 | Raw train | 全samplingで出力・gradientが有限 |
| CORE-PAR-005 | Raw eval | argmax LUTの真理値と一致 |
| CORE-PAR-006 | Warp train／eval | rank 1／2／4／6で基準計算と一致 |
| CORE-PAR-007 | Light train／eval | rank 2／4／6で基準計算と一致 |
| CORE-PAR-008 | LUT抽出 | truth table shapeと値が正しい |
| CORE-PAR-009 | ID制約 | rank 4以下はID一致、高rankはNone |
| CORE-PAR-010 | temperature更新 | 正値のみ受理しforwardへ反映 |
| CORE-PAR-011 | materialize parity | true／false経路が許容誤差内 |
| CORE-PAR-012 | torchlogix parity | 対応する固定weight・inputで同値または承認済み差分 |

### 8.1 C4レビューで追加した数値境界の検証課題

[検証記録の反例](./verification.md#二値評価と抽出lutの不一致) により、通常のランダム入力や連続値の許容誤差比較だけでは、Warpの0／1出力の不一致を検出しきれないことを確認した。

1. float32の正負相殺する係数と、bfloat16・rank 6の固定seedによる反例を回帰対象とする。
2. 二値入力の評価・materializeの有無・抽出真理値表の**bit単位の完全一致**を検証する。将来の回路変換でも同じ基準を引き継ぐ。連続学習値に用いる許容誤差をbitの不一致へ適用しない。
3. 承認済みの [CORE-DES-007](./detailed-design.md#176-warpの二値評価とlut抽出の判定契約承認済み) に従い、CPU float64・固定順FWHT・厳密な負値判定を独立基準として回帰検証する。結果は実行後に検証記録へ記載し、計画だけで合格にしない。
4. 学習の式・勾配を維持し、連続入力のevalを黙って二値化しないことも回帰確認する。

| ID | 対象 | 合格条件 |
|---|---|---|
| CORE-PAR-013 | 判定境界 | float32の相殺例とbfloat16 rank6の全入力で、evalと抽出表がbit単位で一致 |
| CORE-PAR-014 | 生成規則 | 独立したCPU float64の固定順butterfly基準と一致。厳密0・負の0は0、微小な負値へepsilonを適用しない |
| CORE-PAR-015 | 状態と更新 | 重みのin-place変更・置換を次の呼出しへ反映し、過去の返却表、weight、既存grad、RNG、Module状態を壊さない |
| CORE-PAR-016 | 数値拒否 | NaN／Infのweightと変換overflowをValueErrorとし、不正値をboolの0に変えて出力しない |
| CORE-PAR-017 | 二値・連続混在 | 元の入力dtypeでゲート単位に二値性を判定。混合batchでも二値組だけ表を使い、連続組・学習勾配は旧式を維持 |
| CORE-PAR-018 | gate対応 | Dense／Conv軸・並替・broadcast・非連続・空入力で表の行と入力組の対応を維持。二値evalでgate軸を還元する縮約は明示拒否 |
| CORE-PAR-019 | device | CPU生成表を元deviceに返す。CUDAがあれば同一重みからの表・二値出力の一致を検証し、未搭載ならskipを明記 |

## 9. Connectionテスト

学習時の計算方式は、[CORE-DES-008](./detailed-design.md#177-学習可能接続の旧計算との相違承認済み) によりtorchlogixのhard選択・独自勾配を維持することが承認された。CORE-CON-004の有限gradientだけで等価性を保証せず、独立したforward／入力勾配／接続logit勾配の基準と旧コードで照合する。

| ID | 対象 | 合格条件 |
|---|---|---|
| CORE-CON-001 | fixed random Dense | index shape・範囲・state保存が正しい |
| CORE-CON-002 | fixed unique Dense | LUT内重複なし、入力不足を拒否 |
| CORE-CON-003 | Dense forward | 手計算indexと選択結果が一致 |
| CORE-CON-004 | learnable train | connection logitへ有限gradient |
| CORE-CON-005 | learnable eval | argmax接続で決定的 |
| CORE-CON-006 | learnable state | 保存・復元後に同じ離散接続 |
| CORE-CON-007 | Conv2d index | 非正方入力、tuple kernel／stride／paddingで基準一致 |
| CORE-CON-008 | Conv3d index | 非立方入力で基準一致 |
| CORE-CON-009 | Conv tree level | 各levelのrank・node・kernel位置shapeが正しい |
| CORE-CON-010 | invalid shape | 範囲外index、空出力、次元違いを早期拒否 |
| CORE-CON-011 | candidate count | learnable randomのNoneは全特徴候補、正の整数は指定した候補数。不正型・非正値・固定接続での指定を拒否 |
| CORE-CON-012 | learnable unique | 1ゲート内の端子間で候補集合が交わらず、eval・exportでも入力が重複しない。None・候補数過多を拒否 |
| CORE-CON-013 | Conv channel group | None／1／入力channel数でchannel制限を満たし、不正値・Denseやlearnableでの指定を拒否 |
| CORE-CON-014 | unique候補の復元 | state保存・復元後も候補集合、重複禁止、離散出力が維持される |
| CORE-CON-015 | 旧独自勾配 | 6.2節のweight勾配・入力勾配の独立式と旧torchlogixで照合。通常の数値微分やhard-softmax勾配へ置換しない |
| CORE-CON-016 | Gumbelと温度 | 同一seedの学習noiseを再現し、forwardとbackwardで共有。温度更新は入力勾配配分に反映し、評価とget_indicesは乱数を使わない |
| CORE-CON-017 | dtype・shape | Denseの任意先行shape・空batch・非連続を受理し、元の浮動小数点入力をlogitdtypeへ丸めず保持。学習可能trainingの整数入力は明示拒否 |
| CORE-CON-018 | 保存index検証 | 不正dtype・shape・範囲・unique違反を復元前に拒否。直接変更された不正indexも黙って使わない |
| CORE-CON-019 | Conv座標とpadding | 局所座標＋padded開始位置＝絶対index。paddingはlevel0で1回だけ適用し、全levelでrank軸はindex1 |
| CORE-CON-020 | Conv状態 | bufferをdevice移動・保存復元し、座標の合成関係と後段nodeの重複・欠落を検証。3Dの末尾座標数は4 |
| CORE-CON-021 | 接続builder | 型付きのDense／Conv入口から方式・引数を維持して生成し、不正configや未対応方式を拒否 |

## 10. Layerテスト

| ID | 対象 | 合格条件 |
|---|---|---|
| CORE-LAY-001 | LogicDense shape | 任意先行次元を維持して末尾だけ変換 |
| CORE-LAY-002 | LogicDense gradient | inputとweightへ有限gradient |
| CORE-LAY-003 | Dense eval | 同じinput／stateで完全一致 |
| CORE-LAY-004 | Conv2d shape | 5×7等の非MNIST入力で公式どおり |
| CORE-LAY-005 | Conv3d shape | 4×5×6等のvolumeで公式どおり |
| CORE-LAY-006 | Conv gradient | 全tree levelのweightへ有限gradient |
| CORE-LAY-007 | OR Pooling | max経路とbool OR経路が二値入力で一致 |
| CORE-LAY-008 | FixedBinarization | global／feature／channel閾値で期待値一致 |
| CORE-LAY-009 | SoftBinarization | trainは連続、evalは離散、gradient有限 |
| CORE-LAY-010 | LearnableBinarization | CORE-DES-009に従いtrainのsoftplus差分・evalのraw cumsumをそれぞれ旧式と照合。評価の逆順も保持し、gradient clipとeval決定性を確認 |
| CORE-LAY-011 | DummyBinarization | shapeを変えずfloatへ変換 |
| CORE-LAY-012 | GroupSum | groups、tau、biasを手計算と比較 |
| CORE-LAY-013 | GroupSum invalid | 非割切り、非正tau等を拒否 |
| CORE-LAY-014 | Residual | shape一致、gradient、mode伝播 |
| CORE-LAY-015 | state dict | 全parameter／bufferを復元してeval同値 |
| CORE-LAY-016 | device | CPU必須、CUDA利用時はdevice混在なし |
| CORE-LAY-017 | torchlogix parity | 同じstateへ変換可能なケースで出力比較 |
| CORE-LAY-018 | Dense input scale | 入力勾配だけに倍率が適用され、LUT重み・接続logit勾配は不変。0と負の有限倍率も検証 |
| CORE-LAY-019 | Dense最新状態 | 通常evalは重みのin-place／Parameter差替／外部共有更新を次回呼出しへ反映し、微小連続入力を事前に丸めない |
| CORE-LAY-020 | Dense export snapshot | rank 2のbool演算が独立真理値表と一致。再enableで最新の表と接続へ更新、未対応rankを状態変更前に拒否 |
| CORE-LAY-021 | Dense export state | export派生bufferは通常loadでは無視、export loadではchild復元後に再生成。未知keyはstrictで拒否 |
| CORE-LAY-022 | Conv初期化と軸 | residual_catalogのnode単位丸め、全levelのrank軸、paddingの1回適用、先頭levelと後段の独立座標oracleを確認 |
| CORE-LAY-023 | Conv更新の安全性 | 全ゲート平均と全levelの再スケール。途中levelの不正値や共有storageでは元Parameter・値・gradを変更しない |
| CORE-LAY-024 | Conv export state | 全levelのbool snapshotと独立LUTが一致し、state復元後に接続と表を再生成。FXを別の入力でも実行して比較 |
| CORE-LAY-025 | Pooling・GroupSum境界 | 2D／3Dの窓・padding・stride、空batch、連続入力、dtype、勾配、export切替、集約のbias／tau省略を確認 |
| CORE-LAY-026 | 二値化の旧式 | 初期差分、負の生差分、単一・複数閾値、soft／hard／Gumbelを旧コードまたは独立式で検証。閾値を並べ替えない |
| CORE-LAY-027 | 二値化shape | global／feature／channel、非正方2D・非立方3D、1D入力、空batch、bit結合軸と順序を確認 |
| CORE-LAY-028 | 閾値初期化 | uniformの等分・quantileの補間しない順位選択、constantデータ、型・非有限・空dataset拒否を確認 |
| CORE-LAY-029 | 勾配clipの寿命 | 同じParameterへの複数forwardの合算後clip、deepcopy／assign復元、freeze時の安全性、hook重複なしを確認 |
| CORE-LAY-030 | 二値化snapshot | trainからenableしてもeval用閾値を保存し、通常eval・bool export・state復元・FX別入力の判定が一致 |
| CORE-LAY-031 | Residual shape | 2D／3D・非正方入力、2回のpool、明示projectionと生成時shape確認。未知projectionや実出力の不一致を拒否 |
| CORE-LAY-032 | Residual合流 | 通常relaxed ORとbool exportが二値入力で一致し、勾配・state・nested／shared切替を検証 |

## 11. Modeテスト

| ID | 操作 | 合格条件 |
|---|---|---|
| CORE-MODE-001 | `train()` | 連続経路を使用しexport bufferなし |
| CORE-MODE-002 | `eval()` | 離散経路で決定的 |
| CORE-MODE-003 | export enable | model全体eval、対応childを一度ずつ切替 |
| CORE-MODE-004 | export disable | 一時bufferを除去しevalを維持 |
| CORE-MODE-005 | shared child | 同一moduleを重複処理しない |
| CORE-MODE-006 | export中train | `train(True)` はRuntimeError、exportを暗黙に解除しない。明示解除後は通常評価に戻り、続く `train()` で学習できる |
| CORE-MODE-007 | model保護 | `Circuit.from_model()` 前後で元modelのstate・device・mode不変 |

## 12. Circuitテスト

| ID | 対象 | 合格条件 |
|---|---|---|
| CORE-CIR-001 | IR validation正常 | validな小型回路を受理 |
| CORE-CIR-002 | ID重複 | 対象ID付きで拒否 |
| CORE-CIR-003 | 未定義参照 | gateと参照ID付きで拒否 |
| CORE-CIR-004 | topological order | 後続参照とcycleを拒否 |
| CORE-CIR-005 | shape | input／output件数不一致を拒否 |
| CORE-CIR-006 | model tracing | 対応層の小型modelから期待IR |
| CORE-CIR-007 | FX supported ops | 登録済みview／index／logic演算を変換 |
| CORE-CIR-008 | FX unsupported op | node、target、hintを含むerror |
| CORE-CIR-009 | Python evaluate | 全入力でmodel evalと一致 |
| CORE-CIR-010 | simplify each pass | 各passが対象patternだけを簡約 |
| CORE-CIR-011 | simplify parity | 前後で全入力結果が一致 |
| CORE-CIR-012 | simplify idempotence | 2回目の変更件数0、dict同一 |
| CORE-CIR-013 | simplify limit | 未収束を正常扱いしない |
| CORE-CIR-014 | JSON round trip | dict→JSON→Circuit→dictが一致 |
| CORE-CIR-015 | schema invalid | field・型・版・enum違反を拒否 |
| CORE-CIR-016 | compiled runtime | compile前後の出力一致 |
| CORE-CIR-017 | packed runtime | sample境界を跨ぐ件数で一致 |
| CORE-CIR-018 | logical output capture | 簡略化前に元出力順の対応表を保存し、各groupの入力順・定数・重複を保持 |
| CORE-CIR-019 | sum定数吸収 | `[a, 0, b, 1]` の加算簡約後も4出力の位置と全入力の値を保持 |
| CORE-CIR-020 | gate統合・配線置換 | 同値gate共有、wire bypass、NOT融合で対応表の参照も更新され、出力数と値は不変 |
| CORE-CIR-021 | 不要gate除去 | 論理出力からだけ参照されるgateを保持し、両出力から未参照のgateだけを除去 |
| CORE-CIR-022 | 全出力が定数 | reductionの定数簡約後も0／1の出力数・順序・元groupの区切りを維持 |
| CORE-CIR-023 | 保存済み対応表のJSON往復 | 簡略化後のsave→loadでも集約後出力と集約前bit列がそれぞれ元modelと一致 |
| CORE-CIR-024 | 対応表の不正 | 欠落、件数不一致、空tuple、未定義ID、reduction結果への参照を拒否 |
| CORE-CIR-025 | 集約なし・混在・同一信号反復 | 論理出力単体、集約と論理出力の混在、`[a, a, b, a]` で順序と重複を保持 |
| CORE-CIR-026 | 独立した期待値 | 変換元modelの集約前出力、簡略化前IR、簡略化後IR、JSON復元IRのbit列が一致 |
| CORE-CIR-027 | 外部FXの必須対応情報 | CORE-DES-011に従い情報なしでの直接変換を拒否。無効参照・件数不一致を拒否し、定数位置と重複を含む正しい対応情報を受理 |
| CORE-CIR-028 | 通常変換の自動記録 | `from_model()` は利用者の追加指定なしに簡略化前の対応を記録し、外部FXと同じ出力保護規則を満たす |

CORE-CIR-019〜026はCORE-FR-CIR-009の確認ケースとする。小型回路は全入力を列挙する。期待値を `logical_output_ids()` だけから作らず、変換元モデルの集約前Tensorまたは手書きの固定期待値と照合する。ID番号自体の不変性ではなく、各出力位置の論理値の不変性を判定する。

:::info C9の数値契約は承認済み
[CORE-DES-012](./detailed-design.md#1710-c9の集約計算と途中丸め承認済み) に従い、演算順序・数値型・typed scalarを保存し、PyTorchと変換後の結果を照合する。Verilogの加算前0／1列と、Python／Cの集約スコアは比較する境界が異なる。全device・compilerでの無条件なbit一致を、この仕様だけで保証しない。
:::

追加検証対象は、通常GroupSumのbias／tauによる桁落ち、連続したscalar演算の順序、数値型の昇格と丸め、簡略化・JSON往復・Python実行・生成Cを通した集約スコアの維持である。加算前の0／1列と整数値は完全一致、浮動小数点はCORE-DES-014に従い元の演算順序・型を守った微小な丸め差を許容する。非有限値が生じる例はNaN／Infを分類し、0の符号も確認する。これらは開発時の回帰テスト計画であり、運用時に全入力を列挙する処理の追加ではない。

**CORE-DES-014承認済み:** 小数の比較は `torch.testing.assert_close` の標準dtype別設定を基本とする。基準環境の `(rtol, atol)` はfloat16 `(1e-3,1e-5)`、bfloat16 `(1.6e-2,1e-5)`、float32 `(1.3e-6,1e-5)`、float64 `(1e-7,1e-7)`。学習・評価でも適用できる箇所に限定し、既存の厳密テストを機械的に変更しない。NaNは意図した例だけ分類して確認し、bit・整数・構造・元データの不変性には許容誤差を適用しない。経緯は [詳細設計17.12節](./detailed-design.md#1712-c10の浮動小数点一致基準承認済み) を参照する。

| ID | 観点 | 合格基準 |
|---|---|---|
| CORE-CIR-029 | 桁落ちの反例 | 微小tau付きGroupSumと大きな加減算の既知2例をPyTorchと完全一致させる |
| CORE-CIR-030 | scalarの形態 | Python scalarと0次元／多次元Tensor、float16／bfloat16／float32／float64の型昇格を保つ |
| CORE-CIR-031 | 順序とalpha | 演算列・左右位置・alphaを保ち、独立した乗算への展開や係数統合を行わない |
| CORE-CIR-032 | 数値castと分岐 | 明示castとcat／stackの暗黙castを各分岐へ保存し、元の枝を変更しない |
| CORE-CIR-033 | 数値情報の検証 | 不正enum、定数型、rank、operand、cast引数、最終dtype不一致を拒否する |
| CORE-CIR-034 | 周辺設定と非有限値 | 追跡metadataと既定dtypeの不一致を診断し、元の演算で生じるInf／NaNを隠さない。実行API完成時には既定dtype変更後も保存済みの型を使う |

## 13. 外部toolテスト

### 13.1 C

Cも二値化済みの0／1入力を対象とする（CORE-DES-010承認済み）。以下はbool／packedの論理回路・集約のテストであり、uint8画像や浮動小数点値の前処理を生成するテストではない。入力shape・順序を維持し、packedのwordを画素値として解釈しないことを確認する。

| ID | 条件 | 合格条件 |
|---|---|---|
| CORE-EXT-C-001 | 各GateOpの生成source | C compiler error 0件 |
| CORE-EXT-C-002 | 小型回路全入力 | Python IRと全出力一致 |
| CORE-EXT-C-003 | reduction型境界 | 8／16／32／64 bitとfloatが期待どおり |
| CORE-EXT-C-004 | inline true／false | 両方がPython IRと一致 |
| CORE-EXT-C-005 | pack 8／16／32／64 | 端数sampleを含めて一致 |

### 13.2 Verilog

| ID | 条件 | 合格条件 |
|---|---|---|
| CORE-EXT-V-001 | 各GateOpの生成source | Verilator lint error 0件 |
| CORE-EXT-V-002 | reductionなしの小型回路全入力 | testbench出力と `evaluate_logical_outputs()` が一致 |
| CORE-EXT-V-003 | 複数SumReduction | 保存済み対応表の元group順・入力順で生出力を生成し、変換元modelの集約前出力と一致 |
| CORE-EXT-V-004 | inline true／false | 両方で最終論理層の生出力が一致 |
| CORE-EXT-V-005 | 識別子衝突候補 | 一意なsignal名で生成可能 |
| CORE-EXT-V-006 | GroupSumを含むmodel | arithmetic回路、divider、bias加算、`SCORE_SCALE`がなく、入力bitを出力する |
| CORE-EXT-V-007 | 同じwireを複数reductionが参照 | 重複を省かず出力位置と要素数を維持 |
| CORE-EXT-V-008 | MNISTLGN相当のGroupSum | 10個の加算結果ではなく4,000 bitの生論理出力を生成 |
| CORE-EXT-V-009 | 定数吸収・gate共有後にJSON再読込 | inline true／falseともに元modelのbit列と一致し、出力幅・端子順を保持 |
| CORE-EXT-V-010 | 全出力が定数 | gate計算が不要でも期待幅の端子を生成し、定数bit列と一致 |
| CORE-EXT-V-011 | 二値化済み入力 | 0／1の入力信号から論理出力を生成し、uint8画素・浮動小数点値を二値化する前処理回路を含めない。入力端子数は二値化後shapeに一致 |

Verilog生成後のシミュレーション結果はPython IRだけでなく、変換元モデルから取得した集約前出力とも照合する。CORE-EXT-V-006では生成された演算・接続を検査し、説明comment内の `tau` や `bias` という語を演算回路の存在と混同しない。

## 14. 単独packageテスト

| ID | 操作 | 合格条件 |
|---|---|---|
| CORE-PKG-001 | `python -m build` | wheelとsdistを生成 |
| CORE-PKG-002 | wheel内容検査 | source、LICENSE、noticeを含みtests／examplesを含まない |
| CORE-PKG-003 | 隔離環境install | Git情報なしでinstall成功 |
| CORE-PKG-004 | root外import | 公開API import成功、logicNN本体不要 |
| CORE-PKG-005 | basic dependency | Alkaid、Rich、TorchVisionなしでimport成功 |
| CORE-PKG-006 | optional Alkaidなし | adapter以外の機能が正常 |
| CORE-PKG-007 | optional Alkaidあり | plugin importと最小変換が成功 |

## 15. 汎用性受入ケース

| ID | model | 入力 | 合格条件 |
|---|---|---|---|
| CORE-GEN-001 | 2層Dense Raw | 4特徴、3出力group | 学習・eval・回路化成功 |
| CORE-GEN-002 | Dense Warp／Light | rank 4／6 | forward・backward・eval成功 |
| CORE-GEN-003 | Conv2d | 1×5×7、非正方kernel | shape・gradient・eval成功 |
| CORE-GEN-004 | Conv3d | 2×4×5×6 | shape・gradient・eval成功 |
| CORE-GEN-005 | learnable connection | 表形式8特徴 | 接続学習・離散化・state復元成功 |
| CORE-GEN-006 | multi-threshold binarization | 3 channel連続値 | channel別thermometer encoding成功 |
| CORE-GEN-007 | Residual | shape一致の小型画像 | train／eval／export成功 |

これらはMNIST fixtureを使用しない。MNISTLGNは外側logicNNの統合・受入テストで別途検証する。

## 16. torchlogix比較方針

### 16.1 一致を要求するもの

- 16論理関数のIDと真理値。
- 対応する固定weight・indexにおけるforward結果。
- Raw／Warp／Lightの真理値表抽出。
- OR Pooling、GroupSumおよび各二値化方式の定義済み挙動。
- 既存回路が表す論理結果。

### 16.2 差分を許すもの

- 意味を変えないclass内部名、file配置、error文、dict順序。
- 修正対象として記録したassert依存、device固定、不正値の黙認、非決定的状態。
- 明示的に新schemaへ移行したJSON field名。
- 数値安定性のための許容誤差内の連続値。
- 承認済みの補助正則化 `warp` の実装保留。旧式との数値一致に代えて、CORE-REG-001〜004の拒否・状態保護・Warp本体との分離を確認する。

差分を許可する場合は、テストを削除せず、理由、旧結果、新結果、仕様根拠を最終検証結果へ記録する。

## 17. 性能記録

性能は機能合否と分離して次を記録する。

- LogicDenseのforward／backward。
- Conv2dのforward／backward。
- materialize有無の速度とpeak memory。
- Python IR、compiled C、packed Cの推論速度。
- circuit simplifyのgate数、処理時間、peak memory。

固定shape、seed、dtype、device、warmup回数、反復回数を記録する。初回基準を作成し、その後20%以上の退行があれば理由をレビューする。

## 18. 実行順序

```text
1. 実装前に対応単位の仕様を確認し、既存テストの合格状態を確認する
2. 対応実装単位のテストを先に作成する（未実装状態では実行しない）
3. 本体を実装する
4. 対応テストと全coreテストを最初の不合格で終了する設定で実行する
5. logicNN全体の既存テストを同じ設定で実行する
6. Ruff、compile、依存境界検査を行う
7. torchlogix parityと自己レビューを行う
8. memoと検証記録を更新する
```

テスト不合格時は原因を分類する。テストコードの誤り、lint、仕様が明確な実装不具合は、仕様や合格基準を弱めず修正・再検証して継続する。仕様定義を変更する必要がある場合や不明点がある場合だけ中断し、根拠と修正案を報告する（2026-09-12更新）。手順は引き続きテスト作成→本体実装→実行とし、意図的なred確認は省く。

## 19. 最終合格条件

1. CORE-AC-001〜006をすべて満たす。
2. 機能維持表の全項目に実装・移行・明示判断のいずれかがある。
3. 必須CPUテスト、packageテスト、Cの最終出力等価性およびVerilogの最終論理層等価性が合格する。
4. 利用可能な場合はCUDAとAlkaidの追加結果を記録する。
5. logicNN本体から `torchlogix` のimportと外部path依存が0件になる。
6. MNISTLGNの学習・評価・回路出力と精度比較が外側の受入条件を満たす。
