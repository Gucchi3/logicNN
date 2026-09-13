---
sidebar_position: 5
title: 実装・検証記録
---

# logicNN_core 実装・検証記録

## C1: パッケージ基盤・設定型・例外

最終更新日: 2026-09-13。状態は **C1〜C10完了、C11の生成・compiled実行とC13のアプリ統合を検証中、C12の参照モデルは部分実装**。[C11以降の記録](#c11以降-cverilog生成とアプリ統合) を末尾へ追記する。過去の各合格件数はその工程時点の範囲であり、現在の追加実装を含む合計ではない。C・Verilogの二値化済み入力、外部FXの必須出力対応情報、回路実行結果のCPU Tensor返却は承認済み。補助正則化 `warp` は計算を実装せず、指定時の明示拒否のみ実装している。[C4の二値出力の不一致](#二値評価と抽出lutの不一致) とC9の初期反例は修正前の再現記録であり、各修正後の検証結果と区別する。

### 実装内容

1. `utils/logicNN_core/` の独立パッケージ構成、静的版0.1.0、MIT/帰属文書。
2. `LUTConfig` と `ConnectionConfig` の不変設定型、型・数値・方式別条件の検証。
3. `LogicNNCoreError` のメッセージと診断情報。
4. logicNNからのローカルeditable依存と開発用テスト。

回路変換・論理層・学習計算はまだ未実装であり、利用できるように見せるstubは公開していない。

### 実行結果

| 検証 | 結果 |
|---|---|
| 変更前の既存テスト | 95件合格、3.59秒 |
| `uv sync --extra dev` | ローカル依存のbuild・install成功 |
| C1を含む全体pytest（`-x -q`、core行カバレッジ付き） | 388件合格、17.66秒 |
| 独立packageテスト | 上記のうち4件。Gitなしsdist/wheel build、帰属文書・metadata、隔離wheel install・ルート外importが合格 |
| C1本体の独立読取りレビュー | 実害のある指摘なし |
| Ruff | import区切り修正後に合格 |
| Python 3.10構文・AST検査 | 合格。全関数docstring、行長160以内、core依存方向も検査 |
| C1行カバレッジ | 100%（92行、未実行0行） |
| 文書build | C2の文書更新と併せて成功 |

### 品質指摘の解消と今後の中断条件

`tests/core/test_exceptions.py` と `tests/core/test_layer_settings.py` の、pytestとlogicnn_coreのimport間に不要な空行がある。Ruffは両者を同じ外部依存グループと判定するため、I001を報告した。計算ロジックの不一致やpytestの失敗ではない。

初回は当時の停止条件に従って中断した。その後、利用者が「テスト側の問題は修正して再実行してよい。中断は仕様変更が必要なときや不明点があるとき」と指示を更新したため、2箇所のimport区切りを整理し、品質チェックと全テストを再実行した。巨大整数の数値検証とソース品質のテストも追加して合格した。

以後は、仕様が明確なテスト・実装不具合は修正して再検証する。仕様や期待値を不合格回避のために緩めず、仕様変更または未確定判断が必要なら作業を区切って報告する。

## C2: 論理関数・sampling

### 実装内容

1. `functional/logic.py`: 二入力16論理関数、broadcast対応の連続式、bool演算、FX追跡可能なexport LUT。
2. `functional/sampling.py`: 温度付きsoftmax／sigmoid、Gumbel sigmoid、straight-through、forward不変の勾配倍率。
3. `functional/__init__.py`: 実装済み7関数だけの公開入口。未実装の層・回路APIを公開しない。
4. 由来文書・MIT表示・README・公開引数とテスト仕様の追記。

### 実行結果

| 検証 | 結果 |
|---|---|
| 全体pytest＋core行カバレッジ | **707件合格、1件スキップ、12.75秒** |
| sampling単体 | 上記のうち207件。低precision、温度、seed／generator、gradcheck、hard勾配、巨大有限整数を含む |
| logic単体 | 上記のうち112件合格、CUDA実機1件スキップ。真理値表、独立連続式、勾配、FX別入力／ID再実行を含む |
| core行カバレッジ | **99%**（236行、未実行2行）。未実行はCUDAとのdevice混在拒否。samplingは100% |
| 独立配布検証 | sdist→wheel、帰属文書、隔離install・アプリ外importを全体テストで再実行して合格 |
| Ruff・構文・docstring・行長・依存方向 | 合格 |
| 独立レビュー | 巨大有限整数のscalar変換問題を修正後、追加の修正必須事項なし |
| 旧samplingとの比較 | 担当者が旧定義だけをAST抽出してread-only比較。forward／backward 16ケース合格 |
| Docusaurus | `npm run build` 成功 |

全体テストの再実行コマンドは、`logicNN/` をカレントディレクトリにして次を使用した。

```shell
.venv/Scripts/python.exe -m pytest -x -q --cov=logicnn_core --cov-report=term-missing
.venv/Scripts/python.exe -m ruff check .
```

### 仕様内で修正した境界不具合

- 高温・低precisionでsoftmax確率が同値へ丸まっても、hard選択は元logitの最大indexを選ぶ。
- 定数論理関数は、大きな有限入力の和がoverflowしても定数値とゼロ勾配を保つ。
- Pythonの巨大な有限整数は、温度・閾値・倍率の検証後にfloatへ正規化する。PyTorchへの整数scalar変換で起きたOverflowErrorを解消した。
- いずれも期待値を緩めず、回帰テストを追加して再実行した。sampling担当でcoverage起動が一度無出力終了したが、親で同一環境の全体coverageを正常完了して結果を確認した。

### 次の工程の中断理由

C3の先行レビューで、[正則化の定義不足](./detailed-design.md#173-c3開始前の確認事項) が見つかった。旧warp正則化の説明と参照する重みが一致せず、loss集約と再スケールのゼロ分母について判断を求めた。

その後、ゼロ分母なら書換え前に止める安全対策の必要性が了承され、文献調査を経て補助正則化 `warp` だけの実装保留が承認された。Warpパラメータ化そのものは維持する。**残る未承認事項はCORE-DES-004の返却・集約契約**であり、C3のコード・テストコードは未実装のまま確認を求める。

## C3開始前: warp正則化保留の仕様反映

2026-09-12、要件の機能維持表・基本設計・詳細設計・テスト仕様・概要・READMEに、承認済みの保留範囲を反映した。

1. 方式名 `warp` の補助正則化は計算を実装せず、指定された場合は実装保留を示すValueErrorとする。
2. 拒否によって重み・勾配・学習評価状態を変更しない。該当入口を実装するC3・C6・C7のテスト計画へ追記した。
3. Warp方式本体とそのテスト計画は維持する。
4. CORE-DES-004の提案を、層内の全ゲートを等しく平均する案として整理した。未承認のままであり、実装やテストの期待値には反映していない。

今回の変更は文書のみで、Python本体・テストコード・旧logicnnは変更していない。上記707件の合格はC2完了時の結果であり、保留契約の実装・合格を意味しない。

文書検証: `npm run build` は成功し、警告は報告されなかった。生成HTMLに17.4節・17.5節の見出しアンカーがあることも確認した。Pythonコードの変更がないため、pytestは今回再実行していない。

独立した読取りレビューでも、保留範囲、未承認事項の保持、将来の実装と現在の実績の区別、見出しリンクに修正必須の問題は見つからなかった。

## C3: Walsh／Light・組合せ・正則化

CORE-DES-004の承認後、テストを作成してから数学基盤を実装した。層内全levelの集約と層公開入口からのwarp拒否伝播は、層を実装するC6／C7で検証する。

### 実装と確認

- `functional/walsh.py`: 非正規化FWHT、真理値表順のWalsh／Light基底、materialize／streaming縮約。独立した行列・積の基準、数値微分、旧式との読取り比較を実施。
- `functional/combinatorics.py`: colex順のunranking、疎な重複なしsampling、bool真理値表。整数overflow検証と先行shapeの保持を追加。乱数はPyTorch seed／generatorに統一。
- `functional/regularization.py`: 承認済みの平均loss、保留warpの拒否、ゼロ分母・非有限結果・重複メモリを更新前に検証する再スケール。
- `functional/__init__.py`: 実装済み18関数を公開。公開名テストも18関数に更新し、存在しない層や回路のstubは公開しない。

### 検証結果

| 検証 | 結果 |
|---|---|
| 全体pytest＋core行カバレッジ | **1345件合格、7件スキップ、15.14秒** |
| C3の内訳 | Walsh／Light 257件、組合せ202件、正則化179件が合格 |
| 未実行 | CUDA実機を要する7件。合格とは扱わない |
| core行カバレッジ | **99%**（513行中3行未実行）。組合せ・正則化は100% |
| 独立配布・ソース品質 | 全体pytest内でsdist／wheel・隔離install・Python 3.10構文・docstring・160文字行長・依存方向が合格 |
| 再スケールの独立レビュー | 重複記憶位置の問題を修正後、14,424種類の配置で独立offset基準と一致 |
| 組合せの参考計測 | CPUで784入力から2要素tupleを4096個生成、単発約0.021秒。GPU性能を保証する計測ではない |
| Ruff／Docusaurus | `ruff check .` と `npm run build` が成功 |

### 修正・未検証事項

1. 最初の全体実行で、テストの1行が160文字制限を2文字超えた。引数行を折り返して修正し、全体を再実行して合格した。数式や期待値は変更していない。
2. 独立レビューで、同じ記憶位置を複数係数が共有するTensorではin-place再スケールが異なるゲートの結果を保存できないことを確認した。通常の転置・間引きは維持し、重複する更新先だけを書換え前に拒否する。重み・勾配・基底storageの無変更を回帰テストで確認した。
3. 担当者による個別coverage起動でWindows上のtorch import異常終了があった。依存を変更せず、親側で逐次の全体coverageを正常完了した。上表を正式な結果とし、異常終了した起動を合格回数へ含めない。

## C4: Raw／Warp／Lightパラメータ化（検証中断）

### 作成済みの内容

1. `parametrizations/base.py`、`builder.py`、`__init__.py`: 重みを所有しない共通interface、不変のconfigと実行中temperature、dtype／device／shape検証、真理値表とIDの公開入口。
2. `raw.py`: 16ゲートのsoft／hard／Gumbel選択、argmax評価、旧式のresidual／random初期化。
3. `warp.py`: Walsh係数の初期化・縮約・二値化、catalog初期化、真理値表抽出。下記の既知の不一致が未解消である。
4. `light.py`: 真理値表係数のsamplingと連続基底、旧式のresidual／random初期化。
5. 共通・Raw・Warp・Lightのテストを、対応する実装前に作成した。詳細設計5.3〜5.5節と由来文書も更新した。

### 通常テストの結果と限界

| 検証 | 結果 |
|---|---|
| 全体pytest＋core行カバレッジ | **2179件合格、16件スキップ、20.56秒** |
| C4の内訳 | 共通400件、Raw192件、Warp／Light合計242件が合格 |
| 未実行 | CUDA実機を要する16件。合格と扱わない |
| core行カバレッジ | **98%**（732行中13行未実行）。Raw／Warp／Light各本体の行カバレッジは100%だが、全数値境界の正しさを保証する数字ではない |
| Ruff・ソース品質・独立配布 | Ruff合格。構文・docstring・行長・依存方向・sdist／wheelは全体pytestに含み合格 |
| 追加の独立検証 | 下記の出力不一致を検出。**C4全体は未合格・未完了** |
| Docusaurus | 追加見出しのアンカー構文を修正し、自動生成された見出しIDへリンクを統一後、`npm run build` が警告なく成功 |

上記pytestは作成済みの通常ケースに対する実行結果であり、後述の反例を解消した結果ではない。反例をxfailで隠したり、異なるbitを許容誤差で合格にしたりしていない。どの数値境界を基準にするかの決定後、反例を自動回帰テストへ追加して全体を再検証する。

通常の検証中は、Rawの極端な温度でのGumbel計算をC2の安定化softmaxへ統一した。Warp／Lightの旧Gumbelとの比較では、非連続Tensorで同じseedでもノイズの配置が変わることを確認し、比較テストだけで位置の一致した同じノイズを与えた。新実装のseed再現性は別に検証しており、数式の不一致を緩い許容誤差で隠す対応ではない。

### 二値評価と抽出LUTの不一致

CPU、PyTorch実行環境内で読取り再現し、別担当者も独立確認した。

最小例（インストール済みのlogicnn_coreを使用）:

```python
import torch
from logicnn_core.layer_settings import LUTConfig
from logicnn_core.parametrizations import WarpLUTParametrization

model   = WarpLUTParametrization(LUTConfig(kind="warp"))
weights = torch.tensor([[1, -1e-8, -1, -1e-8]], dtype=torch.float32)
inputs  = torch.zeros(1, 2, 1)
print(model(inputs, weights, training=False, contraction="n,bn->bn"))
print(model.truth_tables(weights)[:, 0])
```

現結果は評価が `tensor([[1.]])`、入力00の抽出値が `tensor([False])`。係数 `[1e8, -1, -1e8, -1]` でも再現する。

原因は、数学的には等しい和でも、forwardとFWHTで浮動小数点の加算順が異なること。連続値としての微小差が、0を境界とした判定で1 bitの相違になる。C3のFWHTの数式そのものが誤っているという意味ではない。

旧 `parametrization.py` の `get_luts()` は、`functional.py` の変換を既定の明示Hadamard行列積で実行する。上のfloat32例では旧forwardと旧LUTがともに1となるため、この例だけから旧実装も同じ不具合だとはいえない。

追加の低precision再現条件は以下である。

- CPU、rank 6、`torch.Generator().manual_seed(241)`。
- `torch.randn(128, 64, dtype=torch.bfloat16, generator=generator)` の128ゲート。
- 二進昇順の全64入力を `[64, 6, 128]` に展開し、`contraction="n,bn->bn"` で評価する。
- 比較用Hはbfloat16の64次単位行列に非正規化FWHTを適用した厳密な±1行列。行列抽出値は `(weights @ H < 0).T`。

| 経路の比較 | 今回の8192出力中の不一致 |
|---|---:|
| 新streaming評価と行列抽出 | 11 bit |
| 新materialize評価と行列抽出 | 0 bit |
| 新FWHT抽出と行列抽出 | 10 bit |
| 新streaming評価と新FWHT抽出 | 17 bit |
| 新materialize評価と新FWHT抽出 | 10 bit |
| 旧streaming評価と旧get_luts | 11 bit |
| 旧materialize評価と旧get_luts | 0 bit |

旧get_lutsは内部でfloat32行列積を使うが、この入力では上記行列抽出と同じ表になった。件数は当該環境での再現値であり、すべてのbackendで同件数になる保証ではない。**明示行列積へ戻すだけでは、低precisionの経路差は残る。**

### 判断待ちと再開条件

1. [CORE-DES-007](./detailed-design.md#176-warpの二値評価とlut抽出の判定契約承認済み): この中断後、共通の真理値表を正式な判定元とし、高精度・固定順で表を生成する方針が承認された。学習式を維持してC4の修正・検証を再開する。
2. [CORE-DES-008](./detailed-design.md#177-学習可能接続の旧計算との相違承認済み): C5の先行読取りで、旧学習可能接続のhard forward・独自勾配と、新草案のsoft選択の相違を確認した。その後、利用者がtorchlogixと同じ手法を指定し、旧学習計算を維持することが承認された。
3. 仕様の決定まではC4の既知の問題を未解消として保持し、C5以降を実装しない。旧logicnnを変更せず、依存追加や設定の黙示変更も行っていない。

## C4追加修正: CORE-DES-007の共通判定表

利用者の承認により、学習式を維持し、CPU float64の固定順FWHTで生成したbool真理値表を二値evalと抽出の共通基準とした。

- `truth_tables()` は独立したCPUコピーから表を生成し、非有限係数・変換overflowを拒否したうえで元deviceへ返す。C3のFWHTを再利用し、同じ数式の別実装を増やしていない。
- 二値evalは整数の表indexで参照し、materializeの有無で基準を変えない。混合入力では元dtypeでゲート・位置ごとに二値性を判断し、連続値のevalと学習は従来計算を維持する。
- contractionの軸に沿って表の行・入力bit・maskを対応付ける。二値evalでゲート軸を還元する操作は1つの論理ゲートの評価ではないため明示拒否する。学習と全連続evalでは元の汎用縮約を維持する。
- 内部キャッシュを追加せず毎回最新の重みから再生成する。層・回路側で同じモデル状態の表を共有・保存する統合はC6／C7／C9以降の検証対象であり、今回完了したとは扱わない。

### 修正後の検証結果

| 検証 | 結果 |
|---|---|
| 全体pytest＋core行カバレッジ | **2286件合格、20件CUDAスキップ、22.42秒** |
| 今回の追加テスト | **107件合格、4件CUDAスキップ**。軸対応12件、数値契約・状態・学習回帰95件 |
| 既知の反例 | float32相殺例の入力00はeval・表とも1。bfloat16 rank6の8192出力は共通表と両materialize設定で完全一致 |
| 学習の維持 | 既存の全sampling・数値勾配に加え、二値入力の学習も表を生成せず、旧連続式の出力・入力勾配・係数勾配を維持 |
| 状態の保護 | CPU float64でも独立コピー、非有限・overflow拒否、既存grad・RNG・weight・Module状態の無変更、更新後の再生成、返却済み表の不変を確認 |
| core行カバレッジ | **98%**（771行中14行未実行）。Warpは99%。行カバレッジを全数値条件の正しさ保証とは扱わない |
| 独立配布・ソース品質 | sdist／wheel・隔離install、Python 3.10構文、全関数docstring、160文字行長、依存方向を全体pytestに含み合格 |
| 独立コードレビュー | 整列・安全なindex・混合入力・独立snapshot・非有限伝播・回帰テストを読取り確認し、重大な指摘なし |
| Ruff・Docusaurus | `ruff check .` 合格、`npm run build` が警告なく成功 |

参照元の旧logicnnは変更していない。修正前の診断結果を削除せず、承認した数値契約の変更と修正後の回帰結果を区別して残す。

### CPU参考計測

PyTorch 2.14.0+cpu、10スレッド、float32。warmup後の3回の中央値で、単位はms。小規模な参考計測であり、GPU・学習全体の性能や普遍的な速度比を保証しない。

| rank・ゲート数・batch | 表生成のみ | 新二値eval（毎回の表生成込み） | 旧streaming式のeval |
|---|---:|---:|---:|
| 2・4000・128 | 0.876 | 8.046 | 2.025 |
| 6・128・64 | 0.456 | 1.492 | 2.708 |

二入力ではCPUのindex整列・参照等の負担で遅くなり、六入力では基底計算を省く効果が見られた。層側の評価スナップショット再利用で表生成の重複を省く余地はあるが、表生成以外の時間もあるため、キャッシュだけで速度が戻るとはしない。性能を理由に二値出力の一致を緩めない。

## C5: 固定・学習可能な接続

CORE-DES-008の承認を受けて実装した。学習可能Denseは旧torchlogixのhard選択と独自のlogit／入力勾配を維持する。評価時の乱数なしargmax、温度既定値1.0、候補の重複制約は承認済みの新仕様を維持した。

| 検証 | 結果 |
|---|---|
| C5追加テスト | 274件合格、4件CUDAスキップ |
| 全体pytest＋core行カバレッジ | **2560件合格、24件CUDAスキップ、24.88秒** |
| core行カバレッジ | 97%（1231行中38行未実行）。Dense／Conv接続は各95% |
| 学習方式 | 旧コードとの照合と独立した解析式でforward・両勾配を確認。Gumbelノイズ共有、温度、同点、dtypeも検証 |
| 汎用性 | 任意先行shape、非連続・空batch、非MNIST 2D／3D、tuple kernel／stride／padding、channel groupとuniqueを検証 |
| 保存状態 | indexと座標の往復、値・shape・範囲・合成関係の不正拒否、後段node permutationを検証 |
| 独立レビュー | 数式・dtype・候補分離・座標・groupに修正必須の指摘なし。state保証の範囲を明文化 |
| 品質・文書 | Ruff合格。Docusaurus build成功。既存の単独配布・構文・docstring・160文字行長の検査も合格 |

存在する不正なstate Tensorは接続自身へのコピー前に拒否する。一方、missing／unexpected keyを含む全失敗の原子性や、親子module全体の巻戻しは保証せずPyTorch標準に従う。CUDA実機の動作・性能とC5の大規模性能は未検証である。

## C6: 共通論理層・LogicDense

LUTの重みは層が所有し、C4のparametrizationとC5の接続を組み合わせた。入力勾配だけに倍率を適用し、通常evalでは最新の重みを参照する。rank 2のexportはbool入力と保存済みのLUT ID・接続snapshotだけを使う。

| 検証 | 結果 |
|---|---|
| C6追加テスト | 247件合格、2件CUDAスキップ |
| 全体pytest＋core行カバレッジ | **2807件合格、26件CUDAスキップ、25.11秒** |
| core行カバレッジ | 96%（1351行中48行未実行）。Dense層96%、抽象メソッドを含む共通基底88% |
| 計算 | 全方式・rank・samplingの独立部品結合、任意先行shape・空batch、入力だけの勾配倍率、全16論理関数のeval／export一致 |
| 状態・FX | snapshot再enable、通常／export間のstate往復、子接続復元後の再生成、未知keyのstrict拒否、別入力でのFX再実行 |
| 独立レビュー | 数学・入力dtype・FX・stateに追加修正点なし。余分なroot aliasを除去し、層は仕様どおりlayersだけに公開して再検証 |
| 品質・文書 | Ruff・Docusaurus build成功。独立配布・docstring・160文字行長・Python 3.10構文の検査も合格 |

全modelの再帰的export切替とCircuit生成はC9で実装する。現段階では個別層の切替を提供しており、未実装の全体APIをstubとして公開していない。

## C7: 畳み込み層・OR pooling・GroupSum

LogicConv2d／3dは旧のnode単位初期化と多段のLUT計算を維持した。通常のpaddingとrank軸整列はC5が担当する。正則化は全levelの全ゲート平均、再スケールは全候補を検証してから更新する。OR poolingは通常float32 max／export bool OR、GroupSumは旧のdtypeと条件付きbias／tau演算を維持した。

| 検証 | 結果 |
|---|---|
| C7追加テスト | 347件合格、5件CUDAスキップ |
| 全体pytest＋core行カバレッジ | **3154件合格、31件CUDAスキップ、31.05秒** |
| core行カバレッジ | 97%（1599行中52行未実行）。Conv層97%、pooling／GroupSum各100% |
| Conv | 全方式・対応rank、非MNIST 2D／3D、多段shape・全level勾配、node単位初期化、独立geometry／LUT基準と一致 |
| 更新と状態 | 全ゲート平均、失敗時の全level無変更、共有storageの明示拒否、snapshot再enableとstate復元、別入力のFX実行 |
| 層結合 | Conv→OR pooling→Dense→GroupSumの小型2D／3Dモデルで、全Parameterへの勾配と、eval・export・FX・state往復の一致を確認 |
| 独立レビュー | 多段軸、padding、更新前検証、pooling窓、集約dtypeに追加修正点なし |
| 品質 | Ruff、Python 3.10構文、全関数docstring、160文字行長、独立package build／installに合格。CORE-DES-009の記録を含むDocusaurus buildも警告なく成功 |

通常学習の全体速度やCUDA実機動作の保証ではない。今回の結合試験はC7までの個別層を明示的に切り替えたもので、未実装のCircuit builderやC／Verilog exporterを検証したことにはしない。

## C8事前確認: 学習可能二値化の閾値

旧コードの読取りと、float64で旧式を計算する最小再現により、学習と評価の閾値が異なることを確認した。`raw_diffs=[0.3,-0.1]` で学習時は約 `[0.3,0.33132675096]`、旧評価時は `[0.3,0.2]` となる。入力0.25を同じ厳密な `>` で判定すると `[0,0]` と `[0,1]` に分かれる。

[CORE-DES-009](./detailed-design.md#178-学習可能二値化の閾値の不一致旧方式の維持を承認) として一度実装を停止し、その後利用者がtorchlogixの計算を維持する方針を承認した。共通閾値案は採用せず、trainとevalそれぞれの旧計算を検証基準とする。この診断はC8の実装済みテストの失敗ではなく、仕様との照合で発見した相違である。

## C8: 二値化・Residual・共通export切替

CORE-DES-009の承認に従い、初期閾値の単純差分、trainだけのsoftplus変換、eval／exportのraw cumsumを維持した。逆順の評価閾値を並べ替えない。Fixed／Dummy／Soft／Learnable、閾値生成、Residualを実装し、Residualに必要な共通export walkerだけをC9から前倒しした。

| 検証 | 結果 |
|---|---|
| C8追加テスト | **275件合格、5件CUDAスキップ**。二値化180件、Residual71件、共通mode13件、層結合11件 |
| 全体pytest＋core行カバレッジ | **3429件合格、36件CUDAスキップ、35.16秒** |
| core行カバレッジ | 97%（1942行中57行未実行）。二値化99%、Residual97%、共通mode100% |
| 旧計算 | 閾値のモード差、全sampling、元dtypeでの比較・norm、hardのdtype昇格、uniform／quantile初期化を旧コードまたは独立式で照合 |
| 保存・勾配 | 複数forward合算後のclip、deepcopy／assign復元後のhook、freeze、snapshot再生成と未知keyの拒否を確認 |
| shape・Residual | global／feature／channel、非MNISTの1D系列・2D／3D、空batch、bit順序、旧の2回pooling、明示projection、relaxed ORとbool ORを確認 |
| 層結合 | 二値化→Dense→GroupSumで全Parameterの有限勾配、通常eval・export・FX別入力・state復元の一致を確認 |
| 独立レビュー | 共通切替のeval順序、channel軸の一般化、元dtypeのclipを修正し回帰テストを追加。再レビューに追加修正点なし |
| 品質 | Ruff、独立sdist／wheel build・隔離install、Python 3.10構文、全関数docstring、160文字行長が合格。C8とC9確認事項を含むDocusaurus buildも警告なく成功 |

共通export入口は設定だけのimportへPyTorchを追加せず、必要時に遅延importする。共有・入れ子moduleを一度ずつ処理し、解除後もevalを維持する。全model切替の失敗時transactionは保証しない。CUDA実機と大規模性能は未検証であり、FX追跡の成功をCircuit builderやC／Verilog出力の完成とは扱わない。旧logicnnは変更していない。

## C9事前確認: 入力境界と外部FXの出力情報

詳細設計のIR型、構築手順、対応能力表を旧ソースと独立して照合し、[CORE-DES-010／011](./detailed-design.md#179-c9開始前の入力境界と外部fxの出力情報承認済み) を確認事項として記録した。当初はbool wireだけの回路と実数二値化の範囲、元の出力対応がない外部FXに対する保護保証を未決定として停止した。

旧の `from_model` はbool sampleを使い、builderに入力依存のgt比較handlerはない。外部FXの `sum([a,b])+1` は、簡略化前の4 bit集約と数値だけでは区別できない。この照合は新Circuitの実装済みテストの失敗ではなく、実装前の契約確認である。C9の本体・テストは未着手のまま停止した。C8とこの確認記録を含むDocusaurus buildは警告なく成功した。

その後、C・Verilogとも二値化済み0／1入力とする方針、および外部FXの元出力対応情報必須が承認された。前処理wrapperは生成しない。受渡し情報を `FxSignalReference` とbool定数の列に具体化し、C9をテスト先行で再開する。この承認とテスト計画を、まだC9の実行合格結果とは扱わない。

## C9部分実装と数値契約の確認待ち

2026-09-13の修正前の記録。IR型・参照検証、FX builder、公開Circuit入口を実装したが、[CORE-DES-012](./detailed-design.md#1710-c9の集約計算と途中丸め承認済み) の判断待ちで一度停止した。その後、計算順・数値型を保持する方針が承認され、下記の数値情報修正へ進んだ。

### 実装と通常検証の範囲

| 検証 | 結果 |
|---|---|
| 全体pytest＋core行カバレッジ | **3655件合格、36件CUDAスキップ、39.94秒**。C9追加226件を含む |
| core行カバレッジ | 95%（2543行中115行未実行）。型・validationは100%、builderは86%、Circuit facadeは98% |
| 配布・ソース品質 | 全体pytest内の独立sdist／wheel build・隔離install、Python 3.10構文、全関数docstring、160文字行長が合格 |
| 静的検査・文書 | Ruff合格。CORE-DES-012停止記録を含むDocusaurus buildも警告なく成功 |

- stdlibだけで読めるIR型・検証を追加。shape、参照順序、ID重複、gateの端子数、reduction、論理出力対応を検査する。
- `Circuit.from_model()` はモデルを独立コピーし、CPU上で元weight dtypeを保ってexport・FX追跡する。元のmode、weight、gradient、buffer、CPU乱数状態を変更しない。
- `Circuit.from_fx_graph()` は必須の元出力対応情報を受け取り、元graphを変更せず、順序・定数・重複・group境界を記録する。未知・副作用演算を黙って読み飛ばさない。
- Dense全方式、2D／3D Conv、pooling、GroupSum、二値化、Residualの小型モデルと全二値入力を検証する。内部の数学的IR評価器はテスト用であり、C10の公開runtimeではない。
- 独立レビューに基づき、CPU固定、定数の指定1要素の検証、sumの整数overflow・精度境界の拒否、scalar Tensorのbroadcast shape、device指定の位置／keyword両形式を修正した。sumの境界検証だけで、次項の途中丸め問題が解消したとは扱わない。

### 既知の数値不一致

CPU、PyTorch 2.14環境の追加診断で次を確認した。これは合格済みテストの期待値を変更した結果ではなく、別途見つけた反例である。

| bool入力 `[[True]]` に対する計算 | PyTorch | 現IRの統合式をPython浮動小数点で評価 |
|---|---|---|
| `GroupSum(1, bias=-1.0 + 1e-8, tau=1e-8)` | float32の `0.0` | 約 `1.000000005` |
| `(inputs.sum(-1, keepdim=True) + 16777216.0) - 16777216.0` | float32の `0.0` | `1.0` |

最初の例は修正前のIRでは次の呼出しで再現できた。現行IRはtau／biasを持たないため、このコードは旧状態の診断記録であり実行用サンプルではない。`_data` への参照も診断用であり、公開データ取得APIではない。

```python
import torch
from logicnn_core import Circuit
from logicnn_core.layers import GroupSum

model     = GroupSum(1, bias=-1.0 + 1e-8, tau=1e-8).eval()
inputs    = torch.tensor([[True]], dtype=torch.bool, device="cpu")
circuit   = Circuit.from_model(model, (1,))
reduction = circuit._data.reductions[-1]
print(model(inputs))
print((1 + reduction.bias) / reduction.tau)
```

回路データが数値型・中間演算を保存していないため、式の統合だけでは元の計算を保証できない。通常のGroupSumでも再現するので、外部の追加演算だけを拒否する解決では足りない。仕様拡張案は未承認のまま記録し、dtype項目の追加や数値計算の変更は行っていない。

未実装の範囲はC10のruntime・簡略化・JSON保存、C／Verilog出力、任意backendおよびアプリケーション統合。Verilogは集約前のbit列だけを出す仕様のため、この数値スコアの問題とは分離する。CUDA実機・大規模性能も未検証。旧logicnnは変更していない。

## C9完了: 元の演算順序と数値型の保存

CORE-DES-012の承認に従い、tau／biasへの係数統合を廃止した。sum dtype、順序付きScalarOperation、typed scalar、alpha、左右関係、rank、castおよび最終output_dtypeを記録する。明示castとcat／stackの暗黙昇格を各枝へ保存し、FX metadataと代表Tensorのshape・dtype不一致を診断する。

| 検証 | 結果 |
|---|---|
| C9関連テスト | **553件合格、8.44秒**。型・検証・builder・facade・数値演算列を含む |
| 全体pytest＋core行カバレッジ | **3982件合格、36件CUDAスキップ、45.29秒** |
| core行カバレッジ | 96%（2643行中105行未実行）。型・validationは100%、builderは89%、facadeは98% |
| 既知の反例 | GroupSumの微小tau例、大きな加減算の例とも、保存IRを独立再生した結果がPyTorchと完全一致 |
| 数値型・順序 | float16／bfloat16／float32／float64、Python scalarとTensorの0／1／3次元、alpha、左右関係、明示／暗黙cast、分岐を比較 |
| 非有限値・診断 | 有限の設定から発生するNaN／Infを隠さず、符号付き0、不正enum／rank／dtype／operand、外部FXの既定dtype差を検証 |
| 独立レビュー | builderの型昇格・rsubとalpha・cat cast・分岐保持に追加修正点なし。C10で演算dtypeの意味検証と周囲の既定dtypeに依存しない再生が必要なことを確認 |
| 品質 | Ruff、独立package build／install、Python 3.10構文、docstring、160文字行長が合格。CORE-DES-012の正式仕様を含むDocusaurus buildも成功 |

数値列を再生する `tests/core/circuit_reference.py` は開発テスト用であり、公開runtimeの実装ではない。参照器はモデルのFXを呼び直さず保存IRを読み、論理ゲートの独立したbool評価と元の数値演算を組み合わせる。合格範囲は基準CPU環境でのC9であり、C／Verilogの生成・実行をまだ検証したことにはしない。

## C10事前確認: 実行APIの返却形式

CORE-DES-013として返却型・device・batch形状を確認し、その後CPU Tensorへ統一する承認を得た。旧torchlogixはPython経路でCPU Tensor、compiled経路でNumPy配列を返すが、この違いを引き継がない。通常のmodel(x)や学習をCPUへ変更する話ではないことも利用者と確認した。

## C10部分実装: 保存・簡略化

| 検証 | 結果 |
|---|---|
| C10追加テスト | **328件合格**。保存120件、簡略化208件 |
| 全体pytest＋core行カバレッジ | **4310件合格、36件CUDAスキップ、54.60秒** |
| core行カバレッジ | 96%（2932行中106行未実行）。serialization 99%、simplification 100% |
| 保存 | 全階層の厳密field／型／enum／版、重複key、NaN／Infinity／1e999、int／floatと符号付き0、元出力対応、独立container、原子的file保存を検証 |
| 簡略化 | 固定6pass、全真理値・NOT融合・定数fold・WIRE・重複・未使用node、全参照更新、raw-only出力、整数sumの同dtype ADD、浮動演算列の保持、収束を検証 |
| 独立レビュー | 相互read-onlyレビューで修正必須の指摘なし。公開facadeのcopy→成功時置換は今後の統合で検証する |
| 品質 | Ruff、独立package build／install、stdlibだけのserialization往復import、Python 3.10構文、docstring・行長が合格。CORE-DES-014確認待ちを含むDocusaurus build成功、詳細設計ページのHTTP 200も確認 |

実装済みは `circuit/serialization.py` と `circuit/simplification.py` の内部入口である。公開 `Circuit` のJSON／simplify統合、runtime、C生成・compile、Verilog生成まで完成したことにはしない。テストの非Collection iterable警告をtuple指定で修正し、最終全体実行では警告なし。

### 数値実行の追加診断

CORE-DES-014のfloat32例を親側でも再現し、1要素と16要素の差が `9.313225746154785e-10`、1 ULPであることを `torch.nextafter` で確認した。

```python
import torch

one  = torch.add(torch.full((1, 1), 1e-8, dtype=torch.float32), 0.1, alpha=0.1)
many = torch.add(torch.full((1, 16), 1e-8, dtype=torch.float32), 0.1, alpha=0.1)
print(one[0, 0].item(), many[0, 0].item())
```

独立processの低精度診断では、整数sourceを先に出力dtypeへcastする方法にも、高精度で演算して最後にcastする方法にも反例があった。scalarの左右・型・要素数で経路が変わるため、部分的に一致した方法を一般保証へ広げていない。現IRの小型・各要素再生テストの合格とは分けて記録する。

これらはC10公開runtimeのテスト結果ではなく、実装前の再生方法の診断である。[CORE-DES-014](./detailed-design.md#1712-c10の浮動小数点一致基準承認済み) の数値一致基準を確認するため一度停止した。その後、微小な丸め差を許容し過剰な特殊処理を避ける方針の承認を受け、実装を再開した。旧logicnnは変更していない。

## C10完了: CPU回路実行・公開API統合

CORE-DES-013／014に従い、CPU runtimeを `Circuit` の実行APIへ統合した。元のモデルを呼び直さず保存IRを評価し、scoreと生bitを別の入口で返す。JSON／dict往復と、候補コピーを成功時だけ置換する簡略化も公開した。C生成・compile・VerilogはC11以降であり、未実装APIを成功するstubとして公開していない。

| 検証 | 結果 |
|---|---|
| CPU実行・数値境界・lifecycle関連 | **157件合格、1件CUDAスキップ、3.25秒** |
| 最終全体pytest＋core行カバレッジ | **4467件合格、37件CUDAスキップ、46.64秒、警告なし** |
| core行カバレッジ | 96%（3092行中113行未実行）。runtime95%、facade99%、serialization99%、simplification100% |
| 入力・返却 | Tensor／NumPy、厳密0／1、非連続・逆stride、batch 0／1／複数、任意shape、CPU返却、元入力・grad保護、不正型・meta・sparse等の診断 |
| 論理出力 | 全GateOpの全真理値、rawとscoreの境界、定数・重複・順序、簡略化・JSON／dict往復後の元bit列 |
| 数値 | dtype・rank・scalar左右・alpha・cast・分岐、float16／bfloat16の既知境界、周囲dtype変更でも保存IRの結果不変 |
| 既知の回帰 | 0対約1の二反例は公開変換・評価・簡略化・保存復元を通して厳密0を維持。NaN／Infの分類と0の符号は別途確認 |
| 状態・失敗時保護 | 元モデルのweight・grad・mixed mode・export flag・CPU RNG、簡略化失敗時の所有IR、保存失敗時の既存fileを保護 |
| 品質 | Ruff、全体テスト内のpackage build／install、Python 3.10構文、全関数docstring、160文字行長が合格 |

独立レビューで入力・rank・出力境界・失敗時保護を確認し、数値専任の追加確認では初期実装が低精度のInfを有限化する箇所を修正した。有限値の大差や非有限値の違いを、小数許容誤差の拡大で隠していない。修正後の代表的な境界58テストも上記の最終全体実行に含まれる。これらは対応範囲と基準環境の検証であり、すべてのCPU kernelや極端な演算列のbit完全一致を保証するものではない。

次のC11についてはread-onlyの環境確認のみ実施した。native MSVCとWindows SDKは存在し、Cygwin gccも見つかったが、VerilatorはPATH等の限定検索で未検出。生成Cのcompile・実行やVerilogシミュレーションの合格とは扱わない。追加toolの自動install、旧logicnnの変更は行っていない。

## C11以降: C／Verilog生成とアプリ統合

### 実装と境界

- Cは二値入力から集約後スコア、Verilogは元の順序の生bitだけを出力する。Cの8／16／32／64 bit packingはsample方向に行う。
- Cは保存した整数／小数型と演算列を維持する。float16／bfloat16の演算と浮動小数点から整数へのcastは未対応として診断付きで拒否し、別dtypeへ置換しない。
- compileは専用一時directoryを所有し、native libraryの解放後に削除する。compile成功時だけbackendを差し替え、簡略化成功時に無効化する。JSONへnative状態を保存しない。
- 既存MSVCの開発環境を子process内で使用した。OSへのtool自動導入や旧logicnnの編集はしていない。
- 利用者の最小チェック方針に合わせ、forwardの型／device重複検査、接続／重みの全再走査、毎回のIR検証、簡略化各passの重複検証を削減した。読み込み境界と意味が黙って変わる条件は維持する。

### 検証実績

| 検証 | 結果 |
|---|---|
| チェック削減後のアプリ＋core全体 | 4,776件合格、46件skip、94.20秒、対象行coverage97% |
| 参照モデル・追加C・レビュー修正後の最終全体 | **4,815件合格、47件skip、135.54秒**。対象4,539文中147文未実行、coverage97% |
| C追加境界を含む実MSVC検証 | **50件合格、56.45秒**。整数overflow、縮小castの最適化0〜3、NaN／±Inf／±0、Raw／Warp／LightモデルからJSON復元・実C評価 |
| 正式MNISTLGNの出力受入 | **1件合格、18.63秒**。seed0の初期重み、0／1／交互bitの3入力、JSON／C／Verilog生成、4,000生bitの値・順序、checkpoint往復 |
| 実MNISTのアプリ実行 | CPUで1 epoch完了。best/final保存、best再読込の公式test10,000件。test accuracy65.29%。[アプリの記録](../verification.md#実mnistでの動作確認) |

正式モデルの出力受入は環境変数 `LOGICNN_RUN_ACCEPTANCE=1` を明示して `tests/acceptance/test_mnist_export.py` を実行した。初期重みを用いた生成・Python IR比較であり、Cのnative実行やVerilogシミュレーション、学習後の精度を示すものではない。実MNIST試行も回路出力は無効にしている。これらを一つの学習済みモデルの全backend等価性検証として合算しない。

最終全体のskipには、別途合格した上記opt-in受入1件を含む。残りはCUDAとVerilatorの未実行。Ruff、examplesを含む構文／docstring／行長、独立sdist／wheel buildと隔離installが合格し、Docusaurus buildと両検証ページのHTTP 200も確認した。compiled評価の全IR比較を削除し、公開APIによる変更時の無効化と独立snapshotのテストへ置き換えた。

### レビューで修正した点と未検証範囲

型付き負scalarのC literalを符号付きとして復元した。またMSVC最適化時に整数縮小castの段階が消える反例があり、数値中間変数を `volatile` にして保存型の段階を保持した。独自の浮動小数点エンジンは追加せず、期待値・許容誤差も緩めていない。

この段階ではVerilatorが未検出だった。後続の追加検証では既存WSLのVerilatorを発見し、下記の実シミュレーションに合格した。CUDAは実機なしで未実行。AlkaidのAPIは公式sourceで確認できたが、固定小数点への変換範囲・精度の契約が不足しているため、[adapterのみ保留](./alkaid-decision.md)する。C12、C13の最終受入まで完了したとは扱わない。

## C12部分実装: 汎用の参照モデル

`utils/logicNN_core/examples/reference_models/` に `DenseReferenceModel` と `ConvolutionReferenceModel` を作成した。モデル構造は不変presetで表し、LUT方式と二値化moduleを呼出側から明示する。データ取得・学習ループ・旧クラス名の互換aliasは含めず、wheelにも含めない。

- DenseはMNIST、CIFAR-10、Fashion-MNIST、JSCの115構成、ConvはMNISTとCIFAR-10の10構成を整理した。任意のinput shape・幅・段列・クラス数で独自presetも作成できる。
- 小型の実forward／backward、前処理差替え、代表モデルの評価、全presetの構造値を開発テストへ追加した。大規模presetを全て確保・学習した実績ではない。
- 独立レビューでFashion-MNIST DWNのbit配置の違いを検出した。旧既定 `feature_dim=-2` の `(1,28,196)` とpixel→bit順へ修正し、CIFARの `feature_dim=1` と区別した。独立した列挙によるbit順の回帰テストを追加した。
- 旧側で未登録のFashion通常DLGNの `uniform`、MNIST DWNの `distributive`、JSC SmallRank6の幅10666／5クラス非整除は、元の構造値をREADMEへ残して保留した。別方式や丸めた幅へ無断で置換していない。

Alkaid adapterは引き続き保留しており、参照例の追加によってC12全体を完了としない。examplesもPython 3.10構文・関数docstring・160文字行長の開発検査へ含めた。

## 学習済みモデルのCとVerilog実検証

2026-09-13の追加確認。正式MNISTLGNの学習済みbestと公式testの固定65標本を使用し、初期重みの生成試験とは別に実施した。

| 検証 | 結果 |
|---|---|
| Python IR／JSONと通常C・packed 8／16／32／64 | **6件合格、280.93秒**。全スコアの最大絶対誤差0、予測クラス一致 |
| Verilogの小型回帰＋testbench | **61件合格、29.25秒** |
| 正式モデルの通常配線／inline Verilog | **2件合格、264.98秒**。65×4,000生bit完全一致 |

CはMSVC 19.44.35228 x64、Verilogは既存WSL Ubuntu-24.04内のVerilator 5.036を使用した。シミュレーションの期待値は変換前モデルの最終集約入力をhookで独立取得し、全bitの位置まで照合した。WSL makeのclock skew警告も保存し、新規実行ファイルの完走・一致と区別して記録している。

実行証跡は `log/verification_20260913_1130/` と `log/verification_20260913_1142/`。環境・実行方法は [Verilog実検証](./verilog-verification.md)、学習の受入と再現性の保留は [アプリ追加検証](../verification.md#2026-09-13-追加検証) を参照する。CUDA、Alkaid、未成立の参照モデル、100 epoch精度受入まで合格したことにはしない。

## 1時間作業の最終回帰

2026-09-13 12:25時点で **4,866件合格、53件skip、185.58秒、coverage97%（4,552文中150文未実行）**。WSL Verilatorの小型試験、実MSVC、独立package、追加診断の比較器、文書API例を含む。skipはCUDA40件とopt-in受入13件であり、別途実行したAT-009不合格を通常回帰の成功で上書きしない。

実compilerログの文字化けをconsole code pageの明示とMSVC環境のUTF-16LE読込へ修正した。通常forwardの検査は追加せず、回路の数値式も変更していない。小型28条件の性能基準を作成したが、CPU Tensorのnative peakは未測定である。

証跡は `log/final_verification_20260913_1221/`。未完・判断待ち・正式受入は [受入の現在地](../acceptance-status.md) を参照する。
