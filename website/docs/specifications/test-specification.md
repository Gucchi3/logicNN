---
sidebar_position: 4
title: テスト仕様
---

# logicNN テスト仕様書

## 1. 文書情報

- 文書状態: 変更草案（logicNN_core再設計を反映）
- 作成日: 2026-09-12
- 入力文書: `requirements.md`、`basic-design.md`、`detailed-design.md`
- 対象作業フォルダ: `logicNN/`
- 対象工程: 実装中の開発用テスト、実装完了後の結合確認および受入確認

本書は、logicNNの要件と設計をどのように検証するかを定義する。実装後の実測値や合否は本書へ混在させず、別途作成する最終検証結果へ記録する。

## 2. テスト方針

### 2.1 目的

1. 設定、データ、モデル、学習、保存、表示および回路出力が設計どおりに動くことを確認する。
2. モジュール境界とデータ型を固定し、整理後のコードが再び密結合にならないようにする。
3. エラーを学習開始前に検出し、利用者が原因と修正方法を判断できることを確認する。
4. 保存したbest/finalチェックポイントと成果物を、仕様どおり再利用・確認できることを保証する。
5. 現行 `logicnn` と比較して、MNISTのテスト精度低下が1パーセントポイント以内であることを確認する。
6. logicNN_coreの再設計による論理層、回路JSON、CおよびVerilogの退行を、専用テスト仕様と連携して検出する。

### 2.2 開発用テストと運用版の境界

- 自動テストコードは開発、変更、レビューおよび受入確認でだけ使用する。
- 通常の `python main.py` はテストコード、pytest、コンパイラおよびVerilogシミュレータをimportも実行もしない。
- logicNN側のテストは開発中だけ `tests/` に置く。受入承認後の移行工程で、運用版から `tests/`、pytest設定、テスト専用依存およびテスト専用データを削除する。
- logicNN_core専用テストも開発中だけ `tests/core/` に置き、運用版と配布wheelには含めない。専用テスト仕様と検証結果は残す。
- 設定、データ、モデル、チェックポイントおよび実行環境の入力検証は製品機能であり、運用版から削除しない。
- 本書と最終検証結果は `website/docs/specifications/` に残し、将来の変更時にテストを再作成できるようにする。

### 2.3 テスト対象外

- マルチGPU、分散学習およびクラウド実行。
- logicNN_coreが正式対応外と明示するhardware backendの検証。
- 生成された回路の論理合成品質、FPGA配置配線、消費電力およびタイミング収束。
- 異なるOS、PyTorch版、GPU機種またはCPU／GPU間の完全なビット一致。
- 通常運用中の定期的な自己診断や、学習のたびに開発用テストを自動実行する機能。

## 3. テストの分類

| 分類 | 目的 | 主な入力 | 実行頻度 |
|---|---|---|---|
| 静的・構造検査 | 構文、import、責務境界、docstringを確認する | ソースコード | 実装単位の完了時 |
| 単体テスト | 関数、データ型、登録表、保存形式を小さく確認する | 合成データ、mock、`tmp_path` | 実装中に随時 |
| 結合テスト | モジュール間の順序、引渡し、成果物を確認する | 小型fixture、fake MNIST | 機能単位の完了時 |
| システムテスト | `python main.py` の入口、終了コード、Rich表示を確認する | subprocess、テスト設定 | 全機能結合後 |
| 外部ツールテスト | 生成C／Verilogの構文と動作を確認する | Cコンパイラ、Verilator | 回路出力変更時、最終受入前 |
| 受入テスト | 実MNISTと実コマンドで要件を確認する | 取得済みMNIST、正式モデル | 実装完了後に一度 |
| 精度回帰比較 | 現行版からの精度低下を確認する | 現行版と新版の同条件実行 | 最終受入前 |

pytestのmarkerは次の4種類だけを使用する。

| marker | 意味 |
|---|---|
| `integration` | 複数のlogicNNモジュールを結合するテスト |
| `system` | CLIを別プロセスで起動するテスト |
| `external_tool` | CコンパイラまたはVerilatorを必要とするテスト |
| `acceptance` | 実MNISTまたは現行版との比較を行う長時間テスト |

markerなしのテストを高速な単体テストとする。理由のない `skip` や `xfail` は使用せず、CUDAや外部ツールの有無に依存するケースだけを条件付きskipにできる。最終受入では、受入に必要な環境を用意したうえで対象ケースのskipを認めない。

## 4. テストコードの構成

開発中は次の構成とする。

```text
tests/
├── conftest.py
├── fixtures/
│   ├── configs/
│   └── data.py
├── unit/
│   ├── test_config.py
│   ├── test_runtime.py
│   ├── test_data.py
│   ├── test_model.py
│   ├── test_optimization.py
│   ├── test_epoch.py
│   ├── test_results.py
│   └── test_export.py
├── integration/
│   ├── test_validation.py
│   ├── test_workflow.py
│   └── test_checkpoint_initialization.py
├── system/
│   └── test_cli.py
└── acceptance/
    ├── test_mnist_workflow.py
    ├── test_circuit_outputs.py
    └── test_accuracy_regression.py
```

テストファイルは製品モジュールの構成に対応させる。単なるファイル数合わせの分割は行わず、一つのテストは原則として一つの振る舞いと失敗理由だけを確認する。共有fixtureは `conftest.py` または `fixtures/` に置き、テスト間の実行順序や前テストの成果物へ依存させない。

## 5. 共通テストデータと環境

### 5.1 共通fixture

| fixture | 内容 | 用途 |
|---|---|---|
| `valid_config_json` | 詳細設計3.2に一致する最小の有効設定 | 設定検証の基準 |
| `fake_dataset_bundle` | 小さいTensorだけで構成した3分割のDataBundle | epoch、workflowの高速確認 |
| `fake_mnist` | TorchVision MNISTの外部I/Oだけを置換し、公式件数・形状・ラベルを再現するfixture | MNIST分割とDataLoader |
| `tiny_classification_model` | 入出力と更新を追跡しやすい小型 `nn.Module` | 学習順序、best判定、チェックポイント |
| `tiny_logic_model` | logicnn_coreの公開レイヤーだけで構成した小型論理モデル | 回路変換と等価性 |
| `checkpoint_factory` | スキーマの一項目だけを変更できるlogicNNチェックポイント生成器 | 正常・異常読込 |
| `run_dir` | pytestの `tmp_path` 配下に作る実行ディレクトリ | 成果物テスト |
| `rich_console_capture` | Rich Consoleの記録機能を使った出力取得 | 表示内容とtraceback有無 |

`fake_mnist` はネットワークへ接続せず、学習用60,000件とテスト用10,000件のインデックス、`[1, 28, 28]` の画像、`0`から`9`のラベルを遅延生成する。60,000枚の実画像をメモリへ保持しない。

### 5.2 ファイルと外部状態

- ファイルを作るテストは原則としてpytestの `tmp_path` を使用し、利用者の `data/` と `log/` を変更しない。
- 単体・結合テストはネットワークアクセスを禁止し、データ不足時に自動ダウンロードしないことも検証する。
- 日時、CUDA可否、logicNN_core境界およびOSコマンドは、分岐を検証するときだけ境界でmockする。
- PyTorchの計算そのもの、state dictの保存・復元およびTensorの型・形状は過度にmockせず、実オブジェクトで確認する。
- 受入テストだけが、事前に `tools/download_dataset.py` で取得した実MNISTを読み込む。

### 5.3 基準環境

必須基準は単一CPU環境とする。利用可能な場合は単一CUDA GPUで追加のsmoke testを行うが、CPUとGPUの完全一致は求めない。最終検証結果には次を記録する。

- OS、Python、PyTorch、TorchVisionおよびlogicNN_core版情報
- CPU名、GPUを使った場合はGPU名とCUDA情報
- 実行コマンド、設定ファイル、seed、実行デバイス
- CコンパイラとVerilatorの識別情報
- 各テストの成功、失敗、skip、実行時間

## 6. 静的・構造検査

| ID | 検証内容 | 合格条件 |
|---|---|---|
| QA-001 | Ruffで製品コードと開発用テストを検査する | 構文、未使用import、未定義名、import順の違反が0件 |
| QA-002 | すべてのPythonモジュールをimportする | import時に学習、ダウンロード、成果物作成を開始しない |
| QA-003 | ASTで製品関数・メソッドのdocstringを検査する | すべての関数定義直下に空でない簡潔なdocstringがある |
| QA-004 | `main.py` とモジュール依存を検査する | 基本設計5章の禁止依存がなく、`main.py` に学習ループやデータ処理がない |
| QA-005 | core境界を検索する | logicNN側は `model/` と `utils/export/circuit_exporter.py` からlogicnn_coreの公開APIだけを使用し、coreからlogicNN本体への逆依存がない |
| QA-006 | Docusaurusをproduction buildする | broken link、Markdown、MermaidおよびTypeScriptエラーが0件 |
| QA-007 | 運用版の内容を検査する | 移行後にテストコード、pytest依存、テスト設定、キャッシュが含まれず、文書は残っている |

コードカバレッジはlogicNN側の未検証箇所を見つける補助指標として記録するが、単純な行カバレッジ率だけを受入条件にはしない。logicNN_coreは専用テスト仕様で別集計し、Docusaurus生成物およびテストコード自身は対象外とする。

## 7. 単体テスト

### 7.1 設定と実行環境

| ID | 対象 | 検証内容 | 合格条件 |
|---|---|---|---|
| UT-CFG-001 | `load_config()` | 詳細設計3.2のJSONを読み込む | 全値が厳格な不変Pydantic型になり、パスがプロジェクトルート基準で解決される |
| UT-CFG-002 | 必須項目 | 各必須項目を一つずつ欠落させる | 設定パスを含む `LogicNNError` になる |
| UT-CFG-003 | 未知キー | 最上位と各区分へ未知キーを加える | すべて拒否され、該当パスが列挙される |
| UT-CFG-004 | strict型 | 文字列数値、integerのboolean、booleanのintegerを与える | 暗黙変換せず拒否する |
| UT-CFG-005 | 値域 | seed、件数、率、epochの境界値と境界外をparameterizeする | 詳細設計3.4の範囲だけを受理する |
| UT-CFG-006 | 項目間制約 | `minimum_learning_rate > learning_rate` にする | 両方の設定パスと制約を示して拒否する |
| UT-CFG-007 | 廃止キー | `validation_size`、`model.parameters`、`load_checkpoint`、個別回路スイッチを与える | 未知キーとして拒否する |
| UT-CFG-008 | JSON構文 | 壊れたJSONを読み込む | ファイルと構文位置を含む `LogicNNError` になる |
| UT-CFG-009 | 不変性 | 読込後の設定を書き換える | Pydanticが変更を拒否する |
| UT-RUN-001 | device `auto` | CUDA可否をtrue／falseにする | trueで `cuda`、falseで `cpu` を一つだけ選ぶ |
| UT-RUN-002 | device明示 | `cpu` と利用可能な `cuda` を指定する | 指定どおりのdeviceを返す |
| UT-RUN-003 | CUDA不足 | CUDAなしで `cuda` を指定する | CPUへ切り替えず `LogicNNError` になる |
| UT-RUN-004 | seed | Python、NumPy、PyTorchへ同じseedを二度設定する | 同一環境で同じ乱数列を再現する |
| UT-RUN-005 | CUDA seed | CUDA利用可能時にseedを設定する | CUDAのseed設定も一度呼ばれる |

### 7.2 データ

| ID | 対象 | 検証内容 | 合格条件 |
|---|---|---|---|
| UT-DATA-001 | 二値化関数 | 負値、0、正値を入力する | `pixel > 0.0`だけがfloat32の1.0になり、他は0.0になる |
| UT-DATA-002 | 二値化形状 | 複数のTensor形状を入力する | 形状とdeviceを維持し、入力Tensorを破壊しない |
| UT-DATA-003 | 共通型 | `DatasetMetadata.input_size` を参照する | `input_shape` の積を返し、独立値として変更できない |
| UT-DATA-004 | MNIST分割 | fake MNISTをseed付きで分割する | 50,000／10,000／10,000件になり、3集合が重複しない |
| UT-DATA-005 | 分割再現 | 同一seedと異なるseedで分割する | 同一seedでは同じ分割、異なるseedでは検証集合が変わる |
| UT-DATA-006 | 前処理統一 | 学習・検証・テストの標本を取得する | すべて同じ二値化とdtypeを使用する |
| UT-DATA-007 | DataLoader設定 | 3つのloaderを調べる | batch size、worker、shuffle、drop last、pin memoryが詳細設計10.4と一致する |
| UT-DATA-008 | 評価全件 | 端数を含む件数で検証・テストloaderを反復する | `drop_last=False`で全標本を一度ずつ返す |
| UT-DATA-009 | データ不足 | MNIST未取得のrootを指定する | ダウンロードせず、`tools/download_dataset.py` の案内を含む `LogicNNError` になる |
| UT-DATA-010 | 登録表 | 未登録名とテスト用登録名を選ぶ | 未登録名は候補付きエラー、追加した専用関数は共通DataBundleとして返る |
| UT-DATA-011 | 汎用前処理メタデータ | MNIST以外を想定したparametersを保持する | JSON互換の任意parametersを保持でき、閾値フィールドを必須としない |

### 7.3 モデルと最適化方式

| ID | 対象 | 検証内容 | 合格条件 |
|---|---|---|---|
| UT-MDL-001 | model builder | `mnist_lgn` を指定する | 引数なしの新しいMNISTLGNを毎回返す |
| UT-MDL-002 | 未登録モデル | 存在しない名前を指定する | 登録済み候補を含む `LogicNNError` になる |
| UT-MDL-003 | モデル契約 | 共通公開情報を確認する | `input_shape=(1,28,28)`、`input_size=784`、`num_classes=10`が得られる |
| UT-MDL-004 | 構造情報 | `architecture_info()` を確認する | 詳細設計11.1の全定数をJSON互換値で返す |
| UT-MDL-005 | 層構造 | MNISTLGNのモジュールを順に確認する | Conv、OR Pool、Flatten、Dense、Dense、GroupSumの順で、二値化層を含まない |
| UT-MDL-006 | forward | 複数batch sizeの二値画像を入力する | `[batch, 10]` の有限なクラススコアを返す |
| UT-MDL-007 | 学習モード | `train()` で損失を逆伝播する | 学習可能パラメータへ有限で0ではない勾配が流れる |
| UT-MDL-008 | 評価モード | `eval()` で同じ二値入力を複数回評価する | 離散論理演算を使い、同一の出力を返す |
| UT-MDL-009 | 適合性 | modelとdatasetの形状またはクラス数を変える | workflowの事前検証が不一致項目を示して拒否する |
| UT-OPT-001 | loss builder | `cross_entropy` を選択する | 学習用だけ設定済みlabel smoothing、評価用は0.0になる |
| UT-OPT-002 | optimizer builder | `adamw` を選択する | learning rate、weight decayとPyTorch既定のbetas、epsを使用する |
| UT-OPT-003 | scheduler builder | `cosine_annealing` を選択する | `T_max=epochs`、`eta_min=minimum_learning_rate`になる |
| UT-OPT-004 | 未登録方式 | 未登録のloss、optimizer、scheduler名を一つずつ指定する | 種類と登録済み候補を含む `LogicNNError` になる |

### 7.4 epoch処理

| ID | 対象 | 検証内容 | 合格条件 |
|---|---|---|---|
| UT-EPOCH-001 | `train_one_epoch()` | 端数最終batchを含むデータで学習する | `model.train()`を使い、全標本を集計して重みを更新する |
| UT-EPOCH-002 | 学習指標 | batchごとに件数と損失を変える | batch平均の単純平均ではなく全標本の重み付き平均になる |
| UT-EPOCH-003 | 学習精度 | 予測とラベルが既知のデータを使う | `correct / samples` を0.0〜1.0で返す |
| UT-EPOCH-004 | 学習空データ | 空loaderを渡す | 0除算せず `LogicNNError` になる |
| UT-EPOCH-005 | `evaluate()` | 検証用データを評価する | `model.eval()`、`torch.no_grad()`を使い、重みと勾配を変更しない |
| UT-EPOCH-006 | 評価指標 | 端数最終batchを含める | 全標本の重み付き平均と正確なsamplesを返す |
| UT-EPOCH-007 | 評価空データ | 空loaderを渡す | `LogicNNError` になる |
| UT-EPOCH-008 | 責務境界 | 両関数の呼出しを追跡する | scheduler更新、成果物保存、Rich表示を関数内部で行わない |

### 7.5 成果物とチェックポイント

| ID | 対象 | 検証内容 | 合格条件 |
|---|---|---|---|
| UT-RES-001 | run directory | 固定日時で一つ目を作る | `YYYYMMDD_HHMMSS` のディレクトリを作る |
| UT-RES-002 | 名前重複 | 同じ秒に3回作る | 既存内容を上書きせず `_01`、`_02` を付ける |
| UT-RES-003 | `config.json` | 解決済み設定を保存する | 入力と同じ構造で絶対パスを含む有効JSONになる |
| UT-RES-004 | `training_info.json` | 実行情報を保存する | 詳細設計7.3の全区分があり、ユーザー名とホスト名を含まない |
| UT-RES-005 | `metrics.jsonl` | 2epoch分を追記する | 1epochが1行で固定キー順になり、既存行を失わない |
| UT-RES-006 | `curves.png` | 1件と複数件の履歴から保存する | PNGが更新され、2列、軸名、凡例、グリッドを持つ |
| UT-RES-007 | `test_metrics.json` | bestのテスト指標を保存する | checkpoint名、epoch、loss、accuracy、samplesが仕様型で保存される |
| UT-CHK-001 | 共通スキーマ | bestとfinalを保存する | 詳細設計6.1の同じキー構造で、種別と値だけが異なる |
| UT-CHK-002 | CPU保存 | CPU上のstate dict、およびCUDA利用可能時はCUDA上のstate dictを保存する | 保存対象TensorがすべてCPUへコピーされ、元モデルを変更しない |
| UT-CHK-003 | round trip | 保存直後にCPUへ読み込む | state dict、モデル、データ、最適化、指標を欠落なく復元する |
| UT-CHK-004 | 形式異常 | 必須キー、型、形式名、形式版を一つずつ壊す | `load_checkpoint()` がパスと不正項目を示して拒否する |
| UT-CHK-005 | model不一致 | 登録名、形状、サイズ、クラス数、構造を一つずつ変える | workflowが期待値と実際値を示して学習前に拒否する |
| UT-CHK-006 | dataset不一致 | 名前、形状、クラス数、前処理を一つずつ変える | workflowが不一致項目を示して学習前に拒否する |
| UT-CHK-007 | state dict不一致 | 重みキーの欠落、追加、形状変更を行う | `strict=True` 適用時に学習前エラーになる |
| UT-CHK-008 | 旧形式 | 生state dictまたは現行logicnn形式を渡す | フォールバックせずlogicNN形式エラーになる |

### 7.6 回路出力

| ID | 対象 | 検証内容 | 合格条件 |
|---|---|---|---|
| UT-CIR-001 | 無効化 | `circuit.enabled=false` でexporterを呼ぶ状況を作る | workflowがexporterを呼ばず、`circuit/`も作らない |
| UT-CIR-002 | モデル保護 | export前後の元モデルを比較する | device、training mode、state dictを一切変更しない |
| UT-CIR-003 | API順序 | logicNN_core境界をspyする | deepcopy、CPU、eval、`from_model()`、`simplify()`、JSON、C、Verilogの順に一度ずつ実行する |
| UT-CIR-004 | 保存内容 | 3つのAPIから既知の内容を返す | `circuit/`の固定ファイル名へ加工せず保存する |
| UT-CIR-005 | 部分失敗 | C生成後にVerilog生成を失敗させる | `LogicNNError` になり、先に保存したJSONとCを残す |
| UT-CIR-006 | 未対応能力 | modelにcore未対応の演算を含める | node、演算、対応能力および修正方法を示し、自動フォールバックしない |
| UT-CIR-007 | Python等価性 | 小型論理モデルを回路JSONへ変換して再読込する | 固定入力とseed付き入力で評価モードの出力が一致する |
| UT-CIR-008 | MNISTLGN変換 | 正式モデルを初期重みのまま変換する | 3形式を生成でき、各ファイルが空でない |
| UT-CIR-009 | Verilog出力境界 | GroupSumを含む小型modelを変換する | Verilogに加算・tau・biasを生成せず、GroupSum入力を生出力にする |

## 8. 結合・システムテスト

### 8.1 workflow結合

| ID | 検証内容 | 合格条件 |
|---|---|---|
| IT-WF-001 | 実行前検証の呼出順を追跡する | 詳細設計9.4の順番になり、学習開始前に必要情報が揃う |
| IT-WF-002 | 各事前検証を一つずつ失敗させる | optimizer stepを一度も呼ばず、作成済み調査用成果物だけを残す |
| IT-WF-003 | 3epochの小型学習を完了する | 学習・検証が各3回、schedulerが各epoch末に3回実行される |
| IT-WF-004 | 検証精度を `0.7, 0.7, 0.6` に固定する | bestは先のepoch 1、finalはepoch 3になる |
| IT-WF-005 | 学習率と呼出順を記録する | metricsとRichにはscheduler更新前の学習率が記録される |
| IT-WF-006 | 学習終了後の処理を追跡する | final保存後に保存済みbestを読み、公式テストを一度だけ行う |
| IT-WF-007 | 回路出力を有効にする | テストに使った同じbest状態だけをexporterへ渡す |
| IT-WF-008 | 回路出力を無効にする | 通常成果物だけが存在し、回路APIを呼ばない |
| IT-WF-009 | bestチェックポイントを初期値にする | 重みだけを適用し、epoch 1、空履歴、新しいoptimizerとschedulerから開始する |
| IT-WF-010 | finalチェックポイントを初期値にする | bestの場合と同じ新規学習規則で完了する |
| IT-WF-011 | 追加データセットを登録表へ一時追加する | workflowとepochコードを変更せず、対応モデルとの学習を完了する |
| IT-WF-012 | 回路出力だけを失敗させる | 学習、best/final、履歴、グラフ、テスト結果を残し、全体は非0終了になる |
| IT-WF-013 | `TrainingResult` を確認する | run_dir、best epoch、best検証精度、test metricsが保存内容と一致する |
| IT-WF-014 | Richイベントを記録する | 開始条件、各epoch、最良epoch、テスト結果、保存先が重複なく表示される |

### 8.2 CLIと終了コード

| ID | 検証内容 | 合格条件 |
|---|---|---|
| ST-CLI-001 | `python main.py --help` | 使用方法と `--config` を表示して終了コード0になる |
| ST-CLI-002 | `python main.py` | `config/mnist_lgn.json` を選択してworkflowを一度開始する |
| ST-CLI-003 | `python main.py --config <path>` | 指定した設定だけを使用する |
| ST-CLI-004 | 未知のCLI引数 | argparseが使用方法を示して終了コード2になる |
| ST-CLI-005 | 想定済みエラー | Richでmessage、detail、hintを表示し、tracebackなしで終了コード1になる |
| ST-CLI-006 | Ctrl+C | 短い中断表示を行い、終了コード130になる |
| ST-CLI-007 | 想定外例外 | tracebackを隠さず、0以外で終了する |
| ST-CLI-008 | 別プロセスの通常実行で読み込まれたmoduleを確認する | `tests`、pytest、コンパイラ連携およびVerilator連携を読み込まない |

## 9. 外部ツールによる回路検証

### 9.1 共通条件

回路検証は開発時だけ行い、通常の学習には組み込まない。小型論理モデルには全入力または十分小さい固定入力集合を使用し、MNISTLGNにはseedを記録した実テスト画像の固定部分集合を使用する。変換元モデルはCPU・評価モードとし、すべての比較で同じ入力を使う。

### 9.2 Cコード

| ID | 検証内容 | 合格条件 |
|---|---|---|
| EXT-C-001 | `circuit.c` を利用可能なCコンパイラでコンパイルする | warningの有無を記録し、error 0件で実行ファイルを生成できる |
| EXT-C-002 | 小型回路の全テストベクトルをC実装へ与える | logicNN_coreのPython回路と全出力が一致する |
| EXT-C-003 | MNIST固定標本をC実装へ与える | 出力表現のscaleを考慮したクラス順位が変換元モデルと一致する |

コンパイラ名や版は固定せず、実際に使用した実行ファイルと版情報を最終検証結果へ記録する。コンパイラがない開発環境では通常テストからskipできるが、最終受入結果を確定する環境では未実施のまま合格にしない。

### 9.3 Verilogコード

| ID | 検証内容 | 合格条件 |
|---|---|---|
| EXT-V-001 | `circuit.v` をVerilatorでlint・コンパイルする | 構文error 0件でシミュレーション可能になる |
| EXT-V-002 | 小型回路の全テストベクトルをtestbenchから与える | Verilogの生出力と `Circuit.evaluate_logical_outputs()` が全件一致する |
| EXT-V-003 | MNIST固定標本をtestbenchから与える | Verilogの4,000 bit生出力と `Circuit.evaluate_logical_outputs()` が一致する |
| EXT-V-004 | 生成sourceの構造を確認する | GroupSumのadder、tau除算、bias加算、score scaleおよびクラス順位回路が存在しない |
| EXT-V-005 | 定数・重複出力を持つ小型modelを簡略化し、保存JSONを再読込してVerilog生成する | 元modelの集約前Tensorと出力数・順序・値が一致する。Python IRだけでなく独立した期待値と比較する |

Verilatorを直接使えないWindows環境では、明示的に選択したWSL環境を使用できる。使用した環境とコマンドを結果へ記録する。Verilatorがない通常開発環境ではskipできるが、最終受入結果を確定する環境では未実施のまま合格にしない。

## 10. 受入テスト

### 10.1 事前条件

1. `python tools/download_dataset.py --dataset mnist --root data` が正常終了している。
2. 旧版 `logicnn_old/` と正式版 `logicNN/` の両方を比較できる。
3. 同じMNISTファイル、seed、デバイス、依存関係および実行環境を使用できる。
4. CコンパイラとVerilatorを利用できる。
5. 受入用設定は通常設定を勝手に上書きせず、開発用 `tests/acceptance/configs/` に置く。

短時間の機能確認にはepochを1にした受入用smoke設定を使う。精度比較には正式な `config/mnist_lgn.json` と意味的に同じ条件を使い、回路出力だけは比較時間へ影響させないため両方で無効にする。使用した設定JSONは最終検証結果へ添付または全文記録する。

### 10.2 受入ケース

| ID | 対応受入条件 | 操作 | 合格条件 |
|---|---|---|---|
| AT-001 | AC-001 | 実MNISTとsmoke設定で `python main.py` を実行する | 全epoch、毎epoch検証、best再読込、公式テスト、必須成果物を正常完了する |
| AT-002 | AC-002 | AT-001の `model_best.pth` を初期重みにして再実行する | epoch 1から新しい学習として正常完了する |
| AT-003 | AC-002 | AT-001の `model_final.pth` を初期重みにして再実行する | epoch 1から新しい学習として正常完了する |
| AT-004 | AC-003 | 回路出力を有効にして実行する | bestから完全なJSON／Cと最終集約を除くVerilogを生成し、EXT-C／EXT-Vに合格する |
| AT-005 | AC-003 | 回路出力を無効にして実行する | `circuit/`と回路成果物を生成しない |
| AT-006 | AC-004 | 設定、登録名、データ、モデル、チェックポイントの代表的異常をCLIから与える | すべて学習開始前に原因と対応を示して終了コード1になる |
| AT-007 | AC-005 | 10.3の手順で現行版と新版を実行する | 新版の精度低下が1パーセントポイント以内になる |
| AT-008 | AC-006 | 文書サイトと運用版候補を検査する | 仕様・結果文書が閲覧でき、運用版候補に開発用テストがない |
| AT-009 | NFR-003 | 同一条件の新版学習を二度実行する | 分割、データ順、epoch記録、best epoch、重みおよびテスト指標が同一環境で一致する |
| AT-010 | NFR-004・005 | 正常実行と異常実行のRich表示を確認する | 利用者が条件、進捗、結果、保存先または修正方法を読み取れる |

## 11. 現行版との精度比較

### 11.1 比較条件

- 比較対象は移行前の旧版を保存した `logicnn_old/` と正式版 `logicNN/` とする。
- 同一のMNISTファイル、学習50,000件、検証10,000件、公式テスト10,000件を使う。
- seed、実デバイス、batch size、epoch数、loss、label smoothing、optimizer、learning rate、weight decay、schedulerおよびminimum learning rateを一致させる。
- モデル定数は、Conv kernel数、tree depth、receptive field、pooling、hidden size、クラス数およびGroupSum tauを一致させる。
- 現行版のモデル内二値化と新版のデータ前処理が、モデルの最初の論理層へ同じTensorを渡すことを事前に確認する。
- 両方の回路出力を無効にし、精度に関係しない後処理時間を比較から除外する。
- 先に現行版、次に新版を一回ずつ実行し、途中で環境、データまたは設定を変更しない。

### 11.2 記録値

次を最終検証結果へ並べて記録する。

| 項目 | 現行版 | 新版 |
|---|---:|---:|
| best epoch | 記録値 | 記録値 |
| best validation accuracy | 0.0〜1.0 | 0.0〜1.0 |
| official test loss | 記録値 | 記録値 |
| official test accuracy | 0.0〜1.0 | 0.0〜1.0 |
| elapsed time | 参考値 | 参考値 |

精度低下は次式で判定する。

```text
accuracy_drop = legacy_test_accuracy - new_test_accuracy
PASS if accuracy_drop <= 0.01
```

新版の精度が現行版より高い場合、`accuracy_drop` は負値となり合格である。丸めた百分率ではなく、`test_metrics.json` の0.0〜1.0の値で計算する。失敗した場合はseedを変えて合格値だけを選ばず、入力Tensor、分割、モデル定数、初期state dict、学習率系列および評価モードを順に比較して原因を特定する。

## 12. 実行手順

実装時に `pyproject.toml` の開発用追加依存へpytest、pytest-covおよびRuffを定義する。通常依存には含めない。

### 12.1 開発中の高速確認

```text
uv sync --extra dev
uv run ruff check .
uv run pytest -m "not integration and not system and not external_tool and not acceptance"
```

### 12.2 logicNN内の全自動テスト

```text
uv run pytest -m "not external_tool and not acceptance"
```

### 12.3 外部ツールテスト

```text
uv run pytest -m external_tool
```

### 12.4 受入テスト

```text
uv run pytest -m acceptance
```

### 12.5 文書サイト

```text
cd website
npm run build
```

個別テスト、全テスト、外部ツールおよび受入テストは明示的に起動し、`python main.py` からは呼び出さない。失敗後の再実行では、失敗したIDだけでなく、そのIDが属するテスト分類全体を最終的に再実行する。

## 13. 合否判定と中止条件

### 13.1 実装単位の完了条件

1. 対応する単体テストがすべて成功する。
2. 既存の高速テストに退行がない。
3. Ruffと構造検査に違反がない。
4. 仕様変更が発生した場合は、コードより先に要件または設計と本書を更新する。

### 13.2 最終受入の合格条件

1. QA、UT、IT、ST、EXTおよびATの必須ケースがすべて成功する。
2. `xfail` が0件である。
3. 必須環境不足によるskipが0件である。
4. AC-005の精度差が1パーセントポイント以内である。
5. Docusaurusのproduction buildが成功する。
6. すべての結果、環境、設定、未解決事項が最終検証結果へ記録される。
7. ユーザーが結果を確認し、運用版への移行を承認する。

### 13.3 中止条件

次の場合は後続の長時間テストを中止し、先に原因を修正する。

- 設定、データまたはチェックポイントの不整合を検出できず、学習が始まった場合。
- 学習または評価でNaN／Infinityが発生した場合。
- bestまたはfinalチェックポイントを再読込できない場合。
- 公式テストがモデル選択や学習に使用された場合。
- 生成回路と評価モードのモデルで出力またはクラス順位が不一致になった場合。
- テストが利用者の既存データや成果物を上書きした場合。

## 14. 要件トレーサビリティ

| 要件 | 主なテスト |
|---|---|
| FR-RUN-001〜005 | UT-CFG、UT-RUN、ST-CLI、AT-001、AT-009 |
| FR-DATA-001〜006 | UT-DATA、IT-WF-001・002・011、AT-001・006 |
| FR-MODEL-001〜005 | UT-MDL、IT-WF-007、UT-CIR-007・008 |
| FR-TRAIN-001〜004 | UT-OPT、UT-EPOCH、IT-WF-003〜005、AT-001 |
| FR-WEIGHT-001〜004 | UT-CHK、IT-WF-009・010、AT-002・003・006 |
| FR-EVAL-001〜003 | IT-WF-004・006・007、AT-001 |
| FR-OUT-001〜003 | UT-RES、UT-CHK、IT-WF-012・013、AT-001 |
| FR-CIRCUIT-001〜004 | UT-CIR、IT-WF-007・008・012、EXT-C、EXT-V、AT-004・005 |
| FR-UI-001〜002 | IT-WF-014、ST-CLI、AT-010 |
| NFR-001 保守性 | QA-003〜005、テストと製品の構成分離 |
| NFR-002 拡張性 | UT-DATA-010・011、IT-WF-011、UT-OPT-004 |
| NFR-003 再現性 | UT-RUN-004・005、UT-DATA-005、AT-009 |
| NFR-004 可観測性 | UT-RES-003〜007、IT-WF-014、AT-010 |
| NFR-005 エラーの明確性 | UT-CFG、UT-DATA-009・010、UT-CHK-004〜008、ST-CLI-005、AT-006 |
| NFR-006 依存関係 | QA-005・007、UT-CIR-006、EXT-C、EXT-V |
| NFR-007 文書サイト | QA-006、AT-008 |
| AC-001〜006 | AT-001〜010 |

各テスト実装のdocstringには本書のテストIDを記載する。最終検証結果では、テストIDごとに結果を追跡できる形で記録し、仕様上の必須ケースを無言で削除しない。

## 15. 実装への引継ぎ

本書の承認後、機能を次の単位に分けて実装・レビューする。

1. プロジェクト基盤、例外、設定およびruntime
2. データ共通型、前処理、MNISTおよび取得ツール
3. logicNN_core専用仕様の承認と実装単位C1〜C12
4. model builderとlogicnn_coreを使用するMNISTLGN
5. loss、optimizer、schedulerおよびepoch処理
6. 実行ディレクトリ、メタデータ、指標、グラフおよびチェックポイント
7. workflow、Rich表示およびCLI
8. logicNN側のcircuit exporter
9. 結合、システム、外部ツール、受入および精度回帰確認

logicNN_coreの個別ケースと実装順序は [coreテスト仕様](./logicnn-core/test-specification.md) を正本とする。各単位では対応するテストを実装し、成功を確認してから次の単位へ進む。全実装と受入の完了後に最終検証結果を作成し、ユーザー承認後の移行工程で開発用テストを運用版から削除する。
