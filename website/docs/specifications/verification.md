---
sidebar_position: 5
title: 実装・検証記録
---

# logicNN 実装・検証記録

## 現在地

2026-09-13時点。設定・データ・モデル・最適化・epoch処理・成果物・全体workflow・Rich／CLI・回路出力の実装を接続した。小型データによる実学習、best／finalの実保存・再読込、最終評価、回路有効／無効、異常時の成果物保持を確認した。

実MNIST全件を使用したCPUの1 epoch学習も完了した。さらに学習済みbestのPython／C比較とVerilog実シミュレーションに合格した。**受入全体の完了ではない。** 同条件学習の重み再現性、100 epochの旧版精度比較、coreの参照モデル／Alkaid、最終移行は未完である。旧版の廃止や開発テストの削除は行っていない。最新結果は [追加検証](#2026-09-13-追加検証) を参照する。

## チェック機構の最小化

利用者の承認により、検査ON/OFFスイッチは追加せず、次の重複検査を削除した。

- 各forwardのTensor型・layout・device検査、固定接続と全重みの再走査。
- Residualで子層が扱う入力の再検査。合流のshape一致とexport boolだけを維持。
- 構築済みCircuitを評価するたびの全IR検証と全数値演算の試算。
- compiled評価ごとの全IR比較。公開APIの簡略化成功時にbackendを無効化する仕組みを使用する。
- 簡略化の各passごとの検証。構造検証を開始と完了の2回へ集約。
- 保存する実行metadataの全schema再検証と、曲線描画で過去の全履歴を毎回検査する処理。

設定・外部file読込・回路生成／C ABIの境界、黙ったbroadcastや数値変換を防ぐ最小条件、非有限損失・勾配からの更新保護は維持した。正常な数値式と勾配は変更していない。

## 実行済み検証

| 検証 | 結果 |
|---|---|
| 12:25時点の最終全体pytest | **4,866件合格、53件skip、185.58秒** |
| 対象コードの行coverage | **97%**（model、utils、coreと参照例の4,552文中150文未実行） |
| Ruff | 合格 |
| Python 3.10構文・関数docstring・core行長 | 全体テスト内で合格 |
| 独立package build／install | 全体テスト内で合格 |
| Docusaurus build | 成功 |
| 実C compiler | 既存のnative MSVCを使用。全GateOp、packing、整数／小数計算を含む |

skipはCUDA実機が必要な40件と、明示opt-inの受入13件である。受入は別途実行し、12件の合格とAT-009の1件不合格を下記の記録で区別する。最終全体実行ではWSL Verilatorを有効にしており、小型Verilog試験も実行済みである。skipを合格や機能完成に数えない。テスト件数の過去記録はその時点の範囲であり、新旧の件数差から削除機能数を推測しない。

全体のJUnitとcoverage JSONは `log/final_verification_20260913_1221/`。Python153ファイルの3.10構文・関数docstring・160文字行長、Ruff、Docusaurus build／typecheckも確認した。

```shell
.venv/Scripts/python.exe -m pytest -x -q --cov=logicnn_core --cov=model --cov=utils --cov-report=term-missing
.venv/Scripts/python.exe -m ruff check .
```

## 実MNISTでの動作確認

`tools/download_dataset.py --dataset mnist --root data` で取得後、次を実行して終了コード0を確認した。通常の100 epoch設定は変更していない。

```shell
.venv/Scripts/python.exe -X utf8 main.py --config tests/acceptance/configs/mnist_smoke.json
```

- 実行日: 2026-09-13。保存先: `log/20260913_021753/`。
- CPU、seed 0、batch 128、1 epoch、回路出力は無効。
- 学習50,000件、検証10,000件、公式test 10,000件を全て使用した。
- train loss 1.0533008、accuracy 69.388%。validation loss 1.1362705、accuracy 65.57%。
- 保存したbestを再読込したtest loss 1.1277153、accuracy **65.29%**。
- `model_best.pth` と `model_final.pth`、設定、実行情報、epoch記録、曲線、test記録を生成した。1 epochのためbestとfinalは同じepochだが、別の成果物として保存している。
- 1点だけの曲線が見えない表示不具合を修正し、点markerを付けた。保存済み `metrics.jsonl` を変更せず曲線だけ再生成し、画像の表示も確認した。

この試行は学習・保存・再読込・公式testまでの動作確認であり、目標精度達成や旧版と同等の精度を示すものではない。回路出力を無効にしたため、この実行だけで学習済みモデルのC／Verilog等価性を合格としない。

## 追加検証前の予定

以下は1 epochの最初の動作確認が終わった時点の予定である。実施後の結果は直後の「追加検証」と [受入の現在地](./acceptance-status.md) に整理した。

1. 学習済み正式MNISTLGNのC実行と元のPyTorch結果の比較。初期重みのJSON／C／Verilog生成と4,000生bitの受入1件は別途合格した（18.63秒、実C／Verilog実行は含まない）。
2. Verilator、Alkaid、旧版精度比較、性能基準など、まだ実施できていない範囲を解消する。参照モデルは成立する125構成を整理し、不成立の系列は情報を残して保留した。
3. 最終受入後にサイトの案内・API説明を整理し、運用配布から開発者用テストを除外する。現在の開発用テストを先に削除しない。

## 2026-09-13 追加検証

### 学習済み正式モデルの3方式比較

`log/20260913_021753/model_best.pth` と、seed 0で選んだ公式testの固定65標本を使用した。モデルの最終集約直前をhookで観測した4,000 bitを独立期待値とし、回路自身から期待値を作っていない。

1. Python IR／JSON、通常C、packed Cの8／16／32／64 bit: **6件合格、280.93秒**。全スコアの最大絶対誤差0、予測クラス一致。記録は `log/verification_20260913_1130/`。
2. 既存WSLのVerilator 5.036を使用した通常配線／inline: **2件合格、264.98秒**。65標本×4,000 bitの値と位置がすべて一致。記録は `log/verification_20260913_1142/`。
3. Verilogの小型回帰とtestbench: **61件合格、29.25秒**。正式モデルとは分けて記録する。

追加で、best初期値による実CLIが作成したrun `log/training_acceptance_20260913_1159/best/log/20260913_115839/` を使用した。CLI生成済み3ファイルと検証側の再生成物のbyte一致を確認し、そのPython／実C／通常配線Verilogも **3件合格、291.43秒**。全スコアと全生bitが一致した。記録は `log/verification_20260913_1213_cli/` の `cli_artifacts_verified=true` と各結果JSONに残した。

実C compilerはMSVC 19.44.35228 x64。WSLのmakeにはclock skew警告があったためログを残した。新規の実行ファイルによる比較には合格したが、警告がなかったとは扱わない。追加toolのinstallやOS時計の変更は行っていない。

### 同条件学習の再現性: AT-009を局所保留

CPU、seed 0、batch 128、1 epoch、全学習50,000標本の同じ設定で、実際の `main.py` を別processで再実行した。実行自体は正常終了したが、厳密比較には不合格となった。

| 比較対象 | 結果 |
|---|---|
| train accuracy、validation loss／accuracy、test loss／accuracy | 一致 |
| train loss | 元 `1.0533007766723632`、再実行 `1.053300771627426` |
| 最終重み | 一部不一致。最大絶対差は約 `0.001893699` |
| 接続buffer | 一致 |

元runは `log/20260913_021753/`、再実行は `log/training_acceptance_20260913_1148/repeat/log/20260913_114744/`。差を許容するようにテストを変更したり、seedを変えて合格例を選んだりしていない。

詳細設計9.5では決定論的アルゴリズムを強制しない一方、AT-009は同条件の重み一致を要求している。この整合性が判断事項である。まず小型の別process診断で非決定性の発生箇所を調べる。本体の演算方式・スレッド数・許容差を無断で変更せず、この受入だけを保留し、他の検証を続ける。過去runはthread数・決定性状態を記録していないため、同一実行条件であったことも断定しない。

小型診断は新規初期値の先頭3 batch、checkpoint初期値の先頭8 batchをそれぞれ実行した。通常10 threads、1 thread、10 threads＋deterministic algorithmsの3条件を各2 processで比較し、観測した初期state・入力・forward・勾配・更新後重みは各条件内で一致した。記録は `log/cpu_repro_diagnosis_20260913_1211/` と `log/cpu_repro_diagnosis_20260913_1218_initialized/`。

さらにcheckpoint初期値・通常10 threadsで先頭32 batchを2 process比較したが、この範囲も一致した。`log/cpu_repro_diagnosis_20260913_1225_initialized32/` に観測結果を残し、全epochの再現性合格へは広げない。

一方、直近のbest／final初期化2試行でもtrain lossと保存Tensorに差がある。入力checkpointのTensor byte列は同一であり、古いrunとの比較だけの問題とは言えない。小型診断では全epochの差を再現できておらず、原因を確定していない。PyTorch公式も乱数seed制御と非決定的演算の制御を区別しているが、それだけで今回の原因を特定したとは扱わない。[PyTorchの再現性](https://docs.pytorch.org/docs/2.14/notes/randomness.html)

### best／finalからの新規学習

`tests/acceptance/test_training_runs.py` のbest／finalケースは **2件合格、769.39秒**。`log/training_acceptance_20260913_1159/` に、元runを変更せず保存した。

- 両方とも重みだけを初期値として読み込み、epoch 1・初期learning rate `0.01` から新規学習した。
- 1 epoch後のbest／final両方を保存し、bestによる公式test 10,000件を完了した。
- 両試行のtest lossは `0.45463658199310303`、accuracyは `0.854`。100 epoch精度受入の結果ではない。
- best初期値の試行では回路を有効にしてJSON／C／Verilogを生成、final初期値の試行では回路directoryを生成しないことを確認した。
- 元runのファイルhashは変更されていない。

### 旧版精度比較の前提

`tests/acceptance/compare_legacy_smoke.py` の初回preflightではデータ分割・前処理・初回データ順・モデル構造・初期パラメータに不一致はなかった。記録は `log/legacy_diagnostic/20260913_115815_842466/`。ただし独立レビューで、接続bufferが比較されていないことを検出したため、初回結果を初期ネットワーク全体の一致とは扱わない。比較器へ接続bufferの検査を追加して再実行する。

旧版のDense接続抽選と新版の置換ありsamplingは、承認済み仕様上も同seedの一致を要求していない。接続差は比較条件の差として明示し、旧方式へ無断で変更したり重みを再配置したりしない。

修正後のpreflightを `log/legacy_diagnostic/20260913_122028_957926/` に保存した。全非Parameter stateを比較でき、**Parameterと前処理は一致、Dense 2層の接続は不一致、Conv接続は一致**だった。`initial_network.status=different` と記録し、比較条件を確認できたこととネットワーク一致を区別した。

これは比較条件を揃える確認であり、正式なAT-007の100 epoch精度比較ではない。診断用1 epochと正式精度受入を区別する。

### 実CLI異常・文書例・性能計測

`test_cli_errors.py`、benchmark比較器、再現性比較器、旧版比較器、掲載API例は **35件合格、20.94秒**。設定、loss名、モデル名、データ欠落、正式checkpoint構造不適合の5区分を実際のCLIへ渡し、終了1・原因と対応の表示・学習前停止を確認した。元config／pthのhashも保持している。

性能の初回基準を `tests/benchmarks/baselines/` に作成した。Raw rank 2の12条件、Warp rank 4の8条件、Light rank 6の8条件、計28条件を実測した。小型回路の簡略化は82 gateから18 gateとなり、生bitとスコアを維持した。CPU Tensorのnative peakは未測定で、tracemallocの値をその代用にしていない。初回値は性能の合格保証ではない。

### compilerログの文字化け修正

実MSVCはconsoleのUTF-8で出力していたが、Pythonがlocaleのcp932で読んでいたため日本語が文字化けしていた。compiler出力にはconsole code pageを明示し、MSVC環境の取得は `cmd /u` のUTF-16LEを厳密decodeする境界へ分離した。学習計算・回路計算・通常forwardは変更していない。

日本語の診断と、日本語・絵文字・等号を含む環境値について実機境界を確認した。親processの環境は変更せず、文字置換されたPATHを利用しない。修正前の文字化けログを後から書き換えず保持し、回帰テストと全体検証で修正後を確認する。

修正後に正式CLIのbestを実Cで再compileし、**1件合格、71.00秒**。スコア誤差0と日本語ログの復元を `log/verification_20260913_1227_native_log/` に記録した。
