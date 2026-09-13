---
title: Verilogの外部検証
description: 既存Verilatorを使う小型回路と学習済みMNISTLGNの再現手順
---

# Verilogの外部検証

このページは開発用テストの実行手順です。通常の `python main.py` はVerilogを保存するだけで、Verilatorやコンパイラを起動しません。テストは生成された組合せ回路の**生bit列**を検証し、クラス加算・tau・biasの回路は追加しません。

## 1. 確認した既存環境

2026-09-13の非破壊調査では、WindowsのPATHとCygwinでVerilatorを検出できませんでした。一方、既存WSLの `Ubuntu-24.04` には次の環境があり、version表示とWindows作業ディレクトリへのアクセスを確認しました。

| 項目 | 確認結果 |
|---|---|
| Verilator | `/usr/local/bin/verilator`、5.036、2025-04-27 rev v5.036 |
| C++ compiler | g++ 13.3.0 |
| make | GNU Make 4.3 |
| Windowsフォルダ | `wsl --distribution Ubuntu-24.04 --cd <Windowsの絶対path> --exec pwd` でアクセス可能 |

`Ubuntu` 無印ではVerilatorを検出できませんでした。WSL配布環境の新規作成や、OS・グローバルtoolのインストールは行っていません。**version確認だけではシミュレーション合格とはしません。** 実行結果は [検証記録](./verification.md) と各試験のログで区別します。

## 2. 小型回路を実行する

WindowsのPowerShellで `logicNN/` を作業ディレクトリにして実行します。

```powershell
$env:LOGICNN_VERILATOR_WSL = "Ubuntu-24.04"
.\.venv\Scripts\python.exe -m pytest tests/unit/test_verilog_testbench.py tests/core/test_verilog_exporter.py -q
```

`LOGICNN_VERILATOR_WSL` は開発用テストだけが参照します。指定した既存WSLの `verilator` を使い、Windows側でモデル／IRを扱いながらcompileとsimulationだけをWSLへ渡します。名前やパスをshellコマンド文字列に埋め込まず、引数として渡します。

変数を指定しない場合は、pytestを実行する環境のPATH上の `verilator` を使用します。PATHにない場合は外部試験を明示的にskipし、WSLを自動起動・探索しません。明示したWSLのtool起動やcompileが失敗した場合はテスト失敗とし、別環境へ自動fallbackしません。

小型の外部試験は、次の4種類をinline有効／無効の両方で実行します。

1. 全GateOp：生成側の変換表を使わないPython論理式を期待値にする。
2. 複数group：元の順序、定数、同じwireの重複出力を確認する。
3. 簡略化＋JSON往復：保存後も元の独立したbit列と一致することを確認する。
4. 全出力定数：gate計算が不要でも出力幅・各位置の値を維持する。

すべての2 bit入力4通りを調べます。外部tool不要のsource試験は別に残し、4,000 bit幅・識別子・長いinline鎖なども検証します。

## 3. 学習済みの正式MNISTLGNを実行する

既存の学習runにある `config.json` と `model_best.pth`、取得済みの公式MNIST testデータを使用します。テスト中の再学習やデータ取得は行いません。

```powershell
$env:LOGICNN_RUN_ACCEPTANCE = "1"
$env:LOGICNN_ACCEPTANCE_RUN_DIR = "C:\path\to\existing-training-run"
$env:LOGICNN_ACCEPTANCE_OUTPUT = "C:\path\to\new-verification-directory"
$env:LOGICNN_VERILATOR_WSL = "Ubuntu-24.04"
.\.venv\Scripts\python.exe -m pytest tests/acceptance/test_trained_mnist_verilog.py -q
```

`LOGICNN_ACCEPTANCE_OUTPUT` には、**まだ存在しない検証専用ディレクトリ**を指定します。省略時はpytestの一時ディレクトリを使います。元の学習runは読み取りだけに使用し、上書きしません。

共通fixture `trained_mnist_snapshot` は、seed 0で選んだ公式testの固定65標本、同じbest重み、そこから生成・JSON復元した回路を共有します。Verilogの期待値はIRから再生成せず、**元モデルのGroupSum直前をhookで取得した65 × 4,000 bit**を使用します。inline両方式について、各標本のすべての出力位置を完全一致で確認します。

この試験は同じモデルのbackend間の一致を調べるもので、65標本によるMNIST精度の受入判定ではありません。JSON／Cの検証も同時に行う場合は、同じpytest呼出しへ `tests/acceptance/test_trained_mnist.py` を追加すると、同じsession fixtureを共有できます。Cのnative検証には別途対応C compilerが必要です。

## 4. 保存する検証成果物

正式モデルでは `verilog_wires/` と `verilog_inline/` に、次を保存します。

- `logicnn_circuit.v`：検証したsource。モデルの出力内容は修正しない。
- `testbench.sv`：入力と独立期待値。配列位置 `i` をVerilogのbit `i` に対応させる。
- `version.log`、`compile.log`、`simulation.log`：実際のコマンド、終了コード、標準出力・標準エラー。
- `obj/`：Verilatorが生成したC++とシミュレータ。

検証ディレクトリ直下の `verilog_parity_wires.json`／`verilog_parity_inline.json` に、使用version、WSL配布名、標本index、bit幅、実行時間、合格結果を記録します。途中失敗時は完了した段階までのファイルを保持し、合格JSONを作成しません。

生成したsourceとtestbenchのあるディレクトリでは、次のコマンドでも同じ比較を再実行できます。`-Wno-fatal` はwarningをerror扱いしない指定であり、構文エラーや `$fatal` による値の不一致を無視する指定ではありません。

```text
verilator --binary --timing -Wno-fatal --top-module testbench --Mdir obj logicnn_circuit.v testbench.sv
./obj/Vtestbench
```

WindowsからWSLで再現する場合は、各コマンドを `wsl --distribution Ubuntu-24.04 --cd <成果物の絶対path> --exec` の後ろへ渡します。compileの上限は600秒、simulationは60秒です。終了コード0と `LOGICNN_VERILOG_PASS` の両方を合格条件とし、未検出・失敗・timeoutを合格として扱いません。
