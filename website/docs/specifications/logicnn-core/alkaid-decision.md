---
sidebar_position: 7
title: Alkaid連携の判断待ち
---

# Alkaid連携の判断待ち

## 今回確認できたこと

2026-09-13に公式sourceを確認した。pluginの登録先と返却形式自体は判明した。`ALIRTracerPluginBase.apply_model()` はtrace辞書と出力keyの列を返し、通常のtrace結果はflattenした入力・出力の組となる。[公式plugin source](https://raw.githubusercontent.com/calad0i/alkaid/master/src/alkaid/converter/plugin.py)

したがって、現在の保留理由は「呼び出すAPIが分からない」ことではない。

## 残る数値契約

AlkaidのALIRにおける量子化は固定小数点のbit削減を使い、overflowはwrap、丸めはtruncateである。[公式ALIR仕様](https://calad0i.github.io/alkaid/alir.html)

一方、logicNN_coreでは元のTensor dtypeと演算順序を維持し、bit／整数は一致、小数は微小な丸め差の範囲で扱う。旧torchlogixのadapterはdtype変換を恒等扱いし、sum・bias・tauの演算をAlkaidの数値へ渡している。そのまま移植しただけでは、新coreの浮動小数点スコアの契約を満たすとは言えない。

## 利用者の判断が必要なこと

次の範囲を確定するまでは、adapter本体を実装しない。

1. まずboolの生出力や、丸めが不要で厳密に表現できる整数だけを部分対応とするか。
2. 集約後の小数スコアまで対応させる場合、固定小数点の整数部・小数部、overflow、許容誤差をどう指定するか。

小数を勝手に丸める、別のdtypeへ置換する、対応済みに見えるstubを公開することはしない。Alkaidを導入していない通常環境での学習・JSON・C・Verilog出力には影響させない。

このページは調査と判断待ちの記録であり、新たな数値仕様の承認ではない。今回の「仕様変更が必要な箇所のみ停止する」指示に従い、Alkaid以外の実装・検証を継続する。
