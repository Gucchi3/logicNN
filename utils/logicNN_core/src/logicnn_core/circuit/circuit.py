"""モデルを保護して論理回路を生成する、実装済みの公開入口を提供する。"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from numpy import ndarray
from torch import Tensor, nn
from torch.fx import GraphModule
from torch.fx.experimental.proxy_tensor import make_fx

from ..exceptions import LogicNNCoreError
from ..export_mode import set_export_mode
from . import runtime, serialization, simplification
from .builder import _build_traced_graph, build_from_fx_graph
from .exporters import c, verilog, writing
from .types import CircuitData, FxSignalReference
from .validation import normalize_input_shape, validate_circuit

if TYPE_CHECKING:
    from .compiled_runtime import CompiledCircuit


def _trace_model(model: nn.Module, input_shape: tuple[int, ...]) -> GraphModule:
    """独立したCPUコピーをexport追跡し、元のdtype・モデル状態・CPU乱数状態を守る。"""
    stage = "model copy"
    try:
        with torch.random.fork_rng(devices=[]), torch.no_grad():
            copied = deepcopy(model)
            stage  = "CPU export preparation"
            copied.cpu().eval()
            set_export_mode(copied)
            stage = "FX trace"
            return make_fx(copied)(torch.zeros((1, *input_shape), dtype=torch.bool, device="cpu"))
    except LogicNNCoreError:
        raise
    except Exception as error:
        raise LogicNNCoreError(
            "モデルから回路用の処理グラフを作成できません",
            detail=f"stage={stage}: {type(error).__name__}: {error}",
            hint="モデルが独立コピー・CPU移動・bool入力のexport追跡に対応しているか確認してください",
        ) from error


class Circuit:
    """独立した論理IRを所有し、構築・検証・実行・保存・簡略化を提供する。"""

    def __init__(self, data: CircuitData) -> None:
        """構造と数値型の意味を検証したIRを独立コピーして所有する。"""
        validate_circuit(data)
        runtime.validate_numeric_operations(data)
        self._data     = deepcopy(data)
        self._compiled: CompiledCircuit | None = None

    @classmethod
    def from_model(cls, model: nn.Module, input_shape: Sequence[int]) -> Circuit:
        """0／1入力のモデルをコピーして追跡し、元の論理出力対応を自動記録する。"""
        if not isinstance(model, nn.Module):
            raise TypeError("model はtorch.nn.Moduleで指定してください")
        shape = normalize_input_shape(input_shape)
        graph = _trace_model(model, shape)
        return cls(_build_traced_graph(graph, shape))

    @classmethod
    def from_fx_graph(
        cls,
        graph_module: GraphModule,
        input_shape: Sequence[int],
        *,
        logical_outputs: Sequence[Sequence[FxSignalReference | bool]],
    ) -> Circuit:
        """元出力対応を伴う外部FXから、失われた信号を推測せず回路を構築する。"""
        data = build_from_fx_graph(graph_module, input_shape, logical_outputs=logical_outputs)
        return cls(data)

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> Circuit:
        """厳密な回路schemaのdictから、数値型も検証した独立した回路を生成する。"""
        return cls(serialization.from_dict(data))

    @classmethod
    def from_json(cls, path: str | Path) -> Circuit:
        """UTF-8の回路JSONを読み、保存された数値演算と元出力対応を復元する。"""
        return cls(serialization.read_json(path))

    def validate(self) -> None:
        """保持するIRの構造・論理出力対応・数値演算の型を検証する。"""
        validate_circuit(self._data)
        runtime.validate_numeric_operations(self._data)

    def simplify(self, *, max_passes: int = 1000) -> None:
        """候補の構造・数値型が簡略化後も有効な場合だけ、所有IRを置き換える。"""
        candidate = deepcopy(self._data)
        simplification.simplify(candidate, max_passes=max_passes)
        runtime.validate_numeric_operations(candidate)
        self._data     = candidate
        self._compiled = None

    def compile(self, *, optimization_level: int = 1, pack_bits: int | None = None) -> None:
        """現在のIRをnative Cへcompileし、成功した場合だけ旧compiled状態を置き換える。"""
        from .compiled_runtime import compile_circuit

        candidate      = compile_circuit(self._data, optimization_level=optimization_level, pack_bits=pack_bits)
        self._compiled = candidate

    def evaluate(self, inputs: Tensor | ndarray, *, compiled: bool = False) -> Tensor:
        """二値入力を評価し、保存した出力dtypeとshapeの集約後CPU Tensorを返す。"""
        if type(compiled) is not bool:
            raise LogicNNCoreError("compiledはboolが必要です", detail=f"compiled={compiled!r}", hint="Python実行にはcompiled=Falseを指定してください")
        if compiled:
            if self._compiled is None:
                raise LogicNNCoreError(
                    "compiled実行は現在の回路のcompile成功前には利用できません", detail="現在のIRに対応するnative実行状態がありません",
                    hint="compileを実行してください。簡略化後は再compileが必要です。Python評価にはcompiled=Falseを指定してください",
                )
            return self._compiled.evaluate(inputs)
        return runtime.evaluate(self._data, inputs)

    def c_source(self, *, inline_single_use: bool = False, pack_bits: int | None = None) -> str:
        """二値入力から集約後出力までを計算する独立C sourceを返す。"""
        return c.c_source(self._data, inline_single_use=inline_single_use, pack_bits=pack_bits)

    def verilog_source(self, *, inline_single_use: bool = False) -> str:
        """加算を含めず、保存された元の順序で生bitを出力するVerilog sourceを返す。"""
        return verilog.verilog_source(self._data, inline_single_use=inline_single_use)

    def write_c(self, path: str | Path, *, inline_single_use: bool = False, pack_bits: int | None = None) -> None:
        """検証とC生成の完了後に原子的に保存し、生成失敗時に既存fileを保護する。"""
        writing.write_source(self.c_source(inline_single_use=inline_single_use, pack_bits=pack_bits), path)

    def write_verilog(self, path: str | Path, *, inline_single_use: bool = False) -> None:
        """元の生出力境界を持つVerilogを、生成と検証の完了後に原子的に保存する。"""
        writing.write_source(self.verilog_source(inline_single_use=inline_single_use), path)

    def to_dict(self) -> dict[str, object]:
        """構築時に検証した回路を、所有IRとcontainerを共有しないdictへ変換する。"""
        return serialization.to_dict(self._data)

    def write_json(self, path: str | Path) -> None:
        """構築時に検証した回路をJSONへ原子的に保存し、失敗時は既存fileを保つ。"""
        serialization.write_json(self._data, path)

    def logical_output_ids(self) -> tuple[int, ...]:
        """集約前の出力IDを元group順・bit順にflattenし、定数と重複を維持する。"""
        return tuple(node_id for group in self._data.logical_outputs for node_id in group.node_ids)

    def evaluate_logical_outputs(self, inputs: Tensor | ndarray) -> Tensor:
        """保存済みの元出力順に集約前bitを評価し、二次元のCPU bool Tensorを返す。"""
        return runtime.evaluate(self._data, inputs, logical=True)
