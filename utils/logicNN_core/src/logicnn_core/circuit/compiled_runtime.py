"""C compilerと共有libraryの寿命を管理し、CPU回路のcompiled評価を提供する。"""

from __future__ import annotations

import ctypes
import locale
import math
import os
import shutil
import subprocess
import sys
import weakref
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch
from torch import Tensor

from ..exceptions import LogicNNCoreError
from .exporters.c import c_source
from .runtime import _binary_inputs, validate_numeric_operations
from .types import CircuitData


def _compiler_error(message: str, detail: str) -> LogicNNCoreError:
    """compiler選択・実行・library読込の失敗を、実行段階付きで生成する。"""
    return LogicNNCoreError(message, detail=detail, hint="対応するnative C compilerと開発環境を用意してください。Python実行へは自動fallbackしません")


def _compiler_encoding() -> str:
    """MSVCのconsole code pageを使い、consoleがないWindowsではANSI code pageへ従う。"""
    if os.name == "nt":
        kernel   = ctypes.windll.kernel32
        codepage = kernel.GetConsoleOutputCP() or kernel.GetACP()
        return f"cp{codepage}"
    return locale.getpreferredencoding(False)


def _compiler_environment(compiler: str) -> dict[str, str]:
    """MSVCが必要な場合だけ子process用の開発環境を読み、親の環境変数は変更しない。"""
    environment = os.environ.copy()
    if os.name != "nt" or Path(compiler).name.lower() != "cl.exe" or (environment.get("INCLUDE") and environment.get("LIB")):
        return environment
    parents = Path(compiler).resolve().parents
    if len(parents) < 7:
        raise _compiler_error("MSVC環境を初期化できません", f"compiler={compiler}")
    setup = parents[6] / "Auxiliary" / "Build" / "vcvars64.bat"
    if not setup.is_file():
        raise _compiler_error("MSVC環境設定が見つかりません", f"setup={setup}")
    command = f'cmd.exe /u /d /s /c ""{setup}" >nul && set"'
    result  = subprocess.run(command, capture_output=True, timeout=90, check=False)
    if result.returncode:
        # cmd /uの内部診断はUTF-16、呼び出した外部toolの診断はconsole code page。環境値にはこの推定を使わない。
        encoding = "utf-16-le" if b"\0" in result.stderr else _compiler_encoding()
        detail   = result.stderr.decode(encoding, errors="replace")
        raise _compiler_error("MSVC環境の初期化に失敗しました", f"stage=environment, returncode={result.returncode}: {detail[-4000:]}")
    try:
        output = result.stdout.decode("utf-16-le")
    except UnicodeDecodeError as error:
        raise _compiler_error("MSVC環境の文字列を読み取れません", f"stage=environment, expected=UTF-16LE, byte_offset={error.start}") from error
    for line in output.splitlines():
        name, separator, value = line.partition("=")
        if separator and name:
            environment[name] = value
    return environment


def _compiler() -> str:
    """現在のplatformに対応する既存compilerを選び、自動導入や異種DLLへの代替をしない。"""
    names = ("cl",) if os.name == "nt" else ("cc", "gcc", "clang")
    for name in names:
        path = shutil.which(name)
        if path:
            return path
    raise _compiler_error("対応するC compilerが見つかりません", f"platform={sys.platform}, candidates={names}")


def _release_library(library: ctypes.CDLL, directory: TemporaryDirectory[str]) -> None:
    """所有する共有libraryを解放した後、専用の一時成果物だけを片付ける。"""
    import _ctypes

    try:
        if os.name == "nt":
            _ctypes.FreeLibrary(library._handle)
        else:
            _ctypes.dlclose(library._handle)
    finally:
        directory.cleanup()


class CompiledCircuit:
    """成功時点の独立IRとnative関数を所有し、同じ回路のbatchを評価する。"""

    def __init__(self, data: CircuitData, pack_bits: int | None, library: ctypes.CDLL, directory: TemporaryDirectory[str]) -> None:
        """libraryの関数signatureと所有期間を固定し、入力IRを独立コピーする。"""
        self._data      = deepcopy(data)
        self._pack_bits = pack_bits
        self._function = library.logicnn_evaluate
        self._function.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t]
        self._function.restype  = None
        self._finalizer = weakref.finalize(self, _release_library, library, directory)

    def evaluate(self, inputs: Tensor | np.ndarray) -> Tensor:
        """binary入力を必要に応じpackingし、native結果を保存dtypeのCPU Tensorで返す。"""
        binary = _binary_inputs(inputs, self._data.input_shape).numpy()
        batch  = binary.shape[0]
        output = np.empty((batch, math.prod(self._data.output_shape)), dtype=np.dtype(self._data.output_dtype.value))
        if batch:
            prepared = binary
            if self._pack_bits is not None:
                width    = self._pack_bits
                dtype    = np.dtype(f"uint{width}")
                prepared = np.zeros(((batch + width - 1) // width, binary.shape[1]), dtype=dtype)
                for bit_index in range(min(width, batch)):
                    rows = binary[bit_index::width].astype(dtype)
                    prepared[:len(rows)] |= rows << bit_index
            prepared = np.ascontiguousarray(prepared)
            self._function(prepared.ctypes.data, output.ctypes.data, batch)
        return torch.from_numpy(output).reshape(batch, *self._data.output_shape)


def compile_circuit(data: CircuitData, *, optimization_level: int = 1, pack_bits: int | None = None) -> CompiledCircuit:
    """独立directoryでCをcompileし、library読込まで成功した状態だけを返す。"""
    if type(optimization_level) is not int or optimization_level not in range(4):
        raise _compiler_error("optimization_levelは0〜3の整数が必要です", f"optimization_level={optimization_level!r}")
    source   = c_source(data, pack_bits=pack_bits)
    validate_numeric_operations(data)
    compiler = _compiler()
    temporary: TemporaryDirectory[str] = TemporaryDirectory(prefix="logicnn-core-")
    folder   = Path(temporary.name)
    library: ctypes.CDLL | None = None
    try:
        environment = _compiler_environment(compiler)
        source_path = folder / "circuit.c"
        output_path = folder / ("circuit.dll" if os.name == "nt" else "circuit.so")
        source_path.write_text(source, encoding="utf-8")
        if os.name == "nt":
            optimization = "/Od" if optimization_level == 0 else "/O1" if optimization_level == 1 else "/O2"
            command = [compiler, "/nologo", "/std:c11", "/LD", optimization, "/fp:strict", str(source_path), f"/Fe:{output_path}"]
        else:
            command = [
                compiler, "-std=c11", "-shared", "-fPIC", "-fno-fast-math", "-ffp-contract=off", f"-O{optimization_level}",
                str(source_path), "-o", str(output_path),
            ]
        result = subprocess.run(command, cwd=folder, env=environment, capture_output=True, text=True, encoding=_compiler_encoding(), errors="replace",
                                timeout=90, check=False)
        if result.returncode:
            details = f"stage=compile, compiler={compiler}, returncode={result.returncode}\n{result.stdout[-4000:]}\n{result.stderr[-4000:]}"
            raise _compiler_error("回路のC compileに失敗しました", details)
        library = ctypes.CDLL(str(output_path))
        return CompiledCircuit(data, pack_bits, library, temporary)
    except (OSError, subprocess.SubprocessError, AttributeError) as error:
        if library is not None:
            _release_library(library, temporary)
        else:
            temporary.cleanup()
        raise _compiler_error("C実行環境を準備できません", f"stage=compile/load: {type(error).__name__}: {error}") from error
    except Exception:
        if library is not None:
            _release_library(library, temporary)
        else:
            temporary.cleanup()
        raise
