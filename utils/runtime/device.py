"""単一CPUまたは単一CUDA GPUの実行デバイスを選択する。"""

from __future__ import annotations

import torch

from utils.exceptions import LogicNNError


def select_device(requested: str) -> torch.device:
    """設定値とCUDA可否から実際に使用する単一デバイスを返す。"""
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda":
        if torch.cuda.is_available():
            return torch.device("cuda")
        raise LogicNNError(
            "CUDAを利用できません",
            detail="run.deviceはcudaですが、PyTorchが利用可能なCUDAデバイスを検出できませんでした",
            hint="CUDA環境を確認するか、run.deviceをautoまたはcpuへ変更してください",
        )
    raise LogicNNError(
        "実行デバイスの指定が不正です",
        detail=f"run.device: {requested!r}",
        hint="run.deviceにはauto、cpu、cudaのいずれかを指定してください",
    )

