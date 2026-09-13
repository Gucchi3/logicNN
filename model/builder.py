"""登録名だけから、構造を内部定義したモデルの新しいインスタンスを生成する。"""

from __future__ import annotations

from torch import nn

from utils.exceptions import LogicNNError

from .lgn import MNISTLGN

MODEL_REGISTRY: dict[str, type[nn.Module]] = {"mnist_lgn": MNISTLGN}


def build_model(name: str) -> nn.Module:
    """登録名を検証し、外部の構造パラメータを渡さずモデルを生成する。"""
    if not isinstance(name, str) or name not in MODEL_REGISTRY:
        available = ", ".join(sorted(MODEL_REGISTRY))
        raise LogicNNError("未登録のモデルです", detail=f"model.name={name!r}。登録済み: {available}", hint=f"model.nameを次から指定してください: {available}")
    return MODEL_REGISTRY[name]()
