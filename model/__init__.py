"""モデルの登録表と、名前だけを受け取る生成入口を公開する。"""

from .builder import MODEL_REGISTRY, build_model

__all__ = ["MODEL_REGISTRY", "build_model"]
