"""固定・学習可能な接続と、構造別の型付きbuilderを公開する。"""

from .base import Connections
from .builder import build_conv_connections, build_dense_connections
from .convolution import FixedConvConnections
from .dense import FixedDenseConnections, LearnableDenseConnections

__all__ = [
    "Connections",
    "FixedDenseConnections",
    "LearnableDenseConnections",
    "FixedConvConnections",
    "build_dense_connections",
    "build_conv_connections",
]
