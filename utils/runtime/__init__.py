"""実行デバイスの選択と乱数seed設定を公開する。"""

from .device import select_device
from .seed import set_seed

__all__ = ("select_device", "set_seed")

