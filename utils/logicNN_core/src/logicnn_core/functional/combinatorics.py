"""colex順の組合せ変換、疎な接続samplingとMSB-firstのLUT真理値表を提供する。"""

# torchlogix/functional.pyの組合せ・真理値表処理を基に再構成した。MIT License。
# Copyright (c) 2021-2023 Dr. Felix Petersen
# Copyright (c) 2024-present Dr. Lino Gerlach
# 許諾条件の全文と変更の由来は同梱LICENSEおよびTHIRD_PARTY_NOTICESを参照。

from __future__ import annotations

import math

import torch
from torch import Tensor

__all__ = ["build_binomial_table", "sample_unique_combinations", "truth_table_from_id", "unrank_combinations"]

_INT64_MAX = torch.iinfo(torch.int64).max


def _validate_nonnegative_integer(name: str, value: int) -> None:
    """非boolの非負Python整数とTensor indexで扱える上限を検証する。"""
    if type(value) is not int:
        raise TypeError(f"{name}はboolではないPython整数で指定してください。")
    if not 0 <= value <= _INT64_MAX:
        raise ValueError(f"{name}は0以上int64上限以下で指定してください。")


def _validate_dimensions(n: int, k: int) -> None:
    """集合サイズと選択数が重複なし組合せの範囲内か検証する。"""
    _validate_nonnegative_integer("n", n)
    _validate_nonnegative_integer("k", k)
    if k > n:
        raise ValueError("kはn以下で指定してください。")


def _bounded_binomial(n: int, k: int) -> int:
    """二項係数を正確な整数で求め、int64上限超過時は早期に拒否する。"""
    value = 1
    for index in range(1, min(k, n - k) + 1):
        value = value * (n - index + 1) // index
        if value > _INT64_MAX:
            raise ValueError("二項係数または組合せ総数がint64上限を超えます。")
    return value


def build_binomial_table(n: int, k: int, *, device: torch.device | str | None = None) -> Tensor:
    """shape(n, k+1)のint64表へC(row, column)を並べ、表全体のoverflowを事前検証する。"""
    _validate_dimensions(n, k)
    if n > 0:
        _bounded_binomial(n - 1, min(k, (n - 1) // 2))
    if k == _INT64_MAX or n * (k + 1) > _INT64_MAX:
        raise ValueError("二項係数表の要素数または次元がint64上限を超えます。")
    values = [[math.comb(row, column) for column in range(k + 1)] for row in range(n)]
    return torch.tensor(values, dtype=torch.int64, device=device).reshape(n, k + 1)


def _unrank_single(n: int, k: int, rank: int) -> list[int]:
    """二分探索のcombinadicで一つのrankを昇順tupleへ変換する。"""
    combination = [0] * k
    maximum     = n - 1
    for order in range(k, 0, -1):
        low, high = order - 1, maximum
        while low < high:
            middle = (low + high + 1) // 2
            if math.comb(middle, order) <= rank:
                low = middle
            else:
                high = middle - 1
        combination[order - 1] = low
        rank                  = rank - math.comb(low, order)
        maximum               = low - 1
    return combination


def unrank_combinations(n: int, k: int, ranks: Tensor) -> Tensor:
    """colex順のrankをranks.shape+(k,)の昇順index列へ変換し、入力deviceへ戻す。"""
    _validate_dimensions(n, k)
    _bounded_binomial(n, k)
    values = ranks.to(dtype=torch.int64)
    if values.numel() * k > _INT64_MAX:
        raise ValueError("組合せ出力の要素数がint64上限を超えます。")
    # 初期接続用の整数処理としてCPUで探索し、候補総数に比例する表は作らない。
    combinations = [_unrank_single(n, k, rank) for rank in values.reshape(-1).tolist()]
    return torch.tensor(combinations, dtype=torch.int64, device=ranks.device).reshape(*ranks.shape, k)


def _sample_ranks(total: int, sample_size: int, *, device: torch.device, generator: torch.Generator | None) -> list[int]:
    """疎な部分Fisher-Yates法で順序付きの重複なしrankをO(sample_size)メモリで選ぶ。"""
    swaps: dict[int, int] = {}
    selected: list[int]   = []
    for index in range(sample_size):
        remaining = total - index
        position  = int(torch.randint(remaining, (), device=device, generator=generator).item())
        selected.append(swaps.get(position, position))
        swaps[position] = swaps.get(remaining - 1, remaining - 1)
        swaps.pop(remaining - 1, None)
    return selected


def sample_unique_combinations(
    n: int,
    k: int,
    sample_size: int,
    num_sets: int = 1,
    *,
    device: torch.device | str | None = None,
    generator: torch.Generator | None = None,
) -> Tensor:
    """各set内でtupleを重複させず、shape(num_sets, sample_size, k)で指定deviceへ返す。"""
    _validate_dimensions(n, k)
    _validate_nonnegative_integer("sample_size", sample_size)
    _validate_nonnegative_integer("num_sets", num_sets)
    total = _bounded_binomial(n, k)
    if sample_size > total:
        raise ValueError(f"sample_sizeは組合せ総数{total}以下で指定してください。")
    if num_sets * sample_size * max(k, 1) > _INT64_MAX:
        raise ValueError("sampling出力の要素数がint64上限を超えます。")
    output_device = torch.empty(0, device=device).device
    random_device = output_device if generator is None else generator.device
    if num_sets == 0 or sample_size == 0:
        return torch.empty((num_sets, sample_size, k), dtype=torch.int64, device=output_device)
    # 専用generatorは自身のdevice上だけで消費し、global乱数状態を変更しない。
    rank_sets = [_sample_ranks(total, sample_size, device=random_device, generator=generator) for _ in range(num_sets)]
    ranks     = torch.tensor(rank_sets, dtype=torch.int64).reshape(num_sets, sample_size)
    return unrank_combinations(n, k, ranks).to(device=output_device)


def truth_table_from_id(lut_ids: Tensor, rank: int) -> Tensor:
    """rank 1〜4のIDを、入力の二進昇順に対応するMSB-firstのbool真理値表へ展開する。"""
    if type(rank) is not int:
        raise TypeError("rankはboolではないPython整数で指定してください。")
    if not 1 <= rank <= 4:
        raise ValueError("整数IDのrankは1〜4に限ります。高rankでは真理値表を直接使用してください。")
    entries = 2**rank
    values  = lut_ids.to(dtype=torch.int64)
    shifts  = torch.arange(entries - 1, -1, -1, device=lut_ids.device)
    return ((values.unsqueeze(-1) >> shifts) & 1).to(dtype=torch.bool)
