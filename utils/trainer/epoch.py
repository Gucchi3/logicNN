"""保存・表示・scheduler更新を含めず、1 epochの学習または評価を行う。"""

from collections.abc import Iterable
from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.optim import Optimizer


@dataclass(frozen=True)
class EpochMetrics:
    """全標本を同じ重みで集計した損失・正解率・標本数を保持する。"""

    loss: float
    accuracy: float
    samples: int


def train_one_epoch(
    model: nn.Module, loader: Iterable[tuple[Tensor, Tensor]], loss: nn.Module, optimizer: Optimizer, device: torch.device,
) -> EpochMetrics:
    """全batchを順に学習し、更新前スコアから全標本重み付きの指標を返す。"""
    model.train()
    loss_total = 0.0
    correct    = 0
    samples    = 0
    for inputs, targets in loader:
        inputs  = inputs.to(device=device, dtype=torch.float32)
        targets = targets.to(device=device, dtype=torch.int64)
        optimizer.zero_grad(set_to_none=True)
        scores        = model(inputs)
        batch_loss    = loss(scores, targets)
        batch_correct = int((scores.argmax(dim=1) == targets).sum().item())
        batch_loss.backward()
        optimizer.step()
        count       = targets.shape[0]
        loss_total += float(batch_loss.detach().item()) * count
        correct    += batch_correct
        samples    += count
    if samples == 0:
        raise ValueError("DataLoaderが空です")
    return EpochMetrics(loss_total / samples, correct / samples, samples)


def evaluate(model: nn.Module, loader: Iterable[tuple[Tensor, Tensor]], loss: nn.Module, device: torch.device) -> EpochMetrics:
    """重みと勾配を変更せず全batchを評価し、全標本重み付きの指標を返す。"""
    model.eval()
    loss_total = 0.0
    correct    = 0
    samples    = 0
    with torch.no_grad():
        for inputs, targets in loader:
            inputs      = inputs.to(device=device, dtype=torch.float32)
            targets     = targets.to(device=device, dtype=torch.int64)
            scores      = model(inputs)
            batch_loss  = loss(scores, targets)
            count       = targets.shape[0]
            loss_total += float(batch_loss.item()) * count
            correct    += int((scores.argmax(dim=1) == targets).sum().item())
            samples    += count
    if samples == 0:
        raise ValueError("DataLoaderが空です")
    return EpochMetrics(loss_total / samples, correct / samples, samples)
