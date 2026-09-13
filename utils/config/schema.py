"""Pydanticを使ってlogicNNの設定JSONスキーマを定義する。"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

Seed = Annotated[int, Field(ge=0, le=4_294_967_295)]
PositiveInteger = Annotated[int, Field(ge=1)]
NonNegativeInteger = Annotated[int, Field(ge=0)]
Probability = Annotated[float, Field(ge=0.0, le=1.0)]
PositiveFloat = Annotated[float, Field(gt=0.0)]
NonNegativeFloat = Annotated[float, Field(ge=0.0)]
NonEmptyName = Annotated[str, Field(min_length=1)]


class _StrictConfig(BaseModel):
    """全設定区分に共通する厳格で不変な検証規則を定義する。"""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class RunConfig(_StrictConfig):
    """実行環境、出力先および任意の初期重みを定義する。"""

    seed: Seed
    device: Literal["auto", "cpu", "cuda"]
    log_dir: Path
    initial_checkpoint_path: Path | None

    @field_validator("log_dir", "initial_checkpoint_path", mode="before")
    @classmethod
    def validate_non_empty_path(cls, value: object) -> object:
        """空文字列や空白だけのパスを拒否する。"""
        if isinstance(value, str) and not value.strip():
            raise ValueError("空でないパスを指定してください")
        return value

    @field_validator("initial_checkpoint_path")
    @classmethod
    def validate_checkpoint_extension(cls, value: Path | None) -> Path | None:
        """初期チェックポイントの拡張子をpthに限定する。"""
        if value is not None and value.suffix.lower() != ".pth":
            raise ValueError("初期チェックポイントには.pthファイルを指定してください")
        return value


class DataConfig(_StrictConfig):
    """データセットの選択、保存先およびDataLoader設定を定義する。"""

    name: NonEmptyName
    root: Path
    batch_size: PositiveInteger
    num_workers: NonNegativeInteger

    @field_validator("root", mode="before")
    @classmethod
    def validate_non_empty_root(cls, value: object) -> object:
        """空文字列や空白だけのデータルートを拒否する。"""
        if isinstance(value, str) and not value.strip():
            raise ValueError("空でないパスを指定してください")
        return value


class ModelConfig(_StrictConfig):
    """model builderへ渡す登録名を定義する。"""

    name: NonEmptyName


class LossConfig(_StrictConfig):
    """損失関数の登録名と固有設定を定義する。"""

    name: NonEmptyName
    label_smoothing: Probability


class OptimizerConfig(_StrictConfig):
    """optimizerの登録名と固有設定を定義する。"""

    name: NonEmptyName
    learning_rate: PositiveFloat
    weight_decay: NonNegativeFloat


class SchedulerConfig(_StrictConfig):
    """learning-rate schedulerの登録名と固有設定を定義する。"""

    name: NonEmptyName
    minimum_learning_rate: NonNegativeFloat


class TrainConfig(_StrictConfig):
    """学習回数と最適化方式の設定を定義する。"""

    epochs: PositiveInteger
    loss: LossConfig
    optimizer: OptimizerConfig
    scheduler: SchedulerConfig

    @model_validator(mode="after")
    def validate_learning_rate_range(self) -> "TrainConfig":
        """schedulerの最小学習率が初期学習率を超えないことを確認する。"""
        if self.scheduler.minimum_learning_rate > self.optimizer.learning_rate:
            raise ValueError("train.scheduler.minimum_learning_rateはtrain.optimizer.learning_rate以下にしてください")
        return self


class CircuitConfig(_StrictConfig):
    """学習後の回路成果物生成スイッチを定義する。"""

    enabled: bool


class AppConfig(_StrictConfig):
    """logicNNの設定JSON全体を表す。"""

    run: RunConfig
    data: DataConfig
    model: ModelConfig
    train: TrainConfig
    circuit: CircuitConfig
