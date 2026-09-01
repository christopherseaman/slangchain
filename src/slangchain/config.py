from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class DataConfig:
    path: Path
    train_through: int
    validation_through: int


@dataclass(frozen=True)
class ModelConfig:
    context_length: int
    embed_dim: int
    heads: int
    layers: int
    dropout: float
    year_origin: float
    year_scale: float


@dataclass(frozen=True)
class TrainingConfig:
    output_dir: Path
    batch_size: int
    steps: int
    learning_rate: float
    weight_decay: float
    gradient_clip: float
    eval_interval: int
    eval_batches: int
    log_interval: int
    seed: int
    num_workers: int
    mixed_precision: str


@dataclass(frozen=True)
class ExperimentConfig:
    data: DataConfig
    model: ModelConfig
    training: TrainingConfig


def _section(document: dict[str, Any], name: str) -> dict[str, Any]:
    value = document.get(name)
    if not isinstance(value, dict):
        raise ValueError(f"Missing [{name}] section")
    return value


def load_config(path: Path) -> ExperimentConfig:
    with path.open("rb") as handle:
        document = tomllib.load(handle)

    data_values = _section(document, "data")
    model_values = _section(document, "model")
    training_values = _section(document, "training")
    data = DataConfig(
        path=Path(data_values["path"]),
        train_through=data_values["train_through"],
        validation_through=data_values["validation_through"],
    )
    model = ModelConfig(**model_values)
    training = TrainingConfig(
        output_dir=Path(training_values["output_dir"]),
        **{key: value for key, value in training_values.items() if key != "output_dir"},
    )
    _validate(data, model, training)
    return ExperimentConfig(data=data, model=model, training=training)


def _validate(data: DataConfig, model: ModelConfig, training: TrainingConfig) -> None:
    if data.train_through >= data.validation_through:
        raise ValueError("data.train_through must be earlier than data.validation_through")
    if model.context_length < 8:
        raise ValueError("model.context_length must be at least 8")
    if model.heads <= 0:
        raise ValueError("model.heads must be positive")
    if model.embed_dim <= 0 or model.embed_dim % model.heads:
        raise ValueError("model.embed_dim must be positive and divisible by model.heads")
    if model.layers <= 0:
        raise ValueError("model.layers must be positive")
    if not 0 <= model.dropout < 1:
        raise ValueError("model.dropout must be in [0, 1)")
    if model.year_scale == 0:
        raise ValueError("model.year_scale cannot be zero")
    if min(training.batch_size, training.steps, training.eval_interval, training.eval_batches) <= 0:
        raise ValueError("batch size, steps, and evaluation settings must be positive")
    if training.learning_rate <= 0 or training.weight_decay < 0:
        raise ValueError("optimizer settings must be non-negative with a positive learning rate")
    if training.gradient_clip <= 0:
        raise ValueError("training.gradient_clip must be positive")
    if training.log_interval <= 0 or training.num_workers < 0:
        raise ValueError("logging must be positive and workers cannot be negative")
    if training.mixed_precision not in {"no", "fp16", "bf16"}:
        raise ValueError("training.mixed_precision is not supported by Accelerate")
