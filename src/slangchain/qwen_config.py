from __future__ import annotations

import re
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

Mode = Literal["full", "benchmark"]


@dataclass(frozen=True)
class QwenDataConfig:
    source: Path
    packed_dir: Path
    train_through: int
    validation_through: int
    tokenizer_id: str
    tokenizer_revision: str
    tokenizer_batch_size: int
    max_records_per_split: int | None


@dataclass(frozen=True)
class QwenModelConfig:
    model_id: str
    revision: str
    context_length: int
    weights_dtype: str
    gradient_checkpointing: bool


@dataclass(frozen=True)
class QwenTrainingConfig:
    output_dir: Path
    micro_batch_size: int
    gradient_accumulation_steps: int
    epochs: int
    max_steps: int | None
    max_eval_batches: int | None
    learning_rate: float
    weight_decay: float
    gradient_clip: float
    warmup_ratio: float
    checkpoint_interval: int
    log_interval: int
    seed: int
    num_workers: int
    mixed_precision: str
    keep_checkpoints: int = 3


@dataclass(frozen=True)
class QwenExperimentConfig:
    mode: Mode
    data: QwenDataConfig
    model: QwenModelConfig
    training: QwenTrainingConfig


def _section(document: dict[str, Any], name: str) -> dict[str, Any]:
    value = document.get(name)
    if not isinstance(value, dict):
        raise ValueError(f"Missing [{name}] section")
    return value


def _optional_positive_int(values: dict[str, Any], name: str) -> int | None:
    value = values.get(name)
    if value is None:
        return None
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{name} must be a positive integer when set")
    return value


def load_qwen_config(path: Path) -> QwenExperimentConfig:
    with path.open("rb") as handle:
        document = tomllib.load(handle)

    experiment_values = _section(document, "experiment")
    data_values = _section(document, "data")
    model_values = _section(document, "model")
    training_values = _section(document, "training")
    mode = experiment_values.get("mode")
    if mode not in {"full", "benchmark"}:
        raise ValueError("experiment.mode must be 'full' or 'benchmark'")

    data = QwenDataConfig(
        source=Path(data_values["source"]),
        packed_dir=Path(data_values["packed_dir"]),
        train_through=data_values["train_through"],
        validation_through=data_values["validation_through"],
        tokenizer_id=data_values["tokenizer_id"],
        tokenizer_revision=data_values["tokenizer_revision"],
        tokenizer_batch_size=data_values["tokenizer_batch_size"],
        max_records_per_split=_optional_positive_int(data_values, "max_records_per_split"),
    )
    model = QwenModelConfig(
        model_id=model_values["model_id"],
        revision=model_values["revision"],
        context_length=model_values["context_length"],
        weights_dtype=model_values["weights_dtype"],
        gradient_checkpointing=model_values["gradient_checkpointing"],
    )
    training = QwenTrainingConfig(
        output_dir=Path(training_values["output_dir"]),
        micro_batch_size=training_values["micro_batch_size"],
        gradient_accumulation_steps=training_values["gradient_accumulation_steps"],
        epochs=training_values["epochs"],
        max_steps=_optional_positive_int(training_values, "max_steps"),
        max_eval_batches=_optional_positive_int(training_values, "max_eval_batches"),
        learning_rate=training_values["learning_rate"],
        weight_decay=training_values["weight_decay"],
        gradient_clip=training_values["gradient_clip"],
        warmup_ratio=training_values["warmup_ratio"],
        checkpoint_interval=training_values["checkpoint_interval"],
        log_interval=training_values["log_interval"],
        seed=training_values["seed"],
        num_workers=training_values["num_workers"],
        mixed_precision=training_values["mixed_precision"],
        keep_checkpoints=_optional_positive_int(training_values, "keep_checkpoints")
        or 3,
    )
    config = QwenExperimentConfig(mode=mode, data=data, model=model, training=training)
    _validate(config)
    return config


def _validate(config: QwenExperimentConfig) -> None:
    data = config.data
    model = config.model
    training = config.training
    for name, year in (
        ("data.train_through", data.train_through),
        ("data.validation_through", data.validation_through),
    ):
        if not isinstance(year, int) or isinstance(year, bool) or not 1900 <= year <= 2100:
            raise ValueError(f"{name} must be an integer calendar year from 1900 to 2100")
    if data.train_through >= data.validation_through:
        raise ValueError("data.train_through must be earlier than data.validation_through")
    if data.tokenizer_batch_size <= 0:
        raise ValueError("data.tokenizer_batch_size must be positive")
    if not data.tokenizer_id.strip() or not data.tokenizer_revision.strip():
        raise ValueError("The tokenizer ID and revision are required")
    if not model.model_id.strip() or not model.revision.strip():
        raise ValueError("The model ID and revision are required")
    for label, revision in (
        ("data.tokenizer_revision", data.tokenizer_revision),
        ("model.revision", model.revision),
    ):
        if not re.fullmatch(r"[0-9a-f]{40}", revision):
            raise ValueError(f"{label} must be an immutable 40-character commit SHA")
    if model.context_length < 128:
        raise ValueError("model.context_length must be at least 128")
    if model.weights_dtype not in {"float32", "bfloat16"}:
        raise ValueError("model.weights_dtype must be 'float32' or 'bfloat16'")
    positive_training_values = (
        training.micro_batch_size,
        training.gradient_accumulation_steps,
        training.epochs,
        training.checkpoint_interval,
        training.log_interval,
        training.keep_checkpoints,
    )
    if min(positive_training_values) <= 0:
        raise ValueError("Training sizes, epochs, checkpointing, and logging must be positive")
    if training.learning_rate <= 0 or training.weight_decay < 0:
        raise ValueError(
            "Optimizer settings require a positive learning rate and weight decay >= 0"
        )
    if training.gradient_clip <= 0:
        raise ValueError("training.gradient_clip must be positive")
    if not 0 <= training.warmup_ratio < 1:
        raise ValueError("training.warmup_ratio must be in [0, 1)")
    if training.num_workers < 0:
        raise ValueError("training.num_workers cannot be negative")
    if training.mixed_precision not in {"auto", "no", "fp16", "bf16"}:
        raise ValueError("training.mixed_precision is not supported by Accelerate")
    if config.mode == "full":
        if data.max_records_per_split is not None:
            raise ValueError("Full mode cannot cap records; remove data.max_records_per_split")
        if training.max_steps is not None:
            raise ValueError("Full mode trains by epoch; remove training.max_steps")
        if training.max_eval_batches is not None:
            raise ValueError("Full mode evaluates every packed block; remove max_eval_batches")
    else:
        if (
            data.max_records_per_split is None
            or training.max_steps is None
            or training.max_eval_batches is None
        ):
            raise ValueError(
                "Benchmark mode requires record, training-step, and evaluation-batch caps"
            )
