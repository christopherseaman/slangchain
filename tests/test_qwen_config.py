from __future__ import annotations

from pathlib import Path

import pytest

from slangchain.qwen_config import load_qwen_config


def test_full_qwen_config_uses_all_records_at_context_2048() -> None:
    config = load_qwen_config(Path("configs/qwen35-0.8b.toml"))

    assert config.mode == "full"
    assert config.data.max_records_per_split is None
    assert config.training.max_steps is None
    assert config.training.max_eval_batches is None
    assert config.data.train_through == 2023
    assert config.model.context_length == 2048
    assert config.model.model_id == "Qwen/Qwen3.5-0.8B-Base"
    assert config.training.mixed_precision == "auto"
    assert config.training.keep_checkpoints == 3


def test_full_qwen_config_rejects_a_hidden_record_cap(tmp_path: Path) -> None:
    source = Path("configs/qwen35-0.8b.toml").read_text(encoding="utf-8")
    config_path = tmp_path / "capped.toml"
    config_path.write_text(
        source.replace(
            "tokenizer_batch_size = 512",
            "tokenizer_batch_size = 512\nmax_records_per_split = 10",
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Full mode cannot cap records"):
        load_qwen_config(config_path)


def test_qwen_revisions_must_be_immutable(tmp_path: Path) -> None:
    source = Path("configs/qwen35-0.8b.toml").read_text(encoding="utf-8")
    config_path = tmp_path / "moving-revision.toml"
    config_path.write_text(
        source.replace(
            'revision = "dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68"',
            'revision = "main"',
            1,
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="immutable 40-character commit SHA"):
        load_qwen_config(config_path)


def test_checkpoint_retention_must_be_positive(tmp_path: Path) -> None:
    source = Path("configs/qwen35-0.8b.toml").read_text(encoding="utf-8")
    config_path = tmp_path / "no-checkpoints.toml"
    config_path.write_text(
        source.replace("keep_checkpoints = 3", "keep_checkpoints = 0"),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="keep_checkpoints must be a positive integer"):
        load_qwen_config(config_path)
