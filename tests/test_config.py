from __future__ import annotations

from pathlib import Path

import pytest

from slangchain.config import load_config


def test_toml_paths_are_converted_to_paths() -> None:
    config = load_config(Path("configs/smoke.toml"))

    assert config.data.path == Path("data/processed/toy.jsonl")
    assert config.training.output_dir == Path("artifacts/smoke")


@pytest.mark.parametrize("heads", [0, -4])
def test_head_count_must_be_positive(tmp_path: Path, heads: int) -> None:
    source = Path("configs/smoke.toml").read_text(encoding="utf-8")
    config_path = tmp_path / "invalid.toml"
    config_path.write_text(source.replace("heads = 4", f"heads = {heads}"), encoding="utf-8")

    with pytest.raises(ValueError, match="model.heads must be positive"):
        load_config(config_path)
