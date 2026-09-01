from __future__ import annotations

import torch

from slangchain.config import ModelConfig
from slangchain.data import Entry, EntryDataset
from slangchain.model import VOCAB_SIZE, TemporalSlangModel, sample


def _config() -> ModelConfig:
    return ModelConfig(
        context_length=32,
        embed_dim=16,
        heads=4,
        layers=1,
        dropout=0.0,
        year_origin=2020.0,
        year_scale=5.0,
    )


def test_forward_returns_logits_and_finite_loss() -> None:
    model = TemporalSlangModel(_config())
    tokens, targets, year = EntryDataset([Entry(2024, "glow", "a good mood")], 32)[0]

    logits, loss = model(tokens[None, :], year[None], targets[None, :])

    assert logits.shape == (1, 32, VOCAB_SIZE)
    assert loss is not None and torch.isfinite(loss)


def test_sample_preserves_the_prompt() -> None:
    model = TemporalSlangModel(_config())

    result = sample(
        model,
        prompt="WORD: ",
        year=2030,
        max_new_tokens=2,
        temperature=1.0,
        top_k=10,
        device=torch.device("cpu"),
    )

    assert result.startswith("WORD: ")
