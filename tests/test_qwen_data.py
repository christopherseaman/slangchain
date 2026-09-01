from __future__ import annotations

import json
from array import array
from pathlib import Path

import pytest
import torch

import slangchain.qwen_data as qwen_data
from slangchain.data import Entry
from slangchain.qwen_config import (
    QwenDataConfig,
    QwenExperimentConfig,
    QwenModelConfig,
    QwenTrainingConfig,
)
from slangchain.qwen_data import (
    PackedTokenDataset,
    format_qwen_entry,
    load_packed_datasets,
    pack_corpus,
)


class YearTokenizer:
    eos_token_id = 99
    pad_token_id = 0
    vocab_size = 100
    init_kwargs = {"_commit_hash": "a" * 40}

    def __len__(self) -> int:
        return 100

    def __call__(
        self, texts: list[str], *, add_special_tokens: bool
    ) -> dict[str, list[list[int]]]:
        assert not add_special_tokens
        years = [int(text.split("YEAR: ", 1)[1][:4]) - 1900 for text in texts]
        return {"input_ids": [[year] for year in years]}


class CountingYearTokenizer(YearTokenizer):
    calls = 0

    def __call__(self, texts: list[str], *, add_special_tokens: bool) -> dict[str, list[list[int]]]:
        type(self).calls += 1
        return super().__call__(texts, add_special_tokens=add_special_tokens)


def _benchmark_config(source: Path, packed_dir: Path) -> QwenExperimentConfig:
    return QwenExperimentConfig(
        mode="benchmark",
        data=QwenDataConfig(
            source=source,
            packed_dir=packed_dir,
            train_through=2023,
            validation_through=2024,
            tokenizer_id="fake/tokenizer",
            tokenizer_revision="a" * 40,
            tokenizer_batch_size=2,
            max_records_per_split=1,
        ),
        model=QwenModelConfig(
            model_id="fake/model",
            revision="b" * 40,
            context_length=128,
            weights_dtype="float32",
            gradient_checkpointing=False,
        ),
        training=QwenTrainingConfig(
            output_dir=packed_dir.parent / "artifacts",
            micro_batch_size=1,
            gradient_accumulation_steps=1,
            epochs=1,
            max_steps=1,
            max_eval_batches=1,
            learning_rate=1e-5,
            weight_decay=0.1,
            gradient_clip=1.0,
            warmup_ratio=0.0,
            checkpoint_interval=1,
            log_interval=1,
            seed=42,
            num_workers=0,
            mixed_precision="no",
        ),
    )


def _uint32_values(path: Path) -> list[int]:
    values = array("I")
    with path.open("rb") as handle:
        values.fromfile(handle, path.stat().st_size // values.itemsize)
    return list(values)


def test_qwen_record_has_explicit_year_and_canonical_single_line_fields() -> None:
    text = format_qwen_entry(
        Entry(2030, " future\tcrumb ", "two\nlines", "  use   it ")
    )

    assert text == (
        "<ENTRY>\nYEAR: 2030\nWORD: future crumb\nDEFINITION: two lines\n"
        "EXAMPLE: use it\n</ENTRY>\n"
    )


def test_packing_accepts_prepared_integer_years_and_keeps_splits_separate(
    tmp_path: Path,
) -> None:
    source = tmp_path / "prepared.jsonl"
    rows = [
        {"year": 2022, "word": "train", "definition": "d", "example": "e"},
        {"year": 2024, "word": "validation", "definition": "d", "example": "e"},
        {"year": 2025, "word": "test", "definition": "d", "example": "e"},
    ]
    source.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    packed = tmp_path / "packed"
    config = _benchmark_config(source, packed)

    stats = pack_corpus(config, tokenizer=YearTokenizer())

    assert stats.seen == 3
    assert {name: split.entries for name, split in stats.splits.items()} == {
        "train": 1,
        "validation": 1,
        "test": 1,
    }
    assert _uint32_values(packed / "train.bin") == [122, 99]
    assert _uint32_values(packed / "validation.bin") == [124, 99]
    assert _uint32_values(packed / "test.bin") == [125, 99]


@pytest.mark.parametrize(("key", "value"), [("year", 2022.0), ("year", "2022.0"), ("Year", 2022)])
def test_packing_treats_explicit_integer_valued_years_as_calendar_years(
    tmp_path: Path, key: str, value: object
) -> None:
    source = tmp_path / "prepared.jsonl"
    rows = [
        {key: value, "word": "train", "definition": "d"},
        {"year": 2024, "word": "validation", "definition": "d"},
        {"year": 2025, "word": "test", "definition": "d"},
    ]
    source.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    packed = tmp_path / "packed"

    pack_corpus(_benchmark_config(source, packed), tokenizer=YearTokenizer())

    assert _uint32_values(packed / "train.bin") == [122, 99]


def test_packed_dataset_masks_only_the_partial_block_tail(tmp_path: Path) -> None:
    path = tmp_path / "tokens.bin"
    values = array("I", [5, 6, 7])
    path.write_bytes(values.tobytes())
    dataset = PackedTokenDataset(path, token_count=3, context_length=4, pad_id=0)

    sample = dataset[0]

    assert torch.equal(sample["input_ids"], torch.tensor([5, 6, 7, 0]))
    assert torch.equal(sample["attention_mask"], torch.tensor([1, 1, 1, 0]))
    assert torch.equal(sample["labels"], torch.tensor([5, 6, 7, -100]))


def test_publication_can_retry_without_retokenizing_after_symlink_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "prepared.jsonl"
    source.write_text(
        "".join(
            json.dumps({"year": year, "word": str(year), "definition": "d"}) + "\n"
            for year in (2022, 2024, 2025)
        ),
        encoding="utf-8",
    )
    packed = tmp_path / "packed"
    config = _benchmark_config(source, packed)
    original_symlink = qwen_data.os.symlink
    failed = False

    def fail_once(target: str, link: Path, *, target_is_directory: bool = False) -> None:
        nonlocal failed
        if not failed:
            failed = True
            raise OSError("injected publication failure")
        original_symlink(target, link, target_is_directory=target_is_directory)

    monkeypatch.setattr(qwen_data.os, "symlink", fail_once)
    tokenizer = CountingYearTokenizer()
    with pytest.raises(OSError, match="injected"):
        pack_corpus(config, tokenizer=tokenizer)
    assert not packed.exists()
    assert list(tmp_path.glob(".packed.complete.*"))
    first_calls = tokenizer.calls

    pack_corpus(config, tokenizer=tokenizer)

    assert tokenizer.calls == first_calls
    assert packed.is_symlink()
    assert packed.resolve().is_dir()


def test_unknown_destination_is_preserved(tmp_path: Path) -> None:
    source = tmp_path / "prepared.jsonl"
    source.write_text(
        json.dumps({"year": 2022, "word": "train", "definition": "d"}) + "\n",
        encoding="utf-8",
    )
    packed = tmp_path / "packed"
    packed.write_text("user data", encoding="utf-8")

    with pytest.raises(ValueError, match="not a directory"):
        pack_corpus(_benchmark_config(source, packed), tokenizer=YearTokenizer())
    assert packed.read_text(encoding="utf-8") == "user data"


def test_manifest_binds_packing_implementation_provenance(tmp_path: Path) -> None:
    source = tmp_path / "prepared.jsonl"
    source.write_text(
        "".join(
            json.dumps({"year": year, "word": str(year), "definition": "d"}) + "\n"
            for year in (2022, 2024, 2025)
        ),
        encoding="utf-8",
    )
    packed = tmp_path / "packed"
    config = _benchmark_config(source, packed)
    pack_corpus(config, tokenizer=YearTokenizer())
    manifest_path = packed / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["packing_provenance"]["implementation_sha256"] = "changed"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="provenance"):
        pack_corpus(config, tokenizer=YearTokenizer())


def test_training_load_rejects_self_reported_tokenizer_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "prepared.jsonl"
    source.write_text(
        "".join(
            json.dumps({"year": year, "word": str(year), "definition": "d"}) + "\n"
            for year in (2022, 2024, 2025)
        ),
        encoding="utf-8",
    )
    packed = tmp_path / "packed"
    config = _benchmark_config(source, packed)
    pack_corpus(config, tokenizer=YearTokenizer())
    manifest_path = packed / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["tokenizer"]["eos_token_id"] = 98
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    monkeypatch.setattr(
        qwen_data.AutoTokenizer,
        "from_pretrained",
        lambda *_args, **_kwargs: YearTokenizer(),
    )

    with pytest.raises(ValueError, match="provenance"):
        load_packed_datasets(config)
