from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import os
import shutil
import sys
import tempfile
from array import array
from collections import Counter
from dataclasses import asdict, dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Protocol

import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

from . import data as source_data
from .data import Entry, iter_records, normalize_entry
from .qwen_config import QwenExperimentConfig

PACKED_FORMAT_VERSION = 2
SPLITS = ("train", "validation", "test")


class Tokenizer(Protocol):
    eos_token_id: int | None
    pad_token_id: int | None
    vocab_size: int
    init_kwargs: dict[str, Any]

    def __call__(
        self, texts: list[str], *, add_special_tokens: bool
    ) -> dict[str, list[list[int]]]: ...

    def __len__(self) -> int: ...


@dataclass(frozen=True)
class SplitStats:
    entries: int
    tokens: int
    blocks: int
    sha256: str


@dataclass(frozen=True)
class PackingStats:
    seen: int
    rejected: int
    splits: dict[str, SplitStats]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def format_qwen_entry(entry: Entry) -> str:
    def one_line(text: str) -> str:
        return " ".join(text.split())

    return (
        "<ENTRY>\n"
        f"YEAR: {entry.year}\n"
        f"WORD: {one_line(entry.word)}\n"
        f"DEFINITION: {one_line(entry.definition)}\n"
        f"EXAMPLE: {one_line(entry.example)}\n"
        "</ENTRY>\n"
    )


def _split_name(year: int, train_through: int, validation_through: int) -> str:
    if year <= train_through:
        return "train"
    if year <= validation_through:
        return "validation"
    return "test"


class _TokenWriter:
    def __init__(self, path: Path, tokenizer: Tokenizer, batch_size: int) -> None:
        self.path = path
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.entries = 0
        self.tokens = 0
        self._texts: list[str] = []
        self._handle = path.open("xb")

    def add(self, entry: Entry) -> None:
        self._texts.append(format_qwen_entry(entry))
        self.entries += 1
        if len(self._texts) >= self.batch_size:
            self.flush()

    def flush(self) -> None:
        if not self._texts:
            return
        encoded = self.tokenizer(self._texts, add_special_tokens=False)["input_ids"]
        if len(encoded) != len(self._texts):
            raise RuntimeError("Tokenizer returned the wrong number of records")
        eos_token_id = self.tokenizer.eos_token_id
        if eos_token_id is None:
            raise ValueError("The tokenizer must define eos_token_id")
        values = array("I")
        for record in encoded:
            values.extend(record)
            values.append(eos_token_id)
        if sys.byteorder != "little":
            values.byteswap()
        values.tofile(self._handle)
        self.tokens += len(values)
        self._texts.clear()

    def close(self) -> None:
        self.flush()
        self._handle.flush()
        os.fsync(self._handle.fileno())
        self._handle.close()


def _tokenizer_metadata(tokenizer: Tokenizer, config: QwenExperimentConfig) -> dict[str, Any]:
    return {
        "id": config.data.tokenizer_id,
        "revision": config.data.tokenizer_revision,
        "resolved_revision": tokenizer.init_kwargs.get("_commit_hash"),
        "class": type(tokenizer).__name__,
        "vocab_size": tokenizer.vocab_size,
        "length": len(tokenizer),
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.pad_token_id,
    }


def _implementation_provenance() -> dict[str, Any]:
    digest = hashlib.sha256()
    for path in sorted((Path(__file__).resolve(), Path(source_data.__file__).resolve())):
        digest.update(path.name.encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    versions: dict[str, str] = {
        "python": ".".join(str(part) for part in sys.version_info[:3])
    }
    for package in ("slangchain", "torch", "transformers"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "unknown"
    return {
        "implementation_sha256": digest.hexdigest(),
        "dependencies": versions,
    }


def _manifest_identity(
    config: QwenExperimentConfig,
    source_sha256: str,
    tokenizer: dict[str, Any],
) -> dict[str, Any]:
    return {
        "format_version": PACKED_FORMAT_VERSION,
        "mode": config.mode,
        "source_sha256": source_sha256,
        "context_length": config.model.context_length,
        "train_through": config.data.train_through,
        "validation_through": config.data.validation_through,
        "tokenizer_id": config.data.tokenizer_id,
        "tokenizer_revision": config.data.tokenizer_revision,
        "max_records_per_split": config.data.max_records_per_split,
        "record_format": "<ENTRY>/YEAR/WORD/DEFINITION/EXAMPLE/text-eos",
        "tokenizer": tokenizer,
        "packing_provenance": _implementation_provenance(),
    }


def _read_manifest(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Invalid packed manifest: {path}")
    return value


def _verify_packed_directory(
    directory: Path,
    config: QwenExperimentConfig,
    tokenizer: Tokenizer | None = None,
) -> dict[str, Any]:
    manifest_path = directory / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Packed corpus is incomplete: {manifest_path}")
    manifest = _read_manifest(manifest_path)
    tokenizer_metadata = (
        _tokenizer_metadata(tokenizer, config)
        if tokenizer is not None
        else manifest.get("tokenizer", {})
    )
    expected_identity = _manifest_identity(
        config, sha256_file(config.data.source), tokenizer_metadata
    )
    actual_identity = {key: manifest.get(key) for key in expected_identity}
    if actual_identity != expected_identity:
        raise ValueError(
            f"Packed corpus provenance does not match the config or source: {directory}"
        )
    for split in SPLITS:
        path = directory / f"{split}.bin"
        details = manifest.get("splits", {}).get(split, {})
        if not path.is_file() or sha256_file(path) != details.get("sha256"):
            raise ValueError(f"Packed {split} data is missing or changed: {path}")
        if path.stat().st_size != details.get("tokens", -1) * 4:
            raise ValueError(f"Packed {split} token count does not match its file size")
    return manifest


def verify_packed_corpus(config: QwenExperimentConfig) -> dict[str, Any]:
    return _verify_packed_directory(config.data.packed_dir, config)


def _packing_stats(manifest: dict[str, Any]) -> PackingStats:
    return PackingStats(
        seen=manifest["seen"],
        rejected=manifest["rejected"],
        splits={split: SplitStats(**manifest["splits"][split]) for split in SPLITS},
    )


def _normalize_source_entry(raw: dict[str, Any]) -> Entry:
    lowered = {str(key).lower(): key for key in raw}
    if "year" in lowered and not set(lowered).intersection(
        {"written_on", "created_at", "created", "date", "timestamp"}
    ):
        value = raw[lowered["year"]]
        if isinstance(value, bool):
            raise ValueError("invalid explicit year")
        try:
            decimal_year = Decimal(str(value).strip())
        except (InvalidOperation, ValueError) as error:
            raise ValueError("invalid explicit year") from error
        if not decimal_year.is_finite() or decimal_year != decimal_year.to_integral_value():
            raise ValueError("explicit year must be an integer")
        year = int(decimal_year)
        if not 1900 <= year <= 2100:
            raise ValueError("explicit year is outside the supported range")
        raw = {**raw, "date": str(year)}
    return normalize_entry(raw, max_chars=None)


def pack_corpus(
    config: QwenExperimentConfig,
    *,
    tokenizer: Tokenizer | None = None,
) -> PackingStats:
    source = config.data.source
    destination = config.data.packed_dir
    if not source.is_file():
        raise FileNotFoundError(source)
    source_sha256 = sha256_file(source)

    tokenizer = tokenizer or AutoTokenizer.from_pretrained(
        config.data.tokenizer_id,
        revision=config.data.tokenizer_revision,
    )
    if tokenizer.eos_token_id is None or tokenizer.pad_token_id is None:
        raise ValueError("The tokenizer must define EOS and padding token IDs")

    # A destination may be a published symlink.  Validate it before looking for
    # an interrupted publication, while leaving unknown paths untouched.
    if os.path.lexists(destination):
        if destination.is_dir():
            return _packing_stats(_verify_packed_directory(destination, config, tokenizer))
        raise ValueError(f"Packed destination exists and is not a directory: {destination}")

    # A completed hidden directory is retained when publication itself fails.
    # Reusing it makes publication retryable without invoking the tokenizer.
    completed_prefix = f".{destination.name}.complete."
    completed: list[Path] = sorted(
        path
        for path in destination.parent.glob(f"{completed_prefix}*")
        if path.is_dir() and not path.is_symlink()
    )
    for candidate in completed:
        try:
            manifest = _verify_packed_directory(candidate, config, tokenizer)
        except (FileNotFoundError, ValueError, KeyError, TypeError):
            continue
        try:
            os.symlink(candidate.name, destination, target_is_directory=True)
        except FileExistsError as error:
            if os.path.lexists(destination) and destination.is_dir():
                return _packing_stats(_verify_packed_directory(destination, config, tokenizer))
            raise ValueError(
                f"Packed destination appeared with an incompatible path: {destination}"
            ) from error
        return _packing_stats(manifest)

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.partial.", dir=destination.parent)
    )
    writers = {
        split: _TokenWriter(
            temporary / f"{split}.bin", tokenizer, config.data.tokenizer_batch_size
        )
        for split in SPLITS
    }
    counts: Counter[str] = Counter()
    seen = 0
    rejected = 0
    cap = config.data.max_records_per_split
    try:
        for raw in iter_records(source):
            seen += 1
            try:
                entry = _normalize_source_entry(raw)
            except (TypeError, ValueError, OverflowError):
                rejected += 1
                continue
            split = _split_name(
                entry.year, config.data.train_through, config.data.validation_through
            )
            if cap is not None and counts[split] >= cap:
                if all(counts[name] >= cap for name in SPLITS):
                    break
                continue
            writers[split].add(entry)
            counts[split] += 1

        for writer in writers.values():
            writer.close()
        if any(writers[split].entries == 0 for split in SPLITS):
            raise ValueError(
                "Chronological packing produced an empty split: "
                + ", ".join(f"{name}={writers[name].entries}" for name in SPLITS)
            )

        split_stats = {
            split: SplitStats(
                entries=writers[split].entries,
                tokens=writers[split].tokens,
                blocks=math.ceil(writers[split].tokens / config.model.context_length),
                sha256=sha256_file(temporary / f"{split}.bin"),
            )
            for split in SPLITS
        }
        if sha256_file(source) != source_sha256:
            raise ValueError("Source data changed while it was being packed")
        manifest = {
            **_manifest_identity(
                config, source_sha256, _tokenizer_metadata(tokenizer, config)
            ),
            "source": str(source),
            "seen": seen,
            "rejected": rejected,
            "splits": {name: asdict(stats) for name, stats in split_stats.items()},
        }
        with (temporary / "manifest.json").open("x", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2, sort_keys=True)
            handle.write("\n")

        completed = destination.parent / temporary.name.replace(
            f".{destination.name}.partial.", completed_prefix, 1
        )
        try:
            os.rename(temporary, completed)
        except FileExistsError:
            # Another worker finished the same corpus.  Its immutable directory
            # is safe to reuse; do not overwrite it.
            manifest = _verify_packed_directory(completed, config, tokenizer)
        else:
            temporary = completed
            manifest = _verify_packed_directory(completed, config, tokenizer)
        try:
            os.symlink(completed.name, destination, target_is_directory=True)
        except FileExistsError as error:
            if os.path.lexists(destination) and destination.is_dir():
                return _packing_stats(_verify_packed_directory(destination, config, tokenizer))
            raise ValueError(
                f"Packed destination appeared with an incompatible path: {destination}"
            ) from error
        return PackingStats(seen=seen, rejected=rejected, splits=split_stats)
    finally:
        for writer in writers.values():
            if not writer._handle.closed:
                writer._handle.close()
        if (
            temporary.exists()
            and temporary.parent == destination.parent.resolve()
            and ".complete." not in temporary.name
        ):
            shutil.rmtree(temporary)


class PackedTokenDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(self, path: Path, *, token_count: int, context_length: int, pad_id: int) -> None:
        if token_count <= 1:
            raise ValueError("A packed split must contain at least two tokens")
        if path.stat().st_size != token_count * 4:
            raise ValueError(f"Packed token count does not match file size: {path}")
        self.tokens = torch.from_file(
            str(path), shared=False, size=token_count, dtype=torch.int32
        )
        self.token_count = token_count
        self.context_length = context_length
        self.pad_id = pad_id

    def __len__(self) -> int:
        return math.ceil(self.token_count / self.context_length)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        if not 0 <= index < len(self):
            raise IndexError(index)
        start = index * self.context_length
        stop = min(start + self.context_length, self.token_count)
        valid = stop - start
        input_ids = torch.full((self.context_length,), self.pad_id, dtype=torch.long)
        input_ids[:valid] = self.tokens[start:stop].long()
        attention_mask = torch.zeros(self.context_length, dtype=torch.long)
        attention_mask[:valid] = 1
        labels = input_ids.clone()
        labels[valid:] = -100
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


def load_packed_datasets(
    config: QwenExperimentConfig,
) -> tuple[dict[str, PackedTokenDataset], dict[str, Any]]:
    tokenizer = AutoTokenizer.from_pretrained(
        config.data.tokenizer_id,
        revision=config.data.tokenizer_revision,
    )
    manifest = _verify_packed_directory(config.data.packed_dir, config, tokenizer)
    tokenizer_info = manifest["tokenizer"]
    if tokenizer_info["length"] <= tokenizer_info["eos_token_id"]:
        raise ValueError("Packed tokenizer metadata has an invalid EOS token")
    datasets = {
        split: PackedTokenDataset(
            config.data.packed_dir / f"{split}.bin",
            token_count=manifest["splits"][split]["tokens"],
            context_length=config.model.context_length,
            pad_id=tokenizer_info["pad_token_id"],
        )
        for split in SPLITS
    }
    return datasets, manifest
