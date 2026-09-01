from __future__ import annotations

import csv
import json
import os
import random
import sqlite3
from collections import Counter, defaultdict
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from .model import BOS_ID, EOS_ID, PAD_ID, encode_text

WORD_FIELDS = ("word", "term", "phrase")
DEFINITION_FIELDS = ("definition", "meaning", "description")
EXAMPLE_FIELDS = ("example", "usage")
DATE_FIELDS = ("written_on", "created_at", "created", "date", "timestamp")


@dataclass(frozen=True)
class Entry:
    year: int
    word: str
    definition: str
    example: str = ""


@dataclass(frozen=True)
class PreparationStats:
    seen: int
    written: int
    rejected: int
    by_year: dict[int, int]


def _first_present(record: dict[str, Any], fields: tuple[str, ...]) -> Any:
    lowered = {key.lower(): key for key in record if isinstance(key, str)}
    for field in fields:
        if field in lowered:
            return record[lowered[field]]
    return None


def _parse_year(value: Any) -> int:
    if value is None or value == "":
        raise ValueError("missing date")
    if isinstance(value, bool):
        raise ValueError("invalid date")
    if isinstance(value, (int, float)):
        seconds = float(value)
        if abs(seconds) > 10_000_000_000:
            seconds /= 1000
        year = datetime.fromtimestamp(seconds, tz=UTC).year
    else:
        text = str(value).strip()
        numeric_text = text.removeprefix("-").replace(".", "", 1)
        if text.isdigit() and len(text) == 4:
            year = int(text)
        elif numeric_text.isdigit():
            seconds = float(text)
            if abs(seconds) > 10_000_000_000:
                seconds /= 1000
            year = datetime.fromtimestamp(seconds, tz=UTC).year
        else:
            year = datetime.fromisoformat(text.replace("Z", "+00:00")).year
    if not 1900 <= year <= 2100:
        raise ValueError(f"year {year} is outside the supported range")
    return year


def _clean_text(value: Any, max_chars: int | None) -> str:
    text = str(value or "").replace("\x00", "").replace("\r\n", "\n").replace("\r", "\n")
    text = text.strip()
    return text if max_chars is None else text[:max_chars]


def normalize_entry(record: dict[str, Any], max_chars: int | None) -> Entry:
    word = _clean_text(_first_present(record, WORD_FIELDS), max_chars)
    definition = _clean_text(_first_present(record, DEFINITION_FIELDS), max_chars)
    example = _clean_text(_first_present(record, EXAMPLE_FIELDS), max_chars)
    year = _parse_year(_first_present(record, DATE_FIELDS))
    if not word or not definition:
        raise ValueError("word and definition are required")
    return Entry(year=year, word=word, definition=definition, example=example)


def _validate_schema(record: dict[str, Any]) -> None:
    lowered = {key.lower() for key in record if isinstance(key, str)}
    missing = []
    for label, aliases in (
        ("word", WORD_FIELDS),
        ("definition", DEFINITION_FIELDS),
        ("date", DATE_FIELDS),
    ):
        if not lowered.intersection(aliases):
            missing.append(label)
    if missing:
        raise ValueError(
            f"Input schema is missing required field groups: {', '.join(missing)}. "
            f"Accepted date fields are: {', '.join(DATE_FIELDS)}"
        )


def _iter_json_lines(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"Expected an object on line {line_number}")
            yield value


def _iter_csv(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        _validate_schema(dict.fromkeys(reader.fieldnames or ()))
        yield from reader


def _matching_sqlite_source(connection: sqlite3.Connection) -> tuple[str, list[str]]:
    tables = [
        row[0]
        for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' ORDER BY name"
        )
    ]
    for table in tables:
        quoted_table = table.replace('"', '""')
        columns = [row[1] for row in connection.execute(f'PRAGMA table_info("{quoted_table}")')]
        lowered = {column.lower() for column in columns}
        if (
            lowered.intersection(WORD_FIELDS)
            and lowered.intersection(DEFINITION_FIELDS)
            and lowered.intersection(DATE_FIELDS)
        ):
            return table, columns
    raise ValueError(
        "No SQLite table has word, definition, and date columns. "
        f"Inspected tables: {', '.join(tables) or '(none)'}"
    )


def _iter_sqlite(path: Path) -> Iterator[dict[str, Any]]:
    uri = f"{path.resolve().as_uri()}?mode=ro"
    with sqlite3.connect(uri, uri=True) as connection:
        connection.row_factory = sqlite3.Row
        table, columns = _matching_sqlite_source(connection)
        quoted_table = table.replace('"', '""')
        used_fields = {
            *WORD_FIELDS,
            *DEFINITION_FIELDS,
            *EXAMPLE_FIELDS,
            *DATE_FIELDS,
        }
        columns = [column for column in columns if column.lower() in used_fields]
        quoted_columns = ", ".join(
            f'"{column.replace(chr(34), chr(34) * 2)}"' for column in columns
        )
        for row in connection.execute(f'SELECT {quoted_columns} FROM "{quoted_table}"'):
            yield dict(row)


def iter_records(path: Path) -> Iterator[dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix in {".json", ".jsonl", ".ndjson"}:
        yield from _iter_json_lines(path)
    elif suffix == ".csv":
        yield from _iter_csv(path)
    elif suffix in {".db", ".sqlite", ".sqlite3"}:
        yield from _iter_sqlite(path)
    else:
        raise ValueError(
            f"Unsupported input type {suffix or '(none)'}; expected JSONL, CSV, or SQLite"
        )


def prepare_dataset(
    input_path: Path,
    output_path: Path,
    *,
    max_per_year: int | None = None,
    max_chars: int = 1_000,
    seed: int = 42,
) -> PreparationStats:
    if not input_path.is_file():
        raise FileNotFoundError(input_path)
    if input_path.resolve() == output_path.resolve():
        raise ValueError("Input and output paths must differ")
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output_path}")
    if max_per_year is not None and max_per_year <= 0:
        raise ValueError("max_per_year must be positive")
    if max_chars <= 0:
        raise ValueError("max_chars must be positive")

    seen = 0
    rejected = 0
    counts: Counter[int] = Counter()
    randomizer = random.Random(seed)
    reservoirs: dict[int, list[Entry]] = defaultdict(list)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(f".{output_path.name}.partial")
    if temporary_path.exists():
        raise FileExistsError(f"Refusing to overwrite partial output: {temporary_path}")

    try:
        with temporary_path.open("x", encoding="utf-8") as output:
            for raw in iter_records(input_path):
                seen += 1
                try:
                    entry = normalize_entry(raw, max_chars)
                except (TypeError, ValueError, OverflowError):
                    rejected += 1
                    continue

                counts[entry.year] += 1
                if max_per_year is None:
                    output.write(json.dumps(asdict(entry), ensure_ascii=False) + "\n")
                    continue

                bucket = reservoirs[entry.year]
                if len(bucket) < max_per_year:
                    bucket.append(entry)
                else:
                    candidate = randomizer.randrange(counts[entry.year])
                    if candidate < max_per_year:
                        bucket[candidate] = entry

            if max_per_year is not None:
                for year in sorted(reservoirs):
                    for entry in reservoirs[year]:
                        output.write(json.dumps(asdict(entry), ensure_ascii=False) + "\n")

        if not counts:
            raise ValueError("No valid dated entries were found")
        os.link(temporary_path, output_path)
        temporary_path.unlink()
    except BaseException:
        if temporary_path.exists():
            temporary_path.unlink()
        raise

    written_by_year = (
        {year: min(count, max_per_year) for year, count in counts.items()}
        if max_per_year is not None
        else dict(counts)
    )
    return PreparationStats(
        seen=seen,
        written=sum(written_by_year.values()),
        rejected=rejected,
        by_year=dict(sorted(written_by_year.items())),
    )


def load_entries(path: Path) -> list[Entry]:
    entries: list[Entry] = []
    for raw in _iter_json_lines(path):
        try:
            entry = Entry(
                year=int(raw["year"]),
                word=str(raw["word"]).strip(),
                definition=str(raw["definition"]).strip(),
                example=str(raw.get("example", "")).strip(),
            )
            if not 1900 <= entry.year <= 2100 or not entry.word or not entry.definition:
                raise ValueError("prepared entry violates required invariants")
            entries.append(entry)
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"Invalid prepared entry at index {len(entries)}") from error
    if not entries:
        raise ValueError(f"No entries found in {path}")
    return entries


def split_entries(
    entries: list[Entry], train_through: int, validation_through: int
) -> tuple[list[Entry], list[Entry], list[Entry]]:
    train = [entry for entry in entries if entry.year <= train_through]
    validation = [entry for entry in entries if train_through < entry.year <= validation_through]
    test = [entry for entry in entries if entry.year > validation_through]
    if not train or not validation or not test:
        sizes = (len(train), len(validation), len(test))
        raise ValueError(
            "Chronological split produced an empty partition "
            f"(train={sizes[0]}, validation={sizes[1]}, test={sizes[2]})"
        )
    return train, validation, test


def format_entry(entry: Entry) -> str:
    parts = [f"WORD: {entry.word}", f"DEFINITION: {entry.definition}"]
    if entry.example:
        parts.append(f"EXAMPLE: {entry.example}")
    return "\n".join(parts) + "\n"


class EntryDataset(Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]):
    def __init__(self, entries: list[Entry], context_length: int) -> None:
        self.entries = entries
        self.context_length = context_length

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        entry = self.entries[index]
        sequence = [BOS_ID, *encode_text(format_entry(entry)), EOS_ID]
        sample_length = self.context_length + 1
        if len(sequence) > sample_length:
            sequence = [*sequence[: sample_length - 1], EOS_ID]
        sequence.extend([PAD_ID] * (sample_length - len(sequence)))
        values = torch.tensor(sequence, dtype=torch.long)
        return values[:-1], values[1:], torch.tensor(entry.year, dtype=torch.float32)
