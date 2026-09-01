from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from slangchain.data import Entry, load_entries, prepare_dataset, split_entries


def test_prepare_jsonl_samples_each_year_and_rejects_missing_dates(tmp_path: Path) -> None:
    source = tmp_path / "raw.jsonl"
    rows = [
        {"written_on": f"2020-01-0{day}T00:00:00Z", "word": f"w{day}", "definition": "d"}
        for day in range(1, 4)
    ]
    rows.extend(
        [
            {"written_on": "2021-02-01", "word": "next", "definition": "later"},
            {"word": "undated", "definition": "must be rejected"},
        ]
    )
    source.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    output = tmp_path / "prepared.jsonl"

    stats = prepare_dataset(source, output, max_per_year=2, seed=7)

    assert stats.seen == 5
    assert stats.rejected == 1
    assert stats.written == 3
    assert stats.by_year == {2020: 2, 2021: 1}
    assert len(load_entries(output)) == 3


def test_prepare_reads_the_expected_sqlite_shape(tmp_path: Path) -> None:
    source = tmp_path / "urban.sqlite"
    with sqlite3.connect(source) as connection:
        connection.execute(
            "CREATE TABLE ud (word TEXT, definition TEXT, example TEXT, written_on TEXT)"
        )
        connection.execute(
            "INSERT INTO ud VALUES (?, ?, ?, ?)",
            ("future crumb", "A favor for your future self", "Pack lunch early", "2025-04-07"),
        )
    output = tmp_path / "prepared.jsonl"

    stats = prepare_dataset(source, output)

    assert stats.written == 1
    assert load_entries(output) == [
        Entry(2025, "future crumb", "A favor for your future self", "Pack lunch early")
    ]


def test_prepare_refuses_to_overwrite(tmp_path: Path) -> None:
    source = tmp_path / "raw.jsonl"
    source.write_text(json.dumps({"date": "2020-01-01", "word": "x", "definition": "y"}) + "\n")
    output = tmp_path / "existing.jsonl"
    output.write_text("user data")

    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        prepare_dataset(source, output)

    assert output.read_text() == "user data"


@pytest.mark.parametrize(
    ("raw_date", "expected_year"),
    [("1609459200", 2021), ("1609459200000", 2021)],
)
def test_prepare_accepts_timestamp_strings(
    tmp_path: Path, raw_date: str, expected_year: int
) -> None:
    source = tmp_path / "raw.jsonl"
    source.write_text(json.dumps({"timestamp": raw_date, "word": "x", "definition": "y"}) + "\n")
    output = tmp_path / "prepared.jsonl"

    prepare_dataset(source, output)

    assert load_entries(output)[0].year == expected_year


def test_prepare_rejects_a_malformed_date(tmp_path: Path) -> None:
    source = tmp_path / "raw.jsonl"
    source.write_text(
        json.dumps({"date": "2020-not-a-date", "word": "x", "definition": "y"}) + "\n"
    )

    with pytest.raises(ValueError, match="No valid dated entries"):
        prepare_dataset(source, tmp_path / "prepared.jsonl")


def test_prepare_rejects_an_undated_schema_immediately(tmp_path: Path) -> None:
    source = tmp_path / "raw.csv"
    source.write_text("word,definition\nx,y\n")

    with pytest.raises(ValueError, match="missing required field groups: date"):
        prepare_dataset(source, tmp_path / "prepared.jsonl")


def test_prepare_rejects_an_undated_json_row_but_keeps_later_rows(tmp_path: Path) -> None:
    source = tmp_path / "raw.jsonl"
    rows = [
        {"word": "undated", "definition": "reject this row"},
        {"date": "2024-01-02", "word": "dated", "definition": "keep this row"},
    ]
    source.write_text("".join(json.dumps(row) + "\n" for row in rows))
    output = tmp_path / "prepared.jsonl"

    stats = prepare_dataset(source, output)

    assert stats.rejected == 1
    assert load_entries(output) == [Entry(2024, "dated", "keep this row")]


def test_prepare_accepts_negative_numeric_milliseconds(tmp_path: Path) -> None:
    source = tmp_path / "raw.jsonl"
    source.write_text(
        json.dumps({"timestamp": -315_619_200_000, "word": "past", "definition": "from 1960"})
        + "\n"
    )
    output = tmp_path / "prepared.jsonl"

    prepare_dataset(source, output)

    assert load_entries(output)[0].year == 1960


def test_split_entries_is_chronological() -> None:
    entries = [Entry(year, str(year), "definition") for year in range(2020, 2025)]

    train, validation, test = split_entries(entries, 2022, 2023)

    assert [entry.year for entry in train] == [2020, 2021, 2022]
    assert [entry.year for entry in validation] == [2023]
    assert [entry.year for entry in test] == [2024]
