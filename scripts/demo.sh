#!/usr/bin/env bash
set -euo pipefail

readonly DATA_URL="https://data.badmath.org/UrbanDictionary_04-2026.7z"
readonly ARCHIVE_SHA256="90543ec3ecf4bb56e2c774a474467bb5345a8aa185474d8f08ad6c7c779bae6f"
readonly DATABASE_SHA256="6e919a5761a641782e108dfc0e72c61b3e23efcd482030bc2a670774c566f5ce"
readonly ARCHIVE_NAME="UrbanDictionary_04-2026.7z"
readonly DATABASE_NAME="UrbanDictionary_04-2026.db"

readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
readonly PROJECT_DIR="$(cd -- "$SCRIPT_DIR/.." && pwd)"
readonly RAW_DIR="$PROJECT_DIR/data/raw"
readonly ARCHIVE_PATH="$RAW_DIR/$ARCHIVE_NAME"
readonly DATABASE_PATH="$RAW_DIR/$DATABASE_NAME"
readonly PREPARED_PATH="$PROJECT_DIR/data/processed/entries.jsonl"
readonly PREPARED_MANIFEST_PATH="$PROJECT_DIR/data/processed/entries.manifest"
readonly CONFIG_PATH="$PROJECT_DIR/configs/tiny.toml"
readonly TRAINING_DIR="$PROJECT_DIR/artifacts/tiny"
readonly CHECKPOINT_DIR="$PROJECT_DIR/artifacts/tiny/best"
readonly REPORT_PATH="$TRAINING_DIR/report.json"
readonly TRAINING_MANIFEST_PATH="$TRAINING_DIR/demo.manifest"
readonly LOCK_DIR="$PROJECT_DIR/.demo.lock"

readonly FUTURE_YEAR="${1:-2030}"
readonly SAMPLE_COUNT="${2:-10}"
readonly MAX_PER_YEAR="${MAX_PER_YEAR:-20000}"
readonly MAX_CHARS=1000
readonly PREPARATION_SEED=42

DOWNLOAD_TEMP=""
EXTRACTION_TEMP=""
LOCK_ACQUIRED=0
PREPARED_SHA256=""

sha256_file() {
  if command -v shasum >/dev/null 2>&1; then
    shasum -a 256 "$1" | awk '{print $1}'
  elif command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$1" | awk '{print $1}'
  else
    echo "Need shasum or sha256sum to verify the dataset archive." >&2
    return 1
  fi
}

validate_database() {
  uv run --quiet python - "$1" <<'PY'
import sqlite3
import sys
from pathlib import Path

path = Path(sys.argv[1]).resolve()
with sqlite3.connect(f"{path.as_uri()}?mode=ro", uri=True) as connection:
    integrity = connection.execute("PRAGMA quick_check(1)").fetchone()[0]
    if integrity != "ok":
        raise SystemExit(f"SQLite integrity check failed: {integrity}")
    columns = {
        row[1].lower() for row in connection.execute('PRAGMA table_info("ud")')
    }
    missing = {"word", "definition", "written_on"} - columns
    if missing:
        raise SystemExit(f"SQLite table 'ud' is missing columns: {sorted(missing)}")
PY
  verify_database_checksum "$1"
}

verify_database_checksum() {
  local actual_sha256
  actual_sha256="$(sha256_file "$1")"
  if [[ "$actual_sha256" != "$DATABASE_SHA256" ]]; then
    echo "SQLite database checksum mismatch: $1" >&2
    return 1
  fi
}

cleanup() {
  if [[ -n "$DOWNLOAD_TEMP" && "$DOWNLOAD_TEMP" == "$RAW_DIR"/.download.* ]]; then
    rm -f -- "$DOWNLOAD_TEMP"
    DOWNLOAD_TEMP=""
  fi
  if [[ -n "$EXTRACTION_TEMP" && "$EXTRACTION_TEMP" == "$RAW_DIR"/.extract.* ]]; then
    rm -rf -- "$EXTRACTION_TEMP"
    EXTRACTION_TEMP=""
  fi
  if [[ "$LOCK_ACQUIRED" == 1 ]]; then
    rmdir "$LOCK_DIR" 2>/dev/null || true
    LOCK_ACQUIRED=0
  fi
}

handle_signal() {
  local signal_name="$1"
  cleanup
  trap - "$signal_name"
  kill -s "$signal_name" "$$"
}

acquire_lock() {
  if ! mkdir "$LOCK_DIR"; then
    echo "Another demo run, or a stale lock, exists at $LOCK_DIR" >&2
    echo "Confirm no demo is running before removing a stale lock directory." >&2
    return 1
  fi
  LOCK_ACQUIRED=1
  trap cleanup EXIT
  trap 'handle_signal HUP' HUP
  trap 'handle_signal INT' INT
  trap 'handle_signal TERM' TERM
}

install_file() {
  local source_path="$1"
  local destination_path="$2"
  if ! uv run --quiet python - "$source_path" "$destination_path" <<'PY'
import os
import sys

os.link(sys.argv[1], sys.argv[2])
PY
  then
    echo "Refusing to replace an output created concurrently: $destination_path" >&2
    return 1
  fi
}

download_archive() {
  DOWNLOAD_TEMP="$(mktemp "$RAW_DIR/.download.XXXXXX")"
  echo "Downloading $DATA_URL"
  curl \
    --fail \
    --location \
    --progress-bar \
    --retry 3 \
    --output "$DOWNLOAD_TEMP" \
    "$DATA_URL"

  local actual_sha256
  actual_sha256="$(sha256_file "$DOWNLOAD_TEMP")"
  if [[ "$actual_sha256" != "$ARCHIVE_SHA256" ]]; then
    echo "Archive checksum mismatch." >&2
    return 1
  fi
  install_file "$DOWNLOAD_TEMP" "$ARCHIVE_PATH"
  rm -f -- "$DOWNLOAD_TEMP"
  DOWNLOAD_TEMP=""
}

verify_archive() {
  local actual_sha256
  actual_sha256="$(sha256_file "$ARCHIVE_PATH")"
  if [[ "$actual_sha256" != "$ARCHIVE_SHA256" ]]; then
    echo "Archive checksum mismatch: $ARCHIVE_PATH" >&2
    return 1
  fi
}

extract_database() {
  EXTRACTION_TEMP="$(mktemp -d "$RAW_DIR/.extract.XXXXXX")"
  echo "Extracting $ARCHIVE_PATH"

  if command -v bsdtar >/dev/null 2>&1; then
    bsdtar -xf "$ARCHIVE_PATH" -C "$EXTRACTION_TEMP"
  elif command -v 7zz >/dev/null 2>&1; then
    7zz x "$ARCHIVE_PATH" "-o$EXTRACTION_TEMP"
  elif command -v 7z >/dev/null 2>&1; then
    7z x "$ARCHIVE_PATH" "-o$EXTRACTION_TEMP"
  else
    echo "Need bsdtar, 7zz, or 7z to extract the dataset." >&2
    return 1
  fi

  local extracted_path="$EXTRACTION_TEMP/$DATABASE_NAME"
  if [[ ! -s "$extracted_path" ]]; then
    echo "Archive did not contain the expected database: $DATABASE_NAME" >&2
    return 1
  fi
  echo "Validating extracted SQLite database"
  if ! validate_database "$extracted_path"; then
    return 1
  fi
  install_file "$extracted_path" "$DATABASE_PATH"
  rm -f -- "$extracted_path"
  rmdir "$EXTRACTION_TEMP"
  EXTRACTION_TEMP=""
}

prepared_manifest_content() {
  printf '%s\n' \
    "format=1" \
    "archive_sha256=$ARCHIVE_SHA256" \
    "database_name=$DATABASE_NAME" \
    "database_sha256=$DATABASE_SHA256" \
    "max_per_year=$MAX_PER_YEAR" \
    "max_chars=$MAX_CHARS" \
    "seed=$PREPARATION_SEED" \
    "prepared_sha256=$PREPARED_SHA256"
}

training_manifest_content() {
  local config_sha256="$1"
  local weights_sha256="$2"
  local metadata_sha256="$3"
  local report_sha256="$4"
  printf '%s\n' \
    "format=1" \
    "prepared_sha256=$PREPARED_SHA256" \
    "config_sha256=$config_sha256" \
    "weights_sha256=$weights_sha256" \
    "metadata_sha256=$metadata_sha256" \
    "report_sha256=$report_sha256"
}

install_manifest() {
  local destination_path="$1"
  local content="$2"
  local temporary_path
  temporary_path="$(mktemp "$destination_path.tmp.XXXXXX")"
  printf '%s\n' "$content" >"$temporary_path"
  if ! install_file "$temporary_path" "$destination_path"; then
    rm -f -- "$temporary_path"
    return 1
  fi
  rm -f -- "$temporary_path"
}

verify_prepared_data() {
  if [[ ! -f "$PREPARED_MANIFEST_PATH" ]]; then
    echo "Prepared data has no provenance manifest: $PREPARED_PATH" >&2
    echo "Move both prepared files aside and rerun." >&2
    return 1
  fi
  PREPARED_SHA256="$(sha256_file "$PREPARED_PATH")"
  if [[ "$(cat "$PREPARED_MANIFEST_PATH")" != "$(prepared_manifest_content)" ]]; then
    echo "Prepared data does not match this archive or preparation settings." >&2
    echo "Move $PREPARED_PATH and $PREPARED_MANIFEST_PATH aside, then rerun." >&2
    return 1
  fi
}

training_is_complete() {
  [[ -f "$CHECKPOINT_DIR/model.safetensors" && \
     -f "$CHECKPOINT_DIR/metadata.json" && \
     -f "$REPORT_PATH" ]]
}

verify_training_run() {
  local config_sha256
  local weights_sha256
  local metadata_sha256
  local report_sha256
  config_sha256="$(sha256_file "$CONFIG_PATH")"
  weights_sha256="$(sha256_file "$CHECKPOINT_DIR/model.safetensors")"
  metadata_sha256="$(sha256_file "$CHECKPOINT_DIR/metadata.json")"
  report_sha256="$(sha256_file "$REPORT_PATH")"
  if [[ ! -f "$TRAINING_MANIFEST_PATH" ]]; then
    echo "Training artifacts have no provenance manifest: $TRAINING_DIR" >&2
    echo "Move the training directory aside and rerun." >&2
    return 1
  fi
  if [[ "$(cat "$TRAINING_MANIFEST_PATH")" != \
        "$(training_manifest_content \
          "$config_sha256" \
          "$weights_sha256" \
          "$metadata_sha256" \
          "$report_sha256")" ]]; then
    echo "Training artifacts do not match the prepared data or current config." >&2
    echo "Move $TRAINING_DIR aside and rerun." >&2
    return 1
  fi
}

ensure_database() {
  command -v curl >/dev/null 2>&1 || {
    echo "curl is required." >&2
    return 1
  }
  command -v uv >/dev/null 2>&1 || {
    echo "uv is required: https://docs.astral.sh/uv/" >&2
    return 1
  }
  mkdir -p "$RAW_DIR" "$PROJECT_DIR/data/processed" "$PROJECT_DIR/artifacts"
  acquire_lock
  cd "$PROJECT_DIR"

  if [[ ! -f "$DATABASE_PATH" ]]; then
    if [[ ! -f "$ARCHIVE_PATH" ]]; then
      download_archive
    else
      echo "Using existing archive $ARCHIVE_PATH"
    fi
    verify_archive
    extract_database
  else
    verify_database_checksum "$DATABASE_PATH"
    echo "Using existing database $DATABASE_PATH"
  fi
}

main() {
  ensure_database

  if [[ ! -f "$PREPARED_PATH" ]]; then
    if [[ -e "$PREPARED_MANIFEST_PATH" || -L "$PREPARED_MANIFEST_PATH" ]]; then
      echo "Prepared manifest exists without its data: $PREPARED_MANIFEST_PATH" >&2
      return 1
    fi
    echo "Validating SQLite database before preparation"
    validate_database "$DATABASE_PATH"
    uv run slangchain prepare "$DATABASE_PATH" \
      --output "$PREPARED_PATH" \
      --max-per-year "$MAX_PER_YEAR" \
      --max-chars "$MAX_CHARS" \
      --seed "$PREPARATION_SEED"
    PREPARED_SHA256="$(sha256_file "$PREPARED_PATH")"
    install_manifest "$PREPARED_MANIFEST_PATH" "$(prepared_manifest_content)"
  else
    verify_prepared_data
    echo "Using existing prepared data $PREPARED_PATH"
  fi

  if ! training_is_complete; then
    if [[ -d "$TRAINING_DIR" ]] && \
       [[ -n "$(find "$TRAINING_DIR" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
      echo "Training directory exists but is incomplete: $TRAINING_DIR" >&2
      echo "Move it aside and rerun; the script will not replace it." >&2
      return 1
    fi
    local config_sha256_before
    local prepared_sha256_before
    config_sha256_before="$(sha256_file "$CONFIG_PATH")"
    prepared_sha256_before="$PREPARED_SHA256"
    uv run slangchain train --config configs/tiny.toml
    if ! training_is_complete; then
      echo "Training exited without a complete checkpoint and report." >&2
      return 1
    fi
    if [[ "$(sha256_file "$CONFIG_PATH")" != "$config_sha256_before" || \
          "$(sha256_file "$PREPARED_PATH")" != "$prepared_sha256_before" ]]; then
      echo "Config or prepared data changed during training; no run manifest was published." >&2
      return 1
    fi
    install_manifest \
      "$TRAINING_MANIFEST_PATH" \
      "$(training_manifest_content \
        "$config_sha256_before" \
        "$(sha256_file "$CHECKPOINT_DIR/model.safetensors")" \
        "$(sha256_file "$CHECKPOINT_DIR/metadata.json")" \
        "$(sha256_file "$REPORT_PATH")")"
  else
    verify_training_run
    echo "Using existing checkpoint $CHECKPOINT_DIR"
  fi

  uv run slangchain generate \
    --checkpoint "$CHECKPOINT_DIR" \
    --year "$FUTURE_YEAR" \
    --samples "$SAMPLE_COUNT"
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  main "$@"
fi
