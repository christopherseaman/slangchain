#!/usr/bin/env bash
set -euo pipefail

readonly QWEN_SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# Reuse the checksum-pinned, no-overwrite download and extraction path.
source "$QWEN_SCRIPT_DIR/demo.sh"

readonly QWEN_CONFIG_PATH="${QWEN_CONFIG_PATH:-configs/qwen35-0.8b.toml}"
readonly QWEN_FUTURE_YEAR="${1:-2030}"
readonly QWEN_SAMPLE_COUNT="${2:-10}"

main_qwen() {
  ensure_database
  cd "$PROJECT_DIR"

  echo "Packing every valid record into chronological 2048-token Qwen blocks"
  uv run slangchain qwen-pack --config "$QWEN_CONFIG_PATH"

  echo "Continuing Qwen3.5-0.8B pretraining; complete checkpoints resume automatically"
  uv run slangchain qwen-train --config "$QWEN_CONFIG_PATH" --resume auto

  uv run slangchain qwen-generate \
    --config "$QWEN_CONFIG_PATH" \
    --year "$QWEN_FUTURE_YEAR" \
    --samples "$QWEN_SAMPLE_COUNT"
}

main_qwen
