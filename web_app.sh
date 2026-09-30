#!/usr/bin/env bash
set -euo pipefail

readonly PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
readonly WEB_HOST="${WEB_HOST:-0.0.0.0}"
cd "$PROJECT_DIR"
exec uv run --with 'gradio>=6,<7' python scripts/web_app.py --host "$WEB_HOST" "$@"
