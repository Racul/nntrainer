#!/usr/bin/env bash
# Usage: ./eval/android_snapkv_eval.sh [check|sweep] [options]
# Uses the same Python environment and measurement protocol as the baseline.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -z "${PYTHON:-}" ]]; then
    PYTHON="$SCRIPT_DIR/../.venv/bin/python"
    [[ -x "$PYTHON" ]] || PYTHON=python3
fi
exec "$PYTHON" -u "$SCRIPT_DIR/android_snapkv_eval.py" "$@"
