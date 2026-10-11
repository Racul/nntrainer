#!/usr/bin/env bash
# Usage: ./eval/android_baseline_eval.sh [check|run|memory|both|sweep] [options]
# No installation or build is performed. Optional interpreter override: PYTHON.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -z "${PYTHON:-}" ]]; then
    PYTHON="$SCRIPT_DIR/../.venv/bin/python"
    [[ -x "$PYTHON" ]] || PYTHON=python3
fi
exec "$PYTHON" -u "$SCRIPT_DIR/android_baseline_eval.py" "$@"
