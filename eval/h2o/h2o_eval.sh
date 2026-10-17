#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Usage: ./eval/h2o/h2o_eval.sh [MODEL_DIR] [--check] [options]
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -z "${PYTHON:-}" ]]; then
    PYTHON="$SCRIPT_DIR/../../.venv/bin/python"
    [[ -x "$PYTHON" ]] || PYTHON=python3
fi
exec "$PYTHON" -u "$SCRIPT_DIR/h2o_eval.py" "$@"
