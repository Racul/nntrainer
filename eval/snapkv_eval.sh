#!/usr/bin/env bash
# Usage: ./eval/snapkv_eval.sh [MODEL_DIR] [--check] [--resume OUTPUT_DIR] [--contexts 512 1024 2048]
# Optional: PYTHON, BASELINE_BIN, SNAPKV_BIN, EVAL_CPUS, EVAL_OUTPUT.
set -euo pipefail

# 1. 이 파일의 위치를 기준으로 경로를 잡는다.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# 2. 지정한 PYTHON 또는 저장소 안의 .venv를 사용한다.
if [[ -z "${PYTHON:-}" ]]; then
    PYTHON="$SCRIPT_DIR/../.venv/bin/python"
    [[ -x "$PYTHON" ]] || PYTHON=python3
fi
# 3. 실제 측정은 Python 파일에 맡기고 입력 옵션을 그대로 전달한다.
exec "$PYTHON" -u "$SCRIPT_DIR/snapkv_eval.py" "$@"
