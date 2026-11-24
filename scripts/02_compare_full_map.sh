#!/usr/bin/env bash
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
"${INDICATOR_PYTHON:-python3}" "$root/code/delivery_runner.py" compare
