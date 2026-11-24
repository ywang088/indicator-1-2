#!/usr/bin/env bash
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
"${PARKING_PYTHON:-python3}" "$root/code/fresh_build.py" "$@"
