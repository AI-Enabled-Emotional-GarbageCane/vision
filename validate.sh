#!/usr/bin/env sh
set -eu

PYTHON_BIN="${PYTHON:-python3}"

"$PYTHON_BIN" scripts/validate-vision-contract.py
PYTHONPATH=src "$PYTHON_BIN" tests/smoke_stub_inference.py
