#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
export MPLCONFIGDIR="$PWD/.matplotlib"
export JUPYTER_PATH="$PWD/.jupyter${JUPYTER_PATH:+:$JUPYTER_PATH}"
mkdir -p "$MPLCONFIGDIR"
exec .venv/bin/python -m jupyterlab "$@"
