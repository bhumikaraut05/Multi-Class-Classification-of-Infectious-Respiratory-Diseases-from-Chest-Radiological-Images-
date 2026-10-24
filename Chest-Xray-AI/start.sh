#!/usr/bin/env bash
# One-command start (Linux/macOS). First time: ./start.sh --setup   then: ./start.sh
cd "$(dirname "$0")" || exit 1
PY=$(command -v python3 || command -v python) || { echo "Python 3.10-3.12 not found."; exit 1; }
exec "$PY" run.py "$@"
