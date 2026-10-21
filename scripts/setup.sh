#!/bin/sh
set -eu
cd "$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)"
if command -v uv >/dev/null 2>&1; then
    test -d .venv || uv venv --python 3.12 .venv
    uv pip install --python .venv/bin/python -r requirements.txt
else
    test -d .venv || python3 -m venv .venv
    .venv/bin/python -m pip install -r requirements.txt
fi
