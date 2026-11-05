#!/usr/bin/env bash
# Upload aniposelib-pytorch (main branch only).
set -euo pipefail
cd "$(dirname "$0")"
grep -q '^name = "aniposelib-pytorch"$' pyproject.toml \
    || { echo "refusing: pyproject name is not aniposelib-pytorch"; exit 1; }
[ "$(git rev-parse --abbrev-ref HEAD)" = "main" ] \
    || { echo "refusing: not on the main branch"; exit 1; }
rm -frv dist build *.egg-info
python -m build
twine upload dist/*
