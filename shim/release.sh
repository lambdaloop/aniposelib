#!/usr/bin/env bash
# Upload ONLY the aniposelib==0.0.1 placeholder.
set -euo pipefail
cd "$(dirname "$0")"
grep -q '^name = "aniposelib"$' pyproject.toml && grep -q '^version = "0.0.1"$' pyproject.toml \
    || { echo "refusing: shim/pyproject.toml is not aniposelib 0.0.1"; exit 1; }
rm -frv dist build *.egg-info
python -m build
twine upload dist/*
