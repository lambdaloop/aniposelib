#!/usr/bin/env bash

rm -frv dist build *.egg-info
# python3 setup.py sdist bdist_wheel
python -m build
twine upload dist/*
