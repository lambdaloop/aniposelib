#!/usr/bin/env bash

rm -rv dist build *.egg-info
python3 setup.py sdist bdist_wheel
twine upload dist/*
