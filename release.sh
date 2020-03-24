#!/usr/bin/env bash

rm -rv dist build *.egg-info
python3 setup.py bdist_wheel
twine upload dist/*
