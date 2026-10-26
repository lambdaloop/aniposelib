#!/usr/bin/env python3

# aniposelib-pytorch provides the `aniposelib` import, so it cannot coexist with the
# classic numpy `aniposelib` distribution (one would overwrite the other's files).
# The only allowed `aniposelib` distribution is the file-less placeholder 0.0.1 that
# aniposelib-pytorch depends on.
from importlib import metadata as _md
try:
    _classic_version = _md.version('aniposelib')
except _md.PackageNotFoundError:
    _classic_version = None
if _classic_version is not None and _classic_version != '0.0.1':
    raise ImportError(
        "Classic 'aniposelib' {} is installed alongside 'aniposelib-pytorch'. Both provide "
        "the 'aniposelib' import, so one has overwritten the other's files. Fix with:\n"
        "    pip uninstall -y aniposelib aniposelib-pytorch && pip install aniposelib-pytorch\n"
        "and replace any 'aniposelib' requirement (e.g. 'aniposelib>=0.7.0') with "
        "'aniposelib-pytorch'.".format(_classic_version))
del _md, _classic_version

__version__ = '0.9.0'
VERSION = __version__
DISTRIBUTION = 'aniposelib-pytorch'
BACKEND = 'torch'

from . import boards, cameras, utils
