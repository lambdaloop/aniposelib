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

__version__ = '1.0.0'
VERSION = __version__
DISTRIBUTION = 'aniposelib-pytorch'
BACKEND = 'torch'

from . import boards, cameras, params, utils
from .cameras import Camera, FisheyeCamera, CameraGroup
from .boards import CharucoBoard, Checkerboard

# Sub-modules imported on first attribute access (``aniposelib.quality``), so ``import
# aniposelib`` stays fast and never imports optional dependencies (vggt, huggingface_hub,
# onnxruntime, soundfile, av, yaml: those are imported inside the functions that need them).
_LAZY_SUBMODULES = ('affine', 'bundle', 'detectors', 'geometry', 'init', 'intrinsics', 'io',
                    'matching', 'quality', 'scale', 'stations', 'stba', 'sync', 'triangulation')

__all__ = ['boards', 'cameras', 'params', 'utils', 'Camera', 'FisheyeCamera', 'CameraGroup',
           'CharucoBoard', 'Checkerboard', '__version__'] + list(_LAZY_SUBMODULES)


def __getattr__(name):
    if name in _LAZY_SUBMODULES:
        import importlib
        mod = importlib.import_module('.' + name, __name__)
        globals()[name] = mod
        return mod
    raise AttributeError('module {!r} has no attribute {!r}'.format(__name__, name))


def __dir__():
    return sorted(set(globals()) | set(_LAZY_SUBMODULES))
