"""Input/output type conventions shared by all aniposelib modules (1.0, decision D2).

Rules:
- numpy (or list) in -> numpy out
- torch in -> torch out, detached, on the input's device
- ``return_grad=True`` -> torch out, still attached to the autograd graph

Inside, all computation is torch (float64 by default).
"""

import dataclasses
import functools

import numpy as np
import torch

DEFAULT_DTYPE = torch.float64


def is_torch(x):
    return isinstance(x, torch.Tensor)


def to_torch(x, device=None, dtype=DEFAULT_DTYPE):
    """Convert array-like to a tensor; tensors keep their graph (only moved/cast if needed)."""
    if isinstance(x, torch.Tensor):
        if dtype is not None and x.dtype != dtype and x.is_floating_point():
            x = x.to(dtype)
        return x if device is None else x.to(device)
    x = np.asarray(x)
    if x.dtype.kind in 'fc' or x.dtype == object:
        return torch.as_tensor(np.asarray(x, dtype='float64'), device=device, dtype=dtype)
    return torch.as_tensor(x, device=device)


def to_output(x, torch_out, return_grad=False):
    """Convert one computed value (or a nested tuple/list/dict/dataclass of them) to the output type."""
    if return_grad:
        return x
    if isinstance(x, torch.Tensor):
        x = x.detach()
        return x if torch_out else x.cpu().numpy()
    if isinstance(x, tuple) and hasattr(x, '_fields'):  # namedtuple
        return type(x)(*[to_output(v, torch_out) for v in x])
    if isinstance(x, (tuple, list)):
        return type(x)(to_output(v, torch_out) for v in x)
    if isinstance(x, dict):
        return {k: to_output(v, torch_out) for k, v in x.items()}
    if dataclasses.is_dataclass(x) and not isinstance(x, type):
        return dataclasses.replace(x, **{f.name: to_output(getattr(x, f.name), torch_out)
                                         for f in dataclasses.fields(x)})
    return x


def returns_like_input(arg=0):
    """Decorator: the output type follows the type of positional argument `arg` (after self for
    methods, i.e. arg=0 is the first argument after self). Adds a ``return_grad=False`` keyword.
    Without ``return_grad`` the wrapped function runs under ``torch.no_grad()``."""
    def deco(fn):
        @functools.wraps(fn)
        def wrapper(*args, return_grad=False, **kwargs):
            # methods: args[0] is self; pick the first argument after it
            offset = 1 if args and not isinstance(args[0], (np.ndarray, torch.Tensor, list, tuple)) else 0
            ref = args[offset + arg] if len(args) > offset + arg else None
            torch_out = is_torch(ref)
            if return_grad:
                return fn(*args, **kwargs)
            with torch.no_grad():
                out = fn(*args, **kwargs)
            return to_output(out, torch_out)
        return wrapper
    return deco
