"""Input/output type conventions shared by all aniposelib modules (1.0, decision D2).

Rules:
- numpy (or list) in -> numpy out
- torch in -> torch out, detached, on the input's device
- ``return_grad=True`` -> torch out, still attached to the autograd graph

Inside, all computation is torch (float64 by default).
"""

import dataclasses
import functools
import inspect

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


def to_output(x, torch_out, return_grad=False, device=None):
    """Convert one computed value (or a nested tuple/list/dict/dataclass of them) to the output
    type. Torch outputs are moved to `device` when given (the input's device)."""
    if return_grad:
        return x
    if isinstance(x, torch.Tensor):
        x = x.detach()
        if torch_out:
            return x if device is None or x.device == device else x.to(device)
        x = x.cpu().numpy()
        return x[()] if x.ndim == 0 else x
    kw = dict(device=device)
    if isinstance(x, tuple) and hasattr(x, '_fields'):  # namedtuple
        return type(x)(*[to_output(v, torch_out, **kw) for v in x])
    if isinstance(x, (tuple, list)):
        return type(x)(to_output(v, torch_out, **kw) for v in x)
    if isinstance(x, dict):
        return {k: to_output(v, torch_out, **kw) for k, v in x.items()}
    if dataclasses.is_dataclass(x) and not isinstance(x, type):
        return dataclasses.replace(x, **{f.name: to_output(getattr(x, f.name), torch_out, **kw)
                                         for f in dataclasses.fields(x)})
    return x


def returns_like_input(arg=0):
    """Decorator: the output type follows the type of positional argument `arg` (after self for
    methods, i.e. arg=0 is the first argument after self; it may also be passed by keyword).
    Adds a ``return_grad=False`` keyword. Without ``return_grad`` the wrapped function runs under
    ``torch.no_grad()`` and tensor outputs are detached; numpy (or list) input gives numpy
    output, torch input gives torch output on the input's device."""
    def deco(fn):
        try:
            names = list(inspect.signature(fn).parameters)
        except (TypeError, ValueError):  # pragma: no cover
            names = []

        @functools.wraps(fn)
        def wrapper(*args, return_grad=False, **kwargs):
            # methods: args[0] is self; pick the first argument after it
            offset = 1 if args and not isinstance(args[0], (np.ndarray, torch.Tensor, list, tuple)) else 0
            if len(args) > offset + arg:
                ref = args[offset + arg]
            else:
                ix = offset + arg
                ref = kwargs.get(names[ix]) if ix < len(names) else None
            torch_out = is_torch(ref)
            device = ref.device if torch_out else None
            if return_grad:
                out = fn(*args, **kwargs)
                return _move(out, device) if torch_out else out
            with torch.no_grad():
                out = fn(*args, **kwargs)
            return to_output(out, torch_out, device=device)
        return wrapper
    return deco


def _move(x, device):
    if isinstance(x, torch.Tensor):
        return x if x.device == device else x.to(device)
    if isinstance(x, tuple) and hasattr(x, '_fields'):
        return type(x)(*[_move(v, device) for v in x])
    if isinstance(x, (tuple, list)):
        return type(x)(_move(v, device) for v in x)
    return x
