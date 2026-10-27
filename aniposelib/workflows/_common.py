"""Shared pieces of the workflow entry points (plan 05 §1, §3).

``Rule`` / ``evaluate_rules``
    Declarative verdict rules: ``Rule(name, value, warn_at, fail_at, explain)``. A rule with a
    ``value`` past ``fail_at`` gives ``'fail'``, past ``warn_at`` gives ``'warn'``
    (``higher_is_worse=False`` flips the direction). ``options={'verdict': {name: (warn, fail)}}``
    overrides thresholds per rule, ``{name: None}`` disables a rule.
``noise_floor(errors, inliers=None, diag=None)``
    Robust pixel noise estimate shared by W2/W4/W6/W7: the median of the finite (inlier)
    reprojection errors, clipped to ``[MIN_NOISE_PX, NOISE_DIAG_FRAC * diag]``; 0.5 px when
    there are too few errors (synthetic / noiseless data).
``WorkflowResult``
    Base dataclass of every result: ``report`` (dict), ``verdict`` ('ok'|'warn'|'fail'),
    ``reasons`` (list of str), ``settings`` (every derived threshold, for reproducibility),
    ``provenance``; ``to_dict()``, ``to_json(path=None)``, ``summary()``.
``Options``
    ``options=`` handling: a dict (or None) merged over a workflow's defaults; unknown keys raise.
"""

import dataclasses
import json
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import torch

__all__ = ['Rule', 'evaluate_rules', 'noise_floor', 'WorkflowResult', 'merge_options',
           'provenance', 'image_diagonal', 'MIN_NOISE_PX', 'NOISE_DIAG_FRAC', 'to_numpy',
           'Timer', 'worst']

MIN_NOISE_PX = 0.5
NOISE_DIAG_FRAC = 0.01
_ORDER = {'ok': 0, 'warn': 1, 'fail': 2}


def worst(*verdicts):
    """The worst of several verdicts ('fail' > 'warn' > 'ok')."""
    vs = [v for v in verdicts if v is not None]
    return max(vs, key=lambda v: _ORDER[v]) if vs else 'ok'


def to_numpy(x):
    if x is None:
        return None
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _jsonable(x):
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    if isinstance(x, np.ndarray):
        if x.size > 10000:
            return {'shape': list(x.shape), 'dtype': str(x.dtype)}
        return _jsonable(x.tolist())
    if isinstance(x, (np.floating, float)):
        x = float(x)
        return x if np.isfinite(x) else None
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.bool_,)):
        return bool(x)
    if dataclasses.is_dataclass(x) and not isinstance(x, type):
        if hasattr(x, 'to_dict'):
            return _jsonable(x.to_dict())
        return _jsonable(dataclasses.asdict(x))
    if hasattr(x, 'to_dict') and callable(x.to_dict):
        try:
            return _jsonable(x.to_dict())
        except Exception:  # pragma: no cover
            return repr(x)
    if isinstance(x, (str, int, bool)) or x is None:
        return x
    return repr(x)


# ------------------------------------------------------------------ verdict rules

@dataclass
class Rule:
    name: str
    value: Any
    warn_at: Optional[float] = None
    fail_at: Optional[float] = None
    explain: str = ''
    higher_is_worse: bool = True

    def level(self):
        v = self.value
        if v is None:
            return 'ok'
        if isinstance(v, (bool, np.bool_)):
            # boolean rules: value True means the problem is present
            return ('fail' if self.fail_at is not None else 'warn') if bool(v) else 'ok'
        v = float(v)
        if not np.isfinite(v):
            return 'fail' if (self.higher_is_worse and v > 0) or (not self.higher_is_worse and v < 0) \
                or np.isnan(v) else 'ok'
        worse = (lambda a, b: a > b) if self.higher_is_worse else (lambda a, b: a < b)
        if self.fail_at is not None and worse(v, self.fail_at):
            return 'fail'
        if self.warn_at is not None and worse(v, self.warn_at):
            return 'warn'
        return 'ok'

    def message(self):
        v = self.value
        vs = '{:.3g}'.format(float(v)) if isinstance(v, (float, int, np.floating, np.integer)) \
            and not isinstance(v, (bool, np.bool_)) else str(v)
        th = self.fail_at if self.level() == 'fail' else self.warn_at
        th = '' if th is None or isinstance(v, (bool, np.bool_)) else \
            ' ({} {:.3g})'.format('>' if self.higher_is_worse else '<', float(th))
        return '{}: {} = {}{}{}'.format(self.level(), self.name, vs, th,
                                        (' -- ' + self.explain) if self.explain else '')


def evaluate_rules(rules, overrides=None):
    """-> (verdict, reasons, table). ``overrides``: {rule name: (warn_at, fail_at) | None}."""
    overrides = overrides or {}
    unknown = set(overrides) - {r.name for r in rules}
    out, reasons, table = [], [], []
    for r in rules:
        if r.name in overrides:
            o = overrides[r.name]
            if o is None:
                continue
            r = dataclasses.replace(r, warn_at=o[0], fail_at=o[1])
        lv = r.level()
        out.append(lv)
        table.append({'rule': r.name, 'value': _jsonable(r.value), 'warn_at': r.warn_at,
                      'fail_at': r.fail_at, 'level': lv})
        if lv != 'ok':
            reasons.append(r.message())
    del unknown  # overrides for rules that did not apply (e.g. optional inputs) are ignored
    return worst(*out), reasons, table


# ------------------------------------------------------------------ noise floor

def image_diagonal(cgroup):
    """Median image diagonal (px) of the group's cameras, or None if sizes are unknown."""
    d = []
    for cam in getattr(cgroup, 'cameras', []):
        s = cam.get_size()
        if s is None:
            continue
        s = to_numpy(s).astype(float).ravel()
        if len(s) >= 2 and np.all(np.isfinite(s[:2])) and np.all(s[:2] > 0):
            d.append(float(np.hypot(s[0], s[1])))
    return float(np.median(d)) if d else None


def noise_floor(errors, inliers=None, diag=None, min_count=10):
    """Robust pixel noise floor: median of finite (inlier) errors, clipped to
    [MIN_NOISE_PX, NOISE_DIAG_FRAC * diag]. Falls back to MIN_NOISE_PX with < min_count errors."""
    e = to_numpy(errors)
    if e is None:
        return MIN_NOISE_PX
    e = np.abs(np.asarray(e, dtype=float))
    if inliers is not None:
        m = np.asarray(to_numpy(inliers), dtype=bool)
        e = np.where(np.broadcast_to(m, e.shape), e, np.nan)
    e = e[np.isfinite(e)]
    if e.size < min_count:
        return MIN_NOISE_PX
    s = float(np.median(e))
    hi = NOISE_DIAG_FRAC * diag if diag else np.inf
    return float(np.clip(s, MIN_NOISE_PX, max(hi, MIN_NOISE_PX)))


# ------------------------------------------------------------------ options / provenance

def merge_options(defaults, options, where='options'):
    """Merge ``options`` (dict, dataclass or None) over ``defaults``; unknown keys raise."""
    out = dict(defaults)
    if options is None:
        return out
    if dataclasses.is_dataclass(options):
        options = dataclasses.asdict(options)
    unknown = set(options) - set(out)
    if unknown:
        raise ValueError('unknown {} keys {}; allowed: {}'.format(where, sorted(unknown), sorted(out)))
    out.update(options)
    return out


def provenance(workflow, inputs=None, fallbacks=None, timings=None):
    try:
        from .. import __version__ as ver
    except Exception:  # pragma: no cover
        ver = None
    return {'workflow': workflow, 'aniposelib': ver, 'time': time.strftime('%Y-%m-%dT%H:%M:%S'),
            'inputs': inputs or {}, 'fallbacks': list(fallbacks or []), 'timings': timings or {}}


class Timer:
    """``with timer('step'):`` accumulates wall time per step into ``timer.times``."""

    def __init__(self):
        self.times = {}

    def __call__(self, name):
        timer = self

        class _T:
            def __enter__(self):
                self.t = time.perf_counter()

            def __exit__(self, *a):
                timer.times[name] = timer.times.get(name, 0.0) + time.perf_counter() - self.t
        return _T()


# ------------------------------------------------------------------ result base

@dataclass
class WorkflowResult:
    report: Dict[str, Any] = field(default_factory=dict)
    verdict: str = 'ok'
    reasons: List[str] = field(default_factory=list)
    settings: Dict[str, Any] = field(default_factory=dict)
    provenance: Dict[str, Any] = field(default_factory=dict)

    # fields that hold large arrays / objects and are summarised rather than serialised
    _skip = ('cgroup',)

    def to_dict(self):
        d = {}
        for f in dataclasses.fields(self):
            v = getattr(self, f.name)
            if f.name in self._skip:
                if v is not None and hasattr(v, 'get_names'):
                    d[f.name] = {'cameras': list(v.get_names())}
                continue
            d[f.name] = _jsonable(v)
        return d

    def to_json(self, path=None):
        s = json.dumps(self.to_dict(), indent=2, default=repr)
        if path is not None:
            with open(path, 'w') as f:
                f.write(s)
        return s

    def summary(self):
        lines = ['{}: verdict {}'.format(type(self).__name__, self.verdict)]
        for r in self.reasons:
            lines.append('  - ' + r)
        rules = self.report.get('rules') if isinstance(self.report, dict) else None
        if rules:
            lines.append('  {:<28} {:>12} {:>10} {:>10}  level'.format('rule', 'value', 'warn_at', 'fail_at'))
            for r in rules:
                v = r['value']
                vs = '{:.4g}'.format(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else str(v)
                lines.append('  {:<28} {:>12} {:>10} {:>10}  {}'.format(
                    r['rule'], vs, str(r['warn_at']), str(r['fail_at']), r['level']))
        return '\n'.join(lines)

    def __repr__(self):
        return '<{} verdict={!r} reasons={}>'.format(type(self).__name__, self.verdict, len(self.reasons))

    @property
    def ok(self):
        return self.verdict != 'fail'


def save_calibration(res, out):
    """Write ``res.cgroup`` to ``out`` (TOML) and the report to ``<stem>.report.json``."""
    import os
    if out is None or getattr(res, 'cgroup', None) is None:
        return
    out = str(out)
    res.cgroup.dump(out)
    stem = out[:-5] if out.endswith('.toml') else out
    res.to_json(stem + '.report.json')


def save_arrays(res, out, arrays):
    """Write ``arrays`` (dict of arrays) to ``<stem>.npz`` and the report to ``<stem>.report.json``."""
    if out is None:
        return
    out = str(out)
    stem = out[:-4] if out.endswith('.npz') else out
    np.savez_compressed(stem + '.npz', **{k: to_numpy(v) for k, v in arrays.items() if v is not None})
    res.to_json(stem + '.report.json')
