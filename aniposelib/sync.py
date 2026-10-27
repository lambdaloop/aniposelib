"""Time alignment and synchronisation of unsynchronised streams (``aniposelib.sync``).

Sign convention (used by *every* function in this module)
---------------------------------------------------------
Each stream ``x`` (camera, microphone, clip) has its own source clock. Its **offset** relates it to a common
global timeline::

    source_time[x] = global_time + offset[x]

so a positive offset means that an event appears *later* in that stream's own recording.

A **pair estimate** for ``(a, b)`` returns ``offset = offset[b] - offset[a]``: an event at source time ``t`` in
stream ``a`` appears at source time ``t + offset`` in stream ``b`` (equivalently ``b[n + lag] ~ a[n]``). Frame
estimators use the same convention in frames: frame ``f + offset`` of ``b`` shows the instant of frame ``f`` of
``a``. :func:`solve_offset_graph` turns pair estimates into per-stream offsets with this convention, and
:func:`align_streams` consumes them (it maps ``global_time = source_time - offset``).

Conversions to other conventions found downstream:
- disco ``trim_seconds``: identical (``source_time = global_time + trim``).
- disco/dancing ``ClipMap`` / :class:`TimeMap` (``global = c + beta * pts``): ``offset = -c`` when ``beta == 1``.
- mouse-outdoor ``load_data.align(offsets=...)`` ("seconds to ADD to the timestamps"): ``offset = -added``.
- tailcyclenet mouse-sync ``frame_A = frame_B + d``: ``offset(a=B, b=A) = d``.

Nothing here applies an offset silently. Every estimator returns a result object with the estimate, an
uncertainty, diagnostics and an explicit ``identifiable`` flag with ``reasons`` when it is not.

Everything is numpy/scipy, except :func:`soft_dtw` / :func:`soft_dtw_align` (torch, differentiable). Torch
inputs are accepted everywhere (converted to numpy). Audio file decoding (:func:`load_audio`) needs the optional
``[sync]`` extra (``soundfile``/``av``) or an ``ffmpeg`` executable.
"""

import dataclasses
import math
import shutil
import subprocess
from typing import NamedTuple

import numpy as np
from scipy import signal as _signal

__all__ = [
    'OffsetEstimate', 'OffsetGraph', 'DriftFit', 'WarpEstimate', 'TimeMap', 'AlignedStreams',
    'align_streams', 'load_audio', 'media_info', 'audio_offset', 'audio_offset_windowed', 'onset_envelope', 'motion_signal', 'motion_offset',
    'solve_offset_graph', 'fit_drift', 'epipolar_offset', 'frame_offset_sweep', 'soft_dtw', 'soft_dtw_align',
    'SoftDTW', 'fit_fundamental', 'CONVENTION',
]

CONVENTION = 'source_time = global_time + offset; pair offset(a, b) = offset[b] - offset[a]'


# ------------------------------------------------------------------------------------------------ helpers

def _np(x, dtype='float64'):
    """Array-like or tensor -> numpy (detached, on cpu)."""
    try:
        import torch
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()
    except ImportError:  # pragma: no cover
        pass
    return np.asarray(x, dtype=dtype) if dtype is not None else np.asarray(x)


def _jsonable(x):
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return _jsonable(x.tolist())
    if isinstance(x, (np.floating, float)):
        x = float(x)
        return x if math.isfinite(x) else None
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, np.bool_):
        return bool(x)
    if dataclasses.is_dataclass(x) and not isinstance(x, type):
        return x.to_dict() if hasattr(x, 'to_dict') else _jsonable(dataclasses.asdict(x))
    try:
        import torch
        if isinstance(x, torch.Tensor):
            return _jsonable(x.detach().cpu().numpy())
    except ImportError:  # pragma: no cover
        pass
    return x


def _rng(rng):
    if isinstance(rng, np.random.Generator):
        return rng
    return np.random.default_rng(rng)


def _mad(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float('nan')
    return float(1.4826 * np.median(np.abs(x - np.median(x))))


# ------------------------------------------------------------------------------------------------ results

@dataclasses.dataclass
class OffsetEstimate:
    """One time-offset estimate. ``offset`` follows the module convention (pair: ``offset[b] - offset[a]``).

    Attributes:
        offset: the estimate, in ``units`` (``'s'`` or ``'frames'``).
        sigma: 1-sigma uncertainty in the same units (NaN when it cannot be estimated).
        confidence: method-specific peak score (normalised correlation, cost contrast, ...).
        identifiable: False when the estimate should not be trusted; ``reasons`` says why.
        reasons: e.g. ``'search_boundary'``, ``'ambiguous_second_peak'``, ``'low_confidence'``, ``'flat_curve'``.
        method: estimator name.
        units: ``'s'`` or ``'frames'``.
        pair: optional ``(a, b)`` stream names (used by :func:`solve_offset_graph`).
        diagnostics: method-specific details (curves, second peak, windows, ...).
    """
    offset: float
    sigma: float = float('nan')
    confidence: float = float('nan')
    identifiable: bool = True
    reasons: list = dataclasses.field(default_factory=list)
    method: str = ''
    units: str = 's'
    pair: tuple = None
    diagnostics: dict = dataclasses.field(default_factory=dict)

    def to_dict(self, curves=True):
        d = {f.name: getattr(self, f.name) for f in dataclasses.fields(self)}
        if not curves:
            d['diagnostics'] = {k: v for k, v in self.diagnostics.items() if not k.startswith('curve')}
        d['convention'] = CONVENTION
        return _jsonable(d)

    def __float__(self):
        return float(self.offset)


@dataclasses.dataclass
class OffsetGraph:
    """Per-stream offsets solved from pair estimates (:func:`solve_offset_graph`)."""
    offsets: dict
    sigma: dict
    anchor: str
    edges: list
    rejected: list
    unresolved: list
    cycle_rms: float
    n_cycles: int
    reduced_chi2: float
    identifiable: dict

    def to_dict(self):
        d = {f.name: getattr(self, f.name) for f in dataclasses.fields(self)}
        d['convention'] = CONVENTION
        return _jsonable(d)


@dataclasses.dataclass
class DriftFit:
    """``offset(t) = offset0 + rate * (t - t_ref)`` (rate in s/s; 1e-4 = 100 ppm). ``t`` is global time."""
    offset0: float
    rate: float
    t_ref: float
    sigma_offset0: float
    sigma_rate: float
    residuals: np.ndarray
    inliers: np.ndarray
    rms: float
    n: int
    identifiable: bool = True
    reasons: list = dataclasses.field(default_factory=list)

    def predict(self, t):
        return self.offset0 + self.rate * (_np(t) - self.t_ref)

    def timemap(self):
        """The :class:`TimeMap` for this stream (``source = global + offset(global)``)."""
        a = self.offset0 - self.rate * self.t_ref  # offset at global 0
        return TimeMap.from_offset(a, rate=self.rate)

    def to_dict(self):
        d = {f.name: getattr(self, f.name) for f in dataclasses.fields(self)}
        d['convention'] = CONVENTION
        return _jsonable(d)


# ------------------------------------------------------------------------------------------------ TimeMap

@dataclasses.dataclass
class TimeMap:
    """Invertible map between a stream's source clock (``pts``) and global time.

    ``global = g0 + w(g0)`` with ``g0 = c + beta * pts`` and ``w`` a piecewise-linear deviation given by
    ``knots = [(g0_k, w_k), ...]`` (clamped outside the knot range; ``w = 0`` without knots). This is the
    disco/dancing ``ClipMap``. In the module convention the offset at a global time is
    ``to_source(g) - g``; a constant offset ``o`` is ``TimeMap(c=-o)``.
    """
    c: float = 0.0
    beta: float = 1.0
    knots: tuple = ()

    def __post_init__(self):
        self.c = float(self.c)
        self.beta = float(self.beta)
        k = np.asarray(self.knots, float).reshape(-1, 2) if len(self.knots) else np.zeros((0, 2))
        if len(k):
            order = np.argsort(k[:, 0])
            k = k[order]
            if np.any(np.diff(k[:, 0]) <= 0):
                raise ValueError('TimeMap knots must have distinct g0 values')
            slopes = np.diff(k[:, 1]) / np.diff(k[:, 0])
            if np.any(slopes <= -1):
                raise ValueError('TimeMap knots are not invertible (deviation slope <= -1)')
        if not (self.beta > 0):
            raise ValueError('TimeMap beta must be > 0')
        self.knots = tuple(map(tuple, k.tolist()))

    @classmethod
    def from_offset(cls, offset, rate=0.0, knots=()):
        """``source = global + offset + rate * global`` -> ``global = (source - offset) / (1 + rate)``."""
        return cls(c=-offset / (1.0 + rate), beta=1.0 / (1.0 + rate), knots=knots)

    def _w(self, g0):
        if not self.knots:
            return np.zeros_like(g0)
        k = np.asarray(self.knots)
        return np.interp(g0, k[:, 0], k[:, 1])

    def to_global(self, pts):
        """Source time(s) -> global time(s)."""
        g0 = self.c + self.beta * _np(pts)
        return g0 + self._w(g0)

    def to_source(self, g):
        """Global time(s) -> source time(s) (exact inverse)."""
        g = _np(g)
        if self.knots:
            k = np.asarray(self.knots)
            gk = k[:, 0] + k[:, 1]
            g0 = np.where(g < gk[0], g - k[0, 1],
                          np.where(g > gk[-1], g - k[-1, 1], np.interp(g, gk, k[:, 0])))
        else:
            g0 = g
        return (g0 - self.c) / self.beta

    def offset_at(self, g):
        """Offset (``source - global``) at global time(s) ``g``."""
        return self.to_source(g) - _np(g)

    def to_dict(self):
        return {'c': self.c, 'beta': self.beta, 'knots': [list(k) for k in self.knots],
                'convention': 'global = c + beta*pts + w(c + beta*pts)'}

    @classmethod
    def from_dict(cls, d):
        return cls(c=d.get('c', 0.0), beta=d.get('beta', 1.0), knots=tuple(map(tuple, d.get('knots', ()))))


# ------------------------------------------------------------------------------------------------ align_streams

class AlignedStreams(NamedTuple):
    """``values [S, T, ...]`` (NaN = no sample within tolerance), ``t_grid [T]`` (global time; datetime64 if the
    inputs were datetimes), ``src_idx [S, T]`` (index into each input stream, -1 = none). Streams are in the
    order of the input dict."""
    values: object
    t_grid: np.ndarray
    src_idx: np.ndarray


def _times_kind(t):
    if isinstance(t, np.ndarray) and t.dtype.kind == 'M':
        return 'datetime'
    arr = np.asarray(t)
    if arr.dtype.kind == 'M':
        return 'datetime'
    if arr.dtype.kind in 'biuf':
        return 'numeric'
    return 'other'


def _to_datetime64(t):
    arr = np.asarray(t)
    if arr.dtype.kind == 'M':
        return arr.astype('datetime64[ns]')
    try:
        import pandas as pd
    except ImportError as e:  # pragma: no cover
        raise TypeError('non-numeric timestamps need pandas (or pass datetime64 / float seconds)') from e
    ser = pd.Series(np.asarray(t, dtype=object))
    try:  # pandas >= 2: parse each element on its own (mixed formats / precisions)
        dt = pd.to_datetime(ser, errors='coerce', utc=True, format='mixed')
    except (TypeError, ValueError):
        dt = pd.to_datetime(ser, errors='coerce', utc=True)
    return dt.dt.tz_convert(None).to_numpy(dtype='datetime64[ns]')


def align_streams(streams, rate, tol=None, offsets=None, reuse=False, start=None, end=None, span='union'):
    """Put timestamped streams on a common regular grid (nearest sample within ``tol``).

    Args:
        streams: ``{name: (t, values)}`` or ``{name: t}``. ``t`` is float seconds or datetimes (datetime64,
            pandas timestamps or ISO strings; mixed dtypes are normalised, unparseable entries are dropped).
            ``values`` has ``len(t)`` rows (any trailing shape, identical across streams).
        rate: grid rate in Hz.
        tol: max ``|t_sample - t_grid|`` in seconds (default ``0.5 / rate``).
        offsets: ``{name: offset_seconds | TimeMap}`` in the module convention ``source = global + offset``
            (so ``global = t - offset``). Missing names get 0. **Not** the "seconds to add" convention.
        reuse: if False (default) each source sample is used at most once (the closest grid point wins and a
            displaced grid point may take its other neighbour), so a grid faster than the stream never
            duplicates frames.
        start, end: grid limits in global time (seconds, or datetimes for datetime inputs).
        span: ``'union'`` (default) or ``'intersection'`` of the stream time ranges when start/end are not given.

    Returns:
        :class:`AlignedStreams` ``(values, t_grid, src_idx)``.
    """
    names = list(streams)
    if not names:
        raise ValueError('no streams')
    offsets = offsets or {}
    raw = {}
    for n in names:
        s = streams[n]
        if isinstance(s, tuple) and len(s) == 2:
            t, v = s
        else:
            t, v = s, None
        raw[n] = (t, v)
    kinds = {_times_kind(raw[n][0]) for n in names}
    is_dt = kinds != {'numeric'}
    ref = None
    times = {}
    if is_dt:
        dts = {n: _to_datetime64(raw[n][0]) for n in names}
        valid = [d[~np.isnat(d)] for d in dts.values()]
        ref = min(v.min() for v in valid if v.size)
        for n in names:
            d = dts[n]
            sec = np.full(d.shape, np.nan)
            ok = ~np.isnat(d)
            sec[ok] = (d[ok] - ref).astype('timedelta64[ns]').astype('int64') / 1e9
            times[n] = sec
    else:
        times = {n: _np(raw[n][0]).ravel() for n in names}

    glob = {}
    for n in names:
        o = offsets.get(n, 0.0)
        if isinstance(o, TimeMap):
            if is_dt:
                raise ValueError('TimeMap offsets need float-second timestamps')
            glob[n] = o.to_global(times[n])
        else:
            glob[n] = times[n] - float(o)

    def _lim(x):
        if x is None:
            return None
        if is_dt:
            return float((_to_datetime64([x])[0] - ref).astype('timedelta64[ns]').astype('int64') / 1e9)
        return float(x)

    starts = [np.nanmin(g) for g in glob.values() if np.isfinite(g).any()]
    ends = [np.nanmax(g) for g in glob.values() if np.isfinite(g).any()]
    if not starts:
        raise ValueError('no valid timestamps')
    if span == 'union':
        t0, t1 = min(starts), max(ends)
    elif span == 'intersection':
        t0, t1 = max(starts), min(ends)
    else:
        raise ValueError("span must be 'union' or 'intersection'")
    t0 = _lim(start) if start is not None else t0
    t1 = _lim(end) if end is not None else t1
    if t1 < t0:
        raise ValueError('empty grid (end < start)')
    n_grid = int(np.floor((t1 - t0) * rate + 1e-9)) + 1
    grid = t0 + np.arange(n_grid) / rate
    tol = 0.5 / rate if tol is None else float(tol)

    src_idx = np.full((len(names), n_grid), -1, dtype=np.int64)
    for s, n in enumerate(names):
        g = glob[n]
        ok = np.flatnonzero(np.isfinite(g))
        order = ok[np.argsort(g[ok], kind='stable')]
        gs = g[order]
        if gs.size == 0:
            continue
        src_idx[s] = _match_nearest(grid, gs, order, tol, reuse)

    values = None
    vals = [raw[n][1] for n in names]
    if any(v is not None for v in vals):
        is_torch = False
        try:
            import torch
            is_torch = any(isinstance(v, torch.Tensor) for v in vals)
        except ImportError:  # pragma: no cover
            pass
        arrs = [None if v is None else _np(v) for v in vals]
        shapes = {a.shape[1:] for a in arrs if a is not None}
        if len(shapes) > 1:
            raise ValueError('stream values must share their trailing shape, got {}'.format(shapes))
        tail = shapes.pop()
        values = np.full((len(names), n_grid) + tail, np.nan)
        for s, a in enumerate(arrs):
            if a is None:
                continue
            m = src_idx[s] >= 0
            values[s, m] = a[src_idx[s, m]]
        if is_torch:
            import torch
            values = torch.as_tensor(values)
    if is_dt:
        t_grid = ref + np.round(grid * 1e9).astype('int64').astype('timedelta64[ns]')
    else:
        t_grid = grid
    return AlignedStreams(values, t_grid, src_idx)


def _match_nearest(grid, gs, order, tol, reuse):
    """Grid -> index into the original stream (via ``order``), nearest within tol, optionally one-to-one."""
    n = len(gs)
    right = np.clip(np.searchsorted(gs, grid), 0, n - 1)
    left = np.clip(right - 1, 0, n - 1)
    dl, dr = np.abs(grid - gs[left]), np.abs(grid - gs[right])
    first = np.where(dl <= dr, left, right)
    other = np.where(dl <= dr, right, left)
    dfirst = np.minimum(dl, dr)
    dother = np.maximum(dl, dr)
    out = np.where(dfirst <= tol, first, -1)
    if reuse:
        return np.where(out >= 0, order[np.maximum(out, 0)], -1)
    dist = np.where(out >= 0, dfirst, np.inf)
    for _ in range(2):
        out, dist = _resolve_conflicts(out, dist)
        # displaced grid points try their other neighbour if it is free
        free = np.ones(n, bool)
        free[out[out >= 0]] = False
        cand = (out < 0) & (dother <= tol) & free[other] & (other != first)
        if not cand.any():
            break
        out = np.where(cand, other, out)
        dist = np.where(cand, dother, dist)
    out, dist = _resolve_conflicts(out, dist)
    return np.where(out >= 0, order[np.maximum(out, 0)], -1)


def _resolve_conflicts(out, dist):
    idx = np.flatnonzero(out >= 0)
    if idx.size == 0:
        return out, dist
    srt = idx[np.lexsort((dist[idx], out[idx]))]
    keep = np.r_[True, out[srt][1:] != out[srt][:-1]]
    drop = srt[~keep]
    out = out.copy(); dist = dist.copy()
    out[drop] = -1
    dist[drop] = np.inf
    return out, dist


# ------------------------------------------------------------------------------------------------ correlation core

def _xcorr_ncc(a, b):
    """Overlap-normalised (Pearson) cross-correlation with NaN masks.

    Returns ``lags`` (``b[n + lag] ~ a[n]``), ``ncc`` and the overlap count of valid samples per lag."""
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    ma = np.isfinite(a).astype(float)
    mb = np.isfinite(b).astype(float)
    a0 = np.where(ma > 0, a, 0.0)
    b0 = np.where(mb > 0, b, 0.0)
    # standardise first so the moment sums are well-conditioned
    if ma.sum() > 1:
        mu, sd = a0[ma > 0].mean(), a0[ma > 0].std() + 1e-300
        a0 = np.where(ma > 0, (a0 - mu) / sd, 0.0)
    if mb.sum() > 1:
        mu, sd = b0[mb > 0].mean(), b0[mb > 0].std() + 1e-300
        b0 = np.where(mb > 0, (b0 - mu) / sd, 0.0)

    def cc(x, y):
        return _signal.fftconvolve(x[::-1], y, mode='full')

    n = cc(ma, mb)
    sab = cc(a0, b0)
    sa = cc(a0, mb)
    sb = cc(ma, b0)
    saa = cc(a0 * a0, mb)
    sbb = cc(ma, b0 * b0)
    n_safe = np.maximum(np.round(n), 1.0)
    cov = sab - sa * sb / n_safe
    va = saa - sa * sa / n_safe
    vb = sbb - sb * sb / n_safe
    den = np.sqrt(np.maximum(va * vb, 0.0))
    with np.errstate(invalid='ignore', divide='ignore'):
        ncc = np.where(den > 1e-9 * n_safe, cov / den, np.nan)
    lags = np.arange(-len(a) + 1, len(b))
    return lags, ncc, np.round(n).astype(np.int64)


def _xcorr_ncc_rows(A, B, block=64):
    """:func:`_xcorr_ncc` of every row pair ``(A[i], B[i])``, batched (one FFT per block of rows).
    Yields ``(ncc [rows, L], lags, n)`` per block, in row order. Rows with missing values go
    through :func:`_xcorr_ncc` one at a time."""
    A = np.asarray(A, float)
    B = np.asarray(B, float)
    k = 0
    while k < len(A):
        if not (np.isfinite(A[k]).all() and np.isfinite(B[k]).all()) or A.shape[1] < 2 or B.shape[1] < 2:
            lags, ncc, n = _xcorr_ncc(A[k], B[k])
            yield ncc[None], lags, n
            k += 1
            continue
        e = k
        while e < len(A) and e - k < block and np.isfinite(A[e]).all() and np.isfinite(B[e]).all():
            e += 1
        a0, b0 = A[k:e], B[k:e]
        a0 = (a0 - a0.mean(1, keepdims=True)) / (a0.std(1, keepdims=True) + 1e-300)
        b0 = (b0 - b0.mean(1, keepdims=True)) / (b0.std(1, keepdims=True) + 1e-300)
        ma, mb = np.ones(a0.shape[1]), np.ones(b0.shape[1])

        def cc(x, y):
            x = x[..., ::-1]
            if x.ndim == 1 and y.ndim == 1:
                return _signal.fftconvolve(x, y, mode='full')
            x, y = np.broadcast_to(x, (e - k, x.shape[-1])), np.broadcast_to(y, (e - k, y.shape[-1]))
            return _signal.fftconvolve(x, y, mode='full', axes=-1)
        n = cc(ma, mb)
        sab = cc(a0, b0)
        sa = cc(a0, mb)
        sb = cc(ma, b0)
        saa = cc(a0 * a0, mb)
        sbb = cc(ma, b0 * b0)
        n_safe = np.maximum(np.round(n), 1.0)
        cov = sab - sa * sb / n_safe
        va = saa - sa * sa / n_safe
        vb = sbb - sb * sb / n_safe
        den = np.sqrt(np.maximum(va * vb, 0.0))
        with np.errstate(invalid='ignore', divide='ignore'):
            ncc = np.where(den > 1e-9 * n_safe, cov / den, np.nan)
        yield ncc, np.arange(-A.shape[1] + 1, B.shape[1]), np.round(n).astype(np.int64)
        k = e


def _parabolic(x, y, i):
    """Sub-sample vertex of a peak/valley at index i (returns x[i] at the edges)."""
    if i <= 0 or i >= len(y) - 1 or not np.all(np.isfinite(y[i - 1:i + 2])):
        return float(x[i]), float('nan')
    y0, y1, y2 = y[i - 1], y[i], y[i + 1]
    den = y0 - 2 * y1 + y2
    if den == 0:
        return float(x[i]), 0.0
    dx = 0.5 * (y0 - y2) / den
    dx = float(np.clip(dx, -0.5, 0.5))
    step = float(x[i + 1] - x[i])
    curv = den / step ** 2
    return float(x[i]) + dx * step, float(curv)


def _peak(lags, scores, rate, max_lag, exclusion, min_confidence, min_distinctiveness, overlap=None,
          n_eff_per_s=None):
    """Best peak of a lag curve, with second-peak, boundary and sigma diagnostics."""
    lag_s = lags / rate
    ok = np.isfinite(scores)
    if not ok.any():
        raise ValueError('no lag has a usable overlap')
    sc = np.where(ok, scores, -np.inf)
    i = int(np.argmax(sc))
    best = float(sc[i])
    sub, curv = _parabolic(lag_s, np.where(ok, scores, np.nan), i)
    ex = np.abs(lag_s - lag_s[i]) > exclusion
    # only count genuine local maxima as second peaks (not the shoulder of the main one)
    locmax = np.zeros_like(ok)
    locmax[1:-1] = (sc[1:-1] >= sc[:-2]) & (sc[1:-1] >= sc[2:])
    sec_mask = ex & ok & locmax
    if sec_mask.any():
        j = int(np.flatnonzero(sec_mask)[np.argmax(sc[sec_mask])])
        second, second_lag = float(sc[j]), float(lag_s[j])
    else:
        second, second_lag = float('nan'), float('nan')
    lo, hi = lag_s[ok].min(), lag_s[ok].max()
    if max_lag is not None:
        lo, hi = max(lo, -max_lag), min(hi, max_lag)
    boundary = bool(abs(lag_s[i] - lo) <= 1.0 / rate + 1e-12 or abs(lag_s[i] - hi) <= 1.0 / rate + 1e-12)
    distinct = best - second if np.isfinite(second) else float('inf')
    reasons = []
    if boundary:
        reasons.append('search_boundary')
    if not (best >= min_confidence):
        reasons.append('low_confidence')
    if not (distinct >= min_distinctiveness):
        reasons.append('ambiguous_second_peak')
    # curvature-based sigma (CRLB-like): var = (1 - rho^2) / (rho^2 * N_eff * kappa), kappa = -rho''/rho
    sigma = float('nan')
    ov = None if overlap is None else float(overlap[i])
    if np.isfinite(curv) and curv < 0 and best > 0 and ov is not None:
        n_eff = ov / rate * (n_eff_per_s if n_eff_per_s else rate)
        kappa = -curv / best
        rho2 = min(best * best, 1 - 1e-12)
        sigma = float(np.sqrt((1 - rho2) / (rho2 * max(n_eff, 1.0) * kappa)))
    sigma = float(np.hypot(sigma, 1.0 / (rate * np.sqrt(12)))) if np.isfinite(sigma) else sigma
    diag = dict(peak_score=best, second_peak_score=second, second_peak_offset=second_lag,
                distinctiveness=distinct, boundary_hit=boundary, overlap_seconds=ov / rate if ov else None,
                integer_lag=int(lags[i]), search=(float(lo), float(hi)), curvature=curv)
    return sub, sigma, best, reasons, diag


def _lag_curve(a, b, rate, max_lag, min_overlap):
    lags, ncc, n = _xcorr_ncc(a, b)
    keep = n >= max(int(round(min_overlap * rate)), 2)
    if max_lag is not None:
        keep &= np.abs(lags) <= int(round(max_lag * rate))
    if not keep.any():
        raise ValueError('no lag has enough overlap (min_overlap={} s)'.format(min_overlap))
    return lags[keep], ncc[keep], n[keep]


def _window_offsets(a, b, rate, lag0, n_windows, search, min_window=None):
    """Re-estimate the lag in ``n_windows`` windows of ``a`` (search +-``search`` s around ``lag0``)."""
    n0 = int(round(lag0 * rate))
    r = max(int(round(search * rate)), 2)
    lo = max(0, -n0 + r)
    hi = min(len(a), len(b) - n0 - r)
    if n_windows < 2 or hi - lo < 4 * r:
        return np.zeros(0), np.zeros(0), np.zeros(0)
    edges = np.linspace(lo, hi, n_windows + 1).astype(int)
    ts, offs, scores = [], [], []
    for w0, w1 in zip(edges[:-1], edges[1:]):
        if min_window and (w1 - w0) < min_window * rate:
            continue
        sa = a[w0:w1]
        sb = b[w0 + n0 - r:w1 + n0 + r]
        if np.isfinite(sa).sum() < 8 or np.isfinite(sb).sum() < 8:
            continue
        lags, ncc, n = _xcorr_ncc(sa, sb)
        # lag of sb relative to sa, then shift back to the full-signal lag
        keep = (lags >= 0) & (lags <= 2 * r) & (n >= 0.5 * (w1 - w0))
        if not keep.any() or not np.isfinite(ncc[keep]).any():
            continue
        l, c = lags[keep], ncc[keep]
        i = int(np.nanargmax(c))
        sub, _ = _parabolic(l.astype(float), c, i)
        ts.append(0.5 * (w0 + w1) / rate)
        offs.append((sub + n0 - r) / rate)
        scores.append(float(c[i]))
    return np.asarray(ts), np.asarray(offs), np.asarray(scores)


def _finish(est_offset, sigma, conf, reasons, diag, method, units, pair, a_proc, b_proc, rate, windows,
            window_search):
    if windows and windows >= 3:
        wt, wo, ws = _window_offsets(a_proc, b_proc, rate, est_offset, windows, window_search)
        diag['window_times'] = wt
        diag['window_offsets'] = wo
        diag['window_scores'] = ws
        if len(wo) >= 3:
            spread = _mad(wo)
            diag['window_spread'] = spread
            diag['sigma_curvature'] = sigma
            sigma = float(max(spread / np.sqrt(len(wo)), 1.0 / (rate * np.sqrt(12))))
            diag['sigma_method'] = 'windows'
            if len(wo) >= 3 and np.ptp(wt) > 0:
                diag['window_drift_rate'] = float(np.polyfit(wt, wo, 1)[0])
        else:
            diag['sigma_method'] = 'curvature'
    else:
        diag['sigma_method'] = 'curvature'
    return OffsetEstimate(offset=float(est_offset), sigma=float(sigma), confidence=float(conf),
                          identifiable=not reasons, reasons=reasons, method=method, units=units,
                          pair=pair, diagnostics=diag)


# ------------------------------------------------------------------------------------------------ audio

def load_audio(path, sr=16000, start=0.0, duration=None, backend='auto'):
    """Decode mono float32 audio from an audio or video file, resampled to ``sr``.

    Backends (``backend='auto'`` tries them in order): ``'soundfile'`` (audio files; ``[sync]`` extra),
    ``'av'`` (PyAV, any container; ``[sync]`` extra), ``'ffmpeg'`` (the ``ffmpeg`` executable). Returns a 1-D
    numpy array. Sample 0 is the audio stream start (container start offsets are not applied; see
    :func:`media_info` for the audio stream's start time).
    """
    path = str(path)
    errors = []
    order = ['soundfile', 'av', 'ffmpeg'] if backend == 'auto' else [backend]
    for be in order:
        try:
            if be == 'soundfile':
                return _load_soundfile(path, sr, start, duration)
            if be == 'av':
                return _load_av(path, sr, start, duration)
            if be == 'ffmpeg':
                return _load_ffmpeg(path, sr, start, duration)
            raise ValueError('unknown backend {!r}'.format(be))
        except Exception as e:  # try the next backend
            errors.append('{}: {}'.format(be, e))
    raise RuntimeError('could not decode audio from {} ({}). Install the [sync] extra '
                       '(pip install aniposelib-pytorch[sync]) or ffmpeg.'.format(path, '; '.join(errors)))



def media_info(path):
    """Stream timing of a media file: ``{'audio_start': s, 'video_start': s, 'avg_frame_rate': Hz,
    'r_frame_rate': Hz, 'n_frames': int}`` (keys missing / None when unknown), from PyAV or ``ffprobe``.

    ``audio_start - video_start`` is the source (video) time of audio sample 0 (``audio_t0`` of
    :func:`aniposelib.workflows.estimate_offsets`; :func:`load_audio` starts at the audio stream's
    first sample). ``avg_frame_rate`` is the real mean frame rate; ``r_frame_rate`` is the container's
    nominal rate, which can differ (e.g. 20 fps video in a 30 fps container)."""
    path = str(path)
    out = {}
    try:
        import av
        with av.open(path) as c:
            for s in c.streams:
                st = None if s.start_time is None else float(s.start_time * s.time_base)
                if s.type == 'audio' and 'audio_start' not in out:
                    out['audio_start'] = st
                elif s.type == 'video' and 'video_start' not in out:
                    out['video_start'] = st
                    ar = getattr(s, 'average_rate', None)
                    br = getattr(s, 'base_rate', None) or getattr(s, 'guessed_rate', None)
                    out['avg_frame_rate'] = float(ar) if ar else None
                    out['r_frame_rate'] = float(br) if br else None
                    out['n_frames'] = int(s.frames) if s.frames else None
        return out
    except Exception:
        pass
    exe = shutil.which('ffprobe')
    if exe is None:
        return out
    import json
    r = subprocess.run([exe, '-v', 'error', '-show_streams', '-of', 'json', path], capture_output=True)
    if r.returncode:
        return out

    def rate(v):
        try:
            a, b = str(v).split('/')
            return float(a) / float(b) if float(b) else None
        except Exception:
            return None
    for s in json.loads(r.stdout.decode('utf-8', 'replace')).get('streams', []):
        st = s.get('start_time')
        st = float(st) if st not in (None, 'N/A') else None
        if s.get('codec_type') == 'audio' and 'audio_start' not in out:
            out['audio_start'] = st
        elif s.get('codec_type') == 'video' and 'video_start' not in out:
            out['video_start'] = st
            out['avg_frame_rate'] = rate(s.get('avg_frame_rate'))
            out['r_frame_rate'] = rate(s.get('r_frame_rate'))
            out['n_frames'] = int(s['nb_frames']) if str(s.get('nb_frames', '')).isdigit() else None
    return out


def _resample(x, sr_in, sr):
    if sr_in == sr:
        return x
    from fractions import Fraction
    fr = Fraction(int(sr), int(sr_in)).limit_denominator(1000)
    return _signal.resample_poly(x, fr.numerator, fr.denominator)


def _load_soundfile(path, sr, start, duration):
    import soundfile as sf
    info = sf.info(path)
    s0 = int(round(start * info.samplerate))
    frames = -1 if duration is None else int(round(duration * info.samplerate))
    x, sr_in = sf.read(path, start=s0, frames=frames, dtype='float32', always_2d=True)
    return _resample(x.mean(axis=1), sr_in, sr).astype(np.float32)


def _load_av(path, sr, start, duration):
    import av
    with av.open(path) as c:
        stream = c.streams.audio[0]
        res = av.AudioResampler(format='flt', layout='mono', rate=int(sr))
        chunks = []
        for frame in c.decode(stream):
            for f in res.resample(frame):
                chunks.append(f.to_ndarray().reshape(-1))
        for f in res.resample(None) or []:
            chunks.append(f.to_ndarray().reshape(-1))
    x = np.concatenate(chunks) if chunks else np.zeros(0, np.float32)
    s0 = int(round(start * sr))
    s1 = None if duration is None else s0 + int(round(duration * sr))
    return np.asarray(x[s0:s1], dtype=np.float32)


def _load_ffmpeg(path, sr, start, duration):
    exe = shutil.which('ffmpeg')
    if exe is None:
        raise RuntimeError('ffmpeg not found')
    cmd = [exe, '-hide_banner', '-loglevel', 'error', '-ss', '{:.6f}'.format(max(0.0, start)), '-i', path]
    if duration is not None:
        cmd += ['-t', '{:.6f}'.format(duration)]
    cmd += ['-vn', '-map', '0:a:0', '-ac', '1', '-ar', str(int(sr)), '-f', 'f32le', '-acodec', 'pcm_f32le',
            'pipe:1']
    r = subprocess.run(cmd, capture_output=True, check=False)
    if r.returncode:
        raise RuntimeError(r.stderr.decode('utf-8', 'replace')[-500:])
    return np.frombuffer(r.stdout, dtype='<f4').copy()


def _bandpass(x, sr, band):
    x = np.asarray(x, float)
    x = x - np.median(x)
    if band is None:
        return x
    low, high = band
    high = min(high if high else np.inf, sr * 0.45)
    low = low or 0.0
    if low > 0 and high > low:
        sos = _signal.butter(4, (low, high), btype='bandpass', fs=sr, output='sos')
    elif low > 0:
        sos = _signal.butter(4, low, btype='highpass', fs=sr, output='sos')
    elif np.isfinite(high):
        sos = _signal.butter(4, high, btype='lowpass', fs=sr, output='sos')
    else:
        return x
    return _signal.sosfiltfilt(sos, x)


def onset_envelope(x, sr, bin_hz=100, highpass=100.0):
    """Onset-flux envelope (tailcyclenet mouse-sync): high-pass, RMS in ``1/bin_hz`` bins, positive
    log-energy flux. Returns the envelope sampled at ``bin_hz``."""
    y = _bandpass(x, sr, (highpass, None)) if highpass else np.asarray(x, float)
    n = int(round(sr / bin_hz))
    m = len(y) // n
    rms = np.sqrt(np.mean(y[:m * n].reshape(m, n) ** 2, axis=1) + 1e-16)
    loge = np.log(rms + 1e-8)
    flux = np.maximum(np.diff(loge, prepend=loge[0]), 0)
    return flux - flux.mean()


def _spectrogram_feature(x, sr, nperseg, freq_limit):
    f, t, S = _signal.spectrogram(x, sr, nperseg=nperseg, noverlap=nperseg // 2)
    S = np.log(S[f < freq_limit] + 1e-12)
    S = S - S.mean(axis=1, keepdims=True)
    S = S / (S.std(axis=1, keepdims=True) + 1e-12)
    return S, sr / (nperseg - nperseg // 2), t[0] if len(t) else 0.0


def audio_offset(a, b, sr, max_lag=None, min_overlap=8.0, method='gcc', band=(300.0, 8000.0), windows=6,
                 window_search=None, exclusion=None, min_confidence=0.05, min_distinctiveness=0.02,
                 pair=None, nperseg=None, bin_hz=100):
    """Offset between two audio signals: ``b[n + offset*sr] ~ a[n]`` (seconds; module convention).

    Methods:
        ``'gcc'``: band-passed, overlap-normalised cross-correlation (disco ``sync_disco.estimate_offset``),
            with parabolic sub-sample refinement.
        ``'spectrogram'``: multiscale (Lili's legacy ``sync_videos_offset``): per-frequency log-spectrogram
            correlation for a coarse lag (robust to different microphones/gains), refined by ``'gcc'`` within a
            few spectrogram hops.
        ``'onset'``: onset-flux envelope correlation at ``bin_hz`` (tailcyclenet mouse-sync).

    Args:
        a, b: 1-D signals at ``sr`` Hz (see :func:`load_audio`).
        max_lag: search range in seconds (None = everything with ``min_overlap``).
        min_overlap: minimum overlap in seconds for a lag to be considered.
        band: band-pass in Hz for ``'gcc'`` (and the spectrogram refinement); None to disable.
        windows: if >= 3, re-estimate the lag in that many windows around the result; their spread gives
            ``sigma`` (and ``diagnostics['window_drift_rate']`` hints at clock drift, see :func:`fit_drift`).
        exclusion: second-peak exclusion radius in seconds (default 0.05 s, 0.5 s for 'onset').
        min_confidence, min_distinctiveness: identifiability thresholds on the peak score and on
            ``peak - second_peak``.

    Returns:
        :class:`OffsetEstimate` (seconds).
    """
    a = _np(a).ravel()
    b = _np(b).ravel()
    if min(len(a), len(b)) < min_overlap * sr:
        raise ValueError('audio shorter than min_overlap')
    if method == 'gcc':
        pa, pb = _bandpass(a, sr, band), _bandpass(b, sr, band)
        lags, sc, n = _lag_curve(pa, pb, sr, max_lag, min_overlap)
        bw = (min(band[1] or sr / 2, sr * 0.45) - (band[0] or 0)) if band else sr / 2
        off, sig, conf, reasons, diag = _peak(lags, sc, sr, max_lag, exclusion or 0.05, min_confidence,
                                              min_distinctiveness, overlap=n, n_eff_per_s=2 * bw)
        return _finish(off, sig, conf, reasons, diag, 'gcc', 's', pair, pa, pb, sr, windows,
                       window_search or 0.02)
    if method == 'spectrogram':
        nperseg = nperseg or int(2 ** np.round(np.log2(sr * 0.064)))  # ~64 ms
        flim = band[1] if band and band[1] else 8000.0
        Sa, frate, _ = _spectrogram_feature(a, sr, nperseg, flim)
        Sb, _, _ = _spectrogram_feature(b, sr, nperseg, flim)
        hop = nperseg - nperseg // 2
        acc = None
        for ncc_rows, lags, n in _xcorr_ncc_rows(Sa, Sb):
            for ncc in ncc_rows:
                ncc = np.nan_to_num(ncc)
                acc = ncc if acc is None else acc + ncc
        sc = acc / len(Sa)
        keep = n >= max(int(round(min_overlap * frate)), 2)
        if max_lag is not None:
            keep &= np.abs(lags) <= int(round(max_lag * frate)) + 1
        coarse_lags, coarse_sc = lags[keep], sc[keep]
        coff, _, cconf, creasons, cdiag = _peak(coarse_lags, coarse_sc, frate, max_lag,
                                                exclusion or 4.0 / frate, min_confidence,
                                                min_distinctiveness)
        # the spectrogram frames of a and b are both centred at the same in-segment time, so lags map directly
        coarse = coff
        pa, pb = _bandpass(a, sr, band), _bandpass(b, sr, band)
        r = 3.0 * hop / sr
        lags, sc2, n2 = _lag_curve(pa, pb, sr, None, min_overlap)
        m = np.abs(lags / sr - coarse) <= r
        off, sig, conf, reasons, diag = _peak(lags[m], sc2[m], sr, None, exclusion or 0.05, -np.inf, -np.inf,
                                              overlap=n2[m], n_eff_per_s=sr)
        reasons = list(creasons)  # identifiability is decided at the coarse (global) scale
        if max_lag is not None and abs(off) > max_lag:
            reasons.append('search_boundary')
        diag.update(coarse_offset=coarse, coarse_score=cconf, coarse=cdiag, spectrogram_rate=frate)
        return _finish(off, sig, cconf, reasons, diag, 'spectrogram', 's', pair, pa, pb, sr, windows,
                       window_search or 0.02)
    if method == 'onset':
        ea = onset_envelope(a, sr, bin_hz=bin_hz)
        eb = onset_envelope(b, sr, bin_hz=bin_hz)
        lags, sc, n = _lag_curve(ea, eb, bin_hz, max_lag, min_overlap)
        off, sig, conf, reasons, diag = _peak(lags, sc, bin_hz, max_lag, exclusion or 0.5, min_confidence,
                                              min_distinctiveness, overlap=n)
        return _finish(off, sig, conf, reasons, diag, 'onset', 's', pair, ea, eb, bin_hz, windows,
                       window_search or 0.1)
    raise ValueError("method must be 'gcc', 'spectrogram' or 'onset'")



def audio_offset_windowed(a, b, sr, max_lag=None, min_overlap=8.0, band=(300.0, 8000.0), windows=24,
                          min_window=20.0, search=None, min_score=0.01, pair=None):
    """Offset of long recordings: a coarse lag from the whole clips (``method='spectrogram'``), then
    ``'gcc'`` in ``windows`` windows (at least ``min_window`` s each, searched +-``search`` s around the
    coarse lag; default 0.1 s or 100 ppm of the clip length) and a robust line through the window
    estimates (3 rounds, residuals > 4 MAD dropped). Full-length gcc loses coherence when the clocks
    drift by tens of ppm over an hour; the windows do not. Windows with a peak score above
    ``min_score`` are used (different microphones correlate weakly, ~0.03); the line must explain at
    least half of them with a scatter under a tenth of the search half-width (random peaks scatter
    over the whole search), else the estimate falls back to the coarse lag (``windows_failed``).

    Returns an :class:`OffsetEstimate` (``method='gcc_windowed'``): ``offset`` = the line at the mean
    window time (the whole-clip average, like ``audio_offset``), ``sigma`` = the window scatter /
    sqrt(n - 2) (at least the coarse-scale resolution of one sample), ``diagnostics`` with
    ``window_times`` / ``window_offsets`` (for :func:`fit_drift`), ``window_rate`` (s/s) and the
    coarse estimate. Unidentifiable when the coarse lag is, or fewer than 3 windows correlate."""
    a = _np(a).ravel()
    b = _np(b).ravel()
    co = audio_offset(a, b, sr, max_lag=max_lag, min_overlap=min_overlap, method='spectrogram', band=band,
                      windows=0, pair=pair)
    diag = {'coarse_offset': float(co.offset), 'coarse_identifiable': bool(co.identifiable),
            'coarse_reasons': list(co.reasons), 'sigma_method': 'windowed_gcc'}
    if not co.identifiable:
        return OffsetEstimate(offset=float(co.offset), sigma=float(co.sigma), confidence=float(co.confidence),
                              identifiable=False, reasons=list(co.reasons), method='gcc_windowed', units='s',
                              pair=pair, diagnostics=diag)
    dur = min(len(a), len(b)) / sr
    search = float(search) if search is not None else max(0.1, 1e-4 * dur)
    pa, pb = _bandpass(a, sr, band), _bandpass(b, sr, band)
    nw = int(max(3, min(int(windows), max(3, int(dur // min_window)))))
    wt, wo, ws = _window_offsets(pa, pb, sr, co.offset, nw, search, min_window=min_window)
    good = ws > min_score
    reasons = []
    if good.sum() < 3:
        # too few correlating windows: the coarse (whole-clip) estimate, with a coarse sigma
        diag.update(window_times=wt, window_offsets=wo, window_scores=ws, windows_failed=True)
        return OffsetEstimate(offset=float(co.offset), sigma=float(max(co.sigma, 0.01)),
                              confidence=float(co.confidence), identifiable=True, reasons=[],
                              method='gcc_windowed', units='s', pair=pair, diagnostics=diag)
    wt, wo, ws = wt[good], wo[good], ws[good]
    keep = np.ones(len(wt), bool)
    for _ in range(3):
        A = np.c_[np.ones(keep.sum()), wt[keep] - wt[keep].mean()]
        c, *_ = np.linalg.lstsq(A, wo[keep], rcond=None)
        r = wo - (c[0] + c[1] * (wt - wt[keep].mean()))
        s = max(1.4826 * float(np.median(np.abs(r[keep]))), 1.0 / sr)
        new = np.abs(r) < 4 * s
        if new.sum() < 3 or (new == keep).all():
            break
        keep = new
    t_mid = float(wt[keep].mean())
    if keep.sum() < max(3, 0.5 * len(wt)) or s > 0.1 * search:
        # the windows do not agree on a line: keep the coarse (whole-clip) estimate, honestly
        diag.update(window_times=wt, window_offsets=wo, window_scores=ws, window_spread=s,
                    windows_failed=True)
        return OffsetEstimate(offset=float(co.offset), sigma=float(max(co.sigma, s)),
                              confidence=float(co.confidence), identifiable=True,
                              reasons=[], method='gcc_windowed', units='s', pair=pair,
                              diagnostics=diag)
    sig = float(max(s / np.sqrt(max(keep.sum() - 2, 1)), 1.0 / (sr * np.sqrt(12))))
    diag.update(window_times=wt[keep], window_offsets=wo[keep], window_scores=ws[keep],
                window_rate=float(c[1]), window_t_mid=t_mid, window_spread=s,
                n_windows=int(len(wt)), n_windows_used=int(keep.sum()))
    return OffsetEstimate(offset=float(c[0]), sigma=sig, confidence=float(np.median(ws[keep])),
                          identifiable=True, reasons=reasons, method='gcc_windowed', units='s', pair=pair,
                          diagnostics=diag)


# ------------------------------------------------------------------------------------------------ motion

def motion_signal(points, scale=True):
    """View-invariant motion signal from 2D keypoints ``[T, J, 2]`` (or ``[T, 2]``), NaN = missing.

    Speed of the keypoint centroid, divided by the keypoint spread (so near/far views are comparable), then
    robustly standardised (median/MAD). Port of disco ``refine_sync.body_motion_signal``. Returns ``[T]``
    (NaN where a frame or its predecessor is missing)."""
    p = _np(points)
    if p.ndim == 2:
        p = p[:, None]
    ok = np.isfinite(p).all(-1)
    cnt = ok.sum(1)
    import warnings
    with np.errstate(invalid='ignore'), warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        c = np.nansum(np.where(ok[..., None], p, 0.0), axis=1) / np.maximum(cnt, 1)[:, None]
        c[cnt == 0] = np.nan
        if scale and p.shape[1] > 1:
            d = np.linalg.norm(p - c[:, None], axis=-1)
            s = np.nanmedian(np.where(ok, d, np.nan), axis=1)
            s[(cnt < 2) | ~(s > 1e-9)] = np.nan
        else:
            s = np.ones(len(p))
        v = np.full(len(p), np.nan)
        v[1:] = np.linalg.norm(np.diff(c, axis=0), axis=1) / s[1:]
        med = np.nanmedian(v)
        mad = np.nanmedian(np.abs(v - med)) + 1e-12
    return (v - med) / mad


def motion_offset(sig_a, sig_b, rate, max_lag=2.0, min_overlap=None, exclusion=0.5, min_confidence=0.1,
                  min_distinctiveness=0.01, windows=0, pair=None, units='s'):
    """Offset between two motion signals (e.g. :func:`motion_signal`) sampled at ``rate`` Hz.

    Overlap-normalised correlation with NaN masks, with the diagnostics of disco ``refine_sync``: second
    peak outside ``exclusion`` seconds, search-boundary hit and minimum confidence/distinctiveness. A peak at the
    search boundary is never identifiable. ``units='frames'`` returns frames instead of seconds.
    """
    a, b = _np(sig_a).ravel(), _np(sig_b).ravel()
    if min(np.isfinite(a).sum(), np.isfinite(b).sum()) < 8:
        raise ValueError('motion signals are too short')
    if min_overlap is None:
        min_overlap = 0.25 * min(len(a), len(b)) / rate
    lags, sc, n = _lag_curve(a, b, rate, max_lag, min_overlap)
    off, sig, conf, reasons, diag = _peak(lags, sc, rate, max_lag, exclusion, min_confidence,
                                          min_distinctiveness, overlap=n)
    est = _finish(off, sig, conf, reasons, diag, 'motion', 's', pair, a, b, rate, windows, 2.0 / rate)
    diag['curve_lags'] = lags / rate
    diag['curve_scores'] = sc
    if units == 'frames':
        est.offset *= rate
        est.sigma *= rate
        est.units = 'frames'
    return est


# ------------------------------------------------------------------------------------------------ offset graph

def _edge_tuple(e):
    """-> (a, b, offset, sigma, confidence, identifiable, reasons)"""
    if isinstance(e, OffsetEstimate):
        if e.pair is None:
            raise ValueError('OffsetEstimate edges need .pair = (a, b)')
        a, b = e.pair
        return a, b, float(e.offset), float(e.sigma), float(e.confidence), bool(e.identifiable), list(e.reasons)
    if isinstance(e, dict):
        return (e['a'], e['b'], float(e['offset']), float(e.get('sigma', np.nan)),
                float(e.get('confidence', np.nan)), bool(e.get('identifiable', True)), [])
    e = tuple(e)
    a, b, off = e[:3]
    sig = float(e[3]) if len(e) > 3 and e[3] is not None else float('nan')
    return a, b, float(off), sig, float('nan'), True, []


def solve_offset_graph(edges, anchor=None, names=None, max_residual=None, use_unidentifiable=False,
                       shift='anchor', default_sigma=None):
    """Per-stream offsets from pair offsets by weighted least squares.

    Each edge says ``offset[b] - offset[a] = offset_ab`` (module convention). Weights are ``1/sigma^2`` when
    sigma is known, otherwise ``confidence^2`` (disco ``_solve_trim_graph``), otherwise 1.

    Args:
        edges: iterable of :class:`OffsetEstimate` (with ``pair``), dicts ``{a, b, offset[, sigma]}`` or tuples
            ``(a, b, offset[, sigma])``.
        anchor: stream fixed at offset 0 (default: the first name).
        names: all stream names (default: in order of appearance in the edges).
        max_residual: if given, iteratively drop the worst edge while its residual exceeds this (disco's
            cycle-inconsistency rejection); dropped edges are listed in ``rejected``.
        use_unidentifiable: include estimates flagged ``identifiable=False`` (default: rejected, and listed).
        shift: ``'anchor'`` (anchor = 0) or ``'min'`` (smallest offset = 0, disco ``trim_seconds``).

    Returns:
        :class:`OffsetGraph`. ``cycle_rms`` is the RMS edge residual (0 for a tree; a large value means the
        pair estimates contradict each other); ``identifiable[name]`` is False for streams not connected to the
        anchor (their offset is None).
    """
    raw = [_edge_tuple(e) for e in edges]
    if names is None:
        names = []
        for a, b, *_ in raw:
            for n in (a, b):
                if n not in names:
                    names.append(n)
    names = list(names)
    if not names:
        raise ValueError('no streams')
    anchor = names[0] if anchor is None else anchor
    if anchor not in names:
        raise ValueError('anchor {!r} not in names'.format(anchor))
    rejected, active = [], []
    for e in raw:
        a, b, off, sig, conf, ident, reasons = e
        if a not in names or b not in names:
            rejected.append(dict(a=a, b=b, offset=off, reason='unknown_stream'))
        elif not ident and not use_unidentifiable:
            rejected.append(dict(a=a, b=b, offset=off, reason='not_identifiable', reasons=reasons))
        elif not np.isfinite(off):
            rejected.append(dict(a=a, b=b, offset=off, reason='non_finite'))
        else:
            active.append(e)

    def weight(e):
        sig, conf = e[3], e[4]
        if np.isfinite(sig) and sig > 0:
            return 1.0 / sig ** 2
        if default_sigma:
            return 1.0 / default_sigma ** 2
        if np.isfinite(conf):
            return max(1e-3, conf) ** 2
        return 1.0

    unknown = [n for n in names if n != anchor]
    idx = {n: i for i, n in enumerate(unknown)}

    def solve(act):
        if not act:
            return np.zeros(len(unknown)), np.zeros(0), None
        A = np.zeros((len(act), len(unknown)))
        y = np.array([e[2] for e in act])
        w = np.array([weight(e) for e in act])
        for r, e in enumerate(act):
            if e[0] != anchor:
                A[r, idx[e[0]]] -= 1
            if e[1] != anchor:
                A[r, idx[e[1]]] += 1
        sw = np.sqrt(w)
        x = np.linalg.lstsq(A * sw[:, None], y * sw, rcond=None)[0] if unknown else np.zeros(0)
        res = y - A @ x
        return x, res, (A, w)

    while True:
        x, res, aw = solve(active)
        if max_residual is None or len(active) < 3 or len(res) == 0:
            break
        k = int(np.argmax(np.abs(res)))
        if abs(res[k]) <= max_residual:
            break
        e = active.pop(k)
        rejected.append(dict(a=e[0], b=e[1], offset=e[2], reason='graph_cycle_inconsistent',
                             residual=float(res[k])))

    # connectivity from the anchor
    reach = {anchor}
    changed = True
    while changed:
        changed = False
        for e in active:
            if (e[0] in reach) != (e[1] in reach):
                reach.update((e[0], e[1]))
                changed = True
    offs = {anchor: 0.0}
    offs.update({n: float(x[idx[n]]) for n in unknown})
    sig = {n: float('nan') for n in names}
    sig[anchor] = 0.0
    n_cycles = 0
    red_chi2 = float('nan')
    if aw is not None:
        A, w = aw
        cols = [idx[n] for n in unknown if n in reach]
        if cols:
            Ar = A[:, cols]
            N = Ar.T @ (Ar * w[:, None])
            try:
                cov = np.linalg.inv(N)
                for c, n in zip(cols, [n for n in unknown if n in reach]):
                    sig[n] = float(np.sqrt(cov[cols.index(c), cols.index(c)]))
            except np.linalg.LinAlgError:
                pass
        n_comp_nodes = len(reach)
        n_cycles = len(active) - (n_comp_nodes - 1)
        if n_cycles > 0:
            red_chi2 = float(np.sum(w * res ** 2) / n_cycles)
    for n in names:
        if n not in reach:
            offs[n] = None
            sig[n] = float('nan')
    if shift == 'min':
        m = min(v for v in offs.values() if v is not None)
        offs = {k: (None if v is None else v - m) for k, v in offs.items()}
    elif shift != 'anchor':
        raise ValueError("shift must be 'anchor' or 'min'")
    edge_out = [dict(a=e[0], b=e[1], offset=e[2], sigma=e[3], weight=weight(e), residual=float(r))
                for e, r in zip(active, res)]
    cycle_rms = float(np.sqrt(np.mean(res ** 2))) if len(res) and n_cycles > 0 else 0.0
    return OffsetGraph(offsets=offs, sigma=sig, anchor=anchor, edges=edge_out, rejected=rejected,
                       unresolved=[n for n in names if n not in reach], cycle_rms=cycle_rms,
                       n_cycles=int(max(n_cycles, 0)), reduced_chi2=red_chi2,
                       identifiable={n: n in reach for n in names})


# ------------------------------------------------------------------------------------------------ drift

def fit_drift(t, offsets, robust=True, sigma=None, t_ref=None, f_scale=None, max_rate=1e-2):
    """Fit ``offset(t) = offset0 + rate * (t - t_ref)`` to windowed offset measurements (disco
    ``fit_affine_time_map``; robust soft-L1 by default). ``t`` is global time in seconds (``t_ref`` defaults to
    the median ``t``). ``rate`` is the clock-rate difference (s/s): the stream's source clock runs at
    ``1 + rate`` relative to global. Returns :class:`DriftFit` (with ``timemap()`` and ``predict(t)``)."""
    t = _np(t).ravel()
    y = _np(offsets).ravel()
    ok = np.isfinite(t) & np.isfinite(y)
    t, y = t[ok], y[ok]
    w = None if sigma is None else 1.0 / np.maximum(_np(sigma).ravel()[ok], 1e-12)
    if len(t) < 2 or np.ptp(t) <= 0:
        raise ValueError('need at least two measurements at distinct times')
    t_ref = float(np.median(t)) if t_ref is None else float(t_ref)
    X = np.column_stack([np.ones_like(t), t - t_ref])
    ww = np.ones_like(t) if w is None else w
    p = np.linalg.lstsq(X * ww[:, None], y * ww, rcond=None)[0]
    if robust and len(t) >= 3:
        from scipy.optimize import least_squares
        r0 = (y - X @ p) * ww
        fs = f_scale or max(_mad(r0), 1e-9)
        # restart from a Theil-Sen-like median-slope init to resist leverage outliers
        if len(t) <= 400:
            i, j = np.triu_indices(len(t), 1)
            dt = t[j] - t[i]
            m = dt != 0
            slope = float(np.median((y[j] - y[i])[m] / dt[m]))
            inter = float(np.median(y - slope * (t - t_ref)))
            p = np.array([inter, slope])
        sol = least_squares(lambda q: (X @ q - y) * ww, p, loss='soft_l1', f_scale=fs)
        p = sol.x
    res = y - X @ p
    scale = _mad(res * ww) if robust else float(np.sqrt(np.mean((res * ww) ** 2)))
    if not (scale > 0):
        scale = float(np.sqrt(np.mean((res * ww) ** 2))) or 1e-12
    inl = np.abs(res * ww) <= 3 * max(scale, 1e-12)
    dof = max(int(inl.sum()) - 2, 1)
    Xi = X[inl] * ww[inl, None]
    s2 = float(np.sum((res[inl] * ww[inl]) ** 2) / dof) if w is None else 1.0
    try:
        cov = np.linalg.inv(Xi.T @ Xi) * s2
        so, sr = float(np.sqrt(cov[0, 0])), float(np.sqrt(cov[1, 1]))
    except np.linalg.LinAlgError:
        so = sr = float('nan')
    reasons = []
    if inl.sum() < 3:
        reasons.append('too_few_inliers')
    if abs(p[1]) > max_rate:
        reasons.append('implausible_rate')
    return DriftFit(offset0=float(p[0]), rate=float(p[1]), t_ref=t_ref, sigma_offset0=so, sigma_rate=sr,
                    residuals=res, inliers=inl, rms=float(np.sqrt(np.mean(res[inl] ** 2))), n=int(len(t)),
                    identifiable=not reasons, reasons=reasons)


# ------------------------------------------------------------------------------------------------ epipolar

def _skew(t):
    return np.array([[0, -t[2], t[1]], [t[2], 0, -t[0]], [-t[1], t[0], 0]])


def _sampson(F, a, b):
    """Signed Sampson distance for ``x_b^T F x_a = 0``; a, b [..., 2]."""
    ha = np.concatenate([a, np.ones(a.shape[:-1] + (1,))], -1)
    hb = np.concatenate([b, np.ones(b.shape[:-1] + (1,))], -1)
    Fx = ha @ F.T
    Ftx = hb @ F
    num = np.sum(hb * Fx, -1)
    den = np.sqrt(Fx[..., 0] ** 2 + Fx[..., 1] ** 2 + Ftx[..., 0] ** 2 + Ftx[..., 1] ** 2)
    with np.errstate(invalid='ignore', divide='ignore'):
        return num / den


def _hartley(x):
    c = x.mean(0)
    s = np.sqrt(2) / max(np.mean(np.linalg.norm(x - c, axis=1)), 1e-12)
    T = np.array([[s, 0, -s * c[0]], [0, s, -s * c[1]], [0, 0, 1]])
    return T


def fit_fundamental(a, b, n_iter=8, weights=None):
    """Robust normalised 8-point fundamental matrix (``x_b^T F x_a = 0``) with Cauchy IRLS on Sampson
    residuals. numpy only. a, b: [N, 2] pixel coordinates."""
    a, b = _np(a), _np(b)
    Ta, Tb = _hartley(a), _hartley(b)
    ha = np.c_[a, np.ones(len(a))] @ Ta.T
    hb = np.c_[b, np.ones(len(b))] @ Tb.T
    M = np.einsum('ni,nj->nij', hb, ha).reshape(len(a), 9)
    w = np.ones(len(a)) if weights is None else np.asarray(weights, float)
    F = None
    for it in range(n_iter):
        _, _, Vt = np.linalg.svd(M * np.sqrt(w)[:, None], full_matrices=False)
        Fn = Vt[-1].reshape(3, 3)
        U, S, V2 = np.linalg.svd(Fn)
        Fn = U @ np.diag([S[0], S[1], 0]) @ V2
        F = Tb.T @ Fn @ Ta
        r = np.abs(_sampson(F, a, b))
        s = max(1.4826 * np.median(r), 1e-9)
        w = 1.0 / (1.0 + (r / (2.0 * s)) ** 2)
    return F / (np.linalg.norm(F) + 1e-300)


def _pair_geometry(F_or_cgroup, cams):
    """-> (F in ideal-pixel coords, undistort function per camera)"""
    if F_or_cgroup is None:
        return None, None, None
    if hasattr(F_or_cgroup, 'cameras'):
        cg = F_or_cgroup
        names = cg.get_names()
        ia, ib = [names.index(c) if isinstance(c, str) else int(c) for c in cams]
        ca, cb = cg.cameras[ia], cg.cameras[ib]
        Ka = _np(ca.get_camera_matrix())
        Kb = _np(cb.get_camera_matrix())
        Ea, Eb = _np(ca.get_extrinsics_mat()), _np(cb.get_extrinsics_mat())
        R = Eb[:3, :3] @ Ea[:3, :3].T
        t = Eb[:3, 3] - R @ Ea[:3, 3]
        F = np.linalg.inv(Kb).T @ _skew(t) @ R @ np.linalg.inv(Ka)

        def und(cam, K):
            def f(p):
                shape = p.shape
                flat = p.reshape(-1, 2)
                out = np.full_like(flat, np.nan)
                ok = np.isfinite(flat).all(1)
                if ok.any():
                    n = _np(cam.undistort_points(flat[ok]))
                    out[ok] = n @ K[:2, :2].T + K[:2, 2]
                return out.reshape(shape)
            return f
        return F / np.linalg.norm(F), und(ca, Ka), und(cb, Kb)
    F = _np(F_or_cgroup)
    if F.shape != (3, 3):
        raise ValueError('F_or_cgroup must be a CameraGroup, a 3x3 fundamental matrix or None')
    return F / np.linalg.norm(F), None, None


def _interp_frames(X, t):
    """Linear interpolation of X [T, ...] at fractional frames t [...]; NaN outside or next to missing."""
    t0 = np.floor(t).astype(np.int64)
    w = t - t0
    ok = (t0 >= 0) & (t0 + 1 < len(X))
    t0c = np.clip(t0, 0, len(X) - 2)
    extra = (None,) * (X.ndim - 1)
    wv = w[(...,) + extra]
    out = X[t0c] * (1 - wv) + X[t0c + 1] * wv
    # an exact integer lag needs only one frame
    exact = (np.abs(w) < 1e-9)[(...,) + extra]
    out = np.where(exact, X[np.clip(t0, 0, len(X) - 1)], out)
    ok_full = ok | ((np.abs(w) < 1e-9) & (t0 >= 0) & (t0 < len(X)))
    out[~ok_full] = np.nan
    return out


def _vertex(ds, v, halfwidth):
    ok = np.isfinite(v)
    if not ok.any():
        return float('nan')
    i = int(np.nanargmin(v))
    sel = ok & (np.abs(ds - ds[i]) <= halfwidth + 1e-9)
    if sel.sum() >= 3:
        c = np.polyfit(ds[sel], v[sel], 2)
        if c[0] > 0:
            x = -c[1] / (2 * c[0])
            if ds[sel].min() <= x <= ds[sel].max():
                return float(x)
    return float(ds[i])


def epipolar_offset(p2d_a, p2d_b, F_or_cgroup=None, search=(-10, 10), step=None, subframe=True, cams=(0, 1),
                    n_boot=40, block_len=40, rate=None, rng=0, min_pairs=20, min_contrast=0.1,
                    max_sigma=0.5, exclusion=2.0, ambiguity=0.1, halfwidth=1.0, pair=None):
    """Sub-frame time offset between two cameras from epipolar consistency of moving keypoints.

    For each candidate offset ``d`` (frames), camera b's points are linearly interpolated at frame ``f + d``
    and compared with camera a at frame ``f`` (module convention: frame ``f + d`` of b shows the instant of frame
    ``f`` of a). The cost is the median absolute Sampson distance (px). The minimum is refined with a parabola
    and its uncertainty is a moving-block bootstrap over frames. Port of tailcyclenet
    ``scratch/mouse-sync-v1/sync_2d/sync_epipolar.py`` (cv2-free).

    Args:
        p2d_a, p2d_b: ``[T, 2]`` or ``[T, J, 2]`` frame-indexed 2D points (NaN = missing).
        F_or_cgroup: a :class:`~aniposelib.cameras.CameraGroup` (``cams`` selects the pair; points are
            undistorted, then the calibrated F is used), a 3x3 F with ``x_b^T F x_a = 0`` (pixels as given), or
            None to self-fit F per offset (robust 8-point; needs non-degenerate motion, and the bootstrap then
            keeps each offset's F fixed).
        search: ``(lo, hi)`` offsets in frames; ``step`` defaults to 0.25 (``subframe``) or 1.
        n_boot, block_len: bootstrap resamples and block length in frames (0 to skip).
        rate: if given (Hz), the result is reported in seconds.
        min_contrast: minimum ``(median(curve) - min) / min``, below which the curve is 'flat'.
        max_sigma: maximum bootstrap sigma in frames for an identifiable estimate.

    Returns:
        :class:`OffsetEstimate` with ``diagnostics['curve_offsets'/'curve_cost']``, the bootstrap CI, etc.
    """
    A = _np(p2d_a)
    B = _np(p2d_b)
    if A.ndim == 2:
        A, B = A[:, None], B[:, None]
    F, und_a, und_b = _pair_geometry(F_or_cgroup, cams)
    if und_a is not None:
        A, B = und_a(A), und_b(B)
    step = step or (0.25 if subframe else 1.0)
    ds = np.arange(search[0], search[1] + 1e-9, step)
    fa = np.arange(len(A), dtype=float)
    E = np.full((len(ds), len(A), A.shape[1]), np.nan)
    for k, d in enumerate(ds):
        xb = _interp_frames(B, fa + d)
        ok = np.isfinite(A).all(-1) & np.isfinite(xb).all(-1)
        if ok.sum() < max(min_pairs, 8):
            continue
        Fd = F if F is not None else fit_fundamental(A[ok], xb[ok])
        e = np.full(ok.shape, np.nan)
        e[ok] = np.abs(_sampson(Fd, A[ok], xb[ok]))
        E[k] = e
    n_pairs = np.isfinite(E).reshape(len(ds), -1).sum(1)
    with np.errstate(all='ignore'):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            curve = np.nanmedian(E.reshape(len(ds), -1), axis=1)
    curve[n_pairs < min_pairs] = np.nan
    reasons = []
    if not np.isfinite(curve).any():
        return OffsetEstimate(offset=float('nan'), identifiable=False, reasons=['too_few_pairs'],
                              method='epipolar', units='frames', pair=pair,
                              diagnostics=dict(curve_offsets=ds, curve_cost=curve, n_pairs=n_pairs))
    i = int(np.nanargmin(curve))
    d_hat = _vertex(ds, curve, halfwidth) if subframe else float(ds[i])
    cmin = float(curve[i])
    fin = np.flatnonzero(np.isfinite(curve))
    if i in (fin[0], fin[-1]):
        reasons.append('search_boundary')
    contrast = (float(np.nanmedian(curve)) - cmin) / max(cmin, 1e-12)
    if contrast < min_contrast:
        reasons.append('flat_curve')
    # second local minimum
    c = np.where(np.isfinite(curve), curve, np.inf)
    locmin = np.zeros(len(c), bool)
    locmin[1:-1] = (c[1:-1] <= c[:-2]) & (c[1:-1] <= c[2:])
    far = np.abs(ds - ds[i]) > exclusion
    second = float(np.min(c[locmin & far])) if (locmin & far).any() else float('nan')
    if np.isfinite(second) and second <= cmin * (1 + ambiguity):
        reasons.append('ambiguous_second_minimum')
    # block bootstrap over frames of a
    boots = np.zeros(0)
    if n_boot and block_len and len(A) >= 2 * block_len:
        g = _rng(rng)
        starts = np.arange(0, len(A) - block_len + 1, block_len)
        Ef = E.reshape(len(ds), len(A), -1)
        out = []
        import warnings
        for _ in range(n_boot):
            sel = np.concatenate([np.arange(s, s + block_len) for s in g.choice(starts, len(starts))])
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', RuntimeWarning)
                cb = np.nanmedian(Ef[:, sel].reshape(len(ds), -1), axis=1)
            out.append(_vertex(ds, cb, halfwidth) if subframe else float(ds[int(np.nanargmin(cb))]))
        boots = np.asarray(out)
    sigma = float(np.nanstd(boots)) if boots.size else float('nan')
    if boots.size and max_sigma is not None and not (sigma <= max_sigma):
        reasons.append('bootstrap_unstable')
    diag = dict(curve_offsets=ds, curve_cost=curve, n_pairs=n_pairs, min_cost=cmin, contrast=contrast,
                second_min_cost=second, integer_offset=float(ds[i]), boot=boots,
                boot_ci95=(np.percentile(boots, [2.5, 97.5]).tolist() if boots.size else None),
                F_source=('cgroup' if und_a is not None else 'given' if F is not None else 'self_fit'))
    est = OffsetEstimate(offset=d_hat, sigma=sigma, confidence=contrast, identifiable=not reasons,
                         reasons=reasons, method='epipolar', units='frames', pair=pair, diagnostics=diag)
    if rate:
        est.offset /= rate
        est.sigma /= rate
        est.units = 's'
    return est


# ------------------------------------------------------------------------------------------------ frame sweep

def frame_offset_sweep(cgroup, sample_fn, offsets=range(-4, 5), cams=None, ref_cams=None, min_points=10,
                       min_contrast=0.2):
    """Integer frame-offset sweep per camera against 3D from reference cameras (motor observatory
    ``work/offset_sweep.py``): a safety diagnostic for "hardware-synced" rigs.

    For each tested camera ``c``, 3D points are triangulated from ``ref_cams`` (default: all other cameras) at
    offset 0, then ``sample_fn(c, d)`` is reprojected and compared for each ``d``. A sharp minimum at ``d``
    means camera ``c``'s frame ``f + d`` shows the instant of the reference frame ``f``, i.e. offset ``d``
    frames (module convention). Flat curves are reported as not identifiable.

    Args:
        cgroup: :class:`~aniposelib.cameras.CameraGroup`.
        sample_fn: ``sample_fn(cam_index, d) -> [N, 2]`` or ``[N, J, 2]`` 2D points (NaN = missing) of that
            camera at the sampled frames shifted by ``d``; the same N (and J) for every call.
        offsets: integer offsets to test.
        cams: cameras to test (indices or names; default all).
        ref_cams: reference cameras (indices or names; default all except the tested one).

    Returns:
        ``{camera_name: OffsetEstimate}`` (units frames; ``diagnostics['curve_*']`` holds the median error per
        offset).
    """
    names = cgroup.get_names()

    def ix(c):
        return names.index(c) if isinstance(c, str) else int(c)

    cams = list(range(len(names))) if cams is None else [ix(c) for c in cams]
    ref_all = None if ref_cams is None else [ix(c) for c in ref_cams]
    offsets = [int(d) for d in offsets]
    cache = {}

    def sample(c, d):
        if (c, d) not in cache:
            cache[(c, d)] = _np(sample_fn(c, d))
        return cache[(c, d)]

    p3d_cache = {}
    out = {}
    for c in cams:
        refs = tuple(r for r in (ref_all if ref_all is not None else range(len(names))) if r != c)
        if len(refs) < 2:
            raise ValueError('need at least two reference cameras besides {}'.format(names[c]))
        if refs not in p3d_cache:
            pts = np.stack([sample(r, 0) for r in refs])
            shape = pts.shape[1:-1]
            p3d = _np(cgroup.subset_cameras(list(refs)).triangulate(pts.reshape(len(refs), -1, 2)))
            p3d_cache[refs] = (p3d, shape)
        p3d, shape = p3d_cache[refs]
        cam = cgroup.cameras[c]
        ok3 = np.isfinite(p3d).all(-1)
        proj = np.full(p3d.shape[:-1] + (2,), np.nan)
        if ok3.any():
            proj[ok3] = _np(cam.project(p3d[ok3])).reshape(-1, 2)
        curve, counts = [], []
        for d in offsets:
            p = sample(c, d).reshape(-1, 2)
            err = np.linalg.norm(proj - p, axis=-1)
            m = np.isfinite(err)
            counts.append(int(m.sum()))
            curve.append(float(np.median(err[m])) if m.sum() >= min_points else float('nan'))
        curve = np.asarray(curve)
        reasons = []
        if not np.isfinite(curve).any():
            out[names[c]] = OffsetEstimate(offset=float('nan'), identifiable=False, reasons=['too_few_points'],
                                           method='frame_sweep', units='frames',
                                           diagnostics=dict(curve_offsets=offsets, curve_error=curve,
                                                            n_points=counts))
            continue
        i = int(np.nanargmin(curve))
        fin = np.flatnonzero(np.isfinite(curve))
        if i in (fin[0], fin[-1]):
            reasons.append('search_boundary')
        others = np.delete(curve, i)
        contrast = (float(np.nanmedian(others)) - curve[i]) / max(curve[i], 1e-12) if others.size else 0.0
        if contrast < min_contrast:
            reasons.append('flat_curve')
        ds = np.asarray(offsets, float)
        out[names[c]] = OffsetEstimate(
            offset=float(offsets[i]), sigma=float('nan'), confidence=float(contrast), identifiable=not reasons,
            reasons=reasons, method='frame_sweep', units='frames', pair=None,
            diagnostics=dict(curve_offsets=offsets, curve_error=curve, n_points=counts, contrast=contrast,
                             vertex=_vertex(ds, curve, 1.0), ref_cams=[names[r] for r in refs]))
    return out


# ------------------------------------------------------------------------------------------------ soft-DTW

_BIG = 1e10


class SoftDTW(NamedTuple):
    """``value``: soft-DTW cost (torch scalar keeping the graph of ``D``; float for numpy input).
    ``alignment``: expected alignment ``E = d value / d D`` (dense ``[n, m]``, zero outside the band; or
    ``(I, J, E)`` cell triplets when ``D`` is a callable)."""
    value: object
    alignment: object


def _band_cells(n, m, band, center=None):
    """Sakoe-Chiba band cells ordered by anti-diagonal. Returns (I, J, diag_starts, diag_lo)."""
    i = np.arange(n)
    if center is None:
        c = i * ((m - 1) / (n - 1)) if n > 1 else np.zeros(1)
    else:
        c = np.asarray(center, float).reshape(-1)
        if c.shape != (n,) or np.any(np.diff(c) < 0):
            raise ValueError('center must be a non-decreasing array of length n')
    if band is None:
        jlo, jhi = np.zeros(n, np.int64), np.full(n, m - 1, np.int64)
    else:
        w = float(band)
        jlo = np.clip(np.ceil(c - w - 1e-9), 0, m - 1).astype(np.int64)
        jhi = np.clip(np.floor(c + w + 1e-9), 0, m - 1).astype(np.int64)
        if np.any(jhi < jlo) or np.any(jlo[1:] > jhi[:-1] + 1):
            raise ValueError('band too narrow for this shape/center (cells are not connected)')
    cnt = jhi - jlo + 1
    I = np.repeat(i, cnt)
    J = np.concatenate([np.arange(a, b + 1) for a, b in zip(jlo, jhi)])
    K = I + J
    order = np.lexsort((I, K))
    I, J, K = I[order], J[order], K[order]
    ks = np.arange(K.max() + 1)
    starts = np.searchsorted(K, ks, side='left')
    ends = np.searchsorted(K, ks, side='right')
    return I, J, starts, ends


def _soft_dtw_core(costs, I, J, starts, ends, gamma, open_begin, open_end, n, m):
    """Anti-diagonal wavefront soft-DTW on banded cells. costs: torch [n_cells] in (k, i) order."""
    import torch
    dt, dev = costs.dtype, costs.device
    nk = len(starts)
    ln = ends - starts
    lo = np.where(ln > 0, I[np.minimum(starts, len(I) - 1)], 0)
    K = I + J

    def pred_index(dk, di, dj):
        """index into cat([BIG, ZERO], R_{k-dk}) of predecessor (i-di, j-dj) for every cell (vectorised)"""
        r, c = I - di, J - dj
        idx = np.zeros(len(I), np.int64)
        virt = (r < 0) | (c < 0)
        zero_ok = virt & (r < 0) if open_begin else virt & (r < 0) & (c < 0)
        idx[zero_ok] = 1
        kp = K - dk
        okk = ~virt & (kp >= 0)
        kpc = np.clip(kp, 0, nk - 1)
        pos = r - lo[kpc]
        inside = okk & (pos >= 0) & (pos < ln[kpc])
        idx[inside] = pos[inside] + 2
        return torch.as_tensor(idx, device=dev)

    i_d, i_u, i_l = pred_index(2, 1, 1), pred_index(1, 1, 0), pred_index(1, 0, 1)
    head = torch.tensor([_BIG, 0.0], dtype=dt, device=dev)
    empty = costs[:0]
    Rs = []
    for k in range(nk):
        s, e = int(starts[k]), int(ends[k])
        if e <= s:
            Rs.append(empty)
            continue
        pd = torch.cat([head, Rs[k - 2]]) if k >= 2 else head
        pp = torch.cat([head, Rs[k - 1]]) if k >= 1 else head
        prev = torch.stack([pd[i_d[s:e]], pp[i_u[s:e]], pp[i_l[s:e]]])
        Rs.append(costs[s:e] - gamma * torch.logsumexp(-prev / gamma, dim=0))
    R = torch.cat(Rs)
    if open_end:
        last = np.flatnonzero(I == n - 1)
        vals = R[torch.as_tensor(last, device=dev)]
        return -gamma * torch.logsumexp(-vals / gamma, dim=0)
    endc = np.flatnonzero((I == n - 1) & (J == m - 1))
    if endc.size == 0:
        raise ValueError('the end cell (n-1, m-1) is outside the band; use open_end=True')
    return R[int(endc[0])]


def soft_dtw(D, gamma=1.0, band=None, open_begin=False, open_end=False, alignment=True, shape=None,
             center=None):
    """Soft-DTW (Cuturi & Blondel 2017) on a cost matrix, with a Sakoe-Chiba band (torch, differentiable).

    The DP runs along anti-diagonals over the band cells only, so memory is ``O(n * band)``. Gradients flow
    through the soft-min recursion; the expected alignment is ``E = d value / d D``.

    Args:
        D: ``[n, m]`` cost matrix (torch or numpy), or a callable ``D(I, J) -> torch costs`` for the band cells
            (then pass ``shape=(n, m)``; nothing dense is built).
        gamma: soft-min temperature (-> 0 gives classic DTW).
        band: half-width (in columns) around ``center`` (default: the diagonal from (0, 0) to (n-1, m-1));
            None = full matrix.
        open_begin / open_end: let the path start / end anywhere on the first / last *row* (subsequence
            alignment of the rows within the columns); all rows are always consumed.
        alignment: also return ``E``.

    Returns:
        :class:`SoftDTW` ``(value, alignment)``. Numpy in -> float/numpy out; torch in -> torch out (``value``
        keeps the graph of ``D``).
    """
    import torch
    is_fn = callable(D) and not isinstance(D, (np.ndarray, torch.Tensor))
    torch_in = isinstance(D, torch.Tensor)
    if is_fn:
        if shape is None:
            raise ValueError('shape=(n, m) is required when D is a callable')
        n, m = shape
    else:
        Dt = D if torch_in else torch.as_tensor(_np(D))
        n, m = Dt.shape
    I, J, starts, ends = _band_cells(n, m, band, center)
    if is_fn:
        costs = D(torch.as_tensor(I), torch.as_tensor(J))
    else:
        costs = Dt[torch.as_tensor(I, device=Dt.device), torch.as_tensor(J, device=Dt.device)]
    leaf = costs
    if alignment and not costs.requires_grad:
        leaf = costs.detach().requires_grad_(True)
    with torch.enable_grad():
        value = _soft_dtw_core(leaf, I, J, starts, ends, float(gamma), open_begin, open_end, n, m)
        E = None
        if alignment:
            E = torch.autograd.grad(value, leaf, retain_graph=leaf is costs,
                                    create_graph=leaf is costs and costs.requires_grad)[0]
    if leaf is not costs:
        value = value.detach()
    if alignment:
        if is_fn:
            al = (I, J, E)
        else:
            Ed = torch.zeros((n, m), dtype=E.dtype, device=E.device)
            Ed[torch.as_tensor(I, device=E.device), torch.as_tensor(J, device=E.device)] = E
            al = Ed if torch_in else Ed.detach().cpu().numpy()
    else:
        al = None
    if not torch_in and not is_fn:
        value = float(value.detach())
    return SoftDTW(value, al)


@dataclasses.dataclass
class WarpEstimate:
    """Monotone time warp between two signals (:func:`soft_dtw_align`), module convention.

    ``t_a[i]`` (source seconds of a) aligns with ``t_a[i] + lag[i]`` in b's source time; ``dispersion[i]`` is
    the spread (s) of the soft alignment at that row. ``offset``/``rate`` are a robust affine summary
    ``lag(t) ~ offset0 + rate * (t - t_ref)`` (``drift``), ``offset`` is the median lag.
    """
    offset: float
    sigma: float
    rate: float
    drift: object
    t_a: np.ndarray
    lag: np.ndarray
    dispersion: np.ndarray
    identifiable: bool
    reasons: list
    value: float
    method: str = 'soft_dtw'
    units: str = 's'
    pair: tuple = None
    diagnostics: dict = dataclasses.field(default_factory=dict)

    def interp(self, t_a):
        """b source time for a's source time(s) ``t_a`` (piecewise-linear along the soft path)."""
        t = _np(t_a)
        return t + np.interp(t, self.t_a, self.lag)

    def to_offset_estimate(self):
        return OffsetEstimate(offset=self.offset, sigma=self.sigma, confidence=float('nan'),
                              identifiable=self.identifiable, reasons=list(self.reasons), method=self.method,
                              units='s', pair=self.pair,
                              diagnostics=dict(rate=self.rate, median_dispersion=float(np.median(self.dispersion))))

    def to_dict(self):
        d = {f.name: getattr(self, f.name) for f in dataclasses.fields(self)}
        d['convention'] = CONVENTION
        return _jsonable(d)


def _fill_nan(x):
    x = x.copy()
    t = np.arange(len(x))
    for f in range(x.shape[1]):
        ok = np.isfinite(x[:, f])
        if ok.sum() == 0:
            x[:, f] = 0.0
        elif (~ok).any():
            x[~ok, f] = np.interp(t[~ok], t[ok], x[ok, f])
    return x


def soft_dtw_align(sig_a, sig_b, rate=1.0, gamma=0.1, band=1.0, lag0=None, max_lag=None, normalize=True,
                   max_dispersion=None, max_match_ratio=0.15, pair=None, device=None):
    """Non-constant time alignment of two view-invariant signals by banded soft-DTW.

    Use for offsets that change over time (dropped frames, VFR phones, chapter steps), on signals such as audio
    onset envelopes (:func:`onset_envelope`) or motion energy (:func:`motion_signal`). A coarse constant lag
    ``lag0`` (estimated by cross-correlation if not given) centres a Sakoe-Chiba band of ``+-band`` seconds;
    every sample of a's overlap is aligned (subsequence soft-DTW with open ends in b).

    Args:
        sig_a, sig_b: ``[T]`` or ``[T, F]`` signals at the same ``rate`` (Hz); NaN gaps are interpolated.
        gamma: soft-min temperature on z-scored squared differences.
        band: half-width of the band in seconds (the largest deviation from ``lag0`` that can be tracked).
        lag0: coarse lag in seconds (module convention); None = cross-correlation (``max_lag`` limits it).
        max_dispersion: identifiability threshold on the median row dispersion (s); default ``band / 4``.
        max_match_ratio: identifiability threshold on ``match_ratio`` = mean cost along the soft path / mean
            cost over the band (about 1 for unrelated signals: DTW always finds *some* path, so a sharp
            alignment alone does not prove a match).

    Returns:
        :class:`WarpEstimate` (per-row lag and dispersion, robust affine offset/drift, identifiability).
    """
    import torch
    a = _np(sig_a)
    b = _np(sig_b)
    a = a[:, None] if a.ndim == 1 else a
    b = b[:, None] if b.ndim == 1 else b
    if a.shape[1] != b.shape[1]:
        raise ValueError('signals must have the same number of features')
    reasons = []
    diag = {}
    if lag0 is None:
        ca = np.nanmean(a, axis=1) if a.shape[1] > 1 else a[:, 0]
        cb = np.nanmean(b, axis=1) if b.shape[1] > 1 else b[:, 0]
        coarse = motion_offset(ca, cb, rate, max_lag=max_lag if max_lag is not None else
                               0.5 * min(len(a), len(b)) / rate, exclusion=band)
        lag0 = coarse.offset
        diag['coarse'] = coarse.to_dict(curves=False)
        if not coarse.identifiable:
            reasons.append('coarse_lag_unidentifiable')
    a, b = _fill_nan(a), _fill_nan(b)
    if normalize:
        a = (a - a.mean(0)) / (a.std(0) + 1e-12)
        b = (b - b.mean(0)) / (b.std(0) + 1e-12)
    L0 = int(round(lag0 * rate))
    w = max(int(round(band * rate)), 1)
    ia0, ia1 = max(0, -L0), min(len(a), len(b) - L0)
    if ia1 - ia0 < 4 * w:
        raise ValueError('overlap too short for this band')
    jb0, jb1 = max(0, ia0 + L0 - w), min(len(b), ia1 + L0 + w)
    A = torch.as_tensor(a[ia0:ia1], device=device)
    B = torch.as_tensor(b[jb0:jb1], device=device)
    n, m = len(A), len(B)
    center = np.arange(n) + (ia0 + L0 - jb0)

    def cost(I, J):
        return ((A[I.to(A.device)] - B[J.to(A.device)]) ** 2).mean(-1)

    I, J, starts, ends = _band_cells(n, m, w, center)
    res = soft_dtw(cost, gamma=gamma, band=w, open_begin=True, open_end=True, shape=(n, m), center=center)
    _, _, E = res.alignment
    E = E.detach().cpu().numpy()
    with torch.no_grad():
        cc = cost(torch.as_tensor(I), torch.as_tensor(J)).cpu().numpy()
    path_cost = float(np.sum(E * cc) / max(np.sum(E), 1e-12))
    band_cost = float(np.mean(cc))
    match_ratio = path_cost / max(band_cost, 1e-12)
    mass = np.bincount(I, weights=E, minlength=n)
    jm = np.bincount(I, weights=E * J, minlength=n) / np.maximum(mass, 1e-12)
    jv = np.bincount(I, weights=E * J * J, minlength=n) / np.maximum(mass, 1e-12) - jm ** 2
    sd = np.sqrt(np.maximum(jv, 0.0))
    rows = np.arange(n)
    t_a = (rows + ia0) / rate
    lag = ((jm + jb0) - (rows + ia0)) / rate
    disp = sd / rate
    # band-edge hits: the expected column within 1 sample of the band limit
    jlo = np.clip(np.ceil(center - w - 1e-9), 0, m - 1)
    jhi = np.clip(np.floor(center + w + 1e-9), 0, m - 1)
    edge = ((jm - jlo) < 1.0) & (jlo > 0) | ((jhi - jm) < 1.0) & (jhi < m - 1)
    edge_frac = float(edge.mean())
    drift = fit_drift(t_a, lag, robust=True)
    max_dispersion = band / 4 if max_dispersion is None else max_dispersion
    med_disp = float(np.median(disp))
    if med_disp > max_dispersion:
        reasons.append('diffuse_alignment')
    if edge_frac > 0.05:
        reasons.append('band_edge')
    if match_ratio > max_match_ratio:
        reasons.append('weak_match')
    n_eff = max(n / (2 * w + 1), 1.0)
    sigma = float(np.hypot(med_disp, _mad(lag - drift.predict(t_a))) / np.sqrt(n_eff))
    diag.update(lag0=float(lag0), band_samples=w, gamma=float(gamma), median_dispersion=med_disp,
                band_edge_fraction=edge_frac, row_mass=mass, n_rows=n, path_cost=path_cost,
                band_cost=band_cost, match_ratio=match_ratio)
    return WarpEstimate(offset=float(np.median(lag)), sigma=sigma, rate=drift.rate, drift=drift, t_a=t_a,
                        lag=lag, dispersion=disp, identifiable=not reasons, reasons=reasons,
                        value=float(res.value), pair=pair, diagnostics=diag)
