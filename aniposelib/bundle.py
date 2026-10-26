"""Bundle adjustment: sparse observations, Schur-complement Levenberg-Marquardt, robust recipe
and camera resection (1.0, plan 01 D2-D7, F3, F5).

API
---
``Observations(cam_idx, point_idx, uv, weight=None, time=None, group=None, station=None)``
    Sparse observations (one row per 2D detection). ``Observations.from_dense(p2ds, groups=)``
    converts ``C x N x 2`` (or ``C x N x J x 2``; then ``group`` gets ``frame`` and ``keypoint``).
``bundle_adjust_lm(cgroup, obs_or_p2ds, p3d_init=None, spec=None, loss='cauchy', f_scale=5.0,
                   n_iter=25, max_points=None, prune=None, extra=None, ...) -> float | BAReport``
    Torch Schur-complement LM over camera parameters (selected by a
    :class:`~aniposelib.params.ParamSpec`) and 3D points (or rigid board poses with ``extra=``),
    with robust IRLS weights (``'linear' | 'huber' | 'cauchy' | 'soft_l1'``), priors, gauge
    (``fixed_cameras``) and scale (``scale_constraint``) control, per-camera time offsets (F3) and
    stations (F5: several extrinsic segments of one camera sharing its intrinsics).
``bundle_adjust_robust(cgroup, p2ds, groups=None, schedule=..., reject=RejectSchedule(), resect=True,
                       min_inliers=3, final=('extrinsics', 'intrinsics'), ...) -> BAReport | float``
    The converged recipe: robust (MSAC) triangulation -> inlier observations -> LM, annealing the
    threshold; camera resection between rounds; structured (camera x frame / camera x keypoint)
    rejection. **Masks belong to the fit; reports are computed on all observations.**
``resect_cameras(cgroup, p2ds, ...)``, ``place_cameras(cgroup, p2ds, cams)``,
``add_camera(cgroup, camera, p2d_cam, p2ds)``
    Leave-one-out PnP-RANSAC of a camera against points robustly triangulated from the others
    (``sample='points'`` or ``'frames'``: whole frames as the minimal sample, for decoys).

Conventions
-----------
* Units: pixels for residuals, ``f_scale`` and thresholds; the rig's own units for points.
* Gauge: by default (``fixed_cameras='auto'``) the extrinsics of the first camera are held fixed,
  and (``scale_constraint='auto'``) the rig scale (RMS distance of camera centres from their
  centroid) is kept unless the problem has rigid boards (whose geometry fixes the scale).
* Time offsets (F3, ``ParamSpec(time_offset=sigma)``): an observation of camera ``c`` labelled
  ``time = t`` shows the scene at ``t + delta_c``; the model is first order,
  ``X(t + delta) ~ X(t) + delta * dX/dt``, with velocities from the triangulated tracks
  (``group['track']``, else ``group['keypoint']``, else one track). Units of ``obs.time``.
  ``delta`` of the gauge camera(s) is 0; the others have a zero-mean prior of the given sigma.
* Stations (F5): ``obs.station`` (index into ``stations['station_cam']``) selects the extrinsics
  of a camera segment; intrinsics are shared by all stations of a camera. The camera's own
  extrinsics are written from its first station.
"""

import dataclasses
import math
import time as _time
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from . import geometry as G
from . import triangulation as T
from .params import ParamSpec, IntrinsicsPrior, DIST_NAMES, dist_index

__all__ = ['Observations', 'BAReport', 'RejectSchedule', 'ResectResult', 'bundle_adjust_lm',
           'bundle_adjust_robust', 'resect_cameras', 'place_cameras', 'add_camera', 'frame_mask',
           'keypoint_mask', 'LOSSES']

LOSSES = ('linear', 'huber', 'cauchy', 'soft_l1')

# layout of the per-observation local camera vector
_EXT = slice(0, 6)
_FX, _FY, _CX, _CY, _SKEW = 6, 7, 8, 9, 10
_D0 = 11            # 8 distortion slots 11..18
_DELTA = 19
N_LOCAL = 20
_MAT_SLOT = {'fx': _FX, 'fy': _FY, 'cx': _CX, 'cy': _CY, 'skew': _SKEW}


def _np(v):
    if isinstance(v, torch.Tensor):
        v = v.detach().cpu().numpy()
    return v


def _jsonable(v):
    v = _np(v)
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, np.generic):
        return v.item()
    if isinstance(v, dict):
        return {str(k): _jsonable(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    if dataclasses.is_dataclass(v) and not isinstance(v, type):
        return v.to_dict() if hasattr(v, 'to_dict') else _jsonable(dataclasses.asdict(v))
    return v


def _as_rng(rng):
    if isinstance(rng, np.random.Generator):
        return rng
    return np.random.default_rng(rng)


# ====================================================================== observations

@dataclass
class Observations:
    """Sparse 2D observations.

    ``cam_idx [M]`` camera index (group order), ``point_idx [M]`` point index, ``uv [M,2]`` pixels,
    ``weight [M]`` (confidence, default 1), ``time [M]`` (frame time, for time offsets / stations),
    ``group`` dict of ``[M]`` integer arrays (e.g. ``frame``, ``keypoint``, ``track``; used for
    structured rejection and velocities), ``station [M]`` (index of the extrinsic station, F5).
    ``n_points`` defaults to ``max(point_idx) + 1``; ``point_shape`` remembers the dense shape.
    """
    cam_idx: Any
    point_idx: Any
    uv: Any
    weight: Any = None
    time: Any = None
    group: Optional[Dict[str, Any]] = None
    station: Any = None
    n_points: Optional[int] = None
    n_cams: Optional[int] = None
    point_shape: Optional[tuple] = None

    def __post_init__(self):
        self.cam_idx = np.asarray(_np(self.cam_idx), dtype=np.int64).ravel()
        self.point_idx = np.asarray(_np(self.point_idx), dtype=np.int64).ravel()
        self.uv = np.asarray(_np(self.uv), dtype=np.float64).reshape(-1, 2)
        M = len(self.cam_idx)
        if len(self.point_idx) != M or len(self.uv) != M:
            raise ValueError('cam_idx, point_idx and uv must have the same length')
        if self.weight is not None:
            self.weight = np.asarray(_np(self.weight), dtype=np.float64).ravel()
        if self.time is not None:
            self.time = np.asarray(_np(self.time), dtype=np.float64).ravel()
        if self.station is not None:
            self.station = np.asarray(_np(self.station), dtype=np.int64).ravel()
        if self.group is not None:
            self.group = {k: np.asarray(_np(v)).ravel() for k, v in self.group.items()}
            for k, v in self.group.items():
                if len(v) != M:
                    raise ValueError('group {!r} has length {}, expected {}'.format(k, len(v), M))
        if self.n_points is None:
            self.n_points = int(self.point_idx.max()) + 1 if M else 0
        if self.n_cams is None:
            self.n_cams = int(self.cam_idx.max()) + 1 if M else 0

    def __len__(self):
        return len(self.cam_idx)

    @classmethod
    def from_dense(cls, p2ds, groups=None, weights=None, time=None):
        """From ``C x N x 2`` or ``C x N x J x 2`` (NaN = missing).

        ``groups``: dict of per-point arrays (shape ``N`` or ``N x J``) copied to each observation.
        For 4-D input, ``frame`` (= n) and ``keypoint`` (= j) are added unless given.
        ``weights``: ``C x N (x J)``. ``time``: per-point times (``N`` or ``N x J``)."""
        p = np.asarray(_np(p2ds), dtype=np.float64)
        C = p.shape[0]
        pshape = p.shape[1:-1]
        flat = p.reshape(C, -1, 2)
        N = flat.shape[1]
        groups = dict(groups or {})
        if len(pshape) == 2:
            n_, j_ = np.meshgrid(np.arange(pshape[0]), np.arange(pshape[1]), indexing='ij')
            groups.setdefault('frame', n_)
            groups.setdefault('keypoint', j_)
        valid = np.isfinite(flat).all(-1)
        ci, pi = np.nonzero(valid)
        grp = None
        if groups:
            grp = {}
            for k, v in groups.items():
                v = np.broadcast_to(np.asarray(_np(v)).reshape(np.asarray(_np(v)).shape), pshape).reshape(-1) \
                    if np.asarray(_np(v)).size != N else np.asarray(_np(v)).reshape(-1)
                grp[k] = v[pi]
        w = None
        if weights is not None:
            w = np.asarray(_np(weights), dtype=np.float64).reshape(C, -1)[ci, pi]
        t = None
        if time is not None:
            tt = np.asarray(_np(time), dtype=np.float64)
            tt = np.broadcast_to(tt.reshape(tt.shape + (1,) * (len(pshape) - tt.ndim)), pshape).reshape(-1)
            t = tt[pi]
        return cls(ci, pi, flat[ci, pi], weight=w, time=t, group=grp, n_points=N, n_cams=C,
                   point_shape=tuple(pshape))

    def subset(self, mask):
        """Observations selected by a boolean mask or index array (point indexing unchanged)."""
        m = np.asarray(mask)
        sel = lambda v: None if v is None else v[m]
        return Observations(self.cam_idx[m], self.point_idx[m], self.uv[m], weight=sel(self.weight),
                            time=sel(self.time),
                            group=None if self.group is None else {k: v[m] for k, v in self.group.items()},
                            station=sel(self.station), n_points=self.n_points, n_cams=self.n_cams,
                            point_shape=self.point_shape)

    def to_dense(self, values=None, fill=np.nan):
        """Scatter ``values [M,...]`` (default ``uv``) into a dense ``C x N x ...`` array."""
        values = self.uv if values is None else np.asarray(_np(values))
        out = np.full((self.n_cams, self.n_points) + values.shape[1:], fill,
                      dtype=np.result_type(values.dtype, np.asarray(fill).dtype))
        out[self.cam_idx, self.point_idx] = values
        return out

    def to_dict(self):
        return {f.name: _jsonable(getattr(self, f.name)) for f in dataclasses.fields(self)}


# ====================================================================== report

@dataclass
class BAReport:
    """Result of a bundle adjustment (LM or robust recipe). ``float(report)`` = ``error``.

    ``error``: final median reprojection error (px) used as the scalar summary; ``history``:
    per-iteration (LM) or per-stage (robust) dicts; ``per_camera``: ``{name: {...}}`` with
    observation counts and before/after median/p90 errors; ``excluded``: masks that were applied
    to the *fit* (reports always use all observations); ``gauge``: rig scale before/after and
    their ratio, fixed cameras, scale constraint; ``time_offsets`` (F3); ``stations`` (F5).
    """
    method: str = 'lm'
    error: float = float('nan')
    error_before: float = float('nan')
    converged: bool = False
    n_iter: int = 0
    history: List[Dict[str, Any]] = field(default_factory=list)
    per_camera: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    n_obs: int = 0
    n_points: int = 0
    n_obs_pruned: int = 0
    excluded: Dict[str, Any] = field(default_factory=dict)
    gauge: Dict[str, Any] = field(default_factory=dict)
    frozen_cameras: List[str] = field(default_factory=list)
    time_offsets: Optional[Dict[str, float]] = None
    stations: Optional[Dict[str, Any]] = None
    reverted: bool = False
    seed: Any = None
    config: Dict[str, Any] = field(default_factory=dict)
    elapsed_s: float = 0.0
    warnings: List[str] = field(default_factory=list)
    stages: List[Any] = field(default_factory=list)
    p3d: Any = None
    obs_errors: Any = None

    def __float__(self):
        return float(self.error)

    def to_dict(self):
        d = {}
        for f in dataclasses.fields(self):
            if f.name in ('p3d', 'obs_errors'):
                continue
            d[f.name] = _jsonable(getattr(self, f.name))
        return d

    def summary(self):
        lines = ['{} BA: error {:.3f} px (before {:.3f}), {} obs, {} points, {} iterations{}'.format(
            self.method, self.error, self.error_before, self.n_obs, self.n_points, self.n_iter,
            ' (reverted)' if self.reverted else '')]
        for name, pc in self.per_camera.items():
            lines.append('  {:>12s}: n={:6d}  median {:.2f} -> {:.2f} px'.format(
                str(name), int(pc.get('n_obs', 0)), pc.get('median_before', float('nan')),
                pc.get('median_after', float('nan'))))
        return '\n'.join(lines)


def _err_stats(e):
    e = np.asarray(e, dtype=float)
    e = e[np.isfinite(e)]
    if len(e) == 0:
        return float('nan'), float('nan')
    return float(np.median(e)), float(np.percentile(e, 90))


def rig_scale_from_centers(centers):
    c = centers - centers.mean(0)
    return float(torch.sqrt((c * c).sum(-1).mean()))


# ====================================================================== projection kernel

def _rob(e2, loss, f):
    """Robust cost rho(e^2) in px^2 units, and IRLS weight rho'(e^2)."""
    if loss == 'linear':
        return e2, torch.ones_like(e2)
    if loss == 'cauchy':
        s = f * f
        return s * torch.log1p(e2 / s), 1.0 / (1.0 + e2 / s)
    if loss == 'huber':
        e = torch.sqrt(e2)
        rho = torch.where(e <= f, e2, 2 * f * e - f * f)
        w = torch.where(e <= f, torch.ones_like(e), f / e.clamp(min=1e-12))
        return rho, w
    if loss == 'soft_l1':
        s = f * f
        z = torch.sqrt(1 + e2 / s)
        return 2 * s * (z - 1), 1.0 / z
    raise ValueError('unknown loss {!r}; use one of {}'.format(loss, LOSSES))


def _project_local(L, Xw, k10, off, is_kb, has_b, has_k):
    """Per-observation projection. L [M,20] local camera vectors, Xw [M,3] world points."""
    R = G.rodrigues(L[:, 0:3])
    pc = torch.einsum('mij,mj->mi', R, Xw) + L[:, 3:6]
    z = pc[:, 2]
    eps = G.FISHEYE_EPS
    zs = torch.where(torch.abs(z) >= eps, z, torch.full_like(z, eps))
    x = (pc[:, 0] / zs)[:, None]
    y = (pc[:, 1] / zs)[:, None]
    d = L[:, _D0:_D0 + 8]
    if has_b and not has_k:
        xd, yd = G._brown(x, y, d)
    elif has_k and not has_b:
        xd, yd = G._kb(x, y, d)
    else:
        bx, by = G._brown(x, y, d)
        kx, ky = G._kb(x, y, d)
        kb = is_kb[:, None]
        xd, yd = torch.where(kb, kx, bx), torch.where(kb, ky, by)
    xd, yd = xd[:, 0], yd[:, 0]
    u = L[:, _FX] * xd + L[:, _SKEW] * yd + L[:, _CX] - off[:, 0]
    v = k10 * xd + L[:, _FY] * yd + L[:, _CY] - off[:, 1]
    return torch.stack([u, v], -1), z


def _skew(v):
    o = torch.zeros_like(v[..., 0])
    x, y, z = v[..., 0], v[..., 1], v[..., 2]
    return torch.stack([torch.stack([o, -z, y], -1),
                        torch.stack([z, o, -x], -1),
                        torch.stack([-y, x, o], -1)], -2)


def _rot_and_jac(r):
    """R(r) [M,3,3] and the right Jacobian J_r(r) [M,3,3]: R(r + d) ~ R(r) Exp(J_r d)."""
    R = G.rodrigues(r)
    th2 = (r * r).sum(-1)
    small = th2 < 1e-8
    th2s = torch.where(small, torch.ones_like(th2), th2)
    th = torch.sqrt(th2s)
    a = torch.where(small, 0.5 - th2 / 24, (1 - torch.cos(th)) / th2s)
    b = torch.where(small, 1.0 / 6 - th2 / 120, (th - torch.sin(th)) / (th2s * th))
    K = _skew(r)
    eye = torch.eye(3, dtype=r.dtype, device=r.device)
    J = eye - a[..., None, None] * K + b[..., None, None] * (K @ K)
    return R, J


def _dist_coef_jac(x, y, d, is_kb, has_b, has_k):
    """d(xd, yd) / d(dist[0..7]) -> [M,2,8] for normalised x, y [M]."""
    M = x.shape[0]
    out = torch.zeros(M, 2, 8, dtype=x.dtype, device=x.device)
    if has_b:
        k1, k2, p1, p2, k3, k4, k5, k6 = [d[:, i] for i in range(8)]
        r2 = x * x + y * y
        r4, r6 = r2 * r2, r2 * r2 * r2
        num = 1 + r2 * (k1 + r2 * (k2 + r2 * k3))
        den = 1 + r2 * (k4 + r2 * (k5 + r2 * k6))
        dr = [r2 / den, r4 / den, None, None, r6 / den,
              -num * r2 / (den * den), -num * r4 / (den * den), -num * r6 / (den * den)]
        jb = torch.zeros_like(out)
        for i in (0, 1, 4, 5, 6, 7):
            jb[:, 0, i] = x * dr[i]
            jb[:, 1, i] = y * dr[i]
        jb[:, 0, 2] = 2 * x * y
        jb[:, 1, 2] = r2 + 2 * y * y
        jb[:, 0, 3] = r2 + 2 * x * x
        jb[:, 1, 3] = 2 * x * y
        out = jb
    if has_k:
        r2 = x * x + y * y
        r_ok = r2 >= G.FISHEYE_EPS ** 2
        r = torch.sqrt(torch.where(r_ok, r2, torch.ones_like(r2)))
        th = torch.atan(r)
        jk = torch.zeros_like(out)
        for i in range(4):
            ds = torch.where(r_ok, th ** (2 * i + 3) / r, torch.zeros_like(r))
            jk[:, 0, i] = ds * x
            jk[:, 1, i] = ds * y
        out = torch.where(is_kb[:, None, None], jk, out) if has_b else jk
    return out


def _project_local_jac(L, Xw, k10, off, is_kb, has_b, has_k):
    """Projection and analytic Jacobians: uv [M,2], z [M], dL [M,2,20] (slot 19 = 0), dX [M,2,3]."""
    R, Jr = _rot_and_jac(L[:, 0:3])
    pc = torch.einsum('mij,mj->mi', R, Xw) + L[:, 3:6]
    z = pc[:, 2]
    eps = G.FISHEYE_EPS
    zs = torch.where(torch.abs(z) >= eps, z, torch.full_like(z, eps))
    x = pc[:, 0] / zs
    y = pc[:, 1] / zs
    d = L[:, _D0:_D0 + 8]
    if has_b and not has_k:
        xd, yd, Jd = G._brown(x[:, None], y[:, None], d, jac=True)
    elif has_k and not has_b:
        xd, yd, Jd = G._kb(x[:, None], y[:, None], d, jac=True)
    else:
        bx, by, bj = G._brown(x[:, None], y[:, None], d, jac=True)
        kx, ky, kj = G._kb(x[:, None], y[:, None], d, jac=True)
        kb = is_kb[:, None]
        xd, yd = torch.where(kb, kx, bx), torch.where(kb, ky, by)
        Jd = tuple(torch.where(kb, a, b) for a, b in zip(kj, bj))
    xd, yd = xd[:, 0], yd[:, 0]
    jxx, jxy, jyx, jyy = [j[:, 0] for j in Jd]
    fx, fy, cx, cy, sk = L[:, _FX], L[:, _FY], L[:, _CX], L[:, _CY], L[:, _SKEW]
    uv = torch.stack([fx * xd + sk * yd + cx - off[:, 0], k10 * xd + fy * yd + cy - off[:, 1]], -1)
    Am = torch.stack([torch.stack([fx, sk], -1), torch.stack([k10, fy], -1)], -2)      # [M,2,2]
    Jdm = torch.stack([torch.stack([jxx, jxy], -1), torch.stack([jyx, jyy], -1)], -2)
    iz = 1 / zs
    o = torch.zeros_like(iz)
    Jp = torch.stack([torch.stack([iz, o, -x * iz], -1), torch.stack([o, iz, -y * iz], -1)], -2)
    Gpc = Am @ Jdm @ Jp                                                           # [M,2,3]
    dL = torch.zeros(L.shape[0], 2, N_LOCAL, dtype=L.dtype, device=L.device)
    dL[:, :, 0:3] = -Gpc @ R @ _skew(Xw) @ Jr
    dL[:, :, 3:6] = Gpc
    dL[:, 0, _FX] = xd
    dL[:, 1, _FY] = yd
    dL[:, 0, _CX] = 1
    dL[:, 1, _CY] = 1
    dL[:, 0, _SKEW] = yd
    dL[:, :, _D0:_D0 + 8] = Am @ _dist_coef_jac(x, y, d, is_kb, has_b, has_k)
    dX = Gpc @ R
    return uv, z, dL, dX


# ====================================================================== problem assembly

def _cam_index(cgroup, c):
    if isinstance(c, (int, np.integer)):
        return int(c)
    names = cgroup.get_names()
    if c in names:
        return names.index(c)
    raise KeyError('unknown camera {!r}'.format(c))


def _share_groups(cgroup, share):
    """-> (K groups, dist groups) as lists of index lists."""
    if share is None:
        return [], []
    if isinstance(share, dict):
        gk, gd = [], []
        for key, groups in share.items():
            groups = [[_cam_index(cgroup, c) for c in g] for g in groups]
            if key in ('K', 'matrix', 'all', 'intrinsics'):
                gk += groups
            if key in ('dist', 'distortion', 'all', 'intrinsics'):
                gd += groups
            if key not in ('K', 'matrix', 'all', 'intrinsics', 'dist', 'distortion'):
                raise ValueError("share keys must be 'K', 'dist' or 'all', got {!r}".format(key))
        return gk, gd
    groups = [[_cam_index(cgroup, c) for c in g] for g in share]
    return groups, groups


def _velocities(obs, X, st_valid):
    """Per-point velocity [N,3] from tracks (group 'track' > 'keypoint' > one track) and times."""
    N = X.shape[0]
    t_pt = np.full(N, np.nan)
    np.minimum.at if False else None
    # point time: mean of its observation times
    cnt = np.bincount(obs.point_idx, minlength=N)
    tsum = np.bincount(obs.point_idx, weights=obs.time, minlength=N)
    with np.errstate(invalid='ignore', divide='ignore'):
        t_pt = tsum / cnt
    track = np.zeros(N, dtype=np.int64)
    if obs.group is not None:
        key = 'track' if 'track' in obs.group else ('keypoint' if 'keypoint' in obs.group else None)
        if key is not None:
            track[obs.point_idx] = obs.group[key]
    Xn = X.detach().cpu().numpy()
    V = np.zeros((N, 3))
    ok = np.isfinite(t_pt) & np.isfinite(Xn).all(1)
    for tr in np.unique(track[ok]):
        idx = np.nonzero(ok & (track == tr))[0]
        if len(idx) < 2:
            continue
        idx = idx[np.argsort(t_pt[idx], kind='stable')]
        tt, xx = t_pt[idx], Xn[idx]
        g = np.zeros_like(xx)
        dt_f = np.diff(tt)
        good = dt_f > 0
        # central differences where possible
        for i in range(len(idx)):
            lo, hi = max(i - 1, 0), min(i + 1, len(idx) - 1)
            if tt[hi] > tt[lo]:
                g[i] = (xx[hi] - xx[lo]) / (tt[hi] - tt[lo])
        V[idx] = g
    return torch.as_tensor(V, dtype=X.dtype, device=X.device)


class _Problem:
    """Parameter tables, state and per-observation data of one LM problem."""

    def __init__(self, cgroup, obs, spec, device, dtype, stations=None, rigid=None):
        self.cgroup = cgroup
        self.spec = spec
        self.device, self.dtype = device, dtype
        P = G.pack(cgroup, device=device, dtype=dtype)
        self.P = P
        C = P.n_cams
        self.C = C
        cams = list(cgroup.cameras)
        self.cams = cams
        self.names = cgroup.get_names()
        self.models = P.model_names()
        # ---- stations
        if obs.station is None:
            self.station_cam = np.arange(C)
            st_o = obs.cam_idx.copy()
            ext0 = torch.cat([P.rvec, P.tvec], -1)
            self.station_names = list(self.names)
        else:
            st_o = obs.station.copy()
            stations = dict(stations or {})
            if 'station_cam' in stations:
                station_cam = np.asarray(stations['station_cam'], dtype=np.int64)
            else:
                S = int(st_o.max()) + 1
                station_cam = np.full(S, -1, dtype=np.int64)
                station_cam[st_o] = obs.cam_idx
                if np.any(station_cam < 0):
                    raise ValueError('stations without observations need stations["station_cam"]')
            if np.any(station_cam[st_o] != obs.cam_idx):
                raise ValueError('obs.station is inconsistent with obs.cam_idx / station_cam')
            self.station_cam = station_cam
            ext0 = torch.cat([P.rvec, P.tvec], -1)[torch.as_tensor(station_cam, device=device)]
            if stations.get('rvec') is not None:
                ext0[:, :3] = torch.as_tensor(np.asarray(stations['rvec'], dtype='float64'), device=device, dtype=dtype)
            if stations.get('tvec') is not None:
                ext0[:, 3:] = torch.as_tensor(np.asarray(stations['tvec'], dtype='float64'), device=device, dtype=dtype)
            self.station_names = list(stations.get('names') or
                                      ['{}@{}'.format(self.names[c], s) for s, c in enumerate(station_cam)])
        self.S = len(self.station_cam)
        self.ext = ext0.clone()
        intr = torch.zeros(C, 13, dtype=dtype, device=device)
        intr[:, 0] = P.K[:, 0, 0]; intr[:, 1] = P.K[:, 1, 1]
        intr[:, 2] = P.K[:, 0, 2]; intr[:, 3] = P.K[:, 1, 2]
        intr[:, 4] = P.K[:, 0, 1]
        intr[:, 5:13] = P.dist
        self.intr = intr
        self.delta = torch.zeros(C, dtype=dtype, device=device)
        self.k10 = P.K[:, 1, 0].clone()
        self.off = P.offset.clone()
        self.is_kb = (P.model == 1)
        self.has_b = bool((P.model == 0).any())
        self.has_k = bool((P.model == 1).any())
        # ---- observations
        t = lambda a, dt=None: torch.as_tensor(a, device=device, dtype=dt)
        self.st_o = t(st_o, torch.long)
        self.cam_o = t(self.station_cam[st_o], torch.long)
        self.uv = t(obs.uv, dtype)
        self.conf = t(obs.weight, dtype) if obs.weight is not None else torch.ones(len(obs), dtype=dtype, device=device)
        self.M = len(obs)
        self.obs = obs
        self.rigid = rigid      # (set_idx [N] (-1 = free point), objp [N,3]) or None

    # ------------------------------------------------------------------ tables
    def resolve_fixed(self, frozen_cams=()):
        spec = self.spec
        C, S = self.C, self.S
        ext_free = np.ones(S, bool)
        if isinstance(spec.extrinsics, dict):
            for c, v in spec.extrinsics.items():
                ci = _cam_index(self.cgroup, c)
                if v not in ('free', 'fixed'):
                    raise ValueError("extrinsics values must be 'free' or 'fixed'")
                if v == 'fixed':
                    ext_free[self.station_cam == ci] = False
        elif spec.extrinsics == 'fixed':
            ext_free[:] = False
        fixed_st = []
        fc = spec.fixed_cameras
        if isinstance(fc, str) and fc == 'auto':
            if ext_free.all() and S > 0:
                order = np.argsort(self.station_cam, kind='stable')
                fixed_st = [int(order[0])]
        elif fc is not None:
            if isinstance(fc, (str, int, np.integer)):
                fc = [fc]
            for c in fc:
                ci = _cam_index(self.cgroup, c)
                fixed_st += [int(s) for s in np.nonzero(self.station_cam == ci)[0]]
        for s in fixed_st:
            ext_free[s] = False
        for ci in frozen_cams:
            ext_free[self.station_cam == ci] = False
        self.ext_free = ext_free
        self.fixed_stations = sorted(set(fixed_st) | set(np.nonzero(~ext_free)[0].tolist()))
        self.gauge_stations = fixed_st

    def build(self, frozen_cams=()):
        spec = self.spec
        dev, dt = self.device, self.dtype
        C, S = self.C, self.S
        self.frozen_cams = set(int(c) for c in frozen_cams)
        self.resolve_fixed(self.frozen_cams)
        n = 0
        idx_ext = -np.ones((S, 6), np.int64)
        for s in range(S):
            if self.ext_free[s]:
                idx_ext[s] = np.arange(n, n + 6); n += 6
        idx_cam = -np.ones((C, 14), np.int64)
        coef_cam = np.zeros((C, 14))
        gk, gd = _share_groups(self.cgroup, spec.share)
        rep_k = list(range(C)); rep_d = list(range(C))
        for g in gk:
            for c in g:
                rep_k[c] = g[0]
        for g in gd:
            for c in g:
                rep_d[c] = g[0]
        intr_np = self.intr.detach().cpu().numpy()
        # make shared values identical (mean over the group)
        with torch.no_grad():
            for g in gk:
                self.intr[g, 0:5] = self.intr[g, 0:5].mean(0, keepdim=True)
            for g in gd:
                self.intr[g, 5:13] = self.intr[g, 5:13].mean(0, keepdim=True)
        intr_np = self.intr.detach().cpu().numpy()
        slot_of = {}          # (rep, key) -> theta index
        self.free_names = {}
        for c in range(C):
            if c in self.frozen_cams:
                continue
            mat, didx, use_f = spec.camera_names_free(self.cams[c], self.models[c])
            names = []
            for nm in ('fx', 'fy', 'cx', 'cy', 'skew'):
                if nm in mat:
                    key = (rep_k[c], nm)
                    if key not in slot_of:
                        slot_of[key] = n; n += 1
                    k = _MAT_SLOT[nm] - 6
                    idx_cam[c, k] = slot_of[key]; coef_cam[c, k] = 1.0
                    names.append(nm)
            if use_f:
                key = (rep_k[c], 'f')
                if key not in slot_of:
                    slot_of[key] = n; n += 1
                idx_cam[c, 0] = slot_of[key]; coef_cam[c, 0] = 1.0
                idx_cam[c, 1] = slot_of[key]; coef_cam[c, 1] = intr_np[c, 1] / intr_np[c, 0]
                names.append('f')
            for j in didx:
                key = (rep_d[c], 'd{}'.format(j))
                if key not in slot_of:
                    slot_of[key] = n; n += 1
                idx_cam[c, 5 + j] = slot_of[key]; coef_cam[c, 5 + j] = 1.0
                names.append(DIST_NAMES[self.models[c]][j])
            self.free_names[self.names[c]] = names
        # time offsets
        self.use_delta = spec.time_offset is not None
        if self.use_delta:
            if self.obs.time is None:
                raise ValueError('ParamSpec(time_offset=...) needs Observations.time')
            gauge_cams = set(int(self.station_cam[s]) for s in self.gauge_stations) or {0}
            for c in range(C):
                if c in gauge_cams or c in self.frozen_cams:
                    continue
                idx_cam[c, 13] = n; coef_cam[c, 13] = 1.0; n += 1
        self.NC = n
        self.idx_ext, self.idx_cam, self.coef_cam = idx_ext, idx_cam, coef_cam
        idx_loc = np.concatenate([idx_ext, idx_cam[self.station_cam]], 1)          # [S,20]
        coef_loc = np.concatenate([np.ones((S, 6)), coef_cam[self.station_cam]], 1)
        active = np.nonzero((idx_loc >= 0).any(0))[0]
        self.active = torch.as_tensor(active, device=dev, dtype=torch.long)
        Q = len(active)
        self.Q = Q
        Pm = torch.zeros(S, Q, max(n, 1), dtype=dt, device=dev)
        il, cl = idx_loc[:, active], coef_loc[:, active]
        ss, qq = np.nonzero(il >= 0)
        Pm[ss, qq, il[ss, qq]] = torch.as_tensor(cl[ss, qq], dtype=dt, device=dev)
        self.Pm = Pm
        # tables for the update
        self.t_idx_ext = torch.as_tensor(idx_ext, device=dev)
        self.t_idx_cam = torch.as_tensor(idx_cam, device=dev)
        self.t_coef_cam = torch.as_tensor(coef_cam, device=dev, dtype=dt)
        # priors
        self.priors = self._priors()

    def _priors(self):
        spec = self.spec
        pri = spec.priors
        if pri is None:
            return []
        if isinstance(pri, IntrinsicsPrior):
            pri = pri.priors_for(self.cams)
        out = []
        for c, d in pri.items():
            ci = _cam_index(self.cgroup, c)
            for nm, val in d.items():
                if len(val) == 2:
                    mu, sig = val; kind = 'lin'
                else:
                    mu, sig, kind = val
                kind = 'log' if kind == 'log' else 'lin'
                if nm == 'f':
                    keys = [0, 1]
                elif nm in _MAT_SLOT:
                    keys = [_MAT_SLOT[nm] - 6]
                else:
                    j = dist_index(nm, self.models[ci])
                    if j is None:
                        raise ValueError('camera {} ({}) has no distortion {!r}'.format(
                            self.names[ci], self.models[ci], nm))
                    keys = [5 + j]
                for k in keys:
                    if self.idx_cam[ci, k] >= 0:
                        out.append((ci, k, float(mu), float(sig), kind))
        return out

    # ------------------------------------------------------------------ state
    def local(self, ext, intr, delta):
        cs = torch.as_tensor(self.station_cam, device=self.device)
        return torch.cat([ext, intr[cs], delta[cs][:, None]], 1)       # [S,20]

    def apply_step(self, ext, intr, delta, dth):
        dthp = torch.cat([dth, dth.new_zeros(1)])
        ie = torch.where(self.t_idx_ext >= 0, self.t_idx_ext, torch.full_like(self.t_idx_ext, self.NC))
        ext = ext + dthp[ie]
        ic = torch.where(self.t_idx_cam >= 0, self.t_idx_cam, torch.full_like(self.t_idx_cam, self.NC))
        dc = dthp[ic] * self.t_coef_cam
        intr = intr + dc[:, :13]
        delta = delta + dc[:, 13]
        return ext, intr, delta

    def centers(self, ext):
        R = G.rodrigues(ext[:, :3])
        return -(R.transpose(-1, -2) @ ext[:, 3:, None])[..., 0]


# ====================================================================== structures (points / rigid boards)

class _Kind:
    """Observations of one structure type: free 3D points (D=3) or rigid point sets (D=6)."""

    def __init__(self, name, D, sel, sidx, values, objp=None, vel=None):
        self.name, self.D = name, D
        self.sel = sel            # [m] indices into the problem's observations
        self.sidx = sidx          # [m] structure index
        self.values = values      # [Ns, D]
        self.objp = objp          # [m, 3] (rigid)
        self.vel = vel            # [Ns, 3] (points with time offsets)

    @property
    def n(self):
        return self.values.shape[0]

    def world(self, Sv, delta_o, rep=1):
        if self.name == 'rigid':
            R = G.rodrigues(Sv[:, :3])
            objp = self.objp.repeat(rep, 1) if rep > 1 else self.objp
            return torch.einsum('mij,mj->mi', R, objp) + Sv[:, 3:6]
        if self.vel is not None:
            vel = self.vel[self.sidx]
            vel = vel.repeat(rep, 1) if rep > 1 else vel
            return Sv + delta_o[:, None] * vel
        return Sv


def _kind_project(prob, kind, Lst, Sv):
    st = prob.st_o[kind.sel]
    cam = prob.cam_o[kind.sel]
    Lo = Lst[st]
    So = Sv[kind.sidx]
    k10, off, kb = prob.k10[cam], prob.off[cam], prob.is_kb[cam]

    def f(L_, S_, rep=1):
        """rep: rows of L_/S_ are the m observations repeated rep times."""
        rp = (lambda a: a.repeat((rep,) + (1,) * (a.ndim - 1))) if rep > 1 else (lambda a: a)
        return _project_local(L_, kind.world(S_, L_[:, _DELTA], rep), rp(k10), rp(off), rp(kb),
                              prob.has_b, prob.has_k)[0]
    return f, Lo, So, cam


def _kind_residual(prob, kind, Lst, Sv):
    f, Lo, So, cam = _kind_project(prob, kind, Lst, Sv)
    st = prob.st_o[kind.sel]
    uvp, z = _project_local(Lo, kind.world(So, Lo[:, _DELTA]), prob.k10[cam], prob.off[cam],
                            prob.is_kb[cam], prob.has_b, prob.has_k)
    r = uvp - prob.uv[kind.sel]
    return r, z


def _jac_columns(fn, x, basis):
    """d fn / d x along each basis vector -> [K, m, ...out], by forward-mode AD on a batch where
    the m rows are repeated once per basis vector (one dual-number pass, no vmap)."""
    import torch.autograd.forward_ad as fwAD
    K, m = basis.shape[0], x.shape[0]
    xr = x.unsqueeze(0).expand(K, *x.shape).reshape(K * m, *x.shape[1:])
    tr = basis[:, None, :].expand(K, m, basis.shape[1]).reshape(K * m, basis.shape[1])
    with fwAD.dual_level():
        out = fn(fwAD.make_dual(xr, tr), K)
        tan = fwAD.unpack_dual(out).tangent
    if tan is None:
        tan = torch.zeros_like(fwAD.unpack_dual(out).primal)
    return tan.reshape(K, m, *tan.shape[1:])


def _kind_linearize(prob, kind, Lst, Sv):
    """Residuals [m,2], depth [m], camera Jacobian [m,2,Q] (active slots), structure Jacobian [m,2,D]."""
    st = prob.st_o[kind.sel]
    cam = prob.cam_o[kind.sel]
    Lo = Lst[st]
    So = Sv[kind.sidx]
    Xw = kind.world(So, Lo[:, _DELTA])
    uvp, z, dL, dX = _project_local_jac(Lo, Xw, prob.k10[cam], prob.off[cam], prob.is_kb[cam],
                                        prob.has_b, prob.has_k)
    r = uvp - prob.uv[kind.sel]
    if kind.name == 'rigid':
        Rb, Jb = _rot_and_jac(So[:, :3])
        dw = -Rb @ _skew(kind.objp) @ Jb
        Js = torch.cat([dX @ dw, dX], -1)                                           # [m,2,6]
    else:
        Js = dX
        if kind.vel is not None:
            dL[:, :, _DELTA] = (dX @ kind.vel[kind.sidx][..., None])[..., 0]
    Jc = dL[:, :, prob.active]
    return r, z, Jc, Js


def _chunks_by_struct(kind, S, Q, max_elems):
    """Chunks of structure indices [s0, s1) and the observation masks of each."""
    per = max(1, S * Q * kind.D)
    step = max(1, int(max_elems // per))
    out = []
    for s0 in range(0, kind.n, step):
        s1 = min(kind.n, s0 + step)
        m = (kind.sidx >= s0) & (kind.sidx < s1)
        out.append((s0, s1, m))
    return out


# ====================================================================== LM core

def _bad_penalty(loss, f):
    e2 = torch.tensor((30.0 * f) ** 2, dtype=torch.float64)
    return float(_rob(e2, loss, f)[0])


def _lm_solve(prob, kinds, loss='cauchy', f_scale=5.0, n_iter=25, lam=1e-3, tol=1e-7,
              scale_mode=None, scale_pair=None, solver='auto', max_elems=1 << 24, verbose=False):
    """Schur-complement LM. Returns (ext, intr, delta, values list, history, converged)."""
    dev, dt = prob.device, prob.dtype
    S, Q, NC = prob.S, prob.Q, prob.NC
    Pm2 = prob.Pm.reshape(S * Q, -1)
    bad_pen = _bad_penalty(loss, f_scale)
    st_k = [prob.st_o[k.sel] for k in kinds]
    conf_k = [prob.conf[k.sel] for k in kinds]
    chunks = [_chunks_by_struct(k, S, Q, max_elems) for k in kinds]
    if solver == 'auto':
        solver = 'dense' if S * Q <= 1500 else 'pcg'
    sp = prob.spec

    # ---- gauge helpers
    fixed_st = list(prob.gauge_stations)
    scale0 = None
    if scale_mode == 'keep':
        scale0 = rig_scale_from_centers(prob.centers(prob.ext))
    pair_idx = None
    if scale_pair is not None:
        a, b = scale_pair[0], scale_pair[1]
        ia = prob.station_names.index(a) if a in prob.station_names else _cam_index(prob.cgroup, a)
        ib = prob.station_names.index(b) if b in prob.station_names else _cam_index(prob.cgroup, b)
        pair_idx = (ia, ib, float(scale_pair[2]), (float(scale_pair[3]) if len(scale_pair) > 3 and scale_pair[3] else None))

    def renorm(ext, values):
        if scale_mode == 'keep':
            target = scale0
            cur = rig_scale_from_centers(prob.centers(ext))
        elif pair_idx is not None and pair_idx[3] is None:
            cen = prob.centers(ext)
            target = pair_idx[2]
            cur = float(torch.linalg.norm(cen[pair_idx[0]] - cen[pair_idx[1]]))
        else:
            return ext, values
        if not (cur > 0 and np.isfinite(cur)):
            return ext, values
        s = target / cur
        cen = prob.centers(ext)
        pivot = cen[fixed_st[0]] if fixed_st else cen.mean(0)
        R = G.rodrigues(ext[:, :3])
        newc = pivot + s * (cen - pivot)
        ext = torch.cat([ext[:, :3], -(R @ newc[:, :, None])[..., 0]], 1)
        if fixed_st:
            ext[fixed_st] = prob.ext_fixed_ref[fixed_st]
        out = []
        for k, v in zip(kinds, values):
            if k.name == 'rigid':
                v = torch.cat([v[:, :3], pivot + s * (v[:, 3:] - pivot)], 1)
            else:
                v = pivot + s * (v - pivot)
            out.append(v)
        return ext, out
    prob.ext_fixed_ref = prob.ext.clone()

    def prior_terms(ext, intr, delta):
        """(cost, H [NC,NC], g [NC]) of the priors in theta space."""
        H = torch.zeros(NC, NC, dtype=dt, device=dev)
        g = torch.zeros(NC, dtype=dt, device=dev)
        cost = 0.0
        for (c, k, mu, sig, kind) in prob.priors:
            i = int(prob.idx_cam[c, k]); a = float(prob.coef_cam[c, k])
            val = intr[c, k]
            if kind == 'log':
                r = (torch.log(val) - math.log(mu)) / sig
                J = a / (float(val) * sig)
            else:
                r = (val - mu) / sig
                J = a / sig
            cost += float(r * r)
            H[i, i] += J * J
            g[i] += J * r
        if prob.use_delta:
            sig = float(sp.time_offset)
            for c in range(prob.C):
                i = int(prob.idx_cam[c, 13])
                if i >= 0:
                    r = delta[c] / sig
                    cost += float(r * r)
                    H[i, i] += 1.0 / sig ** 2
                    g[i] += r / sig
        if pair_idx is not None and pair_idx[3] is not None:
            ia, ib, d0, sig = pair_idx

            def res(ea, eb):
                Ra, Rb = G.rodrigues(ea[:3]), G.rodrigues(eb[:3])
                ca, cb = -Ra.T @ ea[3:], -Rb.T @ eb[3:]
                return (torch.linalg.norm(ca - cb) - d0) / sig
            r = res(ext[ia], ext[ib])
            Ja, Jb = torch.autograd.functional.jacobian(res, (ext[ia], ext[ib]))
            Jt = torch.zeros(NC, dtype=dt, device=dev)
            for s_, J_ in ((ia, Ja), (ib, Jb)):
                ii = prob.t_idx_ext[s_]
                ok = ii >= 0
                Jt.index_add_(0, ii[ok], J_[ok])
            cost += float(r * r)
            H += torch.outer(Jt, Jt)
            g += Jt * r
        return cost, H, g

    def total_cost(ext, intr, delta, values):
        Lst = prob.local(ext, intr, delta)
        cst = 0.0
        for k, v, cf in zip(kinds, values, conf_k):
            r, z = _kind_residual(prob, k, Lst, v)
            e2 = (r * r).sum(-1)
            ok = (z > 0) & torch.isfinite(e2)
            rho = _rob(torch.where(ok, e2, torch.zeros_like(e2)), loss, f_scale)[0]
            cst += float((cf * torch.where(ok, rho, torch.full_like(rho, bad_pen))).sum())
        return cst + prior_terms(ext, intr, delta)[0]

    ext, intr, delta = prob.ext.clone(), prob.intr.clone(), prob.delta.clone()
    values = [k.values.clone() for k in kinds]
    cost = total_cost(ext, intr, delta, values)
    history = [dict(iter=0, cost=cost, lam=lam, accepted=True)]
    converged = False
    eyeQ = torch.eye(max(Q, 1), dtype=dt, device=dev)[:Q, :Q]
    for it in range(n_iter):
        Lst = prob.local(ext, intr, delta)
        U_loc = torch.zeros(S, Q, Q, dtype=dt, device=dev)
        g_loc = torch.zeros(S, Q, dtype=dt, device=dev)
        lin = []
        for k, v, st, cf in zip(kinds, values, st_k, conf_k):
            r, z, Jc, Js = _kind_linearize(prob, k, Lst, v)
            e2 = (r * r).sum(-1)
            ok = (z > 0) & torch.isfinite(e2) & torch.isfinite(Jc).all(-1).all(-1) & torch.isfinite(Js).all(-1).all(-1)
            w = cf * _rob(torch.where(ok, e2, torch.zeros_like(e2)), loss, f_scale)[1]
            w = torch.where(ok, w, torch.zeros_like(w))
            r, Jc, Js = torch.nan_to_num(r), torch.nan_to_num(Jc), torch.nan_to_num(Js)
            U_loc.index_add_(0, st, torch.einsum('m,mki,mkj->mij', w, Jc, Jc))
            g_loc.index_add_(0, st, torch.einsum('m,mki,mk->mi', w, Jc, r))
            Hcs = torch.einsum('m,mki,mkj->mij', w, Jc, Js)                     # [m,Q,D]
            V = torch.zeros(k.n, k.D, k.D, dtype=dt, device=dev).index_add_(
                0, k.sidx, torch.einsum('m,mki,mkj->mij', w, Js, Js))
            gs = torch.zeros(k.n, k.D, dtype=dt, device=dev).index_add_(
                0, k.sidx, torch.einsum('m,mki,mk->mi', w, Js, r))
            lin.append((Hcs, V, gs, st))
        Uth = torch.einsum('sqa,sqr,srb->ab', prob.Pm, U_loc, prob.Pm)[:NC, :NC]
        gth = torch.einsum('sqa,sq->a', prob.Pm, g_loc)[:NC]
        pc, pH, pg = prior_terms(ext, intr, delta)
        Uth = Uth + pH
        gth = gth + pg
        dU = Uth.diagonal().clone()
        dU = torch.where(dU > 0, dU, torch.full_like(dU, float(dU.max()) * 1e-12 + 1e-12)) if NC else dU
        improved = False
        while lam <= 1e10:
            Vinvs = []
            bcorr = torch.zeros(S, Q, dtype=dt, device=dev)
            for (Hcs, V, gs, st), k in zip(lin, kinds):
                eyeD = torch.eye(k.D, dtype=dt, device=dev)
                Vd = V + lam * V.diagonal(dim1=-2, dim2=-1)[..., None] * eyeD + 1e-9 * eyeD * \
                    (1 + V.diagonal(dim1=-2, dim2=-1).abs().amax(-1)[:, None, None])
                Vinv = torch.linalg.inv(Vd)
                Vinvs.append(Vinv)
                if Q:
                    bcorr.index_add_(0, st, torch.einsum('mqd,md->mq', Hcs, (Vinv @ gs[..., None])[..., 0][k.sidx]))
            b = -(gth - torch.einsum('sqa,sq->a', prob.Pm, bcorr)[:NC])
            if NC == 0:
                dth = torch.zeros(0, dtype=dt, device=dev)
            elif solver == 'dense':
                Sl = torch.zeros(S, Q, S, Q, dtype=dt, device=dev)
                for (Hcs, V, gs, st), k, Vinv, ch in zip(lin, kinds, Vinvs, chunks):
                    for s0, s1, m in ch:
                        if not bool(m.any()):
                            continue
                        A = torch.zeros(s1 - s0, S, Q, k.D, dtype=dt, device=dev)
                        A.index_put_((k.sidx[m] - s0, st[m]), Hcs[m], accumulate=True)
                        AV = torch.einsum('nsqd,nde->nsqe', A, Vinv[s0:s1])
                        Sl -= torch.einsum('nsqe,nrpe->sqrp', AV, A)
                Hs = torch.einsum('sqa,sqrp,rpb->ab', prob.Pm, Sl, prob.Pm)[:NC, :NC]
                Hd = Uth + Hs + torch.diag(lam * dU)
                try:
                    Lc = torch.linalg.cholesky(Hd)
                    dth = torch.cholesky_solve(b[:, None], Lc)[:, 0]
                except Exception:
                    try:
                        dth = torch.linalg.solve(Hd, b)
                    except Exception:
                        lam *= 10
                        continue
            else:
                dth = _pcg(prob, lin, kinds, Vinvs, Uth + torch.diag(lam * dU), b, Pm2)
            if not bool(torch.isfinite(dth).all()):
                lam *= 10
                continue
            dl = (Pm2[:, :NC] @ dth).reshape(S, Q) if NC else torch.zeros(S, Q, dtype=dt, device=dev)
            new_values = []
            for (Hcs, V, gs, st), k, Vinv, v in zip(lin, kinds, Vinvs, values):
                y = torch.zeros(k.n, k.D, dtype=dt, device=dev)
                if Q:
                    y.index_add_(0, k.sidx, torch.einsum('mqd,mq->md', Hcs, dl[st]))
                ds = -(Vinv @ (gs + y)[..., None])[..., 0]
                new_values.append(v + ds)
            ext_n, intr_n, delta_n = prob.apply_step(ext, intr, delta, dth)
            ext_n, new_values = renorm(ext_n, new_values)
            new_cost = total_cost(ext_n, intr_n, delta_n, new_values)
            if np.isfinite(new_cost) and new_cost < cost:
                rel = (cost - new_cost) / max(abs(cost), 1e-300)
                ext, intr, delta, values, cost = ext_n, intr_n, delta_n, new_values, new_cost
                lam = max(lam / 3, 1e-9)
                improved = True
                break
            lam *= 5
        history.append(dict(iter=it + 1, cost=cost, lam=lam, accepted=improved))
        if verbose:
            print('LM iter {:3d}: cost {:.6e}  lambda {:.1e}{}'.format(it + 1, cost, lam, '' if improved else ' (no step)'))
        if not improved:
            converged = True
            break
        if rel < tol:
            converged = True
            break
    return ext, intr, delta, values, history, converged


def _pcg(prob, lin, kinds, Vinvs, Hc, b, Pm2, max_iter=500, rtol=1e-10):
    """Matrix-free preconditioned CG on the reduced camera system (large camera counts)."""
    S, Q, NC = prob.S, prob.Q, prob.NC
    P2 = Pm2[:, :NC]

    def matvec(v):
        out = Hc @ v
        vl = (P2 @ v).reshape(S, Q)
        acc = torch.zeros(S, Q, dtype=v.dtype, device=v.device)
        for (Hcs, V, gs, st), k, Vinv in zip(lin, kinds, Vinvs):
            y = torch.zeros(k.n, k.D, dtype=v.dtype, device=v.device).index_add_(
                0, k.sidx, torch.einsum('mqd,mq->md', Hcs, vl[st]))
            z = (Vinv @ y[..., None])[..., 0]
            acc.index_add_(0, st, torch.einsum('mqd,md->mq', Hcs, z[k.sidx]))
        return out + 0 - P2.T @ acc.reshape(-1)

    Minv = 1.0 / Hc.diagonal().clamp(min=1e-300)
    x = torch.zeros_like(b)
    r = b.clone()
    z = Minv * r
    p = z.clone()
    rz = r @ z
    bn = float(torch.linalg.norm(b)) + 1e-300
    for _ in range(max_iter):
        Ap = matvec(p)
        alpha = rz / (p @ Ap)
        x = x + alpha * p
        r = r - alpha * Ap
        if float(torch.linalg.norm(r)) < rtol * bn:
            break
        z = Minv * r
        rz_new = r @ z
        p = z + (rz_new / rz) * p
        rz = rz_new
    return x


# ====================================================================== LM driver

def _check_group(cgroup, what):
    chk = getattr(cgroup, '_check_not_frozen', None)
    if chk is not None:
        chk(what)


def _resolve_spec(cgroup, spec, only_extrinsics=False):
    if spec is None:
        spec = getattr(cgroup, 'param_spec', None)
    if spec is None:
        spec = ParamSpec(intrinsics=None)
    elif not isinstance(spec, ParamSpec):
        raise TypeError('spec must be a ParamSpec, got {}'.format(type(spec).__name__))
    if only_extrinsics:
        spec = spec.replace(intrinsics=(), n_dist=0, time_offset=spec.time_offset)
    return spec


def _station_packed(prob, ext):
    cs = list(int(c) for c in prob.station_cam)
    Ps = prob.P.subset(cs)
    Ps = dataclasses.replace(Ps, rvec=ext[:, :3].clone(), tvec=ext[:, 3:].clone(), ext=None)
    Ps.K = Ps.K.clone()
    Ps.K[:, 0, 0] = prob.intr[cs, 0]; Ps.K[:, 1, 1] = prob.intr[cs, 1]
    Ps.K[:, 0, 2] = prob.intr[cs, 2]; Ps.K[:, 1, 2] = prob.intr[cs, 3]
    Ps.K[:, 0, 1] = prob.intr[cs, 4]
    Ps.dist = prob.intr[cs, 5:13].clone()
    return Ps


def _triangulate_obs(prob, obs, point_ids):
    """Linear triangulation of the points ``point_ids`` from the observations (per station)."""
    dev, dt = prob.device, prob.dtype
    Ps = _station_packed(prob, prob.ext)
    N = len(point_ids)
    remap = -np.ones(obs.n_points, np.int64)
    remap[point_ids] = np.arange(N)
    st = obs.station if obs.station is not None else obs.cam_idx
    m = remap[obs.point_idx] >= 0
    dense = torch.full((prob.S, N, 2), float('nan'), dtype=dt, device=dev)
    dense[torch.as_tensor(st[m], device=dev), torch.as_tensor(remap[obs.point_idx[m]], device=dev)] = \
        torch.as_tensor(obs.uv[m], dtype=dt, device=dev)
    return T.triangulate_linear(Ps, dense)


def _umeyama(src, dst, with_scale=True):
    mu_s, mu_d = src.mean(0), dst.mean(0)
    a, b = src - mu_s, dst - mu_d
    U, D, Vt = np.linalg.svd(b.T @ a / len(src))
    Sg = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        Sg[2, 2] = -1
    R = U @ Sg @ Vt
    s = float(np.trace(np.diag(D) @ Sg) / max((a * a).sum() / len(src), 1e-300)) if with_scale else 1.0
    t = mu_d - s * R @ mu_s
    return s, R, t


def _init_boards(X, set_idx, objp, board_ids, extra=None, prob=None, obs=None):
    """Board poses [B,6] (rvec, tvec) mapping objp to the triangulated corners; NaN rows if failed.
    Also returns the median similarity scale (objp units -> rig units)."""
    Xn = X.detach().cpu().numpy()
    out = np.full((len(board_ids), 6), np.nan)
    scales = []
    for bi, b in enumerate(board_ids):
        idx = np.nonzero(set_idx == b)[0]
        ok = idx[np.isfinite(Xn[idx]).all(1)]
        if len(ok) >= 3:
            src, dst = objp[ok], Xn[ok]
            if np.linalg.matrix_rank(src - src.mean(0), tol=1e-9 * (np.abs(src).max() + 1)) >= 2:
                s, R, t = _umeyama(src, dst, True)
                scales.append(s)
                out[bi, :3] = G._rotmat_to_rvec_np(R[None])[0]
                out[bi, 3:] = t
                out[bi, 3:] = t   # scale applied later (the rig may be rescaled to board units)
                out[bi] = np.r_[out[bi, :3], t]
                continue
        if extra is not None and 'rvecs' in extra and 'tvecs' in extra and prob is not None:
            # compose a camera's board pose with that camera's extrinsics (as the Adam path)
            rv, tv = np.asarray(extra['rvecs']), np.asarray(extra['tvecs'])
            for p in idx:
                cs = np.nonzero(np.isfinite(rv[:, p]).all(-1))[0] if rv.ndim == 3 else []
                if len(cs):
                    c = cs[0]
                    Rc = G._rotmat_to_rvec_np  # noqa
                    import cv2
                    Rcam = cv2.Rodrigues(prob.P.rvec[c].detach().cpu().numpy())[0]
                    tcam = prob.P.tvec[c].detach().cpu().numpy()
                    Rb = cv2.Rodrigues(rv[c, p].reshape(3))[0]
                    tb = tv[c, p].reshape(3)
                    Rw = Rcam.T @ Rb
                    tw = Rcam.T @ (tb - tcam)
                    out[bi] = np.r_[cv2.Rodrigues(Rw)[0].ravel(), tw]
                    break
    scale = float(np.median(scales)) if scales else 1.0
    return out, scale


def bundle_adjust_lm(cgroup, obs_or_p2ds, p3d_init=None, spec=None, loss='cauchy', f_scale=5.0,
                     n_iter=25, max_points=None, prune=None, extra=None, stations=None,
                     only_extrinsics=False, min_points_per_camera=10, inplace=True,
                     revert_if_worse=False, return_report=False, device=None, rng=None,
                     tol=1e-7, lam=1e-3, solver='auto', verbose=False):
    """Schur-complement Levenberg-Marquardt bundle adjustment.

    Parameters
    ----------
    cgroup : CameraGroup
    obs_or_p2ds : :class:`Observations`, or dense pixels ``C x N x 2`` / ``C x N x J x 2`` (NaN = missing)
    p3d_init : initial points ``[n_points, 3]`` (point indexing of the observations); default:
        linear triangulation with the current cameras
    spec : :class:`~aniposelib.params.ParamSpec`; default ``cgroup.param_spec`` if set, else each
        camera's own masks (``free_intrinsics`` + ``n_dist``), all extrinsics free, first camera fixed
    loss, f_scale : robust loss (``'linear' | 'huber' | 'cauchy' | 'soft_l1'``) and its scale in px
    n_iter : maximum LM iterations; ``tol``: relative cost decrease to stop; ``lam``: initial damping
    max_points : subsample this many points (or boards) with ``rng``
    prune : ``dict(max_px=None, min_depth=0.0)``: drop observations whose initial error exceeds
        ``max_px`` or whose depth is ``<= min_depth``, then points with fewer than 2 observations
    extra : anipose board dict (``ids``, ``objp``, optional ``rvecs``/``tvecs``) for dense input:
        points with the same board id are corners of one rigid board (6 pose parameters per board)
    stations : F5, ``dict(station_cam=[S], rvec=[S,3], tvec=[S,3], names=[S])`` for ``obs.station``
    only_extrinsics : shortcut for a spec with all intrinsics fixed
    min_points_per_camera : cameras with fewer observations are frozen (and reported)
    inplace : write the result into ``cgroup`` (else into a copy, returned as ``report.cgroup``)
    revert_if_worse : keep the old parameters if the median error got worse
    return_report : return a :class:`BAReport` instead of the final median error (float)
    solver : ``'dense'`` (Cholesky on the reduced camera system), ``'pcg'`` (matrix free) or ``'auto'``
    """
    t0 = _time.time()
    _check_group(cgroup, 'bundle_adjust_lm')
    if loss not in LOSSES:
        raise ValueError('unknown loss {!r}; use one of {}'.format(loss, LOSSES))
    spec = _resolve_spec(cgroup, spec, only_extrinsics)
    if not inplace:
        from copy import deepcopy
        cgroup = deepcopy(cgroup)
    rng_ = _as_rng(rng)
    device = torch.device(device) if device is not None else cgroup.cameras[0].matrix.device
    dtype = cgroup.cameras[0].matrix.dtype
    if isinstance(obs_or_p2ds, Observations):
        obs = obs_or_p2ds
    else:
        obs = Observations.from_dense(obs_or_p2ds)
    if obs.n_cams > len(cgroup.cameras) or (len(obs) and obs.cam_idx.max() >= len(cgroup.cameras)):
        raise ValueError('observations refer to {} cameras, the group has {}'.format(obs.n_cams, len(cgroup.cameras)))
    obs.n_cams = len(cgroup.cameras)
    report = BAReport(method='lm', seed=rng if isinstance(rng, (int, np.integer)) else None)
    report.config = dict(loss=loss, f_scale=float(f_scale), n_iter=int(n_iter), max_points=max_points,
                         prune=prune, spec=spec.to_dict(), solver=solver, tol=tol)
    N = obs.n_points

    # ---- rigid sets (boards)
    set_idx = -np.ones(N, np.int64)
    objp = None
    if extra is not None:
        ids = np.asarray(_np(extra['ids']), dtype=float).ravel()
        objp = np.asarray(_np(extra['objp']), dtype=float).reshape(-1, 3)
        if len(ids) != N:
            raise ValueError('extra["ids"] has {} entries, expected {}'.format(len(ids), N))
        set_idx = np.where(np.isfinite(ids), ids, -1).astype(np.int64)

    # ---- subsample
    if max_points is not None:
        free_pts = np.unique(obs.point_idx[set_idx[obs.point_idx] < 0])
        boards = np.unique(set_idx[set_idx >= 0])
        keep = np.zeros(N, bool)
        if len(free_pts) > max_points:
            keep[rng_.choice(free_pts, size=max_points, replace=False)] = True
        else:
            keep[free_pts] = True
        if len(boards):
            nb = len(boards)
            if objp is not None and nb * max(1, int(np.median(np.bincount(set_idx[set_idx >= 0])))) > max_points:
                per = max(1, int(np.median(np.bincount(set_idx[set_idx >= 0])[boards])))
                nb = max(1, max_points // per)
            chosen = boards if nb >= len(boards) else rng_.choice(boards, size=nb, replace=False)
            keep |= np.isin(set_idx, chosen)
        obs_sel = keep[obs.point_idx]
        report.config['n_obs_sampled_out'] = int((~obs_sel).sum())
        obs = obs.subset(obs_sel)

    prob = _Problem(cgroup, obs, spec, device, dtype, stations=stations)
    S = prob.S

    # ---- initial structure
    pts = np.unique(obs.point_idx)
    if p3d_init is not None:
        X_all = torch.as_tensor(np.asarray(_np(p3d_init), dtype='float64').reshape(-1, 3), dtype=dtype, device=device)
        if X_all.shape[0] != N:
            raise ValueError('p3d_init has {} points, expected {}'.format(X_all.shape[0], N))
    else:
        X_all = torch.full((N, 3), float('nan'), dtype=dtype, device=device)
        if len(pts):
            X_all[torch.as_tensor(pts, device=device)] = _triangulate_obs(prob, obs, pts)
    # boards
    board_ids = np.unique(set_idx[pts][set_idx[pts] >= 0]) if len(pts) else np.zeros(0, np.int64)
    B = None
    if len(board_ids):
        poses, bscale = _init_boards(X_all, set_idx, objp, board_ids, extra=extra, prob=prob, obs=obs)
        if abs(bscale - 1) > 0.01 and np.isfinite(bscale) and bscale > 0:
            # the rig is not in board units: rescale translations so the board fixes the scale
            with torch.no_grad():
                prob.ext[:, 3:] /= bscale
            X_all = X_all / bscale
            poses[:, 3:] /= bscale
            report.warnings.append('rig rescaled by {:.4g} to board units'.format(1 / bscale))
        B = torch.as_tensor(poses, dtype=dtype, device=device)

    # ---- prune
    prune = dict(prune or {})
    max_px = prune.get('max_px')
    min_depth = float(prune.get('min_depth', 0.0) or 0.0)
    is_rigid_o = set_idx[obs.point_idx] >= 0
    bmap = -np.ones(int(set_idx.max()) + 2 if len(set_idx) else 1, np.int64)
    if B is not None:
        bmap[board_ids] = np.arange(len(board_ids))
    Xw = torch.full((len(obs), 3), float('nan'), dtype=dtype, device=device)
    fo = ~is_rigid_o
    if fo.any():
        Xw[torch.as_tensor(np.nonzero(fo)[0], device=device)] = X_all[torch.as_tensor(obs.point_idx[fo], device=device)]
    if is_rigid_o.any():
        ro = np.nonzero(is_rigid_o)[0]
        bp = B[torch.as_tensor(bmap[set_idx[obs.point_idx[ro]]], device=device)]
        op = torch.as_tensor(objp[obs.point_idx[ro]], dtype=dtype, device=device)
        Xw[torch.as_tensor(ro, device=device)] = torch.einsum('mij,mj->mi', G.rodrigues(bp[:, :3]), op) + bp[:, 3:]
    Lst0 = prob.local(prob.ext, prob.intr, prob.delta)
    uvp, z = _project_local(Lst0[prob.st_o], torch.nan_to_num(Xw), prob.k10[prob.cam_o], prob.off[prob.cam_o],
                            prob.is_kb[prob.cam_o], prob.has_b, prob.has_k)
    e0 = torch.linalg.norm(uvp - prob.uv, dim=-1)
    ok = torch.isfinite(Xw).all(-1) & torch.isfinite(e0) & (z > min_depth)
    if max_px is not None:
        ok &= e0 < float(max_px)
    ok = ok.cpu().numpy()
    # points need >= 2 observations, boards >= 4
    for _ in range(2):
        cnt_p = np.bincount(obs.point_idx[ok & ~is_rigid_o], minlength=N)
        ok &= is_rigid_o | (cnt_p[obs.point_idx] >= 2)
        if B is not None:
            cnt_b = np.bincount(bmap[set_idx[obs.point_idx[ok & is_rigid_o]]], minlength=len(board_ids))
            bi_o = np.where(is_rigid_o, bmap[np.maximum(set_idx[obs.point_idx], 0)], 0)
            ok &= ~is_rigid_o | (cnt_b[bi_o] >= 4)
    report.n_obs_pruned = int((~ok).sum())
    # ---- min points per camera guard
    cnt_c = np.bincount(obs.cam_idx[ok], minlength=prob.C)
    frozen = [c for c in range(prob.C) if cnt_c[c] < int(min_points_per_camera or 0)]
    if frozen:
        msg = 'cameras {} have fewer than {} observations and are frozen'.format(
            [prob.names[c] for c in frozen], min_points_per_camera)
        warnings.warn(msg, RuntimeWarning, stacklevel=2)
        report.warnings.append(msg)
        report.frozen_cameras = [prob.names[c] for c in frozen]
    prob.build(frozen)

    # ---- kinds
    kinds = []
    okt = torch.as_tensor(ok, device=device)
    sel_p = np.nonzero(ok & ~is_rigid_o)[0]
    if len(sel_p):
        upts, inv = np.unique(obs.point_idx[sel_p], return_inverse=True)
        vel = None
        if prob.use_delta:
            vel = _velocities(obs, X_all, None)[torch.as_tensor(upts, device=device)]
        kinds.append(_Kind('point', 3, torch.as_tensor(sel_p, device=device), torch.as_tensor(inv, device=device),
                           X_all[torch.as_tensor(upts, device=device)].clone(), vel=vel))
        kinds[-1].ids = upts
    sel_r = np.nonzero(ok & is_rigid_o)[0]
    if len(sel_r):
        bo = bmap[set_idx[obs.point_idx[sel_r]]]
        ub, inv = np.unique(bo, return_inverse=True)
        kinds.append(_Kind('rigid', 6, torch.as_tensor(sel_r, device=device), torch.as_tensor(inv, device=device),
                           B[torch.as_tensor(ub, device=device)].clone(),
                           objp=torch.as_tensor(objp[obs.point_idx[sel_r]], dtype=dtype, device=device)))
        kinds[-1].ids = board_ids[ub]
    report.n_obs = int(ok.sum())
    report.n_points = int(sum(k.n for k in kinds))
    if not kinds:
        raise ValueError('no usable observations after pruning')

    # ---- scale gauge
    sc = spec.scale_constraint
    scale_mode, scale_pair = None, None
    if isinstance(sc, str) and sc == 'auto':
        scale_mode = 'keep' if (not any(k.name == 'rigid' for k in kinds) and len(prob.fixed_stations) < 2
                                and prob.ext_free.sum() >= 1) else None
    elif sc == 'keep':
        scale_mode = 'keep'
    elif sc is not None:
        scale_pair = tuple(sc)
    centers0 = prob.centers(prob.ext)
    scale_before = rig_scale_from_centers(centers0)

    # ---- errors before
    def obs_errors(ext, intr, delta, values):
        Lst = prob.local(ext, intr, delta)
        e = torch.full((len(obs),), float('nan'), dtype=dtype, device=device)
        for k, v in zip(kinds, values):
            r, z = _kind_residual(prob, k, Lst, v)
            ek = torch.linalg.norm(r, dim=-1)
            e[k.sel] = torch.where(z > 0, ek, torch.full_like(ek, float('inf')))
        return e.cpu().numpy()
    e_before = obs_errors(prob.ext, prob.intr, prob.delta, [k.values for k in kinds])

    with torch.no_grad():
        ext, intr, delta, values, hist, conv = _lm_solve(
            prob, kinds, loss=loss, f_scale=float(f_scale), n_iter=int(n_iter), lam=lam, tol=tol,
            scale_mode=scale_mode, scale_pair=scale_pair, solver=solver, verbose=verbose)
        e_after = obs_errors(ext, intr, delta, values)

    report.history = hist
    report.converged = bool(conv)
    report.n_iter = len(hist) - 1
    report.error_before = _err_stats(e_before[ok])[0]
    report.error = _err_stats(e_after[ok])[0]
    report.obs_errors = e_after
    for c, name in enumerate(prob.names):
        m = ok & (obs.cam_idx == c)
        mb, pb_ = _err_stats(e_before[m])
        ma, pa = _err_stats(e_after[m])
        report.per_camera[name] = dict(n_obs=int(m.sum()), n_obs_total=int((obs.cam_idx == c).sum()),
                                       median_before=mb, p90_before=pb_, median_after=ma, p90_after=pa,
                                       free=prob.free_names.get(name, []),
                                       frozen=name in report.frozen_cameras)
    scale_after = rig_scale_from_centers(prob.centers(ext))
    report.gauge = dict(scale_before=scale_before, scale_after=scale_after,
                        scale_ratio=scale_after / scale_before if scale_before > 0 else float('nan'),
                        fixed_stations=[prob.station_names[s] for s in prob.fixed_stations],
                        scale_constraint=scale_mode or (list(scale_pair) if scale_pair else None))
    if prob.use_delta:
        report.time_offsets = {n: float(delta[c]) for c, n in enumerate(prob.names)}
    if obs.station is not None:
        report.stations = dict(names=prob.station_names, station_cam=prob.station_cam.tolist(),
                               rvec=ext[:, :3].cpu().numpy(), tvec=ext[:, 3:].cpu().numpy())
    # points out
    p3d = np.full((N, 3), np.nan)
    for k, v in zip(kinds, values):
        if k.name == 'point':
            p3d[k.ids] = v.cpu().numpy()
        else:
            vv = v.cpu().numpy()
            for j, b in enumerate(k.ids):
                idx = np.nonzero(set_idx == b)[0]
                import cv2
                Rb = cv2.Rodrigues(vv[j, :3])[0]
                p3d[idx] = objp[idx] @ Rb.T + vv[j, 3:]
    report.p3d = p3d.reshape(tuple(obs.point_shape or (N,)) + (3,)) if obs.point_shape else p3d
    worse = revert_if_worse and not (report.error <= report.error_before)
    if worse:
        report.reverted = True
    else:
        _write_back(cgroup, prob, ext, intr)
    report.elapsed_s = _time.time() - t0
    report.cgroup = cgroup
    if verbose:
        print(report.summary())
    return report if return_report else float(report.error)


def _write_back(cgroup, prob, ext, intr):
    first = {}
    for s, c in enumerate(prob.station_cam):
        first.setdefault(int(c), s)
    with torch.no_grad():
        for c, cam in enumerate(cgroup.cameras):
            dev, dt = cam.matrix.device, cam.matrix.dtype
            K = cam.matrix.detach().clone()
            iv = intr[c].to(device=dev, dtype=dt)
            K[0, 0], K[1, 1], K[0, 2], K[1, 2], K[0, 1] = iv[0], iv[1], iv[2], iv[3], iv[4]
            cam.matrix.data.copy_(K)
            n = cam.dist.shape[0]
            cam.dist.data.copy_(iv[5:5 + min(n, 8)] if n <= 8 else torch.cat([iv[5:13], cam.dist.data[8:]]))
            if c in first:
                e = ext[first[c]].to(device=dev, dtype=dt)
                cam.rvec.data.copy_(e[:3].reshape(cam.rvec.shape))
                cam.tvec.data.copy_(e[3:].reshape(cam.tvec.shape))
            if hasattr(cam, '_refresh_frozen'):
                try:
                    cam._refresh_frozen()
                except Exception:
                    pass


# ====================================================================== structured rejection

@dataclass
class RejectSchedule:
    """Per-round thresholds (px) for structured rejection in :func:`bundle_adjust_robust`
    (motor observatory ``robust_bundle_adjust``). Round ``r`` uses entry ``min(r, len - 1)``;
    None disables that mask.

    ``frame``: drop every (camera, frame) block whose median error exceeds the threshold
    (decoys: a camera locked on something else for whole frames); ``keypoint``: drop
    (camera, keypoint) pairs a camera systematically gets wrong (occluded body parts);
    ``obs``: drop single observations above the threshold."""
    frame: Optional[tuple] = (40, 20, 12, 10)
    keypoint: Optional[tuple] = (40, 25, 15, 12)
    obs: Optional[tuple] = (60, 30, 20, 15)
    min_frame_pts: int = 8
    min_kpt_obs: int = 20
    protect_kpts: int = 8
    max_kpt_frac: float = 0.4

    @staticmethod
    def pick(seq, r):
        if seq is None:
            return None
        if np.isscalar(seq):
            return float(seq)
        return float(seq[min(r, len(seq) - 1)])

    def to_dict(self):
        return _jsonable(dataclasses.asdict(self))


def frame_mask(errors, frame_id, thresh, min_pts=8):
    """Exclusion mask ``[C,N]`` of every (camera, frame) block whose median error > ``thresh``
    (blocks with fewer than ``min_pts`` finite errors are kept). Also returns ``[C,F]`` bad blocks
    and the frame ids ``F``."""
    errors = np.asarray(errors, dtype=float)
    frame_id = np.asarray(frame_id)
    frames, inv = np.unique(frame_id, return_inverse=True)
    C, F = errors.shape[0], len(frames)
    order = np.argsort(inv, kind='stable')
    counts = np.bincount(inv, minlength=F)
    width = int(counts.max()) if F else 0
    cols = np.concatenate([np.arange(c) for c in counts]) if F else np.zeros(0, int)
    grid = np.full((C, F, max(width, 1)), np.nan)
    grid[:, inv[order], cols] = errors[:, order]
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        med = np.nanmedian(grid, axis=2)
    cnt = np.isfinite(grid).sum(2)
    bad = (med > thresh) & (cnt >= min_pts)
    return bad[:, inv], bad, frames


def keypoint_mask(errors, kpt_id, thresh, min_obs=20, protect=8, max_frac=0.4):
    """Exclusion mask ``[C,N]`` of (camera, keypoint) pairs whose median error > ``thresh``,
    capped at ``max_frac`` of the keypoints per camera and never leaving fewer than ``protect``
    keypoints (motor ``keypoint_mask``). Also returns the ``[C,K]`` mask."""
    errors = np.asarray(errors, dtype=float)
    kpt_id = np.asarray(kpt_id).astype(np.int64)
    C = errors.shape[0]
    K = int(kpt_id.max()) + 1 if len(kpt_id) else 0
    mask = np.zeros((C, K), bool)
    for i in range(C):
        med = np.full(K, np.nan)
        fin = np.isfinite(errors[i])
        for k in range(K):
            sel = (kpt_id == k) & fin
            if sel.sum() >= min_obs:
                med[k] = np.median(errors[i][sel])
        cand = np.isfinite(med) & (med > thresh)
        order = np.argsort(np.where(np.isfinite(med), med, -np.inf))
        if cand.sum() > max_frac * K:
            cand = np.zeros(K, bool)
            cand[order[-int(max_frac * K):]] = True
        if protect and (~cand).sum() < protect:
            keep = order[::-1]
            keep = [k for k in np.argsort(np.where(np.isfinite(med), med, np.inf))[:protect]]
            cand = np.ones(K, bool)
            cand[keep] = False
        mask[i] = cand
    return mask[:, kpt_id], mask


# ====================================================================== resection

@dataclass
class ResectResult:
    """Outcome of resecting one camera. ``n_inliers_before/after``: its detections within
    ``tau_pnp`` px of the points triangulated from the other cameras, before / with the new pose."""
    camera: str
    updated: bool
    n_points: int
    n_inliers_before: int
    n_inliers_after: int
    rvec: Any = None
    tvec: Any = None
    reason: str = ''

    def to_dict(self):
        return _jsonable(dataclasses.asdict(self))


def _pnp_normalized(X, xn, thresh_n, iters, rng_seed=None, guess=None):
    import cv2
    X = np.ascontiguousarray(X, dtype=np.float64)
    xn = np.ascontiguousarray(xn, dtype=np.float64)
    K = np.eye(3)
    if guess is not None:
        ok, rv, tv, inl = cv2.solvePnPRansac(X, xn, K, None, rvec=guess[0].copy(), tvec=guess[1].copy(),
                                             useExtrinsicGuess=True, reprojectionError=thresh_n,
                                             iterationsCount=iters, confidence=0.9999,
                                             flags=cv2.SOLVEPNP_ITERATIVE)
    else:
        if rng_seed is not None:
            cv2.setRNGSeed(int(rng_seed))
        ok, rv, tv, inl = cv2.solvePnPRansac(X, xn, K, None, reprojectionError=thresh_n, iterationsCount=iters,
                                             confidence=0.9999, flags=cv2.SOLVEPNP_EPNP)
    if not ok or inl is None or len(inl) < 6:
        return None
    ii = inl[:, 0]
    rv, tv = cv2.solvePnPRefineLM(X[ii], xn[ii], K, None, rv, tv)
    return rv.ravel(), tv.ravel()


def _count_inliers(cam_P, X, uv, rv, tv, tau):
    Pc = dataclasses.replace(cam_P, rvec=torch.as_tensor(rv, dtype=cam_P.dtype, device=cam_P.device)[None],
                             tvec=torch.as_tensor(tv, dtype=cam_P.dtype, device=cam_P.device)[None], ext=None)
    e = G.reproj_err(Pc, X, uv[None])[0]
    e = np.where(np.isfinite(e), e, np.inf)
    return int((e < tau).sum()), e


def _frame_pnp(X, xn, uv, frame_id, cam_P, tau, n_frames_sample=3, n_iters=200, min_frame_pts=8, rng=None):
    """Motor ``frame_pnp_ransac``: minimal samples are whole frames, scored by inlier frames."""
    rng = _as_rng(rng)
    fr, cnt = np.unique(frame_id, return_counts=True)
    fr = fr[cnt >= min_frame_pts]
    if len(fr) < 2:
        return None
    f = cam_P.K[0, 0, 0].item()
    best = None
    k = min(n_frames_sample, len(fr))
    for _ in range(n_iters):
        sel = np.isin(frame_id, rng.choice(fr, size=k, replace=False))
        if sel.sum() < 6:
            continue
        sol = _pnp_normalized(X[sel], xn[sel], tau / f, 80, rng_seed=int(rng.integers(1 << 30)))
        if sol is None:
            continue
        _, e = _count_inliers(cam_P, X, uv, sol[0], sol[1], tau)
        bad_all, bad, frames = frame_mask(e[None], frame_id, tau, min_pts=min_frame_pts)
        good_frames = frames[~bad[0]]
        score = int(np.isin(fr, good_frames).sum())
        if best is None or score > best[0]:
            best = (score, sol, good_frames)
    if best is None:
        return None
    keep = np.isin(frame_id, best[2])
    sol = _pnp_normalized(X[keep], xn[keep], tau / f, 2000, guess=best[1]) or best[1]
    return sol


def resect_cameras(cgroup, p2ds, cams=None, tau_tri=5.0, tau_pnp=8.0, min_points=50, sample='points',
                   frame_id=None, contrib=None, force=False, rng=None, iters=5000, min_gain=0.0, max_dist=5.0):
    """Leave-one-out resection: for each camera, robustly triangulate (MSAC, ``tau_tri``) from the
    *other* cameras, PnP-RANSAC this camera against those points and keep the new pose if it
    explains more of its detections (within ``tau_pnp`` px; at least ``min_gain`` relative gain).
    All cameras are resected against the input rig, then updated together. Degenerate solutions
    (inliers clustered in a small image region, or the camera more than ``max_dist`` times the
    other cameras' median distance from the points) are rejected. ``force=True`` always takes a valid
    PnP solution (placement of cameras with unknown poses). ``sample='frames'`` draws whole frames
    (``frame_id`` per point) as minimal samples, robust to decoys. Modifies ``cgroup`` in place;
    returns a list of :class:`ResectResult`."""
    _check_group(cgroup, 'resect_cameras')
    rng = _as_rng(rng)
    p = np.asarray(_np(p2ds), dtype=np.float64)
    C = p.shape[0]
    p = p.reshape(C, -1, 2)
    if frame_id is not None:
        frame_id = np.asarray(frame_id).reshape(-1)
    if sample == 'frames' and frame_id is None:
        raise ValueError("sample='frames' needs frame_id")
    names = cgroup.get_names()
    cams = range(C) if cams is None else [_cam_index(cgroup, c) for c in cams]
    valid = np.isfinite(p).all(-1)
    out = []
    P = G.pack(cgroup)       # every camera is resected against the same (input) rig
    updates = []
    for c in cams:
        pm = p.copy()
        pm[c] = np.nan
        idx = np.nonzero(valid[c] & (np.isfinite(pm).all(-1).sum(0) >= 2))[0]
        if len(idx) < min_points:
            out.append(ResectResult(names[c], False, int(len(idx)), 0, 0, reason='too few points'))
            continue
        con = None
        if contrib is not None:
            con = np.asarray(contrib, bool).copy()
            con[c] = False
        R = T.triangulate_robust(P, pm[:, idx], 'msac', tau=tau_tri, contrib=con)
        good = R.n_inliers >= 2
        if good.sum() < min_points:
            out.append(ResectResult(names[c], False, int(good.sum()), 0, 0, reason='too few consistent points'))
            continue
        X = R.p3d[good]
        uv = p[c, idx[good]]
        Pc = P.subset([c])
        n0, _ = _count_inliers(Pc, X, uv, Pc.rvec[0].cpu().numpy(), Pc.tvec[0].cpu().numpy(), tau_pnp)
        xn = G.undistort(Pc, uv[None])[0]
        okn = np.isfinite(xn).all(-1)
        f = float(Pc.K[0, 0, 0])
        if sample == 'frames':
            sol = _frame_pnp(X[okn], xn[okn], uv[okn], frame_id[idx[good]][okn], Pc, tau_pnp, rng=rng)
        else:
            sol = _pnp_normalized(X[okn], xn[okn], tau_pnp / f, iters, rng_seed=int(rng.integers(1 << 30)))
        if sol is None:
            out.append(ResectResult(names[c], False, int(len(X)), n0, 0, reason='PnP failed'))
            continue
        n1, e1 = _count_inliers(Pc, X, uv, sol[0], sol[1], tau_pnp)
        # degeneracy guards: the inliers must cover the image like the detections do (a decoy
        # explains many detections with a camera placed far away, all points on one pixel), and
        # the camera must not end up far outside the rig
        inl = e1 < tau_pnp
        spread_ok = n1 >= 6 and float(np.sqrt(uv[inl].var(0).sum())) >= 0.3 * float(np.sqrt(uv.var(0).sum()))
        Rn = G.rodrigues(torch.as_tensor(sol[0], dtype=torch.float64)).numpy()
        cnew = -Rn.T @ np.asarray(sol[1])
        cen_all = P.centers().cpu().numpy()
        others = np.delete(cen_all, c, axis=0)
        mid = np.nanmedian(X, axis=0)
        far = np.linalg.norm(cnew - mid) > max_dist * max(np.median(np.linalg.norm(others - mid, axis=1)), 1e-12)
        sane = spread_ok and not far
        upd = sane and ((n1 > n0 + min_gain * max(n0, 1)) or (force and n1 >= 6))
        if upd:
            updates.append((c, sol))
        out.append(ResectResult(names[c], bool(upd), int(len(X)), int(n0), int(n1), rvec=sol[0], tvec=sol[1],
                                reason='more inliers' if upd and n1 > n0 else ('forced' if upd else ('kept' if sane else 'degenerate PnP'))))
    with torch.no_grad():
        for c, sol in updates:
            cam = cgroup.cameras[c]
            cam.rvec.data.copy_(torch.as_tensor(sol[0], dtype=cam.rvec.dtype, device=cam.rvec.device).reshape(cam.rvec.shape))
            cam.tvec.data.copy_(torch.as_tensor(sol[1], dtype=cam.tvec.dtype, device=cam.tvec.device).reshape(cam.tvec.shape))
    return out


def place_cameras(cgroup, p2ds, cams, **kwargs):
    """Place cameras with unknown poses (``cams``) by PnP against the others' points
    (:func:`resect_cameras` with ``force=True``)."""
    kwargs.setdefault('force', True)
    return resect_cameras(cgroup, p2ds, cams=cams, **kwargs)


def add_camera(cgroup, camera, p2d_cam, p2ds, **kwargs):
    """A new group with ``camera`` appended and placed by PnP against points triangulated from
    ``cgroup`` (``p2d_cam``: the new camera's detections ``N x 2`` (or ``N x J x 2``) of the same
    points as ``p2ds``). Returns ``(new_group, ResectResult)``."""
    from copy import deepcopy
    from .cameras import CameraGroup
    cams = [deepcopy(c) for c in cgroup.cameras] + [deepcopy(camera)]
    new = CameraGroup(cams, metadata=deepcopy(getattr(cgroup, 'metadata', None)))
    p = np.asarray(_np(p2ds), dtype=np.float64)
    pc = np.asarray(_np(p2d_cam), dtype=np.float64).reshape((1,) + p.shape[1:])
    res = place_cameras(new, np.concatenate([p, pc], 0), [len(cams) - 1], **kwargs)
    return new, res[0]


# ====================================================================== robust recipe

def _robust_errors(cgroup, flat, tau, contrib=None):
    P = G.pack(cgroup)
    R = T.triangulate_robust(P, flat, 'msac', tau=tau, contrib=contrib, min_inliers=2)
    return R


def _per_camera_stats(names, errors, fit=None):
    out = {}
    for c, n in enumerate(names):
        m, p90 = _err_stats(errors[c])
        d = dict(n_obs=int(np.isfinite(errors[c]).sum()), median=m, p90=p90)
        if fit is not None:
            d['n_fit'] = int(fit[c].sum())
        out[n] = d
    return out


def bundle_adjust_robust(cgroup, p2ds, groups=None, schedule=((200, 100, 50), (40, 20, 10, 5), (20, 10, 5)),
                         reject='default', resect=True, min_inliers=3, final=('extrinsics', 'intrinsics'),
                         spec=None, contrib=None, rng=None, max_points=20000, n_iter=25, sig_frac=0.5,
                         only_extrinsics=False, report_tau=None, resect_kw=None, inplace=True,
                         revert_if_worse=False, return_report=True, verbose=False):
    """Robust bundle adjustment recipe (mouse-outdoor / Emory ``calibrate`` + motor masks).

    For each round of ``schedule`` (thresholds ``tau`` in px): MSAC triangulation with ``tau`` ->
    points with ``>= min_inliers`` inlier views -> LM on their inlier observations (Cauchy,
    ``f_scale = sig_frac * tau``, extrinsics only); then optional LOO resection of every camera
    (:func:`resect_cameras`), then structured rejection (``reject``, a :class:`RejectSchedule`;
    needs ``groups['frame']`` / ``groups['keypoint']``, filled automatically for ``C x N x J x 2``
    input). ``final`` stages re-run the last round's thresholds with extrinsics, then with the
    intrinsics of ``spec`` (default: each camera's masks).

    Masks restrict the *fit* only. Errors in the report (per camera, before/after) are computed on
    **all** observations, against points robustly triangulated with ``report_tau`` (default: the
    last threshold of the schedule).

    ``contrib``: bool per camera, cameras allowed to define points (others are passengers).
    Returns a :class:`BAReport` (``method='robust'``), or its ``error`` if ``return_report=False``.
    """
    t0 = _time.time()
    _check_group(cgroup, 'bundle_adjust_robust')
    if not inplace:
        from copy import deepcopy
        cgroup = deepcopy(cgroup)
    rng_ = _as_rng(rng)
    if reject == 'default':
        reject = RejectSchedule()
    p = np.asarray(_np(p2ds), dtype=np.float64)
    C = p.shape[0]
    pshape = p.shape[1:-1]
    flat = p.reshape(C, -1, 2)
    NP = flat.shape[1]
    groups = dict(groups or {})
    if len(pshape) == 2:
        n_, j_ = np.meshgrid(np.arange(pshape[0]), np.arange(pshape[1]), indexing='ij')
        groups.setdefault('frame', n_)
        groups.setdefault('keypoint', j_)
    grp = {k: np.broadcast_to(np.asarray(v), pshape).reshape(-1) for k, v in groups.items()}
    frame_id, kpt_id = grp.get('frame'), grp.get('keypoint')
    names = cgroup.get_names()
    valid = np.isfinite(flat).all(-1)
    schedule = [tuple(np.atleast_1d(s).tolist()) for s in schedule]
    report_tau = float(report_tau or schedule[-1][-1])
    base_spec = _resolve_spec(cgroup, spec)
    ext_spec = base_spec.replace(intrinsics=(), n_dist=0)
    state0 = {k: v.clone() for k, v in cgroup.state_dict().items()}
    scale0 = rig_scale_from_centers(G.pack(cgroup).centers())

    rep = BAReport(method='robust', seed=rng if isinstance(rng, (int, np.integer)) else None)
    rep.config = dict(schedule=schedule, reject=reject.to_dict() if reject else None, resect=resect,
                      min_inliers=min_inliers, final=list(final or ()), spec=base_spec.to_dict(),
                      max_points=max_points, n_iter=n_iter, sig_frac=sig_frac, report_tau=report_tau)
    R0 = _robust_errors(cgroup, flat, report_tau, contrib)
    e_before = np.where(valid, R0.errors, np.nan)
    rep.error_before = _err_stats(e_before[valid])[0]
    before = _per_camera_stats(names, e_before)
    fit = valid.copy()

    def run_stage(taus, sp, label):
        for tau in taus:
            P = G.pack(cgroup)
            pm = np.where(fit[..., None], flat, np.nan)
            R = T.triangulate_robust(P, pm, 'msac', tau=float(tau), contrib=contrib, min_inliers=2)
            n_in = R.n_inliers
            mi = min_inliers
            ok = np.nonzero(n_in >= mi)[0]
            if len(ok) < 50 and mi > 2:
                ok = np.nonzero(n_in >= 2)[0]
                mi = 2
            if len(ok) < 10:
                rep.history.append(dict(stage=label, tau=float(tau), skipped=True, n_points=int(len(ok))))
                continue
            if max_points is not None and len(ok) > max_points:
                ok = np.sort(rng_.choice(ok, size=max_points, replace=False))
            inl = R.inliers[:, ok] & fit[:, ok]
            ci, pj = np.nonzero(inl)
            obs = Observations(ci, ok[pj], flat[ci, ok[pj]], n_points=NP, n_cams=C)
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', RuntimeWarning)
                r = bundle_adjust_lm(cgroup, obs, p3d_init=R.p3d, spec=sp, loss='cauchy',
                                     f_scale=float(tau) * sig_frac, n_iter=n_iter, return_report=True,
                                     min_points_per_camera=6)
            rep.history.append(dict(stage=label, tau=float(tau), n_points=r.n_points, n_obs=r.n_obs,
                                    min_inliers=mi, error_before=r.error_before, error=r.error,
                                    n_iter=r.n_iter, frozen=r.frozen_cameras))
            rep.stages.append(r)
            if verbose:
                print('{} tau {:g}: {} points, {} obs, {:.3f} -> {:.3f} px'.format(
                    label, tau, r.n_points, r.n_obs, r.error_before, r.error))

    rep.excluded = {}
    resections = []
    for rnd, taus in enumerate(schedule):
        run_stage(taus, ext_spec, 'round{}'.format(rnd))
        if resect:
            kw = dict(tau_tri=5.0, tau_pnp=8.0, contrib=contrib, rng=rng_)
            kw.update(resect_kw or {})
            pm = np.where(fit[..., None], flat, np.nan)
            res = resect_cameras(cgroup, pm, **kw)
            resections.append([r.to_dict() for r in res])
            if verbose:
                for r in res:
                    if r.updated:
                        print('  resected {}: inliers {} -> {}'.format(r.camera, r.n_inliers_before, r.n_inliers_after))
        if reject and rnd < len(schedule) - 1 or (reject and not final and rnd == len(schedule) - 1):
            R = _robust_errors(cgroup, flat, max(5.0, float(taus[-1])), contrib)
            err = np.where(valid, R.errors, np.nan)
            new_fit = valid.copy()
            thr = RejectSchedule.pick(reject.frame, rnd)
            if thr is not None and frame_id is not None:
                fm, fbad, _ = frame_mask(err, frame_id, thr, reject.min_frame_pts)
                new_fit &= ~fm
                rep.excluded['frame'] = fbad
            thr = RejectSchedule.pick(reject.keypoint, rnd)
            if thr is not None and kpt_id is not None:
                km, kbad = keypoint_mask(err, kpt_id, thr, reject.min_kpt_obs, reject.protect_kpts,
                                         reject.max_kpt_frac)
                new_fit &= ~km
                rep.excluded['keypoint'] = kbad
            thr = RejectSchedule.pick(reject.obs, rnd)
            if thr is not None:
                new_fit &= ~(np.nan_to_num(err, nan=np.inf) > thr)
            new_fit[:, new_fit.sum(0) < 2] = False
            fit = new_fit
    for stage in (final or ()):
        if stage == 'extrinsics':
            run_stage(schedule[-1], ext_spec, 'final_extrinsics')
        elif stage == 'intrinsics':
            if not only_extrinsics:
                run_stage(schedule[-1][-2:], base_spec, 'final_intrinsics')
        else:
            raise ValueError("final stages must be 'extrinsics' or 'intrinsics', got {!r}".format(stage))

    R1 = _robust_errors(cgroup, flat, report_tau, contrib)
    e_after = np.where(valid, R1.errors, np.nan)
    rep.error = _err_stats(e_after[valid])[0]
    after = _per_camera_stats(names, e_after, fit)
    for n in names:
        rep.per_camera[n] = dict(n_obs=after[n]['n_obs'], n_fit=after[n]['n_fit'],
                                 median_before=before[n]['median'], p90_before=before[n]['p90'],
                                 median_after=after[n]['median'], p90_after=after[n]['p90'])
    rep.excluded['fit'] = (valid & ~fit).reshape((C,) + tuple(pshape))
    rep.excluded['n_fit_excluded'] = int((valid & ~fit).sum())
    rep.n_obs = int(fit.sum())
    rep.n_points = int((fit.sum(0) >= 2).sum())
    rep.n_iter = len(rep.history)
    rep.config['resections'] = resections
    scale1 = rig_scale_from_centers(G.pack(cgroup).centers())
    rep.gauge = dict(scale_before=scale0, scale_after=scale1, scale_ratio=scale1 / scale0 if scale0 > 0 else float('nan'))
    rep.obs_errors = e_after.reshape((C,) + tuple(pshape))
    rep.p3d = np.asarray(R1.p3d).reshape(tuple(pshape) + (3,))
    if revert_if_worse and not (rep.error <= rep.error_before):
        cgroup.load_state_dict(state0)
        rep.reverted = True
        rep.error = rep.error_before
    rep.elapsed_s = _time.time() - t0
    rep.cgroup = cgroup
    if verbose:
        print(rep.summary())
    return rep if return_report else float(rep.error)
