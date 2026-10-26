"""Spatiotemporal bundle adjustment (1.0, plan 01 F4 + F7 STBA side).

Port of the disco dancing ``may05_stba.py`` solver (Vo, Narasimhan & Sheikh, CVPR 2016) onto
:class:`~aniposelib.geometry.PackedCameras`. It refines camera parameters *and* timing from
unsynchronised (or loosely synchronised) 2D tracks.

Model
-----
* Every track (``point`` id) is cut into segments of ``segment_s`` seconds; each segment is a
  uniform cubic B-spline trajectory ``X_q(t) = sum_k b_k(u) C_q[i+k]`` with knot spacing
  ``knot_spacing`` (``h``) seconds. Observations are evaluated on the trajectory at their **own**
  time, never at the nearest frame::

      t = obs.t + delta[clip] + rate[cam] * (obs.t - t_mid[clip]) + rs[cam] * (v / H - 1/2)
          [+ w_clip(obs.t)]

  ``obs.t`` is the observation's *seed* global time (the camera clock mapped with the best sync
  you have). ``delta`` is a per-clip (or per-camera) offset, ``rate`` a per-camera clock-rate
  error, ``rs`` the rolling-shutter readout time (top-to-bottom, seconds) and ``w`` an optional
  monotone piecewise-linear warp per clip (plan 01 F7).
* **Sign convention.** ``true_global_time = obs.t + delta``. If the seed time was computed with
  the :mod:`aniposelib.sync` convention ``source_time = global_time + offset`` (i.e. ``obs.t =
  source_time - offset_seed``), the refined offset is ``offset_seed - delta``
  (:meth:`STBAResult.sync_offsets`).
* A motion prior penalises spline acceleration (``accel_sigma``, units/s^2) and velocity
  (``vel_sigma``, units/s, "least kinetic energy"); ``None`` disables either.
* Gauge and anchoring: the reference clip (that of camera ``ref_camera``) has ``delta = 0``;
  other offsets get a Gaussian prior ``N(offset0, offset_prior_sigma)``, rates
  ``N(0, rate_prior_sigma)``, rolling shutter ``N(rs0, rs_prior_sigma)``. The camera
  ``fix_cameras`` keep their extrinsics, and ``gauge='baseline'`` keeps the distance between the
  first fixed and the first free camera (scale). **Nothing is applied silently**: every time
  parameter comes with normal and cluster-robust sandwich sigmas.

Solver
------
Robust (Cauchy/Huber IRLS) Levenberg-Marquardt with an exact Schur complement: the dense
per-trajectory control-point blocks (3M x 3M) are eliminated with batched Cholesky (GPU-ready,
float64), leaving a small reduced system in the free camera/time parameters. Jacobians are
analytic (projection, distortion, rotation via the SO(3) left Jacobian); the time derivative is
closed form, ``dr/dt = (d pi / d X) X'(t)``. After every parameter step the control points get a
few per-trajectory LM iterations (variable-projection flavour). Uncertainty: the reduced normal
matrix scaled by the robust chi2/dof, and a cluster-robust sandwich (clusters = ``cluster_s``
time windows), which accounts for temporally correlated keypoint noise.

Value seen in disco: a strong *calibration refiner* (found a camera bump), a weak *sync source*
(sigma 10-25 ms on 8 s windows): anchor offsets with an audio/lighting estimate where possible.

API
---
``TimedObservations(cam, point, uv, t, clip=None, weight=None)``, ``.from_dense(p2ds, times)``.
``SpatiotemporalBA(cgroup, obs, ...).fit() -> STBAResult``.
"""

import dataclasses
import math
import warnings
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import torch

from . import geometry as G
from . import triangulation as T

__all__ = ['TimedObservations', 'SpatiotemporalBA', 'STBAResult', 'Trajectories', 'CAM_PARAMS',
           'bspline_weights', 'spline_eval', 'warp_target_from_soft_dtw']

CAM_PARAMS = ('rx', 'ry', 'rz', 'tx', 'ty', 'tz', 'fx', 'fy', 'cx', 'cy', 'skew',
              'd0', 'd1', 'd2', 'd3', 'd4', 'd5', 'd6', 'd7')
NPC = len(CAM_PARAMS)
_INTR = {'fx': 6, 'fy': 7, 'cx': 8, 'cy': 9, 'skew': 10}
NT = 5   # time columns per observation: delta, rate, rs, warp a, warp b


def _np(x, dtype='float64'):
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    return np.asarray(x, dtype=dtype)


def _jsonable(x):
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.floating, np.integer, np.bool_)):
        return x.item()
    return x


# ------------------------------------------------------------------ observations

@dataclass
class TimedObservations:
    """Flat list of 2D observations with their own times.

    ``cam [N]`` camera index (group order), ``point [N]`` track id (one 3D trajectory per id),
    ``uv [N,2]`` pixels, ``t [N]`` seed global time (s), ``clip [N]`` optional clip/recording id
    (default: the camera), ``weight [N]`` optional (keypoint score)."""
    cam: Any
    point: Any
    uv: Any
    t: Any
    clip: Any = None
    weight: Any = None

    def __post_init__(self):
        self.cam = np.asarray(self.cam, dtype='int64').ravel()
        self.point = np.asarray(self.point, dtype='int64').ravel()
        self.uv = _np(self.uv).reshape(-1, 2)
        self.t = _np(self.t).ravel()
        n = len(self.cam)
        if not (len(self.point) == len(self.uv) == len(self.t) == n):
            raise ValueError('cam, point, uv and t must have the same length')
        if self.clip is not None:
            self.clip = np.asarray(self.clip, dtype='int64').ravel()
            if len(self.clip) != n:
                raise ValueError('clip must have the same length as cam')
        if self.weight is not None:
            self.weight = _np(self.weight).ravel()
            if len(self.weight) != n:
                raise ValueError('weight must have the same length as cam')

    def __len__(self):
        return len(self.cam)

    @classmethod
    def from_dense(cls, p2ds, times, clips=None, weights=None):
        """From ``p2ds [C,F,2]`` (one track) or ``[C,F,J,2]`` (J tracks), NaN = missing, with
        ``times [C,F]`` (per-camera frame times) or ``[F]``. ``clips [C,F]`` / ``[C]`` optional."""
        p = _np(p2ds)
        if p.ndim == 3:
            p = p[:, :, None]
        C, F, J, _ = p.shape
        tt = _np(times)
        tt = np.broadcast_to(tt if tt.ndim == 2 else tt[None], (C, F))
        cam = np.broadcast_to(np.arange(C)[:, None, None], (C, F, J))
        pt = np.broadcast_to(np.arange(J)[None, None], (C, F, J))
        tg = np.broadcast_to(tt[:, :, None], (C, F, J))
        ok = np.isfinite(p).all(-1)
        clip = None
        if clips is not None:
            cl = np.asarray(clips)
            cl = np.broadcast_to(cl[:, None] if cl.ndim == 1 else cl, (C, F))
            clip = np.broadcast_to(cl[:, :, None], (C, F, J))[ok]
        w = None
        if weights is not None:
            w = np.broadcast_to(_np(weights).reshape(C, F, -1), (C, F, J))[ok]
        return cls(cam=cam[ok], point=pt[ok], uv=p[ok], t=tg[ok], clip=clip, weight=w)

    @staticmethod
    def coerce(obs):
        if isinstance(obs, TimedObservations):
            return obs
        if isinstance(obs, dict):
            d = dict(obs)
        else:
            d = {k: getattr(obs, k) for k in ('cam', 'point', 'uv', 't', 'time', 'clip', 'weight')
                 if getattr(obs, k, None) is not None}
        if 't' not in d and 'time' in d:
            d['t'] = d.pop('time')
        d.pop('time', None)
        return TimedObservations(**{k: d.get(k) for k in ('cam', 'point', 'uv', 't', 'clip', 'weight')})


# ------------------------------------------------------------------ B-splines

def bspline_weights(u):
    """Uniform cubic B-spline basis: ``u [...]`` in [0,1) -> weights ``[...,4]`` and d/du."""
    u2, u3 = u * u, u * u * u
    b = torch.stack([(1 - u) ** 3 / 6, (3 * u3 - 6 * u2 + 4) / 6, (-3 * u3 + 3 * u2 + 3 * u + 1) / 6, u3 / 6], -1)
    db = torch.stack([-(1 - u) ** 2 / 2, (3 * u2 - 4 * u) / 2, (-3 * u2 + 2 * u + 1) / 2, u2 / 2], -1)
    return b, db


def _spline_index(tau, h, M):
    s = tau / h
    i = torch.clamp(torch.floor(s), 0, M - 4)
    u = torch.clamp(s - i, -0.5, 1.5)
    return i.long(), u


def spline_eval(C, tau, h, derivative=False):
    """``C [M,3]`` control points, ``tau [N]`` seconds from the trajectory start -> ``X [N,3]``
    (and ``X'`` if ``derivative``)."""
    C = torch.as_tensor(C)
    tau = torch.as_tensor(tau, dtype=C.dtype, device=C.device)
    i, u = _spline_index(tau, h, C.shape[0])
    b, db = bspline_weights(u)
    idx = i[:, None] + torch.arange(4, device=C.device)
    Cl = C[idx]
    X = (b[..., None] * Cl).sum(-2)
    if derivative:
        return X, (db[..., None] * Cl).sum(-2) / h
    return X


def _basis_matrix_np(tau, M, h):
    s = np.asarray(tau, float) / h
    i = np.clip(np.floor(s), 0, M - 4).astype(int)
    u = s - i
    b = np.stack([(1 - u) ** 3 / 6, (3 * u ** 3 - 6 * u ** 2 + 4) / 6,
                  (-3 * u ** 3 + 3 * u ** 2 + 3 * u + 1) / 6, u ** 3 / 6], -1)
    A = np.zeros((len(s), M))
    for k in range(4):
        A[np.arange(len(s)), i + k] += b[:, k]
    return A


@dataclass
class Trajectories:
    """Fitted spline trajectories: ``control [Q,M,3]``, start times ``t0 [Q]`` (global, s),
    knot spacing ``h``, track id ``point [Q]`` and segment index ``segment [Q]``."""
    control: Any
    t0: Any
    h: float
    point: Any
    segment: Any
    duration: float

    def evaluate(self, t, point=None):
        """Positions at global times ``t [N]`` (of track ``point``, scalar or ``[N]``); NaN where
        no trajectory covers the time."""
        t = np.atleast_1d(np.asarray(t, float))
        pts = np.broadcast_to(np.asarray(point if point is not None else self.point[0]), t.shape)
        out = np.full(t.shape + (3,), np.nan)
        C = np.asarray(self.control)
        M = C.shape[1]
        for q in range(len(self.t0)):
            m = (pts == self.point[q]) & (t >= self.t0[q]) & (t <= self.t0[q] + (M - 3) * self.h)
            if m.any():
                out[m] = spline_eval(torch.as_tensor(C[q]), torch.as_tensor(t[m] - self.t0[q]), self.h).numpy()
        return out

    def to_dict(self):
        return _jsonable(dataclasses.asdict(self))


def warp_target_from_soft_dtw(est, t0_ref=0.0, t0_clip=0.0, sigma_floor=None):
    """Convert a :class:`aniposelib.sync.WarpEstimate` between the reference clip's signal (a) and
    another clip's signal (b) into STBA warp targets ``(t_seed, correction, sigma)``.

    The signals start at seed times ``t0_ref`` / ``t0_clip``. An event at reference time
    ``t0_ref + t_a`` appears in the clip at seed time ``t0_clip + t_a + lag``, so the clip's
    correction (``true = seed + correction``) there is ``t0_ref - t0_clip - lag``. ``sigma`` is
    the soft-alignment dispersion (floored at ``sigma_floor``, default one sample spacing)."""
    t_a = np.asarray(est.t_a, float)
    lag = np.asarray(est.lag, float)
    disp = np.asarray(est.dispersion, float)
    if sigma_floor is None:
        sigma_floor = float(np.median(np.diff(t_a))) if len(t_a) > 1 else 1e-3
    t_seed = t0_clip + t_a + lag
    corr = t0_ref - t0_clip - lag
    return t_seed, corr, np.maximum(disp, sigma_floor)


# ------------------------------------------------------------------ result

@dataclass
class STBAResult:
    """Output of :meth:`SpatiotemporalBA.fit` (``true_time = obs.t + delta``)."""
    clip_names: list
    delta: Any                  # [n_clips] s
    delta_sigma: Any            # normal-matrix sigma (NaN where fixed)
    delta_sigma_sandwich: Any   # cluster-robust sigma (NaN where fixed or < 2 clusters)
    delta_free: Any
    rate: Any                   # [C] clock-rate error (dimensionless)
    rate_sigma: Any
    rs: Any                     # [C] rolling-shutter readout (s)
    rs_sigma: Any
    warp: Optional[dict]        # clip -> dict(knots, values, sigma) (monotone piecewise-linear)
    group: Any                  # refined CameraGroup (a copy)
    trajectories: Trajectories
    history: list
    cost: float
    residuals: Any              # [N_used] pixel error per used observation
    obs_index: Any              # [N_used] index into the input observations
    sigma_px: float             # robust per-axis noise estimate: median residual / 1.1774
    theta_names: list
    theta: Any                  # full parameter vector (see theta_names)
    free: Any                   # bool mask of free parameters
    covariance: dict            # 'normal', 'sandwich' (free parameters only), 'free_names', 's2'
    info: dict = field(default_factory=dict)

    def sync_offsets(self, seed_offsets=None):
        """Refined offsets in the :mod:`aniposelib.sync` convention: ``seed_offset - delta``."""
        s = np.zeros(len(self.delta)) if seed_offsets is None else np.asarray(seed_offsets, float)
        return s - np.asarray(self.delta)

    def to_dict(self, include_group=True):
        d = {}
        for f in dataclasses.fields(self):
            v = getattr(self, f.name)
            if f.name == 'group':
                if include_group and v is not None:
                    d['group'] = v.to_dict() if hasattr(v, 'to_dict') else None
                continue
            if f.name == 'trajectories':
                d[f.name] = v.to_dict()
                continue
            d[f.name] = _jsonable(v)
        return d


# ------------------------------------------------------------------ analytic projection

def _skew3(v):
    o = torch.zeros_like(v[..., 0])
    return torch.stack([torch.stack([o, -v[..., 2], v[..., 1]], -1),
                        torch.stack([v[..., 2], o, -v[..., 0]], -1),
                        torch.stack([-v[..., 1], v[..., 0], o], -1)], -2)


def _left_jacobian(r):
    th2 = (r * r).sum()
    K = _skew3(r)
    eye = torch.eye(3, dtype=r.dtype, device=r.device)
    if float(th2) < 1e-10:
        return eye + K / 2 + K @ K / 6
    th = torch.sqrt(th2)
    return eye + (1 - torch.cos(th)) / th2 * K + (th - torch.sin(th)) / (th2 * th) * (K @ K)


def _project_cam(p, X, model, k10, off, jac=False):
    """One camera, parameter vector ``p [19]`` (see ``CAM_PARAMS``), points ``X [n,3]`` ->
    ``uv [n,2]`` and, with ``jac``, ``JX [n,2,3]`` and ``Jp [n,2,19]`` (analytic)."""
    r, t = p[0:3], p[3:6]
    R = G.rodrigues(r)
    Xr = X @ R.T
    pc = Xr + t
    z = pc[:, 2]
    eps = G.FISHEYE_EPS
    zs = torch.where(torch.abs(z) >= eps, z, torch.full_like(z, eps))
    x, y = pc[:, 0] / zs, pc[:, 1] / zs
    d = p[11:19][None]
    fn = G._kb if model == 1 else G._brown
    out = fn(x[None], y[None], d, jac=jac)
    xd, yd = out[0][0], out[1][0]
    fx, fy, cx, cy, sk = p[6], p[7], p[8], p[9], p[10]
    u = fx * xd + sk * yd + cx - off[0]
    v = k10 * xd + fy * yd + cy - off[1]
    uv = torch.stack([u, v], -1)
    if not jac:
        return uv, z
    jxx, jxy, jyx, jyy = [a[0] for a in out[2]]
    m00 = fx * jxx + sk * jyx; m01 = fx * jxy + sk * jyy
    m10 = k10 * jxx + fy * jyx; m11 = k10 * jxy + fy * jyy
    iz = 1 / zs
    g0 = torch.stack([m00 * iz, m01 * iz, -(m00 * x + m01 * y) * iz], -1)
    g1 = torch.stack([m10 * iz, m11 * iz, -(m10 * x + m11 * y) * iz], -1)
    g = torch.stack([g0, g1], -2)                                   # [n,2,3] d uv / d pc
    JX = g @ R
    Jr = -(g @ _skew3(Xr)) @ _left_jacobian(r)
    n = X.shape[0]
    o, one = torch.zeros_like(x), torch.ones_like(x)
    Jk = torch.stack([torch.stack([xd, o], -1), torch.stack([o, yd], -1), torch.stack([one, o], -1),
                      torch.stack([o, one], -1), torch.stack([yd, o], -1)], -1)   # [n,2,5]
    # distortion: d(xd, yd) / d dist [n,2,8]
    dd = p[11:19]
    if model == 1:
        r2 = x * x + y * y
        r_ok = r2 >= eps * eps
        rr = torch.sqrt(torch.where(r_ok, r2, torch.ones_like(r2)))
        th = torch.atan(rr)
        cols = []
        for i in range(4):
            dthd = th ** (2 * i + 3)
            fac = torch.where(r_ok, dthd / rr, torch.zeros_like(rr))
            cols.append(torch.stack([x * fac, y * fac], -1))
        cols += [torch.zeros(n, 2, dtype=X.dtype, device=X.device)] * 4
        Jd = torch.stack(cols, -1)
    else:
        k1, k2, p1, p2, k3, k4, k5, k6 = [dd[i] for i in range(8)]
        r2 = x * x + y * y
        num = 1 + r2 * (k1 + r2 * (k2 + r2 * k3))
        den = 1 + r2 * (k4 + r2 * (k5 + r2 * k6))
        def rad_col(dr):
            return torch.stack([x * dr, y * dr], -1)
        r4, r6 = r2 * r2, r2 * r2 * r2
        Jd = torch.stack([rad_col(r2 / den), rad_col(r4 / den),
                          torch.stack([2 * x * y, r2 + 2 * y * y], -1),
                          torch.stack([r2 + 2 * x * x, 2 * x * y], -1),
                          rad_col(r6 / den), rad_col(-num * r2 / den ** 2), rad_col(-num * r4 / den ** 2),
                          rad_col(-num * r6 / den ** 2)], -1)
    A = torch.stack([torch.stack([fx, sk]), torch.stack([k10 + 0 * fx, fy])])  # [2,2]
    Jdist = torch.einsum('ij,njk->nik', A, Jd)
    Jp = torch.cat([Jr, g, Jk, Jdist], -1)
    return uv, z, JX, Jp


def _theta_from_packed(P):
    K = P.K
    return torch.cat([P.rvec, P.tvec, K[:, 0, 0:1], K[:, 1, 1:2], K[:, 0, 2:3], K[:, 1, 2:3], K[:, 0, 1:2],
                      P.dist[:, :8]], 1)


def _packed_from_theta(P0, th):
    """PackedCameras with the parameters of ``th [C,19]`` (other fields from ``P0``)."""
    K = P0.K.clone()
    K[:, 0, 0] = th[:, 6]; K[:, 1, 1] = th[:, 7]; K[:, 0, 2] = th[:, 8]; K[:, 1, 2] = th[:, 9]
    K[:, 0, 1] = th[:, 10]
    return dataclasses.replace(P0, K=K, dist=th[:, 11:19].clone(), rvec=th[:, 0:3].clone(),
                               tvec=th[:, 3:6].clone(), ext=None)


# ------------------------------------------------------------------ solver

class SpatiotemporalBA:
    """Spatiotemporal bundle adjustment of a camera group on timed 2D tracks (see module docstring).

    Args:
        cgroup: initial ``CameraGroup`` (not modified; the result holds a refined copy).
        obs: :class:`TimedObservations` (or a dict / object with ``cam, point, uv, t[, clip, weight]``).
        knot_spacing: spline knot spacing ``h`` (s). ``segment_s``: trajectory segment length.
        accel_sigma, vel_sigma: motion prior scales (units/s^2, units/s; None = off).
        free_offsets: ``'per_clip'`` (delta per ``obs.clip``), ``'per_camera'`` or None.
        offset0, offset_prior_sigma: prior mean (dict clip -> s, or array) and sigma of the offsets.
        clip_names: names of the clip ids (default ``clip<i>`` / camera names for per-camera).
        free_rate, rate_prior_sigma: per-camera clock-rate error (reference camera fixed).
        rolling_shutter: False, True (free, start at 0), or initial readout times ``[C]`` (s);
            ``rs_prior_sigma`` anchors them.
        extrinsics: free camera poses (except ``fix_cameras``); ``gauge='baseline'`` fixes the scale.
        spec: :class:`aniposelib.params.ParamSpec` of free intrinsics (``intrinsics`` names and
            ``n_dist``); None = intrinsics fixed. ``intrinsics_prior_sigma`` (px, for K entries)
            and ``dist_prior_sigma`` anchor them.
        robust: ``'cauchy'``, ``'huber'`` or None; ``sigma_px`` noise scale, ``robust_c`` kernel
            scale in sigmas.
        warp_knot_s: enable a monotone piecewise-linear warp per free clip with this knot
            spacing (s); ``warp_prior_sigma`` (toward the init), ``warp_smooth_sigma`` (second
            differences); ``warp_clips`` restricts the warps to these clips (default: all free
            clips). ``warp_init``: dict clip -> ``(t_seed, correction[, sigma])`` or a
            ``sync.WarpEstimate`` (reference signal a, clip signal b, both starting at seed time 0).
            ``warp_signals``: dict clip -> ``(t, values)`` view-invariant signals (incl. the
            reference clip); when given, ``sync.soft_dtw_align`` initialises the warps and, if
            ``warp_dtw_sigma_scale`` is not None, its expected path (dispersion-weighted) is kept
            as a soft-DTW regulariser. Unidentifiable alignments are reported in ``info`` and not used.
        cluster_s: window length for the sandwich clusters. ``device``: torch device.
    """

    def __init__(self, cgroup, obs, knot_spacing=1 / 30, segment_s=2.0, pad=None,
                 accel_sigma=40.0, vel_sigma=None, free_offsets='per_clip', offset0=None,
                 offset_prior_sigma=0.1, clip_names=None, ref_camera=0,
                 free_rate=False, rate_prior_sigma=1e-3,
                 rolling_shutter=False, rs_prior_sigma=0.05,
                 extrinsics=True, fix_cameras=(0,), gauge='baseline', baseline_sigma=1e-4,
                 spec=None, intrinsics_prior_sigma=None, dist_prior_sigma=None,
                 robust='cauchy', sigma_px=2.0, robust_c=3.0,
                 warp_knot_s=None, warp_prior_sigma=0.05, warp_smooth_sigma=0.005, warp_init=None,
                 warp_signals=None, warp_dtw_sigma_scale=1.0, warp_dtw_kwargs=None, warp_clips=None,
                 cluster_s=8.0, min_obs=8, min_cams=2, mu_c=1e-6, lm_lambda0=1e-3,
                 max_iter=30, tol_rel=1e-8, batch_traj=256, device=None, dtype=torch.float64,
                 verbose=False):
        if robust not in ('cauchy', 'huber', None, 'linear'):
            raise ValueError("robust must be 'cauchy', 'huber' or None")
        if free_offsets not in ('per_clip', 'per_camera', None):
            raise ValueError("free_offsets must be 'per_clip', 'per_camera' or None")
        self.cgroup = cgroup
        self.P0 = G.pack(cgroup, device=device, dtype=dtype)
        self.dev, self.dt = self.P0.device, dtype
        self.nc = nc = self.P0.n_cams
        self.h = float(knot_spacing)
        self.segment_s = float(segment_s)
        self.pad = 2 * self.h if pad is None else float(pad)
        self.robust, self.sigma_px, self.robust_c = robust, float(sigma_px), float(robust_c)
        self.mu_c, self.lm_lambda0, self.max_iter, self.tol_rel = mu_c, lm_lambda0, max_iter, tol_rel
        self.batch_traj, self.verbose = int(batch_traj), verbose
        self.cluster_s = float(cluster_s)
        self.info = {}
        self.history = []
        self.models = [int(m) for m in self.P0.model.tolist()]
        self.k10 = self.P0.K[:, 1, 0]
        self.off = self.P0.offset
        self._build_obs(TimedObservations.coerce(obs), free_offsets, clip_names, ref_camera, min_obs, min_cams)
        self._build_theta(free_offsets, offset0, offset_prior_sigma, free_rate, rate_prior_sigma,
                          rolling_shutter, rs_prior_sigma, extrinsics, fix_cameras, gauge, baseline_sigma,
                          spec, intrinsics_prior_sigma, dist_prior_sigma)
        self._warp_clips = None if warp_clips is None else {self._clip_key(k) for k in warp_clips}
        self._build_warp(warp_knot_s, warp_prior_sigma, warp_smooth_sigma, warp_init, warp_signals,
                         warp_dtw_sigma_scale, warp_dtw_kwargs or {})
        self._finish_theta()
        self._build_prior_C(accel_sigma, vel_sigma)
        self.C = None
        self.mu_q = torch.full((self.Q,), 1e-3, dtype=self.dt, device=self.dev)

    # ---------------------------------------------------------------- problem construction
    def _T(self, a, dtype=None):
        return torch.as_tensor(np.ascontiguousarray(a), dtype=dtype or self.dt, device=self.dev)

    def _build_obs(self, obs, free_offsets, clip_names, ref_camera, min_obs, min_cams):
        nc = self.nc
        ok = np.isfinite(obs.uv).all(1) & np.isfinite(obs.t) & (obs.cam >= 0) & (obs.cam < nc)
        if obs.weight is not None:
            ok &= np.isfinite(obs.weight) & (obs.weight > 0)
        idx = np.flatnonzero(ok)
        cam, pt, uv, t = obs.cam[idx], obs.point[idx], obs.uv[idx], obs.t[idx]
        w = np.ones(len(idx)) if obs.weight is None else obs.weight[idx]
        if free_offsets == 'per_camera' or obs.clip is None:
            clip = cam.copy()
            n_clips = nc
            names = list(self.P0.names) if self.P0.names else ['cam{}'.format(i) for i in range(nc)]
        else:
            uc, clip = np.unique(obs.clip[idx], return_inverse=True)
            n_clips = len(uc)
            names = ['clip{}'.format(int(c)) for c in uc]
            self.info['clip_ids'] = uc.tolist()
        if clip_names is not None and free_offsets != 'per_camera' and obs.clip is not None:
            names = [clip_names[int(c)] if not isinstance(clip_names, dict) else clip_names[int(c)]
                     for c in self.info['clip_ids']]
        elif clip_names is not None and len(clip_names) == n_clips:
            names = list(clip_names)
        self.clip_names, self.n_clips = names, n_clips
        self.tmin = float(t.min())
        seg = np.floor((t - self.tmin) / self.segment_s).astype(int)
        key = np.stack([pt, seg], 1)
        uk, qi = np.unique(key, axis=0, return_inverse=True)
        qi = qi.ravel()
        # drop thin trajectories
        nobs = np.bincount(qi, minlength=len(uk))
        ncam = np.zeros(len(uk), int)
        cq = np.unique(np.stack([qi, cam], 1), axis=0)
        np.add.at(ncam, cq[:, 0], 1)
        keepq = (nobs >= min_obs) & (ncam >= min_cams)
        newq = -np.ones(len(uk), int); newq[keepq] = np.arange(keepq.sum())
        m = keepq[qi]
        idx, cam, pt, uv, t, w, clip, seg, qi = [a[m] for a in (idx, cam, pt, uv, t, w, clip, seg, qi)]
        qi = newq[qi]
        uk = uk[keepq]
        if len(uk) == 0:
            raise ValueError('no trajectory has >= {} observations from >= {} cameras'.format(min_obs, min_cams))
        o = np.lexsort((t, qi))
        idx, cam, pt, uv, t, w, clip, qi = [a[o] for a in (idx, cam, pt, uv, t, w, clip, qi)]
        self.Q = len(uk)
        self.traj_point, self.traj_seg = uk[:, 0], uk[:, 1]
        self.traj_t0 = self.tmin + self.traj_seg * self.segment_s - self.pad
        self.M = int(math.ceil((self.segment_s + 2 * self.pad) / self.h - 1e-9)) + 3
        self.traj_window = np.floor((self.traj_seg * self.segment_s) / self.cluster_s).astype(int)
        self.obs_index, self.obs_cam, self.obs_clip, self.obs_t = idx, cam, clip, t
        self.obs_uv_np, self.obs_q = uv, qi
        self.q_start = np.searchsorted(qi, np.arange(self.Q + 1))
        tmid = np.zeros(n_clips)
        for k in range(n_clips):
            mm = clip == k
            tmid[k] = 0.5 * (t[mm].min() + t[mm].max()) if mm.any() else 0.0
        self.clip_tmid = tmid
        # ref clip: the clip of the reference camera with most observations
        rc = clip[cam == ref_camera]
        self.ref_clip = int(np.bincount(rc).argmax()) if len(rc) else 0
        self.ref_camera = ref_camera
        H = _np(self.P0.size[:, 1])
        vn = np.zeros(len(t))
        for c in range(self.nc):
            mm = cam == c
            if mm.any() and np.isfinite(H[c]):
                vn[mm] = uv[mm, 1] / H[c] - 0.5
        self.obs_vn = vn
        self.cam_t = self._T(cam, torch.long)
        self.q_t = self._T(qi, torch.long)
        self.tau = self._T(t - self.traj_t0[qi])
        self.uv = self._T(uv)
        self.w = self._T(w)
        self.info.update(n_obs_input=len(obs), n_obs_used=len(idx), n_trajectories=self.Q, M=self.M)

    def _build_theta(self, free_offsets, offset0, offset_prior_sigma, free_rate, rate_prior_sigma,
                     rolling_shutter, rs_prior_sigma, extrinsics, fix_cameras, gauge, baseline_sigma,
                     spec, intrinsics_prior_sigma, dist_prior_sigma):
        nc, ncl = self.nc, self.n_clips
        self.i_delta = nc * NPC
        self.i_rate = self.i_delta + ncl
        self.i_rs = self.i_rate + nc
        self.i_warp = self.i_rs + nc
        th_cam = _np(_theta_from_packed(self.P0))
        names = ['{}.{}'.format(self._cam_name(c), p) for c in range(nc) for p in CAM_PARAMS]
        names += ['delta[{}]'.format(n) for n in self.clip_names]
        names += ['rate[{}]'.format(self._cam_name(c)) for c in range(nc)]
        names += ['rs[{}]'.format(self._cam_name(c)) for c in range(nc)]
        n = self.i_warp
        theta = np.zeros(n)
        theta[:nc * NPC] = th_cam.ravel()
        free = np.zeros(n, bool)
        mu = theta.copy()
        sig = np.full(n, np.inf)
        if free_offsets is not None:
            d0 = np.zeros(ncl)
            if offset0 is not None:
                if isinstance(offset0, dict):
                    for k, v in offset0.items():
                        j = self.clip_names.index(k) if isinstance(k, str) else int(k)
                        d0[j] = v
                else:
                    d0[:] = np.asarray(offset0, float)
            theta[self.i_delta:self.i_rate] = d0
            mu[self.i_delta:self.i_rate] = d0
            for k in range(ncl):
                if k != self.ref_clip:
                    free[self.i_delta + k] = True
                    sig[self.i_delta + k] = offset_prior_sigma if offset_prior_sigma else np.inf
        if free_rate:
            for c in range(nc):
                if c != self.ref_camera:
                    free[self.i_rate + c] = True
                    sig[self.i_rate + c] = rate_prior_sigma if rate_prior_sigma else np.inf
        if rolling_shutter is not False and rolling_shutter is not None:
            if not np.all(np.isfinite(_np(self.P0.size[:, 1]))):
                raise ValueError('rolling_shutter needs image sizes on all cameras')
            rs0 = np.zeros(nc) if rolling_shutter is True else np.broadcast_to(np.asarray(rolling_shutter, float), (nc,))
            theta[self.i_rs:self.i_warp] = rs0
            mu[self.i_rs:self.i_warp] = rs0
            free[self.i_rs:self.i_warp] = True
            sig[self.i_rs:self.i_warp] = rs_prior_sigma if rs_prior_sigma else np.inf
        fixed = set(int(c) for c in (fix_cameras or ()))
        self.extra = []
        if extrinsics:
            for c in range(nc):
                if c not in fixed:
                    free[c * NPC:c * NPC + 6] = True
            if gauge == 'baseline' and fixed:
                ca = min(fixed)
                fr = [c for c in range(nc) if c not in fixed]
                if fr:
                    cb = fr[0]
                    Cc = _np(self.P0.centers())
                    L = float(np.linalg.norm(Cc[ca] - Cc[cb]))
                    self.extra.append(self._baseline_residual(ca, cb, L, baseline_sigma * L))
                    self.info['gauge'] = dict(kind='baseline', cams=(ca, cb), length=L)
            elif gauge not in (None, 'baseline'):
                raise ValueError("gauge must be 'baseline' or None")
        if spec is not None:
            for c in range(nc):
                for nm in spec.intrinsics:
                    j = c * NPC + _INTR[nm]
                    free[j] = True
                    if intrinsics_prior_sigma:
                        sig[j] = intrinsics_prior_sigma
                nd = spec.n_dist if spec.n_dist is not None else int(self.P0.dist_len[c] if self.P0.dist_len else 0)
                nd = min(nd, 4 if self.models[c] == 1 else 8)
                for k in range(nd):
                    free[c * NPC + 11 + k] = True
                    if dist_prior_sigma:
                        sig[c * NPC + 11 + k] = dist_prior_sigma
        self.theta, self.free, self.prior_mu, self.prior_sigma, self.theta_names = theta, free, mu, sig, names
        self.lin_A = []   # linear theta-only residuals (rows over the full theta, pre-divided by sigma)
        self.lin_b = []

    def _cam_name(self, c):
        return self.P0.names[c] if self.P0.names else 'cam{}'.format(c)

    def _baseline_residual(self, ca, cb, L, sigma):
        def res(th):
            pa, pb = th[ca * NPC:ca * NPC + 6], th[cb * NPC:cb * NPC + 6]
            Ra, Rb = G.rodrigues(pa[:3]), G.rodrigues(pb[:3])
            Ca = -(Ra.T @ pa[3:6]); Cb = -(Rb.T @ pb[3:6])
            return ((Ca - Cb).norm() - L)[None] / sigma
        return res

    def _build_warp(self, knot_s, prior_sigma, smooth_sigma, warp_init, warp_signals, dtw_scale, dtw_kwargs):
        self.warp_knots = {}
        self.warp_slice = {}
        N = len(self.obs_t)
        tcols = np.full((N, NT), -1, int)
        tlev = np.zeros((N, NT))
        tcols[:, 0] = self.i_delta + self.obs_clip
        tlev[:, 0] = 1.0
        tcols[:, 1] = self.i_rate + self.obs_cam
        tlev[:, 1] = self.obs_t - self.clip_tmid[self.obs_clip]
        tcols[:, 2] = self.i_rs + self.obs_cam
        tlev[:, 2] = self.obs_vn
        self.tcols, self.tlev = tcols, tlev
        if knot_s is None:
            if warp_init or warp_signals:
                raise ValueError('warp_init / warp_signals need warp_knot_s')
            return
        targets = {}
        if warp_signals is not None:
            targets = self._dtw_targets(warp_signals, dtw_kwargs)
        if warp_init:
            for k, v in warp_init.items():
                j = self.clip_names.index(k) if isinstance(k, str) else int(k)
                if hasattr(v, 'lag'):
                    if not v.identifiable:
                        warnings.warn('warp_init for clip {}: soft-DTW alignment not identifiable ({}); '
                                      'not used'.format(self.clip_names[j], v.reasons))
                        self.info.setdefault('warp_rejected', {})[self.clip_names[j]] = list(v.reasons)
                        continue
                    v = warp_target_from_soft_dtw(v)
                targets[j] = v
        names = list(self.theta_names)
        theta, free, mu, sig = list(self.theta), list(self.free), list(self.prior_mu), list(self.prior_sigma)
        A_rows = []
        for k in range(self.n_clips):
            if k == self.ref_clip or not self.free[self.i_delta + k]:
                continue
            if self._warp_clips is not None and k not in self._warp_clips:
                continue
            m = self.obs_clip == k
            if not m.any():
                continue
            t0, t1 = self.obs_t[m].min(), self.obs_t[m].max()
            nk = max(2, int(math.ceil((t1 - t0) / knot_s)) + 1)
            knots = t0 + np.arange(nk) * knot_s
            i0 = len(theta)
            init = np.zeros(nk)
            tgt = targets.get(k)
            if tgt is not None:
                tt, corr = np.asarray(tgt[0], float), np.asarray(tgt[1], float)
                o = np.argsort(tt)
                init = np.interp(knots, tt[o], corr[o]) - self.theta[self.i_delta + k]
            theta += init.tolist(); mu += init.tolist(); free += [True] * nk
            sig += [prior_sigma if prior_sigma else np.inf] * nk
            names += ['warp[{}][{}]'.format(self.clip_names[k], i) for i in range(nk)]
            self.warp_knots[k] = knots
            self.warp_slice[k] = slice(i0, i0 + nk)
            # observation levers (hat functions)
            s = np.clip((self.obs_t[m] - t0) / knot_s, 0, nk - 1 - 1e-12)
            a = np.floor(s).astype(int)
            fr = s - a
            tcols[m, 3] = i0 + a; tlev[m, 3] = 1 - fr
            tcols[m, 4] = i0 + a + 1; tlev[m, 4] = fr
            if smooth_sigma and nk >= 3:
                D2 = np.diff(np.eye(nk), 2, axis=0) / smooth_sigma
                A_rows.append((i0, nk, D2, np.zeros(nk - 2)))
            if tgt is not None and dtw_scale is not None and len(tgt) >= 3:
                tt, corr, sg = [np.asarray(x, float) for x in tgt[:3]]
                inr = (tt >= t0) & (tt <= t1)
                tt, corr, sg = tt[inr], corr[inr], sg[inr] * dtw_scale
                if len(tt):
                    ss = np.clip((tt - t0) / knot_s, 0, nk - 1 - 1e-12)
                    aa = np.floor(ss).astype(int); ff = ss - aa
                    Bm = np.zeros((len(tt), nk))
                    Bm[np.arange(len(tt)), aa] = 1 - ff
                    Bm[np.arange(len(tt)), aa + 1] += ff
                    # residual (delta + w(t) - corr) / sigma; delta handled as an extra column
                    A_rows.append((i0, nk, Bm / sg[:, None], corr / sg, self.i_delta + k, 1 / sg))
                    self.info.setdefault('warp_dtw', {})[self.clip_names[k]] = int(len(tt))
        self.theta, self.free = np.array(theta), np.array(free, bool)
        self.prior_mu, self.prior_sigma, self.theta_names = np.array(mu), np.array(sig), names
        self._warp_rows = A_rows
        self.tcols, self.tlev = tcols, tlev

    def _dtw_targets(self, signals, kw):
        from . import sync
        if self.ref_clip not in [self._clip_key(k) for k in signals]:
            raise ValueError('warp_signals must include the reference clip {!r}'.format(self.clip_names[self.ref_clip]))
        sig = {self._clip_key(k): v for k, v in signals.items()}
        ta, va = [np.asarray(x, float) for x in sig[self.ref_clip]]
        rate = 1.0 / float(np.median(np.diff(ta)))
        out = {}
        for k, (tb, vb) in sig.items():
            if k == self.ref_clip:
                continue
            tb, vb = np.asarray(tb, float), np.asarray(vb, float)
            grid = tb[0] + np.arange(int((tb[-1] - tb[0]) * rate) + 1) / rate
            vbr = np.interp(grid, tb, vb) if vb.ndim == 1 else np.stack([np.interp(grid, tb, vb[:, f]) for f in range(vb.shape[1])], 1)
            est = sync.soft_dtw_align(va, vbr, rate=rate, **kw)
            self.info.setdefault('warp_soft_dtw', {})[self.clip_names[k]] = dict(
                offset=est.offset, sigma=est.sigma, identifiable=est.identifiable, reasons=list(est.reasons))
            if not est.identifiable:
                warnings.warn('soft-DTW alignment of clip {} is not identifiable ({}); warp not initialised '
                              'from it'.format(self.clip_names[k], est.reasons))
                continue
            out[k] = warp_target_from_soft_dtw(est, t0_ref=ta[0], t0_clip=grid[0])
        return out

    def _clip_key(self, k):
        return self.clip_names.index(k) if isinstance(k, str) else int(k)

    def _finish_theta(self):
        n = len(self.theta)
        self.n_theta = n
        rows, bs = [], []
        for r in getattr(self, '_warp_rows', []):
            i0, nk, A, b = r[:4]
            Af = np.zeros((A.shape[0], n)); Af[:, i0:i0 + nk] = A
            if len(r) > 4:
                Af[:, r[4]] = r[5]
            rows.append(Af); bs.append(b)
        self.lin_A = self._T(np.concatenate(rows)) if rows else None
        self.lin_b = self._T(np.concatenate(bs)) if rows else None
        self.free_idx = np.flatnonzero(self.free)
        self.g2f = -np.ones(n + 1, int)
        self.g2f[self.free_idx] = np.arange(len(self.free_idx))
        self.tcols[self.tcols < 0] = n          # dummy slot (theta is padded with a zero)
        cols = np.concatenate([self.obs_cam[:, None] * NPC + np.arange(NPC), self.tcols], 1)
        self.gcols = self._T(cols, torch.long)
        self.tcols_t = self._T(self.tcols, torch.long)
        self.tlev_t = self._T(self.tlev)
        self.g2f_t = self._T(self.g2f, torch.long)

    def _build_prior_C(self, accel_sigma, vel_sigma):
        M, h = self.M, self.h
        Kp = np.zeros((M, M))
        if accel_sigma:
            D2 = np.diff(np.eye(M), 2, axis=0) / h ** 2
            Kp += D2.T @ D2 * h / accel_sigma ** 2
        if vel_sigma:
            D1 = np.diff(np.eye(M), 1, axis=0) / h
            Kp += D1.T @ D1 * h / vel_sigma ** 2
        self.Kprior = self._T(np.kron(Kp, np.eye(3)))

    # ---------------------------------------------------------------- model
    def _theta_t(self):
        return self._T(self.theta)

    def _obs_model(self, theta, a, b, C, q0, jac):
        the = torch.cat([theta, theta.new_zeros(1)])
        sl = slice(a, b)
        cam = self.cam_t[sl]
        dtime = (the[self.tcols_t[sl]] * self.tlev_t[sl]).sum(-1)
        tau = self.tau[sl] + dtime
        i, u = _spline_index(tau, self.h, self.M)
        bw, db = bspline_weights(u)
        qq = self.q_t[sl] - q0
        idx = i[:, None] + torch.arange(4, device=self.dev)
        Cl = C[qq[:, None], idx]
        X = (bw[..., None] * Cl).sum(-2)
        n = X.shape[0]
        pc = theta[:self.nc * NPC].reshape(self.nc, NPC)
        uv = torch.empty(n, 2, dtype=self.dt, device=self.dev)
        z = torch.empty(n, dtype=self.dt, device=self.dev)
        if jac:
            JX = torch.empty(n, 2, 3, dtype=self.dt, device=self.dev)
            Jp = torch.empty(n, 2, NPC, dtype=self.dt, device=self.dev)
        cam_np = self.obs_cam[a:b]
        for c in np.unique(cam_np):
            m = torch.as_tensor(np.flatnonzero(cam_np == c), device=self.dev)
            out = _project_cam(pc[c], X[m], self.models[c], self.k10[c], self.off[c], jac=jac)
            uv[m] = out[0]; z[m] = out[1]
            if jac:
                JX[m] = out[2]; Jp[m] = out[3]
        uv = torch.where((z > 0)[:, None], uv, torch.full_like(uv, float('nan')))
        if not jac:
            return uv, None
        Xdot = (db[..., None] * Cl).sum(-2) / self.h
        Jt = (JX @ Xdot[..., None])[..., 0]
        Jth = torch.cat([Jp, Jt[..., None] * self.tlev_t[sl][:, None, :]], -1)
        return uv, dict(JX=JX, Jth=Jth, b=bw, idx=idx, qq=qq)

    def _robust_w(self, r, w):
        s = (r * r).sum(-1) / self.sigma_px ** 2
        if self.robust == 'cauchy':
            c2 = self.robust_c ** 2
            return w * c2 * torch.log1p(s / c2), w / (1 + s / c2)
        if self.robust == 'huber':
            c = self.robust_c
            e = torch.sqrt(s)
            cost = torch.where(e <= c, s, 2 * c * e - c * c)
            return w * cost, w * torch.where(e <= c, torch.ones_like(e), c / e.clamp(min=1e-12))
        return w * s, w.clone()

    def _theta_cost(self, theta):
        cost = theta.new_zeros(())
        fin = np.isfinite(self.prior_sigma) & self.free
        if fin.any():
            ii = self._T(np.flatnonzero(fin), torch.long)
            cost = cost + (((theta[ii] - self._T(self.prior_mu[fin])) / self._T(self.prior_sigma[fin])) ** 2).sum()
        if self.lin_A is not None:
            cost = cost + ((self.lin_A @ theta - self.lin_b) ** 2).sum()
        for fx in self.extra:
            cost = cost + (fx(theta) ** 2).sum()
        return cost

    def _batches(self):
        q0 = 0
        while q0 < self.Q:
            q1 = min(self.Q, q0 + self.batch_traj)
            yield q0, q1, int(self.q_start[q0]), int(self.q_start[q1])
            q0 = q1

    def _prior_cost_C(self, C):
        Cf = C.reshape(C.shape[0], -1)
        return torch.einsum('qi,ij,qj->q', Cf, self.Kprior, Cf)

    def cost(self, theta=None, C=None, per_traj=False):
        theta = self._theta_t() if theta is None else theta
        C = self.C if C is None else C
        tot = torch.zeros(self.Q, dtype=self.dt, device=self.dev)
        for q0, q1, a, b in self._batches():
            if b > a:
                uvh, _ = self._obs_model(theta, a, b, C[q0:q1], q0, False)
                cst, _ = self._robust_w(uvh - self.uv[a:b], self.w[a:b])
                cst = torch.nan_to_num(cst, nan=1e6)
                tot[q0:q1] += torch.zeros(q1 - q0, dtype=self.dt, device=self.dev).index_add_(0, self.q_t[a:b] - q0, cst)
            tot[q0:q1] += self._prior_cost_C(C[q0:q1])
        if per_traj:
            return tot
        return tot.sum() + self._theta_cost(theta)

    # ---------------------------------------------------------------- normal equations
    def _batch_system(self, theta, q0, q1, a, b, fit_theta, mu_c, per_traj_grad=False):
        M = self.M; nq = q1 - q0; n3 = 3 * M
        C = self.C[q0:q1]
        H = self.Kprior.expand(nq, n3, n3).clone()
        g = torch.einsum('ij,qj->qi', self.Kprior, C.reshape(nq, -1)).contiguous()
        out = dict(H=H, g=g)
        if b <= a:
            self._damp(H, q0, q1, mu_c)
            return out
        uvh, J = self._obs_model(theta, a, b, C, q0, True)
        r = uvh - self.uv[a:b]
        bad = ~torch.isfinite(r).all(-1) | ~torch.isfinite(J['Jth']).all(-1).all(-1) | ~torch.isfinite(J['JX']).all(-1).all(-1)
        r = torch.where(bad[:, None], torch.zeros_like(r), r)
        _, om = self._robust_w(r, self.w[a:b])
        om = torch.where(bad, torch.zeros_like(om), om) / self.sigma_px ** 2
        JX = torch.nan_to_num(J['JX']); bw = J['b']; idx = J['idx']; qq = J['qq']
        Aobs = torch.einsum('nki,n,nkj->nij', JX, om, JX)
        gx = torch.einsum('nki,n,nk->ni', JX, om, r)
        N = r.shape[0]
        ar3 = torch.arange(3, device=self.dev)
        rows = idx[:, :, None] * 3 + ar3
        base = qq * (n3 * n3)
        val = (bw[:, :, None, None, None] * bw[:, None, :, None, None]) * Aobs[:, None, None]
        flat = base[:, None, None, None, None] + rows[:, :, None, :, None] * n3 + rows[:, None, :, None, :]
        H.view(-1).index_add_(0, flat.reshape(-1), val.reshape(-1))
        gv = bw[:, :, None] * gx[:, None, :]
        g.view(-1).index_add_(0, (qq[:, None, None] * n3 + rows).reshape(-1), gv.reshape(-1))
        if fit_theta:
            Jth = torch.nan_to_num(J['Jth'])
            fcols = self.g2f_t[self.gcols[a:b]]
            used = torch.unique(fcols[fcols >= 0])
            nl = int(used.numel())
            out['used'] = used
            if nl:
                loc = -torch.ones(len(self.free_idx) + 1, dtype=torch.long, device=self.dev)
                loc[used] = torch.arange(nl, device=self.dev)
                lcols = torch.where(fcols >= 0, loc[fcols.clamp(min=0)], torch.full_like(fcols, nl))
                Jl = torch.zeros(N, 2, nl + 1, dtype=self.dt, device=self.dev)
                Jl.scatter_add_(2, lcols[:, None, :].expand(N, 2, -1), Jth)
                Jl = Jl[..., :nl]
                out['Htt'] = torch.einsum('nki,n,nkj->ij', Jl, om, Jl)
                gobs = torch.einsum('nki,n,nk->ni', Jl, om, r)
                out['gt'] = gobs.sum(0)
                HXt = torch.einsum('nki,n,nkj->nij', JX, om, Jl)
                HCt = torch.zeros(nq * n3, nl, dtype=self.dt, device=self.dev)
                v = bw[:, :, None, None] * HXt[:, None]
                HCt.index_add_(0, (qq[:, None, None] * n3 + rows).reshape(-1), v.reshape(-1, nl))
                out['HCt'] = HCt.reshape(nq, n3, nl)
                if per_traj_grad:
                    out['gq'] = torch.zeros(nq, nl, dtype=self.dt, device=self.dev).index_add_(0, qq, gobs)
        self._damp(out['H'], q0, q1, mu_c)
        return out

    def _damp(self, H, q0, q1, mu_c):
        dg = H.diagonal(dim1=1, dim2=2)
        dg.mul_(1 + self.mu_q[q0:q1, None]).add_(mu_c)

    def _chol(self, H):
        L, info = torch.linalg.cholesky_ex(H)
        if bool((info > 0).any()):
            bad = info > 0
            H2 = H.clone()
            eye = torch.eye(H.shape[-1], dtype=self.dt, device=self.dev)
            H2[bad] = H2[bad] + 1e-3 * eye * H[bad].diagonal(dim1=-2, dim2=-1).mean(-1)[:, None, None]
            L, info = torch.linalg.cholesky_ex(H2)
        return L

    def solve_C(self, iters=8):
        """Levenberg-Marquardt on the control points only (per-trajectory damping/acceptance)."""
        theta = self._theta_t()
        c_old = self.cost(theta, self.C, per_traj=True)
        prev = float(c_old.sum() + self._theta_cost(theta))
        for it in range(iters):
            Cnew = self.C.clone()
            for q0, q1, a, b in self._batches():
                sysd = self._batch_system(theta, q0, q1, a, b, False, self.mu_c)
                L = self._chol(sysd['H'])
                dx = torch.cholesky_solve(-sysd['g'][..., None], L)[..., 0]
                Cnew[q0:q1] = self.C[q0:q1] + dx.reshape(q1 - q0, self.M, 3)
            c_new = self.cost(theta, Cnew, per_traj=True)
            acc = (c_new <= c_old) & torch.isfinite(c_new)
            self.C = torch.where(acc[:, None, None], Cnew, self.C)
            c_old = torch.where(acc, c_new, c_old)
            self.mu_q = torch.where(acc, (self.mu_q / 3).clamp(min=1e-7), (self.mu_q * 5).clamp(max=1e6))
            cur = float(c_old.sum() + self._theta_cost(theta))
            if abs(prev - cur) < 1e-9 * max(1.0, abs(prev)):
                prev = cur
                break
            prev = cur
        return prev

    def reduced_system(self, theta, mu_c, clusters=None):
        """Schur-reduced system over the free parameters: ``S, b`` (+ caches, cluster scores)."""
        nf = len(self.free_idx)
        S = torch.zeros(nf, nf, dtype=self.dt, device=self.dev)
        bvec = torch.zeros(nf, dtype=self.dt, device=self.dev)
        caches, scores = [], {}
        for q0, q1, a, b in self._batches():
            sysd = self._batch_system(theta, q0, q1, a, b, True, mu_c, per_traj_grad=clusters is not None)
            L = self._chol(sysd['H'])
            y = torch.cholesky_solve(sysd['g'][..., None], L)[..., 0]
            cache = dict(q0=q0, q1=q1, y=y, used=None)
            if 'HCt' in sysd:
                HCt, used = sysd['HCt'], sysd['used']
                Y = torch.cholesky_solve(HCt, L)
                S[used[:, None], used[None, :]] += sysd['Htt'] - torch.einsum('qin,qim->nm', HCt, Y)
                bvec[used] += sysd['gt'] - torch.einsum('qin,qi->n', HCt, y)
                cache.update(used=used, Y=Y)
                if clusters is not None:
                    gq = sysd['gq'] - torch.einsum('qin,qi->qn', HCt, y)
                    ck = clusters[q0:q1]
                    for kk in np.unique(ck):
                        m = torch.as_tensor(ck == kk, device=self.dev)
                        vfull = torch.zeros(nf, dtype=self.dt, device=self.dev)
                        vfull[used] = gq[m].sum(0)
                        scores[kk] = scores.get(kk, 0) + vfull
            caches.append(cache)
        fi = self._T(self.free_idx, torch.long)
        fin = np.isfinite(self.prior_sigma) & self.free
        if fin.any():
            j = self._T(self.g2f[np.flatnonzero(fin)], torch.long)
            sg = self._T(self.prior_sigma[fin])
            S[j, j] += 1 / sg ** 2
            bvec[j] += (theta[self._T(np.flatnonzero(fin), torch.long)] - self._T(self.prior_mu[fin])) / sg ** 2
        if self.lin_A is not None:
            Af = self.lin_A[:, fi]
            S += Af.T @ Af
            bvec += Af.T @ (self.lin_A @ theta - self.lin_b)
        for fx in self.extra:
            Jx = torch.autograd.functional.jacobian(fx, theta)
            rx = fx(theta)
            Jf = Jx[:, fi]
            S += Jf.T @ Jf
            bvec += Jf.T @ rx
        return S, bvec, caches, scores

    def _warp_monotone(self, theta):
        th = _np(theta)
        for k, sl in self.warp_slice.items():
            knots = self.warp_knots[k]
            dk = np.diff(knots)
            slope = np.diff(th[sl]) / dk
            rate = th[self.i_rate:self.i_rs].max() if self.free[self.i_rate:self.i_rs].any() else 0.0
            if np.any(1 + slope - abs(rate) <= 0.05):
                return False
        return True

    def solve(self, iters=None, c_iters=3, c_refine=2, mu_joint=1e-4, step_tol=2e-7):
        """Joint LM on (parameters, control points) via the Schur complement."""
        iters = self.max_iter if iters is None else iters
        lam = self.lm_lambda0
        theta = self._theta_t()
        cur = float(self.cost(theta))
        fi = self._T(self.free_idx, torch.long)
        n_small = 0
        if len(self.free_idx) == 0:
            return cur
        for it in range(iters):
            mu_save = self.mu_q.clone()
            self.mu_q = torch.full_like(self.mu_q, mu_joint)
            S, bvec, caches, _ = self.reduced_system(theta, self.mu_c)
            self.mu_q = mu_save
            accepted = False
            C_old = self.C
            rel = 0.0
            dth = torch.zeros(len(self.free_idx), dtype=self.dt, device=self.dev)
            for _try in range(8):
                Sd = S + lam * torch.diag(torch.diagonal(S).clamp(min=1e-12))
                try:
                    dth = -torch.linalg.solve(Sd, bvec)
                except RuntimeError:
                    lam *= 10
                    continue
                th_new = theta.clone(); th_new[fi] += dth
                if self.warp_slice and not self._warp_monotone(th_new):
                    lam *= 5
                    continue
                C_new = C_old.clone()
                for cch in caches:
                    q0, q1 = cch['q0'], cch['q1']
                    rhs = cch['y']
                    if cch['used'] is not None and cch['used'].numel():
                        rhs = rhs + (cch['Y'] @ dth[cch['used']][:, None])[..., 0]
                    C_new[q0:q1] = C_old[q0:q1] - rhs.reshape(q1 - q0, self.M, 3)
                c_a = self.cost(th_new, C_new, per_traj=True)
                c_b = self.cost(th_new, C_old, per_traj=True)
                self.C = torch.where((c_a <= c_b)[:, None, None], C_new, C_old)
                self.theta = _np(th_new).copy()
                new = self.solve_C(c_refine) if c_refine else float(self.cost(th_new))
                if np.isfinite(new) and new < cur:
                    rel = (cur - new) / max(cur, 1.0)
                    theta = th_new
                    cur = new
                    lam = max(lam / 3, 1e-9)
                    accepted = True
                    break
                self.C = C_old
                self.theta = _np(theta).copy()
                lam *= 5
            self.history.append(dict(it=it, cost=cur, lam=lam, accepted=accepted,
                                     step_max=float(dth.abs().max()) if accepted else 0.0))
            if self.verbose:
                print('[STBA] it {} cost {:.8g} lam {:.2g} acc {}'.format(it, cur, lam, accepted), flush=True)
            small = accepted and float(dth.abs().max()) < step_tol
            n_small = n_small + 1 if small else 0
            if not accepted or rel < self.tol_rel or n_small >= 3:
                break
            if c_iters:
                cur = self.solve_C(c_iters)
        return cur

    def covariance(self, clusters=None):
        """Covariance of the free parameters: ``normal = s2 S^-1`` and the cluster-robust sandwich."""
        theta = self._theta_t()
        mu_save = self.mu_q.clone()
        self.mu_q = torch.full_like(self.mu_q, 1e-9)
        S, _, _, scores = self.reduced_system(theta, self.mu_c, clusters=clusters)
        self.mu_q = mu_save
        S = 0.5 * (S + S.T)
        dsq = torch.sqrt(torch.diagonal(S).clamp(min=1e-30))
        Sn = S / dsq[:, None] / dsq[None, :]
        ev, V = torch.linalg.eigh(Sn)
        inv_ev = torch.where(ev > 1e-12 * ev.max(), 1.0 / ev, torch.zeros_like(ev))
        Sinv = (V * inv_ev) @ V.T / dsq[:, None] / dsq[None, :]
        n_obs = 2 * len(self.obs_t)
        dof = max(n_obs - len(self.free_idx), 1)
        s2 = float(self.cost(theta)) / dof
        out = dict(normal=_np(Sinv * s2), s2=s2, free_names=[self.theta_names[i] for i in self.free_idx])
        if scores and len(scores) >= 2:
            Gm = torch.stack(list(scores.values()))
            out['sandwich'] = _np(Sinv @ (Gm.T @ Gm) @ Sinv)
            out['n_clusters'] = len(scores)
        return out

    def residuals(self, theta=None, C=None):
        theta = self._theta_t() if theta is None else theta
        C = self.C if C is None else C
        res = []
        for q0, q1, a, b in self._batches():
            if b > a:
                uvh, _ = self._obs_model(theta, a, b, C[q0:q1], q0, False)
                res.append((uvh - self.uv[a:b]).norm(dim=-1))
        return _np(torch.cat(res))

    # ---------------------------------------------------------------- initialisation
    def init_control_points(self, max_gap=0.1, lam=1e-2):
        """Triangulate each trajectory on a half-knot grid (2D tracks interpolated in time in
        undistorted coordinates, seed timing) and least-squares fit the spline."""
        M, h, Q, C = self.M, self.h, self.Q, self.nc
        P = _packed_from_theta(self.P0, self._T(self.theta[:C * NPC].reshape(C, NPC)))
        th = self.theta
        # current timing (delta, rate, rs, warp) applied to the seed times
        the = np.r_[th, 0.0]
        tcorr = self.obs_t + (the[self.tcols] * self.tlev).sum(1)
        xn = np.full((len(self.obs_t), 2), np.nan)
        for c in range(C):
            m = self.obs_cam == c
            if m.any():
                uvc = np.full((C, int(m.sum()), 2), np.nan)
                uvc[c] = self.obs_uv_np[m]
                xn[m] = _np(G._undistort_flat(P, self._T(uvc)))[c]
        grid = np.arange(0.5 * h, (M - 3) * h + 1e-9, 0.5 * h)
        ng = len(grid)
        Gd = np.full((C, Q, ng, 2), np.nan)
        for q in range(Q):
            a, b = self.q_start[q], self.q_start[q + 1]
            tq = tcorr[a:b] - self.traj_t0[q]
            cq = self.obs_cam[a:b]
            for c in np.unique(cq):
                mm = cq == c
                tt, xx = tq[mm], xn[a:b][mm]
                o = np.argsort(tt); tt, xx = tt[o], xx[o]
                if len(tt) < 2:
                    continue
                j = np.searchsorted(tt, grid)
                jc = np.clip(j, 1, len(tt) - 1)
                t0, t1 = tt[jc - 1], tt[jc]
                valid = (j > 0) & (j < len(tt)) & (t1 - t0 <= max_gap) & (t1 > t0)
                al = np.where(valid, (grid - t0) / np.where(t1 > t0, t1 - t0, 1), 0)
                val = xx[jc - 1] + (xx[jc] - xx[jc - 1]) * al[:, None]
                Gd[c, q, valid] = val[valid]
        X = _np(T.triangulate_linear(P, self._T(Gd.reshape(C, Q * ng, 2)), undistort=False)).reshape(Q, ng, 3)
        A = _basis_matrix_np(grid, M, h)
        Dm = np.diff(np.eye(M), 2, axis=0)
        C0 = np.zeros((Q, M, 3))
        ok = np.zeros(Q, bool)
        for q in range(Q):
            good = np.isfinite(X[q]).all(1)
            if good.sum() >= 4:
                med = np.median(X[q][good], 0)
                dv = np.linalg.norm(X[q][good] - med, axis=1)
                g2 = np.flatnonzero(good)[dv <= 4 * np.median(dv) + 1e-12]
                Hm = A[g2].T @ A[g2] + lam * Dm.T @ Dm + 1e-9 * np.eye(M)
                C0[q] = np.linalg.solve(Hm, A[g2].T @ X[q][g2])
                ok[q] = True
            elif good.any():
                C0[q] = np.median(X[q][good], 0)
            else:
                C0[q] = np.nan
        # trajectories without any triangulation: the centroid of the others
        bad = ~np.isfinite(C0).all((1, 2))
        if bad.all():
            raise ValueError('could not initialise any trajectory (no time-overlapping views)')
        C0[bad] = np.nanmedian(C0[~bad].reshape(-1, 3), 0)
        self.info['init_ok_fraction'] = float(ok.mean())
        return C0, ok

    # ---------------------------------------------------------------- driver
    def fit(self, C0=None, c_iters_init=5, iters=None):
        """Initialise the trajectories (unless ``C0 [Q,M,3]`` is given), solve, and return an
        :class:`STBAResult`."""
        if C0 is None:
            C0, _ = self.init_control_points()
        self.C = self._T(C0)
        self.solve_C(c_iters_init)
        cost = self.solve(iters)
        cov = self.covariance(clusters=self.traj_window)
        return self._result(cost, cov)

    def _sig(self, cov, key, i):
        if key not in cov or not self.free[i]:
            return float('nan')
        j = self.g2f[i]
        return float(math.sqrt(max(cov[key][j, j], 0.0)))

    def _result(self, cost, cov):
        th = self.theta
        nc, ncl = self.nc, self.n_clips
        Pn = _packed_from_theta(self.P0, self._T(th[:nc * NPC].reshape(nc, NPC)))
        group = deepcopy(self.cgroup)
        G.unpack_into(group, Pn)
        sig = lambda key, idx: np.array([self._sig(cov, key, i) for i in idx])
        di = np.arange(self.i_delta, self.i_rate)
        ri = np.arange(self.i_rate, self.i_rs)
        si = np.arange(self.i_rs, self.i_warp)
        warp = None
        if self.warp_slice:
            warp = {}
            for k, sl in self.warp_slice.items():
                ii = np.arange(sl.start, sl.stop)
                warp[self.clip_names[k]] = dict(knots=self.warp_knots[k], values=th[sl].copy(),
                                                sigma=sig('normal', ii), monotone=bool(self._warp_monotone(self._T(th))))
        res = self.residuals()
        fin = np.isfinite(res)
        med = float(np.median(res[fin])) if fin.any() else float('nan')
        traj = Trajectories(control=_np(self.C), t0=self.traj_t0.copy(), h=self.h, point=self.traj_point.copy(),
                            segment=self.traj_seg.copy(), duration=(self.M - 3) * self.h)
        return STBAResult(
            clip_names=list(self.clip_names), delta=th[di].copy(), delta_sigma=sig('normal', di),
            delta_sigma_sandwich=sig('sandwich', di), delta_free=self.free[di].copy(),
            rate=th[ri].copy(), rate_sigma=sig('normal', ri), rs=th[si].copy(), rs_sigma=sig('normal', si),
            warp=warp, group=group, trajectories=traj, history=list(self.history), cost=float(cost),
            residuals=res, obs_index=self.obs_index.copy(), sigma_px=med / 1.1774,
            theta_names=list(self.theta_names), theta=th.copy(), free=self.free.copy(), covariance=cov,
            info=dict(self.info, ref_clip=self.clip_names[self.ref_clip], n_iter=len(self.history)))
