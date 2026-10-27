"""Camera stations: time-varying extrinsics of physically moved cameras (1.0, plan 01 F5).

A *station* is one extrinsic pose of a physical camera over a time interval; all stations of a
camera share its intrinsics. A bumped or re-aimed camera gets a new station from the bump on, and
the interval in which it was moving can be marked ``'skip'``.

``Stations({cam_name: [(t0, t1, name), ...]})``: half-open ``[t0, t1)`` intervals in global time
(s), normalised per camera to cover ``(-inf, inf)`` without gaps (a gap is given to the
preceding interval, a leading gap to the following one; equal neighbours merge). Cameras that
are not listed have a single station (index 0). ``lookup(cam, t)`` returns the station index
(``-1`` for ``'skip'``) into ``names(cam)``; ``lookup_name`` the names. ``paint``/``split`` edit
intervals. ``to_toml_dict()`` / ``from_toml_dict()`` give the optional ``[stations]`` table::

    [stations.cam_4]
    t0 = [ -inf, 2706.0, 2707.5,]
    t1 = [ 2706.0, 2707.5, inf,]
    name = [ "cam_4", "skip", "cam_4@s1",]

``CameraGroup(stations=...)`` keeps one; its calibration file stores it as
``[metadata.stations.<cam>]`` (so aniposelib <= 0.9 readers ignore it), and
``cgroup.bundle_adjust_lm(obs)`` looks it up with ``obs.time``.

``detect_moves(cgroup, data, times=None, window_s=4, min_step_deg=0.5) -> MoveReport``: per
camera, points are triangulated **without** it (robust MSAC on the other cameras), projected
into it, and a rotation offset (rotation about the camera centre, camera frame) is fitted per
time window by robust Gauss-Newton. Weighted binary segmentation of the offset series finds the
steps (disco ``pose_ba_stations.detect_steps``); each step time is refined to the frame. The
report holds the series, the change points, proposed :class:`Stations` and each station's
rotation, plus :meth:`MoveReport.corrected_extrinsics`. Nothing is applied to the group
Windows with less than ``min_support_frac`` of the median window support are ignored, and with
``retest`` cameras flagged in the first pass are re-tested without the stronger moved cameras.
"""

import dataclasses
import math
import warnings
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from . import geometry as G
from . import triangulation as T

__all__ = ['SKIP', 'Stations', 'MoveReport', 'Recovery', 'detect_moves', 'recover_moves', 'detect_steps',
           'fit_rotation_offsets']

SKIP = 'skip'
INF = float('inf')


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


# ------------------------------------------------------------------ intervals

def _normalize(rows, default):
    rows = sorted([[float(a), float(b), str(n)] for a, b, n in rows if float(b) > float(a)],
                  key=lambda r: (r[0], r[1]))
    if not rows:
        return [[-INF, INF, default]]
    out = []
    for a, b, n in rows:
        if out and a < out[-1][1]:          # overlap: the later interval wins from its start
            out[-1][1] = a
            if out[-1][1] <= out[-1][0]:
                out.pop()
        if out and a > out[-1][1]:          # gap: give it to the preceding interval
            out[-1][1] = a
        out.append([a, b, n])
    out[0][0] = -INF
    out[-1][1] = INF
    merged = []
    for r in out:
        if merged and merged[-1][2] == r[2] and merged[-1][1] == r[0]:
            merged[-1][1] = r[1]
        else:
            merged.append(list(r))
    return merged


def _paint_rows(rows, a, b, name):
    """Paint ``[a, b) = name`` over normalised rows."""
    out = []
    for r0, r1, n in rows:
        if r1 <= a or r0 >= b:
            out.append([r0, r1, n])
            continue
        if r0 < a:
            out.append([r0, a, n])
        if r1 > b:
            out.append([b, r1, n])
    out.append([a, b, name])
    return sorted(out, key=lambda r: r[0])


class Stations:
    """Per-camera station intervals (see module docstring)."""

    def __init__(self, intervals=None, cameras=None, poses=None):
        self.cameras = list(cameras) if cameras is not None else None
        self.poses = dict(poses or {})
        self._iv = {}
        for cam, rows in (intervals or {}).items():
            name = self._cam_name(cam)
            self._iv[name] = _normalize([tuple(r) for r in rows], name)

    def _cam_name(self, cam):
        if isinstance(cam, (int, np.integer)):
            if self.cameras is None:
                raise ValueError('integer camera index needs Stations(cameras=[names])')
            return self.cameras[int(cam)]
        return str(cam)

    # -- queries
    def __contains__(self, cam):
        return self._cam_name(cam) in self._iv

    def __eq__(self, other):
        return isinstance(other, Stations) and self.to_dict() == other.to_dict()

    def __repr__(self):
        return 'Stations({})'.format(self.to_dict())

    @property
    def moved(self):
        """Cameras with more than one interval."""
        return [c for c, r in self._iv.items() if len(r) > 1]

    def intervals(self, cam):
        name = self._cam_name(cam)
        return [tuple(r) for r in self._iv.get(name, [[-INF, INF, name]])]

    def names(self, cam):
        """Station names of a camera (first appearance order, ``'skip'`` excluded)."""
        out = []
        for _, _, n in self.intervals(cam):
            if n != SKIP and n not in out:
                out.append(n)
        return out

    def lookup_name(self, cam, t):
        rows = self.intervals(cam)
        t = np.atleast_1d(np.asarray(t, float))
        starts = np.array([r[0] for r in rows])
        k = np.clip(np.searchsorted(starts, t, side='right') - 1, 0, len(rows) - 1)
        return np.array([r[2] for r in rows], dtype=object)[k]

    def lookup(self, cam, t):
        """Station index of each time ``t`` (``-1`` = skip, NaN times -> -1)."""
        names = self.names(cam)
        t = np.atleast_1d(np.asarray(t, float))
        nm = self.lookup_name(cam, t)
        idx = {n: i for i, n in enumerate(names)}
        out = np.array([idx.get(n, -1) for n in nm], dtype=int)
        out[~np.isfinite(t)] = -1
        return out

    # -- edits (return a new object)
    def copy(self):
        s = Stations(cameras=self.cameras, poses=self.poses)
        s._iv = {k: [list(r) for r in v] for k, v in self._iv.items()}
        return s

    def paint(self, cam, t0, t1, name):
        """Assign ``[t0, t1)`` of camera ``cam`` to station ``name`` (or ``'skip'``)."""
        s = self.copy()
        cn = self._cam_name(cam)
        rows = s._iv.get(cn, [[-INF, INF, cn]])
        s._iv[cn] = _normalize(_paint_rows(rows, float(t0), float(t1), str(name)), cn)
        return s

    def split(self, cam, t, name, skip=None):
        """New station ``name`` from time ``t`` to the end of the interval containing ``t``;
        ``skip=(a, b)`` additionally marks ``[a, b)`` as moving."""
        rows = self.intervals(cam)
        starts = np.array([r[0] for r in rows])
        k = int(np.clip(np.searchsorted(starts, t, side='right') - 1, 0, len(rows) - 1))
        s = self.paint(cam, t, rows[k][1], name)
        if skip is not None:
            s = s.paint(cam, skip[0], skip[1], SKIP)
        return s

    # -- io
    def to_dict(self):
        return {c: [list(r) for r in rows] for c, rows in self._iv.items()}

    def to_toml_dict(self):
        """``{cam: {'t0': [...], 't1': [...], 'name': [...]}}`` for a ``[stations]`` TOML table."""
        return {c: {'t0': [float(r[0]) for r in rows], 't1': [float(r[1]) for r in rows],
                    'name': [str(r[2]) for r in rows]} for c, rows in self._iv.items()}

    @classmethod
    def from_toml_dict(cls, d, cameras=None):
        """Inverse of :meth:`to_toml_dict`; also accepts ``{cam: [[t0, t1, name], ...]}``."""
        iv = {}
        for cam, v in (d or {}).items():
            if isinstance(v, dict):
                iv[cam] = list(zip(v['t0'], v['t1'], v['name']))
            else:
                iv[cam] = [tuple(r) for r in v]
        return cls(iv, cameras=cameras)


# ------------------------------------------------------------------ change points

def _sse(cs, cs2, cw, i, j):
    w = cw[j] - cw[i]
    if w <= 0:
        return 0.0, np.zeros(cs.shape[1])
    s = cs[j] - cs[i]
    mu = s / w
    return float(((cs2[j] - cs2[i]) - s * mu).sum()), mu


def detect_steps(y, w=None, min_step=0.1, min_seg=3, z_thr=6.0, sharp=1.5, max_cp=10):
    """Weighted binary segmentation of a series ``y [W,D]`` (disco ``detect_steps``).

    A split of a segment is accepted when both sides have >= ``min_seg`` samples, the jump of
    the means is >= ``min_step``, significant against the robust noise scale from first
    differences (``z >= z_thr``) and ``sharp`` times larger than the within-side spread (so slow
    drifts are not cut into pieces). Returns change points sorted by index: dicts with
    ``index`` (first sample after the step, in the original indexing), ``step``, ``z``,
    ``left_mean``, ``right_mean``, ``spread``."""
    y = np.asarray(y, float)
    if y.ndim == 1:
        y = y[:, None]
    W = len(y)
    w = np.ones(W) if w is None else np.asarray(w, float)
    ok = np.isfinite(y).all(1) & (w > 0)
    idx = np.flatnonzero(ok)
    if len(idx) < 2 * min_seg:
        return []
    yy, ww = y[idx], w[idx]
    d = np.diff(yy, axis=0)
    sig = float(np.median(np.linalg.norm(d, axis=1)) / np.sqrt(2) / 1.1774) if len(d) else 0.0
    sig = max(sig, 1e-9)
    cw = np.r_[0, np.cumsum(ww)]
    cs = np.vstack([np.zeros(yy.shape[1]), np.cumsum(yy * ww[:, None], 0)])
    cs2 = np.vstack([np.zeros(yy.shape[1]), np.cumsum(yy ** 2 * ww[:, None], 0)])
    cps, stack = [], [(0, len(yy))]
    while stack and len(cps) < max_cp:
        i, j = stack.pop()
        if j - i < 2 * min_seg:
            continue
        tot, _ = _sse(cs, cs2, cw, i, j)
        best = None
        for k in range(i + min_seg, j - min_seg + 1):
            a, mu1 = _sse(cs, cs2, cw, i, k)
            b, mu2 = _sse(cs, cs2, cw, k, j)
            if best is None or tot - a - b > best[0]:
                best = (tot - a - b, k, mu1, mu2)
        _, k, mu1, mu2 = best
        step = float(np.linalg.norm(mu2 - mu1))
        z = step / (sig * np.sqrt(1 / (k - i) + 1 / (j - k)))
        spread = max(float(np.median(np.linalg.norm(yy[i:k] - mu1, axis=1))),
                     float(np.median(np.linalg.norm(yy[k:j] - mu2, axis=1))))
        if step >= min_step and z >= z_thr and step >= sharp * spread:
            cps.append(dict(index=int(idx[k]), step=step, z=float(z), left_mean=mu1.tolist(),
                            right_mean=mu2.tolist(), spread=spread))
            stack.append((i, k))
            stack.append((k, j))
    return sorted(cps, key=lambda c: c['index'])


# ------------------------------------------------------------------ rotation offsets

def _cam_params(P, c):
    from .stba import _theta_from_packed
    return _theta_from_packed(P)[c]


def _rot_project(P, c, Rw, pc0, wid, jac):
    """pc' = Rw[wid] pc0 -> pixels (identity extrinsics) and d uv / d (left increment)."""
    from .stba import _project_cam
    p = _cam_params(P, c).clone()
    p[:6] = 0
    pc = torch.einsum('nij,nj->ni', Rw[wid], pc0)
    out = _project_cam(p, pc, int(P.model[c]), P.K[c, 1, 0], P.offset[c], jac=jac)
    if not jac:
        return out[0], out[1], None
    from .stba import _skew3
    J = -(out[2] @ _skew3(pc))       # [n,2,3]
    return out[0], out[1], J


def _rotvec(R):
    """Rotation matrices [...,3,3] -> rotation vectors (numpy)."""
    R = np.asarray(R, float)
    flat = R.reshape(-1, 3, 3)
    out = np.zeros((len(flat), 3))
    for i, r in enumerate(flat):
        c = np.clip((np.trace(r) - 1) / 2, -1, 1)
        th = math.acos(c)
        if th < 1e-9:
            out[i] = 0.5 * np.array([r[2, 1] - r[1, 2], r[0, 2] - r[2, 0], r[1, 0] - r[0, 1]])
            continue
        if abs(math.pi - th) < 1e-6:
            wv, v = np.linalg.eigh((r + r.T) / 2)
            out[i] = v[:, np.argmax(wv)] * th
            continue
        out[i] = np.array([r[2, 1] - r[1, 2], r[0, 2] - r[2, 0], r[1, 0] - r[0, 1]]) / (2 * math.sin(th)) * th
    return out.reshape(R.shape[:-2] + (3,))


def fit_rotation_offsets(P, c, X, uv, group, n_groups=None, n_iter=8, huber_px=3.0, min_points=20,
                         weights=None):
    """Robust (Huber IRLS) Gauss-Newton fit of one rotation offset per group.

    ``X [n,3]`` world points, ``uv [n,2]`` their detections in camera ``c`` of ``P``, ``group [n]``
    int ids in ``[0, n_groups)``. The offset rotates about the camera centre in the camera frame
    (``x_cam' = dR (R X + t)``). Returns ``(rotvec [G,3] rad, median residual px [G], n [G])``;
    NaN for groups with fewer than ``min_points`` points."""
    dev, dt = P.device, P.dtype
    X = torch.as_tensor(np.asarray(X, float), dtype=dt, device=dev)
    uv = torch.as_tensor(np.asarray(uv, float), dtype=dt, device=dev)
    group = np.asarray(group, int)
    Gn = int(group.max()) + 1 if n_groups is None else int(n_groups)
    w0 = torch.ones(len(X), dtype=dt, device=dev) if weights is None else torch.as_tensor(weights, dtype=dt, device=dev)
    wid = torch.as_tensor(group, device=dev)
    E = P.extrinsics()[c]
    pc0 = X @ E[:3, :3].T + E[:3, 3]
    Rw = torch.eye(3, dtype=dt, device=dev).repeat(Gn, 1, 1)
    cnt = np.bincount(group, minlength=Gn)
    for _ in range(n_iter):
        uvh, z, J = _rot_project(P, c, Rw, pc0, wid, True)
        r = uvh - uv
        e = r.norm(dim=-1)
        good = torch.isfinite(e) & (z > 0) & torch.isfinite(J).all(-1).all(-1)
        wr = w0 * torch.where(e <= huber_px, torch.ones_like(e), huber_px / e.clamp(min=1e-12))
        wr = torch.where(good, wr, torch.zeros_like(wr))
        r = torch.nan_to_num(r); J = torch.nan_to_num(J)
        H = torch.zeros(Gn, 3, 3, dtype=dt, device=dev).index_add_(0, wid, torch.einsum('nai,naj,n->nij', J, J, wr))
        g = torch.zeros(Gn, 3, dtype=dt, device=dev).index_add_(0, wid, torch.einsum('nai,na,n->ni', J, r, wr))
        H = H + 1e-9 * torch.eye(3, dtype=dt, device=dev)
        step = -torch.linalg.solve(H, g[..., None])[..., 0]
        step = torch.nan_to_num(step)
        Rw = G.rodrigues(step) @ Rw
        if float(step.abs().max()) < 1e-10:
            break
    uvh, z, _ = _rot_project(P, c, Rw, pc0, wid, False)
    e = (uvh - uv).norm(dim=-1).cpu().numpy()
    rv = _rotvec(Rw.cpu().numpy())
    med = np.full(Gn, np.nan)
    for k in np.flatnonzero(cnt):
        ek = e[group == k]
        ek = ek[np.isfinite(ek)]
        med[k] = np.median(ek) if len(ek) else np.nan
    bad = cnt < min_points
    rv[bad] = np.nan
    med[bad] = np.nan
    return rv, med, cnt, e


# ------------------------------------------------------------------ detection

@dataclass
class MoveReport:
    """Output of :func:`detect_moves` (angles in degrees, rotation vectors in the camera frame)."""
    cameras: list
    window_t0: Any              # [W] window start times (s)
    window_s: float
    delta_deg: Any              # [C,W,3] per-window rotation offset (NaN: camera skipped / too few points)
    n_points: Any               # [C,W]
    resid_px: Any               # [C,W] median residual after the offset fit
    change_points: list         # dicts: camera, time, window, step_deg, z, left_deg, right_deg
    stations: Stations
    station_rotations: dict     # cam -> {station: rotvec (rad)}
    info: dict = field(default_factory=dict)
    station_poses: dict = field(default_factory=dict)   # cam -> {station: 4x4} (recover_moves: full poses)

    @property
    def moved(self):
        return sorted({cp['camera'] for cp in self.change_points}, key=self.cameras.index)

    def corrected_extrinsics(self, cgroup):
        """``{cam: {station: 4x4}}`` world-to-camera matrices of every station (``R' = dR R``,
        ``t' = dR t``, i.e. rotation about the camera centre)."""
        out = {}
        for i, cam in enumerate(cgroup.cameras):
            name = self.cameras[i]
            E = np.asarray(cam.get_extrinsics_mat().detach().cpu().numpy() if isinstance(
                cam.get_extrinsics_mat(), torch.Tensor) else cam.get_extrinsics_mat(), float)
            out[name] = {}
            for st, rv in self.station_rotations.get(name, {name: np.zeros(3)}).items():
                if st in self.station_poses.get(name, {}):
                    out[name][st] = np.asarray(self.station_poses[name][st], float)
                    continue
                dR = G.rodrigues(torch.as_tensor(np.nan_to_num(np.asarray(rv, float)))).numpy()
                Ei = E.copy()
                Ei[:3, :3] = dR @ E[:3, :3]
                Ei[:3, 3] = dR @ E[:3, 3]
                out[name][st] = Ei
        return out

    def to_dict(self):
        d = {f.name: getattr(self, f.name) for f in dataclasses.fields(self)}
        d['stations'] = self.stations.to_dict()
        d['moved'] = self.moved
        return _jsonable(d)


def _dense_from_obs(obs, time_tol):
    from .stba import TimedObservations
    o = TimedObservations.coerce(obs)
    tk = np.round(o.t / time_tol).astype(np.int64)
    ut, ti = np.unique(tk, return_inverse=True)
    up, pi = np.unique(o.point, return_inverse=True)
    C = int(o.cam.max()) + 1
    p2d = np.full((C, len(ut), len(up), 2), np.nan)
    p2d[o.cam, ti, pi] = o.uv
    return p2d, ut * time_tol


def detect_moves(cgroup, data, times=None, window_s=4.0, min_step_deg=0.5, min_windows=3, z_thr=6.0,
                 sharp=1.5, tau=10.0, min_points=20, huber_px=3.0, cameras=None, refine=True,
                 skip_s=0.0, max_cp=10, fps=None, time_tol=1e-3, min_other=2, device=None,
                 min_support_frac=0.25, retest=True, max_resid_ratio=5.0, suspect_frac=0.5):
    """Detect camera bumps / re-aims (see module docstring).

    Args:
        cgroup: calibration valid at the start (the reference poses).
        data: ``p2ds [C,N,2]`` or ``[C,N,J,2]`` (synchronised frames, NaN missing), or timed
            observations (``stba.TimedObservations`` / dict with ``cam, point, uv, t``; frames
            are formed from times equal within ``time_tol``).
        times: ``[N]`` frame times (s) for dense input (default ``arange(N) / fps``, fps=1).
        window_s: window length for the rotation-offset series; ``min_windows``: minimum
            segment length (windows) between change points.
        min_step_deg, z_thr, sharp, max_cp: change-point acceptance (:func:`detect_steps`).
        tau: MSAC threshold (px) of the leave-one-out triangulation; ``min_other`` other views.
        huber_px, min_points: rotation fit robustness and minimum points per window.
        min_support_frac: a window counts only if its point count is >= ``max(min_points,
            min_support_frac * median)``, the median over the camera's windows with >=
            ``min_points`` points; sparser windows are NaN in ``delta_deg`` (listed in
            ``info['low_support']``) so a fit on a handful of points cannot form a change point.
            ``None`` / 0: only ``min_points`` (<= 1.0 behaviour).
        max_resid_ratio: a window whose median residual after its offset fit is above
            ``max(huber_px, max_resid_ratio x the camera's median window residual)`` is not
            explained by a rotation (decoys, false detections, broken sync) and is dropped like a
            low-support window (``info['bad_fit']``); a real re-aim is explained by the offset, so
            its windows fit as well as the others. ``None``: off.
        retest: explain-away re-test. A moved camera biases the leave-one-out points of its
            neighbours, so they look rotated too (and with robust thresholds the mover itself may
            stay below ``z_thr``). Suspects are the cameras with a step of z >= ``suspect_frac x
            z_thr`` in the first pass. Repeatedly, each suspect's observations are rotated back by
            its fitted offsets and the other suspects re-tested; the suspect whose correction
            removes the most evidence (sum of step angles) from the others is confirmed
            (``info['retest'][cam]['explains']``, it may be a suspect below ``z_thr``), the
            correction is kept and the remaining flagged cameras are re-tested on the corrected
            data: they stay flagged only if they still step. If no correction removes evidence,
            the first-pass flags are kept; a re-test without support keeps the first pass.
            ``False``: one pass (1.1.0 before this change: flagged cameras in order of z, the
            strongest always kept; ``suspect_frac=None`` keeps suspects = flagged).
        cameras: indices to test (default all). refine: refine step times to the frame.
        skip_s: mark ``[t - skip_s/2, t + skip_s/2)`` around each step as ``'skip'``.

    Returns:
        :class:`MoveReport`.
    """
    P = G.pack(cgroup, device=device)
    names = list(P.names) if P.names else ['cam{}'.format(i) for i in range(P.n_cams)]
    if isinstance(data, np.ndarray) or isinstance(data, torch.Tensor):
        p2d = data.detach().cpu().numpy() if isinstance(data, torch.Tensor) else np.asarray(data, float)
        if p2d.ndim == 3:
            p2d = p2d[:, :, None]
        N = p2d.shape[1]
        t = np.arange(N) / (fps or 1.0) if times is None else np.asarray(times, float)
    else:
        p2d, t = _dense_from_obs(data, time_tol)
        if p2d.shape[0] < P.n_cams:
            p2d = np.concatenate([p2d, np.full((P.n_cams - p2d.shape[0],) + p2d.shape[1:], np.nan)], 0)
    C, N, J, _ = p2d.shape
    if C != P.n_cams:
        raise ValueError('data has {} cameras, the group {}'.format(C, P.n_cams))
    t0 = float(np.nanmin(t))
    win = np.floor((t - t0) / window_s).astype(int)
    W = int(win.max()) + 1
    cams = list(range(C)) if cameras is None else [int(c) for c in cameras]
    flat = p2d.reshape(C, N * J, 2)
    tf = np.repeat(t, J)
    wf = np.repeat(win, J)

    def test(c, exclude=(), z=z_thr, fl=None):
        flat_ = flat if fl is None else fl
        others = [o for o in range(C) if o != c and o not in exclude]
        if len(others) < max(min_other, 2):
            return None
        nv = np.isfinite(flat_[others]).all(-1).sum(0)
        sel = np.flatnonzero((nv >= min_other) & np.isfinite(flat_[c]).all(-1))
        if len(sel) < min_points:
            return None
        res = T.triangulate_robust(P.subset(others), flat_[others][:, sel], method='msac', tau=tau)
        X = np.asarray(res.p3d)
        ok = np.isfinite(X).all(1)
        sel, X = sel[ok], X[ok]
        if len(sel) < min_points:
            return None
        rv, med, cnt, _ = fit_rotation_offsets(P, c, X, flat_[c, sel], wf[sel], n_groups=W,
                                               huber_px=huber_px, min_points=min_points)
        d = np.degrees(rv)
        low = np.zeros(W, bool)
        if min_support_frac:
            valid = cnt >= min_points
            if valid.any():
                thr = max(min_points, min_support_frac * float(np.median(cnt[valid])))
                low = valid & (cnt < thr)
                d[low] = np.nan
                med = med.copy()
                med[low] = np.nan
        badfit = np.zeros(W, bool)
        if max_resid_ratio:
            fm = np.isfinite(med) & np.isfinite(d).all(1)
            if fm.sum() >= 3:
                lim = max(huber_px, max_resid_ratio * float(np.median(med[fm])))
                badfit = fm & (med > lim)
                d[badfit] = np.nan
        cps = detect_steps(d, np.sqrt(np.maximum(cnt, 0)) * np.isfinite(d).all(1),
                           min_step=min_step_deg, min_seg=min_windows, z_thr=z, sharp=sharp, max_cp=max_cp)
        return dict(delta=d, cnt=cnt, med=med, cps=cps, X=X, sel=sel, low=low, badfit=badfit, others=others)

    def zmax(r):
        return max([cp['z'] for cp in r['cps']], default=0.0) if r is not None else 0.0

    def stepmax(r):
        return max([cp['step'] for cp in r['cps']], default=0.0) if r is not None else 0.0

    def corrected(fl, m, r):
        """``fl`` with camera m's observations rotated back by its per-segment offsets."""
        cps = r['cps']
        if not cps:
            return fl
        means = [np.asarray(cps[0]['left_mean'])] + [np.asarray(cp['right_mean']) for cp in cps]
        edges = [-INF] + [t0 + cp['index'] * window_s for cp in cps] + [INF]
        fl = fl.copy()
        cam = cgroup.cameras[m]
        for k, mu in enumerate(means):
            sel = np.flatnonzero((tf >= edges[k]) & (tf < edges[k + 1]) & np.isfinite(fl[m]).all(-1))
            if not len(sel) or not np.isfinite(mu).all():
                continue
            xy = np.asarray(cam.undistort_points(np.asarray(fl[m, sel], float)), float).reshape(-1, 2)
            ray = torch.as_tensor(np.c_[xy, np.ones(len(xy))], dtype=P.dtype, device=P.device)
            dR = G.rodrigues(torch.as_tensor(np.radians(mu), dtype=P.dtype, device=P.device))
            uv, z, _ = _proj_pose(P, m, dR.T, torch.zeros(3, dtype=P.dtype, device=P.device), ray,
                                  E0=torch.eye(4, dtype=P.dtype, device=P.device))
            fl[m, sel] = np.where((z > 0).cpu().numpy()[:, None], uv.cpu().numpy(), np.nan)
        return fl

    zc = z_thr * float(suspect_frac) if (retest and suspect_frac) else z_thr
    out = {c: test(c, z=zc) for c in cams}
    z_first = {c: zmax(out[c]) for c in cams}
    first = dict(out)
    suspects = sorted([c for c in cams if z_first[c] >= zc], key=lambda c: -z_first[c])
    for c in cams:                      # change points of the first pass: z >= z_thr only
        if out[c] is not None:
            out[c] = dict(out[c], cps=[cp for cp in out[c]['cps'] if cp['z'] >= z_thr])
    retest_info = {}
    if retest:
        flagged = [c for c in cams if out[c] is not None and out[c]['cps']]
        confirmed = []
        work = flat
        ev = {c: stepmax(first[c]) for c in suspects}   # current evidence (deg of the largest step, z >= zc)
        while flagged:
            # explain away: a moved camera biases its neighbours' leave-one-out points, so they look
            # rotated too. The mover is the suspect whose correction (its observations rotated back
            # by its fitted offsets) removes most of the other suspects' evidence.
            best = None
            for m in [c for c in suspects if c not in confirmed]:
                rm = test(m, fl=work, z=zc)
                if rm is None or not rm['cps']:
                    continue
                wm = corrected(work, m, rm)
                after = {f: stepmax(test(f, fl=wm, z=zc)) for f in suspects if f != m and f not in confirmed}
                gain = sum(ev[f] - after[f] for f in after)
                if best is None or gain > best[0]:
                    best = (gain, m, rm, wm, after)
            if best is None or (best[0] <= 0 and best[1] not in flagged):
                for f in flagged:       # nothing explains it: keep the first pass
                    confirmed.append(f)
                    retest_info[names[f]] = dict(result='kept', z_first=z_first[f], explains=[])
                break
            gain, m, rm, wm, after = best
            rm = dict(rm, cps=[cp for cp in rm['cps'] if cp['z'] >= zc])
            out[m] = rm
            confirmed.append(m)
            cleared = [names[f] for f in flagged if f != m and after.get(f, 0.0) == 0.0]
            retest_info[names[m]] = dict(result='kept', z_first=z_first[m], z_retest=zmax(rm),
                                         explains=cleared, evidence_removed=float(gain))
            work = wm
            rest = []
            for f in flagged:
                if f == m:
                    continue
                r = test(f, fl=work)
                if r is None:
                    confirmed.append(f)
                    retest_info[names[f]] = dict(result='insufficient support (first pass kept)', z_first=z_first[f])
                    continue
                out[f] = r
                if r['cps']:
                    rest.append(f)
                else:
                    retest_info[names[f]] = dict(result='cleared', z_first=z_first[f], explained_by=names[m])
            for f in after:
                ev[f] = after[f]
            flagged = rest

    delta = np.full((C, W, 3), np.nan)
    npts = np.zeros((C, W), int)
    resid = np.full((C, W), np.nan)
    cps_all = []
    stations = Stations(cameras=names)
    rots = {}
    low_support, bad_fit = {}, {}
    for c in cams:
        r = out[c]
        if r is None:
            continue
        delta[c], npts[c], resid[c] = r['delta'], r['cnt'], r['med']
        if r['low'].any():
            low_support[names[c]] = np.flatnonzero(r['low']).tolist()
        if r['badfit'].any():
            bad_fit[names[c]] = np.flatnonzero(r['badfit']).tolist()
        X, sel = r['X'], r['sel']
        times_c = []
        for cp in r['cps']:
            k = cp['index']
            # previous valid window
            kv = np.flatnonzero(np.isfinite(delta[c, :k]).all(1))
            kp = int(kv[-1]) if len(kv) else k - 1
            t_cp = t0 + k * window_s
            if refine:
                t_cp = _refine_step(P, c, X, flat[c, sel], tf[sel], wf[sel], kp, k,
                                    np.radians(cp['left_mean']), np.radians(cp['right_mean']), huber_px, t_cp)
            times_c.append(t_cp)
            cps_all.append(dict(camera=names[c], time=float(t_cp), window=int(k), step_deg=cp['step'], z=cp['z'],
                                left_deg=cp['left_mean'], right_deg=cp['right_mean']))
        # stations and their rotations
        st_names = [names[c]] + ['{}@s{}'.format(names[c], i + 1) for i in range(len(times_c))]
        edges = [-INF] + times_c + [INF]
        for i, nm in enumerate(st_names):
            if i > 0:
                stations = stations.paint(c, edges[i], edges[i + 1], nm)
        if skip_s:
            for tc in times_c:
                stations = stations.paint(c, tc - skip_s / 2, tc + skip_s / 2, SKIP)
        seg = stations.lookup(c, tf[sel])
        keep = seg >= 0
        rv_st, _, _, _ = fit_rotation_offsets(P, c, X[keep], flat[c, sel][keep], seg[keep],
                                              n_groups=len(st_names), huber_px=huber_px, min_points=min_points)
        rots[names[c]] = {nm: rv_st[i] for i, nm in enumerate(st_names)}
    rep = MoveReport(cameras=names, window_t0=t0 + np.arange(W) * window_s, window_s=float(window_s),
                     delta_deg=delta, n_points=npts, resid_px=resid,
                     change_points=sorted(cps_all, key=lambda d: (d['time'], d['camera'])),
                     stations=stations, station_rotations=rots,
                     info=dict(min_step_deg=min_step_deg, z_thr=z_thr, tested=[names[c] for c in cams],
                               min_support_frac=min_support_frac, low_support=low_support,
                               max_resid_ratio=max_resid_ratio, bad_fit=bad_fit,
                               retest=retest_info))
    # each new station starts from its fitted rotation in a station BA (Stations.poses)
    rep.stations.poses.update(_relative_poses(cgroup, rep))
    # leave-one-out points for recover_moves on the same data (not part of the report)
    rep._loo = dict(tau=float(tau), n=int(N * J), min_other=int(min_other),
                    cams={c: (out[c]['sel'], out[c]['X']) for c in cams if out[c] is not None})
    return rep


def _refine_step(P, c, X, uv, t, w, kl, kr, rl, rr, huber_px, t_default):
    """Frame-level step time between windows ``kl`` and ``kr`` from the robust per-frame cost
    under the left and right rotation offsets."""
    m = (w >= kl) & (w <= kr)
    if m.sum() < 4:
        return t_default
    Xm, uvm, tm = X[m], uv[m], t[m]
    dev, dt = P.device, P.dtype
    E = P.extrinsics()[c]
    Xt = torch.as_tensor(Xm, dtype=dt, device=dev)
    pc0 = Xt @ E[:3, :3].T + E[:3, 3]
    costs = []
    for rv in (rl, rr):
        R = G.rodrigues(torch.as_tensor(np.asarray(rv, float), dtype=dt, device=dev))[None]
        uvh, z, _ = _rot_project(P, c, R, pc0, torch.zeros(len(Xm), dtype=torch.long, device=dev), False)
        e = (uvh - torch.as_tensor(uvm, dtype=dt, device=dev)).norm(dim=-1).cpu().numpy()
        e = np.where(np.isfinite(e), e, 10 * huber_px)
        costs.append(np.where(e <= huber_px, e ** 2, 2 * huber_px * e - huber_px ** 2))
    ut, inv = np.unique(tm, return_inverse=True)
    cl = np.bincount(inv, weights=costs[0], minlength=len(ut))
    cr = np.bincount(inv, weights=costs[1], minlength=len(ut))
    # split before frame s: left frames [0, s), right frames [s, F)
    tot = np.r_[0, np.cumsum(cl)] + np.r_[np.cumsum(cr[::-1])[::-1], 0]
    s = int(np.argmin(tot))
    if s == 0:
        return float(ut[0])
    if s >= len(ut):
        return float(ut[-1])
    return float(0.5 * (ut[s - 1] + ut[s]))


# ------------------------------------------------------------------ moved-camera recovery

def _proj_pose(P, c, R, tv, Xw, E0=None, jac=False):
    """World points ``Xw [n,3]`` -> pixels of camera ``c`` at the pose ``x_cam = R (E0 X) + tv``
    (``E0``: the camera's reference extrinsics; ``R [3,3]`` / ``[G,3,3]`` with ``Xw [G,n,3]``).
    Returns ``uv, z`` and, with ``jac``, ``J [n,2,6]`` w.r.t. a left increment (rotation, translation)."""
    from .stba import _project_cam, _skew3
    p = _cam_params(P, c).clone()
    p[:6] = 0
    E = P.extrinsics()[c] if E0 is None else E0
    pc0 = Xw @ E[:3, :3].T + E[:3, 3]
    pc = pc0 @ R.transpose(-1, -2) + tv[..., None, :] if R.ndim == 3 else pc0 @ R.T + tv
    shp = pc.shape[:-1]
    out = _project_cam(p, pc.reshape(-1, 3), int(P.model[c]), P.K[c, 1, 0], P.offset[c], jac=jac)
    uv, z = out[0].reshape(shp + (2,)), out[1].reshape(shp)
    if not jac:
        return uv, z, None
    JX = out[2]
    Jr = -(JX @ _skew3(pc.reshape(-1, 3)))
    return uv, z, torch.cat([Jr, JX], -1)


def _match_score(uv, dets, gate):
    """``uv [..., I, J, 2]`` predicted, ``dets [I, D, J, 2]`` candidates (NaN padded) ->
    per-instance soft inlier score ``[..., I, D]`` (sum over joints of ``max(0, 1 - e^2/gate^2)``)."""
    e2 = ((uv[..., :, None, :, :] - dets) ** 2).sum(-1)
    s = (1 - e2 / gate ** 2).clamp(min=0)
    return torch.nan_to_num(s, nan=0.0).sum(-1)


def _grid_search(P, c, X, dets, gate, max_deg, step_deg, center=None, chunk_cells=4e6):
    """Best rotation offset (camera frame) on a ``(-max_deg, max_deg]^3`` grid of ``step_deg``
    around ``center`` (rotation vector, rad): ``(rotvec, score)``."""
    dev, dt = P.device, P.dtype
    a = np.radians(np.arange(-max_deg, max_deg + 1e-9, step_deg))
    g = np.stack(np.meshgrid(a, a, a, indexing='ij'), -1).reshape(-1, 3)
    Rc = G.rodrigues(torch.as_tensor(np.zeros(3) if center is None else center, dtype=dt, device=dev))
    Rg = G.rodrigues(torch.as_tensor(g, dtype=dt, device=dev)) @ Rc
    I, D, J = dets.shape[:3]
    bs = max(1, int(chunk_cells // max(I * D * J, 1)))
    scores = []
    zero = torch.zeros(3, dtype=dt, device=dev)
    for s in range(0, len(Rg), bs):
        R = Rg[s:s + bs]
        Xg = X.reshape(1, -1, 3).expand(len(R), -1, -1)
        uv, z, _ = _proj_pose(P, c, R, zero.expand(len(R), 3), Xg)
        uv = torch.where((z > 0)[..., None], uv, torch.full_like(uv, float('nan')))
        sc = _match_score(uv.reshape(len(R), I, J, 2), dets, gate)
        scores.append(sc.amax(-1).sum(-1))
    scores = torch.cat(scores)
    k = int(torch.argmax(scores))
    return _rotvec(Rg[k].cpu().numpy()), float(scores[k])


def _refine_pose(P, c, X, dets, R, tv, gate, full=True, n_outer=4, n_iter=8):
    """Robust (truncated) Gauss-Newton of the pose offset ``(R, tv)`` against the best-matching
    detection of each instance (re-selected every outer iteration): PnP on the matched joints.
    Returns ``R, tv, cost, n_inliers, choice [I]``."""
    dev, dt = P.device, P.dtype
    I, D, J = dets.shape[:3]
    nd = 6 if full else 3
    for _ in range(n_outer):
        uv, z, _ = _proj_pose(P, c, R, tv, X.reshape(-1, 3))
        sc = _match_score(uv.reshape(I, J, 2), dets, gate)
        choice = sc.argmax(-1)
        tgt = dets[torch.arange(I, device=dev), choice].reshape(-1, 2)
        Xf = X.reshape(-1, 3)
        ok = torch.isfinite(tgt).all(-1) & torch.isfinite(Xf).all(-1)
        if int(ok.sum()) < 4:
            break
        Xo, To = Xf[ok], tgt[ok]
        for _ in range(n_iter):
            uv, z, Jc = _proj_pose(P, c, R, tv, Xo, jac=True)
            r = uv - To
            e = r.norm(dim=-1)
            w = ((e < gate) & (z > 0) & torch.isfinite(e)).to(dt)
            r = torch.nan_to_num(r) * w[:, None]
            Jc = torch.nan_to_num(Jc[..., :nd]) * w[:, None, None]
            H = torch.einsum('nai,naj->ij', Jc, Jc) + 1e-9 * torch.eye(nd, dtype=dt, device=dev)
            gv = torch.einsum('nai,na->i', Jc, r)
            step = -torch.linalg.solve(H, gv)
            dR = G.rodrigues(step[:3])
            R = dR @ R
            tv = dR @ tv + (step[3:] if full else 0)
            if float(step.abs().max()) < 1e-10:
                break
    uv, z, _ = _proj_pose(P, c, R, tv, X.reshape(-1, 3))
    sc = _match_score(uv.reshape(I, J, 2), dets, gate)
    choice = sc.argmax(-1)
    tgt = dets[torch.arange(I, device=dev), choice].reshape(-1, 2)
    e = (uv - tgt).norm(dim=-1)
    e = torch.where(torch.isfinite(e) & (z > 0), e, torch.full_like(e, float('inf')))
    cost = float(torch.clamp(e, max=gate).pow(2).sum())
    return R, tv, cost, int((e < gate).sum()), choice


def _frame_detections(detections, names, c):
    if isinstance(detections, dict):
        return detections.get(names[c]) or {}
    return detections[c]


def _get_frame(dc, f):
    if isinstance(dc, dict):
        v = dc.get(f, dc.get(int(f)) if not isinstance(f, int) else None)
    else:
        v = dc[int(f)] if 0 <= int(f) < len(dc) else None
    if v is None:
        return None
    v = np.asarray(v, float) if not isinstance(v, list) else np.stack([np.asarray(x, float) for x in v]) \
        if len(v) else None
    if v is None or v.size == 0:
        return None
    return v[..., :2].reshape(v.shape[0], -1, 2)


def _assign(cost, gate):
    from scipy.optimize import linear_sum_assignment
    cc = np.where(np.isfinite(cost), cost, 1e9)
    r, d = linear_sum_assignment(cc)
    keep = cc[r, d] < gate
    return r[keep], d[keep]


@dataclass
class Recovery:
    """Output of :func:`recover_moves`: ``moves`` (the input :class:`MoveReport` plus the recovered
    stations, change points with ``recovered=True`` and ``station_poses``), ``points`` (the input
    points with the recovered cameras re-associated from ``detections``; None without
    detections), ``cameras`` (per camera: collapse time, step, translation, coverage before /
    lost / after, status)."""
    moves: Any
    points: Any
    cameras: dict


def _loo_quality(P, flat, c, tau, min_other, agree_px, cache=None):
    """Per cell: supported by the other cameras (>= min_other views) and, among those, whether
    camera c agrees (LOO error < agree_px). ``cache``: ``(sel, X)`` from :func:`detect_moves`."""
    C = flat.shape[0]
    others = [o for o in range(C) if o != c]
    nv = np.isfinite(flat[others]).all(-1).sum(0)
    sup = nv >= min_other
    hit = np.zeros(flat.shape[1], bool)
    if cache is not None:
        sel, X = cache
        X = torch.as_tensor(np.asarray(X, float), dtype=P.dtype, device=P.device)
    else:
        sel = np.flatnonzero(sup & np.isfinite(flat[c]).all(-1))
        X = None
    if len(sel):
        if X is None:
            r = T.triangulate_robust(P.subset(others), flat[others][:, sel], method='msac', tau=tau)
            X = torch.as_tensor(np.asarray(r.p3d), dtype=P.dtype, device=P.device)
        eye = torch.eye(3, dtype=P.dtype, device=P.device)
        uv, z, _ = _proj_pose(P, c, eye, torch.zeros(3, dtype=P.dtype, device=P.device), X)
        uv = torch.where((z > 0)[:, None], uv, torch.full_like(uv, float('nan'))).cpu().numpy()
        e = np.linalg.norm(uv - flat[c, sel], axis=-1)
        hit[sel] = np.isfinite(e) & (e < agree_px)
    return sup, hit


def _collapses(q, ns, nh, jw, collapse_ratio, min_windows, min_support, max_n=5):
    """Collapsed runs ``(start, end, level_before)`` (indices into the judged windows ``jw``)."""
    out = []
    i = min_windows
    while i <= len(jw) - min_windows and len(out) < max_n:
        before = np.nanmedian(q[jw[:i]])
        after = q[jw[i:i + min_windows]]
        exp = before * ns[jw[i:i + min_windows]]
        if before > 0 and exp.sum() >= min_support and \
                nh[jw[i:i + min_windows]].sum() < collapse_ratio * exp.sum() and np.all(after < 0.5 * before):
            # end of the run: the first window back to >= half the earlier level
            end = len(jw)
            for k in range(i, len(jw)):
                if q[jw[k]] >= 0.5 * before:
                    end = k
                    break
            out.append((i, end, float(before)))
            i = end + min_windows
        else:
            i += 1
    return out


def recover_moves(cgroup, data, times, moves=None, detections=None, frame_ids=None, window_s=None,
                  tau=10.0, agree_px=None, gate_px=None, min_other=2, collapse_ratio=0.3, min_windows=3,
                  min_support=20, max_deg=15.0, grid_step_deg=2.5, max_instances=600, cameras=None,
                  device=None, rng=0, persistent=True, min_step_deg=0.5, max_blocks=40,
                  split_step_deg=1.0):
    """Re-place cameras that were bumped so hard that their LOO coverage collapses.

    A camera whose share of the points supported by the other cameras that it also sees and agrees
    with (LOO error < ``agree_px``) drops below ``collapse_ratio`` x its earlier level, in at least
    ``min_windows`` windows of ``window_s`` (default: ``moves.window_s`` or 4 s) from a time
    ``t`` on, was moved at ``t`` (or lost its data: with people, an association with the old rig
    cannot match its detections any more). Its new pose is found from the points the *other*
    cameras triangulate after ``t``: a rotation grid search (``+-max_deg`` in ``grid_step_deg``
    steps, then a finer grid) scores how many projected keypoints land near a detection of the
    camera, then a robust Gauss-Newton PnP (rotation + translation, kept only if it explains the
    data clearly better than the rotation alone) refines it. Detections come from ``detections``
    (unassociated, the :func:`~aniposelib.workflows.associate_views` input: per camera, per frame
    id a ``(D, J, 2|3)`` array; ``frame_ids [N]`` maps the rows of ``data`` to those ids;
    the best-matching detection of each instance is used) or else from the camera's own rows of
    ``data``. With ``detections`` the camera is then re-associated after ``t`` (Hungarian per
    frame on the median keypoint distance, gate ``2 x gate_px``) and the returned ``points``
    carry its observations.

    Coverage is the share of the cells the other cameras support (>= ``min_other`` views) where the
    camera agrees; without ``detections`` only cells the camera observes count (a camera that stops
    seeing the animal has not moved, and cannot be re-placed from its own data anyway). With
    ``persistent`` (default) only collapses that last to the end of the data are moves (a coverage
    that comes back by itself is broken data or sync, not a bump). Collapses are handled earliest
    first, then the most complete; a camera already re-placed is used at its new pose (and with its
    re-associated detections) for the next ones. A camera that still fits at its old pose (people
    data whose association dropped it), or whose recovered rotation is below ``min_step_deg``, is
    'not moved' (with ``detections`` it is still re-associated).

    Inside a recovered interval the pose is refined per block of windows (at most ``max_blocks``);
    a block more than ``split_step_deg`` (1 deg) from its segment's first block starts a new station
    (a second bump). A camera that still fits at its old pose but whose locally refined rotation
    (>= ``min_step_deg``) lowers the median residual by 30 % is a small re-aim and recovered too.

    A recovered camera gets a new station from ``t`` (an existing change point of ``moves`` within
    two windows is reused). It is accepted when its coverage after recovery is at least half its
    coverage before the move. Returns a :class:`Recovery`.
    """
    P = G.pack(cgroup, device=device)
    names = list(P.names) if P.names else ['cam{}'.format(i) for i in range(P.n_cams)]
    p2d = data.detach().cpu().numpy() if isinstance(data, torch.Tensor) else np.array(data, float)
    squeeze = p2d.ndim == 3
    if squeeze:
        p2d = p2d[:, :, None]
    C, N, J, _ = p2d.shape
    t = np.asarray(times, float).reshape(-1)
    if len(t) != N:
        raise ValueError('times has {} entries for {} rows'.format(len(t), N))
    if detections is not None and frame_ids is None:
        frame_ids = np.arange(N)
    agree_px = float(agree_px if agree_px is not None else 0.8 * tau)
    gate = float(gate_px if gate_px is not None else tau)
    ws = float(window_s or (moves.window_s if moves is not None else 4.0))
    t0 = float(np.nanmin(t))
    win = np.floor((t - t0) / ws).astype(int)
    W = int(win.max()) + 1
    flat = p2d.reshape(C, N * J, 2)
    wf = np.repeat(win, J)
    cams = list(range(C)) if cameras is None else [int(x) for x in cameras]
    out_moves = moves
    st = moves.stations.copy() if moves is not None else Stations(cameras=names)
    rots = dict(moves.station_rotations) if moves is not None else {}
    poses = {k: dict(v) for k, v in (getattr(moves, 'station_poses', None) or {}).items()}
    cps = list(moves.change_points) if moves is not None else []
    new_pts = None
    info = {}
    E_all = P.extrinsics()
    dev, dt = P.device, P.dtype
    gen = np.random.default_rng(rng)
    lc = getattr(moves, '_loo', None)
    if not (lc and lc['tau'] == float(tau) and lc['n'] == N * J and lc['min_other'] == int(min_other)):
        lc = None
    # phase 1: collapsed runs per camera
    jobs = []
    overall = {}
    transient = {}
    for c in cams:
        sup, hit = _loo_quality(P, flat, c, tau, min_other, agree_px,
                                cache=None if lc is None else lc['cams'].get(c))
        if detections is None:
            # only the camera's own data: judge its agreement where it sees the point (a camera
            # that stops seeing the animal is not a moved camera, and cannot be re-placed anyway)
            sup = sup & np.isfinite(flat[c]).all(-1)
        ns = np.bincount(wf, weights=sup, minlength=W)
        nh = np.bincount(wf, weights=hit, minlength=W)
        judged = ns >= min_support
        if judged.sum() < 2 * min_windows:
            continue
        q = np.where(judged, nh / np.maximum(ns, 1), np.nan)
        jw = np.flatnonzero(judged)
        overall[c] = (float(nh[judged].sum() / max(ns[judged].sum(), 1)), jw, q)
        for cand, end, before in _collapses(q, ns, nh, jw, collapse_ratio, min_windows, min_support):
            k0 = jw[cand]
            good_t = t[hit.reshape(N, J).any(1) & (win < k0)]
            t_last = float(good_t.max()) if len(good_t) else t0 + k0 * ws
            rows_after = np.flatnonzero((t > t_last) & (win >= k0) & (sup.reshape(N, J).any(1)))
            t_cp = float(0.5 * (t_last + t[rows_after].min())) if len(rows_after) else t0 + k0 * ws
            t_end = INF if end >= len(jw) else t0 + jw[end] * ws
            lost = float(np.nanmedian(q[jw[cand:end]]))
            if persistent and np.isfinite(t_end):
                transient[names[c]] = transient.get(names[c], 0) + 1      # came back: not a lasting move
                continue
            d = dict(time=t_cp, t_end=t_end, coverage_before=before, coverage_lost=lost, status='failed')
            info.setdefault(names[c], []).append(d)
            jobs.append((k0, lost / max(before, 1e-9), c, t_cp, t_end, before, d))
    # cameras the association lost from the start (people data with detections): every window is
    # re-placed on its own (or found unmoved), equal neighbours share a station
    if detections is not None and len(overall) >= 3:
        for c, (qc, jw, q) in overall.items():
            ref = float(np.median([v[0] for o, v in overall.items() if o != c]))
            if any(j[2] == c for j in jobs) or not qc < collapse_ratio * ref:
                continue
            info.setdefault(names[c], [])
            blocks = np.array_split(jw, min(len(jw), max_blocks))
            for b in blocks:
                t_a = t0 + b[0] * ws if b is not blocks[0] else -INF
                t_b = t0 + (b[-1] + 1) * ws if b is not blocks[-1] else INF
                d = dict(time=float(t0 + b[0] * ws), t_end=t_b, coverage_before=ref,
                         coverage_lost=float(np.nanmedian(q[b])), status='failed', lost_camera=True)
                info[names[c]].append(d)
                jobs.append((b[0], 0.0, c, t_a, t_b, ref, d))
    # phase 2: earliest first, then the most collapsed; each recovered camera's pose and data are
    # used when the next cameras are re-placed
    work = p2d.copy()
    overrides = []                      # (cam, t0, t1, E [4,4] torch) of the re-placed cameras
    jobs.sort(key=lambda j: (j[0], j[1]))
    zero3 = torch.zeros(3, dtype=dt, device=dev)
    eye3 = torch.eye(3, dtype=dt, device=dev)
    for k0, _, c, t_cp, t_end, before, d in jobs:
        name = names[c]
        rows = np.flatnonzero((t >= t_cp) & (t < t_end))
        others = [o for o in range(C) if o != c]
        # points of the other cameras after the move (recovered cameras at their new pose)
        Xr = np.full((len(rows), J, 3), np.nan)
        keys = np.zeros((len(rows), 0), bool)
        ovs = [ov for ov in overrides if ov[0] != c]
        if ovs:
            keys = np.stack([(t[rows] >= ov[1]) & (t[rows] < ov[2]) for ov in ovs], 1)
        ukeys, kinv = np.unique(keys, axis=0, return_inverse=True)
        kinv = np.asarray(kinv).reshape(-1)
        for kk, key in enumerate(ukeys):
            m = kinv == kk
            Pk = P
            on = [ov for ov, b in zip(ovs, key) if b]
            if on:
                cgk = cgroup.copy()
                for o, _, _, E in on:
                    E = E.cpu().numpy()
                    cgk.cameras[o].set_rotation(_rotvec(E[:3, :3]))
                    cgk.cameras[o].set_translation(E[:3, 3])
                Pk = G.pack(cgk, device=dev)
            r = T.triangulate_robust(Pk.subset(others), work[others][:, rows[m]].reshape(len(others), -1, 2),
                                     method='msac', tau=tau)
            Xr[m] = np.asarray(r.p3d, float).reshape(int(m.sum()), J, 3)
        inst = np.flatnonzero(np.isfinite(Xr).all(-1).sum(-1) >= min(3, J))
        # candidate detections of camera c per instance
        if detections is not None:
            dc = _frame_detections(detections, names, c)
            dl = [_get_frame(dc, frame_ids[rows[i]]) for i in inst]
        else:
            dl = [p2d[c, rows[i]][None] if np.isfinite(p2d[c, rows[i]]).any() else None for i in inst]
        has = np.array([x is not None for x in dl], bool)
        inst = inst[has]
        dl = [x for x in dl if x is not None]
        d['n_instances'] = int(len(inst))
        if len(inst) < max(3, min_support // max(J, 1)):
            d['status'] = 'failed: too few points from the other cameras with detections'
            continue
        pick = np.sort(gen.choice(len(inst), max_instances, replace=False)) if len(inst) > max_instances \
            else np.arange(len(inst))
        Dm = max(x.shape[0] for x in dl)

        def pad(ix):
            a = np.full((len(ix), Dm, J, 2), np.nan)
            for k, i in enumerate(ix):
                a[k, :dl[i].shape[0]] = dl[i]
            return torch.as_tensor(a, dtype=dt, device=dev)
        Xa = torch.as_tensor(Xr[inst], dtype=dt, device=dev)
        Da = pad(np.arange(len(inst)))
        cells = np.isfinite(Xr[inst]).all(-1)
        n_sup = max(int(cells.sum()), 1) if detections is None else \
            max(int(np.isfinite(Xr).all(-1).sum()), 1)

        def coverage(R, tv):
            uv, z, _ = _proj_pose(P, c, R, tv, Xa.reshape(-1, 3))
            sc = _match_score(uv.reshape(len(inst), J, 2), Da, gate)
            ch = sc.argmax(-1)
            tgt = Da[torch.arange(len(inst), device=dev), ch].reshape(-1, 2)
            e = (uv - tgt).norm(dim=-1).reshape(len(inst), J).cpu().numpy()
            return float(((e < agree_px) & cells).sum() / n_sup), e
        cov0, e0 = coverage(eye3, zero3)
        d['coverage_unmoved'] = cov0
        small = None
        if cov0 >= 0.5 * before:
            # still fits at its old pose: a small re-aim (within the agreement threshold) shows as a
            # clearly lower residual after a local refinement
            Xs, Ds = Xa[pick], pad(pick)
            Rr, tr, _, _, _ = _refine_pose(P, c, Xs, Ds, eye3, zero3, gate, full=False)
            cov_r, er = coverage(Rr, tr)
            step_r = float(np.degrees(np.linalg.norm(_rotvec(Rr.cpu().numpy()))))
            m0, mr = (e0 < gate) & cells, (er < gate) & cells
            med0 = float(np.median(e0[m0])) if m0.any() else np.inf
            medr = float(np.median(er[mr])) if mr.any() else np.inf
            if step_r >= min_step_deg and medr < 0.7 * med0 and cov_r >= cov0:
                small = (Rr, tr, cov_r, er, step_r, med0, medr)
            else:
                Rb, tb, moved_ = eye3, zero3, False
                d.update(status='not moved: the camera still fits at its old pose (data lost, e.g. association)',
                         step_deg=0.0)
        if small is not None:
            Rb, tb, after, e, step, med0, medr = small
            d.update(coverage_after=after, step_deg=step, rotvec_deg=np.degrees(_rotvec(Rb.cpu().numpy())).tolist(),
                     center_shift=0.0, translation=False, median_px=medr, median_px_unmoved=med0,
                     n_inliers=int(((e < gate) & cells).sum()))
            moved_ = True
        elif cov0 < 0.5 * before:
            Xs, Ds = Xa[pick], pad(pick)
            f = float(P.K[c, 0, 0])
            coarse_gate = max(gate, 0.6 * np.radians(grid_step_deg) * f)
            rv, _ = _grid_search(P, c, Xs, Ds, coarse_gate, max_deg, grid_step_deg)
            rv, _ = _grid_search(P, c, Xs, Ds, max(gate, coarse_gate / 4), grid_step_deg, grid_step_deg / 4,
                                 center=rv)
            R0 = G.rodrigues(torch.as_tensor(rv, dtype=dt, device=dev))
            Rr, tr, _, _, _ = _refine_pose(P, c, Xs, Ds, R0, zero3, 2 * gate, full=False)
            Rr, tr, cost_r, nin_r, _ = _refine_pose(P, c, Xs, Ds, Rr, tr, gate, full=False)
            Rf, tf_, cost_f, nin_f, _ = _refine_pose(P, c, Xs, Ds, Rr, tr, gate, full=True)
            full = cost_f < 0.8 * cost_r and nin_f >= nin_r
            Rb, tb = (Rf, tf_) if full else (Rr, tr)
            after, e = coverage(Rb, tb)
            step = float(np.degrees(np.linalg.norm(_rotvec(Rb.cpu().numpy()))))
            E0 = E_all[c]
            cen0 = -(E0[:3, :3].T @ E0[:3, 3])
            E1 = torch.eye(4, dtype=dt, device=dev)
            E1[:3, :3] = Rb @ E0[:3, :3]
            E1[:3, 3] = Rb @ E0[:3, 3] + tb
            cen1 = -(E1[:3, :3].T @ E1[:3, 3])
            ok = (e < gate) & cells
            d.update(coverage_after=after, step_deg=step, rotvec_deg=np.degrees(_rotvec(Rb.cpu().numpy())).tolist(),
                     center_shift=float((cen1 - cen0).norm()), translation=bool(full),
                     n_inliers=int(nin_f if full else nin_r), median_px=float(np.median(e[ok])) if ok.any() else None)
            if not after >= 0.5 * before:
                d['status'] = 'failed: coverage after recovery {:.3g} < half of {:.3g}'.format(after, before)
                continue
            if after - cov0 < 0.25 * before:
                d['status'] = 'not moved: the recovered pose explains little more than the old one ({:.3g} vs {:.3g})'.format(
                    after, cov0)
                Rb, tb = eye3, zero3
                moved_ = False
            elif step < min_step_deg and not full:
                d['status'] = 'not moved: recovered rotation {:.2g} deg < min_step_deg'.format(step)
                Rb, tb, moved_ = eye3, zero3, False
            else:
                moved_ = True
        segs = [(t_cp, t_end, Rb, tb)]
        if moved_:
            # further steps inside the interval (a second bump): the pose refined per block of
            # windows from the recovered one; blocks within min_step_deg of a segment's first block
            # share its station
            wi = win[rows[inst]]
            uw = np.unique(wi)
            br = []
            for bl in np.array_split(uw, min(len(uw), max_blocks)):
                sub = np.flatnonzero(np.isin(wi, bl))
                if len(sub) < max(3, min_support // max(J, 1)):
                    continue
                if len(sub) > max_instances:
                    sub = np.sort(gen.choice(sub, max_instances, replace=False))
                Rk, tk, _, _, _ = _refine_pose(P, c, Xa[sub], Da[sub], Rb, tb, 2 * gate, full=False)
                Rk, tk, _, _, _ = _refine_pose(P, c, Xa[sub], Da[sub], Rk, tk, gate, full=False)
                br.append((int(bl[0]), int(bl[-1]), Rk))
            groups = []
            for x in br:
                if groups and np.degrees(np.linalg.norm(_rotvec((x[2] @ groups[-1][0][2].T).cpu().numpy()))) \
                        < split_step_deg:
                    groups[-1].append(x)
                else:
                    groups.append([x])
            if len(groups) > 1:
                segs = []
                tin = t[rows[inst]]
                for gi, g in enumerate(groups):
                    ta = t_cp if gi == 0 else t0 + g[0][0] * ws
                    tb_ = t_end if gi == len(groups) - 1 else t0 + groups[gi + 1][0][0] * ws
                    sub = np.flatnonzero((tin >= ta) & (tin < tb_))
                    if len(sub) > max_instances:
                        sub = np.sort(gen.choice(sub, max_instances, replace=False))
                    Rk, tk, _, _, _ = _refine_pose(P, c, Xa[sub], Da[sub], g[0][2], tb, gate, full=False)
                    if bool(d.get('translation')):
                        Rk, tk, _, _, _ = _refine_pose(P, c, Xa[sub], Da[sub], Rk, tk, gate, full=True)
                    segs.append((ta, tb_, Rk, tk))
                d['segments'] = [dict(time=float(a_), t_end=float(b_),
                                      rotvec_deg=np.degrees(_rotvec(R_.cpu().numpy())).tolist()) for a_, b_, R_, _ in segs]
            d['status'] = 'recovered'
        E0 = E_all[c]
        for si, (sa, sb, Rs, ts) in enumerate(segs):
            srows = np.flatnonzero((t[rows] >= sa) & (t[rows] < sb))
            if moved_:
                E1 = torch.eye(4, dtype=dt, device=dev)
                E1[:3, :3] = Rs @ E0[:3, :3]
                E1[:3, 3] = Rs @ E0[:3, 3] + ts
                E1n = E1.cpu().numpy()
                overrides.append((c, sa, sb, E1))
                exist = [cp for cp in cps if cp['camera'] == name and abs(cp['time'] - sa) <= 2 * ws]
                same = [k_ for k_, E_ in poses.get(name, {}).items()
                        if np.degrees(np.linalg.norm(_rotvec(np.asarray(E_)[:3, :3] @ E1n[:3, :3].T))) < min_step_deg
                        and np.linalg.norm(np.asarray(E_)[:3, 3] - E1n[:3, 3])
                        < 0.01 * float(np.linalg.norm(E1n[:3, 3]) + 1e-9)]
                if same:
                    sn = same[0]
                    exist = []
                elif exist:
                    sa = float(exist[0]['time'])
                    sn = str(st.lookup_name(name, [sa + 1e-9])[0])
                    if sn == SKIP:
                        sn = str(st.lookup_name(name, [sa + 2 * ws])[0])
                    cps = [cp for cp in cps if cp is not exist[0]]
                else:
                    k = len(st.names(name))
                    sn = '{}@s{}'.format(name, k)
                    while sn in st.names(name):
                        k += 1
                        sn = '{}@s{}'.format(name, k)
                prev = str(st.lookup_name(name, [sa - 1e-6])[0]) if np.isfinite(sa) else None
                st = st.paint(name, sa, sb, sn)
                rots.setdefault(name, {name: np.zeros(3)})
                rv_s = _rotvec(E1n[:3, :3] @ E0[:3, :3].T.cpu().numpy())
                rots[name][sn] = rv_s
                poses.setdefault(name, {})[sn] = E1n
                if prev != sn:
                    cps.append(dict(camera=name, time=float(sa) if np.isfinite(sa) else float(t0), window=int(k0),
                                    step_deg=float(np.degrees(np.linalg.norm(rv_s))), z=None, recovered=True,
                                    center_shift=float(np.linalg.norm(
                                        -(E1n[:3, :3].T @ E1n[:3, 3]) + (E0[:3, :3].T @ E0[:3, 3]).cpu().numpy())),
                                    left_deg=[0.0, 0.0, 0.0], right_deg=np.degrees(rv_s).tolist()))
                if si == 0:
                    d['station'] = sn
                    if np.isfinite(sa):
                        d['time'] = float(sa)
            # re-association of the camera's detections after the move (or after the data loss)
            if detections is not None and len(srows):
                if new_pts is None:
                    new_pts = p2d.copy()
                dc = _frame_detections(detections, names, c)
                Xs_ = Xr[srows]
                Xall = torch.as_tensor(np.nan_to_num(Xs_), dtype=dt, device=dev)
                uva, _, _ = _proj_pose(P, c, Rs, ts, Xall.reshape(-1, 3))
                uva = uva.reshape(len(srows), J, 2).cpu().numpy()
                uva[~np.isfinite(Xs_).all(-1)] = np.nan
                rr = rows[srows]
                new_pts[c, rr] = np.nan
                fr = np.asarray(frame_ids)[rr]
                n_as = 0
                for fid in np.unique(fr):
                    ri = np.flatnonzero(fr == fid)
                    det = _get_frame(dc, fid)
                    if det is None:
                        continue
                    e2 = np.linalg.norm(uva[ri][:, None] - det[None], axis=-1)       # [r, D, J]
                    with np.errstate(all='ignore'), warnings.catch_warnings():
                        warnings.simplefilter('ignore', RuntimeWarning)
                        nj = np.isfinite(e2).sum(-1)
                        cost = np.where(nj >= min(3, J), np.nanmedian(e2, -1), np.inf)
                    a_, b_ = _assign(cost, 2 * gate)
                    for i, k in zip(a_, b_):
                        new_pts[c, rr[ri[i]]] = det[k]
                        n_as += 1
                work[c, rr] = new_pts[c, rr]
                d['n_reassociated'] = d.get('n_reassociated', 0) + int(n_as)
    if moves is not None or info or transient:
        base = moves if moves is not None else MoveReport(
            cameras=names, window_t0=t0 + np.arange(W) * ws, window_s=ws, delta_deg=np.full((C, W, 3), np.nan),
            n_points=np.zeros((C, W), int), resid_px=np.full((C, W), np.nan), change_points=[], stations=st,
            station_rotations={}, info={})
        out_moves = dataclasses.replace(base, stations=st, station_rotations=rots,
                                        change_points=sorted(cps, key=lambda d: (d['time'], d['camera'])),
                                        station_poses=poses, info=dict(base.info, recovered=_jsonable(info),
                                                                       transient_collapses=transient))
        out_moves.stations.poses.update(_relative_poses(cgroup, out_moves))
    return Recovery(moves=out_moves, points=None if new_pts is None else (new_pts[:, :, 0] if squeeze else new_pts),
                    cameras=_jsonable(info))


def _relative_poses(cgroup, moves):
    """``{station: dict(camera, D, scale)}``: each station's pose relative to its camera's pose in
    ``cgroup`` (``E_station = D E_camera``, translation in units of the rig scale ``scale``)."""
    from .scale import _rig_scale, camera_centers
    s = float(_rig_scale(camera_centers(cgroup)))
    out = {}
    names = cgroup.get_names()
    for cam, d in (getattr(moves, 'station_poses', None) or {}).items():
        E0 = cgroup.cameras[names.index(cam)].get_extrinsics_mat()
        E0 = np.asarray(E0.detach().cpu().numpy() if isinstance(E0, torch.Tensor) else E0, float)
        for sn, E in d.items():
            out[sn] = dict(camera=cam, D=np.asarray(E, float) @ np.linalg.inv(E0), scale=s)
    for cam, d in (moves.station_rotations or {}).items():
        for sn, rv in d.items():
            if sn == cam or sn in out or not np.isfinite(np.asarray(rv, float)).all():
                continue
            D = np.eye(4)
            D[:3, :3] = G.rodrigues(torch.as_tensor(np.asarray(rv, float))).numpy()
            out[sn] = dict(camera=cam, D=D, scale=s)
    return out


def station_init(cgroup, stations, names_order):
    """Initial ``(rvec [S,3], tvec [S,3])`` for the stations ``names_order`` (list of (camera index,
    station name)) of ``cgroup``: ``stations.poses`` (relative, rescaled to this group's rig scale)
    where known, else the camera's own pose. None when no station has a pose."""
    poses = getattr(stations, 'poses', None) or {}
    if not any(sn in poses for _, sn in names_order):
        return None
    from .scale import _rig_scale, camera_centers
    s = float(_rig_scale(camera_centers(cgroup)))
    rv, tv = [], []
    for c, sn in names_order:
        E0 = cgroup.cameras[c].get_extrinsics_mat()
        E0 = np.asarray(E0.detach().cpu().numpy() if isinstance(E0, torch.Tensor) else E0, float)
        p = poses.get(sn)
        if p is not None:
            D = np.array(p['D'], float)
            if p.get('scale'):
                D[:3, 3] *= s / float(p['scale'])
            E0 = D @ E0
        rv.append(_rotvec(E0[:3, :3]))
        tv.append(E0[:3, 3])
    return np.array(rv), np.array(tv)
