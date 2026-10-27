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
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from . import geometry as G
from . import triangulation as T

__all__ = ['SKIP', 'Stations', 'MoveReport', 'detect_moves', 'detect_steps', 'fit_rotation_offsets']

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

    def __init__(self, intervals=None, cameras=None):
        self.cameras = list(cameras) if cameras is not None else None
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
        s = Stations(cameras=self.cameras)
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
                 min_support_frac=0.25, retest=True):
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
        retest: after the first pass, flagged cameras are re-tested in order of decreasing z
            with the already-confirmed moved cameras removed from the triangulating views; a
            camera stays flagged only if it is still flagged without them (the strongest one is
            always kept). If the re-test lacks support (``min_points`` / ``min_other``), the
            first-pass result is kept. Details in ``info['retest']``. ``False``: one pass.
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

    def test(c, exclude=()):
        others = [o for o in range(C) if o != c and o not in exclude]
        if len(others) < max(min_other, 2):
            return None
        nv = np.isfinite(flat[others]).all(-1).sum(0)
        sel = np.flatnonzero((nv >= min_other) & np.isfinite(flat[c]).all(-1))
        if len(sel) < min_points:
            return None
        res = T.triangulate_robust(P.subset(others), flat[others][:, sel], method='msac', tau=tau)
        X = np.asarray(res.p3d)
        ok = np.isfinite(X).all(1)
        sel, X = sel[ok], X[ok]
        if len(sel) < min_points:
            return None
        rv, med, cnt, _ = fit_rotation_offsets(P, c, X, flat[c, sel], wf[sel], n_groups=W,
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
        cps = detect_steps(d, np.sqrt(np.maximum(cnt, 0)) * np.isfinite(d).all(1),
                           min_step=min_step_deg, min_seg=min_windows, z_thr=z_thr, sharp=sharp, max_cp=max_cp)
        return dict(delta=d, cnt=cnt, med=med, cps=cps, X=X, sel=sel, low=low, others=others)

    out = {c: test(c) for c in cams}
    retest_info = {}
    if retest:
        flagged = [c for c in cams if out[c] is not None and out[c]['cps']]
        flagged.sort(key=lambda c: -max(cp['z'] for cp in out[c]['cps']))
        confirmed = []
        for c in flagged:
            if not confirmed:
                confirmed.append(c)
                retest_info[names[c]] = dict(result='kept', excluded=[], z_first=max(cp['z'] for cp in out[c]['cps']))
                continue
            excl = [o for o in confirmed if o != c]
            r = test(c, exclude=excl)
            z1 = max(cp['z'] for cp in out[c]['cps'])
            if r is None:
                retest_info[names[c]] = dict(result='insufficient support (first pass kept)',
                                             excluded=[names[o] for o in excl], z_first=z1)
                confirmed.append(c)
                continue
            out[c] = r
            if r['cps']:
                confirmed.append(c)
                retest_info[names[c]] = dict(result='kept', excluded=[names[o] for o in excl], z_first=z1,
                                             z_retest=max(cp['z'] for cp in r['cps']))
            else:
                retest_info[names[c]] = dict(result='cleared', excluded=[names[o] for o in excl], z_first=z1)

    delta = np.full((C, W, 3), np.nan)
    npts = np.zeros((C, W), int)
    resid = np.full((C, W), np.nan)
    cps_all = []
    stations = Stations(cameras=names)
    rots = {}
    low_support = {}
    for c in cams:
        r = out[c]
        if r is None:
            continue
        delta[c], npts[c], resid[c] = r['delta'], r['cnt'], r['med']
        if r['low'].any():
            low_support[names[c]] = np.flatnonzero(r['low']).tolist()
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
    return MoveReport(cameras=names, window_t0=t0 + np.arange(W) * window_s, window_s=float(window_s),
                      delta_deg=delta, n_points=npts, resid_px=resid,
                      change_points=sorted(cps_all, key=lambda d: (d['time'], d['camera'])),
                      stations=stations, station_rotations=rots,
                      info=dict(min_step_deg=min_step_deg, z_thr=z_thr, tested=[names[c] for c in cams],
                                min_support_frac=min_support_frac, low_support=low_support,
                                retest=retest_info))


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
