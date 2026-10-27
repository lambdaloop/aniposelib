"""Calibration QA and diagnostics (1.0, plan 01 C4; report 07 section 3.7).

Reprojection error cannot audit a calibration on its own: a moved camera that helps
triangulate its own observations pulls the 3D points towards itself and partly hides. The
central tool here is therefore **leave-one-camera-out (LOO)** reprojection: each camera is
scored against points triangulated (robustly) from the *other* cameras only.

API
---
``loo_reprojection(cgroup, p2ds, tau_tri=5, min_others=3, robust=True) -> LOOResult``
    Fixed denominator (Emory ``evaluate_v2``): an observation is *eligible* when the target
    camera sees it and >= ``min_others`` other cameras do. This depends only on the input,
    never on the calibration. Eligible observations without a supported triangulation (fewer
    than ``min_others`` agreeing other views, or behind the target camera) are **misses**
    (error ``inf``): they count against agreement but not in the quantiles, so quantiles must
    be read together with ``coverage``.
``report(cgroup, p2ds, robust=True, loo=True, min_views=3, fit_mask=None, groups=None,
         thresholds=(5, 8, 10)) -> CalibrationReport``
    Per camera: ``n_obs``, ``n_used``, ``coverage``, all-observation and fit-set
    ``median/p90/p95_px``, held-out (not in the fit mask) median, ``inlier_rate``, LOO
    ``loo_median_px`` / ``loo_coverage`` / ``agree@T`` (fixed denominator) and ``flags``.
    ``overall`` summarises; ``per_group`` breaks the LOO agreement down by point group.
    ``to_dict()``, ``to_json(path=None)``, ``passes(max_median_px=12, max_loo_px=60)``.
``compare(cg_a, cg_b, p2ds=None, trim=2.5) -> CompareResult``
    Trimmed Umeyama-aligned camera-centre residuals, scale, rotation, per-camera orientation
    and intrinsics deltas, and (with ``p2ds``) paired LOO/reprojection errors of both groups.
``sanity(cgroup, f_range=None, pp_frac=0.2, k_max=None) -> dict`` and ``lens_validity(camera)``
    Plausibility of intrinsics; lens round-trip and monotone radial mapping (Emory).
``coverage(p2ds, min_cams=3, cgroup=None, grid=(4, 4)) -> dict``
    Observation counts per camera, multi-view counts and image-area occupancy.
``board_shape_residual(cgroup, rows_or_corners, board) -> dict``
    How far the triangulated board is from the real board (board units, scale free).
``ray_angles(cgroup, p3d, p2ds=None, min_deg=2.0) -> dict``
    Triangulation angles: per point maximum, per camera pair median; grazing pairs flagged.
``detect_moves(cgroup, p2ds, times, ...) -> MoveReport``
    Re-exported from :mod:`aniposelib.stations` (bumped / re-aimed cameras over time).

Points are ``C x N x 2`` (or ``C x N x J x 2``, flattened), NaN = missing, in group order.
Results hold numpy arrays and Python numbers (they are reports); torch input is accepted.
"""

import json
from dataclasses import dataclass, field, fields
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from . import geometry as G
from . import triangulation as T
from . import scale as S
from .stations import detect_moves, MoveReport  # re-export (camera move QA lives in stations)

__all__ = ['loo_reprojection', 'report', 'compare', 'sanity',
           'lens_validity', 'coverage', 'board_shape_residual', 'ray_angles',
           'LOOResult', 'CalibrationReport', 'CompareResult',
           'detect_moves', 'MoveReport']

_builtin = S._builtin


def _np(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _flat_points(p2ds, n_cams=None):
    pts = np.asarray(_np(p2ds), dtype='float64')
    if pts.ndim < 3 or pts.shape[-1] != 2:
        raise ValueError('points must be [C,N,2] or [C,N,J,2], got {}'.format(pts.shape))
    if n_cams is not None and pts.shape[0] != n_cams:
        raise ValueError('points have {} cameras on axis 0, group has {}'.format(pts.shape[0], n_cams))
    pts = pts.reshape(pts.shape[0], -1, 2).copy()
    pts[~np.isfinite(pts).all(-1)] = np.nan   # partial NaNs are missing
    return pts


def _q(e, q):
    return float(np.percentile(e, q)) if len(e) else None


def _stats(e, prefix=''):
    e = np.asarray(e, dtype='float64')
    e = e[np.isfinite(e)]
    return {prefix + 'median_px': _q(e, 50), prefix + 'p90_px': _q(e, 90),
            prefix + 'p95_px': _q(e, 95), prefix + 'n': int(len(e))}


def _thr_key(t):
    return 'agree@{:g}'.format(t)


# ------------------------------------------------------------------ leave one camera out

@dataclass
class LOOResult:
    """Leave-one-camera-out reprojection. ``errors [C,N]``: px error of each eligible observation
    against points triangulated from the other cameras (``inf`` = miss, NaN = not eligible);
    ``eligible [C,N]`` bool; ``per_camera[name]``: ``n_eligible``, ``n_supported``, ``coverage``,
    ``median_px``/``p10_px``/``p90_px`` (over supported), ``mean_px``, ``agree@T`` (fraction
    of *eligible* below T px); ``overall``: means over cameras, worst camera."""
    names: List[str]
    errors: Any
    eligible: Any
    per_camera: Dict[str, Dict[str, Any]]
    overall: Dict[str, Any]
    params: Dict[str, Any] = field(default_factory=dict)

    def medians(self):
        """Per-camera LOO medians ``[C]`` (NaN where unmeasured), in group order."""
        return np.array([np.nan if self.per_camera[n]['median_px'] is None else self.per_camera[n]['median_px']
                         for n in self.names])

    def to_dict(self, arrays=False):
        d = dict(names=list(self.names), per_camera=_builtin(self.per_camera),
                 overall=_builtin(self.overall), params=_builtin(self.params))
        if arrays:
            e = np.where(np.isinf(self.errors), -1.0, self.errors)   # JSON has no inf
            d['errors'] = np.where(np.isnan(e), None, e).tolist()
            d['eligible'] = np.asarray(self.eligible).tolist()
        return d


def _summarize_loo(errors, eligible, names, thresholds):
    with np.errstate(over='ignore'):
        return _summarize_loo_(errors, eligible, names, thresholds)


def _summarize_loo_(errors, eligible, names, thresholds):
    per = {}
    for c, name in enumerate(names):
        ok = eligible[c]
        sup = ok & np.isfinite(errors[c])
        e = errors[c, sup]
        n, ns = int(ok.sum()), int(sup.sum())
        d = dict(n_eligible=n, n_supported=ns, coverage=(ns / n) if n else None,
                 median_px=_q(e, 50), p10_px=_q(e, 10), p90_px=_q(e, 90),
                 mean_px=float(np.mean(e, dtype=np.float64)) if ns else None)
        for t in thresholds:
            d[_thr_key(t)] = float(np.count_nonzero(e < t) / n) if n else None
        per[name] = d

    def mean(key):
        v = [p[key] for p in per.values() if p[key] is not None]
        return float(np.mean(v)) if v else None
    meds = [(p['median_px'], n) for n, p in per.items() if p['median_px'] is not None]
    overall = dict(mean_coverage=mean('coverage'),
                   median_of_medians_px=float(np.median([m for m, _ in meds])) if meds else None,
                   worst_median_px=max(meds)[0] if meds else None,
                   worst_camera=max(meds)[1] if meds else None,
                   n_eligible=int(eligible.sum()),
                   unmeasured=[n for n, p in per.items() if p['n_supported'] == 0])
    for t in thresholds:
        overall['mean_' + _thr_key(t)] = mean(_thr_key(t))
    return per, overall


LOO_BATCH = 1 << 20      # held-out columns per batched LOO triangulation


def loo_errors(cgroup, p2ds, tau_tri=5.0, min_others=3, robust=True, chunk=None):
    """Raw LOO errors ``(errors [C,N], eligible [C,N])`` (see :func:`loo_reprojection`)."""
    P = G.pack(cgroup)
    C = P.n_cams
    pts = _flat_points(p2ds, C)
    N = pts.shape[1]
    finite = np.isfinite(pts).all(-1)
    eligible = finite & ((finite.sum(0)[None] - finite) >= min_others)
    errors = np.full((C, N), np.nan)
    errors[eligible] = np.inf
    cams, idxs = [], []
    for c in range(C):
        idx = np.flatnonzero(eligible[c])
        if len(idx):
            cams.append(c)
            idxs.append(idx)
    if not cams:
        return errors, eligible
    # batched triangulations: every target camera's eligible points, with its row removed, in
    # batches of at most LOO_BATCH columns (bounded memory for millions of points)
    off = np.cumsum([0] + [len(i) for i in idxs])
    allcols = torch.as_tensor(np.concatenate(idxs), device=P.device)
    heldcam = torch.as_tensor(np.repeat(cams, [len(i) for i in idxs]), device=P.device)
    pts_t = torch.as_tensor(pts, dtype=P.dtype, device=P.device)
    with torch.no_grad():
        # undistort every observation once, not once per held-out camera
        und = G._undistort_flat(P, pts_t, check_invertible=True) if robust else None
        Xs = []
        for b0 in range(0, int(off[-1]), LOO_BATCH):
            cols, hc = allcols[b0:b0 + LOO_BATCH], heldcam[b0:b0 + LOO_BATCH]
            ar = torch.arange(len(cols), device=P.device)
            held_t = pts_t[:, cols]
            held_t[hc, ar] = float('nan')
            if robust:
                held_u = und[:, cols]
                held_u[hc, ar] = float('nan')
                tr = T.triangulate_robust(P, held_t, tau=tau_tri, min_inliers=min_others,
                                          chunk=chunk, _und=held_u)
                Xb = tr.p3d
                support = (tr.errors < tau_tri).sum(0)   # recompute support after refinement (Emory)
                good = (support >= min_others) & torch.isfinite(Xb).all(-1)
            else:
                Xb = T.triangulate_linear(P, held_t, min_cams=2, chunk=chunk)
                good = torch.isfinite(Xb).all(-1)
            Xs.append(torch.where(good[:, None], Xb, torch.full_like(Xb, float('nan'))))
        X = torch.cat(Xs, 0)
        for k, c in enumerate(cams):
            sl = slice(int(off[k]), int(off[k + 1]))
            Pc = P.subset([c])
            Xc = X[sl]
            uv = torch.as_tensor(pts[c:c + 1, idxs[k]], dtype=P.dtype, device=P.device)
            e = G._reproj_err_flat(Pc, torch.nan_to_num(Xc), uv, Pc.extrinsics())[0]
            e = torch.where(torch.isfinite(Xc).all(-1), e, torch.full_like(e, float('inf')))
            e = torch.nan_to_num(e, nan=float('inf'))
            errors[c, idxs[k]] = e.cpu().numpy()
    return errors, eligible


def loo_reprojection(cgroup, p2ds, tau_tri=5.0, min_others=3, robust=True, thresholds=(5, 8, 10),
                     chunk=None):
    """Leave-one-camera-out reprojection with a fixed denominator (see the module docstring).

    ``robust=True``: points are triangulated from the other cameras by MSAC (inlier threshold
    ``tau_tri`` px) and supported when >= ``min_others`` other views agree, so a decoy or moved
    camera cannot spoil the healthy cameras' scores. ``robust=False``: plain linear
    triangulation from all other views (EvoVision ``leave_one_out_error``)."""
    names = cgroup.get_names()
    errors, eligible = loo_errors(cgroup, p2ds, tau_tri=tau_tri, min_others=min_others,
                                  robust=robust, chunk=chunk)
    per, overall = _summarize_loo(errors, eligible, names, thresholds)
    return LOOResult(names=names, errors=errors, eligible=eligible, per_camera=per, overall=overall,
                     params=dict(tau_tri=tau_tri, min_others=min_others, robust=robust,
                                 thresholds=list(thresholds)))


# ------------------------------------------------------------------ report

@dataclass
class CalibrationReport:
    names: List[str]
    per_camera: Dict[str, Dict[str, Any]]
    overall: Dict[str, Any]
    per_group: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    params: Dict[str, Any] = field(default_factory=dict)
    loo: Optional[LOOResult] = None

    def passes(self, max_median_px=12.0, max_loo_px=60.0, min_coverage=None):
        """True when the overall median <= ``max_median_px`` and every camera is measured with
        LOO median <= ``max_loo_px`` (and LOO coverage >= ``min_coverage`` if given)."""
        return not self.failures(max_median_px, max_loo_px, min_coverage)

    def failures(self, max_median_px=12.0, max_loo_px=60.0, min_coverage=None):
        """List of human-readable reasons why :meth:`passes` is False."""
        out = []
        med = self.overall.get('median_px')
        if med is None:
            out.append('no triangulated points')
        elif max_median_px is not None and med > max_median_px:
            out.append('median {:.2f} px > {}'.format(med, max_median_px))
        for n in self.names:
            p = self.per_camera[n]
            if p.get('n_used', 0) == 0:
                out.append('{}: unmeasured'.format(n))
            if self.loo is not None:
                lm = p.get('loo_median_px')
                if lm is None:
                    out.append('{}: no LOO measurement'.format(n))
                elif max_loo_px is not None and lm > max_loo_px:
                    out.append('{}: LOO median {:.2f} px > {}'.format(n, lm, max_loo_px))
                cov = p.get('loo_coverage')
                if min_coverage is not None and (cov is None or cov < min_coverage):
                    out.append('{}: LOO coverage {} < {}'.format(n, cov, min_coverage))
        return out

    def to_dict(self):
        return dict(names=list(self.names), per_camera=_builtin(self.per_camera),
                    overall=_builtin(self.overall), per_group=_builtin(self.per_group),
                    params=_builtin(self.params))

    def to_json(self, path=None, indent=1):
        """JSON string of :meth:`to_dict`; also written to ``path`` if given."""
        s = json.dumps(self.to_dict(), indent=indent, allow_nan=False, default=str) \
            if _all_finite(self.to_dict()) else json.dumps(_nan_to_none(self.to_dict()), indent=indent)
        if path is not None:
            with open(path, 'w') as f:
                f.write(s)
        return s

    def table(self):
        """Plain-text per-camera table (for logs / the CLI)."""
        cols = ['n_obs', 'coverage', 'median_px', 'p95_px', 'loo_median_px', 'loo_agree', 'flags']
        lines = ['{:>12s} '.format('camera') + ' '.join('{:>13s}'.format(c) for c in cols)]
        thr = self.params.get('agree_threshold')
        for n in self.names:
            p = dict(self.per_camera[n])
            p['loo_agree'] = p.get('loo_' + _thr_key(thr)) if thr is not None else None
            p['flags'] = ','.join(p.get('flags', [])) or '-'
            cells = []
            for c in cols:
                v = p.get(c)
                cells.append('{:>13s}'.format('-' if v is None else ('{:.3g}'.format(v) if isinstance(v, float) else str(v))))
            lines.append('{:>12s} '.format(str(n)) + ' '.join(cells))
        return '\n'.join(lines)


def _all_finite(d):
    try:
        json.dumps(d, allow_nan=False, default=str)
        return True
    except ValueError:
        return False


def _nan_to_none(x):
    if isinstance(x, float) and not np.isfinite(x):
        return None
    if isinstance(x, dict):
        return {k: _nan_to_none(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_nan_to_none(v) for v in x]
    return x


def report(cgroup, p2ds, robust=True, loo=True, min_views=3, fit_mask=None, groups=None,
           thresholds=(5, 8, 10), tau=10.0, tau_tri=5.0, min_others=None, max_loo_px=60.0,
           min_cam_points=10):
    """Per-camera calibration report (see the module docstring).

    All-observation statistics use points seen by >= ``min_views`` cameras (a point seen twice
    reprojects onto both views almost regardless of the calibration), triangulated robustly
    (MSAC, ``tau`` px) or linearly (``robust=False``). ``fit_mask [C,N]`` (bool, observations
    used to fit the calibration) only selects the extra fit-set / held-out columns: it never
    changes which points are triangulated or scored. ``groups [N]`` (labels per point, e.g. a
    keypoint or a time window) adds a ``per_group`` LOO breakdown. LOO uses ``tau_tri`` and
    ``min_others`` (default ``min_views``). Flags per camera: ``unmeasured`` (< ``min_cam_points``
    used observations), ``loo_disagrees`` (LOO median > ``max_loo_px``)."""
    names = cgroup.get_names()
    P = G.pack(cgroup)
    C = P.n_cams
    pts = _flat_points(p2ds, C)
    N = pts.shape[1]
    finite = np.isfinite(pts).all(-1)
    nv = finite.sum(0)
    multi = nv >= min_views
    if min_others is None:
        min_others = min_views
    errors = np.full((C, N), np.nan)
    inl = np.zeros((C, N), bool)
    if multi.any():
        sub = torch.as_tensor(pts[:, multi], dtype=P.dtype, device=P.device)
        with torch.no_grad():
            if robust:
                tr = T.triangulate_robust(P, sub, tau=tau, min_inliers=2)
                X, e = tr.p3d, tr.errors
            else:
                X = T.triangulate_linear(P, sub, min_cams=2)
                e = G.reproj_err(P, X, sub)
        e = _np(e)
        e = np.where(np.isfinite(_np(X)).all(-1)[None], e, np.nan)
        errors[:, multi] = e
        inl[:, multi] = np.isfinite(e) & (e < tau)
    used = np.isfinite(errors)
    if fit_mask is not None:
        fm = np.asarray(_np(fit_mask), dtype=bool).reshape(C, -1)
        if fm.shape[1] != N:
            raise ValueError('fit_mask must be [C,N] like the points')
    lres = loo_reprojection(cgroup, pts, tau_tri=tau_tri, min_others=min_others, robust=robust,
                            thresholds=thresholds) if loo else None
    per = {}
    for c, n in enumerate(names):
        d = dict(n_obs=int(finite[c].sum()), n_used=int(used[c].sum()))
        d['coverage'] = d['n_used'] / d['n_obs'] if d['n_obs'] else None
        st = _stats(errors[c])
        st.pop('n')
        d.update(st)
        d['mean_px'] = float(errors[c][used[c]].mean()) if d['n_used'] else None
        d['inlier_rate'] = float(inl[c].sum() / used[c].sum()) if used[c].any() else None
        for t in thresholds:
            d[_thr_key(t)] = float(np.count_nonzero(errors[c][used[c]] < t) / used[c].sum()) if used[c].any() else None
        if fit_mask is not None:
            st = _stats(errors[c][fm[c]], 'fit_')
            d.update({k: v for k, v in st.items()})
            st = _stats(errors[c][~fm[c] & finite[c]], 'heldout_')
            d.update({k: v for k, v in st.items()})
        if lres is not None:
            lp = lres.per_camera[n]
            d['loo_median_px'] = lp['median_px']
            d['loo_p90_px'] = lp['p90_px']
            d['loo_coverage'] = lp['coverage']
            d['loo_n_eligible'] = lp['n_eligible']
            for t in thresholds:
                d['loo_' + _thr_key(t)] = lp[_thr_key(t)]
        flags = []
        if d['n_used'] < min_cam_points:
            flags.append('unmeasured')
        if lres is not None and d.get('loo_median_px') is not None and max_loo_px is not None \
                and d['loo_median_px'] > max_loo_px:
            flags.append('loo_disagrees')
        d['flags'] = flags
        per[n] = d
    allu = errors[used]
    overall = dict(n_points=int(N), n_multiview_points=int(multi.sum()), n_cams=C,
                   n_obs=int(finite.sum()), n_used=int(used.sum()))
    overall.update({k: v for k, v in _stats(allu).items() if k != 'n'})
    overall['mean_px'] = float(allu.mean()) if len(allu) else None
    overall['inlier_rate'] = float(inl.sum() / used.sum()) if used.any() else None
    meds = [(p['median_px'], n) for n, p in per.items() if p['median_px'] is not None]
    overall['worst_camera_median_px'] = max(meds)[0] if meds else None
    if fit_mask is not None:
        overall.update(_stats(errors[fm & used], 'fit_'))
        overall.update(_stats(errors[~fm & used], 'heldout_'))
    if lres is not None:
        for k, v in lres.overall.items():
            overall['loo_' + k] = v
    overall['flags'] = sorted({f for p in per.values() for f in p['flags']})
    per_group = {}
    if groups is not None:
        g = np.asarray(_np(groups)).reshape(-1)
        if g.shape[0] != N:
            # allow per-joint labels for [C,N,J,2] input
            raw = np.asarray(_np(p2ds))
            if raw.ndim == 4 and g.shape[0] == raw.shape[2]:
                g = np.tile(g, raw.shape[1])
            else:
                raise ValueError('groups must have one label per point')
        for lab in np.unique(g):
            m = g == lab
            gd = dict(n_points=int(m.sum()), **{k: v for k, v in _stats(errors[:, m][used[:, m]]).items() if k != 'n'})
            if lres is not None:
                el = lres.eligible[:, m]
                le = lres.errors[:, m]
                ne = int(el.sum())
                sup = el & np.isfinite(le)
                gd['loo_median_px'] = _q(le[sup], 50)
                gd['loo_coverage'] = float(sup.sum() / ne) if ne else None
                for t in thresholds:
                    gd['loo_' + _thr_key(t)] = float(np.count_nonzero(le[sup] < t) / ne) if ne else None
            per_group[str(_builtin(lab))] = gd
    params = dict(robust=robust, loo=loo, min_views=min_views, thresholds=list(thresholds), tau=tau,
                  tau_tri=tau_tri, min_others=min_others,
                  agree_threshold=8 if 8 in thresholds else thresholds[len(thresholds) // 2])
    return CalibrationReport(names=names, per_camera=per, overall=overall, per_group=per_group,
                             params=params, loo=lres)


# ------------------------------------------------------------------ compare two calibrations

@dataclass
class CompareResult:
    """``b`` relative to ``a`` after aligning ``a``'s centres onto ``b``'s (trimmed Umeyama):
    ``scale``, ``rotation_deg`` of the similarity, per-camera ``center_residual`` (b units) and
    ``center_residual_frac`` (/ rig scale of b), ``rotation_delta_deg`` (camera orientation
    difference after alignment), ``focal_delta_pct``, ``pp_delta_px``; ``errors`` (with p2ds):
    per-camera LOO medians of both and their paired difference."""
    names: List[str]
    scale: float
    rotation_deg: float
    R: Any
    t: Any
    per_camera: Dict[str, Dict[str, Any]]
    overall: Dict[str, Any]
    gauge_excluded: List[str] = field(default_factory=list)
    errors: Optional[Dict[str, Any]] = None

    def to_dict(self):
        return {f.name: _builtin(getattr(self, f.name)) for f in fields(self)}


def _rot_deg(R):
    return float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))))


def compare(cg_a, cg_b, p2ds=None, trim=2.5, min_keep=4, with_scale=True, **loo_kw):
    """Compare two calibrations of the same cameras (matched by name; see :class:`CompareResult`).

    ``p2ds`` (in ``cg_a``'s camera order) adds LOO reprojection of both groups on the same
    observations (``loo_kw`` go to :func:`loo_reprojection`) and their paired difference."""
    na, nb = cg_a.get_names(), cg_b.get_names()
    names = [n for n in na if n in nb]
    if len(names) < 3:
        raise ValueError('need >= 3 shared cameras, got {}'.format(len(names)))
    ia = [na.index(n) for n in names]
    ib = [nb.index(n) for n in names]
    Xa, Xb = S.camera_centers(cg_a)[ia], S.camera_centers(cg_b)[ib]
    s, R, t, r, use = S.fit_similarity(Xa, Xb, trim=trim, min_keep=min_keep, with_scale=with_scale)
    rig = S._rig_scale(Xb)
    Pa, Pb = G.pack(cg_a), G.pack(cg_b)
    Ra, Rb = _np(Pa.R)[ia], _np(Pb.R)[ib]
    Ka, Kb = _np(Pa.K)[ia], _np(Pb.K)[ib]
    per = {}
    for k, n in enumerate(names):
        # camera a after alignment: Rc' = Ra R^T
        dR = Rb[k] @ (Ra[k] @ R.T).T
        per[n] = dict(center_residual=float(r[k]), center_residual_frac=float(r[k] / rig) if rig > 0 else None,
                      rotation_delta_deg=_rot_deg(dR),
                      focal_delta_pct=float(100 * ((Kb[k][0, 0] + Kb[k][1, 1]) / (Ka[k][0, 0] + Ka[k][1, 1]) - 1)),
                      pp_delta_px=float(np.hypot(Kb[k][0, 2] - Ka[k][0, 2], Kb[k][1, 2] - Ka[k][1, 2])),
                      in_gauge=bool(use[k]))
    vals = lambda key: [p[key] for p in per.values()]
    overall = dict(n_shared=len(names), rig_scale=rig,
                   center_residual_median=float(np.median(r)), center_residual_max=float(np.max(r)),
                   center_residual_max_frac=float(np.max(r) / rig) if rig > 0 else None,
                   rotation_delta_max_deg=float(max(vals('rotation_delta_deg'))),
                   focal_delta_max_pct=float(max(np.abs(vals('focal_delta_pct')))),
                   worst_camera=names[int(np.argmax(r))])
    errors = None
    if p2ds is not None:
        pts = _flat_points(p2ds, len(na))
        sub_b = cg_b if ib == list(range(len(nb))) and len(nb) == len(names) else cg_b.subset_cameras(ib)
        sub_a = cg_a if ia == list(range(len(na))) and len(na) == len(names) else cg_a.subset_cameras(ia)
        la = loo_reprojection(sub_a, pts[ia], **loo_kw)
        lb = loo_reprojection(sub_b, pts[ia], **loo_kw)
        pe = {}
        for n in names:
            a, b = la.per_camera[n]['median_px'], lb.per_camera[n]['median_px']
            pe[n] = dict(loo_median_a=a, loo_median_b=b,
                         delta=(b - a) if a is not None and b is not None else None)
        ca = np.isfinite(la.errors) & np.isfinite(lb.errors)
        errors = dict(per_camera=pe, n_common=int(ca.sum()),
                      paired_median_delta=float(np.median(lb.errors[ca] - la.errors[ca])) if ca.any() else None,
                      loo_a=la.overall, loo_b=lb.overall)
    return CompareResult(names=names, scale=float(s), rotation_deg=_rot_deg(R), R=R, t=t,
                         per_camera=per, overall=overall,
                         gauge_excluded=[n for n, u in zip(names, use) if not u], errors=errors)


# ------------------------------------------------------------------ intrinsics sanity

def _radial_slope(P, rmax, n=512):
    """Minimum slope of the radial distortion map over [0, rmax] (normalised radius)."""
    d = _np(P.dist[0])
    r = np.linspace(0, rmax, n)
    if int(P.model[0]) == G.MODELS['kb']:
        th = np.arctan(r)
        k1, k2, k3, k4 = d[:4]
        f = th * (1 + k1 * th ** 2 + k2 * th ** 4 + k3 * th ** 6 + k4 * th ** 8)
        x = th
    else:
        k1, k2, p1, p2, k3, k4, k5, k6 = d
        r2 = r * r
        f = r * (1 + k1 * r2 + k2 * r2 ** 2 + k3 * r2 ** 3) / (1 + k4 * r2 + k5 * r2 ** 2 + k6 * r2 ** 3)
        x = r
    with np.errstate(invalid='ignore', divide='ignore'):
        sl = np.gradient(f, x)
    return float(np.min(sl)) if np.all(np.isfinite(sl)) else float('nan')


def lens_validity(camera, grid=(33, 19), max_roundtrip_px=0.1, min_slope=0.05, size=None,
                  max_nonfinite_frac=0.0):
    """Is the lens model usable over the whole image? (Emory ``calibrate_v2.lens_validity``)

    A grid of pixels over the image is undistorted and re-projected (round trip, px), and the
    radial map ``r -> r_d`` (``theta -> theta_d`` for fisheye) must be monotone over the
    covered radius (minimum slope > ``min_slope``). Grid pixels the model cannot undistort
    (e.g. fisheye image corners beyond the polynomial's range) are counted in ``n_nonfinite``;
    more than ``max_nonfinite_frac`` of them makes the lens invalid (raise it for a fisheye
    whose image circle does not fill the frame). Returns a dict: ``valid``,
    ``roundtrip_max_px``, ``radial_min_slope``, ``n_nonfinite``, ``max_radius``."""
    P = G.pack([camera])
    if size is None:
        size = camera.get_size()
    if size is None:
        K = _np(P.K[0])
        size = (2 * K[0, 2], 2 * K[1, 2])
    w, h = [float(v) for v in size]
    g = np.stack(np.meshgrid(np.linspace(0, w - 1, grid[0]), np.linspace(0, h - 1, grid[1])), -1).reshape(1, -1, 2)
    gt = torch.as_tensor(g, dtype=P.dtype, device=P.device)
    with torch.no_grad():
        xy = G._undistort_flat(P, gt)
        uv = G.distort(P, xy)
    xy, uv = _np(xy)[0], _np(uv)[0]
    err = np.linalg.norm(uv - g[0], axis=-1)
    nonfinite = int((~np.isfinite(err)).sum())
    rr = np.hypot(xy[:, 0], xy[:, 1])
    rmax = float(np.nanmax(rr)) if np.isfinite(rr).any() else 0.0
    slope = _radial_slope(P, rmax) if rmax > 0 else 1.0
    finite = nonfinite <= max_nonfinite_frac * err.size and np.isfinite(slope)
    rt = float(np.nanmax(err)) if np.isfinite(err).any() else None
    return dict(valid=bool(finite and rt is not None and rt < max_roundtrip_px and slope > min_slope),
                roundtrip_max_px=rt, radial_min_slope=slope if np.isfinite(slope) else None,
                n_nonfinite=nonfinite, max_radius=rmax)


def sanity(cgroup, f_range=None, pp_frac=0.2, k_max=None, aspect_max=0.05, lens=True):
    """Plausibility checks of every camera's intrinsics. Returns ``{name: {...}, 'ok': bool}``.

    ``f_range``: (min, max) focal length in px (default: 0.1-5 x the image width when the
    size is known, else unchecked); ``pp_frac``: principal
    point within this fraction of the size from the image centre; ``k_max``: bound on
    ``|k|`` of every distortion coefficient; ``aspect_max``: ``|fy/fx - 1|``; ``lens``: run
    :func:`lens_validity`. Each camera entry has ``issues`` (list of strings) and ``ok``."""
    out, all_ok = {}, True
    for cam in cgroup.cameras:
        K = _np(cam.matrix).astype('float64')
        d = _np(cam.dist).astype('float64').ravel()
        size = cam.get_size()
        fx, fy = K[0, 0], K[1, 1]
        issues = []
        rng = f_range
        if rng is None and size is not None:
            rng = (0.1 * size[0], 5.0 * size[0])
        if rng is not None and not (rng[0] <= fx <= rng[1] and rng[0] <= fy <= rng[1]):
            issues.append('focal ({:.1f}, {:.1f}) outside [{:.1f}, {:.1f}]'.format(fx, fy, rng[0], rng[1]))
        if aspect_max is not None and fx > 0 and abs(fy / fx - 1) > aspect_max:
            issues.append('aspect fy/fx = {:.3f}'.format(fy / fx))
        if size is not None and pp_frac is not None:
            dx = abs(K[0, 2] - size[0] / 2) / size[0]
            dy = abs(K[1, 2] - size[1] / 2) / size[1]
            if dx > pp_frac or dy > pp_frac:
                issues.append('principal point off-centre by ({:.2f}, {:.2f}) of the size'.format(dx, dy))
        if k_max is not None and np.any(np.abs(d) > k_max):
            issues.append('distortion |k| > {}'.format(k_max))
        if not np.all(np.isfinite(K)) or not np.all(np.isfinite(d)):
            issues.append('non-finite intrinsics')
        entry = dict(fx=float(fx), fy=float(fy), cx=float(K[0, 2]), cy=float(K[1, 2]), dist=d.tolist())
        if lens:
            lv = lens_validity(cam)
            entry['lens'] = lv
            if not lv['valid']:
                issues.append('lens model not invertible/monotone over the image')
        entry['issues'] = issues
        entry['ok'] = not issues
        all_ok &= not issues
        out[cam.get_name()] = entry
    out['ok'] = bool(all_ok)
    return out


# ------------------------------------------------------------------ coverage

def coverage(p2ds, min_cams=3, cgroup=None, grid=(4, 4), names=None):
    """Observation coverage. Returns a dict with ``per_camera[name]``: ``n_obs``,
    ``n_multiview`` (observations of points seen by >= ``min_cams`` cameras), and with
    ``cgroup`` (image sizes) ``area`` = fraction of the ``grid`` cells of the image holding a
    multi-view observation; ``views_hist`` (number of points per view count) and
    ``n_multiview_points``."""
    pts = _flat_points(p2ds)
    C = pts.shape[0]
    if names is None:
        names = cgroup.get_names() if cgroup is not None else ['cam_{}'.format(i) for i in range(C)]
    fin = np.isfinite(pts).all(-1)
    nv = fin.sum(0)
    multi = nv >= min(min_cams, C)
    per = {}
    for c, n in enumerate(names):
        m = fin[c] & multi
        d = dict(n_obs=int(fin[c].sum()), n_multiview=int(m.sum()))
        if cgroup is not None and cgroup.cameras[c].get_size() is not None:
            w, h = cgroup.cameras[c].get_size()
            xy = pts[c, m]
            gx = np.clip((xy[:, 0] / w * grid[0]).astype(int), 0, grid[0] - 1)
            gy = np.clip((xy[:, 1] / h * grid[1]).astype(int), 0, grid[1] - 1)
            d['area'] = float(len(set(zip(gx.tolist(), gy.tolist()))) / (grid[0] * grid[1]))
        per[n] = d
    return dict(per_camera=per, n_points=int(pts.shape[1]), n_multiview_points=int(multi.sum()),
                views_hist=np.bincount(nv, minlength=C + 1).tolist(), min_cams=min_cams)


# ------------------------------------------------------------------ board shape

def board_shape_residual(cgroup, rows_or_corners, board, min_cams=2, min_corners=4, robust=False):
    """Shape agreement of the triangulated board with the real one (EvoVision
    ``board_fit_residual``): per frame, the triangulated corners are similarity-fitted (scale
    free, robust) onto the board's object points; the median residual is in **board units**
    (independent of the rig's gauge). A correct rig gives ~ the triangulation noise; a warped
    rig gives a large residual. Returns ``median``, ``p90``, ``per_frame``, ``frames``, ``n_frames``."""
    objp = S._board_object_points(board)
    corners, keys = S.board_points(cgroup, rows_or_corners, board)
    seen = np.isfinite(corners).all(-1).sum(0)
    usable = (seen >= min_cams).sum(1) >= max(min_corners, 3)
    per, frames = [], []
    if usable.any():
        X = S._triangulate_board(cgroup, corners[:, usable], min_cams=min_cams, robust=robust)
        for k, f in enumerate(np.flatnonzero(usable)):
            r = S._board_fit_residual(X[k], objp, min_corners)
            if np.isfinite(r):
                per.append(r)
                frames.append(_builtin(keys[f]))
    per = np.asarray(per)
    return dict(median=float(np.median(per)) if len(per) else None,
                p90=float(np.percentile(per, 90)) if len(per) else None,
                per_frame=per.tolist(), frames=frames, n_frames=int(len(per)),
                units=getattr(board, 'units', None))


# ------------------------------------------------------------------ ray angles

def ray_angles(cgroup, p3d, p2ds=None, min_deg=2.0, in_image=True):
    """Triangulation angles between camera rays at 3D points ``p3d [N,3]`` (or ``[N,J,3]``).

    A camera observes a point where ``p2ds`` is finite (if given), otherwise where the point is
    in front of it (and inside the image when the size is known and ``in_image``). Returns
    ``max_angle_deg [N]`` (largest pairwise angle among the observing cameras; NaN with < 2),
    ``pair_median_deg [C,C]`` (median over co-observed points), ``grazing_pairs`` (pairs with
    median < ``min_deg``) and ``frac_points_below`` (fraction of points whose max angle <
    ``min_deg``)."""
    P = G.pack(cgroup)
    C = P.n_cams
    X = np.asarray(_np(p3d), dtype='float64').reshape(-1, 3)
    N = len(X)
    centers = _np(P.centers())
    ok3 = np.isfinite(X).all(-1)
    Xt = torch.as_tensor(np.nan_to_num(X), dtype=P.dtype, device=P.device)
    with torch.no_grad():
        uv, z = G.project(P, Xt)
    uv, z = _np(uv), _np(z)
    if p2ds is not None:
        vis = np.isfinite(_flat_points(p2ds, C)).all(-1)
        if vis.shape[1] != N:
            raise ValueError('p2ds and p3d have different numbers of points')
    else:
        vis = z > 0
        if in_image:
            size = _np(P.size)
            for c in range(C):
                if np.all(np.isfinite(size[c])):
                    vis[c] &= (uv[c, :, 0] >= 0) & (uv[c, :, 0] < size[c, 0]) & \
                              (uv[c, :, 1] >= 0) & (uv[c, :, 1] < size[c, 1])
    vis &= ok3[None]
    rays = X[None] - centers[:, None]
    rays /= np.linalg.norm(rays, axis=-1, keepdims=True)
    maxang = np.full(N, np.nan)
    pair = np.full((C, C), np.nan)
    for a in range(C):
        for b in range(a + 1, C):
            m = vis[a] & vis[b]
            if not m.any():
                continue
            ang = np.degrees(np.arccos(np.clip((rays[a, m] * rays[b, m]).sum(-1), -1, 1)))
            pair[a, b] = pair[b, a] = np.median(ang)
            cur = maxang[m]
            maxang[m] = np.where(np.isnan(cur), ang, np.maximum(cur, ang))
    names = cgroup.get_names()
    grazing = [(names[a], names[b], float(pair[a, b])) for a in range(C) for b in range(a + 1, C)
               if np.isfinite(pair[a, b]) and pair[a, b] < min_deg]
    fin = np.isfinite(maxang)
    return dict(max_angle_deg=maxang, pair_median_deg=pair, grazing_pairs=grazing,
                frac_points_below=float((maxang[fin] < min_deg).mean()) if fin.any() else None,
                median_max_angle_deg=float(np.median(maxang[fin])) if fin.any() else None)


