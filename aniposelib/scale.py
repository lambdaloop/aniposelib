"""Global scale, gauge and world frame of a :class:`~aniposelib.cameras.CameraGroup` (1.0, plan 01 C5).

A self-calibrated rig is only defined up to a similarity (scale, rotation, translation). This
module fixes that gauge, from a calibration board, known lengths (bones, rulers), a reference
calibration, or the ground plane, and records where the scale came from.

API
---
``rescale(cgroup, s, units=None, source=None, quality=None)``
    World units *= ``s`` (every ``tvec *= s``), in place; returns ``cgroup``.
``apply_similarity(cgroup, s=1, R=None, t=None)``
    Re-express the rig in the world frame ``X' = s R X + t`` (in place).
``anchor_to(cgroup, camera)``
    Make ``camera`` (index or name) the world frame (identity pose), in place.
``align_to(cgroup, target, by='centers', trim=2.5, min_keep=4, with_scale=True) -> AlignResult``
    Trimmed Umeyama fit of the camera centres onto ``target`` (a ``CameraGroup``, matched by
    camera name, or ``[C,3]`` centres), applied in place. Cameras whose residual exceeds
    ``trim`` x the median residual are dropped from the fit (never below ``min_keep``).
``from_board(cgroup, rows_or_corners, board, ...) -> ScaleResult``
    Metric scale from triangulated board corners (per-frame median of known / measured
    corner-pair distances, median over frames), with frame MAD, p10-p90 spread, pair MAD and
    the board shape residual.
``from_lengths(cgroup, p2ds, pairs, ref_lengths, tau=5, min_inliers=3) -> ScaleResult``
    Scale from known lengths between keypoints (bones, rulers): robust triangulation, median
    length per pair, median of the ratios.
``from_reference(cgroup, ref, ...) -> ScaleResult``
    Scale of the (trimmed) similarity mapping the camera centres onto a reference calibration.
``align_world(cgroup, ground_points=None, up='z', origin='centroid') -> AlignResult``
    Rotate (and translate) the world so the ground plane is normal to ``up``.

``ScaleResult`` unpacks as ``s, result = from_board(...)``; ``result.apply(cgroup)`` rescales
the group and writes the provenance. None of the ``from_*`` functions modifies the group.

Provenance (``cgroup.metadata``): ``units`` (e.g. ``'m'``, ``'mm'``), ``scale_source``
(``'charuco'``, ``'board'``, ``'lengths'``, ``'reference'``, ``'manual'``), ``scale_value`` (the
factor applied), ``scale_quality`` (a dict of the main quality numbers). Only TOML-friendly
values are written.

Helpers: ``camera_centers(cgroup)``, ``umeyama(src, dst, with_scale=True)``,
``fit_similarity(src, dst, trim=2.5, min_keep=4, with_scale=True, inlier_frac=None)``.
"""

from dataclasses import dataclass, field, fields
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from . import geometry as G
from . import triangulation as T

__all__ = ['rescale', 'apply_similarity', 'anchor_to', 'align_to', 'from_board', 'from_lengths',
           'from_reference', 'align_world', 'ScaleResult', 'AlignResult', 'camera_centers',
           'umeyama', 'fit_similarity', 'board_points']


# ------------------------------------------------------------------ small helpers

def _builtin(x):
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, dict):
        return {str(k): _builtin(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_builtin(v) for v in x]
    return x


def _toml_clean(d):
    """Drop None / non-finite values (TOML has no null) from a flat-ish dict."""
    out = {}
    for k, v in d.items():
        v = _builtin(v)
        if v is None:
            continue
        if isinstance(v, float) and not np.isfinite(v):
            continue
        if isinstance(v, dict):
            v = _toml_clean(v)
        out[k] = v
    return out


def _rotmat(rvec):
    import cv2
    return cv2.Rodrigues(np.asarray(rvec, dtype='float64').reshape(3))[0]


def _rvec(R):
    import cv2
    return cv2.Rodrigues(np.asarray(R, dtype='float64'))[0].ravel()


def _np(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def camera_centers(cgroup):
    """Camera centres ``[C,3]`` (numpy) in world coordinates."""
    return _np(G.pack(cgroup).centers()).astype('float64')


def _poses(cgroup):
    return ([_rotmat(_np(c.rvec)) for c in cgroup.cameras],
            [_np(c.tvec).astype('float64').reshape(3) for c in cgroup.cameras])


def _set_pose(cam, R, t):
    cam.set_rotation(_rvec(R))
    cam.set_translation(np.asarray(t, dtype='float64'))


def umeyama(src, dst, with_scale=True):
    """Similarity ``(s, R, t)`` minimising ``sum |s R src + t - dst|^2`` (rows are points)."""
    src = np.asarray(src, dtype='float64')
    dst = np.asarray(dst, dtype='float64')
    mu_s, mu_d = src.mean(0), dst.mean(0)
    xs, xd = src - mu_s, dst - mu_d
    U, S, Vt = np.linalg.svd(xd.T @ xs / len(src))
    D = np.eye(3)
    D[2, 2] = np.sign(np.linalg.det(U @ Vt)) or 1.0
    R = U @ D @ Vt
    var = (xs ** 2).sum(1).mean()
    s = float((S * np.diag(D)).sum() / var) if (with_scale and var > 0) else 1.0
    return s, R, mu_d - s * R @ mu_s


def fit_similarity(src, dst, trim=2.5, min_keep=4, with_scale=True, inlier_frac=None):
    """Robust similarity ``dst ~ s R src + t``. Returns ``(s, R, t, residuals [N], used [N])``.

    ``trim`` (Emory ``align_to``): repeatedly refit on the points whose residual is at most
    ``trim`` x the median residual of the current fit set, until stable; ``trim=None`` is the
    plain fit. ``inlier_frac`` (EvoVision ``fit_similarity_robust``) instead refits 5 times on
    the best ``inlier_frac`` quantile. Never fits on fewer than ``min(min_keep, N)`` points.
    Non-finite rows are ignored (residual NaN, ``used`` False)."""
    src = np.asarray(src, dtype='float64')
    dst = np.asarray(dst, dtype='float64')
    ok = np.isfinite(src).all(1) & np.isfinite(dst).all(1)
    if ok.sum() < 3:
        raise ValueError('need >= 3 finite correspondences for a similarity fit, got {}'.format(int(ok.sum())))
    min_keep = min(int(min_keep), int(ok.sum()))
    use = ok.copy()
    # residuals below this floor are numerical noise, never a reason to trim a camera
    floor = 1e-9 * max(_rig_scale(dst[ok]), 1e-300)
    s, R, t = umeyama(src[use], dst[use], with_scale)
    for _ in range(len(src) if inlier_frac is None else 5):
        r = np.linalg.norm(s * src @ R.T + t - dst, axis=1)
        if inlier_frac is not None:
            new = ok & (r <= max(np.quantile(r[ok], inlier_frac), floor))
        elif trim:
            new = ok & (r <= max(trim * np.median(r[use]), floor))
        else:
            break
        if new.sum() < max(min_keep, 3) or (new == use).all():
            break
        use = new
        s, R, t = umeyama(src[use], dst[use], with_scale)
    r = np.linalg.norm(s * src @ R.T + t - dst, axis=1)
    r[~ok] = np.nan
    return s, R, t, r, use


def _rig_scale(centers):
    c = centers[np.isfinite(centers).all(1)]
    return float(np.median(np.linalg.norm(c - c.mean(0), axis=1))) if len(c) else float('nan')


# ------------------------------------------------------------------ result objects

@dataclass
class ScaleResult:
    """Outcome of a ``from_*`` scale estimate. ``scale`` multiplies the group's world units
    (``rescale(cgroup, scale)``); it is None when the estimate failed."""
    scale: Optional[float]
    accepted: bool
    source: str
    reason: Optional[str] = None
    units: Optional[str] = None
    stats: Dict[str, Any] = field(default_factory=dict)
    flags: List[str] = field(default_factory=list)
    details: Dict[str, Any] = field(default_factory=dict)

    def __iter__(self):
        yield self.scale
        yield self

    def quality(self):
        """The compact dict written to ``metadata['scale_quality']``."""
        q = {k: v for k, v in self.stats.items() if isinstance(v, (int, float, np.generic))}
        if self.flags:
            q['flags'] = list(self.flags)
        return q

    def apply(self, cgroup, force=False):
        """Rescale ``cgroup`` in place by ``scale`` and write the provenance metadata.
        Raises ``ValueError`` if the estimate was not accepted (unless ``force=True``)."""
        if self.scale is None or (not self.accepted and not force):
            raise ValueError('scale estimate not accepted: {}'.format(self.reason))
        return rescale(cgroup, self.scale, units=self.units, source=self.source, quality=self.quality())

    def to_dict(self):
        return {f.name: _builtin(getattr(self, f.name)) for f in fields(self)}


@dataclass
class AlignResult:
    """A similarity ``X' = s R X + t`` applied to a group's world frame."""
    scale: float
    R: Any
    t: Any
    residuals: Dict[str, float] = field(default_factory=dict)
    gauge_cameras: List[str] = field(default_factory=list)
    gauge_excluded: List[str] = field(default_factory=list)
    stats: Dict[str, Any] = field(default_factory=dict)

    @property
    def matrix(self):
        """4x4 similarity matrix."""
        M = np.eye(4)
        M[:3, :3] = self.scale * np.asarray(self.R)
        M[:3, 3] = self.t
        return M

    def __iter__(self):
        yield self.matrix
        yield self

    def to_dict(self):
        return {f.name: _builtin(getattr(self, f.name)) for f in fields(self)}


# ------------------------------------------------------------------ gauge operations

def rescale(cgroup, s, units=None, source=None, quality=None):
    """Multiply the world units by ``s`` (every camera's ``tvec *= s``), in place. Returns ``cgroup``.

    Metadata: ``scale_value`` (cumulative factor of successive rescales), ``scale_source``
    (``source`` or ``'manual'``), ``units`` (if given) and ``scale_quality`` (if given)."""
    s = float(s)
    if not np.isfinite(s) or s <= 0:
        raise ValueError('scale must be a positive finite number, got {}'.format(s))
    with torch.no_grad():
        for cam in cgroup.cameras:
            cam.set_translation(cam.tvec.detach() * s)
    md = cgroup.metadata
    prev = md.get('scale_value', 1.0)
    try:
        prev = float(prev)
    except (TypeError, ValueError):
        prev = 1.0
    md['scale_value'] = prev * s
    md['scale_source'] = source or 'manual'
    if units is not None:
        md['units'] = str(units)
    if quality:
        md['scale_quality'] = _toml_clean(dict(quality))
    elif 'scale_quality' in md:
        del md['scale_quality']
    return cgroup


def apply_similarity(cgroup, s=1.0, R=None, t=None):
    """Re-express the rig in the world frame ``X' = s R X + t``, in place. Returns ``cgroup``.

    A camera ``x = Rc X + tc`` becomes ``Rc' = Rc R^T``, ``tc' = s tc - Rc' t`` (projections are
    unchanged). With ``s != 1`` the world units change; ``metadata['scale_value']`` is updated
    (``units`` is left alone)."""
    s = float(s)
    if not np.isfinite(s) or s <= 0:
        raise ValueError('scale must be a positive finite number, got {}'.format(s))
    R = np.eye(3) if R is None else np.asarray(_np(R), dtype='float64').reshape(3, 3)
    t = np.zeros(3) if t is None else np.asarray(_np(t), dtype='float64').reshape(3)
    Rs, ts = _poses(cgroup)
    with torch.no_grad():
        for cam, Rc, tc in zip(cgroup.cameras, Rs, ts):
            Rn = Rc @ R.T
            _set_pose(cam, Rn, s * tc - Rn @ t)
    if s != 1.0:
        md = cgroup.metadata
        try:
            md['scale_value'] = float(md.get('scale_value', 1.0)) * s
        except (TypeError, ValueError):
            md['scale_value'] = s
    return cgroup


def _cam_index(cgroup, camera):
    if isinstance(camera, (int, np.integer)):
        return int(camera)
    names = cgroup.get_names()
    if camera in names:
        return names.index(camera)
    raise ValueError('camera {!r} not in group {}'.format(camera, names))


def anchor_to(cgroup, camera=0):
    """Make ``camera`` (index or name) the world frame: its pose becomes the identity. In place."""
    i = _cam_index(cgroup, camera)
    Rs, ts = _poses(cgroup)
    return apply_similarity(cgroup, 1.0, Rs[i], ts[i])


def align_to(cgroup, target, by='centers', trim=2.5, min_keep=4, with_scale=True):
    """Similarity-align ``cgroup`` onto ``target`` by camera centres (trimmed Umeyama), in place.

    ``target``: a ``CameraGroup`` (cameras matched by name; at least 3 shared) or ``[C,3]``
    centres in ``cgroup`` order (NaN rows are ignored). An outlier camera (a moved or badly
    calibrated one) is excluded from the fit when its residual is above ``trim`` x the median,
    so it cannot rotate or rescale the whole rig. ``with_scale=False`` keeps the units.
    Returns an :class:`AlignResult` (``residuals`` per camera, in target units)."""
    if by != 'centers':
        raise ValueError("only by='centers' is supported, got {!r}".format(by))
    names = cgroup.get_names()
    X = camera_centers(cgroup)
    if hasattr(target, 'cameras'):
        tnames = target.get_names()
        Yt = camera_centers(target)
        Y = np.full_like(X, np.nan)
        for i, n in enumerate(names):
            if n in tnames:
                Y[i] = Yt[tnames.index(n)]
    else:
        Y = np.asarray(_np(target), dtype='float64').reshape(-1, 3)
        if len(Y) != len(X):
            raise ValueError('target has {} centres, group has {} cameras'.format(len(Y), len(X)))
    s, R, t, r, use = fit_similarity(X, Y, trim=trim, min_keep=min_keep, with_scale=with_scale)
    apply_similarity(cgroup, s, R, t)
    ok = np.isfinite(r)
    stats = dict(median_residual=float(np.median(r[use])), max_residual=float(np.nanmax(r)),
                 rig_scale=_rig_scale(Y), n_shared=int(ok.sum()))
    if hasattr(target, 'metadata') and with_scale and 'units' in target.metadata:
        cgroup.metadata['units'] = target.metadata['units']
    return AlignResult(scale=s, R=R, t=t,
                       residuals={n: float(v) for n, v, o in zip(names, r, ok) if o},
                       gauge_cameras=[n for n, u in zip(names, use) if u],
                       gauge_excluded=[n for n, u, o in zip(names, use, ok) if o and not u],
                       stats=stats)


# ------------------------------------------------------------------ board scale

def _board_object_points(board):
    objp = board.get_object_points()
    return np.asarray(objp, dtype='float64').reshape(-1, 3)


def board_points(cgroup, rows_or_corners, board):
    """Board detections as ``corners [C,F,K,2]`` (NaN = not detected) and frame keys ``[F]``.

    Accepts ``calibrate_rows``-style rows (one list per camera of dicts with ``framenum``,
    ``corners`` and ``ids``; frames are matched by ``framenum``), a ``[C,F,K,2]`` array, or a
    ``[C,F*K,2]`` array (as returned by ``boards.extract_points``)."""
    objp = _board_object_points(board)
    K = len(objp)
    C = len(cgroup.cameras)
    if isinstance(rows_or_corners, (list, tuple)) and len(rows_or_corners) and \
            isinstance(rows_or_corners[0], (list, tuple)) and \
            (len(rows_or_corners[0]) == 0 or isinstance(rows_or_corners[0][0], dict)):
        all_rows = rows_or_corners
        if len(all_rows) != C:
            raise ValueError('rows for {} cameras, group has {}'.format(len(all_rows), C))
        keys = sorted({r['framenum'] for rows in all_rows for r in rows}, key=lambda k: (str(type(k)), k))
        index = {k: i for i, k in enumerate(keys)}
        out = np.full((C, len(keys), K, 2), np.nan)
        for c, rows in enumerate(all_rows):
            for r in rows:
                if 'ids' in r and r['ids'] is not None and r.get('corners') is not None:
                    ids = np.asarray(r['ids']).reshape(-1).astype(int)
                    xy = np.asarray(r['corners'], dtype='float64').reshape(-1, 2)
                    good = (ids >= 0) & (ids < K)
                    out[c, index[r['framenum']], ids[good]] = xy[good]
                elif r.get('filled') is not None:
                    out[c, index[r['framenum']]] = np.asarray(r['filled'], dtype='float64').reshape(K, 2)
        return out, keys
    arr = np.asarray(_np(rows_or_corners), dtype='float64')
    if arr.ndim == 3:
        if arr.shape[1] % K:
            raise ValueError('corners [C,N,2] with N={} not a multiple of the board size {}'.format(arr.shape[1], K))
        arr = arr.reshape(arr.shape[0], -1, K, 2)
    if arr.ndim != 4 or arr.shape[0] != C or arr.shape[2] != K:
        raise ValueError('corners must be [C,F,K,2] with C={} and K={}, got {}'.format(C, K, arr.shape))
    return arr, list(range(arr.shape[1]))


def _triangulate_board(cgroup, corners, min_cams=2, robust=False, tau=10.0):
    C, F, K, _ = corners.shape
    P = G.pack(cgroup)
    flat = corners.reshape(C, F * K, 2)
    if robust:
        X = T.triangulate_robust(P, flat, tau=tau, min_inliers=min_cams).p3d
    else:
        X = T.triangulate_linear(P, flat, min_cams=min_cams)
    return np.asarray(X).reshape(F, K, 3)


def _board_fit_residual(p3d, objp, min_corners=4):
    """Median residual (board units) of a similarity fit of triangulated corners onto the board."""
    ok = np.isfinite(p3d).all(-1)
    if ok.sum() < max(min_corners, 3):
        return float('nan')
    try:
        _s, _R, _t, r, _u = fit_similarity(p3d[ok], objp[ok], trim=None, inlier_frac=0.8,
                                           min_keep=min_corners)
    except ValueError:
        return float('nan')
    return float(np.median(r))


def from_board(cgroup, rows_or_corners, board, min_frames=5, min_pairs=30, min_pair_dist=None,
               max_frame_mad=0.05, min_cams=2, min_corners=4, max_pair_mad=0.05,
               max_frame_spread=0.025, max_board_resid=None, robust=False, units=None):
    """Metric scale from board corners triangulated by ``cgroup`` (EvoVision ``scale_from_charuco``).

    Per frame with >= ``min_corners`` corners seen by >= ``min_cams`` cameras: the ratio of the
    known to the triangulated distance of every corner pair at least ``min_pair_dist`` apart
    (board units; None = all pairs), median per frame; ``scale`` is the median over frames
    (frames, not pairs, are the independent measurements).

    Accepted when there are >= ``min_frames`` frames, >= ``min_pairs`` pairs and the relative
    frame MAD is <= ``max_frame_mad``. Flags (never rejects): ``board_pair_mad`` (relative pair
    MAD > ``max_pair_mad``), ``board_scale_spread`` (p10-p90 of the frame scales /
    scale > ``max_frame_spread``), ``board_warp`` (median board shape residual >
    ``max_board_resid``, board units). The board shape residual (similarity fit of the
    triangulated corners onto the board, scale free) measures the rig's *shape*.
    ``units`` defaults to ``board.units`` when the board has that attribute."""
    if units is None:
        units = getattr(board, 'units', None)
    objp = _board_object_points(board)
    corners, keys = board_points(cgroup, rows_or_corners, board)
    C, F, K, _ = corners.shape
    true_d = np.linalg.norm(objp[:, None] - objp[None], axis=-1)
    iu = np.triu_indices(K, 1)
    td = true_d[iu]
    long_enough = td > (0 if min_pair_dist is None else float(min_pair_dist))
    long_enough &= td > 0
    seen = np.isfinite(corners).all(-1).sum(0)  # [F,K]
    usable = (seen >= min_cams).sum(1) >= max(min_corners, 2)
    res = ScaleResult(scale=None, accepted=False, source='charuco' if hasattr(board, 'marker_length') else 'board',
                      units=units)
    if not usable.any():
        res.reason = 'no board frames seen by >= {} cameras'.format(min_cams)
        return res
    p3d = np.full((F, K, 3), np.nan)
    p3d[usable] = _triangulate_board(cgroup, corners[:, usable], min_cams=min_cams, robust=robust)
    ratios, per_frame, resid, used = [], [], [], []
    for f in np.flatnonzero(usable):
        X = p3d[f]
        meas = np.linalg.norm(X[:, None] - X[None], axis=-1)[iu]
        ok = np.isfinite(meas) & (meas > 1e-12) & long_enough
        if ok.sum() < 3:
            continue
        r = td[ok] / meas[ok]
        ratios.append(r)
        per_frame.append(float(np.median(r)))
        resid.append(_board_fit_residual(X, objp, min_corners))
        used.append(keys[f])
    n_frames = len(per_frame)
    res.details = dict(frames=[_builtin(k) for k in used], per_frame_scale=per_frame,
                       per_frame_board_resid=resid)
    if n_frames < min_frames:
        res.reason = 'only {} usable board frames (need {})'.format(n_frames, min_frames)
        res.stats = dict(n_frames=n_frames)
        return res
    allr = np.concatenate(ratios)
    if len(allr) < min_pairs:
        res.reason = 'only {} corner pairs (need {})'.format(len(allr), min_pairs)
        res.stats = dict(n_frames=n_frames, n_pairs=int(len(allr)))
        return res
    pf = np.asarray(per_frame)
    s = float(np.median(pf))
    frame_mad = float(np.median(np.abs(pf - s))) / s
    pair_mad = float(np.median(np.abs(allr - s))) / s
    p10, p90 = float(np.percentile(pf, 10)), float(np.percentile(pf, 90))
    rr = np.asarray(resid)
    rr = rr[np.isfinite(rr)]
    res.scale = s
    res.stats = dict(scale=s, frame_scale_mad=frame_mad, frame_scale_spread=(p90 - p10) / s,
                     pair_scale_mad=pair_mad, n_frames=n_frames, n_pairs=int(len(allr)),
                     per_frame_scale_p10=p10, per_frame_scale_p90=p90,
                     board_resid=float(np.median(rr)) if len(rr) else float('nan'),
                     board_resid_p90=float(np.percentile(rr, 90)) if len(rr) else float('nan'),
                     n_board_fits=int(len(rr)))
    if frame_mad > max_frame_mad:
        res.reason = 'board sightings disagree on scale (frame MAD {:.3f} > {})'.format(frame_mad, max_frame_mad)
        return res
    res.accepted = True
    if max_pair_mad is not None and pair_mad > max_pair_mad:
        res.flags.append('board_pair_mad')
    if max_frame_spread is not None and res.stats['frame_scale_spread'] > max_frame_spread:
        res.flags.append('board_scale_spread')
    if max_board_resid is not None and len(rr) and res.stats['board_resid'] > max_board_resid:
        res.flags.append('board_warp')
    return res


# ------------------------------------------------------------------ known lengths

def _pair_index(pairs, joints):
    out = []
    for a, b in pairs:
        if not isinstance(a, (int, np.integer)):
            if joints is None:
                raise ValueError('pairs given by name need joints=')
            a = list(joints).index(a)
        if not isinstance(b, (int, np.integer)):
            if joints is None:
                raise ValueError('pairs given by name need joints=')
            b = list(joints).index(b)
        out.append((int(a), int(b)))
    return out


def from_lengths(cgroup, p2ds, pairs, ref_lengths, tau=5.0, min_inliers=3, joints=None,
                 min_samples=10, max_spread=0.1, units=None):
    """Scale from known lengths between keypoints (Emory ``anatomical_scale``).

    ``p2ds [C,N,J,2]``; ``pairs`` a list of ``(a, b)`` joint indices (or names with
    ``joints=``); ``ref_lengths`` one reference length per pair (list, or dict keyed by the
    pair). Points are triangulated robustly (MSAC, inlier threshold ``tau`` px) and kept only
    with >= ``min_inliers`` agreeing views. ``scale`` = median over pairs of
    ``ref / median measured length``. Accepted when every pair has >= ``min_samples``
    measurements; flag ``lengths_spread`` when (max - min ratio) / scale > ``max_spread``."""
    pts = np.asarray(_np(p2ds), dtype='float64')
    if pts.ndim != 4:
        raise ValueError('p2ds must be [C,N,J,2], got {}'.format(pts.shape))
    C, N, J, _ = pts.shape
    if isinstance(ref_lengths, dict):
        refs = [float(ref_lengths[p]) if p in ref_lengths else float(ref_lengths[tuple(p)]) for p in
                [tuple(p) for p in pairs]]
    else:
        refs = [float(v) for v in ref_lengths]
    if len(refs) != len(pairs):
        raise ValueError('{} pairs but {} reference lengths'.format(len(pairs), len(refs)))
    idx = _pair_index(pairs, joints)
    P = G.pack(cgroup)
    tr = T.triangulate_robust(P, pts.reshape(C, N * J, 2), tau=tau, min_inliers=2)
    X = np.asarray(tr.p3d).reshape(N, J, 3).copy()
    X[np.asarray(tr.n_inliers).reshape(N, J) < min_inliers] = np.nan
    per, ratios, ok_all = {}, [], True
    for (a, b), ref, p in zip(idx, refs, pairs):
        d = np.linalg.norm(X[:, a] - X[:, b], axis=-1)
        d = d[np.isfinite(d)]
        key = '{}-{}'.format(*p)
        if len(d) == 0:
            per[key] = dict(n=0, reference=ref)
            ok_all = False
            continue
        med = float(np.median(d))
        per[key] = dict(n=int(len(d)), median_length=med, reference=ref, ratio=ref / med,
                        cv=float(np.std(d) / np.mean(d)))
        ratios.append(ref / med)
        if len(d) < min_samples:
            ok_all = False
    res = ScaleResult(scale=None, accepted=False, source='lengths', units=units,
                      details=dict(per_pair=per))
    if not ratios:
        res.reason = 'no pair measured with >= {} agreeing views'.format(min_inliers)
        return res
    s = float(np.median(ratios))
    spread = (max(ratios) - min(ratios)) / s
    res.scale = s
    res.stats = dict(scale=s, ratio_min=float(min(ratios)), ratio_max=float(max(ratios)),
                     ratio_spread=spread, n_pairs=len(ratios),
                     n_samples=int(sum(v['n'] for v in per.values())))
    if not ok_all:
        res.reason = 'some pairs have fewer than {} measurements'.format(min_samples)
        return res
    res.accepted = True
    if max_spread is not None and spread > max_spread:
        res.flags.append('lengths_spread')
    return res


# ------------------------------------------------------------------ reference calibration

def from_reference(cgroup, ref, min_shared=3, trim=2.5, min_keep=3, max_residual=None, units=None):
    """Scale of the similarity mapping ``cgroup``'s camera centres onto ``ref``'s (by name).

    Assumes the cameras did not move between the two calibrations: the per-camera residual
    (``ref`` units) is reported, and flag ``scale_reference_mismatch`` is raised when the largest
    one exceeds ``max_residual`` (default 5 % of the reference rig scale). Outlier cameras are
    trimmed from the fit (see :func:`fit_similarity`)."""
    names, rnames = cgroup.get_names(), ref.get_names()
    shared = [n for n in names if n in rnames]
    if units is None:
        units = ref.metadata.get('units') if hasattr(ref, 'metadata') else None
    res = ScaleResult(scale=None, accepted=False, source='reference', units=units)
    if len(shared) < max(3, min_shared):
        res.reason = 'only {} shared cameras (need {})'.format(len(shared), max(3, min_shared))
        return res
    a, b = camera_centers(cgroup), camera_centers(ref)
    src = np.array([a[names.index(n)] for n in shared])
    dst = np.array([b[rnames.index(n)] for n in shared])
    s, R, t, r, use = fit_similarity(src, dst, trim=trim, min_keep=min_keep)
    rig = _rig_scale(dst)
    if max_residual is None:
        max_residual = 0.05 * rig
    res.scale = float(s)
    res.accepted = True
    res.stats = dict(scale=float(s), n_shared=len(shared), fit_residual_mean=float(np.mean(r)),
                     fit_residual_max=float(np.max(r)), rig_scale=rig)
    res.details = dict(residuals=dict(zip(shared, r.tolist())),
                       excluded=[n for n, u in zip(shared, use) if not u])
    if np.max(r) > max_residual:
        res.flags.append('scale_reference_mismatch')
    return res


# ------------------------------------------------------------------ world frame

_AXES = {'x': [1, 0, 0], 'y': [0, 1, 0], 'z': [0, 0, 1]}


def _axis(up):
    if isinstance(up, str):
        sign = -1.0 if up.startswith('-') else 1.0
        return sign * np.asarray(_AXES[up.lstrip('+-').lower()], dtype='float64')
    v = np.asarray(up, dtype='float64').reshape(3)
    return v / np.linalg.norm(v)


def _rot_between(a, b):
    """Smallest rotation taking unit vector a to unit vector b."""
    v = np.cross(a, b)
    c = float(np.dot(a, b))
    if np.linalg.norm(v) < 1e-12:
        if c > 0:
            return np.eye(3)
        p = np.cross(a, [1.0, 0, 0]) if abs(a[0]) < 0.9 else np.cross(a, [0, 1.0, 0])
        p /= np.linalg.norm(p)
        return 2 * np.outer(p, p) - np.eye(3)
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx * (1 / (1 + c))


def align_world(cgroup, ground_points=None, up='z', origin='centroid'):
    """Rotate the world so the ground plane's normal is ``up`` (``'z'``, ``'-y'``, or a vector), in place.

    ``ground_points [N,3]`` (world points on the floor; NaN rows ignored) are fitted with a plane
    (SVD). Without them the plane through the camera centres is used. The normal's sign is
    chosen so that the cameras are on the ``up`` side (above the floor; for the camera plane:
    cameras look down, i.e. against ``up``). ``origin``: ``'centroid'`` (the centroid of the ground
    points, or of the camera centres), ``'keep'`` (no
    translation), an int camera index (that camera's centre, on the plane) or a 3-vector
    (world point that becomes the origin). Returns an :class:`AlignResult` (``stats``:
    plane RMS residual, tilt in degrees)."""
    centers = camera_centers(cgroup)
    if ground_points is not None:
        g = np.asarray(_np(ground_points), dtype='float64').reshape(-1, 3)
        g = g[np.isfinite(g).all(1)]
        if len(g) < 3:
            raise ValueError('need >= 3 finite ground points')
    else:
        g = centers
    mu = g.mean(0)
    _, S, Vt = np.linalg.svd(g - mu)
    n = Vt[2]
    if ground_points is not None:
        if np.dot(centers.mean(0) - mu, n) < 0:
            n = -n
    else:
        axes = np.stack([_rotmat(_np(c.rvec))[2] for c in cgroup.cameras])  # optical axes in world
        if np.dot(axes.mean(0), n) > 0:
            n = -n
    target = _axis(up)
    R = _rot_between(n, target)
    if isinstance(origin, str) and origin == 'keep':
        o = None
    elif isinstance(origin, str) and origin == 'centroid':
        o = mu
    elif isinstance(origin, (int, np.integer)):
        c = centers[int(origin)]
        o = c - np.dot(c - mu, n) * n
    else:
        o = np.asarray(_np(origin), dtype='float64').reshape(3)
    t = np.zeros(3) if o is None else -R @ o
    apply_similarity(cgroup, 1.0, R, t)
    rms = float(np.sqrt(np.mean(((g - mu) @ n) ** 2)))
    tilt = float(np.degrees(np.arccos(np.clip(np.dot(n, target), -1, 1))))
    return AlignResult(scale=1.0, R=R, t=t, stats=dict(plane_rms=rms, rotation_deg=tilt,
                                                        n_points=int(len(g))))
