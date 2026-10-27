"""Data workflows (plan 05 W5-W7): syncing cameras, 2D -> 3D reconstruction and multi-animal
association. Composition only: the maths lives in the backend modules
(:mod:`~aniposelib.sync`, :mod:`~aniposelib.stba`, :mod:`~aniposelib.triangulation`,
:mod:`~aniposelib.matching`, ``CameraGroup.optim_points``).

API
---
``estimate_offsets(streams, cgroup=None, reference=None, drift=True, refine=False,
                   options=None, rng=0) -> OffsetEstimation``
    W5. Per-stream time offsets (module convention of :mod:`aniposelib.sync`:
    ``source_time = global_time + offset``) from every available signal per pair: audio
    (``audio_offset`` gcc, cross-checked by the spectrogram and onset methods), motion
    (``motion_offset`` on given signals or on ``motion_signal`` of 2D points) and epipolar
    consistency (``epipolar_offset``, when ``cgroup`` is given; its search is centred on the
    audio/motion estimate of the pair). The pair estimates are combined with
    ``solve_offset_graph`` (anchor = ``reference``, default the best-connected stream), then
    ``fit_drift`` per stream on the windowed audio estimates (``drift=True``). ``refine=True``
    runs ``stba.SpatiotemporalBA`` on the 2D points, with offset priors anchored to the graph
    estimates and a motion prior relative to the rig scale. Nothing is applied: use
    ``result.timemaps`` (``sync.TimeMap`` per stream).
``reconstruct_3d(cgroup, points, scores=None, bones=None, smooth=None, fill='interior',
               robust=True, options=None, device='auto') -> Reconstruction``
    W6. Robust triangulation (``triangulation.triangulate_robust``, MSAC, noise-derived
    ``tau``) of ``points C x N x J x 2`` (or ``C x N x 2``; NaN = missing). ``scores`` are soft
    weights (clipped to [0, 1]) in the hypothesis cost, the refit, the Gauss-Newton step and the
    ``optim_points`` reprojection term; there is no score threshold unless
    ``options={'min_score': s}``. With ``bones=[(a, b), ...]`` or ``smooth=`` the result is
    refined by ``CameraGroup.optim_points`` (unit-free, bone targets estimated robustly; long
    sequences in overlapping, blended chunks). ``fill='interior'`` (default) fills gaps between
    the first and last observation of each joint, never extrapolates; ``'none'`` leaves them NaN.
``associate_views(cgroup, detections, track_ids=None, min_views=2, options=None)
                  -> Association``
    W7. Who is who across cameras. ``detections``: per camera (list in group order or dict by
    camera name), per frame a ``(D, J, 2|3)`` array or a list of ``(J, 2|3)`` arrays (x, y[,
    score]). With ``track_ids`` (same structure, ``(D,)`` ids) tracklets are associated by
    ``matching.associate_tracklets``; without, each frame is associated by
    ``matching.associate_points`` on the detection centroids and the instances are linked
    over time into identities. The output is dense: ``points C x F x P x J x 2`` (NaN where
    unmatched), ready for :func:`reconstruct_3d` or ``calibrate_from_points``.

Results subclass :class:`~aniposelib.workflows._common.WorkflowResult` (``report``,
``verdict``, ``reasons``, ``settings``, ``provenance``, ``to_dict()``, ``to_json()``,
``summary()``).

Conventions
-----------
* numpy in -> numpy out. Times are seconds. ``rng=0`` by default.
* No unit-dependent knobs: pixel thresholds derive from the robust noise floor
  (:func:`~aniposelib.workflows._common.noise_floor`) through the *noise unit*
  ``u = noise_floor / 0.5 px`` (backend pixel defaults, tuned for 0.5 px noise, are multiplied
  by ``u``); first-pass gates are fractions of the image diagonal; 3D lengths are fractions of
  the rig scale (median camera distance to the camera centroid); offsets are compared with the
  frame period. Every derived value is in ``result.settings``.
* Verdicts follow the rules in each docstring; override with
  ``options={'verdict': {rule: (warn_at, fail_at) | None}}``. Workflows raise only on unusable
  inputs (wrong shapes, unknown cameras, unreadable files); bad results are ``'fail'``.
"""

import itertools
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from ._common import (MIN_NOISE_PX, NOISE_DIAG_FRAC, Rule, Timer, WorkflowResult, evaluate_rules,
                      image_diagonal, merge_options, noise_floor, provenance, save_arrays,
                      thread_guard, to_numpy, worst)

__all__ = ['estimate_offsets', 'OffsetEstimation', 'reconstruct_3d', 'Reconstruction',
           'associate_views', 'Association']


# ================================================================== shared helpers

def _noise_unit(nf):
    """Noise unit ``u``: backend pixel defaults (tuned for 0.5 px noise) are multiplied by it."""
    return max(float(nf), MIN_NOISE_PX) / MIN_NOISE_PX


def _device(device):
    if device in (None, 'auto'):
        return 'cuda' if torch.cuda.is_available() else 'cpu'
    return str(device)


def _rig_scale(cgroup):
    """Median distance of the camera centres to their centroid (world units)."""
    from .. import geometry as G
    c = to_numpy(G.pack(cgroup).centers()).astype(float)
    c = c[np.isfinite(c).all(1)]
    if len(c) < 2:
        return float('nan')
    return float(np.median(np.linalg.norm(c - c.mean(0), axis=1)))


def _group_on(cgroup, device):
    dev = _device(device)
    if str(cgroup.device) == dev or (dev == 'cuda' and str(cgroup.device).startswith('cuda')):
        return cgroup
    return cgroup.copy().to(dev)


def _verdict_opts(opts):
    v = opts.get('verdict') or {}
    return dict(v)


# ================================================================== W6 reconstruct_3d

@dataclass
class Reconstruction(WorkflowResult):
    """Result of :func:`reconstruct_3d`.

    ``points3d [N,J,3]`` (``[N,3]`` for ``C x N x 2`` input; NaN = not reconstructed),
    ``errors [C,N,J]`` reprojection error (px) of every observation against the final points,
    ``n_views [N,J]`` inlier views used (0 for filled cells), ``inliers [C,N,J]`` observations
    used, ``filled [N,J]`` cells filled (gap interpolation / optim_points) rather than
    triangulated. ``report``: ``per_camera`` usage, ``gaps``, ``bones`` (length CV), ``noise_px``,
    ``tau_px``, ``rules``."""
    points3d: Any = None
    errors: Any = None
    n_views: Any = None
    inliers: Any = None
    filled: Any = None

    def save(self, out):
        save_arrays(self, out, {'points3d': self.points3d, 'errors': self.errors,
                                'n_views': self.n_views, 'inliers': self.inliers})


_W6_DEFAULTS = {
    'tau_px': None,            # fixed MSAC threshold (px); None = 20 x noise floor (10 px at 0.5 px)
    'tau_noise_mult': 20.0,
    'first_tau_diag_frac': 0.01,   # first robust pass: tau = max(10 px, this x image diagonal)
    'min_score': None,         # hard score cut (observations below are removed); None = soft only
    'min_views': 2,
    'bone_weight': 2.0,        # optim_points scale_length (pixel-equivalent, see optim_points)
    'n_iters': 300,            # optim_points Adam iterations per chunk
    'max_lr': 0.1,             # optim_points step (pixels)
    'chunk_frames': 2000,      # optim_points chunk length (frames) for long sequences
    'overlap_frames': 100,     # overlap between chunks (linear blending)
    'verdict': None,
}


def _interior_fill(X):
    """Linear interpolation of interior gaps per (joint, coordinate) column of X [N,J,3]."""
    out = X.copy()
    N = X.shape[0]
    idx = np.arange(N)
    for j in range(X.shape[1]):
        ok = np.isfinite(X[:, j]).all(-1)
        if ok.sum() < 2:
            continue
        a, b = np.nonzero(ok)[0][[0, -1]]
        gap = ~ok & (idx > a) & (idx < b)
        if gap.any():
            for k in range(3):
                out[gap, j, k] = np.interp(idx[gap], idx[ok], X[ok, j, k])
    return out


def _gap_stats(have):
    """have [N,J] bool -> per-joint missing fraction, number of gaps, longest gap (frames)."""
    N, J = have.shape
    per = []
    for j in range(J):
        h = have[:, j].astype(int)
        d = np.diff(np.r_[1, h, 1])
        starts, ends = np.nonzero(d == -1)[0], np.nonzero(d == 1)[0]
        lens = ends - starts
        per.append({'missing_frac': float(1 - h.mean()) if N else 1.0, 'n_gaps': int(len(lens)),
                    'longest_gap': int(lens.max()) if len(lens) else 0})
    return per


def _bone_cv(X, bones):
    out = []
    for a, b in bones:
        L = np.linalg.norm(X[:, a] - X[:, b], axis=-1)
        L = L[np.isfinite(L)]
        out.append(float(np.std(L) / np.mean(L)) if len(L) >= 2 and np.mean(L) > 0 else float('nan'))
    return out


def _chunks(N, chunk, overlap):
    if N <= chunk:
        return [(0, N)]
    step = max(1, chunk - overlap)
    out, s = [], 0
    while True:
        e = min(N, s + chunk)
        out.append((s, e))
        if e >= N:
            break
        s += step
    return out


def _optim_chunked(cg, obs, init, sw, bones, scale_smooth, opts, tau, keep):
    """optim_points over overlapping chunks, blended linearly in the overlaps. keep [N,J]:
    cells to return (the rest NaN)."""
    C, N, J, _ = obs.shape
    out = np.zeros((N, J, 3))
    wsum = np.zeros((N, 1, 1))
    spans = _chunks(N, int(opts['chunk_frames']), int(opts['overlap_frames']))
    for s, e in spans:
        seg = obs[:, s:e]
        X0 = init[s:e]
        if not np.isfinite(X0).any():
            continue
        X = to_numpy(cg.optim_points(
            seg, X0, constraints=list(bones or []), scale_smooth=scale_smooth,
            scale_length=opts['bone_weight'], reproj_error_threshold=tau,
            scores=None if sw is None else np.sqrt(sw[:, s:e]),
            n_iters=int(opts['n_iters']), max_lr=opts['max_lr'], fill='all', normalize=True))
        w = np.ones(e - s)
        ov = int(opts['overlap_frames'])
        if len(spans) > 1 and ov > 0:
            ramp = np.linspace(0, 1, min(ov, e - s) + 2)[1:-1]
            if s > 0:
                w[:len(ramp)] = ramp
            if e < N:
                w[-len(ramp):] = np.minimum(w[-len(ramp):], ramp[::-1])
        out[s:e] += w[:, None, None] * X
        wsum[s:e] += w[:, None, None]
    with np.errstate(invalid='ignore', divide='ignore'):
        out = out / wsum
    out[~keep] = np.nan
    out[(wsum[:, 0, 0] == 0)] = np.nan
    return out, len(spans)


@thread_guard
def reconstruct_3d(cgroup, points, scores=None, bones=None, smooth=None, fill='interior',
                   robust=True, options=None, device='auto'):
    """W6: 2D -> 3D. See the module docstring.

    Args:
        cgroup: calibrated ``CameraGroup``.
        points: ``C x N x J x 2`` (or ``C x N x 2``) pixels, NaN = missing.
        scores: optional ``C x N x J`` confidences: soft observation weights (clipped to [0, 1]).
        bones: optional ``[(a, b), ...]`` joint pairs with constant length (optim_points).
        smooth: None (default, no smoothing) or a strength (optim_points ``scale_smooth``,
            pixel-equivalent and relative to the typical motion, so unit-free; 1-4 is light).
        fill: ``'interior'`` (default: gaps between a joint's first and last observation are
            filled, never extrapolated) or ``'none'``.
        robust: True (MSAC with a noise-derived tau) or False (score-weighted linear DLT).
        options: overrides of ``_W6_DEFAULTS`` (``tau_px``, ``min_score``, ``n_iters``,
            ``chunk_frames``, ``verdict``, ...).
        device: ``'auto'`` (cuda if available), ``'cpu'``, ``'cuda'``.

    Verdict rules:
        ``reconstructed_frac`` (triangulated (frame, joint) cells / all; warn < 0.5,
        fail < 0.05); ``inlier_rate`` (inlier / observed; warn < 0.7, fail < 0.3);
        ``cameras_unused`` (cameras with no observation or < 20 % of their observations used;
        warn > 0); ``bone_cv`` (median bone-length CV, with ``bones``; warn > 0.1, fail > 0.3).
    """
    from .. import geometry as G
    from .. import triangulation as T
    opts = merge_options(_W6_DEFAULTS, options)
    timer = Timer()
    if fill not in ('interior', 'none'):
        raise ValueError("fill must be 'interior' or 'none' (reconstruct_3d never extrapolates), "
                         "got {!r}".format(fill))
    pts = np.array(to_numpy(points), dtype=float)
    squeeze = pts.ndim == 3
    if squeeze:
        pts = pts[:, :, None]
    if pts.ndim != 4 or pts.shape[-1] != 2:
        raise ValueError('points must be C x N x J x 2 or C x N x 2, got {}'.format(pts.shape))
    C, N, J, _ = pts.shape
    names = list(cgroup.get_names())
    if C != len(names):
        raise ValueError('points have {} cameras, the group has {}'.format(C, len(names)))
    sw = None
    if scores is not None:
        sw = np.array(to_numpy(scores), dtype=float)
        if sw.size != C * N * J:
            raise ValueError('scores must be C x N x J = {}, got {}'.format((C, N, J), sw.shape))
        sw = np.clip(np.nan_to_num(sw.reshape(C, N, J), nan=0.0, posinf=1.0, neginf=0.0), 0, 1)
    if opts['min_score'] is not None:
        if sw is None:
            raise ValueError("options['min_score'] needs scores=")
        pts[sw < float(opts['min_score'])] = np.nan
    bones = [tuple(int(v) for v in b) for b in (bones or [])]
    for a, b in bones:
        if not (0 <= a < J and 0 <= b < J):
            raise ValueError('bone {} out of range for {} joints'.format((a, b), J))
    observed = np.isfinite(pts).all(-1)
    diag = image_diagonal(cgroup)
    cg = _group_on(cgroup, device)
    P = G.pack(cg)
    flat = pts.reshape(C, -1, 2)
    swf = None if sw is None else sw.reshape(C, -1)
    settings = {'weighting': 'soft score weights, clipped to [0, 1], on the squared pixel error '
                             '(MSAC cost, refit, Gauss-Newton; optim_points residual x sqrt(s))'
                             if sw is not None else 'none (no scores)',
                'min_score': opts['min_score'], 'robust': bool(robust), 'fill': fill,
                'image_diagonal_px': diag, 'min_views': int(opts['min_views'])}
    fallbacks = []

    with timer('triangulate'):
        if robust:
            tau0 = max(10.0, opts['first_tau_diag_frac'] * diag) if diag else 20.0
            r0 = T.triangulate_robust(P, flat, tau=tau0, scores=swf, min_inliers=opts['min_views'])
            nf = noise_floor(r0.errors, r0.inliers, diag)
            tau = float(opts['tau_px']) if opts['tau_px'] is not None else \
                float(opts['tau_noise_mult']) * nf
            res = T.triangulate_robust(P, flat, tau=tau, scores=swf, min_inliers=opts['min_views'])
            X = to_numpy(res.p3d).reshape(N, J, 3)
            inl = to_numpy(res.inliers).reshape(C, N, J)
            settings.update(first_tau_px=tau0, noise_px=nf, noise_unit=_noise_unit(nf), tau_px=tau)
        else:
            w = None if swf is None else np.sqrt(swf)
            X = to_numpy(T.triangulate_linear(P, flat, weights=w,
                                              min_cams=opts['min_views'])).reshape(N, J, 3)
            inl = observed & np.isfinite(X).all(-1)[None]
            tau = None
    have = np.isfinite(X).all(-1)                       # triangulated cells
    if robust:
        inl = inl & have[None]

    scale = _rig_scale(cgroup)
    settings['rig_scale'] = scale
    optimise = bool(bones) or smooth is not None
    keep = have.copy()
    if fill == 'interior':
        span = np.zeros_like(have)
        for j in range(J):
            ok = np.nonzero(have[:, j])[0]
            if len(ok):
                span[ok[0]:ok[-1] + 1, j] = True
        keep = span
    n_chunks = 0
    with timer('optim_points' if optimise else 'fill'):
        if optimise and have.sum() >= 2:
            obs_in = np.where(inl[..., None], pts, np.nan)
            tau_o = tau if tau is not None else 10.0 * _noise_unit(noise_floor(None))
            Xo, n_chunks = _optim_chunked(cg, obs_in, X, sw, bones,
                                          0.0 if smooth is None else float(smooth), opts, tau_o,
                                          keep)
            final = Xo
            settings['optim_points'] = {
                'bones': bones, 'scale_length': opts['bone_weight'],
                'scale_smooth': 0.0 if smooth is None else float(smooth),
                'reproj_error_threshold_px': tau_o, 'n_iters': int(opts['n_iters']),
                'max_lr_px': opts['max_lr'], 'normalize': True,
                'chunk_frames': int(opts['chunk_frames']), 'overlap_frames': int(opts['overlap_frames']),
                'n_chunks': n_chunks,
                'bone_targets': 'median length over the frames where both joints are triangulated'}
        else:
            if optimise:
                fallbacks.append('optim_points skipped: fewer than 2 triangulated cells')
            final = _interior_fill(X) if fill == 'interior' else X.copy()
            final[~keep] = np.nan
    filled = np.isfinite(final).all(-1) & ~have
    if not optimise:
        final[have] = X[have]

    # errors of every observation against the final points
    fin_flat = final.reshape(-1, 3)
    proj = to_numpy(cg.project(np.nan_to_num(fin_flat))).reshape(C, N, J, 2)
    err = np.linalg.norm(proj - pts, axis=-1)
    err[~np.isfinite(final).all(-1)[None].repeat(C, 0)] = np.nan
    n_views = np.where(have, inl.sum(0), 0)

    # report
    per_camera = {}
    unused = []
    for c, nm in enumerate(names):
        n_obs = int(observed[c].sum())
        n_in = int(inl[c].sum())
        e = err[c][inl[c]]
        per_camera[nm] = {'n_obs': n_obs, 'n_inliers': n_in,
                          'usage': n_in / n_obs if n_obs else 0.0,
                          'median_err_px': float(np.median(e)) if e.size else None}
        if n_obs == 0 or n_in < 0.2 * n_obs:
            unused.append(nm)
    gaps = _gap_stats(have)
    report = {
        'per_camera': per_camera,
        'cameras_unused': unused,
        'gaps': {'per_joint': gaps,
                 'missing_frac': float(1 - have.mean()) if have.size else 1.0,
                 'filled_frac': float(filled.mean()) if filled.size else 0.0,
                 'longest_gap': max([g['longest_gap'] for g in gaps], default=0),
                 'extrapolated': 0},
        'noise_px': settings.get('noise_px'), 'tau_px': tau,
        'median_err_px': float(np.nanmedian(err[inl])) if inl.any() else None,
    }
    rules = [
        Rule('reconstructed_frac', float(have.mean()) if have.size else 0.0, 0.5, 0.05,
             'share of (frame, joint) cells triangulated from >= {} views'.format(opts['min_views']),
             higher_is_worse=False),
        Rule('inlier_rate', float(inl.sum() / max(observed.sum(), 1)), 0.7, 0.3,
             'share of the observations used by the robust triangulation', higher_is_worse=False),
        Rule('cameras_unused', len(unused), 0, None,
             'cameras with no observations or < 20 % of them used: {}'.format(unused)),
    ]
    if bones:
        cv_before = _bone_cv(X, bones)
        cv_after = _bone_cv(final, bones)
        med = float(np.nanmedian(cv_after)) if np.isfinite(cv_after).any() else float('nan')
        report['bones'] = {'pairs': bones, 'cv': cv_after, 'cv_triangulated': cv_before,
                           'median_cv': med,
                           'median_cv_triangulated': float(np.nanmedian(cv_before))
                           if np.isfinite(cv_before).any() else None}
        rules.append(Rule('bone_cv', med, 0.1, 0.3, 'median coefficient of variation of bone lengths'))
    if not have.any():
        rules.append(Rule('no_points', True, fail_at=True,
                          explain='no point is seen by {} cameras'.format(opts['min_views'])))
    verdict, reasons, table = evaluate_rules(rules, opts['verdict'])
    report['rules'] = table
    out = final[:, 0] if squeeze else final
    sq = (lambda a: a[..., 0]) if squeeze else (lambda a: a)
    return Reconstruction(
        points3d=out, errors=sq(err), n_views=sq(n_views), inliers=sq(inl), filled=sq(filled),
        report=report, verdict=verdict, reasons=reasons, settings=settings,
        provenance=provenance('reconstruct_3d',
                              inputs={'shape': [C, N, J], 'cameras': names,
                                      'scores': scores is not None, 'bones': len(bones),
                                      'smooth': smooth},
                              fallbacks=fallbacks, timings=timer.times))


# ================================================================== W7 associate_views

@dataclass
class Association(WorkflowResult):
    """Result of :func:`associate_views`.

    ``points [C,F,P,J,2]`` per identity (frame axis = frame id; NaN where unmatched),
    ``labels``: per camera, a list over frames of ``(D,)`` int arrays (identity of each input
    detection, -1 = unassigned), ``groups``: the backend result (``matching.Groups`` for
    tracklets, a list of per-frame ``matching.Association`` otherwise), ``frame_ids [F]``.
    ``report``: ``n_identities``, ``identities`` (frames / cameras each), ``assigned_frac``,
    ``noise_px``, ``rules``."""
    points: Any = None
    labels: Any = None
    groups: Any = None
    frame_ids: Any = None

    _skip = ('cgroup', 'groups', 'labels')

    def save(self, out):
        save_arrays(self, out, {'points': self.points})


_W7_DEFAULTS = {
    'gate_diag_frac': 0.02,     # first-pass pixel gate: this x image diagonal (>= 10 px)
    'gate_noise_mult': 80.0,    # second-pass gate: this x noise floor (40 px at 0.5 px noise)
    'min_gate_px': 10.0,
    'verify_gate_mult': 2.5,    # tracklets: verify_floor = this x gate
    'min_score': None,          # keypoint score cut (third channel)
    'min_points': 4,            # finite keypoints for a detection to count
    'min_len': 10,              # tracklets: shortest tracklet (frames)
    'link_frac': 0.05,          # per-frame: max 3D jump between frames, x rig scale
    'link_max_gap': 10,         # per-frame: frames an identity may go unseen and still continue
    'verdict': None,
}


def _parse_frames(names, detections, what='detections'):
    """-> per camera {frame: (D,K,ch) array}, sorted frame ids, K."""
    from .. import matching as M
    per = M._per_camera(detections, names, what)
    out, K, fids = [], None, set()
    for c, dc in enumerate(per):
        fr = {}
        for f, v in M._frame_items(dc):
            d = M._as_dets(v)
            if d is None:
                continue
            if K is None:
                K = d.shape[1]
            if d.shape[1] != K:
                raise ValueError('camera {!r} frame {}: {} keypoints, expected {}'.format(
                    names[c], f, d.shape[1], K))
            fr[int(f)] = d
            fids.add(int(f))
        out.append(fr)
    if K is None:
        raise ValueError('no detections in any camera')
    return out, np.array(sorted(fids), dtype=np.int64), K


def _clean(d, min_score):
    xy = d[..., :2].copy()
    if min_score is not None and d.shape[-1] > 2:
        xy[d[..., 2] < min_score] = np.nan
    return xy


def _link_instances(frames_inst, gate, max_gap):
    """frames_inst: list over frames of [M_f,3] instance points -> list of [M_f] identity ids."""
    from scipy.optimize import linear_sum_assignment
    last_pos, last_f, n_id = [], [], 0
    out = []
    for f, X in enumerate(frames_inst):
        ids = np.full(len(X), -1, dtype=np.int64)
        act = [i for i in range(n_id) if f - last_f[i] <= max_gap]
        okX = np.isfinite(X).all(-1) if len(X) else np.zeros(0, bool)
        if act and okX.any():
            A = np.array([last_pos[i] for i in act])
            D = np.linalg.norm(X[:, None] - A[None], axis=-1)
            D = np.where(okX[:, None] & np.isfinite(D), D, np.inf)
            Dc = np.where(D <= gate, D, 1e30)
            r, c = linear_sum_assignment(Dc)
            for a, b in zip(r, c):
                if D[a, b] <= gate:
                    ids[a] = act[b]
        for m in range(len(X)):
            if ids[m] < 0 and okX[m]:
                ids[m] = n_id
                last_pos.append(X[m]); last_f.append(f)
                n_id += 1
            elif ids[m] >= 0:
                last_pos[ids[m]] = X[m]; last_f[ids[m]] = f
        out.append(ids)
    return out, n_id


@thread_guard
def associate_views(cgroup, detections, track_ids=None, min_views=2, options=None):
    """W7: multi-animal / multi-person association across cameras. See the module docstring.

    Args:
        cgroup: calibrated ``CameraGroup``.
        detections: per camera (list in group order, or dict by camera name): per frame (dict
            ``{frame: ...}`` or a sequence indexed by frame) a ``(D, J, 2|3)`` array or a list of
            ``(J, 2|3)`` arrays (x, y[, score]); NaN = missing keypoint.
        track_ids: optional, same structure with ``(D,)`` per-camera track ids: tracklet
            association (``matching.associate_tracklets``); otherwise per-frame association
            (``matching.associate_points`` on detection centroids) and linking over time.
        min_views: minimum cameras per identity (tracklets) / per instance (per frame).
        options: overrides of ``_W7_DEFAULTS``.

    Gates: a first pass with ``gate_diag_frac x image diagonal`` (>= ``min_gate_px``); the noise
    floor of the associated observations then gives ``gate_noise_mult x noise`` and, if that is
    tighter, a second pass. Per-frame linking accepts 3D jumps up to ``link_frac x rig scale``.

    Verdict rules: ``no_identities`` (fail); ``assigned_frac`` (detections assigned to an
    identity; warn < 0.5, fail < 0.1); ``fragmentation`` (per-frame mode: identities / most
    simultaneous instances; warn > 2).
    """
    from .. import geometry as G
    from .. import matching as M
    from .. import triangulation as T
    opts = merge_options(_W7_DEFAULTS, options)
    timer = Timer()
    names = list(cgroup.get_names())
    C = len(names)
    parsed, fids, K = _parse_frames(names, detections)
    if track_ids is not None:
        M._per_camera(track_ids, names, 'track_ids')       # shape check (raises)
    diag = image_diagonal(cgroup)
    gate1 = max(float(opts['min_gate_px']), opts['gate_diag_frac'] * diag) if diag else 40.0
    min_views = max(1, int(min_views))
    settings = {'image_diagonal_px': diag, 'first_gate_px': gate1, 'min_views': min_views,
                'mode': 'tracklets' if track_ids is not None else 'per_frame',
                'min_score': opts['min_score'], 'min_points': int(opts['min_points'])}
    P = G.pack(cgroup)
    F = int(fids.max()) + 1 if len(fids) else 0
    n_dets_total = 0
    countable = []
    for c in range(C):
        for f, d in parsed[c].items():
            ok = np.isfinite(_clean(d, opts['min_score'])).all(-1).sum(-1) >= int(opts['min_points'])
            n_dets_total += int(ok.sum())
            countable.append(ok)
    labels = [[np.full(len(parsed[c][f]) if f in parsed[c] else 0, -1, dtype=np.int64)
               for f in range(F)] for c in range(C)]
    rules_extra = []

    if track_ids is not None:
        def run(gate):
            return M.associate_tracklets(
                cgroup, detections, track_ids, min_cams=max(2, min_views), max_pair_error=gate,
                verify_floor=opts['verify_gate_mult'] * gate, min_score=opts['min_score'],
                min_points=int(opts['min_points']), min_len=int(opts['min_len']))

        def noise_of(groups):
            if not len(groups):
                return None
            pts, _, _ = groups.to_points(n_frames=400, min_cams=2)
            if pts.shape[1] == 0:
                return None
            r = T.triangulate_robust(P, pts.reshape(C, -1, 2), tau=gate1)
            return noise_floor(r.errors, r.inliers, diag)
        with timer('associate'):
            groups = run(gate1)
            nf = noise_of(groups)
            gate = gate1
            if nf is not None:
                g2 = float(np.clip(opts['gate_noise_mult'] * nf, opts['min_gate_px'], gate1))
                if g2 < 0.75 * gate1:
                    gate = g2
                    groups = run(gate)
        P_ = len(groups)
        points = np.full((C, F, P_, K, 2), np.nan)
        for gi, g in enumerate(groups):
            for c in range(C):
                fr, di, xyz = g.observations(c)
                if xyz is None:
                    continue
                points[c, fr, gi] = _clean(xyz, opts['min_score'])
                for f, i in zip(fr, di):
                    labels[c][int(f)][int(i)] = gi
        identities = [{'frames': int(g.n_frames), 'cameras': [names[c] for c in g.cameras]}
                      for g in groups]
        extra = {'tracklets': _jsonable_info(groups.info)}
        backend = groups
    else:
        # only the frames with detections are associated (frame ids may be sparse, e.g. 1e5
        # apart); the others get an empty association
        cent = []
        for f in fids:
            per = []
            for c in range(C):
                d = parsed[c].get(int(f))
                if d is None:
                    per.append(np.zeros((0, 2)))
                    continue
                xy = _clean(d, opts['min_score'])
                ok = np.isfinite(xy).all(-1)
                with np.errstate(invalid='ignore'):
                    cen = np.nanmean(np.where(ok[..., None], xy, np.nan), axis=1)
                cen[ok.sum(-1) < int(opts['min_points'])] = np.nan
                per.append(cen)
            cent.append(per)

        def empty():
            return M.Association(np.zeros((0, 3)), [], np.zeros(0), np.zeros(0, dtype=np.int64),
                                 [np.full(0, -1, dtype=np.int64) for _ in range(C)])

        def run(gate):
            out = [None] * F
            res = M.associate_points_frames(P, cent, max_res_px=gate, min_views=max(2, min_views))
            for f, a in zip(fids, res):
                out[int(f)] = a
            return [a if a is not None else empty() for a in out]
        with timer('associate'):
            assoc = run(gate1)
            res = np.concatenate([np.asarray(a.residuals, float) for a in assoc]) if assoc else \
                np.zeros(0)
            nf = noise_floor(res[np.isfinite(res)], diag=diag) if res.size else None
            gate = gate1
            if nf is not None:
                g2 = float(np.clip(opts['gate_noise_mult'] * nf, opts['min_gate_px'], gate1))
                if g2 < 0.75 * gate1:
                    gate = g2
                    assoc = run(gate)
        scale = _rig_scale(cgroup)
        link_gate = opts['link_frac'] * scale
        with timer('link'):
            ids, P_ = _link_instances([np.asarray(a.points3d, float).reshape(-1, 3) for a in assoc],
                                      link_gate, int(opts['link_max_gap']))
        points = np.full((C, F, P_, K, 2), np.nan)
        n_frames_id = np.zeros(P_, int)
        cams_id = [set() for _ in range(P_)]
        for f, (a, idf) in enumerate(zip(assoc, ids)):
            for m, mem in enumerate(a.members):
                p = int(idf[m])
                if p < 0:
                    continue
                n_frames_id[p] += 1
                for c, i in mem.items():
                    points[c, f, p] = _clean(parsed[c][f][int(i)][None], opts['min_score'])[0]
                    labels[c][f][int(i)] = p
                    cams_id[p].add(names[c])
        identities = [{'frames': int(n_frames_id[p]), 'cameras': sorted(cams_id[p])}
                      for p in range(P_)]
        most = max([len(a) for a in assoc], default=0)
        frag = P_ / most if most else float('inf')
        settings.update(link_gate=link_gate, link_frac=opts['link_frac'], rig_scale=scale,
                        link_max_gap=int(opts['link_max_gap']))
        extra = {'max_simultaneous': int(most), 'fragmentation': frag}
        rules_extra.append(Rule('fragmentation', frag if P_ else 0.0, 2.0, None,
                                'identities per simultaneous instance (track breaks)'))
        backend = assoc
    settings.update(noise_px=nf, gate_px=gate,
                    noise_unit=_noise_unit(nf) if nf is not None else None)
    n_assigned = sum(int((l >= 0).sum()) for lc in labels for l in lc)
    assigned = n_assigned / n_dets_total if n_dets_total else 0.0
    report = {'n_identities': int(P_), 'identities': identities, 'assigned_frac': assigned,
              'n_detections': int(n_dets_total), 'n_assigned': int(n_assigned),
              'noise_px': nf, 'gate_px': gate, **extra}
    rules = [Rule('no_identities', P_ == 0, fail_at=True, explain='no identity seen by '
                  '>= {} cameras'.format(max(2, min_views))),
             Rule('assigned_frac', assigned, 0.5, 0.1, 'share of detections assigned to an identity',
                  higher_is_worse=False)] + rules_extra
    verdict, reasons, table = evaluate_rules(rules, opts['verdict'])
    report['rules'] = table
    return Association(points=points, labels=labels, groups=backend, frame_ids=np.arange(F),
                       report=report, verdict=verdict, reasons=reasons, settings=settings,
                       provenance=provenance('associate_views',
                                             inputs={'cameras': names, 'n_frames': int(F),
                                                     'n_keypoints': int(K),
                                                     'track_ids': track_ids is not None},
                                             timings=timer.times))


def _jsonable_info(info):
    from ._common import _jsonable
    return _jsonable(info)


# ================================================================== W5 estimate_offsets

@dataclass
class OffsetEstimation(WorkflowResult):
    """Result of :func:`estimate_offsets` (:mod:`aniposelib.sync` convention: ``source_time =
    global_time + offset``; the reference stream has offset 0 and rate 0).

    ``offsets`` {stream: s or None (unidentifiable)}: offset at global time 0 (with a clock
    rate, the offset at global time ``g`` is ``offsets + rates * g``), ``sigmas`` {stream: s},
    ``rates`` {stream: clock-rate error, s/s}, ``timemaps`` {stream: ``sync.TimeMap``} (what the caller
    applies), ``graph`` (``sync.OffsetGraph``), ``pairs`` (every pair estimate used or
    rejected), ``drift`` {stream: ``sync.DriftFit``}, ``identifiable`` {stream: bool},
    ``refined`` (STBA summary when ``refine=True``), ``reference``."""
    offsets: Any = None
    sigmas: Any = None
    rates: Any = None
    timemaps: Any = None
    graph: Any = None
    pairs: Any = None
    drift: Any = None
    identifiable: Any = None
    refined: Any = None
    reference: Any = None


_W5_DEFAULTS = {
    'audio_sr': 16000,            # decoding rate for audio given as paths
    'audio_max_lag': None,        # s; None = every lag with min_overlap
    'audio_min_overlap': 8.0,     # s
    'audio_windows': 6,           # windowed re-estimates (sigma, drift)
    'audio_cross_check': True,    # spectrogram + onset cross-checks of the gcc estimate
    'cross_check_frames': 1.0,    # disagreement tolerance, in frame periods (or 1/30 s)
    'motion_max_lag': 5.0,        # s
    'epipolar_search_frames': 10.0,   # half-width of the epipolar search (frames)
    'epipolar_boot': 20,
    'max_residual_frames': 2.0,   # offset graph: drop edges with residual above this (frames)
    'refine_accel_frac': 10.0,    # STBA motion prior: accel_sigma = this x rig scale / s^2
    'refine_offset_sigma_min': 1e-3,   # s, floor on the STBA offset prior sigma
    'refine_rate_sigma': 1e-3,
    'refine_max_iter': 15,
    'refine_extrinsics': False,
    'verdict': None,
}


def _load_stream(name, v, sr):
    """-> dict(audio=(sig, sr, t0)|None, motion=(sig, rate, t0)|None, points=(p2d, rate, t0)|None)."""
    from .. import sync as S
    out = {'audio': None, 'motion': None, 'points': None}
    if isinstance(v, (str, bytes)) or hasattr(v, '__fspath__'):
        v = {'audio': str(v)}
    if not isinstance(v, dict):
        raise ValueError('stream {!r}: expected a video/audio path or a dict with audio= / '
                         'motion= / points=, got {}'.format(name, type(v).__name__))
    unknown = set(v) - {'audio', 'motion', 'points', 'audio_t0', 'motion_t0'}
    if unknown:
        raise ValueError('stream {!r}: unknown keys {}'.format(name, sorted(unknown)))
    a = v.get('audio')
    if a is not None:
        if isinstance(a, (str, bytes)) or hasattr(a, '__fspath__'):
            try:
                sig = S.load_audio(str(a), sr=sr)
            except ImportError as e:
                raise ImportError('stream {!r}: decoding audio needs the [sync] extra '
                                  '(soundfile / av) or ffmpeg: {}'.format(name, e))
            a = (sig, sr)
        sig, asr = a
        out['audio'] = (np.asarray(to_numpy(sig), float).ravel(), float(asr),
                        float(v.get('audio_t0', 0.0)))
    m = v.get('motion')
    if m is not None:
        sig, rate = m
        out['motion'] = (np.asarray(to_numpy(sig), float).ravel(), float(rate),
                         float(v.get('motion_t0', 0.0)))
    p = v.get('points')
    if p is not None:
        p2d, t = p
        p2d = np.asarray(to_numpy(p2d), float)
        if p2d.ndim == 2:
            p2d = p2d[:, None]
        if p2d.ndim != 3 or p2d.shape[-1] != 2:
            raise ValueError('stream {!r}: points must be [T,J,2] or [T,2], got {}'.format(
                name, p2d.shape))
        t = np.asarray(to_numpy(t), float)
        if t.ndim == 0:
            rate, t0 = float(t), 0.0
        else:
            if len(t) != len(p2d):
                raise ValueError('stream {!r}: {} point frames but {} times'.format(
                    name, len(p2d), len(t)))
            rate, t0 = 1.0 / float(np.median(np.diff(t))), float(t[0])
        out['points'] = (p2d, rate, t0)
    if all(x is None for x in out.values()):
        raise ValueError('stream {!r} has no usable signal'.format(name))
    return out


def _resample(x, sr, sr_to):
    if sr == sr_to:
        return x
    from fractions import Fraction
    from scipy.signal import resample_poly
    fr = Fraction(sr_to / sr).limit_denominator(1000)
    return resample_poly(x, fr.numerator, fr.denominator)


def _edge(est, pair, offset_shift=0.0):
    """Copy of an OffsetEstimate in seconds for the graph (pair, shifted to source times)."""
    import dataclasses
    return dataclasses.replace(est, offset=float(est.offset) + offset_shift, pair=pair)


@thread_guard
def estimate_offsets(streams, cgroup=None, reference=None, drift=True, refine=False,
                     options=None, rng=0, device='auto'):
    """W5: time offsets between unsynchronised streams. See the module docstring.

    Args:
        streams: ``{name: video/audio path | dict}``; the dict may hold ``audio=(signal, sr)``
            or a path (decoded with ``sync.load_audio``, ``[sync]`` extra; ``audio_t0`` = source
            time of sample 0, default 0), ``motion=(signal, rate)`` (``motion_t0``) and
            ``points=(p2d [T,J,2], t)`` with ``t`` the source time of every frame or a scalar
            frame rate (Hz). With ``cgroup``, stream names must be camera names for the
            epipolar estimates (and STBA).
        cgroup: optional ``CameraGroup``: epipolar estimates between camera pairs with points.
        reference: anchor stream (offset 0); default the best-connected one.
        drift: fit a per-stream clock rate (``sync.fit_drift``) on the windowed audio estimates.
        refine: run ``stba.SpatiotemporalBA`` on the points (needs ``cgroup`` and points in
            >= 2 cameras) with offset priors anchored to the graph estimates.
        options: overrides of ``_W5_DEFAULTS``.
        rng: seed for the bootstrap resamples.
        device: ``'auto'`` (cuda if available), ``'cpu'`` or ``'cuda'``: where the STBA
            refinement runs.

    Verdict rules: ``unidentifiable_streams`` (streams not connected to the reference; fail);
    ``max_sigma_frames`` (largest offset sigma in frame periods; warn > 0.25, fail > 1);
    ``cycle_rms_frames`` (graph inconsistency; warn > 0.25, fail > 1); ``method_disagreements``
    (audio methods disagreeing beyond ``cross_check_frames``; warn > 0); ``drift_unidentifiable``
    (with ``drift``: streams whose rate could not be fitted although windows exist; warn > 0).
    """
    from .. import sync as S
    opts = merge_options(_W5_DEFAULTS, options)
    timer = Timer()
    if not isinstance(streams, dict) or len(streams) < 2:
        raise ValueError('streams must be a dict of >= 2 streams {name: path | dict}')
    names = [str(n) for n in streams]
    with timer('load'):
        data = {str(n): _load_stream(str(n), v, opts['audio_sr']) for n, v in streams.items()}
    if reference is not None and str(reference) not in data:
        raise ValueError('reference {!r} is not a stream ({})'.format(reference, names))
    cam_names = list(cgroup.get_names()) if cgroup is not None else []
    rates = [d['points'][1] for d in data.values() if d['points'] is not None] + \
            [d['motion'][1] for d in data.values() if d['motion'] is not None]
    frame_rate = float(np.median(rates)) if rates else 30.0
    frame_s = 1.0 / frame_rate
    settings = {'frame_rate': frame_rate, 'frame_period_s': frame_s,
                'cross_check_tol_s': opts['cross_check_frames'] * frame_s,
                'max_residual_s': opts['max_residual_frames'] * frame_s,
                'audio_windows': opts['audio_windows'], 'drift': bool(drift),
                'refine': bool(refine)}
    pairs, edges, windows, disagreements, fallbacks = [], [], {}, [], []

    def record(a, b, est, kind, used=True, note=None):
        d = {'pair': [a, b], 'kind': kind, 'method': est.method, 'offset': float(est.offset),
             'sigma': float(est.sigma), 'confidence': float(est.confidence),
             'identifiable': bool(est.identifiable), 'reasons': list(est.reasons), 'used': used}
        if note:
            d['note'] = note
        pairs.append(d)

    # ---------------------------------------------------------------- pair estimates
    for a, b in itertools.combinations(names, 2):
        da, db = data[a], data[b]
        coarse = None
        if da['audio'] is not None and db['audio'] is not None:
            with timer('audio'):
                (xa, sra, ta), (xb, srb, tb) = da['audio'], db['audio']
                sr = min(sra, srb)
                xa, xb = _resample(xa, sra, sr), _resample(xb, srb, sr)
                shift = tb - ta      # source time of sample 0
                kw = dict(max_lag=opts['audio_max_lag'], min_overlap=min(
                    opts['audio_min_overlap'], 0.5 * min(len(xa), len(xb)) / sr), pair=(a, b))
                g = S.audio_offset(xa, xb, sr, method='gcc', windows=opts['audio_windows'], **kw)
                checks = []
                if opts['audio_cross_check']:
                    for m in ('spectrogram', 'onset'):
                        try:
                            checks.append(S.audio_offset(xa, xb, sr, method=m, windows=0, **kw))
                        except Exception as e:  # pragma: no cover - defensive
                            fallbacks.append('audio {} {}-{}: {}'.format(m, a, b, e))
                tol = opts['cross_check_frames'] * frame_s
                best = g
                agree = [c for c in checks if c.identifiable and abs(c.offset - g.offset) <= tol]
                disagree = [c for c in checks if c.identifiable and abs(c.offset - g.offset) > tol]
                if not g.identifiable:
                    # gcc failed: an identifiable spectrogram estimate stands in
                    alt = [c for c in checks if c.identifiable and c.method == 'spectrogram']
                    if alt:
                        best = alt[0]
                        fallbacks.append('audio {}-{}: gcc unidentifiable, spectrogram used'.format(a, b))
                elif disagree and not agree:
                    disagreements.append({'pair': [a, b], 'gcc': float(g.offset),
                                          **{c.method: float(c.offset) for c in disagree}})
                record(a, b, g, 'audio', used=best is g)
                for c in checks:
                    record(a, b, c, 'audio', used=best is c, note='cross-check')
                e = _edge(best, (a, b), shift)
                edges.append(e)
                if best.identifiable:
                    coarse = e.offset
                    wt, wo = g.diagnostics.get('window_times'), g.diagnostics.get('window_offsets')
                    if best is g and wt is not None and len(wo) >= 3:
                        windows[(a, b)] = (np.asarray(wt, float) + ta,
                                           np.asarray(wo, float) + shift)
        ma = da['motion'] or (None if da['points'] is None else
                              (S.motion_signal(da['points'][0]), da['points'][1], da['points'][2]))
        mb = db['motion'] or (None if db['points'] is None else
                              (S.motion_signal(db['points'][0]), db['points'][1], db['points'][2]))
        if ma is not None and mb is not None:
            with timer('motion'):
                (sa, ra, ta), (sb, rb, tb) = ma, mb
                r = min(ra, rb)
                if ra != rb:
                    sa = np.interp(np.arange(0, len(sa) / ra, 1 / r) * ra, np.arange(len(sa)), sa)
                    sb = np.interp(np.arange(0, len(sb) / rb, 1 / r) * rb, np.arange(len(sb)), sb)
                est = S.motion_offset(sa, sb, r, max_lag=opts['motion_max_lag'], pair=(a, b))
                # motion is the weaker signal: an edge only where the pair has no audio estimate
                use = est.identifiable and coarse is None
                record(a, b, est, 'motion', used=use)
                if use:
                    e = _edge(est, (a, b), tb - ta)
                    edges.append(e)
                    coarse = e.offset
        if cgroup is not None and da['points'] is not None and db['points'] is not None \
                and a in cam_names and b in cam_names:
            (pa, ra, ta), (pb, rb, tb) = da['points'], db['points']
            if abs(ra - rb) > 1e-6 * ra:
                fallbacks.append('epipolar {}-{}: different frame rates, skipped'.format(a, b))
            else:
                with timer('epipolar'):
                    # centre the search on the coarse (audio / motion) estimate, in frames of b
                    c0 = 0.0 if coarse is None else (coarse - (tb - ta)) * ra
                    w = float(opts['epipolar_search_frames'])
                    sub = cgroup.subset_cameras([cam_names.index(a), cam_names.index(b)])
                    est = S.epipolar_offset(pa, pb, sub, search=(c0 - w, c0 + w), rate=ra,
                                            n_boot=int(opts['epipolar_boot']), rng=rng, pair=(a, b))
                    record(a, b, est, 'epipolar', used=bool(est.identifiable))
                    edges.append(_edge(est, (a, b), tb - ta))

    # ---------------------------------------------------------------- graph
    deg = {n: 0 for n in names}
    for e in edges:
        if e.identifiable:
            deg[e.pair[0]] += 1
            deg[e.pair[1]] += 1
    ref = str(reference) if reference is not None else max(names, key=lambda n: (deg[n], -names.index(n)))
    with timer('graph'):
        # estimates without a sigma (motion) count as one frame period
        graph = S.solve_offset_graph(edges, anchor=ref, names=names,
                                     max_residual=opts['max_residual_frames'] * frame_s,
                                     default_sigma=frame_s)
    offsets = dict(graph.offsets)
    sigmas = {n: (float(graph.sigma[n]) if graph.sigma.get(n) is not None else None) for n in names}
    identifiable = dict(graph.identifiable)

    # ---------------------------------------------------------------- drift
    drift_fits, rates_out, drift_fail = {}, {n: 0.0 for n in names}, []
    if drift:
        with timer('drift'):
            meas = {}   # stream -> (t, offset relative to ref)
            for (a, b), (wt, wo) in windows.items():
                if a == ref:
                    meas.setdefault(b, []).append((wt, wo))
                elif b == ref:
                    meas.setdefault(a, []).append((wt + wo, -wo))
            # streams without a direct edge to the reference: compose through a fitted stream
            for (a, b), (wt, wo) in windows.items():
                for x, y, sgn in ((a, b, 1), (b, a, -1)):
                    if y not in meas and x in meas and y != ref:
                        t_x = np.concatenate([m[0] for m in meas[x]])
                        o_x = np.concatenate([m[1] for m in meas[x]])
                        fx = S.fit_drift(t_x, o_x)
                        tt = wt if sgn == 1 else wt + wo
                        meas[y] = [(tt, fx.predict(tt) + sgn * wo)]
            for n, ms in meas.items():
                t = np.concatenate([m[0] for m in ms])
                o = np.concatenate([m[1] for m in ms])
                if len(t) < 3:
                    continue
                fit = S.fit_drift(t, o)
                drift_fits[n] = fit
                if fit.identifiable:
                    rates_out[n] = float(fit.rate)
                else:
                    drift_fail.append(n)
    a0s = {}      # offset at global time 0 of each stream's affine map
    for n in names:
        if offsets.get(n) is None:
            continue
        if n in drift_fits and drift_fits[n].identifiable:
            f = drift_fits[n]
            # keep the graph offset at the fit's reference time, the fitted rate around it
            a0s[n] = offsets[n] - f.rate * f.t_ref
        else:
            a0s[n] = offsets[n]
    settings['drift_t_ref'] = {n: float(f.t_ref) for n, f in drift_fits.items()}

    # ---------------------------------------------------------------- STBA refinement
    refined = None
    if refine:
        seed = {n: S.TimeMap.from_offset(a0s[n], rate=rates_out[n]) for n in a0s}
        refined = _refine_stba(cgroup, data, names, ref, seed, sigmas, drift, opts, settings,
                               fallbacks, timer, device=device)
        if refined.get('ok'):
            # STBA: true global = seed + delta + rc * (seed - tmid), so the refined offset is
            # seed offset - delta - rc * (g - tmid)
            for n, dlt in refined['delta'].items():
                rc = refined['rate_correction'][n] if drift else 0.0
                a0s[n] = a0s[n] - dlt + rc * refined['tmid'][n]
                rates_out[n] = rates_out[n] - rc
                sigmas[n] = refined['sigmas'][n]
    timemaps = {n: (S.TimeMap.from_offset(a0s[n], rate=rates_out[n]) if n in a0s else None)
                for n in names}
    # reported offsets: the constant part of each time map (offset at global time 0); with a
    # rate the offset at global time g is offsets[n] + rates[n] * g
    offsets = {n: (float(a0s[n]) if n in a0s else None) for n in names}

    # ---------------------------------------------------------------- verdict
    unident = [n for n in names if offsets.get(n) is None]
    sig = [s for n, s in sigmas.items() if n != ref and s is not None and np.isfinite(s)]
    max_sigma_f = max(sig) / frame_s if sig else 0.0
    cyc = graph.cycle_rms / frame_s if np.isfinite(graph.cycle_rms) else 0.0
    rules = [Rule('unidentifiable_streams', len(unident), None, 0,
                  'streams with no identifiable path to the reference {!r}: {}'.format(ref, unident)),
             Rule('max_sigma_frames', max_sigma_f, 0.25, 1.0, 'largest offset sigma / frame period'),
             Rule('cycle_rms_frames', cyc, 0.25, 1.0,
                  'RMS residual of the offset graph / frame period (contradicting estimates)'),
             Rule('method_disagreements', len(disagreements), 0, None,
                  'audio pairs where spectrogram / onset contradict gcc')]
    if drift:
        rules.append(Rule('drift_unidentifiable', len(drift_fail), 0, None,
                          'streams whose clock rate could not be fitted: {}'.format(drift_fail)))
    verdict, reasons, table = evaluate_rules(rules, opts['verdict'])
    report = {'reference': ref, 'pairs': pairs, 'n_edges': len(edges),
              'rejected': list(graph.rejected), 'cycle_rms_s': graph.cycle_rms,
              'disagreements': disagreements, 'unidentifiable': unident,
              'drift': {n: f.to_dict() for n, f in drift_fits.items()}, 'rules': table}
    return OffsetEstimation(
        offsets=offsets, sigmas=sigmas, rates=rates_out, timemaps=timemaps, graph=graph,
        pairs=pairs, drift=drift_fits, identifiable={n: offsets.get(n) is not None for n in names},
        refined=refined, reference=ref, report=report, verdict=verdict, reasons=reasons,
        settings=settings,
        provenance=provenance('estimate_offsets',
                              inputs={n: [k for k, v in d.items() if v is not None]
                                      for n, d in data.items()},
                              fallbacks=fallbacks, timings=timer.times))


def _refine_stba(cgroup, data, names, ref, seed, sigmas, drift, opts, settings, fallbacks,
                 timer, device='auto'):
    """SpatiotemporalBA on the 2D points, seed times from the graph/drift time maps; returns
    the per-stream constant corrections ``delta`` (refined offset = seed offset - delta)."""
    from .. import stba
    if cgroup is None:
        fallbacks.append('refine: needs cgroup, skipped')
        return {'ok': False, 'reason': 'no cgroup'}
    cam_names = list(cgroup.get_names())
    use = [n for n in names if data[n]['points'] is not None and n in cam_names and n in seed]
    if len(use) < 2 or ref not in use:
        fallbacks.append('refine: needs points for the reference and >= 1 other camera')
        return {'ok': False, 'reason': 'not enough cameras with points'}
    if len({data[n]['points'][0].shape[1] for n in use}) != 1:
        return {'ok': False, 'reason': 'cameras have different numbers of keypoints'}
    sub = cgroup.subset_cameras([cam_names.index(n) for n in use])
    cams, pts, uvs, ts, tmid = [], [], [], [], {}
    for k, n in enumerate(use):
        p2d, rate, t0 = data[n]['points']
        glob = seed[n].to_global(t0 + np.arange(p2d.shape[0]) / rate)     # seed global time
        fi, ji = np.nonzero(np.isfinite(p2d).all(-1))
        cams.append(np.full(len(fi), k)); pts.append(ji); uvs.append(p2d[fi, ji]); ts.append(glob[fi])
        tmid[n] = 0.5 * (glob[fi].min() + glob[fi].max()) if len(fi) else 0.0
    obs = stba.TimedObservations(cam=np.concatenate(cams), point=np.concatenate(pts),
                                 uv=np.concatenate(uvs), t=np.concatenate(ts))
    scale = _rig_scale(sub)
    prior = [max(float(sigmas[n] or 0.0), opts['refine_offset_sigma_min']) for n in use if n != ref]
    kw = dict(knot_spacing=settings['frame_period_s'], accel_sigma=opts['refine_accel_frac'] * scale,
              free_offsets='per_camera', offset_prior_sigma=float(np.median(prior)),
              ref_camera=use.index(ref), free_rate=bool(drift),
              rate_prior_sigma=opts['refine_rate_sigma'], extrinsics=bool(opts['refine_extrinsics']),
              max_iter=int(opts['refine_max_iter']))
    settings['stba'] = dict(kw, accel_sigma_rel=opts['refine_accel_frac'], cameras=use)
    try:
        with timer('stba'):
            res = stba.SpatiotemporalBA(sub, obs, device=_device(device), **kw).fit()
    except Exception as e:  # a failed refinement is reported, never raised
        fallbacks.append('refine: STBA failed: {}'.format(e))
        return {'ok': False, 'reason': 'STBA failed: {}'.format(e)}
    delta = np.asarray(res.delta, float)
    sg = np.asarray(res.delta_sigma_sandwich, float)
    sg = np.where(np.isfinite(sg), sg, np.asarray(res.delta_sigma, float))
    return {'ok': True, 'cameras': use,
            'delta': {n: float(delta[k]) for k, n in enumerate(use)},
            'sigmas': {n: (0.0 if n == ref else float(sg[k]) if np.isfinite(sg[k]) else sigmas[n])
                       for k, n in enumerate(use)},
            'rate_correction': {n: float(r) for n, r in zip(use, np.asarray(res.rate, float))},
            'tmid': tmid, 'sigma_px': float(res.sigma_px), 'cost': float(res.cost)}
