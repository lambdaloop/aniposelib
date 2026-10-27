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
                      image_diagonal, merge_options, noise_floor, provenance, reindex_cameras,
                      save_arrays, thread_guard, to_numpy, worst)

__all__ = ['estimate_offsets', 'OffsetEstimation', 'reconstruct_3d', 'Reconstruction',
           'associate_views', 'Association']


# ================================================================== shared helpers

def _noise_unit(nf):
    """Noise unit ``u`` (capped; see :func:`aniposelib.workflows._common.noise_unit`)."""
    from ._common import noise_unit
    return noise_unit(nf)


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
    triangulated, ``point_error [N,J]`` median reprojection error (px) of each triangulated
    point over its inlier views (NaN for filled / missing cells), ``out_of_image [C,N,J]``
    observations outside the image (not used with ``mask_out_of_bounds``). ``report``:
    ``per_camera`` usage, ``gaps``, ``bones`` (length CV), ``noise_px``, ``tau_px``, ``rules``."""
    points3d: Any = None
    errors: Any = None
    n_views: Any = None
    inliers: Any = None
    filled: Any = None
    point_error: Any = None
    out_of_image: Any = None

    def save(self, out):
        save_arrays(self, out, {'points3d': self.points3d, 'errors': self.errors,
                                'n_views': self.n_views, 'inliers': self.inliers,
                                'point_error': self.point_error})


_W6_DEFAULTS = {
    'tau_px': None,            # fixed MSAC threshold (px); None = clip(3 x noise floor, 10 px, 1 % of the diagonal)
    'tau_noise_mult': 3.0,
    'first_tau_diag_frac': 0.01,   # first robust pass: tau = max(10 px, this x image diagonal)
    'min_score': None,         # hard score cut (observations below are removed); None = soft only
    'min_views': 2,
    'min_joints_per_view': None,   # camera-frames with fewer finite keypoints are dropped (int, or a fraction of J if < 1)
    'mask_out_of_bounds': True,    # observations outside the image are not used (counted as out_of_image)
    'max_gap': 10,             # fill='interior': longest gap (frames) that is filled; None = every interior gap
    'max_gap_s': None,         # same in seconds (needs times=)
    'max_point_err_px': None,  # triangulated points whose point_error is above this are removed (NaN)
    'cam_names': None,         # camera order of points (reindexed to the group; missing cameras -> NaN)
    'bone_weight': 'auto',     # 'auto' = per-bone weight from the measured length spread; a float = fixed weight
    'bone_max_reproj_rise': 0.5,   # bones/smooth: reject a chunk's refinement when its median inlier error rises more than this x noise floor
    'n_iters': 300,            # optim_points Adam iterations per chunk
    'max_lr': 0.1,             # optim_points step (pixels)
    'chunk_frames': 2000,      # optim_points chunk length (frames) for long sequences
    'overlap_frames': 100,     # overlap between chunks (linear blending)
    'verdict': None,
}

BONE_WEIGHT_MAX = 2.0          # 'auto' bone weights never exceed the fixed 1.1 default
BONE_NOISE_FLOOR = 0.3         # 'auto': true length spread >= this x the expected triangulation noise of the length


def _segment_runs(N, segments):
    """-> list of (start, end) frame runs; a new run starts wherever the segment id changes."""
    if segments is None or N == 0:
        return [(0, N)]
    s = np.asarray(segments).reshape(-1)
    if len(s) != N:
        raise ValueError('segments must have one id per frame ({}), got {}'.format(N, len(s)))
    cut = np.nonzero(s[1:] != s[:-1])[0] + 1
    b = np.r_[0, cut, N]
    return [(int(a), int(e)) for a, e in zip(b[:-1], b[1:])]


def _fill_keep(have, runs, max_gap, times=None, max_gap_s=None):
    """Cells to return with fill='interior': triangulated cells plus interior gaps (within a
    segment run) no longer than ``max_gap`` frames / ``max_gap_s`` seconds."""
    keep = have.copy()
    N, J = have.shape
    for s, e in runs:
        for j in range(J):
            ok = np.nonzero(have[s:e, j])[0] + s
            if len(ok) < 2:
                continue
            for a, b in zip(ok[:-1], ok[1:]):
                if b - a <= 1:
                    continue
                if max_gap is not None and (b - a - 1) > max_gap:
                    continue
                if max_gap_s is not None and times is not None and (times[b] - times[a]) > max_gap_s:
                    continue
                keep[a + 1:b, j] = True
    return keep


def _interior_fill(X, keep, runs, times=None):
    """Linear interpolation (on ``times`` when given) of the cells ``keep & ~finite`` per
    (segment run, joint)."""
    out = X.copy()
    for s, e in runs:
        x = (np.arange(s, e) if times is None else times[s:e]).astype(float)
        for j in range(X.shape[1]):
            ok = np.isfinite(X[s:e, j]).all(-1)
            gap = keep[s:e, j] & ~ok
            if ok.sum() < 2 or not gap.any():
                continue
            for k in range(3):
                out[s:e, j, k][gap] = np.interp(x[gap], x[ok], X[s:e, j, k][ok])
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


def _bone_lengths(X, a, b):
    L = np.linalg.norm(X[:, a] - X[:, b], axis=-1)
    return L[np.isfinite(L)]


def _bone_cv(X, bones):
    """Robust coefficient of variation per bone: 1.4826 x MAD / median of the length."""
    out = []
    for a, b in bones:
        L = _bone_lengths(X, a, b)
        m = np.median(L) if len(L) >= 2 else np.nan
        out.append(float(1.4826 * np.median(np.abs(L - m)) / m) if len(L) >= 2 and m > 0
                   else float('nan'))
    return out


def _bone_cv_std(X, bones):
    out = []
    for a, b in bones:
        L = _bone_lengths(X, a, b)
        out.append(float(np.std(L) / np.mean(L)) if len(L) >= 2 and np.mean(L) > 0 else float('nan'))
    return out


def _length_noise(P, X, inl, a, b, nf, max_frames=2000):
    """Expected triangulation noise (world units, median over frames) of the length a-b:
    ``sqrt(u^T (Cov_a + Cov_b) u)`` with ``Cov = nf^2 (sum_c J_c^T J_c)^-1`` over each joint's
    inlier views (``J`` = projection Jacobian) and ``u`` the bone direction."""
    from .. import geometry as G
    fr = np.nonzero(np.isfinite(X[:, a]).all(-1) & np.isfinite(X[:, b]).all(-1))[0]
    if len(fr) == 0:
        return float('nan')
    fr = fr[np.linspace(0, len(fr) - 1, min(len(fr), max_frames)).astype(int)]
    covs = []
    for j in (a, b):
        Xj = torch.as_tensor(X[fr, j], device=P.device, dtype=P.dtype)
        _, _, Jac = G.project_jacobian(P, Xj)
        Jac = to_numpy(Jac)                                            # [C,M,2,3]
        m = inl[:, fr, j].astype(float)                                # [C,M]
        H = np.einsum('cm,cmki,cmkj->mij', m, np.nan_to_num(Jac), np.nan_to_num(Jac))
        ok = np.linalg.matrix_rank(H) == 3 if H.ndim == 3 else np.zeros(0, bool)
        Cv = np.full_like(H, np.nan)
        Cv[ok] = np.linalg.inv(H[ok])
        covs.append(Cv)
    d = X[fr, a] - X[fr, b]
    u = d / np.linalg.norm(d, axis=-1, keepdims=True)
    v = np.einsum('mi,mij,mj->m', u, covs[0] + covs[1], u)
    v = v[np.isfinite(v) & (v >= 0)]
    return float(nf * np.sqrt(np.median(v))) if len(v) else float('nan')


def _auto_bone_weights(P, X, inl, bones, nf, unit):
    """Per-bone optim_points weights from each bone's measured length spread.

    A bone with robust length spread ``s`` (1.4826 MAD of the triangulated length) and expected
    triangulation noise ``n`` (:func:`_length_noise`) has a true spread
    ``t = sqrt(max(s^2 - n^2, (BONE_NOISE_FLOOR n)^2))``. Its weight is that of a Gaussian length
    prior relative to the squared pixel residuals, ``nf^2 / (10 (t / unit)^2)`` (``unit`` =
    world length per pixel; optim_points multiplies the bone term by 10), capped at
    ``BONE_WEIGHT_MAX``. Rigid bones get the cap, non-rigid ones (tails, spines) little weight."""
    w, info = [], []
    for a, b in bones:
        L = _bone_lengths(X, a, b)
        n = _length_noise(P, X, inl, a, b, nf) if len(L) >= 3 else float('nan')
        if len(L) < 3 or not np.isfinite(n):
            w.append(BONE_WEIGHT_MAX)
            info.append({'spread': None, 'noise': None, 'weight': BONE_WEIGHT_MAX})
            continue
        s = 1.4826 * float(np.median(np.abs(L - np.median(L))))
        true2 = max(s * s - n * n, (BONE_NOISE_FLOOR * n) ** 2, 1e-300)
        wb = float(min(BONE_WEIGHT_MAX, nf * nf * unit * unit / (10.0 * true2)))
        w.append(wb)
        info.append({'spread': s, 'noise': n, 'weight': wb})
    return np.array(w, float), info


OPTIM_GPU_MIN_CELLS = 50000   # device='auto': smaller optim_points chunks (frames x joints) run on the CPU


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


def _optim_chunked(cg, obs, init, sw, bones, scale_smooth, opts, tau, keep, bone_w=None,
                   med_err=None, max_rise=None, base=None, segments=None):
    """optim_points over overlapping chunks, blended linearly in the overlaps. keep [N,J]:
    cells to return (the rest NaN). ``med_err(X, s, e)``: median inlier error of chunk points;
    a chunk whose refinement raises it by more than ``max_rise`` px is replaced by ``base``
    (the triangulated / interpolated points). Returns (points, n_chunks, n_rejected)."""
    C, N, J, _ = obs.shape
    out = np.zeros((N, J, 3))
    wsum = np.zeros((N, 1, 1))
    spans = _chunks(N, int(opts['chunk_frames']), int(opts['overlap_frames']))
    n_rej = 0
    for s, e in spans:
        seg = obs[:, s:e]
        X0 = init[s:e]
        if not np.isfinite(X0).any():
            continue
        X = to_numpy(cg.optim_points(
            seg, X0, constraints=list(bones or []), scale_smooth=scale_smooth,
            scale_length=(BONE_WEIGHT_MAX if bone_w is None else bone_w),
            reproj_error_threshold=tau,
            scores=None if sw is None else np.sqrt(sw[:, s:e]),
            n_iters=int(opts['n_iters']), max_lr=opts['max_lr'], fill='all', normalize=True,
            segments=None if segments is None else segments[s:e]))
        if med_err is not None and max_rise is not None:
            before, after = med_err(X0, s, e), med_err(X, s, e)
            if np.isfinite(before) and np.isfinite(after) and after > before + max_rise:
                X = np.where(np.isfinite(base[s:e]), base[s:e], X)
                n_rej += 1
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
    return out, len(spans), n_rej


def _parse_extrinsics(ext, C, N):
    """Per-frame extrinsics -> [C,N,4,4] float array (group camera order)."""
    if ext is None:
        return None
    if isinstance(ext, (list, tuple)) and len(ext) and hasattr(ext[0], 'cameras'):
        if len(ext) != N:
            raise ValueError('extrinsics: {} camera groups for {} frames'.format(len(ext), N))
        E = np.stack([np.stack([to_numpy(c.get_extrinsics_mat()) for c in g.cameras]) for g in ext], 1)
    else:
        E = np.asarray(to_numpy(ext), dtype=float)
    if E.ndim == 4 and E.shape[-2:] == (3, 4):
        pad = np.zeros(E.shape[:2] + (1, 4))
        pad[..., 0, 3] = 1.0
        E = np.concatenate([E, pad], -2)
    if E.ndim != 4 or E.shape[-2:] != (4, 4):
        raise ValueError('extrinsics must be [C,N,4,4] or [N,C,4,4] (or 3x4), got {}'.format(E.shape))
    if E.shape[:2] == (C, N):
        return E.astype(float)
    if E.shape[:2] == (N, C):
        return E.transpose(1, 0, 2, 3).astype(float)
    raise ValueError('extrinsics {} match neither [C,N]=({},{}) nor [N,C]'.format(E.shape[:2], C, N))


def _affine_group(cgroup, flat):
    """All-AffineCamera group -> (telephoto pinhole group, affine projection fn). The pinholes
    (``AffineCamera.to_pinhole``) match the affine cameras to 0.01 px on the linear affine
    triangulation of the data; final errors use the exact affine projection."""
    from ..affine import AffineCamera, triangulate_affine
    from ..cameras import CameraGroup
    cams = list(cgroup.cameras)
    aff = [isinstance(c, AffineCamera) for c in cams]
    if not any(aff):
        return None, None
    if not all(aff):
        raise ValueError('reconstruct_3d: the group mixes AffineCamera and pinhole cameras; convert '
                         'the affine ones with AffineCamera.to_pinhole(support_points)')
    X = to_numpy(triangulate_affine(cams, flat, min_cams=2)).reshape(-1, 3)
    X = X[np.isfinite(X).all(-1)]
    if len(X) < 4:
        raise ValueError('reconstruct_3d: AffineCamera group but fewer than 4 points seen by 2 '
                         'cameras; convert with AffineCamera.to_pinhole(support_points)')
    lo, hi = np.percentile(X, 1, axis=0), np.percentile(X, 99, axis=0)
    ext = hi - lo
    corners = np.array([[lo[0] if i & 1 else hi[0], lo[1] if i & 2 else hi[1],
                         lo[2] if i & 4 else hi[2]] for i in range(8)])
    support = np.concatenate([corners - 0.25 * ext, corners + 0.25 * ext, X[::max(1, len(X) // 2000)]])
    pin = CameraGroup([c.to_pinhole(support, max_error_px=0.01) for c in cams])
    A = np.stack([to_numpy(c.A) for c in cams])
    b = np.stack([to_numpy(c.b) for c in cams])

    def proj(Xf):
        return np.einsum('cij,nj->cni', A, Xf) + b[:, None]
    return pin, proj


def _error_drift(err, use, n_blocks=8, min_frames=3):
    """Largest / smallest median error over blocks of frames (in order; the smallest at least
    MIN_NOISE_PX). Errors that grow over the recording: cameras moved or sync broke."""
    e = np.where(use, err, np.nan)
    fr = np.nonzero(np.isfinite(e).any((0, 2)))[0]
    nb = min(n_blocks, len(fr) // min_frames)
    if nb < 4:
        return None
    meds = [float(np.nanmedian(e[:, b])) for b in np.array_split(fr, nb)]
    return max(meds) / max(min(meds), MIN_NOISE_PX)


def _reconstruct_identities(cgroup, pts, scores, bones, smooth, fill, robust, options, device,
                            times, segments, extrinsics):
    """reconstruct_3d per identity of ``C x F x P x J x 2`` points; arrays gain a P axis after F."""
    from ._common import worst
    sw = None if scores is None else np.asarray(to_numpy(scores), dtype=float)
    P = pts.shape[2]
    rs = []
    for p in range(P):
        rs.append(reconstruct_3d(cgroup, pts[:, :, p], scores=None if sw is None else sw[:, :, p],
                                 bones=bones, smooth=smooth, fill=fill, robust=robust,
                                 options=options, device=device, times=times, segments=segments,
                                 extrinsics=extrinsics))
    if not rs:
        raise ValueError('points have no identities (axis 2 is empty)')
    st = lambda name, ax: np.stack([getattr(r, name) for r in rs], ax)
    reasons = ['identity {}: {}'.format(p, x) for p, r in enumerate(rs) for x in r.reasons]
    report = {'n_identities': P, 'per_identity': [r.report for r in rs],
              'rules': [dict(t, identity=p) for p, r in enumerate(rs) for t in r.report['rules']]}
    settings = dict(rs[0].settings, per_identity_tau_px=[r.settings.get('tau_px') for r in rs],
                    per_identity_noise_px=[r.settings.get('noise_px') for r in rs])
    return Reconstruction(points3d=st('points3d', 1), errors=st('errors', 2), n_views=st('n_views', 1),
                          inliers=st('inliers', 2), filled=st('filled', 1),
                          point_error=st('point_error', 1), out_of_image=st('out_of_image', 2),
                          report=report, verdict=worst(*[r.verdict for r in rs]), reasons=reasons,
                          settings=settings,
                          provenance=dict(rs[0].provenance, identities=P,
                                          timings={'per_identity': [r.provenance['timings'] for r in rs]}))


@thread_guard
def reconstruct_3d(cgroup, points, scores=None, bones=None, smooth=None, fill='interior',
                   robust=True, options=None, device='auto', times=None, segments=None,
                   extrinsics=None):
    """W6: 2D -> 3D. See the module docstring.

    Args:
        cgroup: calibrated ``CameraGroup`` (pinhole / fisheye cameras, or only
            ``AffineCamera`` s: these are triangulated through exact telephoto pinholes).
        points: ``C x N x J x 2`` (or ``C x N x 2``) pixels, NaN = missing; or ``C x N x P x J
            x 2`` with an identity axis (``associate_views``' output): each identity is
            reconstructed separately (bones and smoothing never mix animals), the arrays get a
            ``P`` axis after ``N`` and ``report['per_identity']`` holds each report.
        scores: optional ``C x N x J`` (``C x N x P x J``) confidences: soft observation weights (clipped to [0, 1]).
        bones: optional ``[(a, b), ...]`` joint pairs with constant length (optim_points).
            With ``bone_weight='auto'`` each bone's weight follows its measured length spread,
            so non-rigid "bones" are barely enforced; a chunk whose refinement raises the
            reprojection error above the noise floor is rejected (``bone_max_reproj_rise``).
        smooth: None (default, no smoothing) or a strength (optim_points ``scale_smooth``,
            pixel-equivalent and relative to the typical motion, so unit-free; 1-4 is light).
        fill: ``'interior'`` (default: gaps of at most ``max_gap`` frames between a joint's
            observations are filled, never extrapolated) or ``'none'``. For sparse label frames
            (frames that are not consecutive) use ``'none'``, or pass ``times=``.
        robust: True (MSAC with a noise-derived tau) or False (score-weighted linear DLT).
        options: overrides of ``_W6_DEFAULTS``:
            ``tau_px`` (fixed MSAC threshold), ``min_score`` (hard score cut), ``min_views``
            (2), ``min_joints_per_view`` (None; int, or a fraction of J if < 1: camera-frames
            with fewer finite keypoints are dropped before triangulation), ``mask_out_of_bounds``
            (True: observations outside the image are not used; counted in ``out_of_image``),
            ``max_gap`` (10 frames; None = every interior gap) / ``max_gap_s`` (seconds, with
            ``times``): longest gap filled, ``max_point_err_px`` (None: triangulated points whose
            ``point_error`` exceeds it are removed before filling), ``cam_names`` (camera order of
            ``points``; reindexed to the group, missing cameras are NaN), ``bone_weight``
            ('auto' or a float, the 1.1 fixed weight was 2.0), ``bone_max_reproj_rise`` (0.5
            noise floors; None = never reject), ``n_iters``, ``chunk_frames``, ``verdict``, ...
        device: ``'auto'`` (cuda if available; ``optim_points`` chunks of fewer than
            ``OPTIM_GPU_MIN_CELLS`` frames x joints stay on the CPU, where they are faster),
            ``'cpu'``, ``'cuda'``.
        times: optional ``[N]`` frame times (s): gaps are interpolated on time and
            ``max_gap_s`` applies.
        segments: optional ``[N]`` segment ids (clips, windows, trials); a new segment starts
            wherever the id changes. Filling, smoothing, bones and chunks never cross a segment.
        extrinsics: optional per-frame world-to-camera matrices (moving cameras): ``[C,N,4,4]``
            or ``[N,C,4,4]`` (3x4 accepted; ``[C,N]`` when C == N), or a list of N camera
            groups; intrinsics come from ``cgroup``. ``bones`` / ``smooth`` are then skipped.

    Verdict rules:
        ``reconstructed_frac`` (triangulated (frame, joint) cells / all; warn < 0.5,
        fail < 0.05); ``inlier_rate`` (inlier / observed inside the image; warn < 0.7,
        fail < 0.3); ``cameras_unused`` (cameras with no observation or < 20 % of their
        observations used; warn > 0); ``error_drift`` (largest / smallest median error over 8 blocks
        of frames, the smallest at least 0.5 px: errors that change over the recording mean
        moved cameras (pass ``extrinsics=``) or broken sync; warn > 3);
        ``bone_cv`` (median robust bone-length CV 1.4826 MAD / median, with ``bones``; warn
        above max(0.1, the triangulated CV), fail above max(0.3, 2 x the triangulated CV)).
    """
    from .. import geometry as G
    from .. import triangulation as T
    opts = merge_options(_W6_DEFAULTS, options)
    timer = Timer()
    if fill not in ('interior', 'none'):
        raise ValueError("fill must be 'interior' or 'none' (reconstruct_3d never extrapolates), "
                         "got {!r}".format(fill))
    pts = np.array(to_numpy(points), dtype=float)
    if pts.ndim == 5:
        # identity axis (associate_views output C x F x P x J x 2): one reconstruction per identity
        return _reconstruct_identities(cgroup, pts, scores, bones, smooth, fill, robust, options,
                                       device, times, segments, extrinsics)
    squeeze = pts.ndim == 3
    if squeeze:
        pts = pts[:, :, None]
    if pts.ndim != 4 or pts.shape[-1] != 2:
        raise ValueError('points must be C x N x J x 2 or C x N x 2, got {}'.format(pts.shape))
    names = list(cgroup.get_names())
    cam_missing, cam_dropped = [], []
    if opts['cam_names'] is not None:
        pts, cam_missing, cam_dropped = reindex_cameras(pts, opts['cam_names'], names)
    C, N, J, _ = pts.shape
    if C != len(names):
        raise ValueError('points have {} cameras, the group has {}'.format(C, len(names)))
    sw = None
    if scores is not None:
        sw = np.array(to_numpy(scores), dtype=float)
        if opts['cam_names'] is not None:
            sw = reindex_cameras(sw.reshape((len(opts['cam_names']),) + sw.shape[1:]),
                                 opts['cam_names'], names)[0]
        if sw.size != C * N * J:
            raise ValueError('scores must be C x N x J = {}, got {}'.format((C, N, J), sw.shape))
        sw = np.clip(np.nan_to_num(sw.reshape(C, N, J), nan=0.0, posinf=1.0, neginf=0.0), 0, 1)
    tms = None
    if times is not None:
        tms = np.asarray(to_numpy(times), dtype=float).reshape(-1)
        if len(tms) != N:
            raise ValueError('times must have one value per frame ({}), got {}'.format(N, len(tms)))
    if opts['max_gap_s'] is not None and tms is None:
        raise ValueError("options['max_gap_s'] needs times=")
    runs = _segment_runs(N, segments)
    E = _parse_extrinsics(extrinsics, C, N)
    n_low_score = 0
    if opts['min_score'] is not None:
        if sw is None:
            raise ValueError("options['min_score'] needs scores=")
        low = (sw < float(opts['min_score'])) & np.isfinite(pts).all(-1)
        n_low_score = int(low.sum())
        pts[low] = np.nan
    n_sparse_views = 0
    if opts['min_joints_per_view'] is not None:
        k = float(opts['min_joints_per_view'])
        k = int(np.ceil(k * J)) if k < 1 else int(k)
        nj = np.isfinite(pts).all(-1).sum(-1)                    # [C,N]
        drop = (nj > 0) & (nj < k)
        n_sparse_views = int(drop.sum())
        pts[drop] = np.nan
    bones = [tuple(int(v) for v in b) for b in (bones or [])]
    for a, b in bones:
        if not (0 <= a < J and 0 <= b < J):
            raise ValueError('bone {} out of range for {} joints'.format((a, b), J))
    flat = pts.reshape(C, -1, 2)
    pin, affine_proj = _affine_group(cgroup, flat)
    work = pin if pin is not None else cgroup
    observed = np.isfinite(pts).all(-1)
    diag = image_diagonal(cgroup)
    cg = _group_on(work, device)
    P = G.pack(cg)
    oob = observed & ~to_numpy(T._valid_obs(P, torch.as_tensor(flat, device=P.device, dtype=P.dtype),
                                            torch.as_tensor(flat, device=P.device, dtype=P.dtype),
                                            True)).reshape(C, N, J)
    mob = bool(opts['mask_out_of_bounds'])
    usable = observed & ~oob if mob else observed
    swf = None if sw is None else sw.reshape(C, -1)
    Ef = None if E is None else torch.as_tensor(E, device=P.device, dtype=P.dtype)
    settings = {'weighting': 'soft score weights, clipped to [0, 1], on the squared pixel error '
                             '(MSAC cost, refit, Gauss-Newton; optim_points residual x sqrt(s))'
                             if sw is not None else 'none (no scores)',
                'min_score': opts['min_score'], 'robust': bool(robust), 'fill': fill,
                'image_diagonal_px': diag, 'min_views': int(opts['min_views']),
                'min_joints_per_view': opts['min_joints_per_view'],
                'mask_out_of_bounds': mob, 'max_gap': opts['max_gap'], 'max_gap_s': opts['max_gap_s'],
                'max_point_err_px': opts['max_point_err_px'], 'n_segments': len(runs),
                'per_frame_extrinsics': E is not None,
                'cameras': 'affine (exact telephoto pinholes, AffineCamera.to_pinhole)'
                           if pin is not None else 'packed'}
    fallbacks = []

    def project(Xnj):
        """[N,J,3] -> [C,N,J,2] pixels (NaN rows -> garbage, masked by the caller)."""
        Xc = np.nan_to_num(Xnj)
        if affine_proj is not None:
            return affine_proj(Xc.reshape(-1, 3)).reshape(C, N, J, 2)
        return to_numpy(G.project(P, torch.as_tensor(Xc, device=P.device, dtype=P.dtype),
                                  extrinsics=Ef)[0]).reshape(C, N, J, 2)

    with timer('triangulate'):
        if robust:
            tau0 = max(10.0, opts['first_tau_diag_frac'] * diag) if diag else 20.0
            kw = dict(scores=swf, min_inliers=opts['min_views'], mask_out_of_bounds=mob,
                      extrinsics=Ef)
            r0 = T.triangulate_robust(P, flat, tau=tau0, **kw)
            nf = noise_floor(r0.errors, r0.inliers, diag)
            # 3 x noise floor, at least the 1.0 default (10 px) and never looser than the first-pass
            # gate (1 % of the diagonal): on real keypoints a 20 x rule gave 60-190 px and let poor
            # cameras pull the points (motor observatory: 5 x -> 3.75 px median, 3 x -> 3.31 px,
            # project IRLS 3.07 px)
            tau = float(opts['tau_px']) if opts['tau_px'] is not None else \
                float(min(max(float(opts['tau_noise_mult']) * nf, 10.0), max(tau0, 10.0)))
            res = T.triangulate_robust(P, flat, tau=tau, **kw)
            X = to_numpy(res.p3d).reshape(N, J, 3)
            inl = to_numpy(res.inliers).reshape(C, N, J)
            settings.update(first_tau_px=tau0, noise_px=nf, noise_unit=_noise_unit(nf), tau_px=tau)
        else:
            w = None if swf is None else np.sqrt(swf)
            lin = np.where(usable.reshape(C, -1)[..., None], flat, np.nan)
            X = to_numpy(T.triangulate_linear(P, lin, weights=w, extrinsics=Ef,
                                              min_cams=opts['min_views'])).reshape(N, J, 3)
            inl = usable & np.isfinite(X).all(-1)[None]
            tau = None
            nf = None
    have = np.isfinite(X).all(-1)                       # triangulated cells
    if robust:
        inl = inl & have[None]

    def point_err(Xnj, inl_):
        e = np.linalg.norm(project(Xnj) - pts, axis=-1)
        e = np.where(inl_ & np.isfinite(Xnj).all(-1)[None], e, np.nan)
        with np.errstate(all='ignore'), _quiet():
            return np.nanmedian(e, axis=0), e
    n_point_err = 0
    if opts['max_point_err_px'] is not None and have.any():
        pe0, _ = point_err(X, inl)
        bad = have & (pe0 > float(opts['max_point_err_px']))
        n_point_err = int(bad.sum())
        X[bad] = np.nan
        inl = inl & ~bad[None]
        have = have & ~bad
    if nf is None:
        nf = noise_floor(point_err(X, inl)[1], diag=diag) if have.any() else MIN_NOISE_PX
        settings['noise_px'] = nf

    scale = _rig_scale(cgroup) if pin is None else float('nan')
    settings['rig_scale'] = scale
    optimise = bool(bones) or smooth is not None
    if optimise and E is not None:
        fallbacks.append('optim_points skipped: bones / smooth are not supported with per-frame '
                         'extrinsics')
        optimise = False
    keep = have.copy()
    if fill == 'interior':
        keep = _fill_keep(have, runs, None if opts['max_gap'] is None else int(opts['max_gap']),
                          tms, opts['max_gap_s'])
    n_chunks = n_rej = 0
    with timer('optim_points' if optimise else 'fill'):
        base = _interior_fill(X, keep, runs, tms)
        base[~keep] = np.nan
        if optimise and have.sum() >= 2:
            obs_in = np.where(inl[..., None], pts, np.nan)
            tau_o = tau if tau is not None else 10.0 * _noise_unit(nf)
            # optim_points is a loop of small steps: a GPU only pays off for large chunks
            cg_o = cg
            if device in (None, 'auto') and str(cg.device).startswith('cuda') and \
                    min(N, int(opts['chunk_frames'])) * J < OPTIM_GPU_MIN_CELLS:
                cg_o = _group_on(work, 'cpu')
            bone_w, bone_info = None, None
            if bones:
                if opts['bone_weight'] == 'auto':
                    unit, _ = cg_o._optim_unit(X, obs_in)
                    bone_w, bone_info = _auto_bone_weights(P, X, inl, bones, nf, unit)
                else:
                    bone_w = float(opts['bone_weight'])
            rise = None if opts['bone_max_reproj_rise'] is None else \
                float(opts['bone_max_reproj_rise']) * nf

            def med_err(Xc, s, e):
                Xs = np.full((N, J, 3), np.nan)
                Xs[s:e] = np.where(have[s:e, :, None], Xc, np.nan)
                ee = np.linalg.norm(project(Xs) - pts, axis=-1)[:, s:e]
                m = inl[:, s:e] & have[None, s:e]
                return float(np.median(ee[m])) if m.any() else float('nan')
            Xo = np.full((N, J, 3), np.nan)
            # bones and reprojection terms are per frame; smoothing is masked at segment boundaries
            # (optim_points segments=) and filled cells are always inside a segment (_fill_keep)
            seg_ids = np.zeros(N, int)
            for k, (s0, e0) in enumerate(runs):
                seg_ids[s0:e0] = k
            for s0, e0 in [(0, N)]:
                if not have[s0:e0].any():
                    continue
                Xs, nc, nr = _optim_chunked(
                    cg_o, obs_in[:, s0:e0], X[s0:e0], None if sw is None else sw[:, s0:e0], bones,
                    0.0 if smooth is None else float(smooth), opts, tau_o, keep[s0:e0],
                    bone_w=bone_w, segments=seg_ids[s0:e0] if len(runs) > 1 else None,
                    med_err=(lambda Xc, s, e, s0=s0: med_err(Xc, s0 + s, s0 + e)),
                    max_rise=rise, base=base[s0:e0])
                Xo[s0:e0] = Xs
                n_chunks += nc
                n_rej += nr
            final = Xo
            if n_rej:
                fallbacks.append('optim_points: {} of {} chunks rejected (median inlier error rose '
                                 'by more than {:.3g} px); triangulated points kept there'.format(
                                     n_rej, n_chunks, rise))
            settings['optim_points'] = {
                'bones': bones, 'bone_weight': opts['bone_weight'],
                'bone_weights': None if bone_w is None else np.atleast_1d(bone_w).tolist(),
                'bone_spread': bone_info,
                'scale_smooth': 0.0 if smooth is None else float(smooth),
                'reproj_error_threshold_px': tau_o, 'n_iters': int(opts['n_iters']),
                'max_lr_px': opts['max_lr'], 'normalize': True,
                'chunk_frames': int(opts['chunk_frames']), 'overlap_frames': int(opts['overlap_frames']),
                'n_chunks': n_chunks, 'n_rejected_chunks': n_rej, 'max_reproj_rise_px': rise,
                'bone_targets': 'median length over the frames where both joints are triangulated'}
        else:
            if optimise:
                fallbacks.append('optim_points skipped: fewer than 2 triangulated cells')
            final = base if fill == 'interior' else X.copy()
            final[~keep] = np.nan
    filled = np.isfinite(final).all(-1) & ~have
    if not optimise:
        final[have] = X[have]

    # errors of every observation against the final points
    err = np.linalg.norm(project(final) - pts, axis=-1)
    err[~np.isfinite(final).all(-1)[None].repeat(C, 0)] = np.nan
    n_views = np.where(have, inl.sum(0), 0)
    with np.errstate(all='ignore'), _quiet():
        pe = np.nanmedian(np.where(inl & have[None], err, np.nan), axis=0)

    # report
    per_camera = {}
    unused = []
    for c, nm in enumerate(names):
        n_obs = int(usable[c].sum())
        n_in = int(inl[c].sum())
        e = err[c][inl[c]]
        per_camera[nm] = {'n_obs': n_obs, 'n_inliers': n_in, 'n_out_of_image': int(oob[c].sum()),
                          'usage': n_in / n_obs if n_obs else 0.0,
                          'median_err_px': float(np.median(e)) if e.size else None}
        if n_obs == 0 or n_in < 0.2 * n_obs:
            unused.append(nm)
    gaps = _gap_stats(have)
    report = {
        'per_camera': per_camera,
        'cameras_unused': unused,
        'cameras_missing': cam_missing, 'cameras_dropped': cam_dropped,
        'gaps': {'per_joint': gaps,
                 'missing_frac': float(1 - have.mean()) if have.size else 1.0,
                 'filled_frac': float(filled.mean()) if filled.size else 0.0,
                 'longest_gap': max([g['longest_gap'] for g in gaps], default=0),
                 'longest_filled': int(max([g['longest_gap'] for g in _gap_stats(~filled)], default=0))
                 if filled.any() else 0,
                 'extrapolated': 0},
        'n_out_of_image': int(oob.sum()),
        'n_removed': {'min_score': n_low_score, 'min_joints_per_view': n_sparse_views,
                      'max_point_err_px': n_point_err},
        'noise_px': settings.get('noise_px'), 'tau_px': tau,
        'median_err_px': float(np.nanmedian(err[inl])) if inl.any() else None,
        'median_point_error_px': float(np.nanmedian(pe)) if np.isfinite(pe).any() else None,
        'two_view_frac': float(((n_views == 2) & have).sum() / max(have.sum(), 1)),
    }
    drift = _error_drift(err, usable & have[None])
    report['error_drift'] = drift
    rules = [
        Rule('reconstructed_frac', float(have.mean()) if have.size else 0.0, 0.5, 0.05,
             'share of (frame, joint) cells triangulated from >= {} views'.format(opts['min_views']),
             higher_is_worse=False),
        Rule('inlier_rate', float(inl.sum() / max(usable.sum(), 1)), 0.7, 0.3,
             'share of the observations (inside the image) used by the triangulation',
             higher_is_worse=False),
        Rule('cameras_unused', len(unused), 0, None,
             'cameras with no observations or < 20 % of them used: {}'.format(unused)),
        Rule('error_drift', drift, 3.0, None,
             'reprojection error changes over the recording (largest / smallest of 8 blocks): moved '
             'cameras (pass extrinsics= for moving cameras) or broken sync'),
    ]
    if bones:
        cv_before = _bone_cv(X, bones)
        cv_after = _bone_cv(final, bones)
        med = float(np.nanmedian(cv_after)) if np.isfinite(cv_after).any() else float('nan')
        med_tri = float(np.nanmedian(cv_before)) if np.isfinite(cv_before).any() else None
        report['bones'] = {'pairs': bones, 'cv': cv_after, 'cv_triangulated': cv_before,
                           'median_cv': med, 'median_cv_triangulated': med_tri,
                           'cv_std': _bone_cv_std(final, bones),
                           'cv_definition': '1.4826 x MAD / median of each bone length'}
        ref = med_tri if med_tri is not None and np.isfinite(med_tri) else 0.0
        rules.append(Rule('bone_cv', med, max(0.1, ref), max(0.3, 2 * ref),
                          'median robust bone-length CV (vs {:.3g} triangulated)'.format(ref)))
    if not have.any():
        rules.append(Rule('no_points', True, fail_at=True,
                          explain='no point is seen by {} cameras'.format(opts['min_views'])))
    verdict, reasons, table = evaluate_rules(rules, opts['verdict'])
    report['rules'] = table
    out = final[:, 0] if squeeze else final
    sq = (lambda a: a[..., 0]) if squeeze else (lambda a: a)
    return Reconstruction(
        points3d=out, errors=sq(err), n_views=sq(n_views), inliers=sq(inl), filled=sq(filled),
        point_error=sq(pe), out_of_image=sq(oob),
        report=report, verdict=verdict, reasons=reasons, settings=settings,
        provenance=provenance('reconstruct_3d',
                              inputs={'shape': [C, N, J], 'cameras': names,
                                      'scores': scores is not None, 'bones': len(bones),
                                      'smooth': smooth, 'times': times is not None,
                                      'segments': segments is not None,
                                      'extrinsics': extrinsics is not None},
                              fallbacks=fallbacks, timings=timer.times))


class _quiet:
    def __enter__(self):
        import warnings
        self._w = warnings.catch_warnings()
        self._w.__enter__()
        warnings.simplefilter('ignore', RuntimeWarning)

    def __exit__(self, *a):
        self._w.__exit__(*a)


# ================================================================== W7 associate_views

@dataclass
class Association(WorkflowResult):
    """Result of :func:`associate_views`.

    ``points [C,F,P,J,2]`` per identity (frame axis: ``frame_ids``; NaN where unmatched),
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
    'gate_noise_mult': 10.0,    # second-pass gate: this x noise floor (1.1: 80, which never tightened on real data)
    'min_gate_px': 10.0,
    'verify_gate_mult': 2.5,    # tracklets: verify_floor = this x gate
    'min_score': None,          # keypoint score cut (third channel)
    'min_points': 4,            # finite keypoints for a detection to count
    'min_len': 10,              # tracklets: shortest tracklet (frames)
    'link_frac': 0.05,          # per-frame: max 3D jump between frames, x rig scale
    'link_max_gap': 30,         # frames an identity may go unseen and still continue (frame ids)
    'link_max_gap_s': None,     # same in seconds (needs fps)
    'fps': None,                # frame rate (Hz) of the frame ids, for link_max_gap_s
    'min_identity_frames': 5,   # per-frame: identities seen in fewer frames are dropped (labels -1)
    'n_identities': None,       # known number of animals: at most this many per frame, and new
                                # instances re-enter the nearest dormant identity instead of a new one
    'split_on_switch': True,    # tracklets: split a tracklet where the verification drops a run of
                                # its frames (an identity switch), associate again
    'switch_min_frames': 10,    # shortest dropped run treated as a switch (3 over-split a real IoU tracker)
    'stitch': True,             # tracklets: join identities that follow each other in time and space
    'frame_axis': 'auto',       # 'dense' (axis = frame id 0..max), 'sparse' (frames with detections,
                                # see frame_ids), 'auto' (sparse when dense would be > 2x larger and > 1000 frames)
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


def _link_instances(frames_inst, gate, max_gap, frame_ids=None, n_identities=None):
    """frames_inst: list over frames of [M_f,3] instance points -> list of [M_f] identity ids.
    ``frame_ids`` (default 0..F-1): the gap test uses frame-id differences. ``n_identities``:
    once that many identities exist, unmatched instances re-enter the nearest identity not seen
    in this frame (no gate) instead of starting a new one."""
    from scipy.optimize import linear_sum_assignment
    if frame_ids is None:
        frame_ids = np.arange(len(frames_inst))
    last_pos, last_f, n_id = [], [], 0
    out = []
    for k, X in enumerate(frames_inst):
        f = int(frame_ids[k])
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
        if n_identities and n_id >= n_identities:
            free = [i for i in range(n_id) if i not in set(ids.tolist())]
            rest = [m for m in range(len(X)) if ids[m] < 0 and okX[m]]
            if free and rest:
                A = np.array([last_pos[i] for i in free])
                D = np.linalg.norm(X[rest][:, None] - A[None], axis=-1)
                r, c = linear_sum_assignment(np.nan_to_num(D, nan=1e30, posinf=1e30))
                for a, b in zip(r, c):
                    ids[rest[a]] = free[b]
        for m in range(len(X)):
            if ids[m] < 0 and okX[m] and not (n_identities and n_id >= n_identities):
                ids[m] = n_id
                last_pos.append(X[m]); last_f.append(f)
                n_id += 1
            elif ids[m] >= 0:
                last_pos[ids[m]] = X[m]; last_f[ids[m]] = f
        out.append(ids)
    return out, n_id


def _split_tracks(track_ids, names, groups, min_run):
    """New per-camera track ids: each member tracklet of a group is cut where the group's
    verification dropped a run of >= ``min_run`` of its frames (an identity switch)."""
    from .. import matching as M
    per = M._per_camera(track_ids, names, 'track_ids')
    cuts = {}         # (c, track) -> sorted frame ids where a new piece starts
    for g in groups:
        for c, fr in (g.dropped or {}).items():
            fr = np.asarray(sorted(int(x) for x in fr))
            if len(fr) < min_run:
                continue
            br = np.nonzero(np.diff(fr) > 1)[0]
            starts, ends = np.r_[fr[0], fr[br + 1]], np.r_[fr[br], fr[-1]]
            for s, e in zip(starts, ends):
                if e - s + 1 < min_run:
                    continue
                for cc, t in g.members:
                    if cc == c:
                        cuts.setdefault((c, t), set()).update({int(s), int(e) + 1})
    if not cuts:
        return None, 0
    out = []
    for c, dc in enumerate(per):
        items = []
        top = -1
        for f, v in M._frame_items(dc):
            ids = np.asarray(to_numpy(v)).reshape(-1).astype(np.int64)
            items.append((int(f), ids))
            if len(ids):
                top = max(top, int(ids.max()))
        fresh = {}
        new = {}
        for f, ids in items:
            res = ids.copy()
            for k, t in enumerate(ids.tolist()):
                cs = cuts.get((c, int(t)))
                if cs:
                    piece = int(np.searchsorted(np.array(sorted(cs)), f, side='right'))
                    if piece:          # piece 0 keeps the original id
                        key = (int(t), piece)
                        if key not in fresh:
                            top += 1
                            fresh[key] = top
                        res[k] = fresh[key]
            new[f] = res
        out.append(new)
    return out, sum(len(v) for v in cuts.values())


def _stitch(cents, link_gate, max_gap):
    """cents: list over identities of (frame ids [n], centroids [n,3]) -> identity -> new id.
    Identities that do not overlap in time are joined when one starts within ``max_gap`` frames
    of the other's end and within ``link_gate`` of its last position (greedy, closest first; a
    limit growing with the gap joined wrong animals on 3dpop)."""
    P = len(cents)
    span = [(int(f[0]), int(f[-1])) if len(f) else (0, -1) for f, _ in cents]
    cand = []
    for a in range(P):
        for b in range(P):
            if a == b or not len(cents[a][0]) or not len(cents[b][0]):
                continue
            gap = span[b][0] - span[a][1]
            if 0 < gap <= max_gap:
                d = float(np.linalg.norm(cents[a][1][-1] - cents[b][1][0]))
                if d <= link_gate:
                    cand.append((d, a, b))
    root = list(range(P))

    def find(x):
        while root[x] != x:
            x = root[x]
        return x
    nxt, prv = {}, {}
    members = {p: [p] for p in range(P)}
    for d, a, b in sorted(cand):
        if a in nxt or b in prv:
            continue
        ra, rb = find(a), find(b)
        if ra == rb:
            continue
        # no temporal overlap between the joined chains
        fa = np.concatenate([cents[m][0] for m in members[ra]])
        fb = np.concatenate([cents[m][0] for m in members[rb]])
        if np.intersect1d(fa, fb).size:
            continue
        nxt[a], prv[b] = b, a
        root[rb] = ra
        members[ra] += members.pop(rb)
    roots = sorted({find(p) for p in range(P)})
    new = {r: i for i, r in enumerate(roots)}
    return [new[find(p)] for p in range(P)], P - len(roots)


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
        options: overrides of ``_W7_DEFAULTS``: ``gate_noise_mult`` (10), ``link_max_gap`` (30
            frame ids) or ``link_max_gap_s`` (seconds, with ``fps``), ``min_identity_frames`` (5;
            per-frame mode), ``n_identities`` (known number of animals), ``split_on_switch``
            (True; tracklets), ``stitch`` (True; tracklets), ``frame_axis`` ('auto', 'dense',
            'sparse'), ...

    Gates: a first pass with ``gate_diag_frac x image diagonal`` (>= ``min_gate_px``); the noise
    floor of the associated observations then gives ``gate_noise_mult x noise`` and, if that is
    tighter, a second pass. Per-frame linking accepts 3D jumps up to ``link_frac x rig scale``
    between frames at most ``link_max_gap`` frame ids apart; with tracklets, identities that
    follow each other within the same limits are stitched (``stitch``), and tracklets whose
    frames the verification drops in a run (an identity switch inside the tracklet) are cut
    there and associated again (``split_on_switch``).

    Output frame axis: ``points[:, i]`` / ``labels[c][i]`` are the frame ``frame_ids[i]``.
    With ``frame_axis='dense'`` (and 'auto' for compact ids) ``frame_ids = 0..max id``;
    'sparse' keeps only the frames with detections (ids around 1e5 would otherwise take
    hundreds of MB).

    Verdict rules: ``no_identities`` (fail); ``assigned_frac`` (detections assigned to an
    identity; warn < 0.5, fail < 0.1); ``fragmentation`` (per-frame mode: identities / most
    simultaneous instances; warn > 2); ``verify_dropped_frac`` (tracklets: share of the
    associated camera-frames the verification dropped, a sign of identity switches inside
    tracklets; warn > 0.05).
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
    if opts['frame_axis'] not in ('auto', 'dense', 'sparse'):
        raise ValueError("options['frame_axis'] must be 'auto', 'dense' or 'sparse'")
    if opts['link_max_gap_s'] is not None and not opts['fps']:
        raise ValueError("options['link_max_gap_s'] needs options['fps']")
    max_gap = int(round(float(opts['link_max_gap_s']) * float(opts['fps']))) \
        if opts['link_max_gap_s'] is not None else int(opts['link_max_gap'])
    diag = image_diagonal(cgroup)
    gate1 = max(float(opts['min_gate_px']), opts['gate_diag_frac'] * diag) if diag else 40.0
    min_views = max(1, int(min_views))
    n_ident = None if opts['n_identities'] is None else int(opts['n_identities'])
    settings = {'image_diagonal_px': diag, 'first_gate_px': gate1, 'min_views': min_views,
                'mode': 'tracklets' if track_ids is not None else 'per_frame',
                'min_score': opts['min_score'], 'min_points': int(opts['min_points']),
                'link_max_gap_frames': max_gap, 'n_identities': n_ident}
    P = G.pack(cgroup)
    F_dense = int(fids.max()) + 1 if len(fids) else 0
    axis = opts['frame_axis']
    if axis == 'auto':
        axis = 'sparse' if F_dense > max(1000, 2 * len(fids)) else 'dense'
    out_ids = np.arange(F_dense) if axis == 'dense' else fids
    F = len(out_ids)
    fpos = {int(f): i for i, f in enumerate(out_ids)}
    settings['frame_axis'] = axis
    n_dets_total = 0
    for c in range(C):
        for f, d in parsed[c].items():
            ok = np.isfinite(_clean(d, opts['min_score'])).all(-1).sum(-1) >= int(opts['min_points'])
            n_dets_total += int(ok.sum())
    labels = [[np.full(len(parsed[c][int(f)]) if int(f) in parsed[c] else 0, -1, dtype=np.int64)
               for f in out_ids] for c in range(C)]
    rules_extra = []
    scale = _rig_scale(cgroup)
    link_gate = opts['link_frac'] * scale
    settings.update(link_gate=link_gate, link_frac=opts['link_frac'], rig_scale=scale)

    if track_ids is not None:
        def run(gate, tids):
            return M.associate_tracklets(
                cgroup, detections, tids, min_cams=max(2, min_views), max_pair_error=gate,
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
        tids = track_ids
        n_cuts = 0
        with timer('associate'):
            groups = run(gate1, tids)
            nf = noise_of(groups)
            gate = gate1
            if nf is not None:
                g2 = float(np.clip(opts['gate_noise_mult'] * nf, opts['min_gate_px'], gate1))
                if g2 < 0.75 * gate1:
                    gate = g2
                    groups = run(gate, tids)
            if opts['split_on_switch']:
                new, n_cuts = _split_tracks(tids, names, groups, int(opts['switch_min_frames']))
                if new is not None:
                    tids = new
                    groups = run(gate, tids)
        settings['tracklet_cuts'] = n_cuts
        P0 = len(groups)
        obs_g = [[g.observations(c) for c in range(C)] for g in groups]
        ident = list(range(P0))
        n_stitched = 0
        if opts['stitch'] and P0 > 1:
            with timer('stitch'):
                cents = []
                for gi in range(P0):
                    fr_all = np.unique(np.concatenate([np.asarray(o[0]) for o in obs_g[gi]
                                                       if o[2] is not None] or [np.zeros(0, int)]))
                    if not len(fr_all):
                        cents.append((fr_all, np.zeros((0, 3))))
                        continue
                    sel = np.unique(np.r_[fr_all[:3], fr_all[-3:]])
                    pts = np.full((C, len(sel), K, 2), np.nan)
                    for c, (fr, di, xyz) in enumerate(obs_g[gi]):
                        if xyz is None:
                            continue
                        for k, f in enumerate(sel):
                            hit = np.nonzero(np.asarray(fr) == f)[0]
                            if len(hit):
                                pts[c, k] = _clean(np.asarray(xyz)[hit[0]][None], opts['min_score'])[0]
                    X = to_numpy(T.triangulate_linear(P, pts.reshape(C, -1, 2))).reshape(len(sel), K, 3)
                    with np.errstate(all='ignore'), _quiet():
                        ce = np.nanmedian(X, axis=1)
                    ok = np.isfinite(ce).all(-1)
                    cents.append((sel[ok], ce[ok]))
                # the centroids of each identity's first and last 3 frames decide the joins
                ident, n_stitched = _stitch(cents, link_gate, max_gap)
        P_ = (max(ident) + 1) if ident else 0
        points = np.full((C, F, P_, K, 2), np.nan)
        frames_of = [set() for _ in range(P_)]
        cams_of = [set() for _ in range(P_)]
        for gi in range(P0):
            p = ident[gi]
            for c in range(C):
                fr, di, xyz = obs_g[gi][c]
                if xyz is None:
                    continue
                pos = np.array([fpos[int(f)] for f in fr], dtype=np.int64)
                points[c, pos, p] = _clean(np.asarray(xyz), opts['min_score'])
                for f, i in zip(fr, di):
                    labels[c][fpos[int(f)]][int(i)] = p
                frames_of[p].update(int(f) for f in fr)
                cams_of[p].add(names[c])
        identities = [{'frames': len(frames_of[p]), 'cameras': sorted(cams_of[p])} for p in range(P_)]
        info = groups.info
        n_cells = sum(len(np.asarray(o[0])) for og in obs_g for o in og if o[2] is not None)
        drop_frac = float(info.get('n_verify_dropped', 0)) / max(n_cells + info.get('n_verify_dropped', 0), 1)
        extra = {'tracklets': _jsonable_info(info), 'n_tracklet_cuts': int(n_cuts),
                 'n_stitched': int(n_stitched), 'n_groups': int(P0)}
        rules_extra.append(Rule('verify_dropped_frac', drop_frac, 0.05, None,
                                'share of associated camera-frames the verification dropped '
                                '(identity switches inside tracklets?)'))
        backend = groups
    else:
        # only the frames with detections are associated (frame ids may be sparse, e.g. 1e5 apart)
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
                with np.errstate(invalid='ignore'), _quiet():
                    cen = np.nanmean(np.where(ok[..., None], xy, np.nan), axis=1)
                cen[ok.sum(-1) < int(opts['min_points'])] = np.nan
                per.append(cen)
            cent.append(per)

        def run(gate):
            return M.associate_points_frames(P, cent, max_res_px=gate, min_views=max(2, min_views),
                                             max_instances=n_ident or 0)
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
        with timer('link'):
            ids, P0 = _link_instances([np.asarray(a.points3d, float).reshape(-1, 3) for a in assoc],
                                      link_gate, max_gap, frame_ids=fids, n_identities=n_ident)
        # short-lived identities (ghost pairs, 1-3 frames) are dropped
        cnt = np.zeros(P0, int)
        for idf in ids:
            for p in idf[idf >= 0]:
                cnt[p] += 1
        keep_id = cnt >= int(opts['min_identity_frames'] or 0)
        remap = np.full(P0, -1, dtype=np.int64)
        remap[keep_id] = np.arange(int(keep_id.sum()))
        P_ = int(keep_id.sum())
        points = np.full((C, F, P_, K, 2), np.nan)
        n_frames_id = np.zeros(P_, int)
        cams_id = [set() for _ in range(P_)]
        n_inst_kept = []
        for f, a, idf in zip(fids, assoc, ids):
            fi = fpos[int(f)]
            kept = 0
            for m, mem in enumerate(a.members):
                p = int(remap[idf[m]]) if idf[m] >= 0 else -1
                if p < 0:
                    continue
                kept += 1
                n_frames_id[p] += 1
                for c, i in mem.items():
                    points[c, fi, p] = _clean(parsed[c][int(f)][int(i)][None], opts['min_score'])[0]
                    labels[c][fi][int(i)] = p
                    cams_id[p].add(names[c])
            n_inst_kept.append(kept)
        identities = [{'frames': int(n_frames_id[p]), 'cameras': sorted(cams_id[p])}
                      for p in range(P_)]
        most = max(n_inst_kept, default=0)
        frag = P_ / most if most else float('inf')
        settings.update(min_identity_frames=int(opts['min_identity_frames'] or 0))
        extra = {'max_simultaneous': int(most), 'fragmentation': frag,
                 'n_short_identities_dropped': int((~keep_id).sum())}
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
    return Association(points=points, labels=labels, groups=backend, frame_ids=out_ids,
                       report=report, verdict=verdict, reasons=reasons, settings=settings,
                       provenance=provenance('associate_views',
                                             inputs={'cameras': names, 'n_frames': int(len(fids)),
                                                     'max_frame_id': int(F_dense - 1),
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
    'audio_min_overlap_frac': 0.25,   # and at least this share of the shorter clip (long clips: no edge lags)
    'audio_windows': 6,           # windowed re-estimates (sigma, drift)
    'audio_cross_check': True,    # spectrogram + onset cross-checks of the gcc estimate
    'audio_strategy': 'auto',     # 'global' | 'windowed' | 'auto' (= 'windowed' for clips > audio_windowed_min_s)
    'audio_windowed_min_s': 300.0,
    'audio_window_s': 20.0,       # windowed strategy: shortest window (s)
    'cross_check_frames': 1.0,    # disagreement tolerance, in frame periods (or 1/30 s)
    'fps': None,                  # frame rate (Hz, or {stream: Hz}) for the frame-unit verdicts; None = from the data
    'max_time_jitter': 0.05,      # points with per-frame times: share of irregular frame steps before the rate comes from the mean step (warning)
    'stream_camera': None,        # {stream: camera name}: clips of one camera (default: stream name = camera name)
    'motion_max_lag': 5.0,        # s
    'epipolar_F': 'auto',         # 'calibrated' (cgroup) | 'self' (F fitted per offset) | 'both' | 'auto'
    'epipolar_search_frames': 10.0,   # half-width of the epipolar search (frames)
    'epipolar_boot': 20,
    'drift_min_z': 3.0,           # a fitted clock rate below this x its sigma counts as no drift (rate 0)
    'max_residual_frames': 2.0,   # offset graph: drop edges with residual above this (frames)
    'refine_accel_frac': 10.0,    # STBA motion prior: accel_sigma = this x rig scale / s^2
    'refine_offset_sigma_min': 1e-3,   # s, floor on the STBA offset prior sigma
    'refine_rate': 'auto',        # STBA clock rate: 'free' | 'seed' (keep the audio/seed rate) | 'auto'
    'refine_free_rate_min_s': 60.0,   # 'auto': free the rate only with a drift fit and points spanning this long
    'refine_rate_sigma': 1e-3,
    'refine_max_iter': 15,
    'refine_extrinsics': False,
    'verdict': None,
}


def _is_path(v):
    return isinstance(v, (str, bytes)) or hasattr(v, '__fspath__')


def _media(path):
    from .. import sync as S
    try:
        return S.media_info(str(path))
    except Exception:  # pragma: no cover - defensive
        return {}


def _load_stream(name, v, sr, decode_audio=True, max_jitter=0.05):
    """-> dict(audio=(sig, sr, t0)|None, motion=(sig, rate, t0)|None, points=(p2d, rate, t0)|None,
    meta={...})."""
    from .. import sync as S
    out = {'audio': None, 'motion': None, 'points': None, 'meta': {}}
    if _is_path(v):
        v = {'audio': str(v)}
    if not isinstance(v, dict):
        raise ValueError('stream {!r}: expected a video/audio path or a dict with audio= / '
                         'motion= / points=, got {}'.format(name, type(v).__name__))
    unknown = set(v) - {'audio', 'motion', 'points', 'audio_t0', 'motion_t0', 'fps'}
    if unknown:
        raise ValueError('stream {!r}: unknown keys {}'.format(name, sorted(unknown)))
    meta = out['meta']
    if v.get('fps') is not None:
        meta['fps'] = float(v['fps'])
    a = v.get('audio')
    if a is not None:
        t0 = v.get('audio_t0')
        meta['audio_t0_from'] = 'given' if t0 is not None else 'default 0'
        if _is_path(a):
            info = _media(a)
            meta['media'] = info
            if info.get('avg_frame_rate') and 'fps' not in meta:
                meta['fps_container'] = info['avg_frame_rate']
            if t0 is None and info.get('audio_start') is not None:
                # source time of audio sample 0 = audio stream start relative to the first video frame
                t0 = float(info['audio_start']) - float(info.get('video_start') or 0.0)
                meta['audio_t0_from'] = 'container (audio start - video start)'
            if decode_audio:
                try:
                    sig = S.load_audio(str(a), sr=sr)
                except ImportError as e:
                    raise ImportError('stream {!r}: decoding audio needs the [sync] extra '
                                      '(soundfile / av) or ffmpeg: {}'.format(name, e))
                a = (sig, sr)
            else:
                a = None
        if a is not None:
            sig, asr = a
            out['audio'] = (np.asarray(to_numpy(sig), float).ravel(), float(asr), float(t0 or 0.0))
        meta['audio_t0'] = float(t0 or 0.0)
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
            d = np.diff(t)
            med = float(np.median(d)) if len(d) else 0.0
            irr = 0.0
            if med > 0 and len(d) >= 2:
                r = d / med
                # steps that are not whole multiples of the median (or shorter than it), or so many
                # long steps that they cannot be dropped frames: irregular sampling (e.g. container pts
                # of 20 fps video in a 30 fps time base: 2, 1, 2, 1 ticks)
                odd = (np.abs(r - np.round(r)) > 0.05) | (r < 0.95)
                irr = max(float(odd.mean()), float((r > 1.5).mean()) if (r > 1.5).mean() > 0.2 else 0.0)
            meta['time_jitter'] = irr
            regular = irr <= max_jitter or len(t) < 3
            rate = 1.0 / med if regular else (len(t) - 1) / float(t[-1] - t[0])
            meta['points_rate_from'] = 'median step' if regular else 'mean step (irregular times)'
            t0 = float(t[0])
        out['points'] = (p2d, rate, t0)
    if out['audio'] is None and out['motion'] is None and out['points'] is None and \
            not (a is None and v.get('audio') is not None and not decode_audio):
        raise ValueError('stream {!r} has no usable signal'.format(name))
    return out


def _resample(x, sr, sr_to):
    if sr == sr_to:
        return x
    from fractions import Fraction
    from scipy.signal import resample_poly
    fr = Fraction(sr_to / sr).limit_denominator(1000)
    return resample_poly(x, fr.numerator, fr.denominator)


def _resample_points(p, rate, rate_to):
    """[T,J,2] frames at ``rate`` -> frames at ``rate_to`` (linear, NaN-aware; same start time)."""
    T = p.shape[0]
    if T < 2:
        return p
    tn = np.arange(0, (T - 1) / rate + 1e-12, 1.0 / rate_to) * rate     # in source frames
    i0 = np.clip(np.floor(tn).astype(int), 0, T - 1)
    i1 = np.clip(i0 + 1, 0, T - 1)
    w = (tn - i0)[:, None, None]
    return (1 - w) * p[i0] + w * p[i1]


def _edge(est, pair, offset_shift=0.0, sigma=None):
    """Copy of an OffsetEstimate in seconds for the graph (pair, shifted to source times)."""
    import dataclasses
    kw = dict(offset=float(est.offset) + offset_shift, pair=pair)
    if sigma is not None:
        kw['sigma'] = float(sigma)
    return dataclasses.replace(est, **kw)


def _timemap_parts(tm):
    """TimeMap -> (offset at global 0, rate); knots are dropped."""
    rate = 1.0 / tm.beta - 1.0
    return -tm.c / tm.beta, rate


def _windows_of(est):
    if est.diagnostics.get('windows_failed'):
        return None
    wt, wo = est.diagnostics.get('window_times'), est.diagnostics.get('window_offsets')
    if wt is None or wo is None or len(wo) < 3:
        return None
    return np.asarray(wt, float), np.asarray(wo, float)


@thread_guard
def estimate_offsets(streams, cgroup=None, reference=None, drift=True, refine=False,
                     options=None, rng=0, device='auto', seed=None):
    """W5: time offsets between unsynchronised streams. See the module docstring.

    Args:
        streams: ``{name: video/audio path | dict}``; the dict may hold ``audio=(signal, sr)``
            or a path (decoded with ``sync.load_audio``, ``[sync]`` extra; ``audio_t0`` = source
            time of sample 0: for paths it is read from the container (audio stream start minus
            video stream start, ``sync.media_info``), else 0), ``motion=(signal, rate)``
            (``motion_t0``), ``points=(p2d [T,J,2], t)`` with ``t`` the source time of every
            frame or a scalar frame rate (Hz), and ``fps`` (the stream's frame rate). With
            ``cgroup``, stream names must be camera names (or mapped by
            ``options['stream_camera']``) for the epipolar estimates (and STBA).
        cgroup: optional ``CameraGroup``: calibrated epipolar estimates between camera pairs with
            points.
        reference: anchor stream (offset 0); default the best-connected one.
        drift: fit a per-stream clock rate (``sync.fit_drift``) on the windowed audio estimates.
        refine: run ``stba.SpatiotemporalBA`` on the points (needs ``cgroup`` and points in
            >= 2 cameras) with offset priors anchored to the graph estimates.
        options: overrides of ``_W5_DEFAULTS``: ``audio_strategy`` ('auto': clips longer than
            ``audio_windowed_min_s`` = 300 s use a coarse spectrogram lag plus gcc in windows and a
            robust line through them, ``sync.audio_offset_windowed``; 'global'; 'windowed'),
            ``audio_min_overlap_frac`` (0.25 of the shorter clip), ``fps`` (frame rate for the
            frame-unit verdicts; default: from the points / motion, else the video's
            ``avg_frame_rate``, else 30), ``epipolar_F`` ('auto' = the calibrated F with ``cgroup``,
            else F fitted per offset; 'calibrated', 'self', 'both'), ``stream_camera``
            ({stream: camera}: several clips of one camera), ``refine_rate`` ('auto': the STBA
            frees the clock rate only when a drift fit exists and the points span at least
            ``refine_free_rate_min_s`` = 60 s; 'free'; 'seed' keeps the given rate), ...
        rng: seed for the bootstrap resamples.
        device: ``'auto'`` (cuda if available), ``'cpu'`` or ``'cuda'``: where the STBA
            refinement runs.
        seed: optional previous result (``OffsetEstimation``) or ``{stream: sync.TimeMap}``: the
            pair estimates, graph and drift are skipped (audio is not decoded) and these time
            maps are used as they are, or refined with ``refine=True``.

    Verdict rules: ``unidentifiable_streams`` (streams not connected to the reference; fail);
    ``max_sigma_frames`` (largest offset sigma in frame periods; warn > 0.25, fail > 1);
    ``cycle_rms_frames`` (graph inconsistency; warn > 0.25, fail > 1); ``method_disagreements``
    (audio methods disagreeing beyond ``cross_check_frames``; warn > 0); ``drift_unidentifiable``
    (with ``drift``: streams with audio whose rate could not be fitted, also when no windowed
    estimate exists; warn > 0); ``irregular_times`` (streams whose per-frame times jitter more
    than ``max_time_jitter``; warn > 0).
    """
    from .. import sync as S
    opts = merge_options(_W5_DEFAULTS, options)
    timer = Timer()
    if not isinstance(streams, dict) or len(streams) < 2:
        raise ValueError('streams must be a dict of >= 2 streams {name: path | dict}')
    if opts['audio_strategy'] not in ('auto', 'global', 'windowed'):
        raise ValueError("options['audio_strategy'] must be 'auto', 'global' or 'windowed'")
    if opts['epipolar_F'] not in ('auto', 'calibrated', 'self', 'both'):
        raise ValueError("options['epipolar_F'] must be 'auto', 'calibrated', 'self' or 'both'")
    if opts['refine_rate'] not in ('auto', 'free', 'seed'):
        raise ValueError("options['refine_rate'] must be 'auto', 'free' or 'seed'")
    names = [str(n) for n in streams]
    seed_maps = None
    if seed is not None:
        seed_maps = dict(seed.timemaps) if hasattr(seed, 'timemaps') else dict(seed)
    with timer('load'):
        data = {str(n): _load_stream(str(n), v, opts['audio_sr'], decode_audio=seed is None,
                                     max_jitter=float(opts['max_time_jitter']))
                for n, v in streams.items()}
    if reference is not None and str(reference) not in data:
        raise ValueError('reference {!r} is not a stream ({})'.format(reference, names))
    cam_names = list(cgroup.get_names()) if cgroup is not None else []
    sc = dict(opts['stream_camera'] or {})
    cam_of = {n: str(sc.get(n, n)) for n in names}
    # frame rate (verdict units): options > points / motion > per-stream fps > container avg_frame_rate
    fo = opts['fps']
    if isinstance(fo, dict):
        rates = [float(v) for v in fo.values()]
        rate_from = 'options fps'
    elif fo is not None:
        rates, rate_from = [float(fo)], 'options fps'
    else:
        rates = [d['points'][1] for d in data.values() if d['points'] is not None] + \
                [d['motion'][1] for d in data.values() if d['motion'] is not None]
        rate_from = 'points / motion'
        if not rates:
            rates = [d['meta']['fps'] for d in data.values() if 'fps' in d['meta']]
            rate_from = 'stream fps'
        if not rates:
            rates = [d['meta']['fps_container'] for d in data.values() if 'fps_container' in d['meta']]
            rate_from = 'video avg_frame_rate'
    if not rates:
        rates, rate_from = [30.0], 'default 30 fps'
    frame_rate = float(np.median(rates))
    frame_s = 1.0 / frame_rate
    settings = {'frame_rate': frame_rate, 'frame_rate_from': rate_from, 'frame_period_s': frame_s,
                'cross_check_tol_s': opts['cross_check_frames'] * frame_s,
                'max_residual_s': opts['max_residual_frames'] * frame_s,
                'audio_windows': opts['audio_windows'], 'audio_strategy': opts['audio_strategy'],
                'drift': bool(drift), 'refine': bool(refine), 'seeded': seed is not None,
                'streams': {n: d['meta'] for n, d in data.items()}}
    if opts['stream_camera']:
        settings['stream_camera'] = cam_of
    pairs, edges, windows, disagreements, fallbacks = [], [], {}, [], []
    irregular = [n for n, d in data.items() if d['meta'].get('time_jitter', 0.0) > opts['max_time_jitter']]
    for n in irregular:
        fallbacks.append('stream {}: irregular frame times (jitter {:.2g}); frame rate from the mean '
                         'step'.format(n, data[n]['meta']['time_jitter']))

    def record(a, b, est, kind, used=True, note=None, shift=0.0, sigma=None):
        d = {'pair': [a, b], 'kind': kind, 'method': est.method, 'offset_raw': float(est.offset),
             'offset': float(est.offset) + shift,
             'sigma': float(est.sigma if sigma is None else sigma), 'confidence': float(est.confidence),
             'identifiable': bool(est.identifiable), 'reasons': list(est.reasons), 'used': used}
        if note:
            d['note'] = note
        pairs.append(d)

    def audio_pair(a, b):
        (xa, sra, ta), (xb, srb, tb) = data[a]['audio'], data[b]['audio']
        sr = min(sra, srb)
        xa, xb = _resample(xa, sra, sr), _resample(xb, srb, sr)
        shift = tb - ta      # source time of sample 0
        dur = min(len(xa), len(xb)) / sr
        mo = min(max(opts['audio_min_overlap'], opts['audio_min_overlap_frac'] * dur), 0.5 * dur)
        kw = dict(max_lag=opts['audio_max_lag'], min_overlap=mo, pair=(a, b))
        strategy = opts['audio_strategy']
        if strategy == 'auto':
            strategy = 'windowed' if dur > opts['audio_windowed_min_s'] else 'global'
        tol = opts['cross_check_frames'] * frame_s
        if strategy == 'windowed':
            nw = int(np.clip(dur // 120, max(3, opts['audio_windows']), 48))
            best = S.audio_offset_windowed(xa, xb, sr, windows=nw, min_window=opts['audio_window_s'],
                                           **kw)
            record(a, b, best, 'audio', used=True, shift=shift)
            return best, shift, ta
        g = S.audio_offset(xa, xb, sr, method='gcc', windows=opts['audio_windows'], **kw)
        checks = []
        if opts['audio_cross_check']:
            for m in ('spectrogram', 'onset'):
                try:
                    checks.append(S.audio_offset(xa, xb, sr, method=m, windows=0, **kw))
                except Exception as e:  # pragma: no cover - defensive
                    fallbacks.append('audio {} {}-{}: {}'.format(m, a, b, e))
        best = g
        agree = [c for c in checks if c.identifiable and abs(c.offset - g.offset) <= tol]
        disagree = [c for c in checks if c.identifiable and abs(c.offset - g.offset) > tol]
        if not g.identifiable:
            # gcc failed: an identifiable spectrogram estimate stands in (re-run with windows, so
            # its sigma and the drift fit have windowed estimates)
            alt = [c for c in checks if c.identifiable and c.method == 'spectrogram']
            if alt:
                best = S.audio_offset(xa, xb, sr, method='spectrogram', windows=opts['audio_windows'],
                                      **kw)
                fallbacks.append('audio {}-{}: gcc unidentifiable, spectrogram used'.format(a, b))
        elif disagree and not agree:
            disagreements.append({'pair': [a, b], 'gcc': float(g.offset),
                                  **{c.method: float(c.offset) for c in disagree}})
        # sigma floor: the window scatter (a curvature sigma ignores it) and the agreement of the
        # identifiable methods
        sig = float(best.sigma)
        w = _windows_of(best)
        if w is not None:
            sig = max(sig, 1.4826 * float(np.median(np.abs(w[1] - np.median(w[1])))) / np.sqrt(len(w[1])))
        others = [c for c in [g] + checks if c is not best and c.identifiable
                  and abs(c.offset - best.offset) <= tol]
        if others:
            sig = max(sig, 0.5 * max(abs(c.offset - best.offset) for c in others))
        sig = max(sig, 1.0 / (sr * np.sqrt(12)))
        best = _edge(best, (a, b), 0.0, sigma=sig)
        record(a, b, g, 'audio', used=best.method == 'gcc', shift=shift,
               sigma=sig if best.method == 'gcc' else None)
        for c in checks:
            record(a, b, c, 'audio', used=best.method == c.method, note='cross-check', shift=shift,
                   sigma=sig if best.method == c.method else None)
        return best, shift, ta

    # ---------------------------------------------------------------- pair estimates
    for a, b in (itertools.combinations(names, 2) if seed is None else []):
        da, db = data[a], data[b]
        coarse = None
        if da['audio'] is not None and db['audio'] is not None:
            with timer('audio'):
                best, shift, ta = audio_pair(a, b)
                e = _edge(best, (a, b), shift)
                edges.append(e)
                if best.identifiable:
                    coarse = e.offset
                    w = _windows_of(best)
                    if w is not None:
                        windows[(a, b)] = (w[0] + ta, w[1] + shift)
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
                record(a, b, est, 'motion', used=use, shift=tb - ta)
                if use:
                    e = _edge(est, (a, b), tb - ta)
                    edges.append(e)
                    coarse = e.offset
        if da['points'] is not None and db['points'] is not None and cam_of[a] != cam_of[b]:
            calib = cgroup is not None and cam_of[a] in cam_names and cam_of[b] in cam_names
            mode = opts['epipolar_F']
            kinds = {'auto': ['calibrated'] if calib else ['self'], 'calibrated': ['calibrated'] if calib else [],
                     'self': ['self'], 'both': (['calibrated'] if calib else []) + ['self']}[mode]
            (pa, ra, ta), (pb, rb, tb) = da['points'], db['points']
            if kinds and abs(ra - rb) > 1e-6 * ra:
                pb = _resample_points(pb, rb, ra)        # b on a's frame rate (same start time)
                fallbacks.append('epipolar {}-{}: {:.4g} Hz points resampled to {:.4g} Hz'.format(
                    a, b, rb, ra))
            for kind in kinds:
                with timer('epipolar'):
                    # centre the search on the coarse (audio / motion) estimate, in frames of a
                    c0 = 0.0 if coarse is None else (coarse - (tb - ta)) * ra
                    w = float(opts['epipolar_search_frames'])
                    Fc = cgroup.subset_cameras([cam_names.index(cam_of[a]), cam_names.index(cam_of[b])]) \
                        if kind == 'calibrated' else None
                    try:
                        est = S.epipolar_offset(pa, pb, Fc, search=(c0 - w, c0 + w), rate=ra,
                                                n_boot=int(opts['epipolar_boot']), rng=rng, pair=(a, b))
                    except Exception as ex:  # a failed estimate is reported, never raised
                        fallbacks.append('epipolar {} {}-{}: {}'.format(kind, a, b, ex))
                        continue
                    record(a, b, est, 'epipolar', used=bool(est.identifiable), shift=tb - ta,
                           note='F ' + kind)
                    edges.append(_edge(est, (a, b), tb - ta))

    # ---------------------------------------------------------------- graph (or the seed)
    drift_fits, rates_out, drift_fail = {}, {n: 0.0 for n in names}, []
    a0s = {}      # offset at global time 0 of each stream's affine map
    if seed is None:
        deg = {n: 0 for n in names}
        for e in edges:
            if e.identifiable:
                deg[e.pair[0]] += 1
                deg[e.pair[1]] += 1
        ref = str(reference) if reference is not None else \
            max(names, key=lambda n: (deg[n], -names.index(n)))
        with timer('graph'):
            # estimates without a sigma (motion) count as one frame period
            graph = S.solve_offset_graph(edges, anchor=ref, names=names,
                                         max_residual=opts['max_residual_frames'] * frame_s,
                                         default_sigma=frame_s)
        offsets = dict(graph.offsets)
        sigmas = {n: (float(graph.sigma[n]) if graph.sigma.get(n) is not None else None) for n in names}

        # ------------------------------------------------------------ drift
        if drift:
            with timer('drift'):
                not_sig = []
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
                    if fit.identifiable and np.isfinite(fit.sigma_rate) and \
                            abs(fit.rate) < opts['drift_min_z'] * fit.sigma_rate:
                        # a rate within the noise of the windows: no drift (0), not a failure
                        not_sig.append(n)
                        continue
                    drift_fits[n] = fit
                    if fit.identifiable:
                        rates_out[n] = float(fit.rate)
                    else:
                        drift_fail.append(n)
                # streams with audio but no windowed estimate: the rate was never fitted
                for n in names:
                    if n != ref and data[n]['audio'] is not None and n not in drift_fits \
                            and n not in not_sig and offsets.get(n) is not None:
                        drift_fail.append(n)
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
        settings['drift_not_significant'] = not_sig if drift else []
    else:
        graph = None
        missing = [n for n in names if seed_maps.get(n) is None]
        for n in names:
            if seed_maps.get(n) is not None:
                a0s[n], rates_out[n] = _timemap_parts(seed_maps[n])
                if getattr(seed_maps[n], 'knots', ()):
                    fallbacks.append('seed {}: time-map knots dropped'.format(n))
        ssig = dict(getattr(seed, 'sigmas', None) or {})
        sigmas = {n: (ssig.get(n) if ssig.get(n) is not None else (frame_s if n in a0s else None))
                  for n in names}
        ref = str(reference) if reference is not None else \
            (str(seed.reference) if getattr(seed, 'reference', None) in names else
             min((n for n in names if n in a0s), key=lambda n: abs(a0s[n]), default=names[0]))
        if missing:
            fallbacks.append('seed: no time map for {}'.format(missing))

    # ---------------------------------------------------------------- STBA refinement
    refined = None
    if refine:
        seedtm = {n: S.TimeMap.from_offset(a0s[n], rate=rates_out[n]) for n in a0s}
        refined = _refine_stba(cgroup, data, names, ref, seedtm, sigmas, drift, opts, settings,
                               fallbacks, timer, device=device, drift_fits=drift_fits, cam_of=cam_of,
                               seeded=seed is not None)
        if refined.get('ok'):
            # STBA: true global = seed + delta + rc * (seed - tmid), so the refined offset is
            # seed offset - delta - rc * (g - tmid)
            free = refined.get('free_rate', False)
            for n, dlt in refined['delta'].items():
                rc = refined['rate_correction'][n] if free else 0.0
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
    cyc = graph.cycle_rms / frame_s if graph is not None and np.isfinite(graph.cycle_rms) else 0.0
    rules = [Rule('unidentifiable_streams', len(unident), None, 0,
                  'streams with no identifiable path to the reference {!r}: {}'.format(ref, unident)),
             Rule('max_sigma_frames', max_sigma_f, 0.25, 1.0, 'largest offset sigma / frame period'),
             Rule('cycle_rms_frames', cyc, 0.25, 1.0,
                  'RMS residual of the offset graph / frame period (contradicting estimates)'),
             Rule('method_disagreements', len(disagreements), 0, None,
                  'audio pairs where spectrogram / onset contradict gcc'),
             Rule('irregular_times', len(irregular), 0, None,
                  'streams with irregular per-frame times (rate from the mean step): {}'.format(irregular))]
    if drift and seed is None:
        drift_fail = sorted(set(drift_fail))
        rules.append(Rule('drift_unidentifiable', len(drift_fail), 0, None,
                          'streams with audio whose clock rate could not be fitted: {}'.format(drift_fail)))
    verdict, reasons, table = evaluate_rules(rules, opts['verdict'])
    report = {'reference': ref, 'pairs': pairs, 'n_edges': len(edges),
              'rejected': list(graph.rejected) if graph is not None else [],
              'cycle_rms_s': graph.cycle_rms if graph is not None else None,
              'disagreements': disagreements, 'unidentifiable': unident,
              'drift': {n: f.to_dict() for n, f in drift_fits.items()}, 'rules': table}
    return OffsetEstimation(
        offsets=offsets, sigmas=sigmas, rates=rates_out, timemaps=timemaps, graph=graph,
        pairs=pairs, drift=drift_fits, identifiable={n: offsets.get(n) is not None for n in names},
        refined=refined, reference=ref, report=report, verdict=verdict, reasons=reasons,
        settings=settings,
        provenance=provenance('estimate_offsets',
                              inputs={n: [k for k, v in d.items() if v is not None and k != 'meta']
                                      for n, d in data.items()},
                              fallbacks=fallbacks, timings=timer.times))


def _refine_stba(cgroup, data, names, ref, seed, sigmas, drift, opts, settings, fallbacks,
                 timer, device='auto', drift_fits=None, cam_of=None, seeded=False):
    """SpatiotemporalBA on the 2D points, seed times from the graph/drift time maps; returns
    the per-stream constant corrections ``delta`` (refined offset = seed offset - delta)."""
    from .. import stba
    if cgroup is None:
        fallbacks.append('refine: needs cgroup, skipped')
        return {'ok': False, 'reason': 'no cgroup'}
    cam_of = cam_of or {n: n for n in names}
    cam_names = list(cgroup.get_names())
    use = [n for n in names if data[n]['points'] is not None and cam_of[n] in cam_names and n in seed]
    if len(use) < 2 or ref not in use:
        fallbacks.append('refine: needs points for the reference and >= 1 other camera')
        return {'ok': False, 'reason': 'not enough cameras with points'}
    if len({data[n]['points'][0].shape[1] for n in use}) != 1:
        return {'ok': False, 'reason': 'cameras have different numbers of keypoints'}
    sub = cgroup.subset_cameras([cam_names.index(cam_of[n]) for n in use])
    cams, pts, uvs, ts, tmid = [], [], [], [], {}
    span = 0.0
    for k, n in enumerate(use):
        p2d, rate, t0 = data[n]['points']
        glob = seed[n].to_global(t0 + np.arange(p2d.shape[0]) / rate)     # seed global time
        fi, ji = np.nonzero(np.isfinite(p2d).all(-1))
        cams.append(np.full(len(fi), k)); pts.append(ji); uvs.append(p2d[fi, ji]); ts.append(glob[fi])
        tmid[n] = 0.5 * (glob[fi].min() + glob[fi].max()) if len(fi) else 0.0
        if len(fi):
            span = max(span, float(glob[fi].max() - glob[fi].min()))
    rr = opts['refine_rate']
    fitted = any(drift_fits and n in drift_fits and drift_fits[n].identifiable for n in use)
    if rr == 'free':
        free_rate = True
    elif rr == 'seed':
        free_rate = False
    else:
        # a clock rate from a short window extrapolates badly: free it only with a drift fit (or
        # a seeded rate) and enough time span
        free_rate = bool(drift) and (fitted or seeded) and span >= opts['refine_free_rate_min_s']
    obs = stba.TimedObservations(cam=np.concatenate(cams), point=np.concatenate(pts),
                                 uv=np.concatenate(uvs), t=np.concatenate(ts))
    scale = _rig_scale(sub)
    prior = [max(float(sigmas[n] or 0.0), opts['refine_offset_sigma_min']) for n in use if n != ref]
    kw = dict(knot_spacing=settings['frame_period_s'], accel_sigma=opts['refine_accel_frac'] * scale,
              free_offsets='per_camera', offset_prior_sigma=float(np.median(prior)),
              ref_camera=use.index(ref), free_rate=free_rate,
              rate_prior_sigma=opts['refine_rate_sigma'], extrinsics=bool(opts['refine_extrinsics']),
              max_iter=int(opts['refine_max_iter']))
    settings['stba'] = dict(kw, accel_sigma_rel=opts['refine_accel_frac'], cameras=use,
                            refine_rate=rr, points_span_s=span, device=_device(device))
    try:
        with timer('stba'):
            res = stba.SpatiotemporalBA(sub, obs, device=_device(device), **kw).fit()
    except Exception as e:  # a failed refinement is reported, never raised
        fallbacks.append('refine: STBA failed: {}'.format(e))
        return {'ok': False, 'reason': 'STBA failed: {}'.format(e)}
    delta = np.asarray(res.delta, float)
    sg = np.asarray(res.delta_sigma_sandwich, float)
    sg = np.where(np.isfinite(sg), sg, np.asarray(res.delta_sigma, float))
    return {'ok': True, 'cameras': use, 'free_rate': free_rate,
            'delta': {n: float(delta[k]) for k, n in enumerate(use)},
            'sigmas': {n: (0.0 if n == ref else float(sg[k]) if np.isfinite(sg[k]) else sigmas[n])
                       for k, n in enumerate(use)},
            'rate_correction': {n: float(r) for n, r in zip(use, np.asarray(res.rate, float))},
            'tmid': tmid, 'sigma_px': float(res.sigma_px), 'cost': float(res.cost)}
