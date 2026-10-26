"""Board-free initial extrinsics from 2D correspondences (keypoints, blobs) and known
per-camera intrinsics (plan 01 E2).

:func:`from_correspondences` is a port of the motor-observatory incremental SfM
(``calibrate_lib.sfm_extrinsics_robust``), which there beat VGGT 5.5 px vs 112 px, with
per-camera intrinsics instead of a shared virtual camera matrix:

1. every camera's observations are undistorted with its own intrinsics and re-expressed as
   *ideal pinhole pixels* ``u = f_c * x_norm`` (``f_c`` = the camera's focal length), so all
   thresholds stay in pixels while the geometry is distortion free;
2. relative poses of all camera pairs from RANSAC essential matrices (:func:`pairwise_poses`);
   the pair with the most inliers seeds the reconstruction (unit baseline);
3. the remaining cameras are added one at a time by **frame RANSAC PnP**
   (:func:`frame_pnp_ransac`): a minimal sample is a few whole *frames*, and hypotheses are
   scored by the number of frames whose median residual is below ``pnp_thresh``. A detector
   that locks onto a decoy moves all keypoints of a frame together, so frames are the right
   sampling and masking unit;
4. a camera is accepted only with at least ``min_inlier_frames`` inlier frames spanning
   ``min_inlier_windows`` distinct windows (``window_id``: frames from one sampled burst count
   once); it becomes a **driver** (may define 3D points) if at least ``driver_frac`` of its
   frames agree, otherwise a **passenger** (placed, never defines points); a placement whose
   centre lies more than ``max_center_ratio`` times further from the scene than the median
   camera is rejected as ill-conditioned; a camera whose addition degrades the existing
   cameras' error is rejected;
5. optional rotation priors (e.g. VGGT ``pairwise_rotations``) rescue weak or runaway
   placements: with the rotation given the translation is linear (:func:`solve_t_given_R`)
   and a minimal sample is a single frame (:func:`place_with_rotation_prior`). Priors are
   hypotheses, always scored on the 2D data;
6. after every addition the cloud is re-triangulated robustly (IRLS) from all drivers, with
   (camera, frame) blocks whose median error exceeds ``decoy_thresh`` masked; every
   ``local_ba_every`` additions a sparse robust bundle adjustment polishes the placed rig.

The result is in an arbitrary similarity gauge: the seed camera is the identity and the seed
baseline has length 1. Use ``aniposelib.scale`` (or a board) to fix the metric scale.
"""

import warnings

import cv2
import numpy as np
import torch

from ._common import (InitReport, as_rng, from_extrinsics, get_extrinsics, make_M, rodrigues,
                      rotvec, to_np)

__all__ = ['from_correspondences', 'pairwise_poses', 'frame_pnp_ransac', 'solve_t_given_R',
           'place_with_rotation_prior', 'ideal_points', 'refine_extrinsics', 'refine_group']


# ------------------------------------------------------------------ data preparation

def _flatten(p2ds, frame_id, window_id):
    """``p2ds [C,N,2]`` or ``[C,N,J,2]`` -> ``[C,M,2]`` plus per-point frame and window ids."""
    p2ds = np.asarray(to_np(p2ds), dtype='float64')
    if p2ds.ndim == 4:
        C, N, J, _ = p2ds.shape
        flat = p2ds.reshape(C, N * J, 2)
        per_frame = N
        rep = J
    elif p2ds.ndim == 3:
        C, N, _ = p2ds.shape
        flat, per_frame, rep = p2ds, N, 1
    else:
        raise ValueError('p2ds must be [C,N,2] or [C,N,J,2], got shape {}'.format(p2ds.shape))
    M = flat.shape[1]

    def expand(ids, what):
        if ids is None:
            return None
        ids = np.asarray(ids).reshape(-1)
        if len(ids) == M:
            return ids
        if len(ids) == per_frame:
            return np.repeat(ids, rep)
        raise ValueError('{} must have one entry per frame ({}) or per point ({}), got {}'
                         .format(what, per_frame, M, len(ids)))

    fid = expand(frame_id, 'frame_id')
    if fid is None:
        fid = np.repeat(np.arange(per_frame), rep)
    wid = expand(window_id, 'window_id')
    if wid is None:
        wid = fid
    return flat, fid, wid


def ideal_points(cgroup, p2ds):
    """Undistort ``p2ds [C,M,2]`` with each camera's intrinsics and return ideal pinhole pixels
    ``u = f_c * x_norm`` (principal point at 0) and the focal lengths ``f [C]``.
    Points that cannot be undistorted are NaN."""
    from aniposelib import geometry as G
    P = G.pack(cgroup, device='cpu')
    x = to_np(G.undistort(P, torch.as_tensor(p2ds, dtype=torch.float64)))
    K = to_np(P.K)
    f = np.sqrt(np.abs(K[:, 0, 0] * K[:, 1, 1]))
    u = x * f[:, None, None]
    u[~np.all(np.isfinite(p2ds), axis=-1)] = np.nan
    return u, f


def _ideal_packed(f, Ms):
    from aniposelib.geometry import PackedCameras
    C = len(f)
    K = np.tile(np.eye(3), (C, 1, 1))
    K[:, 0, 0] = f
    K[:, 1, 1] = f
    return PackedCameras(K=torch.as_tensor(K), dist=torch.zeros(C, 8, dtype=torch.float64),
                         model=torch.zeros(C, dtype=torch.long),
                         ext=torch.as_tensor(np.asarray(Ms, dtype='float64')))


def _Kf(f):
    return np.array([[f, 0, 0], [0, f, 0], [0, 0, 1.0]])


# ------------------------------------------------------------------ geometry kernels

def project_ideal(p3d, R, t, f):
    """Distortion-free projection to ideal pixels; NaN behind the camera."""
    X = p3d @ np.asarray(R).T + np.asarray(t).reshape(3)
    z = X[:, 2]
    with np.errstate(all='ignore'):
        out = f * X[:, :2] / z[:, None]
    out[~(z > 0)] = np.nan
    return out


def solve_t_given_R(R, p3d, p2d_norm):
    """Least-squares camera translation with the rotation held fixed (normalised image points):
    ``tx - xn tz = xn Yz - Yx`` and ``ty - yn tz = yn Yz - Yy`` with ``Y = R X``."""
    Y = p3d @ np.asarray(R).T
    xn, yn = p2d_norm[:, 0], p2d_norm[:, 1]
    n = len(p3d)
    A = np.zeros((2 * n, 3))
    b = np.zeros(2 * n)
    A[0::2, 0] = 1.0
    A[0::2, 2] = -xn
    b[0::2] = xn * Y[:, 2] - Y[:, 0]
    A[1::2, 1] = 1.0
    A[1::2, 2] = -yn
    b[1::2] = yn * Y[:, 2] - Y[:, 1]
    t, *_ = np.linalg.lstsq(A, b, rcond=None)
    return t


class _Frames:
    """Per-camera frame grouping of the valid (cloud & observation) points."""

    def __init__(self, valid, frame_id, min_frame_pts):
        self.all_idx = np.nonzero(valid)[0]
        f = frame_id[self.all_idx]
        self.frames, inv = np.unique(f, return_inverse=True)
        counts = np.bincount(inv, minlength=len(self.frames))
        self.order = np.argsort(inv, kind='stable')
        self.rows = inv[self.order]
        self.cols = (np.concatenate([np.arange(c) for c in counts]) if len(counts)
                     else np.zeros(0, int))
        self.width = int(counts.max()) if len(counts) else 0
        self.counts = counts
        self.keep = counts >= min_frame_pts
        self.usable = self.frames[self.keep]
        starts = np.r_[0, np.cumsum(counts)[:-1]] if len(counts) else np.zeros(0, int)
        self.members = {int(fr): self.all_idx[self.order[s:s + c]]
                        for fr, s, c, k in zip(self.frames, starts, counts, self.keep) if k}

    def score(self, res, thresh):
        """Number of usable frames with median residual < thresh, and those frames."""
        g = np.full((len(self.frames), self.width), np.inf)
        r = res[self.order]
        g[self.rows, self.cols] = np.where(np.isnan(r), np.inf, r)  # behind camera = outlier
        g.sort(axis=1)
        ar = np.arange(len(self.frames))
        med = 0.5 * (g[ar, (self.counts - 1) // 2] + g[ar, self.counts // 2])
        ok = self.keep & (med < thresh)
        return int(ok.sum()), self.frames[ok]

    def sample(self, rng, k, min_points=6):
        """Indices of >= k whole frames holding >= min_points points."""
        order = rng.permutation(len(self.usable))
        sel, n = [], 0
        for i in order:
            pts = self.members[int(self.usable[i])]
            sel.append(pts)
            n += len(pts)
            if len(sel) >= k and n >= min_points:
                break
        return np.concatenate(sel) if sel else np.zeros(0, int)


def _auto_min_frame_pts(frame_id):
    _, counts = np.unique(frame_id, return_counts=True)
    return int(np.clip(round(0.5 * np.median(counts)), 1, 8))


def frame_pnp_ransac(p3d, u, frame_id, f, n_frames_sample=3, n_iters=300, thresh=8.0, rng=None,
                     min_frame_pts=None, inner_iters=80):
    """RANSAC PnP whose minimal sample is whole frames, scored by inlier frames (motor
    ``frame_pnp_ransac``). ``p3d [M,3]`` (NaN = no point), ``u [M,2]`` ideal pixels of one
    camera, ``f`` its focal length. Returns a dict (rvec, tvec, n_inlier_frames, n_frames,
    inlier_frames, n_inliers) or None."""
    rng = as_rng(rng)
    if min_frame_pts is None:
        min_frame_pts = _auto_min_frame_pts(frame_id)
    valid = np.all(np.isfinite(p3d), -1) & np.all(np.isfinite(u), -1)
    if valid.sum() < 12:
        return None
    fr = _Frames(valid, frame_id, min_frame_pts)
    if len(fr.usable) < 2:
        return None
    K = _Kf(f)
    P3a, P2a = p3d[fr.all_idx], u[fr.all_idx]
    best = None
    for _ in range(n_iters):
        sel = fr.sample(rng, n_frames_sample)
        if len(sel) < 6:
            continue
        try:
            ok, rvec, tvec, _inl = cv2.solvePnPRansac(
                p3d[sel], u[sel], K, None, reprojectionError=thresh, confidence=0.999,
                iterationsCount=inner_iters, flags=cv2.SOLVEPNP_EPNP)
        except cv2.error:
            continue
        if not ok:
            continue
        res = np.linalg.norm(project_ideal(P3a, rodrigues(rvec), tvec, f) - P2a, axis=1)
        score, inl = fr.score(res, thresh)
        if best is None or score > best[0]:
            best = (score, rvec.copy(), tvec.copy(), inl)
            if score == len(fr.usable):
                break
    if best is None or best[0] < 2:
        return None
    score, rvec, tvec, inl_frames = best
    rvec, tvec, score, inl_frames, n_in = _polish(p3d, u, valid, frame_id, fr, K, f, rvec, tvec,
                                                  score, inl_frames, thresh)
    return dict(rvec=np.ravel(rvec), tvec=np.ravel(tvec), n_inlier_frames=int(score),
                n_frames=int(len(fr.usable)), inlier_frames=np.asarray(inl_frames),
                n_inliers=int(n_in), n_points=int(valid.sum()))


def _polish(p3d, u, valid, frame_id, fr, K, f, rvec, tvec, score, inl_frames, thresh):
    """Refit on the points of the inlier frames (PnP RANSAC + LM); keep it if it scores
    at least as well."""
    keep = valid & np.isin(frame_id, inl_frames)
    n_in = int(keep.sum())
    try:
        ok, r2, t2, inl = cv2.solvePnPRansac(
            p3d[keep], u[keep], K, None, reprojectionError=thresh, confidence=0.9999,
            iterationsCount=1000, flags=cv2.SOLVEPNP_ITERATIVE, rvec=np.array(rvec, copy=True).reshape(3, 1),
            tvec=np.array(tvec, copy=True).reshape(3, 1), useExtrinsicGuess=True)
        if ok and inl is not None and len(inl) >= 6:
            ii = inl.ravel()
            r2, t2 = cv2.solvePnPRefineLM(p3d[keep][ii], u[keep][ii], K, None, r2, t2)
            res = np.linalg.norm(project_ideal(p3d[fr.all_idx], rodrigues(r2), t2, f) - u[fr.all_idx], axis=1)
            s2, f2 = fr.score(res, thresh)
            if s2 >= score:
                return r2, t2, s2, f2, len(ii)
    except cv2.error:
        pass
    return rvec, tvec, score, inl_frames, n_in


def place_with_rotation_prior(p3d, u, frame_id, f, R_hyps, thresh=8.0, n_iters=400, rng=None,
                              min_frame_pts=None):
    """Best pose over candidate rotations ``R_hyps``: with R fixed the translation is linear,
    so a minimal sample is one frame; scored by inlier frames (motor
    ``place_with_rotation_prior``). Returns a dict like :func:`frame_pnp_ransac` plus
    ``hypothesis`` (index into ``R_hyps``), or None."""
    rng = as_rng(rng)
    if min_frame_pts is None:
        min_frame_pts = _auto_min_frame_pts(frame_id)
    valid = np.all(np.isfinite(p3d), -1) & np.all(np.isfinite(u), -1)
    if valid.sum() < 8:
        return None
    fr = _Frames(valid, frame_id, min_frame_pts)
    if len(fr.usable) < 2:
        return None
    norm = u / f
    P3a, P2a = p3d[fr.all_idx], u[fr.all_idx]
    best = None
    per = max(8, n_iters // max(1, len(R_hyps)))
    for hi, R0 in enumerate(R_hyps):
        for _ in range(per):
            sel = fr.sample(rng, 1, min_points=3)
            if len(sel) < 3:
                continue
            t0 = solve_t_given_R(R0, p3d[sel], norm[sel])
            if not np.all(np.isfinite(t0)):
                continue
            res = np.linalg.norm(project_ideal(P3a, R0, t0, f) - P2a, axis=1)
            score, inl = fr.score(res, thresh)
            if best is None or score > best[0]:
                best = (score, rotvec(R0), t0, inl, hi)
    if best is None or best[0] < 2:
        return None
    score, rvec, tvec, inl_frames, hi = best
    rvec, tvec, score, inl_frames, n_in = _polish(p3d, u, valid, frame_id, fr, _Kf(f), f, rvec,
                                                  tvec, score, inl_frames, thresh)
    return dict(rvec=np.ravel(rvec), tvec=np.ravel(tvec), n_inlier_frames=int(score),
                n_frames=int(len(fr.usable)), inlier_frames=np.asarray(inl_frames),
                n_inliers=int(n_in), hypothesis=int(hi), via_prior=True)


def pairwise_poses(u, f, thresh=2.0, min_points=30, max_iters=4000, rng=None, pairs=None):
    """RANSAC essential-matrix relative pose of every camera pair (motor ``pairwise_poses``).

    ``u [C,M,2]`` ideal pixels (:func:`ideal_points`), ``f [C]`` focal lengths. Returns
    ``{(i, j): dict(R, t, idx, inl, n_in, ratio)}`` with ``x_j = R x_i + t`` (``|t| = 1``)."""
    rng = as_rng(rng)
    C = u.shape[0]
    out = {}
    todo = pairs if pairs is not None else [(i, j) for i in range(C) for j in range(i + 1, C)]
    for i, j in todo:
        fin = np.all(np.isfinite(u[i]), -1) & np.all(np.isfinite(u[j]), -1)
        idx = np.nonzero(fin)[0]
        if len(idx) < min_points:
            continue
        a, b = u[i, idx] / f[i], u[j, idx] / f[j]
        cv2.setRNGSeed(int(rng.integers(2 ** 31 - 1)))
        try:
            E, mask = cv2.findEssentialMat(a, b, np.eye(3), method=cv2.RANSAC, prob=0.9999,
                                           threshold=thresh / np.sqrt(f[i] * f[j]), maxIters=max_iters)
        except cv2.error:
            continue
        if E is None or mask is None or E.shape[0] < 3:
            continue
        E = E[:3]
        n_in, R, t, mp = cv2.recoverPose(E, a, b, np.eye(3), mask=mask.copy())
        if n_in < min_points // 2:
            continue
        out[(i, j)] = dict(R=R, t=t.ravel() / np.linalg.norm(t), idx=idx,
                           inl=mp.ravel().astype(bool), n_in=int(n_in), ratio=n_in / len(idx))
    return out


# ------------------------------------------------------------------ robust cloud

def _retriangulate(u, f, Ms, ks, drivers, frame_id, decoy_thresh, point_max_err, sigma=8.0,
                   min_frame_pts=1):
    """Robust (IRLS) cloud from placed cameras ``ks``; (camera, frame) blocks with median error
    above ``decoy_thresh`` are masked and the cloud refit. Returns p3d [M,3], good [M],
    errors [len(ks), M] (px), the masked observations."""
    from aniposelib.triangulation import triangulate_robust
    P = _ideal_packed(f[ks], Ms)
    sub = u[ks]
    contrib = np.array([k in drivers for k in ks])
    # tau only decides which views count towards min_inliers; IRLS handles the outliers
    kw = dict(method='irls', sigma=sigma, contrib=contrib, mask_out_of_bounds=False, tau=1e12)
    res = triangulate_robust(P, sub, **kw)
    err = np.asarray(res.errors)
    work = _frame_mask(sub, frame_id, err, decoy_thresh, min_frame_pts)
    res = triangulate_robust(P, work, **kw)
    p3d, err = np.asarray(res.p3d), np.asarray(res.errors)
    nobs = np.all(np.isfinite(work[contrib]), -1).sum(0)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        e = np.where(np.isfinite(err), err, np.nan)
        e[~np.all(np.isfinite(work), -1)] = np.nan
        ptmed = np.nanmedian(e, axis=0)
    good = np.all(np.isfinite(p3d), -1) & (nobs >= 2) & (ptmed <= point_max_err)
    e = np.where(np.all(np.isfinite(work), -1), err, np.nan)
    return p3d, good, e, work


def _frame_mask(p2d, frame_id, err, thresh, min_pts):
    """NaN out (camera, frame) blocks whose median error exceeds ``thresh`` (scalar or per camera)."""
    frames, inv = np.unique(frame_id, return_inverse=True)
    out = p2d.copy()
    thr = np.broadcast_to(np.asarray(thresh, dtype='float64'), (p2d.shape[0],))
    for c in range(p2d.shape[0]):
        e = err[c]
        ok = np.isfinite(e)
        if not ok.any():
            continue
        cnt = np.bincount(inv[ok], minlength=len(frames))
        # per-frame medians via sorting
        order = np.lexsort((e[ok], inv[ok]))
        ev = e[ok][order]
        starts = np.r_[0, np.cumsum(cnt)[:-1]]
        med = np.full(len(frames), np.nan)
        nz = cnt > 0
        lo = starts[nz] + (cnt[nz] - 1) // 2
        hi = starts[nz] + cnt[nz] // 2
        med[nz] = 0.5 * (ev[lo] + ev[hi])
        bad = (med > thr[c]) & (cnt >= min_pts)
        if bad.any():
            out[c, bad[inv]] = np.nan
    return out


def _med(x):
    v = np.asarray(x)[np.isfinite(x)]
    return float(np.median(v)) if len(v) else float('nan')


# ------------------------------------------------------------------ main entry

def _prior_table(rotation_priors, names):
    """{(i, j): R_ij} with x_j = R_ij x_i (both directions)."""
    out = {}
    if not rotation_priors:
        return out
    for (a, b), v in rotation_priors.items():
        i = names.index(a) if isinstance(a, str) else int(a)
        j = names.index(b) if isinstance(b, str) else int(b)
        R = np.asarray(v['R'] if isinstance(v, dict) else v, dtype='float64')
        out[(i, j)] = R
        out[(j, i)] = R.T
    return out


def from_correspondences(cgroup, p2ds, frame_id=None, window_id=None, min_inlier_frames=8,
                         min_inlier_windows=4, driver_frac=0.25, max_center_ratio=3.0,
                         rotation_priors=None, local_ba_every=3, rng=None, *,
                         pair_thresh=2.0, pnp_thresh=8.0, point_max_err=12.0, decoy_thresh=25.0,
                         min_pair_points=30, scan_iters=60, place_iters=400,
                         min_frame_points=None, local_ba_kwargs=None, pairs=None, verbose=False):
    """Initial extrinsics from 2D correspondences and per-camera intrinsics (see module doc).

    Parameters
    ----------
    cgroup : CameraGroup
        Supplies names and intrinsics (matrix, distortion, model, offset); its extrinsics are
        ignored. It is not modified.
    p2ds : ``[C,N,2]`` or ``[C,N,J,2]`` (numpy or torch), NaN = missing. Same point index =
        same 3D point across cameras (associated detections).
    frame_id, window_id : per frame (``N``) or per point; default: ``[C,N,J,2]`` input groups the
        ``J`` points of a frame, ``[C,N,2]`` input makes every point its own frame; windows
        default to frames.
    rotation_priors : ``{(cam_a, cam_b): R}`` (names or indices; ``x_b = R x_a``), e.g. from
        ``init.vggt.pairwise_rotations``; used only to rescue weak/runaway placements.
    local_ba_every : sparse extrinsics + points bundle adjustment (:func:`refine_extrinsics`,
        on decoy-masked ideal pixels, with outlier trimming) after every this many placements and
        after the last one (0/None: never). ``local_ba_kwargs`` are passed to it.
    rng : Generator, int or None.

    Returns
    -------
    (CameraGroup, InitReport)
        The group holds the **placed** cameras only (in ``cgroup`` order), with copies of the
        input intrinsics. ``report.drivers``, ``.passengers``, ``.rejected`` (name -> reason
        dict) and ``.missing`` say what happened; ``report.stats`` has the seed pair, per-camera
        inlier frames/windows and the final median error (px).
    """
    rng = as_rng(rng)
    names = cgroup.get_names()
    C = len(names)
    flat, fid, wid = _flatten(p2ds, frame_id, window_id)
    if flat.shape[0] != C:
        raise ValueError('p2ds has {} cameras, cgroup has {}'.format(flat.shape[0], C))
    mfp = _auto_min_frame_pts(fid) if min_frame_points is None else int(min_frame_points)
    u, f = ideal_points(cgroup, flat)

    if pairs is None:
        pairs = pairwise_poses(u, f, thresh=pair_thresh, min_points=min_pair_points, rng=rng)
    if not pairs:
        raise RuntimeError('from_correspondences: no camera pair shares enough correspondences '
                           '(min_pair_points={}) for an essential-matrix seed'.format(min_pair_points))
    (i0, j0), r0 = max(pairs.items(), key=lambda kv: kv[1]['n_in'])
    ext = {i0: np.eye(4), j0: make_M(r0['R'], r0['t'])}
    drivers = {i0, j0}
    passengers, rejected, cam_stats = {}, {}, {}
    priors = _prior_table(rotation_priors, names)
    log = print if verbose else (lambda *a, **k: None)
    log('sfm: seed pair {}/{} ({} inliers)'.format(names[i0], names[j0], r0['n_in']))

    def retri(e=None):
        e = ext if e is None else e
        ks = sorted(e)
        p3d, good, err, work = _retriangulate(u, f, np.stack([e[k] for k in ks]), ks, drivers, fid,
                                              decoy_thresh, point_max_err, min_frame_pts=mfp)
        return p3d, good, err, ks

    def implausible(M, p3d, good):
        pts = p3d[good]
        if len(pts) < 10 or len(ext) < 2:
            return False, 0.0, 0.0
        cen = np.median(pts, axis=0)

        def dist(Mx):
            return float(np.linalg.norm(-Mx[:3, :3].T @ Mx[:3, 3] - cen))
        med_d = float(np.median([dist(ext[k]) for k in ext]))
        d = dist(M)
        return d > max_center_ratio * med_d, d, med_d

    def rescue(c, cloud):
        hyps = [priors[(k, c)] @ ext[k][:3, :3] for k in sorted(ext) if (k, c) in priors]
        if not hyps:
            return None
        return place_with_rotation_prior(cloud, u[c], fid, f[c], hyps, thresh=pnp_thresh,
                                         n_iters=max(place_iters, 60 * len(hyps)), rng=rng,
                                         min_frame_pts=mfp)

    def n_windows(frames):
        if len(frames) == 0:
            return 0
        sel = np.isin(fid, frames)
        return int(len(np.unique(wid[sel])))

    def weak(r):
        return r['n_inlier_frames'] < min_inlier_frames or n_windows(r['inlier_frames']) < min_inlier_windows

    p3d, good, err, ks = retri()
    remaining = set(range(C)) - set(ext)
    order = 0
    while remaining:
        cloud = np.where(good[:, None], p3d, np.nan)
        cands = []
        for c in sorted(remaining):
            r = frame_pnp_ransac(cloud, u[c], fid, f[c], thresh=pnp_thresh, n_iters=scan_iters,
                                 rng=rng, min_frame_pts=mfp)
            if r is not None:
                r['cam'] = c
                cands.append(r)
        if not cands:
            break
        cands.sort(key=lambda r: -(r['n_inlier_frames'] + r['n_inlier_frames'] / max(1, r['n_frames'])))
        c = cands[0]['cam']
        r = frame_pnp_ransac(cloud, u[c], fid, f[c], thresh=pnp_thresh, n_iters=place_iters,
                             rng=rng, min_frame_pts=mfp)
        if r is None or r['n_inlier_frames'] < cands[0]['n_inlier_frames']:
            r = cands[0]
        remaining.discard(c)
        name = names[c]

        if weak(r):
            r2 = rescue(c, cloud)
            if r2 is not None and r2['n_inlier_frames'] > r['n_inlier_frames']:
                r = r2
            if weak(r):
                rejected[name] = dict(reason='too few independent frames agree with the reconstruction',
                                      inlier_frames=r['n_inlier_frames'],
                                      inlier_windows=n_windows(r['inlier_frames']), n_frames=r['n_frames'])
                log('sfm: REJECT {}: {}'.format(name, rejected[name]))
                continue
        M_new = make_M(rodrigues(r['rvec']), r['tvec'])
        bad, d_new, d_med = implausible(M_new, p3d, good)
        if bad and r['n_inlier_frames'] < r['n_frames']:
            # a far-away camera maps the compact cloud onto one pixel, so the frames a decoy
            # detection put on a fixed spot all agree with it: retry without those frames
            masked = np.isin(fid, r['inlier_frames'])
            r2 = frame_pnp_ransac(np.where(masked[:, None], np.nan, cloud), u[c], fid, f[c],
                                  thresh=pnp_thresh, n_iters=place_iters, rng=rng, min_frame_pts=mfp)
            if r2 is not None and not weak(r2):
                M2 = make_M(rodrigues(r2['rvec']), r2['tvec'])
                if not implausible(M2, p3d, good)[0]:
                    r, M_new, bad = r2, M2, False
                    r['runaway_frames_excluded'] = int(len(np.unique(fid[masked])))
        if bad:
            r2 = rescue(c, cloud)
            if r2 is not None:
                M2 = make_M(rodrigues(r2['rvec']), r2['tvec'])
                if not implausible(M2, p3d, good)[0] and not weak(r2):
                    r, M_new, bad = r2, M2, False
        if bad:
            rejected[name] = dict(reason='implausible camera centre (ill-conditioned pose)',
                                  dist=d_new, median_dist=d_med, ratio=d_new / max(d_med, 1e-12))
            log('sfm: REJECT {}: {}'.format(name, rejected[name]))
            continue
        frac = r['n_inlier_frames'] / max(1, r['n_frames'])
        is_driver = frac >= driver_frac
        before = _med(err)
        trial = dict(ext)
        trial[c] = M_new
        if is_driver:
            drivers.add(c)
        p3d_t, good_t, err_t, ks_t = retri(trial)
        after_old = _med(err_t[[ks_t.index(k) for k in sorted(ext)]])
        if after_old > max(2.0 * before, before + 3.0):
            drivers.discard(c)
            rejected[name] = dict(reason='degrades the existing reconstruction', before_px=before,
                                  after_px=after_old)
            log('sfm: REJECT {}: {}'.format(name, rejected[name]))
            continue
        ext = trial
        nw = n_windows(r['inlier_frames'])
        cam_stats[name] = dict(inlier_frames=r['n_inlier_frames'], n_frames=r['n_frames'],
                               inlier_windows=nw, frac=frac, via_prior=bool(r.get('via_prior')),
                               driver=is_driver)
        if not is_driver:
            passengers[name] = dict(cam_stats[name])
        p3d, good, err, ks = p3d_t, good_t, err_t, ks_t
        order += 1
        log('sfm: placed {} {}/{} frames, {} windows, {:.2f} px{}'.format(
            name, r['n_inlier_frames'], r['n_frames'], nw, _med(err), '' if is_driver else ' [passenger]'))
        if local_ba_every and (order % local_ba_every == 0 or not remaining):
            ks = sorted(ext)
            work = _frame_mask(u[ks], fid, err, decoy_thresh, mfp)
            work[~(err <= max(pnp_thresh, 3 * _med(err)))] = np.nan  # single-view outliers
            cloud = np.where(good[:, None], p3d, np.nan)
            kw = dict(fix=(ks.index(i0),), rng=rng, reject_px=pnp_thresh)
            kw.update(local_ba_kwargs or {})
            Ms = refine_extrinsics(work, f[ks], np.stack([ext[k] for k in ks]), cloud, **kw)
            ext = {k: M for k, M in zip(ks, Ms)}
            p3d, good, err, ks = retri()
            log('sfm:   local BA -> {:.2f} px'.format(_med(err)))

    ks = sorted(ext)
    placed = [names[k] for k in ks]
    template = cgroup.subset_cameras_names(placed)
    out = from_extrinsics(placed, np.stack([ext[k] for k in ks]), template=template)
    per_cam = {names[k]: _med(err[i]) for i, k in enumerate(ks)}
    report = InitReport(
        method='sfm', placed=placed, drivers=[names[k] for k in sorted(drivers)],
        passengers=passengers, rejected=rejected,
        missing=[n for i, n in enumerate(names) if i not in ext and n not in rejected],
        stats=dict(seed_pair=[names[i0], names[j0]], seed_inliers=int(r0['n_in']),
                   n_pairs=len(pairs), cameras=cam_stats, median_px=_med(err), per_camera_px=per_cam,
                   n_points=int(good.sum()), gauge='seed camera {} = identity, seed baseline = 1'.format(names[i0])))
    return out, report


# ------------------------------------------------------------------ local refinement

def _skew(v):
    """[N,3] -> [N,3,3] cross-product matrices."""
    z = np.zeros(len(v))
    return np.stack([np.stack([z, -v[:, 2], v[:, 1]], -1),
                     np.stack([v[:, 2], z, -v[:, 0]], -1),
                     np.stack([-v[:, 1], v[:, 0], z], -1)], 1)


def _right_jacobian(r):
    """SO(3) right Jacobian of exp at rotation vectors ``r [N,3]``."""
    th = np.linalg.norm(r, axis=1)[:, None, None]
    S = _skew(r)
    small = th < 1e-8
    th = np.where(small, 1.0, th)
    a = np.where(small, 0.5, (1 - np.cos(th)) / th ** 2)
    b = np.where(small, 1.0 / 6, (th - np.sin(th)) / th ** 3)
    return np.eye(3)[None] - a * S + b * S @ S


def refine_extrinsics(u, f, Ms, p3d, fix=(0,), max_iter=60, max_points=1500, n_rounds=2,
                      reject_px=8.0, tol=1e-10, rng=None):
    """Sparse extrinsics + points bundle adjustment on ideal pixels: Levenberg-Marquardt with
    an analytic Jacobian and the Schur complement over the points (cameras' 6x6 reduced
    system solved densely). ``u [C,M,2]`` (NaN = unused), ``Ms [C,4,4]``, ``p3d [M,3]`` (NaN =
    unused point). Cameras in ``fix`` are held (the remaining scale gauge is handled by the
    damping). Robustness comes from trimming: after each of ``n_rounds`` fits, observations
    with residual above ``max(reject_px, 3 * median)`` are dropped. Scale-free (unlike the
    Adam BA, whose step size is in absolute parameter units). Returns refined ``Ms``."""
    from scipy.spatial.transform import Rotation
    rng = as_rng(rng)
    C = len(Ms)
    obs = np.all(np.isfinite(u), -1) & np.all(np.isfinite(p3d), -1)[None]
    pts = np.nonzero(obs.sum(0) >= 2)[0]
    if len(pts) < 10:
        return Ms
    if len(pts) > max_points:
        pts = np.sort(rng.choice(pts, max_points, replace=False))
    cam_i, pt_j = np.nonzero(obs[:, pts])
    n_obs, P = len(cam_i), len(pts)
    uu = u[cam_i, pts[pt_j]]
    fixset = set(int(c) for c in fix)
    free = [c for c in range(C) if c not in fixset]
    nf = len(free)
    pos = np.full(C, -1)
    pos[free] = np.arange(nf)
    R = np.stack([M[:3, :3] for M in Ms])
    t = np.stack([M[:3, 3] for M in Ms]).copy()
    rv = np.stack([rotvec(Ri) for Ri in R])
    X = p3d[pts].copy()
    fc = f[cam_i]
    fo = np.nonzero(pos[cam_i] >= 0)[0]          # observations of free cameras
    cf = pos[cam_i[fo]]
    # observation pairs sharing a point (both free cameras), for the Schur complement
    order = np.argsort(pt_j[fo], kind='stable')
    fo_sorted = fo[order]
    ptf = pt_j[fo_sorted]
    starts = np.r_[0, np.nonzero(np.diff(ptf))[0] + 1]
    ends = np.r_[starts[1:], len(ptf)]
    pa, pb = [], []
    for s0, s1 in zip(starts, ends):
        idx = np.arange(s0, s1)
        aa, bb = np.meshgrid(idx, idx, indexing='ij')
        pa.append(aa.ravel())
        pb.append(bb.ravel())
    pa = np.concatenate(pa) if pa else np.zeros(0, int)
    pb = np.concatenate(pb) if pb else np.zeros(0, int)
    # pa/pb index into fo_sorted; map to positions within fo
    inv = np.empty(len(fo), int)
    inv[order] = np.arange(len(fo))
    rank = np.argsort(inv)  # rank[k] = position in fo of fo_sorted[k]
    pa, pb = rank[pa], rank[pb]

    def residual(R, t, X):
        Y = np.einsum('nij,nj->ni', R[cam_i], X[pt_j]) + t[cam_i]
        z = np.where(np.abs(Y[:, 2]) > 1e-12, Y[:, 2], 1e-12)
        return fc[:, None] * Y[:, :2] / z[:, None] - uu, Y, z

    res, Y, z = residual(R, t, X)
    cost = float(np.sum(res ** 2))
    lam = 1e-3
    for it in range(max_iter):
        dY = np.zeros((n_obs, 2, 3))
        dY[:, 0, 0] = fc / z
        dY[:, 1, 1] = fc / z
        dY[:, 0, 2] = -fc * Y[:, 0] / z ** 2
        dY[:, 1, 2] = -fc * Y[:, 1] / z ** 2
        Ri = R[cam_i]
        Jp = dY @ Ri                                                     # [n,2,3]
        V = np.zeros((P, 3, 3))
        np.add.at(V, pt_j, np.einsum('nki,nkj->nij', Jp, Jp))
        gp = np.zeros((P, 3))
        np.add.at(gp, pt_j, np.einsum('nki,nk->ni', Jp, res))
        if nf:
            Jr = -Ri[fo] @ _skew(X[pt_j[fo]]) @ _right_jacobian(rv[cam_i[fo]])
            Jc = np.concatenate([dY[fo] @ Jr, dY[fo]], -1)                # [nfo,2,6]
            U = np.zeros((nf, 6, 6))
            np.add.at(U, cf, np.einsum('nki,nkj->nij', Jc, Jc))
            gc = np.zeros((nf, 6))
            np.add.at(gc, cf, np.einsum('nki,nk->ni', Jc, res[fo]))
            W = np.einsum('nki,nkj->nij', Jc, Jp[fo])                    # [nfo,6,3]
        while True:
            Vd = V + lam * (np.einsum('nii->ni', V)[:, :, None] * np.eye(3) + 1e-12 * np.eye(3))
            Vinv = np.linalg.inv(Vd)
            if nf:
                Ud = U + lam * (np.einsum('nii->ni', U)[:, :, None] * np.eye(6) + 1e-12 * np.eye(6))
                S = np.zeros((nf * 6, nf * 6))
                for k in range(nf):
                    S[6 * k:6 * k + 6, 6 * k:6 * k + 6] += Ud[k]
                ptp = pt_j[fo]
                blocks = W[pa] @ Vinv[ptp[pa]] @ np.transpose(W[pb], (0, 2, 1))  # [npair,6,6]
                Sb = np.zeros((nf, nf, 6, 6))
                np.add.at(Sb, (cf[pa], cf[pb]), blocks)
                S -= Sb.transpose(0, 2, 1, 3).reshape(nf * 6, nf * 6)
                rhs = -gc.copy()
                np.add.at(rhs, cf, np.einsum('nij,nj->ni', W, np.einsum('nij,nj->ni', Vinv[ptp], gp[ptp])))
                try:
                    dc = np.linalg.solve(S, rhs.ravel()).reshape(nf, 6)
                except np.linalg.LinAlgError:
                    dc = np.linalg.lstsq(S, rhs.ravel(), rcond=None)[0].reshape(nf, 6)
                wd = np.zeros((P, 3))
                np.add.at(wd, ptp, np.einsum('nji,nj->ni', W, dc[cf]))
            else:
                dc = np.zeros((0, 6))
                wd = np.zeros((P, 3))
            dp = np.einsum('nij,nj->ni', Vinv, -gp - wd)
            rv2, t2 = rv.copy(), t.copy()
            if nf:
                rv2[free] += dc[:, :3]
                t2[free] += dc[:, 3:]
            R2 = R.copy()
            if nf:
                R2[free] = Rotation.from_rotvec(rv2[free]).as_matrix()
            X2 = X + dp
            res2, Y2, z2 = residual(R2, t2, X2)
            cost2 = float(np.sum(res2 ** 2))
            if np.isfinite(cost2) and cost2 < cost:
                break
            lam *= 5
            if lam > 1e10:
                break
        if not (np.isfinite(cost2) and cost2 < cost):
            break
        done = (cost - cost2) <= tol * cost
        R, rv, t, X, res, Y, z, cost = R2, rv2, t2, X2, res2, Y2, z2, cost2
        lam = max(lam / 3, 1e-9)
        if done:
            break
    if n_rounds > 1:
        rn = np.linalg.norm(res, axis=1)
        bad = rn > max(reject_px, 3 * np.median(rn))
        if bad.any():
            u2 = np.array(u, copy=True)
            u2[cam_i[bad], pts[pt_j[bad]]] = np.nan
            X3 = np.array(p3d, copy=True)
            X3[pts] = X
            Mn = np.stack([make_M(R[c], t[c]) for c in range(C)])
            return refine_extrinsics(u2, f, Mn, X3, fix=fix, max_iter=max_iter,
                                     max_points=max_points, n_rounds=n_rounds - 1,
                                     reject_px=reject_px, tol=tol, rng=rng)
    out = np.stack([make_M(R[c], t[c]) for c in range(C)])
    return out if np.all(np.isfinite(out)) else Ms


def refine_group(cgroup, p2ds, frame_id=None, n_rounds=4, fix=0, decoy_thresh=25.0,
                 reject_px=8.0, max_points=3000, rng=None):
    """Extrinsics-only robust refinement of a whole rig on its 2D data (intrinsics fixed):
    ``n_rounds`` x (IRLS re-triangulation with (camera, frame) decoy masking, then
    :func:`refine_extrinsics` on the observations within ``max(reject_px, 3 x the camera's
    lower-quartile error)``). The gates are per camera and start loose, so a rough init (e.g. VGGT)
    or one misplaced camera is pulled in, then they tighten.
    Camera ``fix`` (index or name) is held. Returns ``(CameraGroup, stats)``; the input is
    not modified."""
    rng = as_rng(rng)
    names = cgroup.get_names()
    if isinstance(fix, str):
        fix = names.index(fix)
    flat, fid, _ = _flatten(p2ds, frame_id, None)
    mfp = _auto_min_frame_pts(fid)
    u, f = ideal_points(cgroup, flat)
    Ms = get_extrinsics(cgroup)
    ks = list(range(len(names)))
    hist = []
    for it in range(n_rounds):
        # gates are per camera and relative to that camera's current error, so a badly placed
        # camera keeps its data (and is pulled in) instead of being masked out for good
        p3d, good, err, work = _retriangulate(u, f, Ms, ks, set(ks), fid, np.inf, np.inf,
                                              min_frame_pts=mfp)
        hist.append(_med(err))
        if it >= 2 and abs(hist[-2] - hist[-1]) <= 0.01 * hist[-1]:
            break
        # lower quartile: a decoy camera's good frames set its gate, its decoy frames fail it
        cam_q = np.array([np.quantile(e[np.isfinite(e)], 0.25) if np.isfinite(e).any() else np.nan
                          for e in err])
        gate = np.maximum(reject_px, 3 * np.nan_to_num(cam_q, nan=reject_px))
        p3d, good, err, work = _retriangulate(u, f, Ms, ks, set(ks), fid,
                                              np.maximum(decoy_thresh, gate), np.inf,
                                              min_frame_pts=mfp)
        work = np.array(work, copy=True)
        work[~(err <= gate[:, None])] = np.nan
        Ms = refine_extrinsics(work, f, Ms, p3d, fix=(fix,), n_rounds=1, max_points=max_points,
                               rng=rng)
    _, good, err, _ = _retriangulate(u, f, Ms, ks, set(ks), fid, decoy_thresh, np.inf,
                                     min_frame_pts=mfp)
    hist.append(_med(err))
    out = from_extrinsics(names, Ms, template=cgroup)
    return out, dict(median_px_history=hist, median_px=hist[-1],
                     per_camera_px={n: _med(err[i]) for i, n in enumerate(names)})
