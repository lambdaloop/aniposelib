"""Calibration workflows (plan 05 W1-W4): board / object calibration, calibration from 2D
points, metric scale and calibration QA. Composition only: the maths lives in the backend
modules (:mod:`~aniposelib.intrinsics`, :mod:`~aniposelib.init`, :mod:`~aniposelib.bundle`,
:mod:`~aniposelib.scale`, :mod:`~aniposelib.quality`, :mod:`~aniposelib.stations`).

API
---
``calibrate_from_board(videos_or_rows, board, cam_names=None, fisheye=False, intrinsics='auto',
                       init=None, detector='auto', bridge_points=None, out=None, options=None,
                       rng=0) -> BoardCalibration``
    W1. Board (``CharucoBoard`` / ``Checkerboard``) or known-3D-points object
    (:class:`~aniposelib.boards.PointObject`, W1b) calibration: detection (videos only;
    DeepCharuco when the ``[deepcharuco]`` extra is installed, else OpenCV), per-camera lens
    model on held-out views (``intrinsics.calibrate_ladder``; per-view DLT for non-planar
    objects), board-graph extrinsics (``init.board``), two-pass Levenberg-Marquardt board BA
    (``bundle.bundle_adjust_lm``). A disconnected board graph gives ``verdict='fail'`` with the
    per-component groups; ``bridge_points=`` (2D tracks seen across components) solves one 6-DoF
    pose per extra component with intrinsics and metric scale fixed.
``calibrate_from_points(points, intrinsics, init='auto', frames=None, groups=None,
                        refine_intrinsics='auto', scale=None, stations=None, options=None, rng=0,
                        device='auto', out=None) -> PointCalibration``
    W2. Extrinsics (and, when the held-out evidence supports it, focal + k1) from 2D
    correspondences: candidate inits (reference / SfM / VGGT / people / given groups) ranked
    by ``init.select``, then ``bundle.bundle_adjust_robust`` with a noise-normalised threshold
    schedule, optional stations and metric scale (W3), then ``quality.report``.
``estimate_scale(cgroup, board=None, board_points=None, lengths=None, reference=None,
                 align='auto', apply=True, options=None, *, object=None, object_points=None)
                 -> ScaleEstimate``
    W3. Metric scale from exactly one source (board / object detections, known lengths, a
    reference rig); refuses (``accepted=False``, ``verdict='fail'``) instead of applying a
    doubtful scale. The input group is never modified: ``result.cgroup`` is a rescaled copy.
``evaluate_calibration(cgroup, points, times=None, reference=None, board=None, options=None, *,
                       board_points=None, object=None, object_points=None)
                       -> CalibrationEvaluation``
    W4. QA: leave-one-camera-out reprojection (fixed denominator), coverage, inlier rates,
    lens validity, ray angles; moved cameras (``stations.detect_moves``) with ``times``;
    ``quality.compare`` with ``reference``; the board shape residual with a board.

Results subclass :class:`~aniposelib.workflows._common.WorkflowResult` (``report``,
``verdict``, ``reasons``, ``settings``, ``provenance``, ``to_dict()``, ``to_json()``,
``summary()``). Calibration results unpack as ``cgroup, result = calibrate_from_board(...)``.

Conventions
-----------
* numpy in -> numpy out; points are ``C x N x 2`` / ``C x N x J x 2`` pixels (NaN = missing),
  in camera order. ``rng=0`` by default.
* No unit-dependent knobs. Pixel thresholds derive from the robust noise floor
  (:func:`~aniposelib.workflows._common.noise_floor`, >= 0.5 px) through the *noise unit*
  ``u = clip(noise_floor / 0.5 px, 1, 2)``: the backend defaults (tuned for 0.5 px noise) are
  multiplied by ``u`` (``options={'noise_unit': 1}`` = the plain pixel defaults). First-pass thresholds are fractions of the image diagonal. Lengths are relative to
  the rig or object size. Every derived value is in ``result.settings``.
* Verdicts follow the rules listed in each docstring; override them with
  ``options={'verdict': {rule: (warn_at, fail_at) | None}}``. Workflows raise only on unusable
  inputs (wrong shapes, missing image sizes, contradictory arguments).
"""

import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from ._common import (MIN_NOISE_PX, Rule, Timer, WorkflowResult, evaluate_rules, image_diagonal,
                      merge_options, noise_floor, provenance, save_calibration, thread_guard, to_numpy)

__all__ = ['calibrate_from_board', 'BoardCalibration', 'calibrate_from_points', 'PointCalibration',
           'estimate_scale', 'ScaleEstimate', 'evaluate_calibration', 'CalibrationEvaluation']


# ================================================================== shared helpers

def _noise_unit(nf, override=None):
    """Noise unit ``u`` (capped; see :func:`aniposelib.workflows._common.noise_unit`).
    ``override``: a workflow's ``options['noise_unit']`` ('auto' / None, or a float)."""
    from ._common import noise_unit
    return noise_unit(nf, override)


def _sigma_opt(v, default):
    """A prior sigma option: a float, or ``{camera name: sigma}`` (+ ``'default'``, else ``default``)."""
    if isinstance(v, dict):
        out = dict(v)
        out.setdefault('default', default)
        return out
    return v


def _check_noise_unit(v):
    if v in (None, 'auto'):
        return
    try:
        ok = float(v) > 0
    except (TypeError, ValueError):
        ok = False
    if not ok:
        raise ValueError("options['noise_unit'] must be 'auto' or a positive number, got {!r}".format(v))


def _device(device):
    if device in (None, 'auto'):
        return 'cuda' if torch.cuda.is_available() else 'cpu'
    return str(device)


def _on_device(cgroup, device):
    """``cgroup`` itself if it already lives on ``device``, else a copy moved there."""
    dev = _device(device)
    cur = str(cgroup.device)
    if cur == dev or (dev == 'cuda' and cur.startswith('cuda')):
        return cgroup
    return cgroup.copy().to(dev)


def _flat(p):
    p = np.asarray(to_numpy(p), dtype='float64')
    return p.reshape(p.shape[0], -1, 2)


def _diag(cgroup, points=None):
    d = image_diagonal(cgroup)
    if d is None and points is not None:
        f = _flat(points)
        with np.errstate(all='ignore'):
            mx = np.nanmax(f.reshape(-1, 2), axis=0)
        d = float(np.hypot(*mx)) if np.all(np.isfinite(mx)) else None
    return d


def _robust_points(cgroup, points, tau, min_inliers=2):
    """Robust (MSAC) triangulation: ``(p3d [N,3], errors [C,N], inliers [C,N])`` numpy."""
    from .. import geometry as G, triangulation as T
    P = G.pack(cgroup)
    flat = torch.as_tensor(_flat(points), dtype=P.dtype, device=P.device)
    with torch.no_grad():
        R = T.triangulate_robust(P, flat, 'msac', tau=float(tau), min_inliers=min_inliers)
    return to_numpy(R.p3d), to_numpy(R.errors), to_numpy(R.inliers)


def _points_noise(cgroup, points, diag):
    """Noise floor of 2D points under a calibration: robust triangulation with a loose
    threshold (2 % of the diagonal), median of the inlier errors of points with >= 3 inlier
    views. Returns ``(noise_floor_px, errors [C,N])``."""
    tau = 0.02 * diag if diag else 20.0
    _, err, inl = _robust_points(cgroup, points, tau)
    sup = inl.sum(0) >= min(3, inl.shape[0])
    return noise_floor(err[:, sup], inl[:, sup], diag=diag), err


def _rules(rules, opts):
    for r in rules:   # readable thresholds (4 significant digits)
        for a in ('warn_at', 'fail_at'):
            v = getattr(r, a)
            if isinstance(v, float) and np.isfinite(v):
                setattr(r, a, float('{:.4g}'.format(v)))
    return evaluate_rules(rules, opts.get('verdict'))



# ================================================================== W1 calibrate_from_board

W1_DEFAULTS = dict(
    sizes=None,               # per-camera (w, h) when neither the videos nor `init` give them
    min_corners=6,            # corners per view used for intrinsics (planar boards)
    ladder_models=None,       # default: intrinsics.DEFAULT_MODELS / FISHEYE_MODELS per camera
    min_n_dist=2,             # board BA frees at least this many leading distortion terms
    n_iter=50,                # LM iterations per BA pass
    f_scale_diag=0.003,       # first-pass Cauchy scale, fraction of the image diagonal
    f_scale_noise=3.0,        # second-pass Cauchy scale, multiple of the noise floor
    max_points=None,          # subsample boards in BA
    # video detection (frame-major, sequential reading): every `skip`-th frame, or `n_frames`
    # evenly spaced, or `frame_indices`; stop once every camera has `max_boards` boards (None: no limit)
    skip=20, n_frames=None, frame_indices=None, max_boards=200, n_workers='auto', detector_kwargs=None,
    prior_sigma_logf=0.05, prior_sigma_k=0.1,  # intrinsics=<CameraGroup> (float or {camera name: sigma})
    object_sigma=None,        # static PointObject: not supported yet (points held fixed)
    min_bridge_points=6,      # bridge: minimum points per extra component
    focal_guess=None,         # lens ladder: extra focal starts (px), a float or {camera name: f} (e.g. phones)
    runaway_ratio=10.0,       # verdict: camera centre farther than this x the median distance from the rig centre
    board_ratio=(3.0, 8.0),   # verdict: worst camera's board error / median camera's (warn, fail)
    noise_unit='auto',        # bridge MSAC threshold = 10 noise units; a float forces the unit
    verdict=None,
)


@dataclass(repr=False)
class BoardCalibration(WorkflowResult):
    """Result of :func:`calibrate_from_board`. ``cgroup`` is None when there is no single
    calibration (disconnected board graph without a bridge); then ``component_groups`` holds
    one calibrated group per connected component (each in its own gauge, metric scale of the
    board). ``detections``: per-camera rows (with board poses); ``lens_models``: per camera,
    the selected lens model and the ladder table (or the DLT / fallback used)."""
    cgroup: Any = None
    detections: Any = None
    lens_models: Dict[str, Any] = field(default_factory=dict)
    components: List[List[str]] = field(default_factory=list)
    component_groups: List[Any] = field(default_factory=list)
    bridge: Optional[Dict[str, Any]] = None

    _skip = ('cgroup', 'detections', 'component_groups')

    def __iter__(self):
        yield self.cgroup
        yield self

    def to_dict(self):
        d = super().to_dict()
        d['component_groups'] = [g.get_names() for g in self.component_groups]
        return d


def _is_videos(x):
    if isinstance(x, (list, tuple)) and len(x) and all(isinstance(v, (list, tuple)) for v in x):
        vals = [v for cam in x for v in cam]
        return bool(vals) and all(isinstance(v, (str, os.PathLike)) for v in vals)
    return False


def _rows_from_dense(arr, board):
    if hasattr(board, 'rows_from_array'):
        return board.rows_from_array(arr)
    arr = np.asarray(arr, dtype='float64')
    K = board.get_empty_detection().reshape(-1, 2).shape[0]
    if arr.ndim != 4 or arr.shape[2] != K or arr.shape[3] != 2:
        raise ValueError('dense detections must be C x F x {} x 2, got {}'.format(K, arr.shape))
    out = []
    for c in range(arr.shape[0]):
        rows = []
        for f in range(arr.shape[1]):
            good = np.all(np.isfinite(arr[c, f]), axis=1)
            if good.any():
                ids = np.nonzero(good)[0]
                rows.append({'framenum': (0, f), 'corners': arr[c, f, ids].reshape(-1, 1, 2).astype('float32'),
                             'ids': ids.reshape(-1, 1).astype('int32')})
        out.append(rows)
    return out


def _copy_rows(all_rows, board):
    out = []
    for rows in all_rows:
        if not isinstance(rows, (list, tuple)) or any(not isinstance(r, dict) for r in rows):
            raise ValueError('rows must be one list of detection dicts per camera')
        cr = []
        for r in rows:
            r = dict(r)
            if r.get('corners') is None:
                continue
            r.pop('rvec', None)
            r.pop('tvec', None)
            r['filled'] = board.fill_points(r['corners'], r.get('ids'))
            cr.append(r)
        out.append(cr)
    return out


def _deepcharuco_available(board):
    if not hasattr(board, 'detector_name') or not hasattr(board, 'marker_length'):
        return False
    try:
        import importlib.util
        if importlib.util.find_spec('huggingface_hub') is None and \
                importlib.util.find_spec('onnxruntime') is None:
            return False
        from ..detectors.deepcharuco import supports_board
        return bool(supports_board(board.squaresX, board.squaresY))
    except Exception:  # pragma: no cover
        return False


def _resolve_detector(board, detector):
    det = detector
    if det == 'auto':
        det = 'deepcharuco' if _deepcharuco_available(board) else 'opencv'
    if not hasattr(board, 'detector_name'):
        det = None
    return det


def _n_workers(n):
    if n is None or n == 'auto':
        try:
            ncpu = len(os.sched_getaffinity(0))
        except (AttributeError, OSError):  # pragma: no cover
            ncpu = os.cpu_count() or 1
        return max(1, min(8, ncpu))
    return max(1, int(n))


def _detect_videos(videos, board, det, opts, names, *, max_boards=None, max_frames=None, min_cameras=2):
    """W1 / W3 board detection in videos: ``videos`` one list of paths per camera (video j of
    every camera = one synchronised segment). Frame-major sequential sampling
    (:func:`aniposelib.boards.detect_videos`): every ``skip``-th frame, or ``n_frames`` evenly
    spaced frames of the segment's shortest video, or ``frame_indices`` (one int sequence, the
    same in every camera); early exit at ``max_boards`` / ``max_frames``. A per-camera / per-video
    ``frame_indices`` falls back to the per-video reader (``CameraGroup.get_rows_videos``).
    Returns per-camera rows (``framenum = (segment, frame)``) and a summary dict."""
    from ..boards import detect_videos
    from ..cameras import _board_with_detector, _is_index_seq
    dkw = opts.get('detector_kwargs')
    if det == 'deepcharuco':
        # input scale from the frame size unless given (scale 1 misses most boards in >= 1080p)
        dkw = dict(getattr(board, 'detector_kwargs', None) or {}) if dkw is None else dict(dkw)
        dkw.setdefault('scale', 'auto')
    b = board if det is None else _board_with_detector(board, det, dkw)
    nw = _n_workers(opts.get('n_workers'))
    fi = opts.get('frame_indices')
    C = len(videos)
    t0 = time.time()
    summary = dict(sampling='sequential', n_workers=nw, skip=int(opts.get('skip') or 1),
                   n_frames=opts.get('n_frames'), max_boards=max_boards, max_frames=max_frames,
                   segments=[])
    if fi is not None and not _is_index_seq(fi):
        from ..cameras import CameraGroup, Camera
        cg = CameraGroup([Camera(name=n) for n in names])
        rows = cg.get_rows_videos(videos, b, verbose=False, frame_indices=fi, n_workers=nw)
        summary['sampling'] = 'per_video_indices'
    else:
        rows = [[] for _ in range(C)]
        n_seg = max((len(v) for v in videos), default=0)
        for j in range(n_seg):
            seg = [v[j] if len(v) > j else None for v in videos]
            idx = None
            if fi is not None:
                idx = [int(i) for i in fi]
            elif opts.get('n_frames'):
                from ..boards import _video_length
                counts = [c for c in (_video_length(v) for v in seg if v is not None) if c > 0]
                if counts:
                    idx = [int(f) for f in np.unique(np.linspace(0, min(counts) - 1, int(opts['n_frames']))
                                                     .round().astype(int))]
            info = {}
            for c, v in enumerate(seg):
                if v is not None and not os.path.exists(str(v)):
                    raise FileNotFoundError('missing video file "{}"'.format(v))
            seg_rows = detect_videos(b, seg, frame_indices=idx, skip=opts.get('skip') or 1,
                                     max_boards=max_boards, max_frames=max_frames,
                                     min_cameras=min(min_cameras, C), n_workers=nw, prefix=j, info=info)
            for c in range(C):
                rows[c].extend(seg_rows[c])
            info['frame_indices'] = idx
            summary['segments'].append(info)
    summary['n_rows'] = [len(r) for r in rows]
    summary['seconds'] = round(time.time() - t0, 2)
    return rows, summary


def _base_group(C, cam_names, init, fisheye, sizes):
    from ..cameras import Camera, CameraGroup, FisheyeCamera
    if init is not None:
        cg = init.copy()
        if cam_names is not None and list(cam_names) != cg.get_names():
            cg = cg.subset_cameras_names(list(cam_names))
        if len(cg.cameras) != C:
            raise ValueError('init has {} cameras, the detections have {}'.format(len(cg.cameras), C))
    else:
        names = list(cam_names) if cam_names is not None else ['cam_{}'.format(i) for i in range(C)]
        if len(names) != C:
            raise ValueError('{} camera names for {} cameras'.format(len(names), C))
        fish = list(fisheye) if isinstance(fisheye, (list, tuple)) else [bool(fisheye)] * C
        if len(fish) != C:
            raise ValueError('fisheye: {} flags for {} cameras'.format(len(fish), C))
        cg = CameraGroup([FisheyeCamera(name=str(n)) if f else Camera(name=str(n))
                          for n, f in zip(names, fish)])
    if sizes is not None:
        s = np.asarray(sizes)
        s = np.tile(s, (C, 1)) if s.ndim == 1 else s
        for cam, sz in zip(cg.cameras, s):
            cam.set_size((int(sz[0]), int(sz[1])))
    return cg


def _set_intrinsics(cam, K, dist=None, n_dist=None):
    cam.set_camera_matrix(np.asarray(K, dtype='float64'))
    nd = to_numpy(cam.get_distortions()).shape[0]
    d = np.zeros(nd)
    if dist is not None:
        dd = np.asarray(dist, dtype='float64').ravel()[:nd]
        d[:len(dd)] = dd
    cam.set_distortions(d)
    if n_dist is not None:
        cam.set_n_dist(int(min(n_dist, nd)))


def _given_intrinsics(cam):
    """Does this camera carry real intrinsics (not the default identity matrix)?"""
    K = to_numpy(cam.get_camera_matrix()).astype(float)
    return bool(np.all(np.isfinite(K)) and K[0, 0] > 1 and K[1, 1] > 1 and not np.allclose(K, np.eye(3)))


def _fit_intrinsics(cg, rows, board, mode, init, opts, rng):
    """Per-camera intrinsics. Returns ``(lens_models, missing, fallbacks)``."""
    from .. import intrinsics as I
    from ..init import board as IB
    lens, missing, fallbacks = {}, [], []
    names = cg.get_names()
    point_object = hasattr(board, 'estimate_intrinsics')
    planar = not point_object or board.planar
    if mode == 'fixed':
        for n in names:
            lens[n] = dict(source='fixed')
        return lens, missing, fallbacks
    prior = mode if hasattr(mode, 'cameras') else None
    for cam, r, n in zip(cg.cameras, rows, names):
        size = cam.get_size()
        has_init = init is not None and _given_intrinsics(cam)
        n_obs = int(sum(np.isfinite(np.asarray(x['filled'], float).reshape(-1, 2)).all(1).sum() for x in r))
        if prior is not None:
            pc = prior.cameras[prior.get_names().index(n)] if n in prior.get_names() else None
            if pc is not None:
                _set_intrinsics(cam, to_numpy(pc.get_camera_matrix()), to_numpy(pc.get_distortions()),
                                max(pc.get_n_dist(), opts['min_n_dist']))
                lens[n] = dict(source='prior')
                continue
        if planar:
            fish = bool(getattr(cam, 'fisheye', False))
            models = opts['ladder_models'] or (I.FISHEYE_MODELS if fish else I.DEFAULT_MODELS)
            if fish:
                models = ('pinhole',) + tuple(m for m in models if m != 'pinhole')
            fg = opts['focal_guess']
            fg = fg.get(n) if isinstance(fg, dict) else fg
            lkw = dict(focal_inits=[float(v) for v in np.atleast_1d(fg)]) if fg is not None else {}
            try:
                lr = I.calibrate_ladder_rows(r, board, size, min_corners=opts['min_corners'], models=models, **lkw)
                if fish and not I.is_fisheye(lr.selected):
                    fits = [m for m in lr.fits if I.is_fisheye(m)]
                    sel = fits[0] if fits else None
                else:
                    sel = lr.selected
                if sel is None or sel == 'rational_8param':
                    raise ValueError('no usable lens model for this camera type')
                fit = lr.fits[sel]
                _set_intrinsics(cam, fit.K, fit.dist, max(fit.n_params - 3, opts['min_n_dist']))
                lens[n] = dict(source='ladder', selected=sel, table=[list(t) for t in lr.table()],
                               n_views=len(lr.train_idx) + len(lr.heldout_idx))
                continue
            except Exception as exc:
                err = '{}: {}'.format(type(exc).__name__, exc)
            objp, imgp = board.get_all_calibration_points(r, min_points=max(opts['min_corners'], 4))
            if len(objp) >= 1:
                seed = IB.init_camera_intrinsics(cam, objp, imgp, size,
                                                 fallback='keep' if has_init else 'nominal')
                cam.set_n_dist(min(opts['min_n_dist'], to_numpy(cam.get_distortions()).shape[0]))
                lens[n] = dict(source='seed:' + seed.source, reason=seed.reason, ladder_error=err)
                fallbacks.append('{}: ladder failed ({}), {} intrinsics'.format(n, err, seed.source))
                if seed.source == 'fallback:nominal':
                    fallbacks.append('{}: nominal intrinsics'.format(n))
                continue
            if has_init:
                lens[n] = dict(source='init', reason='no board views')
                fallbacks.append('{}: no board views, intrinsics from init'.format(n))
            else:
                lens[n] = dict(source=None, reason='no board views')
                missing.append(n)
        else:
            K, info = board.estimate_intrinsics(r, size)
            if K is not None:
                nd = 1 if n_obs >= 100 else 0
                _set_intrinsics(cam, K, None, nd)
                lens[n] = dict(source='dlt', n_dist=nd, **info)
            elif has_init:
                lens[n] = dict(source='init', reason='no view with >= 6 non-coplanar points')
                fallbacks.append('{}: DLT impossible, intrinsics from init'.format(n))
            else:
                lens[n] = dict(source=None, reason='no view with >= 6 non-coplanar annotated points')
                missing.append(n)
    return lens, missing, fallbacks


def _board_ba(cg, imgp, extra, spec, diag, opts, rng, only_extrinsics=False, prune=None):
    """Two-pass LM board BA: Cauchy scale from the diagonal, then from the noise floor."""
    from .. import bundle as B
    f1 = opts['f_scale_diag'] * diag
    kw = dict(extra=extra, spec=spec, loss='cauchy', n_iter=opts['n_iter'], return_report=True,
              max_points=opts['max_points'], rng=rng, only_extrinsics=only_extrinsics, prune=prune)
    r1 = B.bundle_adjust_lm(cg, imgp, f_scale=f1, **kw)
    nf = noise_floor(r1.obs_errors, diag=diag)
    f2 = opts['f_scale_noise'] * nf
    r2 = B.bundle_adjust_lm(cg, imgp, f_scale=f2, **kw)
    nf = noise_floor(r2.obs_errors, diag=diag)
    return r2, dict(f_scale_pass1_px=f1, f_scale_pass2_px=f2, noise_floor_px=nf,
                    error_before_px=r1.error_before, error_px=r2.error)


def _static_anchor(cg, rows, board):
    """Re-express the world in the object's frame: rigid (no scale) fit of the object points
    triangulated by the calibrated rig onto ``board.points3d``."""
    from .. import scale as S
    K = board.n_points
    pts = np.full((len(cg.cameras), K, 2), np.nan)
    for c, r in enumerate(rows):
        if r:
            pts[c] = np.asarray(r[0]['filled'], float).reshape(K, 2)
    X = to_numpy(cg.triangulate(pts))
    ok = np.isfinite(X).all(1)
    if ok.sum() < 3:
        return None
    s, R, t, res, use = S.fit_similarity(X[ok], board.points3d[ok], trim=np.inf, min_keep=3, with_scale=False)
    S.apply_similarity(cg, 1.0, R, t)
    return dict(residual_median=float(np.median(res)), n_points=int(ok.sum()))


def _bridge_components(cg, comp_idx, bridge, nf, diag, opts):
    """Place components 1.. relative to component 0 (and already bridged ones) from 2D tracks
    ``bridge [C,N,2]``: robust points from the placed cameras, PnP of the component's best
    camera, then a robust 6-DoF refinement of the component's rigid transform on all its
    cameras (ideal pixels; intrinsics and scale fixed). Returns a report dict."""
    import cv2
    from scipy.optimize import least_squares
    from ..boards import pnp_inputs
    from ..init._common import M_from_rt, rotvec
    C = len(cg.cameras)
    pts = _flat(bridge)
    Ms = np.stack([to_numpy(c.get_extrinsics_mat()).astype('float64') for c in cg.cameras])
    # normalised (undistorted) observations per camera, and focal lengths
    norm = np.full_like(pts, np.nan)
    focal = np.zeros(C)
    for c, cam in enumerate(cg.cameras):
        v = np.isfinite(pts[c]).all(1)
        if v.any():
            norm[c, v] = np.asarray(to_numpy(cam.undistort_points(pts[c, v].reshape(-1, 1, 2)))).reshape(-1, 2)
        K = to_numpy(cam.get_camera_matrix())
        focal[c] = 0.5 * (K[0, 0] + K[1, 1])
    tau = 10 * _noise_unit(nf, opts.get('noise_unit'))
    placed = set(comp_idx[0])
    todo = list(range(1, len(comp_idx)))
    out = dict(components=[], tau_px=tau, unbridged=[])
    while todo:
        sub = cg.subset_cameras_names([cg.get_names()[i] for i in sorted(placed)])
        pidx = sorted(placed)
        X, _, inl = _robust_points(sub, pts[pidx], tau)
        good = (inl.sum(0) >= 2) & np.isfinite(X).all(1)
        best, best_n = None, 0
        for k in todo:
            n = sum(int((good & np.isfinite(pts[c]).all(1)).sum()) for c in comp_idx[k])
            if n > best_n:
                best, best_n = k, n
        if best is None or best_n < opts['min_bridge_points']:
            out['unbridged'] = [[cg.get_names()[c] for c in comp_idx[k]] for k in todo]
            break
        comp = comp_idx[best]
        counts = [int((good & np.isfinite(pts[c]).all(1)).sum()) for c in comp]
        cb = comp[int(np.argmax(counts))]
        m = good & np.isfinite(pts[cb]).all(1)
        ip, Kp, Dp, pix = pnp_inputs(cg.cameras[cb], pts[cb, m].reshape(-1, 1, 2))
        ok, rv, tv, _ = cv2.solvePnPRansac(np.ascontiguousarray(X[m].reshape(-1, 1, 3)),
                                           np.ascontiguousarray(np.asarray(ip, float).reshape(-1, 1, 2)),
                                           Kp, Dp, reprojectionError=float(tau * pix),
                                           iterationsCount=500, confidence=0.999)
        if not ok:
            out['unbridged'].append([cg.get_names()[c] for c in comp])
            todo.remove(best)
            continue
        Ew = M_from_rt(np.asarray(rv).ravel(), np.asarray(tv).ravel())
        T0 = np.linalg.inv(Ms[cb]) @ Ew       # world -> component frame
        obs = [(c, np.nonzero(good & np.isfinite(norm[c]).all(1))[0]) for c in comp]

        def resid(x):
            T = M_from_rt(x[:3], x[3:])
            res = []
            for c, idx in obs:
                E = Ms[c] @ T
                Y = X[idx] @ E[:3, :3].T + E[:3, 3]
                z = np.where(Y[:, 2] > 1e-12, Y[:, 2], 1e-12)
                res.append(((Y[:, :2] / z[:, None]) - norm[c, idx]).ravel() * focal[c])
            return np.concatenate(res) if res else np.zeros(1)
        x0 = np.r_[rotvec(T0[:3, :3]), T0[:3, 3]]
        sol = least_squares(resid, x0, loss='soft_l1', f_scale=3 * max(nf, MIN_NOISE_PX), max_nfev=200)
        T = M_from_rt(sol.x[:3], sol.x[3:])
        r = np.abs(resid(sol.x)).reshape(-1, 2)
        e = np.hypot(r[:, 0], r[:, 1])
        for c in comp:
            Ms[c] = Ms[c] @ T
            cg.cameras[c].set_rotation(rotvec(Ms[c][:3, :3]))
            cg.cameras[c].set_translation(Ms[c][:3, 3])
        out['components'].append(dict(cameras=[cg.get_names()[c] for c in comp], via=cg.get_names()[cb],
                                      n_points=int(best_n), median_px=float(np.median(e)),
                                      inlier_frac=float(np.mean(e < tau))))
        placed |= set(comp)
        todo.remove(best)
    return out


@thread_guard
def calibrate_from_board(videos_or_rows, board, cam_names=None, fisheye=False, intrinsics='auto',
                         init=None, detector='auto', bridge_points=None, out=None, options=None,
                         rng=0):
    """W1: calibrate a rig from a calibration board or a known-3D-points object.

    Parameters
    ----------
    videos_or_rows : per-camera lists of video paths, per-camera lists of detection rows
        (``board.detect_*`` / ``CameraGroup.get_rows_videos`` output), or a dense
        ``C x F x K x 2`` array of detections / annotations (NaN = missing; the only input of a
        :class:`~aniposelib.boards.PointObject`).
    board : ``CharucoBoard`` | ``Checkerboard`` | ``PointObject`` (its ``units`` are recorded).
    cam_names : camera names (default: ``init``'s, else ``cam_0..``).
    fisheye : bool or per-camera list (cameras built here; ignored with ``init``).
    intrinsics : ``'auto'`` (lens-model ladder on held-out views; per-view DLT for a non-planar
        ``PointObject``), ``'fixed'`` (keep ``init``'s intrinsics) or a ``CameraGroup`` (start
        from its intrinsics, refined in BA under priors: 5 % focal, 0.1 distortion).
    init : ``CameraGroup`` to start from (sizes, intrinsics; also the fallback intrinsics of
        cameras without enough views).
    detector : ``'auto'`` (DeepCharuco when the ``[deepcharuco]`` extra is installed and the
        board has published weights, else OpenCV), ``'opencv'``, ``'deepcharuco'`` or an
        instance. Videos only.
    bridge_points : ``C x N x 2`` (or ``C x N x J x 2``) 2D tracks seen across disconnected
        board-graph components; each extra component gets one 6-DoF transform (intrinsics and
        the board's metric scale fixed), then a joint extrinsics BA with the boards.
    out : path of a calibration TOML to write (plus ``<stem>.report.json``).
    options : overrides of :data:`W1_DEFAULTS` (``sizes``, ``ladder_models``, ``min_n_dist``,
        ``n_iter``, ..., ``verdict``).

    Returns :class:`BoardCalibration` (``cgroup, res = calibrate_from_board(...)``).

    Steps: detection (videos) -> per-camera intrinsics -> board poses (PnP) ->
    ``init.board.initial_extrinsics`` (a static ``PointObject`` places every camera directly by
    PnP in the object's frame) -> two-pass LM board BA (Cauchy scale ``f_scale_diag`` x
    diagonal, then ``f_scale_noise`` x noise floor) -> report (per-camera reprojection, board
    shape residual, lens validity).

    Verdict rules (``nf`` = noise floor of the BA residuals; override with
    ``options={'verdict': {...}}``):

    - ``board_reproj_px``: worst per-camera median reprojection error; warn above
      ``max(1 px, 3 nf)``, fail above twice that;
    - ``uncovered_cameras``: cameras without usable board views (fail if any);
    - ``missing_intrinsics``: cameras whose intrinsics could not be estimated or given (fail);
    - ``disconnected_components``: extra components of the board graph left unbridged (fail);
    - ``board_shape_rel``: median shape residual of the triangulated board / object size;
      warn 0.5 %, fail 2 %;
    - ``lens_invalid``: cameras whose lens model is not invertible over the image (warn);
    - ``board_reproj_ratio``: worst camera's median board error / the median camera's; warn above 3,
      fail above 8 (``options['board_ratio']``): one camera with a wrong lens skews the others;
    - ``runaway_cameras``: centres farther than ``options['runaway_ratio']`` (10) x the median
      distance from the rig centre (fail);
    - ``bridge_inlier_frac`` (with ``bridge_points``): smallest share of a bridged component's
      bridge observations that the final rig explains; warn below 0.5, fail below 0.2. With a
      bridge, the board rules use the board observations alone (re-triangulated), not the joint BA.

    Video detection (``options``): the videos of all cameras are read together, front to back
    (no seeking; video ``j`` of every camera is one synchronised segment), and the board is
    detected in every ``skip``-th frame (20), in ``n_frames`` evenly spaced frames, or in
    ``frame_indices`` (one int sequence, or the per-camera / per-video forms of
    ``CameraGroup.get_rows_videos``, which use the per-video reader). Reading stops once every
    camera has ``max_boards`` boards (200; None = read everything). ``n_workers``: decoding /
    OpenCV-detection threads (``'auto'`` = up to 8). ``settings['detection']`` reports what was
    read and why it stopped. A missing video raises ``FileNotFoundError``.

    ``options['focal_guess']`` (a float or ``{camera name: f}``, px) adds focal starts to the lens
    ladder (phones / unusual lenses); ``prior_sigma_logf`` / ``prior_sigma_k`` may be
    ``{camera name: sigma, 'default': sigma}``; ``noise_unit`` forces the bridge noise unit.
    """
    from .. import quality as Q
    from ..boards import merge_rows, extract_points, extract_rtvecs
    from ..init import board as IB
    from ..params import IntrinsicsPrior, ParamSpec
    timer = Timer()
    opts = merge_options(W1_DEFAULTS, options)
    fallbacks, settings, reasons_extra = [], {}, []
    point_object = hasattr(board, 'estimate_intrinsics')
    static = point_object and getattr(board, 'static', False)
    if intrinsics not in ('auto', 'fixed') and not hasattr(intrinsics, 'cameras'):
        raise ValueError("intrinsics must be 'auto', 'fixed' or a CameraGroup")
    if intrinsics == 'fixed' and init is None:
        raise ValueError("intrinsics='fixed' needs init= (a CameraGroup with the intrinsics)")
    # ---- inputs -> rows
    videos = None
    with timer('detect'):
        if _is_videos(videos_or_rows):
            if point_object:
                raise ValueError('a PointObject has no detector: pass annotations (C x F x K x 2)')
            videos = videos_or_rows
            C = len(videos)
            if cam_names is None and init is None:
                stems = [os.path.splitext(os.path.basename(str(v[0])))[0] if v else 'cam_{}'.format(i)
                         for i, v in enumerate(videos)]
                cam_names = stems if len(set(stems)) == C else None
            cg = _base_group(C, cam_names, init, fisheye, opts['sizes'])
            cg.set_camera_sizes_videos(videos)
            det = _resolve_detector(board, detector)
            settings['detector'] = det if isinstance(det, (str, type(None))) else type(det).__name__
            dkw = dict(max_boards=opts['max_boards'], min_cameras=2)
            try:
                all_rows, settings['detection'] = _detect_videos(videos, board, det, opts, cg.get_names(), **dkw)
            except Exception as exc:
                if det != 'deepcharuco' or detector != 'auto' or isinstance(exc, FileNotFoundError):
                    raise
                fallbacks.append('deepcharuco failed ({}: {}), using opencv'.format(type(exc).__name__, exc))
                settings['detector'] = 'opencv'
                all_rows, settings['detection'] = _detect_videos(videos, board, 'opencv', opts, cg.get_names(),
                                                                 **dkw)
        elif isinstance(videos_or_rows, np.ndarray) or torch.is_tensor(videos_or_rows):
            arr = np.asarray(to_numpy(videos_or_rows), dtype='float64')
            all_rows = _rows_from_dense(arr, board)
            cg = _base_group(len(all_rows), cam_names, init, fisheye, opts['sizes'])
            settings['detector'] = 'given'
        else:
            all_rows = list(videos_or_rows)
            cg = _base_group(len(all_rows), cam_names, init, fisheye, opts['sizes'])
            settings['detector'] = 'given'
    C = len(cg.cameras)
    names = cg.get_names()
    if len(all_rows) != C:
        raise ValueError('detections for {} cameras, expected {}'.format(len(all_rows), C))
    rows = _copy_rows(all_rows, board)
    if static:
        rows = _collapse_static(rows, board)
    for cam in cg.cameras:
        if cam.get_size() is None and intrinsics != 'fixed':
            raise ValueError('camera {} has no image size: pass videos, init= or options={{"sizes": ...}}'
                             .format(cam.get_name()))
    diag = _diag(cg, np.concatenate([np.asarray(r['filled'], float).reshape(1, -1, 2)
                                     for rr in rows for r in rr], axis=1)
                 if any(rows) else None) or 1000.0
    settings['image_diag_px'] = diag
    # ---- intrinsics
    with timer('intrinsics'):
        lens, missing, fb = _fit_intrinsics(cg, rows, board, intrinsics, init, opts, rng)
        fallbacks += fb
    settings['intrinsics'] = intrinsics if isinstance(intrinsics, str) else 'prior'
    # ---- board poses + extrinsics
    with timer('extrinsics'):
        for c, cam in enumerate(cg.cameras):
            if names[c] in missing:
                rows[c] = []
                continue
            board.estimate_pose_rows(cam, rows[c])
        merged = merge_rows(rows, cam_names=list(range(C)))
        imgp, extra = extract_points(merged, board, cam_names=list(range(C)),
                                     min_cameras=1 if static else 2)
        rtvecs = extract_rtvecs(merged, cam_names=list(range(C)))
        seen = np.isfinite(rtvecs).all(-1).any(1) if rtvecs.size else np.zeros(C, bool)
        uncovered = [n for n, s in zip(names, seen) if not s]
        if static:
            for c, cam in enumerate(cg.cameras):
                if seen[c]:
                    k = int(np.nonzero(np.isfinite(rtvecs[c]).all(-1))[0][0])
                    cam.set_rotation(rtvecs[c, k, :3])
                    cam.set_translation(rtvecs[c, k, 3:])
            bi = None
            comps = [[n for n in names if n not in uncovered]]
        else:
            bi = IB.initial_extrinsics(rtvecs, names)
            bi.apply(cg)
            comps = [c for c in bi.components if not (len(c) == 1 and c[0] in uncovered)]
    settings['n_board_frames'] = int(rtvecs.shape[1]) if rtvecs.ndim == 3 else 0
    # ---- BA spec
    if intrinsics == 'fixed':
        spec = ParamSpec(intrinsics=(), n_dist=0)
    elif hasattr(intrinsics, 'cameras'):
        spec = ParamSpec(intrinsics=None, priors=IntrinsicsPrior(intrinsics,
                                                                 sigma_logf=_sigma_opt(opts['prior_sigma_logf'], 0.05),
                                                                 sigma_k=_sigma_opt(opts['prior_sigma_k'], 0.1)))
    else:
        spec = ParamSpec(intrinsics=None)
    if static and opts['object_sigma'] is not None:
        fallbacks.append('object_sigma is not supported yet: object points held fixed')
    # ---- BA
    comp_groups, bridge_rep, ba_rep = [], None, None
    cgroup = None
    with timer('bundle_adjust'):
        usable = len(imgp) and imgp.shape[1] > 0
        if not usable:
            reasons_extra.append('no board view usable for bundle adjustment')
        elif len(comps) <= 1:
            ba_rep, st = _board_ba(cg, imgp, extra, spec, diag, opts, rng)
            settings.update(st)
            cgroup = cg
        else:
            # one BA per component (each its own gauge, board scale)
            comp_idx = [[names.index(n) for n in comp] for comp in comps]
            nfs = []
            for comp, idx in zip(comps, comp_idx):
                sub = cg.subset_cameras_names(comp)
                ip = imgp[idx]
                keep = np.isfinite(ip).all(-1).any(0)
                ex = dict(objp=extra['objp'][keep], ids=extra['ids'][keep],
                          rvecs=extra['rvecs'][idx][:, keep], tvecs=extra['tvecs'][idx][:, keep])
                r, st = _board_ba(sub, ip[:, keep], ex, spec, diag, opts, rng)
                nfs.append(st['noise_floor_px'])
                comp_groups.append(sub)
                for i, cam in zip(idx, sub.cameras):
                    cg.cameras[i].load_state_dict(cam.state_dict())
            settings['component_noise_floor_px'] = nfs
            if bridge_points is not None:
                bp = np.asarray(to_numpy(bridge_points), dtype='float64')
                if bp.shape[0] != C:
                    raise ValueError('bridge_points has {} cameras, expected {}'.format(bp.shape[0], C))
                nf0 = float(np.max(nfs))
                bridge_rep = _bridge_components(cg, comp_idx, bp, nf0, diag, opts)
                bridged = set(n for comp in comps[:1] for n in comp) | \
                    set(n for b in bridge_rep['components'] for n in b['cameras'])
                if not bridge_rep['unbridged']:
                    # joint extrinsics BA: boards (scale) + bridge tracks (link components)
                    bflat = _flat(bp)
                    ip2 = np.concatenate([imgp, bflat], axis=1)
                    nb = bflat.shape[1]
                    ex2 = dict(objp=np.concatenate([extra['objp'], np.zeros((nb, 3))]),
                               ids=np.concatenate([np.asarray(extra['ids'], float), np.full(nb, np.nan)]),
                               rvecs=np.concatenate([extra['rvecs'], np.full((C, nb, 3), np.nan)], axis=1),
                               tvecs=np.concatenate([extra['tvecs'], np.full((C, nb, 3), np.nan)], axis=1))
                    prune = dict(max_px=bridge_rep['tau_px'] * 2)
                    ba_rep, st = _board_ba(cg, ip2, ex2, spec, diag, opts, rng, only_extrinsics=True,
                                           prune=prune)
                    settings.update(st)
                    settings['bridge_prune_px'] = prune['max_px']
                    cgroup = cg
                    comps = [[n for n in names if n not in uncovered]]
                bridge_rep['bridged'] = sorted(bridged, key=names.index)
    # ---- static: the object's frame is the world
    if cgroup is not None and static:
        anchor = _static_anchor(cgroup, rows, board)
        settings['static_anchor'] = anchor
    # ---- report
    report = dict(components=comps, uncovered=uncovered, missing_intrinsics=missing)
    nf = settings.get('noise_floor_px', MIN_NOISE_PX)
    per_cam = {n: {} for n in names}
    worst_px = None
    if ba_rep is not None:
        for n in names:
            pc = ba_rep.per_camera.get(n, {})
            per_cam[n].update(n_obs=pc.get('n_obs', 0), median_px=pc.get('median_after'),
                              p90_px=pc.get('p90_after'))
        meds = [per_cam[n]['median_px'] for n in names if per_cam[n].get('median_px') is not None
                and per_cam[n].get('n_obs', 0) > 0 and n not in uncovered]
        worst_px = float(np.max(meds)) if meds else None
        report['ba'] = dict(error_px=ba_rep.error, error_before_px=ba_rep.error_before,
                            n_iter=ba_rep.n_iter, converged=ba_rep.converged, gauge=ba_rep.gauge,
                            frozen=ba_rep.frozen_cameras)
    shape_rel, n_lens_bad = None, 0
    if cgroup is not None:
        with timer('quality'):
            objp = np.asarray(board.get_object_points(), float).reshape(-1, 3)
            size = float(np.max(np.linalg.norm(objp - objp.mean(0), axis=1)) * 2)
            if not static:
                try:
                    bs = Q.board_shape_residual(cgroup, rows, board)
                    report['board_shape'] = bs
                    if bs.get('median') is not None and size > 0:
                        shape_rel = bs['median'] / size
                except Exception as exc:  # pragma: no cover
                    reasons_extra.append('board shape residual failed: {}'.format(exc))
            elif settings.get('static_anchor'):
                shape_rel = settings['static_anchor']['residual_median'] / size if size > 0 else None
            for n, cam in zip(names, cgroup.cameras):
                try:
                    lv = Q.lens_validity(cam, max_nonfinite_frac=0.05 if getattr(cam, 'fisheye', False) else 0.0)
                    per_cam[n]['lens_valid'] = bool(lv['valid'])
                    n_lens_bad += not lv['valid']
                except Exception:  # pragma: no cover
                    pass
            cgroup.metadata['units'] = getattr(board, 'units', None) or cgroup.metadata.get('units')
            if cgroup.metadata.get('units') is None:
                cgroup.metadata.pop('units', None)
            cgroup.metadata['scale_source'] = 'object' if point_object else \
                ('charuco' if hasattr(board, 'marker_length') else 'board')
    # bridged rig: judge the boards on the board observations alone (the joint BA's noise floor
    # is the people's), and the bridge on how many bridge observations the final rig explains
    bridge_frac = None
    if cgroup is not None and bridge_rep is not None and bridge_rep.get('components') and len(imgp):
        with timer('quality'):
            _, be, bi = _robust_points(cgroup, imgp, 0.02 * diag)
            bsup = bi.sum(0) >= 2
            nf = noise_floor(be[:, bsup], bi[:, bsup], diag=diag)
            settings['noise_floor_board_px'] = nf
            meds = []
            for c, n in enumerate(names):
                e = be[c][bi[c] & bsup]
                per_cam[n]['median_px'] = float(np.median(e)) if e.size else None
                per_cam[n]['board_n_obs'] = int(e.size)
                if e.size and n not in uncovered:
                    meds.append(float(np.median(e)))
            worst_px = float(np.max(meds)) if meds else None
            bt = bridge_rep['tau_px']
            _, _, qi = _robust_points(cgroup, bridge_points, bt)
            bfl = _flat(bridge_points)
            fin = np.isfinite(bfl).all(-1)
            multi = fin.sum(0) >= 2
            fr = []
            for comp in bridge_rep['components']:
                idx = [names.index(n) for n in comp['cameras']]
                m = fin[idx] & multi[None]
                f = float(qi[idx][m].mean()) if m.any() else 0.0
                comp['final_inlier_frac'] = f
                fr.append(f)
            bridge_frac = float(min(fr)) if fr else None
    report['per_camera'] = per_cam
    thr = float('{:.4g}'.format(max(1.0, 3 * nf)))
    settings['board_reproj_warn_px'] = thr
    # runaway cameras and one camera far worse than the others
    runaway, ratio, ratio_cam = [], None, None
    if cgroup is not None:
        from .. import scale as S
        cen = S.camera_centers(cgroup)
        okc = [i for i, n in enumerate(names) if n not in uncovered and n not in missing]
        if len(okc) >= 3:
            d = np.linalg.norm(cen[okc] - np.median(cen[okc], 0), axis=1)
            md = float(np.median(d))
            runaway = [names[i] for i, v in zip(okc, d) if md > 0 and v > opts['runaway_ratio'] * md]
        meds = [(per_cam[n]['median_px'], n) for n in names if per_cam[n].get('median_px') is not None
                and n not in uncovered]
        if len(meds) >= 3:
            mm = float(np.median([m for m, _ in meds]))
            if mm > 0:
                w, ratio_cam = max(meds)
                ratio = w / mm
    rules = [
        Rule('board_reproj_px', worst_px, thr, 2 * thr, 'worst per-camera median board reprojection'),
        Rule('board_reproj_ratio', ratio, opts['board_ratio'][0], opts['board_ratio'][1],
             'worst camera ({}) board error / the median camera\'s'.format(ratio_cam)),
        Rule('runaway_cameras', len(runaway), None, 0,
             'camera centres > {} x the median distance from the rig centre: {}'.format(
                 opts['runaway_ratio'], runaway) if runaway else ''),
        Rule('uncovered_cameras', len(uncovered), None, 0,
             'cameras without usable board views: {}'.format(uncovered) if uncovered else ''),
        Rule('missing_intrinsics', len(missing), None, 0,
             'no intrinsics for {}: give init= / intrinsics=, or annotate >= 6 non-coplanar points '
             'in a view'.format(missing) if missing else ''),
        Rule('disconnected_components', max(len(comps) - 1, 0), None, 0,
             '' if len(comps) <= 1 else
             'board graph components {}: bridge them with bridge_points= (2D tracks seen across '
             'components), a VGGT / keypoint init (calibrate_from_points(init=...)), or more board '
             'frames shared between components'.format(comps)),
        Rule('board_shape_rel', shape_rel, 0.005, 0.02, 'board shape residual / object size'),
        Rule('lens_invalid', n_lens_bad, 0, None, 'lens model not invertible over the image'),
    ]
    if bridge_frac is not None:
        rules.append(Rule('bridge_inlier_frac', bridge_frac, 0.5, 0.2,
                          'smallest share of a bridged component\'s bridge observations explained by the final '
                          'rig within {:.3g} px (a mirrored / wrong bridge explains few)'.format(bridge_rep['tau_px']),
                          higher_is_worse=False))
    report['runaway'] = runaway
    verdict, reasons, table = _rules(rules, opts)
    if cgroup is None and ba_rep is None and not comp_groups:
        verdict = 'fail'
    reasons += ['fail: ' + r for r in reasons_extra]
    if reasons_extra:
        verdict = 'fail'
    report['rules'] = table
    report['noise_floor_px'] = nf
    prov = provenance('calibrate_from_board',
                      inputs=dict(n_cameras=C, cam_names=names, board=repr(board),
                                  units=getattr(board, 'units', None), videos=videos is not None,
                                  n_detections=[len(r) for r in rows]),
                      fallbacks=fallbacks, timings=timer.times)
    res = BoardCalibration(report=report, verdict=verdict, reasons=reasons, settings=settings,
                           provenance=prov, cgroup=cgroup, detections=rows, lens_models=lens,
                           components=comps, component_groups=comp_groups, bridge=bridge_rep)
    if cgroup is not None:
        cgroup.metadata['calibration'] = dict(workflow='calibrate_from_board', verdict=verdict,
                                              noise_floor_px=float(nf))
    _save(res, out)
    return res


def _collapse_static(rows, board):
    """Static object: all frames of a camera are views of the same pose -> one row (median)."""
    out = []
    for rr in rows:
        if not rr:
            out.append([])
            continue
        F = np.stack([np.asarray(r['filled'], float).reshape(-1, 2) for r in rr])
        with np.errstate(all='ignore'):
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', RuntimeWarning)
                med = np.nanmedian(F, axis=0)
        good = np.isfinite(med).all(1)
        ids = np.nonzero(good)[0]
        out.append([{'framenum': (0, 0), 'corners': med[ids].reshape(-1, 1, 2), 'ids': ids.reshape(-1, 1),
                     'filled': board.fill_points(med[ids], ids)}])
    return out


def _save(res, out):
    if out is None:
        return
    if getattr(res, 'cgroup', None) is not None:
        save_calibration(res, out)
    else:
        out = str(out)
        stem = out[:-5] if out.endswith('.toml') else out
        res.to_json(stem + '.report.json')


# ================================================================== W2 calibrate_from_points

W2_DEFAULTS = dict(
    cam_names=None, sizes=None, fisheye=False,   # when `intrinsics` is K / dict (make_intrinsics_group)
    refine='robust',          # 'robust' (bundle_adjust_robust) | 'refine' (sfm.refine_group) | 'iter' | None
    coarse=(0.12, 0.06, 0.03),                    # first-round MSAC thresholds, fraction of the diagonal
    fine=((40, 20, 10, 5), (20, 10, 5)),          # later rounds (px at 0.5 px noise) x noise unit
    resect=True,
    reject='auto',            # structured frame / keypoint rejection in the fine rounds ('auto', noise-scaled) or None
    ba_kwargs=None, sfm_kwargs=None, vggt_kwargs=None, select_kwargs=None, people_kwargs=None,
    track_ids=None,           # C x F x P track ids of 5D points (init='people' / 'vggt' / a CameraGroup)
    intrinsics_free=('f', 'k1'),                  # refine_intrinsics: parameters freed
    intrinsics_sigma_logf=0.05, intrinsics_sigma_k=0.1,   # ... under priors around the given ones (or {name: sigma})
    heldout_every=5,          # refine_intrinsics='auto': every k-th window is held out
    intrinsics_gain=0.02,     # ... accepted if the held-out LOO median improves by this fraction
    intrinsics_share=None,    # refine_intrinsics: None | 'all' | groups of camera names/indices sharing f + k1
    min_inlier_frames=20,     # verdict: frames with inliers per camera
    runaway_ratio=5.0,        # verdict: camera centre farther than this x the median from the rig centre ...
    runaway_move=0.5,         # ... and moved more than this x the rig scale away from its init
    noise_unit='auto',        # 'auto' (capped noise floor / 0.5 px) or a float (1 = absolute pixel defaults)
    auto_skip_sfm=True,       # init='auto': skip SfM when the given extrinsics are complete and fit well
    init_check=True,          # verdict rule init_support (share of points the chosen init explains)
    quality=True, quality_groups=None,
    verdict=None,
)


@dataclass(repr=False)
class PointCalibration(WorkflowResult):
    """Result of :func:`calibrate_from_points`: ``cgroup`` (None if no init succeeded),
    ``init_report`` (:class:`aniposelib.init.RecipeReport`: candidates, per-generator init
    reports, selection, refinement, scale, quality), ``selected_init`` (key of the chosen
    candidate), ``intrinsics`` (what the evidence gate decided), ``scale`` (:class:`ScaleEstimate`
    or None), ``quality`` (:class:`aniposelib.quality.CalibrationReport`)."""
    cgroup: Any = None
    init_report: Any = None
    selected_init: Any = None
    intrinsics: Dict[str, Any] = field(default_factory=dict)
    scale: Any = None
    quality: Any = None

    _skip = ('cgroup', 'quality')

    def __iter__(self):
        yield self.cgroup
        yield self

    def rig_at(self, t):
        """The calibration at time ``t`` (s): a copy of ``cgroup`` whose moved cameras get the pose
        of their station at ``t`` (``stations=``; ``cgroup.stations`` holds the intervals and
        ``cgroup.metadata['station_poses']`` the per-station poses). Without stations: a copy."""
        return rig_at(self.cgroup, t)

    def to_dict(self):
        d = super().to_dict()
        if self.quality is not None and hasattr(self.quality, 'to_dict'):
            q = self.quality.to_dict()
            d['quality'] = {k: q[k] for k in ('per_camera', 'overall') if k in q}
        return d


def rig_at(cgroup, t):
    """A copy of ``cgroup`` with every moved camera at its station's pose at time ``t`` (from
    ``cgroup.stations`` and ``cgroup.metadata['station_poses']``, as written by
    :func:`calibrate_from_points` with ``stations=``). A time in a 'skip' interval keeps the
    camera's first pose."""
    out = cgroup.copy()
    st = getattr(cgroup, 'stations', None)
    poses = cgroup.metadata.get('station_poses') or {}
    if st is None or not poses:
        return out
    for cam in out.cameras:
        n = cam.get_name()
        sn = str(st.lookup_name(n, [float(t)])[0])
        p = poses.get(sn)
        if p is not None and p.get('camera', n) == n:
            cam.set_rotation(np.asarray(p['rvec'], float))
            cam.set_translation(np.asarray(p['tvec'], float))
    return out


def _dense_from_obs(obs):
    C, N = int(obs.n_cams), int(obs.n_points)
    p = np.full((C, N, 2), np.nan)
    p[np.asarray(obs.cam_idx), np.asarray(obs.point_idx)] = np.asarray(to_numpy(obs.uv), float)
    groups = {}
    g = getattr(obs, 'group', None)
    if isinstance(g, dict):
        for k, v in g.items():
            v = np.asarray(v)
            if v.shape[0] == len(obs.point_idx):
                per = np.full(N, -1, dtype=v.dtype if v.dtype.kind in 'iu' else float)
                per[np.asarray(obs.point_idx)] = v
                groups[k] = per
    t = getattr(obs, 'time', None)
    if t is not None:
        per = np.full(N, np.nan)
        per[np.asarray(obs.point_idx)] = np.asarray(to_numpy(t), float)
        groups['time'] = per
    return p, groups


def _parse_groups(groups, pshape):
    if groups is None:
        return {}
    if not isinstance(groups, dict):
        groups = dict(frame=groups)
    out = {}
    for k, v in groups.items():
        if k not in ('frame', 'window', 'keypoint', 'time'):
            raise ValueError("groups keys must be 'frame', 'window', 'keypoint' or 'time', got {!r}".format(k))
        if v is None:
            continue
        v = np.asarray(v)
        if v.shape[:1] != pshape[:1]:
            raise ValueError('groups[{!r}] has shape {}, expected ({},) or {}'.format(k, v.shape, pshape[0], pshape))
        out[k] = v
    return out


def _candidates(base, p, init, frames, gr, opts, rng, rpt, names, dev=None):
    """Candidate inits (the 1.0 recipe of ``init.calibrate_from_points``)."""
    from ..cameras import CameraGroup
    from ..init import _has_extrinsics, from_extrinsics, sfm
    if isinstance(init, str) and init == 'auto':
        specs = (['reference'] if _has_extrinsics(base) else []) + ['sfm'] + (['vggt'] if frames is not None else [])
    elif isinstance(init, (list, tuple)):
        specs = list(init)
    else:
        specs = [init]
    cands, priors = {}, None
    frame_id, window_id = gr.get('frame'), gr.get('window')
    for spec in sorted(specs, key=lambda s: 0 if (isinstance(s, str) and s == 'vggt') else 1):
        try:
            if isinstance(spec, CameraGroup):
                cands['given_{}'.format(len(cands))] = spec
            elif spec == 'reference':
                if not _has_extrinsics(base):
                    raise ValueError("init='reference' needs intrinsics= with extrinsics")
                cands['reference'] = base
            elif spec == 'vggt':
                if frames is None:
                    raise ValueError("init='vggt' needs frames=")
                from ..init import vggt as V
                pred = V.predict(frames, names=names, **(opts['vggt_kwargs'] or {}))
                agg = V.aggregate(pred)
                cands['vggt'] = V.to_cameragroup(agg, template=base)
                for t in range(pred.extrinsics.shape[0]):
                    cands['vggt_t{}'.format(t)] = from_extrinsics(names, pred.extrinsics[t], template=base)
                priors = V.pairwise_rotations(pred)
                rpt.init_reports['vggt'] = dict(stats=agg.stats)
            elif spec == 'sfm':
                kw = dict(frame_id=frame_id, window_id=window_id, rotation_priors=priors, rng=rng, device=dev)
                kw.update(opts['sfm_kwargs'] or {})
                cg, ir = sfm.from_correspondences(base, p, **kw)
                cands['sfm'] = cg
                rpt.init_reports['sfm'] = ir
            elif spec == 'people':
                from ..init import people
                cg, ir = people.from_people(base, p, **(opts['people_kwargs'] or {}))
                cands['people'] = cg
                rpt.init_reports['people'] = ir
            else:
                raise ValueError('unknown init {!r}'.format(spec))
        except Exception as exc:
            rpt.errors[str(spec) if isinstance(spec, str) else 'given'] = '{}: {}'.format(type(exc).__name__, exc)
    return cands


def _robust_ba(cg, pc, gr, diag, opts, rng):
    """Noise-normalised :func:`bundle.bundle_adjust_robust`: a coarse round with thresholds
    relative to the diagonal, then the fine rounds scaled by the noise unit measured after it."""
    from .. import bundle as B
    kw = dict(opts['ba_kwargs'] or {})
    groups = {k: v for k, v in gr.items() if k in ('frame', 'keypoint')}
    if pc.ndim == 4:
        groups = {k: (np.broadcast_to(v[:, None], pc.shape[1:3]) if (k == 'frame' and v.ndim == 1) else v)
                  for k, v in groups.items()}
    coarse = tuple(float(f * diag) for f in opts['coarse'])
    def call(**base):
        base.update(kw)
        return B.bundle_adjust_robust(cg, pc, **base)
    r1 = call(groups=groups or None, schedule=(coarse,), final=(), resect=opts['resect'], reject=None,
              rng=rng, only_extrinsics=True, revert_if_worse=True, report_tau=coarse[-1],
              resect_kw=dict(tau_tri=coarse[-1] / 2, tau_pnp=coarse[-1]))
    e = np.asarray(r1.obs_errors, float)
    nf = noise_floor(e, np.nan_to_num(e, nan=np.inf) < coarse[-1], diag=diag)
    u = _noise_unit(nf, opts['noise_unit'])
    fine = _fine_schedule(opts['fine'], u, coarse[-1])
    rej = B.RejectSchedule()
    rej = None if opts['reject'] is None else B.RejectSchedule(frame=tuple(u * t for t in rej.frame), keypoint=tuple(u * t for t in rej.keypoint),
                           obs=tuple(u * t for t in rej.obs), min_frame_pts=rej.min_frame_pts,
                           min_kpt_obs=rej.min_kpt_obs, protect_kpts=rej.protect_kpts, max_kpt_frac=rej.max_kpt_frac)
    rkw = dict(tau_tri=5.0 * u, tau_pnp=8.0 * u)
    r2 = call(groups=groups or None, schedule=fine, final=('extrinsics',), resect=opts['resect'],
              reject=rej, rng=rng, only_extrinsics=True, revert_if_worse=True, report_tau=fine[-1][-1],
              resect_kw=rkw)
    e = np.asarray(r2.obs_errors, float)
    fit = ~np.asarray(r2.excluded.get('fit', np.zeros(e.shape, bool)), bool)
    nf2 = noise_floor(e, fit & (np.nan_to_num(e, nan=np.inf) < fine[-1][-1]), diag=diag)
    settings = dict(coarse_px=list(coarse), noise_floor_coarse_px=nf, noise_unit=u, fine_px=[list(f) for f in fine],
                    reject_px=None if rej is None else rej.to_dict(), resect_px=rkw, report_tau_px=fine[-1][-1], noise_floor_px=nf2)
    return r2, dict(method='bundle_adjust_robust', before_px=r1.error_before, after_px=r2.error,
                    kept=not (r1.reverted and r2.reverted), result=r2), settings


def _fine_schedule(fine, u, cap):
    """The fine MSAC rounds in px: ``t x u``, dropping thresholds above ``cap`` (the last coarse
    threshold already covered them) so each round keeps its ratios; a round entirely above the cap
    keeps one threshold, at the cap. (1.1.0: ``min(t x u, cap)`` flattened every round to the cap
    when ``u`` was large.)"""
    out = []
    for rnd in fine:
        ts = [float(t * u) for t in rnd]
        keep = [t for t in ts if t <= cap]
        out.append(tuple(keep) if keep else (float(min(min(ts), cap)),))
    return tuple(out)


def _heldout_mask(pc, gr, every):
    """Point mask [N] (or [N,J] -> per frame) of held-out windows: every ``every``-th block."""
    N = pc.shape[1]
    w = gr.get('window', gr.get('frame'))
    if w is None:
        w = np.arange(N)
    w = np.asarray(w)
    if w.ndim > 1:
        w = w[:, 0]
    uw, inv = np.unique(w, return_inverse=True)
    nb = max(5 * every, 1)
    block = np.floor(inv * nb / max(len(uw), 1)).astype(int)
    return (block % every) == (2 % every)


def _loo_stats(cg, p, tau_tri, agree_px):
    """Held-out LOO summary: ``(median of per-camera medians, mean agree@agree_px, mean coverage)``
    (inf / 0 / 0 when nothing could be measured)."""
    from .. import quality as Q
    r = Q.loo_reprojection(cg, p, tau_tri=tau_tri, min_others=min(3, max(len(cg.cameras) - 1, 1)),
                           thresholds=(agree_px,))
    m = r.medians()
    key = [k for k in next(iter(r.per_camera.values())) if k.startswith('agree@')][0]
    ag = [v[key] for v in r.per_camera.values() if v[key] is not None]
    cv = [v['coverage'] for v in r.per_camera.values() if v['coverage'] is not None]
    return (float(np.nanmedian(m)) if np.isfinite(m).any() else np.inf,
            float(np.mean(ag)) if ag else 0.0, float(np.mean(cv)) if cv else 0.0)


def _loo_median(cg, p, tau_tri):
    return _loo_stats(cg, p, tau_tri, 8.0 * tau_tri / 5.0)[0]


def _lens_ok_on(cam, uv, max_roundtrip_px=0.5, min_slope=0.05, n_max=4000):
    """Is the lens model invertible where the camera has data? Its observed pixels ``uv [M,2]``
    are undistorted and re-distorted (99th percentile round trip below ``max_roundtrip_px``) and
    the radial map must be monotone up to their largest radius."""
    from .. import geometry as G
    from ..quality import _radial_slope
    uv = np.asarray(uv, float).reshape(-1, 2)
    uv = uv[np.isfinite(uv).all(1)]
    if len(uv) == 0:
        return True
    if len(uv) > n_max:
        uv = uv[np.random.default_rng(0).choice(len(uv), n_max, replace=False)]
    P = G.pack([cam])
    t = torch.as_tensor(uv[None], dtype=P.dtype, device=P.device)
    with torch.no_grad():
        xy = G._undistort_flat(P, t)
        back = G.distort(P, xy)
    xy, back = to_numpy(xy)[0], to_numpy(back)[0]
    err = np.linalg.norm(back - uv, axis=-1)
    if not np.isfinite(err).mean() >= 0.99:
        return False
    if np.nanpercentile(err, 99) > max_roundtrip_px:
        return False
    rmax = float(np.nanmax(np.hypot(xy[:, 0], xy[:, 1])))
    slope = _radial_slope(P, rmax) if rmax > 0 else 1.0
    return bool(np.isfinite(slope) and slope > min_slope)


def _invalid_lenses(cg, points=None):
    """Names of the cameras whose lens model is not invertible over the image -- or, with
    ``points [C,...,2]``, over the region where the camera has observations (a fisheye model is
    often not invertible in image corners nothing is seen in, even for a correct lens)."""
    from .. import quality as Q
    bad = []
    flat = None if points is None else _flat(points)
    for c, cam in enumerate(cg.cameras):
        try:
            if flat is not None:
                ok = _lens_ok_on(cam, flat[c])
            else:
                ok = Q.lens_validity(cam, max_nonfinite_frac=0.05 if getattr(cam, 'fisheye', False) else 0.0)['valid']
            if not ok:
                bad.append(cam.get_name())
        except Exception:  # pragma: no cover
            pass
    return bad


def _share_spec(share, names):
    """``options['intrinsics_share']`` -> ``ParamSpec(share=)`` groups over the present cameras."""
    if share is None:
        return None
    if isinstance(share, str):
        if share != 'all':
            raise ValueError("options['intrinsics_share'] must be None, 'all' or a list of camera groups")
        return [list(names)] if len(names) > 1 else None
    out = []
    for g in share:
        g = [names[c] if isinstance(c, (int, np.integer)) and not isinstance(c, bool) else str(c) for c in g]
        g = [n for n in g if n in names]
        if len(g) > 1:
            out.append(g)
    return out or None


def _gate_intrinsics(cg, pc, gr, mode, nf, opts, rng):
    """Evidence-gated focal + k1 refinement. Returns ``(cgroup, info)``.

    A refined lens that is not invertible where the camera has observations (round trip of its
    observed pixels, monotone radial map) is put back to the given one (that camera only, then an
    extrinsics re-fit). ``'auto'`` keeps the refinement
    only if, on held-out windows, the LOO median improves by ``intrinsics_gain`` and neither the
    LOO agreement nor the LOO coverage gets worse. Never raises: a failed fit keeps the given
    intrinsics (``decision='kept'``, the error in ``reason``)."""
    try:
        return _gate_intrinsics_(cg, pc, gr, mode, nf, opts, rng)
    except Exception as exc:
        return cg, dict(decision='kept', reason='intrinsics refinement failed ({}: {}); kept the given '
                        'intrinsics'.format(type(exc).__name__, exc), free=list(opts['intrinsics_free']))


def _gate_intrinsics_(cg, pc, gr, mode, nf, opts, rng):
    from .. import bundle as B
    from ..params import IntrinsicsPrior, ParamSpec
    u = _noise_unit(nf, opts['noise_unit'])
    tau = 10.0 * u
    C = len(cg.cameras)
    names = cg.get_names()
    share = _share_spec(opts['intrinsics_share'], names)
    spec = ParamSpec(intrinsics=tuple(opts['intrinsics_free']), share=share,
                     priors=IntrinsicsPrior(cg.copy(), sigma_logf=_sigma_opt(opts['intrinsics_sigma_logf'], 0.05),
                                            sigma_k=_sigma_opt(opts['intrinsics_sigma_k'], 0.1)))
    f_scale = 3 * max(nf, MIN_NOISE_PX)
    min_views = min(3, C)

    def fit(group, p):
        before = group.copy()
        _, _, inl = _robust_points(group, p, tau, min_inliers=2)
        flat = _flat(p)
        keep = inl & (inl.sum(0) >= min_views)[None]
        q = np.where(keep[..., None], flat, np.nan)
        B.bundle_adjust_lm(group, q, spec=spec, loss='cauchy', f_scale=f_scale, n_iter=30, rng=rng)
        bad = _invalid_lenses(group, p)
        if bad:      # revert those lenses, re-settle the extrinsics
            for cam, cb in zip(group.cameras, before.cameras):
                if cam.get_name() in bad:
                    _set_intrinsics(cam, to_numpy(cb.get_camera_matrix()), to_numpy(cb.get_distortions()))
            B.bundle_adjust_lm(group, q, only_extrinsics=True, loss='cauchy', f_scale=f_scale, n_iter=20, rng=rng)
        return group, bad
    base_info = dict(free=list(opts['intrinsics_free']), share=share)
    if mode is True:
        _, bad = fit(cg, pc)
        return cg, dict(base_info, decision='refined', reason='refine_intrinsics=True', reverted_invalid=bad)
    if C < 3:
        return cg, dict(base_info, decision='kept', reason='the held-out LOO check needs >= 3 cameras')
    ho = _heldout_mask(pc, gr, opts['heldout_every'])
    train = pc.copy()
    train[:, ho] = np.nan
    test = pc.copy()
    test[:, ~ho] = np.nan
    if not ho.any() or ho.all():
        return cg, dict(base_info, decision='kept', reason='not enough windows for a held-out check')
    base = cg.copy()
    B.bundle_adjust_lm(base, _flat(train), only_extrinsics=True, loss='cauchy', f_scale=f_scale, n_iter=20, rng=rng)
    trial, bad = fit(base.copy(), train)
    agree_px = 8.0 * u
    e0, a0, c0 = _loo_stats(base, test, 5.0 * u, agree_px)
    e1, a1, c1 = _loo_stats(trial, test, 5.0 * u, agree_px)
    gain = (e0 - e1) / e0 if np.isfinite(e0) and e0 > 0 else 0.0
    info = dict(base_info, heldout_loo_fixed_px=e0, heldout_loo_refined_px=e1, gain=gain,
                heldout_agree_fixed=a0, heldout_agree_refined=a1, heldout_coverage_fixed=c0,
                heldout_coverage_refined=c1, agree_px=agree_px, min_gain=opts['intrinsics_gain'],
                n_heldout_points=int(ho.sum()), invalid_in_trial=bad)
    if bad and len(bad) == C:
        info.update(decision='kept', reason='every refined lens was invalid')
        return cg, info
    why = []
    if not (np.isfinite(e1) and gain > opts['intrinsics_gain']):
        why.append('LOO median gain {:.1%} <= {:.1%}'.format(gain, opts['intrinsics_gain']))
    if a1 < a0 - 0.002:
        why.append('LOO agree@{:.3g}px fell {:.3f} -> {:.3f}'.format(agree_px, a0, a1))
    if c1 < c0 - 0.005:
        why.append('LOO coverage fell {:.3f} -> {:.3f}'.format(c0, c1))
    if not why:
        _, bad = fit(cg, pc)
        info.update(decision='refined', reason='held-out LOO improved by {:.1%} (agree {:.3f} -> {:.3f})'.format(
            gain, a0, a1), reverted_invalid=bad)
    else:
        info.update(decision='kept', reason='held-out check failed: ' + '; '.join(why))
    return cg, info


def _scale_spec(scale, p):
    """W2 ``scale=`` -> keyword arguments of :func:`estimate_scale`."""
    from ..cameras import CameraGroup
    if isinstance(scale, CameraGroup):
        return dict(reference=scale)
    if not isinstance(scale, dict):
        raise ValueError('unsupported scale spec: {!r}'.format(type(scale)))
    s = dict(scale)
    if 'rows' in s:                                 # 1.0 recipe: dict(board=, rows=)
        s['board_points'] = s.pop('rows')
    if 'pairs' in s and 'lengths' not in s or (isinstance(s.get('lengths'), (list, tuple, dict)) and 'pairs' in s):
        s = dict(lengths=dict(points=s.pop('p2ds', p), pairs=s.pop('pairs'), ref=s.pop('lengths')), **s)
    if 'reference' in s and 'align' in s and s['align'] is False:
        s['align'] = None
    if s.get('align') is True:
        s['align'] = 'auto'
    keys = {'board', 'board_points', 'lengths', 'reference', 'align', 'object', 'object_points', 'options'}
    extra = set(s) - keys
    if extra:
        raise ValueError('unknown scale keys {}'.format(sorted(extra)))
    return s


@thread_guard
def calibrate_from_points(points, intrinsics, init='auto', frames=None, groups=None,
                          refine_intrinsics='auto', scale=None, stations=None, options=None, rng=0,
                          device='auto', out=None):
    """W2: calibrate extrinsics from 2D correspondences with known (or nominal) intrinsics.

    Parameters
    ----------
    points : ``C x N x 2`` | ``C x N x J x 2`` pixels (NaN missing; same index = same 3D point),
        an :class:`aniposelib.bundle.Observations`, or ``C x F x P x J x 2`` unassociated people
        with ``init='people'``, ``init='vggt'`` or a ``CameraGroup`` init.
    intrinsics : ``CameraGroup`` (lab calibration, VGGT K, nominal; its extrinsics are the
        ``'reference'`` candidate), a 3x3 / ``[C,3,3]`` K, ``{name: dict(matrix=, dist=)}``, or
        None (nominal; needs ``options={'cam_names':, 'sizes':}``).
    init : ``'auto'`` (reference if ``intrinsics`` has extrinsics, SfM, VGGT only when ``frames``
        is given), ``'reference'``, ``'sfm'``, ``'vggt'``, ``'people'``, a ``CameraGroup`` or a
        list of those. Candidates are ranked by ``init.select`` on the data.
    frames : images for VGGT (one synced set per entry).
    groups : ``dict(frame=, window=, keypoint=, time=)`` ids per point (or an array = frame
        ids): SfM frames/windows, structured rejection, held-out windows, station times.
    refine_intrinsics : ``'auto'`` (focal + k1 under priors, kept only if the held-out LOO
        improves by ``intrinsics_gain``), True (always) or False.
    scale : None, a float, a reference ``CameraGroup``, or keyword arguments of
        :func:`estimate_scale` (``dict(board=, board_points=)``, ``dict(lengths=dict(points=,
        pairs=, ref=))``, ``dict(reference=, align=)``; the 1.0 recipe's ``dict(board=, rows=)`` /
        ``dict(pairs=, lengths=)`` forms are accepted).
    stations : :class:`aniposelib.stations.Stations` of moved cameras (needs
        ``groups['time']``): a final extrinsics LM with one pose per station.
    device : ``'auto'`` (cuda when available) for the heavy steps; the result is on the CPU.

    Returns :class:`PointCalibration` (``cgroup, res = calibrate_from_points(...)``).

    Verdict rules (``nf`` = noise floor of the final fit; LOO over the observations the fit
    kept, so a camera's masked decoy frames do not count twice):

    - ``loo_worst_px``: worst per-camera LOO median; warn above 3 nf, fail above 10 nf;
    - ``unplaced_cameras``: cameras the init could not place (fail);
    - ``runaway_cameras``: centres farther than ``runaway_ratio`` x the median distance from the
      rig centre that also moved more than ``runaway_move`` x the rig scale from their init (fail);
    - ``loo_unsupported``: cameras none of whose held-out points the others triangulate (warn);
    - ``init_support``: share of the multi-view observations the chosen init explains at the last
      coarse threshold; warn below 0.3, fail below 0.05 (a wrong VGGT / reference init);
    - ``two_view_only`` / ``two_view_px`` (fewer than 3 cameras placed): no LOO (warn); worst
      per-camera 2-view reprojection median, warn above 0.5 %, fail above 2 % of the diagonal;
    - ``min_inlier_frames``: fewest frames with inlier observations of a camera; warn below
      ``min_inlier_frames``, fail below a quarter of it;
    - ``min_inlier_frac``: smallest fraction of a camera's observations kept by the fit; warn
      below 0.5, fail below 0.1;
    - ``scale_rejected``: a requested scale was refused (fail).

    Options (``W2_DEFAULTS``; besides the verdict overrides): ``noise_unit`` ('auto' or a float: 1 =
    the backend's absolute pixel defaults); ``intrinsics_share`` (None, 'all' or groups of camera
    names / indices sharing the lens when intrinsics are refined); ``intrinsics_sigma_logf`` /
    ``intrinsics_sigma_k`` (float or ``{camera: sigma, 'default': sigma}``); ``auto_skip_sfm``
    (``init='auto'``: no SfM candidate when the given extrinsics are complete and explain >= 70 % of
    the multi-view observations, >= 50 % per camera); ``track_ids`` (``C x F x P`` ids of 5D points);
    ``reject`` ('auto' or None: structured frame / keypoint rejection); ``runaway_move``;
    ``init_check``. 5D points with ``init='vggt'`` (and ``frames=``) or a ``CameraGroup`` init are
    associated on that rig with :func:`~aniposelib.workflows.associate_views` first. With
    ``stations=``, ``result.cgroup.stations`` and ``metadata['station_poses']`` hold the per-station
    poses; ``result.rig_at(t)`` is the calibration at time ``t``. The refined-lens gate reverts
    lenses that are not invertible over the image. Never raises on bad data (``'fail'`` + reason).
    """
    from ..init import RecipeReport, make_intrinsics_group, select, _has_extrinsics
    from ..init import _refine as _legacy_refine
    from .. import scale as S
    timer = Timer()
    opts = merge_options(W2_DEFAULTS, options)
    _check_noise_unit(opts['noise_unit'])
    rpt = RecipeReport()
    fallbacks, settings = [], {}
    dev = _device(device)
    settings['device'] = dev
    # ---- inputs
    if hasattr(points, 'cam_idx') and hasattr(points, 'uv'):
        p, g0 = _dense_from_obs(points)
        groups = dict(g0, **(groups or {})) if isinstance(groups, dict) or groups is None else groups
    else:
        p = np.asarray(to_numpy(points), dtype='float64')
    if p.ndim not in (3, 4, 5) or p.shape[-1] != 2:
        raise ValueError('points must be C x N x 2 or C x N x J x 2, got {}'.format(p.shape))
    from ..cameras import CameraGroup
    rig_init = isinstance(init, CameraGroup) or (isinstance(init, str) and init == 'vggt')
    if p.ndim == 5 and init != 'people' and not rig_init:
        raise ValueError("C x F x P x J x 2 points need init='people', init='vggt' (with frames=) or a "
                         "CameraGroup init")
    base = make_intrinsics_group(opts['cam_names'], opts['sizes'], intrinsics, fisheye=opts['fisheye'])
    _check_cameras(base, 'intrinsics')
    names = base.get_names()
    if p.shape[0] != len(names):
        raise ValueError('points have {} cameras, intrinsics {}'.format(p.shape[0], len(names)))
    gr = _parse_groups(groups, p.shape[1:-1] if p.ndim < 5 else p.shape[1:2])
    diag = _diag(base, p) or 1000.0
    settings['image_diag_px'] = diag
    tau_coarse = float(opts['coarse'][-1] * diag)
    res = PointCalibration(init_report=rpt)
    shape_in = list(p.shape)
    track_ids = opts['track_ids']
    pk = dict(opts['people_kwargs'] or {})
    if track_ids is None:
        track_ids = pk.get('track_ids')
    elif init == 'people':
        pk.setdefault('track_ids', track_ids)
    opts['people_kwargs'] = pk

    def finish_fail(reason, cands=(), extra=None):
        res.verdict = 'fail'
        res.reasons = ['fail: ' + reason]
        res.report = dict(per_camera={n: dict(placed=False) for n in names}, unplaced=list(names), runaway=[],
                          rules=[], noise_floor_px=None, init=None, candidates=list(cands),
                          errors=dict(rpt.errors), overall=None)
        res.report.update(extra or {})
        res.settings = settings
        res.provenance = provenance('calibrate_from_points', inputs=dict(n_cameras=len(names), shape=shape_in),
                                    fallbacks=fallbacks + ['init {} failed: {}'.format(k, v)
                                                           for k, v in rpt.errors.items()],
                                    timings=timer.times)
        _save(res, out)
        return res
    # ---- candidates + selection
    init_spec = init
    ref_support = None
    if isinstance(init, str) and init == 'auto' and opts['auto_skip_sfm'] and p.ndim < 5 and _has_extrinsics(base):
        with timer('init_check'):
            ref_support = _init_support(_on_device(base, dev), p, tau_coarse)
        settings['auto_reference_support'] = ref_support
        if ref_support['min_camera'] is not None and ref_support['min_camera'] >= 0.5 \
                and ref_support['overall'] >= 0.7:
            init_spec = ['reference'] + (['vggt'] if frames is not None else [])
            fallbacks.append('init auto: SfM skipped, the given extrinsics explain {:.0%} of the multi-view '
                             'observations (worst camera {:.0%})'.format(ref_support['overall'],
                                                                         ref_support['min_camera']))
    with timer('init'):
        cands = _candidates(base, p, init_spec, frames, gr, opts, rng, rpt, names, dev=dev)
        if p.ndim == 5 and 'people' in cands:
            from ..init import people
            akw = dict(pk.get('association_kwargs') or {})
            akw.setdefault('track_ids', track_ids)
            assoc, _ = people.associate_people(base, p, **akw)
            C_, F_, G_, J_, _ = assoc.shape
            p = assoc.reshape(C_, F_ * G_, J_, 2)
            gr = dict(frame=np.arange(F_ * G_), window=np.repeat(np.arange(F_), G_))
        elif p.ndim == 5 and cands:
            # unassociated detections + a rig (VGGT / given): associate_views, then as for 4D input
            from .reconstruction import associate_views
            key0 = 'vggt' if 'vggt' in cands else next(iter(cands))
            try:
                av = associate_views(cands[key0], [p[c] for c in range(p.shape[0])],
                                     track_ids=None if track_ids is None else
                                     [np.asarray(track_ids)[c] for c in range(p.shape[0])])
                ap = np.asarray(av.points, float)
            except Exception as exc:
                rpt.errors['associate'] = '{}: {}'.format(type(exc).__name__, exc)
                ap = np.zeros(p.shape[:2] + (0,) + p.shape[3:])
            rpt.init_reports['associate'] = dict(rig=key0, n_identities=int(ap.shape[2]))
            if ap.shape[2] == 0:
                return finish_fail('the {} rig associated no identity across cameras: that rig is probably '
                                   'wrong{}'.format(key0, _vggt_spread_note(rpt)), cands)
            C_, F_, G_, J_, _ = ap.shape
            p = ap.reshape(C_, F_ * G_, J_, 2)
            gr = dict(frame=np.arange(F_ * G_), window=np.repeat(np.arange(F_), G_))
    rpt.candidates = list(cands)
    if not cands:
        return finish_fail('no initialisation succeeded: {}'.format(rpt.errors), cands)
    complete = {k: v for k, v in cands.items() if set(v.get_names()) == set(names)}
    pool = complete if complete else cands
    with timer('select'):
        if len(pool) == 1:
            key = next(iter(pool))
        else:
            common = [n for n in names if all(n in v.get_names() for v in pool.values())]
            idx = [names.index(n) for n in common]
            skw = dict(rng=rng, min_inliers=min(3, len(common)))
            skw.update(opts['select_kwargs'] or {})
            _, rpt.selection = select(pool, p[idx], cam_names=common, **skw)
            key = rpt.selection.best
    rpt.init = key
    chosen = pool[key].copy()
    present = [n for n in names if n in chosen.get_names()]
    if present != chosen.get_names():
        chosen = chosen.subset_cameras_names(present)
    unplaced = [n for n in names if n not in present]
    pc = p[[names.index(n) for n in present]]
    init_centers = S.camera_centers(chosen)
    cg = chosen.to(dev) if dev != 'cpu' else chosen
    init_support = None
    if opts['init_check']:
        with timer('init_check'):
            init_support = ref_support if (key == 'reference' and ref_support is not None and not unplaced) \
                else _init_support(cg, pc, tau_coarse)
    # ---- refinement
    ba = None
    try:
        with timer('bundle_adjust'):
            method = opts['refine']
            if method in ('robust', 'auto'):
                ba, rpt.refine, st = _robust_ba(cg, pc, gr, diag, opts, rng)
                settings.update(st)
            elif method:
                cg, rpt.refine = _legacy_refine(cg, pc, gr.get('frame'), method, rng, opts['ba_kwargs'])
            else:
                rpt.refine = dict(method=None)
    except Exception as exc:
        rpt.errors['refine'] = '{}: {}'.format(type(exc).__name__, exc)
        return finish_fail('refinement of the {} init failed: {}: {}'.format(key, type(exc).__name__, exc),
                           cands, dict(init=key))
    # verdict noise floor: the points under the refined rig at a loose threshold (as W4), not the
    # errors at the last schedule threshold (that one is ~threshold / 3 by construction)
    # (the fit's own floor still sets the intrinsics gate and the station fit, as before)
    with timer('noise'):
        nf, _ = _points_noise(cg, pc, diag)
    nf_fit = float(settings.get('noise_floor_px', nf))
    settings['noise_floor_fit_px'] = nf_fit
    settings['noise_floor_px'] = nf
    u = _noise_unit(nf, opts['noise_unit'])
    settings['noise_unit_final'] = u
    # ---- intrinsics (evidence gated)
    with timer('intrinsics'):
        if refine_intrinsics in (False, None):
            res.intrinsics = dict(decision='kept', reason='refine_intrinsics=False')
        elif refine_intrinsics in ('auto', True):
            cg, res.intrinsics = _gate_intrinsics(cg, pc, gr, refine_intrinsics, nf_fit, opts, rng)
        else:
            raise ValueError("refine_intrinsics must be 'auto', True or False")
    # ---- stations
    if stations is not None:
        with timer('stations'):
            res.report['stations'] = _stations_ba(cg, pc, gr, stations, nf_fit, rng, opts['noise_unit'])
    if dev != 'cpu':
        cg = cg.to('cpu')
    # ---- scale
    scale_rejected = False
    if scale is not None:
        with timer('scale'):
            if isinstance(scale, (int, float)) and not isinstance(scale, bool):
                S.rescale(cg, float(scale), source='manual')
                rpt.scale = dict(s=float(scale), source='manual')
            else:
                se = estimate_scale(cg, **_scale_spec(scale, pc))
                rpt.scale = res.scale = se
                if se.accepted and se.cgroup is not None:
                    cg = se.cgroup
                else:
                    scale_rejected = True
    # ---- quality
    Cp = len(present)
    fit_mask = None
    if ba is not None and 'fit' in ba.excluded:
        fit_mask = ~np.asarray(ba.excluded['fit'], bool).reshape(Cp, -1)
    with timer('quality'):
        if opts['quality']:
            from .. import quality as Q
            try:
                rpt.quality = res.quality = Q.report(
                    _on_device(cg, dev), pc, groups=opts['quality_groups'], tau=10.0 * u, tau_tri=5.0 * u,
                    max_loo_px=60.0 * u, thresholds=tuple(t * u for t in (5, 8, 10)), fit_mask=fit_mask,
                    min_views=min(3, Cp), loo=Cp >= 3, min_others=min(3, max(Cp - 1, 1)))
            except Exception as exc:  # pragma: no cover
                rpt.errors['quality'] = '{}: {}'.format(type(exc).__name__, exc)
    # ---- verdict
    flat = _flat(pc)
    valid = np.isfinite(flat).all(-1)
    per_cam = {}
    loo_worst, unsupported = None, []
    if res.quality is not None and res.quality.loo is not None:
        le = np.asarray(res.quality.loo.errors, float)
        use = res.quality.loo.eligible & (fit_mask if fit_mask is not None else True)
        meds = []
        for c, n in enumerate(present):
            ne = int(use[c].sum())
            e = le[c][use[c] & np.isfinite(le[c])]
            m = float(np.median(e)) if e.size else None
            per_cam[n] = dict(loo_fit_median_px=m, loo_fit_supported_frac=(e.size / ne) if ne else None)
            if m is not None:
                meds.append(m)
            elif ne:
                unsupported.append(n)
        loo_worst = float(np.max(meds)) if meds else None
    two_view = None
    if Cp < 3 and res.quality is not None:
        mv = [res.quality.per_camera[n].get('median_px') for n in present]
        mv = [v for v in mv if v is not None]
        two_view = float(max(mv)) if mv else None
        for n in present:
            per_cam.setdefault(n, {})['two_view_median_px'] = res.quality.per_camera[n].get('median_px')
    fm = fit_mask if fit_mask is not None else valid
    frame_ids = gr.get('frame')
    if frame_ids is None:
        frame_ids = np.arange(pc.shape[1])
    fr = np.asarray(frame_ids)
    if fr.ndim == 1 and pc.ndim == 4:
        fr = np.broadcast_to(fr[:, None], pc.shape[1:3])
    fr = fr.reshape(-1)
    min_frames, min_frac, worst_frac_cam = None, None, None
    for c, n in enumerate(present):
        kept = fm[c] & valid[c]
        nfr = int(len(np.unique(fr[kept]))) if kept.any() else 0
        frac = float(kept.sum() / max(valid[c].sum(), 1))
        per_cam.setdefault(n, {}).update(n_obs=int(valid[c].sum()), n_fit=int(kept.sum()),
                                         inlier_frames=nfr, inlier_frac=frac)
        if min_frames is None or nfr < min_frames:
            min_frames = nfr
        if min_frac is None or frac < min_frac:
            min_frac, worst_frac_cam = frac, n
    for n in unplaced:
        per_cam[n] = dict(placed=False)
    runaway, moved_frac = _runaways(cg, init_centers, present, opts)
    for n, v in moved_frac.items():
        per_cam[n]['moved_from_init_frac'] = v
    worst_loo_cam = None
    vals = [(v.get('loo_fit_median_px') or -1, n) for n, v in per_cam.items() if n in present]
    if vals:
        worst_loo_cam = max(vals)[1]
    mf = opts['min_inlier_frames']
    loo_note = 'worst per-camera LOO median ({})'.format(worst_loo_cam)
    if unsupported:
        loo_note += '; unsupported (no held-out point triangulated by the other cameras): {}'.format(unsupported)
    rules = [
        Rule('loo_worst_px', loo_worst, 3 * nf, 10 * nf, loo_note),
        Rule('loo_unsupported', len(unsupported) if res.quality is not None and res.quality.loo is not None
             else None, 0, None, 'cameras whose LOO is unsupported: {}'.format(unsupported) if unsupported else ''),
        Rule('unplaced_cameras', len(unplaced), None, 0, 'not placed: {}'.format(unplaced) if unplaced else ''),
        Rule('runaway_cameras', len(runaway), None, 0, 'runaway: {}'.format(runaway) if runaway else ''),
        Rule('min_inlier_frames', min_frames, mf, mf / 4.0, 'fewest frames with inliers', higher_is_worse=False),
        Rule('min_inlier_frac', min_frac, 0.5, 0.1, 'smallest kept fraction ({})'.format(worst_frac_cam),
             higher_is_worse=False),
        Rule('scale_rejected', scale_rejected, None, True,
             'scale refused: {}'.format(getattr(res.scale, 'reasons', '')) if scale_rejected else ''),
    ]
    if init_support is not None:
        rules.append(Rule('init_support', init_support['overall'], 0.3, 0.05,
                          'share of multi-view observations the {} init explains within {:.3g} px{}'.format(
                              key, tau_coarse, _vggt_spread_note(rpt) if str(key).startswith('vggt') else ''),
                          higher_is_worse=False))
    if Cp < 3:
        rules.append(Rule('two_view_only', True, None, None,
                          'only {} camera(s) placed: no leave-one-camera-out check, only the 2-view '
                          'reprojection error'.format(Cp)))
        rules.append(Rule('two_view_px', two_view, 0.005 * diag, 0.02 * diag,
                          'worst per-camera 2-view reprojection median'))
    verdict, reasons, table = _rules(rules, opts)
    settings['loo_warn_px'], settings['loo_fail_px'] = 3 * nf, 10 * nf
    res.report.update(per_camera=per_cam, unplaced=unplaced, runaway=runaway, rules=table, noise_floor_px=nf,
                      init=key, candidates=list(cands), errors=dict(rpt.errors), init_support=init_support,
                      loo_unsupported=unsupported,
                      overall=None if res.quality is None else res.quality.overall)
    res.cgroup, res.selected_init = cg, key
    res.verdict, res.reasons, res.settings = verdict, reasons, settings
    res.provenance = provenance('calibrate_from_points',
                                inputs=dict(n_cameras=len(names), cam_names=names, shape=shape_in,
                                            init=init if isinstance(init, str) else 'given',
                                            frames=frames is not None, stations=stations is not None),
                                fallbacks=fallbacks + ['init {} failed: {}'.format(k, v) for k, v in rpt.errors.items()],
                                timings=timer.times)
    cg.metadata['calibration'] = dict(workflow='calibrate_from_points', verdict=verdict, init=str(key),
                                      noise_floor_px=float(nf))
    if stations is not None and 'stations' in res.report:
        srep = res.report['stations']
        cg.stations = stations.copy()
        cg.metadata['station_poses'] = _station_poses(cg, srep)
        srep.pop('_relative', None)
        srep.pop('_scale', None)
    _save(res, out)
    return res


def _check_cameras(cg, what):
    """Workflows need pinhole / fisheye cameras: a clear error for affine (telecentric) ones."""
    bad = [getattr(c, 'name', i) for i, c in enumerate(getattr(cg, 'cameras', [])) if type(c).__name__ == 'AffineCamera']
    if bad:
        raise ValueError('{}: cameras {} are AffineCamera (telecentric) models, which the workflows do not '
                         'support; convert them first with AffineCamera.to_pinhole(support) (a long-focal '
                         'pinhole that reproduces the affine projection over the working volume)'.format(what, bad))


def _init_support(cg, p, tau):
    """How well a rig explains the data: per camera, the share of its observations of points seen by
    >= min(3, C) cameras that are MSAC inliers at ``tau`` px; ``overall`` pooled; ``min_camera``."""
    _, _, inl = _robust_points(cg, p, tau)
    flat = _flat(p)
    fin = np.isfinite(flat).all(-1)
    multi = fin.sum(0) >= min(3, fin.shape[0])
    per = {}
    for c, n in enumerate(cg.get_names()):
        obs = fin[c] & multi
        per[n] = float(inl[c][obs].mean()) if obs.any() else None
    vals = [v for v in per.values() if v is not None]
    tot = fin[:, multi].sum()
    return dict(per_camera=per, overall=float(inl[:, multi].sum() / tot) if tot else 0.0,
                min_camera=float(min(vals)) if vals else None, tau_px=float(tau))


def _vggt_spread_note(rpt):
    st = (rpt.init_reports.get('vggt') or {}).get('stats') if hasattr(rpt, 'init_reports') else None
    if not st or st.get('rot_disp_deg') is None:
        return ''
    rd = np.asarray(st['rot_disp_deg'], float)
    cd = np.asarray(st.get('center_disp', np.nan), float)
    return ' (VGGT templates: median rotation spread {:.1f} deg, centre spread {:.2f} rig units, {} used)'.format(
        float(np.nanmedian(rd)), float(np.nanmedian(cd)), st.get('n_used'))


def _runaways(cg, init_centers, present, opts):
    """Runaway cameras: centre farther than ``runaway_ratio`` x the median from the rig centre AND
    moved more than ``runaway_move`` x the rig scale from its init (after a similarity fit of the
    result onto the init). A telephoto camera that was already far away in the init is no runaway."""
    from .. import scale as S
    cen = S.camera_centers(cg)
    d = np.linalg.norm(cen - np.median(cen, 0), axis=1)
    md = np.median(d) if len(d) else 0.0
    far = [md > 0 and v > opts['runaway_ratio'] * md for v in d]
    moved = np.full(len(cen), np.inf)
    try:
        if len(cen) >= 3:
            _, _, _, r, _ = S.fit_similarity(cen, init_centers)
            rig = S._rig_scale(init_centers)
            if rig > 0 and np.isfinite(rig):
                moved = r / rig
    except Exception:  # pragma: no cover
        pass
    frac = {n: (float(v) if np.isfinite(v) else None) for n, v in zip(present, moved)}
    run = [n for n, f, m in zip(present, far, moved) if f and not (m <= opts['runaway_move'])]
    return run, frac


def _stations_ba(cg, pc, gr, stations, nf, rng, noise_unit=None):
    """Final extrinsics LM with one pose per station (``groups['time']`` per frame)."""
    from .. import bundle as B
    if 'time' not in gr:
        raise ValueError("stations= needs groups['time'] (frame times, s)")
    u = _noise_unit(nf, noise_unit)
    # a moved camera's later observations are outliers of the static rig: select loosely (5 % of
    # the diagonal) for a first station fit, then keep the observations within 10 noise units of it
    diag = _diag(cg, pc) or 1000.0
    _, _, inl = _robust_points(cg, pc, max(10 * u, 0.05 * diag))
    flat = _flat(pc)
    t = np.asarray(gr['time'], float)
    if pc.ndim == 4 and t.ndim == 1:
        t = np.broadcast_to(t[:, None], pc.shape[1:3])
    t = t.reshape(-1)
    keep = inl & (inl.sum(0) >= 2)[None]
    ci, pj = np.nonzero(keep)
    kw = dict(stations=stations, only_extrinsics=True, loss='cauchy', f_scale=3 * max(nf, MIN_NOISE_PX),
              return_report=True, rng=rng)
    obs = B.Observations(ci, pj, flat[ci, pj], time=t[pj], n_points=flat.shape[1], n_cams=flat.shape[0])
    r = B.bundle_adjust_lm(cg, obs, **kw)
    e = np.asarray(r.obs_errors, float)
    if e.shape == ci.shape:
        k2 = np.nan_to_num(e, nan=np.inf) < 10 * u
        if k2.sum() >= 20 and not k2.all():
            obs = B.Observations(ci[k2], pj[k2], flat[ci[k2], pj[k2]], time=t[pj[k2]], n_points=flat.shape[1],
                                 n_cams=flat.shape[0])
            r = B.bundle_adjust_lm(cg, obs, **kw)
    # each station relative to its camera's (first-station) pose: invariant to a later rescale /
    # alignment of the world apart from the translation's scale
    from .. import scale as S
    from ..init._common import M_from_rt
    names = cg.get_names()
    rel = {}
    st = r.stations or {}
    for k, sn in enumerate(st.get('names') or []):
        c = int(st['station_cam'][k])
        E0 = np.asarray(to_numpy(cg.cameras[c].get_extrinsics_mat()), float)
        Es = M_from_rt(np.asarray(st['rvec'][k], float), np.asarray(st['tvec'][k], float))
        rel[str(sn)] = dict(camera=names[c], D=Es @ np.linalg.inv(E0))
    return dict(error_px=r.error, error_before_px=r.error_before, stations=r.stations, _relative=rel,
                _scale=S._rig_scale(S.camera_centers(cg)))


def _station_poses(cg, srep):
    """Absolute per-station poses ``{station: dict(camera, rvec, tvec)}`` of the final group."""
    from .. import scale as S
    from ..init._common import rotvec
    s = S._rig_scale(S.camera_centers(cg)) / srep['_scale'] if srep.get('_scale') else 1.0
    names = cg.get_names()
    out = {}
    for sn, v in srep['_relative'].items():
        D = np.array(v['D'])
        D[:3, 3] *= s
        E = D @ np.asarray(to_numpy(cg.cameras[names.index(v['camera'])].get_extrinsics_mat()), float)
        out[sn] = dict(camera=v['camera'], rvec=[float(x) for x in rotvec(E[:3, :3])],
                       tvec=[float(x) for x in E[:3, 3]])
    return out


# ================================================================== W3 estimate_scale

W3_DEFAULTS = dict(
    board_kwargs=None,        # passed to scale.from_board (relative gates: frame MAD, spread)
    lengths_kwargs=None,      # passed to scale.from_lengths (tau defaults to 5 px; 5 x noise_unit if noise_unit is given)
    reference_kwargs=None,    # passed to scale.from_reference
    ground_points=None, up='z', origin='centroid',   # align='ground'
    noise_unit='auto',        # 'auto' (capped noise floor / 0.5 px) or a float (1 = absolute pixel defaults)
    lengths_check=0.03,       # lengths: re-measured bones (strict threshold) off by more than this -> warn
    # board_points={cam: video(s)}: frame-major sequential detection (as W1): `n_frames` evenly spaced
    # frames per video (None: every `skip`-th); stop once `max_frames` frames show the board in >= 2 cameras
    n_frames=None, skip=20, frame_indices=None, max_frames=200, detector='auto', detector_kwargs=None, n_workers='auto',
    verdict=None,
)


@dataclass(repr=False)
class ScaleEstimate(WorkflowResult):
    """Result of :func:`estimate_scale`. ``scale`` multiplies the world units; ``sigma`` its
    robust standard error (same units as ``scale``, None if unknown); ``source``
    ('charuco' | 'board' | 'object' | 'lengths' | 'reference'); ``accepted``; ``details``
    (the backend :class:`aniposelib.scale.ScaleResult` as a dict, plus ``align`` when the gauge
    was aligned); ``cgroup``: a rescaled (and aligned) copy of the input group, None when the
    scale was refused or ``apply=False``."""
    scale: Optional[float] = None
    sigma: Optional[float] = None
    source: Optional[str] = None
    accepted: bool = False
    details: Dict[str, Any] = field(default_factory=dict)
    cgroup: Any = None

    def __iter__(self):
        yield self.scale
        yield self


def _scale_sigma(r):
    st = r.stats or {}
    s = r.scale
    if s is None:
        return None
    n = st.get('n_frames') or st.get('n_shared') or st.get('n_pairs')
    if 'frame_scale_mad' in st and n:
        return float(1.4826 * st['frame_scale_mad'] * s / np.sqrt(n))
    return None


def _board_rows_from_videos(cgroup, videos, board, opts):
    """``{camera name: video | [videos]}`` -> per-camera rows (W1's frame-major detection: the
    same frames in every camera, the videos are assumed synchronised). ``options['n_frames']``:
    that many evenly spaced frames of the shortest video, else every ``skip``-th frame; reading
    stops once ``max_frames`` sampled frames show the board in at least two cameras."""
    names = cgroup.get_names()
    unknown = [k for k in videos if str(k) not in names]
    if unknown:
        raise ValueError('board_points has videos for cameras not in the group: {}'.format(unknown))
    per = []
    for n in names:
        v = videos.get(n, [])
        per.append([v] if isinstance(v, (str, os.PathLike)) else list(v))
    det = _resolve_detector(board, opts['detector'])
    rows, summary = _detect_videos(per, board, det, opts, names, max_frames=opts['max_frames'], min_cameras=2)
    idx = [s.get('frame_indices') for s in summary['segments']]
    summary.update(detector=det if isinstance(det, (str, type(None))) else type(det).__name__,
                   n_videos=sum(len(v) for v in per), frame_indices=idx[0] if len(idx) == 1 else idx)
    return rows, summary


@thread_guard
def estimate_scale(cgroup, board=None, board_points=None, lengths=None, reference=None, align='auto',
                   apply=True, options=None, *, object=None, object_points=None):
    """W3: metric scale (and optionally the world frame) of a calibrated rig.

    Exactly one scale source:

    - ``board`` + ``board_points`` (rows, ``[C,F,K,2]`` corners, or videos ``{camera name: video
      | [videos]}``, detected as in W1: synchronised videos read front to back together, every
      ``options['skip']``-th frame (20) or ``options['n_frames']`` evenly spaced frames (the same
      indices in every camera), stopping once ``options['max_frames']`` (200; None = no limit)
      frames show the board in at least two cameras; ``report['detection']`` says what was read;
      also ``object=`` + ``object_points=`` for a
      :class:`~aniposelib.boards.PointObject`): ``scale.from_board``;
    - ``lengths=dict(points=[C,N,J,2], pairs=[(a, b), ...], ref=[...], joints=None)``:
      ``scale.from_lengths`` (MSAC threshold 5 noise units, >= min(3, C) agreeing views); the bones
      are then re-measured on the scaled rig with a threshold half as large
      (``report['lengths_check']``): rule ``lengths_check``;
    - ``reference`` (a ``CameraGroup`` of the same cameras): ``scale.from_reference``.

    ``reference`` may also be given together with ``board`` / ``lengths`` and
    ``align='reference'``: the scale comes from the board / lengths, the world frame (rotation and
    translation only) from the reference; ``details['reference_scale']`` is the scale the
    reference alone would give.

    ``align``: ``'auto'`` (= ``'reference'`` when a reference is given, else None),
    ``'reference'`` (similarity-align the gauge onto the reference: ``scale.align_to``),
    ``'ground'`` (``scale.align_world`` after scaling; ``options`` ``ground_points`` / ``up`` /
    ``origin``) or None. ``apply``: return a rescaled copy in ``result.cgroup`` (units, source and
    quality in its metadata). The input group is never modified.

    ``options['noise_unit']``: 'auto' (default; capped noise floor / 0.5 px) or a float that forces
    the unit of the lengths thresholds (1 = 5 px).

    Verdict rules: ``scale_rejected`` (the backend refused the estimate: too few frames / pairs,
    frame MAD above its relative gate; fail); ``scale_flags`` (number of quality flags such as
    ``board_scale_spread`` / ``lengths_spread`` / ``scale_reference_mismatch``; warn);
    ``lengths_check`` (lengths: the bones re-measured on the scaled rig at the strict
    threshold; warn above ``options['lengths_check']`` (3 %), fail above 10 %).
    """
    from .. import scale as S
    opts = merge_options(W3_DEFAULTS, options)
    _check_noise_unit(opts['noise_unit'])
    timer = Timer()
    if object is not None:
        if board is not None:
            raise ValueError('give board= or object=, not both')
        board, board_points = object, object_points if object_points is not None else board_points
    sources = [n for n, v in (('board', board), ('lengths', lengths), ('reference', reference)) if v is not None]
    frame_ref = False
    if len(sources) == 2 and 'reference' in sources and align == 'reference':
        sources.remove('reference')        # scale from the other source, frame from the reference
        frame_ref = True
    if len(sources) != 1:
        raise ValueError('estimate_scale needs exactly one of board=, lengths=, reference= (got {}); a '
                         "reference may be added to board= / lengths= with align='reference'".format(
                             sources or 'none'))
    src = sources[0]
    if align == 'auto':
        align = 'reference' if src == 'reference' else None
    if align not in (None, 'reference', 'ground'):
        raise ValueError("align must be 'auto', 'reference', 'ground' or None")
    if align == 'reference' and reference is None:
        raise ValueError("align='reference' needs reference=")
    settings = dict(source=src, align=align, frame_from_reference=frame_ref)
    extra = {}
    check = None
    with timer('estimate'):
        if src == 'board':
            if board_points is None:
                raise ValueError('board= needs board_points= (rows, corners or {camera: video})')
            if isinstance(board_points, dict):
                board_points, extra['detection'] = _board_rows_from_videos(cgroup, board_points, board, opts)
            elif hasattr(board, 'rows_from_array') and not isinstance(board_points, (list, tuple)):
                bp = np.asarray(to_numpy(board_points), float)
                if getattr(board, 'static', False) and bp.ndim == 4 and bp.shape[1] > 1:
                    with np.errstate(all='ignore'):
                        import warnings
                        with warnings.catch_warnings():
                            warnings.simplefilter('ignore', RuntimeWarning)
                            bp = np.nanmedian(bp, axis=1, keepdims=True)
                board_points = bp
            kw = dict(opts['board_kwargs'] or {})
            if getattr(board, 'static', False):
                kw.setdefault('min_frames', 1)
            r = S.from_board(cgroup, board_points, board, **kw)
            if hasattr(board, 'rows_from_array'):
                r.source = 'object'
        elif src == 'lengths':
            if not isinstance(lengths, dict) or 'pairs' not in lengths or \
                    not any(k in lengths for k in ('ref', 'lengths', 'ref_lengths')):
                raise ValueError('lengths must be dict(points=, pairs=, ref=)')
            pts = lengths.get('points', lengths.get('p2ds'))
            if pts is None:
                raise ValueError('lengths needs points= ([C,N,J,2])')
            ref = lengths.get('ref', lengths.get('lengths', lengths.get('ref_lengths')))
            diag = _diag(cgroup, pts)
            nf, _ = _points_noise(cgroup, pts, diag)
            u = _noise_unit(nf, opts['noise_unit'])
            # absolute 5 px (the length pairs are scored on their best views; noise-scaled taus
            # let heavy-tailed keypoints bias the scale: Emory +2 % at 5 u vs -0.8 % at 5 px)
            kw = dict(tau=5.0 * (u if opts['noise_unit'] not in (None, 'auto') else 1.0),
                      min_inliers=min(3, len(cgroup.cameras)))
            kw.update(opts['lengths_kwargs'] or {})
            settings.update(noise_floor_px=nf, noise_unit=u, tau_px=kw['tau'], min_inliers=kw['min_inliers'])
            r = S.from_lengths(cgroup, pts, lengths['pairs'], ref, joints=lengths.get('joints'), **kw)
            if r.scale is not None and r.accepted:
                # self-consistency: the bones re-measured on the scaled rig at a stricter threshold
                tmp = cgroup.copy()
                S.rescale(tmp, r.scale)
                kc = dict(kw, tau=0.5 * kw['tau'])
                rc = S.from_lengths(tmp, pts, lengths['pairs'], ref, joints=lengths.get('joints'), **kc)
                per = {k: dict(length_after=v.get('median_length'), reference=v.get('reference'),
                               ratio=(v['median_length'] / v['reference']) if v.get('median_length') else None,
                               n=v.get('n'))
                       for k, v in (rc.details or {}).get('per_pair', {}).items()}
                rat = [v['ratio'] for v in per.values() if v['ratio'] is not None]
                mr = float(np.median(rat)) if rat else None
                check = abs(mr - 1) if mr is not None else None
                extra['lengths_check'] = dict(tau_px=kc['tau'], per_pair=per, median_ratio=mr, rel_dev=check)
        else:
            r = S.from_reference(cgroup, reference, **(opts['reference_kwargs'] or {}))
    details = r.to_dict()
    if frame_ref:
        try:
            rr = S.from_reference(cgroup, reference, **(opts['reference_kwargs'] or {}))
            details['reference_scale'] = rr.scale
            if r.scale and rr.scale:
                details['reference_scale_ratio'] = float(rr.scale / r.scale)
        except Exception as exc:  # pragma: no cover
            details['reference_scale_error'] = str(exc)
    out = None
    if apply and r.accepted and r.scale is not None:
        with timer('apply'):
            out = cgroup.copy()
            if align == 'reference' and not frame_ref:
                al = S.align_to(out, reference)
                md = out.metadata
                md['scale_source'] = 'reference'
                if r.units is not None:
                    md['units'] = str(r.units)
                q = r.quality()
                if q:
                    md['scale_quality'] = S._toml_clean(dict(q))
                details['align'] = al.to_dict()
            else:
                r.apply(out)
                if frame_ref:
                    units = out.metadata.get('units')
                    al = S.align_to(out, reference, with_scale=False)
                    if units is not None:
                        out.metadata['units'] = units
                    details['align'] = al.to_dict()
            if align == 'ground':
                al = S.align_world(out, opts['ground_points'], up=opts['up'], origin=opts['origin'])
                details['align'] = al.to_dict()
    rules = [Rule('scale_rejected', not r.accepted, None, True,
                  'scale refused: {}'.format(r.reason) if not r.accepted else ''),
             Rule('scale_flags', len(r.flags), 0, None, 'flags: {}'.format(r.flags) if r.flags else '')]
    if src == 'lengths':
        rules.append(Rule('lengths_check', check, opts['lengths_check'], 0.1,
                          '|median bone length / reference - 1| re-measured on the scaled rig at half the threshold'))
    verdict, reasons, table = _rules(rules, opts)
    report = dict(rules=table, stats=r.stats, flags=list(r.flags), reason=r.reason, units=r.units)
    report.update(extra)
    return ScaleEstimate(report=report,
                         verdict=verdict, reasons=reasons, settings=settings,
                         provenance=provenance('estimate_scale', inputs=dict(source=src, cameras=cgroup.get_names(),
                                                                             frame_from_reference=frame_ref),
                                               timings=timer.times),
                         scale=r.scale, sigma=_scale_sigma(r), source=r.source, accepted=bool(r.accepted),
                         details=details, cgroup=out)


# ================================================================== W4 evaluate_calibration

W4_DEFAULTS = dict(
    min_others=3,             # LOO: other cameras that must see a point
    detect_moves_kwargs=None, # passed to stations.detect_moves (tau / huber scaled by the noise unit; z_thr 10)
    compare_kwargs=None,      # passed to quality.compare
    min_ray_deg=2.0,          # ray-angle degeneracy threshold
    noise_unit='auto',        # 'auto' (capped noise floor / 0.5 px) or a float (1 = absolute pixel defaults)
    cam_names=None,           # camera names of the points' first axis (default: the group's order)
    tau_px=None,              # robust triangulation threshold (default 10 noise units)
    tau_tri_px=None,          # LOO triangulation threshold (default 5 noise units)
    thresholds_px=None,       # LOO agreement thresholds (default (5, 8, 10) noise units)
    loo_window_s='auto',      # time-blocked LOO with times=: window (s); 'auto' ~ 20 windows; None = off
    min_window_obs=200,       # ... windows with fewer eligible observations are not judged
    bad_window_ratio=0.5,     # ... a camera is bad in a window when its LOO coverage / agreement < this x its median
    min_bad_cameras=None,     # ... a window is bad when this many cameras are (None: max(2, 25 % of the cameras))
    compare_to=None,          # {name: CameraGroup}: paired LOO agreement vs other calibrations
    clusters=None,            # [N] bootstrap cluster per point (default: the LOO windows / 100 frame blocks)
    n_boot=1000,              # bootstrap resamples of the paired comparison
    verdict=None,
)


@dataclass(repr=False)
class CalibrationEvaluation(WorkflowResult):
    """Result of :func:`evaluate_calibration`: ``per_camera`` (LOO median / coverage / agreement,
    reprojection median, inlier rate, lens validity, area coverage, flags), ``moves``
    (:class:`aniposelib.stations.MoveReport` or None), ``comparison``
    (:class:`aniposelib.quality.CompareResult` or None), ``quality`` (the full
    :class:`aniposelib.quality.CalibrationReport`), ``per_window`` (time-blocked LOO with
    ``times=``: per window start time, LOO agreement / median / coverage)."""
    per_camera: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    per_window: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    moves: Any = None
    comparison: Any = None
    quality: Any = None

    _skip = ('quality', 'moves')

    def to_dict(self):
        d = super().to_dict()
        if self.moves is not None:
            d['moves'] = dict(moved=list(self.moves.moved), change_points=_to_builtin(self.moves.change_points))
        return d


def _to_builtin(x):
    from ._common import _jsonable
    return _jsonable(x)


def _paired_agree(e0, e1, eligible, clusters, thresholds, n_boot, seed=0):
    """Paired LOO agreement difference ``e1 - e0`` (both ``[C,N]``, same eligibility), clusters
    resampled (block bootstrap; port of Emory ``evaluate_v2.paired_compare``). Agreement is the
    mean over cameras of (hits / eligible). Returns ``{agree@T: dict(a, b, delta, ci95)}``."""
    windows, inv = np.unique(clusters, return_inverse=True)
    nw, nc = len(windows), e0.shape[0]
    rng = np.random.default_rng(seed)
    boots = [rng.integers(nw, size=nw) for _ in range(int(n_boot))]
    den = np.zeros((nw, nc))
    for c in range(nc):
        den[:, c] = np.bincount(inv, weights=eligible[c], minlength=nw)

    def macro(h, d):
        with np.errstate(all='ignore'):
            v = np.where(d > 0, h / d, np.nan)
        return float(np.nanmean(v)) if np.isfinite(v).any() else float('nan')
    out = {}
    for thr in thresholds:
        h0 = np.zeros_like(den)
        h1 = np.zeros_like(den)
        for c in range(nc):
            with np.errstate(invalid='ignore'):
                h0[:, c] = np.bincount(inv, weights=eligible[c] & np.isfinite(e0[c]) & (e0[c] < thr), minlength=nw)
                h1[:, c] = np.bincount(inv, weights=eligible[c] & np.isfinite(e1[c]) & (e1[c] < thr), minlength=nw)
        d = macro((h1 - h0).sum(0), den.sum(0))
        bs = [macro((h1[k] - h0[k]).sum(0), den[k].sum(0)) for k in boots]
        bs = [b for b in bs if np.isfinite(b)]
        out['agree@{:.4g}'.format(thr)] = dict(a=macro(h0.sum(0), den.sum(0)), b=macro(h1.sum(0), den.sum(0)),
                                               delta=d, ci95=np.percentile(bs, [2.5, 97.5]).tolist() if bs else None,
                                               n_clusters=int(nw))
    return out


def _auto_window(times):
    t = np.asarray(times, float)
    t = t[np.isfinite(t)]
    if len(t) < 2:
        return None
    span = float(t.max() - t.min())
    if span <= 0:
        return None
    w = span / 20.0
    e = 10 ** np.floor(np.log10(w))       # a round number: 1, 2 or 5 x 10^k
    return float(min((m * e for m in (1, 2, 5, 10)), key=lambda v: abs(np.log(v / w))))


@thread_guard
def evaluate_calibration(cgroup, points, times=None, reference=None, board=None, options=None, *,
                         board_points=None, object=None, object_points=None, device='auto'):
    """W4: how good is this calibration on these observations?

    Parameters
    ----------
    cgroup : the calibration to evaluate (pinhole / fisheye cameras; convert an AffineCamera group
        with ``AffineCamera.to_pinhole`` first).
    points : ``C x N x 2`` / ``C x N x J x 2`` pixels in ``cgroup`` order (NaN missing), or in the
        order of ``options['cam_names']``: then only the group's cameras are kept
        (``report['cameras_dropped']``) and group cameras without points are left out of the
        evaluation (``report['cameras_missing']``).
    times : ``[N]`` frame times (s): per-window moved-camera check (``stations.detect_moves``) and a
        time-blocked LOO (``per_window``).
    reference : a ``CameraGroup`` to compare with (``quality.compare``: aligned centre residuals,
        rotation / focal deltas, paired LOO).
    board, board_points : a board and its detections (rows or corners) -> board shape residual
        (``object=`` / ``object_points=`` for a ``PointObject``).
    device : ``'auto'`` (cuda when available), ``'cpu'`` or ``'cuda'``: where the triangulations
        run (a copy of ``cgroup`` is moved there; ``cgroup`` is not modified, results are numpy).

    Options (besides the verdict overrides): ``noise_unit`` ('auto' or a float: 1 = the 1.0
    absolute pixel defaults), ``tau_px`` / ``tau_tri_px`` / ``thresholds_px`` (the robust and LOO
    triangulation thresholds and the LOO agreement thresholds in px, e.g. Emory's yardstick
    ``tau_tri_px=5, thresholds_px=(5, 8, 10)``), ``loo_window_s`` (time-blocked LOO window with
    ``times=``; 'auto' ~ 20 windows), ``compare_to={name: CameraGroup}`` with ``clusters=[N]``
    (paired LOO agreement difference with a cluster bootstrap: ``report['compare_to']``),
    ``detect_moves_kwargs`` (``z_thr`` defaults to 10), ``cam_names``.

    Contents: LOO per camera (fixed denominator), coverage, inlier rates, lens validity, ray
    angles, plus moves / comparison / board shape when their inputs are given. Project policy
    (e.g. a reuse ladder) belongs downstream, on top of ``per_camera``. With 2 cameras there is no
    LOO: the 2-view reprojection error is reported instead.

    Verdict rules (``nf`` = noise floor of the observations under this calibration):

    - ``loo_worst_px``: worst per-camera LOO median; warn above 3 nf, fail above 10 nf;
    - ``loo_coverage_min``: smallest per-camera LOO coverage; warn below 0.5, fail below 0.2
      (the reason names cameras with a low inlier rate that probably cause it);
    - ``loo_unsupported``: cameras whose held-out points never triangulate from the others (warn);
    - ``lens_invalid``: cameras whose lens model is not invertible over the image (warn);
    - ``grazing_frac``: fraction of points whose largest ray angle is below ``min_ray_deg``
      (warn above 0.5);
    - ``moved_cameras``: cameras with a detected bump / re-aim (fail);
    - ``bad_windows_frac``: share of time windows in which at least ``min_bad_cameras`` (default
      max(2, 25 % of the cameras)) cameras have a LOO coverage, or a LOO agreement among the
      supported points, below ``bad_window_ratio`` x their median window (desync / broken data;
      warn above 5 %);
    - ``center_residual_frac``: worst aligned centre residual vs the reference / rig scale;
      warn 1 % (warn only: a reference is often rougher than the calibration under test);
    - ``board_shape_rel``: median board shape residual / object size; warn 0.5 %, fail 2 %;
    - ``two_view_only`` / ``two_view_px`` (2 cameras): no LOO (warn); worst per-camera 2-view
      reprojection median, warn above 0.5 %, fail above 2 % of the image diagonal.
    """
    from .. import quality as Q
    from ._common import reindex_cameras
    opts = merge_options(W4_DEFAULTS, options)
    _check_noise_unit(opts['noise_unit'])
    timer = Timer()
    _check_cameras(cgroup, 'evaluate_calibration')
    if object is not None:
        board, board_points = object, object_points if object_points is not None else board_points
    p = np.asarray(to_numpy(points), dtype='float64')
    gnames = cgroup.get_names()
    missing, dropped = [], []
    if opts['cam_names'] is not None:
        if p.ndim not in (3, 4):
            raise ValueError('points must be C x N x 2 or C x N x J x 2, got {}'.format(p.shape))
        p, missing, dropped = reindex_cameras(p, opts['cam_names'], gnames)
        keep = [n for n in gnames if n not in missing]
        if len(keep) < 2:
            raise ValueError('only {} of the group cameras {} are in cam_names'.format(keep, gnames))
        if missing:
            p = p[[gnames.index(n) for n in keep]]
            cgroup = cgroup.subset_cameras_names(keep)
    names = cgroup.get_names()
    if p.ndim not in (3, 4) or p.shape[0] != len(names) or p.shape[-1] != 2:
        raise ValueError('points must be C x N x 2 or C x N x J x 2 with C={}, got {}'.format(len(names), p.shape))
    if board is not None and board_points is None:
        raise ValueError('board= needs board_points= (rows or corners)')
    C = len(names)
    J = p.shape[2] if p.ndim == 4 else 1
    diag = _diag(cgroup, p) or 1000.0
    cg = _on_device(cgroup, device)
    with timer('noise'):
        nf, _ = _points_noise(cg, p, diag)
    u = _noise_unit(nf, opts['noise_unit'])
    tau = float(opts['tau_px']) if opts['tau_px'] is not None else 10.0 * u
    tau_tri = float(opts['tau_tri_px']) if opts['tau_tri_px'] is not None else 5.0 * u
    thr = tuple(float(t) for t in opts['thresholds_px']) if opts['thresholds_px'] is not None \
        else tuple(t * u for t in (5, 8, 10))
    agree_t = thr[1] if len(thr) >= 3 else thr[len(thr) // 2]
    min_others = min(int(opts['min_others']), max(C - 1, 1))
    use_loo = C >= 3
    settings = dict(image_diag_px=diag, noise_floor_px=nf, noise_unit=u, tau_px=tau, tau_tri_px=tau_tri,
                    thresholds_px=list(thr), min_others=min_others, loo_warn_px=3 * nf, loo_fail_px=10 * nf)
    # time windows of the blocked LOO
    win = None
    if times is not None and use_loo and opts['loo_window_s'] is not None:
        tt = np.asarray(times, float).reshape(-1)
        if tt.shape[0] != p.shape[1]:
            raise ValueError('times has {} entries for {} frames'.format(tt.shape[0], p.shape[1]))
        ws = _auto_window(tt) if opts['loo_window_s'] == 'auto' else float(opts['loo_window_s'])
        if ws:
            t0 = float(np.nanmin(tt))
            win = np.floor((tt - t0) / ws).astype(int)
            settings['loo_window_s'] = ws
    with timer('report'):
        qr = Q.report(cg, p, tau=tau, tau_tri=tau_tri, max_loo_px=60.0 * u, thresholds=thr,
                      min_others=min_others, loo=use_loo, min_views=min(3, C),
                      groups=None if win is None else np.repeat(win, J))
    per = {n: dict(qr.per_camera[n]) for n in names}
    with timer('lens'):
        n_lens_bad = 0
        for n, cam in zip(names, cgroup.cameras):
            try:
                lv = Q.lens_validity(cam, max_nonfinite_frac=0.05 if getattr(cam, 'fisheye', False) else 0.0)
                per[n]['lens_valid'] = bool(lv['valid'])
                n_lens_bad += not lv['valid']
            except Exception:  # pragma: no cover
                pass
        cov = Q.coverage(p, min_cams=min(opts['min_others'], C), cgroup=cgroup)
        for n in names:
            per[n]['area'] = cov['per_camera'][n].get('area')
    with timer('rays'):
        X, _, inl = _robust_points(cg, p, tau)
        X[inl.sum(0) < 2] = np.nan
        ok = np.isfinite(X).all(1)
        grazing = None
        if ok.any():
            ra = Q.ray_angles(cg, X[ok], _flat(p)[:, ok], min_deg=opts['min_ray_deg'])
            grazing = float(ra['frac_points_below'])
    report = dict(overall=qr.overall, coverage=cov, rules=None, cameras_missing=missing, cameras_dropped=dropped)
    moves = comparison = None
    moved, center_frac, shape_rel = [], None, None
    if times is not None:
        with timer('moves'):
            from ..stations import detect_moves
            kw = dict(tau=tau, huber_px=3.0 * u, z_thr=10.0)
            kw.update(opts['detect_moves_kwargs'] or {})
            moves = detect_moves(cg, p, times, **kw)
            moved = list(moves.moved)
            settings['detect_moves'] = {k: v for k, v in kw.items() if isinstance(v, (int, float))}
            report['moves_info'] = {k: v for k, v in moves.info.items() if k in ('retest', 'low_support')}
            for n in names:
                per[n]['moved'] = n in moved
    # time-blocked LOO: per window and camera, LOO agreement at the middle threshold
    per_window, bad_frac, bad_windows = {}, None, []
    if win is not None and qr.loo is not None:
        ws = settings['loo_window_s']
        t0 = float(np.nanmin(np.asarray(times, float)))
        wf = np.repeat(win, J)
        le, el = np.asarray(qr.loo.errors, float), np.asarray(qr.loo.eligible, bool)
        labs, inv = np.unique(wf, return_inverse=True)
        nW = len(labs)
        den = np.zeros((C, nW))
        hit = np.zeros((C, nW))
        sup = np.zeros((C, nW))
        with np.errstate(invalid='ignore'):
            for c in range(C):
                den[c] = np.bincount(inv, weights=el[c], minlength=nW)
                fin = el[c] & np.isfinite(le[c])
                sup[c] = np.bincount(inv, weights=fin, minlength=nW)
                hit[c] = np.bincount(inv, weights=fin & (le[c] < agree_t), minlength=nW)
            # agreement among the supported points (as the projects' hourly LOO tables): a window of
            # false / unmatched detections lowers the coverage, not the agreement
            ag = np.where(sup > 0, hit / np.maximum(sup, 1), np.nan)
            cv = np.where(den > 0, sup / np.maximum(den, 1), np.nan)
        # a window is judged on its eligible observations; it is bad for a camera when the share of
        # its held-out points the others can triangulate (coverage) or the agreement among those
        # collapses relative to that camera's typical window (desync breaks both)
        min_cw = max(10.0, opts['min_window_obs'] / C)
        judged = (den >= min_cw)
        r_ = opts['bad_window_ratio']

        def cmed(a, m):
            return np.array([np.median(a[c, m[c]]) if m[c].any() else np.nan for c in range(C)])
        cov_med = cmed(cv, judged)
        jag = judged & (sup >= min_cw)
        cam_med = cmed(ag, jag)
        with np.errstate(invalid='ignore'):
            badcw = judged & ((cv < r_ * cov_med[:, None]) | (jag & (ag < r_ * cam_med[:, None])))
        okw = judged.any(0) & (den.sum(0) >= opts['min_window_obs'])
        # a window-wide problem (desync, broken data) hits several cameras; one camera dipping in
        # one window is not reported (it shows up in its per-camera LOO)
        min_bad = opts['min_bad_cameras'] or max(2, int(np.ceil(0.25 * C)))
        settings['min_bad_cameras'] = min_bad
        for k, lab in enumerate(labs):
            key = '{:.6g}'.format(t0 + int(lab) * ws)
            gd = qr.per_group.get(str(lab), {})
            tot = den[:, k].sum()
            per_window[key] = dict(
                window=int(lab), n_eligible=int(tot),
                loo_agree=float(hit[:, k].sum() / tot) if tot else None,
                loo_agree_supported=float(hit[:, k].sum() / sup[:, k].sum()) if sup[:, k].sum() else None,
                loo_median_px=gd.get('loo_median_px'), loo_coverage=float(sup[:, k].sum() / tot) if tot else None,
                per_camera={n: dict(agree=None if not np.isfinite(ag[c, k]) else float(ag[c, k]),
                                    coverage=None if not np.isfinite(cv[c, k]) else float(cv[c, k]))
                            for c, n in enumerate(names)},
                bad_cameras=[n for c, n in enumerate(names) if badcw[c, k]], judged=bool(okw[k]))
            per_window[key]['bad'] = bool(okw[k] and badcw[:, k].sum() >= min_bad)
        if okw.sum() >= 3:
            bad_windows = [k for k, v in per_window.items() if v['bad']]
            bad_frac = len(bad_windows) / int(okw.sum())
            report['windows'] = dict(window_s=ws, n_windows=len(per_window), n_judged=int(okw.sum()),
                                     camera_median_agree={n: (None if not np.isfinite(v) else float(v))
                                                          for n, v in zip(names, cam_med)},
                                     camera_median_coverage={n: (None if not np.isfinite(v) else float(v))
                                                             for n, v in zip(names, cov_med)},
                                     bad=bad_windows, agree_px=agree_t)
    if reference is not None:
        with timer('compare'):
            try:
                comparison = Q.compare(cgroup, reference, **(opts['compare_kwargs'] or {}))
                center_frac = comparison.overall.get('center_residual_max_frac')
                for n, d in comparison.per_camera.items():
                    per[n]['reference'] = d
            except ValueError as exc:
                report['compare_error'] = str(exc)
    if opts['compare_to']:
        with timer('compare_to'):
            report['compare_to'] = _compare_to(cg, p, qr, opts, win, J, tau_tri, min_others, thr, device)
    if board is not None:
        with timer('board'):
            bs = Q.board_shape_residual(cg, board_points, board)
            report['board_shape'] = bs
            objp = np.asarray(board.get_object_points(), float).reshape(-1, 3)
            size = float(np.max(np.linalg.norm(objp - objp.mean(0), axis=1)) * 2)
            if bs.get('median') is not None and size > 0:
                shape_rel = bs['median'] / size
    loo = np.array([np.nan if per[n].get('loo_median_px') is None else per[n]['loo_median_px'] for n in names])
    covs = np.array([np.nan if per[n].get('loo_coverage') is None else per[n]['loo_coverage'] for n in names])
    unsupported = [n for n in names if use_loo and per[n].get('loo_median_px') is None
                   and (per[n].get('loo_n_eligible') or 0) > 0]
    for n in names:
        if per[n].get('loo_coverage') is not None:
            per[n]['loo_unsupported_frac'] = 1.0 - per[n]['loo_coverage']
    worst_cam = names[int(np.nanargmax(loo))] if np.isfinite(loo).any() else None
    loo_worst = float(np.nanmax(loo)) if np.isfinite(loo).any() else None
    low_cov = names[int(np.nanargmin(covs))] if np.isfinite(covs).any() else None
    # cameras that disagree with the others in the all-camera robust triangulation: a camera's
    # unsupported LOO points are usually caused by one of them (the bad camera is among the 'others')
    rates = {n: per[n].get('inlier_rate') for n in names if per[n].get('inlier_rate') is not None}
    suspects = []
    if len(rates) >= 3:
        mr = float(np.median(list(rates.values())))
        suspects = [n for n, v in rates.items() if v < 0.5 * mr]
    report['low_inlier_cameras'] = suspects
    cov_note = 'smallest LOO coverage ({})'.format(low_cov)
    if low_cov is not None and suspects and low_cov not in suspects:
        cov_note += '; probably caused by {} (low inlier rate: disagrees with the other cameras)'.format(suspects)
    loo_note = 'worst per-camera LOO median ({})'.format(worst_cam)
    if unsupported:
        loo_note += '; unsupported (no held-out point triangulated by the other cameras): {}'.format(unsupported)
    two_view = None
    if not use_loo:
        mv = [per[n].get('median_px') for n in names if per[n].get('median_px') is not None]
        two_view = float(max(mv)) if mv else None
    rules = [
        Rule('loo_worst_px', loo_worst, 3 * nf, 10 * nf, loo_note),
        Rule('loo_coverage_min', float(np.nanmin(covs)) if np.isfinite(covs).any() else None, 0.5, 0.2,
             cov_note, higher_is_worse=False),
        Rule('loo_unsupported', len(unsupported) if use_loo else None, 0, None,
             'cameras whose LOO is unsupported: {}'.format(unsupported) if unsupported else ''),
        Rule('loo_unmeasured', (use_loo and not np.isfinite(loo).any() and not unsupported) or None, 0, None,
             'no point is seen by {} other cameras: no leave-one-camera-out check'.format(min_others)),
        Rule('lens_invalid', n_lens_bad, 0, None, 'lens model not invertible over the image'),
        Rule('grazing_frac', grazing, 0.5, None, 'points with all ray angles < {} deg'.format(opts['min_ray_deg'])),
        Rule('moved_cameras', len(moved) if times is not None else None, None, 0,
             'moved: {}'.format(moved) if moved else ''),
        Rule('center_residual_frac', center_frac, 0.01, None, 'aligned centre residual vs reference / rig scale'),
        Rule('board_shape_rel', shape_rel, 0.005, 0.02, 'board shape residual / object size'),
        Rule('bad_windows_frac', bad_frac, 0.05, None,
             'time windows where >= {} cameras\' LOO coverage or agreement is < {} x their median window: {}'.format(
                 min_bad, opts['bad_window_ratio'], bad_windows[:10]) if bad_windows else ''),
    ]
    if missing:
        rules.append(Rule('cameras_missing', len(missing), 0, None, 'group cameras without points: {}'.format(missing)))
    if not use_loo:
        rules.append(Rule('two_view_only', True, None, None,
                          'only {} cameras: no leave-one-camera-out check, only the 2-view reprojection error'.format(C)))
        rules.append(Rule('two_view_px', two_view, 0.005 * diag, 0.02 * diag,
                          'worst per-camera 2-view reprojection median'))
    verdict, reasons, table = _rules(rules, opts)
    report['rules'] = table
    report['loo_unsupported'] = unsupported
    report['per_camera_flags'] = {n: per[n].get('flags', []) for n in names}
    return CalibrationEvaluation(report=report, verdict=verdict, reasons=reasons, settings=settings,
                                 provenance=provenance('evaluate_calibration',
                                                       inputs=dict(cameras=names, shape=list(p.shape),
                                                                   times=times is not None,
                                                                   reference=reference is not None,
                                                                   board=board is not None,
                                                                   compare_to=list(opts['compare_to'] or [])),
                                                       timings=timer.times),
                                 per_camera=per, per_window=per_window, moves=moves, comparison=comparison,
                                 quality=qr)


def _compare_to(cg, p, qr, opts, win, J, tau_tri, min_others, thr, device):
    """``options['compare_to']``: paired LOO agreement of this calibration vs each other one, on
    the same observations (other - this; positive = the other one agrees more)."""
    from .. import quality as Q
    names = cg.get_names()
    N = p.shape[1]
    if opts['clusters'] is not None:
        cl = np.asarray(opts['clusters']).reshape(-1)
        if cl.shape[0] == N and J > 1:
            cl = np.repeat(cl, J)
        if cl.shape[0] != N * J:
            raise ValueError('clusters must have one label per point ({}), got {}'.format(N, cl.shape[0]))
    elif win is not None:
        cl = np.repeat(win, J)
    else:
        cl = np.repeat(np.arange(N) // max(1, int(np.ceil(N / 100))), J)
    if qr.loo is not None:
        e0, el = qr.loo.errors, qr.loo.eligible
    else:
        e0, el = Q.loo_errors(cg, p, tau_tri=tau_tri, min_others=min_others)
    out = {}
    for name, other in dict(opts['compare_to']).items():
        try:
            on = other.get_names()
            miss = [n for n in names if n not in on]
            if miss:
                raise ValueError('cameras {} are not in {!r}'.format(miss, name))
            og = other if on == names else other.subset_cameras_names(names)
            og = _on_device(og, device)
            e1, _ = Q.loo_errors(og, p, tau_tri=tau_tri, min_others=min_others)
            d = _paired_agree(e0, e1, el, cl, thr, opts['n_boot'])
            out[str(name)] = d
        except Exception as exc:
            out[str(name)] = dict(error='{}: {}'.format(type(exc).__name__, exc))
    return out
