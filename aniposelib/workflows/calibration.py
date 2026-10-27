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
  ``u = noise_floor / 0.5 px``: the backend defaults (tuned for 0.5 px noise) are multiplied by
  ``u``. First-pass thresholds are fractions of the image diagonal. Lengths are relative to
  the rig or object size. Every derived value is in ``result.settings``.
* Verdicts follow the rules listed in each docstring; override them with
  ``options={'verdict': {rule: (warn_at, fail_at) | None}}``. Workflows raise only on unusable
  inputs (wrong shapes, missing image sizes, contradictory arguments).
"""

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from ._common import (MIN_NOISE_PX, Rule, Timer, WorkflowResult, evaluate_rules, image_diagonal,
                      merge_options, noise_floor, provenance, save_calibration, to_numpy)

__all__ = ['calibrate_from_board', 'BoardCalibration', 'calibrate_from_points', 'PointCalibration',
           'estimate_scale', 'ScaleEstimate', 'evaluate_calibration', 'CalibrationEvaluation']


# ================================================================== shared helpers

def _noise_unit(nf):
    """Noise unit ``u``: backend pixel defaults (tuned for 0.5 px noise) are multiplied by it."""
    return max(float(nf), MIN_NOISE_PX) / MIN_NOISE_PX


def _device(device):
    if device in (None, 'auto'):
        return 'cuda' if torch.cuda.is_available() else 'cpu'
    return str(device)


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
    sup = inl.sum(0) >= 3
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
    skip=20, frame_indices=None, n_workers=1, detector_kwargs=None,   # video detection
    prior_sigma_logf=0.05, prior_sigma_k=0.1,                          # intrinsics=<CameraGroup>
    object_sigma=None,        # static PointObject: not supported yet (points held fixed)
    min_bridge_points=6,      # bridge: minimum points per extra component
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
            try:
                lr = I.calibrate_ladder_rows(r, board, size, min_corners=opts['min_corners'], models=models)
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
    tau = 10 * _noise_unit(nf)
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
    - ``lens_invalid``: cameras whose lens model is not invertible over the image (warn).
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
            det = detector
            if det == 'auto':
                det = 'deepcharuco' if _deepcharuco_available(board) else 'opencv'
            if not hasattr(board, 'detector_name'):
                det = None
            settings['detector'] = det if isinstance(det, (str, type(None))) else type(det).__name__
            kw = dict(verbose=False, frame_indices=opts['frame_indices'], n_workers=opts['n_workers'],
                      skip=opts['skip'], detector_kwargs=opts['detector_kwargs'])
            try:
                all_rows = cg.get_rows_videos(videos, board, detector=det, **kw)
            except Exception as exc:
                if det != 'deepcharuco' or detector != 'auto':
                    raise
                fallbacks.append('deepcharuco failed ({}: {}), using opencv'.format(type(exc).__name__, exc))
                settings['detector'] = 'opencv'
                all_rows = cg.get_rows_videos(videos, board, detector='opencv', **kw)
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
        spec = ParamSpec(intrinsics=None, priors=IntrinsicsPrior(intrinsics, sigma_logf=opts['prior_sigma_logf'],
                                                                 sigma_k=opts['prior_sigma_k']))
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
    report['per_camera'] = per_cam
    thr = float('{:.4g}'.format(max(1.0, 3 * nf)))
    settings['board_reproj_warn_px'] = thr
    rules = [
        Rule('board_reproj_px', worst_px, thr, 2 * thr, 'worst per-camera median board reprojection'),
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
    ba_kwargs=None, sfm_kwargs=None, vggt_kwargs=None, select_kwargs=None, people_kwargs=None,
    intrinsics_free=('f', 'k1'),                  # refine_intrinsics: parameters freed
    intrinsics_sigma_logf=0.05, intrinsics_sigma_k=0.1,   # ... under priors around the given ones
    heldout_every=5,          # refine_intrinsics='auto': every k-th window is held out
    intrinsics_gain=0.02,     # ... accepted if the held-out LOO median improves by this fraction
    min_inlier_frames=20,     # verdict: frames with inliers per camera
    runaway_ratio=5.0,        # verdict: camera centre farther than this x the median from the rig centre
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

    def to_dict(self):
        d = super().to_dict()
        if self.quality is not None and hasattr(self.quality, 'to_dict'):
            q = self.quality.to_dict()
            d['quality'] = {k: q[k] for k in ('per_camera', 'overall') if k in q}
        return d


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


def _candidates(base, p, init, frames, gr, opts, rng, rpt, names):
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
                kw = dict(frame_id=frame_id, window_id=window_id, rotation_priors=priors, rng=rng)
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
    u = _noise_unit(nf)
    fine = tuple(tuple(min(float(t * u), coarse[-1]) for t in rnd) for rnd in opts['fine'])
    rej = B.RejectSchedule()
    rej = B.RejectSchedule(frame=tuple(u * t for t in rej.frame), keypoint=tuple(u * t for t in rej.keypoint),
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
                    reject_px=rej.to_dict(), resect_px=rkw, report_tau_px=fine[-1][-1], noise_floor_px=nf2)
    return r2, dict(method='bundle_adjust_robust', before_px=r1.error_before, after_px=r2.error,
                    kept=not (r1.reverted and r2.reverted), result=r2), settings


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


def _loo_median(cg, p, tau_tri):
    from .. import quality as Q
    r = Q.loo_reprojection(cg, p, tau_tri=tau_tri)
    m = r.medians()
    return float(np.nanmedian(m)) if np.isfinite(m).any() else np.inf


def _gate_intrinsics(cg, pc, gr, mode, nf, opts, rng):
    """Evidence-gated focal + k1 refinement. Returns ``(cgroup, info)``."""
    from .. import bundle as B
    from ..params import IntrinsicsPrior, ParamSpec
    u = _noise_unit(nf)
    tau = 10.0 * u
    spec = ParamSpec(intrinsics=tuple(opts['intrinsics_free']),
                     priors=IntrinsicsPrior(cg.copy(), sigma_logf=opts['intrinsics_sigma_logf'],
                                            sigma_k=opts['intrinsics_sigma_k']))

    def fit(group, p):
        _, _, inl = _robust_points(group, p, tau, min_inliers=2)
        flat = _flat(p)
        keep = inl & (inl.sum(0) >= 3)[None]
        q = np.where(keep[..., None], flat, np.nan)
        B.bundle_adjust_lm(group, q, spec=spec, loss='cauchy', f_scale=3 * max(nf, MIN_NOISE_PX),
                           n_iter=30, rng=rng)
        return group
    if mode is True:
        return fit(cg, pc), dict(decision='refined', reason='refine_intrinsics=True', free=list(opts['intrinsics_free']))
    ho = _heldout_mask(pc, gr, opts['heldout_every'])
    train = pc.copy()
    train[:, ho] = np.nan
    test = pc.copy()
    test[:, ~ho] = np.nan
    if not ho.any() or ho.all():
        return cg, dict(decision='kept', reason='not enough windows for a held-out check')
    base = cg.copy()
    B.bundle_adjust_lm(base, _flat(train), only_extrinsics=True, loss='cauchy',
                       f_scale=3 * max(nf, MIN_NOISE_PX), n_iter=20, rng=rng)
    trial = fit(base.copy(), train)
    e0, e1 = _loo_median(base, test, 5.0 * u), _loo_median(trial, test, 5.0 * u)
    gain = (e0 - e1) / e0 if np.isfinite(e0) and e0 > 0 else 0.0
    info = dict(heldout_loo_fixed_px=e0, heldout_loo_refined_px=e1, gain=gain,
                min_gain=opts['intrinsics_gain'], free=list(opts['intrinsics_free']),
                n_heldout_points=int(ho.sum()))
    if np.isfinite(e1) and gain > opts['intrinsics_gain']:
        fit(cg, pc)
        info.update(decision='refined', reason='held-out LOO improved by {:.1%}'.format(gain))
    else:
        info.update(decision='kept', reason='held-out LOO did not improve enough ({:.1%})'.format(gain))
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


def calibrate_from_points(points, intrinsics, init='auto', frames=None, groups=None,
                          refine_intrinsics='auto', scale=None, stations=None, options=None, rng=0,
                          device='auto', out=None):
    """W2: calibrate extrinsics from 2D correspondences with known (or nominal) intrinsics.

    Parameters
    ----------
    points : ``C x N x 2`` | ``C x N x J x 2`` pixels (NaN missing; same index = same 3D point),
        an :class:`aniposelib.bundle.Observations`, or ``C x F x P x J x 2`` with
        ``init='people'`` (unassociated people).
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
      rig centre (fail);
    - ``min_inlier_frames``: fewest frames with inlier observations of a camera; warn below
      ``min_inlier_frames``, fail below a quarter of it;
    - ``min_inlier_frac``: smallest fraction of a camera's observations kept by the fit; warn
      below 0.5, fail below 0.1;
    - ``scale_rejected``: a requested scale was refused (fail).
    """
    from ..init import RecipeReport, make_intrinsics_group, select
    from ..init import _refine as _legacy_refine
    timer = Timer()
    opts = merge_options(W2_DEFAULTS, options)
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
    if p.ndim == 5 and init != 'people':
        raise ValueError("C x F x P x J x 2 points need init='people'")
    base = make_intrinsics_group(opts['cam_names'], opts['sizes'], intrinsics, fisheye=opts['fisheye'])
    names = base.get_names()
    if p.shape[0] != len(names):
        raise ValueError('points have {} cameras, intrinsics {}'.format(p.shape[0], len(names)))
    gr = _parse_groups(groups, p.shape[1:-1] if p.ndim < 5 else p.shape[1:2])
    diag = _diag(base, p) or 1000.0
    settings['image_diag_px'] = diag
    # ---- candidates + selection
    with timer('init'):
        cands = _candidates(base, p, init, frames, gr, opts, rng, rpt, names)
        if p.ndim == 5 and 'people' in cands:
            from ..init import people
            assoc, _ = people.associate_people(base, p, **(opts['people_kwargs'] or {}).get('association_kwargs', {}) or {})
            C_, F_, G_, J_, _ = assoc.shape
            p = assoc.reshape(C_, F_ * G_, J_, 2)
            gr = dict(frame=np.arange(F_ * G_), window=np.repeat(np.arange(F_), G_))
    rpt.candidates = list(cands)
    res = PointCalibration(init_report=rpt)
    if not cands:
        res.verdict = 'fail'
        res.reasons = ['fail: no initialisation succeeded: {}'.format(rpt.errors)]
        res.report = dict(errors=rpt.errors)
        res.settings = settings
        res.provenance = provenance('calibrate_from_points', inputs=dict(n_cameras=len(names), shape=list(p.shape)),
                                    fallbacks=fallbacks, timings=timer.times)
        _save(res, out)
        return res
    complete = {k: v for k, v in cands.items() if set(v.get_names()) == set(names)}
    pool = complete if complete else cands
    with timer('select'):
        if len(pool) == 1:
            key = next(iter(pool))
        else:
            common = [n for n in names if all(n in v.get_names() for v in pool.values())]
            idx = [names.index(n) for n in common]
            skw = dict(rng=rng)
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
    cg = chosen.to(dev) if dev != 'cpu' else chosen
    # ---- refinement
    nf = MIN_NOISE_PX
    ba = None
    with timer('bundle_adjust'):
        method = opts['refine']
        if method in ('robust', 'auto'):
            ba, rpt.refine, st = _robust_ba(cg, pc, gr, diag, opts, rng)
            settings.update(st)
            nf = st['noise_floor_px']
        elif method:
            cg, rpt.refine = _legacy_refine(cg, pc, gr.get('frame'), method, rng, opts['ba_kwargs'])
            nf, _ = _points_noise(cg, pc, diag)
            settings['noise_floor_px'] = nf
        else:
            rpt.refine = dict(method=None)
            nf, _ = _points_noise(cg, pc, diag)
            settings['noise_floor_px'] = nf
    u = _noise_unit(nf)
    # ---- intrinsics (evidence gated)
    with timer('intrinsics'):
        if refine_intrinsics in (False, None):
            res.intrinsics = dict(decision='kept', reason='refine_intrinsics=False')
        elif refine_intrinsics in ('auto', True):
            cg, res.intrinsics = _gate_intrinsics(cg, pc, gr, refine_intrinsics, nf, opts, rng)
        else:
            raise ValueError("refine_intrinsics must be 'auto', True or False")
    # ---- stations
    if stations is not None:
        with timer('stations'):
            res.report['stations'] = _stations_ba(cg, pc, gr, stations, nf, rng)
    if dev != 'cpu':
        cg = cg.to('cpu')
    # ---- scale
    scale_rejected = False
    if scale is not None:
        with timer('scale'):
            if isinstance(scale, (int, float)) and not isinstance(scale, bool):
                from .. import scale as S
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
    fit_mask = None
    if ba is not None and 'fit' in ba.excluded:
        fit_mask = ~np.asarray(ba.excluded['fit'], bool).reshape(len(present), -1)
    with timer('quality'):
        if opts['quality']:
            from .. import quality as Q
            try:
                rpt.quality = res.quality = Q.report(cg, pc, groups=opts['quality_groups'], tau=10.0 * u,
                                                     tau_tri=5.0 * u, max_loo_px=60.0 * u,
                                                     thresholds=tuple(t * u for t in (5, 8, 10)),
                                                     fit_mask=fit_mask)
            except Exception as exc:  # pragma: no cover
                rpt.errors['quality'] = '{}: {}'.format(type(exc).__name__, exc)
    # ---- verdict
    flat = _flat(pc)
    valid = np.isfinite(flat).all(-1)
    per_cam = {}
    loo_worst = None
    if res.quality is not None and res.quality.loo is not None:
        le = np.asarray(res.quality.loo.errors, float)
        use = res.quality.loo.eligible & (fit_mask if fit_mask is not None else True)
        meds = []
        for c, n in enumerate(present):
            e = le[c][use[c] & np.isfinite(le[c])]
            m = float(np.median(e)) if e.size else None
            per_cam[n] = dict(loo_fit_median_px=m)
            if m is not None:
                meds.append(m)
        loo_worst = float(np.max(meds)) if meds else None
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
    from .. import scale as S
    cen = S.camera_centers(cg)
    d = np.linalg.norm(cen - np.median(cen, 0), axis=1)
    md = np.median(d) if len(d) else 0.0
    runaway = [n for n, v in zip(present, d) if md > 0 and v > opts['runaway_ratio'] * md]
    worst_loo_cam = None
    if per_cam:
        vals = [(v.get('loo_fit_median_px') or -1, n) for n, v in per_cam.items()]
        worst_loo_cam = max(vals)[1]
    mf = opts['min_inlier_frames']
    rules = [
        Rule('loo_worst_px', loo_worst, 3 * nf, 10 * nf, 'worst per-camera LOO median ({})'.format(worst_loo_cam)),
        Rule('unplaced_cameras', len(unplaced), None, 0, 'not placed: {}'.format(unplaced) if unplaced else ''),
        Rule('runaway_cameras', len(runaway), None, 0, 'runaway: {}'.format(runaway) if runaway else ''),
        Rule('min_inlier_frames', min_frames, mf, mf / 4.0, 'fewest frames with inliers', higher_is_worse=False),
        Rule('min_inlier_frac', min_frac, 0.5, 0.1, 'smallest kept fraction ({})'.format(worst_frac_cam),
             higher_is_worse=False),
        Rule('scale_rejected', scale_rejected, None, True,
             'scale refused: {}'.format(getattr(res.scale, 'reasons', '')) if scale_rejected else ''),
    ]
    verdict, reasons, table = _rules(rules, opts)
    settings['loo_warn_px'], settings['loo_fail_px'] = 3 * nf, 10 * nf
    res.report.update(per_camera=per_cam, unplaced=unplaced, runaway=runaway, rules=table, noise_floor_px=nf,
                      init=key, candidates=list(cands), errors=dict(rpt.errors),
                      overall=None if res.quality is None else res.quality.overall)
    res.cgroup, res.selected_init = cg, key
    res.verdict, res.reasons, res.settings = verdict, reasons, settings
    res.provenance = provenance('calibrate_from_points',
                                inputs=dict(n_cameras=len(names), cam_names=names, shape=list(p.shape),
                                            init=init if isinstance(init, str) else 'given',
                                            frames=frames is not None, stations=stations is not None),
                                fallbacks=fallbacks + ['init {} failed: {}'.format(k, v) for k, v in rpt.errors.items()],
                                timings=timer.times)
    cg.metadata['calibration'] = dict(workflow='calibrate_from_points', verdict=verdict, init=str(key),
                                      noise_floor_px=float(nf))
    _save(res, out)
    return res


def _stations_ba(cg, pc, gr, stations, nf, rng):
    """Final extrinsics LM with one pose per station (``groups['time']`` per frame)."""
    from .. import bundle as B
    if 'time' not in gr:
        raise ValueError("stations= needs groups['time'] (frame times, s)")
    u = _noise_unit(nf)
    _, _, inl = _robust_points(cg, pc, 10 * u)
    flat = _flat(pc)
    t = np.asarray(gr['time'], float)
    if pc.ndim == 4 and t.ndim == 1:
        t = np.broadcast_to(t[:, None], pc.shape[1:3])
    t = t.reshape(-1)
    keep = inl & (inl.sum(0) >= 2)[None]
    ci, pj = np.nonzero(keep)
    obs = B.Observations(ci, pj, flat[ci, pj], time=t[pj], n_points=flat.shape[1], n_cams=flat.shape[0])
    r = B.bundle_adjust_lm(cg, obs, stations=stations, only_extrinsics=True, loss='cauchy',
                           f_scale=3 * max(nf, MIN_NOISE_PX), return_report=True, rng=rng)
    return dict(error_px=r.error, error_before_px=r.error_before, stations=r.stations)


# ================================================================== W3 estimate_scale

W3_DEFAULTS = dict(
    board_kwargs=None,        # passed to scale.from_board (relative gates: frame MAD, spread)
    lengths_kwargs=None,      # passed to scale.from_lengths (tau defaults to 10 noise units)
    reference_kwargs=None,    # passed to scale.from_reference
    ground_points=None, up='z', origin='centroid',   # align='ground'
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


def estimate_scale(cgroup, board=None, board_points=None, lengths=None, reference=None, align='auto',
                   apply=True, options=None, *, object=None, object_points=None):
    """W3: metric scale (and optionally the world frame) of a calibrated rig.

    Exactly one source:

    - ``board`` + ``board_points`` (rows or ``[C,F,K,2]`` corners; also ``object=`` +
      ``object_points=`` for a :class:`~aniposelib.boards.PointObject`): ``scale.from_board``;
    - ``lengths=dict(points=[C,N,J,2], pairs=[(a, b), ...], ref=[...], joints=None)``:
      ``scale.from_lengths`` (MSAC threshold 10 noise units);
    - ``reference`` (a ``CameraGroup`` of the same cameras): ``scale.from_reference``.

    ``align``: ``'auto'`` (= ``'reference'`` when a reference is given, else None),
    ``'reference'`` (similarity-align the gauge onto the reference: ``scale.align_to``),
    ``'ground'`` (``scale.align_world`` after scaling; ``options`` ``ground_points`` / ``up`` /
    ``origin``) or None. ``apply``: return a rescaled copy in ``result.cgroup`` (units, source and
    quality in its metadata). The input group is never modified.

    Verdict rules: ``scale_rejected`` (the backend refused the estimate: too few frames / pairs,
    frame MAD above its relative gate; fail); ``scale_flags`` (number of quality flags such as
    ``board_scale_spread`` / ``lengths_spread`` / ``scale_reference_mismatch``; warn).
    """
    from .. import scale as S
    opts = merge_options(W3_DEFAULTS, options)
    timer = Timer()
    if object is not None:
        if board is not None:
            raise ValueError('give board= or object=, not both')
        board, board_points = object, object_points if object_points is not None else board_points
    sources = [n for n, v in (('board', board), ('lengths', lengths), ('reference', reference)) if v is not None]
    if len(sources) != 1:
        raise ValueError('estimate_scale needs exactly one of board=, lengths=, reference= (got {})'.format(
            sources or 'none'))
    src = sources[0]
    if align == 'auto':
        align = 'reference' if src == 'reference' else None
    if align not in (None, 'reference', 'ground'):
        raise ValueError("align must be 'auto', 'reference', 'ground' or None")
    if align == 'reference' and src != 'reference':
        raise ValueError("align='reference' needs reference=")
    settings = dict(source=src, align=align)
    with timer('estimate'):
        if src == 'board':
            if board_points is None:
                raise ValueError('board= needs board_points= (rows or corners)')
            if hasattr(board, 'rows_from_array') and not isinstance(board_points, (list, tuple)):
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
            kw = dict(tau=10.0 * _noise_unit(nf))
            kw.update(opts['lengths_kwargs'] or {})
            settings.update(noise_floor_px=nf, tau_px=kw['tau'])
            r = S.from_lengths(cgroup, pts, lengths['pairs'], ref, joints=lengths.get('joints'),
                               **{k: v for k, v in kw.items()})
        else:
            r = S.from_reference(cgroup, reference, **(opts['reference_kwargs'] or {}))
    details = r.to_dict()
    out = None
    if apply and r.accepted and r.scale is not None:
        with timer('apply'):
            out = cgroup.copy()
            if align == 'reference':
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
            if align == 'ground':
                al = S.align_world(out, opts['ground_points'], up=opts['up'], origin=opts['origin'])
                details['align'] = al.to_dict()
    rules = [Rule('scale_rejected', not r.accepted, None, True,
                  'scale refused: {}'.format(r.reason) if not r.accepted else ''),
             Rule('scale_flags', len(r.flags), 0, None, 'flags: {}'.format(r.flags) if r.flags else '')]
    verdict, reasons, table = _rules(rules, opts)
    return ScaleEstimate(report=dict(rules=table, stats=r.stats, flags=list(r.flags), reason=r.reason,
                                     units=r.units),
                         verdict=verdict, reasons=reasons, settings=settings,
                         provenance=provenance('estimate_scale', inputs=dict(source=src, cameras=cgroup.get_names()),
                                               timings=timer.times),
                         scale=r.scale, sigma=_scale_sigma(r), source=r.source, accepted=bool(r.accepted),
                         details=details, cgroup=out)


# ================================================================== W4 evaluate_calibration

W4_DEFAULTS = dict(
    min_others=3,             # LOO: other cameras that must see a point
    detect_moves_kwargs=None, # passed to stations.detect_moves (tau / huber scaled by the noise unit)
    compare_kwargs=None,      # passed to quality.compare
    min_ray_deg=2.0,          # ray-angle degeneracy threshold
    verdict=None,
)


@dataclass(repr=False)
class CalibrationEvaluation(WorkflowResult):
    """Result of :func:`evaluate_calibration`: ``per_camera`` (LOO median / coverage / agreement,
    reprojection median, inlier rate, lens validity, area coverage, flags), ``moves``
    (:class:`aniposelib.stations.MoveReport` or None), ``comparison``
    (:class:`aniposelib.quality.CompareResult` or None), ``quality`` (the full
    :class:`aniposelib.quality.CalibrationReport`)."""
    per_camera: Dict[str, Dict[str, Any]] = field(default_factory=dict)
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


def evaluate_calibration(cgroup, points, times=None, reference=None, board=None, options=None, *,
                         board_points=None, object=None, object_points=None):
    """W4: how good is this calibration on these observations?

    Parameters
    ----------
    cgroup : the calibration to evaluate.
    points : ``C x N x 2`` / ``C x N x J x 2`` pixels in ``cgroup`` order (NaN missing).
    times : ``[N]`` frame times (s): per-window moved-camera check (``stations.detect_moves``).
    reference : a ``CameraGroup`` to compare with (``quality.compare``: aligned centre residuals,
        rotation / focal deltas, paired LOO).
    board, board_points : a board and its detections (rows or corners) -> board shape residual
        (``object=`` / ``object_points=`` for a ``PointObject``).

    Contents: LOO per camera (fixed denominator), coverage, inlier rates, lens validity, ray
    angles, plus moves / comparison / board shape when their inputs are given. Project policy
    (e.g. a reuse ladder) belongs downstream, on top of ``per_camera``.

    Verdict rules (``nf`` = noise floor of the observations under this calibration):

    - ``loo_worst_px``: worst per-camera LOO median; warn above 3 nf, fail above 10 nf;
    - ``loo_coverage_min``: smallest per-camera LOO coverage; warn below 0.5, fail below 0.2;
    - ``lens_invalid``: cameras whose lens model is not invertible over the image (warn);
    - ``grazing_frac``: fraction of points whose largest ray angle is below ``min_ray_deg``
      (warn above 0.5);
    - ``moved_cameras``: cameras with a detected bump / re-aim (fail);
    - ``center_residual_frac``: worst aligned centre residual vs the reference / rig scale;
      warn 1 %, fail 5 %;
    - ``board_shape_rel``: median board shape residual / object size; warn 0.5 %, fail 2 %.
    """
    from .. import quality as Q
    opts = merge_options(W4_DEFAULTS, options)
    timer = Timer()
    if object is not None:
        board, board_points = object, object_points if object_points is not None else board_points
    p = np.asarray(to_numpy(points), dtype='float64')
    names = cgroup.get_names()
    if p.ndim not in (3, 4) or p.shape[0] != len(names) or p.shape[-1] != 2:
        raise ValueError('points must be C x N x 2 or C x N x J x 2 with C={}, got {}'.format(len(names), p.shape))
    if board is not None and board_points is None:
        raise ValueError('board= needs board_points= (rows or corners)')
    diag = _diag(cgroup, p) or 1000.0
    with timer('noise'):
        nf, _ = _points_noise(cgroup, p, diag)
    u = _noise_unit(nf)
    settings = dict(image_diag_px=diag, noise_floor_px=nf, noise_unit=u, tau_px=10 * u, tau_tri_px=5 * u,
                    loo_warn_px=3 * nf, loo_fail_px=10 * nf)
    with timer('report'):
        qr = Q.report(cgroup, p, tau=10.0 * u, tau_tri=5.0 * u, max_loo_px=60.0 * u,
                      thresholds=tuple(t * u for t in (5, 8, 10)), min_others=opts['min_others'])
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
        cov = Q.coverage(p, min_cams=opts['min_others'], cgroup=cgroup)
        for n in names:
            per[n]['area'] = cov['per_camera'][n].get('area')
    with timer('rays'):
        X, _, inl = _robust_points(cgroup, p, 10.0 * u)
        X[inl.sum(0) < 2] = np.nan
        ok = np.isfinite(X).all(1)
        grazing = None
        if ok.any():
            ra = Q.ray_angles(cgroup, X[ok], _flat(p)[:, ok], min_deg=opts['min_ray_deg'])
            grazing = float(ra['frac_points_below'])
    report = dict(overall=qr.overall, coverage=cov, rules=None)
    moves = comparison = None
    moved, center_frac, shape_rel = [], None, None
    if times is not None:
        with timer('moves'):
            from ..stations import detect_moves
            kw = dict(tau=10.0 * u, huber_px=3.0 * u)
            kw.update(opts['detect_moves_kwargs'] or {})
            moves = detect_moves(cgroup, p, times, **kw)
            moved = list(moves.moved)
            settings['detect_moves'] = {k: v for k, v in kw.items() if isinstance(v, (int, float))}
            for n in names:
                per[n]['moved'] = n in moved
    if reference is not None:
        with timer('compare'):
            try:
                comparison = Q.compare(cgroup, reference, **(opts['compare_kwargs'] or {}))
                center_frac = comparison.overall.get('center_residual_max_frac')
                for n, d in comparison.per_camera.items():
                    per[n]['reference'] = d
            except ValueError as exc:
                report['compare_error'] = str(exc)
    if board is not None:
        with timer('board'):
            bs = Q.board_shape_residual(cgroup, board_points, board)
            report['board_shape'] = bs
            objp = np.asarray(board.get_object_points(), float).reshape(-1, 3)
            size = float(np.max(np.linalg.norm(objp - objp.mean(0), axis=1)) * 2)
            if bs.get('median') is not None and size > 0:
                shape_rel = bs['median'] / size
    loo = np.array([np.nan if per[n].get('loo_median_px') is None else per[n]['loo_median_px'] for n in names])
    covs = np.array([np.nan if per[n].get('loo_coverage') is None else per[n]['loo_coverage'] for n in names])
    worst_cam = names[int(np.nanargmax(loo))] if np.isfinite(loo).any() else None
    loo_worst = float(np.nanmax(loo)) if np.isfinite(loo).any() else float('nan')
    low_cov = names[int(np.nanargmin(covs))] if np.isfinite(covs).any() else None
    rules = [
        Rule('loo_worst_px', loo_worst, 3 * nf, 10 * nf, 'worst per-camera LOO median ({})'.format(worst_cam)),
        Rule('loo_coverage_min', float(np.nanmin(covs)) if np.isfinite(covs).any() else None, 0.5, 0.2,
             'smallest LOO coverage ({})'.format(low_cov), higher_is_worse=False),
        Rule('lens_invalid', n_lens_bad, 0, None, 'lens model not invertible over the image'),
        Rule('grazing_frac', grazing, 0.5, None, 'points with all ray angles < {} deg'.format(opts['min_ray_deg'])),
        Rule('moved_cameras', len(moved) if times is not None else None, None, 0,
             'moved: {}'.format(moved) if moved else ''),
        Rule('center_residual_frac', center_frac, 0.01, 0.05, 'aligned centre residual vs reference / rig scale'),
        Rule('board_shape_rel', shape_rel, 0.005, 0.02, 'board shape residual / object size'),
    ]
    verdict, reasons, table = _rules(rules, opts)
    report['rules'] = table
    report['per_camera_flags'] = {n: per[n].get('flags', []) for n in names}
    return CalibrationEvaluation(report=report, verdict=verdict, reasons=reasons, settings=settings,
                                 provenance=provenance('evaluate_calibration',
                                                       inputs=dict(cameras=names, shape=list(p.shape),
                                                                   times=times is not None,
                                                                   reference=reference is not None,
                                                                   board=board is not None),
                                                       timings=timer.times),
                                 per_camera=per, moves=moves, comparison=comparison, quality=qr)
