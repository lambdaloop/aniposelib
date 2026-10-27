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
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from ._common import (MIN_NOISE_PX, NOISE_DIAG_FRAC, Rule, Timer, WorkflowResult, evaluate_rules,
                      image_diagonal, merge_options, noise_floor, provenance, save_calibration,
                      to_numpy, worst)

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
    return evaluate_rules(rules, opts.get('verdict'))


def _cam_list(names, values, pred):
    return [n for n, v in zip(names, values) if pred(v)]


def _jsonable_list(x):
    return [str(v) for v in x]


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
                                                 fallback='keep' if init is not None else 'nominal')
                cam.set_n_dist(min(opts['min_n_dist'], to_numpy(cam.get_distortions()).shape[0]))
                lens[n] = dict(source='seed:' + seed.source, reason=seed.reason, ladder_error=err)
                fallbacks.append('{}: ladder failed ({}), {} intrinsics'.format(n, err, seed.source))
                if seed.source == 'fallback:nominal':
                    fallbacks.append('{}: nominal intrinsics'.format(n))
                continue
            if init is not None:
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
            elif init is not None:
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
