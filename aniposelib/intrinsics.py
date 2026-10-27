"""Single-camera intrinsics from planar board views, with held-out model selection.

:func:`calibrate_ladder` fits a ladder of lens models of increasing complexity on training
views (each model initialised from the simpler fit below it), scores every model on held-out
views (board poses re-fitted with the intrinsics frozen), and selects the most parsimonious
model whose held-out error is within ``tol`` of the best. This avoids the usual failure of
picking the richest model by its training error. Ported from the disco-dancing pipeline
(``board_intrinsics_ba.py``, ``all_sessions_layout_fix_calibration.py``).

Models (square pixels, ``fx = fy = f``):

====================  =====================================================
``pinhole``           f, cx, cy
``k1k2``              + k1, k2
``k1k2p``             + k1, k2, p1, p2
``standard_5param``   + k1, k2, p1, p2, k3
``rational_8param``   + k1, k2, p1, p2, k3, k4, k5, k6 (no aniposelib camera)
``fisheye_k1k2``      f, cx, cy, k1, k2 (OpenCV equidistant fisheye)
``fisheye``           f, cx, cy, k1..k4
====================  =====================================================

Distortion vectors use OpenCV order. The solver (:func:`lm_board_adjust`) is the shared
Levenberg-Marquardt engine (:mod:`aniposelib._lm`) with a Schur complement on the per-view pose
blocks and analytic Jacobians, fast for a few hundred views.
"""

from dataclasses import dataclass, field

import cv2
import numpy as np

__all__ = ['BROWN_MODELS', 'FISHEYE_MODELS', 'MODELS', 'ModelFit', 'LadderResult',
           'calibrate_ladder', 'calibrate_ladder_rows', 'block_holdout', 'select_model']

BROWN_MODELS = ('pinhole', 'k1k2', 'k1k2p', 'standard_5param', 'rational_8param')
FISHEYE_MODELS = ('fisheye_k1k2', 'fisheye')
MODELS = BROWN_MODELS + FISHEYE_MODELS
DEFAULT_MODELS = ('pinhole', 'k1k2', 'k1k2p', 'standard_5param')

_FISHEYE_FREE = {'fisheye_k1k2': [0, 1], 'fisheye': [0, 1, 2, 3]}
_BROWN_FREE = {
    'pinhole': [],
    'k1k2': [0, 1],
    'k1k2p': [0, 1, 2, 3],
    'standard_5param': [0, 1, 2, 3, 4],
    'rational_8param': [0, 1, 2, 3, 4, 5, 6, 7],
}
NESTED_PARENT = {'k1k2': 'pinhole', 'k1k2p': 'k1k2', 'standard_5param': 'k1k2p',
                 'rational_8param': 'standard_5param', 'fisheye_k1k2': 'pinhole',
                 'fisheye': 'fisheye_k1k2'}


def is_fisheye(model):
    return model in _FISHEYE_FREE


def n_free_params(model):
    if model not in MODELS:
        raise ValueError('unknown model {!r}; choose from {}'.format(model, MODELS))
    return 3 + len(_FISHEYE_FREE[model] if is_fisheye(model) else _BROWN_FREE[model])


def unpack_intrinsics(model, intr):
    f, cx, cy = intr[:3]
    K = np.array([[f, 0.0, cx], [0.0, f, cy], [0.0, 0.0, 1.0]])
    if is_fisheye(model):
        d4 = np.zeros(4)
        d4[_FISHEYE_FREE[model]] = intr[3:]
        return K, d4
    full = np.zeros(8)
    full[_BROWN_FREE[model]] = intr[3:]
    return K, full[:8 if model == 'rational_8param' else 5]


def pack_intrinsics(model, K, dist=None):
    f = 0.5 * (float(K[0, 0]) + float(K[1, 1]))
    head = [f, float(K[0, 2]), float(K[1, 2])]
    dist = np.zeros(8) if dist is None else np.asarray(dist, dtype=np.float64).reshape(-1)
    if is_fisheye(model):
        d4 = dist[:4] if len(dist) == 4 else np.zeros(4)
        return np.array(head + [d4[i] for i in _FISHEYE_FREE[model]])
    full = np.zeros(8)
    full[:min(8, len(dist))] = dist[:8]
    return np.array(head + [full[i] for i in _BROWN_FREE[model]])


def _kernel(model, intr, poses, obj, view, jac=False):
    """Board corners ``obj [m,3]`` of views ``view [m]`` (poses ``[V,6]``, intrinsics vector
    ``intr`` of ``model``) -> pixels ``[m,2]`` (torch), with the :mod:`aniposelib.bundle` kernel;
    with ``jac`` also the Jacobians on the pose ``[m,2,6]`` and on ``intr`` ``[m,2,len(intr)]``."""
    import torch
    from . import bundle as B
    m, fish = obj.shape[0], is_fisheye(model)
    free = [11 + j for j in (_FISHEYE_FREE[model] if fish else _BROWN_FREE[model])]
    L = torch.zeros(m, B.N_LOCAL, dtype=torch.float64)
    L[:, :6] = poses[view]
    L[:, 6] = L[:, 7] = intr[0]                    # square pixels: f moves fx and fy
    L[:, 8], L[:, 9] = intr[1], intr[2]
    L[:, free] = intr[3:]
    args = (torch.zeros(m, dtype=torch.float64), torch.zeros(m, 2, dtype=torch.float64),
            torch.full((m,), fish), not fish, fish)
    if not jac:
        return B._project_local(L, obj, *args)[0]
    uv, _, dL, _ = B._project_local_jac(L, obj, *args)
    Ji = torch.stack([dL[:, :, 6] + dL[:, :, 7], dL[:, :, 8], dL[:, :, 9]] + [dL[:, :, s] for s in free], -1)
    return uv, dL[:, :, :6], Ji


def _tt(a):
    import torch
    return torch.as_tensor(np.ascontiguousarray(a, dtype=np.float64))


def _project_views(model, intr, poses, obj, view):
    import torch
    return _kernel(model, _tt(intr), _tt(poses), _tt(np.asarray(obj).reshape(-1, 3)),
                   torch.as_tensor(np.asarray(view))).numpy()


def _stack(objpoints, imgpoints, indices):
    obj = np.concatenate([np.asarray(objpoints[i], dtype=np.float64).reshape(-1, 3) for i in indices])
    img = np.concatenate([np.asarray(imgpoints[i], dtype=np.float64).reshape(-1, 2) for i in indices])
    view = np.concatenate([np.full(len(np.asarray(objpoints[i]).reshape(-1, 3)), k)
                           for k, i in enumerate(indices)])
    return obj, img, view


def _undistort(model, intr, img):
    K, dist = unpack_intrinsics(model, intr)
    pts = np.asarray(img, dtype=np.float64).reshape(-1, 1, 2)
    if is_fisheye(model):
        return cv2.fisheye.undistortPoints(pts, K, dist.reshape(4, 1)).reshape(-1, 2)
    return cv2.undistortPoints(pts, K, dist).reshape(-1, 2)


def _init_pose(model, intr, obj, img):
    """Planar PnP (IPPE, then iterative) on undistorted normalised points."""
    und = _undistort(model, intr, img)
    obj = np.asarray(obj, dtype=np.float64).reshape(-1, 3)
    for flags in (cv2.SOLVEPNP_IPPE, cv2.SOLVEPNP_ITERATIVE):
        try:
            ok, rvec, tvec = cv2.solvePnP(obj, und, np.eye(3), None, flags=flags)
        except cv2.error:
            continue
        if ok:
            return np.concatenate([np.asarray(rvec).reshape(3), np.asarray(tvec).reshape(3)])
    raise ValueError('PnP failed on a view')


def lm_board_adjust(model, intr0, poses0, obj, img, view, fix_intr=False, max_iter=100, tol=1e-10):
    """Levenberg-Marquardt over intrinsics + per-view board poses (Schur on the pose blocks), on
    the shared engine (:mod:`aniposelib._lm`) with the analytic projection Jacobians of
    :mod:`aniposelib.bundle` (linear loss).

    ``view`` must be contiguous, 0..n_views-1. Returns ``(intr, poses, info)``."""
    import torch
    from . import _lm
    poses = np.asarray(poses0, dtype=np.float64)
    n_views = len(poses)
    view = np.asarray(view)
    if np.any(np.diff(view) < 0) or len(np.unique(view)) != n_views:
        raise ValueError('points must be grouped by view (0..n_views-1, contiguous)')
    obj_t, img_t, view_t = _tt(np.asarray(obj).reshape(-1, 3)), _tt(np.asarray(img).reshape(-1, 2)), torch.as_tensor(view)
    m = len(view)
    n_i = 0 if fix_intr else len(intr0)

    class _P:
        NC = n_i

        def cost(self, st):
            return float(((_kernel(model, st[0], st[1], obj_t, view_t) - img_t) ** 2).sum())

        def linearize(self, st):
            uv, Jp, Ji = _kernel(model, st[0], st[1], obj_t, view_t, jac=True)
            return [_lm.Blocks(r=uv - img_t, w=torch.ones(m, dtype=torch.float64), Js=Jp, sidx=view_t, n=n_views,
                               D=6, Jc=Ji if n_i else None, cols=torch.arange(n_i).expand(m, n_i) if n_i else None)]

        def apply(self, st, dth, dv):
            return (st[0] + dth if n_i else st[0], st[1] + dv[0])

    res = _lm.levenberg_marquardt(_P(), (_tt(intr0), _tt(poses)), n_iter=max_iter, tol=tol, lam_up=10.0, max_tries=12)
    intr, poses = (s.numpy() for s in res.state)
    status = {'no_decrease': 'converged_no_decrease', 'tol': 'converged'}.get(res.reason, 'max_iter')
    return intr, poses, {'status': status, 'iterations': len(res.history) - 1, 'cost': res.cost,
                         'rms_px': float(np.sqrt(res.cost / max(len(view), 1)))}


def _initial_pinhole(objpoints, imgpoints, image_size):
    w, h = image_size
    f = None
    try:
        K0 = cv2.initCameraMatrix2D([np.asarray(o, np.float32).reshape(-1, 3) for o in objpoints],
                                    [np.asarray(i, np.float32).reshape(-1, 2) for i in imgpoints],
                                    tuple(int(s) for s in image_size), 1.0)
        f = float(K0[0, 0])
    except cv2.error:
        pass
    if f is None or not np.isfinite(f) or f <= 0:
        f = 0.85 * w
    return np.array([f, w / 2.0, h / 2.0])


def _stats(res):
    if res is None or not len(res):
        return dict(rms=None, median=None, p95=None)
    return dict(rms=float(np.sqrt(np.mean(res ** 2))), median=float(np.median(res)),
                p95=float(np.percentile(res, 95)))


@dataclass
class ModelFit:
    """One rung of the ladder. Errors are per-corner pixel distances."""
    model: str
    n_params: int
    K: np.ndarray
    dist: np.ndarray
    train: dict
    heldout: dict
    status: str
    iterations: int
    pnp_failures: int = 0

    @property
    def fisheye(self):
        return is_fisheye(self.model)

    def to_camera(self, size=None, name=None):
        """An aniposelib ``Camera`` / ``FisheyeCamera`` with these intrinsics.

        ``n_dist`` is the number of coefficients this model fits, so later bundle adjustment
        keeps the same model."""
        from .cameras import Camera, FisheyeCamera
        n_dist = self.n_params - 3
        if self.fisheye:
            return FisheyeCamera(matrix=self.K, dist=self.dist[:4], size=size, name=name,
                                 n_dist=n_dist)
        if self.model == 'rational_8param':
            raise ValueError('aniposelib cameras do not support the rational model')
        return Camera(matrix=self.K, dist=self.dist[:5], size=size, name=name, n_dist=n_dist)

    def to_dict(self):
        return dict(model=self.model, n_params=self.n_params, K=np.asarray(self.K).tolist(),
                    dist=np.asarray(self.dist).tolist(), train=dict(self.train),
                    heldout=dict(self.heldout), status=self.status,
                    iterations=self.iterations, pnp_failures=self.pnp_failures)


@dataclass
class LadderResult:
    """Output of :func:`calibrate_ladder`: every fitted model and the selected one."""
    fits: dict
    selected: str
    rule: str
    train_idx: list
    heldout_idx: list
    image_size: tuple = None
    extra: dict = field(default_factory=dict)

    @property
    def best(self):
        return self.fits[self.selected]

    def to_camera(self, name=None):
        return self.best.to_camera(size=self.image_size, name=name)

    def table(self):
        """Rows ``(model, n_params, train_rms, heldout_rms, heldout_p95)`` for printing."""
        return [(m, f.n_params, f.train['rms'], f.heldout['rms'], f.heldout['p95'])
                for m, f in self.fits.items()]

    def to_dict(self):
        return dict(fits={m: f.to_dict() for m, f in self.fits.items()}, selected=self.selected,
                    rule=self.rule, train_idx=[int(i) for i in self.train_idx],
                    heldout_idx=[int(i) for i in self.heldout_idx],
                    image_size=None if self.image_size is None else list(self.image_size),
                    extra=dict(self.extra))


def block_holdout(times, block=None, every=5):
    """Train/held-out view indices by contiguous time blocks.

    Views are grouped into blocks of ``block`` time units (a new block also starts after a
    gap > ``block``); every ``every``-th block (offset 2, so held-out blocks are interior) is
    held out, so near-duplicate neighbouring frames never straddle train and held-out sets.
    ``block=None`` makes ~``5 * every`` blocks."""
    times = np.asarray(times, dtype=np.float64)
    n = len(times)
    if n == 0:
        return [], []
    if block is None:
        span = float(np.ptp(times)) if n > 1 else 1.0
        block = max(span / (5 * every), 1e-12)
    order = np.argsort(times, kind='stable')
    blocks = np.zeros(n, dtype=int)
    cur, start = 0, times[order[0]]
    for rank, i in enumerate(order):
        if rank and times[i] - start > block:
            cur += 1
            start = times[i]
        blocks[i] = cur
    held_blocks = {b for b in range(cur + 1) if b % every == (2 % every)}
    if not held_blocks and cur >= 1:
        held_blocks = {cur}
    held = [i for i in range(n) if blocks[i] in held_blocks]
    train = [i for i in range(n) if blocks[i] not in held_blocks]
    return train, held


def select_model(fits, tol=0.15, plausible=None, metric='p95'):
    """Pick a model: held-out ``metric`` within ``(1 + tol)`` of the best, prefer
    ``plausible(fit)`` ones, then fewest parameters. Returns ``(name, rule)``."""
    pool = {m: f for m, f in fits.items()
            if f.status != 'failed' and f.heldout.get(metric) is not None
            and np.isfinite(f.heldout[metric])}
    if not pool:
        raise ValueError('no model could be scored on the held-out views')
    best = min(f.heldout[metric] for f in pool.values())
    cands = {m: f for m, f in pool.items() if f.heldout[metric] <= best * (1 + tol)}

    def parsimonious(d):
        return min(d, key=lambda m: (d[m].n_params, d[m].heldout[metric]))

    rule = 'held-out {} within {:.0%} of the best, then fewest parameters'.format(metric, tol)
    if plausible is not None:
        good = {m: f for m, f in cands.items() if plausible(f)}
        if good:
            return parsimonious(good), rule + ' (plausible models only)'
        return parsimonious(cands), rule + ' (no plausible candidate: flagged)'
    return parsimonious(cands), rule


def calibrate_ladder(objpoints, imgpoints, image_size, models=DEFAULT_MODELS, *,
                     train_idx=None, heldout_idx=None, times=None, block=None, every=5,
                     focal_inits=None, tol=0.15, plausible=None, max_iter=300):
    """Fit a ladder of lens models on board views and select one on held-out views.

    Parameters
    ----------
    objpoints, imgpoints : lists of (N_i, 3) / (N_i, 2) arrays
        One entry per board view (planar object points, e.g. from
        ``board.get_all_calibration_points(rows)``).
    image_size : (width, height)
    models : sequence of model names (see the module docstring); ``'pinhole'`` is always fit.
    train_idx, heldout_idx : view indices, optional
        Default: :func:`block_holdout` on ``times`` (default: view order), every ``every``-th
        block held out.
    focal_inits : extra pinhole focal-length starts (px); the lowest training cost wins.
    tol, plausible : see :func:`select_model`.

    Returns a :class:`LadderResult`; ``result.to_camera()`` gives an aniposelib camera.
    """
    n = len(objpoints)
    if len(imgpoints) != n:
        raise ValueError('objpoints and imgpoints differ in length')
    for m in models:
        n_free_params(m)
    if train_idx is None or heldout_idx is None:
        t = np.arange(n) if times is None else np.asarray(times)
        tr, ho = block_holdout(t, block=block, every=every)
        train_idx = tr if train_idx is None else list(train_idx)
        heldout_idx = ho if heldout_idx is None else list(heldout_idx)
    train_idx, heldout_idx = list(train_idx), list(heldout_idx)
    if len(train_idx) < 3:
        raise ValueError('need at least 3 training views, got {}'.format(len(train_idx)))
    if len(heldout_idx) < 1:
        raise ValueError('need at least 1 held-out view')

    obj, img, view = _stack(objpoints, imgpoints, train_idx)
    starts = [_initial_pinhole([objpoints[i] for i in train_idx],
                               [imgpoints[i] for i in train_idx], image_size)]
    for f0 in (focal_inits or ()):
        starts.append(np.array([float(f0), image_size[0] / 2.0, image_size[1] / 2.0]))
    best = None
    for intr0 in starts:
        try:
            poses0 = np.array([_init_pose('pinhole', intr0, objpoints[i], imgpoints[i])
                               for i in train_idx])
        except ValueError:
            continue
        intr, poses, info = lm_board_adjust('pinhole', intr0, poses0, obj, img, view,
                                            max_iter=max_iter)
        if best is None or info['cost'] < best[0]:
            best = (info['cost'], intr, poses, info)
    if best is None:
        raise ValueError('could not initialise any board pose')
    raw = {'pinhole': (best[1], best[2], best[3])}
    order = [m for m in MODELS if m in models and m != 'pinhole']
    for model in order:
        parent = NESTED_PARENT[model]
        while parent not in raw:
            parent = NESTED_PARENT.get(parent, 'pinhole')
        p_intr, p_poses, _ = raw[parent]
        Kp, dp = unpack_intrinsics(parent, p_intr)
        intr0 = pack_intrinsics(model, Kp, dp if is_fisheye(model) == is_fisheye(parent) else None)
        raw[model] = lm_board_adjust(model, intr0, p_poses, obj, img, view, max_iter=max_iter)

    fits = {}
    for model in ['pinhole'] + order:
        if model != 'pinhole' and model not in models:
            continue
        intr, poses, info = raw[model]
        K, dist = unpack_intrinsics(model, intr)
        finite = bool(np.all(np.isfinite(intr)) and K[0, 0] > 0)
        tr_res = np.linalg.norm(_project_views(model, intr, poses, obj, view) - img, axis=1)
        held, fails = _evaluate_heldout(model, intr, objpoints, imgpoints, heldout_idx) \
            if finite else (None, 0)
        fits[model] = ModelFit(model=model, n_params=n_free_params(model), K=K, dist=dist,
                               train=_stats(tr_res), heldout=_stats(held),
                               status=info['status'] if finite else 'failed',
                               iterations=int(info['iterations']), pnp_failures=int(fails))
    if 'pinhole' not in models:
        fits.pop('pinhole')
    selected, rule = select_model(fits, tol=tol, plausible=plausible)
    return LadderResult(fits=fits, selected=selected, rule=rule, train_idx=train_idx,
                        heldout_idx=heldout_idx, image_size=tuple(image_size))


def _evaluate_heldout(model, intr, objpoints, imgpoints, indices):
    obj, img, view = _stack(objpoints, imgpoints, indices)
    poses0, fails = [], 0
    for i in indices:
        try:
            poses0.append(_init_pose(model, intr, objpoints[i], imgpoints[i]))
        except ValueError:
            poses0.append(np.array([0, 0, 0, 0, 0, 5.0]))
            fails += 1
    _, poses, _ = lm_board_adjust(model, intr, np.asarray(poses0), obj, img, view,
                                  fix_intr=True, max_iter=100)
    return np.linalg.norm(_project_views(model, intr, poses, obj, view) - img, axis=1), fails


def calibrate_ladder_rows(rows, board, image_size, min_corners=6, **kwargs):
    """:func:`calibrate_ladder` on board detection rows (``board.detect_video`` output).

    Views are ordered by ``framenum``, which is used as the time for the held-out blocks."""
    views = []
    objp_all = board.get_object_points().reshape(-1, 3)
    for r in rows:
        filled = board.fill_points(r['corners'], r['ids']).reshape(-1, 2)
        good = np.all(np.isfinite(filled), axis=1)
        if good.sum() >= min_corners:
            fn = r['framenum']
            key = tuple(fn) if isinstance(fn, (tuple, list)) else (0, fn)
            views.append((key, objp_all[good], filled[good]))
    views.sort(key=lambda v: v[0])
    if 'times' not in kwargs and 'train_idx' not in kwargs:
        # prefix (video number) separates blocks: offset each video far apart
        keys = [v[0] for v in views]
        prefixes = {p: k for k, p in enumerate(sorted({k[0] for k in keys}, key=str))}
        span = max([k[1] for k in keys] + [1]) + 1
        kwargs['times'] = [prefixes[k[0]] * 10 * span + k[1] for k in keys]
        kwargs.setdefault('block', max(1.0, span / 25.0))
    res = calibrate_ladder([v[1] for v in views], [v[2] for v in views], image_size, **kwargs)
    res.extra['framenums'] = [v[0] for v in views]
    return res
