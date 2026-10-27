"""Board-based initialisation that survives sparse and disconnected board graphs (plan 01 E1).

``initial_extrinsics(rtvecs, cam_names)`` is the robust counterpart of
:func:`aniposelib.utils.get_initial_extrinsics`: instead of raising when some cameras never
share a board view, it returns one **gauge per connected component** (its anchor camera is the
identity) and reports the components. Each spanning-tree edge uses the classic aniposelib
estimate (cluster + mean, ``utils.get_transform``) and falls back to a **robust estimate**
(rotation chordal mean with rejection + translation median of the trimmed set) when the
classic one fails or is non-finite (disco ``robust_fallback_extrinsics``).

``init_camera_intrinsics(camera, objp, imgp, size, bounds=..., fallback=...)`` wraps
:func:`aniposelib.cameras.init_camera_intrinsics` with a plausibility guard (disco
``valid_intrinsic_seed``): degenerate planar views give absurd focal lengths or principal
points, which are replaced by a nominal seed instead of poisoning bundle adjustment.

Conventions: ``rtvecs [C, F, 6]`` = per camera, per board frame, ``(rvec, tvec)`` of the board
in camera coordinates, NaN when unseen (``boards.extract_rtvecs``).
"""

from dataclasses import dataclass, field, fields

import numpy as np

from ._common import (M_from_rt, jsonable, make_M, robust_rotation_mean, rotation_angle_deg,
                      rotvec, to_np)

__all__ = ['board_graph', 'board_components', 'robust_relative_transform', 'initial_extrinsics',
           'BoardInit', 'valid_intrinsics', 'init_camera_intrinsics', 'IntrinsicsSeed',
           'DEFAULT_INTRINSICS_BOUNDS']


# ------------------------------------------------------------------ graph

def board_graph(rtvecs, min_shared=1):
    """Co-detection counts ``W [C, C]`` (number of board frames seen by both cameras); pairs
    with fewer than ``min_shared`` frames are 0."""
    rtvecs = np.asarray(rtvecs, dtype='float64')
    seen = np.all(np.isfinite(rtvecs), axis=-1).astype(int)  # [C, F]
    W = seen @ seen.T
    np.fill_diagonal(W, 0)
    W[W < min_shared] = 0
    return W


def board_components(rtvecs, cam_names=None, min_shared=1):
    """Connected components of the co-detection graph, as lists of camera indices (or names
    when ``cam_names`` is given), largest first, ties by first camera."""
    W = board_graph(rtvecs, min_shared)
    comps = _components(W)
    if cam_names is not None:
        return [[cam_names[i] for i in c] for c in comps]
    return comps


def _components(W):
    C = len(W)
    unseen = set(range(C))
    comps = []
    while unseen:
        root = min(unseen)
        stack, found = [root], []
        unseen.discard(root)
        while stack:
            n = stack.pop()
            found.append(n)
            for m in np.nonzero(W[n])[0]:
                if m in unseen:
                    unseen.discard(m)
                    stack.append(int(m))
        comps.append(sorted(found))
    comps.sort(key=lambda c: (-len(c), c[0]))
    return comps


def _max_spanning_tree(W, nodes, root):
    """Prim on co-detection counts: edges (parent, child) in placement order."""
    placed, edges = {root}, []
    nodes = set(nodes)
    while placed != nodes:
        best = None
        for a in placed:
            for b in nodes - placed:
                if W[a, b] > 0 and (best is None or W[a, b] > best[2]):
                    best = (a, b, W[a, b])
        if best is None:
            break
        edges.append((best[0], best[1]))
        placed.add(best[1])
    return edges


# ------------------------------------------------------------------ transforms

def _pair_transforms(rtvecs, a, b):
    """Per shared frame, the transform camera a -> camera b: ``M_b @ inv(M_a)``."""
    out = []
    for f in range(rtvecs.shape[1]):
        da, db = rtvecs[a, f], rtvecs[b, f]
        if np.all(np.isfinite(da)) and np.all(np.isfinite(db)):
            out.append(M_from_rt(db[:3], db[3:]) @ np.linalg.inv(M_from_rt(da[:3], da[3:])))
    return np.array(out).reshape(-1, 4, 4)


def robust_relative_transform(Ms, trim=0.25, rot_tol_deg=8.0):
    """Robust mean of 4x4 transforms: keep the ``1 - trim`` fraction of translations closest
    to their median, then rotation = robust chordal mean (iterative rejection), translation =
    median of the kept frames that also pass the rotation test. Returns ``(M, stats)``."""
    Ms = np.asarray(Ms, dtype='float64').reshape(-1, 4, 4)
    if len(Ms) == 0:
        raise ValueError('robust_relative_transform: no transforms')
    ts = Ms[:, :3, 3]
    d = np.linalg.norm(ts - np.median(ts, axis=0), axis=1)
    keep = d <= np.quantile(d, 1 - trim) + 1e-12 if len(Ms) > 2 else np.ones(len(Ms), bool)
    R, rkeep, disp = robust_rotation_mean(Ms[keep, :3, :3], tol_deg=rot_tol_deg)
    kk = np.nonzero(keep)[0][rkeep]
    t = np.median(ts[kk], axis=0)
    stats = dict(n=int(len(Ms)), n_kept=int(len(kk)), rot_disp_deg=float(disp),
                 trans_spread=float(np.median(np.linalg.norm(ts[kk] - t, axis=1))))
    return make_M(R, t), stats


def _classic_transform(rtvecs, a, b):
    from aniposelib.utils import get_transform
    return get_transform(rtvecs, b, a)  # = mean of M_b inv(M_a)


@dataclass
class BoardInit:
    """Board-graph initial extrinsics.

    ``rvecs, tvecs [C,3]``: world->camera of every camera; each component has its own gauge
    (its anchor camera is the identity, translations in board units). Cameras of different
    components are **not** in a common frame; ``component_of[c]`` says which one.
    ``edges``: per spanning-tree edge, ``(from, to, n_shared, method, rot_disp_deg, ...)``."""
    rvecs: np.ndarray
    tvecs: np.ndarray
    cam_names: list
    components: list            # lists of names, largest first
    anchors: list               # anchor camera name per component
    component_of: np.ndarray    # [C] component index
    edges: list = field(default_factory=list)

    @property
    def connected(self):
        return len(self.components) <= 1

    def Ms(self):
        return np.stack([M_from_rt(r, t) for r, t in zip(self.rvecs, self.tvecs)])

    def apply(self, cgroup, component=None):
        """Set the extrinsics of ``cgroup`` (cameras in ``cam_names`` order). With
        ``component=k`` return the subset group of that component instead (in place for the
        others is meaningless across gauges)."""
        if component is None:
            cgroup.set_rotations(self.rvecs)
            cgroup.set_translations(self.tvecs)
            return cgroup
        names = self.components[component]
        sub = cgroup.subset_cameras_names(names)
        idx = [self.cam_names.index(n) for n in names]
        sub.set_rotations(self.rvecs[idx])
        sub.set_translations(self.tvecs[idx])
        return sub

    def to_dict(self):
        return {f.name: jsonable(getattr(self, f.name)) for f in fields(self)} | {'connected': self.connected}


def initial_extrinsics(rtvecs, cam_names=None, method='auto', min_shared=1, anchor=None,
                       trim=0.25, rot_tol_deg=8.0, tree='max'):
    """Initial extrinsics from board poses, one gauge per connected component.

    ``method``: ``'classic'`` (``utils.get_transform``: cluster + mean, as
    ``get_initial_extrinsics``), ``'robust'`` (:func:`robust_relative_transform`) or ``'auto'``
    (classic, robust if it raises or is non-finite). ``anchor``: camera index (or name) that is
    the identity of its component (default: the lowest index of each component, i.e. camera 0
    for the main component as in ``get_initial_extrinsics``). Never raises on disconnected
    graphs: inspect ``BoardInit.components`` / ``.connected``.

    ``tree``: ``'max'`` (Prim's maximum spanning tree of the co-detection counts from the
    anchor) or ``'classic'`` (for a connected graph, the spanning tree and traversal of
    ``utils.get_calibration_graph`` / ``find_calibration_pairs``, so that the result is exactly
    that of aniposelib <= 1.0 whenever the classic estimate succeeds; disconnected graphs use
    ``'max'`` per component)."""
    if tree not in ('max', 'classic'):
        raise ValueError("tree must be 'max' or 'classic'")
    if method not in ('auto', 'classic', 'robust'):
        raise ValueError("method must be 'auto', 'classic' or 'robust'")
    rtvecs = np.asarray(rtvecs, dtype='float64')
    C = rtvecs.shape[0]
    names = list(cam_names) if cam_names is not None else list(range(C))
    if isinstance(anchor, str):
        anchor = names.index(anchor)
    W = board_graph(rtvecs, min_shared)
    comps = _components(W)
    Ms = np.tile(np.eye(4), (C, 1, 1))
    comp_of = np.zeros(C, int)
    anchors, edges = [], []
    for k, comp in enumerate(comps):
        root = anchor if (anchor is not None and anchor in comp) else comp[0]
        anchors.append(names[root])
        comp_of[comp] = k
        if tree == 'classic' and len(comps) == 1 and len(comp) > 1:
            from aniposelib.utils import get_calibration_graph, find_calibration_pairs
            tree_edges = find_calibration_pairs(get_calibration_graph(rtvecs), source=root)
        else:
            tree_edges = _max_spanning_tree(W, comp, root)
        for a, b in tree_edges:
            Mab, info = None, dict(n_shared=int(W[a, b]))
            if method in ('auto', 'classic'):
                try:
                    Mab = np.asarray(_classic_transform(rtvecs, a, b), dtype='float64')
                    if not np.all(np.isfinite(Mab)):
                        raise ValueError('non-finite transform')
                    info['method'] = 'classic'
                except Exception as exc:  # degenerate clustering / empty selections
                    if method == 'classic':
                        raise
                    Mab = None
                    info['classic_error'] = '{}: {}'.format(type(exc).__name__, exc)
            Ts = _pair_transforms(rtvecs, a, b)
            if Mab is None:
                Mab, st = robust_relative_transform(Ts, trim=trim, rot_tol_deg=rot_tol_deg)
                info.update(st, method='robust')
            else:
                info['rot_disp_deg'] = float(np.median(rotation_angle_deg(Mab[:3, :3], Ts[:, :3, :3])))
            Ms[b] = Mab @ Ms[a]
            edges.append(dict(src=names[a], dst=names[b], **info))
    rvecs = np.stack([rotvec(M[:3, :3]) for M in Ms])
    tvecs = Ms[:, :3, 3].copy()
    return BoardInit(rvecs=rvecs, tvecs=tvecs, cam_names=names,
                     components=[[names[i] for i in c] for c in comps], anchors=anchors,
                     component_of=comp_of, edges=edges)


# ------------------------------------------------------------------ intrinsics

#: Plausibility bounds of an intrinsics seed. ``focal`` is relative to ``max(w, h)``,
#: ``principal`` relative to the image size, ``aspect`` = fx / fy.
DEFAULT_INTRINSICS_BOUNDS = dict(focal=(0.2, 5.0), aspect=(0.6, 1.67), principal=(0.2, 0.8),
                                 max_abs_k1=1.0, focal_px=None)


def valid_intrinsics(matrix, dist, size, bounds=None):
    """``(ok, reason)``: is this camera matrix / distortion a plausible seed? (disco
    ``valid_intrinsic_seed`` with bounds relative to the image size; ``focal_px=(lo, hi)``
    overrides the relative focal bounds)."""
    b = dict(DEFAULT_INTRINSICS_BOUNDS)
    b.update(bounds or {})
    K = np.asarray(to_np(matrix), dtype='float64')
    d = np.asarray(to_np(dist), dtype='float64').ravel()
    if K.shape != (3, 3) or not np.all(np.isfinite(K)):
        return False, 'nonfinite_or_wrong_shape_matrix'
    if not np.all(np.isfinite(d)):
        return False, 'nonfinite_distortion'
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    if size is None:
        w = h = None
    else:
        w, h = float(size[0]), float(size[1])
    if b.get('focal_px') is not None:
        lo, hi = b['focal_px']
    elif w is not None and b.get('focal') is not None:
        lo, hi = b['focal'][0] * max(w, h), b['focal'][1] * max(w, h)
    else:
        lo, hi = 0.0, np.inf
    if not (fx > 0 and fy > 0):
        return False, 'nonpositive_focal'
    if not (lo <= fx <= hi and lo <= fy <= hi):
        return False, 'focal_outside_bounds'
    if b.get('aspect') is not None and not (b['aspect'][0] <= fx / fy <= b['aspect'][1]):
        return False, 'focal_aspect_outside_bounds'
    if w is not None and b.get('principal') is not None:
        plo, phi = b['principal']
        if not (plo * w <= cx <= phi * w):
            return False, 'principal_x_outside_bounds'
        if not (plo * h <= cy <= phi * h):
            return False, 'principal_y_outside_bounds'
    if b.get('max_abs_k1') is not None and len(d) and abs(d[0]) > b['max_abs_k1']:
        return False, 'k1_outside_bounds'
    return True, 'accepted'


@dataclass
class IntrinsicsSeed:
    """Outcome of :func:`init_camera_intrinsics`: ``source`` is ``'board'`` (accepted
    estimate), ``'fallback:<kind>'`` or ``'failed'``; ``reason`` is the guard verdict or the
    exception; ``estimate`` is the rejected/accepted board estimate (matrix, dist)."""
    camera: str
    accepted: bool
    source: str
    reason: str
    estimate: dict = None

    def __bool__(self):
        return bool(self.accepted)

    def to_dict(self):
        return {f.name: jsonable(getattr(self, f.name)) for f in fields(self)}


def nominal_matrix(size, focal=None, fisheye=False):
    """A generic seed: principal point at the image centre, ``focal`` px (default
    ``max(w, h)`` for pinhole, ``0.35 * max(w, h)`` for fisheye, about a GoPro wide lens)."""
    w, h = float(size[0]), float(size[1])
    if focal is None:
        focal = (0.35 if fisheye else 1.0) * max(w, h)
    return np.array([[focal, 0, (w - 1) / 2], [0, focal, (h - 1) / 2], [0, 0, 1.0]])


def init_camera_intrinsics(camera, objp, imgp, size, bounds=None, fallback='nominal',
                           nominal_focal=None):
    """Guarded :func:`aniposelib.cameras.init_camera_intrinsics`.

    Runs the classic seed (``cv2.initCameraMatrix2D`` / ``cv2.fisheye.calibrate``), then checks
    it with :func:`valid_intrinsics` (``bounds`` updates :data:`DEFAULT_INTRINSICS_BOUNDS`;
    ``bounds=False`` disables the guard). On failure (exception or implausible values):

    - ``fallback='nominal'``: :func:`nominal_matrix` (``nominal_focal`` px) and zero distortion;
    - ``'keep'``: restore the camera's previous matrix and distortions;
    - ``'raise'``: raise ``ValueError``;
    - a 3x3 matrix, or ``dict(matrix=..., dist=...)``: use it.

    Returns an :class:`IntrinsicsSeed` (truthy when the board estimate was accepted)."""
    from aniposelib.cameras import init_camera_intrinsics as _classic
    name = camera.get_name()
    old_K = to_np(camera.get_camera_matrix()).astype('float64').copy()
    old_d = to_np(camera.get_distortions()).astype('float64').copy()
    estimate, reason = None, None
    try:
        _classic(camera, objp, imgp, size)
        K = to_np(camera.get_camera_matrix()).astype('float64').copy()
        d = to_np(camera.get_distortions()).astype('float64').copy()
        estimate = dict(matrix=K, dist=d)
        if bounds is False:
            ok, reason = bool(np.all(np.isfinite(K)) and np.all(np.isfinite(d))), 'unchecked'
        else:
            ok, reason = valid_intrinsics(K, d, size, bounds)
    except Exception as exc:  # cv2.error, degenerate inputs
        ok, reason = False, 'init_error: {}: {}'.format(type(exc).__name__, exc)
    if ok:
        return IntrinsicsSeed(name, True, 'board', reason, estimate)
    if isinstance(fallback, str) and fallback == 'raise':
        camera.set_camera_matrix(old_K)
        camera.set_distortions(old_d)
        raise ValueError('implausible intrinsics seed for camera {}: {}'.format(name, reason))
    if isinstance(fallback, str) and fallback == 'keep':
        K, d, kind = old_K, old_d, 'keep'
    elif isinstance(fallback, str) and fallback == 'nominal':
        K = nominal_matrix(size, nominal_focal, fisheye=getattr(camera, 'fisheye', False))
        d, kind = np.zeros_like(old_d), 'nominal'
    elif isinstance(fallback, dict):
        K = np.asarray(fallback['matrix'], dtype='float64')
        d = np.asarray(fallback.get('dist', np.zeros_like(old_d)), dtype='float64')
        kind = 'given'
    else:
        K, d, kind = np.asarray(fallback, dtype='float64'), np.zeros_like(old_d), 'given'
    camera.set_camera_matrix(K)
    camera.set_distortions(d)
    return IntrinsicsSeed(name, False, 'fallback:' + kind, reason, estimate)
