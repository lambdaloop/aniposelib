"""Shared helpers for :mod:`aniposelib.init`: rotation averaging, similarity alignment,
building a :class:`~aniposelib.cameras.CameraGroup` from 4x4 extrinsics, and the
:class:`InitReport` result object.

Conventions: extrinsics ``M`` are 4x4 *world -> camera* matrices (``x_cam = R x_world + t``),
the same as ``Camera.get_extrinsics_mat()``. Everything here is numpy (float64).
"""

from copy import deepcopy
from dataclasses import dataclass, field, fields

import numpy as np
import torch


def as_rng(rng):
    """``np.random.Generator`` from a Generator, an int seed or None (never global state)."""
    if isinstance(rng, np.random.Generator):
        return rng
    return np.random.default_rng(rng)


def to_np(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def jsonable(x):
    """numpy/torch -> lists, recursively (for ``to_dict``)."""
    if isinstance(x, torch.Tensor):
        x = to_np(x)
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.floating, np.integer, np.bool_)):
        return x.item()
    if isinstance(x, dict):
        return {(k if isinstance(k, (str, int, float, bool)) or k is None else str(k)): jsonable(v)
                for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    if hasattr(x, 'to_dict') and callable(x.to_dict):
        return x.to_dict()
    return x


# ------------------------------------------------------------------ rotations

def rodrigues(rvec):
    """Rotation vector -> matrix (numpy, no cv2)."""
    r = np.asarray(rvec, dtype='float64').reshape(3)
    th = np.linalg.norm(r)
    if th < 1e-15:
        return np.eye(3)
    k = r / th
    Kx = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(th) * Kx + (1 - np.cos(th)) * Kx @ Kx


def rotvec(R):
    """Rotation matrix -> rotation vector (numpy; robust near pi)."""
    from scipy.spatial.transform import Rotation
    return Rotation.from_matrix(project_to_so3(R)).as_rotvec()


def project_to_so3(A):
    """Closest rotation (Frobenius) to a 3x3 matrix (or a stack [...,3,3])."""
    A = np.asarray(A, dtype='float64')
    U, _, Vt = np.linalg.svd(A)
    d = np.sign(np.linalg.det(U @ Vt))
    D = np.ones(A.shape[:-1])
    D[..., -1] = d
    return (U * D[..., None, :]) @ Vt


def chordal_mean(Rs, weights=None):
    """Chordal L2 mean of rotations: the rotation closest to the (weighted) mean matrix."""
    Rs = np.asarray(Rs, dtype='float64').reshape(-1, 3, 3)
    w = np.ones(len(Rs)) if weights is None else np.asarray(weights, dtype='float64').reshape(-1)
    return project_to_so3(np.einsum('n,nij->ij', w / w.sum(), Rs))


def rotation_angle_deg(R1, R2):
    """Geodesic angle (degrees) between rotations; broadcasts over leading axes."""
    R1, R2 = np.asarray(R1, dtype='float64'), np.asarray(R2, dtype='float64')
    c = (np.einsum('...ji,...ji->...', R1, R2) - 1.0) / 2.0
    return np.degrees(np.arccos(np.clip(c, -1.0, 1.0)))


def robust_rotation_mean(Rs, tol_deg=8.0, n_iter=5, weights=None):
    """Chordal mean with iterative rejection of rotations further than
    ``max(tol_deg, 1.5 * median angle)`` from the current mean (motor ``_robust_mean_rotation``).
    Returns ``(R, keep [n] bool, median_angle_deg)``."""
    Rs = np.asarray(Rs, dtype='float64').reshape(-1, 3, 3)
    w = np.ones(len(Rs)) if weights is None else np.asarray(weights, dtype='float64')
    R = chordal_mean(Rs, w)
    keep = np.ones(len(Rs), bool)
    if len(Rs) < 3:
        return R, keep, float(np.median(rotation_angle_deg(R, Rs)))
    for _ in range(n_iter):
        d = rotation_angle_deg(R, Rs)
        new = d <= max(tol_deg, np.median(d) * 1.5)
        if new.sum() < 2:
            new = d <= np.percentile(d, 60)
        if new.sum() < 2:
            break
        R = chordal_mean(Rs[new], w[new])
        if (new == keep).all():
            break
        keep = new
    return R, keep, float(np.median(rotation_angle_deg(R, Rs)))


def trimmed_mean(x, trim=0.25, axis=0):
    """Mean of the points within the ``1 - trim`` quantile of distance to the median."""
    x = np.asarray(x, dtype='float64')
    med = np.median(x, axis=axis)
    d = np.linalg.norm(x - med, axis=-1)
    keep = d <= np.quantile(d, 1 - trim) + 1e-12
    return x[keep].mean(axis=axis), keep


# ------------------------------------------------------------------ transforms

def make_M(R, t):
    M = np.eye(4)
    M[:3, :3] = np.asarray(R, dtype='float64').reshape(3, 3)
    M[:3, 3] = np.asarray(t, dtype='float64').reshape(3)
    return M


def M_from_rt(rvec, tvec):
    return make_M(rodrigues(rvec), tvec)


def centers(Ms):
    """Camera centres in world coordinates of world->camera matrices ``[...,4,4]``."""
    Ms = np.asarray(Ms, dtype='float64')
    R, t = Ms[..., :3, :3], Ms[..., :3, 3]
    return -np.einsum('...ji,...j->...i', R, t)


def umeyama(src, dst, with_scale=True, weights=None):
    """Similarity ``(s, R, t)`` minimising ``sum w |s R src + t - dst|^2`` (Umeyama 1991)."""
    src, dst = np.asarray(src, dtype='float64'), np.asarray(dst, dtype='float64')
    w = np.ones(len(src)) if weights is None else np.asarray(weights, dtype='float64')
    w = w / w.sum()
    mu_s, mu_d = w @ src, w @ dst
    xs, xd = src - mu_s, dst - mu_d
    S = (xd * w[:, None]).T @ xs
    U, D, Vt = np.linalg.svd(S)
    E = np.eye(3)
    E[2, 2] = np.sign(np.linalg.det(U @ Vt)) or 1.0
    R = U @ E @ Vt
    var_s = float(w @ (xs ** 2).sum(1))
    s = float((D * np.diag(E)).sum() / var_s) if (with_scale and var_s > 0) else 1.0
    return s, R, mu_d - s * R @ mu_s


def transform_extrinsics(Ms, s, R, t):
    """Apply the world similarity ``X' = s R X + t`` to world->camera matrices.
    Camera centres move as ``C' = s R C + t``; rotations become ``R_c R^T``; the metric
    of the camera frame is scaled by ``s`` (so translations scale by ``s``)."""
    Ms = np.asarray(Ms, dtype='float64')
    out = np.array(Ms, copy=True)
    Rc = Ms[..., :3, :3]
    out[..., :3, :3] = Rc @ R.T
    C = centers(Ms)
    Cn = s * C @ R.T + t
    out[..., :3, 3] = -np.einsum('...ij,...j->...i', out[..., :3, :3], Cn)
    return out


def relative_to(Ms, ref=0):
    """Re-express extrinsics so that camera ``ref`` is the identity (the board/SfM gauge)."""
    Ms = np.asarray(Ms, dtype='float64')
    inv = np.linalg.inv(Ms[ref])
    return Ms @ inv


# ------------------------------------------------------------------ camera groups

def from_extrinsics(names, Ms, K=None, sizes=None, model='pinhole', dist=None, n_dist=None,
                    template=None, dtype=None):
    """A :class:`CameraGroup` from per-camera world->camera matrices ``Ms [C,4,4]``.

    Intrinsics come from ``template`` (a ``CameraGroup`` whose cameras are copied, matched by
    name if possible, else by position) or from ``K`` (one 3x3 for all cameras or ``[C,3,3]``),
    ``dist`` (one vector or ``[C,D]``), ``sizes`` (``(w, h)`` or ``[C,2]``) and ``model``
    (``'pinhole'`` | ``'fisheye'``, or a per-camera list).
    """
    from aniposelib.cameras import Camera, CameraGroup, FisheyeCamera
    Ms = np.asarray(Ms, dtype='float64')
    names = [str(n) for n in names]
    C = len(names)
    if Ms.shape != (C, 4, 4):
        raise ValueError('Ms must be [C,4,4] with C=len(names), got {}'.format(Ms.shape))
    cams = []
    if template is not None:
        tnames = template.get_names()
        for i, n in enumerate(names):
            j = tnames.index(n) if n in tnames else i
            cam = template.cameras[j].copy()
            cam.set_name(n)
            cams.append(cam)
    else:
        models = [model] * C if isinstance(model, str) else list(model)
        Ks = np.broadcast_to(np.eye(3) if K is None else np.asarray(K, dtype='float64'), (C, 3, 3))
        szs = [None] * C if sizes is None else (
            [tuple(sizes)] * C if np.ndim(sizes) == 1 else [tuple(s) for s in sizes])
        for i, n in enumerate(names):
            fish = models[i] in ('fisheye', 'kb', True)
            if dist is None:
                d = np.zeros(4 if fish else 5)
            else:
                d = np.asarray(dist, dtype='float64')
                d = d[i] if d.ndim == 2 else d
            kw = {} if n_dist is None else dict(n_dist=n_dist)
            cls = FisheyeCamera if fish else Camera
            cams.append(cls(matrix=Ks[i].copy(), dist=np.array(d, copy=True), size=szs[i], name=n, **kw))
    for cam, M in zip(cams, Ms):
        cam.set_rotation(rotvec(M[:3, :3]))
        cam.set_translation(M[:3, 3])
    cg = CameraGroup(cams, metadata=deepcopy(template.metadata) if template is not None else None)
    if dtype is not None:
        cg = cg.to(dtype)
    return cg


def get_extrinsics(cgroup):
    """``[C,4,4]`` world->camera matrices of a group (numpy)."""
    return np.stack([to_np(c.get_extrinsics_mat()).astype('float64') for c in cgroup.cameras])


def with_extrinsics(cgroup, Ms, names=None):
    """A copy of ``cgroup`` (or of its cameras ``names``) with extrinsics ``Ms``."""
    if names is not None:
        cgroup = cgroup.subset_cameras_names(list(names))
    return from_extrinsics(cgroup.get_names(), Ms, template=cgroup)


# ------------------------------------------------------------------ result

@dataclass
class InitReport:
    """What an initialiser did. ``drivers`` defined the reconstruction; ``passengers`` were
    placed but did not define points; ``rejected`` maps a camera name to the reason (dict);
    ``missing`` were never placed. ``components`` (board init) lists camera-name groups that
    have independent gauges. ``stats`` holds method-specific numbers."""
    method: str
    placed: list = field(default_factory=list)
    drivers: list = field(default_factory=list)
    passengers: dict = field(default_factory=dict)
    rejected: dict = field(default_factory=dict)
    missing: list = field(default_factory=list)
    components: list = field(default_factory=list)
    stats: dict = field(default_factory=dict)

    @property
    def ok(self):
        return len(self.missing) == 0 and len(self.rejected) == 0 and len(self.components) <= 1

    def to_dict(self):
        out = {f.name: jsonable(getattr(self, f.name)) for f in fields(self)}
        out['ok'] = self.ok
        return out
