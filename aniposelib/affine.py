"""Affine (telecentric) cameras (1.0, plan 01 phase G, stretch).

A telecentric lens projects orthographically: ``uv = A X + b`` with ``A`` 2x3 and ``b`` 2
(8 degrees of freedom; the projection matrix is ``[[A, b], [0, 0, 0, 1]]``). There is no depth,
so a pinhole model only approximates it with a huge focal length at a huge distance (the
"telephoto pinhole" that saturates gates and breaks float32 downstream).

API
---
``AffineCamera(A, b, size=None, name=None)``: an ``nn.Module`` with float64 parameters ``A``
    [2,3] and ``b`` [2]. ``project(X)`` -> ``[N,2]``; ``reprojection_error(X, uv)``;
    ``get_projection_matrix()`` (3x4); ``get_direction()`` (unit viewing direction, the null
    space of ``A``, sign arbitrary); ``get_dict()`` / ``AffineCamera.from_dict(d)`` (``model =
    'affine'``, ``affine = [[A, b]]`` 2x4); ``from_projection_matrix(P)``; ``copy()``;
    ``to_pinhole(support, max_error_px=0.5)``: the closed-form telephoto-pinhole approximation
    (tailcyclenet ``johnson_fly``), checked on ``support`` points.
``triangulate_affine(cameras, points, weights=None, min_cams=2) -> [N,3]``
    Exact linear least squares for affine cameras (``points [C,N,2]``, NaN = missing).
``affine_from_points(X, uv) -> AffineCamera``
    Least-squares fit of ``A, b`` from 3D-2D correspondences (>= 4 non-coplanar points).

numpy in -> numpy out, torch in -> torch out (``return_grad=True`` keeps the graph).
``aniposelib.io.from_dlt`` / ``from_projection_matrix`` return an ``AffineCamera`` when the
third row of the projection is ``[0, 0, 0, *]``.
"""

import numpy as np
import torch
from torch import nn

from ._types import returns_like_input, to_torch

__all__ = ['AffineCamera', 'triangulate_affine', 'affine_from_points']


def _f64(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy().astype('float64')
    return np.asarray(x, dtype='float64')


class AffineCamera(nn.Module):
    """Telecentric camera ``uv = A X + b`` (see module docstring)."""

    fisheye = False
    affine = True

    def __init__(self, A=None, b=None, size=None, name=None, *, extra=None, dtype=torch.float64):
        super().__init__()
        A = np.array([[1.0, 0, 0], [0, 1.0, 0]]) if A is None else _f64(A).reshape(2, 3)
        b = np.zeros(2) if b is None else _f64(b).reshape(2)
        self.A = nn.Parameter(torch.as_tensor(A, dtype=dtype))
        self.b = nn.Parameter(torch.as_tensor(b, dtype=dtype))
        self.size = None if size is None else [int(round(float(s))) for s in list(size)[:2]]
        self.name = str(name)
        self.extra = dict(extra) if extra else {}

    # -- basic accessors (mirroring Camera)
    def get_name(self):
        return self.name

    def set_name(self, name):
        self.name = str(name)

    def get_size(self):
        return self.size

    def set_size(self, size):
        self.size = None if size is None else [int(round(float(s))) for s in list(size)[:2]]

    def get_projection_matrix(self):
        P = np.zeros((3, 4))
        P[:2, :3] = _f64(self.A)
        P[:2, 3] = _f64(self.b)
        P[2, 3] = 1.0
        return P

    def get_direction(self):
        """Unit viewing direction (null space of A; the sign is arbitrary)."""
        A = _f64(self.A)
        d = np.cross(A[0], A[1])
        return d / np.linalg.norm(d)

    @classmethod
    def from_projection_matrix(cls, P, size=None, name=None):
        P = _f64(P).reshape(3, 4)
        if abs(P[2, 3]) < 1e-300:
            raise ValueError('affine projection matrix needs P[2,3] != 0')
        P = P / P[2, 3]
        return cls(P[:2, :3], P[:2, 3], size=size, name=name)

    def get_dict(self):
        d = {'name': self.get_name(), 'model': 'affine',
             'affine': np.concatenate([_f64(self.A), _f64(self.b)[:, None]], 1).tolist()}
        if self.size is not None:
            d['size'] = list(self.size)
        for k, v in self.extra.items():
            d.setdefault(k, v)
        return d

    @classmethod
    def from_dict(cls, d, dtype=torch.float64):
        M = np.asarray(d['affine'], dtype='float64').reshape(2, 4)
        extra = {k: v for k, v in d.items() if k not in ('name', 'model', 'affine', 'size')}
        return cls(M[:, :3], M[:, 3], size=d.get('size'), name=d.get('name'), extra=extra, dtype=dtype)

    def copy(self):
        return AffineCamera.from_dict(self.get_dict(), dtype=self.A.dtype)

    # -- geometry
    def _project(self, X):
        X = to_torch(X, device=self.A.device, dtype=self.A.dtype)
        return X @ self.A.T + self.b

    @returns_like_input()
    def project(self, points):
        """World points ``[...,3]`` -> pixels ``[...,2]``."""
        return self._project(points)

    @returns_like_input()
    def reprojection_error(self, p3d, p2d):
        p2d = to_torch(p2d, device=self.A.device, dtype=self.A.dtype)
        return self._project(p3d) - p2d

    def to_pinhole(self, support, max_error_px=0.5, distance=None):
        """Closed-form telephoto pinhole that matches this camera to ``max_error_px`` on the
        ``support`` points ``[N,3]`` (ported from tailcyclenet ``scripts/convert/johnson_fly.py``).
        Keeps skew. Raises if the approximation is worse than ``max_error_px``. Returns a
        :class:`~aniposelib.cameras.Camera`."""
        from .io import camera_from_extrinsics_mat
        X = _f64(support).reshape(-1, 3)
        X = X[np.all(np.isfinite(X), 1)]
        if not len(X):
            raise ValueError('no finite support points')
        A, b = _f64(self.A), _f64(self.b)
        pd = self.get_direction()
        if np.median(X @ pd) < 0:
            pd = -pd
        if distance is None:
            A_pinv = A.T @ np.linalg.inv(A @ A.T)
            origins = (A_pinv @ (A @ X.T)).T
            depth = X @ pd
            sigma = np.linalg.svd(A, compute_uv=False)[0]
            d_pin = (2.0 / max_error_px) * sigma * float(np.max(np.linalg.norm(origins, axis=1) * np.abs(depth)))
            ext = float(depth.max() - depth.min())
            D = max(d_pin, float(-depth.min()) + 0.1 * ext)
        else:
            D = float(distance)
        uy = A[1] / np.linalg.norm(A[1])
        ux_proper = np.cross(uy, pd)
        if float(A[0] @ ux_proper) >= 0:
            uz, ux, center = pd, ux_proper, -D * pd
        else:
            uz, ux, center = -pd, -ux_proper, D * pd
        R = np.stack([ux, uy, uz])
        t = -R @ center
        K = np.array([[D * float(A[0] @ ux), D * float(A[0] @ uy), b[0]],
                      [0.0, D * float(np.linalg.norm(A[1])), b[1]], [0.0, 0.0, 1.0]])
        E = np.eye(4)
        E[:3, :3], E[:3, 3] = R, t
        cam = camera_from_extrinsics_mat(K, E, None, self.size, self.name)
        err = np.linalg.norm(_f64(cam.project(X)) - (X @ A.T + b), axis=1)
        if not np.all(np.isfinite(err)) or err.max() >= max_error_px:
            raise ValueError('pinhole approximation error {:.3g} px >= {} px'.format(
                float(np.nanmax(err)), max_error_px))
        return cam


class _Cams:
    def __init__(self, cams):
        self.cams = cams


def triangulate_affine(cameras, points, weights=None, min_cams=2, return_grad=False):
    """Linear least-squares triangulation for affine cameras (a list of :class:`AffineCamera`
    or a group of them). ``points [C,N,2]`` (or ``[C,...,2]``) -> ``[...,3]``; NaN where fewer
    than ``min_cams`` cameras see a point."""
    cams = list(cameras.cameras) if hasattr(cameras, 'cameras') else list(cameras)
    return _triangulate_affine(_Cams(cams), points, weights, min_cams, return_grad=return_grad)


@returns_like_input()
def _triangulate_affine(holder, points, weights=None, min_cams=2):
    cams = holder.cams
    A = torch.stack([c.A for c in cams])          # [C,2,3]
    b = torch.stack([c.b for c in cams])          # [C,2]
    pts = to_torch(points, device=A.device, dtype=A.dtype)
    C = pts.shape[0]
    lead = pts.shape[1:-1]
    pts = pts.reshape(C, -1, 2)
    valid = torch.isfinite(pts).all(-1)           # [C,N]
    w = valid.to(A.dtype)
    if weights is not None:
        w = w * torch.nan_to_num(to_torch(weights, device=A.device, dtype=A.dtype).reshape(C, -1)) ** 2
    r = torch.nan_to_num(pts) - b[:, None]        # [C,N,2]
    # normal equations sum_c w A^T A x = sum_c w A^T r
    AtA = A.transpose(1, 2) @ A                   # [C,3,3]
    M = torch.einsum('cn,cij->nij', w, AtA)
    v = torch.einsum('cn,cji,cnj->ni', w, A, r)
    ok = (valid.sum(0) >= min_cams) & (torch.linalg.det(M).abs() > 1e-300)
    eye = torch.eye(3, dtype=A.dtype, device=A.device)
    M = torch.where(ok[:, None, None], M, eye)
    X = torch.linalg.solve(M, v[..., None])[..., 0]
    X = torch.where(ok[:, None], X, torch.full_like(X, float('nan')))
    return X.reshape(tuple(lead) + (3,))


def affine_from_points(X, uv, size=None, name=None):
    """Fit ``uv = A X + b`` by least squares (NaN rows ignored) -> :class:`AffineCamera`."""
    X, uv = _f64(X).reshape(-1, 3), _f64(uv).reshape(-1, 2)
    ok = np.all(np.isfinite(X), 1) & np.all(np.isfinite(uv), 1)
    if ok.sum() < 4:
        raise ValueError('need at least 4 correspondences')
    H = np.concatenate([X[ok], np.ones((ok.sum(), 1))], 1)
    M, *_ = np.linalg.lstsq(H, uv[ok], rcond=None)   # [4,2]
    return AffineCamera(M[:3].T, M[3], size=size, name=name)
