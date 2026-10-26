"""Packed, vectorised camera geometry (1.0, plan 01 C1).

All cameras of a group are stacked into one :class:`PackedCameras` (tensors with a leading
camera axis ``C``) and projected / undistorted at once, instead of looping over an
``nn.ModuleList``. Everything here is pure torch and differentiable; ``CameraGroup`` methods
delegate to these kernels.

API
---
``pack(cgroup, device=None, dtype=torch.float64, detach=True) -> PackedCameras``
    Stack a ``CameraGroup`` (pinhole ``Camera`` -> ``'brown'``, ``FisheyeCamera`` -> ``'kb'``).
``unpack_into(cgroup, P)``
    Write the parameters of ``P`` back into the group's cameras (in place, no autograd).
``PackedCameras``
    ``K [C,3,3]`` full intrinsic matrix (skew ``K[0,1]`` honoured; the bottom row is ignored),
    ``dist [C,8]`` zero padded (``brown``: OpenCV ``k1 k2 p1 p2 k3 k4 k5 k6`` rational model;
    ``kb``: Kannala-Brandt / ``cv2.fisheye`` ``k1 k2 k3 k4``), ``model [C]`` int (0 brown, 1 kb),
    ``rvec, tvec [C,3]`` and/or ``ext [C,4,4]`` (or per-frame ``[C,T,4,4]``; takes precedence
    over rvec/tvec), ``offset [C,2]`` (origin of the stored image in sensor pixels: projection
    subtracts it, undistortion adds it back), ``size [C,2]`` (width, height of the stored image,
    NaN if unknown). Helpers: ``extrinsics()``, ``R``, ``t``, ``centers()``, ``to()``,
    ``subset()``, ``detach()``, ``to_dict()``.
``project(P, X, extrinsics=None) -> (uv [C,...,2], depth [C,...])``
    ``X [...,3]`` world points. With per-frame extrinsics ``[C,T,4,4]``, ``X`` is ``[T,...,3]``.
``undistort(P, uv, n_iter=20, tol=1e-12, check_invertible=False) -> [C,...,2]``
    Pixels -> normalised (undistorted) image coordinates. Brown: fixed-point warm start then
    Newton; KB: Newton on theta. KB points with no solution are NaN (as ``FisheyeCamera``);
    ``check_invertible=True`` also NaNs points where the forward model is not locally
    invertible (non-positive Jacobian determinant / non-monotone radial map, or no
    convergence).
``distort(P, xy) -> [C,...,2]``
    Normalised coordinates -> pixels (no extrinsics).
``reproj_err(P, X, uv, extrinsics=None) -> [C,...]``
    Pixel error; NaN where ``uv`` is missing, ``inf`` where the point is behind the camera.
``project_jacobian(P, X, extrinsics=None) -> (uv, depth, J [C,N,2,3])``
    Analytic d(pixel)/d(world point), used by Gauss-Newton point refinement.

Public functions accept numpy or torch for the point arguments: numpy in -> numpy out, torch in
-> detached torch out (on ``P``'s device), ``return_grad=True`` keeps the autograd graph.
"""

import dataclasses
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import torch

from ._types import DEFAULT_DTYPE, returns_like_input, to_torch

__all__ = ['PackedCameras', 'pack', 'unpack_into', 'project', 'project_jacobian', 'undistort',
           'distort', 'reproj_err', 'rodrigues', 'make_extrinsics', 'MODELS']

MODELS = {'brown': 0, 'kb': 1}
MODEL_NAMES = {v: k for k, v in MODELS.items()}
N_DIST = 8
FISHEYE_EPS = 1e-12
_HALF_PI = float(np.pi / 2)


# ------------------------------------------------------------------ rotations

def rodrigues(rvec):
    """Rotation vectors ``[...,3]`` -> rotation matrices ``[...,3,3]`` (differentiable, also at 0)."""
    th2 = (rvec * rvec).sum(-1)
    small = th2 < 1e-8
    th2s = torch.where(small, torch.ones_like(th2), th2)
    th = torch.sqrt(th2s)
    A = torch.where(small, 1 - th2 / 6, torch.sin(th) / th)
    B = torch.where(small, 0.5 - th2 / 24, (1 - torch.cos(th)) / th2s)
    x, y, z = rvec[..., 0], rvec[..., 1], rvec[..., 2]
    o = torch.zeros_like(x)
    Kx = torch.stack([torch.stack([o, -z, y], -1),
                      torch.stack([z, o, -x], -1),
                      torch.stack([-y, x, o], -1)], -2)
    eye = torch.eye(3, dtype=rvec.dtype, device=rvec.device)
    return eye + A[..., None, None] * Kx + B[..., None, None] * (Kx @ Kx)


def make_extrinsics(rvec, tvec):
    """``[...,3]``, ``[...,3]`` -> 4x4 world-to-camera matrices ``[...,4,4]``."""
    R = rodrigues(rvec)
    top = torch.cat([R, tvec[..., :, None]], -1)
    bottom = torch.zeros(top.shape[:-2] + (1, 4), dtype=top.dtype, device=top.device)
    bottom[..., 0, 3] = 1
    return torch.cat([top, bottom], -2)


def _rotmat_to_rvec_np(R):
    import cv2
    R = np.asarray(R, dtype='float64')
    return np.array([cv2.Rodrigues(r)[0].ravel() for r in R.reshape(-1, 3, 3)]).reshape(R.shape[:-2] + (3,))


# ------------------------------------------------------------------ packed container

@dataclass
class PackedCameras:
    """Stacked camera parameters (see module docstring). All tensors share device and dtype."""
    K: torch.Tensor
    dist: torch.Tensor
    model: torch.Tensor
    rvec: Optional[torch.Tensor] = None
    tvec: Optional[torch.Tensor] = None
    ext: Optional[torch.Tensor] = None
    offset: Optional[torch.Tensor] = None
    size: Optional[torch.Tensor] = None
    names: Optional[List[str]] = None
    dist_len: Optional[List[int]] = None   # original per-camera distortion length (for unpack)

    def __post_init__(self):
        C = self.K.shape[0]
        dev, dt = self.K.device, self.K.dtype
        if self.dist.shape[-1] < N_DIST:
            pad = torch.zeros(C, N_DIST - self.dist.shape[-1], dtype=dt, device=dev)
            self.dist = torch.cat([self.dist, pad], -1)
        self.model = torch.as_tensor(self.model, dtype=torch.long, device=dev).reshape(C)
        if self.offset is None:
            self.offset = torch.zeros(C, 2, dtype=dt, device=dev)
        if self.size is None:
            self.size = torch.full((C, 2), float('nan'), dtype=dt, device=dev)
        if self.ext is None and (self.rvec is None or self.tvec is None):
            raise ValueError('PackedCameras needs rvec and tvec, or ext')

    # -- basic properties
    @property
    def n_cams(self):
        return self.K.shape[0]

    def __len__(self):
        return self.n_cams

    @property
    def device(self):
        return self.K.device

    @property
    def dtype(self):
        return self.K.dtype

    @property
    def per_frame(self):
        return self.ext is not None and self.ext.ndim == 4

    def extrinsics(self):
        """World-to-camera matrices ``[C,4,4]`` (or ``[C,T,4,4]`` for per-frame extrinsics)."""
        if self.ext is not None:
            return self.ext
        return make_extrinsics(self.rvec, self.tvec)

    @property
    def R(self):
        return self.extrinsics()[..., :3, :3]

    @property
    def t(self):
        return self.extrinsics()[..., :3, 3]

    def centers(self):
        """Camera centres in world coordinates ``[C,3]`` (``[C,T,3]`` per frame)."""
        R, t = self.R, self.t
        return -(R.transpose(-1, -2) @ t[..., None])[..., 0]

    def model_names(self):
        return [MODEL_NAMES[int(m)] for m in self.model.tolist()]

    # -- conversions
    def _map(self, fn):
        kw = {}
        for f in dataclasses.fields(self):
            v = getattr(self, f.name)
            kw[f.name] = fn(v) if isinstance(v, torch.Tensor) and f.name != 'model' else v
        kw['model'] = self.model
        return PackedCameras(**kw)

    def to(self, device=None, dtype=None):
        out = self._map(lambda v: v.to(device=device, dtype=dtype))
        out.model = out.model.to(device)
        return out

    def detach(self):
        return self._map(lambda v: v.detach())

    def clone(self):
        return self._map(lambda v: v.clone())

    def subset(self, idx):
        idx = list(idx)
        it = torch.as_tensor(idx, dtype=torch.long, device=self.device)
        kw = {}
        for f in dataclasses.fields(self):
            v = getattr(self, f.name)
            if isinstance(v, torch.Tensor):
                kw[f.name] = v[it]
            elif isinstance(v, list):
                kw[f.name] = [v[i] for i in idx]
            else:
                kw[f.name] = v
        return PackedCameras(**kw)

    def with_extrinsics(self, ext):
        """A copy whose extrinsics are replaced by ``ext`` ([C,4,4] or [C,T,4,4])."""
        return dataclasses.replace(self, ext=torch.as_tensor(ext, dtype=self.dtype, device=self.device))

    def to_dict(self):
        """JSON-friendly dict (numpy -> lists); also the ``as_tensors`` contract when kept as tensors."""
        out = {}
        for f in dataclasses.fields(self):
            v = getattr(self, f.name)
            if isinstance(v, torch.Tensor):
                v = v.detach().cpu().numpy().tolist()
            out[f.name] = v
        out['model'] = self.model_names()
        return out


def _cam_offset(cam):
    off = getattr(cam, 'offset', None)
    if off is None:
        extra = getattr(cam, 'extra', None)
        if isinstance(extra, dict):
            off = extra.get('offset')
    if off is None:
        return [0.0, 0.0]
    off = off.detach().cpu().numpy() if isinstance(off, torch.Tensor) else np.asarray(off, dtype='float64')
    return [float(off.ravel()[0]), float(off.ravel()[1])]


def pack(cgroup, device=None, dtype=DEFAULT_DTYPE, detach=True):
    """Stack the cameras of ``cgroup`` (or a list of cameras) into a :class:`PackedCameras`.

    ``detach=False`` keeps the parameters in the autograd graph (gradients flow back to the
    cameras' ``nn.Parameter`` s)."""
    cams = list(cgroup.cameras) if hasattr(cgroup, 'cameras') else list(cgroup)
    if len(cams) == 0:
        raise ValueError('no cameras to pack')
    if device is None:
        device = cams[0].matrix.device

    def get(t):
        t = t.detach() if detach else t
        return t.to(device=device, dtype=dtype)

    K = torch.stack([get(c.matrix) for c in cams])
    dists = []
    for c in cams:
        d = get(c.dist).reshape(-1)
        if d.shape[0] > N_DIST:
            raise ValueError('camera {} has {} distortion coefficients; at most {} are supported'
                             .format(c.get_name(), d.shape[0], N_DIST))
        dists.append(torch.cat([d, torch.zeros(N_DIST - d.shape[0], dtype=dtype, device=device)]))
    dist = torch.stack(dists)
    model = [MODELS['kb'] if getattr(c, 'fisheye', False) else MODELS['brown'] for c in cams]
    rvec = torch.stack([get(c.rvec).reshape(3) for c in cams])
    tvec = torch.stack([get(c.tvec).reshape(3) for c in cams])
    size = torch.tensor([[float('nan')] * 2 if c.get_size() is None else [float(s) for s in c.get_size()]
                         for c in cams], dtype=dtype, device=device)
    offset = torch.tensor([_cam_offset(c) for c in cams], dtype=dtype, device=device)
    return PackedCameras(K=K, dist=dist, model=torch.tensor(model, device=device), rvec=rvec, tvec=tvec,
                         offset=offset, size=size, names=[c.get_name() for c in cams],
                         dist_len=[int(c.dist.shape[0]) for c in cams])


def unpack_into(cgroup, P):
    """Write ``P`` back into ``cgroup``'s cameras (matrix, distortions, rvec/tvec, size, offset).

    Per-frame extrinsics cannot be stored on a ``Camera`` and raise ``ValueError``."""
    cams = list(cgroup.cameras) if hasattr(cgroup, 'cameras') else list(cgroup)
    if len(cams) != P.n_cams:
        raise ValueError('group has {} cameras, packed has {}'.format(len(cams), P.n_cams))
    if P.per_frame:
        raise ValueError('per-frame extrinsics cannot be written into Camera objects')
    if P.ext is not None:
        ext = P.ext.detach().cpu().numpy()
        rvecs = _rotmat_to_rvec_np(ext[:, :3, :3])
        tvecs = ext[:, :3, 3]
    else:
        rvecs = P.rvec.detach().cpu().numpy()
        tvecs = P.tvec.detach().cpu().numpy()
    K = P.K.detach().cpu().numpy()
    dist = P.dist.detach().cpu().numpy()
    size = P.size.detach().cpu().numpy()
    offset = P.offset.detach().cpu().numpy()
    with torch.no_grad():
        for i, cam in enumerate(cams):
            dev, dt = cam.matrix.device, cam.matrix.dtype
            n = cam.dist.shape[0]
            if np.any(dist[i, n:] != 0):
                raise ValueError('camera {} stores {} distortion coefficients but the packed '
                                 'value uses more'.format(cam.get_name(), n))
            cam.matrix.data = torch.as_tensor(K[i], dtype=dt, device=dev).clone()
            cam.dist.data = torch.as_tensor(dist[i, :n], dtype=dt, device=dev).clone()
            cam.rvec.data = torch.as_tensor(rvecs[i], dtype=dt, device=dev).reshape(cam.rvec.shape).clone()
            cam.tvec.data = torch.as_tensor(tvecs[i], dtype=dt, device=dev).reshape(cam.tvec.shape).clone()
            if np.all(np.isfinite(size[i])):
                cam.set_size([int(round(s)) for s in size[i]])
            if hasattr(cam, 'offset') or np.any(offset[i] != 0):
                if hasattr(cam, 'offset') and isinstance(cam.offset, torch.Tensor):
                    cam.offset.data = torch.as_tensor(offset[i], dtype=cam.offset.dtype, device=cam.offset.device)
                elif hasattr(cam, 'extra') and isinstance(getattr(cam, 'extra'), dict) and not hasattr(cam, 'offset'):
                    cam.extra['offset'] = offset[i].tolist()
                else:
                    cam.offset = offset[i].tolist()
    return cgroup


# ------------------------------------------------------------------ distortion models

def _col(v):
    """[C] -> [C,1] for broadcasting against [C,N]."""
    return v[:, None]


def _brown(x, y, d, jac=False):
    k1, k2, p1, p2, k3, k4, k5, k6 = [_col(d[:, i]) for i in range(8)]
    r2 = x * x + y * y
    num = 1 + r2 * (k1 + r2 * (k2 + r2 * k3))
    den = 1 + r2 * (k4 + r2 * (k5 + r2 * k6))
    rad = num / den
    xd = x * rad + 2 * p1 * x * y + p2 * (r2 + 2 * x * x)
    yd = y * rad + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y
    if not jac:
        return xd, yd
    dnum = k1 + r2 * (2 * k2 + 3 * k3 * r2)
    dden = k4 + r2 * (2 * k5 + 3 * k6 * r2)
    drad = (dnum * den - num * dden) / (den * den)      # d rad / d r2
    jxx = rad + 2 * x * x * drad + 2 * p1 * y + 6 * p2 * x
    jxy = 2 * x * y * drad + 2 * p1 * x + 2 * p2 * y
    jyx = 2 * x * y * drad + 2 * p1 * x + 2 * p2 * y
    jyy = rad + 2 * y * y * drad + 6 * p1 * y + 2 * p2 * x
    return xd, yd, (jxx, jxy, jyx, jyy)


def _kb_poly(th, d):
    k1, k2, k3, k4 = [_col(d[:, i]) for i in range(4)]
    t2 = th * th
    thd = th * (1 + t2 * (k1 + t2 * (k2 + t2 * (k3 + t2 * k4))))
    dthd = 1 + t2 * (3 * k1 + t2 * (5 * k2 + t2 * (7 * k3 + t2 * 9 * k4)))
    return thd, dthd


def _kb(x, y, d, jac=False, eps=FISHEYE_EPS):
    r2 = x * x + y * y
    r_ok = r2 >= eps * eps
    r = torch.sqrt(torch.where(r_ok, r2, torch.ones_like(r2)))
    th = torch.atan(r)
    thd, dthd = _kb_poly(th, d)
    s = torch.where(r_ok, thd / r, torch.ones_like(r))
    xd, yd = s * x, s * y
    if not jac:
        return xd, yd
    # ds/dr / r, with its limit 2 (k1 - 1/3) at r -> 0
    dsdr_r = torch.where(r_ok, (dthd / (1 + r * r) * r - thd) / (r * r * r),
                         2 * (_col(d[:, 0]) - 1.0 / 3) * torch.ones_like(r))
    jxx = s + dsdr_r * x * x
    jxy = dsdr_r * x * y
    jyy = s + dsdr_r * y * y
    return xd, yd, (jxx, jxy, jxy, jyy)


def _models_present(P):
    m = P.model
    return bool((m == 0).any()), bool((m == 1).any())


def _distort_norm(P, x, y, jac=False):
    """Normalised [C,N] -> distorted normalised, per camera model."""
    has_b, has_k = _models_present(P)
    if has_b and not has_k:
        return _brown(x, y, P.dist, jac)
    if has_k and not has_b:
        return _kb(x, y, P.dist, jac)
    ob = _brown(x, y, P.dist, jac)
    ok = _kb(x, y, P.dist, jac)
    is_k = _col(P.model == 1)
    if not jac:
        return tuple(torch.where(is_k, a, b) for a, b in zip(ok, ob))
    xd = torch.where(is_k, ok[0], ob[0])
    yd = torch.where(is_k, ok[1], ob[1])
    J = tuple(torch.where(is_k, a, b) for a, b in zip(ok[2], ob[2]))
    return xd, yd, J


def _pixels(P, xd, yd):
    K = P.K
    u = _col(K[:, 0, 0]) * xd + _col(K[:, 0, 1]) * yd + _col(K[:, 0, 2]) - _col(P.offset[:, 0])
    v = _col(K[:, 1, 0]) * xd + _col(K[:, 1, 1]) * yd + _col(K[:, 1, 2]) - _col(P.offset[:, 1])
    return torch.stack([u, v], -1)


def _normalize_pixels(P, uv):
    """Pixels [C,N,2] -> distorted normalised (inverse of the affine part of K, plus offset)."""
    K = P.K
    u = uv[..., 0] + _col(P.offset[:, 0]) - _col(K[:, 0, 2])
    v = uv[..., 1] + _col(P.offset[:, 1]) - _col(K[:, 1, 2])
    a, b, c, e = _col(K[:, 0, 0]), _col(K[:, 0, 1]), _col(K[:, 1, 0]), _col(K[:, 1, 1])
    det = a * e - b * c
    return (e * u - b * v) / det, (a * v - c * u) / det


# ------------------------------------------------------------------ projection kernels (torch)

def _ext_for(P, extrinsics):
    E = P.extrinsics() if extrinsics is None else to_torch(extrinsics, device=P.device, dtype=P.dtype)
    return E


def _cam_points(E, X):
    """World points -> camera coordinates.

    E [C,4,4] with X [N,3] -> [C,N,3];  E [C,N,4,4] (per point) with X [N,3] -> [C,N,3]."""
    R, t = E[..., :3, :3], E[..., :3, 3]
    if E.ndim == 3:
        return torch.einsum('cij,nj->cni', R, X) + t[:, None, :]
    return torch.einsum('cnij,nj->cni', R, X) + t


def _flatten_points(P, X, E):
    """Bring X [...,3] to [N,3] and E to [C,4,4] or [C,N,4,4]; returns (X, E, lead_shape)."""
    lead = X.shape[:-1]
    if E.ndim == 4 and E.shape[1] != (X.reshape(-1, 3).shape[0]):
        # per-frame extrinsics [C,T,4,4] with X [T,...,3]
        T = E.shape[1]
        if X.ndim < 2 or X.shape[0] != T:
            raise ValueError('per-frame extrinsics with T={} need points shaped (T, ..., 3), got {}'
                             .format(T, tuple(X.shape)))
        M = int(np.prod(X.shape[1:-1])) if X.ndim > 2 else 1
        E = E[:, :, None].expand(E.shape[0], T, M, 4, 4).reshape(E.shape[0], T * M, 4, 4)
    return X.reshape(-1, 3), E, lead


def _project_flat(P, X, E, eps=FISHEYE_EPS):
    """X [N,3], E [C,4,4] or [C,N,4,4] -> uv [C,N,2], depth [C,N]."""
    pc = _cam_points(E, X)
    z = pc[..., 2]
    zs = torch.where(torch.abs(z) >= eps, z, torch.full_like(z, eps))
    x, y = pc[..., 0] / zs, pc[..., 1] / zs
    xd, yd = _distort_norm(P, x, y)
    return _pixels(P, xd, yd), z


def _project_jac_flat(P, X, E, eps=FISHEYE_EPS):
    """uv [C,N,2], depth [C,N], J [C,N,2,3] = d uv / d X."""
    pc = _cam_points(E, X)
    z = pc[..., 2]
    zs = torch.where(torch.abs(z) >= eps, z, torch.full_like(z, eps))
    x, y = pc[..., 0] / zs, pc[..., 1] / zs
    xd, yd, (jxx, jxy, jyx, jyy) = _distort_norm(P, x, y, jac=True)
    uv = _pixels(P, xd, yd)
    K = P.K
    a, b, c, e = _col(K[:, 0, 0]), _col(K[:, 0, 1]), _col(K[:, 1, 0]), _col(K[:, 1, 1])
    # A (2x2 of K) @ Jd
    m00 = a * jxx + b * jyx; m01 = a * jxy + b * jyy
    m10 = c * jxx + e * jyx; m11 = c * jxy + e * jyy
    iz = 1 / zs
    # Jp = [[1/z, 0, -x/z], [0, 1/z, -y/z]]
    g0 = torch.stack([m00 * iz, m01 * iz, -(m00 * x + m01 * y) * iz], -1)   # [C,N,3] d u / d pc
    g1 = torch.stack([m10 * iz, m11 * iz, -(m10 * x + m11 * y) * iz], -1)
    R = E[..., :3, :3]
    if E.ndim == 3:
        J0 = torch.einsum('cni,cij->cnj', g0, R)
        J1 = torch.einsum('cni,cij->cnj', g1, R)
    else:
        J0 = torch.einsum('cni,cnij->cnj', g0, R)
        J1 = torch.einsum('cni,cnij->cnj', g1, R)
    return uv, z, torch.stack([J0, J1], -2)


def _undistort_flat(P, uv, n_iter=20, tol=1e-12, check_invertible=False, n_fixed=None):
    """uv [C,N,2] pixels -> normalised [C,N,2] (NaN in = NaN out)."""
    xd0, yd0 = _normalize_pixels(P, uv)
    finite = torch.isfinite(xd0) & torch.isfinite(yd0)
    xd = torch.where(finite, xd0, torch.zeros_like(xd0))
    yd = torch.where(finite, yd0, torch.zeros_like(yd0))
    has_b, has_k = _models_present(P)
    out_x = torch.zeros_like(xd)
    out_y = torch.zeros_like(yd)
    good = finite.clone()
    if has_b:
        bx, by, bok = _undistort_brown(P.dist, xd, yd, n_iter, tol, check_invertible, n_fixed)
        is_b = _col(P.model == 0)
        out_x = torch.where(is_b, bx, out_x); out_y = torch.where(is_b, by, out_y)
        good = good & (bok | ~is_b)
    if has_k:
        kx, ky, kok = _undistort_kb(P.dist, xd, yd, n_iter, check_invertible)
        is_k = _col(P.model == 1)
        out_x = torch.where(is_k, kx, out_x); out_y = torch.where(is_k, ky, out_y)
        good = good & (kok | ~is_k)
    out = torch.stack([out_x, out_y], -1)
    return torch.where(good[..., None], out, torch.full_like(out, float('nan')))


def _undistort_brown(d, xd, yd, n_iter, tol, check_invertible, n_fixed=None):
    # fixed-point warm start (OpenCV's iteration; exact for the radial-only rational model)
    k = [_col(d[:, i]) for i in range(8)]
    k1, k2, p1, p2, k3, k4, k5, k6 = k
    x, y = xd, yd
    n_fixed = min(5, n_iter) if n_fixed is None else n_fixed
    for _ in range(n_fixed):
        r2 = x * x + y * y
        icdist = (1 + r2 * (k4 + r2 * (k5 + r2 * k6))) / (1 + r2 * (k1 + r2 * (k2 + r2 * k3)))
        dx = 2 * p1 * x * y + p2 * (r2 + 2 * x * x)
        dy = p1 * (r2 + 2 * y * y) + 2 * p2 * x * y
        x = (xd - dx) * icdist
        y = (yd - dy) * icdist
    fx, fy = x, y
    # Newton
    for it in range(n_iter - n_fixed):
        ex, ey, (jxx, jxy, jyx, jyy) = _brown(x, y, d, jac=True)
        ex = ex - xd; ey = ey - yd
        det = jxx * jyy - jxy * jyx
        det = torch.where(torch.abs(det) > 1e-30, det, torch.full_like(det, 1e-30))
        sx = (jyy * ex - jxy * ey) / det
        sy = (jxx * ey - jyx * ex) / det
        x = x - sx; y = y - sy
        if it % 4 == 3:
            step = torch.nan_to_num(torch.abs(sx) + torch.abs(sy), nan=0.0, posinf=0.0)
            if step.numel() == 0 or float(step.max()) < tol:
                break
    # keep the better of fixed point and Newton (Newton can run off in non-invertible regions)
    rx, ry, (jxx, jxy, jyx, jyy) = _brown(x, y, d, jac=True)
    res_n = torch.hypot(rx - xd, ry - yd)
    fxd, fyd = _brown(fx, fy, d)
    res_f = torch.hypot(fxd - xd, fyd - yd)
    use_n = torch.isfinite(res_n) & ~(res_n > res_f)
    x = torch.where(use_n, x, fx); y = torch.where(use_n, y, fy)
    ok = torch.isfinite(x) & torch.isfinite(y)
    if check_invertible:
        res = torch.where(use_n, res_n, res_f)
        det = torch.where(use_n, jxx * jyy - jxy * jyx, torch.ones_like(res))
        if not bool(use_n.all()):
            _, _, (a, b, c, e) = _brown(x, y, d, jac=True)
            det = a * e - b * c
        # the solution must lie in the central, monotone part of the radial map (the outer
        # branch past a fold also has det > 0, as both radial factors are negative there)
        r2 = x * x + y * y
        num = 1 + r2 * (k1 + r2 * (k2 + r2 * k3))
        den = 1 + r2 * (k4 + r2 * (k5 + r2 * k6))
        dnum = k1 + r2 * (2 * k2 + 3 * k3 * r2)
        dden = k4 + r2 * (2 * k5 + 3 * k6 * r2)
        rad = num / den
        slope = rad + 2 * r2 * (dnum * den - num * dden) / (den * den)
        ok = ok & (res <= 1e-9) & (det > 0) & (rad > 0) & (slope > 0)
    return x, y, ok


def _undistort_kb(d, xd, yd, n_iter, check_invertible, eps=FISHEYE_EPS, tol=1e-8):
    td2 = xd * xd + yd * yd
    r_ok = td2 >= eps * eps
    thd = torch.sqrt(torch.where(r_ok, td2, torch.ones_like(td2)))
    thd_c = torch.clamp(thd, max=_HALF_PI)
    th = thd_c.clone()
    for _ in range(n_iter):
        f, fp = _kb_poly(th, d)
        f = f - thd_c
        fp = torch.where(torch.abs(fp) > eps, fp, torch.ones_like(fp))
        th = th - f / fp
    f, fp = _kb_poly(th, d)
    resid = torch.abs(f - thd_c)
    ok = torch.isfinite(th) & (th >= 0) & (resid <= tol) & (thd <= _HALF_PI) & (th < _HALF_PI)
    if check_invertible:
        ok = ok & (fp > 0)
    thc = torch.clamp(th, max=_HALF_PI - 1e-9)
    s = torch.where(r_ok, torch.tan(thc) / thd, torch.ones_like(th))
    return s * xd, s * yd, ok


# ------------------------------------------------------------------ public API

def _flat_uv(P, uv):
    uv = to_torch(uv, device=P.device, dtype=P.dtype)
    if uv.shape[0] != P.n_cams:
        raise ValueError('points have {} cameras on axis 0, expected {}'.format(uv.shape[0], P.n_cams))
    return uv.reshape(P.n_cams, -1, 2), uv.shape


@returns_like_input()
def project(P, X, extrinsics=None):
    """World points ``X [...,3]`` -> pixels ``[C,...,2]`` and depth ``[C,...]``.

    ``extrinsics`` overrides ``P``'s: ``[C,4,4]``, ``[C,N,4,4]`` (one per point) or
    ``[C,T,4,4]`` (per frame, with ``X`` shaped ``[T,...,3]``)."""
    X = to_torch(X, device=P.device, dtype=P.dtype)
    Xf, E, lead = _flatten_points(P, X, _ext_for(P, extrinsics))
    uv, z = _project_flat(P, Xf, E)
    C = P.n_cams
    return uv.reshape((C,) + tuple(lead) + (2,)), z.reshape((C,) + tuple(lead))


@returns_like_input()
def project_jacobian(P, X, extrinsics=None):
    """Like :func:`project` for ``X [N,3]``, plus the Jacobian ``J [C,N,2,3]`` of pixels w.r.t. X."""
    X = to_torch(X, device=P.device, dtype=P.dtype)
    Xf, E, lead = _flatten_points(P, X, _ext_for(P, extrinsics))
    return _project_jac_flat(P, Xf, E)


@returns_like_input()
def distort(P, xy):
    """Normalised image coordinates ``[C,...,2]`` -> pixels ``[C,...,2]``."""
    xy, shape = _flat_uv(P, xy)
    xd, yd = _distort_norm(P, xy[..., 0], xy[..., 1])
    return _pixels(P, xd, yd).reshape(shape)


@returns_like_input()
def undistort(P, uv, n_iter=20, tol=1e-12, check_invertible=False):
    """Pixels ``[C,...,2]`` -> undistorted normalised coordinates ``[C,...,2]`` (NaN = missing).

    ``n_iter`` bounds the solver iterations (brown: 5 fixed-point + Newton with early exit;
    kb: Newton on theta). ``check_invertible=True`` returns NaN where the distortion model is
    not locally invertible at the solution or did not converge."""
    uvf, shape = _flat_uv(P, uv)
    return _undistort_flat(P, uvf, n_iter=n_iter, tol=tol, check_invertible=check_invertible).reshape(shape)


def _reproj_err_flat(P, X, uv, E):
    proj, z = _project_flat(P, X, E)
    e = torch.linalg.norm(proj - uv, dim=-1)
    e = torch.where(z > 0, e, torch.full_like(e, float('inf')))
    missing = ~torch.isfinite(uv).all(-1)
    return torch.where(missing, torch.full_like(e, float('nan')), e)


@returns_like_input()
def reproj_err(P, X, uv, extrinsics=None):
    """Pixel reprojection error ``[C,...]`` of world points ``X [...,3]`` against ``uv [C,...,2]``.

    NaN where ``uv`` is missing; ``inf`` where the point is behind the camera."""
    X = to_torch(X, device=P.device, dtype=P.dtype)
    uvf, shape = _flat_uv(P, uv)
    Xf, E, lead = _flatten_points(P, X, _ext_for(P, extrinsics))
    return _reproj_err_flat(P, Xf, uvf, E).reshape(shape[:-1])
