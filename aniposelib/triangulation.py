"""Fast and robust multi-view triangulation on :class:`~aniposelib.geometry.PackedCameras`
(1.0, plan 01 C2/C3).

Every observation ``(camera c, point n)`` contributes a 4x4 normal-equation block of the
linear (DLT) system; the DLT of *any* subset of cameras is the sum of its blocks. Solving the
inhomogeneous 3x3 system (world origin recentred on the camera centroid, scale-aware ridge
``ridge * mean(diag)``, default 1e-12 so well-posed points are unbiased; closed-form adjugate solve) is exact, cheap, batched and well conditioned for far scenes.
Robust variants are thin layers on top of the blocks.

API
---
``dlt_blocks(P, und, weights=None, extrinsics=None, center=None) -> [C,N,4,4]``
    Blocks from undistorted normalised points ``und [C,N,2]`` (NaN = missing -> zero block).
    ``weights`` multiply the DLT *rows*, as in ``CameraGroup.triangulate`` (block weight w**2).
``solve_blocks(B, center=None, ridge=1e-12) -> [...,3]``
    Solve summed blocks; adds ``center`` back.
``triangulate_linear(P, points, weights=None, undistort=True, extrinsics=None, min_cams=2,
                     ridge=1e-12, chunk=None) -> p3d [...,3]``
    Weighted linear triangulation of ``points [C,...,2]`` (pixels, or normalised if
    ``undistort=False``). NaN where fewer than ``min_cams`` cameras observe the point.
``triangulate_robust(P, points, method='msac', ...) -> TriangulationResult``
    * ``'msac'``: every pair of (contributing) cameras observing a point is a hypothesis, scored
      by ``sum_c min(err_c^2, tau^2)``; the best hypothesis' inliers (err < tau) are refit and
      refined by Gauss-Newton on the pixel error (the inlier set may grow, never shrink).
    * ``'irls'``: iteratively reweighted least squares with Cauchy weights
      ``1 / (1 + (err/sigma)^2)``; *single-ray guard*: a point left with fewer than two
      observations of weight > ``min_weight`` is reset to uniform weights.
    * ``'reject'``: fit all, then drop observations with err > ``max(reject_px, rel_factor *
      median err of the point)``, plus the single worst view if it is above ``reject_px``, and
      refit (``n_iter`` rounds, never below ``max(2, min_inliers)`` views).
    ``contrib`` (bool ``[C]``) restricts which cameras may *define* points (passengers are
    still reported in ``errors``/``inliers``); points with fewer than two contributing views
    fall back to all views. ``scores [C,...]`` (1.1) are soft confidence weights (clipped to
    [0, 1]) on the squared pixel error in the hypothesis cost, the refit and the Gauss-Newton
    step; they never drop a view by themselves. ``check_invertible=True`` (1.1, default): lens
    fold / ray-consistency check (observations past the fold of a strong barrel distortion are
    not fit; inliers whose undistorted ray disagrees with the point are dropped and the point
    redone), formerly only in the ``CameraGroup`` wrappers. ``mask_out_of_bounds`` treats observations outside the image
    (``P.size``) as missing. ``chunk`` bounds memory (default: automatic from ``max_elems``).
``triangulate_possible(P, points, tau=10, min_inliers=2, ...) -> PossibleResult``
    ``points [C,N,K,2]``: up to K candidate detections per camera. Hypotheses are pairs
    ``(cam a, cand i, cam b, cand j)``; each camera scores its best candidate. No subset
    enumeration, memory bounded by chunking. ``PossibleResult.legacy()`` gives the
    ``(p3d, picked_vals, points_2d, errors)`` tuple of ``CameraGroup.triangulate_possible``.
``refine_points(P, X, uv, weights, n_iter=2)``
    Batched Gauss-Newton of 3D points on pixel error.

``TriangulationResult``: ``p3d [N,3]`` (NaN if < ``min_inliers``), ``inliers [C,N]`` bool,
``errors [C,N]`` pixel error of every observation (all cameras, NaN = missing),
``weights [C,N]``, ``n_inliers [N]``; ``to_dict()``; unpacks as ``p3d, errors = result``.
Leading point shapes are kept: ``points [C,N,J,2]`` gives ``p3d [N,J,3]``, ``errors [C,N,J]``.

numpy in -> numpy out; torch in -> detached torch (on ``P``'s device); ``return_grad=True``
keeps the graph (``triangulate_linear`` is differentiable w.r.t. points, weights and cameras).
"""

import itertools
from dataclasses import dataclass, fields
from typing import Any

import numpy as np
import torch

from ._types import returns_like_input, to_torch
from . import geometry as G

__all__ = ['dlt_blocks', 'solve_blocks', 'triangulate_linear', 'triangulate_robust',
           'triangulate_possible', 'refine_points', 'TriangulationResult', 'PossibleResult']

DEFAULT_MAX_ELEMS = 1 << 22   # per-chunk budget of [cameras x hypotheses] elements


def _np(v):
    if isinstance(v, torch.Tensor):
        v = v.detach().cpu().numpy()
    if isinstance(v, np.ndarray):
        return v.tolist()
    return v


@dataclass
class TriangulationResult:
    p3d: Any
    inliers: Any
    errors: Any
    weights: Any
    n_inliers: Any

    def __iter__(self):
        yield self.p3d
        yield self.errors

    def to_dict(self):
        return {f.name: _np(getattr(self, f.name)) for f in fields(self)}


@dataclass
class PossibleResult:
    p3d: Any          # [N,3]
    picked: Any       # [C,N] int: chosen candidate per camera, -1 if none (not an inlier)
    points_2d: Any    # [C,N,2] chosen 2D points (NaN where none)
    errors: Any       # [C,N] pixel error of the chosen candidates (NaN where none)
    n_inliers: Any    # [N]
    n_candidates: int

    def legacy(self):
        """``(p3d, picked_vals [C,N,K] bool, points_2d [C,N,2], mean_error [N])`` as returned by
        ``CameraGroup.triangulate_possible`` (mean error over picked views)."""
        picked, err = self.picked, self.errors
        if isinstance(picked, torch.Tensor):
            K = self.n_candidates
            pv = (picked[..., None] == torch.arange(K, device=picked.device)) & (picked[..., None] >= 0)
            e = torch.where(picked >= 0, torch.nan_to_num(err, nan=0.0), torch.zeros_like(err))
            mean = e.sum(0) / (picked >= 0).sum(0).clamp(min=1)
            mean = torch.where((picked >= 0).sum(0) > 0, mean, torch.full_like(mean, float('nan')))
        else:
            K = self.n_candidates
            pv = (picked[..., None] == np.arange(K)) & (picked[..., None] >= 0)
            with np.errstate(invalid='ignore', divide='ignore'):
                e = np.where(picked >= 0, np.nan_to_num(err, nan=0.0), 0.0)
                cnt = (picked >= 0).sum(0)
                mean = np.where(cnt > 0, e.sum(0) / np.maximum(cnt, 1), np.nan)
        return self.p3d, pv, self.points_2d, mean

    def to_dict(self):
        return {f.name: _np(getattr(self, f.name)) for f in fields(self)}


# ------------------------------------------------------------------ linear algebra kernels

def _solve3(A, b):
    """Batched symmetric 3x3 solve via the adjugate (elementwise; no batched-LAPACK limits)."""
    a00, a01, a02 = A[..., 0, 0], A[..., 0, 1], A[..., 0, 2]
    a11, a12, a22 = A[..., 1, 1], A[..., 1, 2], A[..., 2, 2]
    c00 = a11 * a22 - a12 * a12
    c01 = a02 * a12 - a01 * a22
    c02 = a01 * a12 - a02 * a11
    c11 = a00 * a22 - a02 * a02
    c12 = a01 * a02 - a00 * a12
    c22 = a00 * a11 - a01 * a01
    det = a00 * c00 + a01 * c01 + a02 * c02
    b0, b1, b2 = b[..., 0], b[..., 1], b[..., 2]
    return torch.stack([(c00 * b0 + c01 * b1 + c02 * b2) / det,
                        (c01 * b0 + c11 * b1 + c12 * b2) / det,
                        (c02 * b0 + c12 * b1 + c22 * b2) / det], -1)


def _default_center(E):
    """Camera centroid: [3] for E [C,4,4], [N,3] for E [C,N,4,4]."""
    R, t = E[..., :3, :3], E[..., :3, 3]
    c = -(R.transpose(-1, -2) @ t[..., None])[..., 0]
    return c.mean(0)


def _recentred_rows(E, center):
    """Top 3 rows of the extrinsics with the world origin moved to ``center``: [C,(N,)3,4]."""
    R, t = E[..., :3, :3], E[..., :3, 3]
    if E.ndim == 3:
        t2 = t + torch.einsum('cij,j->ci', R, center)
    else:
        t2 = t + torch.einsum('cnij,nj->cni', R, center)
    return torch.cat([R, t2[..., None]], -1)


def _blocks(Pm, und, w2):
    """Pm [C,3,4] or [C,N,3,4]; und [C,N,2] finite; w2 [C,N] block weights -> [C,N,4,4]."""
    if Pm.ndim == 3:
        P0, P1, P2 = Pm[:, None, 0], Pm[:, None, 1], Pm[:, None, 2]
    else:
        P0, P1, P2 = Pm[:, :, 0], Pm[:, :, 1], Pm[:, :, 2]
    ax = und[..., 0, None] * P2 - P0
    ay = und[..., 1, None] * P2 - P1
    M = ax[..., :, None] * ax[..., None, :] + ay[..., :, None] * ay[..., None, :]
    return M * w2[..., None, None]


def _solve(B, center, ridge):
    A = B[..., :3, :3]
    reg = ridge * A.diagonal(dim1=-2, dim2=-1).mean(-1)
    reg = torch.where(reg > 0, reg, torch.zeros_like(reg))
    A = A + reg[..., None, None] * torch.eye(3, dtype=A.dtype, device=A.device)
    X = _solve3(A, -B[..., :3, 3])
    return X if center is None else X + center


def _prep_ext(P, extrinsics, n_points):
    E = P.extrinsics() if extrinsics is None else to_torch(extrinsics, device=P.device, dtype=P.dtype)
    if E.ndim == 4 and E.shape[1] != n_points:
        T = E.shape[1]
        if n_points % T != 0:
            raise ValueError('per-frame extrinsics with T={} do not match {} points'.format(T, n_points))
        M = n_points // T
        E = E[:, :, None].expand(E.shape[0], T, M, 4, 4).reshape(E.shape[0], n_points, 4, 4)
    return E


def _sub_ext(E, idx):
    return E if E.ndim == 3 else E[:, idx]


def _sub_center(center, idx):
    return center if center.ndim == 1 else center[idx]


def _prep_points(P, points):
    pts = to_torch(points, device=P.device, dtype=P.dtype)
    if pts.shape[0] != P.n_cams:
        raise ValueError('points have {} cameras on axis 0, expected {}'.format(pts.shape[0], P.n_cams))
    if pts.shape[-1] != 2:
        raise ValueError('points must end in a coordinate axis of size 2, got {}'.format(tuple(pts.shape)))
    lead = pts.shape[1:-1]
    return pts.reshape(P.n_cams, -1, 2), lead


@returns_like_input(arg=0)
def dlt_blocks(P, und, weights=None, extrinsics=None, center=None):
    """DLT normal blocks ``[C,N,4,4]`` of undistorted points ``und [C,N,2]`` in the frame recentred
    on ``center`` (default: no recentring, i.e. world coordinates). ``weights [C,N]`` multiply the
    DLT rows (block weight ``w**2``); missing points give zero blocks."""
    und = to_torch(und, device=P.device, dtype=P.dtype)
    C, N = und.shape[:2]
    E = _prep_ext(P, extrinsics, N)
    if center is None:
        center = torch.zeros(3, dtype=P.dtype, device=P.device)
    else:
        center = to_torch(center, device=P.device, dtype=P.dtype)
    valid = torch.isfinite(und).all(-1)
    w2 = valid.to(P.dtype)
    if weights is not None:
        w = to_torch(weights, device=P.device, dtype=P.dtype)
        w2 = w2 * torch.nan_to_num(w) ** 2
    return _blocks(_recentred_rows(E, center), torch.nan_to_num(und), w2)


@returns_like_input(arg=0)
def solve_blocks(B, center=None, ridge=1e-12):
    """Solve (summed) DLT blocks ``[...,4,4]`` for 3D points ``[...,3]`` (inhomogeneous, ridge
    ``ridge * mean(diag)``), adding ``center`` back. NaN for singular blocks."""
    B = to_torch(B)
    if center is not None:
        center = to_torch(center, device=B.device, dtype=B.dtype)
    return _solve(B, center, ridge)


# ------------------------------------------------------------------ linear triangulation

def _chunk_bounds(pts, method, n_pairs, chunk, max_elems):
    """``[(s0, s1)]`` point chunks of :func:`_robust_flat`. MSAC chunks are sized by the pair
    hypotheses the points actually have on a GPU (``v (v - 1) / 2`` for ``v`` observing cameras,
    not all camera pairs), so sparse observations get few large chunks; at most ``4 max_elems / 16C``
    points per chunk (the ``[C,n,4,4]`` DLT blocks). Points are independent, so the chunking
    never changes the result."""
    C, N = pts.shape[:2]
    if N == 0:
        return []
    # on the CPU small chunks stay in cache (and are what 1.1.0 did); adaptive chunks are for
    # GPUs, where every chunk costs a few hundred kernel launches
    if chunk is not None or method != 'msac' or pts.device.type == 'cpu':
        step = _auto_chunk(C * max(1, n_pairs) if method == 'msac' else C * 16, chunk, max_elems)
        return [(s0, min(N, s0 + step)) for s0 in range(0, N, step)]
    v = torch.isfinite(pts).all(-1).sum(0)
    cost = (C * torch.clamp(v * (v - 1) // 2, min=1)).cpu().numpy().astype(np.int64)
    cap = max(1, int(4 * max_elems // (16 * C)))
    out, s0, cum = [], 0, np.cumsum(cost)
    while s0 < N:
        base = cum[s0 - 1] if s0 > 0 else 0
        s1 = int(np.searchsorted(cum, base + max_elems, side='right'))
        s1 = min(N, max(s1, s0 + 1), s0 + cap)
        out.append((s0, s1))
        s0 = s1
    return out


def _auto_chunk(n_per_point, chunk, max_elems):
    if chunk is not None:
        return max(1, int(chunk))
    return max(1, int(max_elems // max(1, n_per_point)))


def _valid_obs(P, pts, und, mask_out_of_bounds):
    valid = torch.isfinite(pts).all(-1) & torch.isfinite(und).all(-1)
    if mask_out_of_bounds:
        w, h = P.size[:, 0, None], P.size[:, 1, None]
        known = torch.isfinite(w) & torch.isfinite(h)
        inb = (pts[..., 0] >= 0) & (pts[..., 0] < w) & (pts[..., 1] >= 0) & (pts[..., 1] < h)
        valid = valid & (inb | ~known)
    return valid


def _linear_flat(P, und, w2, E, center, ridge, min_cams):
    """und [C,N,2] (NaN-free where w2 > 0), w2 [C,N] block weights -> [N,3]."""
    B = _blocks(_recentred_rows(E, center), torch.nan_to_num(und), w2).sum(0)
    X = _solve(B, center, ridge)
    enough = (w2 > 0).sum(0) >= min_cams
    return torch.where(enough[:, None], X, torch.full_like(X, float('nan')))


@returns_like_input(arg=0)
def triangulate_linear(P, points, weights=None, undistort=True, extrinsics=None, min_cams=2,
                       ridge=1e-12, chunk=None, max_elems=DEFAULT_MAX_ELEMS):
    """Linear (block DLT, recentred, ridge) triangulation of ``points [C,...,2]`` -> ``[...,3]``.

    ``weights [C,...]`` multiply the DLT rows (as ``CameraGroup.triangulate``). ``extrinsics``
    overrides ``P``'s: ``[C,4,4]``, per point ``[C,N,4,4]``, or per frame ``[C,T,4,4]`` with
    points shaped ``[C,T,...,2]``."""
    pts, lead = _prep_points(P, points)
    C, N, _ = pts.shape
    E = _prep_ext(P, extrinsics, N)
    if weights is not None:
        weights = to_torch(weights, device=P.device, dtype=P.dtype).reshape(C, N)
    out = []
    step = _auto_chunk(C * 16, chunk, max_elems)
    for s0 in range(0, N, step):
        s1 = min(N, s0 + step)
        p = pts[:, s0:s1]
        und = G._undistort_flat(P, p) if undistort else p
        w2 = torch.isfinite(und).all(-1).to(P.dtype)
        if weights is not None:
            w2 = w2 * torch.nan_to_num(weights[:, s0:s1]) ** 2
        Es = E if E.ndim == 3 else E[:, s0:s1]
        center = _default_center(Es)
        out.append(_linear_flat(P, und, w2, Es, center, ridge, min_cams))
    X = torch.cat(out, 0) if out else torch.zeros(0, 3, dtype=P.dtype, device=P.device)
    return X.reshape(tuple(lead) + (3,))


# ------------------------------------------------------------------ Gauss-Newton refinement

def _refine_flat(P, X, uv, w, E, n_iter=2, damping=1e-6):
    """Weighted GN on pixel error. X [N,3], uv [C,N,2], w [C,N] (0 = ignore)."""
    valid = torch.isfinite(uv).all(-1) & (w > 0) & torch.isfinite(X).all(-1)[None]
    uv0 = torch.nan_to_num(uv)
    eye = torch.eye(3, dtype=X.dtype, device=X.device)
    X = torch.nan_to_num(X)
    for _ in range(n_iter):
        proj, z, J = G._project_jac_flat(P, X, E)
        res = proj - uv0
        ok = valid & (z > 0) & torch.isfinite(res).all(-1) & torch.isfinite(J).flatten(-2).all(-1)
        wk = torch.where(ok, w, torch.zeros_like(w))
        J = torch.where(ok[..., None, None], J, torch.zeros_like(J))
        res = torch.where(ok[..., None], res, torch.zeros_like(res))
        H = torch.einsum('cnki,cnkj,cn->nij', J, J, wk)
        g = torch.einsum('cnki,cnk,cn->ni', J, res, wk)
        H = H + (damping * H.diagonal(dim1=-2, dim2=-1).mean(-1)[:, None, None] + 1e-300) * eye
        dx = _solve3(H, g)
        good = torch.isfinite(dx).all(-1) & ((ok.sum(0)) >= 2)
        X = torch.where(good[:, None], X - dx, X)
    return X


@returns_like_input(arg=0)
def refine_points(P, X, uv, weights=None, n_iter=2, extrinsics=None):
    """Gauss-Newton refinement of 3D points ``X [N,3]`` on the weighted pixel error against
    ``uv [C,N,2]``. Points with fewer than two usable observations are left unchanged."""
    X = to_torch(X, device=P.device, dtype=P.dtype).reshape(-1, 3)
    uv = to_torch(uv, device=P.device, dtype=P.dtype).reshape(P.n_cams, -1, 2)
    E = _prep_ext(P, extrinsics, X.shape[0])
    w = torch.ones(uv.shape[:2], dtype=P.dtype, device=P.device) if weights is None else \
        to_torch(weights, device=P.device, dtype=P.dtype).reshape(uv.shape[:2])
    nanX = ~torch.isfinite(X).all(-1)
    out = _refine_flat(P, X, uv, w, E, n_iter=n_iter)
    return torch.where(nanX[:, None], torch.full_like(out, float('nan')), out)


# ------------------------------------------------------------------ robust triangulation

def _pairs(C, device):
    pairs = list(itertools.combinations(range(C), 2))
    pa = torch.tensor([a for a, _ in pairs], dtype=torch.long, device=device)
    pb = torch.tensor([b for _, b in pairs], dtype=torch.long, device=device)
    return pa, pb


def _definers(valid, contrib):
    if contrib is None:
        return valid
    d = valid & contrib[:, None]
    few = d.sum(0) < 2
    return torch.where(few[None], valid, d)


def _msac_chunk(P, p2, und, valid, defn, B, E, center, tau, n_refine, pa, pb, ridge, sw=None):
    C, n = valid.shape
    dev, dt = P.device, P.dtype
    tau2 = tau * tau
    si, ni = torch.nonzero(defn[pa] & defn[pb], as_tuple=True)
    X = torch.full((n, 3), float('nan'), dtype=dt, device=dev)
    inl = torch.zeros((C, n), dtype=torch.bool, device=dev)
    if si.numel() == 0:
        return X, inl
    cen_h = _sub_center(center, ni)
    Xh = _solve(B[pa[si], ni] + B[pb[si], ni], cen_h, ridge)
    vh = valid[:, ni]
    # errors are only needed where the point is observed (the rest is ignored below)
    eh = G._reproj_err_masked(P, Xh, p2[:, ni], _sub_ext(E, ni), vh)
    big = torch.full_like(eh, tau2)
    capped = torch.minimum(torch.nan_to_num(eh, nan=tau, posinf=tau) ** 2, big)
    if sw is not None:
        capped = capped * sw[:, ni]
    cost = torch.where(vh, capped, torch.zeros_like(eh)).sum(0)
    best = torch.full((n,), float('inf'), dtype=dt, device=dev).scatter_reduce(0, ni, cost, reduce='amin')
    is_best = cost <= best[ni]
    H = ni.numel()
    hid = torch.arange(H, device=dev)
    first = torch.full((n,), H, dtype=torch.long, device=dev).scatter_reduce(
        0, ni[is_best], hid[is_best], reduce='amin')
    pts = torch.nonzero(first < H)[:, 0]
    hsel = first[pts]
    inl_p = (eh[:, hsel] < tau) & vh[:, hsel]
    fit = inl_p & defn[:, pts]
    # a winning pair is always an inlier pair unless tau is tiny; keep it as the fallback
    pair_mask = torch.zeros_like(fit)
    pair_mask[pa[si[hsel]], torch.arange(len(pts), device=dev)] = True
    pair_mask[pb[si[hsel]], torch.arange(len(pts), device=dev)] = True
    fit = torch.where((fit.sum(0) >= 2)[None], fit, pair_mask)
    Ep = _sub_ext(E, pts)
    sp = torch.ones_like(fit, dtype=dt) if sw is None else sw[:, pts]
    Xp = _solve(torch.einsum('cn,cnij->nij', fit.to(dt) * sp, B[:, pts]), _sub_center(center, pts), ridge)
    p2s = p2[:, pts]
    for _ in range(n_refine):
        Xp = _refine_flat(P, Xp, p2s, fit.to(dt) * sp, Ep, n_iter=2)
        e = G._reproj_err_flat(P, Xp, p2s, Ep)
        inl2 = (e < tau) & valid[:, pts]
        fit2 = inl2 & defn[:, pts]
        grow = fit2.sum(0) >= fit.sum(0)
        inl_p = torch.where(grow[None], inl2, inl_p)
        fit = torch.where(grow[None], fit2, fit)
    X[pts] = Xp
    inl[:, pts] = inl_p
    return X, inl


def _pixel_weights(P, X, E):
    """(f / z)^2 per observation [C,N]: turns DLT (algebraic) residuals into ~pixel residuals."""
    pc = G._cam_points(E, torch.nan_to_num(X))
    z = pc[..., 2].abs()
    f = 0.5 * (P.K[:, 0, 0].abs() + P.K[:, 1, 1].abs())
    w = (f[:, None] / z.clamp(min=1e-12)) ** 2
    w = w / w.mean(0, keepdim=True).clamp(min=1e-300)
    return torch.where(torch.isfinite(w), w, torch.ones_like(w))


def _irls_chunk(P, p2, und, valid, defn, E, center, sigma, n_iter, min_weight, ridge, sw=None):
    """Cauchy IRLS. The scale starts at the point's median residual and shrinks to ``sigma``
    (graduated), so a start dragged by gross outliers does not trip the single-ray guard."""
    dt = P.dtype
    base = defn.to(dt) if sw is None else defn.to(dt) * sw
    w = base.clone()
    X = _linear_flat(P, und, w, E, center, ridge, 2)
    for it in range(n_iter):
        e = G._reproj_err_flat(P, X, p2, E)
        mag = torch.nan_to_num(e, nan=0.0, posinf=1e30)
        med = torch.nanmedian(torch.where(defn, mag, torch.full_like(mag, float('nan'))), dim=0).values
        scale = torch.clamp(torch.nan_to_num(med, nan=sigma), min=sigma)
        if it >= n_iter // 2:
            scale = torch.full_like(scale, sigma)
        w = base / (1.0 + (mag / scale[None]) ** 2)
        strong = ((w > min_weight) & defn).sum(0)
        w = torch.where((strong < 2)[None], base, w)
        X = _linear_flat(P, und, w * _pixel_weights(P, X, E), E, center, ridge, 2)
    return X, w


def _reject_chunk(P, p2, und, valid, defn, E, center, reject_px, rel_factor, n_iter, min_inliers, ridge,
                  sw=None):
    dt = P.dtype
    sc = 1.0 if sw is None else sw
    keep = defn.clone()
    X = _linear_flat(P, und, keep.to(dt) * sc, E, center, ridge, 2)
    floor = max(2, int(min_inliers))
    for _ in range(n_iter):
        e = G._reproj_err_flat(P, X, p2, E)
        ek = torch.where(keep, torch.nan_to_num(e, nan=float('inf')), torch.full_like(e, float('nan')))
        med = torch.nanmedian(ek, dim=0).values
        thr = torch.clamp(rel_factor * med, min=reject_px)
        ef = torch.where(keep, torch.nan_to_num(e, nan=float('inf')), torch.full_like(e, -1.0))
        bad = keep & ~(ef <= thr[None])
        # and the single worst view (disco prune_groups), which a dragged fit can hide from the
        # relative threshold
        worst = ef.argmax(0)
        wv = ef.gather(0, worst[None])[0]
        drop = (wv > reject_px) & (keep.sum(0) > floor)
        bad[worst[drop], torch.nonzero(drop)[:, 0]] = True
        new = keep & ~bad
        # never drop below `floor` defining views: keep the best ones instead
        short = (new.sum(0) < floor) & (keep.sum(0) >= floor)
        if bool(short.any()):
            ranked = torch.where(keep, torch.nan_to_num(e, nan=float('inf')), torch.full_like(e, float('inf')))
            order = ranked.argsort(0)
            rank = torch.empty_like(order)
            rank.scatter_(0, order, torch.arange(order.shape[0], device=order.device)[:, None].expand_as(order))
            best = (rank < floor) & keep
            new = torch.where(short[None], best, new)
        new = torch.where((keep.sum(0) < floor)[None], keep, new)
        if bool((new == keep).all()):
            break
        keep = new
        X = _linear_flat(P, und, keep.to(dt) * sc * _pixel_weights(P, X, E), E, center, ridge, 2)
    return X, keep


def _score_weights(scores, C, N, dev, dt):
    """Soft observation weights from confidences: clipped to [0, 1], NaN -> 0."""
    if scores is None:
        return None
    s = to_torch(scores, device=dev, dtype=dt).reshape(C, N)
    return torch.nan_to_num(s, nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)


def _mask_noninvertible(P, pts, return_und=False):
    """NaN out pixel observations ``pts [C,N,2]`` where the lens model is not invertible (past
    the fold of a strong barrel distortion): their undistorted rays are wrong and would bias the
    linear refit. ``return_und=True`` also returns the undistorted points (NaN where masked),
    which are the undistortion of the masked points."""
    und = G._undistort_flat(P, pts, check_invertible=True)
    bad = ~torch.isfinite(und).all(-1)
    out = torch.where(bad[..., None], torch.full_like(pts, float('nan')), pts)
    return (out, und) if return_und else out


def _ray_inconsistent(P, X, E, pts, tau, und=None):
    """[C,N] True where an observation's undistorted ray disagrees with the direction to the
    point X (normalised distance > 3 tau / f) although its pixel error may be small: the
    projection lies past the fold of a strong barrel distortion, where the pixel has a second
    (central) preimage. Such rays bias the linear refit. ``und``: the undistorted ``pts`` if
    already known."""
    if und is None:
        und = G._undistort_flat(P, pts)
    pc = G._cam_points(E, torch.nan_to_num(X))
    z = pc[..., 2]
    zs = torch.where(z.abs() > 1e-12, z, torch.full_like(z, 1e-12))
    xn = pc[..., :2] / zs[..., None]
    f = 0.5 * (P.K[:, 0, 0].abs() + P.K[:, 1, 1].abs())
    d = torch.linalg.norm(und - xn, dim=-1)
    _, _, (a, b, c, e) = G._distort_norm(P, xn[..., 0], xn[..., 1], jac=True)
    return (d * f[:, None] > 3 * tau) | ~(a * e - b * c > 0)


def _robust_flat(P, pts, E, method, weights, sw, contrib, tau, min_inliers, n_refine, sigma,
                 n_iter, min_weight, reject_px, rel_factor, mask_out_of_bounds, undistort,
                 ridge, chunk, max_elems, und_all=None):
    """Core of :func:`triangulate_robust` on flat ``pts [C,N,2]`` -> (p3d, inliers, errors,
    weights, n_inliers). ``sw [C,N]`` (or None): soft score weights in [0, 1]. ``und_all``:
    the undistorted ``pts`` if already known (else computed per chunk)."""
    C, N, _ = pts.shape
    dev, dt = P.device, P.dtype
    Pr = P if undistort else _identity_intrinsics(P)
    p3d = torch.full((N, 3), float('nan'), dtype=dt, device=dev)
    inliers = torch.zeros((C, N), dtype=torch.bool, device=dev)
    errors = torch.full((C, N), float('nan'), dtype=dt, device=dev)
    wout = torch.zeros((C, N), dtype=dt, device=dev)

    pa, pb = _pairs(C, dev)
    floor = max(2, int(min_inliers))
    for s0, s1 in _chunk_bounds(pts, method, len(pa), chunk, max_elems):
        p2 = pts[:, s0:s1]
        if not undistort:
            und = p2
        elif und_all is not None:
            und = und_all[:, s0:s1]
        else:
            und = G._undistort_flat(P, p2)
        valid = _valid_obs(Pr, p2, und, mask_out_of_bounds and undistort)
        if weights is not None:
            valid = valid & (torch.nan_to_num(weights[:, s0:s1]) > 0)
        s = None if sw is None else sw[:, s0:s1]
        # zero-score observations are passengers: reported, never used to fit
        defn = _definers(valid if s is None else valid & (s > 0), contrib)
        sc = torch.ones_like(p2[..., 0]) if s is None else s
        Es = E if E.ndim == 3 else E[:, s0:s1]
        center = _default_center(Es)
        und0 = torch.where(valid[..., None], und, torch.full_like(und, float('nan')))
        p2v = torch.where(valid[..., None], p2, torch.full_like(p2, float('nan')))
        if method == 'msac':
            B = _blocks(_recentred_rows(Es, center), torch.nan_to_num(und0), defn.to(dt))
            X, inl = _msac_chunk(Pr, p2v, und0, valid, defn, B, Es, center, tau, n_refine, pa, pb,
                                 ridge, sw=s)
            fitw = (inl & defn).to(dt) * sc
        elif method == 'irls':
            und_d = torch.where(defn[..., None], und0, torch.full_like(und0, float('nan')))
            X, fitw = _irls_chunk(Pr, p2v, torch.nan_to_num(und_d), valid, defn, Es, center, sigma,
                                  n_iter, min_weight, ridge, sw=s)
            if weights is not None:
                fitw = fitw * torch.nan_to_num(weights[:, s0:s1])
            if n_refine:
                X = _refine_flat(Pr, X, p2v, fitw, Es, n_iter=2 * n_refine)
        else:
            X, keep = _reject_chunk(Pr, p2v, torch.nan_to_num(und0), valid, defn, Es, center,
                                    reject_px, rel_factor, n_iter, min_inliers, ridge, sw=s)
            fitw = keep.to(dt) * sc
            if n_refine:
                X = _refine_flat(Pr, X, p2v, fitw, Es, n_iter=2 * n_refine)
        e = G._reproj_err_flat(Pr, torch.nan_to_num(X), p2v, Es)
        e = torch.where(torch.isfinite(X).all(-1)[None], e, torch.full_like(e, float('nan')))
        inl = valid & (torch.nan_to_num(e, nan=float('inf')) < tau)
        if method == 'msac':
            inl = inl | ((fitw > 0) & valid)  # GN can push a fitted view just past tau
        n_def = (inl & defn).sum(0)
        good = (n_def >= floor) & torch.isfinite(X).all(-1)
        p3d[s0:s1] = torch.where(good[:, None], X, torch.full_like(X, float('nan')))
        inliers[:, s0:s1] = inl & good[None]
        errors[:, s0:s1] = e
        wout[:, s0:s1] = fitw
    return p3d, inliers, errors, wout


@returns_like_input(arg=0)
def triangulate_robust(P, points, method='msac', *, tau=10.0, min_inliers=2, n_refine=2,
                       sigma=8.0, n_iter=None, min_weight=0.1,
                       reject_px=20.0, rel_factor=5.0,
                       contrib=None, weights=None, scores=None, mask_out_of_bounds=True,
                       undistort=True, check_invertible=True,
                       extrinsics=None, ridge=1e-12, chunk=None, max_elems=DEFAULT_MAX_ELEMS,
                       _und=None):
    """Robust triangulation of pixel observations ``points [C,...,2]`` (see module docstring).

    Parameters by method:
      msac:   ``tau`` (inlier threshold, px), ``n_refine`` (GN rounds with inlier update)
      irls:   ``sigma`` (Cauchy scale, px), ``n_iter`` (default 6), ``min_weight`` (guard)
      reject: ``reject_px``, ``rel_factor``, ``n_iter`` (default 4)
    Common: ``min_inliers`` (points with fewer inlier defining views are NaN), ``contrib``
    (bool [C], cameras allowed to define points), ``weights [C,...]`` (0 removes an
    observation; for 'irls' they scale the robust weights), ``mask_out_of_bounds``,
    ``undistort`` (False: ``points`` are already normalised, errors are then in normalised
    units), ``extrinsics`` (as in :func:`triangulate_linear`).
    ``scores [C,...]``: detection confidences used as **soft weights** (clipped to [0, 1],
    NaN = 0) on the squared pixel error: in the MSAC hypothesis cost ``sum_c s_c min(e_c^2,
    tau^2)``, the inlier refit (DLT blocks weighted by ``s``) and the Gauss-Newton refinement
    (and the IRLS / reject fits). Low-score views count less but are never dropped for their
    score alone; score-0 views are reported but never fit. Geometric inlier tests (``tau``)
    ignore the scores.
    ``check_invertible=True`` (default; skipped when ``undistort=False``): observations where
    the lens model is not invertible (past the fold of a strong barrel distortion) are not used
    to fit, and inliers whose undistorted ray disagrees with the fitted point (a second,
    central preimage of the pixel) are dropped and their points triangulated again. Their
    errors are still reported. Identical to ``check_invertible=False`` when no lens folds.
    ``inliers`` are observations with error < ``tau`` (all methods), ``errors`` are reported for
    every observation, ``weights`` are the final fit weights (IRLS weights or the fit mask,
    times the scores).
    """
    if method not in ('msac', 'irls', 'reject'):
        raise ValueError("method must be 'msac', 'irls' or 'reject', got {!r}".format(method))
    if n_iter is None:
        n_iter = 6 if method == 'irls' else 4
    raw, lead = _prep_points(P, points)
    C, N, _ = raw.shape
    dev, dt = P.device, P.dtype
    E = _prep_ext(P, extrinsics, N)
    if weights is not None:
        weights = to_torch(weights, device=dev, dtype=dt).reshape(C, N)
    sw = _score_weights(scores, C, N, dev, dt)
    if contrib is not None:
        contrib = torch.as_tensor(np.asarray(contrib) if not isinstance(contrib, torch.Tensor) else contrib,
                                  device=dev).to(torch.bool).reshape(C)
    check = bool(check_invertible) and undistort
    # ``_und`` (internal): the undistorted points, computed by the caller (with
    # ``check_invertible`` as here), e.g. once for all the subsets of a LOO evaluation
    und = None if _und is None or not undistort else to_torch(_und, device=dev, dtype=dt).reshape(C, N, 2)
    if check:
        # one undistortion, shared by the fold mask, the fit and the ray check
        if und is None:
            pts, und = _mask_noninvertible(P, raw, return_und=True)
        else:
            pts = torch.where(torch.isfinite(und).all(-1)[..., None], raw, torch.full_like(raw, float('nan')))
    else:
        pts = raw
    kw = dict(method=method, contrib=contrib, tau=tau, min_inliers=min_inliers, n_refine=n_refine,
              sigma=sigma, n_iter=n_iter, min_weight=min_weight, reject_px=reject_px,
              rel_factor=rel_factor, mask_out_of_bounds=mask_out_of_bounds, undistort=undistort,
              ridge=ridge, chunk=chunk, max_elems=max_elems)
    p3d, inliers, errors, wout = _robust_flat(P, pts, E, weights=weights, sw=sw, und_all=und, **kw)
    if check and N > 0:
        # an inlier explained by a point past its lens fold had a wrong ray in the linear
        # refit: drop those observations and triangulate the affected points again
        # chunked over points (independent): bounded memory for millions of points
        step = max(1, int(4 * max_elems // max(1, C)))
        fold = torch.cat([_ray_inconsistent(P, p3d[s:s + step], _sub_ext(E, slice(s, s + step)),
                                            pts[:, s:s + step], float(tau), und=und[:, s:s + step])
                          for s in range(0, N, step)], 1) & inliers & torch.isfinite(p3d).all(-1)[None]
        redo = torch.nonzero(fold.any(0))[:, 0]
        if redo.numel() > 0:
            sub = pts[:, redo]
            sub = torch.where(fold[:, redo, None], torch.full_like(sub, float('nan')), sub)
            usub = und[:, redo]
            usub = torch.where(fold[:, redo, None], torch.full_like(usub, float('nan')), usub)
            r = _robust_flat(P, sub, _sub_ext(E, redo),
                             weights=None if weights is None else weights[:, redo],
                             sw=None if sw is None else sw[:, redo], und_all=usub, **kw)
            p3d[redo] = r[0]
            inliers[:, redo] = r[1]
            errors[:, redo] = r[2]
            wout[:, redo] = r[3]
        # report errors for every observation, including the masked ones
        masked = _valid_obs(P, raw, raw, mask_out_of_bounds) & ~torch.isfinite(pts).all(-1)
        if weights is not None:
            masked = masked & (torch.nan_to_num(weights) > 0)
        cols = torch.nonzero(masked.any(0))[:, 0]
        if cols.numel() > 0:
            # only the points with masked observations
            Xc = p3d[cols]
            e = G._reproj_err_flat(P, torch.nan_to_num(Xc), raw[:, cols], _sub_ext(E, cols))
            e = torch.where(torch.isfinite(Xc).all(-1)[None], e, torch.full_like(e, float('nan')))
            errors[:, cols] = torch.where(masked[:, cols], e, errors[:, cols])
    n_inl = inliers.sum(0)
    return TriangulationResult(
        p3d=p3d.reshape(tuple(lead) + (3,)),
        inliers=inliers.reshape((C,) + tuple(lead)),
        errors=errors.reshape((C,) + tuple(lead)),
        weights=wout.reshape((C,) + tuple(lead)),
        n_inliers=n_inl.reshape(tuple(lead)))


def _identity_intrinsics(P):
    """Packed cameras with K = I, no distortion, no offset (for already-normalised points)."""
    C = P.n_cams
    K = torch.eye(3, dtype=P.dtype, device=P.device).expand(C, 3, 3).clone()
    return G.PackedCameras(K=K, dist=torch.zeros_like(P.dist), model=torch.zeros_like(P.model),
                           rvec=P.rvec, tvec=P.tvec, ext=P.ext,
                           offset=torch.zeros_like(P.offset), size=None, names=P.names)


# ------------------------------------------------------------------ multi-candidate kernel

@returns_like_input(arg=0)
def triangulate_possible(P, points, tau=10.0, min_inliers=2, n_refine=2, undistort=True,
                         contrib=None, extrinsics=None, ridge=1e-12, chunk=None,
                         max_elems=DEFAULT_MAX_ELEMS, check_invertible=True):
    """Triangulate with up to ``K`` candidate detections per camera, ``points [C,N,K,2]``.

    Each hypothesis is a camera pair with one candidate each; a camera scores its closest
    candidate (``min_k err``), capped at ``tau`` (MSAC). The winner's inlier candidates are
    refit and refined by Gauss-Newton. Returns :class:`PossibleResult`.
    ``check_invertible`` as in :func:`triangulate_robust`: candidates past a lens fold are not
    used, and chosen views whose ray disagrees with the point are dropped and the point redone."""
    pts = to_torch(points, device=P.device, dtype=P.dtype)
    if pts.ndim != 4 or pts.shape[0] != P.n_cams:
        raise ValueError('points must be [C,N,K,2], got {}'.format(tuple(pts.shape)))
    C, N, K, _ = pts.shape
    E = _prep_ext(P, extrinsics, N)
    if contrib is not None:
        contrib = torch.as_tensor(np.asarray(contrib) if not isinstance(contrib, torch.Tensor) else contrib,
                                  device=P.device).to(torch.bool).reshape(C)
    check = bool(check_invertible) and undistort
    if check:
        pts = _mask_noninvertible(P, pts.reshape(C, -1, 2)).reshape(pts.shape)
    kw = dict(tau=tau, min_inliers=min_inliers, n_refine=n_refine, undistort=undistort,
              contrib=contrib, ridge=ridge, chunk=chunk, max_elems=max_elems)
    res = _possible_core(P, pts, E, **kw)
    if check and N > 0:
        # chosen views whose ray disagrees with the point (past the lens fold) biased the refit;
        # drop that camera for those points and redo them
        chosen = res.picked >= 0
        bad = _ray_inconsistent(P, res.p3d, E, res.points_2d, float(tau)) & chosen \
            & torch.isfinite(res.p3d).all(-1)[None]
        redo = torch.nonzero(bad.any(0))[:, 0]
        if redo.numel() > 0:
            sub = pts[:, redo]
            sub = torch.where(bad[:, redo, None, None], torch.full_like(sub, float('nan')), sub)
            r2 = _possible_core(P, sub, _sub_ext(E, redo), **kw)
            res.p3d[redo] = r2.p3d
            res.picked[:, redo] = r2.picked
            res.points_2d[:, redo] = r2.points_2d
            res.errors[:, redo] = r2.errors
            res.n_inliers[redo] = r2.n_inliers
    return res


def _possible_core(P, pts, E, tau, min_inliers, n_refine, undistort, contrib, ridge, chunk,
                   max_elems):
    C, N, K, _ = pts.shape
    dev, dt = P.device, P.dtype
    Pr = P if undistort else _identity_intrinsics(P)
    pa, pb = _pairs(C, dev)
    # candidate pairs (a, i, b, j) for all camera pairs
    ii, jj = torch.meshgrid(torch.arange(K, device=dev), torch.arange(K, device=dev), indexing='ij')
    ha = pa[:, None].expand(-1, K * K).reshape(-1)
    hb = pb[:, None].expand(-1, K * K).reshape(-1)
    hi = ii.reshape(-1).repeat(len(pa))
    hj = jj.reshape(-1).repeat(len(pa))

    p3d = torch.full((N, 3), float('nan'), dtype=dt, device=dev)
    picked = torch.full((C, N), -1, dtype=torch.long, device=dev)
    pts2d = torch.full((C, N, 2), float('nan'), dtype=dt, device=dev)
    errs = torch.full((C, N), float('nan'), dtype=dt, device=dev)
    tau2 = tau * tau
    step = _auto_chunk(C * K * max(1, len(ha)), chunk, max_elems)
    floor = max(2, int(min_inliers))
    for s0 in range(0, N, step):
        s1 = min(N, s0 + step)
        n = s1 - s0
        p2 = pts[:, s0:s1]                                    # [C,n,K,2]
        und = G._undistort_flat(P, p2.reshape(C, -1, 2)).reshape(C, n, K, 2) if undistort else p2
        valid = torch.isfinite(p2).all(-1) & torch.isfinite(und).all(-1)    # [C,n,K]
        cam_ok = valid if contrib is None else valid & contrib[:, None, None]
        Es = E if E.ndim == 3 else E[:, s0:s1]
        center = _default_center(Es)
        rows = _recentred_rows(Es, center)
        if rows.ndim == 3:
            rows_k = rows
        else:
            rows_k = rows[:, :, None].expand(C, n, K, 3, 4).reshape(C, n * K, 3, 4)
        Bk = _blocks(rows_k, torch.nan_to_num(und).reshape(C, n * K, 2),
                     cam_ok.reshape(C, n * K).to(dt)).reshape(C, n, K, 4, 4)
        ok_h = cam_ok[ha, :, hi] & cam_ok[hb, :, hj]          # [H, n]
        hh, nn = torch.nonzero(ok_h, as_tuple=True)
        if hh.numel() == 0:
            continue
        Xh = _solve(Bk[ha[hh], nn, hi[hh]] + Bk[hb[hh], nn, hj[hh]], _sub_center(center, nn), ridge)
        # error of every candidate of every camera against each hypothesis
        Eh = _sub_ext(Es, nn)
        e_all = []
        for k in range(K):
            e_all.append(G._reproj_err_flat(Pr, Xh, p2[:, nn, k], Eh))
        e_all = torch.stack(e_all, -1)                         # [C,M,K]
        e_all = torch.where(valid[:, nn], torch.nan_to_num(e_all, nan=float('inf')), torch.full_like(e_all, float('inf')))
        emin = e_all.min(-1).values                            # [C,M]
        cost = torch.minimum(emin ** 2, torch.full_like(emin, tau2))
        has = valid[:, nn].any(-1)
        cost = torch.where(has, cost, torch.zeros_like(cost)).sum(0)
        best = torch.full((n,), float('inf'), dtype=dt, device=dev).scatter_reduce(0, nn, cost, reduce='amin')
        is_best = cost <= best[nn]
        M = nn.numel()
        first = torch.full((n,), M, dtype=torch.long, device=dev).scatter_reduce(
            0, nn[is_best], torch.arange(M, device=dev)[is_best], reduce='amin')
        sel_pts = torch.nonzero(first < M)[:, 0]
        hs = first[sel_pts]
        kbest = e_all[:, hs].argmin(-1)                        # [C,m]
        ebest = e_all[:, hs].gather(-1, kbest[..., None])[..., 0]
        inl = ebest < tau
        idx = torch.arange(len(sel_pts), device=dev)
        # chosen 2D points
        p2sel = p2[:, sel_pts].gather(2, kbest[:, :, None, None].expand(C, len(sel_pts), 1, 2))[:, :, 0]
        Bsel = Bk[:, sel_pts].gather(2, kbest[:, :, None, None, None].expand(C, len(sel_pts), 1, 4, 4))[:, :, 0]
        cam_ok_sel = cam_ok[:, sel_pts].gather(2, kbest[:, :, None])[..., 0]
        fit = inl & cam_ok_sel
        # winning pair as fallback
        pair = torch.zeros_like(fit)
        pair[ha[hh[hs]], idx] = True
        pair[hb[hh[hs]], idx] = True
        fit = torch.where((fit.sum(0) >= 2)[None], fit, pair)
        if not bool(torch.equal(fit, fit & inl)):
            # fallback pair: make sure the pair's own candidates are the chosen ones
            pk = torch.stack([hi[hh[hs]], hj[hh[hs]]])
            for r, cam in enumerate((ha[hh[hs]], hb[hh[hs]])):
                kbest[cam, idx] = torch.where(fit[cam, idx] & ~inl[cam, idx], pk[r], kbest[cam, idx])
            p2sel = p2[:, sel_pts].gather(2, kbest[:, :, None, None].expand(C, len(sel_pts), 1, 2))[:, :, 0]
            Bsel = Bk[:, sel_pts].gather(2, kbest[:, :, None, None, None].expand(C, len(sel_pts), 1, 4, 4))[:, :, 0]
        Ep = _sub_ext(Es, sel_pts)
        Xp = _solve(torch.einsum('cn,cnij->nij', fit.to(dt), Bsel), _sub_center(center, sel_pts), ridge)
        for _ in range(n_refine):
            Xp = _refine_flat(Pr, Xp, p2sel, fit.to(dt), Ep, n_iter=2)
            # re-pick each camera's closest candidate against the refined point
            e_k = torch.stack([G._reproj_err_flat(Pr, Xp, p2[:, sel_pts, k], Ep) for k in range(K)], -1)
            e_k = torch.where(valid[:, sel_pts], torch.nan_to_num(e_k, nan=float('inf')), torch.full_like(e_k, float('inf')))
            k2 = e_k.argmin(-1)
            e2 = e_k.gather(-1, k2[..., None])[..., 0]
            ok2 = cam_ok[:, sel_pts].gather(2, k2[:, :, None])[..., 0]
            fit2 = (e2 < tau) & ok2
            grow = fit2.sum(0) >= fit.sum(0)
            fit = torch.where(grow[None], fit2, fit)
            kbest = torch.where(grow[None], k2, kbest)
            p2sel = p2[:, sel_pts].gather(2, kbest[:, :, None, None].expand(C, len(sel_pts), 1, 2))[:, :, 0]
        e = G._reproj_err_flat(Pr, Xp, p2sel, Ep)
        chosen = (torch.nan_to_num(e, nan=float('inf')) < tau) | fit
        good = ((chosen & (cam_ok[:, sel_pts].gather(2, kbest[:, :, None])[..., 0])).sum(0) >= floor) \
            & torch.isfinite(Xp).all(-1)
        chosen = chosen & good[None]
        gi = s0 + sel_pts
        p3d[gi] = torch.where(good[:, None], Xp, torch.full_like(Xp, float('nan')))
        picked[:, gi] = torch.where(chosen, kbest, torch.full_like(kbest, -1))
        pts2d[:, gi] = torch.where(chosen[..., None], p2sel, torch.full_like(p2sel, float('nan')))
        errs[:, gi] = torch.where(chosen, e, torch.full_like(e, float('nan')))
    return PossibleResult(p3d=p3d, picked=picked, points_2d=pts2d, errors=errs,
                          n_inliers=(picked >= 0).sum(0), n_candidates=K)
