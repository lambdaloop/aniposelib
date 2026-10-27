import cv2
import numpy as np
from copy import copy, deepcopy
from contextlib import contextmanager
from scipy.sparse import lil_matrix, dok_matrix
from scipy.linalg import inv
from scipy import optimize
from collections import defaultdict, Counter, namedtuple
import io
import re
import toml
import itertools
from tqdm import trange
from pprint import pprint
import time
import warnings

import torch
import torch.nn as nn
import torch.optim as optim

from einops import rearrange, repeat

from aniposelib.boards import merge_rows, extract_points, \
    extract_rtvecs, get_video_params
from aniposelib.utils import get_initial_extrinsics, make_M, get_rtvec, \
    get_connections
from aniposelib._types import returns_like_input, to_torch as _to_torch
from aniposelib import geometry as _G
from aniposelib import triangulation as _T
from aniposelib.params import DEFAULT_FREE_INTRINSICS, validate_intrinsics, matrix_mask


class CameraOrderWarning(UserWarning):
    """CameraGroup.load found camera keys whose numeric order differs from their lexical order
    (aniposelib <= 0.9 sorted them lexically, so cam_10 came before cam_2), or duplicate names."""


def _as_rng(rng):
    """np.random.Generator from a Generator, an int seed or None (fresh entropy).
    Never touches the global numpy RNG."""
    if isinstance(rng, np.random.Generator):
        return rng
    if isinstance(rng, np.random.RandomState):
        return np.random.default_rng(rng.randint(0, 2**31 - 1))
    return np.random.default_rng(rng)


def _check_kwargs(fname, kwargs, allowed):
    """Raise TypeError naming any keyword not accepted by `allowed` (a set of names)."""
    bad = sorted(set(kwargs) - set(allowed))
    if bad:
        raise TypeError("{}() got unexpected keyword argument(s) {}; valid options are {}".format(
            fname, ', '.join(repr(b) for b in bad), sorted(allowed)))

def _triangulate_svd(points, camera_mats, weights):
    """Homogeneous SVD DLT (aniposelib <= 0.9), with static [C,4,4] or per-point [C,N,4,4]
    extrinsics. points [C,N,2] (finite), weights [C,N] -> [N,3]."""
    if camera_mats.ndim == 3:
        camera_mats = camera_mats[:, None].expand(-1, points.shape[1], 4, 4)
    x = points[..., 0, None]
    y = points[..., 1, None]
    w = weights[..., None]
    eq_x = w * (x * camera_mats[:, :, 2] - camera_mats[:, :, 0])   # [C,N,4]
    eq_y = w * (y * camera_mats[:, :, 2] - camera_mats[:, :, 1])
    A = torch.stack([eq_x, eq_y], 1)                                # [C,2,N,4]
    A = A.reshape(-1, A.shape[2], 4).transpose(0, 1)                # [N,2C,4]
    _, _, vh = torch.linalg.svd(A, full_matrices=False)
    p = vh[:, -1, :]
    return p[:, :3] / p[:, 3:4]


def get_error_dict(errors_full, min_points=10):
    n_cams = errors_full.shape[0]
    errors_norm = np.linalg.norm(errors_full, axis=2)

    good = ~np.isnan(errors_full[:, :, 0])

    error_dict = dict()

    for i in range(n_cams):
        for j in range(i+1, n_cams):
            subset = good[i] & good[j]
            err_subset = errors_norm[:, subset][[i, j]]
            err_subset_mean = np.mean(err_subset, axis=0)
            if np.sum(subset) > min_points:
                percents = np.percentile(err_subset_mean, [15, 75])
                # percents = np.percentile(err_subset, [25, 75])
                error_dict[(i, j)] = (err_subset.shape[1], percents)
    return error_dict

def check_errors(cgroup, imgp):
    p3ds = cgroup.triangulate(imgp)
    errors_full = cgroup.reprojection_error(p3ds, imgp, mean=False)
    return get_error_dict(errors_full)

def subset_extra(extra, ixs):
    if extra is None:
        return None

    new_extra = {
        'objp': extra['objp'][ixs],
        'ids': extra['ids'][ixs],
        'rvecs': extra['rvecs'][:, ixs],
        'tvecs': extra['tvecs'][:, ixs]
    }
    return new_extra

def resample_points_extra(imgp, extra, n_samp=25, rng=None):
    rng = _as_rng(rng)
    n_cams, n_points, _ = imgp.shape
    ids = remap_ids(extra['ids'])
    n_ids = np.max(ids)+1
    good = ~np.isnan(imgp[:, :, 0])
    ixs = np.arange(n_points)

    cam_counts = np.zeros((n_ids, n_cams), dtype='int64')
    for idnum in range(n_ids):
        cam_counts[idnum] = np.sum(good[:, ids == idnum], axis=1)
    cam_counts_random = cam_counts + rng.random(size=cam_counts.shape)
    best_boards = np.argsort(-cam_counts_random, axis=0)

    cam_totals = np.zeros(n_cams, dtype='int64')

    include = set()
    for cam_num in range(n_cams):
        for board_id in best_boards[:, cam_num]:
            include.update(ixs[ids == board_id])
            cam_totals += cam_counts[board_id]
            if cam_totals[cam_num] >= n_samp or \
               cam_counts_random[board_id, cam_num] < 1:
                break

    final_ixs = sorted(include)
    newp = imgp[:, final_ixs]
    extra = subset_extra(extra, final_ixs)
    return newp, extra

def resample_points(imgp, extra=None, n_samp=25, rng=None):
    """Subsample points seen by >= 2 cameras (weighted towards points seen by many cameras).
    `rng`: np.random.Generator, int seed or None (fresh entropy); the global RNG is not used."""
    n_cams = imgp.shape[0]
    n_points = imgp.shape[1]
    good = ~np.isnan(imgp[:, :, 0])
    num_cams = np.sum(good, axis=0)

    valid_mask = num_cams >= 2
    valid_ixs = np.where(valid_mask)[0]

    if len(valid_ixs) == 0:
        return imgp, extra

    target = int(n_samp * 2.5)

    if len(valid_ixs) <= target:
        final_ixs = valid_ixs
    else:
        rng = _as_rng(rng)
        num_cams_valid = num_cams[valid_ixs]
        weights = num_cams_valid.astype(float) ** 3
        weights /= weights.sum()
        chosen = rng.choice(len(valid_ixs), size=target, replace=False, p=weights)
        final_ixs = np.sort(valid_ixs[chosen])

    newp = imgp[:, final_ixs]
    new_extra = subset_extra(extra, final_ixs)
    return newp, new_extra

def medfilt_data(values, size=15):
    padsize = size+5
    vpad = np.pad(values, (padsize, padsize), mode='reflect')
    from scipy import signal
    vpadf = signal.medfilt(vpad, kernel_size=size)
    return vpadf[padsize:-padsize]

def nan_helper(y):
    return np.isnan(y), lambda z: z.nonzero()[0]

def interpolate_data(vals):
    nans, ix = nan_helper(vals)
    out = np.copy(vals)
    try:
        out[nans] = np.interp(ix(nans), ix(~nans), vals[~nans])
    except ValueError:
        out[:] = 0
    return out

def interior_span(observed):
    """observed: (N, ...) bool, axis 0 is time. True from the first to the last observed frame
    of each column (interior gaps included), False before and after (and for columns that are
    never observed)."""
    observed = np.asarray(observed, dtype=bool)
    n = observed.shape[0]
    if n == 0:
        return observed.copy()
    fwd = np.maximum.accumulate(observed, axis=0)
    bwd = np.maximum.accumulate(observed[::-1], axis=0)[::-1]
    return fwd & bwd


def remap_ids(ids):
    unique_ids = np.unique(ids)
    ids_out = np.copy(ids)
    for i, num in enumerate(unique_ids):
        ids_out[ids == num] = i
    return ids_out

def transform_points(points, rvecs, tvecs):
    """Rotate points [N,3] by rotation vectors [N,3] (Rodrigues) and translate by [N,3].
    Differentiable also at exactly zero rotation."""
    R = _G.rodrigues(rvecs)
    rotated = torch.einsum('nij,nj->ni', R, points)
    return rotated + tvecs

Ray = namedtuple('Ray', ['origin', 'direction'])

def closest_point_between_rays(r1, r2):
    """
    Find the closest points between two rays.
    
    Parameters:
    Takes two rays (tuple Ray with origin and direction)
    Returns:
    p1: closest point on ray 1
    p2: closest point on ray 2
    distance: minimum distance between the rays
    """

    a1 = r1.origin
    d1 = r1.direction
    a2 = r2.origin
    d2 = r2.direction
    
    # Normalize direction vectors
    d1 = d1 / torch.linalg.norm(d1)
    d2 = d2 / torch.linalg.norm(d2)
    
    # Vector between origins
    w0 = a1 - a2
    
    # Dot products
    a = torch.dot(d1, d1)  # always 1 if normalized
    b = torch.dot(d1, d2)
    c = torch.dot(d2, d2)  # always 1 if normalized
    d = torch.dot(d1, w0)
    e = torch.dot(d2, w0)
    
    # Solve for parameters
    denom = a * c - b * b
    
    if abs(denom) < 1e-10:
        # Parallel rays
        return None
    else:
        t1 = (b * e - c * d) / denom
        t2 = (a * e - b * d) / denom

    # best point behind rays
    if t1 < 0 or t2 < 0:
        return None
        
    # Closest points
    p1 = a1 + t1 * d1
    p2 = a2 + t2 * d2
    
    return (p1 + p2) / 2.0

def to_tensor(x, device=None, dtype=torch.float64):
    if x is None:
        return None
    if torch.is_tensor(x):
        t = x.to(dtype=dtype)
    else:
        if isinstance(x, (list, tuple)) and len(x) > 0 and torch.is_tensor(x[0]):
            t = torch.stack(x).to(dtype=dtype)
        else:
            t = torch.tensor(np.array(x), dtype=dtype)
    if device is not None:
        t = t.to(device)
    return t

def to_homogeneous(p):
    one_size = p.shape[:-1] + (1,)
    ones = torch.ones(size=one_size, dtype=p.dtype, device=p.device)
    return torch.cat([p, ones], dim=-1)


def from_homogeneous(p, eps=1e-10):
    return p[..., :-1] / (p[..., -1, None] + eps) 

def rodrigues(rvec):
    """Rotation vector (3,) or (3, 1) (or batched [..., 3]) -> rotation matrix (3, 3).

    Differentiable everywhere, including the exact zero rotation: near 0 the coefficients
    sin(t)/t and (1-cos(t))/t^2 use their series, so d R / d rvec at rvec = 0 is the
    skew-symmetric generator (aniposelib <= 0.9 returned a constant identity there, so a
    camera at exactly zero rotation got no rotation gradient in bundle adjustment)."""
    if rvec.dim() == 2 and rvec.shape[-1] == 1:
        rvec = rvec.reshape(-1)
    return _G.rodrigues(rvec)

def _apply_intrinsics(p, matrix):
    """Normalised (distorted) coordinates -> sensor pixels with the full upper 2x3 of K
    (fx, skew, cx / K[1,0], fy, cy). With zero skew this is bit-identical to indexing
    fx, fy, cx, cy. The bottom row of K is assumed to be (0, 0, 1)."""
    x, y = p[..., 0], p[..., 1]
    u = matrix[0, 0] * x + matrix[0, 1] * y + matrix[0, 2]
    v = matrix[1, 0] * x + matrix[1, 1] * y + matrix[1, 2]
    return torch.stack([u, v], dim=-1)


def _invert_intrinsics(u, v, matrix):
    """Sensor pixels -> normalised coordinates, inverting the upper 2x2 of K.

    Written so that with zero skew it reduces exactly to (u - cx) / fx, (v - cy) / fy."""
    fx, s, cx = matrix[0, 0], matrix[0, 1], matrix[0, 2]
    k10, fy, cy = matrix[1, 0], matrix[1, 1], matrix[1, 2]
    du = u - cx
    dv = v - cy
    y = (dv - k10 * du / fx) / (fy - k10 * s / fx)
    x = (du - s * y) / fx
    return x, y


def project_points(p3d, ext, matrix, dist, offset=None):
    """Brown-Conrady (OpenCV k1, k2, p1, p2, k3) projection with the full camera matrix.

    Unlike cv2.projectPoints, the skew K[0,1] is honoured (decision D3). BA keeps it, and the
    bottom row of K, frozen by default (see aniposelib.params). `offset` (x, y) is the stored
    image origin in sensor pixels and is subtracted from the result."""
    p2d_proj_raw = torch.matmul(to_homogeneous(p3d), ext.T)
    p2d_proj_raw = from_homogeneous(p2d_proj_raw[..., :3])

    k1, k2, p1, p2, k3, k4, k5, k6 = _brown_dist_terms(dist)
    r2 = torch.sum(torch.square(p2d_proj_raw), axis=1)
    r4 = r2 * r2
    r6 = r4 * r2
    kscale = (1 + k1 * r2 + k2 * r4 + k3 * r6) / (1 + k4 * r2 + k5 * r4 + k6 * r6)

    x = p2d_proj_raw[..., 0]
    y = p2d_proj_raw[..., 1]
    dx = 2*p1*x*y + p2 * (r2 + 2*x*x)
    dy = p1*(r2 + 2*y*y) + 2*p2*x*y
    p1_p2_add = torch.stack([dx, dy], dim=-1)

    p2d_dist = kscale[:, None] * p2d_proj_raw + p1_p2_add

    p2d = _apply_intrinsics(p2d_dist, matrix)
    if offset is not None:
        p2d = p2d - offset
    return p2d


def _brown_dist_terms(dist):
    """k1, k2, p1, p2, k3, k4, k5, k6 (OpenCV rational model) from a distortion vector;
    pads short vectors with zeros. Terms beyond the eighth are ignored."""
    if dist.shape[0] < 8:
        pad = torch.zeros(8 - dist.shape[0], dtype=dist.dtype, device=dist.device)
        dist = torch.cat([dist, pad])
    return tuple(dist[i] for i in range(8))

FISHEYE_EPS = 1e-12
_HALF_PI = float(np.pi / 2)


def _fisheye_dist_terms(dist):
    """k1..k4 from a distortion vector; pads short vectors, ignores extra terms."""
    if dist.shape[0] < 4:
        pad = torch.zeros(4 - dist.shape[0], dtype=dist.dtype, device=dist.device)
        dist = torch.cat([dist, pad])
    return dist[0], dist[1], dist[2], dist[3]


def project_points_fisheye(p3d, ext, matrix, dist, eps=FISHEYE_EPS, offset=None):
    """Differentiable OpenCV fisheye (equidistant) projection.

    The full upper 2x3 of the camera matrix is used, so a skew K[0,1] = alpha * fx
    matches cv2.fisheye's alpha. The bottom row is not read. BA keeps skew frozen
    by default (aniposelib.params). `offset` is subtracted from the result.

    NaN and inf rows pass through as NaN, elementwise, as triangulate expects.
    """
    # Sanitize the input before doing any arithmetic on it. Masking only the
    # output leaves the values finite but the *gradients* NaN, since the
    # torch.where backward computes 0 * NaN = NaN on matrix / rvec / tvec.
    bad = ~torch.isfinite(p3d).all(dim=-1)
    p3d = torch.where(bad[..., None], torch.zeros_like(p3d), p3d)

    p_cam = torch.matmul(to_homogeneous(p3d), ext.T)[..., :3]
    x, y, z = p_cam[..., 0], p_cam[..., 1], p_cam[..., 2]

    # cv2 substitutes z = 1 when z underflows
    z_safe = torch.where(torch.abs(z) >= eps, z, torch.ones_like(z))
    a, b = x / z_safe, y / z_safe

    r2 = a * a + b * b
    r_ok = r2 >= eps * eps
    # guard the sqrt as well, its derivative is infinite at r2 == 0
    r = torch.sqrt(torch.where(r_ok, r2, torch.ones_like(r2)))

    theta = torch.atan(r)
    th2 = theta * theta
    th4 = th2 * th2
    th6 = th4 * th2
    th8 = th4 * th4
    k1, k2, k3, k4 = _fisheye_dist_terms(dist)
    theta_d = theta * (1 + k1 * th2 + k2 * th4 + k3 * th6 + k4 * th8)

    # theta_d / r tends to 1 as r tends to 0, with zero derivative, so the
    # constant branch is the exact value and the exact gradient
    scale = torch.where(r_ok, theta_d / r, torch.ones_like(r))

    p2d = _apply_intrinsics(torch.stack([scale * a, scale * b], dim=-1), matrix)
    if offset is not None:
        p2d = p2d - offset

    return torch.where(bad[..., None], torch.full_like(p2d, float('nan')), p2d)

def make_M_torch(rvec, tvec):
    R = rodrigues(rvec)
    M = torch.eye(4, device=rvec.device, dtype=rvec.dtype)
    M[:3, :3] = R
    M[:3, 3] = tvec.reshape(3)
    return M


# keys of a camera dict that Camera understands; anything else goes to Camera.extra
_CAMERA_KEYS = {'name', 'size', 'matrix', 'distortions', 'rotation', 'translation',
                'n_dist', 'fisheye', 'offset', 'free_intrinsics'}


def _to_builtin(x):
    """Make a value TOML/JSON friendly (numpy/torch -> lists and Python scalars)."""
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, dict):
        return {k: _to_builtin(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_to_builtin(v) for v in x]
    return x


# rig scale (median camera-to-point distance) at which the Adam BA learning rates were tuned;
# see CameraGroup._bundle_adjust_adam(trans_scale=...)
ADAM_REF_SCALE = 1.0

class Camera(nn.Module):
    """Pinhole camera with Brown-Conrady (OpenCV) distortion, as a torch module.

    Conventions (1.0):
    - `matrix` is in sensor pixels and is used in full (skew K[0,1] honoured; the bottom row is
      assumed to be (0, 0, 1)). Only fx, fy, cx, cy are free in bundle adjustment by default,
      see set_free_intrinsics / aniposelib.params.
    - `offset` (x, y) is the origin of the stored image in the sensor frame (a crop):
      project() returns stored-image pixels (sensor minus offset) and undistort_points() takes
      stored-image pixels (adds offset back). `size` is the stored image size.
    - `extra` holds unknown per-camera keys from a calibration file (e.g. `moving`,
      `calibrated`, `model`) and writes them back unchanged.
    - Public read-only methods follow the input type: numpy in -> numpy out, torch in ->
      detached torch out; pass return_grad=True to keep the autograd graph.
    """
    # lets boards.py dispatch on the projection model without importing this module
    fisheye = False
    # length of the distortion vector of this model (shorter ones are zero-padded on load)
    n_dist_model = 5

    def __init__(self,
                 matrix=np.eye(3),
                 dist=np.zeros(5),
                 size=None,
                 rvec=np.zeros(3),
                 tvec=np.zeros(3),
                 name=None,
                 *,
                 n_dist=2,
                 offset=None,
                 extra=None,
                 free_intrinsics=DEFAULT_FREE_INTRINSICS,
                 dtype=torch.float64):
        super().__init__()

        self.register_parameter('matrix', nn.Parameter(to_tensor(matrix, dtype=dtype)))
        self.register_parameter('dist', nn.Parameter(to_tensor(dist, dtype=dtype).view(-1)))
        self.register_parameter('rvec', nn.Parameter(to_tensor(rvec, dtype=dtype).view(-1)))
        self.register_parameter('tvec', nn.Parameter(to_tensor(tvec, dtype=dtype).view(-1)))
        # non-persistent: moves with .to() and is deep-copied, but stays out of state_dict so
        # state dicts saved by aniposelib <= 0.9 still load strictly
        self.register_buffer('offset', torch.zeros(2, dtype=dtype), persistent=False)
        if offset is not None:
            self.set_offset(offset)

        self.set_size(size)
        self.set_name(name)
        self.extra = dict(extra) if extra else {}
        self._frozen_ext = None

        # clamp rather than raise here, so a camera built with a short dist and
        # the default n_dist is not an error; set_n_dist validates strictly
        self.n_dist = min(int(n_dist), self.dist.shape[0])
        self.free_intrinsics = validate_intrinsics(free_intrinsics)
        self._install_grad_masks()

    # ------------------------------------------------------------ parameter masks

    def _install_grad_masks(self):
        self.dist.register_hook(self._mask_dist_grad)
        self.matrix.register_hook(self._mask_matrix_grad)

    # kept for code that called the old name
    _install_dist_mask = _install_grad_masks

    def _mask_dist_grad(self, grad):
        """Zero the gradient of the distortion coefficients beyond n_dist.

        Running as a hook means the mask applies during backward, before
        clip_grad_norm_, so frozen coefficients also stop contributing to the
        gradient norm. Adam then leaves them exactly untouched -- which relies
        on weight_decay being 0 in bundle_adjust.
        """
        n = self.n_dist
        if n is None or n >= grad.shape[0]:
            return grad
        # build the mask from grad at call time, so it survives a
        # set_distortions() that changes the length of the vector
        keep = torch.arange(grad.shape[0], device=grad.device) < n
        return grad * keep.to(grad.dtype)

    def _mask_matrix_grad(self, grad):
        """Zero the gradient of the camera matrix entries that are not free
        (by default skew, K[1,0] and the bottom row). Same mechanism as n_dist."""
        free = getattr(self, 'free_intrinsics', DEFAULT_FREE_INTRINSICS)
        return grad * matrix_mask(free, dtype=grad.dtype, device=grad.device)

    def __setstate__(self, state):
        # Parameter.__deepcopy__ and __reduce_ex__ do not carry backward hooks,
        # so a deepcopied or unpickled camera would keep n_dist while silently
        # losing the freezing it stands for. Put the hooks back.
        super().__setstate__(state)
        if 'offset' not in self._buffers:  # pickled by aniposelib <= 0.9
            self.register_buffer('offset', torch.zeros(2, dtype=self.matrix.dtype,
                                                       device=self.matrix.device),
                                 persistent=False)
        self.__dict__.setdefault('extra', {})
        self.__dict__.setdefault('free_intrinsics', DEFAULT_FREE_INTRINSICS)
        self.__dict__['_frozen_ext'] = None
        self._install_grad_masks()

    def get_n_dist(self):
        return self.n_dist

    def set_n_dist(self, n_dist):
        """Set how many distortion coefficients bundle_adjust may move.

        The first n_dist entries of dist are free and the rest keep whatever
        value they currently hold, so seeding known-good distortions and
        freezing them works. The layout is [k1, k2, p1, p2, k3] for pinhole and
        [k1, k2, k3, k4] for fisheye, so 2 means radial only, and 4 is the
        meaningful step up from it for a pinhole camera.

        This does not freeze the camera matrix: fx, fy, cx, cy are still free in
        a full bundle adjust. Use only_extrinsics=True, or set_free_intrinsics((),)
        for that.

        n_dist never grows on its own, so a later set_distortions() with a
        longer vector leaves the extra coefficients frozen.
        """
        if isinstance(n_dist, bool):
            raise TypeError("n_dist must be an int, got a bool")

        n_dist = int(n_dist)
        n_max = self.dist.shape[0]
        if not 0 <= n_dist <= n_max:
            raise ValueError(
                "n_dist must be between 0 and {} for this camera, got {}".format(
                    n_max, n_dist))

        if n_dist == 3 and n_max == 5:
            # p1 without p2 pins the decentering axis vertical, which is almost
            # always a typo for 4
            warnings.warn(
                'n_dist=3 frees p1 but not p2, which is rarely what you want; '
                'use 2 for radial only or 4 to include tangential',
                stacklevel=2)

        self.n_dist = n_dist

    def get_free_intrinsics(self):
        """Names of the camera-matrix entries bundle adjustment may move."""
        return self.free_intrinsics

    def set_free_intrinsics(self, names=DEFAULT_FREE_INTRINSICS):
        """Choose which camera-matrix entries bundle adjustment may move, out of
        'fx', 'fy', 'cx', 'cy', 'skew'. Default: fx, fy, cx, cy. K[1,0] and the bottom row
        are always frozen. Like n_dist this is a gradient mask, so it also applies to
        optimizers you build yourself over cam.parameters()."""
        self.free_intrinsics = validate_intrinsics(names)

    # ------------------------------------------------------------ (de)serialisation

    def get_dict(self):
        d = {
            'name': self.get_name(),
            'size': _to_builtin(list(self.get_size())) if self.get_size() is not None else None,
            'matrix': self.matrix.detach().cpu().numpy().tolist(),
            'distortions': self.dist.detach().cpu().numpy().tolist(),
            'rotation': self.rvec.detach().cpu().numpy().tolist(),
            'translation': self.tvec.detach().cpu().numpy().tolist(),
            'n_dist': int(self.n_dist),
        }
        if d['size'] is None:
            del d['size']
        if bool(torch.any(self.offset != 0)):
            d['offset'] = self.offset.detach().cpu().numpy().tolist()
        if tuple(self.free_intrinsics) != tuple(DEFAULT_FREE_INTRINSICS):
            d['free_intrinsics'] = list(self.free_intrinsics)
        for k, v in self.extra.items():
            if k not in d:
                d[k] = _to_builtin(v)
        return d

    def load_dict(self, d):
        self.set_camera_matrix(d['matrix'])
        self.set_rotation(d['rotation'])
        self.set_translation(d['translation'])
        dist = np.asarray(d['distortions'], dtype='float64').ravel()
        if len(dist) < self.n_dist_model:
            # pad short vectors (e.g. k1, k2 only) to the model length
            dist = np.concatenate([dist, np.zeros(self.n_dist_model - len(dist))])
        self.set_distortions(dist)
        self.set_name(d['name'])
        self.set_size(d.get('size'))
        self.set_offset(d.get('offset', (0.0, 0.0)))
        if 'free_intrinsics' in d:
            self.set_free_intrinsics(d['free_intrinsics'])
        self.extra = {k: deepcopy(v) for k, v in d.items() if k not in _CAMERA_KEYS}
        # after set_distortions, so its clamp does not eat the loaded value.
        # A calibration written before n_dist existed gets the default.
        if 'n_dist' in d:
            self.set_n_dist(min(int(d['n_dist']), self.dist.shape[0]))

    @classmethod
    def from_dict(cls, d, dtype=torch.float64):
        dist = np.asarray(d.get('distortions', np.zeros(cls.n_dist_model))).ravel()
        cam = cls(dist=np.zeros(max(len(dist), cls.n_dist_model)), dtype=dtype)
        cam.load_dict(d)
        return cam

    # ------------------------------------------------------------ getters / setters

    def get_camera_matrix(self):
        return self.matrix

    def get_distortions(self):
        return self.dist

    def set_camera_matrix(self, matrix):
        self.matrix.data = to_tensor(matrix, device=self.matrix.device, dtype=self.matrix.dtype)

    def set_focal_length(self, fx, fy=None):
        if fy is None:
            fy = fx
        self.matrix.data[0, 0] = float(fx)
        self.matrix.data[1, 1] = float(fy)

    def get_focal_length(self, both=False):
        fx = self.matrix[0, 0].item()
        fy = self.matrix[1, 1].item()
        if both:
            return (fx, fy)
        else:
            return (fx + fy) / 2.0

    def set_distortions(self, dist):
        self.dist.data = to_tensor(dist, device=self.dist.device, dtype=self.dist.dtype).view(-1)
        # a shorter vector has to pull n_dist down with it; a longer one does
        # not push it up, since freeing coefficients should be deliberate
        self.n_dist = min(self.n_dist, self.dist.shape[0])

    def zero_distortions(self):
        self.dist.data.zero_()

    def set_rotation(self, rvec):
        self.rvec.data = to_tensor(rvec, device=self.rvec.device, dtype=self.rvec.dtype).view(-1)
        self._refresh_frozen()

    def get_rotation(self):
        return self.rvec

    def set_translation(self, tvec):
        self.tvec.data = to_tensor(tvec, device=self.tvec.device, dtype=self.tvec.dtype).view(-1)
        self._refresh_frozen()

    def get_translation(self):
        return self.tvec

    def get_offset(self):
        """(x, y) origin of the stored image in sensor pixels (crop offset)."""
        return self.offset

    def set_offset(self, offset):
        off = to_tensor(offset, device=self.offset.device, dtype=self.offset.dtype).view(-1)
        if off.shape[0] != 2:
            raise ValueError("offset must be (x, y), got {}".format(tuple(off.shape)))
        self.offset.copy_(off)

    def _refresh_frozen(self):
        if getattr(self, '_frozen_ext', None) is not None:
            self._frozen_ext = make_M_torch(self.rvec, self.tvec).detach()

    def get_extrinsics_mat(self):
        frozen = getattr(self, '_frozen_ext', None)
        if frozen is not None:
            return frozen
        return make_M_torch(self.rvec, self.tvec)

    def get_extrinsics_params(self):
        return [self.rvec, self.tvec]

    def get_name(self):
        return self.name

    def set_name(self, name):
        self.name = str(name)

    def set_size(self, size):
        """set size as (width, height)"""
        self.size = size

    def get_size(self):
        """get size as (width, height)"""
        return self.size

    @property
    def device(self):
        return self.matrix.device

    @property
    def dtype(self):
        return self.matrix.dtype

    def resize_camera(self, scale):
        """resize the camera by scale factor, updating intrinsics (and offset) to match"""
        size = self.get_size()
        new_size = size[0] * scale, size[1] * scale
        self.set_size(new_size)

        self.matrix.data *= scale
        self.matrix.data[2, 2] = 1
        self.offset.mul_(scale)

    # ------------------------------------------------------------ geometry

    def _tensor(self, x):
        return to_tensor(x, device=self.matrix.device, dtype=self.matrix.dtype)

    @returns_like_input()
    def distort_points(self, points):
        """Normalised (undistorted) coordinates -> stored-image pixels."""
        return self._distort_points(points)

    def _distort_points(self, points):
        points = to_homogeneous(self._tensor(points).reshape(-1, 2))
        eye = torch.eye(4, dtype=self.matrix.dtype, device=self.matrix.device)
        return project_points(points, eye, self.matrix, self.dist, self.offset)

    @returns_like_input()
    def undistort_points(self, points, n_iter=15):
        """Stored-image pixels (..., 2) -> normalised camera coordinates (..., 2)."""
        return self._undistort_points(points, n_iter=n_iter)

    def _undistort_points(self, points, n_iter=15):
        t_points = self._tensor(points)
        shape = t_points.shape
        t_points = t_points.reshape(-1, 2)
        x, y = _invert_intrinsics(t_points[:, 0] + self.offset[0],
                                  t_points[:, 1] + self.offset[1], self.matrix)
        x0, y0 = x.clone(), y.clone()
        k1, k2, p1, p2, k3, k4, k5, k6 = _brown_dist_terms(self.dist)
        for _ in range(n_iter):
            r2 = x*x + y*y
            r4 = r2*r2
            r6 = r4*r2
            radial = (1 + k1*r2 + k2*r4 + k3*r6) / (1 + k4*r2 + k5*r4 + k6*r6)
            dx = 2*p1*x*y + p2*(r2 + 2*x*x)
            dy = p1*(r2 + 2*y*y) + 2*p2*x*y
            x = (x0 - dx) / radial
            y = (y0 - dy) / radial
        return torch.stack([x, y], dim=1).reshape(shape)

    @returns_like_input()
    def project(self, points):
        """(N, 3) world points -> (N, 2) stored-image pixels."""
        return self._project(points)

    def _project(self, points):
        points = self._tensor(points).reshape(-1, 3)
        return project_points(points, self.get_extrinsics_mat(),
                              self.matrix, self.dist, self.offset)

    @returns_like_input()
    def reprojection_error(self, p3d, p2d):
        return self._reprojection_error(p3d, p2d)

    def _reprojection_error(self, p3d, p2d):
        p2d = self._tensor(p2d)
        proj = self._project(p3d).reshape(p2d.shape)
        return p2d - proj

    def copy(self):
        # type(self) so subclasses copy as themselves, and an explicit dist of
        # the right length because load_state_dict is strict about shapes -- the
        # default would break any camera whose dist is not 5 long.
        new_cam = type(self)(name=self.name, size=self.size,
                             dist=np.zeros(self.dist.shape[0]),
                             n_dist=self.n_dist,
                             extra=deepcopy(self.extra),
                             free_intrinsics=self.free_intrinsics,
                             dtype=self.matrix.dtype)
        new_cam.load_state_dict(self.state_dict())
        new_cam.offset.copy_(self.offset.detach().cpu())
        # load_state_dict copies values in place and does not change device, so a
        # copy of a cuda camera would otherwise come back on CPU. Preserve device.
        return new_cam.to(self.matrix.device)

    @returns_like_input()
    def is_point_visible(self, p3d, margin=0):
        """
        Takes as input a set of 3D points: (N, 3).
        Check if 3D points project into camera view.
        margin: pixels from border (e.g., 10 to avoid edge effects)
        """
        p3d = self._tensor(p3d).reshape(-1, 3)
        p2d = self._project(p3d)
        w, h = self.get_size()

        # check if in bounds
        in_bounds = (
            (p2d[:, 0] >= margin) & 
            (p2d[:, 0] < w - margin) &
            (p2d[:, 1] >= margin) & 
            (p2d[:, 1] < h - margin)
        )

        # check if point is in front of camera
        R = rodrigues(self.rvec)
        t = self.tvec
        p_cam = (R @ p3d.T).T + t

        in_front = (p_cam[:, 2] > 0)

        return in_bounds & in_front

    @returns_like_input()
    def projection_sensitivity(self, p):
        """
        Compute the sensitivity (Jacobian) of camera projection to 3D point movement.

        Parameters:
        -----------
        p : array_like, shape (3,) or (N, 3)
            3D point(s) in world coordinates [x, y, z]

        Returns:
        --------
        J : ndarray, shape (N, 2, 3)
            Jacobian matrix where J[i,j] = d pixel_i / d world_j (pinhole part, distortion
            ignored; the full upper 2x2 of K, including skew, is used)
        """
        return self._projection_sensitivity(p)

    def _projection_sensitivity(self, p):
        p = self._tensor(p).reshape(-1, 3)

        n_points = p.shape[0]

        R = rodrigues(self.rvec)
        t = self.tvec

        p_cam = (R @ p.T).T + t

        X = p_cam[:, 0]
        Y = p_cam[:, 1]
        Z = p_cam[:, 2]

        # d(x/z, y/z) / d(X, Y, Z)
        J_q = torch.zeros((n_points, 2, 3), dtype=p.dtype, device=p.device)
        J_q[:, 0, 0] = 1 / Z
        J_q[:, 0, 2] = -X / (Z**2)
        J_q[:, 1, 1] = 1 / Z
        J_q[:, 1, 2] = -Y / (Z**2)

        J = torch.einsum('ab,nbj,jk->nak', self.matrix[:2, :2], J_q, R)
        return J

    def get_center_world(self):
        R = rodrigues(self.rvec)
        t = self.tvec
        return (-R.T @ t) 

    def get_camera_rays(self):
        """
        Shoot rays from the 4 corners and center of a camera into 3D space.

        Returns:
        --------
        rays : list of Ray(origin, direction)
            Order: [top_left, top_right, bottom_right, bottom_left, center]
        """
        if self.get_size() is None:
            raise ValueError("Camera must have size set")

        w, h = self.get_size()

        img_points = np.array([
            [0, 0],           # top left
            [w-1, 0],         # top right
            [w-1, h-1],       # bottom right
            [0, h-1],         # bottom left
            [w/2, h/2]        # center
        ], dtype='float64')

        norm_points = self._undistort_points(img_points)

        rays_camera = to_homogeneous(norm_points)
        rays_camera = rays_camera / torch.linalg.norm(rays_camera, dim=1, keepdim=True)

        R = rodrigues(self.rvec)
        center = self.get_center_world()
        directions = (R.T @ rays_camera.T).T

        rays = [Ray(origin=center, direction=d)
                for d in directions]

        return rays


    # -- 1.0 convenience aliases (thin delegates)
    @classmethod
    def from_extrinsics_mat(cls, K, E, dist=None, size=None, name=None, **kwargs):
        """A camera from a 3x3 ``K`` and a 4x4 / 3x4 world->camera matrix ``E``
        (``x_cam = R X + t``); see :func:`aniposelib.geometry.camera_from_extrinsics_mat`.
        ``FisheyeCamera.from_extrinsics_mat`` builds a fisheye camera."""
        from aniposelib import geometry as _io
        kwargs.setdefault('fisheye', issubclass(cls, FisheyeCamera))
        cam = _io.camera_from_extrinsics_mat(K, E, dist, size, name, **kwargs)
        if not isinstance(cam, cls):
            raise TypeError('{} is not a {}'.format(type(cam).__name__, cls.__name__))
        return cam

    def lens_validity(self, **kwargs):
        """Is the lens model usable over the whole image (undistort/distort round trip,
        monotone radial map)? See :func:`aniposelib.quality.lens_validity` (e.g.
        ``max_nonfinite_frac=`` for a fisheye whose image circle does not fill the frame)."""
        from aniposelib import quality as _Q
        return _Q.lens_validity(self, **kwargs)


class FisheyeCamera(Camera):
    """OpenCV fisheye (equidistant) camera, differentiable end to end.

    Distortions are k1..k4 of the equidistant model, so the vector is 4 long
    rather than the 5 of the Brown-Conrady pinhole model.
    """

    fisheye = True
    n_dist_model = 4

    def __init__(self,
                 matrix=np.eye(3),
                 dist=np.zeros(4),
                 size=None,
                 rvec=np.zeros(3),
                 tvec=np.zeros(3),
                 name=None,
                 *,
                 n_dist=2,
                 **kwargs):
        super().__init__(matrix=matrix, dist=dist, size=size,
                         rvec=rvec, tvec=tvec, name=name,
                         n_dist=n_dist, **kwargs)

    def get_dict(self):
        d = super().get_dict()
        d['fisheye'] = True
        return d

    def _project(self, points):
        points = self._tensor(points).reshape(-1, 3)
        return project_points_fisheye(points, self.get_extrinsics_mat(),
                                      self.matrix, self.dist, offset=self.offset)

    def _distort_points(self, points):
        points = to_homogeneous(self._tensor(points).reshape(-1, 2))
        eye = torch.eye(4, dtype=self.matrix.dtype, device=self.matrix.device)
        return project_points_fisheye(points, eye, self.matrix, self.dist, offset=self.offset)

    def _solve_theta(self, points, n_iter=20, eps=FISHEYE_EPS, tol=1e-8, step_tol=1e-12):
        """Invert the equidistant model for a set of pixel coordinates.

        Newton stops early once no point inside the invertible domain moves by more than
        step_tol (checked at iterations 5, 8, 12, 16, so on CUDA it costs a few host syncs);
        it converges in ~4 iterations in practice. step_tol=0 runs all n_iter iterations.

        Returns (theta, x, y, theta_d, r_ok, ok), all flattened to (K,):
          theta    incidence angle
          x, y     normalized coordinates, sanitized to 0 where the input was not finite
          theta_d  radius of (x, y), sanitized to 1 where r_ok is False
          r_ok     radius above eps, i.e. not the principal point
          ok       Newton converged and the point is inside the invertible domain
        """
        t_points = self._tensor(points).reshape(-1, 2)
        k1, k2, k3, k4 = _fisheye_dist_terms(self.dist)

        x, y = _invert_intrinsics(t_points[:, 0] + self.offset[0],
                                  t_points[:, 1] + self.offset[1], self.matrix)

        finite = torch.isfinite(x) & torch.isfinite(y)
        x = torch.where(finite, x, torch.zeros_like(x))
        y = torch.where(finite, y, torch.zeros_like(y))

        td2 = x * x + y * y
        r_ok = td2 >= eps * eps
        theta_d = torch.sqrt(torch.where(r_ok, td2, torch.ones_like(td2)))
        theta_d_c = torch.clamp(theta_d, max=_HALF_PI)  # as cv2 does
        domain = finite & (theta_d <= _HALF_PI)

        theta = theta_d_c.clone()
        for i in range(n_iter):
            th2 = theta * theta
            th4 = th2 * th2
            th6 = th4 * th2
            th8 = th4 * th4
            f = theta * (1 + k1 * th2 + k2 * th4 + k3 * th6 + k4 * th8) - theta_d_c
            fp = 1 + 3 * k1 * th2 + 5 * k2 * th4 + 7 * k3 * th6 + 9 * k4 * th8
            fp = torch.where(torch.abs(fp) > eps, fp, torch.ones_like(fp))
            step = f / fp
            theta = theta - step
            if step_tol > 0 and (i + 1) in (5, 8, 12, 16):
                moving = domain & ~(torch.abs(step) <= step_tol)
                if not bool(torch.any(moving)):
                    break

        th2 = theta * theta
        th4 = th2 * th2
        th6 = th4 * th2
        th8 = th4 * th4
        resid = torch.abs(
            theta * (1 + k1 * th2 + k2 * th4 + k3 * th6 + k4 * th8) - theta_d_c)

        # With a negative k1 the forward model stops being monotone, and a
        # theta_d past its maximum has no solution at all. cv2 writes -1000000
        # there; we say NaN, because everything downstream reads NaN as missing
        # and would happily triangulate a finite -1000000.
        ok = (finite & torch.isfinite(theta) & (theta >= 0)
              & (resid <= tol) & (theta_d <= _HALF_PI))

        return theta, x, y, theta_d, r_ok, ok

    def _undistort_points(self, points, n_iter=20):
        shape = self._tensor(points).shape
        theta, x, y, theta_d, r_ok, ok = self._solve_theta(points, n_iter=n_iter)

        theta_c = torch.clamp(theta, max=_HALF_PI - 1e-9)  # keep tan() finite
        # scale tends to 1 as the radius tends to 0
        scale = torch.where(r_ok, torch.tan(theta_c) / theta_d,
                            torch.ones_like(theta))
        out = torch.stack([scale * x, scale * y], dim=-1)

        good = ok & (theta < _HALF_PI)
        out = torch.where(good[:, None], out, torch.full_like(out, float('nan')))
        return out.reshape(shape)

    @returns_like_input()
    def undistort_points(self, points, n_iter=20):
        """Stored-image pixels (..., 2) -> normalised camera coordinates (..., 2);
        NaN outside the invertible domain of the model."""
        return self._undistort_points(points, n_iter=n_iter)

    def get_camera_rays(self):
        """Same contract as Camera.get_camera_rays.

        Overridden because the base version lifts the normalized point to z=1,
        which needs tan(theta) and so loses all precision as theta approaches 90
        degrees, and because it would propagate the NaN we emit for pixels
        outside the invertible domain. Building the direction from theta gives a
        unit vector directly, for any field of view.
        """
        if self.get_size() is None:
            raise ValueError("Camera must have size set")

        w, h = self.get_size()
        img_points = np.array([
            [0, 0],           # top left
            [w-1, 0],         # top right
            [w-1, h-1],       # bottom right
            [0, h-1],         # bottom left
            [w/2, h/2]        # center
        ], dtype='float64')

        theta, x, y, theta_d, r_ok, ok = self._solve_theta(img_points)
        # outside the invertible domain fall back to the pure equidistant angle
        # rather than NaN, which would poison get_point_cloud's quantiles
        theta = torch.where(ok, theta, torch.clamp(theta_d, max=_HALF_PI))

        ux = torch.where(r_ok, x / theta_d, torch.zeros_like(x))
        uy = torch.where(r_ok, y / theta_d, torch.zeros_like(y))
        st, ct = torch.sin(theta), torch.cos(theta)
        rays_camera = torch.stack([st * ux, st * uy, ct], dim=-1)  # already unit

        R = rodrigues(self.rvec)
        center = self.get_center_world()
        directions = (R.T @ rays_camera.T).T

        return [Ray(origin=center, direction=d) for d in directions]

    def _projection_sensitivity(self, p):
        """Same contract as Camera.projection_sensitivity, by autograd.

        The base version is a pinhole Jacobian that ignores distortion
        altogether, which for an equidistant lens is wrong by d(theta_d)/dr.
        """
        p = self._tensor(p).reshape(-1, 3)
        ext = self.get_extrinsics_mat().detach()
        matrix = self.matrix.detach()
        dist = self.dist.detach()

        with torch.enable_grad():
            p = p.detach().requires_grad_(True)
            out = project_points_fisheye(p, ext, matrix, dist)  # offset is a constant shift
            # each output row depends only on its own input row, so one backward
            # pass per output column gives all the per-row gradients
            rows = [torch.autograd.grad(out[:, i].sum(), p, retain_graph=(i == 0))[0]
                    for i in range(2)]

        return torch.stack(rows, dim=1)


# cv2.fisheye has no CALIB_FIX_K0, the flags start at k1
_FISHEYE_FIX_K = [cv2.fisheye.CALIB_FIX_K1, cv2.fisheye.CALIB_FIX_K2,
                  cv2.fisheye.CALIB_FIX_K3, cv2.fisheye.CALIB_FIX_K4]


def init_camera_intrinsics(camera, objp, imgp, size, *, guard=False):
    """Seed a camera's intrinsics from board detections, per projection model.

    ``guard=True`` (or a dict of keyword arguments) runs the guarded
    :func:`aniposelib.init.board.init_camera_intrinsics` instead (plausibility check of the
    seed with a fallback, returns an ``IntrinsicsSeed``); default False: the classic seed.

    The pinhole path gets a camera matrix from cv2.initCameraMatrix2D and starts
    from zero distortion. There is no fisheye equivalent of that, so the fisheye
    path runs a full cv2.fisheye.calibrate, which also gives usable k1..k4 --
    starting a fisheye camera at zero distortion throws away the equidistant
    term the model exists to capture.

    Only the first n_dist coefficients are seeded; the rest keep the values they
    already had. Note that with n_dist=0 the fisheye fit is done with every
    coefficient pinned to zero, which will not reproduce a camera whose frozen
    coefficients are non-zero -- pass init_intrinsics=False for that.
    """
    if guard:
        from aniposelib.init import board as _board
        return _board.init_camera_intrinsics(camera, objp, imgp, size,
                                             **(guard if isinstance(guard, dict) else {}))
    n_dist = camera.get_n_dist()
    # cv2 overwrites the coefficients it is told to fix with 0 rather than
    # holding them where they were, so keep a copy to restore afterwards
    old_dist = camera.get_distortions().detach().cpu().numpy().copy()

    def restore_frozen(new_dist):
        new_dist = np.asarray(new_dist, dtype='float64').ravel()
        keep = min(len(new_dist), len(old_dist))
        new_dist[n_dist:keep] = old_dist[n_dist:keep]
        return new_dist

    if camera.fisheye:
        # cv2.fisheye.calibrate is picky: it wants (1, N, 3) / (1, N, 2) float64
        obj64 = [np.asarray(o, dtype='float64').reshape(1, -1, 3) for o in objp]
        img64 = [np.asarray(i, dtype='float64').reshape(1, -1, 2) for i in imgp]
        K = np.eye(3)
        D = np.zeros((4, 1))
        flags = (cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC
                 | cv2.fisheye.CALIB_FIX_SKEW)
        for fix in _FISHEYE_FIX_K[n_dist:]:
            flags |= fix
        try:
            cv2.fisheye.calibrate(obj64, img64, tuple(size), K, D, flags=flags)
            camera.set_camera_matrix(K.copy())
            camera.set_distortions(restore_frozen(D))
            return
        except cv2.error:
            # degenerate views, fall through to the pinhole seed
            print("W: fisheye intrinsics failed for camera {}, "
                  "falling back to a pinhole seed".format(camera.get_name()))

    matrix = cv2.initCameraMatrix2D(objp, imgp, tuple(size))
    camera.set_camera_matrix(matrix.copy())
    # initCameraMatrix2D estimates no distortion, so the free coefficients start
    # at zero; the frozen ones stay where they were
    camera.set_distortions(restore_frozen(np.zeros_like(old_dist)))


def _natural_key(key):
    """Sort key: 'cam_10' -> ('cam_', 10). Keys without a numeric suffix sort lexically
    (and before numbered keys with the same prefix)."""
    m = re.match(r'^(.*?)(\d+)$', str(key))
    if m is None:
        return (str(key), -1)
    return (m.group(1), int(m.group(2)))


def _order_keys(keys, order='natural'):
    keys = list(keys)
    if order == 'natural':
        out = sorted(keys, key=_natural_key)
        if out != sorted(keys):
            warnings.warn(
                "camera keys sorted numerically ({}) differ from their lexical order ({}). "
                "aniposelib <= 0.9 used the lexical order, so results computed with it on this "
                "file had the cameras permuted. Pass order='lexical' to get the old order.".format(
                    out, sorted(keys)), CameraOrderWarning, stacklevel=3)
        return out
    if order == 'lexical':
        return sorted(keys)
    if order == 'file':
        return keys
    raise ValueError("order must be 'natural', 'lexical' or 'file', got {!r}".format(order))


def _warn_duplicate_names(names):
    dup = sorted(n for n, c in Counter(names).items() if c > 1)
    if dup:
        warnings.warn("duplicate camera names {}: subset_cameras_names and name lookups will "
                      "be ambiguous".format(dup), CameraOrderWarning, stacklevel=3)


def _board_with_detector(board, detector, detector_kwargs=None):
    """A shallow copy of a CharucoBoard that uses another detector (board is not modified)."""
    if not hasattr(board, 'detector_name'):
        raise TypeError('detector= is only supported for CharucoBoard, got {}'.format(
            type(board).__name__))
    b = copy(board)
    b.detector_kwargs = dict(board.detector_kwargs if detector_kwargs is None else detector_kwargs)
    if isinstance(detector, str):
        if detector not in ('opencv', 'deepcharuco'):
            raise ValueError("detector must be 'opencv', 'deepcharuco' or a detector instance")
        b.detector_name = detector
        b._detector = None
    else:
        if not hasattr(detector, 'detect'):
            raise TypeError('a detector instance needs a detect(image) method')
        b.detector_name = type(detector).__name__
        b._detector = detector
    b.validate = b.detector_name != 'opencv' or board.validate
    return b


def _is_index_seq(x):
    if isinstance(x, (str, bytes, dict)):
        return False
    arr = np.asarray(x, dtype=object) if not isinstance(x, np.ndarray) else x
    if arr.ndim != 1:
        return False
    return all(isinstance(v, (int, np.integer)) for v in arr.tolist())


def _frames_for(frame_indices, cix, vnum, vidname, cam_name, n_cams):
    """Resolve get_rows_videos(frame_indices=) for one video (None = default sampling)."""
    if frame_indices is None:
        return None
    if isinstance(frame_indices, dict):
        if vidname in frame_indices:
            return frame_indices[vidname]
        if cam_name in frame_indices:
            v = frame_indices[cam_name]
            return v if _is_index_seq(v) else v[vnum]
        return None
    if _is_index_seq(frame_indices):
        return frame_indices
    if len(frame_indices) != n_cams:
        raise ValueError('frame_indices must be a sequence of ints, one entry per camera '
                         '({}), or a dict'.format(n_cams))
    v = frame_indices[cix]
    if v is None or _is_index_seq(v):
        return v
    return v[vnum]


class CameraGroup(nn.Module):
    """A list of cameras (an nn.Module). Points are C x N x 2 in group order, NaN = missing.

    Public read-only methods (project, triangulate*, reprojection_error, undistort_points,
    average_error, optim_points, ...) follow the input type: numpy in -> numpy out, torch in
    -> detached torch out on the input's device; return_grad=True keeps the autograd graph.

    ``stations``: optional :class:`aniposelib.stations.Stations` (time intervals of moved
    cameras). It is saved in the calibration file (``[metadata.stations]``), kept by
    ``copy``/``subset_cameras*``, and used by ``bundle_adjust_lm`` for timed observations."""

    def __init__(self, cameras, metadata=None, dtype=None, stations=None):
        super().__init__()
        self.cameras = nn.ModuleList(cameras)
        self.metadata = {} if metadata is None else metadata
        if stations is not None and not hasattr(stations, 'lookup'):
            from aniposelib.stations import Stations
            stations = Stations.from_toml_dict(stations)
        self.stations = stations
        if dtype is not None:
            self.to(dtype)

    @property
    def device(self):
        for p in self.parameters():
            return p.device
        return torch.device('cpu')

    @property
    def dtype(self):
        for p in self.parameters():
            return p.dtype
        return torch.float64

    def _tensor(self, x):
        return to_tensor(x, device=self.device, dtype=self.dtype)

    def _stations_copy(self):
        st = getattr(self, 'stations', None)
        return None if st is None else st.copy()

    def subset_cameras(self, indices):
        cams = [self.cameras[ix].copy() for ix in indices]
        return CameraGroup(cams, deepcopy(self.metadata), stations=self._stations_copy())

    def subset_cameras_names(self, names, missing='raise'):
        """A new group with the cameras ``names``, in that order. ``missing='raise'`` (default)
        raises IndexError for a name not in the group; ``'drop'`` skips it."""
        if missing not in ('raise', 'drop'):
            raise ValueError("missing must be 'raise' or 'drop'")
        cur_names = self.get_names()
        cur_names_dict = dict(zip(cur_names, range(len(cur_names))))
        indices = []
        for name in names:
            if name not in cur_names_dict:
                if missing == 'drop':
                    continue
                raise IndexError(
                    "name {} not part of camera names: {}".format(
                        name, cur_names
                    ))
            indices.append(cur_names_dict[name])
        return self.subset_cameras(indices)

    def reorder(self, names):
        """A new group with the same cameras in the order ``names`` (a permutation of
        ``get_names()``). Reorder point arrays with :func:`aniposelib.geometry.reindex_points`."""
        names = list(names)
        if sorted(map(str, names)) != sorted(map(str, self.get_names())) or len(names) != len(self.cameras):
            raise ValueError('reorder needs a permutation of {}, got {}'.format(self.get_names(), names))
        return self.subset_cameras_names(names)

    @contextmanager
    def frozen(self):
        """Context for read-only loops (tracking, per-frame triangulation): each camera's
        extrinsics matrix is built once and reused, instead of a Rodrigues per call.
        Writes through set_rotation/set_translation refresh the cache. Solvers
        (bundle_adjust*, calibrate_*) refuse to run inside it, and gradients do not flow to
        rvec/tvec while it is active."""
        prev = [getattr(cam, '_frozen_ext', None) for cam in self.cameras]
        try:
            with torch.no_grad():
                for cam in self.cameras:
                    cam._frozen_ext = make_M_torch(cam.rvec, cam.tvec).detach()
            yield self
        finally:
            for cam, p in zip(self.cameras, prev):
                cam._frozen_ext = p

    # kept as the name suggested in the EvoVision report
    frozen_extrinsics = frozen

    def _check_not_frozen(self, what):
        if any(getattr(cam, '_frozen_ext', None) is not None for cam in self.cameras):
            raise RuntimeError("{} cannot run inside CameraGroup.frozen(): the extrinsics "
                               "would be held fixed".format(what))

    # ------------------------------------------------------------ packed view (1.0, C1)

    def _packable(self):
        """True when every camera is a plain Camera / FisheyeCamera (no overridden geometry)
        that PackedCameras represents exactly (pinhole: <= 8 distortion coefficients)."""
        if len(self.cameras) == 0:
            return False
        for cam in self.cameras:
            cls = type(cam)
            if cls._project not in (Camera._project, FisheyeCamera._project):
                return False
            if cls._undistort_points not in (Camera._undistort_points, FisheyeCamera._undistort_points):
                return False
            if cls.get_extrinsics_mat is not Camera.get_extrinsics_mat:
                return False
            if not getattr(cam, 'fisheye', False) and cam.dist.shape[0] > _G.N_DIST:
                return False
            if getattr(cam, 'fisheye', False) != (cls._project is FisheyeCamera._project):
                return False
        return True

    def _pack_internal(self, grad=None):
        """PackedCameras for internal delegation: attached to the camera parameters when autograd
        is on (so bundle adjustment and return_grad=True still get gradients), and using the
        cached extrinsics inside frozen()."""
        if grad is None:
            grad = torch.is_grad_enabled()
        P = _G.pack(self, dtype=self.dtype, detach=not grad)
        frozen = [getattr(cam, '_frozen_ext', None) for cam in self.cameras]
        if all(f is not None for f in frozen):
            P.ext = torch.stack([f.to(device=P.device, dtype=P.dtype) for f in frozen])
        return P

    def _extrinsics_stack(self):
        return torch.stack([cam.get_extrinsics_mat() for cam in self.cameras])

    def packed(self, device=None, dtype=None, detach=True):
        """Stacked camera parameters as a :class:`aniposelib.geometry.PackedCameras`
        (``K [C,3,3]``, ``dist [C,8]``, model tag brown/kb, ``rvec/tvec [C,3]``,
        ``offset [C,2]``, ``size [C,2]``). ``dtype`` defaults to the cameras' dtype.
        ``detach=False`` keeps the tensors attached to the cameras' parameters.
        Write an externally optimised copy back with :meth:`load_packed_`."""
        return _G.pack(self, device=device, dtype=self.dtype if dtype is None else dtype,
                       detach=detach)

    def load_packed_(self, P):
        """Write a :class:`~aniposelib.geometry.PackedCameras` back into the cameras (in place;
        matrix, distortions, rvec/tvec, size, offset). Returns self."""
        _G.unpack_into(self, P)
        for cam in self.cameras:
            cam._refresh_frozen()
        return self

    def as_tensors(self, device=None, dtype=torch.float64, detach=True, extrinsics=None):
        """Stacked tensors in tracktail's ``format_camera_group`` layout (one dict for the group):

        ``K [C,3,3]``, ``dist [C,D]`` (D = 5 if any pinhole camera, else 4; up to 8 if a camera
        uses rational terms), ``ext [C,4,4]`` world->camera, ``ext_inv``, ``center [C,3]``
        (camera centres in world coordinates), ``size [C,2]`` (width, height; NaN if unknown),
        ``offset [C,2]``, ``type`` (list of 'pinhole' / 'fisheye'), plus ``names``, ``model``
        ('brown' / 'kb') and ``moving``.

        ``extrinsics`` gives per-frame extrinsics for moving rigs: a tensor/array ``[C,T,4,4]``
        or a dict ``{camera name: [T,4,4]}`` (cameras missing from the dict keep their static
        extrinsics, repeated over T). ``ext``/``ext_inv`` are then ``[C,T,4,4]`` and ``center``
        is ``[C,T,3]``."""
        P = _G.pack(self, device=device, dtype=dtype, detach=detach)
        ext = P.extrinsics()
        if extrinsics is not None:
            if isinstance(extrinsics, dict):
                T = None
                for v in extrinsics.values():
                    T = int(np.shape(v)[0]) if not torch.is_tensor(v) else int(v.shape[0])
                    break
                exts = []
                for i, name in enumerate(P.names):
                    if name in extrinsics:
                        exts.append(_to_torch(extrinsics[name], device=P.device, dtype=P.dtype))
                    else:
                        exts.append(ext[i][None].expand(T, 4, 4))
                ext = torch.stack(exts)
            else:
                ext = _to_torch(extrinsics, device=P.device, dtype=P.dtype)
        ext_inv = torch.linalg.inv(ext)
        R, t = ext[..., :3, :3], ext[..., :3, 3]
        center = -torch.einsum('...ji,...j->...i', R, t)
        n_d = max([5 if m == 0 else 4 for m in P.model.tolist()])
        nz = (P.dist != 0).any(0).nonzero()
        if nz.numel() > 0:
            n_d = max(n_d, int(nz.max()) + 1)
        return {
            'K': P.K,
            'dist': P.dist[:, :n_d],
            'ext': ext,
            'ext_inv': ext_inv,
            'center': center,
            'size': P.size,
            'offset': P.offset,
            'type': ['fisheye' if m == 1 else 'pinhole' for m in P.model.tolist()],
            'model': P.model_names(),
            'names': list(P.names),
            'moving': ext.ndim == 4,
        }

    # ------------------------------------------------------------ projection

    @returns_like_input()
    def project(self, points):
        """Given an Nx3 array of points, this returns an CxNx2 array of 2D points,
        where C is the number of cameras"""
        return self._project(points)

    def _project(self, points):
        points = self._tensor(points)
        points = points.reshape(-1, 3)
        n_points = points.shape[0]
        if len(self.cameras) == 0:
            return torch.empty((0, n_points, 2), dtype=points.dtype, device=points.device)
        P = self._robust_packed('project')
        # sanitise non-finite points before any arithmetic so that their gradients stay
        # finite (torch.where's backward would otherwise give 0 * NaN on the cameras)
        bad = ~torch.isfinite(points).all(-1)
        X = torch.where(bad[:, None], torch.zeros_like(points), points)
        uv, _ = _G._project_flat(P, X, P.extrinsics())
        return torch.where(bad[None, :, None], torch.full_like(uv, float('nan')), uv)

    @returns_like_input()
    def undistort_points(self, points, batch_size=None, progress=False, desc="undistort"):
        """Undistort per camera: (C, ..., 2) stored-image pixels -> normalised coordinates of
        the same shape, NaN passing through. The result can go to
        triangulate(..., undistort=False), which is the fast path when the same detections
        are triangulated many times (undistortion only depends on the pixels).
        batch_size=None (default) does all points at once."""
        return self._undistort_points(points, batch_size=batch_size, progress=progress, desc=desc)

    # three downstream projects call the private name
    _undistort_points_batch = undistort_points

    def _undistort_points(self, points, batch_size=None, progress=False, desc="undistort"):
        points = self._tensor(points)
        orig_shape = points.shape
        C = orig_shape[0]
        if C == 0:
            return points.clone()
        K = points[0].numel() // 2
        points_flat = points.reshape(C, K, 2)
        P = self._robust_packed('undistort_points')
        fn = lambda p: _G._undistort_flat(P, p)
        if batch_size is None or batch_size >= K:
            return fn(points_flat).reshape(orig_shape)
        undistort_iter = trange(0, K, batch_size, ncols=70, desc=desc) if progress \
            else range(0, K, batch_size)
        out = [fn(points_flat[:, start:min(start + batch_size, K)]) for start in undistort_iter]
        return torch.cat(out, 1).reshape(orig_shape)

    # ------------------------------------------------------------ triangulation

    # points per chunk when triangulate(batch_size=None); bounds memory on huge inputs
    TRIANGULATE_CHUNK = 65536

    @returns_like_input()
    def triangulate(self, points, weights=None, batch_size=None, undistort=True, progress=False,
                    method='linear', extrinsics=None, min_cams=2):
        """Triangulate (C, N, 2) pixel observations (NaN = missing) -> (N, 3); (C, 2) -> (3,);
        (C, ..., 2) -> (..., 3). Points seen by fewer than ``min_cams`` cameras are NaN.

        method: 'linear' (default; recentred inhomogeneous block DLT with a tiny ridge, see
          aniposelib.triangulation.triangulate_linear: fast, well conditioned for far scenes)
          or 'svd' (the homogeneous SVD DLT of aniposelib <= 0.9; ~1e-6 relative difference
          on noisy data, identical to ~1e-10 on noiseless data).
        weights: (C, N) multiply each camera's DLT rows (0 removes an observation).
        undistort=False takes points already passed through undistort_points.
        extrinsics: world->camera overrides, [C,4,4], per point [C,N,4,4] (moving rigs), or
          per frame [C,T,4,4] with points shaped (C, T, ..., 2).
        batch_size=None (default) undistorts all points at once and solves in chunks of
        CameraGroup.TRIANGULATE_CHUNK points (the result does not depend on the chunking)."""
        return self._triangulate(points, weights=weights, batch_size=batch_size,
                                 undistort=undistort, progress=progress, method=method,
                                 extrinsics=extrinsics, min_cams=min_cams)

    def _group_extrinsics(self, extrinsics, n_points):
        if extrinsics is None:
            return self._extrinsics_stack()
        E = self._tensor(extrinsics)
        if E.shape[0] != len(self.cameras) or E.shape[-2:] != (4, 4):
            raise ValueError('extrinsics must be [C,4,4], [C,N,4,4] or [C,T,4,4] with C = {}, '
                             'got {}'.format(len(self.cameras), tuple(E.shape)))
        if E.ndim == 4 and E.shape[1] != n_points:
            T = E.shape[1]
            if n_points % T != 0:
                raise ValueError('per-frame extrinsics with T={} do not match {} points'.format(
                    T, n_points))
            M = n_points // T
            E = E[:, :, None].expand(E.shape[0], T, M, 4, 4).reshape(E.shape[0], n_points, 4, 4)
        return E

    def _triangulate(self, points, weights=None, batch_size=None, undistort=True, progress=False,
                     method='linear', extrinsics=None, min_cams=2):
        if method not in ('linear', 'svd'):
            raise ValueError("method must be 'linear' or 'svd', got {!r}".format(method))
        assert points.shape[0] == len(self.cameras), \
            "Invalid points shape, first dim should be equal to" \
            " number of cameras ({}), but shape is {}".format(
                len(self.cameras), points.shape
            )

        points = self._tensor(points)
        dtype = points.dtype
        if points.shape[-1] != 2:
            raise ValueError('points must end in a coordinate axis of size 2, got {}'.format(
                tuple(points.shape)))
        lead = tuple(points.shape[1:-1])
        n_cams = points.shape[0]
        points = points.reshape(n_cams, -1, 2)
        n_points = points.shape[1]

        if undistort:
            points = self._undistort_points(points, batch_size=batch_size, progress=progress)

        E = self._group_extrinsics(extrinsics, n_points)

        if weights is None:
            weights = torch.ones((n_cams, n_points), dtype=dtype, device=points.device)
        else:
            weights = to_tensor(weights, device=points.device, dtype=dtype).reshape(n_cams, n_points)

        valid_mask = torch.isfinite(points).all(dim=-1)  # [C, N]
        points = torch.where(valid_mask[..., None], points, torch.zeros_like(points))
        weights = weights * valid_mask.to(dtype)
        valid_counts = torch.sum(weights != 0, dim=0)  # [N]
        enough = valid_counts >= max(2, int(min_cams))

        out = torch.full((n_points, 3), torch.nan, device=points.device, dtype=dtype)
        chunk = self.TRIANGULATE_CHUNK if batch_size is None else batch_size
        iterator = trange(0, n_points, chunk, ncols=70, desc="triangulate") if progress \
            else range(0, n_points, chunk)
        for start in iterator:
            end = min(start + chunk, n_points)
            sel = torch.nonzero(enough[start:end])[:, 0] + start
            if sel.numel() == 0:
                continue
            Es = E if E.ndim == 3 else E[:, sel]
            p, w = points[:, sel], weights[:, sel]
            if method == 'svd':
                out[sel] = _triangulate_svd(p, Es, w)
            else:
                center = _T._default_center(Es)
                out[sel] = _T._linear_flat(None, p, w * w, Es, center, 1e-12, 0)
        return out.reshape(lead + (3,))

    @returns_like_input()
    def triangulate_robust(self, points, method='msac', **kwargs):
        """Robust triangulation of (C, ..., 2) pixel observations (thin wrapper over
        :func:`aniposelib.triangulation.triangulate_robust`) -> :class:`TriangulationResult`
        (``p3d``, ``inliers [C,...]``, ``errors [C,...]`` px for every observation,
        ``weights``, ``n_inliers``; unpacks as ``p3d, errors = result``).

        method: 'msac' (camera-pair hypotheses, capped cost, refit + Gauss-Newton), 'irls'
        (Cauchy) or 'reject' (fit, drop outliers, refit). Keywords: tau, min_inliers,
        n_refine, sigma, n_iter, min_weight, reject_px, rel_factor, contrib (bool [C]: cameras
        allowed to define points), weights, mask_out_of_bounds, undistort, extrinsics, ridge,
        chunk, scores (soft confidence weights in [0, 1]); and check_invertible=True (default):
        observations where the lens model is not invertible are not used to fit (their errors
        are still reported).
        Needs cameras that packed() can represent."""
        return self._triangulate_robust(points, method=method, **kwargs)

    def _robust_packed(self, what):
        if not self._packable():
            raise NotImplementedError(
                '{} needs plain Camera/FisheyeCamera cameras (pinhole: <= 8 distortion '
                'coefficients); this group has cameras the packed kernels cannot '
                'represent'.format(what))
        return self._pack_internal()

    @staticmethod
    def _mask_noninvertible(P, pts):
        """See aniposelib.triangulation._mask_noninvertible (any leading shape)."""
        C = pts.shape[0]
        return _T._mask_noninvertible(P, pts.reshape(C, -1, 2)).reshape(pts.shape)

    def _triangulate_robust(self, points, method='msac', check_invertible=True, **kwargs):
        # the lens-fold / ray-consistency check lives in triangulation.triangulate_robust (1.1)
        P = self._robust_packed('triangulate_robust')
        if kwargs.get('extrinsics') is not None:
            kwargs['extrinsics'] = self._tensor(kwargs['extrinsics'])
        return _T.triangulate_robust(P, self._tensor(points), method=method,
                                     check_invertible=check_invertible, return_grad=True, **kwargs)

    @returns_like_input()
    def triangulate_possible(self, points, undistort=True, min_cams=2, progress=False,
                             batch_size=None, tau=10.0, n_refine=2):
        """Given an CxNxPx2 array (P candidate detections per camera, NaN = none), returns
        (out Nx3, picked_vals CxNxP bool, points_2d CxNx2, errors N).

        1.0: pairwise hypotheses ``(cam a, cand i, cam b, cand j)`` scored MSAC-style (each
        camera scores its closest candidate, capped at ``tau`` px), refit on the inliers and
        refined (aniposelib.triangulation.triangulate_possible; memory bounded, no subset
        enumeration). A camera is picked when its chosen candidate is within ``tau``; errors
        are the mean pixel error over the picked views (NaN where no point). batch_size:
        points per chunk (None = automatic memory bound)."""
        return self._triangulate_possible(points, undistort=undistort, min_cams=min_cams,
                                          progress=progress, batch_size=batch_size, tau=tau,
                                          n_refine=n_refine)

    def _triangulate_possible(self, points, undistort=True, min_cams=2, progress=False,
                              batch_size=None, tau=10.0, n_refine=2):
        # the lens-fold check lives in triangulation.triangulate_possible (1.1)
        P = self._robust_packed('triangulate_possible')
        res = _T.triangulate_possible(P, self._tensor(points), tau=tau, min_inliers=min_cams,
                                      n_refine=n_refine, undistort=undistort, chunk=batch_size,
                                      check_invertible=undistort, return_grad=True)
        return res.legacy()

    @returns_like_input()
    def triangulate_ransac(self, points, undistort=True, min_cams=2, progress=False, tau=10.0,
                           n_refine=2):
        """Given an CxNx2 array, returns (p3ds Nx3, picked CxNx1 bool, p2ds CxNx2, errors N).

        1.0: MSAC on camera pairs (``triangulate_robust(method='msac', tau=tau,
        min_inliers=min_cams)``): picked = inlier views (error < ``tau`` px), p2ds = the
        picked observations (NaN elsewhere), errors = mean pixel error over the picked views
        (NaN where no point). For the full result (per-view errors, inlier masks) use
        triangulate_robust."""
        assert points.shape[0] == len(self.cameras), \
            "Invalid points shape, first dim should be equal to" \
            " number of cameras ({}), but shape is {}".format(
                len(self.cameras), points.shape
            )
        n_cams, n_points, _ = points.shape
        pts = self._tensor(points)
        res = self._triangulate_robust(pts, method='msac', tau=tau, min_inliers=min_cams,
                                       n_refine=n_refine, undistort=undistort,
                                       mask_out_of_bounds=False)
        inl = res.inliers & torch.isfinite(res.p3d).all(-1)[None]
        p2d = torch.where(inl[..., None], pts, torch.full_like(pts, float('nan')))
        e = torch.where(inl, torch.nan_to_num(res.errors, nan=0.0), torch.zeros_like(res.errors))
        cnt = inl.sum(0)
        err = e.sum(0) / cnt.clamp(min=1)
        err = torch.where(cnt > 0, err, torch.full_like(err, float('nan')))
        return res.p3d, inl[..., None], p2d, err

    @returns_like_input()
    def reprojection_error(self, p3ds, p2ds, mean=False):
        """Given an Nx3 array of 3D points and an CxNx2 array of 2D points,
        where N is the number of points and C is the number of cameras,
        this returns an CxNx2 array of errors.
        Optionally mean=True, this averages the errors and returns array of length N of errors"""
        return self._reprojection_error(p3ds, p2ds, mean=mean)

    def _reprojection_error(self, p3ds, p2ds, mean=False):
        p3ds = self._tensor(p3ds)
        p2ds = self._tensor(p2ds)

        one_point = False
        if len(p3ds.shape) == 1 and len(p2ds.shape) == 2:
            p3ds = p3ds.reshape(1, 3)
            p2ds = p2ds.reshape(-1, 1, 2)
            one_point = True

        n_cams, n_points, _ = p2ds.shape
        assert p3ds.shape == (n_points, 3), \
            "shapes of 2D and 3D points are not consistent: " \
            "2D={}, 3D={}".format(p2ds.shape, p3ds.shape)

        errors = p2ds - self._project(p3ds)

        if mean:
            errors_norm = torch.linalg.norm(errors, dim=2)
            good = ~torch.isnan(errors_norm)
            errors_norm = torch.where(good, errors_norm, torch.zeros_like(errors_norm))
            denom = torch.sum(good, dim=0).to(errors_norm.dtype)
            errors = torch.sum(errors_norm, axis=0) / denom

        if one_point:
            if mean:
                errors = float(errors[0])
            else:
                errors = errors.reshape(-1, 2)

        return errors


    def bundle_adjust_iter(self, p2ds, extra=None,
                           n_iters=8, start_mu=12, end_mu=0.06,
                           max_nfev=850, ftol=1e-8,
                           n_samp_iter=450, n_samp_full=1000,
                           error_threshold=0.7, only_extrinsics=False,
                           verbose=False, rng=None, *, method='lm', spec=None,
                           return_report=False, max_points=20000, resect=True,
                           trans_scale='auto'):
        """Iterative robust bundle adjustment of C x N x 2 (or C x N x J x 2) points.

        method='lm' (default since 1.0): Levenberg-Marquardt (aniposelib.bundle).
          * without `extra`: :func:`aniposelib.bundle.bundle_adjust_robust` -- MSAC triangulation,
            LM on inlier observations with an annealed threshold (`n_iters` steps from
            ~1.5 x the initial error down to 10 px, then 20/10/5 px), LOO resection of cameras
            (`resect`), structured rejection, then a final extrinsics (+ intrinsics unless
            `only_extrinsics`) stage. `max_points` bounds the points per LM solve.
          * with `extra` (boards from calibrate_rows): each board is a rigid body (6 pose
            parameters); an extrinsics-only LM pass then a full pass (Cauchy loss).
          Returns the final median reprojection error (px, python float) or a BAReport if
          `return_report`. `spec` (aniposelib.params.ParamSpec) selects the free parameters
          (default: each camera's free_intrinsics + n_dist). `start_mu`, `end_mu`, `max_nfev`,
          `ftol`, `n_samp_iter`, `n_samp_full` and `error_threshold` only affect method='adam'.
        method='adam': the aniposelib 0.9 Adam/OneCycle code path, unchanged (`spec` and
          `return_report` are not supported); returns a float.
        rng: np.random.Generator, int seed or None (fresh entropy); the global RNG is never used.
        """
        if method == 'adam':
            if spec is not None or return_report:
                raise TypeError("spec= and return_report= need method='lm'")
            from .bundle import BAError
            return BAError(self._bundle_adjust_iter_adam(
                p2ds, extra, n_iters=n_iters, start_mu=start_mu, end_mu=end_mu, max_nfev=max_nfev,
                ftol=ftol, n_samp_iter=n_samp_iter, n_samp_full=n_samp_full,
                error_threshold=error_threshold, only_extrinsics=only_extrinsics, verbose=verbose,
                rng=rng, trans_scale=trans_scale))
        if method != 'lm':
            raise ValueError("method must be 'lm' or 'adam', got {!r}".format(method))
        self._check_not_frozen('bundle_adjust_iter')
        from . import bundle as _B
        p2ds = p2ds.detach().cpu().numpy() if torch.is_tensor(p2ds) else np.asarray(p2ds, dtype='float64')
        assert p2ds.shape[0] == len(self.cameras), \
            "Invalid points shape, first dim should be equal to" \
            " number of cameras ({}), but shape is {}".format(len(self.cameras), p2ds.shape)
        if extra is not None:
            r1 = _B.bundle_adjust_lm(self, p2ds, extra=extra, spec=spec, only_extrinsics=True,
                                     loss='cauchy', f_scale=10.0, n_iter=50, rng=rng,
                                     return_report=True, verbose=verbose)
            rep = r1
            if not only_extrinsics:
                rep = _B.bundle_adjust_lm(self, p2ds, extra=extra, spec=spec, loss='cauchy',
                                          f_scale=2.0, n_iter=100, rng=rng, return_report=True,
                                          verbose=verbose)
                rep.error_before = r1.error_before
                rep.stages = [r1]
        else:
            flat = p2ds.reshape(p2ds.shape[0], -1, 2)
            from . import geometry as _G, triangulation as _T
            R0 = _T.triangulate_robust(_G.pack(self), flat, 'msac', tau=200.0)
            e0 = R0.errors[np.isfinite(R0.errors)]
            err0 = float(np.median(e0)) if len(e0) else 50.0
            tau0 = float(np.clip(1.5 * err0, 20.0, 200.0))
            n1 = max(2, int(np.ceil(n_iters / 2)))
            round1 = tuple(float(t) for t in np.geomspace(tau0, 10.0, n1))
            final = ('extrinsics',) if only_extrinsics else ('extrinsics', 'intrinsics')
            if spec is None and not only_extrinsics and getattr(self, 'param_spec', None) is None:
                # points rarely cover the image the way boards do: anchor the intrinsics to their
                # starting values so they move only where the data really asks for it
                from copy import deepcopy as _dc
                from .params import ParamSpec as _PS, IntrinsicsPrior as _IP
                spec = _PS(intrinsics=None, priors=_IP(_dc(self), **_DEFAULT_ITER_PRIOR))
            rep = _B.bundle_adjust_robust(self, p2ds, schedule=(round1, (20.0, 10.0, 5.0)),
                                          resect=resect, final=final, spec=spec, rng=rng,
                                          max_points=max_points, only_extrinsics=only_extrinsics,
                                          verbose=verbose)
        return rep if return_report else _B.BAError(rep.error)

    def bundle_adjust(self, p2ds, extra=None,
                      loss='huber',
                      ftol=1e-4,
                      max_nfev=500,
                      lr=8.0e-4,
                      only_extrinsics=False,
                      verbose=True,
                      f_scale=None,
                      rng=None, *, method='lm', spec=None, return_report=False,
                      trans_scale='auto'):
        """Bundle adjustment of C x N x 2 points (one solve, no outlier schedule).

        method='lm' (default since 1.0): Schur-complement Levenberg-Marquardt
          (:func:`aniposelib.bundle.bundle_adjust_lm`) on all observations with the robust
          `loss` ('huber' | 'cauchy' | 'soft_l1' | 'linear') of scale `f_scale` px (default 1.0
          for huber, 1.5 otherwise, as before); at most min(max_nfev, 100) iterations.
          Free parameters: extrinsics (first camera fixed as gauge), plus (unless
          only_extrinsics) each camera's free_intrinsics and first n_dist distortion
          coefficients, or `spec`. With `extra` boards are rigid bodies.
        method='adam': the aniposelib 0.9 Adam code path (lr, ftol, rng used). Its translation
          step sizes are scaled to the rig (``trans_scale='auto'``) so it no longer diverges or
          stalls depending on the length units; ``trans_scale=None`` restores the 0.9 steps.
        Returns the mean reprojection error after triangulating p2ds (python float, as before
        but no longer a tensor), or a BAReport if `return_report` (method='lm').
        """
        if method == 'adam':
            if spec is not None or return_report:
                raise TypeError("spec= and return_report= need method='lm'")
            from .bundle import BAError
            return BAError(self._bundle_adjust_adam(p2ds, extra, loss=loss, ftol=ftol, max_nfev=max_nfev,
                                                  lr=lr, only_extrinsics=only_extrinsics,
                                                  verbose=verbose, f_scale=f_scale, rng=rng,
                                                  trans_scale=trans_scale))
        if method != 'lm':
            raise ValueError("method must be 'lm' or 'adam', got {!r}".format(method))
        self._check_not_frozen('bundle_adjust')
        from . import bundle as _B
        assert p2ds.shape[0] == len(self.cameras), \
            "Invalid points shape, first dim should be equal to" \
            " number of cameras ({}), but shape is {}".format(len(self.cameras), p2ds.shape)
        if f_scale is None:
            f_scale = 1.0 if loss == 'huber' else 1.5
        rep = _B.bundle_adjust_lm(self, p2ds, extra=extra, spec=spec, loss=loss, f_scale=f_scale,
                                  n_iter=min(int(max_nfev), 100), only_extrinsics=only_extrinsics,
                                  rng=rng, return_report=True, verbose=verbose)
        if return_report:
            return rep
        with torch.no_grad():
            return _B.BAError(self._average_error(self._tensor(p2ds)))

    def bundle_adjust_lm(self, obs_or_p2ds, **kwargs):
        """Schur-complement Levenberg-Marquardt bundle adjustment; see
        :func:`aniposelib.bundle.bundle_adjust_lm` (``return_report=True`` for a BAReport)."""
        from . import bundle as _B
        st = getattr(self, 'stations', None)
        if 'stations' not in kwargs and st is not None and getattr(obs_or_p2ds, 'time', None) is not None:
            kwargs['stations'] = st  # the group's stations, looked up by obs.time
        return _B.bundle_adjust_lm(self, obs_or_p2ds, **kwargs)

    def bundle_adjust_robust(self, p2ds, **kwargs):
        """Robust bundle adjustment recipe (MSAC inliers, annealed LM, resection, structured
        rejection) -> BAReport; see :func:`aniposelib.bundle.bundle_adjust_robust`."""
        from . import bundle as _B
        return _B.bundle_adjust_robust(self, p2ds, **kwargs)

    def resect_cameras(self, p2ds, cams=None, **kwargs):
        """Leave-one-out PnP-RANSAC resection; see :func:`aniposelib.bundle.resect_cameras`."""
        from . import bundle as _B
        return _B.resect_cameras(self, p2ds, cams=cams, **kwargs)

    def place_cameras(self, p2ds, cams, **kwargs):
        """Place cameras with unknown poses; see :func:`aniposelib.bundle.place_cameras`."""
        from . import bundle as _B
        return _B.place_cameras(self, p2ds, cams, **kwargs)

    def add_camera(self, camera, p2d_cam, p2ds, **kwargs):
        """New group with ``camera`` appended and placed; see :func:`aniposelib.bundle.add_camera`."""
        from . import bundle as _B
        return _B.add_camera(self, camera, p2d_cam, p2ds, **kwargs)

    def _bundle_adjust_iter_adam(self, p2ds, extra=None,
                           n_iters=8, start_mu=12, end_mu=0.06,
                           max_nfev=850, ftol=1e-8,
                           n_samp_iter=450, n_samp_full=1000,
                           error_threshold=0.7, only_extrinsics=False,
                           verbose=False, rng=None, trans_scale='auto'):
        """bundle_adjust_iter(method='adam'): the aniposelib 0.9 code path (translation steps
        scaled to the rig unless trans_scale=None; see _bundle_adjust_adam).

        Given an CxNx2 array of 2D points,
        where N is the number of points and C is the number of cameras,
        this performs iterative bundle adjustsment to fine-tune the parameters of the cameras.
        That is, it performs bundle adjustment multiple times, adjusting the weights given to points
        to reduce the influence of outliers.
        This is inspired by the algorithm for Fast Global Registration by Zhou, Park, and Koltun

        rng: np.random.Generator, int seed or None (fresh entropy) for the point resampling;
        the global numpy RNG is never used. Pass an int for a reproducible run.
        """
        self._check_not_frozen('bundle_adjust_iter')
        rng = _as_rng(rng)

        assert p2ds.shape[0] == len(self.cameras), \
            "Invalid points shape, first dim should be equal to" \
            " number of cameras ({}), but shape is {}".format(
                len(self.cameras), p2ds.shape
            )

        if torch.is_tensor(p2ds):
            p2ds = p2ds.detach().cpu().numpy()
        p2ds_full = p2ds
        extra_full = extra

        device = self.device

        p2ds, extra = resample_points(p2ds_full, extra_full,
                                      n_samp=n_samp_full, rng=rng)
        p2ds_t = torch.as_tensor(p2ds, device=device)
        error = self._average_error(p2ds_t, median=True).item()

        if verbose:
            print('initial error: ', error)

        adaptive_start_mu = max(start_mu, error * 1.5)
        mus = np.exp(np.linspace(np.log(adaptive_start_mu), np.log(end_mu), num=n_iters))

        if verbose:
            print('n_samples: {}'.format(n_samp_iter))

        best_error = error
        best_cam_state = {k: v.clone() for k, v in self.state_dict().items()}

        f_scales = np.exp(np.linspace(np.log(8.0), np.log(0.5), num=n_iters))

        n_ext_only = 4 if error > 50 else 2

        for i in range(n_iters):
            p2ds, extra = resample_points(p2ds_full, extra_full,
                                          n_samp=n_samp_full, rng=rng)
            p2ds_t = torch.as_tensor(p2ds, device=device)
            p3ds = self._triangulate_ng(p2ds_t)
            errors_norm = self._reprojection_error_ng(p3ds, p2ds_t, mean=True).detach().cpu().numpy()

            finite_errors = errors_norm[np.isfinite(errors_norm)]
            if len(finite_errors) > 0:
                median_err = np.median(finite_errors)
                p75 = np.percentile(finite_errors, 75)
                p25 = np.percentile(finite_errors, 25)
                iqr = p75 - p25
                scheduled_mu = mus[i]
                data_mu = median_err + 0.7 * iqr
                mu = max(scheduled_mu, min(data_mu, np.percentile(finite_errors, 72)))
                mu = max(mu, end_mu)
            else:
                mu = mus[i]

            good = errors_norm < mu
            n_good = np.sum(good)

            if n_good < 10:
                if len(finite_errors) > 0:
                    mu = np.percentile(finite_errors, 95)
                else:
                    mu = mus[i]
                good = errors_norm < mu

            extra_good = subset_extra(extra, good)
            p2ds_samp, extra_samp = resample_points(
                p2ds[:, good], extra_good, n_samp=n_samp_iter, rng=rng)
            p2ds_samp = torch.as_tensor(p2ds_samp, device=device)

            error = np.median(finite_errors) if len(finite_errors) > 0 else error

            if error < error_threshold:
                break

            if verbose:
                errors_full = self._reprojection_error_ng(p3ds, p2ds_t, mean=False).detach().cpu().numpy()
                print('iter {}: error: {:.2f}, mu: {:.1f}, ratio: {:.3f}, f_scale: {:.1f}'.format(
                    i, error, mu, np.mean(good), f_scales[i]))

            progress_frac = (i + 1) / n_iters
            if progress_frac < 0.2:
                iter_nfev = max(180, max_nfev // 4)
            elif progress_frac < 0.4:
                iter_nfev = max(300, int(max_nfev * 0.5))
            else:
                iter_nfev = max(450, max_nfev)

            iter_only_ext = only_extrinsics or (i < n_ext_only)

            self._bundle_adjust_adam(p2ds_samp, extra_samp,
                               loss='cauchy', ftol=ftol,
                               max_nfev=iter_nfev, only_extrinsics=iter_only_ext,
                               verbose=verbose, f_scale=f_scales[i], rng=rng,
                               trans_scale=trans_scale)

            # Track best state
            cur_p2ds, _ = resample_points(p2ds_full, extra_full, n_samp=n_samp_full, rng=rng)
            cur_p2ds_t = torch.as_tensor(cur_p2ds, device=device)
            cur_error = self._average_error(cur_p2ds_t, median=True).item()
            if cur_error < best_error:
                best_error = cur_error
                best_cam_state = {k: v.clone() for k, v in self.state_dict().items()}

        # Restore best state before final passes
        self.load_state_dict(best_cam_state)

        # Final bundle adjustment with cleaned points
        for final_pass in range(3):
            p2ds, extra = resample_points(p2ds_full, extra_full,
                                          n_samp=n_samp_full, rng=rng)
            p2ds_t = torch.as_tensor(p2ds, device=device)
            p3ds = self._triangulate_ng(p2ds_t)
            errors_norm = self._reprojection_error_ng(p3ds, p2ds_t, mean=True).detach().cpu().numpy()

            if verbose:
                errors_full = self._reprojection_error_ng(p3ds, p2ds_t, mean=False).detach().cpu().numpy()

            finite_errors = errors_norm[np.isfinite(errors_norm)]
            if len(finite_errors) > 0:
                median_err = np.median(finite_errors)
                p75 = np.percentile(finite_errors, 75)
                p25 = np.percentile(finite_errors, 25)
                iqr = p75 - p25
                mu = max(median_err + max(0.3 - 0.08 * final_pass, 0.06) * iqr, end_mu)
            else:
                mu = end_mu

            good = errors_norm < mu
            extra_good = subset_extra(extra, good)
            final_f_scale = max(0.8 - 0.15 * final_pass, 0.3)
            if np.sum(good) >= 10:
                self._bundle_adjust_adam(p2ds_t[:, good], extra_good,
                                   loss='cauchy',
                                   ftol=ftol * 0.1, max_nfev=max(700, max_nfev),
                                   only_extrinsics=only_extrinsics,
                                   verbose=verbose, f_scale=final_f_scale, rng=rng,
                                   trans_scale=trans_scale)

            # Track best after each final pass
            cur_p2ds, _ = resample_points(p2ds_full, extra_full, n_samp=n_samp_full, rng=rng)
            cur_p2ds_t = torch.as_tensor(cur_p2ds, device=device)
            cur_error = self._average_error(cur_p2ds_t, median=True).item()
            if cur_error < best_error:
                best_error = cur_error
                best_cam_state = {k: v.clone() for k, v in self.state_dict().items()}

        # Check if final adjustment improved
        p2ds_check, _ = resample_points(p2ds_full, extra_full, n_samp=n_samp_full, rng=rng)
        p2ds_check_t = torch.as_tensor(p2ds_check, device=device)
        final_error = self._average_error(p2ds_check_t, median=True).item()

        if final_error > best_error:
            self.load_state_dict(best_cam_state)
            error = best_error
            if verbose:
                print('restored best state with error: ', best_error)
        else:
            error = final_error

        if verbose:
            p2ds_t2 = torch.as_tensor(
                resample_points(p2ds_full, extra_full, n_samp=n_samp_full, rng=rng)[0],
                device=device)
            p3ds = self._triangulate_ng(p2ds_t2)
            errors_full = self._reprojection_error_ng(p3ds, p2ds_t2, mean=False).detach().cpu().numpy()
            error_dict = get_error_dict(errors_full)
            pprint(error_dict)
            print('final error: ', error)

        return error

    def _bundle_adjust_adam(self, p2ds, extra=None,
                      loss='huber',
                      ftol=1e-4,
                      max_nfev=500,
                      lr=8.0e-4,
                      only_extrinsics=False,
                      verbose=True,
                      f_scale=None,
                      rng=None,
                      trans_scale='auto'):
        """bundle_adjust(method='adam'): the aniposelib 0.9 Adam optimiser (returns a tensor).

        trans_scale: Adam's step size is in parameter units, so translation-like parameters
        (camera tvecs, 3D points, board translations) get their learning rate multiplied by
        ``rig_scale / ADAM_REF_SCALE`` (and their gradients are clipped in those normalised
        units). This makes the optimiser invariant to the rig's length units (mm, m, unit-scale
        SfM rigs). 'auto' estimates rig_scale as the median camera-to-point distance; a float
        sets rig_scale; None reproduces the 0.9 absolute step sizes exactly.

        Given an CxNx2 array of 2D points,
        where N is the number of points and C is the number of cameras,
        this performs bundle adjustment to fine-tune the parameters of the cameras.

        Free parameters: extrinsics, plus (unless only_extrinsics) the camera-matrix entries
        in each camera's free_intrinsics (default fx, fy, cx, cy) and the first n_dist
        distortion coefficients. rng seeds the board-pose initialisation (extra=...)."""
        self._check_not_frozen('bundle_adjust')
        rng = _as_rng(rng)
        
        assert p2ds.shape[0] == len(self.cameras), \
            "Invalid points shape, first dim should be equal to" \
            " number of cameras ({}), but shape is {}".format(
                len(self.cameras), p2ds.shape
            )

        device = next(self.parameters()).device
        
        p2ds = to_tensor(p2ds, device=device)
        
        if extra is not None:
            extra = dict(extra)
            extra['ids_map'] = remap_ids(extra['ids'])
            extra['objp'] = torch.as_tensor(extra['objp'], device=device)
            objp = extra['objp']
            extra['min_scale'] = torch.amin(objp[objp > 0])

        params = self._initialize_params_bundle(p2ds, extra, rng=rng)
        
        if only_extrinsics:
            cam_params = list(self.get_extrinsics_params())
        else:
            cam_params = list(self.parameters())

        k = self._adam_trans_factor(params['p3d'], trans_scale)
        tvec_ids = {id(cam.tvec) for cam in self.cameras}
        cam_trans = [p for p in cam_params if id(p) in tvec_ids]
        cam_other = [p for p in cam_params if id(p) not in tvec_ids]
        trans_params = cam_trans + [params['p3d']] + ([params['tvecs']] if 'rvecs' in params else [])

        param_groups = [
            {'params': cam_other, 'lr': lr},
            {'params': cam_trans, 'lr': lr * k},
            {'params': [params['p3d']], 'lr': lr * 65.0 * k},
        ]
        max_lr = [lr * 65.0, lr * 65.0 * k, lr * 260.0 * k]
        if 'rvecs' in params:
            param_groups.append({'params': [params['rvecs']], 'lr': lr * 30.0})
            param_groups.append({'params': [params['tvecs']], 'lr': lr * 30.0 * k})
            max_lr += [lr * 130.0, lr * 130.0 * k]

        all_params = cam_params + list(params.values())

        # weight_decay=0 is load-bearing for Camera.n_dist: a frozen distortion
        # coefficient gets a zero gradient, and Adam then leaves it exactly
        # alone, but any weight decay would shrink it every step regardless.
        optimizer = optim.Adam(param_groups, weight_decay=0, amsgrad=True)
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=max_lr,
            total_steps=max_nfev,
            pct_start=0.18,
            anneal_strategy='cos',
            final_div_factor=5000.0
        )

        best_loss = torch.inf
        best_state = None
        best_cam_state = None
        patience_counter = 0
        patience = 150
        min_iters = 20

        old_loss = torch.inf
        for i in range(max_nfev):
            optimizer.zero_grad()
            loss_val = self._error_fun_bundle(params, p2ds, extra, loss=loss, f_scale=f_scale)

            current_loss = loss_val.item()
            if current_loss < best_loss:
                best_loss = current_loss
                best_state = {k: v.data.clone() for k, v in params.items()}
                best_cam_state = {k: v.clone() for k, v in self.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1

            delta = old_loss - current_loss
            if verbose and i % 50 == 0:
                print("iter: {} \t loss: {:.4f}\t delta: {:.5f}".format(i, current_loss, delta))
            
            if i > min_iters and patience_counter > patience:
                if verbose:
                    print("iter: {} \t loss: {:.4f}\t early stop (patience)".format(i, current_loss))
                break
            if i > min_iters and abs(delta) < ftol * 0.01 and old_loss != torch.inf:
                if verbose:
                    print("iter: {} \t loss: {:.4f}\t early stop (ftol)".format(i, current_loss))
                break
            old_loss = current_loss
            loss_val.backward()
            if k != 1.0:
                # clip in normalised units (theta / rig_scale) so clipping is scale-invariant too
                for p in trans_params:
                    if p.grad is not None:
                        p.grad.mul_(k)
            torch.nn.utils.clip_grad_norm_(all_params, max_norm=10.0)
            if k != 1.0:
                for p in trans_params:
                    if p.grad is not None:
                        p.grad.div_(k)
            optimizer.step()
            scheduler.step()

        # Restore best parameters
        if best_state is not None:
            for k, v in params.items():
                v.data.copy_(best_state[k])
            self.load_state_dict(best_cam_state)

        if verbose:
            print("iter: {} \t loss: {:.4f} (best: {:.4f})".format(i, current_loss, best_loss))
            
        with torch.no_grad():
            error = self._average_error(p2ds)
        return error

    def _adam_trans_factor(self, p3d, trans_scale):
        """Learning-rate multiplier for translation-like parameters in the Adam BA."""
        if trans_scale is None:
            return 1.0
        if trans_scale == 'auto':
            with torch.no_grad():
                P = p3d.detach().reshape(-1, 3)
                P = P[torch.isfinite(P).all(dim=1)]
                if len(P) > 2000:
                    P = P[torch.linspace(0, len(P) - 1, 2000, device=P.device).long()]
                centers = torch.stack([torch.as_tensor(c.get_center_world(), device=P.device,
                                                       dtype=P.dtype).reshape(3)
                                       for c in self.cameras])
                if len(P) == 0:
                    d = torch.cdist(centers, centers)
                    scale = float(torch.median(d[d > 0])) if (d > 0).any() else 1.0
                else:
                    scale = float(torch.median(torch.cdist(centers, P)))
        else:
            scale = float(trans_scale)
        if not np.isfinite(scale) or scale <= 0:
            return 1.0
        return scale / ADAM_REF_SCALE

    def _error_fun_bundle(self, params, p2ds, extra, loss='linear', f_scale=None):
        """Error function for bundle adjustment"""
        p3d = params['p3d']
        valid = torch.isfinite(p2ds)
        proj = self._project(p3d)
        diff = proj - p2ds
        diff = torch.where(valid, diff, torch.zeros_like(diff))
        
        n_valid = valid[..., 0].sum().clamp(min=1)
        
        if loss == 'huber':
            delta = 1.0
            abs_diff = torch.abs(diff)
            loss_reproj = torch.where(
                abs_diff < delta,
                0.5 * diff**2,
                delta * (abs_diff - 0.5 * delta)
            )
        elif loss == 'cauchy':
            delta = f_scale if f_scale is not None else 1.5
            loss_reproj = delta**2 * torch.log(1 + (diff / delta)**2)
        else:
            loss_reproj = diff**2

        loss_reproj = torch.sum(loss_reproj) / n_valid
        total_loss = loss_reproj

        if extra is not None:
            ids = extra['ids_map']
            expected = transform_points(extra['objp'],
                                        params['rvecs'][ids],
                                        params['tvecs'][ids])
            loss_obj = torch.mean(torch.linalg.norm(expected - p3d, dim=-1))
            total_loss = total_loss + loss_obj / extra['min_scale']

        return total_loss


    def _initialize_params_bundle(self, p2ds, extra, rng=None):
        """Given an CxNx2 array of 2D points,
        where N is the number of points and C is the number of cameras,
        initializes the parameters for bundle adjustment"""

        n_cams, n_points, _ = p2ds.shape
        assert n_cams == len(self.cameras), \
            "number of cameras in CameraGroup does not " \
            "match number of cameras in 2D points given"

        p3ds = self._triangulate_ng(p2ds)
        params = {
            'p3d': nn.Parameter(p3ds)
        }

        if extra is not None:
            ids = extra['ids_map']
            n_boards = int(np.max(ids[~np.isnan(ids)])) + 1
            valid = torch.isfinite(p2ds[:, :, 0]).detach().cpu().numpy()
            
            # initialize to 0
            rvecs = np.zeros((n_boards, 3), dtype='float64')
            tvecs = np.zeros((n_boards, 3), dtype='float64')

            if 'rvecs' in extra and 'tvecs' in extra:
                rvecs_all = extra['rvecs']
                tvecs_all = extra['tvecs']
                for board_num in range(n_boards):
                    point_id = np.where(ids == board_num)[0][0]
                    cam_ids_possible = np.where(valid[:, point_id])[0]
                    cam_id = _as_rng(rng).choice(cam_ids_possible)
                    M_cam = self.cameras[cam_id].get_extrinsics_mat().detach().cpu().numpy()
                    M_board_cam = make_M(rvecs_all[cam_id, point_id],
                                         tvecs_all[cam_id, point_id])
                    M_board = np.matmul(inv(M_cam), M_board_cam)
                    rvec, tvec = get_rtvec(M_board)
                    rvecs[board_num] = rvec
                    tvecs[board_num] = tvec

            params['rvecs'] = nn.Parameter(torch.as_tensor(rvecs, device=p2ds.device))
            params['tvecs'] = nn.Parameter(torch.as_tensor(tvecs, device=p2ds.device))

        return params
    @returns_like_input()
    def optim_points(self, points, p3ds,
                     constraints=[],
                     constraints_weak=[],
                     scale_smooth=4,
                     scale_length=2, scale_length_weak=0.5,
                     reproj_error_threshold=15, reproj_loss='soft_l1',
                     n_deriv_smooth=1, scores=None, verbose=False,
                     n_fixed=0,
                     max_lr=0.1,
                     n_iters=800,
                     fill='interior',
                     normalize=True):
        """
        Take in an array of 2D points of shape CxNxJx2,
        an array of 3D points of shape NxJx3,
        and an array of constraints of shape Kx2, where
        C: number of camera
        N: number of frames
        J: number of joints
        K: number of constraints

        This function creates an optimized array of 3D points of shape NxJx3.

        Example constraints:
        constraints = [[0, 1], [1, 2], [2, 3]]
        (meaning that lengths of segments 0->1, 1->2, 2->3 are all constant)

        scores: optional CxNxJ confidences; each reprojection residual is multiplied by its
          score (soft weights, use values in [0, 1]).
        normalize: True (default since 1.1): the optimisation runs in *pixel-equivalent*
          units, so it does not depend on the units of the rig (mm or m give the same result).
          The 3D points are parametrised as ``center + u * q`` with ``u`` the median world
          length per pixel at the points (median of z / f over the observations), the Adam
          step ``max_lr`` is in pixels, and the smoothness and bone terms are squared
          pixel-equivalent residuals: smoothness ``0.01 * scale_smooth * |diff_q|^2 /
          mean|diff_q|``,
          bones ``scale_length * 10 * (L_q - l_q)^2`` (strong) and ``scale_length_weak * 10 *
          ...`` (weak), with ``L_q`` a bone length and ``l_q`` its target length in pixel
          units, fixed at the robust estimate (median over the frames where both joints are
          triangulated in ``p3ds``; NaN cells of ``p3ds`` are not used). The bone terms then
          tighten bone lengths within a short budget.
          False: the aniposelib <= 1.0 loss and parametrisation (world units; the effective
          weights and the Adam step depend on the units of the rig).
        fill: what to return where a joint is not seen by at least 2 cameras.
          'interior' (default): gaps between the first and last observed frame of each joint
            are filled by the optimisation; frames before the first / after the last
            observation are NaN (no extrapolation).
          'none': every frame with fewer than 2 views is NaN.
          'all': fill everything (aniposelib <= 0.9 behaviour: leading/trailing runs held
            constant, never-seen joints at the origin, then optimised).
        The optimisation itself always runs on the fully filled initialisation; fill only
        decides which cells are returned.
        """
        return self._optim_points(points, p3ds, constraints=constraints,
                                  constraints_weak=constraints_weak,
                                  scale_smooth=scale_smooth, scale_length=scale_length,
                                  scale_length_weak=scale_length_weak,
                                  reproj_error_threshold=reproj_error_threshold,
                                  reproj_loss=reproj_loss, n_deriv_smooth=n_deriv_smooth,
                                  scores=scores, verbose=verbose, n_fixed=n_fixed,
                                  max_lr=max_lr, n_iters=n_iters, fill=fill,
                                  normalize=normalize)

    def _optim_unit(self, p3ds, points):
        """(u, center): median world length per pixel at the points (z / f over the finite
        observations) and the median point, for the unit-free optim_points parametrisation."""
        X = to_tensor(np.asarray(p3ds, dtype='float64').reshape(-1, 3), device=self.device,
                      dtype=self.dtype)
        okX = torch.isfinite(X).all(-1)
        center = torch.nanmedian(X[okX], dim=0).values.detach() if bool(okX.any()) else \
            torch.zeros(3, dtype=self.dtype, device=self.device)
        pts = to_tensor(points, device=self.device, dtype=self.dtype).reshape(len(self.cameras), -1, 2)
        vals = []
        for ci, cam in enumerate(self.cameras):
            E = cam.get_extrinsics_mat().to(self.dtype)
            K = cam.get_camera_matrix().to(self.dtype)
            f = 0.5 * (K[0, 0].abs() + K[1, 1].abs())
            z = (torch.nan_to_num(X) @ E[:3, :3].T + E[:3, 3])[:, 2]
            ok = okX & (z > 0) & torch.isfinite(pts[ci]).all(-1)
            if bool(ok.any()):
                vals.append(z[ok] / f)
        if vals:
            u = float(torch.median(torch.cat(vals)).detach())
        else:
            d = X[okX] - center
            u = float(d.norm(dim=-1).median()) / 1000.0 if d.shape[0] else 1.0
        if not np.isfinite(u) or u <= 0:
            u = 1.0
        return u, center

    def _optim_points(self, points, p3ds,
                      constraints=[],
                      constraints_weak=[],
                      scale_smooth=4,
                      scale_length=2, scale_length_weak=0.5,
                      reproj_error_threshold=15, reproj_loss='soft_l1',
                      n_deriv_smooth=1, scores=None, verbose=False,
                      n_fixed=0,
                      max_lr=0.1,
                      n_iters=800,
                      fill='interior',
                      normalize=True):
        if fill not in ('interior', 'none', 'all'):
            raise ValueError("fill must be 'interior', 'none' or 'all', got {!r}".format(fill))
        assert points.shape[0] == len(self.cameras), \
            "Invalid points shape, first dim should be equal to" \
            " number of cameras ({}), but shape is {}".format(
                len(self.cameras), points.shape
            )

        n_cams, n_frames, n_joints, _ = points.shape
        device = self.device
        dtype = self.dtype
        constraints = np.array(constraints)
        constraints_weak = np.array(constraints_weak)

        if torch.is_tensor(p3ds):
            p3ds = p3ds.detach().cpu().numpy()
        else:
            p3ds = np.array(p3ds)

        if fill != 'all':
            pts_np = points.detach().cpu().numpy() if torch.is_tensor(points) else np.asarray(points)
            observed = np.sum(np.all(np.isfinite(pts_np), axis=-1), axis=0) >= 2  # (N, J)
            keep = interior_span(observed) if fill == 'interior' else observed
        else:
            keep = None

        p3ds_intp = np.apply_along_axis(interpolate_data, 0, p3ds)

        if normalize:
            unit, center = self._optim_unit(p3ds, points)
            center_np = center.detach().cpu().numpy()
            # never-triangulated columns start at the scene centre, not at the world origin
            never = ~np.isfinite(p3ds).all(-1).any(axis=0)   # (J,)
            if never.any():
                p3ds_intp = p3ds_intp.copy()
                p3ds_intp[:, never] = center_np
            q_intp = (p3ds_intp - center_np) / unit
        else:
            unit, center = None, None
            q_intp = p3ds_intp

        p3ds_med = np.apply_along_axis(medfilt_data, 0, q_intp, size=7)
        motion = np.mean(np.abs(np.diff(p3ds_med, axis=0))) if n_frames > 1 else 0.0
        if normalize:
            motion = max(motion, 1e-3)  # pixel units: a static scene does not blow up
        default_smooth = 1.0/motion
        scale_smooth_full = scale_smooth * default_smooth
        if normalize:
            # pixel-equivalent smoothness: scale_smooth=4 is a light regulariser that does not
            # bias a moving subject (see optim_points)
            scale_smooth_full = 0.01 * scale_smooth_full

        t1 = time.time()

        params = self._initialize_params_triangulation(
            q_intp, constraints, constraints_weak, device=device,
            p3ds_lengths=((p3ds - center_np) / unit) if normalize else None)

        if n_fixed > 0:
            p3ds_fixed = to_tensor(q_intp[:n_fixed], device=device, dtype=dtype)
        else:
            p3ds_fixed = None

        points_tensor = to_tensor(points, device=device, dtype=dtype)
        scores = to_tensor(scores, device=device, dtype=dtype)
        if normalize:
            # squared pixel-equivalent bone residuals (see optim_points)
            scale_length_eff = 10.0 * scale_length
            scale_length_weak_eff = 10.0 * scale_length_weak
        else:
            scale_length_eff, scale_length_weak_eff = scale_length, scale_length_weak

        all_params = [v for v in params.values() if isinstance(v, nn.Parameter)]
        optimizer = optim.Adam(all_params, lr=1e-2, weight_decay=0, amsgrad=True)
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=max_lr, total_steps=n_iters,
            pct_start=0.2, anneal_strategy='cos', final_div_factor=5000.0)

        # Best-params tracking is done entirely on-GPU. Calling `loss.item()` every
        # iteration forces a CPU<->GPU sync that serializes the loop and leaves the GPU
        # idle between tiny kernels (nvtop shows ~0% util). Instead we keep best_loss as a
        # tensor and update best_state branchlessly with torch.where, so nothing is read
        # back to the host inside the loop and the iterations can pipeline.
        best_loss = torch.tensor(float('inf'), device=device, dtype=dtype)
        best_state = {k: v.detach().clone() for k, v in params.items()}

        # the cameras are fixed here: pack them once, detached (no per-iteration packing,
        # rodrigues or camera-parameter gradients; the point gradients are unchanged)
        proj_fn = None
        if len(self.cameras) and self._packable():
            P_fix = self._pack_internal(grad=False)
            E_fix = P_fix.extrinsics().detach()

            def proj_fn(X):
                bad = ~torch.isfinite(X).all(-1)
                X = torch.where(bad[:, None], torch.zeros_like(X), X)
                uv, _ = _G._project_flat(P_fix, X, E_fix)
                return torch.where(bad[None, :, None], torch.full_like(uv, float('nan')), uv)

        # the public wrapper runs under no_grad; the fit itself needs autograd
        with torch.enable_grad():
            for i in range(n_iters):
                optimizer.zero_grad()
                loss = self._error_fun_triangulation(params, points_tensor, proj_fn=proj_fn,
                                                     constraints=constraints,
                                                     constraints_weak=constraints_weak,
                                                     scores=scores,
                                                     scale_smooth=scale_smooth_full,
                                                     scale_length=scale_length_eff,
                                                     scale_length_weak=scale_length_weak_eff,
                                                     reproj_error_threshold=reproj_error_threshold,
                                                     reproj_loss=reproj_loss,
                                                     n_deriv_smooth=n_deriv_smooth,
                                                     p3ds_fixed=p3ds_fixed,
                                                     unit=unit, center=center
                                                     )

                # snapshot best params before stepping; pure GPU ops, no host sync
                with torch.no_grad():
                    improved = loss < best_loss
                    best_loss = torch.where(improved, loss.detach(), best_loss)
                    for k, v in params.items():
                        best_state[k] = torch.where(improved, v.detach(), best_state[k])

                if verbose and i % 20 == 0:
                    # opt-in debug print; the .item() here only runs when verbose is set
                    print("iter: {} \t loss: {:.3f}".format(i, loss.item()))

                loss.backward()
                torch.nn.utils.clip_grad_norm_(all_params, max_norm=10.0)
                optimizer.step()
                scheduler.step()

        # Restore best parameters (GPU copy, no host sync)
        with torch.no_grad():
            for k, v in params.items():
                v.data.copy_(best_state[k])

        t2 = time.time()

        if verbose:
            print("iter: {} \t best loss: {:.3f}".format(i, best_loss.item()))
            print('optimization took {:.2f} seconds'.format(t2 - t1))

        out = params['p3d'].detach()
        if p3ds_fixed is not None:
            out = torch.vstack([p3ds_fixed, out[p3ds_fixed.shape[0]:]])
        if normalize:
            out = center + unit * out
        if keep is not None:
            out = out.clone()
            out[~torch.as_tensor(keep, device=out.device)] = float('nan')
        return out

    def optim_points_possible(self, points, p3ds,
                              constraints=[],
                              constraints_weak=[],
                              scale_smooth=4,
                              scale_length=2, scale_length_weak=0.5,
                              reproj_error_threshold=15, reproj_loss='soft_l1',
                              n_deriv_smooth=1, scores=None, verbose=False):
        """
        Take in an array of 2D points of shape CxNxJxPx2,
        an array of 3D points of shape NxJx3,
        and an array of constraints of shape Kx2, where
        C: number of camera
        N: number of frames
        J: number of joints
        P: number of possible options per point
        K: number of constraints

        This function creates an optimized array of 3D points of shape NxJx3.

        Example constraints:
        constraints = [[0, 1], [1, 2], [2, 3]]
        (meaning that lengths of segments 0->1, 1->2, 2->3 are all constant)

        """
        assert points.shape[0] == len(self.cameras), \
            "Invalid points shape, first dim should be equal to" \
            " number of cameras ({}), but shape is {}".format(
                len(self.cameras), points.shape
            )

        raise NotImplementedError("optim_points_possible not converted to pytorch yet, post an issue on github or email Lili Karashchuk if you need this function")
        

    @returns_like_input()
    def triangulate_optim(self, points, init_ransac=False, init_progress=False,
                          **kwargs):
        """
        Take in an array of 2D points of shape CxNxJx2, and an array of constraints of shape Kx2, where
        C: number of camera
        N: number of frames
        J: number of joints
        K: number of constraints

        This function creates an optimized array of 3D points of shape NxJx3.

        Example constraints:
        constraints = [[0, 1], [1, 2], [2, 3]]
        (meaning that lengths of segments 0->1, 1->2, 2->3 are all constant)

        Under the hood, this runs optim_points after initializing with triangulation.
        See optim_points for all keyword parameters
        """

        assert points.shape[0] == len(self.cameras), \
            "Invalid points shape, first dim should be equal to" \
            " number of cameras ({}), but shape is {}".format(
                len(self.cameras), points.shape
            )

        n_cams, n_frames, n_joints, _ = points.shape
        # constraints = np.array(constraints)
        # constraints_weak = np.array(constraints_weak)

        points = self._tensor(points)
        points_shaped = points.reshape(n_cams, n_frames*n_joints, 2)
        if init_ransac:
            p3ds, picked, p2ds, errors = self._triangulate_possible(
                points_shaped.reshape(n_cams, n_frames*n_joints, 1, 2), progress=init_progress)
            points = p2ds.reshape(points.shape)
        else:
            p3ds = self._triangulate(points_shaped, progress=init_progress)
        p3ds = p3ds.reshape((n_frames, n_joints, 3))

        c = torch.isfinite(p3ds[:, :, 0])
        if torch.sum(c) < 20:
            print("warning: not enough 3D points to run optimization")
            return p3ds

        return self._optim_points(points, p3ds, **kwargs)


    # @torch.compile
    def _error_fun_triangulation(self, params, p2ds,
                                 constraints=[],
                                 constraints_weak=[],
                                 scores=None,
                                 scale_smooth=10000,
                                 scale_length=1,
                                 scale_length_weak=0.2,
                                 reproj_error_threshold=100,
                                 reproj_loss='soft_l1',
                                 n_deriv_smooth=1,
                                 p3ds_fixed=None, unit=None, center=None, proj_fn=None):
        """optim_points loss. ``proj_fn``: world points [M,3] -> pixels [C,M,2] (default
        ``self._project``). With ``unit``/``center`` (normalised mode) ``params['p3d']`` are
        pixel-equivalent coordinates ``q`` (world = center + unit * q); smoothness and bone
        terms are then on ``q``, and the bone term is a plain squared difference."""
        n_cams, n_frames, n_joints, _ = p2ds.shape

        n_3d = n_frames * n_joints * 3
        n_constraints = len(constraints)
        n_constraints_weak = len(constraints_weak)

        # load params
        p3ds = params['p3d']
        joint_lengths = params['joint_lengths']
        joint_lengths_weak = params['joint_lengths_weak']

        ## if fixed points, first n_fixed parameter points are ignored
        ## and replacement points are put in
        ## this way we can keep rest of code the same
        if p3ds_fixed is not None:
            n_fixed = p3ds_fixed.shape[0]
            p3ds = torch.vstack([p3ds_fixed, p3ds[n_fixed:]])

        # reprojection errors
        p3ds_flat = p3ds.reshape(-1, 3)
        if unit is not None:
            p3ds_flat = center + unit * p3ds_flat
        p2ds_flat = p2ds.reshape((n_cams, -1, 2))
        if proj_fn is None:
            errors = self._reprojection_error(p3ds_flat, p2ds_flat)
        else:
            errors = p2ds_flat - proj_fn(p3ds_flat).reshape(p2ds_flat.shape)
        if scores is not None:
            scores_flat = scores.reshape((n_cams, -1))
            errors = errors * scores_flat[:, :, None]
        errors_reproj = errors[torch.isfinite(p2ds_flat)]

        rp = reproj_error_threshold
        errors_reproj = torch.abs(errors_reproj)
        if reproj_loss == 'huber':
            bad = errors_reproj > rp
            errors_reproj[bad] = rp*(2*torch.sqrt(errors_reproj[bad]/rp) - 1)
        elif reproj_loss == 'linear':
            pass
        elif reproj_loss == 'soft_l1':
            errors_reproj = rp*2*(torch.sqrt(1+errors_reproj/rp)-1)

        # print(torch.mean(errors_reproj))
        sum_reproj = torch.sum(torch.square(errors_reproj)) 
        total_error = torch.tensor(0.0, dtype=sum_reproj.dtype, device=sum_reproj.device)
        total_error += sum_reproj

        # temporal constraint
        errors_smooth = torch.diff(p3ds, n=n_deriv_smooth, dim=0)
        sum_smooth = torch.sum(torch.square(errors_smooth)) * scale_smooth
        total_error += sum_smooth

        # joint length constraint
        error_lengths = 0
        for cix, (a, b) in enumerate(constraints):
            lengths = torch.linalg.norm(p3ds[:, a] - p3ds[:, b], dim=1)
            expected = torch.abs(joint_lengths[cix])
            if unit is None:
                error_lengths += torch.sum(torch.square(lengths - expected))/expected
            else:
                error_lengths += torch.sum(torch.square(lengths - expected))
        sum_lengths = error_lengths * scale_length * (100 if unit is None else 1)
        total_error += sum_lengths

        error_lengths_weak = 0
        for cix, (a, b) in enumerate(constraints_weak):
            lengths = torch.linalg.norm(p3ds[:, a] - p3ds[:, b], dim=1)
            expected = torch.abs(joint_lengths_weak[cix])
            if unit is None:
                error_lengths_weak += torch.sum(torch.square(lengths - expected))/expected
            else:
                error_lengths_weak += torch.sum(torch.square(lengths - expected))
        sum_lengths_weak = error_lengths_weak * scale_length_weak * (100 if unit is None else 1)
        total_error += sum_lengths_weak

        # print("reproj: {:.2f}   smooth: {:.2f}   len: {:.2f}  len_weak: {:.2f}".format(
        #       sum_reproj.item(), sum_smooth.item(), sum_lengths.item(), sum_lengths_weak.item()))

        return total_error

 


    def _initialize_params_triangulation(self, p3ds, constraints=[], constraints_weak=[], device=None,
                                         p3ds_lengths=None):
        """Parameters of optim_points. ``p3ds_lengths`` (normalised mode): the points (NaN =
        not triangulated) from which each bone's target length is estimated robustly, as the
        median over the frames where both joints are triangulated (no cross-bone clipping)."""
        if p3ds_lengths is not None:
            def _robust(pairs):
                out = np.empty(len(pairs), dtype='float64')
                for cix, (a, b) in enumerate(pairs):
                    L = np.linalg.norm(p3ds_lengths[:, a] - p3ds_lengths[:, b], axis=1)
                    L = L[np.isfinite(L)]
                    if L.size == 0:
                        L = np.linalg.norm(p3ds[:, a] - p3ds[:, b], axis=1)
                    out[cix] = np.median(L) if L.size else 1.0
                return out
            return {
                'p3d': nn.Parameter(to_tensor(p3ds, device=device, dtype=self.dtype)),
                # fixed targets: free lengths drift towards gap-filled frames (shrinking the
                # skeleton); the robust median is the estimate
                'joint_lengths': to_tensor(_robust(constraints), device=device, dtype=self.dtype),
                'joint_lengths_weak': to_tensor(_robust(constraints_weak), device=device,
                                                dtype=self.dtype)
            }
        joint_lengths = np.empty(len(constraints), dtype='float64')
        joint_lengths_weak = np.empty(len(constraints_weak), dtype='float64')

        for cix, (a, b) in enumerate(constraints):
            lengths = np.linalg.norm(p3ds[:, a] - p3ds[:, b], axis=1)
            joint_lengths[cix] = np.median(lengths)


        for cix, (a, b) in enumerate(constraints_weak):
            lengths = np.linalg.norm(p3ds[:, a] - p3ds[:, b], axis=1)
            joint_lengths_weak[cix] = np.median(lengths)

        all_lengths = np.hstack([joint_lengths, joint_lengths_weak])
        med = np.median(all_lengths)
        if med == 0:
            med = 1e-3

        mad = np.median(np.abs(all_lengths - med))

        joint_lengths[joint_lengths == 0] = med
        joint_lengths_weak[joint_lengths_weak == 0] = med
        joint_lengths[joint_lengths > med+mad*5] = med
        joint_lengths_weak[joint_lengths_weak > med+mad*5] = med

        params = {
            'p3d': nn.Parameter(to_tensor(p3ds, device=device, dtype=self.dtype)),
            'joint_lengths': nn.Parameter(to_tensor(joint_lengths, device=device, dtype=self.dtype)),
            'joint_lengths_weak': nn.Parameter(to_tensor(joint_lengths_weak, device=device, dtype=self.dtype))
        }

        return params

    def copy(self):
        cameras = [cam.copy() for cam in self.cameras]
        metadata = deepcopy(self.metadata)
        return CameraGroup(cameras, metadata, stations=self._stations_copy())

    def set_rotations(self, rvecs):
        for cam, rvec in zip(self.cameras, rvecs):
            cam.set_rotation(rvec)

    def set_translations(self, tvecs):
        for cam, tvec in zip(self.cameras, tvecs):
            cam.set_translation(tvec)

    def get_rotations(self):
        rvecs = []
        for cam in self.cameras:
            rvec = cam.get_rotation()
            rvecs.append(rvec)
        return torch.stack(rvecs)

    def get_translations(self):
        tvecs = []
        for cam in self.cameras:
            tvec = cam.get_translation()
            tvecs.append(tvec)
        return torch.stack(tvecs)

    def get_names(self):
        return [cam.get_name() for cam in self.cameras]

    def set_names(self, names):
        for cam, name in zip(self.cameras, names):
            cam.set_name(name)

    def set_n_dist(self, n_dist):
        """Set how many distortion coefficients bundle_adjust may move, on
        every camera. See Camera.set_n_dist."""
        for cam in self.cameras:
            cam.set_n_dist(n_dist)

    def get_n_dists(self):
        return [cam.get_n_dist() for cam in self.cameras]

    def set_free_intrinsics(self, names=DEFAULT_FREE_INTRINSICS):
        """Choose which camera-matrix entries bundle adjustment may move, on every camera.
        See Camera.set_free_intrinsics."""
        for cam in self.cameras:
            cam.set_free_intrinsics(names)

    def get_free_intrinsics(self):
        return [cam.get_free_intrinsics() for cam in self.cameras]

    def get_offsets(self):
        return torch.stack([cam.get_offset() for cam in self.cameras])

    def set_offsets(self, offsets):
        for cam, off in zip(self.cameras, offsets):
            cam.set_offset(off)

    def apply_crop_offsets(self, config, set_size=True):
        """Set per-camera crop offsets from an anipose-style config.

        `config` may be a full anipose config dict (with a 'cameras' table), the 'cameras'
        table itself ({name: {'offset': [x, y, w, h]}}), or a plain {name: [x, y(, w, h)]}.
        (x, y) becomes the camera offset (stored-image origin in sensor pixels). With four
        values and set_size=True the camera size becomes (w, h), the stored image size.
        Cameras missing from the config are left unchanged. Returns self."""
        table = config.get('cameras', config) if isinstance(config, dict) else config
        for cam in self.cameras:
            entry = table.get(cam.get_name())
            if entry is None:
                continue
            values = entry.get('offset') if isinstance(entry, dict) else entry
            if values is None:
                continue
            values = [float(v) for v in values]
            if len(values) not in (2, 4):
                raise ValueError("offset for camera {} must have 2 or 4 values, got {}".format(
                    cam.get_name(), values))
            cam.set_offset(values[:2])
            if len(values) == 4 and set_size:
                cam.set_size((int(values[2]), int(values[3])))
        return self

    def _triangulate_ng(self, *args, **kwargs):
        with torch.no_grad():
            return self._triangulate(*args, **kwargs)

    def _reprojection_error_ng(self, *args, **kwargs):
        with torch.no_grad():
            return self._reprojection_error(*args, **kwargs)

    @returns_like_input()
    def average_error(self, p2ds, median=False):
        """Mean (or median) over points of the per-point mean reprojection error after
        triangulating p2ds (C x N x 2)."""
        return self._average_error(p2ds, median=median)

    def _average_error(self, p2ds, median=False):
        p3ds = self._triangulate(p2ds)
        errors = self._reprojection_error(p3ds, p2ds, mean=True)
        if median:
            return torch.median(errors)
        else:
            return torch.mean(errors)

    def calibrate_rows(self, all_rows, board,
                       init_intrinsics=True, init_extrinsics=True, verbose=True,
                       **kwargs):
        """Calibrate from board detections. Extra keyword arguments go to bundle_adjust_iter
        (unknown ones raise TypeError before any work is done)."""
        _check_kwargs('calibrate_rows', kwargs, _BA_ITER_KWARGS)
        self._check_not_frozen('calibrate_rows')
        assert len(all_rows) == len(self.cameras), \
            "Number of camera detections does not match number of cameras"
        for rows, camera in zip(all_rows, self.cameras):
            size = camera.get_size()

            assert size is not None, \
                "Camera with name {} has no specified frame size".format(camera.get_name())

            if init_intrinsics:
                objp, imgp = board.get_all_calibration_points(rows)
                mixed = [(o, i) for (o, i) in zip(objp, imgp) if len(o) >= 9]
                objp, imgp = zip(*mixed)
                init_camera_intrinsics(camera, objp, imgp, size)

        for i, (row, cam) in enumerate(zip(all_rows, self.cameras)):
            all_rows[i] = board.estimate_pose_rows(cam, row)

        new_rows = [[r for r in rows if r['ids'].size >= 8] for rows in all_rows]
        merged = merge_rows(new_rows)
        imgp, extra = extract_points(merged, board, min_cameras=2)

        if init_extrinsics:
            rtvecs = extract_rtvecs(merged)
            # if verbose:
            #     pprint(get_connections(rtvecs, self.get_names()))
            rvecs, tvecs = get_initial_extrinsics(rtvecs, self.get_names())
            self.set_rotations(rvecs)
            self.set_translations(tvecs)

        error = self.bundle_adjust_iter(imgp, extra, verbose=verbose, **kwargs)

        return error

    def get_rows_videos(self, videos, board, verbose=True, *, frame_indices=None, n_workers=1,
                        detector=None, detector_kwargs=None, skip=20):
        """Detect the board in each camera's videos -> one list of rows per camera.

        videos: one list of video filenames per camera (group order).
        frame_indices: None (default: every ``skip``-th frame, densified after detections), or
          the frames to decode, passed to ``board.detect_video(frame_indices=)``: one sequence of
          ints for every video, a per-camera list of per-video sequences, or a dict keyed by
          video filename or camera name.
        n_workers: number of videos decoded/detected concurrently (threads; OpenCV releases
          the GIL). The result does not depend on it.
        detector / detector_kwargs: board detector to use for this call ('opencv',
          'deepcharuco' or a detector instance; CharucoBoard only), without changing ``board``.
        Failed videos are reported and give no rows, as before."""
        if detector is not None:
            board = _board_with_detector(board, detector, detector_kwargs)
        jobs = []
        for cix, (cam, cam_videos) in enumerate(zip(self.cameras, videos)):
            for vnum, vidname in enumerate(cam_videos):
                jobs.append((cix, vnum, vidname,
                             _frames_for(frame_indices, cix, vnum, vidname, cam.get_name(),
                                         len(self.cameras))))

        def run(job):
            cix, vnum, vidname, idx = job
            if verbose:
                print(vidname)
            kw = dict(prefix=vnum, progress=verbose and n_workers <= 1)
            if idx is not None:
                kw['frame_indices'] = idx
            elif skip != 20:
                kw['skip'] = skip
            try:
                rows = board.detect_video(vidname, **kw)
            except Exception as e:
                print("WARNING: board detection failed for video {}".format(vidname))
                print(e)
                rows = []
            if verbose:
                print("{} boards detected".format(len(rows)))
            return rows

        if n_workers is not None and n_workers > 1 and len(jobs) > 1:
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=int(n_workers)) as ex:
                results = list(ex.map(run, jobs))
        else:
            results = [run(j) for j in jobs]

        all_rows = [[] for _ in self.cameras]
        for (cix, _, _, _), rows in zip(jobs, results):
            all_rows[cix].extend(rows)
        return all_rows

    def set_camera_sizes_videos(self, videos):
        for cix, (cam, cam_videos) in enumerate(zip(self.cameras, videos)):
            rows_cam = []
            for vnum, vidname in enumerate(cam_videos):
                try:
                    params = get_video_params(vidname)
                    size = (params['width'], params['height'])
                    cam.set_size(size)
                except Exception as e:
                    print("WARNING: camera size detection failed for video {}".format(vidname))
                    print(e)


    def calibrate_videos(self, videos, board,
                         init_intrinsics=True, init_extrinsics=True, verbose=True,
                         *, frame_indices=None, n_workers=1, detector=None,
                         detector_kwargs=None, **kwargs):
        """Takes as input a list of list of video filenames, one list of each camera.
        Also takes a board which specifies what should be detected in the videos.
        frame_indices, n_workers, detector, detector_kwargs go to get_rows_videos.
        Extra keyword arguments go to bundle_adjust_iter (unknown ones raise TypeError
        before any video is read)."""
        _check_kwargs('calibrate_videos', kwargs, _BA_ITER_KWARGS)

        all_rows = self.get_rows_videos(videos, board, verbose=verbose,
                                        frame_indices=frame_indices, n_workers=n_workers,
                                        detector=detector, detector_kwargs=detector_kwargs)
        if init_extrinsics:
            self.set_camera_sizes_videos(videos)

        error = self.calibrate_rows(all_rows, board,
                                    init_intrinsics=init_intrinsics,
                                    init_extrinsics=init_extrinsics,
                                    verbose=verbose, **kwargs)
        return error, all_rows

    def get_extrinsics_params(self):
        for cam in self.cameras:
            for p in cam.get_extrinsics_params():
                yield p

    def get_dicts(self):
        out = []
        for cam in self.cameras:
            out.append(cam.get_dict())
        return out

    @staticmethod
    def from_dicts(arr, metadata=None, dtype=None):
        cameras = []
        for d in arr:
            if 'fisheye' in d and d['fisheye']:
                cam = FisheyeCamera.from_dict(d)
            else:
                cam = Camera.from_dict(d)
            cameras.append(cam)
        _warn_duplicate_names([c.get_name() for c in cameras])
        return CameraGroup(cameras, metadata=metadata, dtype=dtype)

    @staticmethod
    def from_names(names, fisheye=False, n_dist=2):
        cameras = []
        for name in names:
            if fisheye:
                cam = FisheyeCamera(name=name, n_dist=n_dist)
            else:
                cam = Camera(name=name, n_dist=n_dist)
            cameras.append(cam)
        return CameraGroup(cameras)

    def load_dicts(self, arr):
        for cam, d in zip(self.cameras, arr):
            cam.load_dict(d)

    def to_dict(self):
        """The calibration file as a dict: {'cam_0': {...}, ..., 'metadata': {...}}, cameras
        in group order. JSON/TOML friendly."""
        dicts = self.get_dicts()
        names = ['cam_{}'.format(i) for i in range(len(dicts))]
        master_dict = dict(zip(names, dicts))
        master_dict['metadata'] = _to_builtin(deepcopy(self.metadata))
        st = getattr(self, 'stations', None)
        if st is not None:
            # under [metadata.stations]: readers <= 0.9 treat every other top-level table as a camera
            master_dict['metadata']['stations'] = st.to_toml_dict()
        return master_dict

    @staticmethod
    def from_dict(master_dict, order='natural', dtype=None):
        """Inverse of to_dict. Camera sections are every key other than 'metadata', ordered
        by `order`: 'natural' (default; cam_2 before cam_10), 'lexical' (aniposelib <= 0.9)
        or 'file' (dict order)."""
        keys = [k for k in master_dict.keys() if k not in ('metadata', 'stations')]
        keys = _order_keys(keys, order=order)
        items = [master_dict[k] for k in keys]
        metadata = deepcopy(master_dict.get('metadata', {}))
        st = metadata.pop('stations', None) if isinstance(metadata, dict) else None
        if st is None:
            st = master_dict.get('stations')
        cg = CameraGroup.from_dicts(items, metadata=metadata, dtype=dtype)
        if st:
            from aniposelib.stations import Stations
            cg.stations = Stations.from_toml_dict(st, cameras=cg.get_names())
        return cg

    def dumps(self):
        """The calibration as a TOML string (same content as dump)."""
        return toml.dumps(self.to_dict())

    @staticmethod
    def loads(s, order='natural', dtype=None):
        """Load a calibration from a TOML string (see from_dict for `order`)."""
        return CameraGroup.from_dict(toml.loads(s), order=order, dtype=dtype)

    def dump(self, fname):
        with open(fname, 'w') as f:
            toml.dump(self.to_dict(), f)

    @staticmethod
    def load(fname, order='natural', dtype=None):
        """Load a calibration TOML. Cameras are ordered by key with a numeric sort
        (cam_2 before cam_10); order='lexical' restores the aniposelib <= 0.9 order and
        order='file' keeps the file's section order. A CameraOrderWarning is issued when
        the numeric and lexical orders differ, and on duplicate camera names."""
        master_dict = toml.load(fname)
        return CameraGroup.from_dict(master_dict, order=order, dtype=dtype)

    def resize_cameras(self, scale):
        for cam in self.cameras:
            cam.resize_camera(scale)

    @returns_like_input()
    def is_point_visible(self, p3d, margin=0):
        """Takes a Nx3 set of 3D points.
        Returns a boolean array of CxN where C is the number of cameras,
        representing if each camera can see each point.
        """
        p3d = self._tensor(p3d)
        all_visible = [cam.is_point_visible(p3d, margin=margin)
                       for cam in self.cameras]
        return torch.stack(all_visible)

    @returns_like_input()
    def get_triangulation_sensitivity(self, p):
        """
        Compute triangulation sensitivity for a set of 3D points.
        This is the condition number of the stacked projection sensitivities of the cameras
        that see the point. Lower is more reliable; NaN where fewer than 2 cameras see it.

        Parameters:
        -----------
        p : array_like, shape (N, 3)
            3D point(s) in world coordinates

        Returns:
        --------
        sensitivity : (N,) array
        """
        p = self._tensor(p).reshape(-1, 3)

        n_points = p.shape[0]
        n_cams = len(self.cameras)

        J_all = torch.zeros((n_points, 2*n_cams, 3), dtype=p.dtype, device=p.device)
        visibles = self.is_point_visible(p)

        for i, cam in enumerate(self.cameras):
            J_cam = cam.projection_sensitivity(p)
            J_cam = J_cam * visibles[i, :, None, None]
            J_all[:, 2*i:2*i+2, :] = J_cam

        # handle nans
        J_all[~torch.isfinite(J_all)] = 0

        # Compute conditioning
        s = torch.linalg.svdvals(J_all)
        sensitivity = s[:, 0] / (s[:, -1] + 1e-10)

        count = torch.sum(visibles, dim=0)
        sensitivity[count < 2] = torch.nan

        return sensitivity

    def get_point_cloud(self, sensitivity_threshold=4, rng=None):
        """Sample points that at least two cameras see with good triangulation sensitivity.
        Returns (points Mx3, sensitivity M) as torch tensors. rng: np.random.Generator, int
        seed or None (fresh entropy); the global torch RNG is not used."""
        rng = _as_rng(rng)
        with torch.no_grad():
            points = []
            for c1, c2 in itertools.combinations(self.cameras, 2):
                rays1 = c1.get_camera_rays()
                rays2 = c2.get_camera_rays()
                for r1, r2 in itertools.product(rays1, rays2):
                    p = closest_point_between_rays(r1, r2)
                    if p is not None:
                        points.append(p)
            points = torch.stack(points)

            low, high = torch.quantile(points,
                                       to_tensor([0.05, 0.95], device=points.device,
                                                 dtype=points.dtype),
                                       dim=0)
            scale = torch.max(high - low) * 0.15

            all_points = [points]
            for i in range(15):
                noise = torch.as_tensor(rng.standard_normal(points.shape),
                                        dtype=points.dtype, device=points.device)
                pp = points + noise * scale
                all_points.append(pp)
            all_points = torch.vstack(all_points)

            s = self.get_triangulation_sensitivity(all_points)
            s[torch.isnan(s)] = torch.inf

            good = s < sensitivity_threshold

            return all_points[good], s[good]

    # -- 1.0 convenience aliases (thin delegates to aniposelib.scale / quality / init)
    def rescale(self, s, units=None, source=None, quality=None):
        """World units *= ``s``, in place; returns self. See :func:`aniposelib.scale.rescale`."""
        from aniposelib import scale as _S
        return _S.rescale(self, s, units=units, source=source, quality=quality)

    def apply_similarity(self, s=1.0, R=None, t=None):
        """Re-express the rig in the world frame ``X' = s R X + t``, in place; returns self.
        See :func:`aniposelib.scale.apply_similarity`."""
        from aniposelib import scale as _S
        return _S.apply_similarity(self, s, R, t)

    def anchor_to(self, camera=0):
        """Make ``camera`` (index or name) the world frame, in place; returns self.
        See :func:`aniposelib.scale.anchor_to`."""
        from aniposelib import scale as _S
        return _S.anchor_to(self, camera)

    def align_to(self, target, **kwargs):
        """Trimmed similarity alignment of the camera centres onto ``target`` (a group, matched
        by name, or ``[C,3]`` centres), in place -> ``AlignResult``.
        See :func:`aniposelib.scale.align_to`."""
        from aniposelib import scale as _S
        return _S.align_to(self, target, **kwargs)

    def align_world(self, ground_points=None, up='z', origin='centroid'):
        """Rotate the world so the floor normal is ``up``, in place -> ``AlignResult``.
        See :func:`aniposelib.scale.align_world`."""
        from aniposelib import scale as _S
        return _S.align_world(self, ground_points=ground_points, up=up, origin=origin)

    def report(self, p2ds, **kwargs):
        """Calibration QA (per-camera reprojection + leave-one-camera-out) -> ``CalibrationReport``.
        See :func:`aniposelib.quality.report`."""
        from aniposelib import quality as _Q
        return _Q.report(self, p2ds, **kwargs)

    def loo_reprojection(self, p2ds, **kwargs):
        """Leave-one-camera-out reprojection -> ``LOOResult``.
        See :func:`aniposelib.quality.loo_reprojection`."""
        from aniposelib import quality as _Q
        return _Q.loo_reprojection(self, p2ds, **kwargs)

    @staticmethod
    def from_extrinsics(names, Ms, K=None, sizes=None, model='pinhole', **kwargs):
        """A group from world->camera matrices ``Ms [C,4,4]`` (+ ``K``, ``dist``, ``sizes``,
        ``model`` or a ``template`` group for the intrinsics).
        See :func:`aniposelib.init.from_extrinsics`."""
        from aniposelib.init._common import from_extrinsics as _fe
        return _fe(names, Ms, K=K, sizes=sizes, model=model, **kwargs)


# intrinsics anchoring of bundle_adjust_iter(method='lm') on points (spec=None): sigma of log
# focal length, principal point (px) and distortion coefficients; each prior residual weighs like
# one observation with that many pixels of error
_DEFAULT_ITER_PRIOR = dict(sigma_logf=0.02, sigma_c=10.0, sigma_k=0.02)

_BA_ITER_KWARGS = {'n_iters', 'start_mu', 'end_mu', 'max_nfev', 'ftol', 'n_samp_iter',
                   'n_samp_full', 'error_threshold', 'only_extrinsics', 'rng', 'method', 'spec',
                   'return_report', 'max_points', 'resect', 'trans_scale'}
