import cv2
import numpy as np
from copy import copy
from scipy.sparse import lil_matrix, dok_matrix
from scipy.linalg import inv
from scipy import optimize
from scipy import signal
from numba import jit
from collections import defaultdict, Counter, namedtuple
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

# @jit(nopython=True, parallel=True)
# @torch.compile
def triangulate_simple(points, camera_mats):
    num_cams = len(camera_mats)
    A = torch.zeros((num_cams * 2, 4), dtype=points.dtype, device=points.device)
    for i in range(num_cams):
        x, y = points[i]
        mat = camera_mats[i]
        A[(i * 2):(i * 2 + 1)] = x * mat[2] - mat[0]
        A[(i * 2 + 1):(i * 2 + 2)] = y * mat[2] - mat[1]
    u, s, vh = torch.linalg.svd(A, full_matrices=True)
    p3d = vh[-1]
    p3d = p3d[:3] / p3d[3]
    return p3d

def triangulate_simple_batch(points, camera_mats, weights):
    '''
    Inputs:
        points: [C, N, 2] 2d points to triangulate
        camera_mats: [C, 4, 4] camera extrinsics
        weights: [C, N] weight for each camera
    Outputs:
        p3d: [N, 3] triangulated 3d point
    '''
    C, N, _ = points.shape

    points = rearrange(points, 'c n r -> n c r')
    
    # Expand camera_mats to [N, C, 4, 4]
    cam_mats = repeat(camera_mats, 'c i j -> n c i j', n=N)
    
    # Extract x, y coordinates and reshape weights
    x = points[:, :, 0:1, None]  # [N, C, 1]
    y = points[:, :, 1:2, None]  # [N, C, 1]
    w = rearrange(weights, 'c n -> n c 1 1')  # [N, C, 1]
    
    # Build equations for each camera
    # x * mat[2] - mat[0] and y * mat[2] - mat[1]
    eq_x = w * (x * cam_mats[:, :, 2:3, :] - cam_mats[:, :, 0:1, :])  # [N, C, 1, 4]
    eq_y = w * (y * cam_mats[:, :, 2:3, :] - cam_mats[:, :, 1:2, :])  # [N, C, 1, 4]
    
    # Stack and reshape to [N, C*2, 4]
    A = rearrange([eq_x, eq_y], 'two n c 1 j -> n (c two) j')
    
    # SVD decomposition
    u, s, vh = torch.linalg.svd(A, full_matrices=False)  # vh: [N, 4, 4]; only vh used downstream
    
    # Take last row of vh for each point
    p3d_homogeneous = vh[:, -1, :]  # [N, 4]
    
    # Convert from homogeneous to 3D coordinates
    p3d = p3d_homogeneous[:, :3] / p3d_homogeneous[:, 3:4]  # [N, 3]
    
    return p3d


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

def resample_points_extra(imgp, extra, n_samp=25):
    n_cams, n_points, _ = imgp.shape
    ids = remap_ids(extra['ids'])
    n_ids = np.max(ids)+1
    good = ~np.isnan(imgp[:, :, 0])
    ixs = np.arange(n_points)

    cam_counts = np.zeros((n_ids, n_cams), dtype='int64')
    for idnum in range(n_ids):
        cam_counts[idnum] = np.sum(good[:, ids == idnum], axis=1)
    cam_counts_random = cam_counts + np.random.random(size=cam_counts.shape)
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

def resample_points(imgp, extra=None, n_samp=25):
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
        num_cams_valid = num_cams[valid_ixs]
        weights = num_cams_valid.astype(float) ** 3
        weights /= weights.sum()
        chosen = np.random.choice(len(valid_ixs), size=target, replace=False, p=weights)
        final_ixs = np.sort(valid_ixs[chosen])
    
    newp = imgp[:, final_ixs]
    new_extra = subset_extra(extra, final_ixs)
    return newp, new_extra

def medfilt_data(values, size=15):
    padsize = size+5
    vpad = np.pad(values, (padsize, padsize), mode='reflect')
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

def remap_ids(ids):
    unique_ids = np.unique(ids)
    ids_out = np.copy(ids)
    for i, num in enumerate(unique_ids):
        ids_out[ids == num] = i
    return ids_out

def transform_points(points, rvecs, tvecs):
    """Rotate points by given rotation vectors and translate.
    Rodrigues' rotation formula is used.
    """
    theta = torch.linalg.norm(rvecs, dim=1, keepdim=True)
    
    # Handle division by zero
    v = rvecs / torch.where(theta > 1e-10, theta, torch.ones_like(theta))
    
    dot = torch.sum(points * v, dim=1, keepdim=True)
    cos_theta = torch.cos(theta)
    sin_theta = torch.sin(theta)
    
    rotated = cos_theta * points + \
              sin_theta * torch.cross(v, points, dim=1) + \
              dot * (1 - cos_theta) * v
    
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
    """
    Convert rotation vector to rotation matrix using Rodrigues formula.
    
    Args:
        rvec: rotation vector of shape (3,) or (3, 1)
    
    Returns:
        R: rotation matrix of shape (3, 3)
    """
    if rvec.dim() == 2:
        rvec = rvec.squeeze()
    
    theta = torch.norm(rvec)
    
    if theta < 1e-10:
        # Small angle approximation
        return torch.eye(3, dtype=rvec.dtype, device=rvec.device)
    
    # Normalized axis
    k = rvec / theta
    
    # Skew-symmetric matrix (use stack/cat to preserve autograd)
    zero = torch.zeros(1, dtype=rvec.dtype, device=rvec.device)
    K = torch.stack([
        torch.cat([zero, -k[2:3], k[1:2]]),
        torch.cat([k[2:3], zero, -k[0:1]]),
        torch.cat([-k[1:2], k[0:1], zero])
    ])
    
    # Rodrigues formula: R = I + sin(theta)*K + (1-cos(theta))*K^2
    I = torch.eye(3, dtype=rvec.dtype, device=rvec.device)
    R = I + torch.sin(theta) * K + (1 - torch.cos(theta)) * torch.mm(K, K)
    
    return R

def project_points(p3d, ext, matrix, dist):
    p2d_proj_raw = torch.matmul(to_homogeneous(p3d), ext.T)
    p2d_proj_raw = from_homogeneous(p2d_proj_raw[..., :3])

    k1, k2, p1, p2, k3 = dist[:5]
    k4 = k5 = k6 = 0
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

    # Index the intrinsics rather than multiplying by the full 3x3. Going
    # through the matrix makes matrix[2,0] and matrix[2,1] real projective
    # degrees of freedom, and bundle_adjust hands Adam the whole tensor, so
    # they drift and the result stops being a pinhole camera at all.
    # cv2.projectPoints ignores the skew too, so nothing is lost.
    fx, fy = matrix[0, 0], matrix[1, 1]
    cx, cy = matrix[0, 2], matrix[1, 2]
    p2d = torch.stack([fx * p2d_dist[..., 0] + cx,
                       fy * p2d_dist[..., 1] + cy], dim=-1)

    #TODO: handle offset

    return p2d

FISHEYE_EPS = 1e-12
_HALF_PI = float(np.pi / 2)


def _fisheye_dist_terms(dist):
    """k1..k4 from a distortion vector; pads short vectors, ignores extra terms."""
    if dist.shape[0] < 4:
        pad = torch.zeros(4 - dist.shape[0], dtype=dist.dtype, device=dist.device)
        dist = torch.cat([dist, pad])
    return dist[0], dist[1], dist[2], dist[3]


def project_points_fisheye(p3d, ext, matrix, dist, eps=FISHEYE_EPS):
    """Differentiable OpenCV fisheye (equidistant) projection.

    Only matrix[0,0], [1,1], [0,2], [1,2] are read. cv2.fisheye ignores the skew
    too (its alpha is a separate argument), and indexing keeps the unused entries
    out of the autograd graph, so bundle_adjust cannot drift them away from the
    0, 0, 0, 1 that makes a camera matrix a camera matrix.

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

    fx, fy = matrix[0, 0], matrix[1, 1]
    cx, cy = matrix[0, 2], matrix[1, 2]
    p2d = torch.stack([fx * (scale * a) + cx,
                       fy * (scale * b) + cy], dim=-1)

    return torch.where(bad[..., None], torch.full_like(p2d, float('nan')), p2d)


def make_M_torch(rvec, tvec):
    R = rodrigues(rvec)
    M = torch.eye(4, device=rvec.device, dtype=rvec.dtype)
    M[:3, :3] = R
    M[:3, 3] = tvec.reshape(3)
    return M


class Camera(nn.Module):
    # lets boards.py dispatch on the projection model without importing this module
    fisheye = False

    def __init__(self,
                 matrix=np.eye(3),
                 dist=np.zeros(5),
                 size=None,
                 rvec=np.zeros(3),
                 tvec=np.zeros(3),
                 name=None,
                 *,
                 n_dist=2):
        super().__init__()

        self.register_parameter('matrix', nn.Parameter(to_tensor(matrix)))
        self.register_parameter('dist', nn.Parameter(to_tensor(dist).view(-1)))
        self.register_parameter('rvec', nn.Parameter(to_tensor(rvec).view(-1)))
        self.register_parameter('tvec', nn.Parameter(to_tensor(tvec).view(-1)))

        # self.set_camera_matrix(matrix)
        # self.set_distortions(dist)
        # self.set_rotation(rvec)
        # self.set_translation(tvec)

        self.set_size(size)
        self.set_name(name)

        # clamp rather than raise here, so a camera built with a short dist and
        # the default n_dist is not an error; set_n_dist validates strictly
        self.n_dist = min(int(n_dist), self.dist.shape[0])
        self._install_dist_mask()

    def _install_dist_mask(self):
        self.dist.register_hook(self._mask_dist_grad)

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

    def __setstate__(self, state):
        # Parameter.__deepcopy__ and __reduce_ex__ do not carry backward hooks,
        # so a deepcopied or unpickled camera would keep n_dist while silently
        # losing the freezing it stands for. Put the hook back.
        super().__setstate__(state)
        self._install_dist_mask()

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
        a full bundle adjust. Use only_extrinsics=True for that.

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

    def get_dict(self):
        return {
            'name': self.get_name(),
            'size': list(self.get_size()),
            'matrix': self.matrix.detach().cpu().numpy().tolist(),
            'distortions': self.dist.detach().cpu().numpy().tolist(),
            'rotation': self.rvec.detach().cpu().numpy().tolist(),
            'translation': self.tvec.detach().cpu().numpy().tolist(),
            'n_dist': int(self.n_dist),
        }

    def load_dict(self, d):
        self.set_camera_matrix(d['matrix'])
        self.set_rotation(d['rotation'])
        self.set_translation(d['translation'])
        self.set_distortions(d['distortions'])
        self.set_name(d['name'])
        self.set_size(d['size'])
        # after set_distortions, so its clamp does not eat the loaded value.
        # A calibration written before n_dist existed gets the default.
        if 'n_dist' in d:
            self.set_n_dist(min(int(d['n_dist']), self.dist.shape[0]))

    def from_dict(d):
        cam = Camera()
        cam.load_dict(d)
        return cam

    def get_camera_matrix(self):
        return self.matrix

    def get_distortions(self):
        return self.dist

    def set_camera_matrix(self, matrix):
        # self.matrix = np.array(matrix, dtype='float64')
        self.matrix.data = to_tensor(matrix, device=self.matrix.device)

    def set_focal_length(self, fx, fy=None):
        if fy is None:
            fy = fx
        # self.matrix[0, 0] = fx
        # self.matrix[1, 1] = fy
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
        # self.dist = np.array(dist, dtype='float64').ravel()
        self.dist.data = to_tensor(dist, device=self.dist.device).view(-1)
        # a shorter vector has to pull n_dist down with it; a longer one does
        # not push it up, since freeing coefficients should be deliberate
        self.n_dist = min(self.n_dist, self.dist.shape[0])

    def zero_distortions(self):
        # self.dist = self.dist * 0
        self.dist.data.zero_()

    def set_rotation(self, rvec):
        # self.rvec = np.array(rvec, dtype='float64').ravel()
        self.rvec.data = to_tensor(rvec, device=self.rvec.device).view(-1)

    def get_rotation(self):
        return self.rvec

    def set_translation(self, tvec):
        # self.tvec = np.array(tvec, dtype='float64').ravel()
        self.tvec.data = to_tensor(tvec, device=self.tvec.device).view(-1)

    def get_translation(self):
        return self.tvec

    def get_extrinsics_mat(self):
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

    def resize_camera(self, scale):
        """resize the camera by scale factor, updating intrinsics to match"""
        size = self.get_size()
        new_size = size[0] * scale, size[1] * scale
        self.set_size(new_size)

        self.matrix.data *= scale
        self.matrix.data[2, 2] = 1
        
        # matrix = self.get_camera_matrix()
        # new_matrix = matrix * scale
        # new_matrix[2, 2] = 1
        # self.set_camera_matrix(new_matrix)

    def distort_points(self, points):
        # shape = points.shape
        # points = points.reshape(-1, 1, 2)
        # new_points = np.dstack([points, np.ones((points.shape[0], 1, 1))])
        # out, _ = cv2.projectPoints(new_points, np.zeros(3), np.zeros(3),
        #                            self.matrix.astype('float64'),
        #                            self.dist.astype('float64'))
        # return out.reshape(shape)
        points = to_homogeneous(to_tensor(points).reshape(-1, 2))
        eye = torch.eye(4, dtype=torch.float64, device=self.rvec.device)
        out = project_points(points, eye, self.matrix, self.dist)
        return out

    # def undistort_points(self, points):
    #     shape = points.shape
    #     points = points.reshape(-1, 1, 2)
    #     out = cv2.undistortPoints(points,
    #                               self.matrix.astype('float64'),
    #                               self.dist.astype('float64'))
    #     return out.reshape(shape)

    def undistort_points(self, points):
        t_points = to_tensor(points, device=self.matrix.device)
        shape = t_points.shape
        t_points = t_points.reshape(-1, 2)
        fx, fy = self.matrix[0, 0], self.matrix[1, 1]
        cx, cy = self.matrix[0, 2], self.matrix[1, 2]
        x = (t_points[:, 0] - cx) / fx
        y = (t_points[:, 1] - cy) / fy
        x0, y0 = x.clone(), y.clone()
        for _ in range(15):
            r2 = x*x + y*y
            r4 = r2*r2
            r6 = r4*r2
            k1, k2, p1, p2 = self.dist[0], self.dist[1], self.dist[2], self.dist[3]
            if self.dist.shape[0] > 4:
                k3 = self.dist[4]
            else:
                k3 = torch.tensor(0.0, device=self.dist.device, dtype=self.dist.dtype)
            radial = 1 + k1*r2 + k2*r4 + k3*r6
            dx = 2*p1*x*y + p2*(r2 + 2*x*x)
            dy = p1*(r2 + 2*y*y) + 2*p2*x*y
            x = (x0 - dx) / radial
            y = (y0 - dy) / radial
        return torch.stack([x, y], dim=1).reshape(shape)
    
    def project(self, points):
        # points = points.reshape(-1, 1, 3)
        # out, _ = cv2.projectPoints(points, self.rvec, self.tvec,
        #                            self.matrix.astype('float64'),
        #                            self.dist.astype('float64'))
        # return out.reshape(points.shape[0], 2)
        points = to_tensor(points).reshape(-1, 3)
        out = project_points(points, self.get_extrinsics_mat(),
                             self.matrix, self.dist)
        return out

    def reprojection_error(self, p3d, p2d):
        proj = self.project(p3d).reshape(p2d.shape)
        return p2d - proj

    def copy(self):
        # type(self) so subclasses copy as themselves, and an explicit dist of
        # the right length because load_state_dict is strict about shapes -- the
        # default would break any camera whose dist is not 5 long.
        new_cam = type(self)(name=self.name, size=self.size,
                             dist=np.zeros(self.dist.shape[0]),
                             n_dist=self.n_dist)
        new_cam.load_state_dict(self.state_dict())
        # load_state_dict copies values in place and does not change device, so a
        # copy of a cuda camera would otherwise come back on CPU. Preserve device.
        return new_cam.to(self.matrix.device)

    def is_point_visible(self, p3d, margin=0):
        """
        Takes as input a set of 3D points: (N, 3).
        Check if 3D points project into camera view.
        margin: pixels from border (e.g., 10 to avoid edge effects)
        """
        p3d = to_tensor(p3d)
        p2d = self.project(p3d)
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

        # ext = self.get_extrinsics_mat()
        # p_cam = torch.matmul(to_homogeneous(p3d), ext.T)
        # p_cam = from_homogeneous(p_cam[..., :3])

        in_front = (p_cam[:, 2] > 0)

        return in_bounds & in_front

    def projection_sensitivity(self, p):
        """
        Compute the sensitivity (Jacobian) of camera projection to 3D point movement.

        Parameters:
        -----------
        p : array_like, shape (3,) or (N, 3)
            3D point(s) in world coordinates [x, y, z]

        Returns:
        --------
        J : ndarray, shape (2, 3) or (N, 2, 3)
            Jacobian matrix where J[i,j] = ∂pixel_i/∂world_j
            Each column gives pixel sensitivity to movement in x, y, z directions
            If multiple points: J[k] is the Jacobian for point k
        """
        p = to_tensor(p)

        n_points = p.shape[0]

        # Get camera parameters
        R = rodrigues(self.rvec)  # Convert rotation vector to matrix
        t = self.tvec
        fx, fy = self.get_focal_length(both=True)

        # Transform to camera coordinates
        p_cam = (R @ p.T).T + t

        X = p_cam[:, 0]
        Y = p_cam[:, 1]
        Z = p_cam[:, 2]
        
        # Jacobian of projection w.r.t camera coordinates
        # J_proj[i] has shape (2, 3) for point i
        J_proj = torch.zeros((n_points, 2, 3), dtype=torch.float64)
        J_proj[:, 0, 0] = fx / Z
        J_proj[:, 0, 2] = -fx * X / (Z**2)
        J_proj[:, 1, 1] = fy / Z
        J_proj[:, 1, 2] = -fy * Y / (Z**2)
        
        # Chain rule: J = J_proj @ R
        # Using einsum for batch matrix multiplication
        J = torch.einsum('nij,jk->nik', J_proj, R)

        return J

    def get_center_world(self):
        R = rodrigues(self.rvec)
        t = self.tvec
        return (-R.T @ t) 

    def get_camera_rays(self):
        """
        Shoot rays from the 4 corners and center of a camera into 3D space.

        Parameters:
        -----------
        camera : Camera
            The camera to shoot rays from

        Returns:
        --------
        rays : dict
            Dictionary with keys 'origins' (5x3) and 'directions' (5x3)
            Order: [top_left, top_right, bottom_right, bottom_left, center]
        """
        if self.get_size() is None:
            raise ValueError("Camera must have size set")

        w, h = self.get_size()

        # Define corner and center points in image coordinates
        # Order: top_left, top_right, bottom_right, bottom_left, center
        img_points = np.array([
            [0, 0],           # top left
            [w-1, 0],         # top right
            [w-1, h-1],       # bottom right
            [0, h-1],         # bottom left
            [w/2, h/2]        # center
        ], dtype='float64')

        # Undistort to get normalized camera coordinates
        norm_points = self.undistort_points(img_points)

        # Convert to 3D rays in camera frame
        # Normalized points are already x/z, y/z, so we append z=1
        rays_camera = to_homogeneous(norm_points)

        # Normalize the direction vectors
        rays_camera = rays_camera / torch.linalg.norm(rays_camera, dim=1, keepdim=True)

        # Transform rays to world coordinates
        R = rodrigues(self.rvec)
        t = self.tvec

        # Ray origins are all at camera center in world coordinates
        center = self.get_center_world()

        # Ray directions in world frame
        directions = (R.T @ rays_camera.T).T

        rays = [Ray(origin=center, direction=d)
                for d in directions]

        return rays


class FisheyeCamera(Camera):
    """OpenCV fisheye (equidistant) camera, differentiable end to end.

    Distortions are k1..k4 of the equidistant model, so the vector is 4 long
    rather than the 5 of the Brown-Conrady pinhole model.
    """

    fisheye = True

    def __init__(self,
                 matrix=np.eye(3),
                 dist=np.zeros(4),
                 size=None,
                 rvec=np.zeros(3),
                 tvec=np.zeros(3),
                 name=None,
                 *,
                 n_dist=2):
        super().__init__(matrix=matrix, dist=dist, size=size,
                         rvec=rvec, tvec=tvec, name=name,
                         n_dist=n_dist)

    def from_dict(d):
        cam = FisheyeCamera()
        cam.load_dict(d)
        return cam

    def get_dict(self):
        d = super().get_dict()
        d['fisheye'] = True
        return d

    def project(self, points):
        points = to_tensor(points, device=self.matrix.device).reshape(-1, 3)
        return project_points_fisheye(points, self.get_extrinsics_mat(),
                                      self.matrix, self.dist)

    def distort_points(self, points):
        points = to_homogeneous(
            to_tensor(points, device=self.matrix.device).reshape(-1, 2))
        eye = torch.eye(4, dtype=torch.float64, device=self.matrix.device)
        return project_points_fisheye(points, eye, self.matrix, self.dist)

    def _solve_theta(self, points, n_iter=20, eps=FISHEYE_EPS, tol=1e-8):
        """Invert the equidistant model for a set of pixel coordinates.

        Returns (theta, x, y, theta_d, r_ok, ok), all flattened to (K,):
          theta    incidence angle
          x, y     normalized coordinates, sanitized to 0 where the input was not finite
          theta_d  radius of (x, y), sanitized to 1 where r_ok is False
          r_ok     radius above eps, i.e. not the principal point
          ok       Newton converged and the point is inside the invertible domain
        """
        t_points = to_tensor(points, device=self.matrix.device).reshape(-1, 2)
        fx, fy = self.matrix[0, 0], self.matrix[1, 1]
        cx, cy = self.matrix[0, 2], self.matrix[1, 2]
        k1, k2, k3, k4 = _fisheye_dist_terms(self.dist)

        x = (t_points[:, 0] - cx) / fx
        y = (t_points[:, 1] - cy) / fy

        finite = torch.isfinite(x) & torch.isfinite(y)
        x = torch.where(finite, x, torch.zeros_like(x))
        y = torch.where(finite, y, torch.zeros_like(y))

        td2 = x * x + y * y
        r_ok = td2 >= eps * eps
        theta_d = torch.sqrt(torch.where(r_ok, td2, torch.ones_like(td2)))
        theta_d_c = torch.clamp(theta_d, max=_HALF_PI)  # as cv2 does

        theta = theta_d_c.clone()
        for _ in range(n_iter):
            th2 = theta * theta
            th4 = th2 * th2
            th6 = th4 * th2
            th8 = th4 * th4
            f = theta * (1 + k1 * th2 + k2 * th4 + k3 * th6 + k4 * th8) - theta_d_c
            fp = 1 + 3 * k1 * th2 + 5 * k2 * th4 + 7 * k3 * th6 + 9 * k4 * th8
            fp = torch.where(torch.abs(fp) > eps, fp, torch.ones_like(fp))
            theta = theta - f / fp

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

    def undistort_points(self, points):
        shape = to_tensor(points, device=self.matrix.device).shape
        theta, x, y, theta_d, r_ok, ok = self._solve_theta(points)

        theta_c = torch.clamp(theta, max=_HALF_PI - 1e-9)  # keep tan() finite
        # scale tends to 1 as the radius tends to 0
        scale = torch.where(r_ok, torch.tan(theta_c) / theta_d,
                            torch.ones_like(theta))
        out = torch.stack([scale * x, scale * y], dim=-1)

        good = ok & (theta < _HALF_PI)
        out = torch.where(good[:, None], out, torch.full_like(out, float('nan')))
        return out.reshape(shape)

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

    def projection_sensitivity(self, p):
        """Same contract as Camera.projection_sensitivity, by autograd.

        The base version is a pinhole Jacobian that ignores distortion
        altogether, which for an equidistant lens is wrong by d(theta_d)/dr.
        """
        p = to_tensor(p, device=self.matrix.device).reshape(-1, 3)
        ext = self.get_extrinsics_mat().detach()
        matrix = self.matrix.detach()
        dist = self.dist.detach()

        with torch.enable_grad():
            p = p.detach().requires_grad_(True)
            out = project_points_fisheye(p, ext, matrix, dist)
            # each output row depends only on its own input row, so one backward
            # pass per output column gives all the per-row gradients
            rows = [torch.autograd.grad(out[:, i].sum(), p, retain_graph=(i == 0))[0]
                    for i in range(2)]

        return torch.stack(rows, dim=1)


# cv2.fisheye has no CALIB_FIX_K0, the flags start at k1
_FISHEYE_FIX_K = [cv2.fisheye.CALIB_FIX_K1, cv2.fisheye.CALIB_FIX_K2,
                  cv2.fisheye.CALIB_FIX_K3, cv2.fisheye.CALIB_FIX_K4]


def init_camera_intrinsics(camera, objp, imgp, size):
    """Seed a camera's intrinsics from board detections, per projection model.

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


class CameraGroup(nn.Module):
    def __init__(self, cameras, metadata={}):
        super().__init__()
        self.cameras = nn.ModuleList(cameras)
        self.metadata = metadata

    def subset_cameras(self, indices):
        cams = [self.cameras[ix].copy() for ix in indices]
        return CameraGroup(cams, self.metadata)

    def subset_cameras_names(self, names):
        cur_names = self.get_names()
        cur_names_dict = dict(zip(cur_names, range(len(cur_names))))
        indices = []
        for name in names:
            if name not in cur_names_dict:
                raise IndexError(
                    "name {} not part of camera names: {}".format(
                        name, cur_names
                    ))
            indices.append(cur_names_dict[name])
        return self.subset_cameras(indices)

    def project(self, points):
        """Given an Nx3 array of points, this returns an CxNx2 array of 2D points,
        where C is the number of cameras"""
        points = to_tensor(points)
        
        points = points.reshape(-1, 3)
        n_points = points.shape[0]
        n_cams = len(self.cameras)

        out = torch.empty((n_cams, n_points, 2), dtype=points.dtype, device=points.device)
        for cnum, cam in enumerate(self.cameras):
            out[cnum] = cam.project(points).reshape(n_points, 2)

        return out

    def _undistort_points_batch(self, points, batch_size=1000, progress=False, desc="undistort"):
        """Undistort per-camera. points: (C, ..., 2) torch tensor. Returns same shape.
        NaN passes through (cam.undistort_points is arithmetic-only)."""
        orig_shape = points.shape
        C = orig_shape[0]
        K = points[0].numel() // 2
        points_flat = points.reshape(C, K, 2)
        new_points = torch.empty_like(points_flat)
        undistort_iter = trange(0, K, batch_size, ncols=70, desc=desc) if progress \
            else range(0, K, batch_size)
        for start in undistort_iter:
            end = min(start + batch_size, K)
            for cnum, cam in enumerate(self.cameras):
                new_points[cnum, start:end] = cam.undistort_points(points_flat[cnum, start:end])
        return new_points.reshape(orig_shape)

    def triangulate(self, points, weights=None, batch_size=1000, undistort=True, progress=False):
        """Given an CxNx2 array, this returns an Nx3 array of points,
        where N is the number of points and C is the number of cameras"""

        assert points.shape[0] == len(self.cameras), \
            "Invalid points shape, first dim should be equal to" \
            " number of cameras ({}), but shape is {}".format(
                len(self.cameras), points.shape
            )

        device = next(self.parameters()).device
        points = to_tensor(points, device=device)

        one_point = False
        if len(points.shape) == 2:
            points = points.reshape(-1, 1, 2)
            one_point = True

        n_cams, n_points, _ = points.shape

        if undistort:
            points = self._undistort_points_batch(points, batch_size=batch_size, progress=progress)

        out = torch.full((n_points, 3), torch.nan,
                         device=points.device, dtype=torch.float64)

        cam_mats = torch.stack([cam.get_extrinsics_mat() for cam in self.cameras])

        # Initialize weights if not provided
        if weights is None:
            weights = torch.ones((n_cams, n_points), dtype=torch.float64, device=points.device)
        else:
            weights = to_tensor(weights, device=points.device)

        # Mask NaN points and zero their weights
        valid_mask = ~torch.isnan(points).any(dim=-1)  # [C, N]
        points = torch.nan_to_num(points, nan=0.0)
        weights = weights * valid_mask.float()

        # Count how many cameras have valid data per point
        valid_counts = torch.sum(weights > 0, dim=0)  # [N]

        if progress:
            iterator = trange(0, n_points, batch_size, ncols=70, desc="triangulate")
        else:
            iterator = range(0, n_points, batch_size)

        for start in iterator:
            end = min(start + batch_size, n_points)
            batch_indices = torch.arange(start, end, device=points.device)

            # Only process points observed by at least 2 cameras
            valid_in_batch = valid_counts[batch_indices] >= 2
            valid_batch_indices = batch_indices[valid_in_batch]

            if valid_batch_indices.numel() > 0:
                batch_points = points[:, valid_batch_indices]
                batch_weights = weights[:, valid_batch_indices]
                out[valid_batch_indices] = triangulate_simple_batch(batch_points, cam_mats, batch_weights)

        if one_point:
            out = out[0]

        return out

    def triangulate_possible(self, points, undistort=True, min_cams=2, progress=False, batch_size=1000):
        """Given an CxNxPx2 array, returns (out, picked_vals, points_2d, errors) by
        triangulating all valid camera-subset/candidate combinations and picking the
        one with lowest mean reprojection error per point.

        C: number of cameras, N: number of points, P: candidate detections per camera.
        batch_size controls the outer loop: points are processed in chunks of batch_size,
        so peak GPU working set is O(batch_size) rather than O(N). For large P (e.g. P=2
        with C=9 cameras), use a smaller batch_size (e.g. 25) to stay memory-bounded.
        """
        device = next(self.parameters()).device
        points_raw_cpu = to_tensor(points, device='cpu')  # master stays on CPU

        assert points_raw_cpu.shape[0] == len(self.cameras), \
            "Invalid points shape, first dim should be equal to" \
            " number of cameras ({}), but shape is {}".format(
                len(self.cameras), points_raw_cpu.shape
            )

        n_cams, n_points, n_possible, _ = points_raw_cpu.shape

        # Output tensors — O(N), small for typical N
        out         = torch.full((n_points, 3),                    torch.nan, dtype=torch.float64, device=device)
        picked_vals = torch.zeros((n_cams, n_points, n_possible),             dtype=torch.bool,    device=device)
        errors_out  = torch.zeros(n_points,                                   dtype=torch.float64, device=device)
        points_2d   = torch.full((n_cams, n_points, 2),            torch.nan, dtype=torch.float64, device=device)

        cam_mats = torch.stack([cam.get_extrinsics_mat() for cam in self.cameras])

        # Observation mask kept on CPU — (C, N, P) bool, one allocation for all chunks
        obs_mask_np_full = ~np.isnan(points_raw_cpu[..., 0].numpy())

        chunk_iter = trange(0, n_points, batch_size, ncols=70, desc="triangulate_possible") \
            if progress else range(0, n_points, batch_size)

        for p0 in chunk_iter:
            p1 = min(p0 + batch_size, n_points)
            chunk_size = p1 - p0

            # Upload chunk to GPU
            points_raw_chunk = points_raw_cpu[:, p0:p1].to(device)  # (C, chunk, P, 2)

            if undistort:
                points_undist_chunk = self._undistort_points_batch(points_raw_chunk, batch_size=batch_size)
            else:
                points_undist_chunk = points_raw_chunk

            # Enumerate combos for this chunk — CPU numpy, no GPU involvement
            obs_mask_chunk = obs_mask_np_full[:, p0:p1]  # (C, chunk, P)

            flat_c_chunks: list = []
            flat_q_chunks: list = []
            flat_m_chunks: list = []
            flat_p_chunks: list = []  # chunk-local point indices [0, chunk_size)
            point_to_m_range: dict = {}  # chunk-local p_local -> (start_m, end_m)

            m = 0
            for p_local in range(chunk_size):
                obs_cams_p = np.flatnonzero(obs_mask_chunk[:, p_local].any(axis=-1))
                n_obs = obs_cams_p.size
                if n_obs < min_cams:
                    continue

                qs_per_cam = [np.flatnonzero(obs_mask_chunk[c, p_local]) for c in obs_cams_p]
                P_per_cam = np.fromiter((qs.size for qs in qs_per_cam), dtype=np.int64, count=n_obs)
                shape = tuple((P_per_cam + 1).tolist())

                idx_matrix = np.indices(shape).reshape(n_obs, -1)
                is_kept    = idx_matrix < P_per_cam[:, None]
                n_picked   = is_kept.sum(axis=0)
                keep_combo = (n_picked >= min_cams) & (n_picked < n_obs)

                n_kept = int(keep_combo.sum())
                if n_kept == 0:
                    continue

                idx_k     = idx_matrix[:, keep_combo]
                is_kept_k = is_kept[:, keep_combo]

                q_2d = np.zeros((n_obs, n_kept), dtype=np.int64)
                for i, qs in enumerate(qs_per_cam):
                    v = is_kept_k[i]
                    q_2d[i, v] = qs[idx_k[i, v]]

                cam_2d = np.broadcast_to(obs_cams_p[:, None], (n_obs, n_kept))
                m_2d   = np.broadcast_to(np.arange(m, m + n_kept, dtype=np.int64)[None, :], (n_obs, n_kept))

                mask = is_kept_k.ravel()
                flat_c_chunks.append(cam_2d.ravel()[mask].astype(np.int64, copy=False))
                flat_q_chunks.append(q_2d.ravel()[mask])
                flat_m_chunks.append(m_2d.ravel()[mask])
                flat_p_chunks.append(np.full(int(mask.sum()), p_local, dtype=np.int64))

                point_to_m_range[p_local] = (m, m + n_kept)
                m += n_kept

            M_chunk = m
            if M_chunk == 0:
                continue

            flat_c = torch.from_numpy(np.concatenate(flat_c_chunks)).to(device=device)
            flat_q = torch.from_numpy(np.concatenate(flat_q_chunks)).to(device=device)
            flat_m = torch.from_numpy(np.concatenate(flat_m_chunks)).to(device=device)
            flat_p = torch.from_numpy(np.concatenate(flat_p_chunks)).to(device=device)

            # Build (C, M_chunk, 2) and (C, M_chunk) buffers — bounded by chunk, not N
            batch_points_undist = torch.zeros((n_cams, M_chunk, 2), dtype=torch.float64, device=device)
            batch_points_raw    = torch.zeros((n_cams, M_chunk, 2), dtype=torch.float64, device=device)
            batch_weights       = torch.zeros((n_cams, M_chunk),    dtype=torch.float64, device=device)

            # flat_p is chunk-local: indexes into points_raw_chunk[:, 0:chunk_size, ...]
            batch_points_undist[flat_c, flat_m] = points_undist_chunk[flat_c, flat_p, flat_q].to(torch.float64)
            batch_points_raw   [flat_c, flat_m] = points_raw_chunk   [flat_c, flat_p, flat_q].to(torch.float64)
            batch_weights      [flat_c, flat_m] = 1.0

            # One SVD over all M_chunk records; full_matrices=False keeps u bounded
            p3ds_batched = triangulate_simple_batch(batch_points_undist, cam_mats, batch_weights)

            proj     = self.project(p3ds_batched)  # (C, M_chunk, 2)
            err_norm = torch.linalg.norm(proj.to(torch.float64) - batch_points_raw, dim=-1)
            err      = (err_norm * batch_weights).sum(dim=0) / batch_weights.sum(dim=0)

            # Per-point argmin — write into global output tensors using p0 offset
            winner_mask = torch.zeros(M_chunk, dtype=torch.bool, device=device)
            for p_local, (m_s, m_e) in point_to_m_range.items():
                j = m_s + int(err[m_s:m_e].argmin().item())
                p_global = p0 + p_local
                out[p_global]        = p3ds_batched[j]
                errors_out[p_global] = err[j]
                winner_mask[j]       = True

            # Scatter picked_vals / points_2d; flat_p is chunk-local for gather,
            # flat_p + p0 is global for output scatter
            sel          = winner_mask[flat_m]
            p_global_idx = flat_p[sel] + p0
            picked_vals[flat_c[sel], p_global_idx, flat_q[sel]] = True
            points_2d  [flat_c[sel], p_global_idx] = \
                points_raw_chunk[flat_c[sel], flat_p[sel], flat_q[sel]].to(torch.float64)

        return out, picked_vals, points_2d, errors_out

    def triangulate_ransac(self, points, undistort=True, min_cams=2, progress=False):
        """Given an CxNx2 array, this returns an Nx3 array of points,
        where N is the number of points and C is the number of cameras"""

        assert points.shape[0] == len(self.cameras), \
            "Invalid points shape, first dim should be equal to" \
            " number of cameras ({}), but shape is {}".format(
                len(self.cameras), points.shape
            )

        n_cams, n_points, _ = points.shape

        points_ransac = points.reshape(n_cams, n_points, 1, 2)

        return self.triangulate_possible(points_ransac,
                                         undistort=undistort,
                                         min_cams=min_cams,
                                         progress=progress)


    def reprojection_error(self, p3ds, p2ds, mean=False):
        """Given an Nx3 array of 3D points and an CxNx2 array of 2D points,
        where N is the number of points and C is the number of cameras,
        this returns an CxNx2 array of errors.
        Optionally mean=True, this averages the errors and returns array of length N of errors"""

        device = next(self.parameters()).device
        p3ds = to_tensor(p3ds, device=device)
        p2ds = to_tensor(p2ds, device=device)
        
        one_point = False
        if len(p3ds.shape) == 1 and len(p2ds.shape) == 2:
            p3ds = p3ds.reshape(1, 3)
            p2ds = p2ds.reshape(-1, 1, 2)
            one_point = True

        n_cams, n_points, _ = p2ds.shape
        assert p3ds.shape == (n_points, 3), \
            "shapes of 2D and 3D points are not consistent: " \
            "2D={}, 3D={}".format(p2ds.shape, p3ds.shape)

        errors = []
        for cnum, cam in enumerate(self.cameras):
            errors.append(cam.reprojection_error(p3ds, p2ds[cnum]))
        errors = torch.stack(errors)
            
        if mean:
            errors_norm = torch.linalg.norm(errors, dim=2)
            good = ~torch.isnan(errors_norm)
            errors_norm[~good] = 0
            denom = torch.sum(good, dim=0).to(torch.float64)
            # denom[denom < 1.5] = torch.nan # less than 2 cameras
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
                           verbose=False):
        """Given an CxNx2 array of 2D points,
        where N is the number of points and C is the number of cameras,
        this performs iterative bundle adjustsment to fine-tune the parameters of the cameras.
        That is, it performs bundle adjustment multiple times, adjusting the weights given to points
        to reduce the influence of outliers.
        This is inspired by the algorithm for Fast Global Registration by Zhou, Park, and Koltun
        """

        assert p2ds.shape[0] == len(self.cameras), \
            "Invalid points shape, first dim should be equal to" \
            " number of cameras ({}), but shape is {}".format(
                len(self.cameras), p2ds.shape
            )

        if torch.is_tensor(p2ds):
            p2ds = p2ds.detach().cpu().numpy()
        p2ds_full = p2ds
        extra_full = extra

        device = next(self.parameters()).device

        p2ds, extra = resample_points(p2ds_full, extra_full,
                                      n_samp=n_samp_full)
        p2ds_t = torch.as_tensor(p2ds, device=device)
        error = self.average_error(p2ds_t, median=True).item()

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
                                          n_samp=n_samp_full)
            p2ds_t = torch.as_tensor(p2ds, device=device)
            p3ds = self.triangulate(p2ds_t)
            errors_norm = self.reprojection_error(p3ds, p2ds_t, mean=True).detach().cpu().numpy()

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
                p2ds[:, good], extra_good, n_samp=n_samp_iter)
            p2ds_samp = torch.as_tensor(p2ds_samp, device=device)

            error = np.median(finite_errors) if len(finite_errors) > 0 else error

            if error < error_threshold:
                break

            if verbose:
                errors_full = self.reprojection_error(p3ds, p2ds_t, mean=False).detach().cpu().numpy()
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

            self.bundle_adjust(p2ds_samp, extra_samp,
                               loss='cauchy', ftol=ftol,
                               max_nfev=iter_nfev, only_extrinsics=iter_only_ext,
                               verbose=verbose, f_scale=f_scales[i])

            # Track best state
            cur_p2ds, _ = resample_points(p2ds_full, extra_full, n_samp=n_samp_full)
            cur_p2ds_t = torch.as_tensor(cur_p2ds, device=device)
            cur_error = self.average_error(cur_p2ds_t, median=True).item()
            if cur_error < best_error:
                best_error = cur_error
                best_cam_state = {k: v.clone() for k, v in self.state_dict().items()}

        # Restore best state before final passes
        self.load_state_dict(best_cam_state)

        # Final bundle adjustment with cleaned points
        for final_pass in range(3):
            p2ds, extra = resample_points(p2ds_full, extra_full,
                                          n_samp=n_samp_full)
            p2ds_t = torch.as_tensor(p2ds, device=device)
            p3ds = self.triangulate(p2ds_t)
            errors_norm = self.reprojection_error(p3ds, p2ds_t, mean=True).detach().cpu().numpy()

            if verbose:
                errors_full = self.reprojection_error(p3ds, p2ds_t, mean=False).detach().cpu().numpy()

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
                self.bundle_adjust(p2ds_t[:, good], extra_good,
                                   loss='cauchy',
                                   ftol=ftol * 0.1, max_nfev=max(700, max_nfev),
                                   only_extrinsics=only_extrinsics,
                                   verbose=verbose, f_scale=final_f_scale)

            # Track best after each final pass
            cur_p2ds, _ = resample_points(p2ds_full, extra_full, n_samp=n_samp_full)
            cur_p2ds_t = torch.as_tensor(cur_p2ds, device=device)
            cur_error = self.average_error(cur_p2ds_t, median=True).item()
            if cur_error < best_error:
                best_error = cur_error
                best_cam_state = {k: v.clone() for k, v in self.state_dict().items()}

        # Check if final adjustment improved
        p2ds_check, _ = resample_points(p2ds_full, extra_full, n_samp=n_samp_full)
        p2ds_check_t = torch.as_tensor(p2ds_check, device=device)
        final_error = self.average_error(p2ds_check_t, median=True).item()

        if final_error > best_error:
            self.load_state_dict(best_cam_state)
            error = best_error
            if verbose:
                print('restored best state with error: ', best_error)
        else:
            error = final_error

        if verbose:
            p2ds_t2 = torch.as_tensor(
                resample_points(p2ds_full, extra_full, n_samp=n_samp_full)[0],
                device=device)
            p3ds = self.triangulate(p2ds_t2)
            errors_full = self.reprojection_error(p3ds, p2ds_t2, mean=False).detach().cpu().numpy()
            error_dict = get_error_dict(errors_full)
            pprint(error_dict)
            print('final error: ', error)

        return error

    def bundle_adjust(self, p2ds, extra=None,
                      loss='huber',
                      ftol=1e-4,
                      max_nfev=500,
                      lr=8.0e-4,
                      only_extrinsics=False,
                      verbose=True,
                      f_scale=None):
        """Given an CxNx2 array of 2D points,
        where N is the number of points and C is the number of cameras,
        this performs bundle adjustment to fine-tune the parameters of the cameras"""
        
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

        params = self._initialize_params_bundle(p2ds, extra)
        
        if only_extrinsics:
            cam_params = list(self.get_extrinsics_params())
        else:
            cam_params = list(self.parameters())

        param_groups = [
            {'params': cam_params, 'lr': lr},
            {'params': [params['p3d']], 'lr': lr * 65.0},
        ]
        if 'rvecs' in params:
            param_groups.append({'params': [params['rvecs'], params['tvecs']], 'lr': lr * 30.0})

        all_params = cam_params + list(params.values())

        # weight_decay=0 is load-bearing for Camera.n_dist: a frozen distortion
        # coefficient gets a zero gradient, and Adam then leaves it exactly
        # alone, but any weight decay would shrink it every step regardless.
        optimizer = optim.Adam(param_groups, weight_decay=0, amsgrad=True)
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=[lr * 65.0, lr * 260.0] + ([lr * 130.0] if 'rvecs' in params else []),
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
            torch.nn.utils.clip_grad_norm_(all_params, max_norm=10.0)
            optimizer.step()
            scheduler.step()

        # Restore best parameters
        if best_state is not None:
            for k, v in params.items():
                v.data.copy_(best_state[k])
            self.load_state_dict(best_cam_state)

        if verbose:
            print("iter: {} \t loss: {:.4f} (best: {:.4f})".format(i, current_loss, best_loss))
            
        error = self.average_error(p2ds)
        return error

    def _error_fun_bundle(self, params, p2ds, extra, loss='linear', f_scale=None):
        """Error function for bundle adjustment"""
        p3d = params['p3d']
        valid = torch.isfinite(p2ds)
        proj = self.project(p3d)
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



    def _initialize_params_bundle(self, p2ds, extra):
        """Given an CxNx2 array of 2D points,
        where N is the number of points and C is the number of cameras,
        initializes the parameters for bundle adjustment"""

        n_cams, n_points, _ = p2ds.shape
        assert n_cams == len(self.cameras), \
            "number of cameras in CameraGroup does not " \
            "match number of cameras in 2D points given"

        p3ds = self.triangulate(p2ds)
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
                    cam_id = np.random.choice(cam_ids_possible)
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


    def optim_points(self, points, p3ds,
                     constraints=[],
                     constraints_weak=[],
                     scale_smooth=4,
                     scale_length=2, scale_length_weak=0.5,
                     reproj_error_threshold=15, reproj_loss='soft_l1',
                     n_deriv_smooth=1, scores=None, verbose=False,
                     n_fixed=0,
                     max_lr=0.1,
                     n_iters=800):
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

        """
        assert points.shape[0] == len(self.cameras), \
            "Invalid points shape, first dim should be equal to" \
            " number of cameras ({}), but shape is {}".format(
                len(self.cameras), points.shape
            )

        n_cams, n_frames, n_joints, _ = points.shape
        device = next(self.parameters()).device
        constraints = np.array(constraints)
        constraints_weak = np.array(constraints_weak)

        if torch.is_tensor(p3ds):
            p3ds = p3ds.detach().cpu().numpy()
        else:
            p3ds = np.array(p3ds)
        
        p3ds_intp = np.apply_along_axis(interpolate_data, 0, p3ds)

        p3ds_med = np.apply_along_axis(medfilt_data, 0, p3ds_intp, size=7)
        default_smooth = 1.0/np.mean(np.abs(np.diff(p3ds_med, axis=0)))
        scale_smooth_full = scale_smooth * default_smooth

        t1 = time.time()

        params = self._initialize_params_triangulation(
            p3ds_intp, constraints, constraints_weak, device=device)

        if n_fixed > 0:
            p3ds_fixed = to_tensor(p3ds_intp[:n_fixed], device=device)
        else:
            p3ds_fixed = None

        points_tensor = to_tensor(points, device=device)
        scores = to_tensor(scores, device=device)

        all_params = list(params.values())
        optimizer = optim.Adam(all_params, lr=1e-2, weight_decay=0, amsgrad=True)
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=max_lr, total_steps=n_iters,
            pct_start=0.2, anneal_strategy='cos', final_div_factor=5000.0)

        # Best-params tracking is done entirely on-GPU. Calling `loss.item()` every
        # iteration forces a CPU<->GPU sync that serializes the loop and leaves the GPU
        # idle between tiny kernels (nvtop shows ~0% util). Instead we keep best_loss as a
        # tensor and update best_state branchlessly with torch.where, so nothing is read
        # back to the host inside the loop and the iterations can pipeline.
        best_loss = torch.tensor(float('inf'), device=device, dtype=torch.float64)
        best_state = {k: v.detach().clone() for k, v in params.items()}

        for i in range(n_iters):
            optimizer.zero_grad()
            loss = self._error_fun_triangulation(params, points_tensor,
                                                 constraints=constraints,
                                                 constraints_weak=constraints_weak,
                                                 scores=scores,
                                                 scale_smooth=scale_smooth_full,
                                                 scale_length=scale_length,
                                                 scale_length_weak=scale_length_weak,
                                                 reproj_error_threshold=reproj_error_threshold,
                                                 reproj_loss=reproj_loss,
                                                 n_deriv_smooth=n_deriv_smooth,
                                                 p3ds_fixed=p3ds_fixed
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

        return params['p3d'].detach()

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

        points_shaped = points.reshape(n_cams, n_frames*n_joints, 2)
        if init_ransac:
            p3ds, picked, p2ds, errors = self.triangulate_ransac(points_shaped, progress=init_progress)
            points = p2ds.reshape(points.shape)
        else:
            p3ds = self.triangulate(points_shaped, progress=init_progress)
        p3ds = p3ds.reshape((n_frames, n_joints, 3))

        c = torch.isfinite(p3ds[:, :, 0])
        if torch.sum(c) < 20:
            print("warning: not enough 3D points to run optimization")
            return p3ds

        return self.optim_points(points, p3ds, **kwargs)



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
                                 p3ds_fixed=None):
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
        p2ds_flat = p2ds.reshape((n_cams, -1, 2))
        errors = self.reprojection_error(p3ds_flat, p2ds_flat)
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
            error_lengths += torch.sum(torch.square(lengths - expected))/expected
        sum_lengths = error_lengths * scale_length * 100 
        total_error += sum_lengths

        error_lengths_weak = 0
        for cix, (a, b) in enumerate(constraints_weak):
            lengths = torch.linalg.norm(p3ds[:, a] - p3ds[:, b], dim=1)
            expected = torch.abs(joint_lengths_weak[cix])
            error_lengths_weak += torch.sum(torch.square(lengths - expected))/expected 
        sum_lengths_weak = error_lengths_weak * scale_length_weak * 100
        total_error += sum_lengths_weak

        # print("reproj: {:.2f}   smooth: {:.2f}   len: {:.2f}  len_weak: {:.2f}".format(
        #       sum_reproj.item(), sum_smooth.item(), sum_lengths.item(), sum_lengths_weak.item()))

        return total_error

 


    def _initialize_params_triangulation(self, p3ds, constraints=[], constraints_weak=[], device=None):
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
            'p3d': nn.Parameter(to_tensor(p3ds, device=device)),
            'joint_lengths': nn.Parameter(to_tensor(joint_lengths, device=device)),
            'joint_lengths_weak': nn.Parameter(to_tensor(joint_lengths_weak, device=device))
        }

        return params

    def copy(self):
        cameras = [cam.copy() for cam in self.cameras]
        metadata = copy(self.metadata)
        return CameraGroup(cameras, metadata)

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

    def average_error(self, p2ds, median=False):
        p3ds = self.triangulate(p2ds)
        errors = self.reprojection_error(p3ds, p2ds, mean=True)
        if median:
            return torch.median(errors)
        else:
            return torch.mean(errors)

    def calibrate_rows(self, all_rows, board,
                       init_intrinsics=True, init_extrinsics=True, verbose=True,
                       **kwargs):
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

    def get_rows_videos(self, videos, board, verbose=True):
        all_rows = []

        for cix, (cam, cam_videos) in enumerate(zip(self.cameras, videos)):
            rows_cam = []
            for vnum, vidname in enumerate(cam_videos):
                if verbose: print(vidname)
                try:
                    rows = board.detect_video(vidname, prefix=vnum, progress=verbose)
                except Exception as e:
                    print("WARNING: board detection failed for video {}".format(vidname))
                    print(e)
                    rows = []
                if verbose: print("{} boards detected".format(len(rows)))
                rows_cam.extend(rows)
            all_rows.append(rows_cam)

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
                         **kwargs):
        """Takes as input a list of list of video filenames, one list of each camera.
        Also takes a board which specifies what should be detected in the videos"""

        all_rows = self.get_rows_videos(videos, board, verbose=verbose)
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

    def from_dicts(arr):
        cameras = []
        for d in arr:
            if 'fisheye' in d and d['fisheye']:
                cam = FisheyeCamera.from_dict(d)
            else:
                cam = Camera.from_dict(d)
            cameras.append(cam)
        return CameraGroup(cameras)

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

    def dump(self, fname):
        dicts = self.get_dicts()
        names = ['cam_{}'.format(i) for i in range(len(dicts))]
        master_dict = dict(zip(names, dicts))
        master_dict['metadata'] = self.metadata
        with open(fname, 'w') as f:
            toml.dump(master_dict, f)

    def load(fname):
        master_dict = toml.load(fname)
        keys = sorted(master_dict.keys())
        items = [master_dict[k] for k in keys if k != 'metadata']
        cgroup = CameraGroup.from_dicts(items)
        if 'metadata' in master_dict:
            cgroup.metadata = master_dict['metadata']
        return cgroup

    def resize_cameras(self, scale):
        for cam in self.cameras:
            cam.resize_camera(scale)

    def is_point_visible(self, p3d, margin=0):
        """Takes a Nx3 set of 3D points.
        Returns a boolean array of CxN where C is the number of cameras,
        representing if each camera can see each point.
        """
        all_visible = [cam.is_point_visible(p3d)
                       for cam in self.cameras]
        return torch.stack(all_visible)

    
    def get_triangulation_sensitivity(self, p):
        """
        Compute triangulation sensitivity for a set of 3D points.
        This is the minimum singular value of all the projection sensitivities from each camera.
        Generally, a higher value means that large movements of the 3D point
          would result in small movements in the projections, meaning that the triangulation
          is likely to be less reliable.

        Parameters:
        -----------
        p : array_like, shape (3,) or (N, 3)
            3D point(s) in world coordinates

        Returns:
        --------
        sensitivity : float or ndarray
            Lower number means more reliable triangulation 
        """
        # p = np.array(p, dtype='float64')
        p = to_tensor(p)
        
        n_points = p.shape[0]
        n_cams = len(self.cameras)

        J_all = torch.zeros((n_points, 2*n_cams, 3), dtype=torch.float64)
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
        # sensitivity = 1/(s[:, -1] + 1e-6)

        # sensitivity = np.min(np.max(np.abs(J_all), axis=1), axis=1)
        
        count = torch.sum(visibles, dim=0)
        sensitivity[count < 2] = torch.nan

        return sensitivity


    def get_point_cloud(self, sensitivity_threshold=4):
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
                                   to_tensor([0.05, 0.95], device=points.device),
                                   dim=0)
        scale = torch.max(high - low) * 0.15

        all_points = [points]
        for i in range(15):
            pp = points + torch.randn(points.shape) * scale 
            all_points.append(pp)
        all_points = torch.vstack(all_points)

        s = self.get_triangulation_sensitivity(all_points)
        s[torch.isnan(s)] = torch.inf

        good = s < sensitivity_threshold
        
        return all_points[good], s[good]

