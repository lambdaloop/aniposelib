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
    u, s, vh = torch.linalg.svd(A, full_matrices=True)  # vh: [N, 4, 4]
    
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
    # if extra is not None:
    #     return resample_points_extra(imgp, extra, n_samp)

    n_cams = imgp.shape[0]
    good = ~np.isnan(imgp[:, :, 0])
    ixs = np.arange(imgp.shape[1])

    num_cams = np.sum(~np.isnan(imgp[:, :, 0]), axis=0)

    include = set()

    for i in range(n_cams):
        for j in range(i+1, n_cams):
            subset = good[i] & good[j]
            n_good = np.sum(subset)
            if n_good > 0:
                ## pick points, prioritizing points seen by more cameras
                arr = np.copy(num_cams[subset]).astype('float64')
                arr += np.random.random(size=arr.shape)
                picked_ix = np.argsort(-arr)[:n_samp]
                picked = ixs[subset][picked_ix]
                include.update(picked)

    final_ixs = sorted(include)
    newp = imgp[:, final_ixs]
    extra = subset_extra(extra, final_ixs)
    return newp, extra

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
    
    # Skew-symmetric matrix
    K = torch.tensor([
        [0, -k[2], k[1]],
        [k[2], 0, -k[0]],
        [-k[1], k[0], 0]
    ], dtype=rvec.dtype, device=rvec.device)
    
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

    p2d_raw = torch.matmul(to_homogeneous(p2d_dist), matrix.T)
    p2d = from_homogeneous(p2d_raw)

    #TODO: handle offset

    return p2d

def make_M_torch(rvec, tvec):
    R = rodrigues(rvec)
    M = torch.eye(4, device=rvec.device, dtype=rvec.dtype)
    M[:3, :3] = R
    M[:3, 3] = tvec.reshape(3)
    return M


class Camera(nn.Module):
    def __init__(self,
                 matrix=np.eye(3),
                 dist=np.zeros(5),
                 size=None,
                 rvec=np.zeros(3),
                 tvec=np.zeros(3),
                 name=None,
                 extra_dist=False):
        super().__init__()

        self.register_parameter('matrix', nn.Parameter(to_tensor(matrix)))
        self.register_parameter('dist', nn.Parameter(to_tensor(dist)))
        self.register_parameter('rvec', nn.Parameter(to_tensor(rvec).view(-1)))
        self.register_parameter('tvec', nn.Parameter(to_tensor(tvec).view(-1)))
        
        # self.set_camera_matrix(matrix)
        # self.set_distortions(dist)
        # self.set_rotation(rvec)
        # self.set_translation(tvec)
        
        self.set_size(size)
        self.set_name(name)
        self.extra_dist = extra_dist

    def get_dict(self):
        return {
            'name': self.get_name(),
            'size': list(self.get_size()),
            'matrix': self.matrix.detach().cpu().numpy().tolist(),
            'distortions': self.dist.detach().cpu().numpy().tolist(),
            'rotation': self.rvec.detach().cpu().numpy().tolist(),
            'translation': self.tvec.detach().cpu().numpy().tolist(),
        }

    def load_dict(self, d):
        self.set_camera_matrix(d['matrix'])
        self.set_rotation(d['rotation'])
        self.set_translation(d['translation'])
        self.set_distortions(d['distortions'])
        self.set_name(d['name'])
        self.set_size(d['size'])

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

    # def get_params(self, only_extrinsics=False):
    #     if only_extrinsics:
    #         params = np.zeros(6, dtype='float64')
    #     else:
    #         params = np.zeros(8 + self.extra_dist, dtype='float64')
    #     params[0:3] = self.get_rotation()
    #     params[3:6] = self.get_translation()
    #     if only_extrinsics:
    #         return params
    #     params[6] = self.get_focal_length()
    #     dist = self.get_distortions()
    #     params[7] = dist[0]
    #     if self.extra_dist:
    #         params[8] = dist[1]
    #     return params

    # def set_params(self, params, only_extrinsics=False):
    #     self.set_rotation(params[0:3])
    #     self.set_translation(params[3:6])
    #     if only_extrinsics:
    #         return

    #     self.set_focal_length(params[6])

    #     dist = np.zeros(5, dtype='float64')
    #     dist[0] = params[7]
    #     if self.extra_dist:
    #         dist[1] = params[8]
    #     self.set_distortions(dist)

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
        for _ in range(5):
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
        new_cam = Camera(name=self.name, size=self.size, extra_dist=self.extra_dist)
        new_cam.load_state_dict(self.state_dict())
        return new_cam
    
    # def copy(self):
    #     return \
    #         Camera(matrix=self.get_camera_matrix().copy(),
    #                dist=self.get_distortions().copy(),
    #                size=self.get_size(),
    #                rvec=self.get_rotation().copy(),
    #                tvec=self.get_translation().copy(),
    #                name=self.get_name(),
    #                extra_dist=self.extra_dist)

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
    def __init__(self,
                 matrix=np.eye(3),
                 dist=np.zeros(4),
                 size=None,
                 rvec=np.zeros(3),
                 tvec=np.zeros(3),
                 name=None,
                 extra_dist=False):
        self.set_camera_matrix(matrix)
        self.set_distortions(dist)
        self.set_size(size)
        self.set_rotation(rvec)
        self.set_translation(tvec)
        self.set_name(name)
        self.extra_dist = extra_dist

    def from_dict(d):
        cam = FisheyeCamera()
        cam.load_dict(d)
        return cam

    def get_dict(self):
        d = super().get_dict()
        d['fisheye'] = True
        return d

    def distort_points(self, points):
        shape = points.shape
        points = points.reshape(-1, 1, 2)
        new_points = np.dstack([points, np.ones((points.shape[0], 1, 1))])
        out, _ = cv2.fisheye.projectPoints(new_points,
                                           np.zeros(3), np.zeros(3),
                                           self.matrix.astype('float64'),
                                           self.dist.astype('float64'))
        return out.reshape(shape)

    def undistort_points(self, points):
        shape = points.shape
        points = points.reshape(-1, 1, 2)
        out = cv2.fisheye.undistortPoints(points.astype('float64'),
                                          self.matrix.astype('float64'),
                                          self.dist.astype('float64'))
        return out.reshape(shape)

    def project(self, points):
        points = points.reshape(-1, 1, 3)
        out, _ = cv2.fisheye.projectPoints(points,
                                           self.rvec, self.tvec,
                                           self.matrix.astype('float64'),
                                           self.dist.astype('float64'))
        return out

    # def set_params(self, params, only_extrinsics):
    #     self.set_rotation(params[0:3])
    #     self.set_translation(params[3:6])

    #     if only_extrinsics:
    #         return
        
    #     self.set_focal_length(params[6])

    #     dist = np.zeros(4, dtype='float64')
    #     dist[0] = params[7]
    #     if self.extra_dist:
    #         dist[1] = params[8]
    #     # dist[2] = params[9]
    #     # dist[3] = params[10]
    #     self.set_distortions(dist)

    # def get_params(self, only_extrinsics=False):
    #     if only_extrinsics:
    #         params = np.zeros(6, dtype='float64')
    #     else:
    #         params = np.zeros(8+self.extra_dist, dtype='float64')
    #     params[0:3] = self.get_rotation()
    #     params[3:6] = self.get_translation()
    #     if only_extrinsics:
    #         return params
    #     params[6] = self.get_focal_length()
    #     dist = self.get_distortions()
    #     params[7] = dist[0]
    #     if self.extra_dist:
    #         params[8] = dist[1]
    #     # params[9] = dist[2]
    #     # params[10] = dist[3]
    #     return params

    def copy(self):
        return FisheyeCamera(
            matrix=self.get_camera_matrix().copy(),
            dist=self.get_distortions().copy(),
            size=self.get_size(),
            rvec=self.get_rotation().copy(),
            tvec=self.get_translation().copy(),
            name=self.get_name(),
            extra_dist=self.extra_dist)

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

    def triangulate(self, points, weights=None, batch_size=1000, undistort=True, progress=False):
        """Given an CxNx2 array, this returns an Nx3 array of points,
        where N is the number of points and C is the number of cameras"""

        assert points.shape[0] == len(self.cameras), \
            "Invalid points shape, first dim should be equal to" \
            " number of cameras ({}), but shape is {}".format(
                len(self.cameras), points.shape
            )

        points = to_tensor(points)

        one_point = False
        if len(points.shape) == 2:
            points = points.reshape(-1, 1, 2)
            one_point = True

        n_cams, n_points, _ = points.shape

        if undistort:
            new_points = torch.empty_like(points)
            if progress:
                undistort_iter = trange(0, n_points, batch_size, ncols=70, desc="undistort")
            else:
                undistort_iter = range(0, n_points, batch_size)
            
            for start in undistort_iter:
                end = min(start + batch_size, n_points)
                for cnum, cam in enumerate(self.cameras):
                    new_points[cnum, start:end] = cam.undistort_points(points[cnum, start:end])
            points = new_points

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

    def triangulate_possible(self, points, undistort=True,
                             min_cams=2, progress=False, threshold=0.5):
        """Given an CxNxPx2 array, this returns an Nx3 array of points
        by triangulating all possible points and picking the ones with
        best reprojection error
        where:
        C: number of cameras
        N: number of points
        P: number of possible options per point
        """
        points = to_tensor(points)
        device = points.device

        assert points.shape[0] == len(self.cameras), \
            "Invalid points shape, first dim should be equal to" \
            " number of cameras ({}), but shape is {}".format(
                len(self.cameras), points.shape
            )

        n_cams, n_points, n_possible, _ = points.shape

        cam_nums, point_nums, possible_nums = torch.where(
            ~torch.isnan(points[:, :, :, 0]))

        
        all_iters = defaultdict(dict)

        for cam_num, point_num, possible_num in zip(cam_nums.tolist(),
                                                    point_nums.tolist(),
                                                    possible_nums.tolist()):
            if cam_num not in all_iters[point_num]:
                all_iters[point_num][cam_num] = []
            all_iters[point_num][cam_num].append((cam_num, possible_num))

        for point_num in all_iters.keys():
            for cam_num in all_iters[point_num].keys():
                all_iters[point_num][cam_num].append(None)

                
        out = torch.full((n_points, 3), torch.nan, dtype=torch.float64, device=device)
        picked_vals = torch.zeros((n_cams, n_points, n_possible), dtype=torch.bool, device=device)
        errors = torch.zeros(n_points, dtype=torch.float64, device=device)
        points_2d = torch.full((n_cams, n_points, 2), torch.nan, dtype=torch.float64, device=device)

        if progress:
            iterator = trange(n_points, ncols=70)
        else:
            iterator = range(n_points)

        for point_ix in iterator:
            best_point = None
            best_error = 200

            n_cams_max = len(all_iters[point_ix])
            
            for picked in itertools.product(*all_iters[point_ix].values()):
                picked = [p for p in picked if p is not None]
                if len(picked) < min_cams or len(picked) == n_cams_max:
                    continue

                cnums = [p[0] for p in picked]
                xnums = [p[1] for p in picked]

                pts = points[cnums, point_ix, xnums]
                cc = self.subset_cameras(cnums)
                
                p3d = cc.triangulate(pts, undistort=undistort)
                err = cc.reprojection_error(p3d, pts, mean=True)

                if err < best_error:
                    best_point = {
                        'error': err,
                        'point': p3d[:3],
                        'points': pts,
                        'picked': picked,
                        'joint_ix': point_ix
                    }
                    best_error = err
                    if best_error < threshold:
                        break

            if best_point is not None:
                out[point_ix] = best_point['point']
                picked = best_point['picked']
                cnums = [p[0] for p in picked]
                xnums = [p[1] for p in picked]
                picked_vals[cnums, point_ix, xnums] = True
                errors[point_ix] = best_point['error']
                points_2d[cnums, point_ix] = best_point['points']

        return out, picked_vals, points_2d, errors

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

        p3ds = to_tensor(p3ds)
        p2ds = to_tensor(p2ds)
        
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
                           n_iters=6, start_mu=15, end_mu=1,
                           max_nfev=1000, ftol=1e-4,
                           n_samp_iter=200, n_samp_full=1000,
                           error_threshold=0.3, only_extrinsics=False,
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
        p2ds = torch.as_tensor(p2ds, device=device)
        error = self.average_error(p2ds, median=True).item()

        if verbose:
            print('error: ', error)

        mus = np.exp(np.linspace(np.log(start_mu), np.log(end_mu), num=n_iters))

        if verbose:
            print('n_samples: {}'.format(n_samp_iter))

        for i in range(n_iters):
            p2ds, extra = resample_points(p2ds_full, extra_full,
                                          n_samp=n_samp_full)
            p2ds = torch.as_tensor(p2ds, device=device)
            p3ds = self.triangulate(p2ds)
            errors_full = self.reprojection_error(p3ds, p2ds, mean=False).detach().cpu().numpy()
            errors_norm = self.reprojection_error(p3ds, p2ds, mean=True).detach().cpu().numpy()

            error_dict = get_error_dict(errors_full)
            max_error = 0
            min_error = 0
            for k, v in error_dict.items():
                num, percents = v
                max_error = max(percents[-1], max_error)
                min_error = max(percents[0], min_error)
            mu = max(min(max_error, mus[i]), min_error)

            good = errors_norm < mu
            extra_good = subset_extra(extra, good)
            p2ds_samp, extra_samp = resample_points(
                p2ds[:, good].detach().cpu().numpy(), extra_good, n_samp=n_samp_iter)
            p2ds_samp = torch.as_tensor(p2ds_samp, device=device)
            
            error = np.median(errors_norm)

            if error < error_threshold:
                break

            if verbose:
                pprint(error_dict)
                print('error: {:.2f}, mu: {:.1f}, ratio: {:.3f}'.format(error, mu, np.mean(good)))

            self.bundle_adjust(p2ds_samp, extra_samp,
                               loss='linear', ftol=ftol,
                               max_nfev=max_nfev, only_extrinsics=only_extrinsics,
                               verbose=verbose)


        p2ds, extra = resample_points(p2ds_full, extra_full,
                                      n_samp=n_samp_full)
        p2ds = torch.as_tensor(p2ds, device=device)
        p3ds = self.triangulate(p2ds)
        errors_full = self.reprojection_error(p3ds, p2ds, mean=False).detach().cpu().numpy()
        errors_norm = self.reprojection_error(p3ds, p2ds, mean=True).detach().cpu().numpy()
        error_dict = get_error_dict(errors_full)
        if verbose:
            pprint(error_dict)

        max_error = 0
        min_error = 0
        for k, v in error_dict.items():
            num, percents = v
            max_error = max(percents[-1], max_error)
            min_error = max(percents[0], min_error)
        mu = max(max(max_error, end_mu), min_error)

        good = errors_norm < mu
        extra_good = subset_extra(extra, good)
        self.bundle_adjust(p2ds[:, good], extra_good,
                           loss='linear',
                           ftol=ftol, max_nfev=max(200, max_nfev),
                           only_extrinsics=only_extrinsics,
                           verbose=verbose)

        error = self.average_error(p2ds, median=True).item()

        p3ds = self.triangulate(p2ds)
        errors_full = self.reprojection_error(p3ds, p2ds, mean=False).detach().cpu().numpy()
        error_dict = get_error_dict(errors_full)
        if verbose:
            pprint(error_dict)

        if verbose:
            print('error: ', error)

        return error

    def bundle_adjust(self, p2ds, extra=None,
                      loss='linear',
                      ftol=1e-4,
                      max_nfev=1000,
                      lr=1e-3,
                      only_extrinsics=False,
                      verbose=True):
        """Given an CxNx2 array of 2D points,
        where N is the number of points and C is the number of cameras,
        this performs bundle adjustment to fine-tune the parameters of the cameras"""

        assert p2ds.shape[0] == len(self.cameras), \
            "Invalid points shape, first dim should be equal to" \
            " number of cameras ({}), but shape is {}".format(
                len(self.cameras), p2ds.shape
            )

        p2ds = to_tensor(p2ds)
        
        if extra is not None:
            extra = dict(extra)
            extra['ids_map'] = remap_ids(extra['ids'])
            extra['objp'] = torch.as_tensor(extra['objp'])
            objp = extra['objp']
            extra['min_scale'] = torch.amin(objp[objp > 0])

        params = self._initialize_params_bundle(p2ds, extra)
        
        if only_extrinsics:
            cam_params = self.get_extrinsics_params()
        else:
            cam_params = self.parameters()

        all_params = list(cam_params) + list(params.values())
            
        optimizer = optim.Adam(all_params, lr=1e-4, weight_decay=0, fused=True)
        scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=1e-2, total_steps=max_nfev,
                                                  pct_start=0.1)

        old_loss = torch.inf
        for i in range(max_nfev):
            optimizer.zero_grad()
            loss = self._error_fun_bundle(params, p2ds, extra)

            if verbose and i % 20 == 0:
                print("iter: {} \t loss: {:.3f}\t delta: {:.4f}".format(i, loss.item(), old_loss - loss.item()))
            if i > 100 and old_loss - loss < ftol * loss:
                if verbose:
                    print("iter: {} \t loss: {:.3f}\t delta: {:.4f}".format(i, loss.item(), old_loss - loss.item()))
                    print("termination condition reached (delta loss < ftol * loss)")
                break
            old_loss = loss.item()
            loss.backward()
            optimizer.step()
            scheduler.step()

        if verbose:
            print("iter: {} \t loss: {:.3f}".format(i, loss.item()))
            
        error = self.average_error(p2ds)
        return error

    # @torch.compile
    def _error_fun_bundle(self, params, p2ds, extra):
        """Error function for bundle adjustment"""

        p3d = params['p3d']
        valid = torch.isfinite(p2ds).cpu()
        proj = self.project(p3d)
        loss_reproj = torch.mean(torch.square(proj[valid] - p2ds[valid]))
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

            params['rvecs'] = nn.Parameter(torch.as_tensor(rvecs))
            params['tvecs'] = nn.Parameter(torch.as_tensor(tvecs))

        return params

    def optim_points(self, points, p3ds,
                     constraints=[],
                     constraints_weak=[],
                     scale_smooth=4,
                     scale_length=2, scale_length_weak=0.5,
                     reproj_error_threshold=15, reproj_loss='soft_l1',
                     n_deriv_smooth=1, scores=None, verbose=False,
                     n_fixed=0,
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
            p3ds_intp, constraints, constraints_weak)

        if n_fixed > 0:
            p3ds_fixed = to_tensor(p3ds_intp[:n_fixed])
        else:
            p3ds_fixed = None

        points_tensor = to_tensor(points)
            
        all_params = list(params.values())
        optimizer = optim.Adam(all_params, lr=1e-2, weight_decay=0, fused=True)
        scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=5.0, total_steps=n_iters,
                                                  pct_start=0.2)

        ftol = 1e-3
        old_loss = torch.inf
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
            if verbose and i % 20 == 0:
                print("iter: {} \t loss: {:.3f}\t delta: {:.4f}".format(i, loss.item(), old_loss - loss.item()))

            old_loss = loss.item()
            loss.backward()
            optimizer.step()
            scheduler.step()

        t2 = time.time()

        if verbose:
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

 


    def _initialize_params_triangulation(self, p3ds, constraints=[], constraints_weak=[]):
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
            'p3d': nn.Parameter(to_tensor(p3ds)),
            'joint_lengths': nn.Parameter(to_tensor(joint_lengths)),
            'joint_lengths_weak': nn.Parameter(to_tensor(joint_lengths_weak))
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
                matrix = cv2.initCameraMatrix2D(objp, imgp, tuple(size))
                camera.set_camera_matrix(matrix.copy())
                camera.zero_distortions()

        for i, (row, cam) in enumerate(zip(all_rows, self.cameras)):
            all_rows[i] = board.estimate_pose_rows(cam, row)

        new_rows = [[r for r in rows if r['ids'].size >= 8] for rows in all_rows]
        merged = merge_rows(new_rows)
        imgp, extra = extract_points(merged, board, min_cameras=2)

        if init_extrinsics:
            rtvecs = extract_rtvecs(merged)
            if verbose:
                pprint(get_connections(rtvecs, self.get_names()))
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

    def from_names(names, fisheye=False):
        cameras = []
        for name in names:
            if fisheye:
                cam = FisheyeCamera(name=name)
            else:
                cam = Camera(name=name)
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

