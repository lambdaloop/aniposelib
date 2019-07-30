import cv2
import numpy as np
from copy import copy
from scipy.sparse import lil_matrix
from scipy import optimize
from scipy import signal
from numba import jit
from collections import defaultdict
import toml
import itertools
from tqdm import trange
from pprint import pprint

from .boards import merge_rows, extract_points, \
    extract_rtvecs, get_video_params
from .utils import get_initial_extrinsics

def make_M(rvec, tvec):
    out = np.zeros((4, 4))
    rotmat, _ = cv2.Rodrigues(rvec)
    out[:3, :3] = rotmat
    out[:3, 3] = tvec.flatten()
    out[3, 3] = 1
    return out

@jit(nopython=True, parallel=True)
def triangulate_simple(points, camera_mats):
    num_cams = len(camera_mats)
    A = np.zeros((num_cams * 2, 4))
    for i in range(num_cams):
        x, y = points[i]
        mat = camera_mats[i]
        A[(i * 2):(i * 2 + 1)] = x * mat[2] - mat[0]
        A[(i * 2 + 1):(i * 2 + 2)] = y * mat[2] - mat[1]
    u, s, vh = np.linalg.svd(A, full_matrices=True)
    p3d = vh[-1]
    p3d = p3d[:3] / p3d[3]
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
            if np.sum(subset) > min_points:
                percents = np.percentile(err_subset, [10, 50])
                error_dict[(i, j)] = (err_subset.shape[1], percents)
    return error_dict

def check_errors(cgroup, imgp):
    p3ds = cgroup.triangulate(imgp)
    errors_full = cgroup.reprojection_error(p3ds, imgp, mean=False)
    return get_error_dict(errors_full)

def resample_points(imgp, n_samp=25):
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
                arr = np.copy(num_cams[subset]).astype('float')
                arr += np.random.random(size=arr.shape)
                picked_ix = np.argsort(-arr)[:n_samp]
                picked = ixs[subset][picked_ix]
                include.update(picked)

    newp = imgp[:, sorted(include)]
    return newp

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


class Camera:
    def __init__(self,
                 matrix=np.eye(3),
                 dist=np.zeros(5),
                 size=None,
                 rvec=np.zeros(3),
                 tvec=np.zeros(3),
                 name=None):
        self.matrix = np.array(matrix)
        self.dist = np.array(dist)
        self.size = size
        self.rvec = np.array(rvec)
        self.tvec = np.array(tvec)
        self.name = name

    def get_dict(self):
        return {
            'name': self.get_name(),
            'size': self.get_size(),
            'matrix': self.get_camera_matrix().tolist(),
            'distortions': self.get_distortions().tolist(),
            'rotation': self.get_rotation().tolist(),
            'translation': self.get_translation().tolist(),
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
        self.matrix = np.array(matrix)

    def set_focal_length(self, fx, fy=None):
        if fy is None:
            fy = fx
        self.matrix[0, 0] = fx
        self.matrix[1, 1] = fy

    def get_focal_length(self, both=False):
        fx = self.matrix[0, 0]
        fy = self.matrix[1, 1]
        if both:
            return (fx, fy)
        else:
            return (fx + fy) / 2.0

    def set_distortions(self, dist):
        self.dist = np.array(dist).ravel()

    def set_rotation(self, rvec):
        self.rvec = np.array(rvec).ravel()

    def get_rotation(self):
        return self.rvec

    def set_translation(self, tvec):
        self.tvec = np.array(tvec).ravel()

    def get_translation(self):
        return self.tvec

    def get_extrinsics_mat(self):
        return make_M(self.rvec, self.tvec)

    def get_name(self):
        return self.name

    def set_name(self, name):
        self.name = name

    def set_size(self, size):
        """set size as (width, height)"""
        self.size = size

    def get_size(self):
        """get size as (width, height)"""
        return self.size

    def get_params(self):
        params = np.zeros(8, dtype='float32')
        params[0:3] = self.get_rotation()
        params[3:6] = self.get_translation()
        params[6] = self.get_focal_length()
        dist = self.get_distortions()
        params[7] = dist[0]
        # params[8] = dist[1]

        return params

    def set_params(self, params):
        self.set_rotation(params[0:3])
        self.set_translation(params[3:6])
        self.set_focal_length(params[6])

        dist = np.zeros(5, dtype='float32')
        dist[0] = params[7]
        # dist[1] = params[8]
        self.set_distortions(dist)

    def distort_points(self, points):
        shape = points.shape
        points = points.reshape(-1, 1, 2)
        new_points = np.dstack([points, np.ones((points.shape[0], 1, 1))])
        out, _ = cv2.projectPoints(new_points, np.zeros(3), np.zeros(3),
                                   self.matrix, self.dist)
        return out.reshape(shape)

    def undistort_points(self, points):
        shape = points.shape
        points = points.reshape(-1, 1, 2)
        out = cv2.undistortPoints(points, self.matrix, self.dist)
        return out.reshape(shape)

    def project(self, points):
        points = points.reshape(-1, 1, 3)
        out, _ = cv2.projectPoints(points, self.rvec, self.tvec, self.matrix,
                                   self.dist)
        return out

    def reprojection_error(self, p3d, p2d):
        proj = self.project(p3d).reshape(p2d.shape)
        return p2d - proj

    def copy(self):
        return copy(self)



class CameraGroup:
    def __init__(self, cameras, metadata={}):
        self.cameras = cameras
        self.metadata = metadata

    def subset_cameras(self, indices):
        cams = np.array(self.cameras)
        cams = list(cams[indices])
        return CameraGroup(cams)

    def project(self, points):
        """Given an Nx3 array of points, this returns an CxNx2 array of 2D points,
        where C is the number of cameras"""
        points = points.reshape(-1, 1, 3)
        n_points = points.shape[0]
        n_cams = len(self.cameras)

        out = np.zeros((n_cams, n_points, 2))
        for cnum, cam in enumerate(self.cameras):
            out[cnum] = cam.project(points).reshape(n_points, 2)

        return out

    def triangulate(self, points, undistort=True, progress=False):
        """Given an CxNx2 array, this returns an Nx3 array of points,
        where N is the number of points and C is the number of cameras"""

        one_point = False
        if len(points.shape) == 2:
            points = points.reshape(-1, 1, 2)
            one_point = True

        if undistort:
            new_points = np.empty(points.shape)
            for cnum, cam in enumerate(self.cameras):
                # must copy in order to satisfy opencv underneath
                sub = np.copy(points[cnum])
                new_points[cnum] = cam.undistort_points(sub)
            points = new_points

        n_cams, n_points, _ = points.shape

        out = np.empty((n_points, 3))
        out[:] = np.nan

        cam_mats = np.array([cam.get_extrinsics_mat() for cam in self.cameras])

        if progress:
            iterator = trange(n_points, ncols=70)
        else:
            iterator = range(n_points)

        for ip in iterator:
            subp = points[:, ip, :]
            good = ~np.isnan(subp[:, 0])
            if np.sum(good) >= 2:
                out[ip] = triangulate_simple(subp[good], cam_mats[good])

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

        n_cams, n_points, n_possible, _ = points.shape

        cam_nums, point_nums, possible_nums = np.where(
            ~np.isnan(points[:, :, :, 0]))

        all_iters = defaultdict(dict)

        for cam_num, point_num, possible_num in zip(cam_nums, point_nums,
                                                    possible_nums):
            if cam_num not in all_iters[point_num]:
                all_iters[point_num][cam_num] = []
            all_iters[point_num][cam_num].append((cam_num, possible_num))

        for point_num in all_iters.keys():
            for cam_num in all_iters[point_num].keys():
                all_iters[point_num][cam_num].append(None)

        out = np.full((n_points, 3), np.nan, dtype='float')
        picked_vals = np.zeros((n_cams, n_points, n_possible), dtype='bool')
        errors = np.zeros(n_points, dtype='float')
        points_2d = np.full((n_cams, n_points, 2), np.nan, dtype='float')

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
                if len(picked) < min_cams and len(picked) != n_cams_max:
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

        one_point = False
        if len(p3ds.shape) == 1 and len(p2ds.shape) == 2:
            p3ds = p3ds.reshape(1, 3)
            p2ds = p2ds.reshape(-1, 1, 2)
            one_point = True

        n_cams, n_points, _ = p2ds.shape
        assert p3ds.shape == (n_points, 3), \
            "shapes of 2D and 3D points are not consistent: " \
            "2D={}, 3D={}".format(p2ds.shape, p3ds.shape)

        errors = np.empty((n_cams, n_points, 2))

        for cnum, cam in enumerate(self.cameras):
            errors[cnum] = cam.reprojection_error(p3ds, p2ds[cnum])

        if mean:
            errors_norm = np.linalg.norm(errors, axis=2)
            good = ~np.isnan(errors_norm)
            errors_norm[~good] = 0
            denom = np.sum(good, axis=0).astype('float')
            denom[denom < 0.5] = np.nan
            errors = np.sum(errors_norm, axis=0) / denom

        if one_point:
            if mean:
                errors = float(errors[0])
            else:
                errors = errors.reshape(-1, 2)

        return errors


    def bundle_adjust_iter(self, p2ds, n_iters=15, start_mu=15, end_mu=1,
                           max_nfev=200, ftol=1e-4, verbose=True):
        """Given an CxNx2 array of 2D points,
        where N is the number of points and C is the number of cameras,
        this performs iterative bundle adjustsment to fine-tune the parameters of the cameras.
        That is, it performs bundle adjustment multiple times, adjusting the weights given to points
        to reduce the influence of outliers.
        This is inspired by the algorithm for Fast Global Registration by Zhou, Park, and Koltun
        """


        error = self.average_error(p2ds, median=True)

        if verbose:
            print('error: ', error)

        mus = np.exp(np.linspace(np.log(start_mu), np.log(end_mu), num=n_iters))

        for i in range(n_iters):
            p3ds = self.triangulate(p2ds)
            errors_full = self.reprojection_error(p3ds, p2ds, mean=False)
            errors_norm = self.reprojection_error(p3ds, p2ds, mean=True)

            error_dict = get_error_dict(errors_full)
            max_error = 0
            min_error = 0
            for k, v in error_dict.items():
                num, percents = v
                max_error = max(percents[-1], max_error)
                min_error = max(percents[0], min_error)
            mu = max(min(max_error, mus[i]), min_error)

            good = errors_norm < mu
            p2ds_samp = resample_points(p2ds[:, good], n_samp=30)

            pprint(error_dict)

            self.bundle_adjust(p2ds_samp,
                               loss='linear', ftol=ftol,
                               max_nfev=max_nfev,
                               verbose=verbose)

            if verbose:
                error = self.average_error(p2ds, median=True)
                print('error: {:.2f}, mu: {:.1f}, ratio: {:.3f}'.format(error, mu, np.mean(good)))

        p3ds = self.triangulate(p2ds)
        errors_full = self.reprojection_error(p3ds, p2ds, mean=False)
        errors_norm = self.reprojection_error(p3ds, p2ds, mean=True)
        error_dict = get_error_dict(errors_full)
        pprint(error_dict)

        max_error = 0
        min_error = 0
        for k, v in error_dict.items():
            num, percents = v
            max_error = max(percents[-1], max_error)
            min_error = max(percents[0], min_error)
        mu = max(max(max_error, end_mu), min_error)

        good = errors_norm < mu
        self.bundle_adjust(p2ds[:, good], loss='linear',
                           ftol=ftol, max_nfev=max_nfev,
                           verbose=verbose)

        error = self.average_error(p2ds, median=True)

        p3ds = self.triangulate(p2ds)
        errors_full = self.reprojection_error(p3ds, p2ds, mean=False)
        error_dict = get_error_dict(errors_full)
        pprint(error_dict)

        if verbose:
            print('error: ', error)

        return error


    # TODO: implement bundle adjustment with object points
    def bundle_adjust(self, p2ds,
                      loss='linear',
                      threshold=50,
                      ftol=1e-2,
                      max_nfev=1000,
                      weights=None,
                      start_params=None,
                      verbose=True):
        """Given an CxNx2 array of 2D points,
        where N is the number of points and C is the number of cameras,
        this performs bundle adjustsment to fine-tune the parameters of the cameras"""

        x0, n_cam_params = self._initialize_params_bundle(p2ds)

        if start_params is not None:
            x0 = start_params
            n_cam_params = len(self.cameras[0].get_params())

        error_fun = self._error_fun_bundle

        jac_sparse = self._jac_sparsity_bundle(p2ds, n_cam_params)

        f_scale = threshold
        opt = optimize.least_squares(error_fun,
                                     x0,
                                     jac_sparsity=jac_sparse,
                                     f_scale=f_scale,
                                     x_scale='jac',
                                     loss=loss,
                                     ftol=ftol,
                                     method='trf',
                                     tr_solver='lsmr',
                                     verbose=2 * verbose,
                                     max_nfev=max_nfev,
                                     args=(p2ds, n_cam_params, weights))
        best_params = opt.x

        for i, cam in enumerate(self.cameras):
            a = i * n_cam_params
            b = (i + 1) * n_cam_params
            cam.set_params(best_params[a:b])

        error = self.average_error(p2ds)
        return error

    @jit(nopython=True, parallel=True, forceobj=True)
    def _error_fun_bundle(self, params, p2ds, n_cam_params, weights=None):
        """Error function for bundle adjustment"""
        good = ~np.isnan(p2ds)
        n_cams = len(self.cameras)

        for i in range(n_cams):
            cam = self.cameras[i]
            a = i * n_cam_params
            b = (i + 1) * n_cam_params
            cam.set_params(params[a:b])

        n_cams = len(self.cameras)
        sub = n_cam_params * n_cams
        p3ds_test = params[sub:].reshape(-1, 3)
        errors = self.reprojection_error(p3ds_test, p2ds)

        if weights is not None:
            errors = errors * weights.reshape(-1, 1)

        return errors[good]


    def _jac_sparsity_bundle(self, p2ds, n_cam_params):
        """Given an CxNx2 array of 2D points,
        where N is the number of points and C is the number of cameras,
        compute the sparsity structure of the jacobian for bundle adjustment"""

        point_indices = np.zeros(p2ds.shape, dtype='int32')
        cam_indices = np.zeros(p2ds.shape, dtype='int32')

        for i in range(p2ds.shape[1]):
            point_indices[:, i] = i

        for j in range(p2ds.shape[0]):
            cam_indices[j] = j

        good = ~np.isnan(p2ds)

        n_cams = p2ds.shape[0]
        n_points = p2ds.shape[1]
        n_params = n_cams * n_cam_params + n_points * 3

        n_errors = np.sum(good)

        A_sparse = lil_matrix((n_errors, n_params), dtype='int16')

        cam_indices_good = cam_indices[good]
        point_indices_good = point_indices[good]

        ix = np.arange(n_errors)

        for i in range(n_cam_params):
            A_sparse[ix, cam_indices_good * n_cam_params + i] = 1

        for i in range(3):
            A_sparse[ix, n_cams * n_cam_params + point_indices_good * 3 + i] = 1

        return A_sparse

    def _initialize_params_bundle(self, p2ds):
        """Given an CxNx2 array of 2D points,
        where N is the number of points and C is the number of cameras,
        initializes the parameters for bundle adjustment"""

        cam_params = np.hstack([cam.get_params() for cam in self.cameras])
        n_cam_params = len(cam_params) // len(self.cameras)

        total_cam_params = len(cam_params)

        n_cams, n_points, _ = p2ds.shape
        assert n_cams == len(self.cameras), \
            "number of cameras in CameraGroup does not " \
            "match number of cameras in 2D points given"

        p3ds = self.triangulate(p2ds)

        x0 = np.zeros(total_cam_params + p3ds.size)
        x0[:total_cam_params] = cam_params
        x0[total_cam_params:] = p3ds.ravel()

        return x0, n_cam_params

    def triangulate_optim(self, points,
                          constraints=[],
                          constraints_weak=[],
                          scale_smooth=4,
                          scale_length=2, scale_length_weak=0.5,
                          reproj_error_threshold=15,
                          n_deriv_smooth=1, scores=None, init_progress=False,
                          init_ransac=False, verbose=False):
        """
        Take in an array of 2D points of shape CxNxJx2, and an array of constraints of shape Kx2, where
        C: number of camera
        N: number of frames
        J: number of joints
        K: number of constraints

        This function creates an optimized array of 3D points of shape NxJx3.

        Example constraints:
        constraints = [[0, 1], [1, 2], [2, 3]]
        (meaning that lengths of joints 0->1, 1->2, 2->3 are all constant)

        """

        n_cams, n_frames, n_joints, _ = points.shape
        constraints = np.array(constraints)
        constraints_weak = np.array(constraints_weak)

        points_shaped = points.reshape(n_cams, n_frames*n_joints, 2)
        if init_ransac:
            p3ds, picked, p2ds, errors = self.triangulate_ransac(points_shaped, progress=init_progress)
        else:
            p3ds = self.triangulate(points_shaped, progress=init_progress)
        p3ds = p3ds.reshape((n_frames, n_joints, 3))
        p3ds_intp = np.apply_along_axis(interpolate_data, 0, p3ds)

        p3ds_med = np.apply_along_axis(medfilt_data, 0, p3ds_intp, size=7)

        default_smooth = 1.0/np.mean(np.abs(np.diff(p3ds_med, axis=0)))
        scale_smooth_full = scale_smooth * default_smooth

        jac1 = self._jac_sparsity_triangulation(
            points, [], [], n_deriv_smooth)
        jac2 = self._jac_sparsity_triangulation(
            points, constraints, constraints_weak, n_deriv_smooth)

        x1 = self._initialize_params_triangulation(p3ds_intp)

        opt1 = optimize.least_squares(self._error_fun_triangulation,
                                      x0=x1, jac_sparsity=jac1,
                                      loss='linear',
                                      ftol=1e-3,
                                      verbose=2*verbose,
                                      args=(points,
                                            np.array([]),
                                            np.array([]),
                                            scores,
                                            scale_smooth_full,
                                            0,
                                            0,
                                            reproj_error_threshold,
                                            n_deriv_smooth))

        p3ds_new1 = opt1.x[:p3ds.size].reshape(p3ds.shape)
        x2 = self._initialize_params_triangulation(
            p3ds_new1, constraints, constraints_weak)

        opt2 = optimize.least_squares(self._error_fun_triangulation,
                                      x0=x2, jac_sparsity=jac2,
                                      loss='linear',
                                      ftol=1e-3,
                                      verbose=2*verbose,
                                      args=(points,
                                            constraints,
                                            constraints_weak,
                                            scores,
                                            scale_smooth_full,
                                            scale_length,
                                            scale_length_weak,
                                            reproj_error_threshold,
                                            n_deriv_smooth))

        p3ds_new2 = opt2.x[:p3ds.size].reshape(p3ds.shape)

        return p3ds_new2


    @jit(nopython=True, forceobj=True)
    def _error_fun_triangulation(self, params, p2ds,
                                 constraints=[],
                                 constraints_weak=[],
                                 scores=None,
                                 scale_smooth=10000,
                                 scale_length=1,
                                 scale_length_weak=0.2,
                                 reproj_error_threshold=100,
                                 n_deriv_smooth=1):
        n_cams, n_frames, n_joints, _ = p2ds.shape

        n_3d = n_frames*n_joints*3
        n_constraints = len(constraints)
        n_constraints_weak = len(constraints_weak)

        # load params
        p3ds = params[:n_3d].reshape((n_frames, n_joints, 3))
        joint_lengths = np.array(params[n_3d:n_3d+n_constraints])
        joint_lengths_weak = np.array(params[n_3d+n_constraints:])

        # reprojection errors
        p3ds_flat = p3ds.reshape(-1, 3)
        p2ds_flat = p2ds.reshape((n_cams, -1, 2))
        errors = self.reprojection_error(p3ds_flat, p2ds_flat)
        if scores is not None:
            scores_flat = scores.reshape((n_cams, -1))
            errors = errors * scores_flat[:, :, None]
        errors_reproj = errors[~np.isnan(p2ds_flat)]

        rp = reproj_error_threshold
        errors_reproj = np.abs(errors_reproj)
        bad = errors_reproj > rp
        errors_reproj[bad] = rp*(2*np.sqrt(errors_reproj[bad]/rp) - 1)

        # temporal constraint
        errors_smooth = np.diff(p3ds, n=n_deriv_smooth, axis=0).ravel() * scale_smooth

        # joint length constraint
        errors_lengths = np.empty((n_constraints, n_frames), dtype='float')
        for cix, (a, b) in enumerate(constraints):
            lengths = np.linalg.norm(p3ds[:, a] - p3ds[:, b], axis=1)
            expected = joint_lengths[cix]
            errors_lengths[cix] = 100*(lengths - expected)/expected
        errors_lengths = errors_lengths.ravel() * scale_length

        errors_lengths_weak = np.empty((n_constraints_weak, n_frames), dtype='float')
        for cix, (a, b) in enumerate(constraints_weak):
            lengths = np.linalg.norm(p3ds[:, a] - p3ds[:, b], axis=1)
            expected = joint_lengths_weak[cix]
            errors_lengths_weak[cix] = 100*(lengths - expected)/expected
        errors_lengths_weak = errors_lengths_weak.ravel() * scale_length_weak

        return np.hstack([errors_reproj, errors_smooth,
                          errors_lengths, errors_lengths_weak])

    def _initialize_params_triangulation(self, p3ds,
                                         constraints=[],
                                         constraints_weak=[]):
        joint_lengths = np.empty(len(constraints), dtype='float')
        joint_lengths_weak = np.empty(len(constraints_weak), dtype='float')

        for cix, (a, b) in enumerate(constraints):
            lengths = np.linalg.norm(p3ds[:, a] - p3ds[:, b], axis=1)
            joint_lengths[cix] = np.median(lengths)

        for cix, (a, b) in enumerate(constraints_weak):
            lengths = np.linalg.norm(p3ds[:, a] - p3ds[:, b], axis=1)
            joint_lengths_weak[cix] = np.median(lengths)

        return np.hstack([p3ds.ravel(), joint_lengths, joint_lengths_weak])


    def _jac_sparsity_triangulation(self, p2ds,
                                    constraints=[],
                                    constraints_weak=[],
                                    n_deriv_smooth=1):
        n_cams, n_frames, n_joints, _ = p2ds.shape
        n_constraints = len(constraints)
        n_constraints_weak = len(constraints_weak)

        p2ds_flat = p2ds.reshape((n_cams, -1, 2))

        point_indices = np.zeros(p2ds_flat.shape, dtype='int32')
        for i in range(p2ds_flat.shape[1]):
            point_indices[:, i] = i

        point_indices_3d = np.arange(n_frames*n_joints)\
                             .reshape((n_frames, n_joints))

        good = ~np.isnan(p2ds_flat)
        n_errors_reproj = np.sum(good)
        n_errors_smooth = (n_frames-n_deriv_smooth) * n_joints * 3
        n_errors_lengths = n_constraints * n_frames
        n_errors_lengths_weak = n_constraints_weak * n_frames

        n_errors = n_errors_reproj + n_errors_smooth + \
            n_errors_lengths + n_errors_lengths_weak

        n_3d = n_frames*n_joints*3
        n_params = n_3d + n_constraints + n_constraints_weak

        point_indices_good = point_indices[good]

        A_sparse = lil_matrix((n_errors, n_params), dtype='int16')

        # constraints for reprojection errors
        ix_reproj = np.arange(n_errors_reproj)
        for k in range(3):
            A_sparse[ix_reproj, point_indices_good * 3 + k] = 1

        # sparse constraints for smoothness in time
        frames = np.arange(n_frames-n_deriv_smooth)
        for j in range(n_joints):
            for n in range(n_deriv_smooth+1):
                pa = point_indices_3d[frames, j]
                pb = point_indices_3d[frames+n, j]
                for k in range(3):
                    A_sparse[n_errors_reproj + pa*3 + k, pb*3 + k] = 1

        ## -- strong constraints --
        # joint lengths should change with joint lengths errors
        start = n_errors_reproj + n_errors_smooth
        frames = np.arange(n_frames)
        for cix, (a, b) in enumerate(constraints):
            A_sparse[start + cix*n_frames + frames, n_3d+cix] = 1

        # points should change accordingly to match joint lengths too
        frames = np.arange(n_frames)
        for cix, (a, b) in enumerate(constraints):
            pa = point_indices_3d[frames, a]
            pb = point_indices_3d[frames, b]
            for k in range(3):
                A_sparse[start + cix*n_frames + frames, pa*3 + k] = 1
                A_sparse[start + cix*n_frames + frames, pb*3 + k] = 1

        ## -- weak constraints --
        # joint lengths should change with joint lengths errors
        start = n_errors_reproj + n_errors_smooth + n_errors_lengths
        frames = np.arange(n_frames)
        for cix, (a, b) in enumerate(constraints_weak):
            A_sparse[start + cix*n_frames + frames, n_3d+cix] = 1

        # points should change accordingly to match joint lengths too
        frames = np.arange(n_frames)
        for cix, (a, b) in enumerate(constraints_weak):
            pa = point_indices_3d[frames, a]
            pb = point_indices_3d[frames, b]
            for k in range(3):
                A_sparse[start + cix*n_frames + frames, pa*3 + k] = 1
                A_sparse[start + cix*n_frames + frames, pb*3 + k] = 1

        return A_sparse


    def copy(self):
        cameras = [cam.copy() for cam in self.cameras]
        return CameraGroup(cameras)

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
        return np.array(rvecs)

    def get_translations(self):
        tvecs = []
        for cam in self.cameras:
            tvec = cam.get_translation()
            tvecs.append(tvec)
        return np.array(tvecs)

    def average_error(self, p2ds, median=False):
        p3ds = self.triangulate(p2ds)
        errors = self.reprojection_error(p3ds, p2ds, mean=True)
        if median:
            return np.median(errors)
        else:
            return np.mean(errors)

    def calibrate_rows(self, all_rows, board, verbose=True):
        """Assumes camera sizes are set properly"""
        for rows, camera in zip(all_rows, self.cameras):
            size = camera.get_size()

            assert size is not None, \
                "Camera with name {} has no specified frame size".format(camera.get_name())

            objp, imgp = board.get_all_calibration_points(rows)
            matrix = cv2.initCameraMatrix2D([objp], [imgp], size)
            camera.set_camera_matrix(matrix)

        for i, (row, cam) in enumerate(zip(all_rows, self.cameras)):
            all_rows[i] = board.estimate_pose_rows(cam, row)

        merged = merge_rows(all_rows)
        objp, imgp = extract_points(merged, board, min_cameras=2, ignore_no_pose=False)

        rtvecs = extract_rtvecs(merged)
        rvecs, tvecs = get_initial_extrinsics(rtvecs)
        self.set_rotations(rvecs)
        self.set_translations(tvecs)

        error = self.bundle_adjust_iter(imgp, verbose=verbose)

        return error

    def calibrate_videos(self, videos, board, verbose=True):
        """Takes as input a list of list of video filenames, one list of each camera.
        Also takes a board which specifies what should be detected in the videos"""

        all_rows = []

        for cix, (cam, cam_videos) in enumerate(zip(self.cameras, videos)):
            rows_cam = []
            for vnum, vidname in enumerate(cam_videos):
                if verbose: print(vidname)
                rows = board.detect_video(vidname, prefix=vnum, progress=verbose)
                rows_cam.extend(rows)

                params = get_video_params(vidname)
                size = (params['width'], params['height'])
                cam.set_size(size)
            all_rows.append(rows_cam)

        error = self.calibrate_rows(all_rows, board, verbose=verbose)
        return error, all_rows

    def get_dicts(self):
        out = []
        for cam in self.cameras:
            out.append(cam.get_dict())
        return out

    def from_dicts(arr):
        cameras = []
        for d in arr:
            cam = Camera.from_dict(d)
            cameras.append(cam)
        return CameraGroup(cameras)

    def from_names(names):
        cameras = []
        for name in names:
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
