import cv2
import numpy as np
from copy import copy
from scipy.sparse import lil_matrix
from scipy import optimize
from numba import jit
from collections import defaultdict
import toml
import itertools

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

    def triangulate(self, points, undistort=True):
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

        for ip in range(n_points):
            subp = points[:, ip, :]
            good = ~np.isnan(subp[:, 0])
            if np.sum(good) >= 2:
                out[ip] = triangulate_simple(subp[good], cam_mats[good])

        if one_point:
            out = out[0]

        return out

    def triangulate_possible(self, points, undistort=True, min_cams=2):
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
                all_iters[point_num][cam_num] = [None]
            all_iters[point_num][cam_num].append((cam_num, possible_num))

        out = np.zeros((n_points, 3), dtype='float')
        picked_vals = np.zeros((n_cams, n_points, n_possible), dtype='bool')
        errors = np.zeros(n_points, dtype='float')
        points_2d = np.full((n_cams, n_points, 2), np.nan, dtype='float')

        for point_ix in range(n_points):
            best_point = None
            best_error = 200

            for picked in itertools.product(*all_iters[point_ix].values()):
                picked = [p for p in picked if p is not None]
                if len(picked) < min_cams:
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

            if best_point is not None:
                out[point_ix] = best_point['point']
                picked = best_point['picked']
                cnums = [p[0] for p in picked]
                xnums = [p[1] for p in picked]
                picked_vals[cnums, point_ix, xnums] = True
                errors[point_ix] = best_point['error']
                points_2d[cnums, point_ix] = best_point['points']

        return out, picked_vals, points_2d, errors

    def triangulate_ransac(self, points, undistort=True, min_cams=2):
        """Given an CxNx2 array, this returns an Nx3 array of points,
        where N is the number of points and C is the number of cameras"""

        n_cams, n_points, _ = points.shape

        points_ransac = points.reshape(n_cams, n_points, 1, 2)

        return self.triangulate_possible(points_ransac,
                                         undistort=undistort, min_cams=min_cams)


    @jit(nopython=True, parallel=True, forceobj=True)
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
            errors = np.sum(errors_norm, axis=0) / np.sum(good, axis=0)

        if one_point:
            if mean:
                errors = float(errors[0])
            else:
                errors = errors.reshape(-1, 2)

        return errors

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

        x0, n_cam_params = self._initialize_params(p2ds)
        # error_fun = self._make_error_fun(p2ds, n_cam_params, weights=weights)

        if start_params is not None:
            x0 = start_params
            n_cam_params = len(self.cameras[0].get_params())

        error_fun = self._error_fun

        jac_sparse = self._jac_sparsity_matrix(p2ds, n_cam_params)

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

    def bundle_adjust_iter(self, p2ds, n_iters=7, start_mu=None, end_mu=5,
                           max_nfev=50, ftol=1e-2, verbose=True):
        """Given an CxNx2 array of 2D points,
        where N is the number of points and C is the number of cameras,
        this performs iterative bundle adjustsment to fine-tune the parameters of the cameras.
        That is, it performs bundle adjustment multiple times, adjusting the weights given to points
        to reduce the influence of outliers.
        This is inspired by the algorithm for Fast Global Registration by Zhou, Park, and Koltun
        """
        error = self.average_error(p2ds)

        if verbose:
            print('error: ', error)

        if start_mu is None:
            p3ds = self.triangulate(p2ds)
            errors = self.reprojection_error(p3ds, p2ds, mean=True)
            q1, q3 = np.percentile(errors, [25, 75])
            iqr = q3 - q1
            start_mu = q3 + 7*iqr

        # error = self.bundle_adjust(p2ds, threshold=start_mu, loss='huber', ftol=1e-2, max_nfev=100)
        # if verbose:
        #     print('error: ', error)

        mus = np.exp(np.linspace(np.log(start_mu), np.log(end_mu), num=n_iters))

        for i in range(n_iters):
            p3ds = self.triangulate(p2ds)
            errors = self.reprojection_error(p3ds, p2ds, mean=True)
            mu = mus[i]
            good = errors < mu
            self.bundle_adjust(p2ds[:, good],
                               loss='linear', ftol=ftol,
                               max_nfev=max_nfev,
                               verbose=verbose)
            error = self.average_error(p2ds)

            if verbose:
                print('error: {:.2f}, mu: {:.1f}'.format(error, mu))

        p3ds = self.triangulate(p2ds)
        errors = self.reprojection_error(p3ds, p2ds, mean=True)
        good = errors < end_mu
        self.bundle_adjust(p2ds[:, good], loss='linear', ftol=ftol/10.0, verbose=verbose)

        error = self.average_error(p2ds)

        if verbose:
            print('error: ', error)

        return error

    @jit(nopython=True, parallel=True, forceobj=True)
    def _error_fun(self, params, p2ds, n_cam_params, weights=None):
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


    def _jac_sparsity_matrix(self, p2ds, n_cam_params):
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

    def _initialize_params(self, p2ds):
        """Given an CxNx2 array of 2D points,
        where N is the number of points and C is the number of cameras,
        initializes the parameters"""

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

    def average_error(self, p2ds):
        p3ds = self.triangulate(p2ds)
        errors = self.reprojection_error(p3ds, p2ds, mean=True)
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
        objp, imgp = extract_points(merged, board, min_cameras=2, ignore_no_pose=True)

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
