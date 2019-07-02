import cv2
from cv2 import aruco
import numpy as np
from copy import copy
from scipy.sparse import lil_matrix

def make_M(rvec, tvec):
    out = np.zeros((4,4))
    rotmat, _ = cv2.Rodrigues(rvec)
    out[:3,:3] = rotmat
    out[:3, 3] = tvec.flatten()
    out[3, 3] = 1
    return out

def triangulate_simple(points, camera_mats):
    num_cams = len(camera_mats)
    A = np.zeros((num_cams*2, 4))
    for i in range(num_cams):
        x, y = points[i]
        mat = camera_mats[i]
        A[(i*2):(i*2+1)] = x*mat[2]-mat[0]
        A[(i*2+1):(i*2+2)] = y*mat[2]-mat[1]
    u, s, vh = np.linalg.svd(A, full_matrices=True)
    p3d = vh[-1]
    p3d = p3d[:3] / p3d[3]
    return p3d

class Camera:
    def __init__(self,
                 matrix=np.eye(3), dist=np.zeros(5),
                 size=None,
                 rvec=np.zeros(3), tvec=np.zeros(3),
                 name=None
    ):
        self.matrix = np.array(matrix)
        self.dist = np.array(dist)
        self.size = size
        self.rvec = np.array(rvec)
        self.tvec = np.array(tvec)
        self.name = name

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
            return (fx + fy)/2.0


    def set_distortions(self, dist):
        self.dist = np.array(dist)

    def set_rotation(self, rvec):
        self.rvec = np.array(rvec)

    def get_rotation(self):
        return self.rvec

    def set_translation(self, tvec):
        self.tvec = np.array(tvec)

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
        return params

    def set_params(self, params):
        self.set_rotation(params[0:3])
        self.set_translation(params[3:6])
        self.set_focal_length(params[6])

        dist = np.zeros(5, dtype='float32')
        dist[0] = params[7]
        self.set_distortions(dist)


    def distort_points(self, points):
        shape = points.shape
        points = points.reshape(-1, 1, 2)
        new_points = np.dstack([points, np.ones((points.shape[0],1,1))])
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
        out, _ = cv2.projectPoints(points, self.rvec, self.tvec,
                                   self.matrix, self.dist)
        return out

    def reprojection_error(self, p3d, p2d):
        proj = self.project(p3d).reshape(p2d.shape)
        return p2d - proj

    def copy(self):
        return copy(self)


class CameraGroup:
    def __init__(self, cameras):
        self.cameras = cameras

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

        return out

    def reprojection_error(self, p3ds, p2ds):
        """Given an Nx3 array of 3D points and an CxNx2 array of 2D points,
        where N is the number of points and C is the number of cameras,
        this returns an CxNx2 array of errors"""

        n_cams, n_points, _ = p2ds.shape
        assert p3ds.shape == (n_points, 3), \
            "shapes of 2D and 3D points are not consistent: " \
            "2D={}, 3D={}".format(p2ds.shape, p3ds.shape)

        errors = np.empty((n_cams, n_points, 2))

        for cnum, cam in enumerate(self.cameras):
            errors[cnum] = cam.reprojection_error(p3ds, p2ds[cnum])

        return errors

    # TODO: implement bundle adjustment with object points
    def bundle_adjust(self, p2ds, loss='linear', verbose=True):
        """Given an CxNx2 array of 2D points,
        where N is the number of points and C is the number of cameras,
        this performs bundle adjustsment to fine-tune the parameters of the cameras"""

        x0, n_cam_params = self._initialize_params(p2ds)
        error_fun = self._make_error_fun(p2ds, n_cam_params)

        jac_sparse = self._jac_sparsity_matrix(p2ds, n_cam_params)

        f_scale = 10
        opt = optimize.least_squares(error_fun, x0,
                                     jac_sparsity=jac_sparse, f_scale=f_scale,
                                     x_scale='jac', loss=loss, ftol=1e-4,
                                     method='trf', tr_solver='lsmr',
                                     verbose=2*verbose,
                                     max_nfev=1000)
        best_params = opt.x

        for i, cam in enumerate(self.cameras):
            a = i*n_cam_params
            b = (i+1)*n_cam_params
            cam.set_params(best_params[a:b])


    def _make_error_fun(self, p2ds, n_cam_params):

        cam_group = self.copy()
        good = ~np.isnan(p2ds)

        def error_fun(params):
            for i, cam in enumerate(cam_group.cameras):
                a = i*n_cam_params
                b = (i+1)*n_cam_params
                cam.set_params(params[a:b])

            n_cams = len(cam_group.cameras)
            sub = n_cam_params * n_cams
            p3ds_test = params[sub:].reshape(-1, 3)
            errors = cam_group.reprojection_error(p3ds_test, p2ds)

            return errors[good]

        return error_fun

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
        n_params = n_cams*n_cam_params + n_points*3

        n_errors = np.sum(good)

        A_sparse = lil_matrix((n_errors, n_params), dtype='int16')

        cam_indices_good = cam_indices[good]
        point_indices_good = point_indices[good]

        ix = np.arange(n_errors)

        for i in range(n_cam_params):
            A_sparse[ix, cam_indices_good*n_cam_params + i] = 1

        for i in range(3):
            A_sparse[ix, n_cams*n_cam_params + point_indices_good*3 + i] = 1

        return A_sparse

    def _initialize_params(self, p2ds):
        """Given an CxNx2 array of 2D points,
        where N is the number of points and C is the number of cameras,
        initializes the parameters"""

        cam_params = np.hstack([cam.get_params() for cam in self.cameras])
        n_cam_params = len(cam_params)//len(self.cameras)

        total_cam_params = len(cam_params)

        n_cams, n_points, _ = p2ds.shape
        assert n_cams == len(self.cameras), \
            "number of cameras in CameraGroup does not " \
            "match number of cameras in 2D points given"

        n_point_params = n_points * 3

        p3ds = self.triangulate(p2ds)

        x0 = np.zeros(total_cam_params + p3ds.size)
        x0[:total_cam_params] = cam_params
        x0[total_cam_params:] = p3ds.ravel()

        return x0, n_cam_params

    def copy(self):
        cameras = [cam.copy() for cam in self.cameras]
        return CameraGroup(cameras)
