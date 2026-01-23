#!/usr/bin/env ipython

import pytest
from hypothesis import given, strategies as st
from hypothesis.extra.numpy import arrays

import numpy as np
# from aniposelib.cameras import Camera, CameraGroup

from aniposelib.cameras import Camera, CameraGroup

import cv2

@given(st.integers())
def test_is_integer(n):
    assert isinstance(n, int)

@given(
    arrays(np.float32, (20,3),
           elements=st.floats(
               min_value=-1e10, max_value=1e10,
               allow_nan=False, allow_infinity=False))
)
def test_project_error(p3d):
    """The reprojection error should be 0 when projecting any 3d points"""
    cgroup = CameraGroup.load('tests/data/calib_motobs_7cam.toml')
    p2d = cgroup.project(p3d)
    err = cgroup.reprojection_error(p3d, p2d)
    assert np.allclose(err, 0), err


@given(
    arrays(np.float32, (20,3),
           elements=st.floats(allow_nan=True, allow_infinity=True))
)
@pytest.mark.filterwarnings("ignore:overflow")
def test_project_nan(p3d):
    """project should handle nans and infs gracefully"""
    cgroup = CameraGroup.load('tests/data/calib_motobs_7cam.toml')
    print(p3d)
    p2d = cgroup.project(p3d)

    finite3d = np.all(np.isfinite(p3d), axis=1)

    # all corresponding finite should be finite
    assert np.all(np.isfinite(p2d)[:, finite3d])

    # all corresponding not finite should be not finite
    assert np.all(np.isnan(p2d)[:, ~finite3d])

def test_triangulation_1():
    """
    For some points roughly in center of arena,
    projecting then triangulating recovers 3d points
    """
    cgroup = CameraGroup.load('tests/data/calib_motobs_7cam.toml')
    p3d = np.vstack([cgroup.get_translations() * 0.5,
                     cgroup.get_translations() * 0.3])
    p2d = cgroup.project(p3d)

    bad = ~cgroup.is_point_visible(p3d)
    p2d[bad] = np.nan

    p3d_tri = cgroup.triangulate(p2d)
    # triangulated is close to true 3d
    assert np.allclose(p3d_tri, p3d)


def test_triangulation_point_cloud():
    cgroup = CameraGroup.load('tests/data/calib_motobs_7cam.toml')
    p3d, s = cgroup.get_point_cloud()
    p2d = cgroup.project(p3d)

    bad = ~cgroup.is_point_visible(p3d)
    p2d[bad] = np.nan

    p3d_tri = cgroup.triangulate(p2d, fast=False)

    reprojerr = cgroup.reprojection_error(p3d_tri, p2d, mean=True)
    p3d_tri[reprojerr > 3] = np.nan # triangulation failed

    err = np.linalg.norm(p3d_tri - p3d, axis=1)
    err = err[np.isfinite(err)]

    # triangulated is close to true 3d
    assert np.all(err < 0.01)



@given(
    arrays(np.float32, (3,),
           elements=st.floats(min_value=-1, max_value=1,
                              allow_nan=False, allow_infinity=False)),

)
def test_projection_sensitivity(p):
    """
    Camera.projection_sensitivity should match finite differences
    calculation
    """

    # Create a test camera
    rvec = np.array([0.1, 0.2, 0.3])
    tvec = np.array([1.0, 0.5, 0.2])
    K = np.array([[800, 0, 320],
                  [0, 800, 240],
                  [0, 0, 1]], dtype='float64')
    dist = np.zeros(5)
    
    cam = Camera(matrix=K, rvec=rvec, tvec=tvec, dist=dist)
    
    # Analytical Jacobian
    J_analytical = cam.projection_sensitivity(p[None])[0]
    
    # Numerical Jacobian using finite differences
    epsilon = 1e-5
    J_numerical = np.zeros((2, 3))
    
    pixel_center = cam.project(p)
    
    for i in range(3):
        p_plus = p.copy()
        p_plus[i] += epsilon
        pixel_plus = cam.project(p_plus)
        J_numerical[:, i] = (pixel_plus - pixel_center) / epsilon

    # don't try to test weird analytical edge cases
    if not np.all(np.isfinite(J_analytical)):
        return 

    # error will be relative to rough scale of the jacobian
    dist = np.abs(J_numerical - J_analytical)
    med = np.median(np.abs(J_analytical))

    print(dist/med)
    assert np.all(dist / med < 0.1)
    

