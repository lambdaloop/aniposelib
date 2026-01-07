#!/usr/bin/env ipython

import pytest
from hypothesis import given, strategies as st
from hypothesis.extra.numpy import arrays

import numpy as np
from aniposelib.cameras import CameraGroup

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
    """For some points roughly in center of arena,
    projecting then triangulating recovers 3d points"""
    cgroup = CameraGroup.load('tests/data/calib_motobs_7cam.toml')
    p3d = np.vstack([cgroup.get_translations() * 0.5,
                     cgroup.get_translations() * 0.3])
    p2d = cgroup.project(p3d)

    sizes = np.array([cam.get_size() for cam in cgroup.cameras])

    bad = np.any((p2d < 0) | (p2d > sizes[:, None]), axis=2)
    p2d[bad] = np.nan

    p3d_tri = cgroup.triangulate(p2d)
    # triangulated is close to true 3d
    assert np.allclose(p3d_tri, p3d)
