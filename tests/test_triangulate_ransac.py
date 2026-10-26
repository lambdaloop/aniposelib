#!/usr/bin/env python

import pytest
import numpy as np
import torch
import os

from aniposelib.cameras import CameraGroup

CALIB_PATH = os.path.join(
    os.path.dirname(__file__),
    'data/rigs/outdoor_9cam.toml'
)

# Approximate scene center derived from the mean of camera centers
SCENE_CENTER = np.array([-0.03, -0.38, 0.62])


def load_cgroup():
    return CameraGroup.load(CALIB_PATH)


def synthetic_scene(cgroup, n_points=100, spread=0.05, seed=0):
    """Generate 3D points near the scene center and project to all cameras."""
    rng = np.random.default_rng(seed)
    p3d_gt = SCENE_CENTER + rng.uniform(-spread, spread, size=(n_points, 3))
    p2d = cgroup.project(p3d_gt)  # numpy in -> numpy out
    return p3d_gt, p2d  # p2d: (C, N, 2)


def test_shapes_and_nan_mask():
    """triangulate_ransac returns correct shapes and NaN-mask semantics of points_2d."""
    cgroup = load_cgroup()
    n_cams = len(cgroup.cameras)
    n_points = 50
    rng = np.random.default_rng(10)

    _, p2d = synthetic_scene(cgroup, n_points=n_points, seed=10)

    # Drop one random camera per point so RANSAC must choose a proper subset
    for n in range(n_points):
        c = rng.integers(n_cams)
        p2d[c, n] = np.nan

    out, picked_vals, points_2d, errors = cgroup.triangulate_ransac(p2d)
    errors = errors

    assert out.shape == (n_points, 3)
    assert picked_vals.shape == (n_cams, n_points, 1)
    assert points_2d.shape == (n_cams, n_points, 2)
    assert errors.shape == (n_points,)

    # points_2d must be NaN exactly where picked_vals is False
    for c in range(n_cams):
        picked = picked_vals[c, :, 0]
        assert np.all(np.isnan(points_2d[c, ~picked, 0])), \
            f"cam {c}: unpicked slots should be NaN"
        assert np.all(~np.isnan(points_2d[c, picked, 0])), \
            f"cam {c}: picked slots should not be NaN"


def test_ground_truth_recovery():
    """Triangulated 3D points should be close to the ground truth."""
    cgroup = load_cgroup()
    n_cams = len(cgroup.cameras)
    n_points = 100
    rng = np.random.default_rng(20)

    p3d_gt, p2d = synthetic_scene(cgroup, n_points=n_points, seed=20)

    # Drop one camera per point so RANSAC has to select a proper subset
    for n in range(n_points):
        c = rng.integers(n_cams)
        p2d[c, n] = np.nan

    out, _, _, errors = cgroup.triangulate_ransac(p2d)

    assert np.all(np.isfinite(out)), "All 3D points should be finite"
    assert np.all(np.isfinite(errors)), "All errors should be finite"

    err3d = np.linalg.norm(out - p3d_gt, axis=1)
    assert np.all(err3d < 1e-3), f"Max 3D error: {err3d.max():.2e}"


def test_outlier_exclusion():
    """A camera with large 2D noise should be excluded from the winning subset."""
    cgroup = load_cgroup()
    n_cams = len(cgroup.cameras)
    n_points = 80
    rng = np.random.default_rng(30)

    p3d_gt, p2d = synthetic_scene(cgroup, n_points=n_points, seed=30)

    # Corrupt camera 0 for the first half with large additive noise
    outlier_cam = 0
    n_outlier = n_points // 2
    p2d[outlier_cam, :n_outlier] += rng.normal(0, 30.0, size=(n_outlier, 2))

    out, picked_vals, _, _ = cgroup.triangulate_ransac(p2d)

    # The outlier camera should be excluded for all corrupted points
    for n in range(n_outlier):
        assert not picked_vals[outlier_cam, n, 0], \
            f"Outlier cam {outlier_cam} should be excluded for point {n}"

    # 3D accuracy should be preserved despite the outlier
    err3d = np.linalg.norm(out[:n_outlier] - p3d_gt[:n_outlier], axis=1)
    assert np.all(err3d < 1e-3), f"Max 3D error with outlier: {err3d.max():.2e}"


def test_triangulate_possible_p2():
    """With P=2 candidates, the true detection (q=0) should be selected over the distractor (q=1)."""
    cgroup = load_cgroup()
    n_cams = len(cgroup.cameras)
    n_points = 50
    rng = np.random.default_rng(40)

    p3d_gt, p2d_true = synthetic_scene(cgroup, n_points=n_points, seed=40)

    # Build (C, N, P=2, 2): slot 0 = true, slot 1 = large distractor
    p2d_2 = np.full((n_cams, n_points, 2, 2), np.nan)
    p2d_2[:, :, 0, :] = p2d_true
    p2d_2[:, :, 1, :] = p2d_true + 50.0  # shifted by 50 px

    # Drop one camera per point so a proper subset is required
    for n in range(n_points):
        c = rng.integers(n_cams)
        p2d_2[c, n, :, :] = np.nan

    out, picked_vals, _, _ = cgroup.triangulate_possible(p2d_2)

    # For every (cam, point) where any candidate was picked, it must be q=0
    for c in range(n_cams):
        for n in range(n_points):
            if picked_vals[c, n, 0] or picked_vals[c, n, 1]:
                assert picked_vals[c, n, 0], \
                    f"cam {c} point {n}: distractor selected instead of true detection"
                assert not picked_vals[c, n, 1], \
                    f"cam {c} point {n}: distractor should not be selected"

    # 3D accuracy should still be good
    err3d = np.linalg.norm(out - p3d_gt, axis=1)
    assert np.all(err3d < 1e-3), f"Max 3D error: {err3d.max():.2e}"
