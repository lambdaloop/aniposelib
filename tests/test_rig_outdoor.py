"""Boardless outdoor arena (12 thermal cameras, one tracked blob per frame): triangulation of a
trajectory with false-positive views, and bundle adjustment of a perturbed rig from the animal's
own trajectory."""

import numpy as np
import pytest

import synth

pytestmark = pytest.mark.integration

FAST_BA = dict(n_iters=2, max_nfev=100, n_samp_full=300, n_samp_iter=150)


@pytest.fixture(scope='module')
def scene():
    true = synth.load_rig('outdoor_12cam')
    p3d = synth.trajectory(true, 1500, seed=2)
    p2d = synth.add_noise(synth.project(true, p3d), 0.5, seed=3)
    return true, p3d, p2d


def rel_err(cg, p3d_est, p3d):
    return np.linalg.norm(synth.to_np(p3d_est) - p3d, axis=-1) / synth.rig_scale(cg)


def test_triangulate_clean(scene, device):
    true, p3d, p2d = scene
    out = true.to(device).triangulate(p2d)
    true.to('cpu')
    assert np.nanmedian(rel_err(true, out, p3d)) < 1e-3


def test_ransac_rejects_false_detections(scene):
    true, p3d, p2d = scene
    p2d_bad = synth.add_outliers(p2d[:, :200], frac=0.1, cgroup=true, seed=5)
    plain = rel_err(true, true.triangulate(p2d_bad), p3d[:200])
    robust = rel_err(true, true.triangulate_ransac(p2d_bad)[0], p3d[:200])
    assert np.nanmedian(robust) < 3e-3
    assert np.nanmedian(robust) < np.nanmedian(plain) / 5


def test_ba_recovers_perturbed_rig_from_trajectory(scene):
    true, p3d, p2d = scene
    cg = synth.perturb_extrinsics(true, rot_deg=0.5, trans_frac=0.01, seed=4)
    before = synth.center_error(cg, true).max()
    err = cg.bundle_adjust_iter(p2d, only_extrinsics=True, **FAST_BA)
    assert err < 1.0
    assert synth.center_error(cg, true).max() < before / 4


@pytest.mark.target
@pytest.mark.xfail(reason='1.0: robust BA (robust inner triangulation + resection) should survive a '
                          'camera stuck on a sync LED and 5% false detections', strict=False)
def test_ba_with_outliers_and_decoy_camera(scene):
    true, p3d, p2d = scene
    p2d = synth.add_outliers(p2d, frac=0.05, cgroup=true, seed=6)
    p2d, _ = synth.add_decoy(p2d, cam=5, frac=0.6, cgroup=true, seed=7)
    cg = synth.perturb_extrinsics(true, rot_deg=0.5, trans_frac=0.01, seed=4)
    cg.bundle_adjust_iter(p2d, only_extrinsics=True, **FAST_BA)
    assert synth.center_error(cg, true).max() < 0.005
