"""Keypoint self-calibration rig (motor observatory / Emory style): a mouse skeleton seen by 7
cameras, one of which locks onto a static decoy for most frames. Robust triangulation and
leave-one-camera-out QA must expose the decoy camera; BA on keypoints must recover the rig."""

import numpy as np
import pytest

import synth

pytestmark = pytest.mark.integration

DECOY = 3
FAST_BA = dict(n_iters=2, max_nfev=100, n_samp_full=300, n_samp_iter=150, rng=0)


@pytest.fixture(scope='module')
def scene():
    true = synth.load_rig('motobs_7cam')
    p3d, bones = synth.skeleton(true, 80, model=synth.MOUSE, size=1.0, spread=0.12, seed=1)
    p2d = synth.add_noise(synth.project(true, p3d), 0.7, seed=2)
    p2d_decoy, frames = synth.add_decoy(p2d, cam=DECOY, frac=0.6, cgroup=true, seed=3)
    return true, p3d, p2d, p2d_decoy, frames


def flat(p2d):
    return p2d.reshape(p2d.shape[0], -1, 2)


def rel_err(cg, est, p3d):
    return np.linalg.norm(synth.to_np(est).reshape(p3d.shape) - p3d, axis=-1) / synth.rig_scale(cg)


def test_decoy_camera_corrupts_plain_triangulation(scene):
    true, p3d, p2d, p2d_decoy, frames = scene
    clean = rel_err(true, true.triangulate(flat(p2d)), p3d)
    bad = rel_err(true, true.triangulate(flat(p2d_decoy)), p3d)
    assert np.nanmedian(clean) < 2e-3
    assert np.nanmedian(bad[frames]) > 10 * np.nanmedian(clean)


def test_ransac_survives_decoy(scene):
    true, p3d, p2d, p2d_decoy, frames = scene
    sub = p2d_decoy[:, :30]
    out = true.triangulate_ransac(flat(sub))[0]
    assert np.nanmedian(rel_err(true, out, p3d[:30])) < 3e-3


def test_leave_one_out_flags_decoy_camera(scene):
    """Reprojecting each camera against points triangulated from the others singles out the decoy."""
    true, p3d, p2d, p2d_decoy, frames = scene
    names = true.get_names()
    loo = []
    for i, n in enumerate(names):
        others = true.subset_cameras_names([m for m in names if m != n])
        X = others.triangulate(flat(np.delete(p2d_decoy, i, axis=0)))
        proj = synth.to_np(true.cameras[i].project(X)).reshape(-1, 2)
        loo.append(np.nanmedian(np.linalg.norm(proj - flat(p2d_decoy)[i], axis=-1)))
    loo = np.array(loo)
    assert np.argmax(loo) == DECOY
    assert loo[DECOY] > 3 * np.median(np.delete(loo, DECOY))


def test_robust_leave_one_out_isolates_decoy(scene):
    """quality.loo_reprojection with robust triangulation: healthy cameras score near the noise
    level even while another camera is a decoy."""
    from aniposelib import quality
    true, p3d, p2d, p2d_decoy, frames = scene
    rep = quality.loo_reprojection(true, flat(p2d_decoy))
    med = np.array([rep.per_camera[n]['median_px'] for n in true.get_names()])
    assert np.all(np.delete(med, DECOY) < 3) and med[DECOY] > 50
    assert rep.overall['worst_camera'] == true.get_names()[DECOY]
    # fixed denominator: every eligible observation counts; the decoy frames are disagreements
    agree = rep.per_camera[true.get_names()[DECOY]]['agree@8']
    assert abs(agree - (1 - frames.mean())) < 0.05


def test_ba_on_keypoints_reduces_reprojection(scene):
    true, p3d, p2d, p2d_decoy, frames = scene
    cg = synth.perturb_extrinsics(true, rot_deg=0.5, trans_frac=0.01, seed=4)
    before = cg.average_error(flat(p2d), median=True).item()
    after = cg.bundle_adjust_iter(flat(p2d), only_extrinsics=True, **FAST_BA)
    assert after < before / 5


def test_ba_on_keypoints_recovers_rig(scene):
    true, p3d, p2d, p2d_decoy, frames = scene
    cg = synth.perturb_extrinsics(true, rot_deg=0.5, trans_frac=0.01, seed=4)
    cg.bundle_adjust_iter(flat(p2d), only_extrinsics=True, **FAST_BA)
    assert synth.center_error(cg, true).max() < 0.005


def test_ba_on_keypoints_with_decoy(scene):
    true, p3d, p2d, p2d_decoy, frames = scene
    cg = synth.perturb_extrinsics(true, rot_deg=0.5, trans_frac=0.01, seed=4)
    cg.bundle_adjust_iter(flat(p2d_decoy), only_extrinsics=True, **FAST_BA)
    assert synth.center_error(cg, true).max() < 0.005
