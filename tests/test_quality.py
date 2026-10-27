"""aniposelib.quality on synthetic rigs: LOO, reports, reuse assessment, comparisons."""

import json

import numpy as np
import pytest
import torch

import synth
from aniposelib import quality, scale
from aniposelib.boards import CharucoBoard

pytestmark = pytest.mark.integration

DECOY = 3


@pytest.fixture(scope='module')
def scene():
    true = synth.load_rig('motobs_7cam')
    p3d, bones = synth.skeleton(true, 60, model=synth.MOUSE, size=1.0, spread=0.12, seed=1)
    p2d = synth.add_noise(synth.project(true, p3d), 0.7, seed=2)
    p2d_decoy, frames = synth.add_decoy(p2d, cam=DECOY, frac=0.6, cgroup=true, seed=3)
    return true, p3d, p2d, p2d_decoy, frames, bones


def moved(cg, cam, deg, seed=0):
    out = cg.copy()
    rng = np.random.default_rng(seed)
    ax = rng.normal(size=3)
    ax /= np.linalg.norm(ax)
    with torch.no_grad():
        out.cameras[cam].rvec += torch.as_tensor(ax * np.deg2rad(deg), dtype=torch.float64)
    return out


def test_report_flags_decoy(scene, tmp_path):
    true, p3d, p2d, p2d_decoy, frames, _ = scene
    names = true.get_names()
    C, N, J, _ = p2d.shape
    fit_mask = np.zeros((C, N * J), bool)
    fit_mask[:, : N * J // 2] = True
    rep = quality.report(true, p2d_decoy, fit_mask=fit_mask, groups=np.arange(J))
    assert not rep.passes()
    assert any(names[DECOY] in f for f in rep.failures())
    assert rep.per_camera[names[DECOY]]['flags'] == ['loo_disagrees']
    for i, n in enumerate(names):
        p = rep.per_camera[n]
        assert p['n_obs'] == N * J and p['coverage'] == 1.0
        if i != DECOY:
            assert p['median_px'] < 2 and p['loo_median_px'] < 3 and p['loo_agree@8'] > 0.95
            assert p['fit_median_px'] < 2 and p['heldout_median_px'] < 2
    assert rep.per_camera[names[DECOY]]['inlier_rate'] < 0.6
    assert len(rep.per_group) == J
    d = json.loads(rep.to_json(tmp_path / 'r.json'))
    assert d['overall']['loo_worst_camera'] == names[DECOY]
    assert 'loo_median_px' in rep.table()
    # without the decoy the rig passes
    assert quality.report(true, p2d, loo=True).passes(max_median_px=2, max_loo_px=3)


def test_loo_fixed_denominator(scene):
    """Eligibility depends only on the input: a miscalibrated rig is scored on the same
    observations (misses are inf and count against agreement)."""
    true, p3d, p2d, *_ = scene
    p = synth.drop_views(p2d, frac=0.4, seed=4).reshape(7, -1, 2)
    a = quality.loo_reprojection(true, p)
    b = quality.loo_reprojection(moved(true, 2, 10), p)
    assert np.array_equal(a.eligible, b.eligible)
    fin = np.isfinite(p).all(-1)
    others = fin.sum(0)[None] - fin
    assert np.array_equal(a.eligible, fin & (others >= 3))
    assert np.all(np.isnan(a.errors[~a.eligible]))
    n = true.get_names()[2]
    assert b.per_camera[n]['agree@8'] < 0.2 < 0.9 < a.per_camera[n]['agree@8']
    assert b.per_camera[n]['n_eligible'] == a.per_camera[n]['n_eligible']
    # non-robust variant runs and agrees on a clean rig
    c = quality.loo_reprojection(true, p, robust=False)
    assert np.nanmax(c.medians()) < 3


def test_compare_recovers_similarity(scene):
    import cv2
    true, p3d, p2d, *_ = scene
    cg = true.copy()
    R = cv2.Rodrigues(np.array([0.1, -0.2, 0.3]))[0]
    scale.apply_similarity(cg, 2.5, R, [1.0, 2.0, 3.0])
    c = quality.compare(true, cg)
    assert abs(c.scale - 2.5) < 1e-9 and np.abs(c.R - R).max() < 1e-9
    assert c.overall['center_residual_max_frac'] < 1e-9 and c.overall['rotation_delta_max_deg'] < 1e-6
    # one camera moved: it is the worst, and the trimmed gauge excludes it
    with torch.no_grad():
        cg.cameras[4].tvec += 0.2 * torch.as_tensor(synth.to_np(cg.cameras[4].tvec))
    c = quality.compare(true, cg, p2ds=p2d)
    assert abs(c.scale - 2.5) < 1e-6
    assert c.overall['worst_camera'] == true.get_names()[4]
    assert c.gauge_excluded == [true.get_names()[4]]
    assert c.errors['per_camera'][true.get_names()[4]]['delta'] > 5
    json.dumps(c.to_dict())


def test_coverage_and_ray_angles(scene):
    true, p3d, p2d, *_ = scene
    p = synth.drop_views(p2d, frac=0.5, seed=5)
    cov = quality.coverage(p, min_cams=3, cgroup=true)
    fin = np.isfinite(p.reshape(7, -1, 2)).all(-1)
    n0 = true.get_names()[0]
    assert cov['per_camera'][n0]['n_obs'] == fin[0].sum()
    assert cov['per_camera'][n0]['n_multiview'] == (fin[0] & (fin.sum(0) >= 3)).sum()
    assert 0 < cov['per_camera'][n0]['area'] <= 1
    assert sum(cov['views_hist']) == fin.shape[1]
    ra = quality.ray_angles(true, p3d)
    assert ra['pair_median_deg'].shape == (7, 7) and ra['median_max_angle_deg'] > 10
    # a camera next to another one makes a grazing pair
    cg = true.copy()
    cg.cameras[1].set_rotation(synth.to_np(cg.cameras[0].rvec))
    cg.cameras[1].set_translation(synth.to_np(cg.cameras[0].tvec) + [1.0, 0, 0])
    names = cg.get_names()
    pairs = [(a, b) for a, b, _ in quality.ray_angles(cg, p3d)['grazing_pairs']]
    assert (names[0], names[1]) in pairs


def test_sanity_and_lens_validity():
    cg = synth.load_rig('motobs_7cam')
    assert quality.sanity(cg)['ok']
    bad = cg.copy()
    cam = bad.cameras[0]
    d = synth.to_np(cam.dist).copy()
    d[0] = -0.9  # radial map folds over inside the image
    cam.set_distortions(d)
    lv = quality.lens_validity(cam)
    assert not lv['valid']
    s = quality.sanity(bad, k_max=0.5)
    assert not s['ok'] and not s[cam.get_name()]['ok'] and s[cg.get_names()[1]]['ok']
    assert quality.lens_validity(synth.load_rig('fisheye_5cam').cameras[0])['valid']


def test_board_shape_residual():
    true = synth.load_rig('fisheye_5cam')
    board = CharucoBoard(10, 7, 0.1, 0.078, marker_bits=4, dict_size=50)
    corners = synth.board_corners_3d(board, 6, true, seed=6)
    p2d = synth.add_noise(synth.project(true, corners), 0.3, seed=7)
    good = quality.board_shape_residual(true, p2d, board)
    warped = quality.board_shape_residual(synth.perturb_extrinsics(true, rot_deg=3.0, trans_frac=0.05, seed=1),
                                          p2d, board)
    assert good['n_frames'] == 6 and good['median'] < 0.003
    assert warped['median'] > 2 * good['median']


