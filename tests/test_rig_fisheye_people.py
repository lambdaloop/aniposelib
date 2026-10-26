"""EvoVision-style gait rig: 5 fisheye GoPros, a walking person, metric scale from a ChArUco
board seen during the trial."""

import numpy as np
import pytest

import synth
from aniposelib.boards import CharucoBoard

pytestmark = pytest.mark.integration


@pytest.fixture(scope='module')
def scene():
    true = synth.load_rig('fisheye_5cam')
    p3d, bones = synth.skeleton(true, 120, model=synth.PERSON, size=3.5, spread=0.15, seed=3)
    p2d = synth.add_noise(synth.project(true, p3d), 0.7, seed=4)
    return true, p3d, p2d, bones


def test_triangulate_person(scene, device):
    true, p3d, p2d, _ = scene
    C, N, J, _ = p2d.shape
    out = synth.to_np(true.to(device).triangulate(p2d.reshape(C, -1, 2))).reshape(N, J, 3)
    true.to('cpu')
    err = np.linalg.norm(out - p3d, axis=-1) / synth.rig_scale(true)
    assert np.nanmedian(err) < 2e-3


def test_optim_points_bone_constraints(scene):
    """optim_points with bone constraints keeps the skeleton consistent under missing views."""
    true, p3d, p2d, bones = scene
    p2d = synth.drop_views(p2d, frac=0.3, seed=5)
    C, N, J, _ = p2d.shape
    init = synth.to_np(true.triangulate(p2d.reshape(C, -1, 2))).reshape(N, J, 3)
    init_f = np.where(np.isfinite(init), init, np.nanmean(init, axis=(0, 1)))
    out = synth.to_np(true.optim_points(p2d, init_f, constraints=bones, n_iters=150))
    err = np.linalg.norm(out - p3d, axis=-1) / synth.rig_scale(true)
    assert np.isfinite(out).all()
    assert np.median(err) < 5e-3


def bone_cv(x, bones):
    """Median coefficient of variation of bone lengths over time."""
    L = np.array([np.linalg.norm(x[:, a] - x[:, b], axis=-1) for a, b in bones])
    return np.median(np.std(L, axis=1) / np.mean(L, axis=1))


@pytest.mark.target
@pytest.mark.xfail(reason='1.0: bone constraints should make bone lengths more consistent than plain '
                          'triangulation within a short optimisation budget', strict=False)
def test_optim_points_tightens_bones(scene):
    true, p3d, p2d, bones = scene
    p2d = synth.drop_views(p2d, frac=0.3, seed=5)
    C, N, J, _ = p2d.shape
    init = synth.to_np(true.triangulate(p2d.reshape(C, -1, 2))).reshape(N, J, 3)
    ok = np.all(np.isfinite(init), axis=(1, 2))
    init_f = np.where(np.isfinite(init), init, np.nanmean(init, axis=(0, 1)))
    out = synth.to_np(true.optim_points(p2d, init_f, constraints=bones, n_iters=150))
    assert bone_cv(out[ok], bones) < 0.8 * bone_cv(init[ok], bones)


@pytest.mark.parametrize('true_scale', [0.37, 2.5])
def test_metric_scale_from_board(true_scale):
    """A rig in arbitrary units is rescaled to metres from triangulated board-corner distances."""
    true = synth.load_rig('fisheye_5cam')
    board = CharucoBoard(10, 7, 0.1, 0.078, marker_bits=4, dict_size=50)
    corners = synth.board_corners_3d(board, 8, true, seed=6)
    p2d = synth.add_noise(synth.project(true, corners), 0.5, seed=7)

    arbitrary = synth.perturb_extrinsics(true, rot_deg=0, trans_frac=0, seed=0)
    arbitrary.set_translations(synth.to_np(true.get_translations()) / true_scale)
    F, K, _ = corners.shape
    tri = synth.to_np(arbitrary.triangulate(p2d.reshape(5, -1, 2))).reshape(F, K, 3)
    objp = board.get_object_points().reshape(-1, 3)
    i, j = np.triu_indices(K, 1)
    ratios = [np.nanmedian(np.linalg.norm(objp[i] - objp[j], axis=1) /
                           np.linalg.norm(tri[f, i] - tri[f, j], axis=1)) for f in range(F)]
    assert abs(np.nanmedian(ratios) / true_scale - 1) < 0.005

    # the same through aniposelib.scale.from_board (per-frame medians, MAD, board shape)
    from aniposelib import scale
    res = scale.from_board(arbitrary, p2d, board)
    assert res.accepted and not res.flags
    assert abs(res.scale / true_scale - 1) < 0.005
    assert res.stats['board_resid'] < 0.005  # board units (m): the rig shape is right
    res.apply(arbitrary)
    assert arbitrary.metadata['scale_source'] == 'charuco'
    assert synth.center_error(arbitrary, true, with_scale=False).max() < 0.005


def test_optim_points_does_not_extrapolate(scene):
    true, p3d, p2d, bones = scene
    p2d = p2d.copy()
    p2d[:, :20] = np.nan  # nobody sees the first 20 frames
    C, N, J, _ = p2d.shape
    init = synth.to_np(true.triangulate(p2d.reshape(C, -1, 2))).reshape(N, J, 3)
    init_f = np.where(np.isfinite(init), init, np.nanmean(init, axis=(0, 1)))
    out = synth.to_np(true.optim_points(p2d, init_f, constraints=bones, n_iters=50))
    assert np.all(np.isnan(out[:20]))
