"""Calibration workflows (plan 05 W1-W4, W1b) end to end on the synthetic use-case rigs.

Kept fast (the whole suite must stay <= 60 s on CPU): small scenes, and the lens-model ladder
(the slow part of W1) runs only in the Allen test; other W1 tests start from intrinsics priors.
"""

import copy
import json
import os

import numpy as np
import pytest
import torch

import synth
from aniposelib import geometry as G, scale as S, stations as ST
from aniposelib.boards import CharucoBoard, PointObject
from aniposelib.workflows import (calibrate_from_board, calibrate_from_points, estimate_scale,
                                  evaluate_calibration, BoardCalibration, PointCalibration)

pytestmark = pytest.mark.integration

BOARD = dict(squaresX=6, squaresY=5, square_length=24, marker_length=18, marker_bits=4, dict_size=50)


def board(scale=1.0):
    kw = dict(BOARD, square_length=24 * scale, marker_length=18 * scale)
    return CharucoBoard(**kw, units='mm')


def blank_like(true):
    """Sizes only (no intrinsics, identity extrinsics)."""
    from aniposelib.cameras import CameraGroup
    cg = CameraGroup.from_names(true.get_names())
    for c, ct in zip(cg.cameras, true.cameras):
        c.set_size(ct.get_size())
    return cg


def no_extrinsics(true):
    """True intrinsics, identity extrinsics."""
    cg = true.copy()
    for c in cg.cameras:
        c.set_rotation(np.zeros(3))
        c.set_translation(np.zeros(3))
    return cg


def focal_err(cg, true):
    f = np.array([synth.to_np(c.get_camera_matrix())[0, 0] for c in cg.cameras])
    ft = np.array([synth.to_np(c.get_camera_matrix())[0, 0] for c in true.cameras])
    return np.abs(f / ft - 1).max()


def scaled_rig(true, s):
    cg = true.copy()
    S.rescale(cg, s)
    return cg


# ------------------------------------------------------------------ W1 boards

@pytest.fixture(scope='module')
def allen():
    true = synth.load_rig('allen_7cam')
    return true, synth.board_rows(true, board(), n_frames=40, seed=1)


def test_w1_allen_board_default(allen, tmp_path):
    """Blank cameras (sizes only) -> ladder intrinsics -> board graph -> LM BA: verdict ok,
    metric rig, units recorded, unpacks as (cgroup, result), out= writes TOML + report."""
    true, rows = allen
    out = tmp_path / 'calib.toml'
    res = calibrate_from_board(rows, board(), init=blank_like(true), out=out)
    cg, rep = res
    assert isinstance(rep, BoardCalibration) and rep.verdict == 'ok', res.summary()
    assert synth.center_error(cg, true).max() < 0.005
    assert focal_err(cg, true) < 0.005
    assert abs(synth.rig_scale(cg) / synth.rig_scale(true) - 1) < 0.005     # board fixes the scale
    assert cg.metadata['units'] == 'mm'
    assert all(m['source'] == 'ladder' for m in res.lens_models.values())
    assert res.settings['noise_floor_px'] >= 0.5 and 'f_scale_pass2_px' in res.settings
    assert os.path.exists(out) and os.path.exists(tmp_path / 'calib.report.json')
    d = json.loads(res.to_json())
    assert d['verdict'] == 'ok' and d['cgroup']['cameras'] == true.get_names()
    # the input rows are not modified
    assert 'rvec' not in rows[0][0]


@pytest.mark.parametrize('s', [1e-3, 1.0, 1e3])
def test_w1_no_tuning_unit_scales(allen, s):
    """Same call, default options, world units x 1e-3 / 1 / 1e3: same accuracy and verdict."""
    true, _ = allen
    rig = scaled_rig(true, s)
    rows = synth.board_rows(rig, board(s), n_frames=24, seed=2)
    res = calibrate_from_board(rows, board(s), init=no_extrinsics(rig), intrinsics=rig)
    assert res.verdict == 'ok', res.summary()
    assert synth.center_error(res.cgroup, rig).max() < 0.005
    assert abs(synth.rig_scale(res.cgroup) / synth.rig_scale(rig) - 1) < 0.005


def _split_rows(rows, A):
    """Cameras in A only see even frames, the others odd ones: two board-graph components."""
    return [[r for r in rr if r['framenum'][1] % 2 == (0 if c in A else 1)] for c, rr in enumerate(rows)]


def test_w1_disconnected_fails_then_bridge(allen):
    true, _ = allen
    rows = _split_rows(synth.board_rows(true, board(), n_frames=40, seed=1), [0, 1, 2])
    res = calibrate_from_board(rows, board(), init=no_extrinsics(true), intrinsics=true)
    assert res.verdict == 'fail' and res.cgroup is None
    assert any('disconnected' in r and 'bridge_points' in r for r in res.reasons)
    assert sorted(len(c) for c in res.components) == [3, 4]
    for g in res.component_groups:      # each component is calibrated in its own gauge
        assert synth.center_error(g, true.subset_cameras_names(g.get_names())).max() < 0.005
    # 2D tracks seen across the components bridge them (6-DoF per component, scale kept)
    p3 = synth.random_points(true, 200, min_views=3, seed=3)
    bp = synth.add_noise(synth.project(true, p3), 0.5, seed=4)
    res = calibrate_from_board(rows, board(), init=no_extrinsics(true), intrinsics=true, bridge_points=bp)
    assert res.verdict == 'ok', res.summary()
    assert len(res.bridge['components']) == 1 and not res.bridge['unbridged']
    assert synth.center_error(res.cgroup, true).max() < 0.005
    assert abs(synth.rig_scale(res.cgroup) / synth.rig_scale(true) - 1) < 0.005


def test_w1_bad_inputs():
    true = synth.load_rig('allen_7cam')
    rows = synth.board_rows(true, board(), n_frames=20, seed=1)
    with pytest.raises(ValueError):          # no image sizes, nothing to fit intrinsics on
        calibrate_from_board(rows, board())
    with pytest.raises(ValueError):          # 'fixed' needs init
        calibrate_from_board(rows, board(), intrinsics='fixed')
    rows[3] = []                             # a camera without board views: fail, no exception
    res = calibrate_from_board(rows, board(), init=no_extrinsics(true), intrinsics=true)
    assert res.verdict == 'fail' and res.report['uncovered'] == [true.get_names()[3]]


# ------------------------------------------------------------------ W1b PointObject

def _wand(true, n_frames=30, K=12, seed=0, noise=0.5, spread=0.06):
    import cv2
    rng = np.random.default_rng(seed)
    s, cen = synth.rig_scale(true), synth.scene_center(true)
    P = rng.uniform(-1, 1, (K, 3)) * spread * s
    ann = np.full((len(true.cameras), n_frames, K, 2), np.nan)
    for f in range(n_frames):
        R = cv2.Rodrigues(rng.normal(size=3))[0]
        uv = synth.project(true, P @ R.T + cen + rng.uniform(-1, 1, 3) * 0.1 * s)
        ann[:, f] = uv + rng.normal(scale=noise, size=uv.shape)
    return P, ann


def test_w1b_moving_wand_dlt_intrinsics():
    """A non-planar 12-point wand waved around the Allen rig, blank cameras: intrinsics from a
    per-view DLT, then the board BA path with the wand as a rigid body per frame."""
    true = synth.load_rig('allen_7cam')
    P, ann = _wand(true)
    obj = PointObject(P, units='mm')
    assert not obj.planar
    res = calibrate_from_board(ann, obj, init=blank_like(true))
    assert res.verdict == 'ok', res.summary()
    assert all(m['source'] == 'dlt' for m in res.lens_models.values())
    assert synth.center_error(res.cgroup, true).max() < 0.01
    assert focal_err(res.cgroup, true) < 0.01
    assert abs(synth.rig_scale(res.cgroup) / synth.rig_scale(true) - 1) < 0.01
    # W3 / W4 accept the object like a board
    sc = estimate_scale(scaled_rig(res.cgroup, 0.5), object=obj, object_points=ann)
    assert sc.accepted and sc.source == 'object' and abs(sc.scale - 2) < 0.02
    ev = evaluate_calibration(res.cgroup, ann.reshape(len(true.cameras), -1, 2)[:, :120],
                              object=obj, object_points=ann)
    assert ev.report['board_shape']['median'] is not None and ev.verdict != 'fail', ev.summary()


def test_w1b_static_landmarks_world_frame():
    """8 surveyed landmarks seen by the outdoor rig, given intrinsics: every camera is placed
    by PnP in the landmarks' frame, which is the world (absolute, metric, no W3 step)."""
    true = synth.load_rig('outdoor_12cam')
    X = synth.random_points(true, 8, spread=0.2, min_views=12, seed=0)
    ann = synth.add_noise(synth.project(true, X)[:, None], 0.5, seed=1)
    obj = PointObject(X, names=['p{}'.format(i) for i in range(8)], static=True, units='m')
    res = calibrate_from_board(ann, obj, init=no_extrinsics(true), intrinsics='fixed')
    assert res.verdict == 'ok', res.summary()
    d = np.linalg.norm(synth.camera_centers(res.cgroup) - synth.camera_centers(true), axis=1)
    assert np.median(d) < 0.01 * synth.rig_scale(true) and d.max() < 0.03 * synth.rig_scale(true)
    assert res.cgroup.metadata['units'] == 'm'
    # dict annotations give the same array
    per_cam = {n: {0: {'p{}'.format(k): ann[c, 0, k] for k in range(8)}} for c, n in enumerate(true.get_names())}
    np.testing.assert_array_equal(obj.annotations_from_dicts(per_cam), ann)


def test_w1b_bad_inputs():
    true = synth.load_rig('allen_7cam')
    P, ann = _wand(true, n_frames=12)
    obj = PointObject(P)
    with pytest.raises(NotImplementedError):
        obj.detect_image(np.zeros((10, 10), np.uint8))
    with pytest.raises(ValueError):
        PointObject(P[:3])
    with pytest.raises(ValueError):
        calibrate_from_board([['a.mp4']] * 7, obj)
    # a camera with no annotations: uncovered and no intrinsics -> fail with reasons
    a = ann.copy()
    a[2] = np.nan
    res = calibrate_from_board(a, obj, init=blank_like(true))
    assert res.verdict == 'fail'
    assert true.get_names()[2] in res.report['uncovered'] + res.report['missing_intrinsics']
    # too few points per view (3): no pose anywhere
    a = ann.copy()
    a[:, :, 3:] = np.nan
    res = calibrate_from_board(a, obj, init=no_extrinsics(true), intrinsics='fixed')
    assert res.verdict == 'fail' and len(res.report['uncovered']) == 7
    # coplanar points without given intrinsics on a non-planar-DLT path: planar objects use
    # the board ladder; an object with too few points per view for DLT and no init fails
    a = ann.copy()
    a[:, :, 5:] = np.nan
    res = calibrate_from_board(a, obj, init=blank_like(true))
    assert res.verdict == 'fail' and len(res.report['missing_intrinsics']) == 7
    assert any('intrinsics' in r for r in res.reasons)


# ------------------------------------------------------------------ W2 points

def test_w2_outdoor_trajectory_false_detections():
    """Boardless arena: one blob per frame with 5 % false detections; SfM init from the given
    intrinsics, noise-normalised robust BA, held-out gate keeps the (correct) intrinsics."""
    true = synth.load_rig('outdoor_12cam')
    p3 = synth.trajectory(true, 400, seed=2)
    p2 = synth.add_outliers(synth.add_noise(synth.project(true, p3), 0.5, seed=3), frac=0.05, cgroup=true, seed=6)
    cg, res = calibrate_from_points(p2, no_extrinsics(true))
    assert isinstance(res, PointCalibration) and res.verdict == 'ok', res.summary()
    assert res.selected_init == 'sfm' and res.intrinsics['decision'] == 'kept'
    assert synth.center_error(cg, true).max() < 0.01
    st = res.settings
    assert st['noise_unit'] >= 1 and st['fine_px'][-1][-1] == pytest.approx(5 * st['noise_unit'])
    json.dumps(res.to_dict())


@pytest.fixture(scope='module')
def motobs():
    true = synth.load_rig('motobs_7cam')
    p3, bones = synth.skeleton(true, 80, model=synth.MOUSE, size=1.0, spread=0.12, seed=1)
    p2 = synth.add_noise(synth.project(true, p3), 0.7, seed=2)
    return true, p3, bones, p2


def test_w2_keypoints_with_decoy(motobs):
    """One camera locks onto a decoy for 60 % of frames: the calibration is still right, and
    the verdict warns about that camera (few inliers) instead of failing on its LOO."""
    true, _, _, p2 = motobs
    p2d, _ = synth.add_decoy(p2, cam=3, frac=0.6, cgroup=true, seed=3)
    cg, res = calibrate_from_points(p2d, no_extrinsics(true))
    assert synth.center_error(cg, true).max() < 0.005
    assert res.verdict == 'warn', res.summary()
    assert any('min_inlier_frac' in r and true.get_names()[3] in r for r in res.reasons)


@pytest.mark.parametrize('s', [1e-3, 1.0, 1e3])
def test_w2_no_tuning_unit_scales(motobs, s):
    """Reference init (perturbed lab rig) at world units x 1e-3 / 1 / 1e3: same verdict and
    accuracy with default options; scale from bone lengths in the same units."""
    true, p3, bones, p2 = motobs
    rig = scaled_rig(true, s)
    ref = synth.perturb_extrinsics(rig, 1.0, 0.02, seed=4)
    L = [s * np.median(np.linalg.norm(p3[:, a] - p3[:, b], axis=-1)) for a, b in bones]
    cg, res = calibrate_from_points(p2, ref, init='reference', refine_intrinsics=False,
                                    scale=dict(lengths=dict(points=p2, pairs=bones, ref=L)))
    assert res.verdict == 'ok', res.summary()
    assert synth.center_error(cg, rig).max() < 0.005
    assert abs(synth.rig_scale(cg) / synth.rig_scale(rig) - 1) < 0.01
    assert res.scale.accepted and res.scale.source == 'lengths'


def test_w2_bad_inputs(motobs):
    true, _, _, p2 = motobs
    with pytest.raises(ValueError):
        calibrate_from_points(p2[:3], no_extrinsics(true))
    with pytest.raises(ValueError):
        calibrate_from_points(p2, no_extrinsics(true), groups=dict(bogus=np.arange(80)))
    q = p2.copy()
    q[4] = np.nan                              # a camera with no points: not placed -> fail
    cg, res = calibrate_from_points(q, no_extrinsics(true), refine_intrinsics=False)
    assert res.verdict == 'fail' and res.report['unplaced'] == [true.get_names()[4]]
    res = calibrate_from_points(p2, no_extrinsics(true), init='vggt')   # no frames: no init
    assert res.verdict == 'fail' and res.cgroup is None


# ------------------------------------------------------------------ W3 scale

def test_w3_board_and_bones(allen, motobs):
    true, rows = allen
    res = estimate_scale(scaled_rig(true, 0.37), board=board(), board_points=rows[:])
    assert res.accepted and res.verdict == 'ok' and abs(res.scale * 0.37 - 1) < 1e-3
    assert res.sigma is not None and res.sigma < 1e-2 * res.scale
    assert res.cgroup.metadata['units'] == 'mm'
    assert abs(synth.rig_scale(res.cgroup) / synth.rig_scale(true) - 1) < 1e-3
    true, p3, bones, p2 = motobs
    L = [np.median(np.linalg.norm(p3[:, a] - p3[:, b], axis=-1)) for a, b in bones]
    cg = scaled_rig(true, 2.0)
    res = estimate_scale(cg, lengths=dict(points=p2, pairs=bones, ref=L))
    assert res.accepted and abs(res.scale - 0.5) < 0.005 and 'tau_px' in res.settings
    assert synth.rig_scale(cg) == pytest.approx(2 * synth.rig_scale(true))     # input untouched
    bad = estimate_scale(cg, lengths=dict(points=p2[:, :3], pairs=bones, ref=L))
    assert bad.verdict == 'fail' and not bad.accepted and bad.cgroup is None
    with pytest.raises(ValueError):
        estimate_scale(cg)
    ref = estimate_scale(cg, reference=true)
    assert ref.accepted and 'align' in ref.details
    assert np.abs(synth.camera_centers(ref.cgroup) - synth.camera_centers(true)).max() < 1e-6 * synth.rig_scale(true) + 1e-9


# ------------------------------------------------------------------ W4 evaluation

def _bumped(cg, cam, rot_deg):
    cgb = copy.deepcopy(cg)
    c = cgb.cameras[cam]
    dR = G.rodrigues(torch.tensor(np.radians(rot_deg), dtype=torch.float64))
    with torch.no_grad():
        R = G.rodrigues(c.rvec.reshape(3).double())
        c.rvec.copy_(torch.as_tensor(ST._rotvec((dR @ R).numpy())).reshape(c.rvec.shape))
        c.tvec.copy_((dR @ c.tvec.reshape(3).double()).reshape(c.tvec.shape))
    return cgb


def test_w4_moved_camera_and_decoy():
    rig = synth.load_rig('motobs_7cam')
    fn = synth.smooth_tracks(rig, n_points=6, amp=0.1, seed=1)
    t = np.arange(0, 24, 1 / 15)
    X = np.stack([fn(p, t) for p in range(6)], 1)
    moved = (t >= 13.2)[None, :, None, None]
    p2 = synth.add_noise(np.where(moved, synth.project(_bumped(rig, 2, [1.2, -1.0, 0.8]), X),
                                  synth.project(rig, X)), 0.7, seed=2)
    clean = evaluate_calibration(rig, p2[:, :150])           # before the bump: ok
    assert clean.verdict == 'ok', clean.summary()
    p2d, _ = synth.add_decoy(p2, cam=5, frac=0.6, cgroup=rig, seed=3)
    res = evaluate_calibration(rig, p2d, times=t, reference=synth.perturb_extrinsics(rig, 0.5, 0.03, seed=3))
    names = rig.get_names()
    assert res.verdict == 'fail'
    assert res.moves.moved == [names[2]] and res.per_camera[names[2]]['moved']
    assert any('loo_worst_px' in r and names[5] in r for r in res.reasons)
    assert res.comparison is not None and res.report['rules'][5]['level'] != 'ok'
    assert res.settings['loo_fail_px'] == pytest.approx(10 * res.settings['noise_floor_px'])
    json.loads(res.to_json())
    with pytest.raises(ValueError):
        evaluate_calibration(rig, p2[:3])
