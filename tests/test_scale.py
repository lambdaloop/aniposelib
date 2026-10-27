"""aniposelib.scale: gauge operations, metric scale from boards / lengths / references, world frame."""

import numpy as np
import pytest
import torch

import synth
from aniposelib import scale
from aniposelib.boards import CharucoBoard
from aniposelib.cameras import CameraGroup

pytestmark = pytest.mark.integration


@pytest.fixture(scope='module')
def rig():
    return synth.load_rig('motobs_7cam')


def rotation(rv):
    import cv2
    return cv2.Rodrigues(np.asarray(rv, dtype='float64'))[0]


def test_rescale_and_provenance(rig):
    cg = rig.copy()
    p3d = synth.random_points(rig, 50, seed=0)
    out = scale.rescale(cg, 0.001, units='m', source='manual', quality={'x': 1.0, 'bad': float('nan')})
    assert out is cg
    assert np.allclose(synth.camera_centers(cg), 0.001 * synth.camera_centers(rig))
    assert np.allclose(synth.project(cg, 0.001 * p3d), synth.project(rig, p3d), equal_nan=True)
    md = cg.metadata
    assert md['units'] == 'm' and md['scale_source'] == 'manual' and md['scale_value'] == 0.001
    assert md['scale_quality'] == {'x': 1.0}  # non-finite values are dropped (TOML)
    back = CameraGroup.loads(cg.dumps())
    assert back.metadata['units'] == 'm'
    with pytest.raises(ValueError):
        scale.rescale(cg, -1)


def test_apply_similarity_anchor_align(rig):
    cg = rig.copy()
    R = rotation([0.2, 0.1, -0.4])
    t = np.array([10.0, -20.0, 5.0])
    scale.apply_similarity(cg, 3.0, R, t)
    p3d = synth.random_points(rig, 50, seed=1)
    assert np.allclose(synth.project(cg, 3.0 * p3d @ R.T + t), synth.project(rig, p3d), atol=1e-8, equal_nan=True)
    scale.anchor_to(cg, rig.get_names()[2])
    assert np.allclose(synth.to_np(cg.cameras[2].rvec), 0, atol=1e-9)
    assert np.allclose(synth.to_np(cg.cameras[2].tvec), 0, atol=1e-6)
    res = scale.align_to(cg, rig)
    assert abs(res.scale - 1 / 3.0) < 1e-9 and not res.gauge_excluded
    assert synth.center_error(cg, rig, with_scale=False).max() < 1e-9
    assert res.matrix.shape == (4, 4)


def test_align_to_trims_outlier_camera(rig):
    cg = rig.copy()
    scale.apply_similarity(cg, 2.0, rotation([0, 0, 0.5]), [1.0, 2.0, 3.0])
    with torch.no_grad():
        cg.cameras[4].tvec += torch.as_tensor([300.0, 0, 0], dtype=torch.float64)  # a moved camera
    target = synth.camera_centers(rig)
    res = scale.align_to(cg, target)
    assert res.gauge_excluded == [rig.get_names()[4]]
    assert abs(res.scale - 0.5) < 1e-9
    err = np.linalg.norm(synth.camera_centers(cg) - target, axis=1) / synth.rig_scale(rig)
    assert np.delete(err, 4).max() < 1e-9 and err[4] > 0.05
    # the plain fit is dragged by the outlier
    cg2 = rig.copy()
    with torch.no_grad():
        cg2.cameras[4].tvec += torch.as_tensor([300.0, 0, 0], dtype=torch.float64)
    plain = scale.align_to(cg2, target, trim=None)
    assert abs(plain.scale - 1) > 1e-3


def test_from_lengths_with_bones(rig):
    p3d, bones = synth.skeleton(rig, 40, model=synth.MOUSE, seed=1)
    p2d = synth.add_noise(synth.project(rig, p3d), 0.7, seed=2)
    p2d = synth.add_outliers(p2d, frac=0.05, cgroup=rig, seed=3)
    ref = [np.median(np.linalg.norm(p3d[:, a] - p3d[:, b], axis=-1)) * 1000 for a, b in bones]
    joints = synth.MOUSE['joints']
    named = [(joints[a], joints[b]) for a, b in bones]
    res = scale.from_lengths(rig, p2d, named, ref, joints=joints, units='um')
    assert res.accepted and not res.flags
    assert abs(res.scale / 1000 - 1) < 0.01
    s, rep = res
    cg = rig.copy()
    rep.apply(cg)
    assert cg.metadata['scale_source'] == 'lengths' and cg.metadata['units'] == 'um'
    assert abs(cg.metadata['scale_value'] - s) < 1e-12
    # too few samples -> not accepted, apply refuses
    few = scale.from_lengths(rig, p2d[:, :3], bones, ref, min_samples=10)
    assert not few.accepted
    with pytest.raises(ValueError):
        few.apply(rig.copy())


def test_from_reference(rig):
    cg = rig.copy()
    scale.apply_similarity(cg, 0.25, rotation([0.3, 0, 0]), [0, 1.0, 0])
    res = scale.from_reference(cg, rig)
    assert res.accepted and abs(res.scale - 4.0) < 1e-9 and not res.flags
    with torch.no_grad():
        cg.cameras[1].tvec += torch.as_tensor([0, 0, 50.0], dtype=torch.float64)
    res = scale.from_reference(cg, rig)
    assert 'scale_reference_mismatch' in res.flags
    assert abs(res.scale - 4.0) < 1e-6  # trimmed: the moved camera does not set the scale


def test_from_board_rows_and_rejection():
    true = synth.load_rig('fisheye_5cam')
    board = CharucoBoard(10, 7, 0.1, 0.078, marker_bits=4, dict_size=50)
    rows = synth.board_rows(true, board, n_frames=20, seed=1)
    cg = true.copy()
    scale.rescale(cg, 1 / 0.37)
    res = scale.from_board(cg, rows, board, min_pair_dist=0.15)
    assert res.accepted and abs(res.scale * (1 / 0.37) - 1) < 0.005
    assert res.stats['n_frames'] >= 5 and res.stats['board_resid'] < 0.005
    # corner ids scrambled (per camera) in half of the frames: the sightings disagree
    corners, keys = scale.board_points(cg, rows, board)
    rng = np.random.default_rng(0)
    bad = corners.copy()
    for f in range(1, bad.shape[1], 2):
        for c in range(bad.shape[0]):
            bad[c, f] = bad[c, f][rng.permutation(bad.shape[2])]
    res = scale.from_board(cg, bad, board)
    assert not res.accepted and res.scale is not None
    assert scale.from_board(cg, corners[:, :2], board).reason.startswith('only 2')


def test_align_world(rig):
    p = synth.random_points(rig, 200, seed=3)
    # make a tilted floor: project points onto a plane through the scene
    n = np.array([0.3, -0.2, 1.0])
    n /= np.linalg.norm(n)
    floor = p - ((p - p.mean(0)) @ n)[:, None] * n
    cg = rig.copy()
    res = scale.align_world(cg, ground_points=floor, up='z')
    # in the new frame, floor points have z = 0 (origin at their centroid)
    f2 = floor @ res.R.T + res.t
    assert np.abs(f2[:, 2]).max() < 1e-6 * synth.rig_scale(rig)
    assert np.allclose(synth.project(cg, f2), synth.project(rig, floor), atol=1e-6, equal_nan=True)
    # the cameras are above the floor
    assert synth.camera_centers(cg)[:, 2].mean() > 0


def _runaway(rig, cam=-1, factor=18.0):
    """Copy of ``rig`` with one camera centre moved ``factor`` x the rig radius from the rig centre."""
    bad = rig.copy()
    X = synth.camera_centers(rig)
    c0 = np.median(X, 0)
    r = np.median(np.linalg.norm(X - c0, axis=1))
    c = bad.cameras[cam]
    R = rotation(synth.to_np(c.rvec).reshape(3))
    with torch.no_grad():
        c.set_translation(torch.as_tensor(-R @ (c0 + factor * r * np.array([1.0, 0, 0])), dtype=torch.float64))
    return bad


def test_fit_similarity_runaway_point():
    rig = synth.load_rig('outdoor_12cam')
    src = synth.camera_centers(_runaway(rig))
    dst = synth.camera_centers(rig)
    dst = 0.5 * dst @ rotation([0.3, -0.2, 0.1]).T + [10.0, -5.0, 2.0]
    s, R, t, r, use = scale.fit_similarity(src, dst)
    assert not use[-1] and use[:-1].all()
    assert r[:-1].max() < 1e-9 * synth.rig_scale(rig) and r[-1] > synth.rig_scale(rig)
    assert abs(s - 0.5) < 1e-9
    s2, R2, t2, r2, use2 = scale.fit_similarity(src, dst, trim=None, inlier_frac=0.8)
    assert not use2[-1] and r2[:-1].max() < 1e-9 * synth.rig_scale(rig)
    # pre-1.1 behaviour: the all-point start is dragged and never recovers
    _s, _R, _t, r3, use3 = scale.fit_similarity(src, dst, robust_init=False)
    assert use3[-1]
    # clean (noisy) data: identical to the all-point start; collinear subsets are skipped
    rng = np.random.default_rng(0)
    for X in [dst, np.c_[np.linspace(0, 1, 6), np.zeros(6), np.zeros(6)]
              + rng.normal(0, [0, 1e-3, 1e-3], (6, 3))]:
        Y = X + rng.normal(0, 0.01 * synth.rig_scale(rig), X.shape)
        for kw in [{}, {'inlier_frac': 0.8, 'trim': None}]:
            a, b = scale.fit_similarity(X, Y, **kw), scale.fit_similarity(X, Y, robust_init=False, **kw)
            assert all(np.allclose(u, v, atol=1e-9, equal_nan=True) for u, v in zip(a, b))


def test_compare_runaway_camera():
    from aniposelib import quality
    rig = synth.load_rig('outdoor_12cam')
    c = quality.compare(_runaway(rig), rig)
    assert c.overall['center_residual_median'] < 1e-6 * c.overall['rig_scale']
    names = rig.get_names()
    assert not c.per_camera[names[-1]]['in_gauge']
    assert all(c.per_camera[n]['in_gauge'] and c.per_camera[n]['center_residual'] < 1e-6 * c.overall['rig_scale']
               for n in names[:-1])
