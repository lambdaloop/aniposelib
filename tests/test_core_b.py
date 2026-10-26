"""1.0 Phase B (core): camera model, conventions and I/O."""

import copy
import pickle
import warnings

import cv2
import numpy as np
import pytest
import torch

import synth
from aniposelib.cameras import (Camera, CameraGroup, CameraOrderWarning, FisheyeCamera,
                                interior_span, resample_points)
from aniposelib.params import ParamSpec
from aniposelib.utils import mean_transform_robust, select_matrices

K_SKEW = np.array([[800.0, 40.0, 300.0], [0.0, 780.0, 220.0], [0.0, 0.0, 1.0]])
DIST = np.array([0.01, -0.001, 0.0002, -0.0001, 0.0003])


def small_group(n=3, matrix=K_SKEW, dist=DIST, offset=None, cls=Camera):
    cams = []
    for i in range(n):
        tvec = np.array([-0.4 * i, 0.1 * i, 0.0])
        rvec = np.array([0.0, 0.05 * i, 0.0])
        cam = cls(matrix=matrix, dist=dist, rvec=rvec, tvec=tvec, name='c{}'.format(i),
                  size=(640, 480), offset=offset)
        cams.append(cam)
    return CameraGroup(cams)


WORLD = np.array([[0.2, -0.1, 10.0], [1.1, 0.3, 12.0], [-0.5, 0.4, 9.0], [0.0, 0.0, 11.0]])


# ---------------------------------------------------------------- B1 / B5 ordering and string I/O

def test_load_sorts_numerically_and_warns(tmp_path):
    cg = synth.load_rig('allen_12cam')
    s = cg.dumps()
    with pytest.warns(CameraOrderWarning):
        back = CameraGroup.loads(s)
    assert back.get_names() == cg.get_names()
    lex = CameraGroup.loads(s, order='lexical')   # explicit choice: no warning
    names = cg.get_names()
    assert lex.get_names() == [names[i] for i in [0, 1, 10, 11] + list(range(2, 10))]


def test_mouse_outdoor_names_in_order():
    cg = synth.load_rig('outdoor_12cam')
    assert cg.get_names() == [str(n) for n in range(10, 22)]


def test_no_warning_for_few_cameras():
    cg = synth.load_rig('motobs_7cam')
    with warnings.catch_warnings():
        warnings.simplefilter('error', CameraOrderWarning)
        CameraGroup.loads(cg.dumps())


def test_duplicate_names_warn():
    d = small_group().to_dict()
    d['cam_1']['name'] = d['cam_0']['name']
    with pytest.warns(CameraOrderWarning, match='duplicate'):
        CameraGroup.from_dict(d)


def test_dict_roundtrip_with_metadata():
    cg = small_group(offset=(3.0, -2.0))
    cg.metadata = {'units': 'mm', 'nested': {'a': [1, 2]}}
    back = CameraGroup.from_dict(cg.to_dict())
    assert back.metadata == cg.metadata
    np.testing.assert_array_equal(back.project(WORLD), cg.project(WORLD))
    back2 = CameraGroup.loads(cg.dumps())
    np.testing.assert_allclose(back2.project(WORLD), cg.project(WORLD), rtol=0, atol=1e-9)


# ---------------------------------------------------------------- B2 metadata

def test_metadata_default_not_shared():
    a, b = CameraGroup([Camera()]), CameraGroup([Camera()])
    a.metadata['x'] = 1
    assert b.metadata == {}


def test_metadata_copies_are_isolated():
    cg = small_group()
    cg.metadata = {'nested': {'a': 1}}
    for other in [cg.copy(), cg.subset_cameras([0, 1]), cg.subset_cameras_names(['c2'])]:
        other.metadata['nested']['a'] = 99
        assert cg.metadata['nested']['a'] == 1


# ---------------------------------------------------------------- B3 / B4 per-camera keys

def test_extra_keys_roundtrip(tmp_path):
    d = small_group().to_dict()
    d['cam_0']['moving'] = True
    d['cam_0']['calibrated'] = False
    d['cam_1']['model'] = 'custom'
    cg = CameraGroup.from_dict(d)
    assert cg.cameras[0].extra == {'moving': True, 'calibrated': False}
    assert cg.cameras[1].extra == {'model': 'custom'}
    assert cg.copy().cameras[0].extra == {'moving': True, 'calibrated': False}
    cg.dump(tmp_path / 'c.toml')
    back = CameraGroup.load(tmp_path / 'c.toml')
    assert back.cameras[0].extra == {'moving': True, 'calibrated': False}
    assert back.get_dicts()[1]['model'] == 'custom'


@pytest.mark.parametrize('fisheye,n', [(False, 5), (True, 4)])
def test_short_dist_padded_on_load(fisheye, n):
    d = {'name': 'a', 'size': [640, 480], 'matrix': np.eye(3).tolist(),
         'distortions': [0.1, -0.01], 'rotation': [0, 0, 0], 'translation': [0, 0, 1]}
    if fisheye:
        d['fisheye'] = True
    cam = CameraGroup.from_dicts([d]).cameras[0]
    assert cam.dist.shape == (n,)
    np.testing.assert_array_equal(cam.dist.detach().numpy()[:2], [0.1, -0.01])
    assert np.all(cam.dist.detach().numpy()[2:] == 0)


# ---------------------------------------------------------------- B6 return types

def test_return_type_matrix(device):
    cg = small_group().to(device)
    p2d_np = cg.project(WORLD)
    assert isinstance(p2d_np, np.ndarray)
    p2d_t = cg.project(torch.as_tensor(WORLD))
    assert isinstance(p2d_t, torch.Tensor) and not p2d_t.requires_grad
    assert p2d_t.device.type == 'cpu'   # input device, not camera device
    for out in [cg.triangulate(p2d_np), cg.reprojection_error(WORLD, p2d_np),
                cg.undistort_points(p2d_np), cg.reprojection_error(WORLD, p2d_np, mean=True),
                cg.is_point_visible(WORLD), cg.cameras[0].project(WORLD)]:
        assert isinstance(out, np.ndarray)
    tri = cg.triangulate(p2d_t)
    assert isinstance(tri, torch.Tensor) and not tri.requires_grad
    np.testing.assert_allclose(tri.numpy(), WORLD, atol=1e-8)
    p3, picked, p2, err = cg.triangulate_ransac(p2d_np)
    assert all(isinstance(x, np.ndarray) for x in (p3, picked, p2, err))
    assert isinstance(cg.average_error(p2d_np), (float, np.floating, np.ndarray))


def test_return_grad_keeps_graph():
    cg = small_group()
    p2d = cg.project(torch.as_tensor(WORLD), return_grad=True)
    assert p2d.requires_grad
    p2d.sum().backward()
    assert cg.cameras[1].matrix.grad.abs().sum() > 0 and cg.cameras[1].rvec.grad.abs().sum() > 0
    # triangulate is differentiable in the camera parameters too
    tri = cg.triangulate(cg.project(WORLD), return_grad=True)
    assert tri.requires_grad


def test_keyword_input_type():
    cg = small_group()
    assert isinstance(cg.project(points=torch.as_tensor(WORLD)), torch.Tensor)
    assert isinstance(cg.project(points=WORLD), np.ndarray)


# ---------------------------------------------------------------- B7 / B12 undistort, batching

def test_public_undistort_matches_alias_and_fast_path():
    cg = synth.load_rig('fisheye_5cam')
    p3d = synth.random_points(cg, 200, seed=1)
    p2d = synth.project(cg, p3d)
    und = cg.undistort_points(p2d)
    np.testing.assert_array_equal(und, cg._undistort_points_batch(p2d))
    np.testing.assert_array_equal(cg.triangulate(und, undistort=False), cg.triangulate(p2d))


def test_batch_size_parity():
    cg = synth.load_rig('outdoor_9cam')
    p2d = synth.add_noise(synth.project(cg, synth.random_points(cg, 2500, seed=2)), 1.0)
    a = cg.triangulate(p2d)
    b = cg.triangulate(p2d, batch_size=1000)
    np.testing.assert_allclose(a, b, rtol=0, atol=1e-12)


def test_fisheye_early_exit_parity():
    cam = synth.load_rig('fisheye_5cam').cameras[0]
    w, h = cam.get_size()
    rng = np.random.default_rng(0)
    pix = rng.uniform([0, 0], [w, h], size=(5000, 2))
    pix[:5] = np.nan
    fast = cam._solve_theta(pix)
    full = cam._solve_theta(pix, step_tol=0)
    np.testing.assert_allclose(fast[0].detach().numpy(), full[0].detach().numpy(), rtol=0, atol=1e-12)
    assert torch.equal(fast[5], full[5])


def test_frozen_context():
    cg = small_group()
    p2d = cg.project(WORLD)
    with cg.frozen():
        np.testing.assert_array_equal(cg.project(WORLD), p2d)
        np.testing.assert_array_equal(cg.triangulate(p2d), cg.triangulate(p2d))
        with pytest.raises(RuntimeError):
            cg.bundle_adjust(p2d, max_nfev=5, verbose=False)
        cg.cameras[1].set_translation([-1.0, 0.0, 0.0])
        moved = cg.project(WORLD)
    assert not np.allclose(moved[1], p2d[1])
    np.testing.assert_array_equal(cg.project(WORLD), moved)
    assert all(c._frozen_ext is None for c in cg.cameras)


def test_project_moves_input_to_camera_device(device):
    cg = synth.load_rig('fisheye_5cam').to(device)
    out = cg.project(torch.zeros((3, 3)) + torch.tensor([0.0, 0.0, 3.0]))
    assert out.device.type == 'cpu'


# ---------------------------------------------------------------- B8 kwargs

def test_unknown_kwargs_raise():
    cg = small_group()
    p2d = cg.project(WORLD)
    with pytest.raises(TypeError, match='lossy'):
        cg.bundle_adjust(p2d, lossy='huber')
    with pytest.raises(TypeError, match='seed'):
        cg.bundle_adjust_iter(p2d, seed=0)
    with pytest.raises(TypeError, match='bogus'):
        cg.calibrate_rows([[], [], []], None, bogus=1)
    with pytest.raises(TypeError, match='bogus'):
        cg.calibrate_videos([[], [], []], None, bogus=1)


# ---------------------------------------------------------------- B9 rng

def test_resample_rng_deterministic_and_global_untouched():
    cg = synth.load_rig('outdoor_9cam')
    p2d = synth.project(cg, synth.random_points(cg, 1000, seed=3))
    state = np.random.get_state()[1].copy()
    a, _ = resample_points(p2d, n_samp=50, rng=7)
    b, _ = resample_points(p2d, n_samp=50, rng=np.random.default_rng(7))
    resample_points(p2d, n_samp=50)
    np.testing.assert_array_equal(a, b)
    assert np.array_equal(np.random.get_state()[1], state)


def test_point_cloud_rng():
    cg = synth.load_rig('motobs_7cam')
    a, _ = cg.get_point_cloud(rng=1)
    b, _ = cg.get_point_cloud(rng=1)
    assert torch.equal(a, b)


# ---------------------------------------------------------------- B10 utils

def _M(rvec, tvec):
    M = np.eye(4)
    M[:3, :3] = cv2.Rodrigues(np.asarray(rvec, dtype='float64'))[0]
    M[:3, 3] = tvec
    return M


def test_select_matrices_identical_and_small():
    Ms = [np.eye(4)] * 6   # VGGT-like: every pair agrees exactly
    assert len(select_matrices(Ms)) == 6
    assert len(select_matrices([np.eye(4)])) == 1
    np.testing.assert_allclose(mean_transform_robust(Ms), np.eye(4), atol=1e-12)
    with pytest.raises(ValueError):
        select_matrices([])


def test_mean_transform_robust_median_seed_and_fallback():
    good = [_M([0.01 * i, 0, 0], [1, 0, 0]) for i in range(5)]
    bad = [_M([2.5, 0, 0], [5, 5, 5])] * 2
    M = mean_transform_robust(good + bad)
    np.testing.assert_allclose(M[:3, 3], [1, 0, 0], atol=1e-9)
    far = _M([3.0, 0, 0], [0, 0, 0])
    out = mean_transform_robust(good, approx=far, error=1e-6)
    assert np.all(np.isfinite(out))


# ---------------------------------------------------------------- B11 fill

def test_interior_span():
    obs = np.array([0, 0, 1, 0, 1, 0], dtype=bool)
    np.testing.assert_array_equal(interior_span(obs), [0, 0, 1, 1, 1, 0])
    assert not interior_span(np.zeros((4, 2), bool)).any()


@pytest.mark.parametrize('fill', ['interior', 'none', 'all'])
def test_optim_points_fill_modes(fill):
    cg = small_group()
    rng = np.random.default_rng(0)
    N, J = 20, 2
    p3d = WORLD[None, :J] + np.cumsum(rng.normal(scale=0.01, size=(N, J, 3)), axis=0)
    p2d = cg.project(p3d.reshape(-1, 3)).reshape(3, N, J, 2)
    p2d[:, :3] = np.nan        # leading gap
    p2d[:, 10:12, 0] = np.nan  # interior gap, joint 0
    out = cg.optim_points(p2d, p3d, n_iters=20, fill=fill)
    nan = np.isnan(out[..., 0])
    if fill == 'all':
        assert not nan.any()
    elif fill == 'interior':
        assert nan[:3].all() and not nan[3:].any()
    else:
        assert nan[:3].all() and nan[10:12, 0].all() and not nan[3:10].any()


# ---------------------------------------------------------------- B13 full K (skew)

def test_skew_projection_matches_full_K_and_roundtrips():
    cam = Camera(matrix=K_SKEW, dist=DIST, offset=(7.0, -4.0))
    pix = cam.project(WORLD)
    x = WORLD[:, :2] / WORLD[:, 2:]
    r2 = (x ** 2).sum(1)
    k1, k2, p1, p2, k3 = DIST
    radial = 1 + k1 * r2 + k2 * r2 ** 2 + k3 * r2 ** 3
    xd = x[:, 0] * radial + 2 * p1 * x[:, 0] * x[:, 1] + p2 * (r2 + 2 * x[:, 0] ** 2)
    yd = x[:, 1] * radial + p1 * (r2 + 2 * x[:, 1] ** 2) + 2 * p2 * x[:, 0] * x[:, 1]
    sensor = (K_SKEW @ np.stack([xd, yd, np.ones_like(xd)]))[:2].T
    np.testing.assert_allclose(pix, sensor - [7.0, -4.0], rtol=0, atol=1e-9)
    # port of tailcyclenet test_skew_projection_inverse_and_rays_round_trip
    np.testing.assert_allclose(cam.undistort_points(pix), x, rtol=0, atol=1e-9)
    np.testing.assert_allclose(cam.distort_points(x), pix, rtol=0, atol=1e-7)


def test_skew_triangulation():
    cg = small_group()
    np.testing.assert_allclose(cg.triangulate(cg.project(WORLD)), WORLD, atol=1e-7)


def test_skew_projection_sensitivity_matches_autograd():
    cam = Camera(matrix=K_SKEW, dist=np.zeros(5), rvec=[0.1, -0.2, 0.05], tvec=[0.1, 0.2, 0.3])
    J = cam.projection_sensitivity(WORLD)
    for i, p in enumerate(torch.as_tensor(WORLD)):
        oracle = torch.autograd.functional.jacobian(
            lambda q: cam.project(q[None], return_grad=True)[0], p)
        np.testing.assert_allclose(J[i], oracle.detach().numpy(), rtol=1e-9, atol=1e-9)


def test_fisheye_skew_matches_cv2_alpha():
    K = np.array([[869.2, 25.0, 959.5], [0.0, 869.2, 539.5], [0, 0, 1.0]])
    dist = np.array([0.1846, 0.01, 0.0, 0.0])
    cam = FisheyeCamera(matrix=K, dist=dist, rvec=[0.1, -0.2, 0.3], tvec=[0.5, -0.3, 1.2])
    p3d = np.random.default_rng(0).normal(size=(100, 3)) + [0, 0, 4.0]
    # cv2.fisheye ignores K[0,1] and takes the skew as alpha = K[0,1] / fx instead
    exp, _ = cv2.fisheye.projectPoints(p3d.reshape(-1, 1, 3), np.array([0.1, -0.2, 0.3]),
                                       np.array([0.5, -0.3, 1.2]), K, dist, alpha=25.0 / 869.2)
    np.testing.assert_allclose(cam.project(p3d), exp.reshape(-1, 2), rtol=0, atol=1e-9)
    x = cam.undistort_points(exp.reshape(-1, 2))
    np.testing.assert_allclose(cam.distort_points(x), exp.reshape(-1, 2), atol=1e-6)


# ---------------------------------------------------------------- B14 parameter masks

def _ba_skew_group():
    cg = small_group(dist=np.zeros(5))
    p2d = cg.project(WORLD)
    for cam in cg.cameras:
        cam.matrix.data[2, 0] = 1e-6
        cam.rvec.data += 0.01
    return cg, p2d


def test_ba_keeps_skew_and_bottom_row_frozen():
    cg = synth.load_rig('motobs_7cam')
    for cam in cg.cameras:
        cam.matrix.data[0, 1] = 5.0
    p2d = synth.add_noise(synth.project(cg, synth.random_points(cg, 300, seed=5)), 0.3)
    for cam in cg.cameras:
        cam.matrix.data[0, 0] *= 1.01
        cam.matrix.data[2, 0] = 1e-7
    before = [c.matrix.detach().clone() for c in cg.cameras]
    cg.bundle_adjust(p2d, max_nfev=60, verbose=False)
    for b, c in zip(before, cg.cameras):
        m = c.matrix.detach()
        assert m[0, 1] == b[0, 1] and m[1, 0] == b[1, 0] and torch.equal(m[2], b[2])


def test_mask_applies_to_any_optimizer():
    """The free-intrinsics mask is a gradient hook, so a hand-written optimizer over
    cgroup.parameters() also leaves skew and the bottom row exactly alone."""
    cg = small_group(dist=np.zeros(5))
    p2d = torch.as_tensor(cg.project(WORLD))
    before = [c.matrix.detach().clone() for c in cg.cameras]
    opt = torch.optim.Adam(cg.parameters(), lr=1e-2)
    for cam in cg.cameras:
        cam.matrix.data[0, 0] += 5
    for _ in range(5):
        opt.zero_grad()
        loss = (cg.project(torch.as_tensor(WORLD), return_grad=True) - p2d).square().sum()
        loss.backward()
        opt.step()
    for b, c in zip(before, cg.cameras):
        m = c.matrix.detach()
        assert m[0, 1] == b[0, 1] and m[1, 0] == b[1, 0] and torch.equal(m[2], b[2])
        assert not torch.equal(m[0, 0], b[0, 0] + 5)


@pytest.mark.parametrize('roundtrip', [copy.deepcopy, lambda c: pickle.loads(pickle.dumps(c)),
                                       lambda c: c.copy(),
                                       lambda c: CameraGroup.loads(c.dumps())])
def test_free_intrinsics_mask_survives_copies(roundtrip):
    cg, p2d = _ba_skew_group()
    ParamSpec(intrinsics=('fx', 'fy', 'skew'), n_dist=1).apply(cg)
    cg2 = roundtrip(cg)
    assert cg2.get_free_intrinsics() == [('fx', 'fy', 'skew')] * 3
    assert cg2.get_n_dists() == [1] * 3
    cam = cg2.cameras[0]
    cam.project(WORLD, return_grad=True).sum().backward()
    g = cam.matrix.grad
    assert g[0, 1] != 0 and g[0, 0] != 0 and g[0, 2] == 0 and g[1, 2] == 0
    assert torch.all(g[2] == 0) and g[1, 0] == 0


def test_free_intrinsics_validation():
    with pytest.raises(ValueError):
        Camera().set_free_intrinsics(('fx', 'k1'))
    assert ParamSpec.from_camera(Camera()).intrinsics == ('fx', 'fy', 'cx', 'cy')


# ---------------------------------------------------------------- B15 dtype

def test_float32_smoke():
    cg = synth.load_rig('outdoor_9cam')
    p3d = synth.random_points(cg, 100, seed=4)
    p2d = synth.project(cg, p3d)
    cg32 = CameraGroup.loads(cg.dumps(), dtype=torch.float32)
    assert cg32.dtype == torch.float32
    out = cg32.triangulate(p2d)
    assert out.dtype == np.float32
    err = np.linalg.norm(out - p3d, axis=1) / synth.rig_scale(cg)
    assert np.nanmedian(err) < 1e-3
    assert cg32.copy().dtype == torch.float32


# ---------------------------------------------------------------- B18 crop offsets

def test_offset_project_undistort_and_triangulate():
    sensor = small_group(offset=None)
    cropped = small_group(offset=(100.0, 50.0))
    p_sensor = sensor.project(WORLD)
    p_crop = cropped.project(WORLD)
    np.testing.assert_allclose(p_crop, p_sensor - [100.0, 50.0], rtol=0, atol=1e-9)
    np.testing.assert_allclose(cropped.undistort_points(p_crop), sensor.undistort_points(p_sensor),
                               atol=1e-12)
    np.testing.assert_allclose(cropped.triangulate(p_crop), WORLD, atol=1e-7)


def test_offset_serialised_only_when_nonzero():
    assert 'offset' not in small_group().get_dicts()[0]
    cg = small_group(offset=(3.0, 4.0))
    assert cg.get_dicts()[0]['offset'] == [3.0, 4.0]
    back = CameraGroup.loads(cg.dumps())
    np.testing.assert_array_equal(back.get_offsets().numpy(), [[3.0, 4.0]] * 3)
    assert back.cameras[0].extra == {}


def test_apply_crop_offsets_anipose_config():
    cg = small_group()
    config = {'cameras': {'c0': {'offset': [10, 20, 300, 200]}, 'c2': {'offset': [5, 6]}}}
    cg.apply_crop_offsets(config)
    np.testing.assert_array_equal(cg.get_offsets().numpy(), [[10, 20], [0, 0], [5, 6]])
    assert tuple(cg.cameras[0].get_size()) == (300, 200)
    assert tuple(cg.cameras[2].get_size()) == (640, 480)
    cg.apply_crop_offsets({'c1': [1, 2]})
    np.testing.assert_array_equal(cg.cameras[1].get_offset().numpy(), [1, 2])
