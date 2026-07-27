#!/usr/bin/env python

import os

import cv2
import numpy as np
import pytest
import torch

from aniposelib.boards import Checkerboard
from aniposelib.cameras import (Camera, CameraGroup, FisheyeCamera,
                                init_camera_intrinsics)

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
FISHEYE_TOML = os.path.join(DATA, 'calib_fisheye_5cam.toml')

K = np.array([[869.2167, 0.0, 959.5],
              [0.0, 869.2167, 539.5],
              [0.0, 0.0, 1.0]], dtype='float64')
SIZE = (1920, 1080)
RVEC = np.array([0.1, -0.2, 0.3])
TVEC = np.array([0.5, -0.3, 1.2])

# the two negative k1 cases are the calib-883 cameras whose frame corners fall
# outside the invertible domain of the model
DISTS = [np.zeros(4),
         np.array([0.1846, 0.0, 0.0, 0.0]),
         np.array([-0.1539, 0.0, 0.0, 0.0]),
         np.array([-0.15, 0.05, -0.01, 0.002])]


def make_cam(dist):
    return FisheyeCamera(matrix=K, dist=dist, size=SIZE,
                         rvec=RVEC, tvec=TVEC, name='fish')


def cv_project(p3d, dist, rvec=RVEC, tvec=TVEC, matrix=K):
    # cv2.fisheye.projectPoints wants (N, 1, 3) float64
    out, _ = cv2.fisheye.projectPoints(
        np.ascontiguousarray(p3d, dtype='float64').reshape(-1, 1, 3),
        rvec, tvec, matrix, np.asarray(dist, dtype='float64'))
    return out.reshape(-1, 2)


def frame_grid(nx=40, ny=30):
    gx, gy = np.meshgrid(np.linspace(0, SIZE[0] - 1, nx),
                         np.linspace(0, SIZE[1] - 1, ny))
    return np.stack([gx.ravel(), gy.ravel()], axis=-1)


def to_np(x):
    return x.detach().cpu().numpy()


def test_fisheye_constructs_and_loads():
    """FisheyeCamera used to raise on construction, making every fisheye
    calibration file unloadable."""
    cam = FisheyeCamera(matrix=K, dist=np.zeros(4), size=SIZE)
    assert cam.dist.shape == (4,)
    assert cam.matrix.dtype == torch.float64

    cgroup = CameraGroup.load(FISHEYE_TOML)
    assert len(cgroup.cameras) == 5
    assert all(isinstance(c, FisheyeCamera) for c in cgroup.cameras)
    assert all(c.dist.shape == (4,) for c in cgroup.cameras)


@pytest.mark.parametrize('dist', DISTS)
def test_fisheye_forward_parity_cv2(dist):
    rng = np.random.default_rng(0)
    p3d = rng.normal(size=(500, 3)) * np.array([2.0, 2.0, 0.5]) + np.array([0, 0, 4.0])
    got = to_np(make_cam(dist).project(p3d))
    assert np.max(np.abs(got - cv_project(p3d, dist))) < 1e-9


def test_fisheye_forward_parity_real_calibration():
    cgroup = CameraGroup.load(FISHEYE_TOML)
    rng = np.random.default_rng(1)
    p3d = rng.normal(size=(200, 3)) * 0.5
    got = to_np(cgroup.project(p3d))
    for i, cam in enumerate(cgroup.cameras):
        exp = cv_project(p3d, to_np(cam.dist), to_np(cam.rvec),
                         to_np(cam.tvec), to_np(cam.matrix))
        assert np.max(np.abs(exp - got[i])) < 1e-9


def test_fisheye_behind_camera_matches_cv2():
    """Points behind the camera give a ghost mirrored through the principal
    point. That is what cv2.fisheye does and what the pinhole path here does,
    so we match it rather than inventing a different answer. is_point_visible
    is what screens these out."""
    p = np.array([[0.1, 0.2, 3.0], [0.1, 0.2, -3.0]])

    # with the rig extrinsics, just check we agree with cv2
    got = to_np(make_cam(DISTS[1]).project(p))
    assert np.allclose(got, cv_project(p, DISTS[1]), atol=1e-9)

    # the mirroring itself is only visible with an identity pose
    zero = np.zeros(3)
    cam = FisheyeCamera(matrix=K, dist=DISTS[1], size=SIZE, rvec=zero, tvec=zero)
    got = to_np(cam.project(p))
    assert np.allclose(got, cv_project(p, DISTS[1], zero, zero), atol=1e-9)

    center = np.array([K[0, 2], K[1, 2]])
    assert np.allclose(got[1] - center, -(got[0] - center), atol=1e-9)


@pytest.mark.parametrize('dist', DISTS)
def test_fisheye_inverse_parity_cv2(dist):
    pix = frame_grid(80, 60)
    exp = cv2.fisheye.undistortPoints(
        pix.reshape(-1, 1, 2), K, np.asarray(dist, dtype='float64')).reshape(-1, 2)
    got = to_np(make_cam(dist).undistort_points(pix))

    # cv2 writes -1000000 when its solve does not converge; we write NaN, since
    # everything downstream reads NaN as missing but would triangulate -1000000
    cv_bad = np.any(exp < -1e5, axis=1)
    our_bad = np.any(~np.isfinite(got), axis=1)
    assert np.array_equal(cv_bad, our_bad)

    good = ~cv_bad
    assert np.max(np.abs(exp[good] - got[good])) < 1e-9

    if dist[0] < -0.1 and np.all(dist[1:] == 0):
        # a lone k1 this negative makes the forward model non monotone well
        # inside the frame, so this case really does exercise that branch. The
        # higher order terms in the last case pull it back to invertible.
        assert cv_bad.sum() > 0


@pytest.mark.parametrize('dist', DISTS)
def test_fisheye_roundtrip(dist):
    cam = make_cam(dist)
    pix = frame_grid()
    und = cam.undistort_points(pix)
    red = to_np(cam.distort_points(und))

    finite = np.all(np.isfinite(to_np(und)), axis=1)
    assert finite.mean() > 0.7
    assert np.max(np.abs(red[finite] - pix[finite])) < 1e-6
    assert np.all(np.isnan(red[~finite]))


@pytest.mark.parametrize('dist', [np.zeros(4), np.array([0.1846, 0.0, 0.0, 0.0])])
def test_fisheye_gradients_finite(dist):
    """The singular points of the model must not produce NaN gradients, and a
    NaN row in the batch must not poison the gradients of the shared camera
    parameters. bundle_adjust seeds p3d from triangulate, which returns NaN for
    points seen by fewer than two cameras, so this is a routine input."""
    cam = FisheyeCamera(matrix=K, dist=dist, size=SIZE, rvec=RVEC, tvec=TVEC)
    p = torch.tensor([[0.0, 0.0, 3.0],          # exactly the principal point
                      [0.0, 0.0, 0.0],          # z == 0
                      [5.0, 4.0, 0.3],          # large radius
                      [float('nan'), 1.0, 2.0],
                      [1.0, float('inf'), 2.0]],
                     dtype=torch.float64, requires_grad=True)

    out = cam.project(p)
    assert torch.isnan(out[3]).all() and torch.isnan(out[4]).all()
    assert torch.isfinite(out[:3]).all()

    out[torch.isfinite(out)].sum().backward()
    for name, t in [('p', p), ('matrix', cam.matrix), ('dist', cam.dist),
                    ('rvec', cam.rvec), ('tvec', cam.tvec)]:
        assert torch.isfinite(t.grad).all(), name

    # the skew and bottom row are not part of the model, so they must not move
    g = cam.matrix.grad
    assert g[0, 1] == 0 and g[2, 0] == 0 and g[2, 1] == 0 and g[2, 2] == 0


def test_fisheye_nan_passthrough_project():
    cam = make_cam(DISTS[1])
    rng = np.random.default_rng(3)
    p3d = rng.normal(size=(20, 3)) + np.array([0, 0, 4.0])
    clean = to_np(cam.project(p3d))

    dirty = p3d.copy()
    dirty[3] = [np.nan, 1.0, 2.0]
    dirty[7] = [1.0, np.inf, 2.0]
    dirty[11, 2] = np.nan
    out = to_np(cam.project(dirty))

    bad = np.array([3, 7, 11])
    assert np.all(np.isnan(out[bad]))
    keep = np.setdiff1d(np.arange(20), bad)
    assert np.array_equal(out[keep], clean[keep])


def test_fisheye_nan_passthrough_undistort():
    cam = make_cam(DISTS[1])
    pix = frame_grid(10, 8)
    clean = to_np(cam.undistort_points(pix))

    dirty = pix.copy()
    dirty[2] = [np.nan, 500.0]
    dirty[5] = [800.0, np.nan]
    out = to_np(cam.undistort_points(dirty))

    assert np.all(np.isnan(out[[2, 5]]))
    keep = np.setdiff1d(np.arange(len(pix)), [2, 5])
    assert np.array_equal(out[keep], clean[keep])
    assert out.shape == pix.shape


def test_fisheye_toml_roundtrip(tmp_path):
    cg = CameraGroup.load(FISHEYE_TOML)
    out = str(tmp_path / 'out.toml')
    cg.dump(out)
    cg2 = CameraGroup.load(out)

    for a, b in zip(cg.cameras, cg2.cameras):
        assert isinstance(b, FisheyeCamera)
        assert b.dist.shape == (4,)
        assert torch.equal(a.matrix, b.matrix)
        assert torch.equal(a.dist, b.dist)
        assert torch.equal(a.rvec, b.rvec)
        assert torch.equal(a.tvec, b.tvec)
        assert a.get_name() == b.get_name()
        assert list(a.get_size()) == list(b.get_size())

    assert all(d['fisheye'] is True for d in cg2.get_dicts())
    assert cg2.metadata == cg.metadata


def test_fisheye_copy():
    cg = CameraGroup.load(FISHEYE_TOML)
    cam = cg.cameras[3]

    # copy() has to build the new camera with a 4 element dist, since
    # load_state_dict is strict about shapes
    dup = cam.copy()
    assert isinstance(dup, FisheyeCamera)
    assert dup.dist.shape == (4,)
    assert torch.equal(dup.dist, cam.dist)

    dup.rvec.data += 1.0
    assert not torch.equal(dup.rvec, cam.rvec)

    sub = cg.subset_cameras([0, 2, 4])
    assert all(isinstance(c, FisheyeCamera) for c in sub.cameras)


def test_fisheye_camera_rays_finite():
    """calib-883 cam_3 has corners outside the invertible domain, which the
    inherited get_camera_rays would turn into NaN directions."""
    cg = CameraGroup.load(FISHEYE_TOML)
    for cam in cg.cameras:
        for ray in cam.get_camera_rays():
            assert torch.isfinite(ray.direction).all()
            assert abs(torch.linalg.norm(ray.direction).item() - 1) < 1e-12


def _tri_case():
    cg = CameraGroup.load(FISHEYE_TOML)
    rng = np.random.default_rng(1)
    p3d = rng.normal(size=(300, 3)) * 0.5
    return cg, p3d, cg.project(p3d).detach()


def test_fisheye_triangulate_cpu():
    cg, p3d, p2d = _tri_case()
    got = cg.triangulate(p2d)
    assert torch.max(torch.abs(got - torch.as_tensor(p3d))).item() < 1e-9

    # dropping one view must not fabricate detections
    p2n = p2d.clone()
    p2n[0, :10] = float('nan')
    got2 = cg.triangulate(p2n)
    assert torch.max(torch.abs(got2 - torch.as_tensor(p3d))).item() < 1e-9


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_fisheye_triangulate_cuda():
    cg, p3d, p2d = _tri_case()
    cg = cg.to('cuda')
    p2d = p2d.to('cuda')

    got = cg.triangulate(p2d)
    assert got.device.type == 'cuda'
    assert torch.max(torch.abs(got.cpu() - torch.as_tensor(p3d))).item() < 1e-9

    proj = cg.project(torch.as_tensor(p3d, device='cuda'))
    assert proj.device.type == 'cuda'
    assert torch.max(torch.abs(proj.cpu() - p2d.cpu())).item() < 1e-8


def test_fisheye_board_pose():
    """The pinhole PnP solvers read the 4 equidistant coefficients as
    k1, k2, p1, p2, which is off by more than a degree of rotation."""
    board = Checkerboard(7, 5, square_length=0.03)
    objp = board.get_object_points()
    dist = np.array([0.1846, 0.02, 0.0, 0.0])
    rvec = np.array([0.15, -0.25, 0.10])
    tvec = np.array([0.1, -0.05, 1.4])

    cam = FisheyeCamera(matrix=K, dist=dist, size=SIZE, rvec=rvec, tvec=tvec)
    p2d = to_np(cam.project(objp.reshape(-1, 3))).reshape(-1, 1, 2)

    got_r, got_t = board.estimate_pose_points(cam, p2d)
    assert np.max(np.abs(got_r.ravel() - rvec)) < 1e-5
    assert np.max(np.abs(got_t.ravel() - tvec)) < 1e-5


def test_fisheye_init_intrinsics():
    """calibrate_rows seeds intrinsics per projection model. A fisheye camera
    started at zero distortion throws away the equidistant term."""
    board = Checkerboard(7, 5, square_length=0.03)
    objp = board.get_object_points()
    dist = np.array([0.1846, 0.0, 0.0, 0.0])
    rng = np.random.default_rng(0)

    truth = FisheyeCamera(matrix=K, dist=dist, size=SIZE)
    objs, imgs = [], []
    for _ in range(14):
        truth.set_rotation(rng.normal(scale=0.35, size=3))
        truth.set_translation(np.array([0, 0, 1.1]) + rng.normal(scale=0.28, size=3))
        p2d = to_np(truth.project(objp.reshape(-1, 3)))
        if not (p2d.min() > 0 and p2d[:, 0].max() < SIZE[0]
                and p2d[:, 1].max() < SIZE[1]):
            continue
        objs.append(objp.astype('float32'))
        imgs.append(p2d.reshape(-1, 1, 2).astype('float32'))

    cam = FisheyeCamera(size=SIZE, name='fish')
    init_camera_intrinsics(cam, objs, imgs, SIZE)
    assert abs(to_np(cam.matrix)[0, 0] - K[0, 0]) < 1.0
    assert abs(to_np(cam.dist)[0] - dist[0]) < 1e-3

    # the pinhole seed must be untouched by this
    camp = Camera(size=SIZE, name='pin')
    init_camera_intrinsics(camp, objs, imgs, SIZE)
    assert np.allclose(to_np(camp.dist), 0)


@pytest.mark.integration
def test_fisheye_bundle_adjust_improves():
    cg, p3d, p2d = _tri_case()
    rng = np.random.default_rng(7)
    for cam in cg.cameras:
        cam.rvec.data += torch.as_tensor(rng.normal(size=3) * 0.01)
        cam.tvec.data += torch.as_tensor(rng.normal(size=3) * 0.02)

    before = cg.average_error(p2d).item()
    after = cg.bundle_adjust(p2d, max_nfev=500, verbose=False,
                             only_extrinsics=True).item()
    assert after < before * 0.5

    m = to_np(cg.cameras[2].matrix)
    assert m[0, 1] == 0 and m[2, 0] == 0 and m[2, 1] == 0 and m[2, 2] == 1
