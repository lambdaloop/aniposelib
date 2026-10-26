#!/usr/bin/env python

import copy
import os
import pickle

import numpy as np
import pytest
import torch

from aniposelib.boards import Checkerboard
from aniposelib.cameras import (Camera, CameraGroup, FisheyeCamera,
                                init_camera_intrinsics)

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
FISHEYE_TOML = os.path.join(DATA, 'rigs', 'fisheye_5cam.toml')

K = np.array([[800.0, 0.0, 320.0],
              [0.0, 800.0, 240.0],
              [0.0, 0.0, 1.0]], dtype='float64')


def dist_grad(cam):
    """Which distortion coefficients actually receive gradient."""
    cam.dist.grad = None
    cam.project(np.array([[0.1, 0.2, 3.0], [0.4, -0.2, 2.0]]), return_grad=True).sum().backward()
    return np.nonzero(cam.dist.grad.detach().cpu().numpy())[0].tolist()


def perturbed_group(n_dist, seed=3):
    """The fixture rig, nudged off its optimum so bundle adjust has work to do."""
    cgroup = CameraGroup.load(FISHEYE_TOML)
    rng = np.random.default_rng(seed)
    p3d = rng.normal(size=(400, 3)) * 0.5
    p2d = cgroup.project(p3d) + rng.normal(scale=0.5, size=(5, 400, 2))

    for cam in cgroup.cameras:
        cam.rvec.data += torch.as_tensor(rng.normal(size=3) * 0.01)
        cam.dist.data += torch.as_tensor(rng.normal(size=4) * 0.02)

    cgroup.set_n_dist(n_dist)
    return cgroup, p2d


def test_defaults():
    assert Camera().get_n_dist() == 2
    assert FisheyeCamera().get_n_dist() == 2
    # a dist shorter than the default must clamp, not raise
    assert FisheyeCamera(dist=np.zeros(1)).get_n_dist() == 1


def test_dist_is_flattened():
    """dist used to keep a (5,1) shape, which makes dist.shape[0] a lie."""
    assert Camera(dist=np.zeros((5, 1))).dist.shape == (5,)


@pytest.mark.parametrize('n_dist,expected', [(0, []), (1, [0]), (2, [0, 1]),
                                             (5, [0, 1, 2, 3, 4])])
def test_gradient_mask(n_dist, expected):
    cam = Camera(matrix=K, n_dist=n_dist)
    assert dist_grad(cam) == expected


def test_n_dist_is_keyword_only():
    """extra_dist used to be positional #7. Without keyword-only, a stale
    Camera(..., True) would silently mean n_dist=1 and False would mean 0."""
    with pytest.raises(TypeError):
        Camera(np.eye(3), np.zeros(5), None, np.zeros(3), np.zeros(3), 'cam', 2)


@pytest.mark.parametrize('bad', [-1, 99])
def test_set_n_dist_rejects_out_of_range(bad):
    with pytest.raises(ValueError):
        Camera().set_n_dist(bad)


def test_set_n_dist_rejects_bool():
    # bool is an int subclass, so True would otherwise pass as n_dist=1
    with pytest.raises(TypeError):
        Camera().set_n_dist(True)


def test_lone_p1_warns():
    with pytest.warns(UserWarning):
        Camera().set_n_dist(3)   # p1 without p2


def test_set_distortions_clamps_down_only():
    cam = Camera(n_dist=5)
    cam.set_distortions(np.zeros(3))
    assert cam.get_n_dist() == 3
    # a longer vector must not silently free more coefficients
    cam.set_distortions(np.zeros(5))
    assert cam.get_n_dist() == 3


@pytest.mark.parametrize('n_dist,method', [(0, 'lm'), (1, 'lm'), (2, 'lm'), (2, 'adam')])
def test_bundle_adjust_freezes_exactly(n_dist, method):
    """LM (default) leaves frozen coefficients out of the solve; Adam relies on the gradient
    hooks plus weight_decay=0."""
    cgroup, p2d = perturbed_group(n_dist)
    before = [c.dist.detach().clone() for c in cgroup.cameras]

    cgroup.bundle_adjust(torch.as_tensor(p2d), max_nfev=300, verbose=False,
                         only_extrinsics=False, method=method)

    moved = 0.0
    for b, cam in zip(before, cgroup.cameras):
        after = cam.dist.detach()
        # frozen coefficients must be untouched bit for bit, not merely close
        assert torch.equal(after[n_dist:], b[n_dist:])
        if n_dist > 0:
            moved = max(moved, (after[:n_dist] - b[:n_dist]).abs().max().item())

    if n_dist > 0:
        assert moved > 1e-3, "free coefficients did not move, test proves nothing"


def test_bundle_adjust_iter_freezes_exactly():
    """bundle_adjust_iter restores state dicts between passes, which is where a
    naive freeze could resurrect a different value."""
    cgroup, p2d = perturbed_group(2)
    before = [c.dist.detach().clone() for c in cgroup.cameras]

    cgroup.bundle_adjust_iter(p2d, n_iters=3, n_samp_iter=100, n_samp_full=200,
                              verbose=False, rng=0)

    for b, cam in zip(before, cgroup.cameras):
        assert torch.equal(cam.dist.detach()[2:], b[2:])


def test_frozen_values_are_kept_not_zeroed():
    """Seeding known-good distortions and freezing them is the point."""
    cam = FisheyeCamera(matrix=K, dist=np.array([0.05, 0.01, 0.03, -0.02]), n_dist=1)
    cam.dist.grad = None
    cam.project(np.array([[0.1, 0.2, 3.0]]), return_grad=True).sum().backward()
    assert np.allclose(cam.dist.detach().numpy()[1:], [0.01, 0.03, -0.02])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_bundle_adjust_freezes_exactly_cuda():
    cgroup, p2d = perturbed_group(2)
    cgroup = cgroup.to('cuda')
    before = [c.dist.detach().cpu().clone() for c in cgroup.cameras]

    cgroup.bundle_adjust(torch.as_tensor(p2d, device='cuda'), max_nfev=300,
                         verbose=False, only_extrinsics=False)

    for b, cam in zip(before, cgroup.cameras):
        assert torch.equal(cam.dist.detach().cpu()[2:], b[2:])


@pytest.mark.parametrize('roundtrip', [copy.deepcopy,
                                       lambda c: pickle.loads(pickle.dumps(c))])
def test_mask_survives_deepcopy_and_pickle(roundtrip):
    """Parameter.__deepcopy__ / __reduce_ex__ drop backward hooks, so without
    __setstate__ the copy keeps n_dist while silently losing the freezing."""
    cam = Camera(matrix=K, n_dist=2)
    assert dist_grad(roundtrip(cam)) == [0, 1]


def test_copy_propagates_n_dist():
    cgroup = CameraGroup.load(FISHEYE_TOML)
    cgroup.set_n_dist(1)

    dup = cgroup.cameras[3].copy()
    assert isinstance(dup, FisheyeCamera)
    assert dup.get_n_dist() == 1
    assert dist_grad(dup) == [0]

    sub = cgroup.subset_cameras([0, 2, 4])
    assert sub.get_n_dists() == [1, 1, 1]


def test_copy_handles_non_default_dist_length():
    """load_state_dict is strict, so copy() used to raise for any dist that was
    not 5 long."""
    cam = Camera(dist=np.zeros(8), n_dist=2)
    dup = cam.copy()
    assert dup.dist.shape == (8,)
    assert dup.get_n_dist() == 2


def test_toml_roundtrip(tmp_path):
    cgroup = CameraGroup.load(FISHEYE_TOML)
    # the fixture predates n_dist, so it should come back as the default
    assert cgroup.get_n_dists() == [2] * 5

    cgroup.set_n_dist(1)
    out = str(tmp_path / 'out.toml')
    cgroup.dump(out)

    reloaded = CameraGroup.load(out)
    assert reloaded.get_n_dists() == [1] * 5
    assert all(d['n_dist'] == 1 for d in reloaded.get_dicts())


def test_group_set_n_dist_validates():
    cgroup = CameraGroup.load(FISHEYE_TOML)
    with pytest.raises(ValueError):
        cgroup.set_n_dist(9)   # fisheye cameras only have 4


def _board_views():
    board = Checkerboard(7, 5, square_length=0.03)
    objp = board.get_object_points()
    size = (1920, 1080)
    fk = np.array([[869.2167, 0.0, 959.5], [0.0, 869.2167, 539.5], [0.0, 0.0, 1.0]])
    truth = FisheyeCamera(matrix=fk, dist=np.array([0.1846, 0.0, 0.0, 0.0]), size=size)

    rng = np.random.default_rng(0)
    objs, imgs = [], []
    for _ in range(14):
        truth.set_rotation(rng.normal(scale=0.35, size=3))
        truth.set_translation(np.array([0, 0, 1.1]) + rng.normal(scale=0.28, size=3))
        p2d = truth.project(objp.reshape(-1, 3))
        if not (p2d.min() > 0 and p2d[:, 0].max() < size[0]
                and p2d[:, 1].max() < size[1]):
            continue
        objs.append(objp.astype('float32'))
        imgs.append(p2d.reshape(-1, 1, 2).astype('float32'))
    return objs, imgs, size


@pytest.mark.parametrize('n_dist', [1, 2, 4])
def test_init_intrinsics_respects_n_dist(n_dist):
    objs, imgs, size = _board_views()
    cam = FisheyeCamera(size=size, name='fish', n_dist=n_dist)
    init_camera_intrinsics(cam, objs, imgs, size)

    d = cam.dist.detach().numpy()
    assert abs(d[0] - 0.1846) < 1e-3          # k1 is recovered either way
    assert np.all(d[n_dist:] == 0)            # nothing seeded beyond n_dist


def test_init_intrinsics_keeps_frozen_values():
    """cv2's CALIB_FIX_K flags zero the coefficients they fix rather than
    holding them, and the pinhole path used to zero the whole vector."""
    objs, imgs, size = _board_views()

    cam = FisheyeCamera(size=size, name='fish',
                        dist=np.array([0.0, 0.0, 0.03, -0.02]), n_dist=2)
    init_camera_intrinsics(cam, objs, imgs, size)
    assert np.allclose(cam.dist.detach().numpy()[2:], [0.03, -0.02])

    camp = Camera(size=size, name='pin',
                  dist=np.array([0.0, 0.0, 0.0, 0.0, 0.07]), n_dist=2)
    init_camera_intrinsics(camp, objs, imgs, size)
    d = camp.dist.detach().numpy()
    assert d[4] == 0.07
    assert np.allclose(d[:4], 0)
