"""aniposelib.geometry camera construction / projection matrices and aniposelib.affine (round trips on the use-case rigs)."""

import json

import numpy as np
import pytest
import torch

import synth
from aniposelib import geometry as aio
from aniposelib.affine import AffineCamera, affine_from_points, triangulate_affine
from aniposelib.cameras import Camera, CameraGroup, FisheyeCamera

ALL = list(synth.RIGS)


def proj(cg, X):
    return np.asarray(cg.project(X))


def same_projection(a, b, X, atol=1e-6):
    assert a.get_names() == b.get_names()
    assert [type(c) for c in a.cameras] == [type(c) for c in b.cameras]
    for ca, cb in zip(a.cameras, b.cameras):
        assert list(ca.get_size()) == list(cb.get_size())
    np.testing.assert_allclose(proj(a, X), proj(b, X), rtol=0, atol=atol)


def with_skew(cg, skew=0.7):
    cg = cg.copy()
    for c in cg.cameras:
        K = synth.to_np(c.get_camera_matrix()).copy()
        K[0, 1] = skew
        c.set_camera_matrix(K)
    return cg


@pytest.fixture(scope='module')
def rigs():
    return {r: synth.load_rig(r) for r in ALL}


@pytest.fixture(scope='module')
def points(rigs):
    return {r: synth.random_points(cg, 40, seed=3) for r, cg in rigs.items()}


# ------------------------------------------------------------------ basics

@pytest.mark.parametrize('rig', ALL)
def test_camera_from_extrinsics_mat(rig, rigs, points):
    cg = rigs[rig]
    cams = [aio.camera_from_extrinsics_mat(c.get_camera_matrix(), c.get_extrinsics_mat(), c.get_distortions(),
                                           c.get_size(), c.get_name(), fisheye=c.fisheye)
            for c in cg.cameras]
    same_projection(cg, CameraGroup(cams), points[rig], atol=1e-8)


def test_camera_from_extrinsics_mat_checks():
    E = np.eye(4)
    E[:3, :3] = np.diag([1.0, 1.0, -1.0])
    with pytest.raises(ValueError, match='improper'):
        aio.camera_from_extrinsics_mat(np.eye(3), E)
    with pytest.raises(ValueError):
        aio.camera_from_extrinsics_mat(np.eye(3), np.eye(3))
    cam = aio.camera_from_extrinsics_mat(np.eye(3), np.eye(4)[:3], offset=(3, 4), name='a')
    assert cam.get_name() == 'a' and np.allclose(synth.to_np(cam.get_offset()), [3, 4])


def test_reindex_points():
    p = np.arange(3 * 4 * 2, dtype=float).reshape(3, 4, 2)
    out = aio.reindex_points(p, ['a', 'b', 'c'], ['c', 'x', 'a'])
    np.testing.assert_array_equal(out[0], p[2])
    np.testing.assert_array_equal(out[2], p[0])
    assert np.all(np.isnan(out[1]))
    with pytest.raises(KeyError):
        aio.reindex_points(p, ['a', 'b', 'c'], ['c', 'x'], missing='raise')
    with pytest.raises(ValueError):
        aio.reindex_points(p, ['a', 'b'], ['a'])
    t = aio.reindex_points(torch.as_tensor(p), ['a', 'b', 'c'], ['b'])
    assert isinstance(t, torch.Tensor) and torch.equal(t[0], torch.as_tensor(p[1]))
    cg = CameraGroup.from_names(['a', 'b', 'c'])
    np.testing.assert_array_equal(aio.reindex_points(p, cg, ['b', 'a']), p[[1, 0]])


# ------------------------------------------------------------------ MATLAB


# ------------------------------------------------------------------ OpenCV YAML


# ------------------------------------------------------------------ projection matrix / DLT

def test_projection_matrix_roundtrip(rigs, points):
    cg = with_skew(rigs['motobs_7cam'])
    for c in cg.cameras:
        c.zero_distortions()
    X = points['motobs_7cam']
    for c in cg.cameras:
        P = aio.to_projection_matrix(c)
        for s in (1.0, -3.5):  # scale and sign of P do not matter
            c2 = aio.from_projection_matrix(s * P, size=c.get_size(), name=c.get_name())
            np.testing.assert_allclose(synth.to_np(c2.get_camera_matrix()),
                                       synth.to_np(c.get_camera_matrix()), rtol=1e-9, atol=1e-7)
            np.testing.assert_allclose(np.asarray(c2.project(X)), np.asarray(c.project(X)), atol=1e-6)
        L = aio.to_dlt(c)
        assert L.shape == (11,)
        c3 = aio.from_dlt(L)
        np.testing.assert_allclose(np.asarray(c3.project(X)), np.asarray(c.project(X)), atol=1e-6)
        h = np.c_[X, np.ones(len(X))] @ np.r_[L, 1].reshape(3, 4).T
        np.testing.assert_allclose(h[:, :2] / h[:, 2:], np.asarray(c.project(X)), atol=1e-6)


def test_projection_matrix_singular():
    with pytest.raises(ValueError):
        aio.decompose_projection(np.zeros((3, 4)))


# ------------------------------------------------------------------ tracktail metadata.yaml


# ------------------------------------------------------------------ JSON rig


# ------------------------------------------------------------------ affine cameras

def _affine_rig(seed=0, n=4):
    rng = np.random.default_rng(seed)
    cams = []
    for i in range(n):
        A = rng.normal(size=(2, 3)) * 50
        b = rng.uniform(200, 400, size=2)
        cams.append(AffineCamera(A, b, size=(640, 480), name='a{}'.format(i)))
    return cams


def test_affine_projection_and_triangulation():
    cams = _affine_rig()
    rng = np.random.default_rng(1)
    X = rng.normal(size=(30, 3))
    uv = np.stack([c.project(X) for c in cams])
    uv[0, :5] = np.nan
    Xh = triangulate_affine(cams, uv)
    np.testing.assert_allclose(Xh, X, atol=1e-9)
    tX = triangulate_affine(cams, torch.as_tensor(uv))
    assert isinstance(tX, torch.Tensor)
    one = uv.copy()
    one[1:, 0] = np.nan
    assert np.all(np.isnan(triangulate_affine(cams, one)[0]))
    fit = affine_from_points(X, uv[1])
    np.testing.assert_allclose(synth.to_np(fit.A), synth.to_np(cams[1].A), atol=1e-8)
    back = AffineCamera.from_dict(json.loads(json.dumps(cams[2].get_dict())))
    np.testing.assert_allclose(back.project(X), cams[2].project(X))


def test_affine_from_dlt_and_pinhole_approximation():
    cam = _affine_rig(seed=2, n=1)[0]
    P = cam.get_projection_matrix()
    out = aio.from_projection_matrix(2.0 * P)
    assert isinstance(out, AffineCamera)
    L = aio.to_dlt(cam)
    assert np.allclose(L[8:11], 0) and isinstance(aio.from_dlt(L), AffineCamera)
    rng = np.random.default_rng(3)
    X = rng.normal(size=(50, 3))
    np.testing.assert_allclose(aio.from_dlt(L).project(X), cam.project(X), atol=1e-9)
    pin = cam.to_pinhole(X, max_error_px=0.5)
    err = np.linalg.norm(np.asarray(pin.project(X)) - cam.project(X), axis=1)
    assert err.max() < 0.5
    assert abs(synth.to_np(pin.get_camera_matrix())[0, 1]) > 0  # skew kept
