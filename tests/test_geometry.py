"""geometry.py: packed camera kernels (plan 01 C1). Parity with the per-camera CameraGroup path
on every use-case rig, independent checks against OpenCV, and skew/offset/invertibility units."""

import json

import cv2
import numpy as np
import pytest
import torch

import synth
from aniposelib import geometry as G
from aniposelib.cameras import Camera, CameraGroup, FisheyeCamera

RIGS = list(synth.RIGS)


def visible_points(cg, n=1000, seed=1):
    X = synth.random_points(cg, n, spread=0.3, min_views=1, seed=seed)
    return X, synth.project(cg, X)


@pytest.mark.parametrize('rig', RIGS)
def test_project_parity_with_cameragroup(rig):
    cg = synth.load_rig(rig)
    P = G.pack(cg)
    X, p2d = visible_points(cg)
    ref = synth.to_np(cg.project(X))
    uv, z = G.project(P, X)
    assert uv.shape == ref.shape and z.shape == ref.shape[:2]
    vis = np.isfinite(p2d).all(-1)
    d = np.linalg.norm(uv - ref, axis=-1)
    if all(c.fisheye for c in cg.cameras):
        assert np.all(d[vis] <= 1e-9)
        return
    # the per-camera pinhole path divides by (z + 1e-10), the kernel by z exactly (a ~1e-8 px
    # difference on metre-scale rigs). Feed the kernel's distortion + K the reference's
    # normalised coordinates: then they agree to 1e-9 px.
    E = P.extrinsics().numpy()
    pc = np.einsum('cij,nj->cni', E[:, :3, :3], X) + E[:, None, :3, 3]
    xy_ref = pc[..., :2] / (pc[..., 2:] + 1e-10)
    d_model = np.linalg.norm(G.distort(P, xy_ref) - ref, axis=-1)
    assert np.all(d_model[vis] <= 1e-9)
    assert np.all(d[vis] <= 1e-5)


@pytest.mark.parametrize('rig', RIGS)
def test_undistort_parity_with_cameragroup(rig):
    cg = synth.load_rig(rig)
    P = G.pack(cg)
    X, p2d = visible_points(cg)
    ref = synth.to_np(cg._undistort_points_batch(torch.as_tensor(p2d)))
    und = G.undistort(P, p2d)
    assert np.array_equal(np.isfinite(und), np.isfinite(ref))
    # where the reference (15 fixed-point iterations) has converged the two agree to 1e-9; where it
    # has not, the kernel is the more accurate one: bound by the reference's own residual
    f = P.K[:, 0, 0].numpy()[:, None]
    ref_res = np.linalg.norm(G.distort(P, ref) - p2d, axis=-1) / f
    d = np.linalg.norm(und - ref, axis=-1)
    m = np.isfinite(d)
    assert np.all(d[m] <= 1e-9 + 2 * ref_res[m])
    own_res = np.linalg.norm(G.distort(P, und) - p2d, axis=-1)
    assert np.nanmax(own_res) < 1e-8


def test_project_matches_opencv_rational_and_fisheye():
    rng = np.random.default_rng(0)
    K = np.array([[900.0, 0, 640], [0, 880, 360], [0, 0, 1]])
    rvec, tvec = rng.normal(size=3) * 0.2, np.array([0.1, -0.2, 3.0])
    X = rng.uniform(-1, 1, size=(500, 3))
    d8 = np.array([-0.2, 0.05, 1e-3, -2e-3, 0.01, 0.02, -0.01, 0.003])
    cam = Camera(matrix=K, dist=d8, rvec=rvec, tvec=tvec, size=(1280, 720))
    P = G.pack([cam])
    uv, _ = G.project(P, X)
    ref, _ = cv2.projectPoints(X.reshape(-1, 1, 3), rvec, tvec, K, d8)
    np.testing.assert_allclose(uv[0], ref.reshape(-1, 2), atol=1e-8)
    und = G.undistort(P, uv)
    ref_u = cv2.undistortPointsIter(uv[0].reshape(-1, 1, 2), K, d8, None, None,
                                    (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 100, 1e-15))
    np.testing.assert_allclose(und[0], ref_u.reshape(-1, 2), atol=1e-9)

    d4 = np.array([0.05, -0.01, 0.002, -0.0005])
    fcam = FisheyeCamera(matrix=K, dist=d4, rvec=rvec, tvec=tvec, size=(1280, 720))
    P = G.pack([fcam])
    uv, _ = G.project(P, X)
    ref, _ = cv2.fisheye.projectPoints(X.reshape(-1, 1, 3), rvec, tvec, K, d4)
    np.testing.assert_allclose(uv[0], ref.reshape(-1, 2), atol=1e-8)


def _rand_cam(rng, fisheye=False):
    K = np.array([[800.0, 7.5, 500], [0, 790, 400], [0, 0, 1]])  # skewed
    dist = np.array([0.03, -0.01, 0.001, 0.0005]) if fisheye else np.array([-0.15, 0.04, 1e-3, -1e-3, 0.0])
    cls = FisheyeCamera if fisheye else Camera
    cam = cls(matrix=K, dist=dist, rvec=rng.normal(size=3) * 0.1, tvec=np.array([0.0, 0, 4]), size=(1000, 800))
    return cam


@pytest.mark.parametrize('fisheye', [False, True])
def test_skew_and_offset_direct_formula(fisheye):
    rng = np.random.default_rng(1)
    cam = _rand_cam(rng, fisheye)
    cam.offset = [12.0, -7.0]
    P = G.pack([cam])
    np.testing.assert_array_equal(P.offset.numpy(), [[12.0, -7.0]])
    X = rng.uniform(-1, 1, size=(200, 3))
    uv, z = G.project(P, X)
    # direct formula
    R = cv2.Rodrigues(cam.rvec.detach().numpy())[0]
    pc = X @ R.T + cam.tvec.detach().numpy()
    x, y = pc[:, 0] / pc[:, 2], pc[:, 1] / pc[:, 2]
    if fisheye:
        k = cam.dist.detach().numpy()
        r = np.hypot(x, y); th = np.arctan(r)
        thd = th * (1 + k[0] * th**2 + k[1] * th**4 + k[2] * th**6 + k[3] * th**8)
        xd, yd = x * thd / r, y * thd / r
    else:
        k1, k2, p1, p2, k3 = cam.dist.detach().numpy()
        r2 = x * x + y * y
        rad = 1 + k1 * r2 + k2 * r2**2 + k3 * r2**3
        xd = x * rad + 2 * p1 * x * y + p2 * (r2 + 2 * x * x)
        yd = y * rad + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y
    K = cam.matrix.detach().numpy()
    u = K[0, 0] * xd + K[0, 1] * yd + K[0, 2] - 12.0
    v = K[1, 1] * yd + K[1, 2] + 7.0
    np.testing.assert_allclose(uv[0], np.stack([u, v], -1), atol=1e-9)
    np.testing.assert_allclose(z[0], pc[:, 2], atol=1e-12)
    # undistort inverts it, including skew and offset
    und = G.undistort(P, uv)
    np.testing.assert_allclose(und[0], np.stack([x, y], -1), atol=1e-10)


def test_mixed_models_in_one_pack():
    rng = np.random.default_rng(2)
    cams = [_rand_cam(rng, False), _rand_cam(rng, True), _rand_cam(rng, False)]
    P = G.pack(cams)
    assert P.model_names() == ['brown', 'kb', 'brown']
    X = rng.uniform(-1, 1, size=(100, 3))
    uv, _ = G.project(P, X)
    for i, c in enumerate(cams):
        np.testing.assert_allclose(uv[i], G.project(G.pack([c]), X)[0][0], atol=1e-12)
    und = G.undistort(P, uv)
    np.testing.assert_allclose(G.distort(P, und), uv, atol=1e-8)


def test_check_invertible_brown_and_kb():
    K = np.array([[500.0, 0, 500], [0, 500, 500], [0, 0, 1]])
    # strong barrel distortion: the radial map folds at r^2 = 1/(3*0.4) ~ 0.83
    cam = Camera(matrix=K, dist=[-0.4, 0, 0, 0, 0], size=(1000, 1000))
    P = G.pack([cam])
    # r_d = r (1 - 0.4 r^2) peaks at r = sqrt(1/1.2): larger distorted radii have no preimage
    rmax = np.sqrt(1 / 1.2)
    rdmax = rmax * (1 - 0.4 * rmax ** 2)
    rd = np.linspace(0, 1.0, 80)
    uv = np.stack([500 + 500 * rd, np.full_like(rd, 500)], -1)[None]
    plain = G.undistort(P, uv)
    checked = G.undistort(P, uv, check_invertible=True)
    assert np.all(np.isfinite(plain))            # old behaviour: always a (garbage) answer
    assert np.all(np.isnan(checked[0, rd > rdmax + 1e-3, 0]))
    ok = rd < rdmax - 0.02
    assert np.all(np.isfinite(checked[0, ok]))
    np.testing.assert_allclose(G.distort(P, checked)[0, ok], uv[0, ok], atol=1e-8)
    # kb with a non-monotone theta map: no solution past its maximum -> NaN even without the check
    fcam = FisheyeCamera(matrix=K, dist=[-0.3, 0, 0, 0], size=(1000, 1000))
    P = G.pack([fcam])
    th_d = np.linspace(0.05, 1.5, 50)
    uv = np.stack([500 + 500 * th_d, np.full_like(th_d, 500)], -1)[None]
    out = G.undistort(P, uv)
    thmax = np.sqrt(1 / 0.9)  # d/dth th(1 - 0.3 th^2) = 0
    assert np.all(np.isnan(out[0, th_d > thmax * (1 - 0.3 * thmax**2) + 1e-6]))
    good = th_d < 0.4
    assert np.all(np.isfinite(G.undistort(P, uv, check_invertible=True)[0, good]))


def test_pack_unpack_roundtrip_and_to_dict():
    cg = synth.load_rig('outdoor_12cam')
    P = G.pack(cg)
    assert P.n_cams == 12 and P.dist.shape == (12, 8) and P.names == cg.get_names()
    json.dumps(P.to_dict())
    cg2 = synth.perturb_extrinsics(cg, rot_deg=3, trans_frac=0.1, fix_first=False, seed=3)
    G.unpack_into(cg2, P)
    for a, b in zip(cg.cameras, cg2.cameras):
        for k in ['matrix', 'dist', 'rvec', 'tvec']:
            np.testing.assert_allclose(getattr(a, k).detach().numpy(), getattr(b, k).detach().numpy(), atol=1e-12)
    # via ext: rvec recovered through cv2.Rodrigues
    P2 = P.with_extrinsics(P.extrinsics())
    P2.rvec = None
    G.unpack_into(cg2, P2)
    np.testing.assert_allclose(G.pack(cg2).R.numpy(), P.R.numpy(), atol=1e-10)
    sub = P.subset([3, 1])
    assert sub.names == [P.names[3], P.names[1]]
    np.testing.assert_array_equal(sub.K.numpy(), P.K.numpy()[[3, 1]])


def test_pack_gradients_flow_to_cameras():
    cg = synth.load_rig('motobs_7cam')
    X = synth.random_points(cg, 50, seed=1)
    P = G.pack(cg, detach=False)
    uv, _ = G.project(P, torch.as_tensor(X), return_grad=True)
    uv.nan_to_num().sum().backward()
    assert cg.cameras[2].rvec.grad is not None and torch.any(cg.cameras[2].matrix.grad != 0)


def test_return_types():
    cg = synth.load_rig('fisheye_5cam')
    P = G.pack(cg)
    X = synth.random_points(cg, 10, seed=1)
    uv, z = G.project(P, X)
    assert isinstance(uv, np.ndarray) and isinstance(z, np.ndarray)
    uvt, _ = G.project(P, torch.as_tensor(X))
    assert isinstance(uvt, torch.Tensor) and not uvt.requires_grad
    Xg = torch.as_tensor(X).requires_grad_(True)
    e = G.reproj_err(P, Xg, uvt, return_grad=True)
    assert e.requires_grad
    assert isinstance(G.undistort(P, uv), np.ndarray)


def test_per_frame_extrinsics_project():
    cg = synth.load_rig('allen_7cam')
    P = G.pack(cg)
    T = 4
    rng = np.random.default_rng(0)
    rv = P.rvec[:, None] + torch.as_tensor(rng.normal(size=(7, T, 3)) * 0.01)
    tv = P.tvec[:, None] + torch.as_tensor(rng.normal(size=(7, T, 3)))
    E = G.make_extrinsics(rv, tv)                          # [C,T,4,4]
    X = synth.random_points(cg, T * 5, seed=2).reshape(T, 5, 3)
    uv, z = G.project(P, X, extrinsics=E)
    assert uv.shape == (7, T, 5, 2)
    for t in range(T):
        np.testing.assert_allclose(uv[:, t], G.project(P, X[t], extrinsics=E[:, t])[0], atol=1e-9)


def test_project_jacobian_matches_autograd():
    rng = np.random.default_rng(3)
    cams = [_rand_cam(rng, False), _rand_cam(rng, True)]
    P = G.pack(cams)
    X = torch.as_tensor(rng.uniform(-0.5, 0.5, size=(20, 3)))
    _, _, J = G.project_jacobian(P, X)
    Ja = torch.autograd.functional.jacobian(lambda x: G._project_flat(P, x, P.extrinsics())[0], X)
    Ja = torch.einsum('cnkmj->cnkj', Ja * torch.eye(20, dtype=X.dtype)[None, :, None, :, None])
    np.testing.assert_allclose(J.numpy(), Ja.numpy(), atol=1e-7)


def test_device_parity(device):
    cg = synth.load_rig('outdoor_9cam')
    P = G.pack(cg)
    X, p2d = visible_points(cg, 200)
    uv, _ = G.project(P.to(device), torch.as_tensor(X, device=device))
    assert uv.device.type == device.type
    np.testing.assert_allclose(uv.cpu().numpy(), G.project(P, X)[0], atol=1e-9)
    und = G.undistort(P.to(device), torch.as_tensor(p2d, device=device))
    np.testing.assert_allclose(und.cpu().numpy(), G.undistort(P, p2d), atol=1e-12)
