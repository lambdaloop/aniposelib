"""1.0-wire: CameraGroup delegation to the packed geometry / triangulation kernels, packed(),
as_tensors(), triangulate(method=, extrinsics=, min_cams=), triangulate_robust, the Rodrigues
gradient at zero, and get_rows_videos(frame_indices=, n_workers=, detector=)."""

import numpy as np
import pytest
import torch

import synth
from aniposelib import geometry as G
from aniposelib.cameras import Camera, CameraGroup, FisheyeCamera, rodrigues
from aniposelib.triangulation import TriangulationResult

RIGS = list(synth.RIGS)


def skewed_offset(cg):
    """Add skew and crop offsets so the parity checks cover them."""
    for i, c in enumerate(cg.cameras):
        with torch.no_grad():
            c.matrix[0, 1] = 0.25 * (i + 1)
        c.set_offset([2.0 * i + 1, -1.5 * i])
    return cg


def percam_project(cg, X):
    with torch.no_grad():
        return torch.stack([c._project(torch.as_tensor(X)) for c in cg.cameras]).numpy()


def percam_undistort(cg, p):
    with torch.no_grad():
        return torch.stack([c._undistort_points(torch.as_tensor(p[i]), n_iter=200)
                            if not c.fisheye else c._undistort_points(torch.as_tensor(p[i]))
                            for i, c in enumerate(cg.cameras)]).numpy()


class CustomCamera(Camera):
    """A subclass with its own projection: must use the per-camera fallback."""
    def _project(self, points):
        return super()._project(points) + 0.5


@pytest.mark.parametrize('rig', RIGS)
def test_group_methods_match_per_camera_path(rig):
    cg = skewed_offset(synth.load_rig(rig))
    X = synth.random_points(cg, 300, spread=0.3, min_views=1, seed=1)
    ref = percam_project(cg, X)
    vis = np.isfinite(synth.project(cg, X)).all(-1)
    out = cg.project(X)
    # the per-camera pinhole path divides by z + 1e-10, the kernel by z exactly: ~1e-10/z
    # relative (up to ~2e-8 for the closest points of the metre-scale outdoor rig)
    rel = np.abs(out - ref).max(-1) / np.maximum(1, np.abs(ref).max(-1))
    assert np.all(rel[vis] < 1e-7)
    assert np.median(rel[vis]) < 1e-9
    err = cg.reprojection_error(X, ref)
    assert np.all(np.abs(err[vis]) < 1e-5)
    und = cg.undistort_points(ref)
    und_ref = percam_undistort(cg, ref)
    assert np.nanmax(np.abs(und - und_ref)[vis]) < 1e-9


@pytest.mark.parametrize('rig', ['motobs_7cam', 'fisheye_5cam'])
def test_triangulate_linear_default_and_svd(rig):
    cg = skewed_offset(synth.load_rig(rig))
    X = synth.random_points(cg, 200, seed=2)
    p2d = synth.drop_views(synth.project(cg, X), 0.2, seed=3)
    ok = synth.n_views(p2d) >= 2
    sc = synth.rig_scale(cg)
    lin = cg.triangulate(p2d)
    svd = cg.triangulate(p2d, method='svd')
    assert np.array_equal(np.isfinite(lin).all(-1), ok)
    assert np.nanmax(np.linalg.norm(lin - X, axis=-1)) / sc < 1e-8
    assert np.nanmax(np.linalg.norm(svd - X, axis=-1)) / sc < 1e-8
    noisy = synth.add_noise(p2d, 0.5, seed=4)
    d = np.linalg.norm(cg.triangulate(noisy) - cg.triangulate(noisy, method='svd'), axis=-1) / sc
    assert np.nanmax(d) < 1e-3
    with pytest.raises(ValueError):
        cg.triangulate(p2d, method='nope')


def test_triangulate_shapes_min_cams_and_weights():
    cg = synth.load_rig('allen_7cam')
    X = synth.random_points(cg, 40, seed=5)
    p2d = synth.project(cg, X)
    one = cg.triangulate(p2d[:, 0])
    assert one.shape == (3,) and np.allclose(one, X[0], atol=1e-6)
    p4 = p2d.reshape(7, 8, 5, 2)
    assert cg.triangulate(p4).shape == (8, 5, 3)
    nv = synth.n_views(p2d)
    out = cg.triangulate(p2d, min_cams=int(nv.max()) + 1)
    assert np.all(np.isnan(out))
    w = np.ones(p2d.shape[:2])
    w[3] = 0
    a = cg.triangulate(p2d, weights=w)
    b = cg.subset_cameras([0, 1, 2, 4, 5, 6]).triangulate(p2d[[0, 1, 2, 4, 5, 6]])
    assert np.allclose(a, b, equal_nan=True, atol=1e-9)


def test_triangulate_moving_extrinsics():
    cg = synth.load_rig('motobs_7cam')
    X = synth.random_points(cg, 30, seed=6)
    C = 7
    E = synth.to_np(cg._extrinsics_stack())
    # rig moves: every point has its own rigid shift of the whole rig
    rng = np.random.default_rng(0)
    shifts = rng.normal(size=(30, 3)) * synth.rig_scale(cg) * 0.05
    Ep = np.repeat(E[:, None], 30, axis=1).copy()
    Ep[..., :3, 3] -= np.einsum('cij,nj->cni', E[:, :3, :3], shifts)   # world shifted by +shift
    P = cg.packed()
    p2d = G.project(P, X, extrinsics=Ep)[0]
    for method in ('linear', 'svd'):
        out = cg.triangulate(p2d, extrinsics=Ep, method=method)
        assert np.nanmax(np.abs(out - X)) < 1e-6 * synth.rig_scale(cg)
    # per-frame [C,T,4,4] with points [C,T,J,2]
    Et = Ep[:, :6]
    pts = G.project(P, X[:30].reshape(6, 5, 3), extrinsics=Et)[0]
    out = cg.triangulate(pts, extrinsics=Et)
    assert out.shape == (6, 5, 3)
    assert np.nanmax(np.abs(out - X.reshape(6, 5, 3))) < 1e-6 * synth.rig_scale(cg)


def test_fallback_for_custom_cameras():
    cg = synth.load_rig('allen_7cam')
    cams = [CustomCamera.from_dict(c.get_dict()) for c in cg.cameras]
    cc = CameraGroup(cams)
    assert not cc._packable() and cg._packable()
    X = synth.random_points(cg, 20, seed=7)
    assert np.allclose(cc.project(X), cg.project(X) + 0.5, atol=1e-6)
    p2d = cg.project(X)
    assert np.allclose(cc.triangulate(p2d), cg.triangulate(p2d), atol=1e-8)
    out, picked, p2, err = cc.triangulate_ransac(p2d)   # 0.9 subset search
    assert out.shape == (20, 3) and picked.shape == (7, 20, 1)
    with pytest.raises(NotImplementedError):
        cc.triangulate_robust(p2d)


def test_packed_roundtrip_and_grad():
    cg = skewed_offset(synth.load_rig('fisheye_5cam'))
    P = cg.packed()
    assert P.K.shape == (5, 3, 3) and P.model.tolist() == [1] * 5
    P2 = P.clone()
    P2.tvec = P2.tvec + 0.01
    P2.K[:, 0, 0] *= 1.01
    cg2 = cg.copy()
    cg2.load_packed_(P2)
    assert np.allclose(synth.to_np(cg2.get_translations()), synth.to_np(P2.tvec))
    assert np.allclose(cg2.cameras[0].get_camera_matrix().detach().numpy(), P2.K[0].numpy())
    # attached packing: gradients reach the camera parameters
    Pg = cg.packed(detach=False)
    X = torch.as_tensor(synth.random_points(cg, 10, seed=8))
    G.project(Pg, X, return_grad=True)[0].sum().backward()
    assert cg.cameras[0].rvec.grad is not None and torch.all(torch.isfinite(cg.cameras[0].rvec.grad))


def test_as_tensors_contract():
    cg = skewed_offset(synth.load_rig('allen_7cam'))
    d = cg.as_tensors()
    C = 7
    assert d['K'].shape == (C, 3, 3) and d['dist'].shape == (C, 5)
    assert d['ext'].shape == (C, 4, 4) and d['ext_inv'].shape == (C, 4, 4)
    assert d['center'].shape == (C, 3) and d['size'].shape == (C, 2) and d['offset'].shape == (C, 2)
    assert d['type'] == ['pinhole'] * C and d['moving'] is False
    E0 = cg.cameras[0].get_extrinsics_mat().detach().numpy()
    assert np.allclose(d['ext'][0].numpy(), E0)
    assert np.allclose(d['center'][0].numpy(), cg.cameras[0].get_center_world().detach().numpy())
    assert np.allclose(d['offset'].numpy(), synth.to_np(cg.get_offsets()))
    moving = {cg.cameras[2].get_name(): np.repeat(E0[None], 4, 0)}
    dm = cg.as_tensors(extrinsics=moving, dtype=torch.float32)
    assert dm['ext'].shape == (C, 4, 4, 4) and dm['center'].shape == (C, 4, 3) and dm['moving']
    assert dm['K'].dtype == torch.float32
    assert np.allclose(dm['ext'][2, 3].numpy(), E0, atol=1e-6)
    fd = synth.load_rig('fisheye_5cam').as_tensors()
    assert fd['dist'].shape == (5, 4) and fd['type'] == ['fisheye'] * 5


def test_triangulate_robust_wrapper():
    true = synth.load_rig('motobs_7cam')
    p3d, _ = synth.skeleton(true, 40, model=synth.MOUSE, seed=1)
    p2d = synth.add_noise(synth.project(true, p3d), 0.5, seed=2)
    bad, _ = synth.add_decoy(p2d, cam=3, frac=0.6, cgroup=true, seed=3)
    C = 7
    flat = bad.reshape(C, -1, 2)
    res = true.triangulate_robust(flat, method='msac', tau=8.0)
    assert isinstance(res, TriangulationResult)
    assert res.p3d.shape == (flat.shape[1], 3) and res.errors.shape == flat.shape[:2]
    gt = p3d.reshape(-1, 3)
    sc = synth.rig_scale(true)
    robust = np.nanmedian(np.linalg.norm(res.p3d - gt, axis=-1)) / sc
    plain = np.nanmedian(np.linalg.norm(true.triangulate(flat) - gt, axis=-1)) / sc
    assert robust < plain / 3
    p3, errs = true.triangulate_robust(torch.as_tensor(flat), method='irls')
    assert isinstance(p3, torch.Tensor) and errs.shape == flat.shape[:2]
    # ransac keeps its 4-tuple, picked = inliers
    out, picked, p2, err = true.triangulate_ransac(flat, tau=8.0)
    assert picked.shape == (C, flat.shape[1], 1)
    assert np.array_equal(np.isnan(p2[..., 0]), ~picked[..., 0])


def test_rodrigues_gradient_at_zero():
    r = torch.zeros(3, dtype=torch.float64, requires_grad=True)
    R = rodrigues(r)
    assert torch.allclose(R, torch.eye(3, dtype=torch.float64))
    # dR/dr at 0 is the skew generator: d R[2,1] / d r[0] = 1, d R[1,2] / d r[0] = -1
    (R[2, 1] - R[1, 2]).backward()
    assert torch.allclose(r.grad, torch.tensor([2.0, 0, 0], dtype=torch.float64))
    # matches finite differences near zero, and cv2 away from it
    import cv2
    for v in ([1e-9, -2e-9, 3e-9], [0.3, -0.2, 0.1]):
        rv = np.array(v)
        assert np.allclose(rodrigues(torch.as_tensor(rv)).numpy(), cv2.Rodrigues(rv)[0], atol=1e-12)
    # a camera at exactly zero rotation gets a rotation gradient through the group projection
    cg = synth.load_rig('allen_7cam')
    cg.cameras[0].set_rotation([0.0, 0.0, 0.0])
    X = torch.as_tensor(synth.random_points(cg, 10, min_views=1, seed=9))
    cg.project(X, return_grad=True)[0].sum().backward()
    assert cg.cameras[0].rvec.grad.abs().sum() > 0


class FakeBoard:
    detector_name = 'opencv'
    validate = False
    detector_kwargs = {}

    def __init__(self):
        self.calls = []

    def detect_video(self, vidname, prefix=None, progress=False, frame_indices=None, skip=20):
        self.calls.append((vidname, prefix, None if frame_indices is None else list(frame_indices)))
        if vidname == 'broken.mp4':
            raise IOError('cannot open')
        return [{'framenum': (prefix, 0), 'video': vidname}]


def test_get_rows_videos_frame_indices_and_workers():
    cg = CameraGroup([Camera(name='a'), Camera(name='b')])
    videos = [['a0.mp4', 'a1.mp4'], ['b0.mp4']]
    board = FakeBoard()
    rows = cg.get_rows_videos(videos, board, verbose=False, frame_indices=[3, 1, 7])
    assert [len(r) for r in rows] == [2, 1]
    assert all(c[2] == [3, 1, 7] for c in board.calls)
    board = FakeBoard()
    cg.get_rows_videos(videos, board, verbose=False, frame_indices=[[[1], [2]], [[5]]])
    assert {c[0]: c[2] for c in board.calls} == {'a0.mp4': [1], 'a1.mp4': [2], 'b0.mp4': [5]}
    board = FakeBoard()
    cg.get_rows_videos(videos, board, verbose=False, frame_indices={'b': [4], 'a1.mp4': [9]})
    assert {c[0]: c[2] for c in board.calls} == {'a0.mp4': None, 'a1.mp4': [9], 'b0.mp4': [4]}
    board = FakeBoard()
    seq = cg.get_rows_videos(videos + [], board, verbose=False)
    par = cg.get_rows_videos(videos, FakeBoard(), verbose=False, n_workers=3)
    assert seq == par
    broken = cg.get_rows_videos([['broken.mp4'], ['b0.mp4']], FakeBoard(), verbose=False)
    assert [len(r) for r in broken] == [0, 1]


def test_get_rows_videos_detector_selection():
    from aniposelib.boards import CharucoBoard
    from aniposelib.cameras import _board_with_detector

    class Det:
        def detect(self, image):
            return None, None

    board = CharucoBoard(5, 6, 1.0, 0.8)
    b2 = _board_with_detector(board, Det())
    assert b2.detector_name == 'Det' and b2.validate and board.detector_name == 'opencv'
    assert board._detector is None
    with pytest.raises(ValueError):
        _board_with_detector(board, 'nope')
