"""triangulation.py: block DLT, robust triangulation and the multi-candidate kernel
(plan 01 C2/C3), on the synthetic use-case rigs."""

import json
import time

import numpy as np
import pytest
import torch

import synth
from aniposelib import geometry as G
from aniposelib import triangulation as T

METHODS = ['msac', 'irls', 'reject']


def rel(cg, est, p3d):
    return np.linalg.norm(synth.to_np(est).reshape(p3d.shape) - p3d, axis=-1) / synth.rig_scale(cg)


@pytest.fixture(scope='module')
def outdoor():
    true = synth.load_rig('outdoor_12cam')
    p3d = synth.trajectory(true, 1500, seed=2)
    p2d = synth.add_noise(synth.project(true, p3d), 0.5, seed=3)
    return true, G.pack(true), p3d, p2d


@pytest.fixture(scope='module')
def motobs():
    true = synth.load_rig('motobs_7cam')
    p3d, _ = synth.skeleton(true, 80, model=synth.MOUSE, seed=1)
    p2d = synth.add_noise(synth.project(true, p3d), 0.7, seed=2)
    p2d_decoy, frames = synth.add_decoy(p2d, cam=3, frac=0.6, cgroup=true, seed=3)
    return true, G.pack(true), p3d, p2d, p2d_decoy, frames


# ------------------------------------------------------------------ linear

@pytest.mark.parametrize('rig', list(synth.RIGS))
def test_linear_matches_cameragroup_triangulate(rig):
    cg = synth.load_rig(rig)
    P = G.pack(cg)
    X = synth.random_points(cg, 300, seed=1)
    p2d = synth.drop_views(synth.project(cg, X), 0.2, seed=2)
    ok = synth.n_views(p2d) >= 2
    ref = synth.to_np(cg.triangulate(p2d))
    lin = T.triangulate_linear(P, p2d)
    assert np.array_equal(np.isfinite(lin).all(-1), ok)
    # noiseless: both are exact; agreement to ~1e-9 of rig scale
    assert np.nanmax(rel(cg, lin, ref)) < 1e-8
    assert np.nanmax(rel(cg, lin, X)) < 1e-8


def test_linear_weights_and_blocks(motobs):
    true, P, p3d, p2d, _, _ = motobs
    flat = synth.project(true, p3d).reshape(7, -1, 2)   # noiseless: weighted SVD and block DLT agree
    rng = np.random.default_rng(0)
    w = rng.uniform(0.2, 1, size=flat.shape[:2])
    ref = synth.to_np(true.triangulate(flat, weights=w))
    lin = T.triangulate_linear(P, flat, weights=w)
    assert np.nanmax(np.linalg.norm(lin - ref, axis=-1)) / synth.rig_scale(true) < 1e-6
    # the DLT of any subset = sum of its blocks
    und = G.undistort(P, flat)
    c = P.centers().mean(0).numpy()
    B = T.dlt_blocks(P, und, weights=w, center=c)
    assert B.shape == (7, flat.shape[1], 4, 4)
    X = T.solve_blocks(B.sum(0), center=c)
    np.testing.assert_allclose(X, lin, atol=1e-9 * synth.rig_scale(true))
    X02 = T.solve_blocks(B[[0, 2]].sum(0), center=c)
    sub = T.triangulate_linear(P.subset([0, 2]), flat[[0, 2]], weights=w[[0, 2]])
    np.testing.assert_allclose(X02, sub, atol=1e-6 * synth.rig_scale(true))


def test_linear_shapes_types_and_grad(motobs):
    true, P, p3d, p2d, _, _ = motobs
    out = T.triangulate_linear(P, p2d)                      # [C,N,J,2] -> [N,J,3]
    assert out.shape == p3d.shape and isinstance(out, np.ndarray)
    pt = torch.as_tensor(p2d, dtype=torch.float64).requires_grad_(True)
    out_t = T.triangulate_linear(P, pt)
    assert isinstance(out_t, torch.Tensor) and not out_t.requires_grad
    out_g = T.triangulate_linear(P, pt, return_grad=True)
    out_g.nan_to_num().sum().backward()
    assert torch.isfinite(pt.grad).all() and pt.grad.abs().sum() > 0


def test_linear_far_scene_is_well_conditioned():
    """Homogeneous DLT collapses far from the world origin; the recentred solve does not."""
    cg = synth.load_rig('allen_7cam')
    X = synth.random_points(cg, 200, seed=1)
    p2d = synth.project(cg, X)
    shift = np.array([1e6, -2e6, 5e5])
    P = G.pack(cg)
    t = P.tvec - torch.einsum('cij,j->ci', P.R, torch.as_tensor(shift))   # world moved by +shift
    P2 = G.PackedCameras(K=P.K, dist=P.dist, model=P.model, rvec=P.rvec, tvec=t, size=P.size)
    out = T.triangulate_linear(P2, p2d)
    assert np.nanmax(np.linalg.norm(out - shift - X, axis=-1)) / synth.rig_scale(cg) < 1e-6


def test_per_point_and_per_frame_extrinsics():
    cg = synth.load_rig('allen_7cam')
    P = G.pack(cg)
    Tn, M = 3, 40
    rng = np.random.default_rng(0)
    rv = P.rvec[:, None] + torch.as_tensor(rng.normal(size=(7, Tn, 3)) * 0.02)
    tv = P.tvec[:, None] + torch.as_tensor(rng.normal(size=(7, Tn, 3)) * 5)
    E = G.make_extrinsics(rv, tv)                                     # [C,T,4,4] moving rig
    X = synth.random_points(cg, Tn * M, seed=3).reshape(Tn, M, 3)
    uv, _ = G.project(P, X, extrinsics=E)
    out = T.triangulate_linear(P, uv, extrinsics=E)
    assert out.shape == (Tn, M, 3)
    assert np.nanmax(rel(cg, out, X)) < 1e-8
    res = T.triangulate_robust(P, uv, extrinsics=E)
    assert np.nanmax(rel(cg, res.p3d, X)) < 1e-8


# ------------------------------------------------------------------ robust

@pytest.mark.integration
@pytest.mark.parametrize('method', METHODS)
def test_robust_rejects_false_detections(outdoor, method):
    true, P, p3d, p2d = outdoor
    bad = synth.add_outliers(p2d, frac=0.1, cgroup=true, seed=5)
    plain = rel(true, T.triangulate_linear(P, bad), p3d)
    res = T.triangulate_robust(P, bad, method=method)
    r = rel(true, res.p3d, p3d)
    assert np.mean(np.isfinite(r)) > 0.99
    assert np.nanmedian(r) < 1e-3 and np.nanpercentile(r, 95) < 2e-3
    assert np.nanmedian(r) < np.nanmedian(plain) / 20
    # outliers are flagged: compare against the true per-observation corruption
    corrupted = np.linalg.norm(bad - p2d, axis=-1) > 30
    seen = np.isfinite(bad[..., 0])
    assert np.mean(~res.inliers[corrupted & seen]) > 0.98
    assert np.mean(res.inliers[~corrupted & seen]) > 0.98


@pytest.mark.integration
@pytest.mark.parametrize('method', METHODS)
def test_robust_survives_decoy_camera(motobs, method):
    true, P, p3d, p2d, p2d_decoy, frames = motobs
    res = T.triangulate_robust(P, p2d_decoy, method=method)
    assert res.p3d.shape == p3d.shape and res.errors.shape == p2d.shape[:-1]
    r = rel(true, res.p3d, p3d)
    assert np.nanmedian(r[frames]) < 1e-3
    assert np.isfinite(r).all()
    assert not res.inliers[3][frames].any()             # decoy observations rejected ...
    assert res.inliers[3][~frames].mean() > 0.95         # ... the camera's good frames kept
    assert np.nanmedian(res.errors[3][frames]) > 50      # and reported with their real error


def test_result_object(motobs):
    true, P, p3d, p2d, p2d_decoy, frames = motobs
    res = T.triangulate_robust(P, p2d_decoy)
    p, e = res
    assert p is res.p3d and e is res.errors
    d = json.loads(json.dumps(res.to_dict(), default=float).replace('NaN', 'null'))
    assert set(d) == {'p3d', 'inliers', 'errors', 'weights', 'n_inliers'}
    np.testing.assert_array_equal(res.n_inliers, res.inliers.sum(0))
    rt = T.triangulate_robust(P, torch.as_tensor(p2d_decoy))
    assert isinstance(rt.p3d, torch.Tensor)
    np.testing.assert_allclose(rt.p3d.numpy(), res.p3d, atol=1e-12)
    with pytest.raises(ValueError):
        T.triangulate_robust(P, p2d_decoy, method='ransac')


def test_irls_single_ray_guard():
    """With two views that disagree badly, naive IRLS collapses onto one ray (≈0 px for that
    camera); the guard keeps both views weighted."""
    cg = synth.load_rig('motobs_7cam')
    P = G.pack(cg)
    X = synth.random_points(cg, 50, min_views=7, seed=1)
    p2d = synth.project(cg, X)
    p2d[2:] = np.nan
    p2d[1] += 80.0
    res = T.triangulate_robust(P, p2d, method='irls', sigma=2.0, min_inliers=1, n_refine=0, tau=1e9)
    assert np.all(np.isfinite(res.p3d))
    assert np.all((res.weights[:2] > 0.1).sum(0) == 2)


def test_contrib_passenger_cameras(motobs):
    true, P, p3d, p2d, _, _ = motobs
    biased = p2d.copy()
    biased[[0, 1]] += 6.0           # two systematically biased (unplaced/LOO) cameras
    contrib = np.ones(7, bool); contrib[[0, 1]] = False
    for method in METHODS:
        res = T.triangulate_robust(P, biased, method=method, contrib=contrib)
        drop = p2d.copy()
        drop[~contrib] = np.nan
        ref = T.triangulate_robust(P, drop, method=method)
        np.testing.assert_allclose(res.p3d, ref.p3d, atol=1e-9 * synth.rig_scale(true))
        # passengers are still measured against the drivers' geometry
        assert abs(np.nanmedian(res.errors[0]) - np.hypot(6, 6)) < 1.5


def test_mask_out_of_bounds():
    cg = synth.load_rig('outdoor_9cam')
    P = G.pack(cg)
    X = synth.random_points(cg, 100, min_views=4, seed=1)
    p2d = synth.project(cg, X)
    p2d[0] = -500.0                  # garbage outside the image (e.g. a padded crop prediction)
    p2d[1] = 1e4
    res = T.triangulate_robust(P, p2d, method='irls')
    ok = synth.n_views(p2d[2:]) >= 2
    assert np.nanmax(rel(cg, res.p3d[ok], X[ok])) < 1e-8
    assert not res.inliers[:2].any() and np.isnan(res.errors[:2]).all()


def test_min_inliers(outdoor):
    true, P, p3d, p2d = outdoor
    res = T.triangulate_robust(P, p2d[:, :200], min_inliers=3)
    assert np.all(res.n_inliers[np.isfinite(res.p3d[:, 0])] >= 3)
    two = p2d[:, :50].copy(); two[2:] = np.nan
    res = T.triangulate_robust(P, two, min_inliers=3)
    assert np.isnan(res.p3d).all()


def test_chunking_is_exact(outdoor):
    true, P, p3d, p2d = outdoor
    bad = synth.add_outliers(p2d[:, :300], frac=0.1, cgroup=true, seed=5)
    for method in METHODS:
        a = T.triangulate_robust(P, bad, method=method)
        b = T.triangulate_robust(P, bad, method=method, chunk=37)
        np.testing.assert_allclose(a.p3d, b.p3d, atol=1e-12)
        np.testing.assert_array_equal(a.inliers, b.inliers)
    np.testing.assert_allclose(T.triangulate_linear(P, bad, chunk=11), T.triangulate_linear(P, bad), atol=1e-12)


def test_device_parity(device, outdoor):
    true, P, p3d, p2d = outdoor
    bad = synth.add_outliers(p2d[:, :400], frac=0.1, cgroup=true, seed=5)
    Pd = P.to(device)
    bt = torch.as_tensor(bad, device=device)
    for method in METHODS:
        a = T.triangulate_robust(P, bad, method=method)
        b = T.triangulate_robust(Pd, bt, method=method)
        assert b.p3d.device.type == device.type
        np.testing.assert_allclose(b.p3d.cpu().numpy(), a.p3d, atol=1e-9)
        np.testing.assert_array_equal(b.inliers.cpu().numpy(), a.inliers)
    np.testing.assert_allclose(T.triangulate_linear(Pd, bt).cpu().numpy(), T.triangulate_linear(P, bad), atol=1e-9)


# ------------------------------------------------------------------ multi-candidate

def candidates(cg, n=200, seed=4):
    X = synth.random_points(cg, n, seed=seed)
    p = synth.add_noise(synth.project(cg, X), 0.5, seed=1)
    C, N = p.shape[:2]
    rng = np.random.default_rng(seed)
    pp = np.full((C, N, 3, 2), np.nan)
    pp[:, :, 0] = p
    pp[:, :, 1] = p + 50.0                                     # a nearby distractor
    pp[:, :, 2] = rng.uniform(0, 600, size=(C, N, 2))          # a random false detection
    perm = rng.permuted(np.tile(np.arange(3), (C, N, 1)), axis=-1)
    pp = np.take_along_axis(pp, perm[..., None], 2)
    return X, p, pp, np.argmax(perm == 0, -1)


def test_triangulate_possible_picks_true_candidates():
    cg = synth.load_rig('outdoor_9cam')
    P = G.pack(cg)
    X, p, pp, truek = candidates(cg)
    res = T.triangulate_possible(P, pp)
    vis = np.isfinite(p[..., 0])
    assert np.all(res.picked[vis] == truek[vis])
    assert np.all(res.picked[~vis] == -1)
    assert np.nanmedian(rel(cg, res.p3d, X)) < 1e-3
    p3, picked_vals, pts2d, err = res.legacy()
    assert picked_vals.shape == pp.shape[:-1] and pts2d.shape == p.shape and err.shape == (len(X),)
    assert np.array_equal(np.isfinite(pts2d[..., 0]), picked_vals.any(-1))
    # one candidate per camera = ransac; agrees with msac
    one = T.triangulate_possible(P, p[:, :, None])
    msac = T.triangulate_robust(P, p, mask_out_of_bounds=False)
    np.testing.assert_allclose(one.p3d, msac.p3d, atol=1e-9 * synth.rig_scale(cg))


def test_triangulate_possible_memory_bounded():
    """No subset enumeration: 12 cameras x 4 candidates (5^12 subsets for the old kernel) in
    bounded chunks."""
    cg = synth.load_rig('outdoor_12cam')
    P = G.pack(cg)
    X = synth.random_points(cg, 30, seed=2)
    p = synth.project(cg, X)
    rng = np.random.default_rng(0)
    pp = np.concatenate([p[:, :, None], rng.uniform(0, 600, size=p.shape[:2] + (3, 2))], 2)
    res = T.triangulate_possible(P, pp, max_elems=1 << 18)
    assert np.all(res.picked[np.isfinite(p[..., 0])] == 0)


# ------------------------------------------------------------------ speed
# wall-clock budgets are for an idle machine, so these are opt-in (-m slow)

@pytest.mark.slow
def test_speed_12cams(outdoor):
    true, P, p3d, p2d = outdoor
    X = synth.random_points(true, 10000, spread=0.25, seed=1)
    p = torch.as_tensor(synth.add_noise(synth.project(true, X), 0.5, seed=2))
    T.triangulate_robust(P, p[:, :100])
    t = time.time()
    T.triangulate_robust(P, p)
    assert time.time() - t < 5.0


@pytest.mark.slow
def test_speed_12cams_100k(device):
    true = synth.load_rig('outdoor_12cam')
    X = synth.random_points(true, 100_000, spread=0.25, seed=1)
    p2 = synth.add_outliers(synth.add_noise(synth.project(true, X), 0.5, seed=2), 0.05, true, seed=3)
    P = G.pack(true).to(device)
    pt = torch.as_tensor(p2, device=device)
    T.triangulate_robust(P, pt[:, :1000])
    for method, budget in [('msac', 6.0), ('irls', 3.0), ('reject', 3.0)]:
        t = time.time()
        res = T.triangulate_robust(P, pt, method=method)
        dt = time.time() - t
        assert dt < budget, (method, dt)
        assert np.nanmedian(rel(true, res.p3d.cpu().numpy(), X)) < 1e-3
