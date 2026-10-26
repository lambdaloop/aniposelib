"""aniposelib.bundle (LM bundle adjustment, robust recipe, resection) and the full ParamSpec."""

import json
from copy import deepcopy

import numpy as np
import pytest
import torch

import synth
from aniposelib import bundle, params
from aniposelib.bundle import Observations, bundle_adjust_lm
from aniposelib.params import ParamSpec, IntrinsicsPrior


@pytest.fixture(scope='module')
def outdoor():
    true = synth.load_rig('outdoor_9cam')
    p3d = synth.trajectory(true, 400, seed=2)
    p2d = synth.add_noise(synth.project(true, p3d), 0.3, seed=3)
    return true, p3d, p2d


def K(cam):
    return synth.to_np(cam.get_camera_matrix())


def test_analytic_jacobians_match_finite_differences():
    torch.manual_seed(0)
    M = 40
    for kind in ('brown', 'kb', 'mixed'):
        L = torch.zeros(M, 20, dtype=torch.float64)
        L[:, 0:3] = torch.randn(M, 3) * 0.5
        L[0, 0:3] = 0
        L[:, 3:6] = torch.randn(M, 3) * 0.1 + torch.tensor([0, 0, 5.])
        L[:, 6], L[:, 7], L[:, 8], L[:, 9], L[:, 10] = 800., 820., 320., 240., 3.
        L[:, 11:19] = torch.randn(M, 8) * 0.05
        X = torch.randn(M, 3, dtype=torch.float64)
        k10 = torch.full((M,), 0.5, dtype=torch.float64)
        off = torch.randn(M, 2, dtype=torch.float64)
        is_kb = {'brown': torch.zeros(M, dtype=torch.bool), 'kb': torch.ones(M, dtype=torch.bool),
                 'mixed': torch.arange(M) % 2 == 0}[kind]
        hb, hk = bool((~is_kb).any()), bool(is_kb.any())
        uv, z, dL, dX = bundle._project_local_jac(L, X, k10, off, is_kb, hb, hk)
        f = lambda L_, X_: bundle._project_local(L_, X_, k10, off, is_kb, hb, hk)[0]
        assert torch.allclose(uv, f(L, X))
        h = 1e-6
        for k in range(19):
            e = torch.zeros(20, dtype=torch.float64); e[k] = h
            num = (f(L + e, X) - f(L - e, X)) / (2 * h)
            assert torch.allclose(num, dL[:, :, k], rtol=1e-5, atol=1e-5), (kind, k)
        for k in range(3):
            e = torch.zeros(3, dtype=torch.float64); e[k] = h
            num = (f(L, X + e) - f(L, X - e)) / (2 * h)
            assert torch.allclose(num, dX[:, :, k], rtol=1e-5, atol=1e-5)


def test_observations_dense_roundtrip_and_groups():
    p = np.full((3, 4, 2, 2), np.nan)
    p[0, 1, 0] = [1, 2]; p[2, 3, 1] = [5, 6]; p[1, 1, 0] = [3, 4]
    obs = Observations.from_dense(p, time=np.arange(4.0))
    assert len(obs) == 3 and obs.n_points == 8 and obs.point_shape == (4, 2)
    assert set(obs.group) == {'frame', 'keypoint'}
    assert np.array_equal(np.sort(obs.group['frame']), [1, 1, 3])
    assert np.allclose(obs.to_dense().reshape(p.shape)[np.isfinite(p)], p[np.isfinite(p)])
    sub = obs.subset(obs.cam_idx != 1)
    assert len(sub) == 2 and sub.n_points == 8
    json.dumps(obs.to_dict())


def test_observations_from_timed():
    from aniposelib.stba import TimedObservations
    t = TimedObservations(cam=[0, 1, 0, 1], point=[7, 7, 7, 7], uv=np.zeros((4, 2)), t=[0.0, 0.001, 1 / 30, 1 / 30])
    obs = Observations.from_timed(t, frame_time=1 / 30)
    assert obs.n_points == 2 and np.array_equal(obs.point_idx, [0, 0, 1, 1])
    assert np.all(obs.group['track'] == 7)


def test_lm_recovers_extrinsics_fast_and_keeps_gauge(outdoor):
    true, p3d, p2d = outdoor
    cg = synth.perturb_extrinsics(true, rot_deg=0.5, trans_frac=0.01, seed=4)
    r0 = synth.to_np(cg.cameras[0].rvec).copy()
    rep = bundle_adjust_lm(cg, p2d, only_extrinsics=True, return_report=True)
    assert synth.center_error(cg, true).max() < 4e-3
    assert rep.error < 0.5 and rep.error < rep.error_before
    assert np.array_equal(synth.to_np(cg.cameras[0].rvec), r0)           # gauge camera untouched
    assert abs(rep.gauge['scale_ratio'] - 1) < 1e-9                         # scale kept
    for c, ct in zip(cg.cameras, true.cameras):                             # intrinsics untouched
        assert np.array_equal(K(c), K(ct))
    d = rep.to_dict()
    json.dumps(d)
    assert set(d['per_camera']) == set(true.get_names())
    assert isinstance(bundle_adjust_lm(deepcopy(true), p2d, only_extrinsics=True), float)


def test_pcg_matches_dense(outdoor):
    true, p3d, p2d = outdoor
    a = synth.perturb_extrinsics(true, rot_deg=0.3, trans_frac=0.005, seed=5)
    b = deepcopy(a)
    bundle_adjust_lm(a, p2d, only_extrinsics=True, solver='dense', n_iter=8)
    bundle_adjust_lm(b, p2d, only_extrinsics=True, solver='pcg', n_iter=8)
    ca, cb = synth.camera_centers(a), synth.camera_centers(b)
    assert np.abs(ca - cb).max() < 1e-6 * synth.rig_scale(true)


def test_spec_shared_focal_and_frozen_entries(outdoor):
    true, p3d, p2d = outdoor
    cg = deepcopy(true)
    for c in cg.cameras:
        with torch.no_grad():
            c.matrix[0, 0] *= 1.02; c.matrix[1, 1] *= 1.02
    ratio = [K(c)[1, 1] / K(c)[0, 0] for c in cg.cameras]
    cx = [K(c)[0, 2] for c in cg.cameras]
    d0 = [synth.to_np(c.dist).copy() for c in cg.cameras]
    spec = ParamSpec(intrinsics=('f',), n_dist=0, extrinsics='fixed')
    rep = bundle_adjust_lm(cg, p2d, spec=spec, return_report=True)
    f_err = [abs(K(c)[0, 0] / K(t)[0, 0] - 1) for c, t in zip(cg.cameras, true.cameras)]
    assert max(f_err) < 2e-3
    for c, r, x, d in zip(cg.cameras, ratio, cx, d0):
        assert abs(K(c)[1, 1] / K(c)[0, 0] - r) < 1e-12        # aspect fixed
        assert K(c)[0, 2] == x                                   # cx fixed
        assert np.array_equal(synth.to_np(c.dist), d)            # n_dist=0
    assert rep.per_camera[true.get_names()[2]]['free'] == ['f'] or 'f' in rep.per_camera[true.get_names()[2]]['free']


def test_spec_share_groups_and_priors(outdoor):
    true, p3d, p2d = outdoor
    cg = deepcopy(true)
    spec = ParamSpec(intrinsics=('fx', 'fy'), n_dist=0, share=[[1, 2]], fixed_cameras=[0])
    bundle_adjust_lm(cg, p2d, spec=spec, n_iter=5)
    assert np.allclose(K(cg.cameras[1])[:2, :2], K(cg.cameras[2])[:2, :2])
    # a tight prior pins the focal length to the reference
    cg = deepcopy(true)
    ref = deepcopy(true)
    with torch.no_grad():
        ref.cameras[3].matrix[0, 0] *= 1.1
    spec = ParamSpec(intrinsics=('fx',), n_dist=0, priors=IntrinsicsPrior(ref, sigma_logf=1e-6), fixed_cameras=[0])
    bundle_adjust_lm(cg, p2d, spec=spec, n_iter=10)
    assert abs(K(cg.cameras[3])[0, 0] / K(ref.cameras[3])[0, 0] - 1) < 1e-4


def test_scale_pair_constraint(outdoor):
    true, p3d, p2d = outdoor
    cg = deepcopy(true)
    names = true.get_names()
    C = synth.camera_centers(true)
    target = 2.0 * np.linalg.norm(C[0] - C[3])
    spec = ParamSpec.extrinsics_only(scale_constraint=(names[0], names[3], target))
    rep = bundle_adjust_lm(cg, p2d, spec=spec, return_report=True)
    C2 = synth.camera_centers(cg)
    assert abs(np.linalg.norm(C2[0] - C2[3]) / target - 1) < 1e-6
    assert abs(rep.gauge['scale_ratio'] - 2) < 2e-2
    assert rep.error < 0.5


def test_min_points_guard_and_revert(outdoor):
    true, p3d, p2d = outdoor
    cg = synth.perturb_extrinsics(true, rot_deg=0.5, trans_frac=0.01, seed=4)
    p = p2d.copy()
    p[4, 5:] = np.nan
    r4 = synth.to_np(cg.cameras[4].rvec).copy()
    with pytest.warns(RuntimeWarning, match='frozen'):
        rep = bundle_adjust_lm(cg, p, only_extrinsics=True, return_report=True)
    assert rep.frozen_cameras == [true.get_names()[4]]
    assert np.array_equal(synth.to_np(cg.cameras[4].rvec), r4)
    # revert_if_worse keeps the old parameters when the error grows (a wrong target)
    cg = deepcopy(true)
    before = synth.camera_centers(cg)
    bad = p2d + np.random.default_rng(0).normal(scale=50, size=p2d.shape)
    rep = bundle_adjust_lm(cg, bad, only_extrinsics=True, return_report=True, revert_if_worse=True,
                           loss='linear', n_iter=0)
    assert np.array_equal(synth.camera_centers(cg), before)


def test_not_inplace_returns_copy(outdoor):
    true, p3d, p2d = outdoor
    cg = synth.perturb_extrinsics(true, rot_deg=0.5, trans_frac=0.01, seed=4)
    before = synth.camera_centers(cg)
    rep = bundle_adjust_lm(cg, p2d, only_extrinsics=True, inplace=False, return_report=True)
    assert np.array_equal(synth.camera_centers(cg), before)
    assert synth.center_error(rep.cgroup, true).max() < 4e-3


def test_time_offsets_recovered():
    """F3: per-camera time offsets from a smooth trajectory (first order, prior anchored)."""
    true = synth.load_rig('outdoor_9cam')
    cen, s = synth.scene_center(true), synth.rig_scale(true)
    t = np.arange(400) / 30.0
    X = lambda tt: cen + 0.2 * s * np.stack([np.sin(1.3 * tt), np.cos(0.9 * tt), 0.3 * np.sin(2.1 * tt)], -1)
    delta = np.zeros(9)
    delta[[2, 5]] = [0.01, -0.02]
    p2d = np.stack([synth.project(true, X(t + delta[c]))[c] for c in range(9)])
    p2d = synth.add_noise(p2d, 0.3, seed=1)
    obs = Observations.from_dense(p2d, time=t)
    cg = deepcopy(true)
    rep = bundle_adjust_lm(cg, obs, spec=ParamSpec.extrinsics_only(time_offset=0.05), return_report=True,
                           loss='linear')
    est = np.array([rep.time_offsets[n] for n in true.get_names()])
    assert np.abs(est - delta).max() < 2e-3
    assert synth.center_error(cg, true).max() < 1e-3


def test_stations_give_a_moved_camera_its_own_extrinsics():
    """F5: a camera bumped half way gets a second station; intrinsics stay shared."""
    from aniposelib.stations import Stations
    true = synth.load_rig('outdoor_9cam')
    names = true.get_names()
    moved = deepcopy(true)
    with torch.no_grad():
        moved.cameras[4].rvec += torch.tensor([0.0, 0.02, 0.0], dtype=torch.float64)
    p3d = synth.trajectory(true, 500, seed=2)
    t = np.arange(500) / 30.0
    p2 = synth.project(true, p3d)
    p2[4, 250:] = synth.project(moved, p3d)[4, 250:]
    p2 = synth.add_noise(p2, 0.3, seed=3)
    st = Stations({names[4]: [(-np.inf, t[250], 'a'), (t[250], np.inf, 'b')]}, cameras=names)
    cg = deepcopy(true)
    rep = bundle_adjust_lm(cg, Observations.from_dense(p2, time=t), stations=st, only_extrinsics=True,
                           return_report=True)
    S = rep.stations
    rb = S['rvec'][S['names'].index('b')]
    assert np.abs(rb - synth.to_np(moved.cameras[4].rvec)).max() < 3e-3
    assert np.abs(synth.to_np(cg.cameras[4].rvec) - synth.to_np(true.cameras[4].rvec)).max() < 3e-3
    assert rep.error < 0.5
    # without stations the bump cannot be explained
    cg2 = deepcopy(true)
    assert bundle_adjust_lm(cg2, p2, only_extrinsics=True) > rep.error


def test_resect_and_add_camera(outdoor):
    true, p3d, p2d = outdoor
    cg = deepcopy(true)
    with torch.no_grad():
        cg.cameras[3].rvec += torch.tensor([0.3, -0.2, 0.1], dtype=torch.float64)
    res = bundle.resect_cameras(cg, p2d, rng=0)
    r3 = res[3]
    assert r3.updated and r3.n_inliers_after > r3.n_inliers_before
    assert not any(r.updated for i, r in enumerate(res) if i != 3 and r.n_inliers_before > 0.9 * r.n_points)
    assert synth.center_error(cg, true).max() < 5e-3
    # add_camera: place a new camera from its detections
    sub = true.subset_cameras([0, 1, 2, 3, 4, 5, 6, 7])
    newcam = deepcopy(true.cameras[8])
    with torch.no_grad():
        newcam.rvec.zero_(); newcam.tvec.zero_()
    grown, rr = bundle.add_camera(sub, newcam, p2d[8], p2d[:8], rng=0)
    assert rr.updated and len(grown.cameras) == 9
    assert synth.center_error(grown, true).max() < 5e-3


def test_masks_frame_and_keypoint():
    err = np.ones((2, 6))
    frame = np.array([0, 0, 0, 1, 1, 1])
    err[1, 3:] = 100
    m, bad, frames = bundle.frame_mask(err, frame, 10, min_pts=2)
    assert m[1, 3:].all() and not m[0].any() and not m[1, :3].any()
    kpt = np.tile(np.arange(10), 30)
    e = np.ones((1, 300)); e[0, kpt == 3] = 50
    km, kmask = bundle.keypoint_mask(e, kpt, 10, min_obs=5, protect=2)
    assert kmask[0, 3] and kmask[0].sum() == 1


def test_robust_report_uses_all_observations():
    true = synth.load_rig('motobs_7cam')
    p3d, _ = synth.skeleton(true, 40, model=synth.MOUSE, spread=0.12, seed=1)
    p2d = synth.add_noise(synth.project(true, p3d), 0.7, seed=2)
    p2d, frames = synth.add_decoy(p2d, cam=3, frac=0.6, cgroup=true, seed=3)
    cg = synth.perturb_extrinsics(true, rot_deg=0.5, trans_frac=0.01, seed=4)
    rep = bundle.bundle_adjust_robust(cg, p2d, schedule=((100, 20), (10, 5)), final=('extrinsics',), rng=0)
    assert synth.center_error(cg, true).max() < 5e-3
    name = true.get_names()[3]
    pc = rep.per_camera[name]
    assert pc['n_obs'] == int(np.isfinite(p2d[3, ..., 0]).sum())     # all obs reported
    assert pc['n_fit'] < pc['n_obs'] and pc['median_after'] > 20       # decoy honest in the report
    assert rep.excluded['fit'].shape == p2d.shape[:-1]
    assert rep.excluded['fit'][3][frames].mean() > 0.9
    json.dumps(rep.to_dict())


def test_board_ba_lm_recovers_intrinsics():
    from test_rig_lab_board import initialise, focal_rel_err
    true = synth.load_rig('allen_7cam')
    cg, imgp, extra = initialise(true, n_frames=25)
    rep = bundle_adjust_lm(cg, imgp, extra=extra, f_scale=2.0, return_report=True)
    assert synth.center_error(cg, true).max() < 5e-3
    assert focal_rel_err(cg, true).max() < 5e-3
    assert rep.gauge['scale_constraint'] is None


def test_cameragroup_methods_default_lm_and_adam_pin(outdoor):
    true, p3d, p2d = outdoor
    cg = synth.perturb_extrinsics(true, rot_deg=0.5, trans_frac=0.01, seed=4)
    err = cg.bundle_adjust(p2d, only_extrinsics=True, verbose=False)
    assert isinstance(err, float) and err.item() == float(err) and err < 1
    rep = cg.bundle_adjust(p2d, only_extrinsics=True, verbose=False, return_report=True)
    assert isinstance(rep, bundle.BAReport)
    cg2 = synth.perturb_extrinsics(true, rot_deg=0.5, trans_frac=0.01, seed=4)
    e2 = cg2.bundle_adjust(p2d, only_extrinsics=True, verbose=False, method='adam', max_nfev=20)
    assert isinstance(e2, float)
    with pytest.raises(TypeError):
        cg2.bundle_adjust(p2d, method='adam', spec=ParamSpec())
    with pytest.raises(ValueError):
        cg2.bundle_adjust(p2d, method='sgd')


def test_paramspec_core_api_still_works():
    from aniposelib.cameras import Camera
    cam = Camera(dist=np.zeros(5))
    ParamSpec(intrinsics=('fx', 'fy'), n_dist=1).apply(cam)
    assert cam.get_free_intrinsics() == ('fx', 'fy') and cam.get_n_dist() == 1
    assert ParamSpec.from_camera(cam).intrinsics == ('fx', 'fy')
    ParamSpec(intrinsics=('f', 'cx')).apply(cam)
    assert cam.get_free_intrinsics() == ('fx', 'cx', 'fy') or set(cam.get_free_intrinsics()) == {'fx', 'fy', 'cx'}
    with pytest.raises(ValueError):
        ParamSpec(intrinsics=('focal',))
    with pytest.raises(ValueError):
        ParamSpec(scale_constraint='bad')
    s = ParamSpec(priors={'a': {'f': (1000, 0.1, 'log')}})
    json.dumps(s.to_dict())
