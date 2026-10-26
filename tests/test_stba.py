"""Spatiotemporal BA (aniposelib.stba) and camera stations (aniposelib.stations)."""
import json
from copy import deepcopy

import numpy as np
import pytest
import toml
import torch

import synth
from aniposelib import geometry as G
from aniposelib import stba as S
from aniposelib import stations as ST
from aniposelib.params import ParamSpec


@pytest.fixture(scope='module')
def rig():
    return synth.load_rig('fisheye_5cam')


def _step_obs(cg, corr, cam=1, n_points=8, duration=4.0, seed=4):
    """Timed tracks where camera `cam` sees the scene at seed time + corr(t)."""
    obs, fn = synth.timed_observations(cg, n_points=n_points, duration=duration, fps=60, noise=0.5, seed=seed)
    uv = obs['uv'].copy()
    idx = np.flatnonzero(obs['cam'] == cam)
    rng = np.random.default_rng(seed)
    for p in range(n_points):
        j = idx[obs['point'][idx] == p]
        tt = obs['t'][j]
        uv[j] = synth.project(cg, fn(p, tt + corr(tt)))[cam] + rng.normal(scale=0.5, size=(len(j), 2))
    obs['uv'] = uv
    ok = np.isfinite(uv).all(1)
    return {k: v[ok] for k, v in obs.items()}, fn


# ---------------------------------------------------------------- kernels

@pytest.mark.parametrize('model', [0, 1])
def test_projection_jacobians_match_finite_differences(model):
    dt = torch.float64
    X = torch.randn(40, 3, dtype=dt, generator=torch.Generator().manual_seed(0)) * 0.5 + torch.tensor([0, 0, 4.0], dtype=dt)
    p = torch.tensor([0.1, -0.2, 0.3, 0.1, 0.2, 0.3, 800, 810, 960, 540, 1.5,
                      0.05, -0.01, 0.002, 0.001, 0.003, 0.01, 0.002, 0.001], dtype=dt)
    if model == 1:
        p[15:] = 0
    k10, off = torch.tensor(0.0, dtype=dt), torch.tensor([5.0, 7.0], dtype=dt)
    uv, z, JX, Jp = S._project_cam(p, X, model, k10, off, jac=True)
    # agrees with the geometry kernel
    P = S._packed_from_theta(G.PackedCameras(K=torch.eye(3, dtype=dt)[None], dist=torch.zeros(1, 8, dtype=dt),
                                             model=[model], rvec=torch.zeros(1, 3, dtype=dt),
                                             tvec=torch.zeros(1, 3, dtype=dt), offset=off[None]), p[None])
    np.testing.assert_allclose(G.project(P, X)[0][0].numpy(), uv.numpy(), atol=1e-9)
    e = 1e-6
    for k in range(19):
        q1, q2 = p.clone(), p.clone()
        q1[k] += e; q2[k] -= e
        fd = (S._project_cam(q1, X, model, k10, off)[0] - S._project_cam(q2, X, model, k10, off)[0]) / (2 * e)
        assert torch.allclose(fd, Jp[..., k], rtol=1e-5, atol=1e-5 * (1 + float(fd.abs().max()))), k
    for k in range(3):
        Y1, Y2 = X.clone(), X.clone()
        Y1[:, k] += e; Y2[:, k] -= e
        fd = (S._project_cam(p, Y1, model, k10, off)[0] - S._project_cam(p, Y2, model, k10, off)[0]) / (2 * e)
        assert torch.allclose(fd, JX[..., k], atol=1e-5)


def test_spline_fit_and_eval():
    h, M = 1 / 30, 63
    t = np.linspace(0, 1.9, 200)
    X = np.stack([t ** 3, np.sin(t), 1 - t], -1)
    A = S._basis_matrix_np(t + h, M, h)
    C = np.linalg.lstsq(A, X, rcond=None)[0]
    Xh, V = S.spline_eval(torch.as_tensor(C), torch.as_tensor(t + h), h, derivative=True)
    assert np.abs(Xh.numpy() - X).max() < 1e-4
    V_true = np.stack([3 * t ** 2, np.cos(t), -np.ones_like(t)], -1)
    assert np.abs(V.numpy() - V_true)[10:-10].max() < 1e-2


# ---------------------------------------------------------------- STBA

def test_stba_recovers_offsets_and_rig(rig, device):
    off = np.array([0, 0.012, -0.020, 0.008, 0.015])
    obs, _ = synth.timed_observations(rig, n_points=6, duration=3.0, fps=[60, 60, 30, 50, 60], offsets=off, noise=0.5)
    cgp = synth.perturb_extrinsics(rig, rot_deg=0.5, trans_frac=0.005, seed=3)
    before = synth.to_np(cgp.cameras[1].rvec).copy()
    ba = S.SpatiotemporalBA(cgp, obs, segment_s=1.0, sigma_px=0.5, free_offsets='per_camera', cluster_s=1.0,
                            device=device)
    res = ba.fit()
    np.testing.assert_allclose(synth.to_np(cgp.cameras[1].rvec), before)       # input untouched
    assert res.delta[0] == 0 and not res.delta_free[0]
    np.testing.assert_allclose(res.delta, off, atol=3e-4)                     # sub-frame: < 0.3 ms
    assert np.all(res.delta_sigma[1:] < 2e-4) and np.all(np.isfinite(res.delta_sigma_sandwich[1:]))
    assert synth.center_error(res.group, rig).max() < 1e-3 < synth.center_error(cgp, rig).max()
    assert np.median(res.residuals) < 1.0
    np.testing.assert_allclose(res.sync_offsets(), -res.delta)
    d = json.loads(json.dumps(res.to_dict()))
    assert d['clip_names'] == res.clip_names and 'group' in d
    X = res.trajectories.evaluate(np.array([0.5, 1.5]), point=0)
    assert np.isfinite(X).all()


def test_stba_per_clip_offsets_and_intrinsics(rig):
    # camera 1 has two clips (a chapter step at 1.5 s) with different offsets
    corr = lambda t: np.where(t < 1.5, 0.010, 0.025)
    obs, _ = _step_obs(rig, corr, n_points=6, duration=3.0)
    clip = obs['cam'].copy()
    clip[(obs['cam'] == 1) & (obs['t'] >= 1.5)] = 5
    obs['clip'] = clip
    ba = S.SpatiotemporalBA(rig, obs, segment_s=1.0, sigma_px=0.5, extrinsics=False,
                            spec=ParamSpec(intrinsics=('fx', 'fy'), n_dist=0), intrinsics_prior_sigma=50.0)
    res = ba.fit()
    assert res.clip_names == ['clip0', 'clip1', 'clip2', 'clip3', 'clip4', 'clip5']
    np.testing.assert_allclose(res.delta, [0, 0.010, 0, 0, 0, 0.025], atol=5e-4)
    K = synth.to_np(res.group.cameras[2].get_camera_matrix())
    assert abs(K[0, 0] - synth.to_np(rig.cameras[2].get_camera_matrix())[0, 0]) < 2.0


def test_stba_rolling_shutter_and_rate(rig):
    off = np.array([0, 0.005, 0.010, -0.004, 0.0])
    rs = np.array([0.008, 0.012, 0.0, 0.010, 0.015])
    rates = np.array([0, 2e-3, 0, -1e-3, 0])
    obs, _ = synth.timed_observations(rig, n_points=8, duration=4.0, fps=60, offsets=off, rs=rs, rates=rates,
                                      noise=0.5)
    ba = S.SpatiotemporalBA(rig, obs, segment_s=1.0, sigma_px=0.5, free_offsets='per_camera', rolling_shutter=True,
                            free_rate=True, rate_prior_sigma=1e-2, extrinsics=False, cluster_s=1.0)
    res = ba.fit()
    np.testing.assert_allclose(res.rs, rs, atol=2e-3)
    np.testing.assert_allclose(res.rate, rates, atol=2e-4)
    np.testing.assert_allclose(res.delta, off, atol=5e-4)
    assert np.all(res.rs_sigma < 2e-3)


def test_stba_time_warp_from_soft_dtw(rig):
    # camera 1's clock jumps by 20 ms at t = 2 s (dropped frames): one offset cannot fit it
    corr = lambda t: np.where(t < 2, 0.010, 0.030)
    obs, fn = _step_obs(rig, corr)

    def speed(t):            # a view-invariant signal (total 3D speed), as a stand-in for 2D motion energy
        e = 1e-3
        return sum(np.linalg.norm(fn(p, t + e) - fn(p, t - e), axis=1) / (2 * e) for p in range(8))
    t = np.arange(0, 4.0, 1 / 60)
    ba = S.SpatiotemporalBA(rig, obs, segment_s=1.0, sigma_px=0.5, free_offsets='per_camera', extrinsics=False,
                            warp_knot_s=0.25, warp_smooth_sigma=0.01, cluster_s=1.0,
                            warp_signals={0: (t, speed(t)), 1: (t, speed(t + corr(t)))},
                            warp_dtw_kwargs=dict(band=0.15, gamma=0.05, lag0=0.0))
    assert ba.info['warp_soft_dtw'][ba.clip_names[1]]['identifiable']
    res = ba.fit()
    w = res.warp[res.clip_names[1]]
    total = w['values'] + res.delta[1]
    knots = w['knots']
    away = np.abs(knots - 2.0) > 0.3
    np.testing.assert_allclose(total[away], corr(knots[away]), atol=2e-3)
    assert w['monotone']
    # the other cameras get warps too (prior-anchored at 0): they stay ~0
    for k in (2, 3, 4):
        assert np.abs(res.warp[res.clip_names[k]]['values'] + res.delta[k]).max() < 2e-3
    # restricted to one clip
    ba1 = S.SpatiotemporalBA(rig, obs, segment_s=1.0, sigma_px=0.5, free_offsets='per_camera', extrinsics=False,
                             warp_knot_s=0.5, warp_clips=[1])
    assert set(ba1.warp_slice) == {1}


def test_stba_rejects_bad_input(rig):
    with pytest.raises(ValueError):
        S.SpatiotemporalBA(rig, dict(cam=[0], point=[0], uv=[[1.0, 2.0]], t=[0.0]))
    with pytest.raises(ValueError):
        S.SpatiotemporalBA(rig, dict(cam=[0], point=[0], uv=[[1.0, 2.0]], t=[0.0]), free_offsets='bogus')


# ---------------------------------------------------------------- stations

def test_stations_intervals_lookup_and_toml_roundtrip():
    st = ST.Stations({'cam4': [(10.0, 20.0, 'cam4@s1'), (0.0, 10.0, 'cam4'), (20.0, 22.0, 'skip'),
                               (22.0, 30.0, 'cam4@s2')]}, cameras=['cam0', 'cam4'])
    assert st.intervals('cam4')[0] == (-np.inf, 10.0, 'cam4') and st.intervals('cam4')[-1][1] == np.inf
    assert st.names('cam4') == ['cam4', 'cam4@s1', 'cam4@s2']
    np.testing.assert_array_equal(st.lookup('cam4', [-5, 9.99, 10, 21, 25, 1e6, np.nan]), [0, 0, 1, -1, 2, 2, -1])
    np.testing.assert_array_equal(st.lookup(1, [15.0]), [1])          # index via cameras=
    np.testing.assert_array_equal(st.lookup('cam0', [0.0, 1e9]), [0, 0])  # unlisted camera: one station
    assert st.moved == ['cam4']
    s = toml.dumps({'stations': st.to_toml_dict()})
    back = ST.Stations.from_toml_dict(toml.loads(s)['stations'], cameras=['cam0', 'cam4'])
    assert back == st
    st2 = st.split('cam4', 25.0, 'cam4@s3', skip=(24.5, 25.0))
    assert st2.lookup_name('cam4', [24.7, 26.0]).tolist() == ['skip', 'cam4@s3']
    assert st.lookup_name('cam4', [26.0]).tolist() == ['cam4@s2']    # edits return a copy
    assert ST.Stations({'a': [(0, 1, 'a'), (1, 2, 'a')]}).intervals('a') == [(-np.inf, np.inf, 'a')]


def test_detect_steps_series():
    rng = np.random.default_rng(0)
    y = rng.normal(scale=0.02, size=(30, 3))
    y[17:] += [1.0, -0.5, 0.2]
    cps = ST.detect_steps(y, min_step=0.3)
    assert [c['index'] for c in cps] == [17]
    drift = np.linspace(0, 0.25, 30)  # half-means differ by < min_step[:, None] * np.ones(3) + rng.normal(scale=0.02, size=(30, 3))
    assert ST.detect_steps(drift, min_step=0.3) == []


def _bumped(cg, cam, rot_deg):
    cgb = deepcopy(cg)
    c = cgb.cameras[cam]
    dR = G.rodrigues(torch.tensor(np.radians(rot_deg), dtype=torch.float64))
    with torch.no_grad():
        R = G.rodrigues(c.rvec.reshape(3).double())
        c.rvec.copy_(torch.as_tensor(ST._rotvec((dR @ R).numpy())).reshape(c.rvec.shape))
        c.tvec.copy_((dR @ c.tvec.reshape(3).double()).reshape(c.tvec.shape))
    return cgb


def test_detect_moves_finds_bumped_camera(rig):
    fn = synth.smooth_tracks(rig, n_points=8, amp=0.1, seed=1)
    t = np.arange(0, 40, 1 / 30)
    X = np.stack([fn(p, t) for p in range(8)], 1)
    rot = np.array([1.2, -1.0, 0.8])             # ~1.75 deg
    tb = 23.37
    p2d = np.where((t >= tb)[None, :, None, None], synth.project(_bumped(rig, 2, rot), X), synth.project(rig, X))
    p2d = synth.add_noise(p2d, 0.7, seed=2)
    rep = ST.detect_moves(rig, p2d, times=t, window_s=2.0, min_step_deg=0.5, skip_s=0.2)
    name = rep.cameras[2]
    assert rep.moved == [name]
    (cp,) = rep.change_points
    assert abs(cp['time'] - tb) < 1 / 30 + 1e-9
    assert abs(cp['step_deg'] - np.linalg.norm(rot)) < 0.1
    st = rep.stations
    assert st.names(name) == [name, name + '@s1']
    assert st.lookup_name(name, [tb, tb + 0.2]).tolist() == ['skip', name + '@s1']
    np.testing.assert_allclose(np.degrees(rep.station_rotations[name][name + '@s1']), rot, atol=0.05)
    E = rep.corrected_extrinsics(rig)[name][name + '@s1']
    np.testing.assert_allclose(E, synth.to_np(_bumped(rig, 2, rot).cameras[2].get_extrinsics_mat()), atol=2e-3)
    json.dumps(rep.to_dict())
    # timed observations in, same answer (only the bumped camera tested)
    C, N, J, _ = p2d.shape
    cam, fr, pt = np.meshgrid(np.arange(C), np.arange(N), np.arange(J), indexing='ij')
    ok = np.isfinite(p2d).all(-1)
    obs = dict(cam=cam[ok], point=pt[ok], uv=p2d[ok], t=t[fr[ok]])
    rep2 = ST.detect_moves(rig, obs, window_s=2.0, cameras=[2])
    assert [c['camera'] for c in rep2.change_points] == [name]
    assert abs(rep2.change_points[0]['time'] - tb) < 1 / 30 + 1e-6
