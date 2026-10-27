"""1.1 moves: W6 static-artifact flags and mask_regions; W4 moved-camera recovery (recover_moves)."""
import json

import numpy as np
import pytest

import synth
from aniposelib import workflows as W


# ------------------------------------------------------------------ W6 static flags / mask_regions

@pytest.fixture(scope='module')
def static_scene():
    rig = synth.load_rig('motobs_7cam')
    N = 1800
    t = np.sort(np.random.default_rng(0).uniform(0, 6 * 3600, N))      # sparse frames over 6 h
    X = synth.trajectory(rig, N, spread=0.25, step=0.02, seed=1)        # the animal
    block = synth.scene_center(rig) + 0.2 * synth.rig_scale(rig) * np.array([1.0, 0.5, 0.0])
    Xb = np.repeat(block[None], N, 0)
    on = np.arange(N) % 3 == 0                                          # a static detection every 3rd frame
    p2 = synth.add_noise(synth.project(rig, X), 0.5, seed=2)
    pb = synth.add_noise(synth.project(rig, Xb), 0.5, seed=3)
    # the static point is a false detection in 2 cameras, replacing the animal there
    p2[:2, on] = pb[:2, on]
    p2[2:, on] = np.nan
    return rig, t, p2, on, pb


def test_static_flags_and_mask_regions(static_scene):
    rig, t, p2, on, pb = static_scene
    res = W.reconstruct_3d(rig, p2, times=t, fill='none')
    assert res.static.shape == (len(t),)
    assert res.static[on].mean() > 0.95 and res.static[~on].mean() < 0.01
    st = res.report['static']
    assert st['n_points'] == int(res.static.sum()) and st['n_static_voxels'] >= 1
    assert any(r['rule'] == 'static_frac' and r['level'] == 'warn' for r in res.report['rules'])
    assert np.isfinite(res.points3d[on]).all()                          # flag only by default
    json.dumps(res.to_dict())
    # static_remove drops them; static_flag=False keeps the old output
    rm = W.reconstruct_3d(rig, p2, times=t, fill='none', options={'static_remove': True})
    assert np.isnan(rm.points3d[on]).all() and np.isfinite(rm.points3d[~on]).mean() > 0.95
    off = W.reconstruct_3d(rig, p2, times=t, fill='none', options={'static_flag': False})
    assert not off.static.any() and off.report['static'] is None
    # too short a recording for 5 distinct hours: nothing flagged
    short = W.reconstruct_3d(rig, p2, times=t / 10, fill='none')
    assert not short.static.any()
    # mask_regions: a circle around the static point in camera 0 removes it there
    x, y = np.nanmedian(pb[0, on], 0)
    name = rig.get_names()[0]
    mk = W.reconstruct_3d(rig, p2, times=t, fill='none', options={'mask_regions': {name: [(x, y, 20.0)]}})
    assert int(on.sum()) <= mk.report['n_removed']['mask_regions'] <= int(on.sum()) + 10
    assert np.isnan(mk.points3d[on]).all() and not mk.static.any()


# ------------------------------------------------------------------ W4 moved-camera recovery

def _bumped(cg, cam, rot_deg, shift=None):
    import copy
    import torch
    from aniposelib import geometry as G
    from aniposelib import stations as ST
    cgb = copy.deepcopy(cg)
    c = cgb.cameras[cam]
    dR = G.rodrigues(torch.tensor(np.radians(rot_deg), dtype=torch.float64))
    with torch.no_grad():
        R = G.rodrigues(c.rvec.reshape(3).double())
        c.rvec.copy_(torch.as_tensor(ST._rotvec((dR @ R).numpy())).reshape(c.rvec.shape))
        tv = dR @ c.tvec.reshape(3).double()
        if shift is not None:
            tv = tv + torch.as_tensor(np.asarray(shift, float))
        c.tvec.copy_(tv.reshape(c.tvec.shape))
    return cgb


@pytest.fixture(scope='module')
def lost_camera():
    """3 people, fisheye rig; camera 2 is re-aimed by 5.8 deg at 16 s and the association (with
    the old rig) lost it from then on; the unassociated detections are shuffled per frame."""
    rig = synth.load_rig('fisheye_5cam')
    F, Pn, cb, tb = 300, 3, 2, 16.0
    X = synth.crowd(rig, Pn, F, size=3.5, sep=1.5, seed=0)
    t = np.arange(F) / 10.0
    rigb = _bumped(rig, cb, [3.0, -4.5, 2.0])
    p = np.stack([synth.project(rig, X[i]) for i in range(Pn)], 2)          # C x F x P x J x 2
    pb = np.stack([synth.project(rigb, X[i]) for i in range(Pn)], 2)
    p[cb, t >= tb] = pb[cb, t >= tb]
    p = synth.add_noise(p, 1.0, seed=1)
    rng = np.random.default_rng(0)
    det = [{f: p[c, f][rng.permutation(Pn)] for f in range(F)} for c in range(p.shape[0])]
    assoc = p.copy()
    assoc[cb, t >= tb] = np.nan
    return rig, rigb, cb, tb, t, p, assoc, det


def test_w4_recovers_lost_camera_and_reassociates(lost_camera):
    rig, rigb, cb, tb, t, p, assoc, det = lost_camera
    name = rig.get_names()[cb]
    ev = W.evaluate_calibration(rig, assoc, times=t, options={'detections': det})
    assert ev.moves.moved == [name] and 'moved_cameras' in ' '.join(ev.reasons)
    (r,) = ev.report['recovered_moves'][name]
    assert r['status'] == 'recovered' and abs(r['time'] - tb) < 0.15
    assert abs(r['step_deg'] - np.linalg.norm([3.0, -4.5, 2.0])) < 0.1
    (cp,) = ev.moves.change_points
    assert cp['recovered'] and cp['camera'] == name
    st = ev.moves.stations
    sn = st.lookup_name(name, [tb + 1])[0]
    assert sn != name and st.lookup_name(name, [tb - 1])[0] == name and sn in st.poses
    E = ev.moves.corrected_extrinsics(rig)[name][sn]
    np.testing.assert_allclose(E, synth.to_np(rigb.cameras[cb].get_extrinsics_mat()), atol=2e-3)
    # the recovered camera is re-associated after the move (same shape as the input)
    assert ev.points.shape == p.shape
    post = t >= tb + 0.2
    got = ev.points[cb, post]
    assert np.isfinite(got).mean() > 0.9
    np.testing.assert_allclose(got[np.isfinite(got)], p[cb, post][np.isfinite(got)])
    json.dumps(ev.to_dict())
    # off: the old behaviour (nothing found: the camera has no data after the move)
    ev0 = W.evaluate_calibration(rig, assoc, times=t, options={'recover_moves': False})
    assert ev0.moves.moved == [] and ev0.points is None


def test_recover_without_detections_and_station_ba(lost_camera):
    """A 12 deg re-aim with translation, the camera's own (associated) points: the rotation fit of
    detect_moves cannot follow, recovery re-places it and the station BA starts from it."""
    from aniposelib import bundle as B
    from aniposelib import stations as ST
    rig = synth.load_rig('motobs_7cam')
    t = np.arange(0, 24, 1 / 15.0)
    fn = synth.smooth_tracks(rig, n_points=10, amp=0.1, seed=1)
    X = np.stack([fn(k, t) for k in range(10)], 1)
    s = synth.rig_scale(rig)
    rigb = _bumped(rig, 3, [8.0, -9.0, 0.0], shift=[0.03 * s, 0.0, 0.0])
    p2 = synth.project(rig, X)
    p2[3, t >= 12] = synth.project(rigb, X)[3, t >= 12]
    p2 = synth.add_noise(p2, 0.5, seed=2)
    rec = ST.recover_moves(rig, p2, t, window_s=1.0)
    name = rig.get_names()[3]
    (r,) = rec.cameras[name]
    assert r['status'] == 'recovered' and abs(r['time'] - 12) < 0.1 and rec.points is None
    sn = rec.moves.stations.lookup_name(name, [13])[0]
    E = rec.moves.corrected_extrinsics(rig)[name][sn]
    Et = synth.to_np(rigb.cameras[3].get_extrinsics_mat())
    assert np.abs(E[:3, :3] - Et[:3, :3]).max() < 2e-3
    # station BA from the recovered start pose
    C, N, J, _ = p2.shape
    cam, fr, pt = np.meshgrid(np.arange(C), np.arange(N), np.arange(J), indexing='ij')
    ok = np.isfinite(p2).all(-1)
    obs = B.Observations(cam[ok], (fr * J + pt)[ok], p2[ok], time=t[fr[ok]], n_points=N * J, n_cams=C)
    rep = B.bundle_adjust_lm(rig, obs, stations=rec.moves.stations, only_extrinsics=True, return_report=True,
                             inplace=False)
    assert rep.error < 1.0 and sn in rep.stations['names']
