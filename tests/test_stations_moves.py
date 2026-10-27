"""detect_moves robustness: per-window minimum support and re-test without moved cameras."""
import copy

import numpy as np
import pytest
import torch

import synth
from aniposelib import geometry as G
from aniposelib import stations as ST


def _bumped(cg, cam, rot_deg):
    cgb = copy.deepcopy(cg)
    c = cgb.cameras[cam]
    dR = G.rodrigues(torch.tensor(np.radians(rot_deg), dtype=torch.float64))
    with torch.no_grad():
        R = G.rodrigues(c.rvec.reshape(3).double())
        c.rvec.copy_(torch.as_tensor(ST._rotvec((dR @ R).numpy())).reshape(c.rvec.shape))
        c.tvec.copy_((dR @ c.tvec.reshape(3).double()).reshape(c.tvec.shape))
    return cgb


def _tracks(rig, n_points=8, dur=24.0, fps=15.0):
    fn = synth.smooth_tracks(rig, n_points=n_points, amp=0.1, seed=1)
    t = np.arange(0, dur, 1 / fps)
    return t, np.stack([fn(p, t) for p in range(n_points)], 1)


def _moved(rig, X, t, moves, sigma=0.5):
    """``moves = [(cam, t_move, rot_deg)]`` -> noisy 2D with each camera re-aimed from ``t_move``."""
    p2 = synth.project(rig, X)
    for cam, tm, rot in moves:
        pb = synth.project(_bumped(rig, cam, rot), X)
        p2[cam] = np.where((t >= tm)[:, None, None], pb[cam], p2[cam])
    return synth.add_noise(p2, sigma, seed=2)


def _sparse_bumped_tail():
    rig = synth.load_rig('motobs_7cam')
    t, X = _tracks(rig, n_points=12)
    p2 = synth.add_noise(synth.project(rig, X), 0.5, seed=2)
    pb = synth.add_noise(synth.project(_bumped(rig, 2, [2.0, -1.5, 1.0]), X), 0.5, seed=3)
    rng = np.random.default_rng(0)
    for w in (21, 22, 23):                   # last 3 one-second windows: 25 garbage points each
        m = (t >= w) & (t < w + 1)
        sub, subb = p2[2, m], pb[2, m]
        keep = np.zeros(sub.shape[:2], bool)
        keep.flat[rng.choice(keep.size, 25, replace=False)] = True
        sub[keep] = subb[keep]
        sub[~keep] = np.nan
        p2[2, m] = sub
    return rig, t, p2


def test_sparse_windows_cannot_form_change_point():
    rig, t, p2 = _sparse_bumped_tail()
    name = rig.get_names()[2]
    rep = ST.detect_moves(rig, p2, times=t, window_s=1.0)
    assert rep.change_points == [] and rep.moved == []
    assert rep.info['low_support'] == {name: [21, 22, 23]}
    assert np.isnan(rep.delta_deg[2, 21:]).all() and np.isfinite(rep.delta_deg[2, :21]).all()
    assert (rep.n_points[2, 21:] == 25).all()            # counts are still reported
    old = ST.detect_moves(rig, p2, times=t, window_s=1.0, min_support_frac=None)
    assert old.moved == [name] and old.change_points[0]['window'] == 21
    assert old.info['low_support'] == {}


def test_clean_and_dense_results_unchanged():
    rig = synth.load_rig('motobs_7cam').subset_cameras([1, 2, 3, 5, 6])
    t, X = _tracks(rig)
    p2 = _moved(rig, X, t, [(1, 13.2, [1.2, -1.0, 0.8])])
    new = ST.detect_moves(rig, p2, times=t)
    old = ST.detect_moves(rig, p2, times=t, min_support_frac=None, retest=False)
    np.testing.assert_array_equal(new.delta_deg, old.delta_deg)
    assert new.change_points == old.change_points and new.moved == [rig.get_names()[1]]


def test_retest_clears_cameras_flagged_because_of_the_moved_one():
    # tau=1000 px: the leave-one-out triangulation keeps the re-aimed camera, which biases the
    # other cameras' offsets (the real-data failure mode: a FP next to the true move)
    rig = synth.load_rig('motobs_7cam').subset_cameras([0, 1, 2, 3])
    names = rig.get_names()
    t, X = _tracks(rig)
    p2 = _moved(rig, X, t, [(1, 13.2, [1.2, -1.0, 0.8])])
    first = ST.detect_moves(rig, p2, times=t, tau=1000.0, retest=False)
    assert names[1] in first.moved and len(first.moved) >= 2      # false positives without re-test
    rep = ST.detect_moves(rig, p2, times=t, tau=1000.0)
    assert rep.moved == [names[1]]
    assert rep.info['retest'][names[1]]['result'] == 'kept'
    for n in first.moved:
        if n != names[1]:
            assert rep.info['retest'][n]['result'] == 'cleared'
            assert rep.info['retest'][n]['explained_by'] == names[1]
            assert rep.stations.names(n) == [n]
    assert rep.change_points[0]['z'] > 6 and abs(rep.change_points[0]['time'] - 13.2) < 0.2


def test_retest_keeps_a_second_true_move():
    rig = synth.load_rig('motobs_7cam').subset_cameras([1, 2, 3, 5, 6])
    names = rig.get_names()
    t, X = _tracks(rig)
    p2 = _moved(rig, X, t, [(1, 13.2, [1.2, -1.0, 0.8]), (3, 7.1, [-0.8, 1.0, -1.0])])
    first = ST.detect_moves(rig, p2, times=t, window_s=2.0, tau=1000.0, retest=False)
    assert set(first.moved) > {names[1], names[3]}
    rep = ST.detect_moves(rig, p2, times=t, window_s=2.0, tau=1000.0)
    assert set(rep.moved) == {names[1], names[3]}
    got = {cp['camera']: cp['time'] for cp in rep.change_points}
    assert got[names[1]] == pytest.approx(13.2, abs=0.2) and got[names[3]] == pytest.approx(7.1, abs=0.2)
    assert sorted(r['result'] for r in rep.info['retest'].values()).count('kept') == 2


def test_retest_two_moves_of_three_cameras_kept():
    # 3 cameras, two of them moved: correcting one does not explain the other away
    rig = synth.load_rig('motobs_7cam').subset_cameras([1, 2, 3])
    names = rig.get_names()
    t, X = _tracks(rig)
    p2 = _moved(rig, X, t, [(0, 13.2, [1.5, -1.0, 0.8]), (2, 7.1, [-1.0, 1.0, -1.0])])
    first = ST.detect_moves(rig, p2, times=t, window_s=2.0, retest=False)
    rep = ST.detect_moves(rig, p2, times=t, window_s=2.0)
    assert rep.moved == first.moved and len(rep.moved) >= 2
    assert all(r['result'] == 'kept' for r in rep.info['retest'].values())
