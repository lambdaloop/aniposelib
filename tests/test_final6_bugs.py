"""Regression tests for the six bugs found by the 1.1 final real-data rerun (FINAL.md 'Bugs')."""

import numpy as np
import pytest

import synth
from aniposelib import workflows as W

pytestmark = pytest.mark.integration


# ------------------------------------------------------------------ bug 4: W6 fill with times=

def test_w6_fill_times_never_bridges_time_holes():
    true = synth.load_rig('fisheye_5cam')
    p3d, _ = synth.skeleton(true, 40, model=synth.PERSON, size=3.5, spread=0.15, seed=3)
    p2d = synth.add_noise(synth.project(true, p3d), 0.5, seed=4)
    t = np.arange(40) / 30.0
    t[20:] += 60.0                       # a 60 s hole between frames 19 and 20
    p2d[:, 19] = np.nan                  # one missing frame next to it: 1 index, 60 s
    p2d[:, 10] = np.nan                  # an ordinary 1-frame gap
    r = W.reconstruct_3d(true, p2d, times=t)
    assert np.isnan(r.points3d[19]).all()
    assert np.isfinite(r.points3d[10]).all()
    assert r.settings['max_gap_s'] == pytest.approx(11 / 30.0)
    r = W.reconstruct_3d(true, p2d, times=t, options={'max_gap': None})
    assert np.isfinite(r.points3d[19]).all()       # max_gap=None still fills every gap


# ------------------------------------------------------------------ bug 3: fit_similarity robust start

def test_fit_similarity_robust_init_keeps_inlier_cameras():
    from aniposelib import scale as S
    ang = np.linspace(0, 2 * np.pi, 5, endpoint=False) + 0.3
    c = np.c_[3 * np.cos(ang), 3 * np.sin(ang), 1.5 + 0.3 * np.sin(2 * ang)]
    ref = c.copy()
    ref[2] += [0.05, 0, 0]                  # one camera 50 mm off in the reference
    n3, err = 0, []
    for seed in range(60):
        r = np.random.default_rng(seed)
        src = (c + r.normal(0, 0.018 / np.sqrt(3), c.shape)) * 0.7
        s, R, t, res, use = S.fit_similarity(src, ref, trim=2.5, min_keep=3, robust_init=True)
        n3 += int(use.sum()) <= 3
        err.append(abs(s * 0.7 - 1))
    assert n3 <= 6                          # was most trials (3 of 5 cameras kept)
    assert np.percentile(err, 90) < 0.006
    # a runaway camera is still rejected
    src = c * 0.7
    src[1] += [5.0, 0, 0]
    s, R, t, res, use = S.fit_similarity(src, c, trim=2.5, min_keep=3, robust_init=True)
    assert not use[1] and use.sum() == 4 and abs(s * 0.7 - 1) < 1e-6


# ------------------------------------------------------------------ bug 5: W2 intrinsics gate evidence

def _wrong_lens(true, dk1=0.15):
    from aniposelib.workflows import calibration as WC
    g = true.copy()
    for i, cam in enumerate(g.cameras):
        d = synth.to_np(cam.get_distortions()).reshape(-1).copy()
        d[0] += dk1 * (-1) ** i
        WC._set_intrinsics(cam, synth.to_np(cam.get_camera_matrix()), d)
    return g


def test_w2_intrinsics_gate_needs_heldout_evidence():
    from aniposelib.workflows import calibration as WC
    true = synth.load_rig('outdoor_12cam')
    p3 = synth.trajectory(true, 1000, seed=2)
    p2 = synth.add_noise(synth.project(true, p3), 0.7, seed=2)
    opts = dict(WC.W2_DEFAULTS)
    cg, info = WC._gate_intrinsics(_wrong_lens(true, 0.3), p2, dict(window=np.arange(1000) // 10), 'auto',
                                   0.7, opts, 0)
    assert info['decision'] == 'refined', info
    assert info['p_better'] >= 0.95 and info['n_heldout_windows'] >= 10
    # same lens error, a handful of frames: refused for lack of held-out evidence, and says so
    cg, info = WC._gate_intrinsics(_wrong_lens(true, 0.3), p2[:, :60], dict(window=np.arange(60)), 'auto',
                                   0.7, opts, 0)
    assert info['decision'] == 'kept' and 'held-out evidence' in info['reason'], info


# ------------------------------------------------------------------ bug 1: W4 auto windows on long sessions

def test_w4_auto_window_at_most_an_hour():
    from aniposelib.workflows import calibration as WC
    assert WC._auto_window(np.linspace(0, 46 * 3600, 1000)) == 3600.0      # was 20000 s
    assert WC._auto_window(np.linspace(0, 60, 1000)) == 2.0                 # short trials unchanged
    assert WC._auto_window(np.linspace(0, 3100, 1000)) == 200.0


# ------------------------------------------------------------------ bug 2: W4 static offset is not a move

def _shifted(cg, cam, d):
    from aniposelib.workflows import calibration as WC
    g = cg.copy()
    c = g.cameras[cam]
    R = WC._rotmat(c)
    tv = synth.to_np(c.get_translation()).reshape(3)
    centre = -R.T @ tv + np.asarray(d, float)
    c.set_translation(-R @ centre)
    return g


@pytest.mark.parametrize('frac', [0.03, 0.12])
def test_w4_static_offset_is_not_a_move(frac):
    from aniposelib.workflows import evaluate_calibration
    rig = synth.load_rig('motobs_7cam')
    fn = synth.smooth_tracks(rig, n_points=6, amp=0.4, seed=1)
    t = np.arange(0, 40, 1 / 10)
    X = np.stack([fn(p, t) for p in range(6)], 1)
    p2 = synth.add_noise(synth.project(rig, X), 0.7, seed=2)
    bad = _shifted(rig, 2, [frac * synth.rig_scale(rig), 0, 0])     # wrong for the whole trial
    ev = evaluate_calibration(bad, p2, times=t)
    assert ev.moves.moved == [], (ev.moves.change_points, ev.report.get('moves_demoted'))
    assert any('loo' in r for r in ev.reasons)                       # the LOO rules report it


# ------------------------------------------------------------------ bug 6: W1 focal_guess and the drag verdict

def _w1_board():
    from aniposelib.boards import CharucoBoard
    return CharucoBoard(6, 5, 24, 18, marker_bits=4, dict_size=50, units='mm')


def test_w1_focal_guess_constrains_the_ladder(monkeypatch):
    from aniposelib import intrinsics as I
    from aniposelib.workflows import calibration as WC
    true = synth.load_rig('allen_7cam')
    board = _w1_board()
    rows = synth.board_rows(true, board, n_frames=12, seed=1)
    name = true.get_names()[3]
    f_true = float(synth.to_np(true.cameras[3].get_camera_matrix())[0, 0])
    orig = I.calibrate_ladder_rows

    def far(r, b, size, **kw):            # the ladder of camera 3 converges to a wrong focal (phone)
        lr = orig(r, b, size, **kw)
        if kw.get('focal_inits'):
            for f in lr.fits.values():
                f.K = np.array(f.K, float)
                f.K[0, 0] *= 0.45
                f.K[1, 1] *= 0.45
        return lr
    monkeypatch.setattr(I, 'calibrate_ladder_rows', far)
    cg = WC._base_group(len(rows), true.get_names(), None, False, [true.cameras[0].get_size()] * len(rows))
    opts = WC.merge_options(WC.W1_DEFAULTS, {'focal_guess': {name: 0.95 * f_true}})
    lens, missing, fb = WC._fit_intrinsics(cg, WC._copy_rows(rows, board), board, 'auto', None, opts, 0)
    f = float(synth.to_np(cg.cameras[3].get_camera_matrix())[0, 0])
    assert lens[name]['source'] == 'focal_guess' and f == pytest.approx(0.95 * f_true)
    assert any(name in m for m in fb)
    f0 = float(synth.to_np(cg.cameras[0].get_camera_matrix())[0, 0])
    assert f0 == pytest.approx(f_true, rel=0.05)                  # cameras without a guess: unchanged


def test_w1_drag_check_fails_when_one_camera_drags(monkeypatch):
    from aniposelib.workflows import calibrate_from_board
    from aniposelib.workflows import calibration as WC
    true = synth.load_rig('allen_7cam')
    board = _w1_board()
    rows = synth.board_rows(true, board, n_frames=30, seed=1)
    init = true.copy()
    cam = init.cameras[3]
    K = synth.to_np(cam.get_camera_matrix()).copy()
    K[:2, :2] *= 0.7                     # a wrong, fixed lens: 20 x worse board error than the others
    WC._set_intrinsics(cam, K, synth.to_np(cam.get_distortions()))
    res = calibrate_from_board(rows, board, init=init, intrinsics='fixed')
    lv = {r['rule']: r['level'] for r in res.report['rules']}
    assert res.report['drag_check']['camera'] == true.get_names()[3]
    assert lv['camera_drag_deg'] == 'ok'                    # a well-connected rig is not dragged
    # same, but leaving the camera out moves the others by 5 deg: it dragged them -> fail
    orig = WC._board_ba
    calls = []

    def rerun(cg, *a, **k):
        out = orig(cg, *a, **k)
        calls.append(1)
        if len(calls) > 1:
            c = cg.cameras[0]
            c.set_rotation(synth.to_np(c.get_rotation()).reshape(3) + np.radians([0, 5.0, 0]))
        return out
    monkeypatch.setattr(WC, '_board_ba', rerun)
    res = calibrate_from_board(rows, board, init=init, intrinsics='fixed')
    lv = {r['rule']: r['level'] for r in res.report['rules']}
    assert lv['camera_drag_deg'] == 'fail' and res.verdict == 'fail'
    assert res.report['drag_check']['max_relative_rotation_deg'] > 3
