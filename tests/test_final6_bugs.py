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
