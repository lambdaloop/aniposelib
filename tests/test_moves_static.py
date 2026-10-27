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
