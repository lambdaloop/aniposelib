"""Workflows W5-W7 (plan 05 §5): estimate_offsets, reconstruct_3d, associate_views on the
synthetic use-case rigs. Kept small (few frames, short audio) so the suite stays fast."""

import json

import numpy as np
import pytest

import synth
from aniposelib import workflows as W
from test_sync import _bursts, _smooth_signal

pytestmark = pytest.mark.integration


# ------------------------------------------------------------------ W6 reconstruct_3d

@pytest.fixture(scope='module')
def person():
    true = synth.load_rig('fisheye_5cam')
    p3d, bones = synth.skeleton(true, 100, model=synth.PERSON, size=3.5, spread=0.15, seed=3)
    p2d = synth.add_noise(synth.project(true, p3d), 0.7, seed=4)
    p2d = synth.drop_views(p2d, frac=0.3, seed=5)
    p2d[:, :8] = np.nan                 # nobody sees the first frames: never extrapolated
    p2d[:, 40:46, 3] = np.nan           # an interior gap of joint 3
    p2d = synth.add_outliers(p2d, 0.03, cgroup=true, seed=6)
    return true, p3d, p2d, bones


def _rel(cg, X, p3d, s=1.0):
    return np.linalg.norm(X - p3d * s, axis=-1) / (s * synth.rig_scale(cg))


def bone_cv(x, bones):
    L = np.array([np.linalg.norm(x[:, a] - x[:, b], axis=-1) for a, b in bones])
    return np.nanmedian(np.nanstd(L, axis=1) / np.nanmean(L, axis=1))


def test_reconstruct_person_gaps_bones(person):
    true, p3d, p2d, bones = person
    plain = W.reconstruct_3d(true, p2d)
    res = W.reconstruct_3d(true, p2d, bones=bones, options={'n_iters': 150})
    for r in (plain, res):
        assert r.verdict == 'ok', r.reasons
        assert r.points3d.shape == p3d.shape and r.errors.shape == p2d.shape[:-1]
        assert np.all(np.isnan(r.points3d[:8]))                  # no extrapolation
        assert np.isfinite(r.points3d[40:46, 3]).all() and r.filled[40:46, 3].all()
        assert (r.n_views[40:46, 3] == 0).all()
        assert np.nanmedian(_rel(true, r.points3d, p3d)) < 2e-3
    # outliers are rejected, bones tighten the skeleton
    assert plain.report['per_camera'][true.get_names()[0]]['usage'] > 0.9
    assert bone_cv(res.points3d[8:], bones) < 0.5 * bone_cv(plain.points3d[8:], bones)
    assert res.report['bones']['median_cv'] < res.report['bones']['median_cv_triangulated']
    gaps = res.report['gaps']
    assert gaps['extrapolated'] == 0 and gaps['longest_gap'] >= 8
    for k in ('noise_px', 'tau_px', 'optim_points', 'weighting', 'rig_scale'):
        assert k in res.settings
    assert 0.5 <= res.settings['noise_px'] < 1.5
    json.dumps(res.to_dict())
    assert 'bone_cv' in res.summary()
    nf = W.reconstruct_3d(true, p2d, fill='none')
    assert np.isnan(nf.points3d[40:46, 3]).all()


@pytest.mark.parametrize('scale', [1e-3, 1.0, 1e3])
def test_reconstruct_unit_scale_invariant(person, scale):
    """Same call, default options, at three unit scales: same verdict and relative accuracy."""
    true0, p3d, p2d, bones = person
    true = synth.load_rig('fisheye_5cam')
    true.set_translations(synth.to_np(true.get_translations()) * scale)
    res = W.reconstruct_3d(true, p2d, bones=bones, options={'n_iters': 100})
    ref = W.reconstruct_3d(true0, p2d, bones=bones, options={'n_iters': 100})
    assert res.verdict == 'ok', res.reasons
    np.testing.assert_allclose(res.points3d / scale, ref.points3d, rtol=0, atol=1e-6 * synth.rig_scale(true0))
    assert np.isclose(res.settings['tau_px'], ref.settings['tau_px'])


def test_reconstruct_soft_scores(person):
    """Low scores mark corrupted views (below tau, so MSAC keeps them): weighting beats none."""
    true, p3d, p2d, _ = person
    rng = np.random.default_rng(7)
    bad = (rng.random(p2d.shape[:-1]) < 0.25) & np.isfinite(p2d).all(-1)
    corrupted = p2d + np.where(bad[..., None], rng.normal(scale=4.0, size=p2d.shape), 0.0)
    scores = np.where(bad, 0.05, 1.0)
    plain = W.reconstruct_3d(true, corrupted)
    soft = W.reconstruct_3d(true, corrupted, scores=scores)
    e = lambda r: np.nanmedian(_rel(true, r.points3d, p3d))
    assert e(soft) < 0.75 * e(plain)
    assert soft.inliers[bad].mean() > 0.5                       # soft: no hard drops
    assert 'soft' in soft.settings['weighting']
    # an explicit hard cut is available
    cut = W.reconstruct_3d(true, corrupted, scores=scores, options={'min_score': 0.5})
    assert not cut.inliers[bad].any() and cut.settings['min_score'] == 0.5


def test_reconstruct_bad_inputs(person):
    true, p3d, p2d, _ = person
    r = W.reconstruct_3d(true, np.full_like(p2d, np.nan))
    assert r.verdict == 'fail' and any('no_points' in x or 'reconstructed_frac' in x for x in r.reasons)
    one = p2d.copy(); one[2] = np.nan
    r = W.reconstruct_3d(true, one)
    assert r.verdict == 'warn' and true.get_names()[2] in r.report['cameras_unused']
    with pytest.raises(ValueError):
        W.reconstruct_3d(true, p2d[:3])
    with pytest.raises(ValueError):
        W.reconstruct_3d(true, p2d, fill='all')
    flat = W.reconstruct_3d(true, p2d[:, :, 0])                 # C x N x 2 input
    assert flat.points3d.shape == (p2d.shape[1], 3)


def test_reconstruct_chunked(person):
    """Long sequences are optimised in overlapping, blended chunks."""
    true, p3d, p2d, bones = person
    res = W.reconstruct_3d(true, p2d, bones=bones,
                           options={'n_iters': 60, 'chunk_frames': 40, 'overlap_frames': 10})
    assert res.settings['optim_points']['n_chunks'] >= 3
    assert np.nanmedian(_rel(true, res.points3d, p3d)) < 3e-3
    assert np.all(np.isnan(res.points3d[:8]))


# ------------------------------------------------------------------ W7 associate_views

@pytest.fixture(scope='module')
def crowd():
    cg = synth.load_rig('fisheye_5cam')
    P3 = synth.crowd(cg, 3, 60, sep=1.0, phase=0.35, seed=10)          # [P, T, J, 3]
    p2d = synth.add_noise(synth.project(cg, P3), 1.0, seed=1)            # [C, P, T, J, 2]
    rng = np.random.default_rng(0)
    dets, ids, truth = [], [], []
    for c in range(p2d.shape[0]):
        perm = rng.permutation(3) + 10 * c
        d_c, i_c, t_c = {}, {}, {}
        for t in range(P3.shape[1]):
            o = rng.permutation(3)
            d_c[t], i_c[t], t_c[t] = p2d[c, o, t], perm[o], o
        dets.append(d_c); ids.append(i_c); truth.append(t_c)
    return cg, P3, dets, ids, truth


@pytest.mark.parametrize('mode', ['tracklets', 'per_frame'])
def test_associate_crowd(crowd, mode):
    cg, P3, dets, ids, truth = crowd
    res = W.associate_views(cg, dets, track_ids=ids if mode == 'tracklets' else None)
    C, T = len(cg.cameras), P3.shape[1]
    assert res.verdict == 'ok', res.reasons
    assert res.points.shape == (C, T, 3, P3.shape[2], 2)
    assert res.report['assigned_frac'] > 0.95 and res.settings['mode'] == mode
    assert res.settings['gate_px'] <= res.settings['first_gate_px']
    # every identity is one person, and the dense points reconstruct that person
    for p in range(3):
        who = {int(truth[c][f][i]) for c in range(C) for f in range(T)
               for i in np.nonzero(res.labels[c][f] == p)[0]}
        assert len(who) == 1
        rec = W.reconstruct_3d(cg, res.points[:, :, p])
        assert rec.verdict == 'ok', rec.reasons
        assert np.nanmedian(_rel(cg, rec.points3d, P3[who.pop()])) < 3e-3
    json.dumps(res.to_dict())


def test_associate_bad_inputs(crowd):
    cg, P3, dets, ids, truth = crowd
    only = [d if c == 0 else {} for c, d in enumerate(dets)]   # one camera sees everything
    res = W.associate_views(cg, only)
    assert res.verdict == 'fail' and any('no_identities' in r for r in res.reasons)
    with pytest.raises(ValueError):
        W.associate_views(cg, dets[:2])


# ------------------------------------------------------------------ W5 estimate_offsets

@pytest.fixture(scope='module')
def sync_scene():
    cg = synth.load_rig('allen_7cam').subset_cameras([0, 1, 2])
    names = cg.get_names()
    c0, s = synth.scene_center(cg), synth.rig_scale(cg)
    rng = np.random.default_rng(0)
    fx = [_smooth_signal(10, 1.0, rng, f_max=0.03, n_sines=4) for _ in range(3)]
    offs = rng.normal(scale=0.03 * s, size=(6, 3))

    def X(t):
        base = np.stack([f(np.asarray(t, float)) for f in fx], -1) / 4 * 0.12 * s + c0
        return base[:, None] + offs[None]
    sr, dur, fps = 8000, 20.0, 30.0
    base = _bursts(int(sr * (dur + 4)), sr, rng, n_events=50)
    off = {names[0]: 0.0, names[1]: 0.4137, names[2]: -0.2561}
    rate = {names[0]: 0.0, names[1]: 0.0, names[2]: 3e-4}    # names[2]'s clock drifts
    streams = {}
    for c, n in enumerate(names):
        src = np.arange(int(sr * dur)) / sr
        g = (src - off[n]) / (1 + rate[n])
        audio = np.interp((g + 2) * sr, np.arange(len(base)), base) + 0.05 * rng.normal(size=len(src))
        sf = np.arange(int(fps * dur)) / fps
        p = synth.add_noise(synth.project(cg, X((sf - off[n]) / (1 + rate[n]) * fps + 60))[c], 0.3, seed=c)
        streams[n] = dict(audio=(audio, sr), points=(p, sf))
    return cg, streams, off, rate


def test_offsets_audio_epipolar_drift(sync_scene):
    cg, streams, off, rate = sync_scene
    names = list(streams)
    res = W.estimate_offsets(streams, cgroup=cg, reference=names[0], refine=True)
    assert res.verdict == 'ok', res.reasons
    assert res.reference == names[0] and res.offsets[names[0]] == 0.0
    for n in names:
        assert abs(res.offsets[n] - off[n]) < 1e-3, (n, res.offsets[n], off[n])
        assert res.identifiable[n]
    assert abs(res.rates[names[2]] - rate[names[2]]) < 5e-5
    # the time maps are what the caller applies: source = global + offset(global)
    tm = res.timemaps[names[2]]
    g = 10.0
    assert abs(tm.to_source(g) - ((1 + rate[names[2]]) * g + off[names[2]])) < 1e-3
    kinds = {(p['kind'], p['method']) for p in res.pairs}
    assert {('audio', 'gcc'), ('audio', 'spectrogram'), ('audio', 'onset')} <= kinds
    assert any(k == 'epipolar' for k, _ in kinds)
    assert res.refined['ok'] and res.settings['stba']['accel_sigma_rel'] == 10.0
    json.dumps(res.to_dict())


def test_offsets_unidentifiable_fail():
    rng = np.random.default_rng(3)
    sr = 8000
    streams = {'a': {'audio': (rng.normal(size=sr * 10), sr)},
               'b': {'audio': (rng.normal(size=sr * 10), sr)}}
    res = W.estimate_offsets(streams, options={'audio_windows': 0, 'audio_cross_check': False})
    assert res.verdict == 'fail'
    assert any('unidentifiable_streams' in r for r in res.reasons)
    assert res.timemaps['b'] is None or res.timemaps['a'] is None
    with pytest.raises(ValueError):
        W.estimate_offsets({'a': {'audio': (np.zeros(10), sr)}})
    with pytest.raises(ValueError):
        W.estimate_offsets({'a': {}, 'b': {'audio': (np.zeros(10), sr)}})


def test_offsets_from_audio_files(tmp_path):
    """Streams given as file paths: audio decoded with sync.load_audio ([sync] extra)."""
    sf = pytest.importorskip('soundfile')
    from test_sync import _delayed
    sr = 8000
    rng = np.random.default_rng(5)
    a = _bursts(sr * 12, sr, rng)
    b = _delayed(a, sr, 0.75, rng=rng)
    sf.write(str(tmp_path / 'a.wav'), (0.3 * a).astype('float32'), sr)
    sf.write(str(tmp_path / 'b.wav'), (0.3 * b).astype('float32'), sr)
    res = W.estimate_offsets({'a': str(tmp_path / 'a.wav'), 'b': tmp_path / 'b.wav'},
                             reference='a', options={'audio_sr': sr})
    assert res.verdict == 'ok', res.reasons
    assert abs(res.offsets['b'] - 0.75) < 1e-3
