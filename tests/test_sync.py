"""Tests for aniposelib.sync (F1, F2, F7): synthetic audio, streams, epipolar lag and soft-DTW."""
import json

import numpy as np
import pytest
import torch

import synth
from aniposelib import sync


# ------------------------------------------------------------------ synthetic signals

def _bursts(n, sr, rng, n_events=40):
    x = 0.05 * rng.normal(size=n)
    t = np.arange(n) / sr
    for _ in range(n_events):
        s = rng.integers(0, n - 2000)
        L = int(rng.integers(200, 1500))
        if rng.random() < 0.5:
            x[s:s + L] += rng.normal(size=L) * rng.uniform(0.5, 2)
        else:  # chirp
            tt = t[:L]
            f0, f1 = rng.uniform(300, 1500), rng.uniform(1500, 3500)
            x[s:s + L] += np.sin(2 * np.pi * (f0 * tt + (f1 - f0) * tt ** 2 / (2 * tt[-1]))) * rng.uniform(.5, 2)
    return x


def _delayed(x, sr, offset, rate=0.0, gain=0.7, noise=0.05, rng=None):
    """b(t) = gain * x((t - offset) / (1 + rate)): an event at time t in x appears at offset + (1+rate) t."""
    rng = rng or np.random.default_rng(1)
    tb = np.arange(len(x)) / sr
    ta = (tb - offset) / (1 + rate)
    b = gain * np.interp(ta * sr, np.arange(len(x)), x, left=0, right=0)
    return b + noise * rng.normal(size=len(x))


def _smooth_signal(n, rate, rng, f_max=1.5, n_sines=12):
    t = np.arange(n) / rate
    fr = rng.uniform(f_max / 15, f_max, n_sines)
    ph = rng.uniform(0, 2 * np.pi, n_sines)
    amp = rng.uniform(0.5, 1, n_sines)
    return lambda tt: np.sum(amp[:, None] * np.sin(2 * np.pi * fr[:, None] * np.asarray(tt)[None] + ph[:, None]),
                             axis=0)


# ------------------------------------------------------------------ TimeMap / align_streams

def test_timemap_roundtrip_and_convention():
    tm = sync.TimeMap.from_offset(2.5)
    assert np.isclose(tm.to_source(10.0), 12.5)          # source = global + offset
    assert np.isclose(tm.to_global(12.5), 10.0)
    tm = sync.TimeMap(c=-1.0, beta=1.0001, knots=((0.0, 0.0), (50.0, 0.02), (100.0, -0.01)))
    pts = np.linspace(-20, 150, 500)
    assert np.allclose(tm.to_source(tm.to_global(pts)), pts, atol=1e-9)
    assert sync.TimeMap.from_dict(json.loads(json.dumps(tm.to_dict()))) == tm
    with pytest.raises(ValueError):
        sync.TimeMap(knots=((0, 0), (1, -2)))  # slope -2: not invertible


def test_align_streams_no_reuse_and_offsets():
    # a 25 Hz stream on a 30 Hz grid: no frame may be used twice
    t = np.arange(250) / 25.0
    v = np.arange(250, dtype=float)
    out = sync.align_streams({'a': (t, v)}, rate=30, tol=1 / 30)
    idx = out.src_idx[0]
    used = idx[idx >= 0]
    assert len(used) == len(np.unique(used))
    assert len(used) >= 240
    assert np.all(np.abs(out.t_grid[idx >= 0] - t[used]) <= 1 / 30 + 1e-12)
    reuse = sync.align_streams({'a': (t, v)}, rate=30, tol=1 / 30, reuse=True)
    r = reuse.src_idx[0]
    assert len(r[r >= 0]) > len(np.unique(r[r >= 0]))  # merge_asof-like behaviour duplicates frames

    # offsets in the module convention: source = global + offset
    tb = np.arange(100) / 25.0 + 0.4  # b's clock is 0.4 s ahead
    out = sync.align_streams({'a': (t[:100], v[:100]), 'b': (tb, v[:100] + 1000)}, rate=25,
                             offsets={'b': 0.4}, span='intersection')
    assert out.values.shape == (2, 100)
    assert np.allclose(out.values[1] - out.values[0], 1000)


def test_align_streams_datetimes_and_mixed():
    pd = pytest.importorskip('pandas')
    base = np.datetime64('2026-09-28T19:53:17', 'ns')
    t1 = base + (np.arange(50) * 40e6).astype('timedelta64[ns]')
    t2 = pd.Series(pd.to_datetime(t1[::2]).tz_localize('UTC').astype(str))  # strings, tz-aware
    out = sync.align_streams({'a': (t1, np.ones((50, 2))), 'b': (list(t2) + ['garbage'],
                                                                np.ones((26, 2)))}, rate=25)
    assert out.t_grid.dtype.kind == 'M'
    assert out.values.shape == (2, 50, 2)
    assert np.isfinite(out.values[1, ::2]).all() and np.isnan(out.values[1, 1::2]).all()


# ------------------------------------------------------------------ audio

@pytest.mark.parametrize('method,tol', [('gcc', 2e-4), ('spectrogram', 2e-4), ('onset', 0.012)])
@pytest.mark.parametrize('offset', [1.2345, -0.789])
def test_audio_offset_methods(method, tol, offset):
    sr = 8000
    rng = np.random.default_rng(0)
    a = _bursts(sr * 20, sr, rng)
    b = _delayed(a, sr, offset, rng=rng)
    est = sync.audio_offset(a, b, sr, max_lag=4, method=method)
    assert est.identifiable, est.reasons
    assert abs(est.offset - offset) < tol
    assert est.sigma > 0 and np.isfinite(est.sigma)
    json.dumps(est.to_dict())


def test_audio_offset_reports_unidentifiable():
    sr = 8000
    rng = np.random.default_rng(3)
    a, b = rng.normal(size=sr * 10), rng.normal(size=sr * 10)  # unrelated noise
    est = sync.audio_offset(a, b, sr, max_lag=2, method='gcc', windows=0)
    assert not est.identifiable
    # a true offset outside the search range is a boundary hit, never a measurement
    a = _bursts(sr * 20, sr, rng)
    b = _delayed(a, sr, 3.0, rng=rng)
    est = sync.audio_offset(a, b, sr, max_lag=1.0, method='gcc')
    assert not est.identifiable


def test_audio_drift_and_fit_drift():
    sr = 8000
    rng = np.random.default_rng(2)
    a = _bursts(sr * 40, sr, rng, n_events=80)
    b = _delayed(a, sr, 0.5, rate=4e-4, rng=rng)
    est = sync.audio_offset(a, b, sr, max_lag=2, method='gcc', windows=8)
    wt, wo = est.diagnostics['window_times'], est.diagnostics['window_offsets']
    assert len(wo) >= 6
    fit = sync.fit_drift(wt, wo)
    assert abs(fit.rate - 4e-4) < 5e-5
    # offset at a's time t is 0.5 + rate * t
    assert abs(fit.predict(10.0) - (0.5 + 4e-4 * 10)) < 2e-3
    tm = fit.timemap()
    # source_b = global + offset(global): at global 10 (a's clock), b shows it at ~ 10.504
    assert abs(tm.to_source(10.0) - (10.0 + 0.5 + 4e-3)) < 2e-3


def test_fit_drift_robust():
    rng = np.random.default_rng(0)
    t = np.linspace(0, 3600, 60)
    y = 0.25 + 2e-5 * t + rng.normal(scale=1e-3, size=t.size)
    y[[5, 17, 40]] += [0.5, -0.3, 1.0]
    fit = sync.fit_drift(t, y, t_ref=0.0)
    assert abs(fit.rate - 2e-5) < 2e-6 and abs(fit.offset0 - 0.25) < 3e-3
    assert not fit.inliers[[5, 17, 40]].any()
    json.dumps(fit.to_dict())


# ------------------------------------------------------------------ offset graph

def test_solve_offset_graph():
    true = {'c0': 0.0, 'c1': 1.5, 'c2': -0.75, 'c3': 3.0}
    names = list(true)
    rng = np.random.default_rng(0)
    edges = []
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            edges.append(sync.OffsetEstimate(offset=true[b] - true[a] + rng.normal(scale=1e-3), sigma=1e-3,
                                             pair=(a, b)))
    edges.append(sync.OffsetEstimate(offset=9.0, sigma=1e-3, pair=('c1', 'c3')))  # contradictory
    edges.append(sync.OffsetEstimate(offset=0.1, identifiable=False, reasons=['flat_curve'], pair=('c0', 'c4')))
    g = sync.solve_offset_graph(edges, names=names + ['c4'], max_residual=0.05)
    for n in names:
        assert abs(g.offsets[n] - true[n]) < 3e-3
        assert g.sigma[n] < 2e-3
    assert g.offsets['c4'] is None and g.unresolved == ['c4'] and not g.identifiable['c4']
    reasons = sorted(r['reason'] for r in g.rejected)
    assert reasons == ['graph_cycle_inconsistent', 'not_identifiable']
    assert g.n_cycles == 3 and g.cycle_rms < 3e-3
    json.dumps(g.to_dict())
    g2 = sync.solve_offset_graph([('a', 'b', 1.0), ('b', 'c', 2.0)], shift='min', anchor='b')
    assert g2.offsets == {'a': 0.0, 'b': 1.0, 'c': 3.0}


# ------------------------------------------------------------------ motion

def test_motion_offset_and_signal():
    rate = 30.0
    rng = np.random.default_rng(0)
    T = 900
    f = _smooth_signal(T, rate, rng, f_max=2.0)
    t = np.arange(T) / rate
    true = 0.4  # b shows a's instant t at t + 0.4
    centre_a = np.stack([f(t), f(t + 7.0)], -1) * 50 + 300
    centre_b = np.stack([f(t - true), f(t - true + 7.0)], -1) * 30 + 200
    pa = centre_a[:, None] + rng.normal(scale=10, size=(1, 5, 2))
    pb = centre_b[:, None] + rng.normal(scale=6, size=(1, 5, 2))
    pa[100:130] = np.nan
    sa, sb = sync.motion_signal(pa), sync.motion_signal(pb)
    est = sync.motion_offset(sa, sb, rate, max_lag=2.0)
    assert est.identifiable, est.reasons
    assert abs(est.offset - true) < 1.0 / rate
    assert 'second_peak_score' in est.diagnostics
    # a periodic signal is ambiguous
    per = np.sin(2 * np.pi * 1.0 * t)
    e2 = sync.motion_offset(per, np.roll(per, 3), rate, max_lag=2.0)
    assert not e2.identifiable and 'ambiguous_second_peak' in e2.reasons


# ------------------------------------------------------------------ epipolar / frame sweep

def _rig_scene(name='allen_7cam', T=400, J=6, seed=0):
    cg = synth.load_rig(name)
    c, s = synth.scene_center(cg), synth.rig_scale(cg)
    rng = np.random.default_rng(seed)
    fx = [_smooth_signal(T, 1.0, rng, f_max=0.03, n_sines=4) for _ in range(3)]
    offs = rng.normal(scale=0.03 * s, size=(J, 3))

    def X(t):  # continuous trajectory, t in frames -> [len(t), J, 3]
        t = np.atleast_1d(np.asarray(t, float))
        base = np.stack([f(t) for f in fx], -1) / 4 * 0.12 * s + c
        return base[:, None] + offs[None]
    return cg, X


def test_epipolar_offset_subframe():
    cg, X = _rig_scene()
    T = 400
    true = 2.3  # b's frame f + 2.3 shows the instant of a's frame f
    f = np.arange(T, dtype=float)
    pa = synth.project(cg, X(f))[0]
    pb = synth.project(cg, X(f - true))[1]
    pa = synth.add_noise(pa, 0.3, seed=1)
    pb = synth.add_noise(pb, 0.3, seed=2)
    assert np.isfinite(pa).mean() > 0.9 and np.isfinite(pb).mean() > 0.9
    est = sync.epipolar_offset(pa, pb, cg.subset_cameras([0, 1]), search=(-6, 6), n_boot=20, rng=0)
    assert est.identifiable, est.reasons
    assert abs(est.offset - true) < 0.15
    assert 0 < est.sigma < 0.3
    lo, hi = est.diagnostics['boot_ci95']
    assert lo - 0.1 <= true <= hi + 0.1
    json.dumps(est.to_dict())
    # with a fundamental matrix instead of the cgroup (no distortion in this rig's pinhole model)
    est2 = sync.epipolar_offset(pa, pb, None, search=(-6, 6), n_boot=0)
    assert abs(est2.offset - true) < 0.3


def test_epipolar_offset_static_unidentifiable():
    cg, X = _rig_scene()
    p = X(np.zeros(1))[0]
    pa = np.repeat(synth.project(cg, p)[0][None], 200, 0)
    pb = np.repeat(synth.project(cg, p)[1][None], 200, 0)
    pa = synth.add_noise(pa, 0.3, seed=1)
    pb = synth.add_noise(pb, 0.3, seed=2)
    est = sync.epipolar_offset(pa, pb, cg.subset_cameras([0, 1]), search=(-4, 4), n_boot=10)
    assert not est.identifiable


def test_frame_offset_sweep():
    cg, X = _rig_scene(T=300)
    cg = cg.subset_cameras([0, 1, 2, 3])
    true = {0: 0, 1: 0, 2: 2, 3: -1}
    frames = np.arange(20, 280, 5)
    P = synth.project(cg, X(np.arange(300)))  # [C, T, J, 2], integer frames only
    P_static = synth.project(cg, X(np.full(len(frames), 50.0)))

    def sample_fn(c, d):
        return P[c, frames + d - true[c]]

    res = sync.frame_offset_sweep(cg, sample_fn, cams=[2, 3], ref_cams=[0, 1])
    names = cg.get_names()
    assert res[names[2]].offset == 2 and res[names[2]].identifiable
    assert res[names[3]].offset == -1 and res[names[3]].identifiable

    # static scene: flat curve -> not identifiable
    def static_fn(c, d):
        return P_static[c]
    res = sync.frame_offset_sweep(cg, static_fn, cams=[2], ref_cams=[0, 1])
    assert not res[names[2]].identifiable and 'flat_curve' in res[names[2]].reasons


# ------------------------------------------------------------------ soft-DTW

def _hard_dtw(D):
    n, m = D.shape
    R = np.full((n + 1, m + 1), np.inf)
    R[0, 0] = 0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            R[i, j] = D[i - 1, j - 1] + min(R[i - 1, j - 1], R[i - 1, j], R[i, j - 1])
    return R[n, m]


def test_soft_dtw_value_and_gradient():
    rng = np.random.default_rng(0)
    D = rng.uniform(size=(12, 15))
    v = sync.soft_dtw(D, gamma=1e-3).value
    assert abs(v - _hard_dtw(D)) < 0.02
    Dt = torch.tensor(D, requires_grad=True)
    val, E = sync.soft_dtw(Dt, gamma=0.5, band=4)
    g, = torch.autograd.grad(val, Dt, retain_graph=True)
    assert torch.allclose(g, E, atol=1e-10)
    assert (E >= -1e-12).all() and torch.isclose(E[0, 0], torch.tensor(1.0, dtype=E.dtype))
    # differentiable all the way: E itself has a graph when D requires grad
    assert E.requires_grad
    # finite differences agree with the alignment (= gradient)
    E_np = sync.soft_dtw(D, gamma=0.5, band=4).alignment
    for (i, j) in [(0, 0), (5, 6), (11, 14), (3, 5)]:
        Dp, Dm = D.copy(), D.copy()
        Dp[i, j] += 1e-6
        Dm[i, j] -= 1e-6
        fd = (sync.soft_dtw(Dp, 0.5, 4, alignment=False).value
              - sync.soft_dtw(Dm, 0.5, 4, alignment=False).value) / 2e-6
        assert abs(fd - E_np[i, j]) < 1e-6
    # second order: gradients of a loss of E flow back to D
    loss = (E * torch.arange(15.0, dtype=E.dtype)).sum()
    g2, = torch.autograd.grad(loss, Dt)
    assert torch.isfinite(g2).all() and g2.abs().sum() > 0


def test_soft_dtw_align_piecewise_warp():
    rate = 30.0
    rng = np.random.default_rng(0)
    T = 900
    f = _smooth_signal(T, rate, rng, f_max=1.5)
    tb = np.arange(T) / rate

    def true_lag(ta):  # b source = a source + lag(a); a 0.25 s chapter step at 15 s, plus 0.5% drift
        return 0.3 + 0.005 * ta + np.where(ta > 15.0, 0.25, 0.0)

    # invert tb = ta + lag(ta): before the step, after it, or in the gap the step leaves in b
    ta1, ta2 = (tb - 0.3) / 1.005, (tb - 0.55) / 1.005
    ta = np.where(ta1 <= 15.0, ta1, np.where(ta2 > 15.0, ta2, np.nan))
    a = f(np.arange(T) / rate) + 0.05 * rng.normal(size=T)
    b = f(np.nan_to_num(ta)) + 0.05 * rng.normal(size=T)
    b[np.isnan(ta)] = np.nan
    w = sync.soft_dtw_align(a, b, rate=rate, gamma=0.05, band=0.6, max_lag=2.0)
    assert w.identifiable, w.reasons
    t, lag = w.t_a, w.lag
    away = (np.abs(t - 15.0) > 1.0) & (t > 1.0) & (t < t.max() - 1.0)
    err = np.abs(lag - true_lag(t))[away]
    assert np.median(err) < 0.5 / rate and np.percentile(err, 95) < 2.0 / rate
    # the affine summary absorbs part of the step (and says so); the per-row path above is what is exact
    assert 0 < w.drift.rate < 0.03
    json.dumps(w.to_dict())
    # unrelated signals: diffuse / not identifiable
    w2 = sync.soft_dtw_align(a, rng.normal(size=T), rate=rate, gamma=0.05, band=0.6, lag0=0.3)
    assert not w2.identifiable


def test_soft_dtw_align_dropped_frames_two_cameras():
    """Two cameras of a rig; camera b drops 6 frames mid-recording, so the lag steps from +3 to -3 frames.
    Aligned through the view-invariant motion signal of the 2D tracks."""
    cg, X = _rig_scene(T=900, seed=3)
    rate = 30.0
    f = np.arange(900, dtype=float)
    pa = synth.project(cg, X(4.0 * f))[0]
    inst = f - 3.0                   # b frame f shows a's frame f - 3 ...
    inst[450:] += 6.0                # ... and after 6 dropped frames, a's frame f + 3
    pb = synth.project(cg, X(4.0 * inst))[1]
    pa, pb = synth.add_noise(pa, 0.5, seed=1), synth.add_noise(pb, 0.5, seed=2)
    w = sync.soft_dtw_align(sync.motion_signal(pa), sync.motion_signal(pb), rate=rate, gamma=0.1, band=0.5,
                            max_lag=1.0)
    assert w.identifiable, w.reasons
    fa = w.t_a * rate
    true = np.where(fa < 447, 3.0, -3.0)
    away = np.abs(fa - 447) > 30
    err = np.abs(w.lag * rate - true)[away]
    assert np.median(err) < 0.75 and np.percentile(err, 80) < 1.5
    # a constant-offset estimator cannot represent this: the two halves disagree by 6 frames
    assert np.ptp(np.percentile(w.lag[away] * rate, [10, 90])) > 4


def test_torch_inputs_and_load_audio(tmp_path):
    sr = 8000
    rng = np.random.default_rng(0)
    a = _bursts(sr * 12, sr, rng)
    b = _delayed(a, sr, 0.25, rng=rng)
    est = sync.audio_offset(torch.tensor(a), torch.tensor(b), sr, max_lag=1, method='gcc')
    assert abs(est.offset - 0.25) < 2e-4
    from scipy.io import wavfile
    p = tmp_path / 'b.wav'
    wavfile.write(str(p), sr, (b / np.abs(b).max()).astype('float32'))
    try:
        x = sync.load_audio(p, sr=sr)
    except RuntimeError as e:  # no soundfile / av / ffmpeg in this environment
        pytest.skip(str(e))
    assert x.shape == b.shape
    x2 = sync.load_audio(p, sr=4000, start=1.0, duration=2.0)
    assert len(x2) == 8000
