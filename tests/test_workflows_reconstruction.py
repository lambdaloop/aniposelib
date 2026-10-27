"""Workflows W5-W7 (plan 05 §5): estimate_offsets, reconstruct_3d, associate_views on the
synthetic use-case rigs. Kept small (few frames, short audio) so the suite stays fast."""

import json
import os

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


# ------------------------------------------------------------------ W6 options (1.1 real-data fixes)

def _moving(true, p3d, deg_per_frame=0.02):
    """Per-frame extrinsics [C,N,4,4]: every camera pans at its own rate (the rig is not rigid)."""
    from aniposelib import geometry as G
    E0 = synth.to_np(G.pack(true).extrinsics())
    C, N = E0.shape[0], p3d.shape[0]
    E = np.empty((C, N, 4, 4))
    for c in range(C):
        for t in range(N):
            a = np.deg2rad(deg_per_frame * t * (c - (C - 1) / 2))
            M = np.eye(4)
            M[:3, :3] = [[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]]
            E[c, t] = M @ E0[c]
    p2d = synth.to_np(G.project(G.pack(true), p3d, extrinsics=E)[0])
    return E, p2d


def test_reconstruct_max_gap_segments_times(person):
    true, p3d, p2d, bones = person
    q = p2d.copy()
    q[:, 50:80, 2] = np.nan                       # a 30-frame gap of joint 2
    r = W.reconstruct_3d(true, q)
    assert np.isnan(r.points3d[50:80, 2]).all()                   # default max_gap = 10 frames
    assert np.isfinite(r.points3d[40:46, 3]).all()                # short gaps still filled
    r = W.reconstruct_3d(true, q, options={'max_gap': None})
    assert np.isfinite(r.points3d[50:80, 2]).all()
    # times: interpolation on time, max_gap_s in seconds
    t = np.arange(q.shape[1]) / 30.0
    r = W.reconstruct_3d(true, q, times=t, options={'max_gap': None, 'max_gap_s': 0.5})
    assert np.isnan(r.points3d[50:80, 2]).all() and np.isfinite(r.points3d[40:46, 3]).all()
    with pytest.raises(ValueError):
        W.reconstruct_3d(true, q, options={'max_gap_s': 0.5})
    # segments: a gap at a segment boundary is never bridged, also with bones
    seg = np.r_[np.zeros(43), np.ones(q.shape[1] - 43)]
    for b, sm in ((None, None), (bones, None), (None, 2.0)):
        r = W.reconstruct_3d(true, p2d, segments=seg, bones=b, smooth=sm, options={'n_iters': 40})
        assert np.isnan(r.points3d[40:43, 3]).all() and np.isnan(r.points3d[43:46, 3]).all()
        assert np.isfinite(r.points3d[30:40, 3]).all()
    assert r.settings['n_segments'] == 2


def test_reconstruct_moving_extrinsics(person):
    true, p3d, _, _ = person
    E, p2d = _moving(true, p3d)
    p2d = synth.add_noise(p2d, 0.5, seed=1)
    static = W.reconstruct_3d(true, p2d, fill='none')
    moving = W.reconstruct_3d(true, p2d, fill='none', extrinsics=E)
    assert any('error_drift' in x for x in static.reasons), static.reasons
    assert moving.verdict == 'ok', moving.reasons
    assert np.nanmedian(_rel(true, moving.points3d, p3d)) < 1e-3
    assert np.nanmedian(_rel(true, static.points3d, p3d)) > 5 * np.nanmedian(_rel(true, moving.points3d, p3d))
    # [N,C,4,4], 3x4 and linear triangulation are accepted too
    r = W.reconstruct_3d(true, p2d, fill='none', robust=False, extrinsics=E.transpose(1, 0, 2, 3)[..., :3, :])
    np.testing.assert_allclose(np.nanmedian(_rel(true, r.points3d, p3d)), 0, atol=1e-3)
    with pytest.raises(ValueError):
        W.reconstruct_3d(true, p2d, extrinsics=E[:, :5])


def test_reconstruct_view_point_filters(person):
    true, p3d, p2d, _ = person
    # min_joints_per_view: camera-frames with few keypoints are dropped
    q = p2d.copy()
    q[0, 20:30, 2:] = np.nan                       # camera 0 sees only 2 joints there
    r = W.reconstruct_3d(true, q, options={'min_joints_per_view': 0.5})
    assert r.report['n_removed']['min_joints_per_view'] >= 10
    assert not r.inliers[0, 20:30].any()
    # per-point error and a gate on it
    r = W.reconstruct_3d(true, p2d)
    assert r.point_error.shape == p3d.shape[:2]
    assert np.all(np.isnan(r.point_error[r.filled]))
    thr = float(np.nanpercentile(r.point_error, 80))
    g = W.reconstruct_3d(true, p2d, fill='none', options={'max_point_err_px': thr})
    assert np.nanmax(g.point_error) <= thr + 1e-9
    assert g.report['n_removed']['max_point_err_px'] > 0
    # out-of-image observations are counted separately, and used with mask_out_of_bounds=False
    w, h = synth.to_np(true.cameras[1].get_size())
    q = p2d.copy()
    ok = np.isfinite(q[1, :, 0, 0])
    q[1, ok, 0, 0] = w + 2.0
    r = W.reconstruct_3d(true, q)
    assert r.report['per_camera'][true.get_names()[1]]['n_out_of_image'] == ok.sum()
    assert r.out_of_image[1, ok, 0].all() and not r.inliers[1, ok, 0].any()
    r2 = W.reconstruct_3d(true, q, options={'mask_out_of_bounds': False})
    assert r2.report['n_out_of_image'] == ok.sum() and r2.settings['mask_out_of_bounds'] is False


def test_reconstruct_cam_names(person):
    true, p3d, p2d, _ = person
    names = true.get_names()
    order = [3, 0, 4, 1, 2]
    ref = W.reconstruct_3d(true, p2d, fill='none')
    r = W.reconstruct_3d(true, p2d[order], fill='none',
                         options={'cam_names': [names[i] for i in order]})
    np.testing.assert_allclose(r.points3d, ref.points3d, atol=1e-9)
    r = W.reconstruct_3d(true, p2d[[0, 1, 2, 3]], fill='none',
                         options={'cam_names': names[:4] + []})
    assert r.report['cameras_missing'] == [names[4]]
    r = W.reconstruct_3d(true, np.concatenate([p2d, p2d[:1]]), fill='none',
                         options={'cam_names': names + ['extra']})
    assert r.report['cameras_dropped'] == ['extra']


def test_reconstruct_affine_group():
    from aniposelib.affine import AffineCamera
    from aniposelib.cameras import CameraGroup
    rng = np.random.default_rng(0)
    cams = []
    for c in range(4):
        a = 2 * np.pi * c / 4
        d = np.array([np.cos(a), np.sin(a), 0.4])
        d /= np.linalg.norm(d)
        u = np.cross(d, [0, 0, 1.0]); u /= np.linalg.norm(u)
        v = np.cross(d, u)
        A = 900 * np.stack([u, v]) + rng.normal(scale=5, size=(2, 3))
        cams.append(AffineCamera(A, np.array([640.0, 512.0]), size=(1280, 1024), name='c{}'.format(c)))
    cg = CameraGroup(cams)
    X = rng.normal(scale=0.2, size=(60, 3))
    p2d = np.stack([X @ c.A.detach().numpy().T + c.b.detach().numpy() for c in cams])
    p2d += rng.normal(scale=0.3, size=p2d.shape)
    p2d[1, 5] += 80.0                               # an outlier
    r = W.reconstruct_3d(cg, p2d)
    assert r.verdict == 'ok', r.reasons
    assert 'affine' in r.settings['cameras']
    assert np.median(np.linalg.norm(r.points3d - X, axis=-1)) < 2e-3
    assert not r.inliers[1, 5]
    mixed = CameraGroup([cams[0], synth.load_rig('fisheye_5cam').cameras[0]])
    with pytest.raises(ValueError, match='to_pinhole'):
        W.reconstruct_3d(mixed, p2d[:2])


def test_reconstruct_bones_auto(person):
    """'auto' bone weights follow each bone's spread: rigid bones are enforced, a stretchy one is
    not; on noise-free labels the refinement may not raise the reprojection error."""
    true, p3d, p2d, bones = person
    q = p3d.copy()
    a, b = bones[0]
    stretch = 1 + 0.25 * np.sin(np.arange(len(q)) / 5.0)            # a non-rigid "bone"
    q[:, b] = q[:, a] + (q[:, b] - q[:, a]) * stretch[:, None]
    obs = synth.add_noise(synth.project(true, q), 0.7, seed=4)
    r = W.reconstruct_3d(true, obs, bones=bones, options={'n_iters': 100})
    w = r.settings['optim_points']['bone_weights']
    assert w[0] < 0.01 * np.median(w[1:])
    assert r.verdict == 'ok', r.reasons                              # bone_cv relative to triangulated
    L = np.linalg.norm(r.points3d[:, a] - r.points3d[:, b], axis=-1)
    Lt = np.linalg.norm(q[:, a] - q[:, b], axis=-1)
    assert np.nanmedian(np.abs(L - Lt) / Lt) < 0.02                  # the stretch is kept
    # exact labels: the bone term may not pull the points off their exact projections
    exact = synth.project(true, p3d)
    r = W.reconstruct_3d(true, exact, bones=bones, options={'n_iters': 100, 'bone_weight': 2.0})
    assert np.nanmedian(r.errors[r.inliers]) < 0.3


# ------------------------------------------------------------------ W5 options (1.1 real-data fixes)

def test_offsets_audio_t0_from_container(tmp_path, monkeypatch):
    """Paths: the audio stream's start time (relative to the video) is the source time of sample 0."""
    sf = pytest.importorskip('soundfile')
    from aniposelib import sync as S
    from test_sync import _delayed
    sr = 8000
    rng = np.random.default_rng(5)
    a = _bursts(sr * 12, sr, rng)
    b = _delayed(a, sr, 0.75, rng=rng)
    sf.write(str(tmp_path / 'a.wav'), (0.3 * a).astype('float32'), sr)
    sf.write(str(tmp_path / 'b.wav'), (0.3 * b).astype('float32'), sr)
    starts = {'a.wav': 0.010, 'b.wav': 0.066}
    monkeypatch.setattr(S, 'media_info', lambda p: {'audio_start': starts[os.path.basename(p)],
                                                    'video_start': 0.0, 'avg_frame_rate': 20.0})
    res = W.estimate_offsets({'a': str(tmp_path / 'a.wav'), 'b': str(tmp_path / 'b.wav')},
                             reference='a', options={'audio_sr': sr})
    assert abs(res.offsets['b'] - (0.75 + 0.056)) < 1e-3, res.offsets
    assert res.settings['frame_rate'] == 20.0 and 'avg_frame_rate' in res.settings['frame_rate_from']
    assert 'container' in res.settings['streams']['b']['audio_t0_from']
    p = [x for x in res.pairs if x['used']][0]
    assert abs(p['offset'] - p['offset_raw'] - 0.056) < 1e-9
    # an explicit audio_t0 wins
    res = W.estimate_offsets({'a': {'audio': str(tmp_path / 'a.wav'), 'audio_t0': 0.0},
                              'b': {'audio': str(tmp_path / 'b.wav'), 'audio_t0': 0.0}},
                             reference='a', options={'audio_sr': sr, 'fps': 30.0})
    assert abs(res.offsets['b'] - 0.75) < 1e-3 and res.settings['frame_rate'] == 30.0


def test_offsets_windowed_audio_drift():
    """Long clips: coarse lag + gcc in windows + a line; the clock rate is fitted."""
    from test_sync import _delayed
    sr = 2000
    rng = np.random.default_rng(2)
    a = _bursts(sr * 150, sr, rng, n_events=400)
    b = _delayed(a, sr, 1.3, rate=2e-4, rng=rng)
    res = W.estimate_offsets({'a': {'audio': (a, sr)}, 'b': {'audio': (b, sr)}}, reference='a',
                             options={'audio_strategy': 'windowed', 'audio_window_s': 10.0})
    assert res.verdict != 'fail', res.reasons
    assert [p['method'] for p in res.pairs] == ['gcc_windowed']
    assert abs(res.rates['b'] - 2e-4) < 2e-5, res.rates
    g = 75.0
    assert abs(res.timemaps['b'].to_source(g) - (1.3 + (1 + 2e-4) * g)) < 2e-3
    assert res.sigmas['b'] > 1e-5


def test_offsets_drift_unidentifiable_and_irregular_times(sync_scene):
    cg, streams, off, rate = sync_scene
    names = list(streams)
    # no windowed audio estimates: drift=True cannot fit a rate and says so
    res = W.estimate_offsets({n: {'audio': streams[n]['audio']} for n in names}, reference=names[0],
                             options={'audio_windows': 0})
    assert any('drift_unidentifiable' in r for r in res.reasons), res.reasons
    # container-style per-frame times (2, 1, 2, 1 ticks of 1/30 s = 20 fps on average)
    p, t = streams[names[0]]['points']
    ticks = np.cumsum(np.r_[0, np.tile([2, 1], len(t))[:len(t) - 1]]) / 30.0
    st = {n: {'points': (streams[n]['points'][0], ticks)} for n in names[:2]}
    res = W.estimate_offsets(st, drift=False)
    assert abs(res.settings['frame_rate'] - 20.0) < 0.1
    assert any('irregular_times' in r for r in res.reasons)


def test_offsets_self_F_seed_and_refine_rate(sync_scene):
    cg, streams, off, rate = sync_scene
    names = list(streams)
    pts = {n: {'points': streams[n]['points']} for n in names}
    # epipolar estimates without a calibration: F fitted per offset
    res = W.estimate_offsets(pts, reference=names[0], drift=False,
                             options={'epipolar_F': 'self', 'epipolar_boot': 0})
    assert any(p['kind'] == 'epipolar' and p.get('note') == 'F self' for p in res.pairs)
    # a seed skips the estimation; refine on a short window keeps the seed rate ('auto')
    full = W.estimate_offsets(streams, cgroup=cg, reference=names[0])
    r2 = W.estimate_offsets(streams, cgroup=cg, reference=names[0], seed=full, refine=True)
    assert r2.pairs == [] and r2.settings['seeded'] and r2.refined['ok']
    assert r2.refined['free_rate'] is False
    assert np.isclose(r2.rates[names[2]], full.rates[names[2]], rtol=1e-9, atol=0)
    for n in names:
        assert abs(r2.offsets[n] - off[n]) < 2e-3
    # stream_camera: a clip named differently from its camera still gets the calibrated F
    st = {'clipA': {'points': streams[names[0]]['points']}, names[1]: {'points': streams[names[1]]['points']}}
    r3 = W.estimate_offsets(st, cgroup=cg, drift=False, options={'stream_camera': {'clipA': names[0]},
                                                                   'epipolar_boot': 0})
    assert any(p.get('note') == 'F calibrated' for p in r3.pairs)


def test_reconstruct_smooth_segments_no_crosstalk(person):
    """Smoothing never links frames of different segments (one optim_points call, masked term)."""
    true, p3d, _, _ = person
    q = p3d.copy()
    q[50:] += 0.3 * synth.rig_scale(true) * np.array([1.0, 0.0, 0.0])     # a cut between two clips
    obs = synth.add_noise(synth.project(true, q), 0.5, seed=2)
    seg = (np.arange(len(q)) >= 50).astype(int)
    kw = dict(smooth=50.0, fill='none', options={'n_iters': 150})
    a = W.reconstruct_3d(true, obs, **kw)
    b = W.reconstruct_3d(true, obs, segments=seg, **kw)
    e = lambda r: np.nanmedian(np.linalg.norm(r.points3d[48:52] - q[48:52], axis=-1)) / synth.rig_scale(true)
    assert e(b) < 0.5 * e(a)
