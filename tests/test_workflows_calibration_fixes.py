"""1.1 real-data fixes of the calibration workflows (W2-W4): options and bug fixes, on small
synthetic scenes (fast)."""

import copy
import json

import numpy as np
import pytest
import torch

import synth
from aniposelib import geometry as G, stations as ST
from aniposelib.workflows import calibrate_from_points, estimate_scale, evaluate_calibration
from aniposelib.workflows import calibration as WC

pytestmark = pytest.mark.integration


@pytest.fixture(scope='module')
def mo():
    true = synth.load_rig('motobs_7cam')
    p3, bones = synth.skeleton(true, 60, model=synth.MOUSE, size=1.0, spread=0.12, seed=1)
    p2 = synth.add_noise(synth.project(true, p3), 0.7, seed=2)
    return true, p3, bones, p2


def _bones(p3, bones, s=1.0):
    return [s * np.median(np.linalg.norm(p3[:, a] - p3[:, b], axis=-1)) for a, b in bones]


def _bumped(cg, cam, rot_deg):
    cgb = copy.deepcopy(cg)
    c = cgb.cameras[cam]
    dR = G.rodrigues(torch.tensor(np.radians(rot_deg), dtype=torch.float64))
    with torch.no_grad():
        R = G.rodrigues(c.rvec.reshape(3).double())
        c.rvec.copy_(torch.as_tensor(ST._rotvec((dR @ R).numpy())).reshape(c.rvec.shape))
        c.tvec.copy_((dR @ c.tvec.reshape(3).double()).reshape(c.tvec.shape))
    return cgb


# ------------------------------------------------------------------ noise unit / schedule

def test_fine_schedule_keeps_ratios():
    fine = ((40, 20, 10, 5), (20, 10, 5))
    assert WC._fine_schedule(fine, 1.0, 1e9) == ((40, 20, 10, 5), (20, 10, 5))
    # u = 2, cap 30 px: thresholds above the cap are dropped, not flattened to 30
    assert WC._fine_schedule(fine, 2.0, 30.0) == ((20, 10), (20, 10))
    # every threshold above the cap: one round at the cap
    assert WC._fine_schedule(((40, 20),), 2.0, 10.0) == ((10.0,),)


def test_noise_unit_option(mo):
    true, p3, bones, p2 = mo
    ref = synth.perturb_extrinsics(true, 1.0, 0.02, seed=4)
    cg, res = calibrate_from_points(p2, ref, init='reference', refine_intrinsics=False,
                                    options={'noise_unit': 1, 'quality': False})
    assert res.settings['noise_unit'] == 1 and res.settings['noise_unit_final'] == 1
    assert res.settings['fine_px'][-1][-1] == 5.0
    # the verdict noise floor is measured on the points (W4's estimator), not at the last threshold
    assert 0.5 <= res.settings['noise_floor_px'] < 2.0 and 'noise_floor_fit_px' in res.settings
    ev = evaluate_calibration(cg, p2, options={'noise_unit': 1, 'tau_tri_px': 4, 'thresholds_px': (3, 6)})
    assert ev.settings['tau_tri_px'] == 4 and ev.settings['thresholds_px'] == [3, 6]
    assert 'loo_agree@6' in next(iter(ev.per_camera.values()))
    with pytest.raises(ValueError):
        evaluate_calibration(cg, p2, options={'noise_unit': 'abs'})


# ------------------------------------------------------------------ W3

def test_w3_lengths_tau_check_and_reference_frame(mo):
    true, p3, bones, p2 = mo
    L = _bones(p3, bones)
    cg = synth.perturb_extrinsics(true, 0.0, 0.0)
    from aniposelib import scale as S
    S.rescale(cg, 2.0)
    res = estimate_scale(cg, lengths=dict(points=p2, pairs=bones, ref=L))
    assert res.accepted and abs(res.scale - 0.5) < 0.005
    assert res.settings['tau_px'] == pytest.approx(5.0)            # absolute 5 px by default
    forced = estimate_scale(cg, lengths=dict(points=p2, pairs=bones, ref=L), options={'noise_unit': 2.0})
    assert forced.settings['tau_px'] == pytest.approx(10.0)
    chk = res.report['lengths_check']
    assert abs(chk['median_ratio'] - 1) < 0.01 and res.verdict == 'ok', res.summary()
    # a too-loose threshold with gross outliers biases the scale: the re-measured bones disagree
    q = synth.add_outliers(p2, frac=0.25, cgroup=true, seed=5)
    bad = estimate_scale(cg, lengths=dict(points=q, pairs=bones, ref=L),
                         options={'lengths_kwargs': {'tau': 400.0}})
    assert 'lengths_check' in bad.report
    # frame from a reference, scale from the bones (the reference is in other units: x 3)
    ref = true.copy()
    S.rescale(ref, 3.0)
    moved = cg.copy()
    S.apply_similarity(moved, 1.0, G.rodrigues(torch.tensor([0.3, -0.2, 0.1], dtype=torch.float64)).numpy(),
                       np.array([5.0, -2.0, 1.0]))
    r2 = estimate_scale(moved, lengths=dict(points=p2, pairs=bones, ref=L), reference=ref, align='reference')
    assert r2.accepted and abs(r2.scale - 0.5) < 0.005 and r2.settings['frame_from_reference']
    assert abs(r2.details['reference_scale_ratio'] - 3.0) < 0.03   # reference 1.5 vs bones 0.5
    c_out, c_true = synth.camera_centers(r2.cgroup), synth.camera_centers(true)
    off = c_out - c_out.mean(0) - (c_true - c_true.mean(0))
    assert np.abs(off).max() < 1e-2 * synth.rig_scale(true)       # same orientation, bone scale
    assert np.linalg.norm(c_out.mean(0) - synth.camera_centers(ref).mean(0)) < 1e-6 + 2 * synth.rig_scale(true)
    with pytest.raises(ValueError):
        estimate_scale(moved, lengths=dict(points=p2, pairs=bones, ref=L), reference=ref)   # align not given


def test_w3_board_points_from_videos(monkeypatch, tmp_path):
    import aniposelib.boards as B
    from aniposelib.boards import CharucoBoard
    true = synth.load_rig('allen_7cam')
    board = CharucoBoard(6, 5, 24, 18, marker_bits=4, dict_size=50, units='mm')
    rows = synth.board_rows(true, board, n_frames=25, seed=1)
    seen = {}

    def fake_detect(board_, videos, **kw):
        seen.update(kw, videos=videos, kwargs=board_.detector_kwargs, name=board_.detector_name)
        kw['info'].update(stopped='end')
        return rows
    monkeypatch.setattr(B, 'detect_videos', fake_detect)
    monkeypatch.setattr(B, '_video_length', lambda path: 500)
    vids = {n: str(tmp_path / '{}.mp4'.format(n)) for n in true.get_names()}
    for v in vids.values():
        open(v, 'w').close()
    res = estimate_scale(true, board=board, board_points=vids, options={'n_frames': 11, 'detector': 'opencv'})
    assert res.accepted and abs(res.scale - 1) < 1e-3
    assert seen['frame_indices'] == list(np.linspace(0, 499, 11).round().astype(int))
    assert seen['videos'][0] == vids[true.get_names()[0]] and seen['max_frames'] == 200
    assert seen['min_cameras'] == 2 and seen['prefix'] == 0
    det = res.report['detection']
    assert det['sampling'] == 'sequential' and det['frame_indices'] == seen['frame_indices']
    estimate_scale(true, board=board, board_points=vids, options={'detector': 'deepcharuco'})
    assert seen['name'] == 'deepcharuco' and seen['kwargs'] == {'scale': 'auto'}     # frame-size input scale
    estimate_scale(true, board=board, board_points=vids,
                   options={'detector': 'deepcharuco', 'detector_kwargs': {'scale': 0.3}})
    assert seen['kwargs'] == {'scale': 0.3}
    with pytest.raises(ValueError):
        estimate_scale(true, board=board, board_points={'nope': 'x.mp4'})
    with pytest.raises(FileNotFoundError):
        estimate_scale(true, board=board, board_points={true.get_names()[0]: str(tmp_path / 'missing.mp4')})


def test_w3_board_short_pairs_dropped_by_default():
    """``min_pair_dist='auto'`` (default): only corner pairs more than two squares apart (the motor
    observatory's 50 mm gate on its 24 mm board). Short pairs inflate the triangulated distance under
    corner noise (scale biased low: motobs W2 rig + project detections 0.27 % -> 0.16 % vs the reference)."""
    from aniposelib import scale as S
    from aniposelib.boards import CharucoBoard
    board = CharucoBoard(6, 5, 24, 18, marker_bits=4, dict_size=50, units='mm')
    objp = board.get_object_points().reshape(-1, 3)
    td = np.linalg.norm(objp[:, None] - objp[None], axis=-1)[np.triu_indices(len(objp), 1)]
    assert abs(S._auto_min_pair_dist(board, td) - 48) < 1e-3
    small = CharucoBoard(3, 3, 24, 18, marker_bits=4, dict_size=50)     # 2 x 2 corners: keep the side pairs
    o3 = small.get_object_points().reshape(-1, 3)
    t3 = np.linalg.norm(o3[:, None] - o3[None], axis=-1)[np.triu_indices(len(o3), 1)]
    assert (t3 > S._auto_min_pair_dist(small, t3)).all()
    two = synth.load_rig('allen_7cam').subset_cameras([0, 3])
    new, old = [], []
    for seed in (1, 2, 3):
        rows = synth.board_rows(two, board, n_frames=30, noise=4.0, seed=seed)
        r = estimate_scale(two, board=board, board_points=rows)
        r0 = estimate_scale(two, board=board, board_points=rows, options={'board_kwargs': {'min_pair_dist': None}})
        assert r.accepted and r0.accepted
        assert r.report['stats']['n_pairs'] < r0.report['stats']['n_pairs']
        new.append(r.scale - 1)
        old.append(r0.scale - 1)
    assert np.mean(old) < -1e-3                      # all pairs: biased low
    assert abs(np.mean(new)) < 0.5 * abs(np.mean(old))
    with pytest.raises(ValueError):
        S.from_board(two, rows, board, min_pair_dist='half')


# ------------------------------------------------------------------ W2

def test_w2_intrinsics_gate_share_and_never_raises(mo):
    true, _, _, p2 = mo
    names = true.get_names()
    assert WC._share_spec('all', names) == [names]
    assert WC._share_spec([[0, 1], [names[2], 'unknown']], names) == [[names[0], names[1]]]
    with pytest.raises(ValueError):
        WC._share_spec('some', names)
    ref = synth.perturb_extrinsics(true, 0.5, 0.01, seed=4)
    cg, res = calibrate_from_points(p2, ref, init='reference', refine_intrinsics=True,
                                    options={'intrinsics_share': 'all', 'quality': False})
    f = [synth.to_np(c.get_camera_matrix())[0, 0] for c in cg.cameras]
    assert np.ptp(f) < 1e-6 * np.mean(f) and res.intrinsics['share'] == [names]
    # an unusable gate (no data) keeps the intrinsics instead of raising
    opts = dict(WC.W2_DEFAULTS)
    cg2, info = WC._gate_intrinsics(ref.copy(), np.full_like(p2, np.nan), {}, 'auto', 1.0, opts, 0)
    assert info['decision'] == 'kept'


def test_w2_invalid_lens_is_reverted(mo, monkeypatch):
    true, _, _, p2 = mo
    ref = synth.perturb_extrinsics(true, 0.5, 0.01, seed=4)
    bad_name = true.get_names()[1]
    monkeypatch.setattr(WC, '_invalid_lenses', lambda g, p=None: [bad_name])
    k0 = synth.to_np(ref.cameras[1].get_camera_matrix()).copy()
    cg, res = calibrate_from_points(p2, ref, init='reference', refine_intrinsics=True,
                                    options={'quality': False})
    assert res.intrinsics['reverted_invalid'] == [bad_name]
    assert np.allclose(synth.to_np(cg.cameras[1].get_camera_matrix()), k0)


def test_w2_fail_results_keep_the_report_schema(mo):
    true, _, _, p2 = mo
    res = calibrate_from_points(p2, WC_no_extrinsics(true), init='vggt')       # no frames: no init
    assert res.verdict == 'fail' and res.cgroup is None
    assert set(res.report['per_camera']) == set(true.get_names())
    json.dumps(res.to_dict())
    # people init on garbage detections: a fail result, not an exception
    p5 = np.random.default_rng(0).uniform(0, 1000, (7, 6, 2, 4, 2))
    res = calibrate_from_points(p5, WC_no_extrinsics(true), init='people',
                                options={'track_ids': np.zeros((7, 6, 2), int)})
    assert res.verdict == 'fail' and 'per_camera' in res.report


def WC_no_extrinsics(true):
    cg = true.copy()
    for c in cg.cameras:
        c.set_rotation(np.zeros(3))
        c.set_translation(np.zeros(3))
    return cg


def test_w2_auto_skips_sfm_for_a_good_reference(mo):
    true, _, _, p2 = mo
    ref = synth.perturb_extrinsics(true, 0.5, 0.01, seed=4)
    cg, res = calibrate_from_points(p2, ref, refine_intrinsics=False, options={'quality': False})
    assert 'sfm' not in res.report['candidates'] and res.selected_init == 'reference'
    assert any('SfM skipped' in f for f in res.provenance['fallbacks'])
    assert res.report['init_support']['overall'] > 0.9
    assert synth.center_error(cg, true).max() < 0.005


def test_w2_5d_points_with_a_rig_init(mo):
    """Unassociated people (C x F x P x J x 2) + a rig: associate_views, then the 4D route."""
    true, _, _, _ = mo
    X = synth.crowd(true, 2, 40, model=synth.MOUSE, size=1.0, sep=1.0, spread=0.1, seed=3)  # [P,F,J,3]
    p = np.stack([synth.project(true, X[i]) for i in range(2)], 2)           # C x F x P x J x 2
    p = synth.add_noise(p, 0.5, seed=1)
    perm = np.random.default_rng(0).permuted(np.tile([0, 1], (p.shape[0], p.shape[1], 1)), axis=2)
    p = np.take_along_axis(p, perm[..., None, None], axis=2)                 # slots shuffled per camera
    rig = synth.perturb_extrinsics(true, 0.3, 0.005, seed=2)
    cg, res = calibrate_from_points(p, WC_no_extrinsics(true), init=rig, refine_intrinsics=False,
                                    options={'quality': False})
    assert res.init_report.init_reports['associate']['n_identities'] >= 2
    assert synth.center_error(cg, true).max() < 0.01


def test_w2_runaway_is_relative_to_the_init():
    """A camera that was already far in the init (telephoto) is not a runaway."""
    init = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0], [40, 40, 0]], float)
    from aniposelib.cameras import Camera, CameraGroup
    cams = []
    for i, c in enumerate(init):
        cam = Camera(name=str(i))
        cam.set_translation(-c)
        cams.append(cam)
    cg = CameraGroup(cams)
    run, frac = WC._runaways(cg, init, cg.get_names(), WC.W2_DEFAULTS)
    assert run == [] and frac['4'] < 1e-6
    moved = init.copy()
    moved[4] = [400, 400, 0]
    run, _ = WC._runaways(cg, moved * 0 + init, cg.get_names(), WC.W2_DEFAULTS)
    cams[4].set_translation(-moved[4])
    run, _ = WC._runaways(CameraGroup(cams), init, cg.get_names(), WC.W2_DEFAULTS)
    assert run == ['4']


# ------------------------------------------------------------------ 2-camera rigs

def test_two_camera_rig(mo):
    true, p3, bones, p2 = mo
    two = true.subset_cameras([0, 1])
    q = p2[:2]
    ref = synth.perturb_extrinsics(two, 0.5, 0.0, seed=1)
    cg, res = calibrate_from_points(q, ref, init='reference')
    assert res.verdict != 'fail', res.summary()
    assert any('two_view_only' in r for r in res.reasons) and 'per_camera' in res.report
    sc = estimate_scale(two, lengths=dict(points=q, pairs=bones, ref=_bones(p3, bones)))
    assert sc.accepted and sc.settings['min_inliers'] == 2
    ev = evaluate_calibration(two, q)
    assert ev.verdict == 'warn', ev.summary()
    assert ev.settings['min_others'] == 1
    assert all(r['value'] is None or np.isfinite(r['value']) for r in ev.report['rules']
               if isinstance(r['value'], float))
    json.loads(ev.to_json())


# ------------------------------------------------------------------ W4

def test_w4_cam_names_and_affine(mo):
    true, _, _, p2 = mo
    names = true.get_names()
    order = [names[i] for i in (3, 0, 1, 2, 5, 4, 6)] + ['extra']
    idx = [names.index(n) for n in order[:-1]]
    q = np.concatenate([p2[idx], np.full_like(p2[:1], 5.0)], 0)
    ref = evaluate_calibration(true, p2)
    ev = evaluate_calibration(true, q, options={'cam_names': order})
    assert ev.report['cameras_dropped'] == ['extra'] and ev.report['cameras_missing'] == []
    for n in names:
        assert ev.per_camera[n]['loo_median_px'] == pytest.approx(ref.per_camera[n]['loo_median_px'])
    ev = evaluate_calibration(true, q[1:], options={'cam_names': order[1:]})     # names[3] missing
    assert ev.report['cameras_missing'] == [names[3]] and names[3] not in ev.per_camera
    assert any('cameras_missing' in r for r in ev.reasons)
    from aniposelib.affine import AffineCamera
    from aniposelib.cameras import CameraGroup
    mixed = CameraGroup([AffineCamera(name='a'), true.cameras[0], true.cameras[1]])
    with pytest.raises(ValueError, match='to_pinhole'):
        evaluate_calibration(mixed, np.zeros((3, 4, 2)))


def _affine_rig(n=4, seed=0):
    from aniposelib.affine import AffineCamera
    from aniposelib.cameras import CameraGroup
    rng = np.random.default_rng(seed)
    cams = []
    for c in range(n):
        a = 2 * np.pi * c / n
        d = np.array([np.cos(a), np.sin(a), 0.4])
        d /= np.linalg.norm(d)
        u = np.cross(d, [0, 0, 1.0]); u /= np.linalg.norm(u)
        v = np.cross(d, u)
        A = 900 * np.stack([u, v]) + rng.normal(scale=5, size=(2, 3))
        cams.append(AffineCamera(A, np.array([640.0, 512.0]), size=(1280, 1024), name='c{}'.format(c)))
    X = rng.normal(scale=0.2, size=(80, 3))
    p2d = np.stack([X @ c.A.detach().numpy().T + c.b.detach().numpy() for c in cams])
    p2d += rng.normal(scale=0.3, size=p2d.shape)
    return CameraGroup(cams), X, p2d


def test_w4_affine_group_native():
    # tailcycle johnson-fly: W4 takes an AffineCamera group (exact telephoto pinholes, as W6) and
    # gives the same LOO as the user's own to_pinhole conversion
    from aniposelib.cameras import CameraGroup
    cg, X, p2d = _affine_rig()
    ev = evaluate_calibration(cg, p2d)
    assert 'affine' in ev.settings['cameras'] and ev.verdict == 'ok', ev.summary()
    pin = CameraGroup([c.to_pinhole(X, max_error_px=0.01) for c in cg.cameras])
    ref = evaluate_calibration(pin, p2d)
    for n in cg.get_names():
        assert ev.per_camera[n]['loo_median_px'] == pytest.approx(ref.per_camera[n]['loo_median_px'], abs=0.02)
    bad = copy.deepcopy(cg)
    with torch.no_grad():
        bad.cameras[1].b.add_(torch.tensor([6.0, 0.0], dtype=bad.cameras[1].b.dtype))
    ev2 = evaluate_calibration(cg, p2d, options={'compare_to': {'shifted': bad}, 'n_boot': 100})
    assert 'error' not in ev2.report['compare_to']['shifted']


def test_w4_yardstick_summary_and_ranking(mo):
    # Emory: the default W4 reports the standard yardstick (LOO agree@5/8/10 px, MSAC 5 px) whatever
    # the noise unit, and compare_to ranks calibrations by agree@8px (paired block bootstrap)
    from aniposelib import quality as Q
    true, p3, _, _ = mo
    p2 = synth.add_noise(synth.project(true, p3), 3.0, seed=4)       # noisy: noise unit 2
    rough = synth.perturb_extrinsics(true, 1.5, 0.03, seed=5)
    ev = evaluate_calibration(true, p2, options={'compare_to': {'rough': rough, 'same': true}, 'n_boot': 200})
    assert ev.settings['noise_unit'] > 1 and ev.settings['tau_tri_px'] > 5
    assert ev.settings['yardstick_tau_px'] == 5.0 and ev.settings['yardstick_px'] == [5, 8, 10]
    e, el = Q.loo_errors(true, p2, tau_tri=5.0, min_others=3)
    for c, n in enumerate(true.get_names()):
        want = float(np.sum(el[c] & np.isfinite(e[c]) & (e[c] < 8)) / el[c].sum())
        assert ev.per_camera[n]['loo_agree@8px'] == pytest.approx(want)
    s = ev.report['summary']
    assert s['rank_by'] == 'loo_agree@8px' and 0 < s['loo_agree@8px'] < 1
    pc = ev.report['compare_to']
    assert pc['same']['better'] == 'tie' and pc['same']['agree@8px']['delta'] == 0
    r = pc['rough']
    assert r['rank_by'] == 'agree@8px' and r['better'] == 'this'
    assert r['agree@8px']['a'] == pytest.approx(s['loo_agree@8px'])
    assert r['agree@8px']['this_minus_other'] == -r['agree@8px']['delta'] > 0
    lo, hi = r['agree@8px']['ci95']
    assert r['agree@8px']['this_minus_other_ci95'] == [-hi, -lo]
    rk = ev.report['ranking']
    assert rk['order'][-1] == 'rough' and 'rough' in rk['worse_others'] and not rk['better_others']
    assert not any('compare_to_better' in x for x in ev.reasons)
    # evaluating the worse calibration: the verdict says a compared calibration is better
    evr = evaluate_calibration(rough, p2, options={'compare_to': {'true': true}, 'n_boot': 200})
    assert evr.report['ranking']['order'][0] == 'true' and 'true' in evr.report['ranking']['better_others']
    assert any('compare_to_better' in x for x in evr.reasons) and evr.verdict != 'ok'
    json.loads(evr.to_json())
    # yardstick off: no extra keys
    ev0 = evaluate_calibration(true, p2, options={'yardstick_px': None})
    assert 'loo_agree@8px' not in ev0.per_camera[true.get_names()[0]]


@pytest.mark.slow
def test_w4_time_windows_compare_to_and_moves(mo):
    rig = synth.load_rig('motobs_7cam')
    fn = synth.smooth_tracks(rig, n_points=6, amp=0.1, seed=1)
    t = np.arange(0, 40, 1 / 10)
    X = np.stack([fn(p, t) for p in range(6)], 1)
    p2 = synth.add_noise(synth.project(rig, X), 0.7, seed=2)
    # desync: during 12-16 s two cameras see the scene 0.5 s late
    late = synth.project(rig, np.stack([fn(p, t - 0.5) for p in range(6)], 1))
    bad = (t >= 12) & (t < 16)
    p2[:2, bad] = late[:2, bad]
    other = synth.perturb_extrinsics(rig, 1.0, 0.02, seed=3)
    ev = evaluate_calibration(rig, p2, times=t, options={'compare_to': {'other': other}, 'n_boot': 200})
    assert ev.settings['loo_window_s'] == 2.0 and len(ev.per_window) == 20
    w = ev.report['windows']
    assert set(w['bad']) == {'12', '14'}, ev.per_window
    assert set(ev.per_window['12']['bad_cameras']) == set(rig.get_names()[:2])
    assert any('bad_windows_frac' in r for r in ev.reasons)
    assert ev.moves is not None and ev.settings['detect_moves']['z_thr'] == 10.0
    pc = ev.report['compare_to']['other']
    key = [k for k in pc if k.startswith('agree@')][1]
    assert pc[key]['delta'] < 0 and pc[key]['ci95'][1] < 0          # the perturbed rig agrees less
    json.loads(ev.to_json())
    ev2 = evaluate_calibration(rig, p2, times=t, options={'loo_window_s': None})
    assert ev2.per_window == {}


def test_w4_reference_rule_warns_only(mo):
    true, _, _, p2 = mo
    rough = synth.perturb_extrinsics(true, 3.0, 0.1, seed=7)
    ev = evaluate_calibration(true, p2, reference=rough)
    lv = {r['rule']: r['level'] for r in ev.report['rules']}
    assert lv['center_residual_frac'] == 'warn'


# ------------------------------------------------------------------ W1 (dancing replay)

def _board():
    from aniposelib.boards import CharucoBoard
    return CharucoBoard(6, 5, 24, 18, marker_bits=4, dict_size=50, units='mm')


@pytest.mark.slow
def test_w1_bridge_rules_and_board_floor():
    from aniposelib.workflows import calibrate_from_board
    true = synth.load_rig('allen_7cam')
    rows = synth.board_rows(true, _board(), n_frames=40, seed=1)
    rows = [[r for r in rr if r['framenum'][1] % 2 == (0 if c in (0, 1, 2) else 1)] for c, rr in enumerate(rows)]
    p3 = synth.random_points(true, 200, min_views=3, seed=3)
    bp = synth.add_noise(synth.project(true, p3), 3.0, seed=4)          # people: noisier than the boards
    res = calibrate_from_board(rows, _board(), init=WC_no_extrinsics(true), intrinsics=true, bridge_points=bp)
    lv = {r['rule']: (r['level'], r['value']) for r in res.report['rules']}
    assert res.verdict == 'ok', res.summary()
    assert lv['bridge_inlier_frac'][1] > 0.8 and lv['runaway_cameras'][1] == 0
    # the board threshold comes from the board observations, not from the noisier bridge
    nfb = res.settings['noise_floor_board_px']
    assert nfb < 1.0 and res.settings['board_reproj_warn_px'] == pytest.approx(max(1.0, 3 * nfb), rel=1e-3)
    # a wrong bridge (the second component sees a mirrored scene) is not 'ok'
    c0 = synth.scene_center(true)
    bad = bp.copy()
    bad[3:] = synth.add_noise(synth.project(true, 2 * c0 - p3), 3.0, seed=5)[3:]
    res = calibrate_from_board(rows, _board(), init=WC_no_extrinsics(true), intrinsics=true, bridge_points=bad)
    assert res.verdict == 'fail', res.summary()


def test_w1_focal_guess_is_forwarded(monkeypatch):
    from aniposelib import intrinsics as I
    from aniposelib.workflows import calibrate_from_board
    true = synth.load_rig('allen_7cam').subset_cameras([0, 1, 2])
    rows = synth.board_rows(true, _board(), n_frames=12, seed=1)
    seen = []

    def fake(rows_, board_, size, **kw):
        seen.append(kw.get('focal_inits'))
        raise RuntimeError('stop')
    monkeypatch.setattr(I, 'calibrate_ladder_rows', fake)
    calibrate_from_board(rows, _board(), init=true, options={'focal_guess': {true.get_names()[1]: 1700.0}})
    assert seen == [None, [1700.0], None]


# ------------------------------------------------------------------ W2 stations -> rig_at

def test_w2_stations_poses_in_the_group():
    from aniposelib.stations import Stations
    from aniposelib.workflows.calibration import rig_at
    rig = synth.load_rig('motobs_7cam')
    fn = synth.smooth_tracks(rig, n_points=6, amp=0.1, seed=1)
    t = np.arange(0, 24, 1 / 10)
    X = np.stack([fn(p, t) for p in range(6)], 1)
    bumped = _bumped(rig, 2, [1.2, -1.0, 0.8])
    moved = (t >= 12.0)[None, :, None, None]
    p2 = synth.add_noise(np.where(moved, synth.project(bumped, X), synth.project(rig, X)), 0.5, seed=2)
    name = rig.get_names()[2]
    st = Stations({name: [(-np.inf, 12.0, name), (12.0, np.inf, name + '@s1')]})
    from aniposelib import scale as S
    ref = rig.copy()
    cg, res = calibrate_from_points(p2, ref, init='reference', refine_intrinsics=False, stations=st,
                                    groups=dict(time=t), options={'quality': False}, scale=2.0)
    assert cg.stations is not None and name + '@s1' in cg.metadata['station_poses']
    late = rig_at(cg, 20.0)
    early = res.rig_at(5.0)
    tgt = bumped.copy()
    S.rescale(tgt, 2.0)
    E = lambda g: synth.to_np(g.cameras[2].get_extrinsics_mat())
    ref2 = rig.copy()
    S.rescale(ref2, 2.0)
    assert np.abs(E(late) - E(tgt))[:3, :3].max() < 2e-3
    assert np.abs(E(early) - E(ref2))[:3, :3].max() < 2e-3


# ------------------------------------------------------------------ init.sfm seed robustness

def test_sfm_seed_skips_a_pair_that_triangulates_nothing(mo):
    """A pair can top the essential-matrix inlier count and still give no two-view points (seen on
    the cluster: then every other camera stayed unplaced); the next pair is tried."""
    from aniposelib.init import sfm
    true, _, _, p2 = mo
    base = WC_no_extrinsics(true)
    u, f = sfm.ideal_points(base, p2.reshape(p2.shape[0], -1, 2))
    pairs = sfm.pairwise_poses(u, f, thresh=2.0, min_points=30, rng=0)
    top = max(pairs, key=lambda k: pairs[k]['n_in'])
    bad = dict(pairs[top])
    bad['n_in'] = 10 ** 6
    bad['t'] = -np.asarray(bad['t'])            # wrong twisted-pair solution: points behind the cameras
    bad['R'] = np.asarray(bad['R']).T
    pairs[top] = bad
    cg, rep = sfm.from_correspondences(base, p2, pairs=pairs, rng=0)
    assert len(rep.placed) == 7, rep
    assert len(rep.stats['seed_tried']) >= 2 and rep.stats['seed_tried'][0][2] < 30


def test_intrinsics_prior_per_camera_sigma_and_partial_group():
    from aniposelib.params import IntrinsicsPrior
    true = synth.load_rig('motobs_7cam')
    names = true.get_names()
    pr = IntrinsicsPrior(true.subset_cameras([0, 2]), sigma_logf={names[0]: 0.3, 'default': 0.05}, sigma_k=0.1)
    out = pr.priors_for(true.cameras)
    assert set(out) == {0, 2}                                 # unmatched cameras: no prior (not by position)
    assert out[0]['fx'][1] == 0.3 and out[2]['fx'][1] == 0.05
    assert IntrinsicsPrior(true).priors_for(true.cameras)[3]['fx'][1] == 0.05



def test_detect_moves_drops_windows_a_rotation_cannot_explain():
    """outdoor-mouse false move: the last windows of a camera are scattered detections; their
    rotation fits are wild (47 deg) with residuals far above the camera's usual ones."""
    rig = synth.load_rig('motobs_7cam')
    fn = synth.smooth_tracks(rig, n_points=6, amp=0.1, seed=1)
    t = np.arange(0, 60, 1 / 10)
    X = np.stack([fn(p, t) for p in range(6)], 1)
    p2 = synth.add_noise(synth.project(rig, X), 0.5, seed=2)
    rng = np.random.default_rng(0)
    bad = t >= 48
    p2[2, bad] += rng.normal(0, 40, p2[2, bad].shape)
    mv = ST.detect_moves(rig, p2, t, tau=5.0, z_thr=10.0)
    assert mv.moved == []
    assert len(mv.info['bad_fit'].get(rig.get_names()[2], [])) >= 2
    mv0 = ST.detect_moves(rig, p2, t, tau=5.0, z_thr=10.0, max_resid_ratio=None)
    assert np.isfinite(mv0.delta_deg[2, -2]).all() and not np.isfinite(mv.delta_deg[2, -2]).all()



def test_lens_check_on_the_data_region():
    from aniposelib.cameras import Camera
    cam = Camera(name='a', size=(1000, 800))
    cam.set_camera_matrix(np.array([[600.0, 0, 500], [0, 600, 400], [0, 0, 1]]))
    cam.set_distortions(np.array([-1.0, 0, 0, 0, 0]))       # folds at r ~ 0.58 (normalised)
    centre = np.random.default_rng(0).uniform([400, 300], [600, 500], (200, 2))
    corners = np.random.default_rng(1).uniform([0, 0], [120, 100], (200, 2))
    assert WC._lens_ok_on(cam, centre)
    assert not WC._lens_ok_on(cam, np.concatenate([centre, corners]))
    cam.set_distortions(np.zeros(5))
    assert WC._lens_ok_on(cam, corners)


@pytest.mark.slow
def test_w1_videos_end_to_end(tmp_path):
    """W1 on rendered videos (2 cameras): frame-major sequential detection, early exit, metric rig."""
    import cv2
    from board_render import render_board
    from aniposelib.boards import CharucoBoard
    from aniposelib.workflows import calibrate_from_board
    board = CharucoBoard(6, 5, 24, 18, marker_bits=4, dict_size=50, units='mm')
    K = np.array([[600.0, 0, 320], [0, 600, 240], [0, 0, 1]])
    size = (640, 480)
    R1 = cv2.Rodrigues(np.array([0.0, -0.3, 0.0]))[0]
    t1 = np.array([-150.0, 0, 20.0])
    rng = np.random.default_rng(0)
    c = board.get_object_points().mean(0)
    writers = [cv2.VideoWriter(str(tmp_path / '{}.avi'.format(n)), cv2.VideoWriter_fourcc(*'MJPG'), 10, size)
               for n in 'ab']
    if not writers[0].isOpened():
        pytest.skip('no MJPG video writer')
    for f in range(40):
        rb = rng.normal(0, 0.35, 3)
        Rb = cv2.Rodrigues(rb)[0]
        tb = np.array([rng.uniform(-60, 60), rng.uniform(-40, 40), rng.uniform(380, 520)]) - Rb @ c
        for cam, (R, t) in enumerate([(np.eye(3), np.zeros(3)), (R1, t1)]):
            img = render_board(board, K, cv2.Rodrigues(R @ Rb)[0], R @ tb + t, size) if f % 4 else \
                np.full(size[::-1], 128, np.uint8)
            writers[cam].write(cv2.cvtColor(cv2.GaussianBlur(img, (0, 0), 0.7), cv2.COLOR_GRAY2BGR))
    for w in writers:
        w.release()
    vids = [[str(tmp_path / 'a.avi')], [str(tmp_path / 'b.avi')]]
    from aniposelib.cameras import Camera, CameraGroup
    init = CameraGroup([Camera(matrix=K, dist=np.zeros(5), size=size, name=n) for n in 'ab'])
    cg, res = calibrate_from_board(vids, board, detector='opencv', init=init, intrinsics='fixed',
                                   options={'skip': 1, 'max_boards': 12, 'n_workers': 2})
    det = res.settings['detection']
    assert det['sampling'] == 'sequential' and det['segments'][0]['stopped'] == 'max_boards'
    assert det['n_rows'][1] == 12 and det['segments'][0]['last_frame'] < 39
    assert cg is not None and cg.get_names() == ['a', 'b']
    C = synth.camera_centers(cg)
    assert abs(np.linalg.norm(C[1] - C[0]) - np.linalg.norm(R1.T @ t1)) < 3.0       # mm (render bias ~1 %)


# ------------------------------------------------------------------ init='people' (1.1 gaps: Panoptic 5D)

def _people_fragments(true, n_frames=40, sigma=0.7, seed=4):
    """Two people, every camera's tracker cuts each person's track once (at a different frame per
    camera), track ids have no cross-camera meaning."""
    P3 = np.stack([synth.skeleton(true, n_frames, model=synth.PERSON, size=3.5, spread=0.2, seed=s)[0]
                   for s in (10, 11)], 1)
    p2 = synth.add_noise(synth.project(true, P3), sigma, seed=seed)
    C, F = p2.shape[:2]
    rng = np.random.default_rng(0)
    tid = np.zeros((C, F, 2), int)
    for c in range(C):
        for k in range(2):
            cut = rng.integers(10, F - 10)
            tid[c, :, k] = np.where(np.arange(F) < cut, 10 * k + 1, 10 * k + 2) + 100 * c
    return p2, tid


def test_people_association_merges_track_fragments():
    """A person's track fragments (tracker cuts) join one identity; before 1.1 the fragment
    matches lost the dominance test and the groups with two tracks of a camera were dropped,
    which left most Panoptic detections unassociated (2 of 5 cameras unplaced)."""
    from aniposelib.init import people
    true = synth.load_rig('fisheye_5cam')
    p2, tid = _people_fragments(true)
    assoc, rep = people.associate_people(WC_no_extrinsics(true), p2, track_ids=tid, rng=0, n_windows=3)
    assert assoc.shape[2] == 2 and not rep.dropped
    assert np.isfinite(assoc).all(-1).all(-1).all()            # every camera-frame of both people
    for g in rep.groups:                                       # one person per group
        assert len({(t if isinstance(t, int) else t[0]) % 100 // 10 for t in g.values()}) == 1


def test_sfm_pnp_threshold_follows_the_keypoint_noise():
    """5 px keypoint noise: the fixed 8 px frame test placed 2 of 5 cameras; the threshold now
    scales with the noise measured on the seed pair (noise_mult=None: the old behaviour)."""
    from aniposelib.init import sfm
    true = synth.load_rig('fisheye_5cam')
    p3d, _ = synth.skeleton(true, 40, model=synth.PERSON, size=3.5, spread=0.15, seed=3)
    p2d = synth.add_noise(synth.project(true, p3d), 5.0, seed=4)
    cg, rep = sfm.from_correspondences(WC_no_extrinsics(true), p2d, rng=0)
    assert len(rep.placed) == 5 and 4 < rep.stats['noise_px'] < 6 and rep.stats['pnp_thresh_px'] > 8
    assert synth.center_error(cg, true).max() < 0.03
    _, rep0 = sfm.from_correspondences(WC_no_extrinsics(true), p2d, rng=0, noise_mult=None)
    assert len(rep0.placed) < 5


def test_w2_people_fragmented_tracks_associates_once(monkeypatch):
    """W2 init='people' with fragmented tracks places every camera, and associates only once."""
    from aniposelib.init import people
    true = synth.load_rig('fisheye_5cam')
    p2, tid = _people_fragments(true, sigma=1.0)
    calls = []
    orig = people.associate_people
    monkeypatch.setattr(people, 'associate_people', lambda *a, **k: calls.append(1) or orig(*a, **k))
    cg, res = calibrate_from_points(p2, WC_no_extrinsics(true), init='people', refine_intrinsics=False,
                                    options={'track_ids': tid, 'quality': False,
                                             'people_kwargs': {'association_kwargs': {'n_windows': 3}}})
    assert len(calls) == 1
    assert cg is not None and len(cg.get_names()) == 5, res.reasons
    assert synth.center_error(cg, true).max() < 0.02
