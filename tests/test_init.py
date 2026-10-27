"""aniposelib.init (plan 01 phase E): board graphs, keypoint SfM, people bootstrap, selection
and the calibrate_from_points recipe, on synthetic rigs."""

import json

import numpy as np
import pytest

import synth
from aniposelib.boards import CharucoBoard, merge_rows, extract_rtvecs
from aniposelib.init import board as B, people, sfm
from aniposelib.init import calibrate_from_points, select
from aniposelib.init._common import get_extrinsics, rodrigues
from aniposelib.utils import get_initial_extrinsics

pytestmark = pytest.mark.integration

BOARD = dict(squaresX=6, squaresY=5, square_length=24, marker_length=18, marker_bits=4, dict_size=50)


# ------------------------------------------------------------------ board

@pytest.fixture(scope='module')
def board_scene():
    true = synth.load_rig('allen_7cam')
    bd = CharucoBoard(**BOARD)
    rows = synth.board_rows(true, bd, n_frames=40, seed=1)
    rows = [bd.estimate_pose_rows(c, r) for c, r in zip(true.cameras, rows)]
    return true, extract_rtvecs(merge_rows(rows))


def test_board_robust_matches_classic(board_scene):
    true, rt = board_scene
    for method in ('classic', 'robust', 'auto'):
        bi = B.initial_extrinsics(rt, true.get_names(), method=method)
        assert bi.connected
        assert synth.center_error(bi.apply(true.copy()), true).max() < 0.01


def test_board_disconnected_graph(board_scene):
    """Cameras {0,1,2} and {3..6} never see the board together: the classic initializer
    raises, the robust one returns one gauge per component, each accurate."""
    true, rt = board_scene
    rt = rt.copy()
    A, Bc = [0, 1, 2], [3, 4, 5, 6]
    rt[A, 0::2] = np.nan
    rt[Bc, 1::2] = np.nan
    with pytest.raises(ValueError):
        get_initial_extrinsics(rt, true.get_names())
    names = true.get_names()
    bi = B.initial_extrinsics(rt, names)
    assert not bi.connected
    assert sorted(map(sorted, bi.components)) == sorted([sorted(names[i] for i in A),
                                                         sorted(names[i] for i in Bc)])
    for k, comp in enumerate(bi.components):
        sub = bi.apply(true.copy(), component=k)
        assert synth.center_error(sub, true.subset_cameras_names(comp)).max() < 0.01
        anchor = sub.cameras[[c.get_name() for c in sub.cameras].index(bi.anchors[k])]
        assert np.allclose(synth.to_np(anchor.get_extrinsics_mat()), np.eye(4), atol=1e-9)
    json.dumps(bi.to_dict())
    assert B.board_components(rt) == [Bc, A]


def test_guarded_intrinsics():
    true = synth.load_rig('allen_7cam')
    bd = CharucoBoard(**BOARD)
    rows = synth.board_rows(true, bd, n_frames=30, seed=2)[0]
    objp, imgp = bd.get_all_calibration_points(rows)
    size = true.cameras[0].get_size()
    cam = true.cameras[0].copy()
    seed = B.init_camera_intrinsics(cam, objp, imgp, size)
    assert seed and seed.source == 'board'
    f_true = synth.to_np(true.cameras[0].get_camera_matrix())[0, 0]
    assert abs(synth.to_np(cam.get_camera_matrix())[0, 0] / f_true - 1) < 0.1
    # an impossible bound rejects the estimate -> nominal seed
    cam = true.cameras[0].copy()
    seed = B.init_camera_intrinsics(cam, objp, imgp, size, bounds=dict(focal_px=(1, 2)))
    assert not seed and seed.source == 'fallback:nominal' and seed.reason == 'focal_outside_bounds'
    K = synth.to_np(cam.get_camera_matrix())
    assert K[0, 0] == max(size) and abs(K[0, 2] - (size[0] - 1) / 2) < 1e-9
    with pytest.raises(ValueError):
        B.init_camera_intrinsics(true.cameras[0].copy(), objp, imgp, size,
                                 bounds=dict(focal_px=(1, 2)), fallback='raise')
    # degenerate input (one tiny view) does not raise
    cam = true.cameras[0].copy()
    seed = B.init_camera_intrinsics(cam, objp[:1], [i[:4] for i in imgp[:1]], size)
    assert np.all(np.isfinite(synth.to_np(cam.get_camera_matrix())))
    ok, why = B.valid_intrinsics(np.eye(3) * 1000, np.zeros(5), (1000, 800))
    assert not ok and why == 'principal_x_outside_bounds'


# ------------------------------------------------------------------ keypoint SfM

@pytest.fixture(scope='module')
def mouse_scene():
    true = synth.load_rig('motobs_7cam')
    p3d, bones = synth.skeleton(true, 80, model=synth.MOUSE, size=1.0, spread=0.12, seed=1)
    p2d = synth.add_noise(synth.project(true, p3d), 0.7, seed=2)
    p2d_decoy, frames = synth.add_decoy(p2d, cam=3, frac=0.6, cgroup=true, seed=3)
    return true, p2d, p2d_decoy


def blank(cg):
    cg = cg.copy()
    for c in cg.cameras:
        c.set_rotation(np.zeros(3))
        c.set_translation(np.zeros(3))
    return cg


@pytest.mark.slow
def test_sfm_mouse_with_decoy_recovers_rig(mouse_scene):
    """From intrinsics only, a mouse seen by 7 cameras, one locked onto a decoy for 60 % of
    the frames: every camera placed within 0.5 % of rig scale (similarity aligned)."""
    true, _, p2d_decoy = mouse_scene
    cg, rep = sfm.from_correspondences(blank(true), p2d_decoy, rng=0)
    assert rep.placed == true.get_names() and not rep.rejected and not rep.missing
    assert synth.center_error(cg, true).max() < 0.005
    assert rep.stats['median_px'] < 1.5
    json.dumps(rep.to_dict())


def test_sfm_person_fisheye():
    true = synth.load_rig('fisheye_5cam')
    p3d, _ = synth.skeleton(true, 60, model=synth.PERSON, size=3.5, spread=0.15, seed=3)
    p2d = synth.add_noise(synth.project(true, p3d), 0.7, seed=4)
    cg, rep = sfm.from_correspondences(blank(true), p2d, rng=1)
    assert len(rep.placed) == 5
    assert synth.center_error(cg, true).max() < 0.005


def test_place_with_rotation_prior():
    """With the rotation given, one frame fixes the position (linear solve)."""
    true = synth.load_rig('motobs_7cam')
    p3d, _ = synth.skeleton(true, 30, model=synth.MOUSE, seed=5)
    X = p3d.reshape(-1, 3)
    u, f = sfm.ideal_points(true, synth.project(true, p3d).reshape(7, -1, 2))
    M = get_extrinsics(true)[2]
    fid = np.repeat(np.arange(30), p3d.shape[1])
    # a compact scene: a wrong rotation is partly absorbed by the translation, so only the
    # data (inlier frames, then the PnP polish) decide; the result must be the true pose
    wrong = rodrigues([0.0, 0.5, 0]) @ M[:3, :3]
    r = sfm.place_with_rotation_prior(X, u[2], fid, f[2], [wrong, M[:3, :3]], rng=0)
    assert r['n_inlier_frames'] == 30
    assert np.allclose(rodrigues(r['rvec']), M[:3, :3], atol=1e-4)
    assert np.allclose(r['tvec'], M[:3, 3], atol=1e-3 * np.linalg.norm(M[:3, 3]))
    t = sfm.solve_t_given_R(M[:3, :3], X[:10], u[2, :10] / f[2])
    assert np.allclose(t, M[:3, 3], atol=1e-6 * np.linalg.norm(M[:3, 3]))



# ------------------------------------------------------------------ SfM speed-ups (1.1): parallel RANSAC

def test_frame_pnp_ransac_independent_of_threads():
    """The PnP RANSAC streams have seeds drawn from rng: same result with 1 or 4 threads, and
    the true pose is found."""
    true = synth.load_rig('motobs_7cam')
    p3d, _ = synth.skeleton(true, 40, model=synth.MOUSE, seed=5)
    X = p3d.reshape(-1, 3)
    u, f = sfm.ideal_points(true, synth.add_noise(synth.project(true, p3d), 0.5, seed=1).reshape(7, -1, 2))
    fid = np.repeat(np.arange(40), p3d.shape[1])
    M = get_extrinsics(true)[3]
    rs = [sfm.frame_pnp_ransac(X, u[3], fid, f[3], n_iters=120, rng=7, n_jobs=n) for n in (1, 4)]
    for k in ('rvec', 'tvec', 'inlier_frames'):
        assert np.array_equal(rs[0][k], rs[1][k])
    assert rs[0]['n_inlier_frames'] == 40
    assert np.allclose(rodrigues(rs[0]['rvec']), M[:3, :3], atol=1e-2)


def test_frames_sample_whole_frames():
    fid = np.repeat(np.arange(50), 4)
    valid = np.ones(200, bool)
    valid[::7] = False
    fr = sfm._Frames(valid, fid, 2)
    rng = np.random.default_rng(0)
    for _ in range(20):
        sel = fr.sample(rng, 3)
        frames = np.unique(fid[sel])
        assert len(frames) >= 3 and len(sel) >= 6
        # every valid point of a sampled frame is in the sample
        assert set(sel) == set(np.nonzero(valid & np.isin(fid, frames))[0])


def test_pairwise_poses_usac_vs_ransac():
    """USAC (default) and the 1.0 RANSAC find poses of the same quality; deterministic."""
    true = synth.load_rig('motobs_7cam')
    M = get_extrinsics(true)
    p3d, _ = synth.skeleton(true, 200, model=synth.MOUSE, seed=2)
    p2 = synth.add_outliers(synth.add_noise(synth.project(true, p3d), 0.5, seed=3), frac=0.1, cgroup=true, seed=4)
    u, f = sfm.ideal_points(true, p2.reshape(7, -1, 2))
    prs = [(0, 1), (2, 5), (1, 4)]
    res = {m: sfm.pairwise_poses(u, f, rng=0, pairs=prs, method=m) for m in ('usac', 'ransac')}
    again = sfm.pairwise_poses(u, f, rng=0, pairs=prs, n_jobs=1)
    for i, j in prs:
        assert np.array_equal(again[(i, j)]['R'], res['usac'][(i, j)]['R'])
        Rt = M[j][:3, :3] @ M[i][:3, :3].T
        errs = {m: np.degrees(np.arccos(np.clip((np.trace(Rt.T @ r[(i, j)]['R']) - 1) / 2, -1, 1)))
                for m, r in res.items()}
        # a compact mouse scene: pair rotations are only a seed (a few degrees)
        assert errs['usac'] < max(2 * errs['ransac'], 5.0)
        assert res['usac'][(i, j)]['n_in'] >= 0.95 * res['ransac'][(i, j)]['n_in']
    with pytest.raises(ValueError):
        sfm.pairwise_poses(u, f, pairs=prs, method='lmeds')


@pytest.mark.slow
def test_sfm_deterministic_threads_and_device(mouse_scene):
    """Same rng -> same rig, whatever the number of threads; device= runs the robust
    re-triangulations there (CUDA when available) with the same result to rounding."""
    import torch
    true, _, p2d_decoy = mouse_scene
    a, ra = sfm.from_correspondences(blank(true), p2d_decoy, rng=3, n_jobs=1)
    b, rb = sfm.from_correspondences(blank(true), p2d_decoy, rng=3, n_jobs=4)
    assert ra.placed == rb.placed
    assert np.array_equal(get_extrinsics(a), get_extrinsics(b))
    if torch.cuda.is_available():
        c, rc = sfm.from_correspondences(blank(true), p2d_decoy, rng=3, device='cuda')
        assert rc.placed == ra.placed
        assert np.allclose(get_extrinsics(c), get_extrinsics(a), atol=1e-6)


@pytest.mark.slow
def test_people_bootstrap():
    """Two people, detection slots shuffled per camera: identities are recovered from
    epipolar geometry, then SfM recovers the rig."""
    true = synth.load_rig('fisheye_5cam')
    P3 = np.stack([synth.skeleton(true, 30, model=synth.PERSON, size=3.5, spread=0.2, seed=s)[0]
                   for s in (10, 11)], 1)
    p2 = synth.add_noise(synth.project(true, P3), 0.7, seed=4)
    rng = np.random.default_rng(0)
    perm = np.stack([rng.permutation(2) for _ in range(5)])
    p2s = np.stack([p2[c][:, perm[c]] for c in range(5)])
    cg, rep = people.from_people(blank(true), p2s, rng=0, association_kwargs=dict(n_windows=3))
    groups = rep.stats['association']['groups']
    assert len(groups) == 2
    for g in groups:   # every camera's slot points to the same true person
        assert len({int(perm[c][g[n]]) for c, n in enumerate(true.get_names())}) == 1
    assert synth.center_error(cg, true.subset_cameras_names(rep.placed)).max() < 0.01


# ------------------------------------------------------------------ selection, refinement, recipe

def test_select_picks_best_candidate(mouse_scene):
    true, p2d, _ = mouse_scene
    cands = {'far': synth.perturb_extrinsics(true, rot_deg=3, trans_frac=0.05, seed=1),
             'close': synth.perturb_extrinsics(true, rot_deg=0.1, trans_frac=0.002, seed=2),
             'mid': synth.perturb_extrinsics(true, rot_deg=1, trans_frac=0.02, seed=3)}
    best, res = select(cands, p2d)
    assert res.best == 'close' and res.ranking() == ['close', 'mid', 'far']
    _, res2 = select(list(cands.values()), p2d, score='median_px')
    assert res2.best == 1
    json.dumps(res.to_dict())


def test_refine_group_pulls_in_perturbed_rig(mouse_scene):
    true, _, p2d_decoy = mouse_scene
    cg = synth.perturb_extrinsics(true, rot_deg=1, trans_frac=0.02, seed=4)
    out, st = sfm.refine_group(cg, p2d_decoy, rng=0)
    assert synth.center_error(cg, true).max() > 0.03
    assert synth.center_error(out, true).max() < 0.005
    assert st['median_px'] < 1.0


@pytest.mark.slow
def test_calibrate_from_points_recipe():
    """Blank extrinsics + intrinsics -> SfM -> refinement -> metric scale from a reference
    -> quality report."""
    true = synth.load_rig('fisheye_5cam')
    p3d, _ = synth.skeleton(true, 60, model=synth.PERSON, size=3.5, spread=0.15, seed=3)
    p2d = synth.add_noise(synth.project(true, p3d), 0.7, seed=4)
    cg, rep = calibrate_from_points(p2d, intrinsics=blank(true), rng=0, scale=true)
    assert rep.init == 'sfm' and cg.get_names() == true.get_names()
    assert synth.center_error(cg, true).max() < 0.005
    assert abs(synth.rig_scale(cg) / synth.rig_scale(true) - 1) < 0.01
    # scale=<reference> also aligns the gauge (rotation and origin) to the reference
    assert np.abs(synth.camera_centers(cg) - synth.camera_centers(true)).max() < 0.01 * synth.rig_scale(true)
    assert rep.scale.details['align']['gauge_cameras']
    assert rep.quality is not None
    json.dumps(rep.to_dict())
    # a given candidate competes with SfM through select()
    cg2, rep2 = calibrate_from_points(p2d, intrinsics=blank(true), rng=0, report=False,
                                      init=['sfm', synth.perturb_extrinsics(true, 3, 0.05, seed=1)],
                                      refine=None)
    assert rep2.init == 'sfm' and rep2.selection is not None
