"""Data handed between workflows, as documented in docs/workflows.md ("Getting your data in")."""
import numpy as np

import aniposelib as al
import synth
from test_rig_lab_board import make_board


def test_board_detections_flow_and_lens_only_calibration():
    true = synth.load_rig('allen_7cam')
    board = make_board()
    rows = synth.board_rows(true, board, n_frames=30, seed=1)
    cg, res = al.calibrate_from_board(rows, board, cam_names=true.get_names(), init=true, intrinsics='fixed')
    s = al.estimate_scale(cg, board=board, board_points=res.detections)      # W1 detections -> W3
    assert s.accepted and abs(s.scale - 1) < 0.01
    # every camera filmed the board on its own: positions fail, lenses are still there
    solo = [[dict(r, framenum=(c, r['framenum'])) for r in rr] for c, rr in enumerate(rows)]
    sizes = {c.get_name(): [int(v) for v in synth.to_np(c.get_size())] for c in true.cameras}
    cg2, res2 = al.calibrate_from_board(solo, board, cam_names=true.get_names(), options={'sizes': sizes})
    assert cg2 is None and res2.verdict == 'fail' and res2.lenses is not None
    f = np.array([float(synth.to_np(c.get_camera_matrix())[0, 0]) for c in res2.lenses.cameras])
    f0 = np.array([float(synth.to_np(c.get_camera_matrix())[0, 0]) for c in true.cameras])
    assert np.median(np.abs(f / f0 - 1)) < 0.05


def test_associate_views_output_feeds_reconstruct_and_calibrate():
    true = synth.load_rig('allen_7cam')
    X = synth.crowd(true, 3, 40, seed=0)
    P, F, J, _ = X.shape
    C = len(true.cameras)
    uv = np.asarray(true.project(X.reshape(-1, 3))).reshape(C, P, F, J, 2)
    uv = uv + np.random.default_rng(0).normal(0, 0.5, uv.shape)
    dets = [[uv[c, :, f] for f in range(F)] for c in range(C)]
    a = al.associate_views(true, dets)
    assert a.points.shape[:2] == (C, F) and a.points.shape[2] >= 3
    r = al.reconstruct_3d(true, a.points)
    assert r.points3d.shape == (F, a.points.shape[2], J, 3)
    cg, res = al.calibrate_from_points(a.points.reshape(C, -1, J, 2), synth.perturb_extrinsics(true, 1.0, 0.01),
                                       init='reference', refine_intrinsics=False)
    assert res.verdict != 'fail'
