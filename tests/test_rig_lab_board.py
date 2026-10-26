"""Lab ChArUco rigs (Allen 7/12-camera mouse rigs): the classic anipose board-calibration flow,
from blank cameras (sizes only) to a calibrated rig, on synthetic board detections."""

import numpy as np
import pytest

import synth
from aniposelib.boards import CharucoBoard, merge_rows, extract_points, extract_rtvecs
from aniposelib.cameras import CameraGroup, init_camera_intrinsics
from aniposelib.utils import get_initial_extrinsics

pytestmark = pytest.mark.integration

BOARD = dict(squaresX=6, squaresY=5, square_length=24, marker_length=18, marker_bits=4, dict_size=50)
FAST_BA = dict(n_iters=2, max_nfev=100, n_samp_full=300, n_samp_iter=150, rng=0)


def make_board():
    return CharucoBoard(**BOARD)


def blank_like(true):
    cg = CameraGroup.from_names(true.get_names())
    for c, ct in zip(cg.cameras, true.cameras):
        c.set_size(ct.get_size())
    return cg


def focal_rel_err(cg, true):
    f = np.array([synth.to_np(c.get_camera_matrix())[0, 0] for c in cg.cameras])
    ft = np.array([synth.to_np(c.get_camera_matrix())[0, 0] for c in true.cameras])
    return np.abs(f / ft - 1)


def initialise(true, n_frames=40, seed=1):
    """intrinsics + board poses + initial extrinsics (everything before BA)."""
    board = make_board()
    rows = synth.board_rows(true, board, n_frames=n_frames, seed=seed)
    cg = blank_like(true)
    for r, cam in zip(rows, cg.cameras):
        objp, imgp = board.get_all_calibration_points(r)
        init_camera_intrinsics(cam, objp, imgp, cam.get_size())
    rows = [board.estimate_pose_rows(c, r) for c, r in zip(cg.cameras, rows)]
    merged = merge_rows(rows)
    imgp, extra = extract_points(merged, board, min_cameras=2)
    rvecs, tvecs = get_initial_extrinsics(extract_rtvecs(merged), cg.get_names())
    cg.set_rotations(rvecs)
    cg.set_translations(tvecs)
    return cg, imgp, extra


@pytest.mark.parametrize('rig', ['allen_7cam', 'allen_12cam'])
def test_board_detections_cover_every_camera(rig):
    true = synth.load_rig(rig)
    rows = synth.board_rows(true, make_board(), n_frames=40, seed=1)
    assert all(len(r) >= 10 for r in rows)


@pytest.mark.parametrize('rig', ['allen_7cam', 'allen_12cam'])
def test_board_initialisation(rig):
    """Intrinsics from board views and extrinsics from the pose graph land near the truth."""
    true = synth.load_rig(rig)
    cg, imgp, extra = initialise(true)
    assert focal_rel_err(cg, true).max() < 0.05
    assert synth.center_error(cg, true).max() < 0.08
    assert cg.average_error(imgp, median=True).item() < 5


def test_calibrate_rows_end_to_end():
    """The public one-call API on the 7-camera rig (short BA budget)."""
    true = synth.load_rig('allen_7cam')
    rows = synth.board_rows(true, make_board(), n_frames=40, seed=1)
    cg = blank_like(true)
    err = cg.calibrate_rows(rows, make_board(), verbose=False, **FAST_BA)
    assert err < 3
    assert synth.center_error(cg, true).max() < 0.08


@pytest.mark.target
@pytest.mark.xfail(reason='1.0: LM/robust BA should recover board rigs to <0.5% of rig scale and <0.5% focal', strict=False)
def test_board_ba_recovers_rig_precisely():
    true = synth.load_rig('allen_7cam')
    cg, imgp, extra = initialise(true)
    cg.bundle_adjust_iter(imgp, extra, **FAST_BA)
    assert synth.center_error(cg, true).max() < 0.005
    assert focal_rel_err(cg, true).max() < 0.005
