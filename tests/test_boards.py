"""Board detection on the OpenCV class API (no contrib): rendered boards, pose, validation."""

import os
import pickle
import re
from pathlib import Path

import cv2
import numpy as np
import pytest

from aniposelib.boards import CharucoBoard, Checkerboard, DetectionCheck, empty_corners
from aniposelib.cameras import Camera, FisheyeCamera

from board_render import render_board, look_at_pose

K = np.array([[800.0, 0, 400], [0, 800, 300], [0, 0, 1]])
SIZE = (800, 600)

LEGACY_ARUCO = ('detectMarkers', 'refineDetectedMarkers', 'interpolateCornersCharuco',
                'drawDetectedCornersCharuco', 'drawDetectedMarkers', 'estimatePoseCharucoBoard',
                'estimatePoseSingleMarkers', 'estimatePoseBoard', 'calibrateCameraCharuco',
                'calibrateCameraAruco', 'CharucoBoard_create', 'GridBoard_create',
                'DetectorParameters_create', 'Dictionary_get', 'drawAxis', 'drawMarker',
                'drawPlanarBoard')


def make_board(**kw):
    args = dict(squaresX=6, squaresY=5, square_length=24.0, marker_length=18.0)
    args.update(kw)
    return CharucoBoard(**args)


def view(board, distance=450.0, dist=None, rvec=(0.3, -0.25, 0.1), blur=0.7, K=K):
    rvec, tvec = look_at_pose(board, distance, rvec)
    img = render_board(board, K, rvec, tvec, SIZE, dist=dist)
    if blur:
        img = cv2.GaussianBlur(img, (0, 0), blur)
    gt = cv2.projectPoints(board.get_object_points(), rvec, tvec, K,
                           np.zeros(5) if dist is None else np.asarray(dist))[0].reshape(-1, 2)
    return img, gt, rvec, tvec


@pytest.mark.skipif(not os.environ.get('ANIPOSELIB_EXPECT_NOCONTRIB'),
                    reason='only in the nocontrib env (pixi -e nocontrib)')
def test_env_has_no_contrib():
    """In the guard env, cv2 must be plain opencv-python (no legacy aruco free functions)."""
    assert not hasattr(cv2.aruco, 'interpolateCornersCharuco')
    assert not hasattr(cv2.aruco, 'estimatePoseCharucoBoard')


def test_no_legacy_aruco_calls():
    """Guard: the legacy contrib-only cv2.aruco free functions must not come back (0.9.0 broke on them)."""
    pat = re.compile(r'aruco\s*\.\s*(' + '|'.join(LEGACY_ARUCO) + r')\b')
    root = Path(__file__).resolve().parents[1] / 'aniposelib'
    hits = []
    for f in sorted(root.rglob('*.py')):
        for n, line in enumerate(f.read_text().splitlines(), 1):
            if pat.search(line):
                hits.append('{}:{}: {}'.format(f.name, n, line.strip()))
    assert not hits, 'legacy cv2.aruco functions used:\n' + '\n'.join(hits)


@pytest.mark.parametrize('bgr', [False, True])
def test_detect_rendered_board(bgr):
    board = make_board()
    img, gt, _, _ = view(board)
    if bgr:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    corners, ids = board.detect_image(img)
    assert corners.shape == (board.total_size, 1, 2) and corners.dtype == np.float32
    assert ids.shape == (board.total_size, 1) and ids.dtype == np.int32
    assert sorted(ids.ravel()) == list(range(board.total_size))
    err = np.linalg.norm(corners.reshape(-1, 2) - gt[ids.ravel()], axis=1)
    # OpenCV's ChArUco corners sit ~0.5 px down-right of the geometric corner (as in 0.9.x)
    assert np.median(err) < 0.9 and err.max() < 1.5
    assert board.validate_detection(corners, ids)


def test_detect_empty_image_shapes():
    board = make_board()
    corners, ids = board.detect_image(np.full((480, 640), 128, np.uint8))
    assert corners.shape == (0, 1, 2) and ids.shape == (0, 1)
    assert np.all(np.isnan(board.fill_points(corners, ids)))


def test_legacy_pattern_matters_for_even_rows():
    legacy = make_board(squaresX=7, squaresY=6, legacy_pattern=True)
    modern = make_board(squaresX=7, squaresY=6, legacy_pattern=False)
    img, _, _, _ = view(legacy, distance=520)
    n_right = len(legacy.detect_image(img)[1])
    n_wrong = len(modern.detect_image(img)[1])
    assert n_right == legacy.total_size
    assert n_wrong < n_right / 2


def test_aruco_dict_is_honoured():
    board5 = make_board(aruco_dict='DICT_5X5_100')
    assert make_board(aruco_dict=cv2.aruco.DICT_5X5_100).dictionary.markerSize == 5
    img, _, _, _ = view(board5)
    assert len(board5.detect_image(img)[1]) == board5.total_size
    assert len(make_board().detect_image(img)[1]) == 0  # default 4x4_50 sees nothing
    with pytest.raises(ValueError, match='supported pairs'):
        make_board(marker_bits=3, dict_size=50)
    with pytest.raises(ValueError, match='unknown ArUco dictionary'):
        make_board(aruco_dict='DICT_9X9_1')


def test_from_rows_cols_and_draw():
    b = CharucoBoard.from_rows_cols(5, 6, 24.0, 18.0)
    assert (b.squaresX, b.squaresY, b.rows, b.cols) == (6, 5, 5, 6)
    img = b.draw((600, 500))
    assert img.shape == (500, 600) and img.dtype == np.uint8
    c, i = b.detect_image(cv2.copyMakeBorder(img, 40, 40, 40, 40, cv2.BORDER_CONSTANT, value=255))
    assert len(i) == b.total_size
    cb = Checkerboard(7, 5, square_length=1.0)
    img = cb.draw((640, 480))
    corners, ids = cb.detect_image(cv2.GaussianBlur(img, (0, 0), 0.7))
    assert corners is not None and len(corners) == 35


def test_object_points_origin_convention():
    b = make_board()
    assert np.allclose(b.get_object_points()[0], 0)
    assert np.allclose(b.get_object_points('opencv'), b.board.getChessboardCorners())


def test_validate_detection_rejects_wrong_ids():
    board = make_board()
    img, gt, _, _ = view(board)
    corners, ids = board.detect_image(img)
    chk = board.validate_detection(corners, ids)
    assert isinstance(chk, DetectionCheck) and chk and chk.max_residual_px < 3
    assert set(chk.to_dict()) >= {'ok', 'max_residual_px', 'inlier_fraction'}
    # a transposed board definition: same pixels, ids read on the wrong grid
    transposed = make_board(squaresX=5, squaresY=6)
    bad = transposed.validate_detection(corners[:transposed.total_size], ids[:transposed.total_size])
    assert not bad and bad.inlier_fraction < 0.7
    assert not board.validate_detection(*empty_corners())


def test_transposed_board_warns():
    board = make_board()
    img, _, _, _ = view(board)
    corners, ids = board.detect_image(img)
    # rows from a board read with the wrong layout (ids permuted along the grid)
    t = make_board(squaresX=5, squaresY=6)
    rows = [{'framenum': k, 'corners': corners[:t.total_size], 'ids': ids[:t.total_size]}
            for k in range(6)]
    from aniposelib.boards import _warn_if_bad_layout
    with pytest.warns(RuntimeWarning, match='from_rows_cols'):
        _warn_if_bad_layout(t, rows)


def test_validate_gate_drops_detections():
    board = make_board(validate=True, max_homography_px=1e-6)
    img, _, _, _ = view(board)
    assert len(board.detect_image(img)[1]) == 0
    assert make_board(validate=True).detect_image(img)[1].shape[0] == board.total_size


def _angle_deg(r1, r2):
    R = cv2.Rodrigues(np.asarray(r1, float))[0].T @ cv2.Rodrigues(np.asarray(r2, float))[0]
    return np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1)))


def test_estimate_pose_rendered_pinhole():
    board = make_board()
    img, _, rvec, tvec = view(board)
    cam = Camera(matrix=K, dist=np.zeros(5), size=SIZE)
    corners, ids = board.detect_image(img)
    r, t = board.estimate_pose_points(cam, corners, ids)
    assert _angle_deg(r, rvec) < 0.5
    assert np.linalg.norm(t.ravel() - tvec) < 0.01 * np.linalg.norm(tvec)
    # OpenCV-origin poses differ by one square along the board's x and y axes
    r2, t2 = board.estimate_pose_points(cam, corners, ids, origin='opencv')
    R = cv2.Rodrigues(r2)[0]
    assert _angle_deg(r2, r) < 1e-6
    sl = board.square_length
    assert np.allclose(t2.ravel() + R @ np.array([sl, sl, 0.0]), t.ravel(), atol=1e-4)


@pytest.mark.parametrize('fisheye', [False, True])
def test_estimate_pose_exact(fisheye):
    """Noise-free projected corners (with lens distortion) give the exact pose, pinhole and fisheye."""
    board = make_board()
    rvec, tvec = look_at_pose(board, 400.0, (0.5, 0.2, -0.3), offset=(30, -20))
    objp = board.get_object_points()
    if fisheye:
        cam = FisheyeCamera(matrix=[[400, 0, 400], [0, 400, 300], [0, 0, 1]],
                            dist=[0.05, -0.01, 0.002, 0.0], size=SIZE)
    else:
        cam = Camera(matrix=K, dist=[-0.2, 0.05, 0.001, -0.001, 0.0], size=SIZE)
    cam.set_rotation(rvec)
    cam.set_translation(tvec)
    p2d = cam.project(objp).detach().cpu().numpy()
    ids = np.arange(len(objp))[::2].reshape(-1, 1).astype('int32')  # partial view
    r, t = board.estimate_pose_points(cam, p2d[ids.ravel()].reshape(-1, 1, 2), ids)
    assert _angle_deg(r, rvec) < 1e-3
    assert np.allclose(t.ravel(), tvec, atol=1e-3)


def test_estimate_pose_degenerate():
    board = make_board()
    cam = Camera(matrix=K, dist=np.zeros(5), size=SIZE)
    assert board.estimate_pose_points(cam, *empty_corners()) == (None, None)
    ids = np.arange(5).reshape(-1, 1).astype('int32')       # one board row: collinear
    pts = np.random.default_rng(0).uniform(0, 500, (5, 1, 2)).astype('float32')
    assert board.estimate_pose_points(cam, pts, ids, min_corners=4) == (None, None)


def test_detect_images_arrays_and_video(tmp_path):
    board = make_board()
    imgs = [cv2.cvtColor(view(board, distance=d)[0], cv2.COLOR_GRAY2BGR) for d in (420, 480)]
    rows = board.detect_images(imgs + [np.full_like(imgs[0], 128)])
    assert [len(r['ids']) for r in rows] == [board.total_size, board.total_size, 0]
    assert rows[0]['filled'].shape == (board.total_size, 1, 2)
    fname = str(tmp_path / 'board.avi')
    w = cv2.VideoWriter(fname, cv2.VideoWriter_fourcc(*'MJPG'), 10, SIZE)
    if not w.isOpened():
        pytest.skip('no MJPG video writer')
    for k in range(12):
        w.write(imgs[k % 2] if k % 3 else np.full_like(imgs[0], 128))
    w.release()
    rows = board.detect_video(fname, frame_indices=[1, 2, 3, 7], prefix=0)
    assert [r['framenum'] for r in rows] == [(0, 1), (0, 2), (0, 7)]
    rows = board.detect_video(fname, skip=4)
    assert len(rows) >= 4


def test_board_pickles():
    board = make_board(legacy_pattern=True, aruco_dict='DICT_4X4_100')
    b2 = pickle.loads(pickle.dumps(board))
    assert b2.board.getLegacyPattern() and b2.dictionary.bytesList.shape == board.dictionary.bytesList.shape
    img, _, _, _ = view(board)
    assert len(b2.detect_image(img)[1]) == board.total_size


def test_custom_detector_instance():
    class Fixed:
        def detect(self, image):
            pts = np.random.default_rng(1).uniform(0, 500, (8, 1, 2))
            return pts.astype('float32'), np.arange(8, dtype='int32').reshape(-1, 1)
    board = make_board(detector=Fixed())
    assert board.validate  # learned/custom detectors are validated by default
    c, i = board.detect_image(np.zeros((10, 10), np.uint8))
    assert len(i) == 0  # random corners fail the homography gate
    board = make_board(detector=Fixed(), validate=False)
    assert len(board.detect_image(np.zeros((10, 10), np.uint8))[1]) == 8


def test_detect_and_pose_rendered_fisheye():
    board = make_board()
    Kf = np.array([[420.0, 0, 400], [0, 420, 300], [0, 0, 1]])
    D = np.array([0.08, -0.03, 0.01, 0.0])
    rvec, tvec = look_at_pose(board, 260.0, (0.35, -0.3, 0.2), offset=(25, -10))
    img = cv2.GaussianBlur(render_board(board, Kf, rvec, tvec, SIZE, dist=D, fisheye=True), (0, 0), 0.7)
    corners, ids = board.detect_image(img)
    assert len(ids) >= 0.9 * board.total_size
    cam = FisheyeCamera(matrix=Kf, dist=D, size=SIZE)
    r, t = board.estimate_pose_points(cam, corners, ids)
    assert _angle_deg(r, rvec) < 1.0
    assert np.linalg.norm(t.ravel() - tvec) < 0.02 * np.linalg.norm(tvec)
