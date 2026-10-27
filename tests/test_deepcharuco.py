"""DeepCharuco detector (optional): skipped when the weights cannot be found or downloaded."""

import os

import cv2
import numpy as np
import pytest

from aniposelib.boards import CharucoBoard
from aniposelib.detectors import deepcharuco as dc

from board_render import render_board, look_at_pose

# the nets were trained at 320x240; small frames keep the CPU tests fast
K = np.array([[260.0, 0, 160], [0, 260, 120], [0, 0, 1]])
SIZE = (320, 240)
GEOM = (6, 5)  # squaresX, squaresY -> r5_c6 (small published pair)


def make_board(**kw):
    return CharucoBoard(*GEOM, square_length=24.0, marker_length=18.0, **kw)


def _weights_available():
    """True if the r5_c6 weights are cached, or downloadable (unless ANIPOSELIB_NO_DOWNLOAD is set)."""
    det, ref = dc.checkpoint_names(*GEOM)
    download = not os.environ.get('ANIPOSELIB_NO_DOWNLOAD')
    try:
        dc.fetch_weights(det, download=download)
        dc.fetch_weights(ref, download=download)
        return True
    except Exception:
        return False


needs_weights = pytest.mark.skipif(not _weights_available(),
                                   reason='DeepCharuco weights not cached and not downloadable')


@pytest.fixture(scope='module')
def scene():
    board = make_board()
    rvec, tvec = look_at_pose(board, 300.0)
    img = cv2.GaussianBlur(render_board(board, K, rvec, tvec, SIZE), (0, 0), 0.8)
    gt = cv2.projectPoints(board.get_object_points(), rvec, tvec, K, np.zeros(5))[0].reshape(-1, 2)
    return board, img, gt


@pytest.fixture(scope='module')
def detector():
    import torch
    n = torch.get_num_threads()
    torch.set_num_threads(min(n, 4))  # small convs: many threads only add overhead
    yield dc.DeepCharucoDetector(GEOM, device='cpu', batch_size=4)
    torch.set_num_threads(n)


def test_checkpoint_names_and_errors():
    assert dc.checkpoint_names(9, 7) == ('detectnet_r7_c9.ckpt', 'refinenet_r7_c9.ckpt')
    assert dc.checkpoint_names(5, 6)[1] is None  # r6_c5 has no refinenet
    with pytest.raises(ValueError, match='no published DeepCharuco checkpoint'):
        dc.checkpoint_names(11, 8)
    assert not dc.supports_board(4, 7)


def test_preprocess_roundtrip():
    gray = np.zeros((100, 130), np.uint8)
    x, meta = dc.preprocess(gray, 0.5)
    assert x.shape[0] % 8 == 0 and x.shape[1] % 8 == 0
    assert x.min() == pytest.approx(-128 / 255)
    # cv2.resize pixel-centre convention
    assert np.allclose(dc.to_original([[0.0, 0.0]], {'sx': 0.5, 'sy': 0.5}), [[0.5, 0.5]])


@needs_weights
def test_deepcharuco_detects_rendered_board(scene, detector):
    board, img, gt = scene
    corners, ids = detector.detect(img)
    assert corners.shape[1:] == (1, 2) and ids.dtype == np.int32
    assert len(ids) >= 0.9 * board.total_size
    err = np.linalg.norm(corners.reshape(-1, 2) - gt[ids.ravel()], axis=1)
    assert np.median(err) < 0.5 and err.max() < 1.5
    assert board.validate_detection(corners, ids)


@needs_weights
def test_deepcharuco_batch_matches_single(scene, detector):
    board, img, gt = scene
    bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    out = detector.detect_batch([img, np.full_like(img, 128), bgr, img[:200, :280]])
    single = detector.detect(img)
    assert np.allclose(out[0][0], single[0]) and np.array_equal(out[0][1], single[1])
    assert len(out[1][1]) == 0 and out[1][0].shape == (0, 1, 2)
    assert np.allclose(out[2][0], single[0])
    assert len(out[3][1]) > 0


@needs_weights
def test_board_with_deepcharuco_detector(scene, detector):
    board, img, gt = scene
    b = make_board(detector=detector)
    assert b.validate and b.batch_size == 4
    rows = b.detect_images([img])
    assert len(rows) == 1 and len(rows[0]['ids']) >= 0.9 * b.total_size
    frames = b.detect_frames([img, img])
    assert all(len(i) == len(rows[0]['ids']) for _, i in frames)
    b2 = make_board(detector='deepcharuco', detector_kwargs=dict(device='cpu'))
    assert b2.detector.squaresX == GEOM[0]  # built lazily from the board



@needs_weights
def test_deepcharuco_pixel_budget_and_auto_scale(scene):
    board, img, gt = scene
    det = dc.DeepCharucoDetector(GEOM, device='cpu', batch_size=8, max_batch_pixels=1.0)   # one frame per call
    calls = []
    net = det.detector
    det.detector = lambda x: calls.append(x.shape[0]) or net(x)
    out = det.detect_batch([img, img, img])
    assert calls == [1, 1, 1] and np.allclose(out[0][0], out[2][0])
    auto = dc.DeepCharucoDetector(GEOM, device='cpu', scale='auto')
    assert auto.scale_for((1080, 1440)) == pytest.approx(0.5) and auto.scale_for((480, 640)) == 1.0
    assert np.allclose(auto.detect(img)[0], det.detect(img)[0])        # 800x600: never upscaled
    with pytest.raises(ValueError):
        dc.DeepCharucoDetector(GEOM, device='cpu', scale='big')
