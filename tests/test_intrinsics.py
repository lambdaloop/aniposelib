"""intrinsics.calibrate_ladder: held-out model selection on synthetic board views."""

import cv2
import numpy as np
import pytest

from aniposelib import intrinsics
from aniposelib.boards import CharucoBoard
from aniposelib.cameras import FisheyeCamera

SIZE = (1280, 960)


def views(K, dist, n=40, fisheye=False, noise=0.3, seed=0):
    rng = np.random.default_rng(seed)
    objp = CharucoBoard(9, 7, 30.0, 22.0).get_object_points()
    centred = objp - objp.mean(0)
    O, I = [], []
    while len(O) < n:
        R = cv2.Rodrigues(rng.normal(scale=0.4, size=3))[0]
        z = rng.uniform(400, 900)
        P = centred @ R.T + [rng.uniform(-0.3, 0.3) * z, rng.uniform(-0.25, 0.25) * z, z]
        if fisheye:
            p = cv2.fisheye.projectPoints(P.reshape(-1, 1, 3), np.zeros(3), np.zeros(3), K,
                                          np.asarray(dist, float).reshape(4, 1))[0].reshape(-1, 2)
        else:
            p = cv2.projectPoints(P, np.zeros(3), np.zeros(3), K, np.asarray(dist, float))[0].reshape(-1, 2)
        ok = (p[:, 0] > 0) & (p[:, 0] < SIZE[0]) & (p[:, 1] > 0) & (p[:, 1] < SIZE[1])
        if ok.sum() < 12:
            continue
        O.append(objp[ok])
        I.append(p[ok] + rng.normal(scale=noise, size=(ok.sum(), 2)))
    return O, I


def test_ladder_selects_parsimonious_brown_model():
    K = np.array([[900.0, 0, 640], [0, 900, 480], [0, 0, 1]])
    O, I = views(K, [-0.25, 0.08, 0, 0, 0])
    res = intrinsics.calibrate_ladder(O, I, SIZE)
    assert res.selected == 'k1k2'
    assert res.fits['pinhole'].heldout['p95'] > 2 * res.best.heldout['p95']
    assert abs(res.best.K[0, 0] / 900 - 1) < 0.01
    assert res.best.heldout['rms'] < 0.5
    assert set(res.train_idx).isdisjoint(res.heldout_idx) and len(res.heldout_idx) >= 4
    cam = res.to_camera(name='A')
    assert cam.get_n_dist() == 2 and cam.get_size() == SIZE
    d = res.to_dict()
    assert d['selected'] == 'k1k2' and 'pinhole' in d['fits']


def test_ladder_fisheye_models():
    K = np.array([[500.0, 0, 640], [0, 500, 480], [0, 0, 1]])
    O, I = views(K, [0.05, -0.02, 0.01, 0.0], fisheye=True)
    res = intrinsics.calibrate_ladder(O, I, SIZE, models=('pinhole', 'k1k2', 'fisheye_k1k2', 'fisheye'))
    assert res.best.heldout['rms'] < 0.5
    fit = res.fits['fisheye']
    assert abs(fit.K[0, 0] / 500 - 1) < 0.01
    cam = fit.to_camera(size=SIZE)
    assert isinstance(cam, FisheyeCamera) and cam.get_n_dist() == 4


def test_ladder_rows_and_errors():
    board = CharucoBoard(9, 7, 30.0, 22.0)
    K = np.array([[900.0, 0, 640], [0, 900, 480], [0, 0, 1]])
    O, I = views(K, [-0.1, 0, 0, 0, 0], n=30)
    objp = board.get_object_points()
    rows = []
    for k, (o, i) in enumerate(zip(O, I)):
        ids = np.array([int(np.argmin(np.linalg.norm(objp - p, axis=1))) for p in o])
        rows.append({'framenum': (0, 10 * k), 'corners': i.reshape(-1, 1, 2).astype('float32'),
                     'ids': ids.reshape(-1, 1).astype('int32')})
    res = intrinsics.calibrate_ladder_rows(rows, board, SIZE, models=('pinhole', 'k1k2'))
    assert res.selected == 'k1k2' and len(res.extra['framenums']) == 30
    with pytest.raises(ValueError):
        intrinsics.calibrate_ladder(O[:2], I[:2], SIZE, train_idx=[0, 1], heldout_idx=[0])
    with pytest.raises(ValueError, match='unknown model'):
        intrinsics.calibrate_ladder(O, I, SIZE, models=('pinhole', 'zoom'))


def test_block_holdout():
    tr, ho = intrinsics.block_holdout(np.arange(100))
    assert len(ho) > 0 and len(tr) > 50 and set(tr).isdisjoint(ho)
    # held-out views are contiguous blocks, never single interleaved frames
    ho = np.sort(ho)
    assert np.sum(np.diff(ho) == 1) >= len(ho) // 2
