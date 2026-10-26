"""1.0 integration: top-level lazy modules, CameraGroup/Camera aliases, stations in the TOML,
board units, disconnected board graphs."""
import os
import subprocess
import sys
import warnings

import numpy as np
import pytest
import torch

import aniposelib
from aniposelib.boards import CharucoBoard, Checkerboard
from aniposelib.cameras import Camera, CameraGroup, FisheyeCamera
from aniposelib import utils

import synth

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OPTIONAL = ['vggt', 'huggingface_hub', 'onnxruntime', 'soundfile', 'av', 'yaml']


_SMOKE_CODE = r"""
import sys
import aniposelib
opt = {opt!r}
assert not [m for m in opt if m in sys.modules], [m for m in opt if m in sys.modules]
for m in ('quality', 'scale', 'sync', 'stba', 'stations', 'io', 'affine', 'matching', 'init', 'intrinsics',
          'bundle', 'detectors'):
    assert 'aniposelib.' + m not in sys.modules, m
for m in aniposelib._LAZY_SUBMODULES:
    mod = getattr(aniposelib, m)
    assert mod.__name__ == 'aniposelib.' + m
import aniposelib.init.vggt, aniposelib.detectors.deepcharuco
assert aniposelib.init.vggt is not None and aniposelib.detectors.DeepCharucoDetector is not None
assert not [m for m in opt if m in sys.modules], [m for m in opt if m in sys.modules]
assert set(aniposelib._LAZY_SUBMODULES) <= set(dir(aniposelib))
print('ok')
""".format(opt=OPTIONAL)


def _start_smoke():
    # a fresh interpreter (this one has imported everything already); started at collection so
    # its ~5 s of torch import overlaps the rest of the suite
    env = dict(os.environ, PYTHONPATH=ROOT + os.pathsep + os.environ.get('PYTHONPATH', ''),
               OMP_NUM_THREADS='1')
    return subprocess.Popen([sys.executable, '-c', _SMOKE_CODE], stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, text=True, env=env)


_SMOKE = _start_smoke()


def test_import_is_light_and_modules_lazy():
    out, err = _SMOKE.communicate(timeout=180)
    assert _SMOKE.returncode == 0, err
    assert 'ok' in out


def test_top_level_names():
    assert aniposelib.__version__ == '1.0.0'
    assert aniposelib.CameraGroup is CameraGroup and aniposelib.CharucoBoard is CharucoBoard
    with pytest.raises(AttributeError):
        aniposelib.not_a_module
    assert aniposelib.quality.detect_moves is aniposelib.stations.detect_moves


def _small_group():
    cams = []
    for i in range(4):
        a = 2 * np.pi * i / 4
        C = np.array([np.cos(a), np.sin(a), 0.3]) * 2
        z = -C / np.linalg.norm(C)
        x = np.cross([0, 0, 1.0], z); x /= np.linalg.norm(x)
        y = np.cross(z, x)
        R = np.stack([x, y, z])
        E = np.eye(4); E[:3, :3] = R; E[:3, 3] = -R @ C
        K = np.array([[800, 0, 320], [0, 800, 240], [0, 0, 1.0]])
        cams.append(Camera.from_extrinsics_mat(K, E, size=(640, 480), name='cam_{}'.format(i)))
    return CameraGroup(cams)


def test_camera_from_extrinsics_mat_and_group_from_extrinsics():
    cg = _small_group()
    E = cg.cameras[1].get_extrinsics_mat().detach().numpy()
    assert np.allclose(E[3], [0, 0, 0, 1])
    fc = FisheyeCamera.from_extrinsics_mat(np.eye(3), E, name='f')
    assert isinstance(fc, FisheyeCamera)
    Ms = np.stack([c.get_extrinsics_mat().detach().numpy() for c in cg.cameras])
    cg2 = CameraGroup.from_extrinsics(cg.get_names(), Ms, template=cg)
    assert np.allclose(cg2.project(np.zeros((1, 3))), cg.project(np.zeros((1, 3))))


def test_subset_missing_and_reorder():
    cg = _small_group()
    with pytest.raises(IndexError):
        cg.subset_cameras_names(['cam_0', 'nope'])
    assert cg.subset_cameras_names(['cam_2', 'nope', 'cam_0'], missing='drop').get_names() == ['cam_2', 'cam_0']
    r = cg.reorder(['cam_3', 'cam_1', 'cam_0', 'cam_2'])
    assert r.get_names() == ['cam_3', 'cam_1', 'cam_0', 'cam_2']
    X = np.array([[0.1, 0.0, 0.2]])
    assert np.allclose(r.project(X)[0], cg.project(X)[3])
    with pytest.raises(ValueError):
        cg.reorder(['cam_0', 'cam_1'])


def test_scale_quality_aliases():
    rig = synth.load_rig('motobs_7cam')
    cg = rig.copy()
    assert cg.rescale(2.0, units='mm') is cg and cg.metadata['units'] == 'mm'
    cg.apply_similarity(1.0, None, [1.0, 2.0, 3.0])
    cg.anchor_to(2)
    E = cg.cameras[2].get_extrinsics_mat().detach().numpy()
    assert np.allclose(E, np.eye(4), atol=1e-9)
    al = cg.align_to(rig)
    assert np.allclose(synth.camera_centers(cg), synth.camera_centers(rig), atol=1e-6 * synth.rig_scale(rig))
    assert abs(al.scale - 0.5) < 1e-9
    cg.align_world(up='z')
    p3d = synth.random_points(rig, 150, seed=1)
    p2d = synth.add_noise(synth.project(rig, p3d), 0.3, seed=2)
    rep = rig.report(p2d)
    assert rep.overall is not None and len(rep.per_camera) == len(rig.cameras)
    loo = rig.loo_reprojection(p2d)
    assert set(loo.per_camera) == set(rig.get_names())
    v = rig.cameras[0].lens_validity()
    assert v['valid']


def test_stations_roundtrip(tmp_path):
    from aniposelib.stations import Stations
    cg = _small_group()
    st = Stations({'cam_1': [(-np.inf, 10.0, 'cam_1'), (10.0, 12.0, 'skip'), (12.0, np.inf, 'cam_1@s1')]})
    cg.stations = st
    for g in (cg.copy(), cg.subset_cameras([0, 1]), cg.reorder(cg.get_names()[::-1])):
        assert g.stations == st
    fn = str(tmp_path / 'calib.toml')
    cg.dump(fn)
    cg2 = CameraGroup.load(fn)
    assert cg2.get_names() == cg.get_names() and cg2.stations == st
    assert 'stations' not in cg2.metadata
    cg3 = CameraGroup.loads(cg.dumps())
    assert cg3.stations.lookup('cam_1', [0.0, 11.0, 13.0]).tolist() == [0, -1, 1]
    # a reader that ignores the table (all top-level non-metadata tables are cameras) still works
    import toml
    d = toml.load(fn)
    assert set(d) == {'cam_0', 'cam_1', 'cam_2', 'cam_3', 'metadata'}
    # dict form in the constructor, no stations -> nothing written
    assert CameraGroup(list(cg.cameras), stations=st.to_toml_dict()).stations == st
    assert 'stations' not in toml.loads(_small_group().dumps())['metadata']


def test_board_units():
    b = CharucoBoard(5, 4, 30.0, 22.0, units='mm')
    assert b.units == 'mm' and 'units' in repr(b)
    assert CharucoBoard(5, 4, 30.0, 22.0).units is None
    assert Checkerboard(5, 4, 1.0, units='cm').units == 'cm'
    import pickle
    assert pickle.loads(pickle.dumps(b)).units == 'mm'


def test_from_board_uses_board_units():
    from aniposelib import scale
    true = synth.load_rig('fisheye_5cam')
    board = CharucoBoard(10, 7, 0.1, 0.078, units='m')
    rows = synth.board_rows(true, board, n_frames=12, seed=1)
    cg = true.copy()
    res = scale.from_board(cg, rows, board, min_pair_dist=0.15)
    assert res.accepted and res.units == 'm'
    res.apply(cg)
    assert cg.metadata['units'] == 'm'


def test_get_initial_extrinsics_disconnected():
    rig = synth.load_rig('motobs_7cam')
    rng = np.random.default_rng(0)
    C, F = 4, 12
    Ms = [c.get_extrinsics_mat().detach().numpy() for c in rig.cameras[:C]]
    rtvecs = np.full((C, F, 6), np.nan)
    for f in range(F):
        B = np.eye(4)
        B[:3, 3] = rng.normal(size=3) * 10 + synth.scene_center(rig)
        for c in range(C):
            if (c < 2) == (f % 2 == 0):  # cams {0,1} and {2,3} never see the same board
                M = Ms[c] @ B
                rtvecs[c, f] = np.r_[utils.get_rtvec(M)[0], utils.get_rtvec(M)[1]]
    with pytest.raises(ValueError):
        utils.get_initial_extrinsics(rtvecs)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        rvecs, tvecs = utils.get_initial_extrinsics(rtvecs, disconnected='components')
    assert any('disconnected' in str(x.message) for x in w)
    assert rvecs.shape == (C, 3) and np.all(np.isfinite(tvecs))
    rel = (Ms[1] @ np.linalg.inv(Ms[0]))[:3, 3]
    assert np.allclose(tvecs[1], rel, atol=1e-6 * np.linalg.norm(rel))


def test_bundle_adjust_lm_uses_group_stations(monkeypatch):
    from aniposelib import bundle
    from aniposelib.stations import Stations
    cg = _small_group()
    cg.stations = Stations({'cam_1': [(-np.inf, 1.0, 'cam_1'), (1.0, np.inf, 'cam_1@s1')]})
    seen = {}
    monkeypatch.setattr(bundle, 'bundle_adjust_lm', lambda g, o, **kw: seen.update(kw) or 0.0)
    obs = bundle.Observations(cam_idx=np.array([0, 1]), point_idx=np.array([0, 0]),
                              uv=np.zeros((2, 2)), time=np.array([0.5, 2.0]))
    cg.bundle_adjust_lm(obs)
    assert seen['stations'] is cg.stations
    seen.clear()
    cg.bundle_adjust_lm(np.zeros((4, 3, 2)))  # dense points have no times: stations not used
    assert 'stations' not in seen
