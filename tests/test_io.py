"""aniposelib.io importers/exporters and aniposelib.affine (round trips on the use-case rigs)."""

import json

import numpy as np
import pytest
import torch

import synth
from aniposelib import io as aio
from aniposelib.affine import AffineCamera, affine_from_points, triangulate_affine
from aniposelib.cameras import Camera, CameraGroup, FisheyeCamera

ALL = list(synth.RIGS)


def proj(cg, X):
    return np.asarray(cg.project(X))


def same_projection(a, b, X, atol=1e-6):
    assert a.get_names() == b.get_names()
    assert [type(c) for c in a.cameras] == [type(c) for c in b.cameras]
    for ca, cb in zip(a.cameras, b.cameras):
        assert list(ca.get_size()) == list(cb.get_size())
    np.testing.assert_allclose(proj(a, X), proj(b, X), rtol=0, atol=atol)


def with_skew(cg, skew=0.7):
    cg = cg.copy()
    for c in cg.cameras:
        K = synth.to_np(c.get_camera_matrix()).copy()
        K[0, 1] = skew
        c.set_camera_matrix(K)
    return cg


@pytest.fixture(scope='module')
def rigs():
    return {r: synth.load_rig(r) for r in ALL}


@pytest.fixture(scope='module')
def points(rigs):
    return {r: synth.random_points(cg, 40, seed=3) for r, cg in rigs.items()}


# ------------------------------------------------------------------ basics

@pytest.mark.parametrize('rig', ALL)
def test_camera_from_extrinsics_mat(rig, rigs, points):
    cg = rigs[rig]
    cams = [aio.camera_from_extrinsics_mat(c.get_camera_matrix(), c.get_extrinsics_mat(), c.get_distortions(),
                                           c.get_size(), c.get_name(), fisheye=c.fisheye)
            for c in cg.cameras]
    same_projection(cg, CameraGroup(cams), points[rig], atol=1e-8)


def test_camera_from_extrinsics_mat_checks():
    E = np.eye(4)
    E[:3, :3] = np.diag([1.0, 1.0, -1.0])
    with pytest.raises(ValueError, match='improper'):
        aio.camera_from_extrinsics_mat(np.eye(3), E)
    with pytest.raises(ValueError):
        aio.camera_from_extrinsics_mat(np.eye(3), np.eye(3))
    cam = aio.camera_from_extrinsics_mat(np.eye(3), np.eye(4)[:3], offset=(3, 4), name='a')
    assert cam.get_name() == 'a' and np.allclose(synth.to_np(cam.get_offset()), [3, 4])


def test_reindex_points():
    p = np.arange(3 * 4 * 2, dtype=float).reshape(3, 4, 2)
    out = aio.reindex_points(p, ['a', 'b', 'c'], ['c', 'x', 'a'])
    np.testing.assert_array_equal(out[0], p[2])
    np.testing.assert_array_equal(out[2], p[0])
    assert np.all(np.isnan(out[1]))
    with pytest.raises(KeyError):
        aio.reindex_points(p, ['a', 'b', 'c'], ['c', 'x'], missing='raise')
    with pytest.raises(ValueError):
        aio.reindex_points(p, ['a', 'b'], ['a'])
    t = aio.reindex_points(torch.as_tensor(p), ['a', 'b', 'c'], ['b'])
    assert isinstance(t, torch.Tensor) and torch.equal(t[0], torch.as_tensor(p[1]))
    cg = CameraGroup.from_names(['a', 'b', 'c'])
    np.testing.assert_array_equal(aio.reindex_points(p, cg, ['b', 'a']), p[[1, 0]])


# ------------------------------------------------------------------ MATLAB

def test_matlab_conventions():
    """MATLAB: x_img(1-based) = [X 1] [R; t] IntrinsicMatrix (row vectors)."""
    rng = np.random.default_rng(0)
    K = np.array([[800.0, 1.5, 321.0], [0, 790.0, 241.0], [0, 0, 1]])
    ang = rng.normal(size=3) * 0.3
    import cv2
    R = cv2.Rodrigues(ang)[0]
    t = np.array([0.1, -0.2, 3.0])
    ml = dict(IntrinsicMatrix=K.T, RadialDistortion=[0.0, 0.0], TangentialDistortion=[0.0, 0.0],
              ImageSize=[480, 640], RotationMatrix=R.T, TranslationVector=t)
    X = rng.normal(size=(20, 3)) * 0.3
    h = (np.c_[X, np.ones(len(X))] @ np.vstack([R.T, t])) @ K.T
    uv_matlab = h[:, :2] / h[:, 2:]
    cg = aio.from_matlab(ml, names=['m'])
    assert cg.cameras[0].get_size() == [640, 480]
    np.testing.assert_allclose(proj(cg, X)[0], uv_matlab - 1, atol=1e-9)
    cg0 = aio.from_matlab(ml, names=['m'], one_based=False)
    np.testing.assert_allclose(proj(cg0, X)[0], uv_matlab, atol=1e-9)
    # FocalLength / PrincipalPoint / Skew form, nested Intrinsics
    ml2 = dict(Intrinsics=dict(FocalLength=[800.0, 790.0], PrincipalPoint=[321.0, 241.0], Skew=1.5,
                               ImageSize=[480, 640]), RotationMatrix=R.T, TranslationVector=t)
    np.testing.assert_allclose(proj(aio.from_matlab(ml2), X), proj(cg, X), atol=1e-9)


@pytest.mark.parametrize('rig', ['allen_7cam', 'motobs_7cam'])
def test_matlab_roundtrip(rig, rigs, points, tmp_path):
    cg = with_skew(rigs[rig])
    ml = aio.to_matlab(cg)
    same_projection(cg, aio.from_matlab(ml), points[rig])
    # .mat (scipy) and .json files
    aio.to_matlab(cg, tmp_path / 'c.mat')
    same_projection(cg, aio.from_matlab(str(tmp_path / 'c.mat')), points[rig])
    aio.to_matlab(cg, tmp_path / 'c.json')
    same_projection(cg, aio.from_matlab(tmp_path / 'c.json'), points[rig])
    # Emory-style export keys (convert_calibration.py)
    emory = [dict(name=c['Name'], intrinsic_matrix_matlab=c['IntrinsicMatrix'], radial=c['RadialDistortion'],
                  tangential=c['TangentialDistortion'], image_size_hw=c['ImageSize'],
                  rotation_matlab=c['RotationMatrix'], translation_matlab=c['TranslationVector']) for c in ml]
    same_projection(cg, aio.from_matlab(emory), points[rig])


def test_matlab_stereo(rigs, points):
    cg = rigs['allen_7cam'].subset_cameras([0, 1])
    # put camera 0 at the origin
    E0 = synth.to_np(cg.cameras[0].get_extrinsics_mat())
    E1 = synth.to_np(cg.cameras[1].get_extrinsics_mat())
    rel = E1 @ np.linalg.inv(E0)
    ml = aio.to_matlab(cg)
    stereo = dict(CameraParameters1={k: v for k, v in ml[0].items() if k not in ('RotationMatrix', 'TranslationVector')},
                  CameraParameters2={k: v for k, v in ml[1].items() if k not in ('RotationMatrix', 'TranslationVector')},
                  RotationOfCamera2=rel[:3, :3].T, TranslationOfCamera2=rel[:3, 3])
    out = aio.from_matlab(stereo, names=['l', 'r'])
    X = points['allen_7cam']
    Xc = X @ E0[:3, :3].T + E0[:3, 3]  # world -> camera-1 frame
    np.testing.assert_allclose(proj(out, Xc), proj(cg, X), atol=1e-6)
    # PoseCamera2 (rigidtform3d: camera 2 pose in camera 1 frame) gives the same
    pose = np.linalg.inv(rel)
    stereo2 = dict(stereo)
    del stereo2['RotationOfCamera2'], stereo2['TranslationOfCamera2']
    stereo2['PoseCamera2'] = dict(R=pose[:3, :3], Translation=pose[:3, 3])
    np.testing.assert_allclose(proj(aio.from_matlab(stereo2), Xc), proj(cg, X), atol=1e-6)


def test_matlab_rejects_fisheye(rigs):
    with pytest.raises(ValueError, match='fisheye'):
        aio.to_matlab(rigs['fisheye_5cam'])


# ------------------------------------------------------------------ OpenCV YAML

@pytest.mark.parametrize('rig', ['allen_7cam', 'fisheye_5cam'])
def test_opencv_yaml_roundtrip(rig, rigs, points, tmp_path):
    cg = with_skew(rigs[rig]) if rig != 'fisheye_5cam' else rigs[rig]
    files = aio.to_opencv_yaml(cg, tmp_path / 'yaml')
    assert len(files) == len(cg.cameras)
    same_projection(cg, aio.from_opencv_yaml(files), points[rig])
    same_projection(cg, aio.from_opencv_yaml(str(tmp_path / 'yaml')), points[rig])
    # cv2 reads what we write
    import cv2
    fs = cv2.FileStorage(files[0], cv2.FILE_STORAGE_READ)
    np.testing.assert_allclose(fs.getNode('camera_matrix').mat(), synth.to_np(cg.cameras[0].get_camera_matrix()))
    assert int(fs.getNode('image_width').real()) == cg.cameras[0].get_size()[0]
    fs.release()


def test_opencv_yaml_bare_ints_and_matlab_convention(tmp_path):
    """johnson_mouse-style file: MATLAB-transposed matrices with bare integers under dt: d."""
    K = np.array([[1000.0, 0, 640], [0, 1000, 512], [0, 0, 1]])
    import cv2
    R = cv2.Rodrigues(np.array([0.1, 0.2, -0.3]))[0]
    t = np.array([10.0, -5.0, 400.0])
    txt = ('%YAML:1.0\n---\nimage_width: 0\nimage_height: 0\n'
           'intrinsicMatrix: !!opencv-matrix\n   rows: 3\n   cols: 3\n   dt: d\n'
           '   data: [ 1000, 0, 0, 0, 1000, 0,\n       641, 513, 1 ]\n'
           'distortionCoefficients: !!opencv-matrix\n   rows: 1\n   cols: 5\n   dt: d\n'
           '   data: [ 0, 0, 0, 0, 0 ]\n'
           'R: !!opencv-matrix\n   rows: 3\n   cols: 3\n   dt: d\n   data: [ '
           + ', '.join('%.17g' % v for v in R.T.ravel()) + ' ]\n'
           'T: !!opencv-matrix\n   rows: 3\n   cols: 1\n   dt: d\n   data: [ 10, -5, 400 ]\n')
    f = tmp_path / 'cam1.yaml'
    f.write_text(txt)
    d = aio.read_opencv_yaml(str(f))
    assert d['intrinsicMatrix'][2, 0] == 641.0 and d['image_width'] == 0
    cg = aio.from_opencv_yaml(str(f))
    cam = cg.cameras[0]
    assert cam.get_name() == 'cam1' and cam.get_size() is None
    np.testing.assert_allclose(synth.to_np(cam.get_camera_matrix()), K, atol=1e-12)  # 1-based shift
    np.testing.assert_allclose(synth.to_np(cam.get_extrinsics_mat())[:3, :3], R, atol=1e-9)
    np.testing.assert_allclose(synth.to_np(cam.get_translation()), t)
    cg0 = aio.from_opencv_yaml(str(f), one_based=False)
    assert synth.to_np(cg0.cameras[0].get_camera_matrix())[0, 2] == 641.0


def test_opencv_yaml_parser_misc():
    txt = ('%YAML:1.0\n---\ncamera_name: "cam 2"\nnested:\n   a: 1\n   b: !!opencv-matrix\n'
           '      rows: 1\n      cols: 2\n      dt: f\n      data: [ 1.5e-3, -.Nan ]\nflag: 1\n')
    d = aio.read_opencv_yaml(txt)
    assert d['camera_name'] == 'cam 2' and d['flag'] == 1 and d['nested']['a'] == 1
    assert d['nested']['b'].shape == (1, 2) and np.isnan(d['nested']['b'][0, 1])


# ------------------------------------------------------------------ projection matrix / DLT

def test_projection_matrix_roundtrip(rigs, points):
    cg = with_skew(rigs['motobs_7cam'])
    for c in cg.cameras:
        c.zero_distortions()
    X = points['motobs_7cam']
    for c in cg.cameras:
        P = aio.to_projection_matrix(c)
        for s in (1.0, -3.5):  # scale and sign of P do not matter
            c2 = aio.from_projection_matrix(s * P, size=c.get_size(), name=c.get_name())
            np.testing.assert_allclose(synth.to_np(c2.get_camera_matrix()),
                                       synth.to_np(c.get_camera_matrix()), rtol=1e-9, atol=1e-7)
            np.testing.assert_allclose(np.asarray(c2.project(X)), np.asarray(c.project(X)), atol=1e-6)
        L = aio.to_dlt(c)
        assert L.shape == (11,)
        c3 = aio.from_dlt(L)
        np.testing.assert_allclose(np.asarray(c3.project(X)), np.asarray(c.project(X)), atol=1e-6)
        h = np.c_[X, np.ones(len(X))] @ np.r_[L, 1].reshape(3, 4).T
        np.testing.assert_allclose(h[:, :2] / h[:, 2:], np.asarray(c.project(X)), atol=1e-6)


def test_projection_matrix_singular():
    with pytest.raises(ValueError):
        aio.decompose_projection(np.zeros((3, 4)))


# ------------------------------------------------------------------ tracktail metadata.yaml

@pytest.mark.parametrize('rig', ['allen_12cam', 'fisheye_5cam'])
def test_metadata_yaml_roundtrip(rig, rigs, points, tmp_path):
    cg = rigs[rig].copy()
    cg.cameras[0].set_offset((12.0, 7.0))
    d = aio.to_metadata_yaml(cg, tmp_path / 'metadata.yaml')
    assert d['cam_type'] == ('fisheye' if rig == 'fisheye_5cam' else 'pinhole')
    assert d['offset_dict'][cg.cameras[0].get_name()] == [12.0, 7.0]
    np.testing.assert_allclose(d['extrinsic_matrices'][cg.cameras[1].get_name()],
                               synth.to_np(cg.cameras[1].get_extrinsics_mat()))
    back = aio.from_metadata_yaml(str(tmp_path / 'metadata.yaml'), order='file')
    same_projection(cg, back, points[rig])
    # the written file is plain JSON-flow YAML (PyYAML reads it too, if installed)
    try:
        import yaml
    except ImportError:
        yaml = None
    if yaml is not None:
        same_projection(cg, aio.from_metadata_yaml(yaml.safe_load(open(tmp_path / 'metadata.yaml')), order='file'),
                        points[rig])


def test_metadata_yaml_tracktail_order_and_moving(rigs):
    cg = rigs['allen_12cam'].subset_cameras(list(range(3)))
    cg.set_names(['10', '2', '1'])
    T = 4
    ext = {n: np.repeat(synth.to_np(c.get_extrinsics_mat())[None], T, 0) for n, c in zip(cg.get_names(), cg.cameras)}
    ext['2'][1:, :3, 3] += 5.0
    d = aio.to_metadata_yaml(cg, moving={'2': ext['2']})
    assert d['moving_cams'] is True and np.shape(d['extrinsic_matrices']['10']) == (T, 4, 4)
    with pytest.warns(RuntimeWarning, match='moving_cams'):
        back = aio.from_metadata_yaml(d)
    assert back.get_names() == ['1', '2', '10']   # tracktail: numeric when all digits
    back, moving = aio.from_metadata_yaml(d, return_moving=True, order='file')
    assert back.get_names() == ['10', '2', '1'] and back.cameras[1].extra['moving'] is True
    np.testing.assert_allclose(moving['2'], ext['2'])
    np.testing.assert_allclose(synth.to_np(back.cameras[1].get_extrinsics_mat()), ext['2'][0], atol=1e-9)
    bad = dict(d)
    bad['extrinsic_matrices'] = dict(d['extrinsic_matrices'])
    bad['extrinsic_matrices']['1'] = ext['1'][0].tolist()
    with pytest.raises(ValueError, match='moving'):
        aio.from_metadata_yaml(bad)


def test_metadata_yaml_mixed_models_raise():
    cg = CameraGroup([Camera(name='a', size=(10, 10)), FisheyeCamera(name='b', size=(10, 10))])
    with pytest.raises(ValueError, match='cam_type'):
        aio.to_metadata_yaml(cg)


# ------------------------------------------------------------------ JSON rig

@pytest.mark.parametrize('rig', ['outdoor_9cam', 'fisheye_5cam'])
def test_json_rig_roundtrip(rig, rigs, points, tmp_path):
    cg = rigs[rig] if rig == 'fisheye_5cam' else with_skew(rigs[rig])
    d = aio.to_json_rig(cg, tmp_path / 'rig.json', session='s1', extra={'metric_scale': 1.0})
    json.dumps(d)
    assert d['session'] == 's1' and d['metric_scale'] == 1.0
    same_projection(cg, aio.from_json_rig(str(tmp_path / 'rig.json')), points[rig])
    # x_cam = R X + t
    c = cg.cameras[0]
    st = d['stations'][c.get_name()]
    E = synth.to_np(c.get_extrinsics_mat())
    np.testing.assert_allclose(st['R'], E[:3, :3])
    np.testing.assert_allclose(st['t'], E[:3, 3])


def test_json_rig_disco_layout(rigs, points):
    """A disco pose_ba_core rig (f/aspect/cx/cy + k's, stations cam@N, no full K)."""
    cg = rigs['fisheye_5cam']
    d = aio.to_json_rig(cg)
    for it in d['intrinsics'].values():
        del it['K'], it['dist'], it['skew']
    st = {}
    for n, s in d['stations'].items():
        s = {k: v for k, v in s.items() if k not in ('K', 'camera')}
        st[n + '@1'] = s
        s2 = dict(s)
        s2['t'] = (np.asarray(s['t']) + 1.0).tolist()
        st[n + '@2'] = s2
    disco = {'intrinsics': d['intrinsics'], 'stations': st}
    with pytest.warns(RuntimeWarning, match='more stations'):
        out = aio.from_json_rig(disco)
    same_projection(cg, out, points['fisheye_5cam'], atol=1e-5)
    allst = aio.from_json_rig(disco, stations='all')
    assert len(allst.cameras) == 2 * len(cg.cameras) and allst.get_names()[1].endswith('@2')


# ------------------------------------------------------------------ affine cameras

def _affine_rig(seed=0, n=4):
    rng = np.random.default_rng(seed)
    cams = []
    for i in range(n):
        A = rng.normal(size=(2, 3)) * 50
        b = rng.uniform(200, 400, size=2)
        cams.append(AffineCamera(A, b, size=(640, 480), name='a{}'.format(i)))
    return cams


def test_affine_projection_and_triangulation():
    cams = _affine_rig()
    rng = np.random.default_rng(1)
    X = rng.normal(size=(30, 3))
    uv = np.stack([c.project(X) for c in cams])
    uv[0, :5] = np.nan
    Xh = triangulate_affine(cams, uv)
    np.testing.assert_allclose(Xh, X, atol=1e-9)
    tX = triangulate_affine(cams, torch.as_tensor(uv))
    assert isinstance(tX, torch.Tensor)
    one = uv.copy()
    one[1:, 0] = np.nan
    assert np.all(np.isnan(triangulate_affine(cams, one)[0]))
    fit = affine_from_points(X, uv[1])
    np.testing.assert_allclose(synth.to_np(fit.A), synth.to_np(cams[1].A), atol=1e-8)
    back = AffineCamera.from_dict(json.loads(json.dumps(cams[2].get_dict())))
    np.testing.assert_allclose(back.project(X), cams[2].project(X))


def test_affine_from_dlt_and_pinhole_approximation():
    cam = _affine_rig(seed=2, n=1)[0]
    P = cam.get_projection_matrix()
    out = aio.from_projection_matrix(2.0 * P)
    assert isinstance(out, AffineCamera)
    L = aio.to_dlt(cam)
    assert np.allclose(L[8:11], 0) and isinstance(aio.from_dlt(L), AffineCamera)
    rng = np.random.default_rng(3)
    X = rng.normal(size=(50, 3))
    np.testing.assert_allclose(aio.from_dlt(L).project(X), cam.project(X), atol=1e-9)
    pin = cam.to_pinhole(X, max_error_px=0.5)
    err = np.linalg.norm(np.asarray(pin.project(X)) - cam.project(X), axis=1)
    assert err.max() < 0.5
    assert abs(synth.to_np(pin.get_camera_matrix())[0, 1]) > 0  # skew kept
