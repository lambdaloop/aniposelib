"""aniposelib.init.vggt: Sim(3) aggregation, scale normalisation, intrinsics restoration,
pairwise relative poses (pure numpy, fast) and an optional real-VGGT smoke test (slow, only
if `vggt` is importable and the weights are already in the local HF cache; never downloads)."""

import importlib.util

import numpy as np
import pytest

import synth
from aniposelib.init import vggt
from aniposelib.init._common import (centers, get_extrinsics, relative_to, rodrigues,
                                     rotation_angle_deg, transform_extrinsics, umeyama)


def _rand_rot(rng, deg):
    v = rng.normal(size=3)
    return rodrigues(v / np.linalg.norm(v) * np.radians(deg))


def _templates(Ms, rng, n=8, rot_noise_deg=0.3, cen_noise=0.004, outlier=True, bad_cam=True):
    """VGGT-like templates: per-camera noise, then a random similarity, then VGGT's gauge
    (camera 0 = identity) and an arbitrary scale. Template 2 is garbage (if outlier) and
    template 4 has one camera badly wrong (if bad_cam)."""
    scale = np.linalg.norm(centers(Ms) - centers(Ms).mean(0), axis=1).mean()
    out = []
    for k in range(n):
        Mk = Ms.copy()
        for c in range(len(Ms)):
            R = _rand_rot(rng, rot_noise_deg) @ Ms[c, :3, :3]
            C = centers(Ms[c]) + rng.normal(scale=cen_noise * scale, size=3)
            Mk[c, :3, :3] = R
            Mk[c, :3, 3] = -R @ C
        if outlier and k == 2:
            for c in range(len(Ms)):
                R = _rand_rot(rng, 180 * rng.uniform())
                Mk[c, :3, :3] = R
                Mk[c, :3, 3] = rng.normal(scale=scale, size=3)
        if bad_cam and k == 4:
            Mk[3, :3, :3] = _rand_rot(rng, 40) @ Mk[3, :3, :3]
            Mk[3, :3, 3] += scale * 0.5
        Mk = transform_extrinsics(Mk, rng.uniform(0.2, 5), _rand_rot(rng, 180 * rng.uniform()),
                                  rng.normal(size=3) * scale)
        Mk = relative_to(Mk, 0)
        Mk[:, :3, 3] *= rng.uniform(0.5, 2.0)
        out.append(Mk)
    return np.stack(out)


def _rot_errors(Ms_est, Ms_true):
    """Rotation error (deg) per camera after aligning the estimate to the truth by Sim(3)."""
    s, R, t = umeyama(centers(Ms_est), centers(Ms_true))
    al = transform_extrinsics(Ms_est, s, R, t)
    return rotation_angle_deg(al[:, :3, :3], Ms_true[:, :3, :3])


@pytest.fixture(scope='module')
def rig():
    cg = synth.load_rig('motobs_7cam')
    return cg, get_extrinsics(cg)


@pytest.mark.parametrize('rotation_mean', ['chordal', 'robust', 'median'])
def test_aggregate_recovers_rig(rig, rotation_mean):
    cg, Ms = rig
    E = _templates(Ms, np.random.default_rng(1))
    agg = vggt.aggregate(E, names=cg.get_names(), rotation_mean=rotation_mean)
    assert agg.Ms.shape == (7, 4, 4) and isinstance(agg.Ms, np.ndarray)
    assert 2 in agg.stats['rejected_templates']
    est = vggt.from_extrinsics(cg.get_names(), agg.Ms, template=cg)
    assert synth.center_error(est, cg).max() < 1e-2
    assert _rot_errors(agg.Ms, Ms).max() < 0.5
    # output gauge: unit mean centre distance
    C = centers(agg.Ms)
    assert np.isclose(np.linalg.norm(C - C.mean(0), axis=1).mean(), 1.0)
    assert agg.stats['raw_scale_rel_spread'] > 0.1        # templates had random scales
    assert np.all(agg.kept >= 6) and np.all(agg.kept <= 7)
    assert np.all(agg.stats['rot_disp_deg'] < 1.0)
    d = agg.to_dict()
    assert d['names'] == cg.get_names() and len(d['Ms']) == 7


def test_aggregate_explicit_ref_and_first(rig):
    cg, Ms = rig
    E = _templates(Ms, np.random.default_rng(2), outlier=False, bad_cam=False)
    agg = vggt.aggregate(E, ref=0)
    assert agg.ref == 0
    assert synth.center_error(vggt.from_extrinsics(cg.get_names(), agg.Ms, template=cg), cg).max() < 1e-2
    # 'first' (EvoVision/motor gauge) also works on clean templates
    agg1 = vggt.aggregate(E, align='first', rotation_mean='robust')
    assert np.allclose(agg1.Ms[0], np.eye(4), atol=1e-2)
    assert synth.center_error(vggt.from_extrinsics(cg.get_names(), agg1.Ms, template=cg), cg).max() < 2e-2


def test_normalize_template_scale(rig):
    _, Ms = rig
    E = np.stack([Ms, transform_extrinsics(Ms, 3.0, np.eye(3), np.zeros(3))])
    En, s = vggt.normalize_template_scale(E)
    assert np.isclose(s[1] / s[0], 3.0)
    C = centers(En)
    assert np.allclose(np.linalg.norm(C - C.mean(1, keepdims=True), axis=-1).mean(1), 1.0)
    assert np.allclose(En[0, :, :3, :3], Ms[:, :3, :3])
    M1, s1 = vggt.normalize_template_scale(Ms)
    assert M1.shape == (7, 4, 4) and np.isclose(s1, s[0])


def test_chordal_mean_symmetric():
    from aniposelib.init._common import chordal_mean
    rng = np.random.default_rng(0)
    R0 = _rand_rot(rng, 70)
    Rs = []
    for _ in range(5):
        w = rng.normal(size=3) * 0.2
        Rs += [R0 @ rodrigues(w), R0 @ rodrigues(-w)]
    assert rotation_angle_deg(chordal_mean(Rs), R0) < 1e-4  # arccos precision


def test_restore_intrinsics_pad_roundtrip():
    # disco test numbers: 1920x1080 letterboxed into 518x518
    Km = np.array([[259.0, 0, 259.0], [0, 259.0, 259.0], [0, 0, 1]])
    K = vggt.restore_padded_intrinsics(Km, (1920, 1080), (518, 518))
    assert np.isclose(K[0, 0], 259.0 * 1920 / 518) and np.isclose(K[1, 1], 259.0 * 1080 / 294)
    assert np.isclose(K[0, 2], 960.0) and np.isclose(K[1, 2], 540.0)
    # hand-made resize + pad: project a point with true K, map pixels by resize+pad, and check
    # that K_model = A K reproduces model pixels and restore_intrinsics inverts it
    Ktrue = np.array([[1400.0, 0, 950], [0, 1380.0, 530], [0, 0, 1]])
    s, nh = 518 / 1920, 294
    pad_top = (518 - nh) // 2
    A_hand = np.array([[s, 0, 0], [0, nh / 1080, pad_top], [0, 0, 1]])
    A, out, resized = vggt.preprocess_transform((1920, 1080), mode='pad')
    assert out == (518, 518) and resized == (518, 294) and np.allclose(A, A_hand)
    X = np.array([0.3, -0.2, 2.0])
    x = Ktrue @ X
    xm = A_hand @ x
    Km = A @ Ktrue
    assert np.allclose(Km @ X / (Km @ X)[2], xm / xm[2])
    assert np.allclose(vggt.restore_intrinsics(Km, A), Ktrue)
    # crop mode, portrait: width 518, height 686 cropped to 518 (offset -84)
    A, out, resized = vggt.preprocess_transform((480, 640), mode='crop')
    assert resized == (518, 686) and out == (518, 518) and np.isclose(A[1, 2], -84)
    assert np.allclose(vggt.restore_intrinsics(A @ Ktrue, A), Ktrue)
    # batched [T,C,3,3]
    As = np.broadcast_to(A, (2, 3, 3, 3))
    assert np.allclose(vggt.restore_intrinsics(As @ Ktrue, As), Ktrue)


@pytest.mark.parametrize('mode,size', [('pad', (640, 360)), ('pad', (360, 640)), ('crop', (640, 360)),
                                       ('crop', (300, 640))])
def test_preprocess_image_matches_transform(mode, size):
    """A bright dot drawn in an in-memory frame lands where A says it should."""
    w, h = size
    img = np.zeros((h, w, 3), np.uint8)
    px = np.array([0.62 * w, 0.43 * h])
    yy, xx = np.mgrid[:h, :w]
    img[(xx + 0.5 - px[0]) ** 2 + (yy + 0.5 - px[1]) ** 2 < 36] = 255
    x, A, osz = vggt.preprocess_image(img, mode=mode)
    assert osz == (w, h) and x.dtype == np.float32 and x.shape[0] == 3
    if mode == 'pad':
        assert x.shape[1:] == (518, 518)
    exp = (A @ np.r_[px, 1])[:2]
    g = x[0].copy()
    if mode == 'pad':  # padding is white: mask it out
        A_, (ow, oh), (nw, nh) = vggt.preprocess_transform((w, h), mode=mode)
        mask = np.zeros_like(g, bool)
        mask[int(A[1, 2]):int(A[1, 2]) + nh, int(A[0, 2]):int(A[0, 2]) + nw] = True
        g[~mask] = 0
    g = np.clip(g, 0, None) ** 2
    yy, xx = np.mgrid[:g.shape[0], :g.shape[1]]
    got = np.array([(g * (xx + 0.5)).sum(), (g * (yy + 0.5)).sum()]) / g.sum()
    assert np.allclose(got, exp, atol=0.75)


def test_pairwise_rotations(rig):
    _, Ms = rig
    E = _templates(Ms, np.random.default_rng(3), n=6)
    pw = vggt.pairwise_rotations(E, pairs=[(0, 1), (2, 5), (3, 6)])
    assert set(pw) == {(0, 1), (2, 5), (3, 6)}
    for (i, j), e in pw.items():
        Rt = Ms[j, :3, :3] @ Ms[i, :3, :3].T
        rel = Ms[j] @ np.linalg.inv(Ms[i])
        d = rel[:3, 3] / np.linalg.norm(rel[:3, 3])
        assert rotation_angle_deg(e['R'], Rt) < 0.5
        assert np.degrees(np.arccos(np.clip(e['t_dir'] @ d, -1, 1))) < 2.0
        assert e['n_templates'] == 6 and e['rot_disp_deg'] < 1.0 and e['scale'] > 0
    assert len(vggt.pairwise_rotations(E)) == 21
    assert vggt.pairwise_rotations(E[:2]) == {}


def test_to_cameragroup_from_prediction(rig):
    cg, Ms = rig
    E = _templates(Ms, np.random.default_rng(4), n=5, outlier=False, bad_cam=False)
    K = np.stack([synth.to_np(c.get_camera_matrix()) for c in cg.cameras])
    sizes = np.array([c.get_size() for c in cg.cameras])
    Kt = K[None] * np.random.default_rng(0).uniform(0.97, 1.03, size=(5, 7, 1, 1))
    Kt[..., 2, 2] = 1
    pred = vggt.VGGTPrediction(extrinsics=E, intrinsics=Kt, sizes=sizes, names=cg.get_names())
    out = vggt.to_cameragroup(pred)
    assert out.get_names() == cg.get_names()
    assert np.allclose(synth.to_np(out.cameras[1].get_camera_matrix()), np.median(Kt[:, 1], 0))
    assert tuple(out.cameras[1].get_size()) == tuple(sizes[1])
    assert synth.center_error(out, cg).max() < 1e-2
    assert 'extrinsics' in pred.to_dict()


def test_module_imports_without_vggt():
    # the module must not import vggt/huggingface_hub at import time
    import sys
    assert 'vggt.models' not in sys.modules or importlib.util.find_spec('vggt') is not None
    with pytest.raises(ValueError):
        vggt.aggregate(np.zeros((2, 3, 4)))


def _vggt_ready():
    if importlib.util.find_spec('vggt') is None or importlib.util.find_spec('huggingface_hub') is None:
        return False
    return vggt.weights_cached(vggt.DEFAULT_MODEL)


@pytest.mark.slow
@pytest.mark.skipif(not _vggt_ready(), reason='vggt not installed or VGGT-1B weights not in the local HF cache')
def test_predict_real_vggt(rig):
    """Smoke test on rendered views of a textured plane (never downloads)."""
    import cv2
    rng = np.random.default_rng(0)
    tex = cv2.GaussianBlur((rng.uniform(size=(600, 600, 3)) * 255).astype(np.uint8), (0, 0), 3)
    tex = cv2.resize(tex, (1200, 1200), interpolation=cv2.INTER_NEAREST)
    frames = []
    for k in range(2):
        frs = []
        for c in range(3):
            H = np.array([[1, 0.05 * c, 30.0 * c + 10 * k], [0.03 * c, 1, 20.0 * c], [1e-4 * c, 0, 1]])
            frs.append(cv2.warpPerspective(tex, H, (640, 360)))
        frames.append(frs)
    pred = vggt.predict(frames, local_files_only=True, names=['a', 'b', 'c'])
    assert pred.extrinsics.shape == (2, 3, 4, 4) and pred.intrinsics.shape == (2, 3, 3, 3)
    assert np.isfinite(pred.extrinsics).all()
    assert np.allclose(pred.intrinsics[..., 0, 2], 320, atol=1) and np.allclose(pred.intrinsics[..., 1, 2], 180, atol=1)
    agg = vggt.aggregate(pred)
    assert agg.Ms.shape == (3, 4, 4)


def test_predict_with_fake_model(monkeypatch):
    """predict() plumbing (preprocessing, dtype, K restoration, freeing) with a stand-in
    network and pose decoder, so it runs without vggt."""
    import sys
    import types

    import torch

    class FakeVGGT(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.p = torch.nn.Parameter(torch.zeros(1))

        def aggregator(self, x):
            assert x.ndim == 5 and x.shape[2] == 3
            return x, None

        def camera_head(self, x):
            S = x.shape[1]
            enc = torch.zeros(1, S, 9)
            enc[..., 0] = torch.arange(S, dtype=torch.float32)  # camera s at x = s
            enc[..., 3] = 1.0
            enc[..., 7:] = 1.0
            return [enc]

    def decode(enc, hw):
        H, W = hw
        S = enc.shape[1]
        extr = torch.zeros(1, S, 3, 4)
        extr[..., :3, :3] = torch.eye(3)
        extr[..., 0, 3] = -enc[..., 0]
        K = torch.zeros(1, S, 3, 3)
        K[..., 0, 0] = K[..., 1, 1] = W / 2
        K[..., 0, 2], K[..., 1, 2], K[..., 2, 2] = W / 2, H / 2, 1
        return extr, K

    mod = types.ModuleType('vggt.utils.pose_enc')
    mod.pose_encoding_to_extri_intri = decode
    for name in ('vggt', 'vggt.utils'):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    monkeypatch.setitem(sys.modules, 'vggt.utils.pose_enc', mod)

    rng = np.random.default_rng(0)
    frames = [[rng.integers(0, 255, size=(360, 640, 3), dtype=np.uint8),
               rng.integers(0, 255, size=(640, 360, 3), dtype=np.uint8),     # portrait camera
               rng.integers(0, 255, size=(360, 640, 3), dtype=np.uint8)] for _ in range(2)]
    for mode in ('crop', 'pad'):
        pred = vggt.predict(frames, model=FakeVGGT(), device='cpu', names=['a', 'b', 'c'], mode=mode)
        assert pred.extrinsics.shape == (2, 3, 4, 4) and pred.stats['dtype'] == 'float32'
        assert np.allclose(centers(pred.extrinsics[0])[:, 0], [0, 1, 2])
        assert np.allclose(pred.sizes, [[640, 360], [360, 640], [640, 360]])
        # principal point back at the centre of every original image (incl. batch padding)
        assert np.allclose(pred.intrinsics[0, :, 0, 2], pred.sizes[:, 0] / 2)
        assert np.allclose(pred.intrinsics[0, :, 1, 2], pred.sizes[:, 1] / 2)
        assert pred.intrinsics[0, 0, 0, 0] > 0
