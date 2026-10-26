"""Every use-case rig loads, round-trips through TOML, and subsets by name."""

import numpy as np
import pytest

import synth
from aniposelib.cameras import CameraGroup

ALL = list(synth.RIGS)


def params(cg):
    return [(c.get_name(), synth.to_np(c.get_camera_matrix()), synth.to_np(c.get_distortions()),
             synth.to_np(c.get_rotation()), synth.to_np(c.get_translation()), tuple(c.get_size()))
            for c in cg.cameras]


def same(a, b):
    assert len(a) == len(b)
    for x, y in zip(a, b):
        assert x[0] == y[0] and x[5] == y[5]
        for u, v in zip(x[1:5], y[1:5]):
            np.testing.assert_allclose(u, v, rtol=0, atol=1e-9)


@pytest.mark.parametrize('rig', ALL)
def test_rig_loads_and_projects(rig):
    cg = synth.load_rig(rig)
    p3d = synth.random_points(cg, 50, seed=1)
    p2d = synth.project(cg, p3d)
    assert p2d.shape == (len(cg.cameras), 50, 2)
    assert np.all(synth.n_views(p2d) >= 2)


@pytest.mark.parametrize('rig', [r for r in ALL if len(synth.load_rig(r).cameras) <= 10])
def test_dump_load_roundtrip(rig, tmp_path):
    cg = synth.load_rig(rig)
    cg.dump(tmp_path / 'c.toml')
    same(params(cg), params(CameraGroup.load(tmp_path / 'c.toml')))


@pytest.mark.parametrize('rig', ['allen_12cam', 'outdoor_12cam'])
def test_dump_load_roundtrip_many_cameras(rig, tmp_path):
    cg = synth.load_rig(rig)
    cg.dump(tmp_path / 'c.toml')
    same(params(cg), params(CameraGroup.load(tmp_path / 'c.toml')))


def test_subset_by_name_matches_projection():
    cg = synth.load_rig('outdoor_12cam')
    names = cg.get_names()[::3]
    sub = cg.subset_cameras_names(names)
    assert sub.get_names() == names
    p3d = synth.random_points(cg, 20, seed=2)
    full = synth.project(cg, p3d)
    idx = [cg.get_names().index(n) for n in names]
    np.testing.assert_allclose(synth.project(sub, p3d), full[idx])
