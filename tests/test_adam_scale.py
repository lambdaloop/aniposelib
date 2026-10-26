"""The Adam BA (method='adam') must not depend on the rig's length units (mm, m, unit-scale SfM)."""

from copy import deepcopy

import numpy as np
import pytest

import synth


def scaled_problem(scale, rig='motobs_7cam', n=400):
    base = synth.load_rig(rig)
    f = scale / synth.rig_scale(base)
    p3d = synth.trajectory(base, n, seed=2) * f
    true = deepcopy(base)
    true.set_translations(synth.to_np(base.get_translations()) * f)
    p2d = synth.add_noise(synth.project(true, p3d), 0.5, seed=3)
    pert = synth.perturb_extrinsics(true, rot_deg=0.5, trans_frac=0.01, seed=4)
    return true, pert, p2d


def run(scale, **kw):
    true, cg, p2d = scaled_problem(scale)
    before = synth.center_error(cg, true).max()
    cg.bundle_adjust(p2d, method='adam', only_extrinsics=True, max_nfev=150, verbose=False,
                     rng=0, **kw)
    return before, synth.center_error(cg, true).max()


@pytest.mark.parametrize('scale', [1e-3, 1.0, 1e3])
def test_adam_converges_at_any_rig_scale(scale):
    before, after = run(scale)
    assert after < before / 3


def test_adam_is_scale_invariant():
    errs = [run(s)[1] for s in (1e-3, 1e3)]
    assert abs(errs[0] - errs[1]) < 1e-3


def test_adam_legacy_steps_still_available():
    """trans_scale=None keeps the 0.9 absolute steps (which barely move a tiny rig)."""
    before, after = run(1e-3, trans_scale=None)
    assert after > before / 3
