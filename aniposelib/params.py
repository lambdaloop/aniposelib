"""Which camera parameters bundle adjustment may move (minimal 1.0 precursor of ParamSpec).

Each camera keeps a set of free intrinsics (entries of its camera matrix) next to its
``n_dist`` (how many leading distortion coefficients are free). Both are enforced by backward
hooks on the parameters, so they apply to aniposelib's own solvers *and* to any optimizer a
caller builds over ``cgroup.parameters()``, and they survive ``copy``, ``deepcopy`` and pickle.

By default only fx, fy, cx and cy are free. The skew ``K[0,1]``, ``K[1,0]`` and the bottom row
of K are frozen: projection honours the skew (decision D3), but BA must not use it, or the
bottom row, as extra projective degrees of freedom.
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import torch

# name -> (row, col) in the 3x3 camera matrix
MATRIX_ENTRIES = {
    'fx': (0, 0),
    'fy': (1, 1),
    'cx': (0, 2),
    'cy': (1, 2),
    'skew': (0, 1),
}

DEFAULT_FREE_INTRINSICS = ('fx', 'fy', 'cx', 'cy')


def validate_intrinsics(names):
    """Normalise a collection of intrinsic names to a sorted-by-definition tuple."""
    if isinstance(names, str):
        names = (names,)
    names = tuple(names)
    bad = [n for n in names if n not in MATRIX_ENTRIES]
    if bad:
        raise ValueError('unknown intrinsic name(s) {}; valid names are {}'.format(
            bad, list(MATRIX_ENTRIES)))
    return tuple(n for n in MATRIX_ENTRIES if n in names)


def matrix_mask(names, dtype=torch.float64, device=None):
    """3x3 mask with 1 on the free entries of the camera matrix."""
    mask = torch.zeros((3, 3), dtype=dtype, device=device)
    for n in validate_intrinsics(names):
        mask[MATRIX_ENTRIES[n]] = 1
    return mask


@dataclass
class ParamSpec:
    """Minimal parameter spec: free intrinsics and free distortion count.

    ``intrinsics``: names from ``MATRIX_ENTRIES`` that BA may move (default fx, fy, cx, cy).
    ``n_dist``: number of free distortion coefficients, or None to keep each camera's own.

    ``spec.apply(cgroup_or_camera)`` writes the spec onto the cameras;
    ``ParamSpec.from_camera(cam)`` reads it back. Extrinsics are controlled, as before, by
    ``only_extrinsics=`` on the solvers.
    """
    intrinsics: Tuple[str, ...] = DEFAULT_FREE_INTRINSICS
    n_dist: Optional[int] = None

    def __post_init__(self):
        self.intrinsics = validate_intrinsics(self.intrinsics)

    def apply(self, target):
        cams = getattr(target, 'cameras', [target])
        for cam in cams:
            cam.set_free_intrinsics(self.intrinsics)
            if self.n_dist is not None:
                cam.set_n_dist(self.n_dist)
        return target

    @staticmethod
    def from_camera(cam):
        return ParamSpec(intrinsics=cam.get_free_intrinsics(), n_dist=cam.get_n_dist())
