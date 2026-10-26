"""Which camera parameters bundle adjustment may move: the parameter spec (1.0, plan 01 B14 + D1).

Two layers live here:

1. **Per-camera masks** (B14, unchanged): each camera keeps a set of free camera-matrix entries
   (``cam.free_intrinsics``, default fx, fy, cx, cy) and ``n_dist`` (how many leading distortion
   coefficients are free). They are enforced by gradient hooks, so they apply to the Adam solver
   and to any optimizer built over ``cgroup.parameters()``, and survive copy/deepcopy/pickle.
   The skew ``K[0,1]``, ``K[1,0]`` and the bottom row of K are frozen by default.

2. **The full** :class:`ParamSpec` (D1), read by the Levenberg-Marquardt solver
   (:func:`aniposelib.bundle.bundle_adjust_lm`)::

       spec = ParamSpec(
           intrinsics=('f', 'cx', 'cy', 'k1'),   # names (see below); None = each camera's own masks
           n_dist=None,                          # free leading distortion coefficients (None = names / camera)
           extrinsics='free',                    # 'free' | 'fixed' | {cam: 'free'|'fixed'}
           share=[[0, 1, 2]],                    # identical cameras: one set of intrinsics per group
           priors={'cam_3': {'f': (1820, 0.05, 'log'), 'k1': (-0.14, 0.1)}},  # or IntrinsicsPrior(ref)
           fixed_cameras='auto',                 # gauge: extrinsics held fixed ('auto' = first camera)
           scale_constraint='auto',              # 'keep' | None | (camA, camB, dist, sigma) | 'auto'
           skew='fixed', aspect='free',
           time_offset=None)                     # F3: per-camera time offset prior sigma (None = off)

   Intrinsic names: ``'fx', 'fy', 'cx', 'cy', 'skew'`` (camera-matrix entries), ``'f'`` (one focal
   parameter moving fx and fy together at a fixed aspect ratio), and distortion names. For the
   Brown/OpenCV model the distortion layout is ``k1 k2 p1 p2 k3 k4 k5 k6``; for fisheye
   (Kannala-Brandt) ``k1 k2 k3 k4``. ``'dist'`` frees all of them; ``n_dist=k`` frees the first k.

   Names can be cameras' names or indices everywhere a camera is referred to.

A spec can be stored on a group (``cgroup.param_spec = spec``; it is deep-copied with the group)
and is then used by the LM solver when no ``spec=`` is passed.
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import torch

__all__ = ['ParamSpec', 'IntrinsicsPrior', 'MATRIX_ENTRIES', 'DEFAULT_FREE_INTRINSICS',
           'DIST_NAMES', 'validate_intrinsics', 'validate_names', 'matrix_mask', 'dist_index']

# name -> (row, col) in the 3x3 camera matrix
MATRIX_ENTRIES = {
    'fx': (0, 0),
    'fy': (1, 1),
    'cx': (0, 2),
    'cy': (1, 2),
    'skew': (0, 1),
}

DEFAULT_FREE_INTRINSICS = ('fx', 'fy', 'cx', 'cy')

# distortion names per model (index in the distortion vector)
DIST_NAMES = {
    'brown': ('k1', 'k2', 'p1', 'p2', 'k3', 'k4', 'k5', 'k6'),
    'kb': ('k1', 'k2', 'k3', 'k4'),
}
_ALL_DIST = ('k1', 'k2', 'k3', 'k4', 'k5', 'k6', 'p1', 'p2')
_SPECIAL = ('f', 'dist')
ALL_NAMES = tuple(MATRIX_ENTRIES) + _SPECIAL + _ALL_DIST


def validate_intrinsics(names):
    """Normalise a collection of camera-matrix entry names to a tuple in definition order
    (the per-camera mask of B14: only ``MATRIX_ENTRIES`` names are allowed)."""
    if isinstance(names, str):
        names = (names,)
    names = tuple(names)
    bad = [n for n in names if n not in MATRIX_ENTRIES]
    if bad:
        raise ValueError('unknown intrinsic name(s) {}; valid names are {}'.format(
            bad, list(MATRIX_ENTRIES)))
    return tuple(n for n in MATRIX_ENTRIES if n in names)


def validate_names(names):
    """Normalise the full-spec intrinsic names (matrix entries, 'f', distortion names, 'dist')."""
    if names is None:
        return None
    if isinstance(names, str):
        names = (names,)
    names = tuple(names)
    bad = [n for n in names if n not in ALL_NAMES]
    if bad:
        raise ValueError('unknown intrinsic name(s) {}; valid names are {}'.format(bad, list(ALL_NAMES)))
    return tuple(n for n in ALL_NAMES if n in names)


def dist_index(name, model):
    """Index of distortion ``name`` in the distortion vector of ``model`` ('brown' or 'kb'),
    or None if the model has no such coefficient."""
    names = DIST_NAMES[model]
    return names.index(name) if name in names else None


def matrix_mask(names, dtype=torch.float64, device=None):
    """3x3 mask with 1 on the free entries of the camera matrix."""
    mask = torch.zeros((3, 3), dtype=dtype, device=device)
    for n in validate_intrinsics(names):
        mask[MATRIX_ENTRIES[n]] = 1
    return mask


@dataclass
class IntrinsicsPrior:
    """Gaussian priors on intrinsics, centred on a reference calibration.

    ``reference``: a CameraGroup (cameras matched by name, else by position) or a list of cameras.
    ``sigma_logf``: sigma of log focal length (0.05 = 5 %); ``sigma_c``: principal point, px;
    ``sigma_k``: distortion coefficients (unitless). None skips that prior.
    A prior residual ``(value - mu) / sigma`` weighs like one observation with that pixel error.
    """
    reference: Any
    sigma_logf: Optional[float] = 0.05
    sigma_c: Optional[float] = None
    sigma_k: Optional[float] = None

    def priors_for(self, cameras):
        """``{camera index: {name: (mu, sigma[, 'log'])}}`` for ``cameras``."""
        ref = list(getattr(self.reference, 'cameras', self.reference))
        by_name = {c.get_name(): c for c in ref}
        out = {}
        for i, cam in enumerate(cameras):
            rc = by_name.get(cam.get_name())
            if rc is None:
                if i >= len(ref):
                    continue
                rc = ref[i]
            K = rc.get_camera_matrix().detach().cpu().numpy() if isinstance(rc.matrix, torch.Tensor) \
                else np.asarray(rc.get_camera_matrix())
            d = rc.dist.detach().cpu().numpy() if isinstance(rc.dist, torch.Tensor) else np.asarray(rc.dist)
            pr = {}
            if self.sigma_logf is not None:
                pr['fx'] = (float(K[0, 0]), self.sigma_logf, 'log')
                pr['fy'] = (float(K[1, 1]), self.sigma_logf, 'log')
            if self.sigma_c is not None:
                pr['cx'] = (float(K[0, 2]), self.sigma_c)
                pr['cy'] = (float(K[1, 2]), self.sigma_c)
            if self.sigma_k is not None:
                model = 'kb' if getattr(rc, 'fisheye', False) else 'brown'
                for j, n in enumerate(DIST_NAMES[model][:len(d)]):
                    pr[n] = (float(d[j]), self.sigma_k)
            out[i] = pr
        return out

    def to_dict(self):
        return {'reference': '<CameraGroup>', 'sigma_logf': self.sigma_logf,
                'sigma_c': self.sigma_c, 'sigma_k': self.sigma_k}


@dataclass
class ParamSpec:
    """Parameter spec: what bundle adjustment may move (see the module docstring).

    The first two fields are the minimal B14 spec and keep their meaning:
    ``intrinsics`` (free intrinsic names) and ``n_dist`` (free distortion count, None = keep
    each camera's own). ``spec.apply(cgroup_or_camera)`` writes the camera-matrix part and
    ``n_dist`` onto the cameras' gradient masks (used by the Adam solver);
    ``ParamSpec.from_camera(cam)`` reads them back.
    """
    intrinsics: Optional[Tuple[str, ...]] = DEFAULT_FREE_INTRINSICS
    n_dist: Optional[int] = None
    extrinsics: Union[str, Dict[Any, str]] = 'free'
    share: Optional[Any] = None
    priors: Optional[Any] = None
    fixed_cameras: Any = 'auto'
    scale_constraint: Any = 'auto'
    skew: str = 'fixed'
    aspect: str = 'free'
    time_offset: Optional[float] = None

    def __post_init__(self):
        if self.intrinsics is not None:
            self.intrinsics = validate_names(self.intrinsics)
        if self.skew not in ('fixed', 'free'):
            raise ValueError("skew must be 'fixed' or 'free', got {!r}".format(self.skew))
        if self.aspect not in ('fixed', 'free'):
            raise ValueError("aspect must be 'fixed' or 'free', got {!r}".format(self.aspect))
        if isinstance(self.extrinsics, str) and self.extrinsics not in ('free', 'fixed'):
            raise ValueError("extrinsics must be 'free', 'fixed' or a per-camera dict")
        sc = self.scale_constraint
        if not (sc is None or sc in ('keep', 'auto') or (isinstance(sc, (tuple, list)) and len(sc) in (3, 4))):
            raise ValueError("scale_constraint must be None, 'keep', 'auto' or (camA, camB, dist[, sigma])")

    # -------------------------------------------------- B14 (per-camera masks)
    def apply(self, target):
        """Write the camera-matrix names and ``n_dist`` onto the cameras' gradient masks."""
        cams = getattr(target, 'cameras', [target])
        names = self.matrix_names(skew=self.skew)
        for cam in cams:
            if names is not None:
                cam.set_free_intrinsics(names)
            if self.n_dist is not None:
                cam.set_n_dist(min(self.n_dist, cam.dist.shape[0]))
        return target

    @staticmethod
    def from_camera(cam):
        return ParamSpec(intrinsics=cam.get_free_intrinsics(), n_dist=cam.get_n_dist())

    def matrix_names(self, skew=None):
        """Camera-matrix entry names implied by ``intrinsics`` ('f' -> fx, fy)."""
        if self.intrinsics is None:
            return None
        out = set(n for n in self.intrinsics if n in MATRIX_ENTRIES)
        if 'f' in self.intrinsics:
            out |= {'fx', 'fy'}
        if (skew or self.skew) == 'free':
            out.add('skew')
        return validate_intrinsics(out)

    @classmethod
    def extrinsics_only(cls, **kw):
        """Spec with every intrinsic fixed (``only_extrinsics=True``)."""
        return cls(intrinsics=(), n_dist=0, **kw)

    def replace(self, **kw):
        d = {k: getattr(self, k) for k in self.__dataclass_fields__}
        d.update(kw)
        return ParamSpec(**d)

    def to_dict(self):
        d = {}
        for k in self.__dataclass_fields__:
            v = getattr(self, k)
            if isinstance(v, IntrinsicsPrior):
                v = v.to_dict()
            elif isinstance(v, tuple):
                v = list(v)
            d[k] = v
        return d

    # -------------------------------------------------- resolution (used by bundle.py)
    def camera_names_free(self, cam, model):
        """``(matrix names, distortion indices, use_f)`` free for one camera under this spec."""
        if self.intrinsics is None:
            mat = set(cam.get_free_intrinsics())
            nd = cam.get_n_dist() if self.n_dist is None else self.n_dist
            didx = set(range(min(int(nd), len(DIST_NAMES[model]), cam.dist.shape[0])))
            use_f = False
        else:
            mat = set(n for n in self.intrinsics if n in MATRIX_ENTRIES)
            use_f = 'f' in self.intrinsics
            didx = set()
            if 'dist' in self.intrinsics:
                didx = set(range(min(len(DIST_NAMES[model]), cam.dist.shape[0])))
            for n in self.intrinsics:
                if n in _ALL_DIST:
                    j = dist_index(n, model)
                    if j is not None and j < cam.dist.shape[0]:
                        didx.add(j)
            if self.n_dist is not None:
                didx |= set(range(min(int(self.n_dist), len(DIST_NAMES[model]), cam.dist.shape[0])))
            elif not any(n in _ALL_DIST or n == 'dist' for n in self.intrinsics):
                nd = cam.get_n_dist()
                didx |= set(range(min(int(nd), len(DIST_NAMES[model]), cam.dist.shape[0])))
        if self.skew == 'free':
            mat.add('skew')
        if self.aspect == 'fixed' and ('fx' in mat or 'fy' in mat):
            mat -= {'fx', 'fy'}
            use_f = True
        if use_f:
            mat -= {'fx', 'fy'}
        return mat, sorted(didx), use_f
