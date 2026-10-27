"""Initialisation of camera extrinsics (plan 01 phase E).

Sub-modules (import what you need; nothing heavy is imported here):

- :mod:`aniposelib.init.board`: board pose graphs that may be disconnected (one gauge per
  component + robust fallback) and a guarded ``init_camera_intrinsics``;
- :mod:`aniposelib.init.sfm`: ``from_correspondences`` (keypoint / blob incremental SfM with
  frame RANSAC, drivers/passengers, rotation priors) and ``refine_group`` (robust
  extrinsics-only refinement);
- :mod:`aniposelib.init.vggt`: VGGT hypotheses (optional extra ``[vggt]``, imported lazily);
- :mod:`aniposelib.init.select`: score candidate inits on the 2D data.

:func:`calibrate_from_points` is the 1.0 recipe API, an alias of the workflow
:func:`aniposelib.workflows.calibrate_from_points` since 1.1.
"""

from dataclasses import dataclass, field, fields

import numpy as np

from . import board, select as _select_mod, sfm
from ._common import InitReport, from_extrinsics, get_extrinsics, jsonable, to_np
from .select import select

__all__ = ['board', 'sfm', 'select', 'vggt', 'calibrate_from_points', 'RecipeReport',
           'InitReport', 'from_extrinsics', 'from_correspondences', 'make_intrinsics_group']

from_correspondences = sfm.from_correspondences


def __getattr__(name):  # lazy: aniposelib.init.vggt without importing it eagerly
    if name == 'vggt':
        import importlib
        return importlib.import_module('.vggt', __name__)
    raise AttributeError(name)


@dataclass
class RecipeReport:
    """What :func:`calibrate_from_points` did: the chosen ``init`` (its key), the
    :class:`InitReport` of each candidate generator, the :class:`~.select.SelectResult`, the
    refinement statistics, the scale result and the quality report (if ``aniposelib.quality``
    is available). ``errors`` lists candidate generators that failed and why."""
    init: object = None
    candidates: list = field(default_factory=list)
    init_reports: dict = field(default_factory=dict)
    selection: object = None
    refine: dict = field(default_factory=dict)
    scale: object = None
    quality: object = None
    errors: dict = field(default_factory=dict)

    def to_dict(self):
        return {f.name: jsonable(getattr(self, f.name)) for f in fields(self)}


def make_intrinsics_group(cam_names, sizes=None, intrinsics=None, fisheye=False):
    """A ``CameraGroup`` carrying intrinsics only (identity extrinsics).

    ``intrinsics``: a ``CameraGroup`` (copied; reordered to ``cam_names`` if given), a 3x3 or
    ``[C,3,3]`` matrix, a dict ``{name: dict(matrix=, dist=, fisheye=)}`` or ``{name: K}``, or
    None (nominal: focal = max(w, h), centred principal point; needs ``sizes``)."""
    from aniposelib.cameras import CameraGroup
    if isinstance(intrinsics, CameraGroup):
        cg = intrinsics.copy()
        if cam_names is not None and list(cam_names) != cg.get_names():
            cg = cg.subset_cameras_names(list(cam_names))
        if sizes is not None:
            for c, s in zip(cg.cameras, _sizes(sizes, len(cg.cameras))):
                if c.get_size() is None:
                    c.set_size(s)
        return cg
    if cam_names is None:
        raise ValueError('cam_names is required unless intrinsics is a CameraGroup')
    names = [str(n) for n in cam_names]
    C = len(names)
    szs = _sizes(sizes, C) if sizes is not None else [None] * C
    Ks, dists, models = [], [], []
    for i, n in enumerate(names):
        spec = intrinsics
        if isinstance(intrinsics, dict):
            spec = intrinsics.get(n, intrinsics.get(i))
        fish = fisheye
        dist = None
        if isinstance(spec, dict):
            fish = spec.get('fisheye', fisheye)
            dist = spec.get('dist', spec.get('distortions'))
            spec = spec.get('matrix')
        if spec is None:
            if szs[i] is None:
                raise ValueError('camera {}: no intrinsics and no size'.format(n))
            K = board.nominal_matrix(szs[i], fisheye=fish)
        else:
            K = np.asarray(to_np(spec), dtype='float64')
            if K.ndim == 3:
                K = K[i]
        Ks.append(K)
        models.append('fisheye' if fish else 'pinhole')
        d = np.zeros(4 if fish else 5) if dist is None else np.asarray(to_np(dist), dtype='float64').ravel()
        dists.append(d)
    cams = from_extrinsics(names, np.tile(np.eye(4), (C, 1, 1)), K=np.stack(Ks), sizes=None,
                           model=models)
    for c, d, s in zip(cams.cameras, dists, szs):
        c.set_distortions(d)
        if s is not None:
            c.set_size(s)
    return cams


def _sizes(sizes, C):
    s = np.asarray(sizes)
    if s.ndim == 1:
        return [tuple(int(v) for v in s)] * C
    return [tuple(int(v) for v in row) for row in s]


def _has_extrinsics(cg):
    Ms = get_extrinsics(cg)
    return not np.allclose(Ms, np.eye(4)[None]) and np.all(np.isfinite(Ms))


def _refine(cg, p2ds, frame_id, method, rng, ba_kwargs):
    """Final refinement. 'auto' / 'robust': :func:`aniposelib.bundle.bundle_adjust_robust`
    (extrinsics only, MSAC inliers + LM + resection + structured rejection; reverted if the robust
    median error does not improve; ``ba_kwargs`` are passed through). 'refine': the robust sparse
    extrinsics refinement :func:`sfm.refine_group`. 'iter': ``bundle_adjust_iter`` (extrinsics
    only), kept only if it lowers the robust median error. None/False: no refinement."""
    kw = dict(ba_kwargs or {})
    if not method:
        return cg, dict(method=None)
    if method in ('auto', 'robust'):
        p = np.asarray(to_np(p2ds), dtype='float64')
        kw.setdefault('only_extrinsics', True)
        kw.setdefault('final', ('extrinsics',) if kw['only_extrinsics'] else ('extrinsics', 'intrinsics'))
        kw.setdefault('revert_if_worse', True)
        kw.setdefault('rng', 0 if rng is None else rng)
        if frame_id is not None and 'groups' not in kw:
            fid = np.asarray(frame_id)
            if p.ndim == 3 and fid.shape == p.shape[1:2]:
                kw['groups'] = dict(frame=fid)
        out = cg.bundle_adjust_robust(p, **kw)
        return cg, dict(method='bundle_adjust_robust', result=out, before_px=out.error_before,
                        after_px=out.error, kept=not out.reverted)
    if method == 'refine':
        out, st = sfm.refine_group(cg, p2ds, frame_id=frame_id, rng=rng, **kw)
        st['method'] = 'refine_group'
        return out, st
    if method == 'iter':
        p = np.asarray(to_np(p2ds), dtype='float64')
        flat = p.reshape(p.shape[0], -1, 2)
        before = _select_mod.score_candidate(cg, flat)['median_px']
        trial = cg.copy()
        kw.setdefault('only_extrinsics', True)
        kw.setdefault('rng', 0 if rng is None else rng)
        trial.bundle_adjust_iter(flat, **kw)
        after = _select_mod.score_candidate(trial, flat)['median_px']
        keep = bool(np.isfinite(after) and after <= before)
        return (trial if keep else cg), dict(method='bundle_adjust_iter', before_px=before,
                                             after_px=after, kept=keep)
    raise ValueError("refine must be 'auto', 'refine', 'robust', 'iter' or None")


def calibrate_from_points(p2ds, cam_names=None, sizes=None, intrinsics=None, init='auto',
                          frames=None, groups=None, robust=True, scale=None, *, frame_id=None,
                          window_id=None, refine='auto', ba_kwargs=None, sfm_kwargs=None,
                          vggt_kwargs=None, select_kwargs=None, report=True, rng=None):
    """Calibrate extrinsics from 2D points with known intrinsics (the 1.0 recipe API).

    Since 1.1 an alias of the workflow :func:`aniposelib.workflows.calibrate_from_points`
    (candidate inits -> :func:`select` -> noise-normalised robust BA -> scale -> quality), with
    the 1.0 signature and return value:

    - ``intrinsics`` / ``cam_names`` / ``sizes``: see :func:`make_intrinsics_group`;
    - ``init``: ``'auto'`` | ``'reference'`` | ``'sfm'`` | ``'vggt'`` (needs ``frames``) | a
      ``CameraGroup`` | a list; a single generator that fails raises, as in 1.0;
    - ``frame_id`` / ``window_id``: SfM frames / windows (and structured rejection);
      ``groups``: point labels for the ``quality.report`` per-group breakdown;
    - ``refine``: ``'auto'`` / ``'robust'`` (workflow robust BA), ``'refine'``
      (``sfm.refine_group``), ``'iter'`` (Adam BA, kept if better; ``robust=False`` turns
      ``'auto'`` into it) or None;
    - ``scale``: float | reference group (also aligns the gauge) | ``dict(reference=, align=False)``
      | ``dict(pairs=, lengths=)`` | ``dict(board=, rows=)``.

    Intrinsics are never refined here (``refine_intrinsics=False``). Returns
    ``(CameraGroup, RecipeReport)``; ``report.scale`` is a
    :class:`~aniposelib.workflows.ScaleEstimate`. For the verdict / full result use the workflow."""
    from aniposelib.workflows.calibration import calibrate_from_points as _workflow
    base = make_intrinsics_group(cam_names, sizes, intrinsics)
    method = refine if (robust or refine != 'auto') else 'iter'
    if method == 'auto':
        method = 'robust'
    if method not in ('robust', 'refine', 'iter', None, False):
        raise ValueError("refine must be 'auto', 'refine', 'robust', 'iter' or None")
    grp = {k: v for k, v in (('frame', frame_id), ('window', window_id)) if v is not None}
    if callable(scale) and not hasattr(scale, 'cameras'):
        sc_fn, scale = scale, None
    else:
        sc_fn = None
    res = _workflow(p2ds, base, init=init, frames=frames, groups=grp or None, refine_intrinsics=False,
                    scale=scale, rng=0 if rng is None else rng,
                    options=dict(refine=method or None, ba_kwargs=ba_kwargs, sfm_kwargs=sfm_kwargs,
                                 vggt_kwargs=vggt_kwargs, select_kwargs=select_kwargs, quality=report,
                                 quality_groups=groups))
    rpt = res.init_report
    single = not (isinstance(init, (list, tuple)) or init == 'auto')
    if res.cgroup is None or (single and rpt.errors and not rpt.candidates):
        raise RuntimeError('calibrate_from_points: no initialisation succeeded: {}'.format(rpt.errors))
    if res.scale is not None and not res.scale.accepted:
        raise ValueError('scale estimate not accepted: {}'.format(res.scale.report.get('reason')))
    cg = res.cgroup
    if sc_fn is not None:
        from aniposelib import scale as S
        s = float(sc_fn(cg))
        S.rescale(cg, s, source='manual')
        rpt.scale = dict(s=s, source='callable')
    return cg, rpt
