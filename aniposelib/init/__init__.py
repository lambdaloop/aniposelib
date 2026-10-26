"""Initialisation of camera extrinsics (plan 01 phase E).

Sub-modules (import what you need; nothing heavy is imported here):

- :mod:`aniposelib.init.board`: board pose graphs that may be disconnected (one gauge per
  component + robust fallback) and a guarded ``init_camera_intrinsics``;
- :mod:`aniposelib.init.sfm`: ``from_correspondences`` (keypoint / blob incremental SfM with
  frame RANSAC, drivers/passengers, rotation priors) and ``refine_group`` (robust
  extrinsics-only refinement);
- :mod:`aniposelib.init.vggt`: VGGT hypotheses (optional extra ``[vggt]``, imported lazily);
- :mod:`aniposelib.init.select`: score candidate inits on the 2D data.

:func:`calibrate_from_points` is a recipe composed of those parts.
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


def _scale_from_reference(cg, ref, align, kw):
    """Scale (and, if ``align``, the full similarity gauge) from a reference calibration of the
    same cameras: ``scale.from_reference`` (flags, residuals), then ``scale.align_to``."""
    from aniposelib import scale as S
    res = S.from_reference(cg, ref, **kw)
    if not align:
        res.apply(cg)
        return res
    if res.scale is None or not res.accepted:
        raise ValueError('scale estimate not accepted: {}'.format(res.reason))
    al = S.align_to(cg, ref, trim=kw.get('trim', 2.5), min_keep=kw.get('min_keep', 3))
    md = cg.metadata
    md['scale_source'] = 'reference'
    if res.units is not None:
        md['units'] = str(res.units)
    q = res.quality()
    if q:
        md['scale_quality'] = S._toml_clean(dict(q))
    res.details['align'] = al.to_dict()
    return res


def _apply_scale(cg, scale, p2ds):
    """``scale``: None, a float (multiply translations), a reference ``CameraGroup``
    (``scale.from_reference``, then the gauge is aligned to the reference with
    ``scale.align_to``: same scale, rotation and origin; ``dict(reference=ref, align=False)``
    only rescales), ``dict(pairs=, lengths=)`` (``scale.from_lengths`` on
    ``p2ds [C,N,J,2]``), ``dict(board=, rows=)`` (``scale.from_board``) or a callable
    ``f(cgroup) -> s``."""
    from aniposelib import scale as S
    from aniposelib.cameras import CameraGroup
    if scale is None:
        return None
    if callable(scale) and not isinstance(scale, CameraGroup):
        s = float(scale(cg))
        S.rescale(cg, s, source='manual')
        return dict(s=s, source='callable')
    if isinstance(scale, (int, float)):
        S.rescale(cg, float(scale), source='manual')
        return dict(s=float(scale), source='manual')
    if isinstance(scale, dict) and 'reference' in scale:
        kw = {k: v for k, v in scale.items() if k not in ('reference', 'align')}
        return _scale_from_reference(cg, scale['reference'], scale.get('align', True), kw)
    if isinstance(scale, CameraGroup):
        return _scale_from_reference(cg, scale, True, {})
    elif isinstance(scale, dict) and 'board' in scale:
        res = S.from_board(cg, scale['rows'], scale['board'],
                           **{k: v for k, v in scale.items() if k not in ('board', 'rows')})
    elif isinstance(scale, dict) and 'pairs' in scale:
        res = S.from_lengths(cg, scale.get('p2ds', p2ds), scale['pairs'], scale['lengths'],
                             **{k: v for k, v in scale.items() if k not in ('pairs', 'lengths', 'p2ds')})
    else:
        raise ValueError('unsupported scale spec: {!r}'.format(type(scale)))
    res.apply(cg)
    return res


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
    """Calibrate extrinsics from 2D points with known intrinsics: a recipe.

    1. Intrinsics group from ``intrinsics`` (``CameraGroup`` | K | dict | None, see
       :func:`make_intrinsics_group`) with ``cam_names`` / ``sizes``.
    2. Candidate inits, by ``init``:
       ``'reference'`` (the extrinsics of ``intrinsics``, e.g. a lab calibration),
       ``'sfm'`` (:func:`sfm.from_correspondences`), ``'vggt'`` (needs ``frames``, the extra
       ``[vggt]``: the aggregate and every template are candidates), a ``CameraGroup``, a list
       of those, or ``'auto'`` = reference (if it has extrinsics) + sfm + vggt (if ``frames``).
       Generators that fail are recorded in ``report.errors``.
    3. :func:`select` (robust-inlier fraction, ``select_kwargs``) picks one; if SfM had to drop
       cameras, only complete candidates compete unless none is complete.
    4. Refinement (``refine``, see :func:`_refine`; ``robust=False`` turns ``'auto'`` into
       ``'iter'``, the classic Adam BA).
    5. Scale (``scale``: float | reference group (also aligns the gauge: rotation and origin
       of the reference; ``dict(reference=ref, align=False)`` only rescales) |
       ``dict(pairs=, lengths=)`` | ``dict(board=, rows=)`` | callable), via
       :mod:`aniposelib.scale`.
    6. ``aniposelib.quality.report(cgroup, p2ds, groups=groups)`` if ``report``.

    ``p2ds``: ``[C,N,2]`` or ``[C,N,J,2]`` in ``cam_names`` order (NaN = missing).
    Returns ``(CameraGroup, RecipeReport)``. The group may lack cameras the init could not
    place (see ``report.init_reports``)."""
    from aniposelib.cameras import CameraGroup
    rpt = RecipeReport()
    base = make_intrinsics_group(cam_names, sizes, intrinsics)
    names = base.get_names()
    p = np.asarray(to_np(p2ds), dtype='float64')
    if p.shape[0] != len(names):
        raise ValueError('p2ds has {} cameras, expected {}'.format(p.shape[0], len(names)))
    if init == 'auto':
        specs = (['reference'] if _has_extrinsics(base) else []) + ['sfm'] + (['vggt'] if frames is not None else [])
    elif isinstance(init, (list, tuple)):
        specs = list(init)
    else:
        specs = [init]
    cands = {}
    priors = None
    # vggt first: its pairwise rotations become SfM priors
    for spec in sorted(specs, key=lambda s: 0 if s == 'vggt' else 1):
        try:
            if isinstance(spec, CameraGroup):
                cands['given_{}'.format(len(cands))] = spec
            elif spec == 'reference':
                cands['reference'] = base
            elif spec == 'vggt':
                from . import vggt as V
                pred = V.predict(frames, names=names, **(vggt_kwargs or {}))
                agg = V.aggregate(pred)
                cands['vggt'] = V.to_cameragroup(agg, template=base)
                for t in range(pred.extrinsics.shape[0]):
                    cands['vggt_t{}'.format(t)] = from_extrinsics(names, pred.extrinsics[t], template=base)
                priors = V.pairwise_rotations(pred)
                rpt.init_reports['vggt'] = dict(stats=agg.stats)
            elif spec == 'sfm':
                kw = dict(frame_id=frame_id, window_id=window_id, rotation_priors=priors, rng=rng)
                kw.update(sfm_kwargs or {})
                cg, ir = sfm.from_correspondences(base, p, **kw)
                cands['sfm'] = cg
                rpt.init_reports['sfm'] = ir
            else:
                raise ValueError('unknown init {!r}'.format(spec))
        except Exception as exc:
            if len(specs) == 1:
                raise
            rpt.errors[str(spec)] = '{}: {}'.format(type(exc).__name__, exc)
    if not cands:
        raise RuntimeError('calibrate_from_points: no initialisation succeeded: {}'.format(rpt.errors))
    complete = {k: v for k, v in cands.items() if set(v.get_names()) == set(names)}
    pool = complete if complete else cands
    rpt.candidates = list(cands)
    if len(pool) == 1:
        key = next(iter(pool))
        chosen = pool[key]
    else:
        common = [n for n in names if all(n in v.get_names() for v in pool.values())]
        idx = [names.index(n) for n in common]
        chosen, rpt.selection = select(pool, p[idx], cam_names=common, **(select_kwargs or {}))
        key = rpt.selection.best
        chosen = pool[key]
    rpt.init = key
    present = [n for n in names if n in chosen.get_names()]
    if present != chosen.get_names():
        chosen = chosen.subset_cameras_names(present)
    pc = p[[names.index(n) for n in present]]
    method = refine if (robust or refine != 'auto') else 'iter'
    cgroup, rpt.refine = _refine(chosen, pc, frame_id, method, rng, ba_kwargs)
    if scale is not None:
        rpt.scale = _apply_scale(cgroup, scale, pc)
    if report:
        try:
            from aniposelib import quality
            rpt.quality = quality.report(cgroup, pc, groups=groups)
        except Exception as exc:  # quality is optional here
            rpt.errors['quality'] = '{}: {}'.format(type(exc).__name__, exc)
    return cgroup, rpt
