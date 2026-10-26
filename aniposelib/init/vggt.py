"""VGGT camera initialisation (optional extra ``[vggt]``: ``vggt``, ``huggingface_hub``).

VGGT (Wang et al. 2025, ``facebook/VGGT-1B``) predicts, from one synchronised frame per camera
(a *template*), every camera's world->camera extrinsics and pinhole intrinsics, up to an
arbitrary similarity (VGGT puts camera 0 at the identity and renormalises scene scale on every
inference). One template is noisy; several templates (different moments of the same session)
averaged in a common gauge give a usable *hypothesis* to start bundle adjustment from. It is
not an answer: template self-consistency does not predict accuracy (motor observatory), so
score the result against real 2D data (``aniposelib.init.select``) before trusting it.

Pipeline::

    pred = vggt.predict(frame_sets, names=names)      # needs vggt; frees the 5 GB model after
    agg = vggt.aggregate(pred)                        # numpy only: Sim(3) align + robust means
    cgroup = vggt.to_cameragroup(agg)                 # or from_extrinsics(names, agg.Ms, K=...)

* :func:`predict` runs the network. Intrinsics are mapped back to **original image pixels**
  (VGGT sees a resized, padded/cropped 518-px image; using its K directly gives the wrong
  aspect ratio: disco board-scale MAD 16.5 % -> 1.5 % after the fix).
* :func:`aggregate`, :func:`normalize_template_scale`, :func:`pairwise_rotations`,
  :func:`restore_intrinsics` and :func:`preprocess_transform` are pure numpy and testable
  without the network.

Conventions: extrinsics are 4x4 world->camera (``x_cam = R x_world + t``) like
``Camera.get_extrinsics_mat()``; numpy float64 out. Ported from disco
``calibration_disco.py`` (padding fix, artifact layout), EvoVision ``calibrate.py`` (fp16/bf16
auto, local-cache-first load, per-template scale normalisation, QA stats, freeing the model)
and the motor observatory ``calibrate_lib.py`` (pairwise VGGT, robust rotation/direction means).
"""

import logging
from dataclasses import dataclass, field, fields

import numpy as np

from ._common import (as_rng, centers, chordal_mean, jsonable, make_M, project_to_so3,
                      robust_rotation_mean, rotation_angle_deg, transform_extrinsics, umeyama)
from ._common import from_extrinsics as _from_extrinsics

log = logging.getLogger(__name__)

__all__ = ['VGGTPrediction', 'VGGTAggregate', 'predict', 'predict_pairwise', 'aggregate',
           'normalize_template_scale', 'pairwise_rotations', 'preprocess_transform',
           'restore_intrinsics', 'restore_padded_intrinsics', 'preprocess_image',
           'from_extrinsics', 'to_cameragroup', 'free_model', 'weights_cached']

DEFAULT_MODEL = 'facebook/VGGT-1B'
TARGET_SIZE = 518
PATCH = 14


# ====================================================================== result objects

@dataclass
class VGGTPrediction:
    """Raw per-template VGGT output.

    ``extrinsics [T,C,4,4]`` world->camera per template (each template in its own arbitrary
    gauge and scale); ``intrinsics [T,C,3,3]`` in **original image pixels**;
    ``sizes [C,2]`` original ``(width, height)``; ``model_size (width, height)`` of the
    network input; ``transforms [T,C,3,3]`` original->model pixel maps (see
    :func:`preprocess_transform`); ``stats`` timing/dtype/device info."""
    extrinsics: np.ndarray
    intrinsics: np.ndarray
    sizes: np.ndarray
    names: list
    model_size: tuple = (0, 0)
    transforms: np.ndarray = None
    stats: dict = field(default_factory=dict)

    @property
    def n_templates(self):
        return int(len(self.extrinsics))

    def median_intrinsics(self):
        """``[C,3,3]`` per-camera median of K over templates."""
        return np.median(self.intrinsics, axis=0)

    def to_dict(self):
        return {f.name: jsonable(getattr(self, f.name)) for f in fields(self)}


@dataclass
class VGGTAggregate:
    """Averaged VGGT rig.

    ``Ms [C,4,4]`` world->camera, in the gauge of reference template ``ref`` scaled so that the
    mean camera-centre distance from the centroid is 1. ``kept [C]`` templates used per camera.
    QA (``stats``): ``raw_scale_rel_spread`` (std/mean of the raw per-template rig size; large =
    VGGT unstable), ``raw_scales``, ``rot_disp_deg [C]`` (median angle of a camera's aligned
    template rotations from the mean), ``center_disp [C]`` (median centre distance, rig units),
    ``template_residuals [T]`` (median centre residual after alignment), ``rejected_templates``.
    ``K``/``sizes`` are carried over from a :class:`VGGTPrediction` (median K) when available."""
    Ms: np.ndarray
    names: list
    kept: np.ndarray
    n_templates: int
    ref: int = 0
    align: str = 'sim3'
    rotation_mean: str = 'chordal'
    K: np.ndarray = None
    sizes: np.ndarray = None
    aligned: np.ndarray = None
    stats: dict = field(default_factory=dict)

    @property
    def raw_scale_rel_spread(self):
        return self.stats.get('raw_scale_rel_spread')

    def to_dict(self):
        return {f.name: jsonable(getattr(self, f.name)) for f in fields(self) if f.name != 'aligned'}


# ====================================================================== pure math

def normalize_template_scale(Ms):
    """Scale each template so that the mean camera-centre distance from the centroid is 1.

    ``Ms`` is ``[C,4,4]`` (one template) or ``[T,C,4,4]``. Only translations change (the
    world is scaled about the origin). Returns ``(Ms_normalised, scales)`` where ``scales``
    (scalar or ``[T]``) is the raw rig size that was divided out (EvoVision
    ``vggt_extrinsics``). A degenerate template (all centres equal) keeps scale 1 and gets
    ``nan`` in ``scales``."""
    Ms = np.array(Ms, dtype='float64', copy=True)
    single = Ms.ndim == 3
    if single:
        Ms = Ms[None]
    C = centers(Ms)
    s = np.linalg.norm(C - C.mean(axis=1, keepdims=True), axis=-1).mean(axis=1)
    ok = s > 1e-12
    div = np.where(ok, s, 1.0)
    Ms[..., :3, 3] /= div[:, None, None]
    s = np.where(ok, s, np.nan)
    return (Ms[0], float(s[0])) if single else (Ms, s)


def preprocess_transform(orig_size, mode='crop', target_size=TARGET_SIZE, canvas_size=None):
    """Geometry of VGGT's ``load_and_preprocess_images``: the 3x3 map ``A`` taking original
    pixel coordinates to network-input pixel coordinates, plus the network input size.

    ``orig_size = (width, height)``. ``mode='crop'`` (VGGT default): width -> 518, height
    scaled and rounded to a multiple of 14, centre-cropped to 518 if taller. ``mode='pad'``:
    longest side -> 518, the other rounded to a multiple of 14, then centre-padded to
    518x518. ``canvas_size=(W, H)``: the extra centre padding VGGT adds when a batch mixes
    image shapes. Continuous pixel convention (``x_model = s x_orig + offset``), as in disco.
    Returns ``(A, (width, height), (new_w, new_h))`` where the last is the resized (pre
    crop/pad) size."""
    w, h = int(orig_size[0]), int(orig_size[1])
    if w <= 0 or h <= 0:
        raise ValueError('image sizes must be positive, got {}'.format(orig_size))
    if mode == 'pad':
        if w >= h:
            nw, nh = target_size, int(round(h * (target_size / w) / PATCH)) * PATCH
        else:
            nh, nw = target_size, int(round(w * (target_size / h) / PATCH)) * PATCH
        nw, nh = max(nw, PATCH), max(nh, PATCH)
        hp, wp = max(target_size - nh, 0), max(target_size - nw, 0)
        ox, oy = wp // 2, hp // 2
        out_w, out_h = nw + wp, nh + hp
    elif mode == 'crop':
        nw = target_size
        nh = max(int(round(h * (nw / w) / PATCH)) * PATCH, PATCH)
        ox, oy = 0, 0
        out_w, out_h = nw, nh
        if nh > target_size:
            oy = -((nh - target_size) // 2)
            out_h = target_size
    else:
        raise ValueError("mode must be 'crop' or 'pad', got {!r}".format(mode))
    if canvas_size is not None:
        cw, ch = int(canvas_size[0]), int(canvas_size[1])
        ox += max(cw - out_w, 0) // 2
        oy += max(ch - out_h, 0) // 2
        out_w, out_h = max(cw, out_w), max(ch, out_h)
    A = np.array([[nw / w, 0.0, ox], [0.0, nh / h, oy], [0.0, 0.0, 1.0]])
    return A, (out_w, out_h), (nw, nh)


def restore_intrinsics(K_model, transform):
    """Intrinsics in network-input pixels -> original image pixels: ``K = A^-1 K_model``
    where ``A`` (``[...,3,3]``, from :func:`preprocess_transform`) maps original to model
    pixels. Broadcasts over leading axes."""
    K_model = np.asarray(K_model, dtype='float64')
    A = np.asarray(transform, dtype='float64')
    if K_model.shape[-2:] != (3, 3):
        raise ValueError('intrinsics must end in (3, 3), got {}'.format(K_model.shape))
    return np.linalg.inv(A) @ K_model


def restore_padded_intrinsics(K_model, original_size, model_size=(TARGET_SIZE, TARGET_SIZE), mode='pad'):
    """Disco ``_restore_padded_intrinsics``-compatible wrapper: undo VGGT's resize and
    letterbox for an image of ``original_size=(w, h)`` fed as ``model_size=(W, H)``."""
    A, _, _ = preprocess_transform(original_size, mode=mode, target_size=max(model_size),
                                   canvas_size=model_size)
    return restore_intrinsics(K_model, A)


def _robust_mean_direction(ts, tol_deg=12.0, n_iter=5):
    """Mean unit direction with iterative rejection (motor ``_robust_mean_direction``)."""
    n = np.asarray(ts, dtype='float64').reshape(-1, 3)
    n = n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)

    def unit(v):
        return v / max(np.linalg.norm(v), 1e-12)

    d = unit(n.mean(axis=0))
    keep = np.ones(len(n), bool)
    for _ in range(n_iter):
        ang = np.degrees(np.arccos(np.clip(n @ d, -1, 1)))
        new = ang <= max(tol_deg, np.median(ang) * 1.5)
        if new.sum() < 2:
            new = ang <= np.percentile(ang, 60)
        if new.sum() < 2:
            break
        d = unit(n[new].mean(axis=0))
        if (new == keep).all():
            break
        keep = new
    ang = np.degrees(np.arccos(np.clip(n @ d, -1, 1)))
    return d, keep, float(np.median(ang))


def _median_rotation(Rs):
    """Element-wise median projected to SO(3) (disco ``aggregate_vggt_cameras``)."""
    return project_to_so3(np.median(np.asarray(Rs), axis=0))


def _anchor_points(Ms, axis_len=1.0):
    """Camera centres plus a point along each camera axis (x, y, z) at ``axis_len``: makes the
    Sim(3) fit use orientations too (and well-posed for 2 cameras)."""
    C = centers(Ms)
    R = Ms[..., :3, :3]
    pts = [C] + [C + axis_len * R[..., k, :] for k in range(3)]
    return np.concatenate(pts, axis=-2)


def _fit_sim3(src_Ms, dst_Ms, n_cams, n_iter=2, rel_tol=3.0):
    """Similarity mapping template ``src`` onto ``dst`` (both scale-normalised) using centres
    and orientation anchor points, with one robust re-weighting pass dropping cameras whose
    residual exceeds ``rel_tol`` times the median."""
    a, b = _anchor_points(src_Ms), _anchor_points(dst_Ms)
    w = np.ones(len(a))
    for _ in range(n_iter):
        s, R, t = umeyama(a, b, with_scale=True, weights=w)
        res = np.linalg.norm(s * a @ R.T + t - b, axis=1).reshape(4, n_cams).mean(axis=0)
        med = np.median(res)
        cam_w = (res <= max(rel_tol * med, 1e-9)).astype(float)
        if cam_w.sum() < 2:
            break
        w = np.tile(cam_w, 4)
    C_res = np.linalg.norm(s * centers(src_Ms) @ R.T + t - centers(dst_Ms), axis=1)
    return (s, R, t), C_res


def aggregate(extrinsics, names=None, align='sim3', rotation_mean='chordal', ref=None, rng=None,
              reject_rel=3.0, reject_abs=0.05, center_tol=0.05, rot_tol_deg=8.0, n_iter=3):
    """Average per-template VGGT extrinsics into one rig.

    Parameters
    ----------
    extrinsics : ``[T,C,4,4]`` world->camera per template, or a :class:`VGGTPrediction`
        (then ``names``, median ``K`` and ``sizes`` are carried over).
    align : ``'sim3'`` (default) scale-normalise every template, pick a reference template
        (``ref``; default = the medoid, the template that agrees best with all others after
        a similarity fit), fit each template to it with Umeyama on camera centres plus
        orientation anchors (robust to a bad camera), reject templates whose median centre
        residual exceeds ``max(reject_abs, reject_rel * median)``, average, then re-align to
        the running mean ``n_iter`` times. ``'first'``: VGGT's own gauge (camera 0 = identity)
        with per-template scale normalisation only (EvoVision/motor behaviour). ``'none'``:
        average the inputs as they are (already in one gauge).
    rotation_mean : ``'chordal'`` (L2 chordal mean of the kept templates), ``'robust'``
        (chordal with iterative rejection beyond ``max(rot_tol_deg, 1.5 * median)``) or
        ``'median'`` (element-wise median projected to SO(3), disco).
    center_tol : per camera, templates whose aligned centre is further than
        ``max(center_tol, 2.5 * median)`` (rig units) from the median centre are dropped
        before the centre is averaged.
    rng : accepted for API uniformity (the algorithm is deterministic).

    Returns a :class:`VGGTAggregate` (numpy).
    """
    as_rng(rng)
    pred = extrinsics if isinstance(extrinsics, VGGTPrediction) else None
    if pred is not None:
        names = pred.names if names is None else names
        extrinsics = pred.extrinsics
    E = np.asarray(extrinsics, dtype='float64')
    if E.ndim == 3:
        E = E[None]
    if E.ndim != 4 or E.shape[-2:] != (4, 4):
        raise ValueError('extrinsics must be [T,C,4,4], got {}'.format(E.shape))
    T, C = E.shape[:2]
    names = [str(i) for i in range(C)] if names is None else [str(n) for n in names]
    if len(names) != C:
        raise ValueError('len(names)={} but extrinsics have C={}'.format(len(names), C))
    if rotation_mean not in ('chordal', 'robust', 'median'):
        raise ValueError("rotation_mean must be 'chordal', 'robust' or 'median'")
    finite = np.isfinite(E).all(axis=(1, 2, 3))

    En, raw_scales = normalize_template_scale(E)
    finite &= np.isfinite(raw_scales)
    good = np.flatnonzero(finite)
    if len(good) == 0:
        raise ValueError('no finite, non-degenerate templates')
    rs = raw_scales[good]
    stats = dict(raw_scales=raw_scales, raw_scale_rel_spread=float(rs.std() / max(rs.mean(), 1e-12)))
    template_res = np.full(T, np.nan)
    rejected = [int(t) for t in np.flatnonzero(~finite)]

    if align == 'none':
        A = E.copy()
        use = finite.copy()
        ref = int(good[0]) if ref is None else int(ref)
    elif align == 'first':
        A = En @ np.linalg.inv(En[:, :1])  # camera 0 = identity, scale already normalised
        A, _ = normalize_template_scale(A)
        use = finite.copy()
        ref = int(good[0]) if ref is None else int(ref)
    elif align == 'sim3':
        if ref is None:
            if len(good) <= 2:
                ref = int(good[0])
            else:
                cost = np.zeros(len(good))
                for a_i, ta in enumerate(good):
                    r = [np.median(_fit_sim3(En[tb], En[ta], C)[1]) for tb in good if tb != ta]
                    cost[a_i] = np.median(r)
                ref = int(good[np.argmin(cost)])
        ref = int(ref)
        target = En[ref]
        A = np.full_like(En, np.nan)
        use = finite.copy()
        for it in range(max(1, n_iter)):
            for tt in good:
                (s, R, t), res = _fit_sim3(En[tt], target, C)
                A[tt] = transform_extrinsics(En[tt], s, R, t)
                template_res[tt] = np.median(res)
            med = np.median(template_res[good])
            use = finite & (template_res <= max(reject_abs, reject_rel * med))
            if use.sum() < 1:
                use = finite.copy()
            target = _average(A[use], rotation_mean, rot_tol_deg, center_tol)[0]
            # keep the reference gauge: re-fit the mean onto the reference template
            (s, R, t), _ = _fit_sim3(target, En[ref], C)
            target = transform_extrinsics(target, s, R, t)
            target, _ = normalize_template_scale(target)
        rejected = [int(t) for t in np.flatnonzero(~use)]
        stats['template_residuals'] = template_res
    else:
        raise ValueError("align must be 'sim3', 'first' or 'none'")

    Ms, kept, rot_disp, cen_disp = _average(A[use], rotation_mean, rot_tol_deg, center_tol)
    if align == 'sim3':
        (s, R, t), _ = _fit_sim3(Ms, En[ref], C)
        Ms, _ = normalize_template_scale(transform_extrinsics(Ms, s, R, t))
    stats.update(rot_disp_deg=rot_disp, center_disp=cen_disp, rejected_templates=rejected,
                 n_used=int(use.sum()))
    K = sizes = None
    if pred is not None:
        K, sizes = pred.median_intrinsics(), np.asarray(pred.sizes)
    return VGGTAggregate(Ms=Ms, names=names, kept=kept, n_templates=int(T), ref=ref, align=align,
                         rotation_mean=rotation_mean, K=K, sizes=sizes, aligned=A, stats=stats)


def _average(A, rotation_mean, rot_tol_deg, center_tol):
    """Per-camera robust mean of aligned templates ``A [T,C,4,4]``."""
    T, C = A.shape[:2]
    Cs = centers(A)
    Ms = np.zeros((C, 4, 4))
    kept = np.zeros(C, int)
    rot_disp = np.zeros(C)
    cen_disp = np.zeros(C)
    for c in range(C):
        Rs = A[:, c, :3, :3]
        if rotation_mean == 'robust':
            R, rkeep, _ = robust_rotation_mean(Rs, tol_deg=rot_tol_deg)
        elif rotation_mean == 'median':
            R, rkeep = _median_rotation(Rs), np.ones(T, bool)
        else:
            R, rkeep = chordal_mean(Rs), np.ones(T, bool)
        med = np.median(Cs[:, c], axis=0)
        d = np.linalg.norm(Cs[:, c] - med, axis=1)
        ckeep = d <= max(center_tol, 2.5 * np.median(d))
        keep = rkeep & ckeep
        if keep.sum() == 0:
            keep = ckeep if ckeep.any() else np.ones(T, bool)
        if rotation_mean == 'chordal' and not keep.all():
            R = chordal_mean(Rs[keep])
        cen = Cs[keep, c].mean(axis=0)
        Ms[c] = make_M(R, -R @ cen)
        kept[c] = int(keep.sum())
        rot_disp[c] = float(np.median(rotation_angle_deg(R, Rs)))
        cen_disp[c] = float(np.median(np.linalg.norm(Cs[:, c] - cen, axis=1)))
    return Ms, kept, rot_disp, cen_disp


def pairwise_rotations(extrinsics, pairs=None, min_templates=3, rot_tol_deg=8.0, dir_tol_deg=12.0):
    """Relative pose of camera pairs from per-template extrinsics (gauge-free: no alignment).

    ``extrinsics [T,C,4,4]`` or a :class:`VGGTPrediction`. For each pair ``(i, j)`` (default:
    all ``i < j``) and template, the relative transform ``M_j M_i^-1`` (camera i frame ->
    camera j frame) is formed; its rotation is averaged with :func:`robust_rotation_mean`
    and its translation direction with a robust mean direction (motor
    ``vggt_pairwise_rotations``). Returns ``{(i, j): dict(R, t_dir, scale, rot_disp_deg,
    dir_disp_deg, n_templates)}``; ``scale`` is the median baseline in scale-normalised
    template units. Pairs with fewer than ``min_templates`` finite templates are omitted.

    These are hypotheses: dispersion does not predict accuracy (motor: correlation 0.19).
    Score every edge against real 2D data before use."""
    if isinstance(extrinsics, VGGTPrediction):
        extrinsics = extrinsics.extrinsics
    E = np.asarray(extrinsics, dtype='float64')
    T, C = E.shape[:2]
    En, s = normalize_template_scale(E)
    if pairs is None:
        pairs = [(i, j) for i in range(C) for j in range(i + 1, C)]
    out = {}
    for (i, j) in pairs:
        i, j = int(i), int(j)
        rel = En[:, j] @ np.linalg.inv(En[:, i])
        ok = np.isfinite(rel).all(axis=(1, 2)) & np.isfinite(s)
        rel = rel[ok]
        if len(rel) < min_templates:
            continue
        R, rkeep, rdisp = robust_rotation_mean(rel[:, :3, :3], tol_deg=rot_tol_deg)
        d, dkeep, ddisp = _robust_mean_direction(rel[:, :3, 3], tol_deg=dir_tol_deg)
        keep = rkeep & dkeep
        if keep.sum() < 2:
            keep = rkeep
        out[(i, j)] = dict(R=R, t_dir=d, scale=float(np.median(np.linalg.norm(rel[keep, :3, 3], axis=1))),
                           rot_disp_deg=rdisp, dir_disp_deg=ddisp, n_templates=int(len(rel)))
    return out


# ====================================================================== camera groups

def from_extrinsics(names, Ms, K=None, sizes=None, model='pinhole', **kw):
    """A :class:`~aniposelib.cameras.CameraGroup` from world->camera ``Ms [C,4,4]``
    (re-export of :func:`aniposelib.init._common.from_extrinsics`)."""
    return _from_extrinsics(names, Ms, K=K, sizes=sizes, model=model, **kw)


def to_cameragroup(result, names=None, K=None, sizes=None, template=None, model='pinhole', **kw):
    """CameraGroup from a :class:`VGGTAggregate` or :class:`VGGTPrediction` (aggregated first
    with ``aggregate(result)``). Intrinsics: ``template`` CameraGroup (copied, matched by
    name) > explicit ``K``/``sizes`` > the per-camera median VGGT intrinsics (zero
    distortion). Extra ``kw`` go to :func:`from_extrinsics` (``dist``, ``n_dist``, ``dtype``)."""
    agg = aggregate(result) if isinstance(result, VGGTPrediction) else result
    names = agg.names if names is None else names
    if template is not None:
        return _from_extrinsics(names, agg.Ms, template=template, **kw)
    K = agg.K if K is None else K
    sizes = agg.sizes if sizes is None else sizes
    if sizes is not None:
        sizes = np.asarray(sizes).astype(int)
        sizes = [int(v) for v in sizes] if sizes.ndim == 1 else [tuple(int(v) for v in s) for s in sizes]
    return _from_extrinsics(names, agg.Ms, K=K, sizes=sizes, model=model, **kw)


# ====================================================================== network

_MODEL_CACHE = {}


def free_model():
    """Drop cached VGGT models (``keep_model=True``) and empty the CUDA cache."""
    _MODEL_CACHE.clear()
    _empty_cuda()


def _empty_cuda():
    import gc
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:  # pragma: no cover
        pass


def weights_cached(model=DEFAULT_MODEL):
    """True if ``model``'s weights are in the local Hugging Face cache (never downloads)."""
    try:
        from huggingface_hub import try_to_load_from_cache
    except ImportError:
        return False
    for fname in ('model.safetensors', 'model.pt', 'pytorch_model.bin'):
        p = try_to_load_from_cache(model, fname)
        if isinstance(p, str):
            return True
    return False


def _resolve_device(device):
    import torch
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    return torch.device(device)


def _resolve_dtype(dtype, device):
    import torch
    if isinstance(dtype, torch.dtype):
        return dtype
    names = {'bf16': torch.bfloat16, 'bfloat16': torch.bfloat16, 'fp16': torch.float16,
             'float16': torch.float16, 'half': torch.float16, 'fp32': torch.float32,
             'float32': torch.float32}
    if dtype != 'auto':
        return names[str(dtype)]
    if device.type == 'cuda':
        cap = torch.cuda.get_device_capability(device)[0]
        return torch.bfloat16 if cap >= 8 else torch.float16  # sm_75 has no bf16
    return torch.float32


def _load_model(model, device, local_files_only):
    """VGGT from a repo id / local path, local cache first (EvoVision: avoids a ~40 s
    re-validation round trip and a hard dependency on huggingface.co being reachable)."""
    import torch
    if isinstance(model, torch.nn.Module):
        return model.to(device).eval(), False
    key = (str(model), str(device))
    if key in _MODEL_CACHE:
        return _MODEL_CACHE[key], True
    try:
        from vggt.models.vggt import VGGT
    except ImportError as exc:
        raise ImportError('aniposelib.init.vggt.predict needs the optional `vggt` package '
                          '(pip install "aniposelib-pytorch[vggt]" or the VGGT repo)') from exc
    if local_files_only == 'auto':
        try:
            net = VGGT.from_pretrained(model, local_files_only=True)
        except Exception as exc:
            log.info('VGGT %s not in the local cache (%s); downloading', model, type(exc).__name__)
            net = VGGT.from_pretrained(model)
    else:
        net = VGGT.from_pretrained(model, local_files_only=bool(local_files_only))
    return net.to(device).eval(), False


def _read_image(img):
    """Path or HxWx3 uint8 RGB array (HxW gray, HxWx4 RGBA blended on white) -> uint8 RGB."""
    if isinstance(img, (str, bytes)) or hasattr(img, '__fspath__'):
        try:
            from PIL import Image
            im = Image.open(img)
            if im.mode == 'RGBA':
                bg = Image.new('RGBA', im.size, (255, 255, 255, 255))
                im = Image.alpha_composite(bg, im)
            return np.asarray(im.convert('RGB'))
        except ImportError:
            import cv2
            arr = cv2.imread(str(img), cv2.IMREAD_COLOR)
            if arr is None:
                raise FileNotFoundError(img)
            return arr[..., ::-1].copy()
    arr = np.asarray(img)
    if arr.dtype != np.uint8:
        arr = np.clip(arr * (255.0 if arr.max() <= 1.0 else 1.0), 0, 255).astype(np.uint8)
    if arr.ndim == 2:
        arr = np.repeat(arr[..., None], 3, axis=-1)
    if arr.shape[-1] == 4:
        a = arr[..., 3:].astype('float64') / 255.0
        arr = (arr[..., :3] * a + 255.0 * (1 - a)).astype(np.uint8)
    return arr


def _resize(arr, size):
    try:
        from PIL import Image
        return np.asarray(Image.fromarray(arr).resize(size, Image.Resampling.BICUBIC))
    except ImportError:
        import cv2
        return cv2.resize(arr, size, interpolation=cv2.INTER_CUBIC)


def preprocess_image(img, mode='crop', target_size=TARGET_SIZE):
    """Mirror of VGGT ``load_and_preprocess_images`` for one image (path or HxWx3 uint8 RGB
    array): returns ``(float32 [3,H,W] in [0,1], A, (orig_w, orig_h))`` with ``A`` the
    original->model pixel map (before any batch padding)."""
    arr = _read_image(img)
    h, w = arr.shape[:2]
    A, (ow, oh), (nw, nh) = preprocess_transform((w, h), mode=mode, target_size=target_size)
    x = _resize(arr, (nw, nh)).astype(np.float32) / 255.0
    x = np.ascontiguousarray(x.transpose(2, 0, 1))
    if mode == 'crop' and nh > target_size:
        y0 = (nh - target_size) // 2
        x = x[:, y0:y0 + target_size]
    if mode == 'pad' and (oh > nh or ow > nw):
        top, left = int(A[1, 2]), int(A[0, 2])
        out = np.ones((3, oh, ow), np.float32)
        out[:, top:top + nh, left:left + nw] = x
        x = out
    return x, A, (w, h)


def _prepare_template(frames, mode, target_size):
    """Stack one template's frames as VGGT does (mixed shapes centre-padded with white)."""
    xs, As, sizes = zip(*[preprocess_image(f, mode=mode, target_size=target_size) for f in frames])
    H = max(x.shape[1] for x in xs)
    W = max(x.shape[2] for x in xs)
    out, Aout = [], []
    for x, A in zip(xs, As):
        hp, wp = H - x.shape[1], W - x.shape[2]
        if hp or wp:
            t, l = hp // 2, wp // 2
            y = np.ones((3, H, W), np.float32)
            y[:, t:t + x.shape[1], l:l + x.shape[2]] = x
            x = y
            A = A.copy()
            A[0, 2] += l
            A[1, 2] += t
        out.append(x)
        Aout.append(A)
    return np.stack(out), np.stack(Aout), np.array(sizes), (W, H)


def _run_template(net, images, device, dtype):
    """One forward pass (aggregator + camera head only) -> (extr [S,4,4], K_model [S,3,3])."""
    import torch
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri
    x = torch.as_tensor(images, device=device)
    with torch.no_grad():
        if dtype != torch.float32:
            with torch.autocast(device.type, dtype=dtype):
                tokens, _ = net.aggregator(x[None])
        else:
            tokens, _ = net.aggregator(x[None])
        pose_enc = net.camera_head(tokens)[-1]
        extr, intr = pose_encoding_to_extri_intri(pose_enc.float(), x.shape[-2:])
    S = x.shape[0]
    M = np.tile(np.eye(4), (S, 1, 1))
    M[:, :3, :] = extr[0].detach().cpu().numpy().astype('float64')
    return M, intr[0].detach().cpu().numpy().astype('float64')


class _Session:
    """Load VGGT once, run many templates, free on exit (unless keep_model)."""

    def __init__(self, model, device, dtype, local_files_only, keep_model):
        self.model, self.keep = model, keep_model
        self.device = _resolve_device(device)
        self.dtype = _resolve_dtype(dtype, self.device)
        self.local_files_only = local_files_only

    def __enter__(self):
        self.net, self.cached = _load_model(self.model, self.device, self.local_files_only)
        if self.keep and not self.cached and isinstance(self.model, str):
            _MODEL_CACHE[(str(self.model), str(self.device))] = self.net
        return self

    def __exit__(self, *exc):
        if not self.keep and isinstance(self.model, str):
            _MODEL_CACHE.pop((str(self.model), str(self.device)), None)
            self.net = None
            _empty_cuda()
        return False


def predict(frame_sets, model=DEFAULT_MODEL, device=None, dtype='auto', local_files_only='auto',
            names=None, keep_model=False, mode='crop', target_size=TARGET_SIZE):
    """Run VGGT on every template.

    Parameters
    ----------
    frame_sets : list of templates; each template is a list with one frame per camera (same
        camera order in every template): an image path, or an in-memory ``HxWx3`` **uint8 RGB**
        array (convert OpenCV BGR with ``frame[..., ::-1]``).
    model : HF repo id or local directory (``VGGT.from_pretrained``), or a loaded VGGT module.
    device : default cuda if available else cpu.
    dtype : ``'auto'`` = bf16 on cuda with compute capability >= 8, fp16 on older cuda,
        float32 on cpu; or ``'bf16'|'fp16'|'fp32'``/a torch dtype (autocast for the
        aggregator; the camera head runs in float32).
    local_files_only : ``'auto'`` tries the local HF cache first and downloads only if that
        fails; ``True`` never downloads; ``False`` always lets HF check the Hub.
    keep_model : keep the loaded model in a module cache for later calls (``free_model()``
        drops it); by default the ~5 GB model is freed and the CUDA cache emptied.
    mode : VGGT preprocessing, ``'crop'`` (VGGT default) or ``'pad'`` (keeps all pixels of
        portrait/mixed-orientation frames). Intrinsics are corrected for either.

    Returns a :class:`VGGTPrediction`.
    """
    import time
    frame_sets = [list(t) for t in frame_sets]
    if not frame_sets:
        raise ValueError('frame_sets is empty')
    C = len(frame_sets[0])
    if any(len(t) != C for t in frame_sets):
        raise ValueError('every template must have one frame per camera')
    names = [str(i) for i in range(C)] if names is None else [str(n) for n in names]
    t0 = time.time()
    ext, Ks, As, sizes, msize = [], [], [], None, None
    with _Session(model, device, dtype, local_files_only, keep_model) as sess:
        t_load = time.time() - t0
        for frames in frame_sets:
            imgs, A, sz, msize = _prepare_template(frames, mode, target_size)
            M, Km = _run_template(sess.net, imgs, sess.device, sess.dtype)
            ext.append(M)
            Ks.append(restore_intrinsics(Km, A))
            As.append(A)
            sizes = sz if sizes is None else sizes
        dev, dt = str(sess.device), str(sess.dtype).replace('torch.', '')
    stats = dict(device=dev, dtype=dt, load_s=t_load, total_s=time.time() - t0, mode=mode)
    return VGGTPrediction(extrinsics=np.stack(ext), intrinsics=np.stack(Ks), sizes=np.asarray(sizes),
                          names=names, model_size=tuple(int(v) for v in msize), transforms=np.stack(As),
                          stats=stats)


def predict_pairwise(frame_sets, pairs=None, model=DEFAULT_MODEL, device=None, dtype='auto',
                     local_files_only='auto', keep_model=False, mode='crop', min_templates=3, **kw):
    """VGGT on **camera pairs** (motor ``vggt_pairwise_rotations``): each two-image
    reconstruction has exactly one relative pose, so a failure stays local to its edge.
    ``pairs`` defaults to all ``i < j``. The model is loaded once for all pairs.
    Returns ``{(i, j): dict(R, t_dir, scale, rot_disp_deg, dir_disp_deg, n_templates)}``
    like :func:`pairwise_rotations` (``R``: camera i frame -> camera j frame); extra ``kw``
    go to :func:`pairwise_rotations`."""
    frame_sets = [list(t) for t in frame_sets]
    C = len(frame_sets[0])
    if pairs is None:
        pairs = [(i, j) for i in range(C) for j in range(i + 1, C)]
    out = {}
    with _Session(model, device, dtype, local_files_only, keep_model) as sess:
        for (i, j) in pairs:
            ext = []
            for frames in frame_sets:
                imgs, _, _, _ = _prepare_template([frames[i], frames[j]], mode, TARGET_SIZE)
                ext.append(_run_template(sess.net, imgs, sess.device, sess.dtype)[0])
            r = pairwise_rotations(np.stack(ext), pairs=[(0, 1)], min_templates=min_templates, **kw)
            if (0, 1) in r:
                out[(int(i), int(j))] = r[(0, 1)]
    return out
