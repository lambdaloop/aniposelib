"""Cross-camera correspondence: who is who across cameras (1.0, plan F6).

Two entry points, for the two shapes of the problem:

* :func:`associate_tracklets`: per-camera *tracked* detections (a 2D tracker such as ByteTrack
  gives ids that are consistent over time within one camera, with no meaning across cameras)
  -> groups of tracklets, one per subject, ready for bundle adjustment or triangulation.
  Correspondence is decided once per *tracklet* from a sample of frames, and then checked
  frame by frame (``verify``) and camera by camera (ambiguity abstention).
* :func:`associate_points`: one frame of untracked single-point detections per camera
  (e.g. box centres or animal centroids) -> 3D instances, by triangulating every cross-camera
  pair and growing the best-supported ones greedily.

API
---
``associate_tracklets(cgroup, detections, track_ids, min_cams=3, max_pair_error=40.0,
min_overlap=8, ambiguity_ratio=3.0, dedupe_ratio=0.2, *, min_len=10, sample_frames=40,
backbone_frames=120, verify=True, verify_mult=4.0, verify_floor=100.0, min_score=None,
min_points=4) -> Groups``
    ``detections``: one entry per camera (a list in group order, or a dict keyed by camera
    name). Each entry maps frame -> detections: a dict ``{frame: array (D, K, 2|3)}`` or a
    sequence indexed by frame; the value per frame is a ``(D, K, 2|3)`` array or a list of
    ``(K, 2|3)`` arrays (x, y[, score]); NaN = missing keypoint. ``track_ids`` has the same
    structure with ``(D,)`` integer ids per frame. Mismatched frames / lengths raise
    ``ValueError``. Keypoints with ``score < min_score`` are treated as missing; a detection
    needs ``min_points`` finite keypoints to enter a tracklet.

    Pipeline:

    1. tracklets per camera (``(camera, id)``); shorter than ``min_len`` frames are dropped;
    2. duplicate tracklets per camera (two ids on one body, overlapping in time) are merged when
       the median keypoint distance over >= 5 shared frames is below ``dedupe_ratio`` x the
       body's pixel size (the longest tracklet wins shared frames);
    3. every cross-camera tracklet pair sharing >= ``min_overlap`` frames is scored by the median
       two-view reprojection error over up to ``sample_frames`` uniformly sampled shared frames
       (one batched triangulation per camera pair);
    4. greedy union-find clustering over edges <= ``max_pair_error`` px: pairs sharing
       >= ``backbone_frames`` frames first, then by error; two components may not merge if any
       camera has both of them in the same frame (one camera cannot see one subject twice),
       which is what lets a fragmented subject come back together without merging two subjects;
    5. groups seen by fewer than ``min_cams`` cameras are dropped;
    6. ``verify``: per group, (camera, frame) cells whose median reprojection error exceeds
       ``max(verify_floor, verify_mult * that camera's own median)`` are removed (id-switch
       guard; relative, so it works on a rough calibration). Per frame the worst offending
       camera is removed first and the frame re-triangulated, so a switched camera does not
       also cost the cameras it pulled the triangulation away from. The camera count is
       re-checked afterwards;
    7. ambiguity abstention per (group, camera), only with >= ``min_cams`` cameras: the camera
       is held out, the group is triangulated from the others and projected into it; ``own`` =
       median distance of its own detections to the prediction, ``alt`` = the same for the
       closest *other* detection of that camera in those frames (duplicates of the chosen body,
       by ``dedupe_ratio``, are not competitors). If ``alt < ambiguity_ratio * own`` the camera
       abstains from that group (its observations are removed). Decided on up to
       ``sample_frames`` frames, once per (group, camera), never per frame.

``Groups``: ``.groups`` (list of :class:`Group`), ``.n_cams``, ``.cam_names``, ``.info``
(``n_tracklets``, ``n_pairs``, ``merged`` duplicates, ``abstentions``, ...), ``len()``,
iteration, ``to_dict()``, ``to_points()`` and ``dense()``.

``Groups.to_points(n_frames=None, score_thr=None, min_cams=None, frames='sample',
min_points=4) -> TrackletPoints(points [C,F,K,2], frame_id [F], group_id [F])``
    Rows for bundle adjustment / QA. ``frames='sample'`` spreads one budget of ``n_frames``
    rows uniformly over each group's frames (budget shared across groups, ``n_frames //
    n_groups`` each; None = every frame); ``frames='all'`` ignores the budget. A camera
    contributes to a row only with >= ``min_points`` finite keypoints (after ``score_thr``),
    and a row is kept only with >= ``min_cams`` (default: the association's) cameras. Unpacks
    as ``points, frame_id, group_id = groups.to_points(...)``.

``Groups.dense(n_frames=None) -> [G, C, T, K, 2]`` with ``T = n_frames`` (default: last frame
+ 1), frames indexed by their id.

``associate_points(cgroup, points_per_cam, max_res_px=30.0, min_views=2, *, max_instances=0,
scores=None) -> Association``
    ``points_per_cam``: list (len C) of ``(n_i, 2)`` pixel arrays (NaN rows = missing).
    Every cross-camera pair of detections is triangulated in one batched call; pairs whose
    two-view residual is above ``max_res_px`` are dropped; each candidate is ranked by
    (-support, residual) where support = how many *other* cameras have a detection within
    ``max_res_px`` of its reprojection (an integer count; a phantom pair of two different
    animals needs a third ray to land on it by chance). With ``scores`` (list of ``(n_i,)``)
    the ranking is (-support, -min pair score, residual). Candidates are accepted greedily:
    each grows into every other camera (nearest unused detection within the gate), is refit
    over its members (linear triangulation), and is kept if the refit median residual is
    <= ``max_res_px`` and it has >= ``max(2, min_views)`` members. Each detection is used once.
    ``min_views=1`` additionally emits leftover detections as single-view instances (NaN
    point, ``inf`` residual). ``max_instances`` caps the output (0 = no cap).
    Torch input gives torch ``points3d`` / ``residuals`` / ``labels``.

``Association``: ``points3d [M,3]``, ``members`` (list of ``{cam: det_idx}``), ``residuals [M]``,
``support [M]``, ``labels`` (per camera ``(n_i,)`` instance index, -1 = unassigned),
``to_points(points_per_cam) -> [C, M, 2]``, ``to_dict()``, ``len()``.

``associate_points_frames(cgroup, frames, **kw) -> list of Association`` loops over frames
(``frames`` = list over frames of ``points_per_cam``).

The associate -> BA -> re-associate loop
----------------------------------------
Both functions decide correspondence with the *current* calibration, which is circular when
the goal is to calibrate. It is benign in practice because a wrong correspondence is a large
error that no rough calibration produces for the right one, and the thresholds are generous
(``max_pair_error``) or relative (``verify``). For self-calibration from subjects, iterate::

    groups = associate_tracklets(cgroup, dets, ids, max_pair_error=150)   # loose, rough rig
    points, frame_id, group_id = groups.to_points(n_frames=400, score_thr=0.5)
    cgroup.bundle_adjust_iter(points.reshape(len(cgroup.cameras), -1, 2), ...)
    groups = associate_tracklets(cgroup, dets, ids)                       # tighter, refined rig
    ...                                                                   # until groups stop changing

Provenance
----------
Ported from the EvoVision gait pipeline (``pipeline/identify_main_3d.py``: ``build_tracklets``,
``merge_duplicate_tracklets``, ``_tracklet_pair_errors``, ``_cluster_tracklets``,
``_verify_and_repair``, and ``pipeline/calib_reuse.py``: ``tracklet_points``,
``_ambiguity_margin``, ``_sample_frames``, ``_view_rows``) and from tailcyclenet
(``tailcyclenet/infer/associate.py``: ``associate``, ``_support_count``, ``_fast_candidates``),
generalised (no logging, no skeleton subsets, arbitrary keypoint count) and batched on the
packed-camera kernels of :mod:`aniposelib.geometry` / :mod:`aniposelib.triangulation`.
"""

import itertools
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, NamedTuple, Optional, Tuple

import numpy as np
import torch

from . import geometry as G
from . import triangulation as T

__all__ = ['associate_tracklets', 'Groups', 'Group', 'TrackletPoints',
           'associate_points', 'associate_points_frames', 'Association']

# A pair of tracklets in one camera needs this many shared frames before the duplicate test
# (median separation / body size) means anything.
_DUP_MIN_SHARED = 5


# ------------------------------------------------------------------ helpers

def _jsonable(x):
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    if isinstance(x, np.ndarray):
        return [_jsonable(v) for v in x.tolist()] if x.dtype == object else _clean_list(x.tolist())
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating, float)):
        x = float(x)
        return x if np.isfinite(x) else None
    if isinstance(x, (np.bool_,)):
        return bool(x)
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    return x


def _clean_list(v):
    if isinstance(v, list):
        return [_clean_list(a) for a in v]
    if isinstance(v, float) and not np.isfinite(v):
        return None
    return v


def _nanmedian(x, axis=None):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', category=RuntimeWarning)
        return np.nanmedian(x, axis=axis)


def _sample(idx, n):
    """``n`` elements of ``idx``, uniformly spread (all of them when ``n`` is None or larger)."""
    idx = np.asarray(idx)
    if n is None or len(idx) <= n:
        return idx
    return idx[np.linspace(0, len(idx) - 1, max(int(n), 1)).astype(int)]


def _body_size(xy):
    """Pixel size of one detection ``(..., K, 2)``: norm of its bounding-box diagonal."""
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', category=RuntimeWarning)
        span = np.nanmax(xy, axis=-2) - np.nanmin(xy, axis=-2)
    return np.linalg.norm(span, axis=-1)


def _per_camera(x, names, what):
    if isinstance(x, dict) and all(n in x for n in names) and len(names) > 0:
        return [x[n] for n in names]
    if isinstance(x, dict):
        raise ValueError(f'{what}: dict keys {sorted(map(str, x))[:5]}... do not match the camera '
                         f'names {names}')
    x = list(x)
    if len(x) != len(names):
        raise ValueError(f'{what}: got {len(x)} cameras, the group has {len(names)}')
    return x


def _frame_items(per_cam):
    if per_cam is None:
        return []
    if isinstance(per_cam, dict):
        return [(int(k), v) for k, v in per_cam.items()]
    return list(enumerate(per_cam))


def _as_dets(v):
    """One frame's detections -> float64 ``(D, K, ch)``."""
    if v is None:
        return None
    if isinstance(v, torch.Tensor):
        v = v.detach().cpu().numpy()
    if isinstance(v, (list, tuple)):
        if len(v) == 0:
            return None
        v = np.stack([np.asarray(a.detach().cpu().numpy() if isinstance(a, torch.Tensor) else a,
                                 dtype=np.float64) for a in v])
    v = np.asarray(v, dtype=np.float64)
    if v.size == 0:
        return None
    if v.ndim == 2:
        v = v[None]
    if v.ndim != 3 or v.shape[-1] not in (2, 3):
        raise ValueError(f'detections must be (D, K, 2|3) per frame, got {v.shape}')
    return v


# ------------------------------------------------------------------ result types

class TrackletPoints(NamedTuple):
    """``points [C,F,K,2]``, ``frame_id [F]``, ``group_id [F]`` (unpacks as a tuple)."""
    points: np.ndarray
    frame_id: np.ndarray
    group_id: np.ndarray


@dataclass
class Group:
    """One subject: tracklets of several cameras that agree geometrically.

    ``members``: ``[(cam_idx, track_id)]`` (representative ids; ``duplicates`` maps a member to
    the ids merged into it); ``cameras``: cameras still contributing after verify/abstention;
    ``frames``: frame ids with at least one observation; ``n_frames``; ``dropped``: per camera,
    frame ids removed by the verify step; ``abstained``: ``{cam: (own_px, alt_px)}`` cameras
    removed by the ambiguity guard; ``margins``: ``{cam: (own_px, alt_px)}`` for every checked
    camera (alt None = no competitor); ``pair_errors``: ``[((c, t), (c, t), px)]`` the edges
    inside the group."""
    id: int
    members: List[Tuple[int, int]]
    cameras: List[int]
    frames: np.ndarray
    dropped: Dict[int, np.ndarray] = field(default_factory=dict)
    abstained: Dict[int, Tuple[float, float]] = field(default_factory=dict)
    margins: Dict[int, Tuple[float, Optional[float]]] = field(default_factory=dict)
    pair_errors: List[Tuple[Tuple[int, int], Tuple[int, int], float]] = field(default_factory=list)
    duplicates: Dict[Tuple[int, int], List[int]] = field(default_factory=dict)
    # per camera: (frame_ids [n], det_idx [n], xyz [n, K, 2|3]) of the kept observations
    _obs: Dict[int, Tuple[np.ndarray, np.ndarray, np.ndarray]] = field(
        default_factory=dict, repr=False, compare=False)

    @property
    def n_frames(self):
        return len(self.frames)

    def observations(self, cam):
        """``(frame_ids, det_idx, keypoints [n, K, 2|3])`` this group uses in camera ``cam``
        (``det_idx`` indexes the input detections of that frame)."""
        if cam not in self._obs:
            return np.zeros(0, int), np.zeros(0, int), None
        return self._obs[cam]

    def to_dict(self):
        return {
            'id': int(self.id),
            'members': [[int(c), int(t)] for c, t in self.members],
            'cameras': [int(c) for c in self.cameras],
            'n_frames': int(self.n_frames),
            'frames': _jsonable(self.frames),
            'dropped': {str(c): _jsonable(v) for c, v in self.dropped.items()},
            'abstained': {str(c): _jsonable(v) for c, v in self.abstained.items()},
            'margins': {str(c): _jsonable(v) for c, v in self.margins.items()},
            'pair_errors': [[[int(a[0]), int(a[1])], [int(b[0]), int(b[1])], float(e)]
                            for a, b, e in self.pair_errors],
            'duplicates': [[int(c), int(t), [int(x) for x in v]]
                           for (c, t), v in self.duplicates.items()],
        }


@dataclass
class Groups:
    """Result of :func:`associate_tracklets`. Iterable over :class:`Group`."""
    groups: List[Group]
    n_cams: int
    cam_names: List[str]
    n_keypoints: int
    min_cams: int
    info: Dict[str, Any] = field(default_factory=dict)

    def __len__(self):
        return len(self.groups)

    def __iter__(self):
        return iter(self.groups)

    def __getitem__(self, i):
        return self.groups[i]

    def to_dict(self):
        return {'n_cams': int(self.n_cams), 'cam_names': list(self.cam_names),
                'n_keypoints': int(self.n_keypoints), 'min_cams': int(self.min_cams),
                'groups': [g.to_dict() for g in self.groups], 'info': _jsonable(self.info)}

    def _cell(self, xyz, score_thr):
        xy = xyz[..., :2].copy()
        if score_thr is not None and xyz.shape[-1] > 2:
            xy[xyz[..., 2] < score_thr] = np.nan
        return xy

    def to_points(self, n_frames=None, score_thr=None, min_cams=None, frames='sample',
                  min_points=4):
        """Corresponded rows ``TrackletPoints(points [C,F,K,2], frame_id [F], group_id [F])``.

        See the module docstring. Rows are ordered by group, then frame."""
        if frames not in ('sample', 'all'):
            raise ValueError("frames must be 'sample' or 'all'")
        min_cams = self.min_cams if min_cams is None else int(min_cams)
        C, K = self.n_cams, self.n_keypoints
        per_group = None
        if frames == 'sample' and n_frames is not None and len(self.groups):
            per_group = max(1, int(n_frames) // len(self.groups))
        rows, fids, gids = [], [], []
        for g in self.groups:
            sel = _sample(g.frames, per_group)
            if len(sel) == 0:
                continue
            pts = np.full((C, len(sel), K, 2), np.nan)
            for c, (fr, _di, xyz) in g._obs.items():
                pos = np.searchsorted(sel, fr)
                ok = (pos < len(sel)) & (sel[np.minimum(pos, len(sel) - 1)] == fr)
                pts[c, pos[ok]] = self._cell(xyz[ok], score_thr)
            good = np.isfinite(pts).all(-1).sum(-1) >= min_points        # [C, F]
            pts[~good] = np.nan
            keep = good.sum(0) >= min_cams
            rows.append(pts[:, keep])
            fids.append(sel[keep])
            gids.append(np.full(int(keep.sum()), g.id))
        if not rows:
            return TrackletPoints(np.full((C, 0, K, 2), np.nan), np.zeros(0, int), np.zeros(0, int))
        return TrackletPoints(np.concatenate(rows, 1), np.concatenate(fids).astype(int),
                              np.concatenate(gids).astype(int))

    def dense(self, n_frames=None, score_thr=None):
        """``[G, C, T, K, 2]`` keypoints indexed by frame id (``T`` default: last frame + 1)."""
        if n_frames is None:
            n_frames = 1 + max([int(g.frames.max()) for g in self.groups if len(g.frames)],
                               default=-1)
        out = np.full((len(self.groups), self.n_cams, int(n_frames), self.n_keypoints, 2), np.nan)
        for gi, g in enumerate(self.groups):
            for c, (fr, _di, xyz) in g._obs.items():
                ok = (fr >= 0) & (fr < n_frames)
                out[gi, c, fr[ok]] = self._cell(xyz[ok], score_thr)
        return out


# ------------------------------------------------------------------ tracklets

@dataclass
class _Tracklet:
    cam: int
    tid: int
    frames: np.ndarray      # compact frame index, sorted
    det: np.ndarray         # detection index within the frame
    xyz: np.ndarray         # [n, K, ch] raw (masked by min_score)
    xy: np.ndarray          # [n, K, 2] geometry (NaN = missing)
    und: Optional[np.ndarray] = None
    merged: List[int] = field(default_factory=list)


def _parse_inputs(names, detections, track_ids, min_score):
    """-> per camera: {frame_id: (dets [D,K,ch] masked, ids [D])}, K, ch."""
    dets = _per_camera(detections, names, 'detections')
    tids = _per_camera(track_ids, names, 'track_ids')
    out, K, ch = [], None, None
    for c, (dc, tc) in enumerate(zip(dets, tids)):
        tmap = {}
        for f, v in _frame_items(tc):
            v = None if v is None else np.asarray(
                v.detach().cpu().numpy() if isinstance(v, torch.Tensor) else v).reshape(-1)
            tmap[int(f)] = v
        frames = {}
        for f, v in _frame_items(dc):
            d = _as_dets(v)
            if d is None:
                continue
            ids = tmap.get(int(f))
            if ids is None or len(ids) != len(d):
                raise ValueError(f'camera {names[c]!r} frame {f}: {0 if ids is None else len(ids)} '
                                 f'track ids for {len(d)} detections')
            if K is None:
                K, ch = d.shape[1], d.shape[2]
            if d.shape[1] != K:
                raise ValueError(f'camera {names[c]!r} frame {f}: {d.shape[1]} keypoints, '
                                 f'expected {K}')
            d = d.copy()
            if min_score is not None and d.shape[-1] > 2:
                d[d[..., 2] < min_score, :2] = np.nan
            frames[int(f)] = (d, ids.astype(np.int64))
        for f, ids in tmap.items():
            if ids is not None and len(ids) and f not in frames:
                raise ValueError(f'camera {names[c]!r} frame {f}: {len(ids)} track ids but no '
                                 f'detections')
        out.append(frames)
    return out, (K or 0), (ch or 2)


def _build_tracklets(c, frames, fidx, min_points, min_len):
    by_tid = {}
    for f, (d, ids) in frames.items():
        n_ok = np.isfinite(d[..., :2]).all(-1).sum(-1)
        for i in np.flatnonzero(n_ok >= min_points):
            by_tid.setdefault(int(ids[i]), []).append((fidx[f], int(i), d[i]))
    out = {}
    for t, rows in by_tid.items():
        rows.sort(key=lambda r: r[0])
        fr = np.array([r[0] for r in rows])
        if len(np.unique(fr)) != len(fr):
            raise ValueError(f'camera {c}: track id {t} appears twice in one frame')
        if len(fr) < min_len:
            continue
        xyz = np.stack([r[2] for r in rows])
        out[t] = _Tracklet(c, t, fr, np.array([r[1] for r in rows]), xyz, xyz[..., :2].copy())
    return out


def _merge_duplicates(tracks, dedupe_ratio):
    """Fold tracklets of one camera that are the same body (see the module docstring)."""
    tids = sorted(tracks, key=lambda t: (-len(tracks[t].frames), t))
    parent = {t: t for t in tids}

    def find(t):
        while parent[t] != t:
            parent[t] = parent[parent[t]]
            t = parent[t]
        return t

    merges = []
    for i, ta in enumerate(tids):
        for tb in tids[i + 1:]:
            if find(ta) == find(tb):
                continue
            A, B = tracks[ta], tracks[tb]
            shared, ia, ib = np.intersect1d(A.frames, B.frames, return_indices=True)
            if len(shared) < _DUP_MIN_SHARED:
                continue
            a, b = A.xy[ia], B.xy[ib]
            dist = _nanmedian(np.linalg.norm(a - b, axis=-1))
            size = _nanmedian(_body_size(a))
            if np.isfinite(dist) and size > 0 and dist / size < dedupe_ratio:
                merges.append((ta, tb, float(dist), float(size), int(len(shared))))
                parent[find(tb)] = find(ta)
    if not merges:
        return tracks, []
    out = {}
    for t in tids:                    # longest first, so it wins shared frames
        root = find(t)
        if root not in out:
            out[root] = tracks[t]
            continue
        R, S = out[root], tracks[t]
        new = ~np.isin(S.frames, R.frames)
        fr = np.concatenate([R.frames, S.frames[new]])
        order = np.argsort(fr, kind='stable')
        out[root] = _Tracklet(R.cam, R.tid, fr[order],
                              np.concatenate([R.det, S.det[new]])[order],
                              np.concatenate([R.xyz, S.xyz[new]])[order],
                              np.concatenate([R.xy, S.xy[new]])[order],
                              merged=R.merged + [t] + S.merged)
    return out, merges


def _pair_errors(P, per_cam, min_overlap, sample_frames):
    """Median two-view reprojection error of every cross-camera tracklet pair:
    ``{(key_a, key_b): (err, n_shared)}``; one triangulation per camera pair."""
    out = {}
    for ca, cb in itertools.combinations(range(len(per_cam)), 2):
        combos, ra, rb, xa, xb = [], [], [], [], []
        for ta, A in per_cam[ca].items():
            for tb, B in per_cam[cb].items():
                shared, ia, ib = np.intersect1d(A.frames, B.frames, return_indices=True)
                if len(shared) < min_overlap:
                    continue
                take = _sample(np.arange(len(shared)), sample_frames)
                combos.append(((ca, ta), (cb, tb), len(take), len(shared)))
                ra.append(A.und[ia[take]]); rb.append(B.und[ib[take]])
                xa.append(A.xy[ia[take]]); xb.append(B.xy[ib[take]])
        if not combos:
            continue
        und = np.stack([np.concatenate(ra), np.concatenate(rb)])     # [2, S, K, 2]
        uv = np.stack([np.concatenate(xa), np.concatenate(xb)])
        S, K = und.shape[1:3]
        sub = P.subset([ca, cb])
        X = T.triangulate_linear(sub, und.reshape(2, -1, 2), undistort=False)
        err = G.reproj_err(sub, X, uv.reshape(2, -1, 2))              # [2, S*K]
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=RuntimeWarning)
            err = np.nanmean(err, axis=0).reshape(S, K)
        at = 0
        for ka, kb, n, n_shared in combos:
            e = _nanmedian(err[at:at + n])
            at += n
            if np.isfinite(e):
                out[(ka, kb)] = (float(e), int(n_shared))
    return out


def _cluster(keys, occ, cams, errors, max_error, backbone_frames):
    """Greedy union-find over the edges, backbone tier first, with the same-camera-same-frame
    exclusion."""
    comp = list(range(len(keys)))
    occ_by = {i: {cams[i]: occ[i].copy()} for i in range(len(keys))}
    index = {k: i for i, k in enumerate(keys)}

    def find(i):
        while comp[i] != i:
            comp[i] = comp[comp[i]]
            i = comp[i]
        return i

    ranked = sorted((0 if n_shared >= backbone_frames else 1, e, index[ka], index[kb])
                    for (ka, kb), (e, n_shared) in errors.items() if e <= max_error)
    for _tier, _e, ia, ib in ranked:
        ra, rb = find(ia), find(ib)
        if ra == rb:
            continue
        if any(np.any(occ_by[ra][c] & occ_by[rb][c]) for c in occ_by[ra].keys() & occ_by[rb].keys()):
            continue
        comp[rb] = ra
        for c, m in occ_by.pop(rb).items():
            if c in occ_by[ra]:
                occ_by[ra][c] |= m
            else:
                occ_by[ra][c] = m
    groups = {}
    for i in range(len(keys)):
        groups.setdefault(find(i), []).append(keys[i])
    return list(groups.values())


def _ambiguity_margin(chosen, alternatives, projected, dup_ratio=0.0):
    """``(own_px, alt_px)`` for one camera of one group (``alt_px`` None without competitors).

    ``chosen``, ``projected``: ``[N, K, 2]`` over frames; ``alternatives``: per frame a
    ``[D, K, 2]`` array of the other detections of that camera."""
    own_f = _nanmedian(np.linalg.norm(chosen - projected, axis=-1), axis=-1)
    own = float(_nanmedian(own_f)) if np.isfinite(own_f).any() else np.nan
    alt_f = []
    for fi, others in enumerate(alternatives):
        if others is None or len(others) == 0:
            continue
        mine = chosen[fi]
        if dup_ratio:
            size = float(_body_size(mine))
            if size > 0:
                sep = _nanmedian(np.linalg.norm(others - mine[None], axis=-1), axis=-1)
                others = others[~(np.isfinite(sep) & (sep / size < dup_ratio))]
        if len(others) == 0:
            continue
        d = _nanmedian(np.linalg.norm(others - projected[fi][None], axis=-1), axis=-1)
        d = d[np.isfinite(d)]
        if len(d):
            alt_f.append(float(d.min()))
    alt = float(np.median(alt_f)) if alt_f else None
    return own, alt


def associate_tracklets(cgroup, detections, track_ids, min_cams=3, max_pair_error=40.0,
                        min_overlap=8, ambiguity_ratio=3.0, dedupe_ratio=0.2, *, min_len=10,
                        sample_frames=40, backbone_frames=120, verify=True, verify_mult=4.0,
                        verify_floor=100.0, min_score=None, min_points=4):
    """Group per-camera tracklets into cross-camera subjects. See the module docstring."""
    P = G.pack(cgroup)
    names = list(cgroup.get_names()) if hasattr(cgroup, 'get_names') else \
        [str(i) for i in range(P.K.shape[0])]
    C = len(names)
    parsed, K, ch = _parse_inputs(names, detections, track_ids, min_score)
    frame_ids = np.array(sorted({f for fr in parsed for f in fr}), dtype=np.int64)
    fidx = {int(f): i for i, f in enumerate(frame_ids)}
    F = len(frame_ids)
    info = dict(n_frames=int(F))

    # 1-2. tracklets, duplicates
    per_cam, merged = [], []
    for c in range(C):
        tr = _build_tracklets(c, parsed[c], fidx, min_points, min_len)
        tr, m = _merge_duplicates(tr, dedupe_ratio)
        merged += [dict(cam=c, kept=a, merged=b, dist_px=d, size_px=s, n_shared=n)
                   for a, b, d, s, n in m]
        if tr:                         # undistort once per camera
            stack = np.concatenate([t.xy for t in tr.values()])
            und = G.undistort(P.subset([c]), stack[None])[0]
            at = 0
            for t in tr.values():
                t.und = und[at:at + len(t.frames)]
                at += len(t.frames)
        per_cam.append(tr)
    keys = [(c, t) for c in range(C) for t in sorted(per_cam[c])]
    info.update(n_tracklets=len(keys), n_tracklets_per_camera=[len(t) for t in per_cam],
                merged=merged)
    empty = Groups([], C, names, K, min_cams, info)
    if not keys:
        info.update(n_pairs=0, n_edges=0, abstentions=[], n_verify_dropped=0, n_rejected=0)
        return empty
    occ = np.zeros((len(keys), F), dtype=bool)
    for i, (c, t) in enumerate(keys):
        occ[i, per_cam[c][t].frames] = True

    # 3-4. pair errors, clustering
    errors = _pair_errors(P, per_cam, min_overlap, sample_frames)
    info['n_pairs'] = len(errors)
    info['n_edges'] = int(sum(e <= max_pair_error for e, _ in errors.values()))
    clusters = _cluster(keys, occ, [k[0] for k in keys], errors, max_pair_error, backbone_frames)

    groups, abst, n_drop_cells, n_rejected = [], [], 0, 0
    for members in clusters:
        if len({c for c, _ in members}) < min_cams:
            continue
        # gather [C, Fg, K, ...] over the group's frames
        gf = np.unique(np.concatenate([per_cam[c][t].frames for c, t in members]))
        Fg = len(gf)
        xy = np.full((C, Fg, K, 2), np.nan)
        und = np.full((C, Fg, K, 2), np.nan)
        xyz = np.full((C, Fg, K, ch), np.nan)
        det = np.full((C, Fg), -1, dtype=np.int64)
        for c, t in members:
            tr = per_cam[c][t]
            pos = np.searchsorted(gf, tr.frames)
            xy[c, pos], und[c, pos], xyz[c, pos], det[c, pos] = tr.xy, tr.und, tr.xyz, tr.det
        present = np.isfinite(xy[..., 0]).any(-1)                     # [C, Fg]
        dropped = {}

        # 6. verify and repair: per frame, drop the worst camera while it is above its own
        # relative limit, re-triangulate and re-check (so one switched camera does not also
        # cost the cameras it pulled the triangulation away from)
        if verify:
            bad = np.zeros_like(present)
            limit = None
            for _round in range(max(C - 2, 1)):
                use = np.where((present & ~bad)[..., None, None], und, np.nan)
                X = T.triangulate_linear(P, use.reshape(C, -1, 2), undistort=False)
                err = G.reproj_err(P, X, xy.reshape(C, -1, 2)).reshape(C, Fg, K)
                pcf = _nanmedian(err, axis=2)                         # [C, Fg]; inf = behind
                if limit is None:
                    base = _nanmedian(np.where(np.isfinite(pcf), pcf, np.nan), axis=1)
                    limit = np.maximum(verify_floor,
                                       verify_mult * np.nan_to_num(base, nan=0.0))[:, None]
                over = np.where(present & ~bad & (pcf > limit), pcf / limit, 0.0)
                worst = np.argmax(over, axis=0)                       # [Fg]
                hit = over[worst, np.arange(Fg)] > 0
                if not hit.any():
                    break
                bad[worst[hit], np.flatnonzero(hit)] = True
            for c in np.flatnonzero(bad.any(1)):
                dropped[int(c)] = frame_ids[gf[bad[c]]]
            n_drop_cells += int(bad.sum())
            present &= ~bad
            if int(present.any(1).sum()) < min_cams:
                n_rejected += 1
                continue

        # 7. ambiguity abstention, per (group, camera)
        margins, abstained = {}, {}
        cams_present = [int(c) for c in np.flatnonzero(present.any(1))]
        if len(cams_present) >= min_cams and ambiguity_ratio:
            sel = _sample(np.arange(Fg), sample_frames)
            for c in list(cams_present):
                rows = sel[present[c, sel]]
                if len(rows) == 0:
                    continue
                held = np.where(present[:, rows, None, None], und[:, rows], np.nan)
                held[c] = np.nan
                Xh = T.triangulate_linear(P, held.reshape(C, -1, 2), undistort=False)
                proj, _ = G.project(P.subset([c]), Xh)
                proj = proj[0].reshape(len(rows), K, 2)
                alts = []
                for r in rows:
                    d, _ids = parsed[c][int(frame_ids[gf[r]])]
                    others = np.delete(d[..., :2], det[c, r], axis=0)
                    alts.append(others)
                own, alt = _ambiguity_margin(xy[c, rows], alts, proj, dup_ratio=dedupe_ratio)
                margins[c] = (own, alt)
                if alt is not None and np.isfinite(own) and alt < ambiguity_ratio * own:
                    abstained[c] = (own, alt)
                    abst.append(dict(group=len(groups), cam=c, own_px=own, alt_px=alt))
                    present[c] = False
                    cams_present.remove(c)
            if len(cams_present) < min_cams:
                n_rejected += 1
                for a in abst:
                    if a['group'] == len(groups):
                        a['group'] = None
                continue

        obs = {}
        for c in cams_present:
            m = present[c]
            obs[c] = (frame_ids[gf[m]], det[c, m], xyz[c, m])
        live = present.any(0)
        mset = set(members)
        g = Group(id=len(groups), members=sorted(members), cameras=sorted(cams_present),
                  frames=frame_ids[gf[live]], dropped=dropped, abstained=abstained,
                  margins=margins,
                  pair_errors=sorted([(ka, kb, e) for (ka, kb), (e, _n) in errors.items()
                                      if ka in mset and kb in mset and e <= max_pair_error],
                                     key=lambda x: x[2]),
                  duplicates={(c, t): list(per_cam[c][t].merged) for c, t in members
                              if per_cam[c][t].merged},
                  _obs=obs)
        groups.append(g)
    info.update(abstentions=abst, n_verify_dropped=n_drop_cells, n_rejected=n_rejected,
                n_clusters=len(clusters))
    return Groups(groups, C, names, K, min_cams, info)


# ------------------------------------------------------------------ per-frame points

@dataclass
class Association:
    """Result of :func:`associate_points` (see the module docstring)."""
    points3d: Any
    members: List[Dict[int, int]]
    residuals: Any
    support: Any
    labels: List[Any]

    def __len__(self):
        return len(self.members)

    def to_points(self, points_per_cam):
        """``[C, M, 2]``: each instance's member detections (NaN where a camera has none)."""
        torch_out = any(isinstance(p, torch.Tensor) for p in points_per_cam)
        pts = [np.asarray(p.detach().cpu().numpy() if isinstance(p, torch.Tensor) else p,
                          dtype=np.float64).reshape(-1, 2) for p in points_per_cam]
        out = np.full((len(pts), len(self.members), 2), np.nan)
        for m, mem in enumerate(self.members):
            for c, i in mem.items():
                out[c, m] = pts[c][i]
        return torch.as_tensor(out) if torch_out else out

    def to_dict(self):
        return {'points3d': _jsonable(self.points3d), 'residuals': _jsonable(self.residuals),
                'support': _jsonable(self.support),
                'members': [{str(c): int(i) for c, i in m.items()} for m in self.members],
                'labels': [_jsonable(l) for l in self.labels]}


def associate_points(cgroup, points_per_cam, max_res_px=30.0, min_views=2, *, max_instances=0,
                     scores=None):
    """Per-frame multi-instance association of single-point detections. See the module docstring."""
    return _associate_batch(cgroup, [points_per_cam], max_res_px=max_res_px, min_views=min_views,
                            max_instances=max_instances, scores=scores)[0]


def associate_points_frames(cgroup, frames, batch_frames=256, **kwargs):
    """:func:`associate_points` over a list of frames (each a ``points_per_cam`` list). Frames are
    processed ``batch_frames`` at a time: one undistortion, one pair triangulation and one refit
    per greedy round for the whole batch (the same result as one call per frame)."""
    P = cgroup if isinstance(cgroup, G.PackedCameras) else G.pack(cgroup)
    frames = list(frames)
    out = []
    for s in range(0, len(frames), max(1, int(batch_frames))):
        out += _associate_batch(P, frames[s:s + max(1, int(batch_frames))], **kwargs)
    return out


def _prep_frame(points_per_cam, C, scores):
    if len(points_per_cam) != C:
        raise ValueError(f'points_per_cam has {len(points_per_cam)} cameras, the group has {C}')
    torch_out = any(isinstance(p, torch.Tensor) for p in points_per_cam)
    device = next((p.device for p in points_per_cam if isinstance(p, torch.Tensor)), None)
    pts = [np.asarray(p.detach().cpu().numpy() if isinstance(p, torch.Tensor) else p,
                      dtype=np.float64).reshape(-1, 2) for p in points_per_cam]
    n = np.array([len(p) for p in pts])
    N = max(int(n.max()) if C else 0, 1)
    pad = np.full((C, N, 2), np.nan)
    for c, p in enumerate(pts):
        pad[c, :len(p)] = p
    spad = None
    if scores is not None:
        spad = np.full((C, N), -np.inf)
        for c, s in enumerate(scores):
            s = np.asarray(s.detach().cpu().numpy() if isinstance(s, torch.Tensor) else s,
                           dtype=np.float64).reshape(-1)
            spad[c, :len(s)] = s
    return dict(torch_out=torch_out, device=device, n=n, pad=pad, spad=spad)


def _pair_candidates(valid, C):
    ca_l, ia_l, cb_l, ib_l = [], [], [], []
    for ca, cb in itertools.combinations(range(C), 2):
        ia, ib = np.flatnonzero(valid[ca]), np.flatnonzero(valid[cb])
        if len(ia) == 0 or len(ib) == 0:
            continue
        A, B = np.meshgrid(ia, ib, indexing='ij')
        ca_l.append(np.full(A.size, ca)); cb_l.append(np.full(A.size, cb))
        ia_l.append(A.ravel()); ib_l.append(B.ravel())
    if not ca_l:
        return None
    return tuple(np.concatenate(v) for v in (ca_l, ia_l, cb_l, ib_l))


def _greedy_frame(C, fr, und, valid, cand, X, proj, depth, max_res_px, min_views, max_instances):
    """Greedy instance growing of one frame (a generator: yields refit requests ``(sub [C,2]
    undistorted, obs [C,2] pixels)`` and receives ``(Xr [3], e [C])``); returns the
    :class:`Association`."""
    pad, spad, n = fr['pad'], fr['spad'], fr['n']
    members, points3d, residuals, support = [], [], [], []
    used = ~valid
    if cand is not None:
        ca_, ia_, cb_, ib_ = cand
        M = len(ca_)
        r_ = np.arange(M)
        proj = np.where((depth > 0)[..., None], proj, np.nan)
        pair_res = 0.5 * (np.linalg.norm(proj[ca_, r_] - pad[ca_, ia_], axis=-1)
                          + np.linalg.norm(proj[cb_, r_] - pad[cb_, ib_], axis=-1))
        ok = np.isfinite(pair_res) & (pair_res <= max_res_px) & np.isfinite(X).all(-1)
        # distances of every candidate's reprojection to every detection [C, M, N]
        dist = np.linalg.norm(proj[:, :, None] - pad[:, None], axis=-1)
        dist = np.where(valid[:, None], dist, np.inf)
        dist = np.nan_to_num(dist, nan=np.inf)
        near = (dist < max_res_px).any(-1)                             # [C, M]
        near[ca_, r_] = False
        near[cb_, r_] = False
        sup = near.sum(0)
        cand_ok = np.flatnonzero(ok)
        if spad is not None:
            ps = np.minimum(spad[ca_, ia_], spad[cb_, ib_])
            order = cand_ok[np.lexsort((pair_res[cand_ok], -ps[cand_ok], -sup[cand_ok]))]
        else:
            order = cand_ok[np.lexsort((pair_res[cand_ok], -sup[cand_ok]))]
        for k in order:
            a, ia, b, ib = int(ca_[k]), int(ia_[k]), int(cb_[k]), int(ib_[k])
            if used[a, ia] or used[b, ib]:
                continue
            mem = {a: ia, b: ib}
            for c in range(C):
                if c in mem:
                    continue
                d = np.where(used[c], np.inf, dist[c, k])
                j = int(np.argmin(d))
                if d[j] < max_res_px:
                    mem[c] = j
            if len(mem) < max(2, min_views):
                continue
            cams = sorted(mem)
            sub = np.full((C, 2), np.nan)
            obs = np.full((C, 2), np.nan)
            for c in cams:
                sub[c] = und[c, mem[c]]
                obs[c] = pad[c, mem[c]]
            Xr, e = yield sub, obs
            res = float(np.median(e[cams]))
            if not res <= max_res_px:
                continue
            members.append({c: mem[c] for c in cams})
            points3d.append(Xr)
            residuals.append(res)
            support.append(int(sup[k]))
            for c in cams:
                used[c, mem[c]] = True
            if max_instances and len(members) >= max_instances:
                break
    if min_views == 1:
        for c in range(C):
            for i in range(n[c]):
                if max_instances and len(members) >= max_instances:
                    break
                if used[c, i]:
                    continue
                members.append({c: i})
                points3d.append(np.full(3, np.nan))
                residuals.append(np.inf)
                support.append(0)
                used[c, i] = True
    labels = [np.full(n[c], -1, dtype=np.int64) for c in range(C)]
    for m, mem in enumerate(members):
        for c, i in mem.items():
            labels[c][i] = m
    p3 = np.array(points3d, dtype=np.float64).reshape(-1, 3)
    res = np.array(residuals, dtype=np.float64)
    sup = np.array(support, dtype=np.int64)
    if fr['torch_out']:
        device = fr['device']
        p3, res, sup = (torch.as_tensor(v, device=device) for v in (p3, res, sup))
        labels = [torch.as_tensor(l, device=device) for l in labels]
    return Association(p3, members, res, sup, labels)


def _associate_batch(cgroup, frames, max_res_px=30.0, min_views=2, *, max_instances=0, scores=None):
    P = cgroup if isinstance(cgroup, G.PackedCameras) else G.pack(cgroup)
    C = P.K.shape[0]
    frs = [_prep_frame(f, C, scores) for f in frames]
    if not frs:
        return []
    # one undistortion of every detection of every frame
    widths = [f['pad'].shape[1] for f in frs]
    cuts = np.cumsum([0] + widths)
    und_all = G.undistort(P, np.concatenate([f['pad'] for f in frs], axis=1))
    unds, valids, cands = [], [], []
    for k, f in enumerate(frs):
        und = und_all[:, cuts[k]:cuts[k + 1]]
        valid = np.isfinite(f['pad']).all(-1) & np.isfinite(und).all(-1)
        unds.append(und); valids.append(valid)
        cands.append(_pair_candidates(valid, C))
    # one linear triangulation + projection of every cross-camera candidate pair
    sizes = [0 if cd is None else len(cd[0]) for cd in cands]
    mcuts = np.cumsum([0] + sizes)
    cu = np.full((C, int(mcuts[-1]), 2), np.nan)
    for k, cd in enumerate(cands):
        if cd is None:
            continue
        ca_, ia_, cb_, ib_ = cd
        r_ = mcuts[k] + np.arange(len(ca_))
        cu[ca_, r_], cu[cb_, r_] = unds[k][ca_, ia_], unds[k][cb_, ib_]
    if cu.shape[1]:
        X_all = T.triangulate_linear(P, cu, undistort=False)          # [M, 3]
        proj_all, depth_all = G.project(P, X_all)                     # [C, M, 2]
    gens, out = [], [None] * len(frs)
    for k, f in enumerate(frs):
        sl = slice(int(mcuts[k]), int(mcuts[k + 1]))
        args = (None, None, None) if cands[k] is None else (X_all[sl], proj_all[:, sl], depth_all[:, sl])
        gens.append(_greedy_frame(C, f, unds[k], valids[k], cands[k], *args, max_res_px, min_views,
                                  max_instances))
    # drive the greedy loops: one batched refit per round over all frames
    pending = {}
    for k, g in enumerate(gens):
        try:
            pending[k] = next(g)
        except StopIteration as stop:
            out[k] = stop.value
    while pending:
        keys = list(pending)
        sub = np.stack([pending[k][0] for k in keys], 1)               # [C, R, 2]
        obs = np.stack([pending[k][1] for k in keys], 1)
        Xr = T.triangulate_linear(P, sub, undistort=False)             # [R, 3]
        e = G.reproj_err(P, Xr, obs)                                   # [C, R]
        nxt = {}
        for r, k in enumerate(keys):
            try:
                nxt[k] = gens[k].send((Xr[r], e[:, r]))
            except StopIteration as stop:
                out[k] = stop.value
        pending = nxt
    return out
