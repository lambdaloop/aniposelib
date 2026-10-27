"""People / tracklet bootstrap: calibrate from several unassociated people (plan 01 E2,
disco ``pose_ba_boot``).

With several people in view, a camera's detection slot ``p`` (or track id) says nothing about
which person it is in another camera. :func:`associate_people` recovers the cross-camera
identity from geometry alone, then :func:`from_people` runs the ordinary keypoint SfM
(:func:`aniposelib.init.sfm.from_correspondences`) on the associated detections.

Per camera pair (disco ``hypotheses`` / ``solve_pair`` / ``match_tracklets``):

1. **person hypotheses**: in the busiest windows, every (track a in camera i, track b in
   camera j) pair seen together gives >= ``min_corr`` joint correspondences -> a 5-point
   RANSAC essential matrix (on ideal, undistorted pixels of each camera's own intrinsics);
2. each hypothesis is scored by the number of detections with an **unambiguous** partner:
   mutual best, median symmetric epipolar distance below ``gate`` px, and the second best in
   row and column at least ``ratio`` times worse;
3. with the best E, tracks are matched by votes of those anchors (a track anchored to several
   partners keeps one only if it dominates), and E is re-estimated from all matched joints.

Accepted track pairs are merged across all pairs by union-find on ``(camera, track)`` nodes;
groups that would contain two tracks of the same camera in the same frame are split off
(dropped). Track ids must be consistent within a camera (``track_ids``; default: the slot
index ``p`` is the track).
"""

from dataclasses import dataclass, field, fields

import cv2
import numpy as np

from ._common import as_rng, jsonable, to_np
from . import sfm as _sfm

__all__ = ['associate_people', 'from_people', 'AssociationReport']


def _sym_epipolar(xa, xb, E):
    """Symmetric epipolar distance (normalised units) between homogeneous-less points
    ``xa, xb [...,2]`` (``x_b^T E x_a = 0``)."""
    ha = np.concatenate([xa, np.ones(xa.shape[:-1] + (1,))], -1)
    hb = np.concatenate([xb, np.ones(xb.shape[:-1] + (1,))], -1)
    la = ha @ E.T          # epipolar lines in b
    lb = hb @ E            # epipolar lines in a
    num = np.abs(np.sum(hb * la, -1))
    with np.errstate(all='ignore'):
        return 0.5 * num * (1 / np.linalg.norm(la[..., :2], axis=-1) + 1 / np.linalg.norm(lb[..., :2], axis=-1))


def _pair_cost(xa, xb, E, f, min_j=6):
    """[F,Pa,Pb] median over joints of the symmetric epipolar distance (px); inf if fewer than
    ``min_j`` joints are visible in both."""
    d = _sym_epipolar(xa[:, :, None], xb[:, None], E) * f      # [F,Pa,Pb,J]
    ok = np.isfinite(d)
    n = ok.sum(-1)
    # nanmedian by sorting (NaN last): thread safe (no warnings filter) and faster
    srt = np.sort(np.where(ok, d, np.nan), axis=-1)
    lo = np.maximum((n - 1) // 2, 0)[..., None]
    hi = np.maximum(n // 2, 0)[..., None]
    hi = np.minimum(hi, d.shape[-1] - 1)
    med = 0.5 * (np.take_along_axis(srt, lo, -1) + np.take_along_axis(srt, hi, -1))[..., 0]
    med[(n < min_j) | ~np.isfinite(med)] = np.inf
    return med


def _mutual(D, gate, ratio):
    F, A, B = D.shape
    if A == 0 or B == 0:
        return np.zeros(D.shape, bool)
    best = (D == D.min(2, keepdims=True)) & (D == D.min(1, keepdims=True)) & (D < gate)
    r2 = np.sort(D, 2)[:, :, 1:2] if B > 1 else np.full((F, A, 1), np.inf)
    c2 = np.sort(D, 1)[:, 1:2, :] if A > 1 else np.full((F, 1, B), np.inf)
    return best & (r2 >= ratio * D) & (c2 >= ratio * D)


def _essential(a, b, thr, rng, max_iters=1000):
    if len(a) < 8:
        return None
    cv2.setRNGSeed(int(rng.integers(2 ** 31 - 1)))
    try:
        E, mask = cv2.findEssentialMat(a, b, np.eye(3), method=cv2.RANSAC, prob=0.999,
                                       threshold=thr, maxIters=max_iters)
    except cv2.error:
        return None
    if E is None or E.shape[0] < 3:
        return None
    E = E[:3]
    inl = mask.ravel().astype(bool)
    return E / np.linalg.norm(E), float(inl.mean())


def _solve_pair(xa, xb, ta, tb, f, gate, ratio, win, n_windows, min_corr, max_tracks, min_votes,
                dominance, rng, n_score=120):
    """Relative geometry + track matches of one camera pair. ``xa [F,P,J,2]`` normalised."""
    F = xa.shape[0]
    va = np.isfinite(xa).all(-1).sum(-1) >= 6       # [F,P] person visible
    vb = np.isfinite(xb).all(-1).sum(-1) >= 6
    wins = [np.arange(s, min(F, s + win)) for s in range(0, F, win)]
    busy = np.array([(va[w].sum(1) * vb[w].sum(1)).sum() for w in wins], float)
    hyps = []
    for k in np.argsort(-busy)[:n_windows]:
        if busy[k] <= 0:
            break
        w = wins[k]
        ua, ca = np.unique(ta[w][va[w]], return_counts=True)
        ub, cb = np.unique(tb[w][vb[w]], return_counts=True)
        for A in ua[np.argsort(-ca)][:max_tracks]:
            for B in ub[np.argsort(-cb)][:max_tracks]:
                pa, pb = [], []
                for fi in w:
                    ia = np.nonzero((ta[fi] == A) & va[fi])[0]
                    ib = np.nonzero((tb[fi] == B) & vb[fi])[0]
                    if len(ia) and len(ib):
                        m = np.isfinite(xa[fi, ia[0]]).all(-1) & np.isfinite(xb[fi, ib[0]]).all(-1)
                        pa.append(xa[fi, ia[0]][m])
                        pb.append(xb[fi, ib[0]][m])
                if not pa or sum(len(p) for p in pa) < min_corr:
                    continue
                h = _essential(np.concatenate(pa), np.concatenate(pb), 4.0 / f, rng, max_iters=200)
                if h is not None and h[1] >= 0.5:
                    hyps.append(h[0])
    if not hyps:
        return None
    ov_a, ov_b = _overlap_fn(ta, va), _overlap_fn(tb, vb)
    sel = np.sort(rng.choice(F, min(F, n_score), replace=False))
    scores = [int(_mutual(_pair_cost(xa[sel], xb[sel], E, f), gate, ratio).sum()) for E in hyps]
    E = hyps[int(np.argmax(scores))]
    for _ in range(3):  # match tracks, re-estimate E from all matched joints
        D = _pair_cost(xa, xb, E, f)
        fr, i, j = np.nonzero(_mutual(D, gate, ratio))
        if len(fr) == 0:
            return None
        votes = {}
        for a, b in zip(ta[fr, i], tb[fr, j]):
            votes[(int(a), int(b))] = votes.get((int(a), int(b)), 0) + 1
        acc = _dominant(votes, dominance, min_votes, ov_a, ov_b)
        if not acc:
            return None
        pa, pb = [], []
        for fi in range(F):
            for a, b in acc:
                ia = np.nonzero((ta[fi] == a) & va[fi])[0]
                ib = np.nonzero((tb[fi] == b) & vb[fi])[0]
                if len(ia) and len(ib) and D[fi, ia[0], ib[0]] < 2 * gate:
                    m = np.isfinite(xa[fi, ia[0]]).all(-1) & np.isfinite(xb[fi, ib[0]]).all(-1)
                    pa.append(xa[fi, ia[0]][m])
                    pb.append(xb[fi, ib[0]][m])
        h = _essential(np.concatenate(pa), np.concatenate(pb), 2.0 / f, rng)
        if h is None:
            break
        E = h[0]
    return dict(E=E, matches=acc, votes={k: votes.get(k, 0) for k in acc}, score=max(scores),
                n_hypotheses=len(hyps))


def _overlap_fn(t, v):
    """``overlap(x, y)``: whether tracks x and y (ids in ``t [F,P]``, visible where ``v``) share a frame."""
    frames = {}
    fr, sl = np.nonzero(v & (t >= 0))
    for tid in np.unique(t[fr, sl]):
        frames[int(tid)] = set(fr[t[fr, sl] == tid].tolist())
    return lambda x, y: bool(frames.get(int(x), set()) & frames.get(int(y), set()))


def _dominant(votes, dominance, min_votes, overlap_a=None, overlap_b=None):
    """Track pairs (a, b) that are each other's dominant partner. ``overlap_a(x, y)`` /
    ``overlap_b(x, y)``: whether two tracks of camera a / b are visible in a common frame; only
    those compete (fragments of one person's track, cut by the tracker, are not rivals: each
    keeps its own match). None: every other track is a rival (the 1.0 rule)."""
    acc = []
    for (a, b), n in votes.items():
        if n < min_votes:
            continue
        rivals_a = [m for (x, y), m in votes.items() if x == a and y != b
                    and (overlap_b is None or overlap_b(y, b))]
        rivals_b = [m for (x, y), m in votes.items() if y == b and x != a
                    and (overlap_a is None or overlap_a(x, a))]
        if all(n >= dominance * m for m in rivals_a + rivals_b):
            acc.append((a, b))
    return acc


@dataclass
class AssociationReport:
    """``pairs``: per camera pair, the accepted track matches and votes; ``groups``: per
    global identity, ``{camera: track}``; ``dropped``: groups with conflicting tracks."""
    pairs: dict = field(default_factory=dict)
    groups: list = field(default_factory=list)
    dropped: list = field(default_factory=list)

    def to_dict(self):
        d = {f.name: jsonable(getattr(self, f.name)) for f in fields(self)}
        d['pairs'] = {'{}-{}'.format(*k): jsonable({kk: vv for kk, vv in v.items() if kk != 'E'})
                      for k, v in self.pairs.items()}
        return d


def associate_people(cgroup, p2ds, track_ids=None, gate=10.0, ratio=1.8, win=20, n_windows=8,
                     min_corr=40, max_tracks=8, min_votes=5, dominance=2.0, min_pair_cams=2,
                     rng=None):
    """Cross-camera identities of people from epipolar geometry (see module doc).

    ``p2ds [C,F,P,J,2]``: per camera, frame and detection slot, J keypoints (NaN = missing);
    ``track_ids [C,F,P]`` (int, -1 = none; default: slot index). Returns
    ``(assoc [C,F,G,J,2], AssociationReport)`` where ``G`` global people are consistent across
    cameras (NaN where a camera does not see them)."""
    rng = as_rng(rng)
    p = np.asarray(to_np(p2ds), dtype='float64')
    C, F, Pn, J, _ = p.shape
    if track_ids is None:
        track_ids = np.broadcast_to(np.arange(Pn)[None, None], (C, F, Pn))
    track_ids = np.asarray(track_ids)
    u, f = _sfm.ideal_points(cgroup, p.reshape(C, -1, 2))
    x = (u / f[:, None, None]).reshape(C, F, Pn, J, 2)
    rep = AssociationReport()
    parent = {}

    def find(n):
        while parent.setdefault(n, n) != n:
            parent[n] = parent[parent[n]]
            n = parent[n]
        return n

    # the camera pairs are independent (cv2 releases the GIL): run them on a thread pool, each
    # with its own seed drawn from rng in pair order (the result does not depend on the threads)
    todo = [(i, j, int(rng.integers(2 ** 31 - 1))) for i in range(C) for j in range(i + 1, C)]

    def one(job):
        i, j, seed = job
        return _solve_pair(x[i], x[j], track_ids[i], track_ids[j], float(np.sqrt(f[i] * f[j])),
                           gate, ratio, win, n_windows, min_corr, max_tracks, min_votes,
                           dominance, np.random.default_rng(seed))
    for (i, j, _), r in zip(todo, _sfm._pmap(one, todo)):
        if r is not None:
            rep.pairs[(i, j)] = r
    # merge the accepted track pairs, strongest first. A person's tracks fragment (an IoU tracker
    # cuts them at occlusions), so a group may hold several tracks of one camera, but never two
    # that are visible in the same frame (that merge is refused, i.e. the weaker edge is dropped;
    # before 1.1 any group with two tracks of one camera was dropped whole, which lost most data)
    vis = np.isfinite(p).all(-1).sum(-1) >= 1                       # [C,F,P]
    frames_of = {}
    for c in range(C):
        fr, sl = np.nonzero(vis[c] & (track_ids[c] >= 0))
        for t in np.unique(track_ids[c][fr, sl]):
            frames_of[(c, int(t))] = set(fr[track_ids[c][fr, sl] == t].tolist())
    edges = sorted(((n, (i, int(a)), (j, int(b))) for (i, j), r in rep.pairs.items()
                    for (a, b), n in r['votes'].items()), key=lambda e: -e[0])
    occ = {}                                                        # root -> {camera: frames}

    def occupancy(n):
        rt = find(n)
        if rt not in occ:
            occ[rt] = {n[0]: set(frames_of.get(n, ()))}
        return rt, occ[rt]

    for n, na, nb in edges:
        ra, oa = occupancy(na)
        rb, ob = occupancy(nb)
        if ra == rb:
            continue
        if any(c in oa and len(oa[c] & fs) > 0 for c, fs in ob.items()):
            rep.dropped.append(dict(pair=[list(na), list(nb)], votes=int(n), reason='same-frame conflict'))
            continue
        parent[rb] = ra
        for c, fs in ob.items():
            oa.setdefault(c, set()).update(fs)
        del occ[rb]
    groups = {}
    for nd in list(parent):
        groups.setdefault(find(nd), []).append(nd)
    members = []
    for g in groups.values():
        m = {}
        for c, t in sorted(g):
            m.setdefault(c, []).append(t)
        if len(m) >= min_pair_cams:
            members.append(m)
    members.sort(key=lambda m: -sum(len(frames_of.get((c, t), ())) for c, ts in m.items() for t in ts))
    rep.groups = [{cgroup.get_names()[c]: (int(ts[0]) if len(ts) == 1 else [int(t) for t in ts])
                   for c, ts in m.items()} for m in members]
    G = len(members)
    out = np.full((C, F, max(G, 1), J, 2), np.nan)[:, :, :G]
    for g, m in enumerate(members):
        for c, ts in m.items():
            hit = np.isin(track_ids[c], ts)               # [F,P]
            fr, sl = np.nonzero(hit)
            out[c, fr, g] = p[c, fr, sl]
    return out, rep


def from_people(cgroup, p2ds, track_ids=None, association_kwargs=None, association=None, **sfm_kwargs):
    """Calibrate extrinsics from several unassociated people: :func:`associate_people`, then
    :func:`~aniposelib.init.sfm.from_correspondences` on the associated keypoints (one SfM
    'frame' per person and frame; windows = frames). ``association``: a precomputed
    ``(assoc, AssociationReport)`` of :func:`associate_people` to reuse (then ``track_ids`` and
    ``association_kwargs`` are ignored). Returns ``(CameraGroup, InitReport)``; the association
    report is in ``report.stats['association']``."""
    if association is None:
        association = associate_people(cgroup, p2ds, track_ids=track_ids, **(association_kwargs or {}))
    assoc, arep = association
    C, F, G, J, _ = assoc.shape
    if G == 0:
        raise RuntimeError('from_people: no person could be associated across cameras')
    flat = assoc.reshape(C, F * G, J, 2)
    kw = dict(frame_id=np.arange(F * G), window_id=np.repeat(np.arange(F), G))
    kw.update(sfm_kwargs)
    cg, rep = _sfm.from_correspondences(cgroup, flat, **kw)
    rep.method = 'people'
    rep.stats['association'] = arep.to_dict()
    return cg, rep
