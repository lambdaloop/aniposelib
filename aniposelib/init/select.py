"""Choose between candidate initialisations by how well they explain the 2D data (plan 01 E4).

Candidates can come from anywhere: a lab calibration, keypoint SfM, every VGGT template, a
previous session's rig. Each is scored on the observations by robust (MSAC) triangulation:

- ``score='robust_inliers'`` (Emory ``vggt_pose_init``): the fraction of observations that are
  inliers (error < ``tau`` px) of points with at least ``min_inliers`` inlier views; higher is
  better, ties are broken by the inlier median error;
- ``score='median_px'`` (motor ``select_vggt_template``): median robust reprojection error of
  all observations of supported points; lower is better.

Documented lesson (motor, Emory): a template's agreement with the *other* templates does not
predict its accuracy. Only the data can rank hypotheses, so score them, do not average them.
"""

from dataclasses import dataclass, field, fields

import numpy as np

from ._common import jsonable, to_np

__all__ = ['select', 'score_candidate', 'SelectResult']


def _flat(p2ds):
    p = np.asarray(to_np(p2ds), dtype='float64')
    if p.ndim == 4:
        p = p.reshape(p.shape[0], -1, 2)
    if p.ndim != 3:
        raise ValueError('p2ds must be [C,N,2] or [C,N,J,2]')
    return p


def score_candidate(cgroup, p2ds, tau=30.0, min_inliers=3, max_points=20000, rng=0):
    """Robust-triangulation statistics of one candidate (cameras in ``p2ds`` order)."""
    from aniposelib import geometry as G
    from aniposelib.triangulation import triangulate_robust
    p = _flat(p2ds)
    valid = np.all(np.isfinite(p), -1)
    keep = np.nonzero(valid.sum(0) >= max(2, min_inliers))[0]
    if len(keep) > max_points:
        keep = np.sort(np.random.default_rng(rng).choice(keep, max_points, replace=False))
    n_obs = int(valid[:, keep].sum())
    if n_obs == 0:
        return dict(inlier_frac=0.0, median_px=float('inf'), inlier_median_px=float('inf'),
                    n_obs=0, n_points=0, n_supported=0)
    P = G.pack(cgroup, device='cpu')
    res = triangulate_robust(P, p[:, keep], method='msac', tau=tau, min_inliers=min_inliers,
                             mask_out_of_bounds=False)
    inl = np.asarray(res.inliers, dtype=bool)
    err = np.asarray(res.errors, dtype='float64')
    supported = np.asarray(res.n_inliers) >= min_inliers
    ok_inl = inl & supported[None] & valid[:, keep]
    e_sup = err[:, supported][valid[:, keep][:, supported]]
    e_sup = np.where(np.isfinite(e_sup), e_sup, np.inf)
    e_in = err[ok_inl]
    per_cam = {}
    for c, n in enumerate(cgroup.get_names()):
        vc = valid[c, keep]
        per_cam[n] = float(ok_inl[c].sum() / max(1, vc.sum()))
    return dict(inlier_frac=float(ok_inl.sum() / n_obs),
                median_px=float(np.median(e_sup)) if len(e_sup) else float('inf'),
                inlier_median_px=float(np.median(e_in)) if len(e_in) else float('inf'),
                n_obs=n_obs, n_points=int(len(keep)), n_supported=int(supported.sum()),
                per_camera_inlier_frac=per_cam)


@dataclass
class SelectResult:
    """``best``: index (or key) of the winner; ``scores``: one dict per candidate (in input
    order, with its ``key``); ``score``: the criterion used."""
    best: object
    score: str
    scores: list = field(default_factory=list)

    def ranking(self):
        """Candidate keys, best first."""
        return [s['key'] for s in sorted(self.scores, key=_sort_key(self.score))]

    def to_dict(self):
        return {f.name: jsonable(getattr(self, f.name)) for f in fields(self)}


def _sort_key(score):
    if score == 'robust_inliers':
        return lambda s: (-s['inlier_frac'], s['inlier_median_px'])
    return lambda s: (s['median_px'], -s['inlier_frac'])


def select(candidates, p2ds, score='robust_inliers', tau=30.0, min_inliers=3, cam_names=None,
           max_points=20000, rng=0):
    """Pick the candidate init that best explains ``p2ds``.

    ``candidates``: a list or dict of ``CameraGroup`` s. ``p2ds``: ``[C,N,2]`` / ``[C,N,J,2]``
    in the order of ``cam_names`` (default: the first candidate's camera order); every
    candidate is reordered by name (a candidate missing a camera is scored ``-inf``/``inf``).
    Returns ``(best_cgroup, SelectResult)``."""
    if score not in ('robust_inliers', 'median_px'):
        raise ValueError("score must be 'robust_inliers' or 'median_px'")
    items = list(candidates.items()) if isinstance(candidates, dict) else list(enumerate(candidates))
    if not items:
        raise ValueError('select: no candidates')
    names = list(cam_names) if cam_names is not None else items[0][1].get_names()
    scores = []
    for key, cg in items:
        if set(names) - set(cg.get_names()):
            s = dict(inlier_frac=-np.inf, median_px=np.inf, inlier_median_px=np.inf,
                     error='missing cameras: {}'.format(sorted(set(names) - set(cg.get_names()))))
        else:
            sub = cg if cg.get_names() == names else cg.subset_cameras_names(names)
            s = score_candidate(sub, p2ds, tau=tau, min_inliers=min_inliers, max_points=max_points,
                                rng=rng)
        s['key'] = key
        scores.append(s)
    best = sorted(scores, key=_sort_key(score))[0]['key']
    cg = dict(items)[best]
    if cg.get_names() != names:
        cg = cg.subset_cameras_names(names)
    return cg, SelectResult(best=best, score=score, scores=scores)
