"""aniposelib.matching: cross-camera correspondence from tracklets and per-frame points."""

import json

import numpy as np
import pytest
import torch

import synth
from aniposelib import geometry as G
from aniposelib import triangulation as T
from aniposelib.matching import (Association, Groups, TrackletPoints, _ambiguity_margin,
                                 associate_points, associate_points_frames,
                                 associate_tracklets)

SWITCH_CAM, SWITCH_A, SWITCH_B, SWITCH_FRAMES = 2, 1, 2, range(60, 70)
FRAG_CAM, FRAG_PERSON, FRAG_AT, FRAG_ID = 1, 0, 80, 77
DUP_CAM, DUP_PERSON, DUP_SPAN, DUP_ID = 3, 2, (30, 120), 99


def make_people_scene(cg, n_people=3, n_frames=160, noise=1.0, seed=0):
    """3 people seen by every camera, per-camera ids (random, no cross-camera meaning), with
    a fragmented track, a short id switch and a duplicate track. Returns the 3D truth, the
    detections and ids per camera ({frame: (D,K,3)} / {frame: (D,)}) and the true person of
    every detection."""
    rng = np.random.default_rng(seed)
    P3 = synth.crowd(cg, n_people, n_frames, sep=1.0, phase=0.35, seed=10)
    p2d = synth.add_noise(synth.project(cg, P3), noise, seed=seed + 1)   # [C, P, T, J, 2]
    dets, ids, truth = [], [], []
    for c in range(p2d.shape[0]):
        perm = rng.permutation(n_people) + 10 * c + 3
        d_c, i_c, t_c = {}, {}, {}
        for t in range(n_frames):
            rows, tid, who = [], [], []
            for p in range(n_people):
                body = p
                if c == SWITCH_CAM and t in SWITCH_FRAMES and p in (SWITCH_A, SWITCH_B):
                    body = SWITCH_A + SWITCH_B - p
                tp = int(perm[p])
                if c == FRAG_CAM and p == FRAG_PERSON and t >= FRAG_AT:
                    tp = FRAG_ID
                rows.append(p2d[c, body, t]); tid.append(tp); who.append(body)
            if c == DUP_CAM and DUP_SPAN[0] <= t < DUP_SPAN[1]:
                rows.append(p2d[c, DUP_PERSON, t] + rng.normal(scale=1.5, size=p2d.shape[-2:]))
                tid.append(DUP_ID); who.append(DUP_PERSON)
            o = rng.permutation(len(rows))
            score = rng.uniform(0.6, 1.0, size=(len(rows), p2d.shape[3], 1))
            d_c[t] = np.concatenate([np.stack(rows)[o], score], -1)
            i_c[t] = np.array(tid)[o]
            t_c[t] = np.array(who)[o]
        dets.append(d_c); ids.append(i_c); truth.append(t_c)
    return P3, dets, ids, truth


@pytest.fixture(scope='module')
def people():
    cg = synth.load_rig('fisheye_5cam')
    P3, dets, ids, truth = make_people_scene(cg)
    groups = associate_tracklets(cg, dets, ids)
    return cg, P3, dets, ids, truth, groups


@pytest.mark.integration
def test_tracklets_group_people(people):
    cg, P3, dets, ids, truth, groups = people
    C = len(cg.cameras)
    assert isinstance(groups, Groups) and len(groups) == 3
    found = set()
    for g in groups:
        assert len(g.cameras) >= 3
        who = set()
        for c in g.cameras:
            fr, di, kp = g.observations(c)
            w = {truth[c][int(f)][int(i)] for f, i in zip(fr, di)}
            who |= w
            assert len(w) == 1, f'group {g.id} camera {c} mixes people {w}'
            if c == SWITCH_CAM and w <= {SWITCH_A, SWITCH_B}:   # switched window removed
                assert not set(SWITCH_FRAMES) & set(fr.tolist()) and len(fr) == 150
        assert len(who) == 1
        found |= who
        # the fragmented person comes back together
        if who == {FRAG_PERSON}:
            assert sum(1 for c, _ in g.members if c == FRAG_CAM) == 2
            assert len(g.observations(FRAG_CAM)[0]) == 160
    assert found == {0, 1, 2}
    assert all(len(g.cameras) == C for g in groups)
    # id switch: exactly the switched cells of the two switched people, in that camera only
    dropped = {g.id: {c: set(v.tolist()) for c, v in g.dropped.items()} for g in groups}
    assert sum(len(v) for d in dropped.values() for v in d.values()) == 2 * len(SWITCH_FRAMES)
    for d in dropped.values():
        assert set(d) <= {SWITCH_CAM}
    # the duplicate was merged and is not a competitor (no abstention)
    assert [(m['cam'], m['merged']) for m in groups.info['merged']] == [(DUP_CAM, DUP_ID)]
    assert groups.info['abstentions'] == []
    assert groups.info['n_tracklets'] == 3 * C + 1        # + fragment (duplicate merged)


@pytest.mark.integration
def test_tracklets_to_points(people):
    cg, P3, dets, ids, truth, groups = people
    C, K = len(cg.cameras), P3.shape[2]
    out = groups.to_points(n_frames=60)
    assert isinstance(out, TrackletPoints)
    pts, fid, gid = out
    assert pts.shape == (C, 60, K, 2) and fid.shape == (60,) and gid.shape == (60,)
    assert set(gid.tolist()) == {0, 1, 2}
    P = G.pack(cg)
    X = T.triangulate_linear(P, pts.reshape(C, -1, 2))
    err = G.reproj_err(P, X, pts.reshape(C, -1, 2))
    assert np.nanmedian(err) < 2.0 and np.nanmax(err) < 10.0
    # 3D matches the person each group stands for
    X = X.reshape(-1, K, 3)
    for g in groups:
        c = g.cameras[0]
        fr, di, _ = g.observations(c)
        person = truth[c][int(fr[0])][int(di[0])]
        sel = gid == g.id
        d = np.linalg.norm(X[sel] - P3[person, fid[sel]], axis=-1)
        assert np.nanmedian(d) < 0.02 * synth.rig_scale(cg)
    # all frames / score threshold / min_cams
    pa = groups.to_points(frames='all')
    assert pa.points.shape[1] == 3 * 160
    hi = groups.to_points(n_frames=30, score_thr=0.99, min_points=4)
    assert hi.points.shape[1] < 30
    assert np.isfinite(groups.to_points(n_frames=30).points).sum() > np.isfinite(hi.points).sum()
    dense = groups.dense()
    assert dense.shape == (3, C, 160, K, 2)


def test_tracklets_to_dict_json(people):
    groups = people[-1]
    d = groups.to_dict()
    s = json.dumps(d)
    assert len(json.loads(s)['groups']) == 3


def test_tracklets_dict_by_name_and_lists(people):
    cg, P3, dets, ids, truth, groups = people
    names = cg.get_names()
    # dict keyed by name, per frame a list of (K, 3) arrays, ids as lists
    dets2 = {n: {f: list(v) for f, v in d.items()} for n, d in zip(names, dets)}
    ids2 = {n: {f: v.tolist() for f, v in i.items()} for n, i in zip(names, ids)}
    g2 = associate_tracklets(cg, dets2, ids2, verify=False, ambiguity_ratio=0)
    assert len(g2) == 3
    assert sorted(g.members for g in g2) == sorted(g.members for g in groups)
    # without the verify step the switched window stays in (so the guard is what removes it)
    mixed = [g for g in g2 if len({truth[SWITCH_CAM][int(f)][int(i)]
                                   for f, i in zip(*g.observations(SWITCH_CAM)[:2])}) > 1]
    assert len(mixed) == 2


def test_tracklets_mismatched_ids_raise(people):
    cg, P3, dets, ids, truth, groups = people
    bad = [dict(i) for i in ids]
    bad[1][5] = bad[1][5][:-1]
    with pytest.raises(ValueError, match='track ids'):
        associate_tracklets(cg, dets, bad)
    bad = [dict(i) for i in ids]
    del bad[0][7]
    with pytest.raises(ValueError):
        associate_tracklets(cg, dets, bad)
    with pytest.raises(ValueError):
        associate_tracklets(cg, dets[:-1], ids[:-1])


def test_tracklets_duplicate_merge_small():
    """Two ids on one body in one camera are merged; two different bodies are not."""
    from aniposelib.matching import _Tracklet, _merge_duplicates
    rng = np.random.default_rng(0)
    body = rng.uniform(0, 300, size=(1, 10, 2)) + np.arange(40)[:, None, None]
    other = body + 250.0
    def tr(t, xy, frames):
        return _Tracklet(0, t, np.asarray(frames), np.zeros(len(frames), int), xy, xy)
    tracks = {1: tr(1, body, range(40)),
              2: tr(2, body[10:30] + rng.normal(scale=1.0, size=(20, 10, 2)), range(10, 30)),
              3: tr(3, other, range(40))}
    out, merges = _merge_duplicates(tracks, 0.2)
    assert sorted(out) == [1, 3] and out[1].merged == [2]
    assert len(out[1].frames) == 40 and [m[:2] for m in merges] == [(1, 2)]


def test_ambiguity_margin_and_abstention(people):
    chosen = np.zeros((5, 4, 2))
    proj = chosen + 2.0
    near = [chosen[i:i + 1] + 3.0 for i in range(5)]
    own, alt = _ambiguity_margin(chosen, near, proj)
    assert own == pytest.approx(2 * np.sqrt(2)) and alt == pytest.approx(np.sqrt(2))
    assert _ambiguity_margin(chosen, [np.zeros((0, 4, 2))] * 5, proj)[1] is None
    # a huge ratio makes every camera abstain -> no group survives
    cg, P3, dets, ids, truth, groups = people
    g = associate_tracklets(cg, dets, ids, ambiguity_ratio=1e6)
    assert len(g) == 0 and g.info['n_rejected'] == 3 and len(g.info['abstentions']) > 0


def test_tracklets_min_score():
    cg = synth.load_rig('fisheye_5cam')
    P3, dets, ids, truth = make_people_scene(cg, n_frames=40)
    g = associate_tracklets(cg, dets, ids, min_score=1.01)     # every keypoint masked
    assert len(g) == 0 and g.info['n_tracklets'] == 0
    assert g.to_points().points.shape[1] == 0


# ------------------------------------------------------------------ per-frame points

def make_animal_frames(cg, n_animals=4, n_frames=5, noise=1.0, seed=0):
    rng = np.random.default_rng(seed)
    frames, truth = [], []
    for t in range(n_frames):
        X = synth.random_points(cg, n_animals, spread=0.15, min_views=7, seed=seed + 100 * t)
        p2d = synth.add_noise(synth.project(cg, X), noise, seed=seed + t)
        per_cam, labels = [], []
        for c in range(p2d.shape[0]):
            keep = np.flatnonzero(rng.random(n_animals) > 0.15)        # some missed
            pts = list(p2d[c, keep]); lab = list(keep)
            if c == t % p2d.shape[0]:                                  # one false detection
                pts.append(rng.uniform([0, 0], [1440, 1080])); lab.append(-1)
            o = rng.permutation(len(pts))
            per_cam.append(np.array(pts).reshape(-1, 2)[o])
            labels.append(np.array(lab, dtype=int)[o])
        frames.append(per_cam)
        truth.append((X, labels))
    return frames, truth


@pytest.mark.integration
def test_associate_points_animals():
    cg = synth.load_rig('motobs_7cam')
    frames, truth = make_animal_frames(cg)
    results = associate_points_frames(cg, frames, max_res_px=15.0, min_views=3)
    for res, frame, (X, lab) in zip(results, frames, truth):
        assert isinstance(res, Association)
        assert len(res) == len(X)
        assert res.points3d.shape == (len(X), 3)
        assert np.all(res.residuals < 5)
        seen = set()
        for m, mem in enumerate(res.members):
            who = {int(lab[c][i]) for c, i in mem.items()}
            assert len(who) == 1 and -1 not in who
            a = who.pop()
            seen.add(a)
            assert np.linalg.norm(res.points3d[m] - X[a]) < 0.01 * synth.rig_scale(cg)
            for c, i in mem.items():
                assert res.labels[c][i] == m
        assert seen == set(range(len(X)))
        # every assigned label is consistent with members; false detections unassigned
        for c in range(len(cg.cameras)):
            assert np.all(res.labels[c][lab[c] == -1] == -1)
            assert (res.labels[c] >= 0).sum() == sum(c in mem for mem in res.members)
        P2 = res.to_points(frame)
        assert P2.shape == (len(cg.cameras), len(X), 2)
        json.dumps(res.to_dict())


def test_associate_points_torch_and_single_views():
    cg = synth.load_rig('motobs_7cam')
    frames, truth = make_animal_frames(cg, n_frames=1, seed=3)
    pts = [torch.as_tensor(p) for p in frames[0]]
    res = associate_points(cg, pts, max_res_px=15.0)
    assert isinstance(res.points3d, torch.Tensor) and isinstance(res.labels[0], torch.Tensor)
    ref = associate_points(cg, frames[0], max_res_px=15.0)
    assert isinstance(ref.points3d, np.ndarray)
    np.testing.assert_allclose(res.points3d.numpy(), ref.points3d)
    assert isinstance(res.to_points(pts), torch.Tensor)
    # min_views=1: leftovers (the false detection) become single-view instances
    one = associate_points(cg, frames[0], max_res_px=15.0, min_views=1)
    n_det = sum(len(p) for p in frames[0])
    assert sum(len(m) for m in one.members) == n_det
    single = [m for m, mem in enumerate(one.members) if len(mem) == 1]
    assert len(single) >= 1 and np.all(np.isnan(one.points3d[single]))
    assert np.all(np.isinf(one.residuals[single]))
    # max_instances caps; NaN rows are missing; wrong camera count raises
    assert len(associate_points(cg, frames[0], max_res_px=15.0, max_instances=2)) == 2
    nan = [np.vstack([p, np.full((1, 2), np.nan)]) for p in frames[0]]
    r2 = associate_points(cg, nan, max_res_px=15.0)
    assert len(r2) == len(ref) and all(l[-1] == -1 for l in r2.labels)
    with pytest.raises(ValueError):
        associate_points(cg, frames[0][:-1])
