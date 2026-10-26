"""Synthetic data generators for rig-based integration tests.

Each rig is a real calibration (a small TOML in tests/data/rigs/) taken from a downstream use
case. All 2D data is synthesised from it with a seeded RNG, so every test has exact ground truth
and runs in seconds. Arrays are numpy and float64 throughout; cameras are aniposelib objects.
"""

import os
from copy import deepcopy

import numpy as np
import torch

from aniposelib.cameras import CameraGroup

RIGS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'rigs')

# name -> (file, short description of the use case it stands in for)
RIGS = {
    'allen_7cam': ('allen_7cam.toml', 'lab mouse rig, 7 pinhole cams, ChArUco board (mm)'),
    'allen_12cam': ('allen_12cam.toml', 'lab mouse rig, 12 pinhole cams, ChArUco board (mm)'),
    'motobs_7cam': ('motobs_7cam.toml', 'motor observatory mouse rig, keypoint self-calibration (mm)'),
    'outdoor_12cam': ('outdoor_12cam.toml', 'outdoor thermal mouse arena, 12 cams, no board'),
    'outdoor_9cam': ('outdoor_9cam.toml', 'outdoor thermal mouse arena, 9 cams'),
    'fisheye_5cam': ('fisheye_5cam.toml', 'EvoVision gait rig, 5 fisheye GoPros, people (m)'),
}


def to_np(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def load_rig(name):
    return CameraGroup.load(os.path.join(RIGS_DIR, RIGS[name][0]))


# ---------------------------------------------------------------- geometry helpers

def camera_centers(cgroup):
    return np.array([to_np(c.get_center_world()).ravel() for c in cgroup.cameras])


def camera_axes(cgroup):
    """Viewing direction (optical axis) of each camera in world coordinates."""
    out = []
    for c in cgroup.cameras:
        R = to_np(c.get_extrinsics_mat())[:3, :3]
        out.append(R.T @ np.array([0.0, 0.0, 1.0]))
    return np.array(out)


def scene_center(cgroup):
    """Least-squares point closest to all optical axes: where the rig looks."""
    C, D = camera_centers(cgroup), camera_axes(cgroup)
    A = np.zeros((3, 3)); b = np.zeros(3)
    for c, d in zip(C, D):
        P = np.eye(3) - np.outer(d, d)
        A += P; b += P @ c
    return np.linalg.solve(A, b)


def rig_scale(cgroup):
    """Median camera-to-scene distance (sets the units of everything synthetic)."""
    return float(np.median(np.linalg.norm(camera_centers(cgroup) - scene_center(cgroup), axis=1)))


def project(cgroup, p3d):
    """Project (N, 3) or (N, J, 3) points; NaN where behind a camera or outside the image."""
    p3d = np.asarray(p3d, dtype='float64')
    shape = p3d.shape[:-1]
    flat = p3d.reshape(-1, 3)
    p2d = to_np(cgroup.project(flat)).reshape(len(cgroup.cameras), -1, 2).copy()
    for i, cam in enumerate(cgroup.cameras):
        E = to_np(cam.get_extrinsics_mat())
        z = flat @ E[:3, :3].T + E[:3, 3]
        w, h = cam.get_size()
        bad = (z[:, 2] <= 0) | ~np.all(np.isfinite(p2d[i]), axis=1) \
            | (p2d[i, :, 0] < 0) | (p2d[i, :, 0] >= w) | (p2d[i, :, 1] < 0) | (p2d[i, :, 1] >= h)
        p2d[i, bad] = np.nan
    return p2d.reshape((len(cgroup.cameras),) + shape + (2,))


def n_views(p2d):
    return np.sum(np.all(np.isfinite(p2d), axis=-1), axis=0)


# ---------------------------------------------------------------- scenes

def random_points(cgroup, n, spread=0.15, min_views=2, seed=0):
    """Points scattered around the scene centre (spread is a fraction of rig scale),
    kept only if at least `min_views` cameras see them."""
    rng = np.random.default_rng(seed)
    c, s = scene_center(cgroup), rig_scale(cgroup)
    out = []
    while sum(len(o) for o in out) < n:
        p = c + rng.uniform(-1, 1, size=(4 * n, 3)) * spread * s
        ok = n_views(project(cgroup, p)) >= min_views
        out.append(p[ok])
    return np.concatenate(out)[:n]


def trajectory(cgroup, n, spread=0.25, step=0.01, min_views=2, seed=0):
    """A smooth random walk (an animal moving around the arena), one point per frame.
    Frames seen by fewer than `min_views` cameras are dropped (the animal left the arena)."""
    rng = np.random.default_rng(seed)
    c, s = scene_center(cgroup), rig_scale(cgroup)
    out = []
    p = np.zeros(3)
    while sum(len(o) for o in out) < n:
        steps = rng.normal(scale=step, size=(4 * n, 3))
        walk = np.empty_like(steps)
        for t in range(len(steps)):
            p = np.clip(0.98 * p + steps[t], -spread, spread)
            walk[t] = p
        x = c + walk * s
        out.append(x[n_views(project(cgroup, x)) >= min_views])
    return np.concatenate(out)[:n]


# simple stick figures; lengths are fractions of rig scale
MOUSE = dict(
    joints=['nose', 'ear_l', 'ear_r', 'neck', 'spine', 'tail_base', 'paw_fl', 'paw_fr', 'paw_bl', 'paw_br'],
    offsets=np.array([[0.06, 0, 0.01], [0.04, 0.012, 0.015], [0.04, -0.012, 0.015], [0.03, 0, 0.01],
                      [0, 0, 0.012], [-0.04, 0, 0.008], [0.025, 0.012, -0.01], [0.025, -0.012, -0.01],
                      [-0.025, 0.014, -0.01], [-0.025, -0.014, -0.01]]),
    bones=[(0, 3), (1, 3), (2, 3), (3, 4), (4, 5), (3, 6), (3, 7), (5, 8), (5, 9)])

PERSON = dict(
    joints=['head', 'neck', 'sh_l', 'sh_r', 'el_l', 'el_r', 'wr_l', 'wr_r', 'hip_l', 'hip_r',
            'kn_l', 'kn_r', 'an_l', 'an_r'],
    offsets=np.array([[0, 0, 0.17], [0, 0, 0.15], [0, 0.04, 0.14], [0, -0.04, 0.14], [0, 0.05, 0.11],
                      [0, -0.05, 0.11], [0, 0.05, 0.08], [0, -0.05, 0.08], [0, 0.02, 0.08],
                      [0, -0.02, 0.08], [0, 0.02, 0.04], [0, -0.02, 0.04], [0, 0.02, 0.0], [0, -0.02, 0.0]]),
    bones=[(0, 1), (1, 2), (1, 3), (2, 4), (3, 5), (4, 6), (5, 7), (2, 8), (3, 9), (8, 9),
           (8, 10), (9, 11), (10, 12), (11, 13)])


def _rotz(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def skeleton(cgroup, n_frames, model=MOUSE, size=1.0, up=None, spread=0.2, seed=0):
    """Rigid-ish stick figure walking around the scene: (N, J, 3) and its bone list.

    The figure is placed with its local z axis along `up` (default: average camera 'up' is not
    known, so the mean optical axis is used as 'down'), translated along a random walk and
    yawed slowly; joints get a small independent jitter so bones are not perfectly rigid."""
    rng = np.random.default_rng(seed)
    s = rig_scale(cgroup)
    if up is None:
        up = -camera_axes(cgroup).mean(0)
        up /= np.linalg.norm(up)
    # frame with z = up
    x = np.cross(up, [1.0, 0, 0]) if abs(up[0]) < 0.9 else np.cross(up, [0, 1.0, 0])
    x /= np.linalg.norm(x); y = np.cross(up, x)
    B = np.stack([x, y, up], axis=1)
    path = trajectory(cgroup, n_frames, spread=spread, step=0.004, min_views=3, seed=seed)
    yaw = np.cumsum(rng.normal(scale=0.05, size=n_frames))
    off = model['offsets'] * size * s
    p3d = np.empty((n_frames, len(off), 3))
    for t in range(n_frames):
        local = off @ _rotz(yaw[t]).T + rng.normal(scale=0.0005 * s, size=off.shape)
        p3d[t] = path[t] + local @ B.T
    return p3d, list(model['bones'])


# ---------------------------------------------------------------- corruption

def add_noise(p2d, sigma=0.5, seed=0):
    rng = np.random.default_rng(seed)
    return p2d + rng.normal(scale=sigma, size=p2d.shape)


def drop_views(p2d, frac=0.1, seed=0):
    """Randomly remove observations (occlusion / missed detections)."""
    rng = np.random.default_rng(seed)
    p2d = p2d.copy()
    m = rng.random(p2d.shape[:-1]) < frac
    p2d[m] = np.nan
    return p2d


def add_outliers(p2d, frac=0.05, cgroup=None, seed=0):
    """Replace a fraction of observations by uniformly random pixels (false detections)."""
    rng = np.random.default_rng(seed)
    p2d = p2d.copy()
    for i in range(p2d.shape[0]):
        w, h = cgroup.cameras[i].get_size() if cgroup is not None else (1000, 1000)
        m = (rng.random(p2d.shape[1:-1]) < frac) & np.all(np.isfinite(p2d[i]), axis=-1)
        p2d[i][m] = rng.uniform([0, 0], [w, h], size=(int(m.sum()), 2))
    return p2d


def add_decoy(p2d, cam, frac=0.6, pixel=None, cgroup=None, jitter=2.0, seed=0):
    """One camera locks onto a static decoy (sync LED, fixture) for a fraction of frames.
    `frac` of that camera's frames (all joints) are replaced by a fixed pixel + jitter."""
    rng = np.random.default_rng(seed)
    p2d = p2d.copy()
    if pixel is None:
        w, h = cgroup.cameras[cam].get_size()
        pixel = np.array([0.8 * w, 0.2 * h])
    frames = rng.random(p2d.shape[1]) < frac
    sub = p2d[cam, frames]
    p2d[cam, frames] = pixel + rng.normal(scale=jitter, size=sub.shape)
    return p2d, frames


# ---------------------------------------------------------------- boards

def board_rows(cgroup, board, n_frames=60, noise=0.3, min_corners=8, dist_frac=0.6,
               tilt=0.5, seed=0):
    """Synthetic board detections in the `all_rows` format of calibrate_rows.

    Each frame places the board near the scene centre, facing a random camera (so different
    frames are seen by different camera subsets, like waving a board around a rig), projects
    its inner corners and keeps a camera's detection if >= min_corners corners land in-image."""
    rng = np.random.default_rng(seed)
    objp = board.get_object_points().reshape(-1, 3)
    objp = objp - objp.mean(0)
    C, cen, s = camera_centers(cgroup), scene_center(cgroup), rig_scale(cgroup)
    all_rows = [[] for _ in cgroup.cameras]
    for f in range(n_frames):
        k = rng.integers(len(C))
        # board normal points to camera k, with random tilt and in-plane roll
        n = C[k] - cen; n /= np.linalg.norm(n)
        n = n + rng.normal(scale=tilt, size=3); n /= np.linalg.norm(n)
        a = np.cross(n, [0, 0, 1.0]) if abs(n[2]) < 0.9 else np.cross(n, [1.0, 0, 0])
        a /= np.linalg.norm(a); b = np.cross(n, a)
        roll = rng.uniform(0, 2 * np.pi)
        a, b = np.cos(roll) * a + np.sin(roll) * b, -np.sin(roll) * a + np.cos(roll) * b
        R = np.stack([a, b, n], axis=1)
        center = cen + rng.uniform(-1, 1, 3) * 0.1 * s + (C[k] - cen) * rng.uniform(0, dist_frac) * 0.5
        p3d = objp @ R.T + center
        p2d = project(cgroup, p3d)
        for i in range(len(C)):
            ok = np.all(np.isfinite(p2d[i]), axis=1)
            if ok.sum() < min_corners:
                continue
            ids = np.nonzero(ok)[0]
            corners = p2d[i, ids] + rng.normal(scale=noise, size=(len(ids), 2))
            all_rows[i].append({
                'framenum': (0, f),
                'corners': corners.reshape(-1, 1, 2).astype('float32'),
                'ids': ids.reshape(-1, 1).astype('int32'),
            })
    return all_rows


def board_corners_3d(board, n_frames, cgroup, seed=0):
    """Board corner positions (F, K, 3) at random poses near the scene centre (for scale tests)."""
    rng = np.random.default_rng(seed)
    objp = board.get_object_points().reshape(-1, 3)
    cen, s = scene_center(cgroup), rig_scale(cgroup)
    out = []
    for _ in range(n_frames):
        R = _random_rotation(rng, 0.6)
        out.append((objp - objp.mean(0)) @ R.T + cen + rng.uniform(-1, 1, 3) * 0.1 * s)
    return np.array(out)


def _random_rotation(rng, scale):
    import cv2
    R, _ = cv2.Rodrigues(rng.normal(scale=scale, size=3))
    return R


# ---------------------------------------------------------------- rig perturbation / comparison

def perturb_extrinsics(cgroup, rot_deg=1.0, trans_frac=0.01, fix_first=True, seed=0):
    """A copy of the group with every camera's pose nudged (camera 0 kept if fix_first)."""
    rng = np.random.default_rng(seed)
    cg = deepcopy(cgroup)
    s = rig_scale(cgroup)
    for i, cam in enumerate(cg.cameras):
        if fix_first and i == 0:
            continue
        with torch.no_grad():
            cam.rvec += torch.as_tensor(rng.normal(size=3) * np.deg2rad(rot_deg) / np.sqrt(3),
                                        dtype=cam.rvec.dtype, device=cam.rvec.device)
            cam.tvec += torch.as_tensor(rng.normal(size=3) * trans_frac * s / np.sqrt(3),
                                        dtype=cam.tvec.dtype, device=cam.tvec.device)
    return cg


def umeyama(src, dst, with_scale=True):
    """Similarity (s, R, t) minimising |s R src + t - dst|."""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    xs, xd = src - mu_s, dst - mu_d
    U, S, Vt = np.linalg.svd(xd.T @ xs / len(src))
    D = np.eye(3); D[2, 2] = np.sign(np.linalg.det(U @ Vt))
    R = U @ D @ Vt
    s = (S * np.diag(D)).sum() / (xs ** 2).sum(1).mean() if with_scale else 1.0
    return s, R, mu_d - s * R @ mu_s


def center_error(cg_est, cg_true, with_scale=True):
    """Camera-centre error after similarity alignment, as a fraction of rig scale."""
    a, b = camera_centers(cg_est), camera_centers(cg_true)
    s, R, t = umeyama(a, b, with_scale)
    return np.linalg.norm(s * a @ R.T + t - b, axis=1) / rig_scale(cg_true)


def crowd(cgroup, n_people, n_frames, model=PERSON, size=3.5, sep=1.0, spread=0.05, phase=0.0,
          seed=0):
    """Several skeletons walking near the scene centre, kept apart: (n_people, N, J, 3).

    Each figure follows its own small random walk (`skeleton`, seed + i) recentred on a point
    of a circle of radius `sep` / 3 of rig scale (starting at angle `phase`) in the plane orthogonal to the mean
    viewing direction, so the subjects are never in the same place."""
    s = rig_scale(cgroup)
    up = -camera_axes(cgroup).mean(0)
    up /= np.linalg.norm(up)
    x = np.cross(up, [1.0, 0, 0]) if abs(up[0]) < 0.9 else np.cross(up, [0, 1.0, 0])
    x /= np.linalg.norm(x); y = np.cross(up, x)
    out = []
    for i in range(n_people):
        p3d, _ = skeleton(cgroup, n_frames, model=model, size=size, spread=spread, seed=seed + i)
        a = phase + 2 * np.pi * i / max(n_people, 1)
        centre = scene_center(cgroup) + sep * s / 3 * (np.cos(a) * x + np.sin(a) * y)
        out.append(p3d - p3d.reshape(-1, 3).mean(0) + centre)
    return np.stack(out)



# ---------------------------------------------------------------- timed tracks (STBA / stations)

def smooth_tracks(cgroup, n_points=6, amp=0.08, freq=(0.4, 1.6), seed=0):
    """``f(point, t) -> X [len(t),3]``: smooth 3D tracks (sums of sinusoids, amplitude a fraction of
    rig scale) around the scene centre, like dancers' joints (disco ``synthetic_problem``)."""
    rng = np.random.default_rng(seed)
    c, s = scene_center(cgroup), rig_scale(cgroup)
    base = c + rng.uniform(-0.12, 0.12, (n_points, 3)) * s
    a = rng.uniform(0.3, 1.0, (n_points, 3, 3)) * amp * s
    f = rng.uniform(freq[0], freq[1], (n_points, 3, 3))
    ph = rng.uniform(0, 2 * np.pi, (n_points, 3, 3))

    def fn(p, t):
        t = np.asarray(t, float)
        return base[p] + np.einsum('kd,nkd->nd', a[p], np.sin(2 * np.pi * f[p][None] * t[:, None, None, None][..., 0]
                                                                + ph[p][None]))
    return fn


def timed_observations(cgroup, n_points=6, duration=3.0, fps=60.0, offsets=None, rates=None, rs=None,
                       noise=0.5, amp=0.08, seed=0):
    """Unsynchronised 2D tracks: camera ``c`` records frames at its own times ``tt`` (``fps``, a
    random sub-frame start) and stamps them ``tt``, but sees the scene at true time
    ``tt + offsets[c] + rates[c] * (tt - duration/2) + rs[c] * (v/H - 1/2)`` (STBA convention
    ``true = seed + delta``). Returns ``(obs dict(cam, point, uv, t), track_fn)``."""
    rng = np.random.default_rng(seed + 1)
    C = len(cgroup.cameras)
    fps = np.broadcast_to(np.asarray(fps, float), (C,))
    offsets = np.zeros(C) if offsets is None else np.asarray(offsets, float)
    rates = np.zeros(C) if rates is None else np.asarray(rates, float)
    rs = np.zeros(C) if rs is None else np.asarray(rs, float)
    fn = smooth_tracks(cgroup, n_points, amp=amp, seed=seed)
    out = dict(cam=[], point=[], uv=[], t=[])
    for c in range(C):
        tt = rng.uniform(0, 1 / fps[c]) + np.arange(0, duration, 1 / fps[c])
        H = cgroup.cameras[c].get_size()[1]
        for p in range(n_points):
            ttrue = tt + offsets[c] + rates[c] * (tt - duration / 2)
            uv = project(cgroup, fn(p, ttrue))[c]
            if rs[c]:
                for _ in range(3):
                    uv = project(cgroup, fn(p, ttrue + rs[c] * (np.nan_to_num(uv[:, 1]) / H - 0.5)))[c]
            uv = uv + rng.normal(scale=noise, size=uv.shape)
            ok = np.all(np.isfinite(uv), axis=1)
            out['cam'].append(np.full(ok.sum(), c)); out['point'].append(np.full(ok.sum(), p))
            out['uv'].append(uv[ok]); out['t'].append(tt[ok])
    return {k: np.concatenate(v) for k, v in out.items()}, fn
