# 1.0-bundle: Levenberg-Marquardt bundle adjustment (plan 01 D1-D7, F3, F5 BA side)

## Default change (D5)
- **`CameraGroup.bundle_adjust` and `bundle_adjust_iter` now default to `method='lm'`** (torch
  Schur-complement Levenberg-Marquardt, `aniposelib.bundle`). The 0.9 Adam/OneCycle code is kept
  unchanged as `method='adam'`. *Old behaviour:* pass `method='adam'` (also accepted by
  `calibrate_rows` / `calibrate_videos`).
  - `bundle_adjust_iter(method='lm')` without `extra`: `bundle_adjust_robust` (MSAC inlier
    observations, annealed threshold from ~1.5x the initial error down to 5 px, LOO resection of
    disagreeing cameras, structured rejection, final extrinsics(+intrinsics) stage). With `extra`
    (boards, i.e. `calibrate_rows`): each board is a rigid body with 6 pose parameters (the Adam
    path had free corners + a soft board-shape penalty); an extrinsics-only pass, then a full pass.
    `start_mu`, `end_mu`, `max_nfev`, `ftol`, `n_samp_iter`, `n_samp_full`, `error_threshold`
    only affect `method='adam'`; new keyword-only `spec=`, `return_report=`, `max_points=20000`,
    `resect=True`. Returns the median reprojection error over all observations (robust points).
  - `bundle_adjust(method='lm')`: one LM solve with the robust `loss` (`f_scale` default 1.0 for
    huber, 1.5 otherwise, as before), `n_iter = min(max_nfev, 100)`; `lr` is Adam-only. Returns the
    mean reprojection error as before.
  - **Gauge**: LM holds the first camera's extrinsics fixed and keeps the rig scale (RMS distance of
    the camera centres from their centroid) unless boards fix it; the world frame and scale no
    longer drift. *Old behaviour:* `spec=ParamSpec(intrinsics=None, fixed_cameras=None,
    scale_constraint=None)` (free gauge, damped) or `method='adam'`.
  - Frozen parameters (beyond `n_dist`, skew, `K[1,0]`, bottom row of K, or not in the spec) are not
    in the LM problem at all: bit-for-bit unchanged, as with Adam's hooks.
  - `bundle_adjust_iter(method='lm')` on points (no `extra`) with the default `spec=None` anchors the
    intrinsics stage with Gaussian priors on the starting values (`IntrinsicsPrior(start,
    sigma_logf=0.02, sigma_c=10 px, sigma_k=0.02)`, `cameras._DEFAULT_ITER_PRIOR`): trajectories /
    keypoints rarely cover the image, and unconstrained k2 ran away (k2 +144 on a synthetic 9-cam
    trajectory, 3 % camera-centre drift). *Old behaviour (free intrinsics):* pass
    `spec=ParamSpec(intrinsics=None)`.
- New `CameraGroup.bundle_adjust_lm`, `bundle_adjust_robust`, `resect_cameras`, `place_cameras`,
  `add_camera` (thin delegates to `aniposelib.bundle`).
- `init.calibrate_from_points(refine='auto')` now uses `bundle_adjust_robust` (extrinsics only,
  `revert_if_worse=True`); the previous default is `refine='refine'` (`sfm.refine_group`).
- **Return type (D2):** `bundle_adjust*` return a python `float` (subclass `bundle.BAError`, which
  still answers `.item()`, `.detach()`, `.cpu()`, `.numpy()` for 0.9-style code) instead of a 0-d
  tensor, or a `BAReport` with `return_report=True` (LM only).
- Rodrigues zero-rotation gradient: fixed on `1.0-wire` (no change here).

## New: `aniposelib.bundle`
- `Observations(cam_idx, point_idx, uv, weight, time, group, station)`, `from_dense(p2ds, groups=,
  weights=, time=)` (4-D input adds `frame`/`keypoint` groups), `from_timed(stba.TimedObservations,
  frame_time=)`, `subset`, `to_dense`, `to_dict`.
- `bundle_adjust_lm(cgroup, obs_or_p2ds, p3d_init=None, spec=None, loss='cauchy', f_scale=5,
  n_iter=25, max_points=None, prune=dict(max_px, min_depth), extra=None, stations=None,
  only_extrinsics=False, min_points_per_camera=10, inplace=True, revert_if_worse=False,
  return_report=False, solver='auto'|'dense'|'pcg', rng=None)`: analytic Jacobians (no autograd /
  torch.func), Schur complement on points (or 6-dof rigid boards), dense Cholesky on the reduced
  camera system or matrix-free PCG for large camera counts, Cauchy/Huber/soft-L1 IRLS with true
  robust cost for step acceptance. ~0.1 s for 12 cameras x 1500 points on CPU (Adam: seconds).
- `bundle_adjust_robust(cgroup, p2ds, groups=None, schedule=((200,100,50),(40,20,10,5),(20,10,5)),
  reject=RejectSchedule(frame, keypoint, obs), resect=True, resect_below=0.9, min_inliers=3,
  final=('extrinsics','intrinsics'), spec, contrib, rng)` -> `BAReport` (masks restrict the fit;
  per-camera before/after errors are on all observations).
- `resect_cameras(cgroup, p2ds, cams, tau_tri=5, tau_pnp=8, sample='points'|'frames', frame_id,
  force, min_gain, max_dist)` (LOO PnP-RANSAC, all cameras against the same input rig, degenerate
  solutions rejected: clustered inliers / camera far outside the rig), `place_cameras`,
  `add_camera(cgroup, camera, p2d_cam, p2ds)`; `frame_mask`, `keypoint_mask`, `RejectSchedule`,
  `ResectResult`.
- `BAReport`: error before/after, LM history, per-camera counts and median/p90 before/after, free
  parameter names, frozen cameras, excluded masks, gauge (scale before/after/ratio, fixed
  stations, scale constraint), time offsets, stations, seed/config, `to_dict()`, `summary()`.
- **F3** per-camera time offsets: `ParamSpec(time_offset=sigma)` adds a first-order delta per camera
  (`X(t+delta) ~ X + delta * dX/dt`, velocities from tracks via `obs.time` and `group['track'|
  'keypoint']`), zero-mean prior, gauge camera at 0; result in `report.time_offsets`. Off by default.
- **F5** stations: `stations=` a `stations.Stations` (lookup by `obs.time`, 'skip' dropped) or
  `dict(station_cam, rvec, tvec, names)` with `obs.station`; each station has its own extrinsics,
  intrinsics shared per camera; result in `report.stations`, the camera keeps its first station.

## `aniposelib.params` (full ParamSpec, D1)
- `ParamSpec(intrinsics, n_dist, extrinsics='free'|'fixed'|{cam: ...}, share=[[...]] | {'K':..,
  'dist':..}, priors={cam: {name: (mu, sigma[, 'log'])}} | IntrinsicsPrior(ref, sigma_logf,
  sigma_c, sigma_k), fixed_cameras='auto', scale_constraint='auto'|'keep'|None|(camA, camB, dist[,
  sigma]), skew='fixed', aspect='free', time_offset=None)`. Names: `fx fy cx cy skew f` + distortion
  names (`k1 k2 p1 p2 k3 k4 k5 k6` brown, `k1..k4` fisheye) + `dist`. `intrinsics=None` = each
  camera's own masks (LM default). B14 API (`apply`, `from_camera`, `validate_intrinsics`,
  `matrix_mask`) unchanged; `apply` maps `'f'` to fx, fy. `cgroup.param_spec` (if set) is used by
  LM when no `spec=` is passed. New: `ParamSpec.extrinsics_only()`, `replace()`, `to_dict()`.

## Tests
- `tests/test_bundle.py` (21 tests, incl. scale invariance on 1e-3 / 1 / 1e3 rigs); targets flipped (markers removed, original calls and
  thresholds): `test_rig_lab_board::test_board_ba_recovers_rig_precisely`,
  `test_rig_outdoor::test_ba_with_outliers_and_decoy_camera`,
  `test_rig_keypoint_decoy::test_ba_on_keypoints_recovers_rig`, `::test_ba_on_keypoints_with_decoy`.
- `test_n_dist::test_bundle_adjust_freezes_exactly` runs LM and (n_dist=2) `method='adam'`.
