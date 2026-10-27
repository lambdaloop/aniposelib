# Changelog (aniposelib-pytorch)

## Unreleased

- **Removed (1.1, plan 04 decision 15):** the first-order per-camera time offset in `bundle_adjust_lm`
  (`ParamSpec(time_offset=...)`, `BAReport.time_offsets`). Time offsets live only in `stba.SpatiotemporalBA`
  (`free_offsets='per_camera'` gives the same per-camera deltas, with a spline motion model instead of finite
  differences). Stations (`stations=`) stay in `bundle_adjust_lm`.
- **One LM core (1.1, plan 04 §3):** a shared Levenberg-Marquardt engine (`aniposelib._lm`: IRLS losses, Marquardt
  damping with acceptance on the true cost, Schur complement over generic point-like blocks, dense Cholesky or PCG
  reduced solve). `bundle_adjust_lm` (cost histories unchanged to ~1e-11 relative), `stba.SpatiotemporalBA`
  (unchanged to ~1e-12, about 2x faster; it now uses the bundle projection kernel), `init.sfm.refine_extrinsics`
  (now `bundle_adjust_lm` on ideal pinhole cameras; observations behind a camera are pruned) and
  `intrinsics.lm_board_adjust` (analytic instead of numeric Jacobians; ladder fits agree to ~1e-6 px, more
  iterations to reach `tol`) all run on it. `method='adam'` is unchanged.

- Dev env: the default pixi env now includes all optional extras (vggt @ a288dd0, torchvision cu128, huggingface_hub, safetensors, pillow, onnxruntime, onnx, soundfile, av, pyyaml); the real VGGT-1B smoke test (`-m slow`) now runs when the weights are in the local HF cache.
- `CharucoBoard.draw` / `Checkerboard.draw` default to `margin='auto'` (half a square of white border), so rendered boards are directly detectable (edge-to-edge renders lost most edge corners: 6/20 on a 6x5 board). `margin=0` restores the old edge-to-edge output.

## 1.0.0

Still imports as `aniposelib`; still cannot be installed alongside classic `aniposelib` (see README).
The sections below are the per-branch change notes (`changes/*.md`); each default change says how to get the
old behaviour. Summary of what can change results of existing code:

### Upgrading from 0.9: default and behaviour changes
- **Return types (D2):** public read-only methods return numpy for numpy input, detached torch for torch input;
  `return_grad=True` keeps the graph (old behaviour). Drop `.detach().cpu().numpy()` on results of numpy input.
- **Camera order on load:** numeric (`cam_2` < `cam_10`); `order='lexical'` for <= 0.9. Rigs with more than 10
  cameras loaded with <= 0.9 were permuted.
- **Bundle adjustment:** `bundle_adjust` / `bundle_adjust_iter` default to `method='lm'` (Schur-complement
  Levenberg-Marquardt, gauge fixed: first camera and rig scale), and return a python float (`BAError`).
  The 0.9 Adam path is `method='adam'`. `bundle_adjust_iter` on points anchors intrinsics with priors
  (`spec=ParamSpec(intrinsics=None)` frees them). Boards are rigid bodies in LM.
- **Triangulation:** `triangulate(method='linear')` (block DLT; `method='svd'` is the old one);
  `triangulate_ransac` is MSAC (`tau=10` px), `triangulate_possible` uses the pairwise kernel.
- **Projection:** full K (skew honoured), rational Brown terms k4..k6 used, exact depth division, converged pinhole
  undistortion; zero-skew 5-coefficient calibrations are unchanged up to ~1e-10.
- **Board poses** are in the zero-origin frame of `get_object_points()` (`origin='opencv'` for the old one);
  marker detection `adaptiveThreshWinSizeMin` 50 -> 3; `opencv-python` is enough (no contrib).
- **`optim_points(fill='interior')`**; unknown keyword arguments raise; no global RNG (`rng=` everywhere).
- **Crop offsets** (`Camera.offset`, the tailcyclenet `offset` key) are applied by aniposelib: do not subtract
  them again downstream (tailcyclenet's own offset handling would apply them twice).
- `init.calibrate_from_points(scale=<reference>)` aligns the gauge too; `init.vggt.predict(mode='pad')`.

### Known limitations
- `AffineCamera` (telecentric) is not usable inside a `CameraGroup` (TOML, triangulation, BA); use
  `affine.triangulate_affine` and `AffineCamera.to_pinhole`.
- `CameraGroup.triangulate_robust` raises `NotImplementedError` for groups that cannot be packed (custom camera
  subclasses); `progress=` is ignored on the kernel paths; `as_tensors()['size']` is float (NaN if unknown).
- `aniposelib.sync` estimators return numpy inside their result dataclasses whatever the input type (deliberate
  deviation from D2: they are reports).
- The VGGT network has not been run in CI (no `vggt` in the test environment).
- BA stations: only `bundle_adjust_lm` with timed `Observations` reads `cgroup.stations` automatically.

### 1.0-core (Phase B: correctness and conventions)

- **Camera order on load (B1).** `CameraGroup.load/loads/from_dict` sort camera keys numerically
  (`cam_2` < `cam_10`); files with more than 10 cameras were permuted by aniposelib <= 0.9.
  A `CameraOrderWarning` is issued when numeric and lexical order differ, and on duplicate camera
  names. Old behaviour: `CameraGroup.load(f, order='lexical')`; `order='file'` keeps section order.
- **`metadata=None` default (B2).** `CameraGroup(cameras)` no longer shares one mutable `{}`;
  `copy`, `subset_cameras*` deep-copy the metadata.
- **Unknown per-camera keys round-trip (B3).** Keys `Camera` does not know (`moving`, `calibrated`,
  `model`, ...) are kept in `Camera.extra` and written back by `get_dict`/`dump`.
- **Distortion padding (B4).** Short `distortions` are zero-padded on load to the model length
  (5 for pinhole, 4 for fisheye). Longer vectors are kept.
- **String/dict I/O (B5).** New `CameraGroup.dumps()`, `CameraGroup.loads(s)`, `to_dict()`,
  `CameraGroup.from_dict(d)` (with metadata). `load`, `from_dicts`, `from_names` are now
  staticmethods; `Camera.from_dict` is a classmethod.
- **Return types follow the input (B6, D2).** Public read-only methods (`project`, `triangulate`,
  `triangulate_possible`, `triangulate_ransac`, `triangulate_optim`, `reprojection_error`,
  `undistort_points`, `distort_points`, `average_error`, `optim_points`, `is_point_visible`,
  `projection_sensitivity`, `get_triangulation_sensitivity`) return numpy for numpy/list input and
  detached torch (on the input's device) for torch input; they run under `no_grad`.
  Old behaviour (tensors with autograd graph): pass `return_grad=True`. Callers that did
  `.detach().cpu().numpy()` on results of numpy input can drop it (it now fails on numpy).
  Getters (`get_rotations`, `get_camera_matrix`, ...) still return the parameters.
- **Public `CameraGroup.undistort_points` (B7)**; `_undistort_points_batch` is an alias.
  `triangulate(cgroup.undistort_points(p), undistort=False)` is the fast path for repeated calls.
- **Unknown keyword arguments raise (B8).** `bundle_adjust*` raise `TypeError` (as before);
  `calibrate_rows`/`calibrate_videos` now validate their `**kwargs` *before* detecting boards.
- **No global RNG (B9).** `bundle_adjust_iter(rng=)`, `bundle_adjust(rng=)`, `resample_points(rng=)`,
  `get_point_cloud(rng=)` take a `np.random.Generator`, int seed or None. None draws fresh entropy
  (it used to consume `np.random` global state, so `np.random.seed(...)` no longer makes these
  reproducible: pass `rng=<int>`).
- **`utils.select_matrices` / `mean_transform_robust` (B10).** No crash on identical (zero
  variance) transforms or fewer than 2; `mean_transform_robust(approx=None)` seeds with the median
  transform (was: plain mean of all); if nothing is within `error` the closest transform is used
  instead of a NaN mean. New `utils.median_transform`.
- **`optim_points(fill='interior')` default (B11).** Frames before a joint's first / after its last
  observation (>= 2 views) are returned as NaN instead of extrapolated; interior gaps are still
  filled. `fill='none'` NaNs every unobserved frame. Old behaviour: `fill='all'`.
- **Speed (B12).** Fisheye undistort Newton loop exits early (<= 1e-12 difference);
  `triangulate`/`undistort_points` default `batch_size=None` (no chunking of the undistort; the SVD
  runs in chunks of `CameraGroup.TRIANGULATE_CHUNK`=65536; results unchanged). `project` moves input
  to the camera device (results come back on the input's device). New `with cgroup.frozen():`
  caches extrinsics matrices for read-only loops (solvers refuse to run inside it).
  Old chunking: `triangulate(..., batch_size=1000)`.
- **Full-K projection (B13, D3).** `project`/`undistort_points`/`distort_points` use the skew
  `K[0,1]` (and `K[1,0]`), for pinhole and fisheye (fisheye skew = cv2's `alpha * fx`). Board
  calibrations (zero skew) are bit-identical. `projection_sensitivity` includes the skew.
  There is no switch back; zero the skew in the matrix to get the old projection.
- **Parameter masks (B14).** New `aniposelib.params` (`ParamSpec`, `MATRIX_ENTRIES`). Each camera
  has `free_intrinsics` (default `('fx','fy','cx','cy')`); skew, `K[1,0]` and the bottom row of K
  are frozen by gradient hooks (like `n_dist`), also for user-built optimizers, surviving
  copy/deepcopy/pickle/TOML (`free_intrinsics` key written only when non-default).
  `cam.set_free_intrinsics(('fx','fy','cx','cy','skew'))` frees the skew. Replaces downstream
  `register_hook` / `freeze_matrix_structure` hacks.
- **`dtype=` (B15).** `Camera(dtype=)`, `CameraGroup(cameras, dtype=)`, `CameraGroup.load(f, dtype=)`,
  `cgroup.to(torch.float32)`; computation follows the camera dtype (default float64).
- **numba dropped (B16)** from imports and dependencies.
- **Crop offsets (B18).** `Camera(offset=(x, y))`: the stored-image origin in sensor pixels
  (matrix in sensor px, `size` = stored image). `project` subtracts it, `undistort_points` adds it;
  written as `offset = [x, y]` only when non-zero (the tailcyclenet `calibration.toml` key, which
  is now applied by aniposelib, so do not subtract it again downstream). `resize_camera` scales it.
  `CameraGroup.apply_crop_offsets(config)` accepts an anipose config (`[cameras.<name>] offset =
  [x, y, w, h]`, sets size to `(w, h)` unless `set_size=False`) or `{name: [x, y]}`.
  `get_offsets`/`set_offsets` on the group.

### 1.0-wire: `CameraGroup` on the packed kernels (plan 01 C1–C3 wiring, D8 video sampling)

- **Delegation.** `CameraGroup.project`, `reprojection_error`, `undistort_points` (= `_undistort_points_batch`),
  `triangulate`, `triangulate_ransac`, `triangulate_possible` (and everything built on them: `average_error`,
  `optim_points`, `triangulate_optim`, the inner triangulation/projection of `bundle_adjust*`) now run on
  `aniposelib.geometry` / `aniposelib.triangulation` (one batched kernel instead of a per-camera loop), whenever
  every camera is a plain `Camera`/`FisheyeCamera` (pinhole ≤ 8 coefficients). Subclasses that override
  `_project`/`_undistort_points`/`get_extrinsics_mat` keep the per-camera path. Gradients still reach the
  camera parameters (BA, `return_grad=True`), the n_dist / free_intrinsics masks still apply, and
  `with cgroup.frozen():` is honoured. The default test suite went from ~45 s to ~25 s on CPU.
  Numerical differences against 0.9 (per-camera code is unchanged):
  - projection divides by the exact depth (the pinhole path used `z + 1e-10`): ~1e-10/z relative;
  - pinhole undistortion converges (fixed point + Newton, ≤1e-12) instead of 15 fixed-point steps: identical
    where the old iteration had converged, more accurate where it had not (strong distortion near the edges);
  - **rational Brown terms k4..k6** (distortion vectors of length 6–8) are now used everywhere, also by the
    per-camera `Camera.project`/`undistort_points` (they were silently ignored). 5-coefficient calibrations
    are unchanged. *Old behaviour:* truncate the distortions to 5 entries.
  *Old per-camera results:* `torch.stack([cam.project(X) for cam in cgroup.cameras])`, `cam.undistort_points`.
- **`triangulate(method='linear')` is the new default** (recentred inhomogeneous block DLT with a 1e-12 ridge;
  faster and better conditioned for far scenes). Differs from the SVD DLT by ~1e-6 relative on noisy data,
  ~1e-10 on noiseless data. This also changes the inner triangulation of `bundle_adjust*`/`average_error`
  slightly. *Old behaviour:* `triangulate(..., method='svd')`.
- `triangulate` new keywords: `extrinsics=` (`[C,4,4]` override, per point `[C,N,4,4]`, or per frame `[C,T,4,4]`
  with points `(C,T,...,2)`, for moving rigs; both methods) and `min_cams=2`. Points may have any leading
  shape `(C,...,2)` → `(...,3)`. `batch_size`/`progress` keep their meaning.
- **`CameraGroup.packed(device=None, dtype=None, detach=True)`** → `geometry.PackedCameras`,
  **`load_packed_(P)`** (write back after an external optimiser) and **`as_tensors(device, dtype=float64,
  detach=True, extrinsics=None)`**: tracktail's `format_camera_group` contract as one dict of stacked tensors
  (`K`, `dist [C,5]` (4 for all-fisheye, up to 8 with rational terms), `ext`, `ext_inv`, `center`, `size`
  (float, NaN if unknown), `offset`, `type` ('pinhole'/'fisheye'), plus `model`, `names`, `moving`).
  `extrinsics=` takes `[C,T,4,4]` or `{name: [T,4,4]}` (missing cameras keep their static pose).
- **`CameraGroup.triangulate_robust(points, method='msac'|'irls'|'reject', **kw)`** → `TriangulationResult`
  (thin wrapper over `triangulation.triangulate_robust`). CameraGroup-only `check_invertible=True` (default):
  observations where the lens model is not invertible, and inliers whose undistorted ray disagrees with the
  fitted point (projection past the fold of a strong barrel distortion, where a pixel has a second preimage),
  are not used to fit; those points are re-triangulated without them. Errors are still reported for every
  observation.
- **`triangulate_ransac` is MSAC** (`triangulate_robust(method='msac', tau=10, min_inliers=min_cams)`), same
  4-tuple `(p3d, picked [C,N,1], points_2d, errors)`: picked = inlier views (error < `tau` px), errors = mean
  pixel error over picked views. New keywords `tau=10.0`, `n_refine=2`. Differences: a view within `tau` of
  the consensus is kept even if it is slightly off (the 0.9 search picked the lowest-mean-error subset and
  excluded the full set); points with exactly 2 views are now triangulated (0.9 returned NaN); points
  without a solution have error NaN (was 0). *Old behaviour:* none built in (subset search is kept only as the
  fallback for non-packable cameras); use a smaller `tau` to be stricter.
- **`triangulate_possible` uses the pairwise kernel** (no subset enumeration, memory bounded; fixes the OOM),
  same return tuple. New `tau=10.0`, `n_refine=2`; `batch_size` default `None` (automatic memory bound; was
  1000). Errors NaN (was 0) where no point.
- **`rodrigues()` gradient at zero rotation.** `cameras.rodrigues` (and `transform_points`, used for board
  poses in BA) use the series-expanded `geometry.rodrigues`: exact zero rotation vectors (e.g. the reference
  camera) now get the correct rotation gradient (0.9 returned a constant identity, i.e. zero gradient, so BA
  could not rotate such a camera until something else moved it). This changes BA trajectories slightly.
  No switch back.
- **`get_rows_videos(videos, board, verbose=True, *, frame_indices=None, n_workers=1, detector=None,
  detector_kwargs=None, skip=20)`**: `frame_indices` (one int sequence for all videos, per-camera lists of
  per-video sequences, or a dict keyed by video filename or camera name) is passed to
  `board.detect_video(frame_indices=)`; `n_workers>1` detects videos concurrently in threads (same result);
  `detector=` ('opencv', 'deepcharuco' or an instance) uses a copy of the CharucoBoard with that detector.
  `calibrate_videos` accepts the same four keywords. Defaults reproduce the old behaviour.
- Tests: `tests/test_geometry.py`/`test_triangulation.py` parity tests now compare against the per-camera path
  / `method='svd'` explicitly (the group methods delegate); `test_triangulate_ransac.py::test_outlier_exclusion`
  passes `tau=2.0` (some N(0,30 px) corruptions are within the default 10 px threshold). New `tests/test_wire.py`.

### kernels (1.0-kernels): packed geometry + fast/robust triangulation (plan 01 C1-C3)

New, additive modules (not yet wired into `CameraGroup`; that happens at merge):

- `aniposelib.geometry`: `PackedCameras` (stacked `K [C,3,3]`, `dist [C,8]` zero-padded with a
  `brown`/`kb` model tag, `rvec/tvec [C,3]` or `ext [C,4,4]` / per-frame `[C,T,4,4]`, `offset [C,2]`,
  `size [C,2]`), `pack(cgroup)`, `unpack_into(cgroup, P)`, vectorised `project` (→ uv, depth),
  `undistort(check_invertible=)`, `distort`, `reproj_err`, `project_jacobian`.
- `aniposelib.triangulation`: `dlt_blocks`, `solve_blocks`, `triangulate_linear` (recentred inhomogeneous
  block DLT + ridge, `extrinsics=` for moving rigs), `triangulate_robust(method='msac'|'irls'|'reject',
  contrib=, mask_out_of_bounds=, ...)` → `TriangulationResult`, `triangulate_possible` (pairwise
  multi-candidate kernel, memory-bounded) → `PossibleResult` (`.legacy()` = old 4-tuple), `refine_points`.

Numerical differences from the per-camera `CameraGroup` path (relevant once methods delegate):
- Projection honours the full K (skew `K[0,1]`, and `K[1,0]`); the old path ignored skew (D3). The bottom
  row of K is still ignored. Zero-skew calibrations are unchanged.
- Projection divides by the exact depth; the old pinhole path divided by `z + 1e-10` (≈1e-8 px on
  metre-scale rigs). Behind-camera points get `inf` in `reproj_err`.
- Brown model uses all 8 OpenCV coefficients (`k4..k6` rational terms were silently ignored).
- Brown undistortion converges (5 fixed-point steps + Newton, ≤1e-12) instead of 15 fixed-point steps
  (old residual up to ~1e-5 px on the outdoor rig). Old behaviour: `CameraGroup._undistort_points_batch`.
- Sensor crop `offset` (from `cam.offset` or `cam.extra['offset']`) is subtracted on projection and added
  on undistortion.
- `triangulate_linear` differs from the SVD `triangulate` by ~1e-6 relative on noisy data (different
  algebraic objective; identical to ~1e-10 on noiseless data). The ridge default is 1e-12 (tracktail's 1e-8
  biases well-posed points by ~1e-8 of rig scale).

### 1.0-bundle: Levenberg-Marquardt bundle adjustment (plan 01 D1-D7, F3, F5 BA side)

#### Default change (D5)
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

#### New: `aniposelib.bundle`
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

#### `aniposelib.params` (full ParamSpec, D1)
- `ParamSpec(intrinsics, n_dist, extrinsics='free'|'fixed'|{cam: ...}, share=[[...]] | {'K':..,
  'dist':..}, priors={cam: {name: (mu, sigma[, 'log'])}} | IntrinsicsPrior(ref, sigma_logf,
  sigma_c, sigma_k), fixed_cameras='auto', scale_constraint='auto'|'keep'|None|(camA, camB, dist[,
  sigma]), skew='fixed', aspect='free', time_offset=None)`. Names: `fx fy cx cy skew f` + distortion
  names (`k1 k2 p1 p2 k3 k4 k5 k6` brown, `k1..k4` fisheye) + `dist`. `intrinsics=None` = each
  camera's own masks (LM default). B14 API (`apply`, `from_camera`, `validate_intrinsics`,
  `matrix_mask`) unchanged; `apply` maps `'f'` to fx, fy. `cgroup.param_spec` (if set) is used by
  LM when no `spec=` is passed. New: `ParamSpec.extrinsics_only()`, `replace()`, `to_dict()`.

#### Tests
- `tests/test_bundle.py` (21 tests, incl. scale invariance on 1e-3 / 1 / 1e3 rigs); targets flipped (markers removed, original calls and
  thresholds): `test_rig_lab_board::test_board_ba_recovers_rig_precisely`,
  `test_rig_outdoor::test_ba_with_outliers_and_decoy_camera`,
  `test_rig_keypoint_decoy::test_ba_on_keypoints_recovers_rig`, `::test_ba_on_keypoints_with_decoy`.
- `test_n_dist::test_bundle_adjust_freezes_exactly` runs LM and (n_dist=2) `method='adam'`.

### boards (B17, D8, D9, D10)

#### OpenCV / dependencies
- **`opencv-contrib-python` → `opencv-python>=4.8,<5`.** `boards.py` now uses only the main-module ArUco class
  API (`cv2.aruco.ArucoDetector`, `CharucoDetector`, `CharucoBoard.generateImage`, `solvePnP`). The ChArUco path
  works again on every OpenCV ≥ 4.8 (it silently found nothing on opencv-python ≥ 4.7 / contrib ≥ 4.10).
  *Old behaviour:* having `opencv-contrib-python` installed instead still works (it also provides `cv2`); do not
  install both.
- New pixi env `nocontrib` (`pixi run -e nocontrib test-nocontrib`) with only `opencv-python`, plus a guard test
  that fails if any legacy contrib `cv2.aruco.*` free function (`detectMarkers`, `interpolateCornersCharuco`,
  `estimatePoseCharucoBoard`, ...) reappears in `aniposelib/`.
- New optional extra `aniposelib-pytorch[deepcharuco]` (`huggingface_hub`, `onnxruntime`, `onnx`).

#### `CharucoBoard`
- New keyword-only arguments: `legacy_pattern=False` (calib.io / OpenCV < 4.6 boards with an even number of rows
  need `True`), `detector='opencv'|'deepcharuco'|instance`, `detector_kwargs=`, `detector_params=`
  (`cv2.aruco.DetectorParameters` or a dict of fields), `validate=None`, `max_homography_px=3.0`.
- `aruco_dict=` is now honoured (a `cv2.aruco.Dictionary`, a `cv2.aruco.DICT_*` constant or a name such as
  `'DICT_5X5_100'` / `'APRILTAG_36h11'`). It used to be ignored. An unsupported `(marker_bits, dict_size)` now
  raises `ValueError` listing the valid pairs (was a bare `KeyError`).
- `CharucoBoard.from_rows_cols(rows, cols, ...)` (= `CharucoBoard(squaresX=cols, squaresY=rows, ...)`), and
  `.rows` / `.cols` properties.
- `draw(size, margin=0, border_bits=1)` works (uses `generateImage`; it raised `AttributeError` on OpenCV ≥ 4.7).
  `Checkerboard.draw(size)` now renders a board (it returned `None`).
- **Marker detection default `adaptiveThreshWinSizeMin` 50 → 3** (OpenCV's default; finds ~25 % more small or
  distant markers). *Old behaviour:* `CharucoBoard(..., detector_params={'adaptiveThreshWinSizeMin': 50})`.
- No bare `except`: only `cv2.error` is caught (with a warning); programming errors propagate.
- `detect_image` returns corners `(N, 1, 2)` float32 and ids `(N, 1)` int32; **an empty detection is now
  `(0, 1, 2)` / `(0, 1)`** instead of `np.float64([])` of shape `(0,)`. `len(...)`/`.size` checks behave the same.
- **Board poses (`estimate_pose_points`) are in the zero-origin frame of `get_object_points()`** (first inner
  corner at the origin), consistent with `extract_points` and the bundle adjustment. 0.9.x used
  `estimatePoseCharucoBoard`, whose frame has the outer board corner at the origin, i.e. `tvec` differed by
  `R @ (square_length, square_length, 0)`. Relative camera extrinsics are unaffected. *Old behaviour:*
  `estimate_pose_points(..., origin='opencv')`; also `get_object_points(origin='opencv')`
  (= `board.getChessboardCorners()`). Pose = planar `solvePnP` (IPPE + LM refine; fisheye via undistorted points);
  collinear corner sets return `(None, None)`. New `min_corners=7` keyword.
- `validate_detection(corners, ids, max_homography_px=3.0)` → `DetectionCheck` (truthy; `max_residual_px`,
  `inlier_fraction`, `to_dict()`): the homography gate (wrong ids / transposed board / wrong `legacy_pattern`).
  Applied automatically when `validate=True` (default only for non-OpenCV detectors).
- `detect_video`/`detect_images` warn (`RuntimeWarning`) when most detections do not fit the board layout
  (typical of swapped `squaresX`/`squaresY` or a wrong `legacy_pattern`).
- `detect_video(..., frame_indices=[...])` detects exactly those frames (batched for batched detectors);
  `detect_images` also accepts decoded image arrays; new `detect_frames(frames)`.
- `CharucoBoard` pickles (cv2 objects are rebuilt), so it can be sent to worker processes.
- `Checkerboard.estimate_pose_points` returns `(None, None)` when `solvePnPRansac` fails (`retval` False) instead
  of returning its garbage output, and warns instead of printing.
- New helpers in `aniposelib.boards`: `draw_corners` (replaces `cv2.aruco.drawDetectedCornersCharuco` in manual
  verification), `homography_check`, `get_aruco_dictionary`, `default_detector_params`, `solve_board_pnp`,
  `empty_corners`.

#### `aniposelib.detectors` (new, D8)
- `DeepCharucoDetector(board, device='auto', backend='torch'|'onnx', batch_size=16, refine=True, scale=1.0,
  min_score=0.0, subpix=False, color_order='bgr', weights_dir=None, repo='lambdaloop/deepcharuco-weights')`:
  Lightning-free batched DeepCharuco nets (from disco `deepcharuco_disco.py`) with EvoVision's coordinate
  conventions (isotropic scale, pad to /8, `cv2.resize` pixel-centre inverse) and ONNX export. Weights are
  resolved from `weights_dir` / `$DEEPCHARUCO_WEIGHTS`, the Hugging Face cache, `~/.cache/aniposelib/deepcharuco`,
  then downloaded (huggingface_hub if installed, else plain HTTPS). Unsupported boards raise `ValueError` listing
  the published geometries (5–9 rows × 5–9 columns; r6_c5 has no refinenet).
- Use as `CharucoBoard(..., detector='deepcharuco', detector_kwargs=dict(device='cuda'))`; returns the same rows
  format, so `calibrate_rows`/`calibrate_videos` work unchanged.

#### `aniposelib.intrinsics` (new, D9)
- `calibrate_ladder(objpoints, imgpoints, image_size, models=...)` / `calibrate_ladder_rows(rows, board, size)`:
  fits pinhole → k1k2 → k1k2p → standard_5param (→ rational, fisheye_k1k2, fisheye) with nested inits and a
  Schur-LM solver, scores each on held-out time blocks, and picks the most parsimonious model within 15 % of the
  best held-out p95 (ported from disco `board_intrinsics_ba.py`). `LadderResult.to_camera()` gives a
  `Camera`/`FisheyeCamera` with the matching `n_dist`.

- `bundle_adjust(method='adam')` / `bundle_adjust_iter(method='adam')` are now scale-invariant. Translation-like
  parameters (camera tvecs, 3D points, board translations) get learning rates scaled by
  `rig_scale / ADAM_REF_SCALE` (rig_scale = median camera-to-point distance, `ADAM_REF_SCALE = 1.0`), and their
  gradients are clipped in normalised units. Before, the absolute step sizes stalled on mm-scale or tiny rigs and
  drifted or diverged on others: camera-centre error after BA ranged 0.09–3.5% of rig scale depending only on
  units; it is now identical at 1e-3, 1 and 1e3. Old behaviour: `trans_scale=None`. A fixed scale:
  `trans_scale=<float>`.

### quality (1.0-quality): calibration QA and global scale/gauge (plan 01 C4, C5)

New, additive modules (functions taking a `CameraGroup`; `CameraGroup` method aliases such as
`cgroup.rescale`/`cgroup.align_to` are added at merge). Nothing existing changes behaviour.

- `aniposelib.quality`:
  - `loo_reprojection(cgroup, p2ds, tau_tri=5, min_others=3, robust=True)` -> `LOOResult`:
    leave-one-camera-out reprojection with a **fixed denominator** (eligible = target sees the
    point and >= `min_others` other cameras do; unsupported triangulations are misses, `inf`).
    Per camera `median_px`, `p10/p90_px`, `coverage`, `agree@5/8/10`; `overall['worst_camera']`.
    Robust (MSAC from the other cameras, one batched triangulation for all cameras) by default;
    `robust=False` is the EvoVision plain-linear LOO. `loo_errors` returns the raw arrays.
  - `report(cgroup, p2ds, robust=True, loo=True, min_views=3, fit_mask=None, groups=None,
    thresholds=(5, 8, 10))` -> `CalibrationReport` (`per_camera`, `overall`, `per_group`,
    `passes(max_median_px=12, max_loo_px=60)`, `failures()`, `to_dict()`, `to_json(path)`,
    `table()`). Fit masks only add fit-set / held-out columns; they never change the scoring.
  - `paired_bootstrap(res_a, res_b, clusters, n_boot=2000, seed=0, threshold=8)`,
    `temporal_split(starts, block, guard, fractions=(.6, .2, .2), seed=0, window=0)` (Emory).
  - `compare(cg_a, cg_b, p2ds=None, trim=2.5)` -> `CompareResult` (trimmed-Umeyama centre
    residuals, scale, rotation, per-camera orientation / focal / principal-point deltas, paired
    LOO errors).
  - `sanity(cgroup, f_range=None, pp_frac=0.2, k_max=None)`, `lens_validity(camera)` (grid
    round trip + monotone radial map; Emory), `coverage(p2ds, min_cams=3, cgroup=None)`,
    `board_shape_residual(cgroup, rows|corners, board)` (board units, scale free; EvoVision),
    `ray_angles(cgroup, p3d, p2ds=None, min_deg=2)` (grazing camera pairs),
    `assess_reuse(cgroup, p2ds, good=(12, 18), medium=(25, 40))` -> 'good'/'medium'/'bad'
    (EvoVision `calib_reuse.assess/classify`, coverage gate first).
- `aniposelib.scale`:
  - in place, return the group: `rescale(cgroup, s, units=None, source=None, quality=None)`,
    `apply_similarity(cgroup, s, R, t)` (world' = s R X + t), `anchor_to(cgroup, camera)`.
  - `align_to(cgroup, target, by='centers', trim=2.5, min_keep=4, with_scale=True)` ->
    `AlignResult` (trimmed Umeyama on camera centres; an outlier camera is excluded from the
    gauge; residuals below 1e-9 x rig scale never trigger trimming). In place.
  - estimates (do not modify the group; `ScaleResult` unpacks as `s, res`; `res.apply(cgroup)`
    rescales and writes provenance, refusing unaccepted estimates):
    `from_board(cgroup, rows|corners, board, min_frames=5, min_pairs=30, min_pair_dist=None,
    max_frame_mad=0.05)` (EvoVision `scale_from_charuco`: per-frame medians, frame MAD /
    spread, pair MAD, board shape residual, flags), `from_lengths(cgroup, p2ds, pairs,
    ref_lengths, tau=5, min_inliers=3, joints=None)` (Emory anatomical scale),
    `from_reference(cgroup, ref)` (trimmed centre similarity, `scale_reference_mismatch` flag).
  - `align_world(cgroup, ground_points=None, up='z', origin='centroid')`: floor plane -> `up`.
  - Metadata provenance keys written by `rescale`/`ScaleResult.apply`: `units`, `scale_source`,
    `scale_value` (cumulative factor), `scale_quality` (TOML-safe dict; None/NaN dropped).
    `CharucoBoard` has no `units` attribute yet; `from_board(units=...)` (or `board.units` if
    present) sets it.
  - helpers: `camera_centers`, `umeyama`, `fit_similarity(trim=|inlier_frac=)`, `board_points`.
- Tests: `tests/test_quality.py`, `tests/test_scale.py`; target
  `test_rig_keypoint_decoy.py::test_robust_leave_one_out_isolates_decoy` now passes (marker
  removed); `test_rig_fisheye_people.py::test_metric_scale_from_board` also checks
  `scale.from_board`.

### 1.0-init: `aniposelib.init` (plan 01 phase E: E1-E5)

New sub-package `aniposelib.init` (additive; nothing existing changes, `utils.get_initial_extrinsics` and
`cameras.init_camera_intrinsics` are untouched; aliases/wiring into `CameraGroup` can follow at merge). Not imported
by `aniposelib/__init__.py`; `import aniposelib.init` imports no optional dependency.

- **`init.board` (E1).** `initial_extrinsics(rtvecs, cam_names, method='auto'|'classic'|'robust')` -> `BoardInit`:
  never raises on **disconnected board graphs**; one gauge per connected component (anchor camera = identity),
  `components`, `anchors`, `component_of`, per-edge `method`/dispersion, `apply(cgroup, component=None)`. Edges of a
  maximum spanning tree (most co-detections) use the classic cluster+mean estimate and fall back to a robust one
  (`robust_relative_transform`: trimmed translations, robust chordal rotation mean, median translation; disco
  `robust_fallback_extrinsics`). `board_graph`, `board_components`.
  `init_camera_intrinsics(camera, objp, imgp, size, bounds=None, fallback='nominal'|'keep'|'raise'|K|dict)` ->
  `IntrinsicsSeed`: guards the classic seed with `valid_intrinsics` (focal 0.2-5 x max(w,h), aspect 0.6-1.67,
  principal point in the central 60 %, |k1| <= 1; `DEFAULT_INTRINSICS_BOUNDS`, `bounds=False` disables) and
  catches cv2 errors (disco `valid_intrinsic_seed`). *Old behaviour:* `aniposelib.cameras.init_camera_intrinsics`.
- **`init.sfm` (E2).** `from_correspondences(cgroup, p2ds, frame_id=None, window_id=None, min_inlier_frames=8,
  min_inlier_windows=4, driver_frac=0.25, max_center_ratio=3, rotation_priors=None, local_ba_every=3, rng=None)`
  -> `(CameraGroup of the placed cameras, InitReport)`: port of the motor-observatory robust incremental SfM
  (pairwise essential seed, frame-RANSAC PnP, drivers/passengers, window-count gate, runaway-centre rejection
  with a retry that excludes the frames of the runaway solution (decoy cameras), rotation-prior rescue, robust
  IRLS cloud with (camera, frame) decoy masking). Uses **each camera's own intrinsics** (undistorted ideal pixels,
  no shared virtual K). Local and final refinement by `refine_extrinsics` (new scale-free Schur-complement LM on
  ideal pixels with outlier trimming; the Adam `bundle_adjust_iter` diverges on unit-baseline SfM rigs).
  Also `refine_group(cgroup, p2ds)` (robust extrinsics-only refinement with per-camera gates), `pairwise_poses`,
  `frame_pnp_ransac`, `place_with_rotation_prior`, `solve_t_given_R`, `ideal_points`.
  `InitReport` (method, placed, drivers, passengers, rejected{name: reason}, missing, components, stats, `ok`,
  `to_dict()`).
- **`init.people` (E2, people bootstrap).** `associate_people(cgroup, p2ds [C,F,P,J,2], track_ids=None)` recovers
  cross-camera person identities from epipolar geometry (person-hypothesis essential RANSAC scored by mutual-best
  ratio-tested matches, dominant tracklet votes, union-find; disco `pose_ba_boot`), `from_people(...)` then runs
  `from_correspondences`. Not ported: the phone focal grid search and the depth-ratio spanning-tree composition.
- **`init.vggt` (E3, optional extra `[vggt]` = `vggt`, `huggingface_hub`, imported only on use).** `predict(frame_sets,
  model='facebook/VGGT-1B', device=None, dtype='auto', local_files_only='auto', ...)` on paths or in-memory RGB
  frames (bf16/fp16/fp32 auto, local cache first, model freed unless `keep_model=True`), intrinsics restored to
  **original image pixels** (resize, crop/pad and mixed-batch padding: `preprocess_transform`, `restore_intrinsics`,
  `restore_padded_intrinsics`). `aggregate(..., align='sim3', rotation_mean='chordal'|'robust'|'median')` -> robust
  Sim(3) alignment to a medoid template + outlier-template rejection + QA (`raw_scale_rel_spread`, kept templates,
  rotation/centre dispersion); EvoVision/motor averaging in camera 0's frame is `align='first'`.
  `pairwise_rotations` / `predict_pairwise` (SfM rotation priors; their dispersion does not predict accuracy),
  `to_cameragroup`, `from_extrinsics`. The real-network test is `slow` and runs only if the weights are cached.
- **`init.select` (E4).** `select(candidates, p2ds, score='robust_inliers'|'median_px', tau=30, min_inliers=3)` ->
  `(best CameraGroup, SelectResult)`: MSAC robust-triangulation inlier fraction (Emory `vggt_pose_init`) or median
  robust reprojection (motor `select_vggt_template`); `score_candidate`.
- **`init.calibrate_from_points(p2ds, cam_names=None, sizes=None, intrinsics=None, init='auto', frames=None,
  groups=None, robust=True, scale=None, ...)` (E5)** -> `(CameraGroup, RecipeReport)`: candidates (reference
  extrinsics / SfM / VGGT aggregate + templates / given groups), `select`, refinement (`refine='auto'`:
  `sfm.refine_group`; `'robust'`: `cgroup.bundle_adjust_robust`; `'iter'`: `bundle_adjust_iter`, kept only if it
  lowers the robust error),
  scale via `aniposelib.scale` (float, reference group, `dict(pairs=, lengths=)`, `dict(board=, rows=)`), and
  `quality.report`. `make_intrinsics_group(cam_names, sizes, intrinsics)`.
- `pyproject.toml`: package `aniposelib.init`, extra `[vggt]`.

### 1.0-sync: `aniposelib.sync` (plan 01 F1, F2, F7)

New optional module; nothing existing changes. Import with `from aniposelib import sync` (not imported by
`aniposelib/__init__.py`). numpy/scipy only, except soft-DTW (torch). No offset is ever applied silently: every
estimator returns a result with `identifiable` + `reasons`.

- **Sign convention** (module docstring, `sync.CONVENTION`): `source_time = global_time + offset`; a pair estimate
  `(a, b)` is `offset[b] - offset[a]` (an event at `t` in a appears at `t + offset` in b). Same as disco
  `trim_seconds`; opposite of mouse-outdoor `load_data.align(offsets=)` ("seconds to add"); TC mouse-sync
  `frame_A = frame_B + d` is `offset(a=B, b=A) = d`.
- `align_streams({name: (t, values)}, rate, tol, offsets, reuse=False)` -> `(values [S,T,...], t_grid, src_idx)`:
  nearest sample within `tol`, **no frame reuse by default** (`reuse=True` gives the old `merge_asof` behaviour),
  float seconds or datetimes (datetime64 / pandas / mixed ISO strings; unparseable entries dropped), offsets as
  floats or `TimeMap`s.
- `TimeMap(c, beta, knots)`: `global = c + beta*pts` + piecewise-linear deviation knots (disco `ClipMap`), exact
  inverse, `from_offset`, `to_dict`/`from_dict`.
- `OffsetEstimate` dataclass (offset, sigma, confidence, identifiable, reasons, method, units, pair, diagnostics,
  `to_dict()`).
- `audio_offset(a, b, sr, method='gcc'|'spectrogram'|'onset')` (disco overlap-normalised GCC, legacy multiscale
  spectrogram, TC onset flux), with second-peak / boundary / confidence checks and window-spread sigma;
  `onset_envelope`; `load_audio(path, sr)` via soundfile / PyAV (`[sync]` extra) or an `ffmpeg` executable.
- `motion_signal(points)` + `motion_offset(sig_a, sig_b, rate)` (disco `refine_sync` diagnostics; NaN-aware).
- `solve_offset_graph(edges, anchor)` -> `OffsetGraph` (weighted LS, sigma per stream, cycle RMS, iterative
  rejection of inconsistent edges, unidentifiable edges rejected and listed, unreachable streams -> None).
- `fit_drift(t, offsets)` -> `DriftFit` (robust affine offset + clock rate, `.timemap()`).
- `epipolar_offset(p2d_a, p2d_b, cgroup|F|None)`: fractional lag scan (median Sampson), parabola vertex, block
  bootstrap sigma/CI; cv2-free (`fit_fundamental` robust 8-point when F is not given).
- `frame_offset_sweep(cgroup, sample_fn, offsets)`: integer per-camera sweep against 3D from reference cameras.
- `soft_dtw(D, gamma, band)` (banded anti-diagonal soft-DTW in torch, differentiable; returns value + expected
  alignment; accepts a callable cost for band-only memory) and `soft_dtw_align(sig_a, sig_b, rate, gamma, band)` ->
  `WarpEstimate` (per-row lag + dispersion, robust offset/drift summary, identifiability: dispersion, band-edge hits,
  path-vs-band cost ratio).
- `pyproject.toml`: new optional extra `[sync] = soundfile, av`.

### 1.0-stba: `aniposelib.stba` and `aniposelib.stations` (plan 01 F4, F5, F7 STBA side)

New optional modules; nothing existing changes (not imported by `aniposelib/__init__.py`).

- `aniposelib.stba.SpatiotemporalBA(cgroup, obs, ...).fit() -> STBAResult`: port of disco `may05_stba.py` onto
  `geometry.PackedCameras` (brown and KB models, full K with skew, crop offsets; no private camera vector).
  - Observations with their own times: `TimedObservations(cam, point, uv, t, clip=None, weight=None)` (or a dict),
    `TimedObservations.from_dense(p2ds, times)`.
  - Model: per-track uniform cubic B-spline segments (`knot_spacing`, `segment_s`), evaluated at
    `t = obs.t + delta[clip] + rate[cam] (obs.t - t_mid) + rs[cam] (v/H - 1/2) [+ warp_clip(obs.t)]`.
    **Sign:** `true_time = obs.t + delta`; in the `sync` convention the refined offset is `seed_offset - delta`
    (`STBAResult.sync_offsets`).
  - `free_offsets='per_clip'|'per_camera'|None` (reference clip of `ref_camera` fixed, others prior-anchored with
    `offset_prior_sigma`, default 0.1 s), `free_rate`, `rolling_shutter=False|True|[C]`, `extrinsics=True` with
    `fix_cameras=(0,)` and a baseline scale gauge, free intrinsics from `params.ParamSpec` (+ priors),
    `robust='cauchy'|'huber'|None`, motion prior `accel_sigma` / `vel_sigma` (**units of the rig**: 40 is m/s^2).
  - Solver: Cauchy-IRLS LM with exact Schur elimination of the per-trajectory control-point blocks (batched
    Cholesky, float64, GPU-ready), **analytic** Jacobians (projection, distortion, SO(3) left Jacobian; time
    derivative from the spline) instead of disco's `vmap(jacfwd)` (which costs ~4 s of first-call warm-up on CPU).
  - Uncertainty: reduced normal matrix (`s2 S^-1`) and cluster-robust sandwich (clusters = `cluster_s` windows).
  - F7: optional monotone piecewise-linear per-clip warp (`warp_knot_s`, `warp_clips`, prior + second-difference
    smoothness, monotonicity enforced in the step acceptance). Initialised from `warp_init` (targets or a
    `sync.WarpEstimate`) or from `warp_signals` via `sync.soft_dtw_align`; the soft-DTW expected path
    (dispersion-weighted) is kept as a regulariser (`warp_dtw_sigma_scale`, None = init only). Unidentifiable
    alignments warn and are not used.
  - `STBAResult`: delta + normal/sandwich sigmas, rate, rs, warp, refined `group` (a copy; the input group is not
    modified), `trajectories` (`Trajectories.evaluate(t, point)`), history, residuals, covariance, `to_dict()`.
- `aniposelib.stations`:
  - `Stations({cam: [(t0, t1, name|'skip'), ...]}, cameras=None)`: half-open intervals normalised to cover
    `(-inf, inf)`; `lookup(cam, t)` -> station index (-1 = skip), `lookup_name`, `names`, `paint`, `split`,
    `to_toml_dict()` / `from_toml_dict()` for an optional `[stations.<cam>] t0/t1/name` TOML table.
    `CameraGroup(stations=...)` and the BA station lookup are not wired (cameras.py / bundle.py are owned by
    other branches).
  - `detect_moves(cgroup, p2ds|timed obs, times, window_s, min_step_deg) -> MoveReport`: leave-one-camera-out MSAC
    triangulation (`triangulation.triangulate_robust`), per-window robust rotation offsets (batched Huber GN),
    weighted binary segmentation (disco `detect_steps`), frame-level step-time refinement, proposed `Stations`
    (+ optional `skip_s`), per-station rotations and `corrected_extrinsics(cgroup)`. Nothing is applied.
    (Plan 01 placed it in `quality`; it lives in `stations` per the 1.0 ownership table.)
- `tests/synth.py`: new helpers `smooth_tracks`, `timed_observations` (unsynchronised cameras with offsets, clock
  rates and rolling shutter; the disco `synthetic_problem` as a fast fixture).

### 1.0-iomatch: `aniposelib.io`, `aniposelib.affine`, `aniposelib.matching` (plan 01 phase G, F6)

All additive (new modules); no existing behaviour changes.

#### `aniposelib.io` (new)
- `camera_from_extrinsics_mat(K, E4x4, dist, size, name, *, fisheye=, offset=, extra=)`: the one place that turns a
  world->camera matrix (`x_cam = R X + t`) into a `Camera`/`FisheyeCamera` (nearly-orthonormal R is projected onto
  SO(3); an improper R raises instead of silently producing a mirrored camera).
- `reindex_points(points, from_names, to_names, missing='nan'|'raise')`: reorder the camera axis between camera
  orders (e.g. tracktail order vs TOML order); absent cameras become NaN.
- `from_matlab(params, names=None, *, one_based=True)` / `to_matlab(cgroup, path=None)`: MATLAB
  `cameraParameters`/`stereoParameters` (`toStruct` dicts, lists, `{name: dict}`, `.mat` via scipy, `.json`;
  the Emory export keys are accepted). Transposes `IntrinsicMatrix` and row-vector rotations, packs radial/tangential
  into OpenCV order, `ImageSize` is [h w], and **shifts the 1-based principal point by -1**.
  *Old behaviour of the hand-written converters (Emory `convert_calibration.py`, tailcyclenet `johnson_mouse.py`), which
  did not shift it:* `one_based=False`.
- `read_opencv_yaml` / `write_opencv_yaml` (cv2-free FileStorage YAML parser; cv2 misreads bare integers under
  `dt: d`), `from_opencv_yaml(paths, *, convention='auto'|'opencv'|'matlab', one_based=None)` /
  `to_opencv_yaml(cam_or_group, path)` (one file per camera; written files are readable by `cv2.FileStorage`).
  A transposed K is detected as the MATLAB convention (K and R transposed, 1-based principal point by default;
  `one_based=False` for the old johnson_mouse numbers).
- `from_projection_matrix(P)` / `from_dlt(L11)` (RQ decomposition, **skew kept**: the projection honours it now;
  tailcyclenet `rodent3d.py` dropped it), `to_projection_matrix`, `to_dlt`, `decompose_projection`. An affine `P`
  (`P[2,:3] = 0`) / DLT with `L9..L11 = 0` returns an `AffineCamera`.
- `from_metadata_yaml(path_or_dict, *, order='tracktail', return_moving=False)` / `to_metadata_yaml(cgroup, path,
  *, moving=None)`: tracktail `metadata.yaml` (`intrinsic_matrices`, `extrinsic_matrices`, `distortion_matrices`,
  `camera_heights/widths`, `offset_dict` -> `Camera.offset`, `cam_type`, `moving_cams` with `(T,4,4)`). Camera order
  defaults to tracktail's (numeric if all names are digits, else lexical) so indices match tracktail arrays
  (`order='natural'`/`'file'` otherwise). PyYAML is optional: files written here are read without it.
- `to_json_rig(cgroup, path, *, session=, extra=)` / `from_json_rig(path_or_dict, *, stations='first'|'all')`:
  JSON rig in the disco `pose_ba_core` layout (`intrinsics` + `stations` with `R`, `t`), plus full `K`/`dist`.

#### `aniposelib.affine` (new)
- `AffineCamera(A, b)` (telecentric `uv = A X + b`, `nn.Module`): `project`, `reprojection_error`,
  `get_projection_matrix`, `get_direction`, dict round trip (`model = 'affine'`), and `to_pinhole(support,
  max_error_px=0.5)` (the tailcyclenet `johnson_fly` telephoto-pinhole construction, checked on support points).
- `triangulate_affine(cameras, points)` (exact linear LS), `affine_from_points(X, uv)`.
- Not (yet) usable inside `CameraGroup` (projection/BA there assume pinhole/fisheye).

#### `aniposelib.matching` (new, F6)
- `associate_tracklets(cgroup, detections, track_ids, min_cams=3, max_pair_error=40, min_overlap=8,
  ambiguity_ratio=3.0, dedupe_ratio=0.2, *, min_len=10, sample_frames=40, backbone_frames=120, verify=True,
  verify_mult=4.0, verify_floor=100.0, min_score=None, min_points=4) -> Groups`: per-camera 2D tracklets (tracker ids
  with no cross-camera meaning) -> cross-camera subject groups. Merges duplicate ids, scores tracklet pairs by two-view
  error (one batched triangulation per camera pair), union-find clustering (backbone tier first, same-camera-same-frame
  exclusion), relative per-camera id-switch guard (`verify`), leave-one-camera-out ambiguity abstention. Ported from
  EvoVision `identify_main_3d` / `calib_reuse.tracklet_points`. Differences from the original: `verify` drops the worst
  camera per frame and re-triangulates (the one-shot version also dropped cameras the switched one had pulled off);
  the tracklet entry gate is `min_points` finite keypoints instead of a score sum on body-17.
- `Groups.to_points(n_frames=None, score_thr=None, min_cams=None, frames='sample'|'all', min_points=4)` ->
  `TrackletPoints(points [C,F,K,2], frame_id, group_id)` (frame budget shared across groups) for BA/QA;
  `Groups.dense()`, `Group.observations(cam)` (maps back to detection indices), `to_dict()`.
  Documented loop: associate -> bundle adjust -> re-associate.
- `associate_points(cgroup, points_per_cam, max_res_px=30.0, min_views=2, *, max_instances=0, scores=None) ->
  Association` (+ `associate_points_frames`): per-frame multi-animal association of single-point detections (all
  cross-camera pairs triangulated in one batch, ranked by support count then residual, greedy growth, refit gate, each
  detection used once). Ported from tailcyclenet `infer/associate.py`.
- tests: `synth.crowd()` helper (several separated skeletons).

### 1.0-integrate: final integration (aliases, stations in the group, lazy modules, docs)

- **Version 1.0.0** (`aniposelib.__version__`, `pyproject.toml`, `pixi.toml`). Top level also exports `Camera`,
  `FisheyeCamera`, `CameraGroup`, `CharucoBoard`, `Checkerboard`.
- **Lazy sub-modules.** `import aniposelib; aniposelib.quality` (also `scale`, `sync`, `stba`, `stations`, `io`,
  `affine`, `matching`, `init`, `intrinsics`, `detectors`, `bundle`, `geometry`, `triangulation`) works without an
  explicit import, via a module `__getattr__`. `import aniposelib` imports none of the optional dependencies (vggt,
  huggingface_hub, onnxruntime, soundfile, av, yaml); a smoke test checks this in a fresh interpreter.
  `pandas` (only `utils.load_pose2d_fnames` uses it) and `scipy.signal` (only `cameras.medfilt_data`) are now imported
  on use: `import aniposelib` takes ~6 s instead of ~9.5 s on the cluster file system (most of it is torch).
  *Old behaviour:* `aniposelib.utils.pd` / `aniposelib.cameras.signal` no longer exist as module attributes.
- **`CameraGroup` aliases** (thin delegates, documented in each docstring): `rescale`, `apply_similarity`,
  `anchor_to`, `align_to`, `align_world` (`aniposelib.scale`; in place), `report`, `loo_reprojection`
  (`aniposelib.quality`), `CameraGroup.from_extrinsics(names, Ms, K=, sizes=, model=, dist=, template=)`
  (`init.from_extrinsics`). Already there from other branches: `triangulate_robust`, `bundle_adjust_lm/robust`,
  `resect_cameras`, `place_cameras`, `add_camera`.
- **`Camera` aliases:** `Camera.from_extrinsics_mat(K, E, dist, size, name, ...)` classmethod
  (`io.camera_from_extrinsics_mat`; `FisheyeCamera.from_extrinsics_mat` builds a fisheye camera) and
  `cam.lens_validity(**kw)` (`quality.lens_validity`). Note: on `tests/data/rigs/fisheye_5cam` camera `..._32`
  is reported invalid by default (137 of 627 grid pixels, 22 %, cannot be undistorted: the image corners are beyond
  its KB polynomial's range, min radial slope 0.13); `lens_validity(max_nonfinite_frac=0.25)` accepts it.
- `subset_cameras_names(names, missing='raise'|'drop')` (default unchanged) and `reorder(names)` (a permutation;
  returns a new group).
- `cameras.init_camera_intrinsics(..., guard=False)`: `guard=True` (or a dict of keyword arguments) runs the guarded
  `init.board.init_camera_intrinsics` (plausibility check + fallback). Default unchanged.
- `utils.get_initial_extrinsics(rtvecs, cam_names=None, disconnected='raise'|'components')`: `'components'`
  delegates to `init.board.initial_extrinsics` (one gauge per connected component, warning). Default unchanged
  (`ValueError` on disconnected board graphs).
- **`CharucoBoard(..., units=None)`** / `Checkerboard(..., units=None)`: unit of the square length (`'mm'`, `'m'`);
  `scale.from_board` writes it to `metadata['units']`. Pickles of older boards load with `units=None`.
- **`CameraGroup(cameras, metadata=None, dtype=None, stations=None)`**: optional
  `aniposelib.stations.Stations` (or its TOML dict). `copy`, `subset_cameras*`, `reorder` keep it. `dump`/`dumps`/
  `to_dict` write it as **`[metadata.stations.<cam>]`** (`t0`, `t1`, `name` arrays) and `load`/`loads`/`from_dict`
  read it back into `cgroup.stations` (a top-level `[stations]` table is also accepted on read). It lives under
  `metadata` (not a top-level `[stations]` table as the stations module docstring suggested) because aniposelib
  <= 0.9 treats every top-level table except `metadata` as a camera; 0.9 readers see it as an extra metadata key.
  Files without stations are unchanged.
  `cgroup.bundle_adjust_lm(obs)` uses `cgroup.stations` when `obs` has times (`Observations.time`) and no
  `stations=` is passed. The other BA entry points (`bundle_adjust_robust`, `bundle_adjust_iter`, dense points)
  do not look stations up (dense points carry no times); pass `stations=` to `bundle.bundle_adjust_lm` explicitly.
- `quality.detect_moves` / `quality.MoveReport` re-export `stations.detect_moves`.
- **`init.calibrate_from_points(scale=<reference CameraGroup>)` now also aligns the gauge** to the reference
  (`scale.from_reference` for the estimate and its flags, then `scale.align_to`: rotation, origin and scale of the
  reference; outlier cameras are trimmed from the fit). `report.scale.details['align']` has the similarity.
  *Old behaviour (only rescale):* `scale=dict(reference=ref, align=False)`.
- **`init.vggt.predict` / `predict_pairwise` default `mode='pad'`** (was `'crop'`): letterboxing keeps every pixel of
  portrait / mixed-orientation frames; intrinsics are restored to original pixels in both modes. *Old behaviour:*
  `mode='crop'` (VGGT's own preprocessing). `preprocess_transform`/`preprocess_image` keep `mode='crop'` as their
  default. VGGT itself has never been run in the test environment (no `vggt` installed); the plumbing is tested with
  a stand-in network.
- New optional extras `[io]` (`pyyaml`, only for hand-written / block-style tracktail `metadata.yaml`; files written by
  `aniposelib.io` are read without it) and `[all]`.
- Tests: `tests/test_integrate.py` (lazy import smoke test in a subprocess started at collection, aliases, stations
  round trip and BA hook, board units, disconnected board graph); `test_init.py::test_calibrate_from_points_recipe`
  checks the aligned gauge.

## 0.9.1

- Fix: the dependency is back to `opencv-contrib-python>=4.7,<5`. 0.9.0 depended on `opencv-python`, whose aruco module
  lacks `detectMarkers`, `interpolateCornersCharuco` and `estimatePoseCharucoBoard`, so ChArUco board detection and
  pose estimation (`CharucoBoard`, `calibrate_rows`/`calibrate_videos` with ChArUco) failed with `AttributeError`.
  0.9.0 is yanked.

## 0.9.0

First release under the `aniposelib-pytorch` name, from the `pytorch` branch of
`lambdaloop/anipose-lib`. It still imports as `aniposelib`.

- The code is identical to `pytorch@05b769a`, which downstream projects previously pinned via git.
- `torch>=2.1` is now a declared dependency.
- `opencv-contrib-python` was replaced by `opencv-python>=4.7,<5`. (This broke ChArUco and was reverted in 0.9.1.)
- New dependency `aniposelib==0.0.1`, a file-less placeholder. It makes classic aniposelib (numpy, 0.8.x)
  impossible to install in the same environment.
- `import aniposelib` raises `ImportError` if a classic `aniposelib` distribution is also installed.
- New `aniposelib.DISTRIBUTION = 'aniposelib-pytorch'` and `aniposelib.BACKEND = 'torch'`.
- `requires-python >= 3.9`.

Differences from classic aniposelib 0.8.0, inherited from the pytorch branch:
- `Camera`, `FisheyeCamera` and `CameraGroup` are `torch.nn.Module`s with float64 parameters, and support `.to(device)`.
- Methods return torch tensors. This changes in 1.0: numpy in → numpy out, with `return_grad=True` for autograd.
- `bundle_adjust_iter` / `bundle_adjust` are torch (Adam) optimizers.
- `n_dist` / `set_n_dist` control which distortion terms are optimised, and fisheye is differentiable.
- `triangulate_possible` is GPU-batched.
