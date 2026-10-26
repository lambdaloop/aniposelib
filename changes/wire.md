# 1.0-wire: `CameraGroup` on the packed kernels (plan 01 C1–C3 wiring, D8 video sampling)

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
