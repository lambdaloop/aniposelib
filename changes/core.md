# 1.0-core (Phase B: correctness and conventions)

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
