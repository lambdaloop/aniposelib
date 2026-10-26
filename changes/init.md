# 1.0-init: `aniposelib.init` (plan 01 phase E: E1-E5)

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
