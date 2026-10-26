# quality (1.0-quality): calibration QA and global scale/gauge (plan 01 C4, C5)

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
