# Workflows (aniposelib 1.1)

aniposelib has a large backend (camera models, board detection, triangulation, bundle adjustment, QA,
synchronisation, ...). Most uses come down to seven workflows, and each has **one entry point**, re-exported at
the top level (`aniposelib.calibrate_from_board`, ...). They compose the backend; they do not add maths.

| | workflow | function | result |
|---|---|---|---|
| W1 | calibration board or known-3D object in the videos | `calibrate_from_board` | `BoardCalibration` |
| W2 | calibrate from the animal / people (no board) | `calibrate_from_points` | `PointCalibration` |
| W3 | metric units and the world frame | `estimate_scale` | `ScaleEstimate` |
| W4 | is this calibration (still) good? | `evaluate_calibration` | `CalibrationEvaluation` |
| W5 | unsynchronised cameras | `estimate_offsets` | `OffsetEstimation` |
| W6 | 2D poses to 3D | `reconstruct_3d` | `Reconstruction` |
| W7 | several animals / people | `associate_views` | `Association` |

## Shared conventions

- **Results.** Every result has `.report` (dict), `.verdict` (`'ok'` / `'warn'` / `'fail'`), `.reasons` (one line per
  rule that is not ok), `.settings` (every derived threshold, so a run is reproducible), `.provenance` (version,
  inputs, fallbacks that fired, timings), `.to_dict()`, `.to_json(path)` and `.summary()`. Calibration results
  unpack: `cgroup, res = calibrate_from_board(...)`.
- **No tuning knobs in pixels or world units.** Pixel thresholds come from the robust *noise floor* (median inlier
  reprojection error, at least 0.5 px, at most 1 % of the image diagonal); first-pass gates are fractions of the
  image diagonal; 3D lengths are fractions of the rig scale; time is in seconds and compared with the frame period.
  The same default call works on a rig in mm, m or km.
- **Failures are results.** A workflow raises only on unusable inputs (wrong shapes, unknown cameras, unreadable
  files). A bad result is `verdict='fail'` with reasons.
- **Overrides.** `options={...}` overrides internals (each module has `W*_DEFAULTS`; unknown keys raise).
  Verdict thresholds: `options={'verdict': {'rule_name': (warn_at, fail_at)}}`, or `{'rule_name': None}` to disable a
  rule (for example a project's own pixel gates).
- **Saving.** `out='calibration.toml'` writes the calibration and `calibration.report.json`; the data workflows write
  `<stem>.npz` + `<stem>.report.json`. Nothing is written without `out=`.
- **Types.** numpy in, numpy out. Points are `C x N x 2` or `C x N x J x 2` in camera order, NaN = missing.
  `rng=0` by default.

## W1 `calibrate_from_board`

**Use when** you recorded a ChArUco board, a checkerboard, or any object with known 3D points.

```python
board = al.CharucoBoard(7, 5, square_length=25, marker_length=18.75, units='mm')
cgroup, res = al.calibrate_from_board([['A.mp4'], ['B.mp4'], ['C.mp4']], board,
                                      cam_names=['A', 'B', 'C'], out='calibration.toml')
print(res.summary())          # res.lens_models, res.detections
```

Inputs: per-camera video lists, per-camera detection rows, or a dense `C x F x K x 2` array. `fisheye=` (bool or
per camera). `intrinsics='auto'` picks a lens model per camera on held-out views (pinhole ... rational, or
fisheye); `'fixed'` keeps `init`'s intrinsics; a `CameraGroup` starts from its intrinsics under priors.
`detector='auto'` uses DeepCharuco when the `[deepcharuco]` extra is installed (falling back to OpenCV, recorded in
`provenance`), else OpenCV.

Steps: detection -> intrinsics -> board poses -> board-graph initial extrinsics -> two-pass LM board BA -> report.

**Disconnected board graph** (some cameras never saw the board at the same time as the others): the result is
`verdict='fail'` with `res.components` and one calibration per component (`res.component_groups`, each in its own
frame). To bridge them, pass `bridge_points=` (`C x N x 2` or `C x N x J x 2`) 2D tracks of anything seen across
components (the animal, a person, a wand): each extra component gets one 6-DoF transform with the board's
intrinsics and metric scale fixed, then a joint BA. Recording more board frames in the overlap also works.

Verdict rules: `board_reproj_px` (worst per-camera median; warn above max(1 px, 3 x noise floor), fail above twice
that), `uncovered_cameras`, `missing_intrinsics`, `disconnected_components` (fail), `board_shape_rel` (shape
residual / board size; warn 0.5 %, fail 2 %), `lens_invalid` (warn).

### W1b `PointObject`: generic calibration objects

Any object with known 3D points (a non-planar wand or fixture, a 3D-printed target, surveyed landmarks). There is
no detector: pass annotated 2D points.

```python
obj = al.PointObject(points3d, names=['tip', 'mid', ...], static=False, units='mm')   # K x 3
ann = obj.annotations_from_dicts({'A': {0: {'tip': (x, y), ...}, ...}, ...})          # -> C x F x K x 2
cgroup, res = al.calibrate_from_board(ann, obj, cam_names=['A', 'B', 'C'])
```

- `static=False`: the object moves (one pose per frame), like a board.
- `static=True`: fixed landmarks. Each camera is placed by PnP against them and the calibration is in the object's
  world frame and units, so no W3 step is needed.
- Intrinsics: planar objects as boards; non-planar objects by per-view DLT when a view has >= 6 annotated points;
  otherwise pass intrinsics (`init=` / `intrinsics=`), or the result fails with that reason.

## W2 `calibrate_from_points`

**Use when** there is no board, or the board only gave intrinsics: keypoints, people, a blob trajectory.

```python
cgroup, res = al.calibrate_from_points(p2d, intrinsics=lab_cgroup,       # or K, per-camera dicts, nominal
                                       init='auto', frames=None,          # frames= opts in to VGGT
                                       scale=dict(board=board, board_points=rows),
                                       out='calibration.toml')
print(res.selected_init, res.summary())
```

Steps: candidate initial rigs (the given rig, SfM from the correspondences, VGGT only when `frames=` is given) ranked
on the data -> robust BA (MSAC inliers, >= 3 views per point, re-placing stuck cameras, a threshold schedule scaled
by the noise floor; `groups=` enables structured frame / keypoint rejection) -> `refine_intrinsics='auto'` frees
focal + k1 under priors and keeps them only if the held-out leave-one-out error improves -> optional scale (W3) ->
report. `stations=` handles cameras that moved mid-session.

Verdict rules: `loo_worst_px` (worst per-camera leave-one-out median; warn 3 x, fail 10 x noise floor),
`unplaced_cameras`, `runaway_cameras` (fail), `min_inlier_frames`, `min_inlier_frac` (warn / fail),
`scale_rejected` (fail).

## W3 `estimate_scale`

**Use when** the calibration is in arbitrary units (W2, VGGT) or should be aligned to a reference / the ground.

```python
res = al.estimate_scale(cgroup, board=board, board_points=rows)                    # board seen in the trial
res = al.estimate_scale(cgroup, lengths=dict(points=p2d, pairs=[(0, 3)], ref=[25.0]))   # known lengths
res = al.estimate_scale(cgroup, reference=lab_cgroup, align='reference')           # an older calibration
scaled = res.cgroup if res.accepted else None
```

Exactly one source. The estimate is the median of per-frame ratios with relative gates; a doubtful scale is refused
(`accepted=False`, `verdict='fail'`) rather than applied. With `apply=True` (default) `res.cgroup` is a rescaled copy
with `units`, `scale_source` and `scale_quality` in its metadata; the input is never modified. `align='ground'`
puts the floor at z = 0 (`options={'ground_points': ...}`).

## W4 `evaluate_calibration`

**Use when** checking a new calibration, deciding whether an old one can be reused, looking for moved cameras, or
comparing candidates.

```python
res = al.evaluate_calibration(cgroup, p2d, times=t, reference=old_cgroup)
res.per_camera     # leave-one-camera-out median / p90, coverage, inlier rate per camera
res.moves          # cameras that moved, with times
```

Always: leave-one-camera-out reprojection (a failed triangulation counts against the camera), coverage, inlier
rates, lens validity, grazing camera pairs. With `times`: moved-camera detection. With `reference`: an aligned
comparison. With `board` + `board_points` (or `object` + `object_points`): the board shape residual. Project policy
(for example a reuse ladder) goes on top of `res.per_camera`.

Verdict rules: `loo_worst_px` (warn 3 x, fail 10 x noise floor), `loo_coverage_min`, `lens_invalid`, `grazing_frac`,
`moved_cameras` (fail), `center_residual_frac` (vs the reference; warn 1 %, fail 5 % of rig scale),
`board_shape_rel`.

## W5 `estimate_offsets`

**Use when** the cameras were not hardware-synchronised (GoPros, phones).

```python
res = al.estimate_offsets({'gopro1': 'g1.mp4', 'phone2': 'p2.mov'}, cgroup=cgroup, drift=True)
res.offsets, res.sigmas, res.timemaps      # source_time = global_time + offset
```

Streams are video / audio paths or dicts (`audio=(signal, sr)`, `motion=(signal, rate)`, `points=(p2d, t or fps)`).
Pairwise estimates from every available signal (audio cross-correlation cross-checked by spectrogram and onset
methods, motion energy, epipolar consistency of 2D tracks when `cgroup` is given) are combined in an offset graph
with inconsistent pairs rejected, then a per-stream clock rate is fitted (`drift=True`). `refine=True` runs the
spatiotemporal BA (sub-frame offsets, clock rate, rolling shutter) with priors anchored to the graph. Nothing is
applied: use `res.timemaps`.

Verdict rules: `unidentifiable_streams` (fail), `max_sigma_frames` and `cycle_rms_frames` (warn 0.25, fail 1 frame),
`method_disagreements`, `drift_unidentifiable` (warn).

## W6 `reconstruct_3d`

**Use when** turning 2D keypoints into 3D tracks.

```python
res = al.reconstruct_3d(cgroup, p2d, scores=scores, bones=[(0, 1), (1, 2)])
res.points3d       # N x J x 3, NaN where there are not enough views (never extrapolated)
```

Robust MSAC triangulation with a threshold from the noise floor. **Scores are soft weights** (clipped to [0, 1]):
low-confidence views count less but are not dropped; geometrically inconsistent views are rejected whatever their
score. A hard cut only with `options={'min_score': s}`. With `bones=` or `smooth=`, the points are refined with
bone-length consistency and / or smoothing (unit-free; long sequences in overlapping chunks). No smoothing by
default. `fill='interior'` fills gaps between a joint's first and last observation.

Verdict rules: `reconstructed_frac`, `inlier_rate`, `cameras_unused`, `bone_cv` (with `bones`).

## W7 `associate_views`

**Use when** several animals or people are in view.

```python
res = al.associate_views(cgroup, detections, track_ids=track_ids)   # track_ids optional
res.points         # C x F x P x J x 2 per identity (NaN where unmatched) -> W2 / W6
res.labels         # per camera, per frame: identity of each detection (-1 = unassigned)
```

With `track_ids`, per-camera tracklets are associated across cameras; without, detections are associated frame by
frame and linked over time. Gates are relative to the image diagonal and then the noise floor.

Verdict rules: `no_identities` (fail), `assigned_frac`, `fragmentation`.

## How the reference projects chain them

| project | pipeline |
|---|---|
| anipose / lab board rigs | W1 -> W6 |
| EvoVision | W4 on the last rig; if not ok: W7 -> W2 (VGGT frames, fixed intrinsics) -> W3 (board in trial) -> W6 |
| outdoor mouse | W2 (VGGT frames, trajectory) -> W4 with `times` -> W6 |
| motor observatory | W2 (keypoints, SfM) -> W3 (board videos) -> W4 -> W6 |
| Emory | W2 (lab rig as intrinsics and init) -> W3 (reference + bone lengths) -> W4 -> W6 |
| dancing | W5 -> W1 per board session (`bridge_points` from people) -> W7 -> W2 (stations) -> W5 `refine=True` -> W6 |
| tailcyclenet / tracktail | W6 for labels, W7 at inference; geometry via `CameraGroup.as_tensors` |
