# Changelog (aniposelib-pytorch)

## 1.1.0 (unreleased)

Plans 04 (scope trim, one LM core) and 05 (one entry point per workflow). Upgrade notes: everything under
**Removed** is gone without a deprecation cycle (verbatim copies for downstream projects are kept outside the
package); everything under **Changed** says how to get the 1.0 behaviour.

### Workflows (new)

Seven entry points, re-exported at the top level, documented in [docs/workflows.md](docs/workflows.md). Each
returns a result with a noise-relative verdict (`'ok' | 'warn' | 'fail'`), reasons, every derived threshold in
`.settings`, `.to_json()` and `.summary()`; `out=` saves the calibration TOML / arrays plus a report JSON.

- **Calibration workflows (1.1, plan 05 W1-W4):** `aniposelib.calibrate_from_board` (boards and the new
  `boards.PointObject`: known-3D-points objects, moving wands with per-view DLT intrinsics or static landmarks that
  define the world frame), `calibrate_from_points`, `estimate_scale`, `evaluate_calibration`. Results carry a
  noise-relative verdict (`'ok' | 'warn' | 'fail'`) with reasons and every derived threshold in `.settings`;
  calibrations unpack as `cgroup, result = ...`. A disconnected board graph is a `'fail'` with per-component groups;
  `bridge_points=` solves one 6-DoF pose per extra component.
- **Workflows W5-W7** (`aniposelib.workflows`, plan 05): `estimate_offsets` (audio gcc with spectrogram/onset cross-checks, motion, epipolar; offset graph, drift, optional STBA refinement; returns time maps, applies nothing), `reconstruct_3d` (noise-derived MSAC tau, soft score weights, optional bones/smoothing through `optim_points` with chunking, `fill='interior'` never extrapolates) and `associate_views` (tracklets or per-frame association, dense `C x F x P x J x 2` output; gates from the noise floor and image diagonal), with `OffsetEstimation`, `Reconstruction` and `Association` results.
- `aniposelib.workflows._common`: `Rule` / `evaluate_rules` (declarative verdicts, overridable with
  `options={'verdict': {rule: (warn_at, fail_at) | None}}`), `noise_floor`, `WorkflowResult`.

### Performance (1.1, perf pass)

Same results (bitwise, or within float rounding where noted), less time and memory:

- **Thread guard:** every workflow caps torch's intra-op threads for the call at `ANIPOSELIB_NUM_THREADS`, else
  `OMP_NUM_THREADS`, else 8 (never above the CPUs the process may use; an explicit lower
  `torch.set_num_threads` is kept, the previous count is restored afterwards). Torch's default (one thread per core)
  made the workflows 10-40x slower on a shared machine. Old behaviour: `ANIPOSELIB_NUM_THREADS=<cores>`.
- **`device=`** on `evaluate_calibration` (keyword) and `estimate_offsets` (STBA refinement), default `'auto'`
  (cuda when available; a copy of the group is moved, results stay numpy). `calibrate_from_points` now runs its
  quality report on `device` too (it ran on the CPU). Old behaviour: `device='cpu'`. On a GPU W4/W5 results agree
  with the CPU to float rounding.
- Undistortion runs on the finite observations only, per lens model, in bounded batches; distortion terms with a
  zero coefficient are skipped (exact). `triangulate_robust` undistorts once (fold mask, fit and ray check share
  it), the LOO evaluation undistorts every observation once for all held-out cameras and triangulates in batches of
  1M columns (no more OOM at 12 x 2M points); MSAC hypothesis errors are computed only where a point is observed;
  on a GPU MSAC chunks are sized by the actual hypotheses (CPU chunks unchanged).
- `init.sfm`: pairwise essential matrices on a thread pool (same per-pair seeds); vectorised frame grouping.
- **`init.sfm.from_correspondences` 2-4x faster** (the top cost of `calibrate_from_points` with SfM: motor
  observatory 43 -> 13 s, outdoor mouse 44 -> 22 s, Emory 36 -> 10 s on an L4 node; W2 results equivalent, see
  below). RANSAC results change slightly for a given `rng` (same seed -> same result, whatever the thread count):
  - the frame PnP RANSAC (`frame_pnp_ransac`, new `n_jobs=`) runs its hypotheses in independent streams of 50
    iterations whose seeds are drawn from `rng`, on a thread pool; the per-camera scan runs all remaining cameras
    at once. Frames are sampled with O(k) draws instead of a permutation of all frames. No way back to the 1.0
    random stream (statistically the same search).
  - **`pair_method='usac'`** (new default; `pairwise_poses(method=)`): OpenCV `USAC_DEFAULT` essential matrices,
    5-10x faster on real keypoints and more inliers per pair. This can change the seed pair (Emory: a pair with
    8035 vs 5906 inliers). Old behaviour: `pair_method='ransac'` (`calibrate_from_points(...,
    options={'sfm_kwargs': {'pair_method': 'ransac'}})`).
  - **`device=`** (None: the group's device): the robust re-triangulations run there. `calibrate_from_points` passes
    its `device` (they ran on the CPU). Old behaviour: `options={'sfm_kwargs': {'device': 'cpu'}}`.
  - the re-triangulations fit only points seen by >= 2 cameras; single-view observations no longer get a
    (meaningless, often hundreds of px) error that entered the per-frame decoy medians and the reported medians.
- `optim_points` packs the fixed cameras once per call (the cameras no longer accumulate `.grad` during it), has no
  host syncs in its loop and evaluates all bone terms in one op (float rounding). `reconstruct_3d(device='auto')`
  runs `optim_points` chunks smaller than 50k frame x joint cells on the CPU (a GPU was 3-7x slower there); pass
  `device='cuda'` for the old behaviour.
- `bundle_adjust_lm` and the lens ladder compute rotations (and their Jacobians) once per station / board view.
- `matching.associate_points_frames` batches frames (new `batch_frames=256`); `associate_views` (per-frame mode)
  uses it and only associates frames that have detections (frame ids around 1e5 no longer cost minutes).
- `sync.audio_offset(method='spectrogram')`: batched per-frequency correlations; `_lm` Schur blocks via a flat
  `index_add_`.

### Calibration workflows: real-data fixes (1.1)

From the real-data checks of the reference projects (Emory, motor observatory, EvoVision, outdoor mouse, tailcycle,
dancing). Behaviour changes say how to get the previous behaviour.

- **`options['noise_unit']`** (`'auto'` | float) in `calibrate_from_points`, `estimate_scale`, `evaluate_calibration`
  and `calibrate_from_board` (bridge): a float forces the unit that multiplies the backend's pixel defaults (1 =
  absolute pixels). Default `'auto'` = the capped unit (base commit).
- W2 fine threshold schedule: thresholds above the last coarse threshold are dropped instead of clipped to it, so the
  rounds keep their ratios (`min(t*u, coarse)` flattened every round to one threshold).
- W2 verdict noise floor (`settings['noise_floor_px']`, the `loo_worst_px` limits and the quality thresholds): measured on
  the points under the refined rig at 2 % of the diagonal (as W4), not at the last schedule threshold (that one is
  ~threshold / 3 by construction); the old value is `settings['noise_floor_fit_px']`.
- W2 `refine_intrinsics='auto'`: a refined lens that is not invertible over the image (`quality.lens_validity`) is put
  back to the given one (that camera, then an extrinsics re-fit); the refinement is kept only if the held-out LOO
  median gains `intrinsics_gain` **and** the held-out LOO agreement and coverage do not drop. The gate never raises
  (a failing fit keeps the intrinsics). Old behaviour: none (Emory: 6/11 invalid lenses were accepted).
- W2 `options['intrinsics_share']` (None | `'all'` | groups of camera names / indices): identical cameras share the lens
  when intrinsics are refined. `intrinsics_sigma_logf` / `intrinsics_sigma_k` (and W1 `prior_sigma_*`) may be
  `{camera name: sigma, 'default': sigma}`; `params.IntrinsicsPrior` takes per-camera sigma dicts and matches cameras
  by position only when no name matches (before: by position for any unmatched camera).
- W2 `init='auto'` skips SfM when the given extrinsics are complete and explain >= 70 % of the multi-view observations
  (>= 50 % per camera) at the coarse threshold (`options['auto_skip_sfm']=False` for the old behaviour); new rule
  `init_support` (share of observations the chosen init explains; warn < 0.3, fail < 0.05).
- W2 `runaway_cameras`: a camera far from the rig centre is a runaway only if it also moved more than
  `runaway_move` (0.5) x the rig scale from its init (telephoto rigs). Old: `options={'runaway_move': -1}`.
- W2 5D points (`C x F x P x J x 2`) with `init='vggt'` (needs `frames=`) or a `CameraGroup` init: the people are
  associated with `associate_views` on that rig (`options['track_ids']`), then calibrated as 4D input; no identity ->
  `'fail'` with the VGGT template spread in the reason. `init='people'` forwards `track_ids` to the second
  association; a failing refinement returns `'fail'` instead of raising; the report always has `per_camera`.
- W2 `stations=`: two-pass station fit (loose selection, then 10 noise units), the result group carries
  `cgroup.stations` and `metadata['station_poses']`; `PointCalibration.rig_at(t)` / `workflows.calibration.rig_at`.
  W2 `options['reject']=None` turns off the structured frame / keypoint rejection.
- 2-camera rigs: W2 and W4 run (no LOO: rules `two_view_only` (warn) and `two_view_px`), W3 lengths use
  `min_inliers=min(3, C)`. LOO cameras without any supported point are reported as None with an `loo_unsupported` rule
  (never 1.8e308); W4 names cameras with a low inlier rate as the likely cause of a low LOO coverage.
- `init.sfm.from_correspondences`: when the best seed pair triangulates fewer than `min_pair_points` points, the next
  pairs are tried (`stats['seed_tried']`); this made `test_w2_bad_inputs` flaky across machines.
- **W2 `init='people'` (EvoVision, Panoptic 5 cameras: 2 of 5 were unplaced, 102 s).** `init.people.associate_people`:
  a person's track fragments (an IoU tracker cuts tracks at occlusions) are joined: fragments of one camera that are
  never visible in the same frame do not compete in the pair matching, and the matched track pairs are merged
  strongest first, refusing only merges that put two tracks of one camera in the same frame (before: any identity with
  two tracks of one camera was dropped whole, which left 3 small identities). `report.groups` values are a track id or
  a list of them. The camera pairs run on threads with per-pair seeds; W2 associates once (it ran twice) and passes its
  `rng` / `device` (the association was unseeded). `from_people(association=)` reuses an association. Panoptic per-camera
  K: all 5 cameras, centres 11.9 mm median / 15.6 mm max (seed route 17.9 mm, EvoVision 13.2 mm), 27 s.
- `init.sfm.from_correspondences(noise_mult=2.5)`: the frame-RANSAC threshold is `max(pnp_thresh, noise_mult x noise)`,
  with the noise the median symmetric epipolar distance of the seed pair (`stats['noise_px']`,
  `stats['pnp_thresh_px']`). At ~2 px noise nothing changes (8 px); at 4-7 px the fixed 8 px rejected correctly posed
  cameras or made them passengers. Old: `sfm_kwargs={'noise_mult': None}` (or `people_kwargs` for `init='people'`).
- `init.sfm.from_correspondences`: a camera rejected for too few agreeing frames is retried once against the final
  cloud (after a local BA), instead of being rejected for good; `report.rejected` lists it only if the retry fails.
- W3 lengths: MSAC threshold an absolute 5 px (was 10 noise units: +16 % scale on Emory; 5 noise units still gave +2 %;
  `options['noise_unit']` makes it 5 x that unit), and the bones are re-measured on the scaled
  rig at half the threshold (`report['lengths_check']`, rule `lengths_check`, warn above `options['lengths_check']`
  = 3 %). Old: `options={'lengths_kwargs': {'tau': 10 * u}}`. `lengths=` / `board=` together with `reference=` and
  `align='reference'`: scale from the lengths / board, frame from the reference (`details['reference_scale']`).
  `board_points={camera: video}` detects the board in the (synchronised) videos as W1 does (see below).
- W4: `options['cam_names']` (points in another camera order; `report['cameras_missing' / 'cameras_dropped']`),
  `tau_px` / `tau_tri_px` / `thresholds_px` (absolute pixels), time-blocked LOO with `times=` (`loo_window_s='auto'`
  ~ 20 windows, `res.per_window`, rule `bad_windows_frac`; `loo_window_s=None` = off), `compare_to={name: cgroup}` +
  `clusters` (paired LOO agreement with a cluster bootstrap, `report['compare_to']`). `center_residual_frac` is now
  warn-only (old: `options={'verdict': {'center_residual_frac': (0.01, 0.05)}}`). `detect_moves` defaults in W4:
  `z_thr=10` (was 6), and `stations.detect_moves` gets `min_support_frac=0.25` (windows with few points cannot form a
  step), `max_resid_ratio=5` (windows whose offset fit leaves residuals above 5 x the camera's median window are
  dropped: a rotation does not explain them) and `retest=True` (flagged cameras re-tested without the confirmed moved
  ones). Old: `detect_moves_kwargs=dict(z_thr=6, min_support_frac=None, max_resid_ratio=None, retest=False)`. An `AffineCamera` group raises a
  `ValueError` pointing to `AffineCamera.to_pinhole`.
- `scale.fit_similarity(robust_init=True)`: LMedS start (3-point subsets + leave-one-out), so one runaway camera no
  longer drags `align_to` / `from_reference` / `quality.compare` (motobs: 669 mm -> 0 median residual). Results on
  clean data are unchanged; `robust_init=False` for the old fit.
- W1: `options['focal_guess']` (float or `{camera: f}`, extra lens-ladder starts), `intrinsics=<CameraGroup>` may
  cover only some cameras; new rules `runaway_cameras` (> `runaway_ratio`=10 x the median distance, fail),
  `board_reproj_ratio` (worst / median camera, warn 3, fail 8) and, with `bridge_points`, `bridge_inlier_frac`
  (warn < 0.5, fail < 0.2); with a bridge the board rules use the board observations alone
  (`settings['noise_floor_board_px']`).
- **W1 / W3 video detection (motor-observatory calibration videos, 12 x 1440x1080):** frame-major sequential sampling,
  new `boards.detect_videos(board, videos, frame_indices=|skip=, max_boards=, max_frames=, min_cameras=2,
  n_workers=)`: all cameras' videos are read together front to back (`grab`/`retrieve`, no OpenCV seeking), the board
  is detected in every `skip`-th frame (or `n_frames` evenly spaced, or `frame_indices`) of every camera at once
  (batched across cameras for DeepCharuco), and reading stops early once every camera has `max_boards` boards (W1,
  default 200) or `max_frames` frames show the board in >= 2 cameras (W3, default 200); `None` = read everything.
  The densify-after-detection sampling of `detect_video` is no longer used by W1. `n_workers` default `'auto'`
  (up to 8 threads; was 1). New W1 option `n_frames`, W3 `frame_indices`. `settings['detection']` (W1) /
  `report['detection']` (W3) say what was read and why it stopped. A missing video raises `FileNotFoundError`
  (the per-video reader printed a warning and gave no rows); per-camera `frame_indices` still use
  `CameraGroup.get_rows_videos`. Old behaviour: `CameraGroup.get_rows_videos` (unchanged) and pass the rows.
  Same 60 frames: identical detections and scale, 238 -> 75 s (W3, L4 node, 35 s clips); W3 default 589 -> 78 s,
  W1 617 -> 93 s (DeepCharuco), 2214 -> 210 s (OpenCV).
- **DeepCharuco in the workflows: input scale `'auto'`** (`DeepCharucoDetector(scale='auto')`: longest side to
  720 px, never up), unless `detector_kwargs['scale']` is given. At scale 1, 1440x1080 frames gave 112 boards in 840
  frames (OpenCV 312, scale 0.5: 249), missed a camera entirely (W1 `fail`) and was 4x slower. Old:
  `options={'detector_kwargs': {'scale': 1.0}}`. `DeepCharucoDetector` itself keeps `scale=1.0`.
- `DeepCharucoDetector(max_batch_pixels=4e6)`: frames per network call capped by pixels, halved on CUDA OOM
  (batch 16 of 1440x1080 frames ran out of memory on a 22 GB L4).

### Data workflows W5-W7: real-data fixes (1.1)

Tested on the reference projects' data (`dev/scratch/realdata`). Every new default says how to get the earlier
1.1 development behaviour.

- **`reconstruct_3d` (W6)**
  - Default MSAC threshold `tau = clip(3 x noise floor, 10 px, 1 % of the diagonal)` (`tau_noise_mult` 5 -> 3;
    motor observatory median reprojection 3.75 -> 3.31 px vs 3.07 px for the project's IRLS, EvoVision/Panoptic
    MPJPE unchanged to better, other datasets sit at the 10 px floor). Old: `options={'tau_noise_mult': 5}`.
  - `fill='interior'` fills gaps of at most `options['max_gap']=10` frames (it filled 90 % of a sparse outdoor
    trajectory, runs up to 4 minutes). New `times=` (interpolate on time; `options['max_gap_s']`). Old:
    `options={'max_gap': None}`. For label frames that are not consecutive, use `fill='none'` (or pass `times=`).
  - New `segments=` (clip / window id per frame): filling, smoothing (masked at the boundaries, one
    `optim_points` call), bones and chunks never cross a segment.
  - New `extrinsics=` (per-frame cameras `[C,N,4,4]` / `[N,C,4,4]` / a list of groups) for moving rigs, and a
    verdict rule `error_drift` (largest / smallest median error over 8 blocks of frames; warn > 3) that catches
    a static rig used on moving cameras (or broken sync).
  - Bones: `options['bone_weight']='auto'` (default) gives each bone a weight from its measured length spread
    minus the expected triangulation noise (non-rigid "bones" are barely enforced; capped at the old 2.0), and a
    chunk whose refinement raises the median inlier error by more than `bone_max_reproj_rise=0.5` noise floors
    keeps the triangulated points. `report['bones']` CVs are now robust (1.4826 MAD / median; the std/mean CV is
    `cv_std`), and the `bone_cv` rule is relative to the triangulated CV (warn > max(0.1, CV_tri), fail >
    max(0.3, 2 CV_tri)). Old: `options={'bone_weight': 2.0, 'bone_max_reproj_rise': None}`.
  - New outputs: `point_error [N,J]` (median inlier error per point) with `options['max_point_err_px']`,
    `out_of_image [C,N,J]` and `report['n_out_of_image']` / per-camera `n_out_of_image` (out-of-image
    observations no longer count as rejections; `inlier_rate` uses the in-image observations);
    `options['mask_out_of_bounds']` (True), `options['min_joints_per_view']`, `options['cam_names']`,
    `report['two_view_frac']`.
  - `AffineCamera` groups are supported (exact telephoto pinholes from `AffineCamera.to_pinhole`, errors from the
    affine projection); mixed affine / pinhole groups raise a `ValueError` pointing to `to_pinhole` (both
    crashed in `geometry.pack`).
  - `points` may have an identity axis (`C x F x P x J x 2`, `associate_views` output): one reconstruction per
    identity.
  - `CameraGroup.optim_points`: `scale_length` may be one weight per bone; new `segments=`.
- **`estimate_offsets` (W5)**
  - Bug: audio decoded from a path ignored the audio stream's start time (1.1 frames off on the mouse-sync
    GoPros, verdict ok). `audio_t0` now defaults to audio start minus video start from the container (new
    `sync.media_info`); an explicit `audio_t0` wins.
  - Long clips (> `audio_windowed_min_s=300` s): `options['audio_strategy']='auto'` uses a coarse spectrogram lag
    plus gcc in windows and a robust line through them (`sync.audio_offset_windowed`; full-length gcc was
    unidentifiable under tens of ppm of drift). The minimum overlap is also relative
    (`audio_min_overlap_frac=0.25` of the shorter clip; edge lags won on hour-long clips). Old:
    `options={'audio_strategy': 'global', 'audio_min_overlap_frac': 0}`.
  - Audio sigmas are floored by the window scatter, the agreement of the audio methods and one sample (the
    spectrogram fallback reported 1.8e-5 s for ~5 ms of scatter); the fallback runs with windows, so `drift=True`
    can fit it; streams with audio and no windowed estimate are listed by `drift_unidentifiable` (it was silent);
    a fitted rate below `drift_min_z=3` sigmas is no drift (rate 0; 6 noisy windows over 2 minutes gave a spurious
    100 ppm). Old: `options={'drift_min_z': 0}`.
  - The STBA frees the clock rate only with a drift fit (or seed) and points spanning
    `refine_free_rate_min_s=60` s (`options['refine_rate']='auto'`; a 10-30 s window gave spurious rates). Old:
    `options={'refine_rate': 'free'}` (with `drift=True`).
  - New: `seed=` (a previous result or time maps: skip the audio/graph, refine only), `options['epipolar_F']`
    (`'auto'`: calibrated with `cgroup`, else F fitted per offset, so points-only pairs get epipolar estimates;
    `'self'`, `'both'`), `options['stream_camera']` (clips of one camera), `options['fps']` / per-stream `fps`
    (else the video's `avg_frame_rate`), epipolar pairs at different frame rates (resampled; were skipped),
    irregular per-frame times use the mean rate and warn (`irregular_times`), `pairs` report the raw and the
    shifted offset (`offset_raw`, `offset`).
  - Clips of one device share one clock (`options['stream_device']`, default the `stream_camera` mapping;
    `options['shared_clock']=True`): one rate in the drift fit (new `sync.fit_drift_shared`) and in the STBA (new
    `SpatiotemporalBA(rate_groups=)`). Old: `options={'shared_clock': False}`. `options['clip_start']`
    (`{stream: s | (s, sigma)}` on the device clock, or `'container'` = the files' `creation_time`, new in
    `sync.media_info`, sigma 1 s) keeps the clips on one timeline: `clip_start` graph edges, tied intercepts in the
    drift fit, tied offsets in the STBA (`SpatiotemporalBA(clip_ties=)`).
  - New `refine_windows=` (window length in s, or `(t0, t1[, cgroup])` windows): STBA per window, rates from the
    seed / drift fit; the time maps are the seed plus a piecewise-linear correction (new `sync.TimeMap.corrected`;
    continuous, monotone). Per-window results in `refined['windows']`, new rule `refine_windows_failed`. A
    piecewise seed keeps its knots (they were dropped).
  - `options['audio_delay_sigma']` (s, default 0 = old): added to every audio sigma (speaker distance, 2.9 ms/m,
    and phone A/V latency, which audio cannot see).
  - STBA: points with regular per-frame times (gaps of whole frames allowed) use the given times (frames after a
    gap were placed at `t0 + i / rate`); the rate correction is applied around the STBA's own clip mid-times.
- **`associate_views` (W7)**
  - Defaults: `gate_noise_mult` 80 -> 10 (the second, tighter pass never fired on real 3-4 px detections),
    `link_max_gap` 10 -> 30 frame ids (also `link_max_gap_s` with `fps`), `min_identity_frames=5` (1-3-frame ghost
    identities are dropped). Old: `options={'gate_noise_mult': 80, 'link_max_gap': 10, 'min_identity_frames': 0}`.
  - Tracklets: `split_on_switch=True` cuts a tracklet where the verification drops a run of its frames (an identity
    switch) and associates again; `stitch=True` joins identities that follow each other in time and space. New
    rule `verify_dropped_frac`. Old: `options={'split_on_switch': False, 'stitch': False}`.
  - `options['n_identities']`: known number of animals (at most that many per frame; re-entry instead of new ids).
  - Frame ids: linking uses frame-id differences; `options['frame_axis']='auto'` returns only the frames with
    detections (`frame_ids`) when the dense axis would be > 2x larger and > 1000 frames (ids around 1e5 took
    hundreds of MB). Old: `options={'frame_axis': 'dense'}`.

### Moved cameras and static artifacts (1.1, real-data follow-up)

- **`evaluate_calibration` (W4) `options['recover_moves']=True`** (new, on by default with `times=`;
  `stations.recover_moves`): a camera whose leave-one-out coverage collapses from a time on, and stays collapsed, is
  re-placed from the points the other cameras triangulate after that time (rotation grid search +-15 deg, then a
  robust PnP; translation kept only if it clearly helps) and gets a new station in `res.moves` (change point with
  `recovered=True`, `moves.station_poses`, `res.report['recovered_moves']`). With
  `options['detections']` (the `associate_views` input) and `options['frame_ids']` (or `C x F x P x J x 2` points,
  now accepted) the camera's detections are re-associated after the move and returned in `res.points` (same shape as
  the input); a camera that only lost its association but still fits its old pose is re-associated and reported as
  'not moved'. Inside a recovered interval the pose is refined per block of windows, so a second bump gets its own
  station (dancing G4: 6.2 deg + 1.6 deg). A camera the association lost from the start (tracklets crossing an 8 deg re-aim were all dropped) is
  placed window by window, equal neighbours sharing a station (Panoptic 00_15 8 deg: found at 20.0 s, 8.0 deg; before:
  missed, and a 0.6 deg false station on 00_05). A recovered pose must explain clearly more than the old one
  (coverage gain >= 25 % of the level before). Without detections only a camera's agreement where it sees the point
  counts. Why: the dancing G4 6.2
  deg bump was invisible to the whole chain because W7 with the old rig could not associate G4 any more.
  Old behaviour: `options={'recover_moves': False}`.
- **`detect_moves(retest=True)` is now an explain-away test** (`suspect_frac=0.5`): cameras with a step of z >= 0.5 x
  `z_thr` are suspects; each suspect's observations are rotated back by its fitted offsets and the other suspects
  re-tested; the suspect that removes the most step evidence from the others is the mover (`info['retest'][cam]
  ['explains']`) and the cameras it explains are cleared. A moved camera biases its neighbours' leave-one-out points,
  so they looked rotated too, and with W4's robust thresholds the mover itself could stay below `z_thr`: on Panoptic
  with an injected 1 deg re-aim of 00_15, W4 reported 00_03 + 00_24 (both false) and missed 00_15; now 00_15 only.
  The old re-test (exclude the strongest camera, keep it always) is gone; `retest=False` gives one pass.
- W4 `options['moves_bad_window_frac']=0.8` (new): a detected step whose shorter side has >= 80 % of the camera's
  observations in bad time windows (`bad_windows_frac`: several cameras broke there) is dropped from `moves`
  (`report['moves_demoted']`). Outdoor mouse 5_19: the cam 11 '13 deg step' at t = 1051 s lay in the broken first
  hour. Old: `options={'moves_bad_window_frac': None}`.
- `stations.Stations.poses` (new): start poses of stations (relative to the camera, rescaled to the rig).
  `detect_moves` fills them from its fitted rotations, `recover_moves` from its recovered poses, and
  `bundle_adjust_lm(stations=...)` / `calibrate_from_points(stations=...)` start each station there (before: every
  station started at its camera's first pose, so a large bump could not converge); W2's station fit selects its
  observations with the stations at those poses. `MoveReport.corrected_extrinsics` uses `station_poses` when present.
- `calibrate_from_points(stations=...)`: the per-camera LOO of the verdict judges each observation with its station's
  pose (`settings['loo_per_station']`); before, a moved camera was judged with its first pose (dancing P2: 70 px
  instead of ~4 px).
- **`reconstruct_3d` (W6) static artifacts** (`res.static [N,J]`, `report['static']`, rule `static_frac` warn > 5 %):
  points in a voxel of `static_voxel` (0.011) x rig scale that is occupied for >= `static_min_s` (40) s in >=
  `static_min_bins` (5) distinct `static_bin_s` (3600) s bins (the outdoor-mouse sync block; port of the project's
  `static_flags`). Flag only; `static_remove=True` drops them before filling. Needs hours of data, so short
  recordings are never flagged; time from `times=` or the frame index at `options['fps']` (30). Off:
  `options={'static_flag': False}`.
- `reconstruct_3d` `options['mask_regions']={cam: [(x, y, r), ...]}`: observations inside these image circles are
  dropped (`report['n_removed']['mask_regions']`).

### Defaults settled after the real-data check

- `associate_views`: `switch_min_frames` 3 -> 10 (3 over-split a real IoU tracker: 7 identities for 4 people);
  `options={'switch_min_frames': 3}` for trackers that swap often.
- `evaluate_calibration`: `min_window_obs` 50 -> 200, so sparse hours are not judged; `bad_windows_frac` is warn-only.

### Removed

- **`aniposelib.io`.** The file formats are gone: MATLAB (`from/to_matlab`), OpenCV YAML
  (`read/write/from/to_opencv_yaml`), tracktail `metadata.yaml` (`read/from/to_metadata_yaml`) and the disco JSON rig
  (`to/from_json_rig`), with the `[io]` extra (pyyaml). The geometry helpers moved to `aniposelib.geometry`:
  `camera_from_extrinsics_mat`, `reindex_points`, `decompose_projection`, `from/to_projection_matrix`, `from/to_dlt`
  (`Camera.from_extrinsics_mat` still works).
- `quality.paired_bootstrap`, `quality.temporal_split`, `quality.assess_reuse` (and their result classes):
  project statistics / policy, back to Emory and EvoVision.
- `cameras.triangulate_simple`, `cameras.triangulate_simple_batch`; `triangulate(method='svd')` stays.
- **The per-camera slow path.** Groups the packed kernels cannot represent (custom `Camera` subclasses that override
  projection, pinhole cameras with > 8 distortion coefficients) now raise `NotImplementedError` in `project`,
  `undistort_points`, `triangulate*` instead of falling back to a per-camera loop.
- **Manual board verification:** the `manually_verify` argument of `Checkerboard` / `CharucoBoard` and
  `manually_verify_board_detection` (the `cv2.imshow` popup). Passing `manually_verify=` is now a `TypeError`.
- **Removed (1.1, plan 04 decision 15):** the first-order per-camera time offset in `bundle_adjust_lm`
  (`ParamSpec(time_offset=...)`, `BAReport.time_offsets`). Time offsets live only in `stba.SpatiotemporalBA`
  (`free_offsets='per_camera'` gives the same per-camera deltas, with a spline motion model instead of finite
  differences). Stations (`stations=`) stay in `bundle_adjust_lm`.

### Changed

- **One LM core (1.1, plan 04 §3):** a shared Levenberg-Marquardt engine (`aniposelib._lm`: IRLS losses, Marquardt
  damping with acceptance on the true cost, Schur complement over generic point-like blocks, dense Cholesky or PCG
  reduced solve). `bundle_adjust_lm` (cost histories unchanged to ~1e-11 relative), `stba.SpatiotemporalBA`
  (unchanged to ~1e-12, about 2x faster; it now uses the bundle projection kernel), `init.sfm.refine_extrinsics`
  (now `bundle_adjust_lm` on ideal pinhole cameras; observations behind a camera are pruned) and
  `intrinsics.lm_board_adjust` (analytic instead of numeric Jacobians; ladder fits agree to ~1e-6 px, same
  iteration counts) all run on it. `method='adam'` is unchanged.

- **`utils.get_initial_extrinsics` / `CameraGroup.calibrate_rows` route through `init.board` (decision 22):**
  connected graphs give exactly the 1.0 result (same spanning tree and per-edge estimate, `tree='classic'`), plus a
  robust fallback for edges where the classic estimate raises or is non-finite (`method='classic'` disables it).
  Disconnected graphs still raise `ValueError`, now listing the components; `disconnected='components'` opts in to
  per-component gauges. `init.board.initial_extrinsics(tree=)` is new (default `'max'`, unchanged).
- **`init.calibrate_from_points` is an alias of the workflow** (same signature and `(cgroup, RecipeReport)` return).
  Changes: the robust refinement uses the noise-normalised threshold schedule (coarse round relative to the image
  diagonal, then multiples of the noise floor; identical to the 1.0 schedule for 0.5 px noise), `rng=None` now means
  seed 0, and `report.scale` is a `workflows.ScaleEstimate` (`.details['align']` as before).
- **`triangulation.triangulate_robust` / `triangulate_possible` do the lens-fold / ray-consistency check** (`check_invertible=True`, plan 04 decision 24), formerly only in the `CameraGroup` wrappers; function and method now agree. Results are unchanged where no lens folds; for strongly barrel-distorted lenses the function results change like the method's did in 1.0. `check_invertible=False` restores the 1.0 function behaviour.
- `triangulate_robust(scores=...)`: confidences as soft weights (clipped to [0, 1]) in the MSAC hypothesis cost, the inlier refit and the Gauss-Newton refinement (and the IRLS / reject fits). New keyword; no default change.
- **`optim_points(normalize=True)` default (plan 04 decision 24, plan 05 §6.3).** The optimisation runs in pixel-equivalent units (points parametrised as `center + u * q`, `u` = median z/f at the points), so its result no longer depends on the rig's units (mm vs m vs km give the same answer; before, the Adam step `max_lr` and the bone/smoothness weights were in world units). Bone targets are fixed at a robust per-bone median (frames where both joints are triangulated) instead of free parameters clipped across bones, and the bone term is a squared pixel-equivalent residual (`10 * scale_length`), so bone constraints now tighten bone lengths within a short budget (the 1.0 target `test_optim_points_tightens_bones` passes). Smoothness is `0.01 * scale_smooth / mean|diff|` on pixel-equivalent differences (a light default). `normalize=False` restores the 1.0 loss and parametrisation exactly.
- `optim_points(normalize=True)`: joints that are never triangulated start at the scene centre instead of the world origin.
- `CharucoBoard.draw` / `Checkerboard.draw` default to `margin='auto'` (half a square of white border), so rendered boards are directly detectable (edge-to-edge renders lost most edge corners: 6/20 on a 6x5 board). `margin=0` restores the old edge-to-edge output.
- Dev env: the default pixi env now includes all optional extras (vggt @ a288dd0, torchvision cu128, huggingface_hub, safetensors, pillow, onnxruntime, onnx, soundfile, av); the real VGGT-1B smoke test (`-m slow`) now runs when the weights are in the local HF cache.

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
