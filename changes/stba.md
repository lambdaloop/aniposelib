# 1.0-stba: `aniposelib.stba` and `aniposelib.stations` (plan 01 F4, F5, F7 STBA side)

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
