# 1.0-integrate: final integration (aliases, stations in the group, lazy modules, docs)

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
