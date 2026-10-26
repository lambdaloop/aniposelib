# 1.0-iomatch: `aniposelib.io`, `aniposelib.affine`, `aniposelib.matching` (plan 01 phase G, F6)

All additive (new modules); no existing behaviour changes.

## `aniposelib.io` (new)
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

## `aniposelib.affine` (new)
- `AffineCamera(A, b)` (telecentric `uv = A X + b`, `nn.Module`): `project`, `reprojection_error`,
  `get_projection_matrix`, `get_direction`, dict round trip (`model = 'affine'`), and `to_pinhole(support,
  max_error_px=0.5)` (the tailcyclenet `johnson_fly` telephoto-pinhole construction, checked on support points).
- `triangulate_affine(cameras, points)` (exact linear LS), `affine_from_points(X, uv)`.
- Not (yet) usable inside `CameraGroup` (projection/BA there assume pinhole/fisheye).
