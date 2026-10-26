# kernels (1.0-kernels): packed geometry + fast/robust triangulation (plan 01 C1-C3)

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
