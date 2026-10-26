# boards (B17, D8, D9, D10)

## OpenCV / dependencies
- **`opencv-contrib-python` → `opencv-python>=4.8,<5`.** `boards.py` now uses only the main-module ArUco class
  API (`cv2.aruco.ArucoDetector`, `CharucoDetector`, `CharucoBoard.generateImage`, `solvePnP`). The ChArUco path
  works again on every OpenCV ≥ 4.8 (it silently found nothing on opencv-python ≥ 4.7 / contrib ≥ 4.10).
  *Old behaviour:* having `opencv-contrib-python` installed instead still works (it also provides `cv2`); do not
  install both.
- New pixi env `nocontrib` (`pixi run -e nocontrib test-nocontrib`) with only `opencv-python`, plus a guard test
  that fails if any legacy contrib `cv2.aruco.*` free function (`detectMarkers`, `interpolateCornersCharuco`,
  `estimatePoseCharucoBoard`, ...) reappears in `aniposelib/`.
- New optional extra `aniposelib-pytorch[deepcharuco]` (`huggingface_hub`, `onnxruntime`, `onnx`).

## `CharucoBoard`
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

## `aniposelib.detectors` (new, D8)
- `DeepCharucoDetector(board, device='auto', backend='torch'|'onnx', batch_size=16, refine=True, scale=1.0,
  min_score=0.0, subpix=False, color_order='bgr', weights_dir=None, repo='lambdaloop/deepcharuco-weights')`:
  Lightning-free batched DeepCharuco nets (from disco `deepcharuco_disco.py`) with EvoVision's coordinate
  conventions (isotropic scale, pad to /8, `cv2.resize` pixel-centre inverse) and ONNX export. Weights are
  resolved from `weights_dir` / `$DEEPCHARUCO_WEIGHTS`, the Hugging Face cache, `~/.cache/aniposelib/deepcharuco`,
  then downloaded (huggingface_hub if installed, else plain HTTPS). Unsupported boards raise `ValueError` listing
  the published geometries (5–9 rows × 5–9 columns; r6_c5 has no refinenet).
- Use as `CharucoBoard(..., detector='deepcharuco', detector_kwargs=dict(device='cuda'))`; returns the same rows
  format, so `calibrate_rows`/`calibrate_videos` work unchanged.

## `aniposelib.intrinsics` (new, D9)
- `calibrate_ladder(objpoints, imgpoints, image_size, models=...)` / `calibrate_ladder_rows(rows, board, size)`:
  fits pinhole → k1k2 → k1k2p → standard_5param (→ rational, fisheye_k1k2, fisheye) with nested inits and a
  Schur-LM solver, scores each on held-out time blocks, and picks the most parsimonious model within 15 % of the
  best held-out p95 (ported from disco `board_intrinsics_ba.py`). `LadderResult.to_camera()` gives a
  `Camera`/`FisheyeCamera` with the matching `n_dist`.
