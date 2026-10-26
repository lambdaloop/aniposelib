# Tests

- `pixi run test`: the default run (unit + fast `integration` tests). Must stay under ~60 s on CPU.
- `pixi run test-unit`: unit tests only.
- `pixi run test-targets`: **1.0 goals** (`@pytest.mark.target`, xfail). XPASS means the goal is met: remove the
  `target`/`xfail` markers so the test joins the default run.
- `pixi run test-all`: everything, including `slow`.
- The `device` fixture runs a test on cpu and (if available) cuda. To force CPU, set `CUDA_VISIBLE_DEVICES=`.

## Use-case rigs (`data/rigs/*.toml`, see `synth.RIGS`)
Real calibrations from downstream projects. **All 2D data is synthetic** (`synth.py`, seeded), so there is exact
ground truth and no large fixtures.

| rig | use case | tests |
|---|---|---|
| `allen_7cam`, `allen_12cam` | lab mouse rig, ChArUco board (mm) | `test_rig_lab_board.py` |
| `outdoor_12cam`, `outdoor_9cam` | outdoor thermal arena, trajectory, false detections | `test_rig_outdoor.py`, `test_triangulate_ransac.py` |
| `fisheye_5cam` | EvoVision fisheye GoPros, people, metric scale | `test_rig_fisheye_people.py`, `test_fisheye.py`, `test_n_dist.py` |
| `motobs_7cam` | keypoint self-calibration, decoy camera, LOO QA | `test_rig_keypoint_decoy.py`, `test_cameragroup.py` |

Generators in `synth.py`:
- scenes: `random_points`, `trajectory`, `skeleton` (`MOUSE`/`PERSON`), `board_rows` (`calibrate_rows` format),
  `board_corners_3d`;
- corruption: `add_noise`, `drop_views`, `add_outliers`, `add_decoy`;
- rig helpers: `perturb_extrinsics`, `center_error` (similarity-aligned camera centres, as a fraction of rig scale),
  `rig_scale`, `scene_center`.
