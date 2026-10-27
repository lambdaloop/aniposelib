# Anipose lib

 [![License](https://img.shields.io/badge/License-BSD%202--Clause-orange.svg)](https://opensource.org/licenses/BSD-2-Clause)
 
An easy-to-use library for calibrating cameras and triangulation in Python.

This is the backend library for the [Anipose](https://github.com/lambdaloop/anipose) package.

## aniposelib-pytorch

This branch is published on PyPI as **`aniposelib-pytorch`**: the PyTorch/GPU version of aniposelib
(cameras are `torch.nn.Module`s, GPU-batched triangulation and bundle adjustment). It still
**imports as `aniposelib`**.

```bash
pip install aniposelib-pytorch
```

Install torch with the CUDA build that matches your GPU first if you need a specific one
(see https://pytorch.org/get-started/locally/).

**It cannot be installed alongside classic `aniposelib`** (numpy, 0.8.x) because both provide the
`aniposelib` import. `aniposelib-pytorch` depends on a file-less placeholder `aniposelib==0.0.1`, so
pip/uv refuse environments that also require classic aniposelib. Replace any `aniposelib`
requirement in your project with `aniposelib-pytorch`. If both somehow end up installed, `import
aniposelib` raises an error; fix it with:

```bash
pip uninstall -y aniposelib aniposelib-pytorch && pip install aniposelib-pytorch
```

Do not `pip install aniposelib==0.8.x` into an environment with aniposelib-pytorch: it overwrites
the package files. Classic aniposelib is still available with `pip install aniposelib` in a
separate environment.

## Quick start: the seven workflows

Each workflow is one function, re-exported at the top level. Each returns a result with `.verdict` (`'ok'` |
`'warn'` | `'fail'`), `.reasons`, `.report`, `.settings` (every derived threshold), `.summary()` and `.to_json()`.
Thresholds come from the measured pixel noise and the rig scale, so the defaults work at any unit scale without
tuning. A bad result is a `'fail'` with reasons, not an exception. Full guide: [docs/workflows.md](docs/workflows.md).

```python
import aniposelib as al

board = al.CharucoBoard(7, 5, square_length=25, marker_length=18.75, units='mm')

cgroup, res = al.calibrate_from_board(videos, board, out='calibration.toml')         # W1 board (or al.PointObject)
cgroup, res = al.calibrate_from_points(p2d, intrinsics=lab_cgroup)                  # W2 from keypoints / people
res = al.estimate_scale(cgroup, board=board, board_points=rows)                     # W3 metric scale, world frame
res = al.evaluate_calibration(cgroup, p2d, times=t, reference=old_cgroup)           # W4 QA, reuse, moved cameras
res = al.estimate_offsets({'cam1': 'cam1.mp4', 'cam2': 'cam2.mov'}, cgroup=cgroup)  # W5 unsynchronised cameras
res = al.reconstruct_3d(cgroup, p2d, scores=scores, bones=[(0, 1), (1, 2)])          # W6 2D -> 3D
res = al.associate_views(cgroup, detections)                                        # W7 several animals / people
print(res.summary())
```

Optional extras: `[deepcharuco]` (learned ChArUco detector: `huggingface_hub`, `onnxruntime`, `onnx`), `[vggt]`
(VGGT initialisation: `vggt`, `huggingface_hub`), `[sync]` (audio decoding: `soundfile`, `av`; an `ffmpeg` executable
also works), `[all]`. `import aniposelib` imports none of them; sub-modules (`aniposelib.quality`, `.scale`, `.sync`,
`.init`, ...) load on first access.

**Types.** Points are `C x N x 2` (or `C x N x J x 2`) in camera order, NaN = missing; 3D points are `N x 3`.
Public methods accept numpy or torch: **numpy in gives numpy out, torch in gives detached torch out** (on the
input's device), and `return_grad=True` keeps the autograd graph. Computation is float64 torch, on the cameras'
device (`cgroup.to('cuda')`).

## Advanced: the backend

The workflows only compose these; use them directly when you need finer control.

### Board calibration (anipose CLI API)
```python
from aniposelib import CameraGroup, CharucoBoard
cgroup = CameraGroup.from_names(['A', 'B', 'C'], fisheye=False)
error, rows = cgroup.calibrate_videos([['A.mp4'], ['B.mp4'], ['C.mp4']], board)   # LM bundle adjustment
cgroup.dump('calibration.toml')
cgroup = CameraGroup.load('calibration.toml')     # numeric camera order: cam_2 < cam_10
```

### Triangulation
```python
p3d = cgroup.triangulate(p2d)                       # p2d [C,N,2] -> [N,3], linear block DLT
res = cgroup.triangulate_robust(p2d, tau=10, scores=scores)   # MSAC: res.p3d, res.inliers, res.errors
err = cgroup.reprojection_error(p3d, p2d)           # [C,N,2]
p3d = cgroup.optim_points(p2d_joints, p3d_joints)   # smoothed / bone-constrained refinement (unit-free)
```

### Bundle adjustment, quality, scale, gauge
```python
cgroup.bundle_adjust_robust(p2d)                    # robust LM BA -> BAReport (resection, rejection)
rep = cgroup.report(p2d)                            # per-camera + leave-one-camera-out reprojection
from aniposelib import scale
s, res = scale.from_board(cgroup, rows, board)      # or scale.from_lengths / scale.from_reference
cgroup.align_to(reference_group)                    # trimmed similarity onto another calibration
cgroup.align_world(ground_points, up='z')           # floor -> z = 0
```

### Synchronisation and moving cameras
```python
from aniposelib import sync
est = sync.audio_offset(audio_a, audio_b, sr)       # est.offset (s), est.sigma, est.identifiable
graph = sync.solve_offset_graph(edges)              # per-stream offsets from pairwise estimates
from aniposelib.stba import SpatiotemporalBA         # sub-frame offsets, clock rate, rolling shutter
moves = aniposelib.quality.detect_moves(cgroup, p2d, times)   # bumped cameras -> Stations
```

Other modules: `aniposelib.init` (board graph, keypoint SfM, VGGT, candidate selection),
`aniposelib.matching` (cross-camera association), `aniposelib.intrinsics` (lens-model ladder),
`aniposelib.params.ParamSpec` (which parameters BA may change, priors, sharing), `aniposelib.geometry`
(batched camera kernels on `PackedCameras`, plus `camera_from_extrinsics_mat`, `reindex_points`, projection-matrix /
DLT conversion), `aniposelib.triangulation`, `aniposelib.affine`.

**Upgrading:** 1.1 removes the file-format importers (`aniposelib.io`) and a few project-specific helpers (see CHANGELOG, "Removed in 1.1"). 1.0 changes several defaults (numpy-out returns, LM bundle adjustment with a fixed gauge
instead of Adam, numeric camera order on load, linear triangulation, full-K projection). Every change and how to get
the old behaviour is in [CHANGELOG.md](CHANGELOG.md).

The [documentation](https://anipose.readthedocs.io/en/latest/aniposelib-tutorial.html) is located in Anipose repository.

