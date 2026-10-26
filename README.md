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

## 1.0 API overview

Optional extras: `[deepcharuco]` (learned ChArUco detector: `huggingface_hub`, `onnxruntime`, `onnx`), `[vggt]`
(VGGT initialisation: `vggt`, `huggingface_hub`), `[sync]` (audio decoding: `soundfile`, `av`; an `ffmpeg` executable
also works), `[io]` (`pyyaml`, for hand-written tracktail `metadata.yaml`), `[all]`. `import aniposelib` imports none
of them; sub-modules (`aniposelib.quality`, `.scale`, `.sync`, `.init`, ...) load on first access.

**Types.** Points are `C x N x 2` (or `C x N x J x 2`) in camera order, NaN = missing; 3D points are `N x 3`.
Public methods accept numpy or torch: **numpy in gives numpy out, torch in gives detached torch out** (on the
input's device), and `return_grad=True` keeps the autograd graph. Computation is float64 torch, on the cameras'
device (`cgroup.to('cuda')`). Result objects are dataclasses with `to_dict()`.

### Board calibration
```python
import aniposelib
from aniposelib import CameraGroup, CharucoBoard

board = CharucoBoard(7, 5, square_length=25, marker_length=18.75, marker_bits=4, dict_size=50, units='mm')
# board = CharucoBoard(..., detector='deepcharuco')   # learned detector, [deepcharuco] extra
cgroup = CameraGroup.from_names(['A', 'B', 'C'], fisheye=False)
error, rows = cgroup.calibrate_videos([['A.mp4'], ['B.mp4'], ['C.mp4']], board)   # LM bundle adjustment
cgroup.dump('calibration.toml')
cgroup = CameraGroup.load('calibration.toml')     # numeric camera order: cam_2 < cam_10
```

### Triangulation
```python
p3d = cgroup.triangulate(p2d)                       # p2d [C,N,2] -> [N,3], linear block DLT
res = cgroup.triangulate_robust(p2d, tau=10)        # MSAC: res.p3d, res.inliers, res.errors, res.n_inliers
err = cgroup.reprojection_error(p3d, p2d)           # [C,N,2]
p3d = cgroup.optim_points(p2d_joints, p3d_joints)   # smoothed / constrained refinement
```

### Keypoint self-calibration (no board)
```python
from aniposelib.init import calibrate_from_points
cgroup, rep = calibrate_from_points(p2d, intrinsics=intrinsics_group)   # [C,N,2] or [C,N,J,2]
# candidates (reference / SfM / VGGT with frames=...), selection, bundle_adjust_robust, quality report
print(rep.init, rep.quality.table())
cgroup.bundle_adjust_robust(p2d)                    # robust LM BA -> BAReport (resection, rejection)
```

### Quality, scale, gauge
```python
rep = cgroup.report(p2d)                # per-camera + leave-one-camera-out reprojection (CalibrationReport)
rep.passes(), rep.failures(), rep.to_json('qa.json')
from aniposelib import scale
s, res = scale.from_board(cgroup, rows, board)      # or scale.from_lengths / scale.from_reference
res.apply(cgroup)                                   # rescales, writes metadata['units'] / provenance
cgroup.align_to(reference_group)                    # trimmed similarity onto another calibration
cgroup.align_world(ground_points, up='z')           # floor -> z = 0
cgroup.anchor_to('A')                               # camera A is the world frame
```

### Synchronisation and moving cameras
```python
from aniposelib import sync
est = sync.audio_offset(audio_a, audio_b, sr)       # est.offset (s), est.sigma, est.identifiable, est.reasons
graph = sync.solve_offset_graph(edges)              # per-stream offsets from pairwise estimates
from aniposelib.stba import SpatiotemporalBA         # sub-frame offsets, clock rate, rolling shutter
moves = aniposelib.quality.detect_moves(cgroup, p2d, times)   # bumped cameras -> Stations
cgroup.stations = moves.stations                     # saved in the TOML ([metadata.stations])
```

Other modules: `aniposelib.io` (MATLAB, OpenCV YAML, DLT/P matrices, tracktail `metadata.yaml`, JSON rigs),
`aniposelib.matching` (cross-camera tracklet / multi-animal association), `aniposelib.intrinsics` (model ladder),
`aniposelib.params.ParamSpec` (which parameters BA may change, priors, sharing), `aniposelib.geometry` /
`aniposelib.triangulation` (batched kernels on `PackedCameras`).

**Upgrading from 0.9:** 1.0 changes several defaults (numpy-out returns, LM bundle adjustment with a fixed gauge
instead of Adam, numeric camera order on load, linear triangulation, full-K projection). Every change and how to get
the old behaviour is in [CHANGELOG.md](CHANGELOG.md).

The [documentation](https://anipose.readthedocs.io/en/latest/aniposelib-tutorial.html) is located in Anipose repository.

