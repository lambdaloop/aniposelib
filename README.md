# Anipose lib

[![License](https://img.shields.io/badge/License-BSD%202--Clause-orange.svg)](https://opensource.org/licenses/BSD-2-Clause)

aniposelib calibrates groups of cameras and turns 2D points seen by several cameras into 3D. It is the library behind
[Anipose](https://github.com/lambdaloop/anipose), and you can also use it on its own.

This branch is the PyTorch version, published on PyPI as **`aniposelib-pytorch`**. It runs on the GPU when one is
available, and it still imports as `aniposelib`.

## Installing

```bash
pip install aniposelib-pytorch
```

If you need a particular CUDA build of PyTorch, install that first (see https://pytorch.org/get-started/locally/).

A few features need extra packages, which you can add with pip extras:

- `aniposelib-pytorch[deepcharuco]`: a learned ChArUco board detector, which finds boards that OpenCV misses.
- `aniposelib-pytorch[vggt]`: VGGT, which can guess camera positions straight from images.
- `aniposelib-pytorch[sync]`: reading audio from video files, for synchronising cameras.
- `aniposelib-pytorch[all]`: all of the above.

**Don't install it next to the classic `aniposelib` package** (the numpy version, 0.8.x). Both provide the
`aniposelib` import, so one would overwrite the other. If your project lists `aniposelib` as a requirement, replace it
with `aniposelib-pytorch`. If both end up installed anyway, `import aniposelib` stops with an error; fix it with:

```bash
pip uninstall -y aniposelib aniposelib-pytorch && pip install aniposelib-pytorch
```

## What you can do with it

Most projects need some of these seven steps. Each one is a single function.

| I want to... | Function |
|---|---|
| calibrate my cameras with a calibration board (or another object of known shape) | `calibrate_from_board` |
| calibrate my cameras from the animals or people in the videos, without a board | `calibrate_from_points` |
| put a calibration into real units (mm) or line it up with the floor or an older calibration | `estimate_scale` |
| check whether a calibration is good, or still good, or whether a camera was bumped | `evaluate_calibration` |
| line up cameras that weren't recording in sync | `estimate_offsets` |
| turn 2D keypoints from several cameras into 3D | `reconstruct_3d` |
| work out which animal is which when several are in view | `associate_views` |

A typical lab setup with a ChArUco board looks like this:

```python
import aniposelib as al

board = al.CharucoBoard(7, 5, square_length=25, marker_length=18.75, units='mm')

# 1. calibrate the cameras from videos of the board
cgroup, result = al.calibrate_from_board([['cam1.mp4'], ['cam2.mp4'], ['cam3.mp4']], board,
                                         cam_names=['cam1', 'cam2', 'cam3'],
                                         out='calibration.toml')
print(result.summary())

# 2. later, turn 2D keypoints into 3D
#    points2d has shape (cameras, frames, keypoints, 2), with NaN where a keypoint wasn't found
recon = al.reconstruct_3d(cgroup, points2d, scores=scores)
points3d = recon.points3d            # (frames, keypoints, 3)
```

Every function returns a result that tells you how it went:

- `result.verdict` is `'ok'`, `'warn'` or `'fail'`.
- `result.reasons` says, in plain words, what was wrong.
- `result.summary()` prints a short table of the checks.

A bad result comes back as `'fail'` with a reason; it doesn't raise an error. You shouldn't need to tune thresholds:
they're worked out from how noisy your data is and how big your camera setup is, so the same call works whether
your setup is measured in millimetres or metres.

The [workflow guide](docs/workflows.md) walks through each step with examples.

## The lower-level toolbox

The seven functions are built from a larger set of tools, which you can use directly when you need more control:
camera models, board detection, triangulation, bundle adjustment, quality reports, scale and alignment, and
synchronisation. Some common ones:

```python
from aniposelib import CameraGroup

cgroup = CameraGroup.load('calibration.toml')
cgroup.dump('calibration.toml')

p3d = cgroup.triangulate(p2d)                      # simple triangulation, p2d has shape (cameras, points, 2)
res = cgroup.triangulate_robust(p2d)               # ignores views that disagree with the others
err = cgroup.reprojection_error(p3d, p2d)
cgroup.bundle_adjust_robust(p2d)                   # refine the calibration on your points
report = cgroup.report(p2d)                        # per-camera errors, including leave-one-camera-out
```

Anipose itself uses the classic calibration call, which still works the same way:

```python
cgroup = CameraGroup.from_names(['A', 'B', 'C'])
error, rows = cgroup.calibrate_videos([['A.mp4'], ['B.mp4'], ['C.mp4']], board)
```

Where to find the rest:

- `aniposelib.init`: ways to get a first guess of the camera positions (from a board, from points, from VGGT).
- `aniposelib.quality`: calibration checks.
- `aniposelib.scale`: scale and alignment.
- `aniposelib.sync` and `aniposelib.stba`: camera synchronisation.
- `aniposelib.matching`: matching animals across cameras.
- `aniposelib.intrinsics`: picking a lens model.
- `aniposelib.geometry` and `aniposelib.triangulation`: the batched maths underneath.

**A note on data shapes.** 2D points are arrays with the camera as the first axis, `(cameras, points, 2)` or
`(cameras, frames, keypoints, 2)`, with NaN for missing points. If you pass numpy arrays you get numpy arrays back;
if you pass torch tensors you get torch tensors back.

## Upgrading

Version 1.1 removed the file-format converters (`aniposelib.io`) and a few helpers that were specific to one project.
Version 1.0 changed several defaults compared with 0.9. [CHANGELOG.md](CHANGELOG.md) lists every change and how to
get the old behaviour back.

The [Anipose documentation](https://anipose.readthedocs.io/en/latest/aniposelib-tutorial.html) has a tutorial.
