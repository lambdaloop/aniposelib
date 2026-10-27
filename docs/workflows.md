# A guide to the aniposelib workflows

aniposelib has a lot of tools in it. Most projects, though, only ever do a handful of things: calibrate the cameras,
check the calibration, sometimes sync the cameras, and turn 2D keypoints into 3D. Each of those jobs has one function
that does the whole thing with sensible defaults. This guide explains when to use each one and what it gives back.

- [Before you start: what every function has in common](#before-you-start)
- [Calibrating with a board](#calibrating-with-a-board-calibrate_from_board)
- [Calibrating with an object of known shape](#calibrating-with-an-object-of-known-shape-pointobject)
- [Calibrating from the animals themselves](#calibrating-from-the-animals-themselves-calibrate_from_points)
- [Getting real-world units](#getting-real-world-units-estimate_scale)
- [Checking a calibration](#checking-a-calibration-evaluate_calibration)
- [Syncing cameras](#syncing-cameras-estimate_offsets)
- [Going from 2D to 3D](#going-from-2d-to-3d-reconstruct_3d)
- [Several animals at once](#several-animals-at-once-associate_views)
- [Putting it together](#putting-it-together)

## Before you start

**Data shapes.** 2D points always have the camera as the first axis: `(cameras, points, 2)`, or
`(cameras, frames, keypoints, 2)` for poses. Use NaN for a point a camera didn't see. Pass numpy and you get numpy
back.

**Every function tells you how it went.** Each one returns a result object with:

- `verdict`: `'ok'`, `'warn'` or `'fail'`;
- `reasons`: a list of plain-language explanations for anything that isn't ok;
- `summary()`: a printable table of every check, its value and its limits;
- `to_json(path)`: the whole report, to keep next to your results.

```python
cgroup, result = al.calibrate_from_board(videos, board)
if result.verdict == 'fail':
    print(result.summary())
```

A poor result is reported as `'fail'`; the function doesn't crash. It only raises an error when the input itself
can't be used, for example arrays of the wrong shape.

**You shouldn't need to tune anything.** The thresholds aren't fixed numbers of pixels or millimetres. They're worked
out from your data: how noisy the 2D points are, how big the images are, and how big your camera setup is. The same
call works on a mouse rig measured in millimetres and a field setup measured in metres. All the values that were
worked out are saved in `result.settings`, so a run can be reproduced.

If you do need to change something, every function takes an `options={...}` dictionary. For example, to use your own
pixel limits for the checks:

```python
result = al.evaluate_calibration(cgroup, points2d,
                                 options={'verdict': {'loo_worst_px': (5, 15)}})   # warn above 5 px, fail above 15
```

**Saving.** Calibration functions accept `out='calibration.toml'`, which saves the calibration and a report next to it
(`calibration.report.json`). Without `out`, nothing is written to disk.

---

## Calibrating with a board: `calibrate_from_board`

Use this when you've filmed a calibration board (ChArUco or checkerboard) with all your cameras. This is the usual lab
setup.

```python
import aniposelib as al

board = al.CharucoBoard(7, 5, square_length=25, marker_length=18.75, units='mm')
cgroup, result = al.calibrate_from_board(
    [['cam1.mp4'], ['cam2.mp4'], ['cam3.mp4']],   # one list of videos per camera
    board,
    cam_names=['cam1', 'cam2', 'cam3'],
    out='calibration.toml')
```

What it does:

1. Finds the board in the videos. If the learned DeepCharuco detector is installed it uses that, otherwise OpenCV.
2. Works out each camera's lens (focal length and distortion). It tries a few lens models, from simple to complex,
   and keeps the simplest one that predicts board views it hasn't seen.
3. Works out where the cameras are relative to each other, from frames where two cameras see the board at once.
4. Refines everything together.

Use `fisheye=True` for fisheye lenses. If you already know the lenses, pass an earlier calibration as `intrinsics=` to
start from it.

**If some cameras never saw the board at the same time as the others.** Imagine cameras on the left of the arena and
cameras on the right, and the board was only ever shown to one side at a time. Each side can be calibrated, but
nothing says where the left side is relative to the right side. In that case you get `'fail'`, plus a separate
calibration for each group in `result.component_groups`. To join them you have two options:

- record a few more board frames where both sides can see it; or
- pass `bridge_points=`, 2D tracks of anything both sides saw at the same time (the animal, a person walking through).
  aniposelib then works out how the groups sit relative to each other, keeping the lenses and the real-world scale
  from the board.

The checks: how far the board corners land from where the calibration predicts (compared with the noise in the
detections), whether every camera saw the board, whether all cameras are connected, and whether the board comes out
the right shape in 3D.

## Calibrating with an object of known shape: `PointObject`

Sometimes you don't have a printed board, but you do have something whose 3D shape you know: a wand with markers on
it, a 3D-printed fixture, or points in the arena you've measured. You mark those points by hand in each camera, then
calibrate with the same function.

```python
wand = al.PointObject(points3d,                       # (points, 3) known positions, in mm
                      names=['tip', 'middle', 'base', ...],
                      units='mm')
annotations = wand.annotations_from_dicts({
    'cam1': {0: {'tip': (512, 300), 'middle': (530, 410)}, 5: {...}},   # camera -> frame -> point -> (x, y)
    'cam2': {...},
})
cgroup, result = al.calibrate_from_board(annotations, wand, cam_names=['cam1', 'cam2', 'cam3'])
```

There are two kinds of object:

- **Moving objects** (the default): you wave the object around, like a board.
- **Fixed landmarks** (`static=True`): the points never move, such as measured marks on the floor. Each camera is
  placed directly against them, and the calibration comes out in the landmarks' own coordinates and units. That gives
  you a world frame and real-world scale for free.

If the object isn't flat, aniposelib can estimate the lenses from it, as long as a camera sees at least 6 of the
points in a frame. With fewer points, pass known lenses as `intrinsics=`.

## Calibrating from the animals themselves: `calibrate_from_points`

Use this when there's no board in the recording, or the board was only good enough to get the lenses. The 2D points
can be keypoints from a pose model, people, or anything else you can track in several cameras.

```python
cgroup, result = al.calibrate_from_points(
    points2d,                       # (cameras, frames, keypoints, 2), NaN where missing
    intrinsics=lab_calibration,     # the lenses: an earlier calibration, or nominal values
    out='calibration.toml')
print(result.summary())
```

What it does:

1. Gets a few first guesses of where the cameras are: the calibration you passed in (if it has camera positions), a
   reconstruction from the points themselves, and, if you pass video frames with `frames=`, VGGT. It keeps the guess
   that fits the data best.
2. Refines the camera positions while ignoring points that don't fit, such as wrong detections or a camera confusing
   one animal for another.
3. Optionally fine-tunes the focal length and main distortion term, but keeps the change only if it really predicts
   held-out data better (`refine_intrinsics='auto'`).
4. Optionally puts the result into real units (see `estimate_scale`, which you can pass through `scale=`).

The result has no real-world scale unless you give it one: the points alone can't tell a small rig from a big one.

The main check is a **leave-one-camera-out** error: each camera is hidden in turn, the points are rebuilt in 3D from
the other cameras, and then projected into the hidden camera to see how far off they land. This catches a camera in
the wrong place, which a plain reprojection error can miss.

## Getting real-world units: `estimate_scale`

Calibrations from points (or from VGGT) come out in arbitrary units. This puts them into millimetres, metres or
whatever you use. Give it one of three things:

```python
# a board seen in the recording
result = al.estimate_scale(cgroup, board=board, board_points=board_detections)

# known distances, e.g. a bone length or two markers 25 mm apart
result = al.estimate_scale(cgroup, lengths=dict(points=points2d, pairs=[(0, 3)], ref=[25.0]))

# an older calibration of the same cameras
result = al.estimate_scale(cgroup, reference=old_calibration, align='reference')

scaled = result.cgroup           # a rescaled copy; the original is left alone
```

It measures the scale in many frames separately and takes the middle value. If those measurements disagree too much,
it refuses (`result.accepted` is False and the verdict is `'fail'`) rather than give you a scale it doesn't trust.
`align='ground'` also turns the calibration so that the floor is flat at height zero.

## Checking a calibration: `evaluate_calibration`

Use this to check a new calibration, to decide whether last week's calibration can still be used today, or to find
out whether a camera was bumped during a session.

```python
result = al.evaluate_calibration(cgroup, points2d)
result.per_camera      # errors and coverage for each camera
```

It reports, for each camera, the leave-one-camera-out error, how many of its points were usable, and whether its
lens model behaves well across the whole image. It also flags pairs of cameras that look at the scene from almost
the same direction, which triangulate poorly. Some inputs switch on extra checks:

- `times=`, the time of each frame: it looks for cameras that moved during the recording, and when.
- `reference=`, another calibration: it compares the two.
- a board or `PointObject` and its detections: it checks that the board comes out the right shape.

The function only reports how good the calibration is. The decision is up to you, for example a rule that a
calibration is reused only while every camera stays under some error.

## Syncing cameras: `estimate_offsets`

Use this when the cameras weren't started together, such as GoPros or phones.

```python
result = al.estimate_offsets({'gopro1': 'gopro1.mp4', 'phone': 'phone.mov'})
result.offsets         # how far each camera's clock is from the reference camera, in seconds
result.timemaps        # converts between global time and each camera's own time
```

It uses whatever it has for each pair of cameras: the soundtrack (which works well when there's music or clapping),
how much is moving in the picture, and, if you give it a calibration and 2D tracks, how well the tracks line up in
3D at different offsets. It combines all the pairs, throws out ones that don't agree with the rest, and can also
correct cameras whose clocks run slightly fast or slow (`drift=True`, the default). With `refine=True` it goes further
and fits sub-frame offsets from the 2D tracks.

It never changes your data. Use `result.timemaps` to line the frames up yourself. A camera it couldn't place in time
(no sound overlap, nothing to compare) is reported, not guessed.

## Going from 2D to 3D: `reconstruct_3d`

Use this to turn 2D keypoints from several cameras into 3D tracks.

```python
result = al.reconstruct_3d(cgroup, points2d, scores=scores)
points3d = result.points3d       # (frames, keypoints, 3), NaN where there weren't enough cameras
```

For every point it picks the cameras that agree with each other and ignores ones that don't, so a single bad
detection doesn't drag the 3D point away.

If your pose model gives a confidence for each keypoint, pass it as `scores=`. Confident detections count more and
unsure ones count less, but nothing is thrown out just for having a low score. You don't need to pick a cutoff. If you
want one anyway, use `options={'min_score': 0.3}`.

Two optional extras:

- `bones=[(0, 1), (1, 2), ...]`: pairs of keypoints whose distance should stay the same (a bone). The 3D points are
  adjusted so those lengths stay steady.
- `smooth=`: smooths the tracks over time. Off by default.

Gaps in the middle of a track are filled in, but a track is never extended before its first or after its last
detection.

## Several animals at once: `associate_views`

Use this when more than one animal or person is in view, so you first have to work out which detection in each
camera belongs to which individual.

```python
result = al.associate_views(cgroup, detections)      # detections: per camera, per frame, all the instances found
result.points          # (cameras, frames, individuals, keypoints, 2), ready for reconstruct_3d
result.labels          # for each detection, which individual it was matched to (-1 if none)
```

If your 2D tracker already links each animal over time within a camera, pass those IDs as `track_ids=`. Whole tracks
are then matched across cameras, which is more reliable than matching frame by frame. Without them, each frame is
matched on its own and the matches are joined up over time.

---

## Putting it together

Here's how the projects that aniposelib was built around would use these steps.

**A lab rig with a board** (for example, the classic Anipose setup):
calibrate with `calibrate_from_board`, then `reconstruct_3d` for every session.

**A rig that's recalibrated from the animal every session** (for example, the motor observatory):
`calibrate_from_points` on the pose keypoints → `estimate_scale` from a board video → `evaluate_calibration` as a
quality gate → `reconstruct_3d`.

**Reusing a calibration while it's still good** (for example, EvoVision):
`evaluate_calibration` on the last calibration. Only if it's no longer good: `associate_views` →
`calibrate_from_points` → `estimate_scale` → `reconstruct_3d`.

**Starting from a lab calibration and adjusting it** (for example, Emory):
`calibrate_from_points` with the lab calibration as the starting point → `estimate_scale` against the lab calibration
and known bone lengths → `evaluate_calibration` → `reconstruct_3d`.

**Outdoor cameras that get knocked about** (for example, the outdoor mouse project):
`calibrate_from_points` from VGGT and the tracked trajectory → `evaluate_calibration` with `times=` to catch moved
cameras → `reconstruct_3d`.

**Unsynced consumer cameras with many people** (for example, the dancing project):
`estimate_offsets` from the audio → `calibrate_from_board` for each board session, joined with `bridge_points` from
the dancers → `associate_views` → `calibrate_from_points` → `estimate_offsets(refine=True)` → `reconstruct_3d`.

## When you need more control

The seven functions are built from lower-level tools: camera models, board detection, triangulation, bundle
adjustment, quality reports, scale and alignment, and synchronisation. They're all still there (see the README for
the main ones, and the docstrings of each module). The workflow code in `aniposelib/workflows/` is a good place to see
how they fit together.
