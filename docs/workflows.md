# A guide to the aniposelib workflows

aniposelib has a lot of tools in it. Most projects, though, only ever do a handful of things: calibrate the cameras,
check the calibration, sometimes sync the cameras, and turn 2D keypoints into 3D. Each of those jobs has one function
that does the whole thing with sensible defaults.

This guide starts with the big picture: how the seven functions hand data to each other, and where each kind of
input comes from. Then it walks through the common routes from start to finish, and ends with a section on each
function.

- [The big picture](#the-big-picture)
- [Getting your data in](#getting-your-data-in)
- [Common routes, start to finish](#common-routes-start-to-finish)
- [Things every function has in common](#things-every-function-has-in-common)
- The seven functions:
  [calibrate_from_board](#calibrating-with-a-board-calibrate_from_board) ·
  [PointObject](#calibrating-with-an-object-of-known-shape-pointobject) ·
  [calibrate_from_points](#calibrating-from-the-animals-themselves-calibrate_from_points) ·
  [estimate_scale](#getting-real-world-units-estimate_scale) ·
  [evaluate_calibration](#checking-a-calibration-evaluate_calibration) ·
  [estimate_offsets](#syncing-cameras-estimate_offsets) ·
  [reconstruct_3d](#going-from-2d-to-3d-reconstruct_3d) ·
  [associate_views](#several-animals-at-once-associate_views)

## The big picture

Everything revolves around one object: the **calibration**, a `CameraGroup` that knows each camera's lens and where
it sits. Three functions make one, two improve or check one, and two use one:

```
                     videos of a board ──► calibrate_from_board ──┐
                                                                  │
   2D keypoints / people (no board) ─────► calibrate_from_points ─┤
                                                                  ▼
                                                            calibration
                                                                  │
            estimate_scale   (puts it in mm, lines it up with the floor or an old calibration)
            evaluate_calibration   (is it good? did a camera move?)
                                                                  │
                                                                  ▼
   2D keypoints ──────────────────────────────────────────────► reconstruct_3d ──► 3D tracks

   Two helpers sit in front, when you need them:
     estimate_offsets   cameras not started together      → time maps to line up the frames
     associate_views    several animals or people in view → one 2D track per individual
```

In words:

1. **If the cameras weren't synchronised**, sync them first with `estimate_offsets`, and line up your 2D data with the
   time maps it returns. Everything after this assumes that frame `i` is the same moment in every camera.
2. **Make a calibration.** With a board in the videos, use `calibrate_from_board`. Without one, use
   `calibrate_from_points` on the animals themselves, starting from lenses you already know.
3. **Give it real units if it needs them** with `estimate_scale`. A board calibration is already in the board's units;
   a calibration from points has no scale until you give it one.
4. **Check it** with `evaluate_calibration`, on the same kind of points you'll triangulate. This also tells you
   whether a camera got bumped during the session.
5. **Several animals?** Work out who is who across cameras with `associate_views`, then use its output wherever 2D
   points are expected.
6. **Turn 2D into 3D** with `reconstruct_3d`.

The functions don't read or write files for you (except videos, and `out=` to save a calibration). They take numpy
arrays and calibrations, and give back numpy arrays and calibrations, so you can stop, save and continue at any
step: `cgroup.dump('calibration.toml')` and `al.CameraGroup.load('calibration.toml')`.

## Getting your data in

### 2D keypoints

Most functions want 2D points as one array with **the camera first**: `(cameras, frames, keypoints, 2)`, in pixels,
with NaN wherever a camera didn't see a keypoint. A pose model usually gives you one table per camera, so stack them:

```python
cam_names = ['cam1', 'cam2', 'cam3']
points2d = np.stack([load_keypoints(name) for name in cam_names])   # each (frames, keypoints, 2)
scores   = np.stack([load_scores(name) for name in cam_names])      # each (frames, keypoints), optional
points2d[scores < 0.05] = np.nan                                    # only drop the hopeless ones
```

Two things to get right:

- **The same frame number must be the same moment in every camera.** If the cameras weren't synchronised, see
  [syncing](#syncing-cameras-estimate_offsets) below; the time maps it returns tell you which frame of each camera
  goes with which moment.
- **The cameras must be in the calibration's order** (`cgroup.get_names()`). If they aren't, reorder with
  `al.geometry.reindex_points(points2d, your_names, cgroup)`, or pass `options={'cam_names': your_names}` to
  `evaluate_calibration` and `reconstruct_3d`.

Keep the scores. `reconstruct_3d` uses them to trust confident detections more, so you don't need a strict cut-off.

### Board detections

You don't usually need to find the board yourself: `calibrate_from_board` and `estimate_scale` both take videos
directly. But board detections are also useful later (to check the scale, or the board's shape in 3D), and there are
three ways to get them:

```python
# 1. reuse the ones calibrate_from_board found: one list of detections per camera
cgroup, result = al.calibrate_from_board(videos, board)
board_detections = result.detections

# 2. find the board in a set of synchronised videos, one per camera
board_detections = al.boards.detect_videos(board, ['cam1.mp4', 'cam2.mp4', 'cam3.mp4'], max_frames=200)

# 3. let estimate_scale read the videos itself
result = al.estimate_scale(cgroup, board=board, board_points={'cam1': 'cam1.mp4', 'cam2': 'cam2.mp4'})
```

Any of these can be passed as `board_points=` to `estimate_scale` and `evaluate_calibration`. An array of corner
positions, `(cameras, frames, corners, 2)`, works too.

### Lenses (intrinsics)

`calibrate_from_points` needs to know each camera's lens, because the animals alone are a poor way to measure focal
length and distortion. Where to get them, from best to worst:

1. **A board calibration of the same cameras**, even an old one, even one where the board was only waved in front of
   each camera separately: `intrinsics=al.CameraGroup.load('lab_calibration.toml')`. If it also has good camera
   positions, it's used as the first guess of where the cameras are.
2. **A board calibration done just for the lenses**: film the board in front of each camera, one camera at a time,
   and run `calibrate_from_board` on those videos. Since no two cameras saw the board together it reports `'fail'`
   for the camera positions, but the lenses are still calibrated: use `intrinsics=result.lenses`.
3. **Nominal values** from the camera's spec sheet, or from VGGT (`init='vggt'` with `frames=`).
   `calibrate_from_points` can then fine-tune the focal length and main distortion term, when the data support it.

If all your cameras are the same model, say so with `options={'intrinsics_share': 'all'}`.

### Times

Several functions are better with the time of each frame in seconds (`times=`): `evaluate_calibration` uses them to
find moved cameras and bad hours, `calibrate_from_points` to drop bad hours and to handle moved cameras, and
`reconstruct_3d` to fill gaps by time. For a normal video that's just `np.arange(n_frames) / fps`. For a recording
that stops and starts, use the real timestamps, so gaps show up as gaps.

### Frames for VGGT

If `calibrate_from_points` has no good first guess of the camera positions, VGGT can make one from images. Pass
`frames=`: a few sets of synchronised video frames, one image per camera in each set (for example 5 to 10 moments
spread over the session).

## Common routes, start to finish

### A lab rig with a board

The classic Anipose setup: film the board, calibrate once, then triangulate every session.

```python
import aniposelib as al

board = al.CharucoBoard(7, 5, square_length=25, marker_length=18.75, units='mm')
cgroup, result = al.calibrate_from_board([['cam1.mp4'], ['cam2.mp4'], ['cam3.mp4']], board,
                                         cam_names=['cam1', 'cam2', 'cam3'], out='calibration.toml')

# later, for each session
cgroup = al.CameraGroup.load('calibration.toml')
check = al.evaluate_calibration(cgroup, points2d)            # still good for this session?
recon = al.reconstruct_3d(cgroup, points2d, scores=scores)
```

The calibration comes out in the board's units (mm here), so there's no need for `estimate_scale`.

### No board in the session: calibrate from the animal

The cameras get moved between sessions, so each session is calibrated from the animal's keypoints, starting from known
lenses. A board video is used for the scale.

```python
lenses = al.CameraGroup.load('lab_calibration.toml')         # lenses (and maybe rough positions)
cgroup, result = al.calibrate_from_points(points2d, intrinsics=lenses)

scaled = al.estimate_scale(cgroup, board=board,
                           board_points={'cam1': 'board_cam1.mp4', 'cam2': 'board_cam2.mp4'})
cgroup = scaled.cgroup                                       # now in mm

check = al.evaluate_calibration(cgroup, points2d_other_trial)  # a quality gate on data it wasn't fitted to
recon = al.reconstruct_3d(cgroup, points2d, scores=scores)
```

No board video? Use known distances instead, such as bone lengths:
`estimate_scale(cgroup, lengths=dict(points=points2d, pairs=bones, ref=bone_mm))`, or an older calibration of the
same cameras: `estimate_scale(cgroup, reference=old, align='reference')`.

### Several animals or people

When more than one individual is in view, match them across cameras first. The output of `associate_views` has an
extra axis for the individuals, `(cameras, frames, individuals, keypoints, 2)`.

```python
matched = al.associate_views(cgroup, detections, track_ids=track_ids)   # track_ids from your 2D tracker, optional
recon = al.reconstruct_3d(cgroup, matched.points)                        # (frames, individuals, keypoints, 3)
```

To calibrate from them, treat each individual in each frame as one set of keypoints:

```python
C, F, P, J, _ = matched.points.shape
cgroup, result = al.calibrate_from_points(matched.points.reshape(C, F * P, J, 2), intrinsics=cgroup)
```

Matching needs a calibration that's roughly right. If you don't have one yet, `calibrate_from_points` can do the
matching itself: pass the unmatched detections as `(cameras, frames, people, keypoints, 2)` with `init='vggt'` and
`frames=` (or `init='people'`, from the geometry of the camera pairs alone).

### Cameras that weren't synchronised

Phones and GoPros start whenever you press the button. Sync them from their soundtracks first, then put your 2D
points on one common clock:

```python
sync = al.estimate_offsets({'gopro1': 'gopro1.mp4', 'gopro2': 'gopro2.mp4', 'phone': 'phone.mov'})

from aniposelib.sync import align_streams
aligned = align_streams({name: (frame_times[name], keypoints[name]) for name in cam_names},
                        rate=30, offsets=sync.timemaps)
points2d = aligned.values             # (cameras, frames, keypoints, 2) on the common clock
times = aligned.t_grid                # the common clock, in seconds
```

`frame_times[name]` is each camera's own time of each frame (for a video, `np.arange(n) / fps`). From here on, the
other routes work as usual. Sound can only sync to a few milliseconds (sound travels about 3 ms per metre); once you
have a calibration, `estimate_offsets(..., cgroup=cgroup, seed=sync, refine=True)` refines the sync from the 2D
tracks themselves.

### Long sessions where cameras get bumped

Outdoors, or in a busy room, a camera can get knocked partway through. `evaluate_calibration` with the frame times
finds when and which camera moved, and `calibrate_from_points` then gives that camera one pose before and one after:

```python
check = al.evaluate_calibration(cgroup, points2d, times=times)
print(check.moves.moved, check.moves.change_points)           # which cameras moved, and when

cgroup2, result = al.calibrate_from_points(points2d, intrinsics=cgroup,
                                           stations=check.moves.stations, groups={'time': times})
rig_before = result.rig_at(100.0)                            # the calibration at t = 100 s
rig_after = result.rig_at(5000.0)
```

Triangulate each stretch between moves with the calibration for that stretch (`result.rig_at(t)` for a time inside
it), or pass the cameras' pose in every frame to `reconstruct_3d` as `extrinsics=`. `check.per_window` also points
out hours where the data broke (cameras out of sync, for example), which is worth knowing before you trust those
hours' 3D.

### Reusing a calibration while it's still good

If calibrating every session is expensive, check the last calibration on the new data first, and only recalibrate
when it no longer fits:

```python
check = al.evaluate_calibration(last_calibration, points2d)
worst = max(c['loo_median_px'] for c in check.per_camera.values())
if check.verdict == 'fail' or worst > 15:                    # your own rule
    cgroup, result = al.calibrate_from_points(points2d, intrinsics=last_calibration)
```

To compare two calibrations on the same data, use `options={'compare_to': {'old': other_calibration}}`.

---

## Things every function has in common

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

**Pixel thresholds and noisy keypoints.** Internally, the pixel limits start from values that suit clean detections
(about half a pixel of noise) and grow with the measured noise, but at most twofold, because pose keypoints often have
a few very wrong detections that make the noise look larger than it is. If you want the limits in plain pixels,
exactly as the lower-level functions use them, pass `options={'noise_unit': 1}` to `calibrate_from_points`,
`estimate_scale` or `evaluate_calibration` (`'auto'` is the default).

**Speed.** Functions with a `device=` argument (`calibrate_from_points`, `evaluate_calibration`,
`estimate_offsets`, `reconstruct_3d`) run their heavy steps on the GPU when there is one (`device='auto'`; pass
`device='cpu'` to stay on the CPU), and the results always come back as numpy. On the CPU each function uses at most 8 threads, because more
threads make the many small steps slower, especially on a shared machine; set `ANIPOSELIB_NUM_THREADS` (or
`OMP_NUM_THREADS`) to change that.

**Saving.** Calibration functions accept `out='calibration.toml'`, which saves the calibration and a report next to it
(`calibration.report.json`). Without `out`, nothing is written to disk.

---

## Calibrating with a board: `calibrate_from_board`

**In context:** the usual first step when you have a board. Its input is your board videos (or detections you
already have, see [board detections](#board-detections)). Its calibration goes straight to `reconstruct_3d`, and its
`result.detections` can be reused by `estimate_scale` and `evaluate_calibration`. Its `result.lenses` are the best
`intrinsics=` for `calibrate_from_points`, even when the cameras never saw the board together.


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
   The videos of all cameras are read together from start to end, and the board is looked for in every 20th frame
   (`options={'skip': 20}`). It stops once every camera has seen the board 200 times (`'max_boards'`; `None` reads
   everything). For a quick look at a long video, `options={'n_frames': 300}` uses 300 frames spread evenly over the
   whole video instead. `result.settings['detection']` says how much was read and why it stopped.
2. Works out each camera's lens (focal length and distortion). It tries a few lens models, from simple to complex,
   and keeps the simplest one that predicts board views it hasn't seen.
3. Works out where the cameras are relative to each other, from frames where two cameras see the board at once.
4. Refines everything together.

Use `fisheye=True` for fisheye lenses. If you already know the lenses, pass an earlier calibration as `intrinsics=` to
start from it; it may hold only some of the cameras, and the others are worked out from the board. If a camera's lens
comes out wrong (phones sometimes do), give a rough focal length in pixels as
`options={'focal_guess': {'phone1': 1700}}`: the lens is then kept within 25 % of it, and held near it while
everything is refined. If one camera's lens is so wrong that it pulls the other cameras out of place, the check
`camera_drag_deg` fails: it refines the rig again without that camera and sees how far the others move.

**If some cameras never saw the board at the same time as the others.** Imagine cameras on the left of the arena and
cameras on the right, and the board was only ever shown to one side at a time. Each side can be calibrated, but
nothing says where the left side is relative to the right side. In that case you get `'fail'`, plus a separate
calibration for each group in `result.component_groups`. To join them you have two options:

- record a few more board frames where both sides can see it; or
- pass `bridge_points=`, 2D tracks of anything both sides saw at the same time (the animal, a person walking through).
  aniposelib then works out how the groups sit relative to each other, keeping the lenses and the real-world scale
  from the board.

The checks: how far the board corners land from where the calibration predicts (compared with the noise in the
detections), whether one camera is much worse than the others, whether a camera ended up far away from the rest,
whether every camera saw the board, whether all cameras are connected, and whether the board comes out the right shape
in 3D. With `bridge_points=`, the board check uses the board alone, and the bridge must agree with the final
calibration (a group of cameras placed back to front fails).

## Calibrating with an object of known shape: `PointObject`

**In context:** a stand-in for the board in `calibrate_from_board`. Everything downstream works the same.


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

**In context:** the first step when there's no board in the session. It needs 2D keypoints
([getting them in](#2d-keypoints)) and lenses ([where to get them](#lenses-intrinsics)). The result has no real-world
scale, so it usually goes to `estimate_scale` next; then to `evaluate_calibration` and `reconstruct_3d`. With several
animals, feed it the output of `associate_views`.


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
   held-out data better (`refine_intrinsics='auto'`): at least 2 % better, on enough held-out points from enough time
   windows, and consistently so across those windows. `result.intrinsics['reason']` says why it was kept or not.
4. Optionally puts the result into real units (see `estimate_scale`, which you can pass through `scale=`).

The result has no real-world scale unless you give it one: the points alone can't tell a small rig from a big one.

The main check is a **leave-one-camera-out** error: each camera is hidden in turn, the points are rebuilt in 3D from
the other cameras, and then projected into the hidden camera to see how far off they land. This catches a camera in
the wrong place, which a plain reprojection error can miss. With only two cameras that check is impossible; you get a
`'warn'` saying so, and the plain two-camera error instead.

Some situations it handles for you:

- **A good starting calibration.** If the calibration you pass in already explains most of the points, the slower
  reconstruction from the points alone is skipped.
- **Speed of the reconstruction from the points alone.** Its random searches run on several threads and its 3D steps on
  `device`. The result depends only on `rng`, not on the number of threads. To get the slower pair search of 1.0
  back, pass `options={'sfm_kwargs': {'pair_method': 'ransac'}}`.
- **Lens refinement.** A refined lens is only kept if it predicts held-out data better on every count (the typical
  error, how many points agree, how many can be checked) and if the lens model still behaves well over the whole
  image. A camera whose refined lens doesn't is put back to the lens you gave.
- **Identical cameras.** `options={'intrinsics_share': 'all'}` makes all cameras share one focal length and
  distortion when lenses are refined (or give groups of camera names, e.g. `[['gopro1', 'gopro2'], ['phone1']]`).
- **People not yet matched across cameras.** Pass the detections as `(cameras, frames, people, keypoints, 2)` with
  `init='vggt'` and `frames=` (or a calibration as `init=`), plus your tracker's IDs as
  `options={'track_ids': ...}`. The people are matched across cameras with that first calibration
  (`associate_views`), then the calibration is refined from them. If nobody can be matched, you get `'fail'` with
  the reason: the first calibration was probably wrong. Without any first calibration, use `init='people'`: the
  people are matched across cameras from the geometry of each camera pair alone. A tracker usually cuts a person's
  track into several pieces; the pieces are joined as long as two of them are never seen in the same frame.
- **Noisy keypoints.** The reconstruction from the points alone measures the keypoint noise on its first camera
  pair and loosens its tests to match, so noisy detections (people at 1080p, a few pixels off) no longer leave
  cameras out. A camera that still fails gets a second try once the others are placed. To use the fixed 1.0 test,
  pass `options={'sfm_kwargs': {'noise_mult': None}}` (or `'people_kwargs'` with `init='people'`).
- **Cameras that were bumped.** Give `stations=` (for example from `evaluate_calibration(...).moves.stations`) and the
  frame times in `groups={'time': ...}`. Each position of a bumped camera gets its own pose. `result.rig_at(t)` gives
  you the calibration at time `t`.
- **Hours of broken data.** Give the frame times as `times=` (in seconds). The calibration is then checked hour by
  hour, or rather in about 20 time windows (the same check as in `evaluate_calibration`). Windows where several cameras
  suddenly stop agreeing, for example because the cameras went out of sync, are dropped, and the camera positions are
  fitted again without them. This only happens when the bad windows clearly stand out and hold at most half of the
  data. `result.report['bad_windows']` lists what was dropped, and why nothing was dropped otherwise.
  `options={'drop_bad_windows': False}` turns this off. It is not done with `stations=`.

Other options: `intrinsics_sigma_logf` can be given per camera (`{'phone1': 0.3, 'default': 0.05}`) for cameras whose
focal length is less certain, and `reject=None` turns off the removal of whole frames and keypoints that don't fit.

## Getting real-world units: `estimate_scale`

**In context:** after `calibrate_from_points` (or anything else that leaves the calibration in arbitrary units).
Its `result.cgroup` is the calibration to use from then on. The board detections can come from
`calibrate_from_board`, `al.boards.detect_videos`, or straight from videos ([board detections](#board-detections)).


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
With a board, only corners more than two squares apart are compared: on close corners, small detection errors make the
board look slightly too big. To use every pair as before 1.1, pass `options={'board_kwargs': {'min_pair_dist': None}}`.
`align='ground'` also turns the calibration so that the floor is flat at height zero.

With known lengths, the lengths are measured again on the rescaled calibration with a stricter cut-off. If they don't
come out right this time, you get a `'warn'` (`result.report['lengths_check']`).

To get the scale from bone lengths but the position and orientation of an older calibration, give both:

```python
result = al.estimate_scale(cgroup, lengths=dict(points=points2d, pairs=bones, ref=bone_mm),
                           reference=lab_calibration, align='reference')
```

The board can also be found directly in the videos: `board_points={'cam1': 'cam1.mp4', 'cam2': 'cam2.mp4', ...}`. The
videos must be in sync (frame 100 in one camera is frame 100 in all of them). As in `calibrate_from_board`, every 20th
frame is checked, and it stops once 200 frames show the board in at least two cameras (`options={'max_frames': 200}`).
`options={'n_frames': 60}` looks at only 60 evenly spaced frames instead.
The corners the learned (DeepCharuco) detector finds are then sharpened on the full-size frame, because it looks at
a shrunken copy of the frame to find the board. To turn that off, pass `options={'detector_kwargs': {'subpix': False}}`.

## Checking a calibration: `evaluate_calibration`

**In context:** after any calibration, before you trust its 3D; or at the start of a session, to decide whether an
old calibration can be reused. Give it the same kind of 2D points you'll triangulate, ideally from frames the
calibration wasn't fitted to. Its `result.moves.stations` goes to `calibrate_from_points(stations=...)`.


Use this to check a new calibration, to decide whether last week's calibration can still be used today, or to find
out whether a camera was bumped during a session.

```python
result = al.evaluate_calibration(cgroup, points2d)
result.per_camera      # errors and coverage for each camera
```

It reports, for each camera, the leave-one-camera-out error, how many of its points were usable, and whether its
lens model behaves well across the whole image. It also flags pairs of cameras that look at the scene from almost
the same direction, which triangulate poorly. Some inputs switch on extra checks:

- `times=`, the time of each frame: it looks for cameras that moved during the recording, and when. It also splits
  the recording into about 20 time windows (at most an hour each) and repeats the leave-one-camera-out check in each one
  (`result.per_window`). A window where a camera suddenly agrees much less than usual is flagged; this is usually
  a sync problem or broken data, not the calibration. Set the window length with `options={'loo_window_s': 3600}`.
  A camera that is off from the very start is a calibration error (the leave-one-camera-out checks report it), not
  a move, and a "move" no bigger than the jitter of the cameras that stayed put is ignored
  (`result.report['moves_demoted']`).
- With `times=` it also tries to re-place a camera that was bumped so hard that the others can't use it any more
  (its points stop agreeing from some time on, for good). It looks for the new aim that makes the camera agree with
  the points the other cameras see, and adds it as a new station in `result.moves`. For people tracked with
  `associate_views`, also pass the raw detections, `options={'detections': detections, 'frame_ids': ...}` (or the
  `(cameras, frames, individuals, keypoints, 2)` points directly): the bumped camera's detections are then matched
  to the people again and returned in `result.points`, ready for `calibrate_from_points(..., stations=
  result.moves.stations)`. Turn it off with `options={'recover_moves': False}`.
- `reference=`, another calibration: it compares the two. A big difference only gives a `'warn'`, because the
  reference is often the rougher of the two.
- a board or `PointObject` and its detections: it checks that the board comes out the right shape.
- `options={'compare_to': {'lab': lab_calibration}}`: compares how well each calibration predicts the same points,
  with a confidence interval (`result.report['compare_to']`). Give `options['clusters']` (for example a recording or
  block ID per frame) so the interval accounts for frames that aren't independent. Each entry says which one is
  `'better'` (`'this'`, `'other'` or `'tie'` when the interval includes zero). `delta` is the other calibration minus
  this one; `this_minus_other` is the opposite sign, the one Emory's `paired_compare` uses (candidate minus
  reference). `result.report['ranking']` lists all the calibrations, best first. When another calibration is clearly
  better (by at least 0.01), the verdict warns with `compare_to_better`.

To rank calibrations, look at `result.report['summary']['loo_agree@8px']`: the share of each camera's points that
land within 8 px of where the other cameras put them (leave-one-camera-out, triangulated with a 5 px cut-off, averaged
over cameras; higher is better). These standard pixel cut-offs (`loo_agree@5px`, `@8px`, `@10px`, also per camera)
are the same for every data set, so the numbers can be compared between calibrations and match the Emory
yardstick. The checks behind the verdict use cut-offs that grow with the noise in the points instead. `compare_to`
ranks by `agree@8px` too.

An `AffineCamera` (telecentric) group can be checked directly: it is converted to equivalent long-focal pinhole
cameras, exactly as in `reconstruct_3d`.

If your points include cameras the calibration doesn't have, or are in another order, pass their camera names as
`options={'cam_names': [...]}`. Cameras that aren't in the calibration are dropped, and calibration cameras without
points are left out of the check; both are listed in `result.report`. To use your own pixel cut-offs for the
noise-scaled leave-one-camera-out check, pass `options={'tau_tri_px': 5, 'thresholds_px': (5, 8, 10)}`; for the
standard ones, `options={'yardstick_px': (5, 8, 10), 'yardstick_tau_px': 5}` (the defaults).

With only two cameras there's no leave-one-camera-out check; you get the plain two-camera error and a `'warn'`.

The function only reports how good the calibration is. The decision is up to you, for example a rule that a
calibration is reused only while every camera stays under some error.

## Syncing cameras: `estimate_offsets`

**In context:** the very first step, before anything else, when the cameras weren't started together. Its
`result.timemaps` go to `align_streams` to put your 2D data on one clock
([example](#cameras-that-werent-synchronised)). Once there's a calibration, it can be run again with `seed=` and
`refine=True` to sharpen the sync.


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

A few things that help:

- When you pass video files, the sound is lined up with the first video frame using the start times stored in
  the file. Long recordings (more than 5 minutes) are matched in pieces, so slightly fast or slow clocks don't
  blur the match. Each soundtrack is filtered once, and the camera pairs are compared in parallel: 7 cameras with an
  hour of sound each take about 15 s on 8 CPUs. To run the pairs one at a time (less memory), pass
  `options={'audio_threads': 1}`.
- Without a calibration, 2D tracks are still compared camera pair by camera pair (`options={'epipolar_F':
  'self'}` is what it does then; `'calibrated'` or `'both'` with a calibration).
- If one camera recorded several files, say which camera each file belongs to:
  `options={'stream_camera': {'P2-clip1': 'phone2', 'P2-clip2': 'phone2'}}`. The files then share one clock speed.
  If you also know where each file starts on the camera's own clock (GoPro chapters follow each other exactly), pass
  `options={'clip_start': {'P2-clip1': 0.0, 'P2-clip2': 820.0}}` and the files stay on one timeline.
  `'clip_start': 'container'` uses the times stored in the files, which are only good to a second.
- Sound takes about 3 ms per metre to reach a camera, and phones add their own audio delay, so the sound can't sync
  cameras better than that. Say how much you expect with `options={'audio_delay_sigma': 0.01}` (in seconds) to get
  honest uncertainties. The 2D-track refinement measures the real delay.
- The frame rate used for the checks comes from the 2D tracks, else from the video files. Set it with
  `options={'fps': 30}` if neither is right.
- To refine an earlier result without redoing the sound, pass it back: `estimate_offsets(streams, cgroup,
  seed=result, refine=True)`. A short window only shifts the clocks; their speed is changed only when there's a
  long enough stretch of tracks (`options['refine_rate']`).
- For a long session, refine in pieces: `refine_windows=30` (seconds), or a list of `(start, end)` times, also
  `(start, end, calibration)` when a camera moved. Each piece is fitted on its own and the time maps join them
  smoothly, so a fix found in one piece isn't stretched over the whole hour.

## Going from 2D to 3D: `reconstruct_3d`

**In context:** the last step. It takes a calibration (from any of the calibration functions, after `estimate_scale`
if you want real units) and 2D keypoints in the same camera order, or the output of `associate_views`.


Use this to turn 2D keypoints from several cameras into 3D tracks.

```python
result = al.reconstruct_3d(cgroup, points2d, scores=scores)
points3d = result.points3d       # (frames, keypoints, 3), NaN where there weren't enough cameras
```

For every point it picks the cameras that agree with each other and ignores ones that don't, so a single bad
detection doesn't drag the 3D point away. The cameras it keeps all count fully in the final position (a plain
least-squares fit).

`options={'inlier_refit': 'auto'}` gives the cameras that agree closely more say than the ones that are only roughly
right, when the errors are spread out more than clean noise would be. That lowers the reprojection error of the
cameras used, but on real data it did not make the 3D points more accurate. The test that matters is a camera that
was left out: on the motor observatory recordings, the default fit already predicts it better than the project's own
fit (median 4.98 vs 5.20 px), while the reprojection on the cameras used was 3.29 vs 3.06 px. On CMU Panoptic people
against ground truth, the option made the 3D error slightly worse (by about 0.05 cm). So it stays off by default.

If your pose model gives a confidence for each keypoint, pass it as `scores=`. Confident detections count more and
unsure ones count less, but nothing is thrown out just for having a low score. You don't need to pick a cutoff. If you
want one anyway, use `options={'min_score': 0.3}`.

Two optional extras:

- `bones=[(0, 1), (1, 2), ...]`: pairs of keypoints whose distance should stay the same (a bone). The 3D points are
  adjusted so those lengths stay steady.
- `smooth=`: smooths the tracks over time. Off by default.

Short gaps in the middle of a track (up to 10 frames, `options={'max_gap': ...}`) are filled in, but a track is never
extended before its first or after its last detection, and longer gaps stay empty. Pass `times=` (seconds) to fill
by time instead (`options={'max_gap_s': 0.5}`; by default a gap is filled only if it lasts at most 11 typical frame
periods, so a long pause in the recording is never bridged). If your frames aren't a continuous video, for example labelled frames
picked here and there, use `fill='none'`. If the recording is made of separate clips or windows, pass their ids as
`segments=`: nothing is filled, smoothed or joined across two clips.

Bones are kept steady only as far as the data say they are: a bone whose length really changes (a tail, a spine)
gets little weight, and if keeping bones steady would make the points fit the images worse, the triangulated points
are kept. So `bones=` is safe to pass, but it helps most for truly rigid pairs.

Other useful things:

- `result.point_error`: how well each 3D point fits its cameras, in pixels. `options={'max_point_err_px': 5}` drops
  points worse than that.
- Moving cameras: pass the cameras' position in every frame as `extrinsics=` (one 4x4 matrix per camera and frame).
  If the cameras moved and you didn't say so, the check `error_drift` notices that the errors change over the
  recording.
- Points outside the image are not used, and are counted separately (`result.out_of_image`), not as bad detections.
- `options={'min_joints_per_view': 10}` ignores a camera in a frame where it saw fewer than 10 keypoints;
  `options={'min_views': 3}` only keeps points seen by 3 cameras or more (fewer points, but each one checked).
- If the points are in a different camera order than the calibration, say so with `options={'cam_names': [...]}`.
- The output of `associate_views` (with an axis for each animal) can be passed in directly; each animal is
  rebuilt on its own.
- Telecentric (affine) cameras work too.
- Static false detections (a sync light, a reflection) that the cameras see for hours show up as 3D points that
  never move. With `times=` they are flagged in `result.static` (a point that stays within about 1 % of the rig size
  for 40 s or more, in 5 different hours or more); `options={'static_remove': True}` drops them. If you know where
  such a thing is in the image, drop it up front: `options={'mask_regions': {'cam1': [(x, y, radius)]}}`.
- Animals that walk on the ground: `options={'surface_flag': True}` fits the ground (a height map) to the points that
  3 or more cameras agree on. Points seen by only 2 cameras that float well above or below it (about 3 % of the rig
  size) are probably wrong matches; they are flagged in `result.off_surface`. `result.surface_residual` gives each
  point's height above the ground.

## Several animals at once: `associate_views`

**In context:** between a calibration and everything that uses 2D points. It needs a calibration that's roughly
right. Its `result.points` can go straight to `reconstruct_3d`, or, reshaped to one set of keypoints per individual
and frame, to `calibrate_from_points` ([example](#several-animals-or-people)).


Use this when more than one animal or person is in view, so you first have to work out which detection in each
camera belongs to which individual.

```python
result = al.associate_views(cgroup, detections)      # detections: per camera, per frame, all the instances found
result.points          # (cameras, frames, individuals, keypoints, 2), ready for reconstruct_3d
result.labels          # for each detection, which individual it was matched to (-1 if none)
```

If your 2D tracker already links each animal over time within a camera, pass those IDs as `track_ids=`. Whole tracks
are then matched across cameras, which is more reliable than matching frame by frame. Without them, each frame is
matched on its own and the matches are joined up over time. Individuals that show up for fewer than 5 frames are
dropped (they are usually ghosts: two wrong detections that happen to line up), and an individual can be out of
view for up to 30 frames (`options={'link_max_gap': ...}`, or `link_max_gap_s` with `fps`) and keep its identity.
If you know how many animals there are, say so: `options={'n_identities': 4}`.

Trackers sometimes swap two animals. When a track stops agreeing with the other cameras part way through, it is cut
there and the pieces are matched again, and identities that follow each other in time and space are joined up.

`result.frame_ids` says which frame each row of `result.points` is. When the frame numbers are large and far apart,
only the frames with detections are kept (`options={'frame_axis': 'dense'}` keeps every frame number).

## When you need more control

The seven functions are built from lower-level tools: camera models, board detection, triangulation, bundle
adjustment, quality reports, scale and alignment, and synchronisation. They're all still there (see the README for
the main ones, and the docstrings of each module). The workflow code in `aniposelib/workflows/` is a good place to see
how they fit together.
