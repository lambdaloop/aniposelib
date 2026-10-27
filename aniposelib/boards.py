import warnings
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, asdict

import cv2
import numpy as np
from tqdm import trange


def get_video_params_cap(cap):
    params = dict()
    params['width'] = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    params['height'] = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    params['nframes'] = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    params['fps'] = cap.get(cv2.CAP_PROP_FPS)
    return params


def get_video_params(fname):
    cap = cv2.VideoCapture(fname)
    if not cap.isOpened():
        raise FileNotFoundError(f'missing file "{fname}"')
    params = get_video_params_cap(cap)
    cap.release()
    return params


def pnp_inputs(camera, points):
    """Massage (points, K, D, pix_scale) into what the pinhole cv2 PnP solvers expect.

    A fisheye camera's coefficients are k1..k4 of the equidistant model, which
    cv2.solvePnP and friends would read as the pinhole k1, k2, p1, p2. So
    undistort with the right model first and hand cv2 normalized points with
    identity intrinsics and no distortion instead.

    pix_scale converts a threshold in pixels to the units of the returned
    points, which is 1 for the pinhole path and 1/f for the normalized one.

    Pixels outside the invertible domain of the fisheye model come back as NaN;
    callers are expected to drop those rows.
    """
    K = camera.get_camera_matrix().cpu().detach().numpy()

    if getattr(camera, 'fisheye', False):
        undistorted = np.asarray(camera.undistort_points(np.asarray(points)))
        undistorted = undistorted.reshape(points.shape)
        focal = 0.5 * (K[0, 0] + K[1, 1])
        return undistorted, np.eye(3), np.zeros(5), 1.0 / focal

    return points, K, camera.get_distortions().cpu().detach().numpy(), 1.0


def fix_rvec(rvec, tvec):
    # https://github.com/opencv/opencv/issues/8813
    T = tvec.ravel()[0]
    R = cv2.Rodrigues(rvec)[0]

    # Unrelated -- makes Y the up axis, Z forward
    R = R @ np.array([
        [1, 0, 0],
        [0, 0, 1],
        [0, -1, 0],
    ])
    if 0 < R[1, 1] < 1:
        # If it gets here, the pose is flipped.

        # Flip the axes. E.g., Y axis becomes [-y0, -y1, y2].
        R *= np.array([
            [1, -1, 1],
            [1, -1, 1],
            [-1, 1, -1],
        ])

        # Fixup: rotate along the plane spanned by camera's forward (Z) axis and vector to marker's position
        forward = np.array([0, 0, 1])
        tnorm = T / np.linalg.norm(T)
        axis = np.cross(tnorm, forward)
        angle = -2 * np.arccos(tnorm @ forward)
        R = cv2.Rodrigues(angle * axis)[0] @ R

    return cv2.Rodrigues(R)[0]


def merge_rows(all_rows, cam_names=None):
    """Takes a list of rows returned from detect_images or detect_videos.
    Returns a merged version of the rows, wherein rows from different videos/images with same framenum are grouped.
    Optionally takes a list of cam_names, which determines what the keys are for each row.
    """

    assert cam_names is None or len(all_rows) == len(cam_names), \
        "number of rows does not match the number of camera names"

    if cam_names is None:
        cam_names = range(len(all_rows))

    rows_dict = defaultdict(dict)
    framenums = set()

    for cname, rows in zip(cam_names, all_rows):
        for r in rows:
            num = r['framenum']
            rows_dict[cname][num] = r
            framenums.add(num)

    framenums = sorted(framenums)
    merged = []

    for num in framenums:
        d = dict()
        for cname in cam_names:
            if num in rows_dict[cname]:
                d[cname] = rows_dict[cname][num]
        merged.append(d)

    return merged


def extract_points(merged,
                   board,
                   cam_names=None,
                   min_cameras=1,
                   min_points=4,
                   check_rtvecs=True):
    """Takes a list of merged rows (output of merge_rows) and a board object.
    Returns an array of object points and another array of image points, both of size CxNx2,
    where C is the number of cameras, N is the number of points.
    Optionally takes a list of cam_names, which determines what the keys are for each row. If cam_names are not given, then it is automatically determined from the rows, used in sorted order.
    It also takes a parameter min_cameras, which specifies how many cameras must see a point in order to keep it.
    """

    if cam_names is None:
        s = set.union(*[set(r.keys()) for r in merged])
        cam_names = sorted(s)

    test = board.get_empty_detection().reshape(-1, 2)
    n_cams = len(cam_names)
    n_points_per_detect = test.shape[0]
    n_detects = len(merged)

    objp_template = board.get_object_points().reshape(-1, 3)

    imgp = np.full((n_cams, n_detects, n_points_per_detect, 2),
                   np.nan, dtype='float64')

    rvecs = np.full((n_cams, n_detects, n_points_per_detect, 3),
                    np.nan, dtype='float64')

    tvecs = np.full((n_cams, n_detects, n_points_per_detect, 3),
                    np.nan, dtype='float64')

    objp = np.empty((n_detects, n_points_per_detect, 3),
                    dtype='float64')

    board_ids = np.empty((n_detects, n_points_per_detect),
                         dtype='int32')

    for rix, row in enumerate(merged):
        objp[rix] = np.copy(objp_template)
        board_ids[rix] = rix

        for cix, cname in enumerate(cam_names):
            if cname in row:
                filled = row[cname]['filled'].reshape(-1, 2)
                bad = np.any(np.isnan(filled), axis=1)
                num_good = np.sum(~bad)
                if num_good < min_points:
                    continue

                if row[cname].get('rvec', None) is None or \
                   row[cname].get('tvec', None) is None:
                    if check_rtvecs:
                        continue
                    else:
                        row[cname]['rvec'] = np.full(3, np.nan, dtype='float64')
                        row[cname]['tvec'] = np.full(3, np.nan, dtype='float64')
                
                imgp[cix, rix] = filled

                rvecs[cix, rix, ~bad] = row[cname]['rvec'].ravel()
                tvecs[cix, rix, ~bad] = row[cname]['tvec'].ravel()

    imgp = np.reshape(imgp, (n_cams, -1, 2))
    rvecs = np.reshape(rvecs, (n_cams, -1, 3))
    tvecs = np.reshape(tvecs, (n_cams, -1, 3))
    objp = np.reshape(objp, (-1, 3))
    board_ids = np.reshape(board_ids, (-1))

    num_good = np.sum(~np.isnan(imgp), axis=0)[:, 0]
    good = num_good >= min_cameras

    imgp = imgp[:, good]
    rvecs = rvecs[:, good]
    tvecs = tvecs[:, good]
    objp = objp[good]
    board_ids = board_ids[good]

    extra = {
        'objp': objp,
        'ids': board_ids,
        'rvecs': rvecs,
        'tvecs': tvecs
    }

    return imgp, extra


def extract_rtvecs(merged,
                   cam_names=None,
                   min_cameras=1,
                   board=None,
                   cameras=None):
    """Takes a list of merged rows (output of merge_rows) and a board object.
    Returns an array of rvecs and tvecs appended together, of size CxNx6,
    where C is the number of cameras, N is the number of detections.
    Optionally takes a list of cam_names, which determines what the keys are for each row. If cam_names are not given, then it is automatically determined from the rows, used in sorted order.
    It also takes a parameter min_cameras, which specifies how many cameras must see a point in order to keep it.

    board.estimate_pose_rows should have been run on the rows before merging.
    If not, the board and cameras must be passed as arguments.
    """

    if cam_names is None:
        s = set.union(*[set(r.keys()) for r in merged])
        cam_names = sorted(s)

    n_cams = len(cam_names)
    n_detects = len(merged)

    rtvecs = np.empty((n_cams, n_detects, 6), dtype='float64')
    rtvecs[:] = np.nan

    for rix, row in enumerate(merged):
        for cix, cname in enumerate(cam_names):
            if cname in row:
                r = row[cname]
                if 'rvec' not in r or 'tvec' not in r:
                    if board is None:
                        raise ValueError(
                            'rvec or tvec not found in rows. '
                            'board.estimate_pose_rows should have been run on '
                            'the rows before merging.'
                            'If not, the board and cameras must be passed as arguments.'
                        )
                    else:
                        rvec, tvec = board.estimate_pose_points(
                            cameras[cix], r['corners'], r['ids'])
                        r['rvec'] = rvec
                        r['tvec'] = tvec

                if r['rvec'] is None or r['tvec'] is None:
                    continue

                rvec = r['rvec'].ravel()
                tvec = r['tvec'].ravel()

                rtvec = np.hstack([rvec, tvec])
                rtvecs[cix, rix] = rtvec

    num_good = np.sum(~np.isnan(rtvecs), axis=0)[:, 0]
    rtvecs = rtvecs[:, num_good >= min_cameras]

    return rtvecs



def empty_corners():
    """An empty detection, shaped like a non-empty one: corners (0, 1, 2) float32, ids (0, 1) int32."""
    return np.zeros((0, 1, 2), dtype='float32'), np.zeros((0, 1), dtype='int32')


def _to_gray(image):
    if image.ndim == 3:
        if image.shape[2] == 4:
            return cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image


def draw_corners(image, corners, ids=None, color=(0, 0, 255), radius=None):
    """Draw detected corners (and their ids) on a copy of ``image`` (BGR or gray).

    A local replacement for OpenCV's ``drawDetectedCornersCharuco`` (contrib-only on
    some OpenCV builds). Returns the annotated BGR image."""
    out = image.copy()
    if out.ndim == 2:
        out = cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)
    if corners is None or len(corners) == 0:
        return out
    pts = np.asarray(corners, dtype='float64').reshape(-1, 2)
    if radius is None:
        radius = max(2, int(round(min(out.shape[:2]) / 200)))
    ids_flat = None if ids is None else np.asarray(ids).ravel()
    for n, (x, y) in enumerate(pts):
        if not (np.isfinite(x) and np.isfinite(y)):
            continue
        c = (int(round(x)), int(round(y)))
        cv2.circle(out, c, radius, color, -1, cv2.LINE_AA)
        if ids_flat is not None:
            cv2.putText(out, str(int(ids_flat[n])), (c[0] + radius + 1, c[1] - radius - 1),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)
    return out


@dataclass
class DetectionCheck:
    """Result of :meth:`CharucoBoard.validate_detection` (truthy when the detection passes).

    ``max_residual_px`` / ``median_residual_px``: residuals of a least-squares homography from
    the board plane to the image fitted on *all* corners; ``inlier_fraction``: fraction of
    corners within ``max_homography_px`` of a RANSAC homography. Wrong ids (a transposed
    board, wrong ``legacy_pattern``, a confused learned detector) cannot be fitted by any
    homography, so they show up as a large residual and a low inlier fraction."""
    ok: bool
    n_corners: int
    max_residual_px: float
    median_residual_px: float
    inlier_fraction: float
    reason: str = ''

    def __bool__(self):
        return bool(self.ok)

    def to_dict(self):
        return asdict(self)


def homography_check(objp, corners, ids, max_homography_px=3.0, min_corners=6):
    """Homography gate for one planar-board detection (see :class:`DetectionCheck`).

    ``objp`` is the (K, 2|3) board layout indexed by corner id."""
    nan = float('nan')
    if corners is None or ids is None or len(corners) == 0:
        return DetectionCheck(False, 0, nan, nan, 0.0, 'no corners')
    objp = np.asarray(objp, dtype='float64').reshape(len(objp), -1)[:, :2]
    pts = np.asarray(corners, dtype='float64').reshape(-1, 2)
    ids = np.asarray(ids).astype('int64').ravel()
    n = len(ids)
    if len(pts) != n:
        return DetectionCheck(False, n, nan, nan, 0.0, 'corners and ids differ in length')
    good = np.all(np.isfinite(pts), axis=1)
    pts, ids = pts[good], ids[good]
    n = len(ids)
    if n and (ids.min() < 0 or ids.max() >= len(objp)):
        return DetectionCheck(False, n, nan, nan, 0.0, 'corner id out of range for this board')
    if len(np.unique(ids)) != n:
        return DetectionCheck(False, n, nan, nan, 0.0, 'duplicate corner ids')
    if n < max(int(min_corners), 4):
        return DetectionCheck(False, n, nan, nan, 0.0,
                              'fewer than {} corners'.format(max(int(min_corners), 4)))
    src = objp[ids]
    try:
        H, _ = cv2.findHomography(src, pts, 0)
    except cv2.error:
        H = None
    if H is None:
        return DetectionCheck(False, n, float('inf'), float('inf'), 0.0, 'homography fit failed')
    proj = cv2.perspectiveTransform(src.reshape(-1, 1, 2), H).reshape(-1, 2)
    res = np.linalg.norm(proj - pts, axis=1)
    try:
        Hr, mask = cv2.findHomography(src, pts, cv2.RANSAC, float(max_homography_px))
    except cv2.error:
        Hr, mask = None, None
    if Hr is None or mask is None:
        frac = float(np.mean(res <= max_homography_px))
    else:
        projr = cv2.perspectiveTransform(src.reshape(-1, 1, 2), Hr).reshape(-1, 2)
        frac = float(np.mean(np.linalg.norm(projr - pts, axis=1) <= max_homography_px))
    mx = float(np.max(res))
    ok = mx <= max_homography_px
    reason = '' if ok else 'homography residual {:.2f} px > {:.2f} px'.format(mx, max_homography_px)
    return DetectionCheck(bool(ok), n, mx, float(np.median(res)), frac, reason)


def _warn_if_bad_layout(board, rows, min_checked=5):
    """Warn when most detections do not fit the board layout (e.g. squaresX/squaresY swapped)."""
    if not hasattr(board, 'validate_detection'):
        return
    fracs = []
    for r in rows:
        ids = r.get('ids')
        if ids is None or len(ids) < 6:
            continue
        pts = np.asarray(r['corners'], dtype='float64').reshape(-1, 2)
        pts = pts[np.all(np.isfinite(pts), axis=1)]
        if len(pts) < 6:
            continue
        # tolerant threshold (a fraction of the corner spacing), so that lens distortion
        # does not trigger it but wrong ids (whole squares off) do
        span = np.ptp(pts, axis=0)
        spacing = np.sqrt(max(span[0] * span[1], 1.0) / len(pts))
        chk = board.validate_detection(r['corners'], ids,
                                       max_homography_px=max(3.0, 0.3 * spacing))
        fracs.append(chk.inlier_fraction)
    if len(fracs) >= min_checked and np.median(fracs) < 0.6:
        warnings.warn(
            'most board detections do not fit the board layout (median homography inlier '
            'fraction {:.2f} over {} detections). Check the board definition: squaresX is the '
            'number of squares across (columns) and squaresY down (rows) -- use '
            'CharucoBoard.from_rows_cols(rows, cols, ...) to be explicit -- and legacy_pattern '
            '(boards made by calib.io or OpenCV < 4.6 with an even number of rows need '
            'legacy_pattern=True).'.format(float(np.median(fracs)), len(fracs)),
            RuntimeWarning, stacklevel=3)



def _auto_margin(margin, width, height, nx, ny, frac=0.5):
    """Margin in pixels; 'auto' = `frac` of a square, with the square size chosen so that
    board + margin on both sides fits in width x height."""
    if margin is None or (isinstance(margin, str) and margin == 'auto'):
        p = min(width / (nx + 2 * frac), height / (ny + 2 * frac))
        return max(int(round(frac * p)), 1)
    margin = int(margin)
    if margin < 0:
        raise ValueError('margin must be >= 0 or "auto", got {}'.format(margin))
    return margin


def _no_manual_verify(flag):
    if flag:
        raise ValueError('manually_verify was removed in aniposelib 1.1 (the cv2.imshow popup); '
                         'inspect detections with board.draw_corners / your own viewer instead')

class CalibrationObject(ABC):
    @abstractmethod
    def draw(self, size):
        pass

    @abstractmethod
    def detect_image(self, image):
        pass

    @abstractmethod
    def get_object_points(self):
        pass

    @abstractmethod
    def estimate_pose_points(self, camera, corners, ids):
        pass

    @abstractmethod
    def fill_points(self, corners, ids):
        pass

    @abstractmethod
    def get_empty_detection(self):
        pass

    def estimate_pose_image(self, camera, image):
        corners, ids = self.detect_image(image)
        return self.estimate_pose_points(camera, corners, ids)

    def detect_frames(self, frames):
        """Detect the board in a list of decoded frames (BGR or gray arrays).

        Returns a list of ``(corners, ids)``, one per frame. Batched detectors
        (``detector='deepcharuco'``) run whole batches at once."""
        return [self.detect_image(f) for f in frames]

    def detect_images(self, images, progress=False, prefix=None):
        """Detect the board in a list of image filenames (or decoded image arrays).

        Returns rows (dicts with ``framenum``, ``corners``, ``ids``, ``filled``, and
        ``fname`` for filenames)."""
        length = len(images)
        rows = []

        if progress:
            it = trange(length, ncols=70)
        else:
            it = range(length)

        for framenum in it:
            imname = images[framenum]
            if isinstance(imname, np.ndarray):
                frame = imname
                imname = None
            else:
                frame = cv2.imread(imname)
                if frame is None:
                    raise FileNotFoundError(f'could not read image "{imname}"')

            corners, ids = self.detect_image(frame)

            if corners is not None:

                if prefix is None:
                    key = framenum
                else:
                    key = (prefix, framenum)

                row = {
                    'framenum': key,
                    'corners': corners,
                    'ids': ids,
                }
                if imname is not None:
                    row['fname'] = imname

                rows.append(row)

        rows = self.fill_points_rows(rows)
        _warn_if_bad_layout(self, rows)

        return rows

    def detect_video(self, vidname, prefix=None, skip=20, progress=False,
                     frame_indices=None):
        """Detect the board in a video. Returns rows (see ``detect_images``).

        By default every ``skip``-th frame is checked, densifying to every frame for
        ``skip/2`` frames after each detection. Pass ``frame_indices`` to detect on
        exactly those frames instead (decoded in order; batched for batched detectors)."""
        cap = cv2.VideoCapture(vidname)
        if not cap.isOpened():
            raise FileNotFoundError(f'missing video file "{vidname}"')
        length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if frame_indices is not None:
            rows = self._detect_video_indices(cap, frame_indices, prefix, progress)
            cap.release()
            rows = self.fill_points_rows(rows)
            _warn_if_bad_layout(self, rows)
            return rows

        if length < 10:
            length = int(1e9)
            progress = False
        rows = []

        go = int(skip / 2)

        if progress:
            it = trange(length, ncols=70)
        else:
            it = range(length)

        for framenum in it:
            ret, frame = cap.read()
            if not ret:
                break
            if framenum % skip != 0 and go <= 0:
                continue

            corners, ids = self.detect_image(frame)

            if corners is not None and len(corners) > 0:
                if prefix is None:
                    key = framenum
                else:
                    key = (prefix, framenum)
                go = int(skip / 2)
                row = {'framenum': key, 'corners': corners, 'ids': ids}
                rows.append(row)

            go = max(0, go - 1)

        cap.release()

        rows = self.fill_points_rows(rows)
        _warn_if_bad_layout(self, rows)

        return rows

    def _detect_video_indices(self, cap, frame_indices, prefix, progress):
        wanted = sorted(set(int(i) for i in frame_indices))
        batch = max(1, int(getattr(self, 'batch_size', 1) or 1))
        rows, buf, nums = [], [], []

        def flush():
            for num, (corners, ids) in zip(nums, self.detect_frames(buf)):
                if corners is not None and len(corners) > 0:
                    key = num if prefix is None else (prefix, num)
                    rows.append({'framenum': key, 'corners': corners, 'ids': ids})
            buf.clear()
            nums.clear()

        it = trange(len(wanted), ncols=70) if progress else range(len(wanted))
        pos = 0
        for n in it:
            target = wanted[n]
            if target - pos > 30:
                cap.set(cv2.CAP_PROP_POS_FRAMES, target)
                pos = target
            while pos < target:
                if not cap.grab():
                    break
                pos += 1
            ret, frame = cap.read()
            pos += 1
            if not ret:
                break
            buf.append(frame)
            nums.append(target)
            if len(buf) >= batch:
                flush()
        if buf:
            flush()
        return rows

    def estimate_pose_rows(self, camera, rows):
        for row in rows:
            rvec, tvec = self.estimate_pose_points(camera,
                                                   row['corners'],
                                                   row['ids'])
            row['rvec'] = rvec
            row['tvec'] = tvec
        return rows

    def fill_points_rows(self, rows):
        for row in rows:
            row['filled'] = self.fill_points(row['corners'], row['ids'])
        return rows

    def get_all_calibration_points(self, rows, min_points=5):
        rows = self.fill_points_rows(rows)

        objpoints = self.get_object_points()
        objpoints = objpoints.reshape(-1, 3)

        all_obj = []
        all_img = []

        for row in rows:
            filled_test = row['filled'].reshape(-1, 2)
            good = np.all(~np.isnan(filled_test), axis=1)
            filled_app = row['filled'].reshape(-1, 2)
            objp = np.copy(objpoints)
            if np.sum(good) >= min_points:
                all_obj.append(np.float32(objp[good]))
                all_img.append(np.float32(filled_app[good]))

        return all_obj, all_img


class Checkerboard(CalibrationObject):
    DETECT_PARAMS = \
        cv2.CALIB_CB_NORMALIZE_IMAGE + \
        cv2.CALIB_CB_ADAPTIVE_THRESH + \
        cv2.CALIB_CB_FAST_CHECK

    SUBPIX_CRITERIA = (cv2.TERM_CRITERIA_EPS +
                       cv2.TERM_CRITERIA_MAX_ITER,
                       30, 0.01)

    def __init__(self, squaresX, squaresY, square_length=1, manually_verify=False, *, units=None):
        self.squaresX = squaresX
        self.squaresY = squaresY
        self.square_length = square_length
        self.units = units
        _no_manual_verify(manually_verify)

        total_size = squaresX * squaresY

        objp = np.zeros((total_size, 3), np.float64)
        objp[:, :2] = np.mgrid[0:squaresX, 0:squaresY].T.reshape(-1, 2)
        objp *= square_length
        self.objPoints = objp

        self.ids = np.arange(total_size)

        self.empty_detection = np.zeros((total_size, 1, 2)) * np.nan

    def get_size(self):
        size = (self.squaresX, self.squaresY)
        return size

    def get_empty_detection(self):
        return np.copy(self.empty_detection)

    def get_square_length(self):
        return self.square_length

    def draw(self, size, margin='auto'):
        """Render the checkerboard as a uint8 image of ``size = (width, height)`` pixels.

        The board has ``(squaresX + 1) x (squaresY + 1)`` squares (squaresX/Y count inner
        corners), centred on a white background, top-left square black. margin: minimum white
        border in pixels; ``'auto'`` (default) = half a square, ``0`` = as large as fits."""
        width, height = int(size[0]), int(size[1])
        nx, ny = self.squaresX + 1, self.squaresY + 1
        margin = _auto_margin(margin, width, height, nx, ny)
        p = int(min((width - 2 * margin) // nx, (height - 2 * margin) // ny))
        if p <= 0:
            raise ValueError('size {} too small for a {}x{} checkerboard'.format(size, nx, ny))
        img = np.full((height, width), 255, dtype='uint8')
        x0, y0 = (width - p * nx) // 2, (height - p * ny) // 2
        for j in range(ny):
            for i in range(nx):
                if (i + j) % 2 == 0:
                    img[y0 + j * p:y0 + (j + 1) * p, x0 + i * p:x0 + (i + 1) * p] = 0
        return img

    def get_empty(self):
        return np.copy(self.empty_detection)

    def fill_points(self, corners, ids=None):
        out = self.get_empty_detection()
        if corners is None or len(corners) == 0:
            return out
        if ids is None:
            return corners
        else:
            ids = ids.ravel()
            for i, cxs in zip(ids, corners):
                out[i] = cxs
            return out

    def detect_image(self, image, subpix=True):

        gray = _to_gray(image)

        size = self.get_size()
        pattern_was_found, corners = cv2.findChessboardCorners(gray, size, self.DETECT_PARAMS)

        if corners is not None:

            if subpix:
                corners = cv2.cornerSubPix(gray, corners, (3, 3), (-1, -1), self.SUBPIX_CRITERIA)

        if corners is None:
            ids = None
        else:
            ids = self.ids

        return corners, ids

    def get_object_points(self):
        return self.objPoints

    def estimate_pose_points(self, camera, points, ids=None):
        if points is None:
            return None, None
        ngood = np.sum(~np.isnan(points)) // 2
        if ngood < 7:
            return None, None

        n_points = points.size // 2
        points = np.reshape(points, (n_points, 1, 2))

        obj_points = self.get_object_points()

        if points.shape[0] != obj_points.shape[0]:
            return None, None

        points, K, D, pix_scale = pnp_inputs(camera, points)

        # a fisheye undistort returns NaN outside its invertible domain
        good = np.all(np.isfinite(points), axis=(1, 2))
        if np.sum(good) < 7:
            return None, None
        points = points[good]
        obj_points = obj_points[good]

        try:
            retval, rvec, tvec, inliers = cv2.solvePnPRansac(obj_points,
                                                             points,
                                                             K,
                                                             D,
                                                             confidence=0.9,
                                                             reprojectionError=30 * pix_scale)
        except cv2.error as e:
            warnings.warn("failed to find checkerboard pose in image: {}".format(e))
            return None, None
        if not retval:
            return None, None
        return rvec, tvec



ARUCO_DICTS = {
    (4, 50): cv2.aruco.DICT_4X4_50,
    (5, 50): cv2.aruco.DICT_5X5_50,
    (6, 50): cv2.aruco.DICT_6X6_50,
    (7, 50): cv2.aruco.DICT_7X7_50,
    (4, 100): cv2.aruco.DICT_4X4_100,
    (5, 100): cv2.aruco.DICT_5X5_100,
    (6, 100): cv2.aruco.DICT_6X6_100,
    (7, 100): cv2.aruco.DICT_7X7_100,
    (4, 250): cv2.aruco.DICT_4X4_250,
    (5, 250): cv2.aruco.DICT_5X5_250,
    (6, 250): cv2.aruco.DICT_6X6_250,
    (7, 250): cv2.aruco.DICT_7X7_250,
    (4, 1000): cv2.aruco.DICT_4X4_1000,
    (5, 1000): cv2.aruco.DICT_5X5_1000,
    (6, 1000): cv2.aruco.DICT_6X6_1000,
    (7, 1000): cv2.aruco.DICT_7X7_1000
}


def get_aruco_dictionary(aruco_dict=None, marker_bits=4, dict_size=50):
    """Resolve an ArUco dictionary.

    ``aruco_dict`` may be a ``cv2.aruco.Dictionary``, a predefined-dictionary constant
    (e.g. ``cv2.aruco.DICT_APRILTAG_36h11``) or its name (``'DICT_4X4_50'``, ``'4X4_50'``,
    ``'APRILTAG_36h11'``). If it is None, ``(marker_bits, dict_size)`` picks one of the 16
    ``DICT_{bits}X{bits}_{size}`` dictionaries."""
    if aruco_dict is None:
        key = (int(marker_bits), int(dict_size))
        if key not in ARUCO_DICTS:
            raise ValueError(
                'unsupported (marker_bits, dict_size) = {}; supported pairs are {} '
                '(or pass aruco_dict=)'.format(key, sorted(ARUCO_DICTS)))
        return cv2.aruco.getPredefinedDictionary(ARUCO_DICTS[key])
    if isinstance(aruco_dict, cv2.aruco.Dictionary):
        return aruco_dict
    if isinstance(aruco_dict, str):
        name = aruco_dict.strip()
        if not name.upper().startswith('DICT_'):
            name = 'DICT_' + name
        const = getattr(cv2.aruco, name, None)
        if const is None:
            const = getattr(cv2.aruco, name.upper(), None)
        if const is None or not isinstance(const, int):
            valid = sorted(n for n in dir(cv2.aruco) if n.startswith('DICT_'))
            raise ValueError('unknown ArUco dictionary {!r}; valid names: {}'.format(aruco_dict, valid))
        return cv2.aruco.getPredefinedDictionary(const)
    if isinstance(aruco_dict, (int, np.integer)):
        return cv2.aruco.getPredefinedDictionary(int(aruco_dict))
    raise TypeError('aruco_dict must be a cv2.aruco.Dictionary, an int constant or a name, '
                    'not {}'.format(type(aruco_dict).__name__))


def default_detector_params():
    """The tuned ``cv2.aruco.DetectorParameters`` used by :class:`CharucoBoard`."""
    params = cv2.aruco.DetectorParameters()
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_CONTOUR
    # 3 is OpenCV's default; 0.9.x used 50, which misses small/distant markers
    params.adaptiveThreshWinSizeMin = 3
    params.adaptiveThreshWinSizeMax = 700
    params.adaptiveThreshWinSizeStep = 50
    params.adaptiveThreshConstant = 0
    return params


class CharucoBoard(CalibrationObject):
    """A ChArUco board.

    Parameters
    ----------
    squaresX, squaresY : int
        Number of squares **across** (columns) and **down** (rows) the board, in the order
        ``cv2.aruco.CharucoBoard`` takes them. A board described as "R rows x C columns" is
        ``CharucoBoard(C, R, ...)``; use :meth:`from_rows_cols` to avoid transposing it.
    square_length, marker_length : float
        Side lengths, in the units the calibration should come out in.
    marker_bits, dict_size : int
        Pick the predefined ``DICT_{bits}X{bits}_{size}`` dictionary (ignored if
        ``aruco_dict`` is given).
    aruco_dict : cv2.aruco.Dictionary, int or str, optional
        Any ArUco/AprilTag dictionary (see :func:`get_aruco_dictionary`).
    manually_verify : bool
        Removed in 1.1 (kept in the signature for positional compatibility; ``True`` raises).
    legacy_pattern : bool
        Use the checkerboard phase of OpenCV < 4.6 / calib.io boards. It only matters for an
        even number of rows (``squaresY``); with the wrong setting, few corners are found.
    detector : 'opencv', 'deepcharuco' or a detector instance
        How :meth:`detect_image` finds corners. ``'deepcharuco'`` builds a
        :class:`aniposelib.detectors.DeepCharucoDetector` (``detector_kwargs`` are passed on);
        an instance needs ``detect(image) -> (corners, ids)`` (and optionally ``detect_batch``).
    detector_params : cv2.aruco.DetectorParameters or dict, optional
        OpenCV marker-detection parameters (a dict updates :func:`default_detector_params`).
    validate : bool, optional
        Drop detections that fail :meth:`validate_detection`. Default: on for learned
        detectors, off for ``'opencv'`` (compatibility).
    max_homography_px : float
        Threshold of the validation gate.
    units : str, optional
        Unit of ``square_length``/``marker_length`` (e.g. ``'mm'``, ``'m'``). Informational:
        :func:`aniposelib.scale.from_board` writes it to ``cgroup.metadata['units']``.

    Object points
    -------------
    :meth:`get_object_points` puts the **first inner corner at the origin**: corner ``id``
    sits at ``((id % (squaresX-1)) * square_length, (id // (squaresX-1)) * square_length, 0)``.
    Ids are row-major with x fastest, the same ids ``cv2.aruco.CharucoDetector`` returns.
    OpenCV's own ``board.getChessboardCorners()`` is this plus ``(square_length,
    square_length, 0)`` (its origin is the outer corner of the board); pass
    ``origin='opencv'`` to get those. Board poses (:meth:`estimate_pose_points`) are in the
    zero-origin frame by default, consistent with ``extract_points``.
    """

    def __init__(self,
                 squaresX,
                 squaresY,
                 square_length,
                 marker_length,
                 marker_bits=4,
                 dict_size=50,
                 aruco_dict=None,
                 manually_verify=False,
                 *,
                 legacy_pattern=False,
                 detector='opencv',
                 detector_params=None,
                 detector_kwargs=None,
                 validate=None,
                 max_homography_px=3.0,
                 units=None):
        self.units = units
        self.squaresX = int(squaresX)
        self.squaresY = int(squaresY)
        self.square_length = square_length
        self.marker_length = marker_length
        _no_manual_verify(manually_verify)
        self.marker_bits = marker_bits
        self.dict_size = dict_size
        self.aruco_dict = aruco_dict
        self.legacy_pattern = bool(legacy_pattern)

        self.dictionary = get_aruco_dictionary(aruco_dict, marker_bits, dict_size)

        self.board = cv2.aruco.CharucoBoard((self.squaresX, self.squaresY),
                                            square_length, marker_length,
                                            self.dictionary)
        self.board.setLegacyPattern(self.legacy_pattern)

        if detector_params is None:
            params = default_detector_params()
        elif isinstance(detector_params, dict):
            params = default_detector_params()
            for key, value in detector_params.items():
                if not hasattr(params, key):
                    raise ValueError('unknown cv2.aruco.DetectorParameters field {!r}'.format(key))
                setattr(params, key, value)
        else:
            params = detector_params
        self.detector_params = params
        self._aruco_detector = cv2.aruco.ArucoDetector(self.dictionary, params)
        self._charuco_detector = cv2.aruco.CharucoDetector(self.board)

        self.detector_kwargs = dict(detector_kwargs or {})
        if isinstance(detector, str):
            if detector not in ('opencv', 'deepcharuco'):
                raise ValueError("detector must be 'opencv', 'deepcharuco' or a detector instance")
            self.detector_name = detector
            self._detector = None
        else:
            if not hasattr(detector, 'detect'):
                raise TypeError('a detector instance needs a detect(image) method')
            self.detector_name = type(detector).__name__
            self._detector = detector
        if validate is None:
            validate = self.detector_name != 'opencv'
        self.validate = bool(validate)
        self.max_homography_px = float(max_homography_px)

        total_size = (self.squaresX - 1) * (self.squaresY - 1)

        objp = np.zeros((total_size, 3), np.float64)
        objp[:, :2] = np.mgrid[0:(self.squaresX - 1), 0:(self.squaresY - 1)].T.reshape(
            -1, 2)
        objp *= square_length
        self.objPoints = objp

        self.empty_detection = np.zeros((total_size, 1, 2)) * np.nan
        self.total_size = total_size

    @classmethod
    def from_rows_cols(cls, rows, cols, square_length, marker_length, **kwargs):
        """Build a board from its number of square **rows** (down) and **columns** (across).

        Equivalent to ``CharucoBoard(squaresX=cols, squaresY=rows, ...)``."""
        return cls(cols, rows, square_length, marker_length, **kwargs)

    @property
    def rows(self):
        """Number of square rows (= squaresY)."""
        return self.squaresY

    @property
    def cols(self):
        """Number of square columns (= squaresX)."""
        return self.squaresX

    def __repr__(self):
        return ('CharucoBoard(squaresX={}, squaresY={}, square_length={}, marker_length={}, '
                'legacy_pattern={}, detector={!r}{})'.format(
                    self.squaresX, self.squaresY, self.square_length, self.marker_length,
                    self.legacy_pattern, self.detector_name,
                    '' if getattr(self, 'units', None) is None else ', units={!r}'.format(self.units)))

    def __getstate__(self):
        # cv2 objects do not pickle; rebuild them on unpickling
        state = self.__dict__.copy()
        for key in ('dictionary', 'board', '_aruco_detector', '_charuco_detector',
                    'detector_params'):
            state.pop(key, None)
        if isinstance(state.get('aruco_dict'), cv2.aruco.Dictionary):
            raise TypeError('a CharucoBoard with a custom cv2.aruco.Dictionary cannot be pickled')
        return state

    def __setstate__(self, state):
        state.setdefault('units', None)
        self.__dict__.update(state)
        self.dictionary = get_aruco_dictionary(self.aruco_dict, self.marker_bits, self.dict_size)
        self.board = cv2.aruco.CharucoBoard((self.squaresX, self.squaresY),
                                            self.square_length, self.marker_length,
                                            self.dictionary)
        self.board.setLegacyPattern(self.legacy_pattern)
        self.detector_params = default_detector_params()
        self._aruco_detector = cv2.aruco.ArucoDetector(self.dictionary, self.detector_params)
        self._charuco_detector = cv2.aruco.CharucoDetector(self.board)

    def get_size(self):
        size = (self.squaresX, self.squaresY)
        return size

    def get_square_length(self):
        return self.square_length

    def get_empty_detection(self):
        return np.copy(self.empty_detection)

    def draw(self, size, margin='auto', border_bits=1):
        """Render the board as a uint8 image of ``size = (width, height)`` pixels.

        margin: white border in pixels around the board. ``'auto'`` (default) leaves half a
        square on every side, the quiet zone the outer markers need to be detected
        (edge-to-edge renders lose most edge corners). ``0`` renders edge to edge (0.9 output)."""
        width, height = int(size[0]), int(size[1])
        margin = _auto_margin(margin, width, height, self.squaresX, self.squaresY)
        return self.board.generateImage((width, height),
                                        marginSize=int(margin), borderBits=int(border_bits))

    def fill_points(self, corners, ids):
        out = self.get_empty_detection()
        if corners is None or len(corners) == 0:
            return out
        ids = np.asarray(ids).ravel()
        for i, cxs in zip(ids, corners):
            out[i] = cxs
        return out

    # ------------------------------------------------------------ detection

    @property
    def detector(self):
        """The detector instance (None for the built-in OpenCV path)."""
        if self._detector is None and self.detector_name == 'deepcharuco':
            from .detectors.deepcharuco import DeepCharucoDetector
            self._detector = DeepCharucoDetector(self, **self.detector_kwargs)
        return self._detector

    @property
    def batch_size(self):
        det = self.detector
        return int(getattr(det, 'batch_size', 1) or 1) if det is not None else 1

    def detect_markers(self, image, camera=None, refine=True):
        """Detect ArUco markers (OpenCV). Returns ``(corners, ids)``, or ``([], [])``."""
        gray = _to_gray(image)

        try:
            corners, ids, rejectedImgPoints = self._aruco_detector.detectMarkers(gray)
        except cv2.error as e:
            warnings.warn('cv2 marker detection failed on this image: {}'.format(e))
            return [], []

        if ids is None or len(ids) == 0:
            return [], []

        if not refine:
            return corners, ids

        if camera is None or getattr(camera, 'fisheye', False):
            # refineDetectedMarkers has no fisheye variant, and handing it the
            # equidistant coefficients would have it undistort with the wrong
            # model. Refinement is optional, so skip the intrinsics instead.
            K = D = None
        else:
            K = camera.get_camera_matrix().cpu().detach().numpy()
            D = camera.get_distortions().cpu().detach().numpy()

        try:
            detectedCorners, detectedIds, rejectedCorners, recoveredIdxs = \
                self._aruco_detector.refineDetectedMarkers(
                    gray, self.board, corners, ids, rejectedImgPoints, K, D)
        except cv2.error as e:
            warnings.warn('cv2 marker refinement failed on this image: {}'.format(e))
            detectedCorners, detectedIds = corners, ids

        return detectedCorners, detectedIds

    def _detect_opencv(self, image, camera=None):
        gray = _to_gray(image)
        corners, ids = self.detect_markers(gray, camera, refine=True)
        if len(corners) == 0:
            return empty_corners()
        try:
            ch_corners, ch_ids, _, _ = self._charuco_detector.detectBoard(
                gray, markerCorners=corners, markerIds=ids)
        except cv2.error as e:
            warnings.warn('cv2 ChArUco corner interpolation failed on this image: {}'.format(e))
            return empty_corners()
        if ch_ids is None or len(ch_ids) == 0:
            return empty_corners()
        return (np.asarray(ch_corners, dtype='float32').reshape(-1, 1, 2),
                np.asarray(ch_ids, dtype='int32').reshape(-1, 1))

    def _postprocess(self, image, corners, ids):
        if self.validate and len(corners) > 0 and \
                not self.validate_detection(corners, ids):
            return empty_corners()
        return corners, ids

    def detect_image(self, image, camera=None):
        """Detect ChArUco corners in one image (BGR or gray).

        Returns ``(corners, ids)``: corners ``(N, 1, 2)`` float32 in pixels and ids ``(N, 1)``
        int32. No detection gives shapes ``(0, 1, 2)`` / ``(0, 1)``."""
        det = self.detector
        if det is None:
            corners, ids = self._detect_opencv(image, camera)
        else:
            corners, ids = det.detect(image)
        return self._postprocess(image, corners, ids)

    def detect_frames(self, frames):
        det = self.detector
        if det is None or not hasattr(det, 'detect_batch'):
            return [self.detect_image(f) for f in frames]
        out = det.detect_batch(list(frames))
        return [self._postprocess(f, c, i) for f, (c, i) in zip(frames, out)]

    def validate_detection(self, corners, ids, max_homography_px=None, min_corners=6):
        """Check a detection against the board geometry with a homography (planar board).

        Returns a truthy/falsy :class:`DetectionCheck`. ``max_homography_px`` defaults to
        the board's ``max_homography_px`` (3 px). Strong lens distortion also inflates the
        residual, so with wide-angle lenses raise the threshold."""
        if max_homography_px is None:
            max_homography_px = self.max_homography_px
        return homography_check(self.objPoints, corners, ids,
                                max_homography_px=max_homography_px,
                                min_corners=min_corners)

    # ------------------------------------------------------------ geometry / pose

    def get_object_points(self, origin='zero'):
        """Corner positions ``(K, 3)`` indexed by id (see the class docstring).

        ``origin='zero'`` (default): first inner corner at the origin.
        ``origin='opencv'``: ``board.getChessboardCorners()`` (outer board corner at origin)."""
        if origin == 'zero':
            return self.objPoints
        if origin == 'opencv':
            return self.objPoints + np.array([self.square_length, self.square_length, 0.0])
        raise ValueError("origin must be 'zero' or 'opencv'")

    def estimate_pose_points(self, camera, corners, ids, *, min_corners=7, origin='zero'):
        """Board pose ``(rvec, tvec)`` in the camera frame from detected corners.

        ``solvePnP`` (planar IPPE, refined with LM) on the corners matched to the board's
        object points by id; a fisheye camera is handled by solving on undistorted points.
        The pose maps ``get_object_points(origin)`` into the camera frame (zero-origin by
        default; 0.9.x used OpenCV's origin, ``origin='opencv'``). Returns ``(None, None)``
        with fewer than ``min_corners`` usable corners, or when they are collinear."""
        if corners is None or ids is None or len(corners) < min_corners:
            return None, None

        n_corners = corners.size // 2
        corners = np.reshape(np.asarray(corners, dtype='float64'), (n_corners, 1, 2))
        ids = np.asarray(ids).astype('int64').reshape(-1)
        if len(ids) != n_corners:
            return None, None

        corners, K, D, _ = pnp_inputs(camera, corners)

        # a fisheye undistort returns NaN outside its invertible domain
        good = np.all(np.isfinite(corners), axis=(1, 2))
        good &= (ids >= 0) & (ids < self.total_size)
        if np.sum(good) < min_corners:
            return None, None
        corners = corners[good]
        ids = ids[good]

        if self.board.checkCharucoCornersCollinear(ids.reshape(-1, 1).astype('int32')):
            return None, None

        objp = self.get_object_points(origin)[ids].reshape(-1, 1, 3)
        return solve_board_pnp(objp, corners, K, D)


def solve_board_pnp(objp, imgp, K, D):
    """Planar PnP: IPPE (picks the better of the two planar solutions), then LM refinement.

    Falls back to the iterative solver. Returns ``(rvec, tvec)`` (3x1 each) or ``(None, None)``."""
    objp = np.ascontiguousarray(objp, dtype='float64').reshape(-1, 1, 3)
    imgp = np.ascontiguousarray(imgp, dtype='float64').reshape(-1, 1, 2)
    K = np.asarray(K, dtype='float64')
    D = np.asarray(D, dtype='float64').ravel()
    for flags in (cv2.SOLVEPNP_IPPE, cv2.SOLVEPNP_ITERATIVE):
        try:
            ok, rvec, tvec = cv2.solvePnP(objp, imgp, K, D, flags=flags)
        except cv2.error:
            continue
        if not ok or not np.all(np.isfinite(rvec)) or not np.all(np.isfinite(tvec)):
            continue
        try:
            rvec, tvec = cv2.solvePnPRefineLM(objp, imgp, K, D, rvec, tvec)
        except cv2.error:
            pass
        return rvec, tvec
    return None, None



# ------------------------------------------------------------------ generic 3D objects

def _dlt_projection(objp, imgp):
    """Normalised (Hartley) DLT: 3x4 ``P`` with ``imgp ~ P [objp; 1]`` from >= 6 points."""
    X = np.asarray(objp, dtype='float64').reshape(-1, 3)
    x = np.asarray(imgp, dtype='float64').reshape(-1, 2)
    mx, mX = x.mean(0), X.mean(0)
    sx = np.sqrt(2) / max(np.mean(np.linalg.norm(x - mx, axis=1)), 1e-12)
    sX = np.sqrt(3) / max(np.mean(np.linalg.norm(X - mX, axis=1)), 1e-12)
    Tx = np.array([[sx, 0, -sx * mx[0]], [0, sx, -sx * mx[1]], [0, 0, 1]])
    TX = np.eye(4)
    TX[:3, :3] *= sX
    TX[:3, 3] = -sX * mX
    xn = (x - mx) * sx
    Xn = np.hstack([(X - mX) * sX, np.ones((len(X), 1))])
    A = np.zeros((2 * len(X), 12))
    A[0::2, 0:4] = Xn
    A[0::2, 8:12] = -xn[:, :1] * Xn
    A[1::2, 4:8] = Xn
    A[1::2, 8:12] = -xn[:, 1:] * Xn
    _, _, Vt = np.linalg.svd(A)
    Pn = Vt[-1].reshape(3, 4)
    return np.linalg.solve(Tx, Pn @ TX)


def _coplanar(points, tol=1e-3):
    """Are the points (nearly) coplanar? Smallest / largest singular value of the centred cloud."""
    X = np.asarray(points, dtype='float64').reshape(-1, 3)
    if len(X) < 4:
        return True
    s = np.linalg.svd(X - X.mean(0), compute_uv=False)
    return bool(s[-1] <= tol * max(s[0], 1e-300))


class PointObject(CalibrationObject):
    """Any calibration object with **known 3D points** (plan 05 W1b): a non-planar wand or
    fixture, a 3D-printed target, surveyed landmarks. There is no detector: the 2D points are
    annotations (a dense ``C x F x K x 2`` array, NaN = not annotated; see
    :meth:`annotations_from_dicts`), and :func:`aniposelib.workflows.calibrate_from_board`
    takes them in place of board detections.

    Parameters
    ----------
    points3d : (K, 3) array
        Known coordinates of the object points, in ``units``.
    names : list of K str, optional
        Point names (for dict annotations).
    static : bool
        False: a rigid object moved around (one pose per frame, like a board). True: fixed
        landmarks (one pose) that define the world frame and the metric scale.
    units : str, optional
        Unit of ``points3d`` (informational, written to ``cgroup.metadata['units']``).

    ``get_object_points()`` is ``points3d``; ``estimate_pose_points(camera, corners, ids)`` is a
    PnP (>= 4 points; IPPE for planar objects, SQPnP/EPnP + LM refinement otherwise).
    ``planar`` says whether the points are coplanar (then the usual planar intrinsics
    initialisation applies; otherwise :meth:`estimate_intrinsics` uses a per-view DLT).
    """

    def __init__(self, points3d, names=None, static=False, units=None):
        P = np.asarray(points3d, dtype='float64')
        if P.ndim != 2 or P.shape[1] != 3 or len(P) < 4:
            raise ValueError('points3d must be K x 3 with K >= 4, got shape {}'.format(P.shape))
        if not np.all(np.isfinite(P)):
            raise ValueError('points3d must be finite')
        self.points3d = P
        self.n_points = len(P)
        self.total_size = len(P)
        if names is not None:
            names = [str(n) for n in names]
            if len(names) != len(P) or len(set(names)) != len(names):
                raise ValueError('names must be {} unique strings'.format(len(P)))
        self.names = names
        self.static = bool(static)
        self.units = units
        self.planar = _coplanar(P)
        self.ids = np.arange(len(P))
        self.empty_detection = np.full((len(P), 1, 2), np.nan)

    def __repr__(self):
        return 'PointObject({} points, {}, static={}, units={!r})'.format(
            self.n_points, 'planar' if self.planar else 'non-planar', self.static, self.units)

    # ---------------------------------------------- CalibrationObject interface
    def draw(self, size):
        raise NotImplementedError('a PointObject cannot be drawn')

    def detect_image(self, image):
        raise NotImplementedError(
            'PointObject has no detector: annotate the points and pass a C x F x K x 2 array '
            '(see PointObject.annotations_from_dicts) to calibrate_from_board')

    def get_object_points(self):
        return self.points3d.copy()

    def get_empty_detection(self):
        return np.copy(self.empty_detection)

    def get_size(self):
        return (self.n_points,)

    def fill_points(self, corners, ids):
        out = self.get_empty_detection()
        if corners is None or len(corners) == 0:
            return out
        corners = np.asarray(corners, dtype='float64').reshape(-1, 2)
        if ids is None:
            if len(corners) != self.n_points:
                raise ValueError('corners without ids must have one row per object point')
            return corners.reshape(-1, 1, 2)
        ids = np.asarray(ids).astype('int64').ravel()
        good = (ids >= 0) & (ids < self.n_points)
        out[ids[good], 0] = corners[good]
        return out

    def estimate_pose_points(self, camera, corners, ids=None, min_points=4):
        """Pose ``(rvec, tvec)`` of the object in camera coordinates by PnP, or ``(None, None)``."""
        if corners is None:
            return None, None
        filled = self.fill_points(corners, ids).reshape(-1, 2)
        good = np.all(np.isfinite(filled), axis=1)
        if good.sum() < max(4, min_points):
            return None, None
        pts, K, D, _ = pnp_inputs(camera, filled[good].reshape(-1, 1, 2))
        pts = np.asarray(pts, dtype='float64').reshape(-1, 1, 2)
        objp = self.points3d[good]
        ok = np.all(np.isfinite(pts), axis=(1, 2))
        if ok.sum() < 4:
            return None, None
        pts, objp = pts[ok], objp[ok]
        if self.planar or _coplanar(objp):
            return solve_board_pnp(objp, pts, K, D)
        objp = np.ascontiguousarray(objp.reshape(-1, 1, 3))
        D = np.asarray(D, dtype='float64').ravel()
        for flags in (cv2.SOLVEPNP_SQPNP, cv2.SOLVEPNP_EPNP, cv2.SOLVEPNP_ITERATIVE):
            try:
                ok, rvec, tvec = cv2.solvePnP(objp, pts, K, D, flags=flags)
            except cv2.error:
                continue
            if not ok or not np.all(np.isfinite(rvec)) or not np.all(np.isfinite(tvec)):
                continue
            try:
                rvec, tvec = cv2.solvePnPRefineLM(objp, pts, K, D, rvec, tvec)
            except cv2.error:
                pass
            if tvec.ravel()[2] <= 0:
                continue
            return rvec, tvec
        return None, None

    # ---------------------------------------------- annotations
    def annotations_from_dicts(self, per_cam, cam_names=None, return_frames=False):
        """``{cam: {frame: {name_or_index: (x, y)}}}`` -> ``C x F x K x 2`` (NaN = not
        annotated). Cameras in ``cam_names`` order (default: the dict's order); frames are the
        sorted union of all frame keys (``return_frames=True`` also returns them)."""
        cams = list(per_cam) if cam_names is None else list(cam_names)
        frames = sorted({f for c in cams for f in per_cam.get(c, {})}, key=lambda k: (str(type(k)), k))
        index = {f: i for i, f in enumerate(frames)}
        out = np.full((len(cams), len(frames), self.n_points, 2), np.nan)
        for ci, c in enumerate(cams):
            for f, pts in per_cam.get(c, {}).items():
                for key, xy in pts.items():
                    if isinstance(key, str):
                        if self.names is None or key not in self.names:
                            raise KeyError('unknown point name {!r}'.format(key))
                        k = self.names.index(key)
                    else:
                        k = int(key)
                        if not 0 <= k < self.n_points:
                            raise KeyError('point index {} out of range'.format(k))
                    if xy is not None:
                        out[ci, index[f], k] = np.asarray(xy, dtype='float64').reshape(2)
        return (out, frames) if return_frames else out

    def rows_from_array(self, points):
        """``C x F x K x 2`` annotations -> per-camera rows (``framenum``, ``corners``, ``ids``,
        ``filled``), the format of board detections."""
        arr = np.asarray(points, dtype='float64')
        if arr.ndim != 4 or arr.shape[2] != self.n_points or arr.shape[3] != 2:
            raise ValueError('annotations must be C x F x {} x 2, got {}'.format(self.n_points, arr.shape))
        all_rows = []
        for c in range(arr.shape[0]):
            rows = []
            for f in range(arr.shape[1]):
                good = np.all(np.isfinite(arr[c, f]), axis=1)
                if not good.any():
                    continue
                ids = np.nonzero(good)[0]
                rows.append({'framenum': (0, f), 'corners': arr[c, f, ids].reshape(-1, 1, 2),
                             'ids': ids.reshape(-1, 1).astype('int32'),
                             'filled': self.fill_points(arr[c, f, ids], ids)})
            all_rows.append(rows)
        return all_rows

    # ---------------------------------------------- intrinsics
    def estimate_intrinsics(self, rows, size=None, min_points=6):
        """Camera matrix from a per-view DLT (non-planar objects only): every view with >=
        ``min_points`` non-coplanar annotated points gives ``P``, RQ-decomposed with
        :func:`aniposelib.geometry.from_projection_matrix`; the median over views of fx, fy,
        cx, cy seeds K (zero skew). Returns ``(K or None, info)``."""
        from .geometry import decompose_projection
        Ks = []
        for r in rows:
            filled = self.fill_points(r['corners'], r['ids']).reshape(-1, 2)
            good = np.all(np.isfinite(filled), axis=1)
            if good.sum() < max(6, min_points) or _coplanar(self.points3d[good]):
                continue
            try:
                K, _, _ = decompose_projection(_dlt_projection(self.points3d[good], filled[good]))
            except (ValueError, np.linalg.LinAlgError):
                continue
            if np.all(np.isfinite(K)) and K[0, 0] > 0 and K[1, 1] > 0:
                Ks.append(K)
        info = dict(n_views=len(Ks), method='dlt')
        if not Ks:
            return None, info
        Ks = np.array(Ks)
        K = np.eye(3)
        K[0, 0], K[1, 1] = np.median(Ks[:, 0, 0]), np.median(Ks[:, 1, 1])
        K[0, 2], K[1, 2] = np.median(Ks[:, 0, 2]), np.median(Ks[:, 1, 2])
        if size is not None:
            info['focal_rel_spread'] = float(np.median(np.abs(Ks[:, 0, 0] / K[0, 0] - 1)))
        return K, info
