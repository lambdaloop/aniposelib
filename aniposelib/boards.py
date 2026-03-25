import cv2
import numpy as np
from abc import ABC, abstractmethod
from tqdm import trange
from collections import defaultdict

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


class CalibrationObject(ABC):
    @abstractmethod
    def draw(self, size):
        pass

    @abstractmethod
    def detect_image(self, image):
        pass

    @abstractmethod
    def manually_verify_board_detection(self, image, corners):
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

    def detect_images(self, images, progress=False, prefix=None):
        length = len(images)
        rows = []

        if progress:
            it = trange(length, ncols=70)
        else:
            it = range(length)

        for framenum in it:
            imname = images[framenum]
            frame = cv2.imread(imname)

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
                    'fname': imname
                }

                rows.append(row)

        rows = self.fill_points_rows(rows)

        return rows

    def detect_video(self, vidname, prefix=None, skip=20, progress=False):
        cap = cv2.VideoCapture(vidname)
        if not cap.isOpened():
            raise FileNotFoundError(f'missing video file "{vidname}"')
        length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
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

        # all_obj = np.vstack(all_obj)
        # all_img = np.vstack(all_img)

        # all_obj = np.array(all_obj, dtype='float64')
        # all_img = np.array(all_img, dtype='float64')

        return all_obj, all_img


class Checkerboard(CalibrationObject):
    DETECT_PARAMS = \
        cv2.CALIB_CB_NORMALIZE_IMAGE + \
        cv2.CALIB_CB_ADAPTIVE_THRESH + \
        cv2.CALIB_CB_FAST_CHECK

    SUBPIX_CRITERIA = (cv2.TERM_CRITERIA_EPS +
                       cv2.TERM_CRITERIA_MAX_ITER,
                       30, 0.01)

    def __init__(self, squaresX, squaresY, square_length=1, manually_verify=False):
        self.squaresX = squaresX
        self.squaresY = squaresY
        self.square_length = square_length
        self.manually_verify = manually_verify

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

    # TODO: implement checkerboard draw function
    def draw(self, size):
        pass

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

        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        size = self.get_size()
        pattern_was_found, corners = cv2.findChessboardCorners(gray, size, self.DETECT_PARAMS)

        if corners is not None:

            if subpix:
                corners = cv2.cornerSubPix(gray, corners, (3, 3), (-1, -1), self.SUBPIX_CRITERIA)

        if corners is not None \
            and self.manually_verify \
                and not self.manually_verify_board_detection(gray, corners):
            corners = None

        if corners is None:
            ids = None
        else:
            ids = self.ids

        return corners, ids

    def manually_verify_board_detection(self, image, corners):

        height, width = image.shape[:2]
        image = cv2.drawChessboardCorners(image, self.get_size(), corners, 1)
        cv2.putText(image, '(a) Accept (d) Reject', (int(width/1.35), int(height/16)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 255, 1, cv2.LINE_AA)
        cv2.imshow('verify_detection', image)
        while 1:
            key = cv2.waitKey(0) & 0xFF
            if key == ord('a'):
                cv2.putText(image, 'Accepted!', (int(width/2.5), int(height/1.05)), cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 2, cv2.LINE_AA)
                cv2.imshow('verify_detection', image)
                cv2.waitKey(100)
                return True
            elif key == ord('d'):
                cv2.putText(image, 'Rejected!', (int(width/2.5), int(height/1.05)), cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 2, cv2.LINE_AA)
                cv2.imshow('verify_detection', image)
                cv2.waitKey(100)
                return False

    def get_object_points(self):
        return self.objPoints

    def estimate_pose_points(self, camera, points, ids=None):
        ngood = np.sum(~np.isnan(points)) // 2
        if points is None or ngood < 7:
            return None, None

        n_points = points.size // 2
        points = np.reshape(points, (n_points, 1, 2))

        K = camera.get_camera_matrix()
        D = camera.get_distortions()
        obj_points = self.get_object_points()

        if points.shape[0] != obj_points.shape[0]:
            return None, None

        try:
            retval, rvec, tvec, inliers = cv2.solvePnPRansac(obj_points,
                                                             points,
                                                             K,
                                                             D,
                                                             confidence=0.9,
                                                             reprojectionError=30)
            return rvec, tvec

        except:
            print("W: failed to find checkerboard pose in image")
            return None, None




class CharucoBoard(CalibrationObject):
    def __init__(self,
                 squaresX,
                 squaresY,
                 square_length,
                 marker_length,
                 marker_bits=4,
                 dict_size=50,
                 aruco_dict=None,
                 manually_verify=False):
        self.squaresX = squaresX
        self.squaresY = squaresY
        self.square_length = square_length
        self.marker_length = marker_length
        self.manually_verify = manually_verify

        # import aruco only here so that we only require opencv-contrib-python when using ChArUco module
        global aruco
        from cv2 import aruco

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

        dkey = (marker_bits, dict_size)
        self.dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICTS[dkey])

        self.board = cv2.aruco.CharucoBoard([squaresX, squaresY],
                                            square_length, marker_length,
                                            self.dictionary)
        # set up detector parameters for ArUco marker detection
        self.detector_params = cv2.aruco.DetectorParameters()
        self.detector_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_CONTOUR
        self.detector_params.adaptiveThreshWinSizeMin = 50
        self.detector_params.adaptiveThreshWinSizeMax = 700
        self.detector_params.adaptiveThreshWinSizeStep = 50
        self.detector_params.adaptiveThreshConstant = 0
        # create detector instances using the OpenCV 4.7+ class-based API
        self.detector = cv2.aruco.ArucoDetector(self.dictionary,
                                                self.detector_params)
        self.charuco_detector = cv2.aruco.CharucoDetector(self.board)
        self.charuco_detector.setDetectorParameters(self.detector_params)

        total_size = (squaresX - 1) * (squaresY - 1)

        objp = np.zeros((total_size, 3), np.float64)
        objp[:, :2] = np.mgrid[0:(squaresX - 1), 0:(squaresY - 1)].T.reshape(
            -1, 2)
        objp *= square_length
        self.objPoints = objp

        self.empty_detection = np.zeros((total_size, 1, 2)) * np.nan
        self.total_size = total_size

    def get_size(self):
        size = (self.squaresX, self.squaresY)
        return size

    def get_square_length(self):
        return self.square_length

    def get_empty_detection(self):
        return np.copy(self.empty_detection)

    def draw(self, size):
        return self.board.generateImage(size)

    def fill_points(self, corners, ids):
        out = self.get_empty_detection()
        if corners is None or len(corners) == 0:
            return out
        ids = ids.ravel()
        for i, cxs in zip(ids, corners):
            out[i] = cxs
        return out

    def detect_markers(self, image, camera=None, refine=True):
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        try:
            corners, ids, rejectedImgPoints = self.detector.detectMarkers(gray)
        except Exception:
            ids = None

        if ids is None:
            return [], []

        if camera is None:
            K = D = None
        else:
            K = camera.get_camera_matrix()
            D = camera.get_distortions()

        if refine:
            detectedCorners, detectedIds, rejectedCorners, recoveredIdxs = \
                self.detector.refineDetectedMarkers(gray, self.board,
                    corners, ids,
                    rejectedImgPoints,
                    K, D)
        else:
            detectedCorners, detectedIds = corners, ids

        return detectedCorners, detectedIds

    def detect_image(self, image, camera=None):

        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        # use CharucoDetector which combines marker detection + charuco
        # corner interpolation in one step (OpenCV 4.7+ API)
        detectedCorners, detectedIds, _, _ = self.charuco_detector.detectBoard(gray)
        if detectedCorners is None:
            detectedCorners = detectedIds = np.float64([])

        if len(detectedCorners) > 0 \
            and self.manually_verify \
            and not self.manually_verify_board_detection(gray, detectedCorners, detectedIds):
            detectedCorners = detectedIds = np.float64([])

        return detectedCorners, detectedIds


    def manually_verify_board_detection(self, image, corners, ids=None):

        height, width = image.shape[:2]
        image = cv2.aruco.drawDetectedCornersCharuco(image, corners, ids)
        cv2.putText(image, '(a) Accept (d) Reject', (int(width/1.35), int(height/16)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 255, 1, cv2.LINE_AA)
        cv2.imshow('verify_detection', image)
        while 1:
            key = cv2.waitKey(0) & 0xFF
            if key == ord('a'):
                cv2.putText(image, 'Accepted!', (int(width/2.5), int(height/1.05)), cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 2, cv2.LINE_AA)
                cv2.imshow('verify_detection', image)
                cv2.waitKey(100)
                return True
            elif key == ord('d'):
                cv2.putText(image, 'Rejected!', (int(width/2.5), int(height/1.05)), cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 2, cv2.LINE_AA)
                cv2.imshow('verify_detection', image)
                cv2.waitKey(100)
                return False

    def get_object_points(self):
        return self.objPoints

    def estimate_pose_points(self, camera, corners, ids):
        if corners is None or ids is None or len(corners) < 7:
            return None, None

        n_corners = corners.size // 2
        corners = np.reshape(corners, (n_corners, 1, 2))

        K = camera.get_camera_matrix()
        D = camera.get_distortions()
        # use getChessboardCorners + solvePnP (OpenCV 4.7+ API)
        all_obj_points = self.board.getChessboardCorners()
        if len(ids) > 0 and all_obj_points is not None:
            detected_obj_points = []
            detected_img_points = []
            # match detected corner IDs to object points
            for i, corner_id in enumerate(ids.flatten()):
                if corner_id < len(all_obj_points):
                    detected_obj_points.append(all_obj_points[corner_id])
                    detected_img_points.append(corners[i].reshape(2))
            if len(detected_obj_points) >= 7:
                obj_points_array = np.array(detected_obj_points,
                                            dtype=np.float32).reshape(-1, 3)
                img_points_array = np.array(detected_img_points,
                                            dtype=np.float32).reshape(-1, 2)
                ret, rvec, tvec = cv2.solvePnP(obj_points_array,
                                               img_points_array, K, D)
                if ret:
                    return rvec, tvec
        return None, None
