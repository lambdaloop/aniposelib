import cv2
from cv2 import aruco
import numpy as np
from abc import ABC, abstractmethod

def fix_rvec(rvec, tvec):
    # https://github.com/opencv/opencv/issues/8813
    T = tvec.ravel()[0]
    R = cv2.Rodrigues(rvec)[0]

    # Unrelated -- makes Y the up axis, Z forward
    R = R @ np.array([
            [1, 0, 0],
            [0, 0, 1],
            [0,-1, 0],
    ])
    if 0 < R[1,1] < 1:
        # If it gets here, the pose is flipped.

        # Flip the axes. E.g., Y axis becomes [-y0, -y1, y2].
        R *= np.array([
                [ 1, -1,  1],
                [ 1, -1,  1],
                [-1,  1, -1],
        ])

        # Fixup: rotate along the plane spanned by camera's forward (Z) axis and vector to marker's position
        forward = np.array([0, 0, 1])
        tnorm = T / np.linalg.norm(T)
        axis = np.cross(tnorm, forward)
        angle = -2*math.acos(tnorm @ forward)
        R = cv2.Rodrigues(angle * axis)[0] @ R

    return cv2.Rodrigues(R)[0]

class CalibrationObject(ABC):

    @abstractmethod
    def draw(self, size):
        pass

    @abstractmethod
    def detect_image(self, image):
        pass

    def detect_video(self, vidname):
        pass

    @abstractmethod
    def get_object_points(self):
        pass

    def estimate_pose_image(self, camera, image):
        corners, ids = self.detect_image(image)
        return self.estimate_pose_points(camera, corners, ids)

    @abstractmethod
    def estimate_pose_points(self, camera, points):
        pass



class Checkerboard(CalibrationObject):
    DETECT_PARAMS = \
        cv2.CALIB_CB_FAST_CHECK + \
        cv2.CALIB_CB_ADAPTIVE_THRESH + \
        cv2.CALIB_CB_FILTER_QUADS


    SUBPIX_CRITERIA = (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER,
                       30, 0.1)

    
    def __init__(self, squaresX, squaresY, square_length=1):
        self.squaresX = squaresX
        self.squaresY = squaresY
        self.square_length = square_length

        total_size = squaresX * squaresY
        
        objp = np.zeros((total_size, 3), np.float32)
        objp[:, :2] = np.mgrid[0:squaresY, 0:squaresX].T.reshape(-1, 2)
        objp *= square_length
        self.objPoints = objp

        self.ids = np.arange(total_size)

        self.empty_detection = np.zeros((total_size, 1, 2))*np.nan
        
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
            ids = np.squeeze(ids)
            for i, cxs in zip(ids, corners):
                out[i] = cxs
            return out

    def detect_image(self, image, subpix=True):
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        size = self.get_size()
        ret, corners = cv2.findChessboardCorners(gray, size, self.DETECT_PARAMS)

        if ret and subpix:
            corners = cv2.cornerSubPix(
                gray, corners, (3,3), (-1,-1), self.SUBPIX_CRITERIA)

        if corners is None:
            ids = None
        else:
            ids = self.ids
            
        return corners, ids

    def get_object_points(self):
        return self.objPoints

    def estimate_pose_image(self, camera, image):
        corners, ids = self.detect_image(image)
        return self.estimate_pose_points(camera, corners, ids)

    def estimate_pose_points(self, camera, points, ids=None):
        ngood = np.sum(np.isnan(corners)) // 2
        if points is None or ngood < 3:
            return None, None

        n_points = points.size // 2
        points = np.reshape(points, (n_points, 1, 2))

        K = camera.get_camera_matrix()
        D = camera.get_distortions()
        obj_points = self.get_object_points()
        
        retval, rvec, tvec, inliers = cv2.solvePnPRansac(
            obj_points, corners, K, D,
            confidence=0.9, reprojectionError=10)

        return rvec, tvec

    
ARUCO_DICTS = {
    (4, 50): aruco.DICT_4X4_50,
    (5, 50): aruco.DICT_5X5_50,
    (6, 50): aruco.DICT_6X6_50,
    (7, 50): aruco.DICT_7X7_50,

    (4, 100): aruco.DICT_4X4_100,
    (5, 100): aruco.DICT_5X5_100,
    (6, 100): aruco.DICT_6X6_100,
    (7, 100): aruco.DICT_7X7_100,

    (4, 250): aruco.DICT_4X4_250,
    (5, 250): aruco.DICT_5X5_250,
    (6, 250): aruco.DICT_6X6_250,
    (7, 250): aruco.DICT_7X7_250,

    (4, 1000): aruco.DICT_4X4_1000,
    (5, 1000): aruco.DICT_5X5_1000,
    (6, 1000): aruco.DICT_6X6_1000,
    (7, 1000): aruco.DICT_7X7_1000
}


class CharucoBoard(CalibrationObject):
    def __init__(self, squaresX, squaresY,
                 square_length, marker_length, 
                 marker_bits=4, dict_size=50,
                 aruco_dict=None):
        self.squaresX = squaresX
        self.squaresY = squaresY
        self.square_length = square_length
        self.marker_length = marker_length

        dkey = (marker_bits, dict_size)
        self.dictionary = aruco.getPredefinedDictionary(ARUCO_DICTS[dkey])

        self.board = aruco.CharucoBoard_create(
                squaresX, squaresY,
                square_length, marker_length,
                self.dictionary)

        total_size = (squaresX-1) * (squaresY-1)
        
        objp = np.zeros((total_size, 3), np.float32)
        objp[:, :2] = np.mgrid[0:(squaresY-1), 0:(squaresX-1)].T.reshape(-1, 2)
        objp *= square_length
        self.objPoints = objp

        self.empty_detection = np.zeros((total_size, 1, 2))*np.nan
        self.total_size = total_size
        
        
    def get_size(self):
        size = (self.squaresX, self.squaresY)
        return size

    def get_square_length(self):
        return self.square_length

    def get_empty_detection(self):
        return np.copy(self.empty_detection)
    
    def draw(self, size):
        return self.board.draw(size)
    
    def fill_points(self, corners, ids):
        out = self.get_empty_detection()
        if corners is None or len(corners) == 0:
            return out
        ids = np.squeeze(ids)
        for i, cxs in zip(ids, corners):
            out[i] = cxs
        return out

    def detect_markers(self, image, camera=None):
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        params = aruco.DetectorParameters_create()
        params.cornerRefinementMethod = aruco.CORNER_REFINE_CONTOUR
        params.adaptiveThreshWinSizeMin = 100
        params.adaptiveThreshWinSizeMax = 700
        params.adaptiveThreshWinSizeStep = 50
        params.adaptiveThreshConstant = 5

        corners, ids, rejectedImgPoints = aruco.detectMarkers(
            gray, self.dictionary, parameters=params)

        if ids is None:
            return [], []

        if camera is None:
            K = D = None
        else:
            K = camera.get_camera_matrix()
            D = camera.get_distortions()

        detectedCorners, detectedIds, rejectedCorners, recoveredIdxs = \
            aruco.refineDetectedMarkers(gray, self.board, corners, ids,
                                        rejectedImgPoints,
                                        K, D,
                                        parameters=params)

        return detectedCorners, detectedIds

    
    def detect_image(self, image, camera=None):
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        corners, ids = self.detect_markers(image, camera)
        if len(corners) > 0:
            ret, detectedCorners, detectedIds = aruco.interpolateCornersCharuco(
                corners, ids, gray, self.board)
            if detectedIds is None:
                detectedCorners = detectedIds = np.float32([])
        else:
            detectedCorners = detectedIds = np.float32([])
                
        return detectedCorners, detectedIds

    def get_object_points(self):
        return self.objPoints

    def estimate_pose_image(self, camera, image):
        corners, ids = self.detect_image(image)
        return self.estimate_pose_points(camera, corners, ids)

    def estimate_pose_points(self, camera, corners, ids):
        if corners is None or ids is None or len(corners) < 3:
            return None, None

        n_corners = corners.size // 2
        corners = np.reshape(corners, (n_corners, 1, 2))
        
        K = camera.get_camera_matrix()
        D = camera.get_distortions()

        ret, rvec, tvec = aruco.estimatePoseCharucoBoard(
            corners, ids, self.board, K, D)

        return rvec, tvec

        
