"""Render synthetic camera views of calibration boards (for detection tests)."""

import numpy as np

def render_board(board, K, rvec, tvec, size, dist=None, fisheye=False, background=128, ppu=None):
    """Render a board seen by a camera (pinhole or fisheye), via inverse mapping.

    The board pose maps zero-origin object points into the camera frame. Returns a uint8 gray image."""
    import cv2
    W, H = size
    sl = board.square_length
    nx, ny = board.squaresX, board.squaresY
    if ppu is None:
        ppu = 120.0 / sl  # texture pixels per board unit
    tex = board.draw((int(round(nx * sl * ppu)), int(round(ny * sl * ppu))))
    xs, ys = np.meshgrid(np.arange(W, dtype=np.float64), np.arange(H, dtype=np.float64))
    pix = np.stack([xs.ravel(), ys.ravel()], 1).reshape(-1, 1, 2)
    if fisheye:
        D = np.zeros(4) if dist is None else np.asarray(dist, dtype=np.float64)
        und = cv2.fisheye.undistortPoints(pix, K, D.reshape(4, 1)).reshape(-1, 2)
    else:
        D = np.zeros(5) if dist is None else np.asarray(dist, dtype=np.float64)
        und = cv2.undistortPoints(pix, K, D).reshape(-1, 2)
    R = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64).reshape(3))[0]
    t = np.asarray(tvec, dtype=np.float64).reshape(3)
    d = np.concatenate([und, np.ones((len(und), 1))], 1)
    db = d @ R               # R^T d, per row
    tb = R.T @ t
    lam = tb[2] / db[:, 2]
    Xb = lam[:, None] * db - tb
    # zero-origin board coords -> OpenCV board coords -> texture pixels
    u = (Xb[:, 0] + sl) * ppu - 0.5
    v = (Xb[:, 1] + sl) * ppu - 0.5
    bad = (lam <= 0) | ~np.isfinite(u) | ~np.isfinite(v)
    u[bad] = -10
    v[bad] = -10
    img = cv2.remap(tex, u.reshape(H, W).astype(np.float32), v.reshape(H, W).astype(np.float32),
                    cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_CONSTANT, borderValue=background)
    return img


def look_at_pose(board, distance, rvec=(0.3, -0.25, 0.1), offset=(0.0, 0.0)):
    """A board pose (rvec, tvec) with the board centre in front of the camera at ``distance``."""
    import cv2
    rvec = np.asarray(rvec, dtype=np.float64)
    R = cv2.Rodrigues(rvec)[0]
    c = board.get_object_points().mean(0)
    tvec = np.array([offset[0], offset[1], distance]) - R @ c
    return rvec, tvec
