#!/usr/bin/env python

import pytest

import numpy as np

import importlib
import aniposelib.cameras
importlib.reload(aniposelib.cameras)

from aniposelib.cameras import Camera, CameraGroup
from aniposelib.boards import CharucoBoard, Checkerboard

import os
import toml
import pickle

import torch
import time

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# copied from anipose
def get_calibration_board(config):
    calib = config['calibration']
    board_size = calib['board_size']
    board_type = calib['board_type'].lower()

    manually_verify = False
    
    if board_type == 'aruco':
        raise NotImplementedError("aruco board is not implemented with the current pipeline")
    elif board_type == 'charuco':
        board = CharucoBoard(
            board_size[0], board_size[1],
            calib['board_square_side_length'],
            calib['board_marker_length'],
            calib['board_marker_bits'],
            calib['board_marker_dict_number'],
            manually_verify=manually_verify)



    elif board_type == 'checkerboard':
        board = Checkerboard(board_size[0], board_size[1],
                             calib['board_square_side_length'], manually_verify=manually_verify)
    else:
        raise ValueError("board_type should be one of "
                         "'aruco', 'charuco', or 'checkerboard' not '{}'".format(
                             board_type))

    return board

@pytest.mark.integration
def test_7cam_allen_calibration():
    prefix = 'tests/data/calib/7cam-allen-mouse'
    calib_path = os.path.join(prefix, 'calibration.toml')
    detection_path = os.path.join(prefix, 'detections.pickle')
    config_path = os.path.join(prefix, 'config.toml')

    cgroup_ref = CameraGroup.load(calib_path)

    cgroup = CameraGroup.from_names(cgroup_ref.get_names())
    for cam, cref in zip(cgroup.cameras, cgroup_ref.cameras):
        cam.set_size(cref.get_size())

    cgroup = cgroup.to(device)

    with open(detection_path, 'rb') as f:
        all_rows = pickle.load(f)

    config = toml.load(config_path)
    board = get_calibration_board(config)

    err = cgroup.calibrate_rows(all_rows, board)
    # print(err)
    assert err < 1.5

    return err

@pytest.mark.integration
def test_12cam_allen_calibration():
    prefix = 'tests/data/calib/12cam-allen-mouse'
    calib_path = os.path.join(prefix, 'calibration.toml')
    detection_path = os.path.join(prefix, 'detections.pickle')
    config_path = os.path.join(prefix, 'config.toml')

    cgroup_ref = CameraGroup.load(calib_path)

    cgroup = CameraGroup.from_names(cgroup_ref.get_names())
    for cam, cref in zip(cgroup.cameras, cgroup_ref.cameras):
        cam.set_size(cref.get_size())

    cgroup = cgroup.to(device)

    with open(detection_path, 'rb') as f:
        all_rows = pickle.load(f)

    config = toml.load(config_path)
    board = get_calibration_board(config)

    err = cgroup.calibrate_rows(all_rows, board)
    # print(err)
    assert err < 1.5

    return err


def see_mouse_outdoor_4_10():
    prefix = 'tests/data/calib/mouse-outdoor'
    calib_path = os.path.join(prefix, 'calibration_right_4_10_2026_adjusted.toml')
    point_path = os.path.join(prefix, 'points_right_4_10_2026_sub.npz')

    cgroup = CameraGroup.load(calib_path).to(device)
    dd = np.load(point_path)

    p2d_sub = dd['p2d']

    cgroup.bundle_adjust_iter(p2d_sub, verbose=True)

    p3d = cgroup.triangulate(p2d_sub, progress=False)
    err = cgroup.reprojection_error(p3d, p2d_sub, mean=True)
    outerr = torch.median(err).item()

    # print(outerr)
    assert outerr < 10

    return outerr

# start_time = time.time()

# print(test_7cam_allen_calibration())
# print(test_12cam_allen_calibration())
# print(see_mouse_outdoor_4_10())

# end_time = time.time()

# total_time = end_time - start_time

# print("total time: ", total_time)
