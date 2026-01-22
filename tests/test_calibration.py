#!/usr/bin/env python

import pytest

import numpy as np

from aniposelib.cameras import Camera, CameraGroup
from aniposelib.boards import CharucoBoard, Checkerboard

import os
import toml
import pickle

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


    with open(detection_path, 'rb') as f:
        all_rows = pickle.load(f)

    config = toml.load(config_path)
    board = get_calibration_board(config)

    err = cgroup.calibrate_rows(all_rows, board)
    assert err < 1

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


    with open(detection_path, 'rb') as f:
        all_rows = pickle.load(f)

    config = toml.load(config_path)
    board = get_calibration_board(config)

    err = cgroup.calibrate_rows(all_rows, board)
    assert err < 1
