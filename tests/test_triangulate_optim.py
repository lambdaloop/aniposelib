#!/usr/bin/env python

import pytest

import numpy as np
from aniposelib.cameras import Camera, CameraGroup

import os
import toml

def load_constraints(config, bodyparts, key='constraints'):
    constraints_names = config['triangulation'].get(key, [])
    bp_index = dict(zip(bodyparts, range(len(bodyparts))))
    constraints = []
    for a, b in constraints_names:
        assert a in bp_index, 'Bodypart {} from constraints not found in list of bodyparts'.format(a)
        assert b in bp_index, 'Bodypart {} from constraints not found in list of bodyparts'.format(b)
        con = [bp_index[a], bp_index[b]]
        constraints.append(con)
    return constraints

def load_offsets_dict(config, cam_names, video_folder=None):
    offsets_dict = dict()
    for cname in cam_names:
        if 'cameras' not in config or cname not in config['cameras']:
            offsets_dict[cname] = (0, 0)
        else:
            offsets_dict[cname] = tuple(config['cameras'][cname]['offset'])

    return offsets_dict

@pytest.mark.integration
def test_7cam_allen_triangulation():
    prefix = 'tests/data/triangulate/7cam-allen-mouse'
    n_pts = 1000
    
    calib_path = os.path.join(prefix, 'calibration.toml')
    config_path = os.path.join(prefix, 'config.toml')
    points_path = os.path.join(prefix, 'points.npz')

    config = toml.load(config_path)

    cgroup = CameraGroup.load(calib_path)
    d = np.load(points_path)

    bodyparts = list(d['bodyparts'])
    p2d = d['p2d'][:, :n_pts]
    p3d_ref = d['p3d'][:n_pts]
    scores = d['scores'][:, :n_pts]

    # threshold based on score
    p2d[scores < config['triangulation']['score_threshold']] = np.nan

    # add offsets
    offsets_dict = load_offsets_dict(config, cgroup.get_names())
    for i, cname in enumerate(cgroup.get_names()):
        offset = offsets_dict[cname]
        p2d[i, :, :, 0] += offset[0] 
        p2d[i, :, :, 1] += offset[1] 

    constraints = load_constraints(config, bodyparts)
    constraints_weak = load_constraints(config, bodyparts, 'constraints_weak')

    p3d_pred = cgroup.triangulate_optim(
        p2d, 
        constraints=constraints,
        constraints_weak=constraints_weak,
        scale_smooth=config['triangulation']['scale_smooth'],
        scale_length=config['triangulation']['scale_length'],
        scale_length_weak=config['triangulation']['scale_length_weak'],
        n_deriv_smooth=config['triangulation']['n_deriv_smooth'],
        reproj_error_threshold=config['triangulation'].get('reproj_error_threshold', 5),
        verbose=True)

    err = np.linalg.norm(p3d_pred - p3d_ref, axis=2)
    assert np.all(np.median(err, axis=0) < 2)

@pytest.mark.integration
def test_12cam_allen_triangulation():
    prefix = 'tests/data/triangulate/12cam-allen-mouse'
    n_pts = 1000

    calib_path = os.path.join(prefix, 'calibration.toml')
    config_path = os.path.join(prefix, 'config.toml')
    points_path = os.path.join(prefix, 'points.npz')

    config = toml.load(config_path)

    cgroup = CameraGroup.load(calib_path)
    d = np.load(points_path)

    bodyparts = list(d['bodyparts'])
    p2d = d['p2d'][:, :n_pts]
    p3d_ref = d['p3d'][:n_pts]
    scores = d['scores'][:, :n_pts]

    # threshold based on score
    p2d[scores < config['triangulation']['score_threshold']] = np.nan

    # add offsets
    offsets_dict = load_offsets_dict(config, cgroup.get_names())
    for i, cname in enumerate(cgroup.get_names()):
        offset = offsets_dict[cname]
        p2d[i, :, :, 0] += offset[0] 
        p2d[i, :, :, 1] += offset[1] 

    constraints = load_constraints(config, bodyparts)
    constraints_weak = load_constraints(config, bodyparts, 'constraints_weak')

    p3d_pred = cgroup.triangulate_optim(
        p2d, 
        constraints=constraints,
        constraints_weak=constraints_weak,
        scale_smooth=config['triangulation']['scale_smooth'],
        scale_length=config['triangulation']['scale_length'],
        scale_length_weak=config['triangulation']['scale_length_weak'],
        n_deriv_smooth=config['triangulation']['n_deriv_smooth'],
        reproj_error_threshold=config['triangulation'].get('reproj_error_threshold', 5),
        verbose=True)

    err = np.linalg.norm(p3d_pred - p3d_ref, axis=2)
    assert np.all(np.median(err, axis=0) < 2)
