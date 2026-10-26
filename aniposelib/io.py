"""Importers and exporters for camera calibrations (1.0, plan 01 phase G).

Every loader builds aniposelib cameras through :func:`camera_from_extrinsics_mat`, so the
world -> camera convention is the same everywhere: ``x_cam = R X + t`` (OpenCV), ``E`` is the
4x4 world-to-camera matrix, ``K`` is in sensor pixels with a 0-based pixel origin (the centre
of the top-left pixel is (0, 0)), and the full ``K`` (including skew ``K[0, 1]``) is used.

API
---
``camera_from_extrinsics_mat(K, E, dist=None, size=None, name=None, *, fisheye=False,
offset=None, extra=None) -> Camera``
    ``E`` is 4x4 (or 3x4) world -> camera. A rotation that is not quite orthonormal is
    projected onto SO(3); an improper one (det < 0) raises.
``reindex_points(points, from_names, to_names, missing='nan') -> points``
    Reorder the camera axis (axis 0) of ``points`` from one camera order to another. Cameras in
    ``to_names`` that are not in ``from_names`` become NaN (``missing='raise'`` raises).
``from_matlab(params, names=None, *, one_based=True) / to_matlab(cgroup, path=None, *,
one_based=True)``
    MATLAB ``cameraParameters`` / ``stereoParameters`` (as a dict, a list or dict of per-camera
    dicts, a ``.mat`` file of structs (``toStruct``), or a ``.json`` export). MATLAB stores
    ``IntrinsicMatrix`` and rotations transposed (row-vector convention ``x = X R + t``) and
    uses **1-based pixel coordinates**, so the principal point is shifted by -1 on import
    (``one_based=False`` keeps it, as the hand-written converters did). Stereo parameters give
    camera 1 at the origin and camera 2 from ``RotationOfCamera2``/``TranslationOfCamera2`` (or
    ``PoseCamera2``, a camera-to-world pose, inverted).
``read_opencv_yaml(path_or_text) / write_opencv_yaml(path, data)``
    OpenCV ``FileStorage`` YAML (``!!opencv-matrix``, scalars, strings, nested maps) without
    cv2: ``cv2.FileStorage`` silently misreads bare integers in ``dt: d`` matrices. Files
    written here are also readable by ``cv2.FileStorage``.
``from_opencv_yaml(paths, names=None, *, convention='auto', one_based=None) /
to_opencv_yaml(cam_or_group, path)``
    One camera per file (keys ``camera_matrix``/``K``/``intrinsicMatrix``/...,
    ``distortion_coefficients``/``D``/..., ``R`` or ``rvec``, ``T``/``tvec``,
    ``image_width``/``image_height``). ``convention='matlab'`` (auto-detected from a transposed
    K) transposes K and R and, by default, applies the 1-based principal point shift.
    A group is written as one file per camera into a directory.
``from_projection_matrix(P, size=None, name=None, dist=None) / to_projection_matrix(cam)``
    RQ decomposition of a finite 3x4 camera ``P ~ K [R | t]`` (skew kept). An affine camera
    (``P[2, :3] = 0``) gives an :class:`~aniposelib.affine.AffineCamera`.
``from_dlt(L, size=None, name=None, *, affine_tol=1e-12) / to_dlt(cam)``
    11 (or 12) DLT coefficients ``u = (L1 X + L2 Y + L3 Z + L4) / (L9 X + L10 Y + L11 Z + 1)``.
    ``|L9..L11| <= affine_tol`` -> :class:`~aniposelib.affine.AffineCamera`.
``from_metadata_yaml(path_or_dict, *, order='tracktail', return_moving=False) /
to_metadata_yaml(cgroup, path=None, *, moving=None)``
    tracktail ``metadata.yaml``: ``intrinsic_matrices``, ``extrinsic_matrices`` (world -> cam
    4x4, or ``(T,4,4)`` with ``moving_cams: true``), ``distortion_matrices``,
    ``camera_heights``/``camera_widths``, optional ``offset_dict`` (``{name: [ox, oy, ...]}``,
    applied as :attr:`Camera.offset`) and ``cam_type`` (``pinhole``/``fisheye``).
    ``order='tracktail'`` sorts names like tracktail (numerically when all names are digits,
    else lexically), so camera indices match tracktail arrays; ``'natural'`` and ``'file'``
    are also accepted. Reading needs PyYAML unless the file was written by
    :func:`to_metadata_yaml` (which writes a YAML subset that is read without it).
``to_json_rig(cgroup, path=None, *, session=None, extra=None) / from_json_rig(path_or_dict, *,
stations='first')``
    JSON rig compatible with the disco ``pose_ba_core`` rig files (``intrinsics`` +
    ``stations`` with ``R``, ``t``: ``x_cam = R X + t``); the full ``K`` and distortion vector
    are also stored, so skew and rational coefficients round-trip.
"""

import json
import os
import re
import warnings
from collections import OrderedDict

import numpy as np
import torch

from ._types import is_torch

__all__ = ['camera_from_extrinsics_mat', 'reindex_points',
           'from_matlab', 'to_matlab',
           'read_opencv_yaml', 'write_opencv_yaml', 'from_opencv_yaml', 'to_opencv_yaml',
           'from_projection_matrix', 'to_projection_matrix', 'from_dlt', 'to_dlt',
           'decompose_projection',
           'from_metadata_yaml', 'to_metadata_yaml', 'read_metadata_yaml',
           'to_json_rig', 'from_json_rig']


# ------------------------------------------------------------------ small helpers

def _np(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy().copy()  # never alias camera parameters
    return np.asarray(x)


def _f64(x):
    return np.array(_np(x), dtype='float64')


def _cameras(cgroup):
    if hasattr(cgroup, 'cameras'):
        return list(cgroup.cameras)
    if isinstance(cgroup, (list, tuple)):
        return list(cgroup)
    return [cgroup]


def _rvec_from_R(R):
    import cv2
    return cv2.Rodrigues(np.asarray(R, dtype='float64'))[0].ravel()


def _nearest_rotation(R, name=None):
    R = _f64(R).reshape(3, 3)
    if not np.all(np.isfinite(R)):
        raise ValueError('camera {}: rotation has non-finite entries'.format(name))
    U, s, Vt = np.linalg.svd(R)
    d = np.linalg.det(U @ Vt)
    if d < 0:
        raise ValueError('camera {}: rotation is improper (det < 0); is it transposed or '
                         'from a left-handed frame?'.format(name))
    if np.max(np.abs(s - 1)) > 1e-3:
        warnings.warn('camera {}: rotation is not orthonormal (singular values {}); using the '
                      'nearest rotation'.format(name, np.round(s, 6)), RuntimeWarning)
    return U @ Vt


def _extrinsics(cam):
    return _f64(cam.get_extrinsics_mat())


def _K(cam):
    return _f64(cam.get_camera_matrix())


def _dist(cam):
    return _f64(cam.get_distortions()).ravel()


def _size(cam):
    s = cam.get_size()
    if s is None:
        return None
    return [int(round(float(v))) for v in list(s)[:2]]


def _offset(cam):
    off = getattr(cam, 'offset', None)
    return [0.0, 0.0] if off is None else [float(v) for v in _f64(off).ravel()[:2]]


def _group(cams, metadata=None):
    from .cameras import CameraGroup
    return CameraGroup(cams, metadata=metadata)


def _order_names(names, order):
    names = list(names)
    if order == 'file':
        return names
    if order == 'tracktail':
        if all(str(n).isdigit() for n in names):
            return sorted(names, key=lambda n: int(n))
        return sorted(names, key=str)
    if order in ('natural', 'lexical'):
        from .cameras import _order_keys
        return _order_keys(names, order)
    raise ValueError("order must be 'tracktail', 'natural', 'lexical' or 'file', got {!r}".format(order))


# ------------------------------------------------------------------ camera construction

def camera_from_extrinsics_mat(K, E, dist=None, size=None, name=None, *, fisheye=False,
                               offset=None, extra=None, n_dist=None, dtype=torch.float64):
    """Build a :class:`~aniposelib.cameras.Camera` (or ``FisheyeCamera``) from a 3x3 ``K``, a
    4x4 (or 3x4) world -> camera matrix ``E`` (``x_cam = R X + t``), distortions and an image
    ``size`` (width, height). ``offset`` is the crop origin in sensor pixels."""
    from .cameras import Camera, FisheyeCamera
    K = _f64(K).reshape(3, 3)
    E = _f64(E)
    if E.shape not in ((4, 4), (3, 4)):
        raise ValueError('extrinsics must be 4x4 or 3x4, got {}'.format(E.shape))
    R = _nearest_rotation(E[:3, :3], name)
    t = E[:3, 3]
    cls = FisheyeCamera if fisheye else Camera
    if dist is None:
        dist = np.zeros(cls.n_dist_model)
    dist = _f64(dist).ravel()
    if len(dist) < cls.n_dist_model:
        dist = np.concatenate([dist, np.zeros(cls.n_dist_model - len(dist))])
    kw = dict(matrix=K, dist=dist, size=None if size is None else [int(round(float(s))) for s in list(size)[:2]],
              rvec=_rvec_from_R(R), tvec=t, name=name, offset=offset, extra=extra, dtype=dtype)
    if n_dist is not None:
        kw['n_dist'] = int(n_dist)
    return cls(**kw)


def reindex_points(points, from_names, to_names, missing='nan'):
    """Reorder axis 0 (cameras) of ``points`` from camera order ``from_names`` to ``to_names``.

    ``from_names``/``to_names`` may be name lists or ``CameraGroup`` s. Cameras of ``to_names``
    absent from ``from_names`` are filled with NaN (``missing='nan'``) or raise
    (``missing='raise'``). numpy in -> numpy out, torch in -> torch out."""
    def names_of(x):
        return list(x.get_names()) if hasattr(x, 'get_names') else [str(n) for n in x]
    src, dst = names_of(from_names), names_of(to_names)
    if len(set(src)) != len(src) or len(set(dst)) != len(dst):
        raise ValueError('camera names must be unique')
    if missing not in ('nan', 'raise'):
        raise ValueError("missing must be 'nan' or 'raise'")
    if len(points) != len(src):
        raise ValueError('points have {} cameras, from_names has {}'.format(len(points), len(src)))
    lookup = {n: i for i, n in enumerate(src)}
    absent = [n for n in dst if n not in lookup]
    if absent and missing == 'raise':
        raise KeyError('cameras not in from_names: {}'.format(absent))
    if is_torch(points):
        out = torch.full((len(dst),) + tuple(points.shape[1:]), float('nan'),
                         dtype=points.dtype if points.is_floating_point() else torch.float64,
                         device=points.device)
        for j, n in enumerate(dst):
            if n in lookup:
                out[j] = points[lookup[n]]
        return out
    points = np.asarray(points)
    dt = points.dtype if points.dtype.kind in 'fc' else np.float64
    out = np.full((len(dst),) + points.shape[1:], np.nan, dtype=dt)
    for j, n in enumerate(dst):
        if n in lookup:
            out[j] = points[lookup[n]]
    return out


# ------------------------------------------------------------------ MATLAB

_ML_KEYS = {
    'K': ('k',),
    'IntrinsicMatrix': ('intrinsicmatrix', 'intrinsic_matrix_matlab', 'intrinsic_matrix'),
    'FocalLength': ('focallength',),
    'PrincipalPoint': ('principalpoint',),
    'Skew': ('skew',),
    'RadialDistortion': ('radialdistortion', 'radial'),
    'TangentialDistortion': ('tangentialdistortion', 'tangential'),
    'ImageSize': ('imagesize', 'image_size_hw'),
    'RotationMatrix': ('rotationmatrix', 'rotation_matlab', 'rotationofcamera2'),
    'TranslationVector': ('translationvector', 'translation_matlab', 'translationofcamera2'),
    'Pose': ('pose', 'posecamera2', 'absolutepose'),
    'Intrinsics': ('intrinsics',),
    'Name': ('name', 'cameraname'),
}


def _ml_get(d, key):
    low = {str(k).lower(): k for k in d}
    for alias in _ML_KEYS[key]:
        if alias in low:
            v = d[low[alias]]
            if v is not None and not (isinstance(v, (list, np.ndarray)) and np.size(v) == 0):
                return v
    return None


def _upper_K(K):
    """MATLAB IntrinsicMatrix is K.T; return the upper-triangular (OpenCV) form either way."""
    K = _f64(K).reshape(3, 3)
    lower = np.any(K[2, :2] != 0) and np.all(K[:2, 2] == 0)
    return K.T.copy() if lower else K.copy()


def _is_ml_camera(d):
    return isinstance(d, dict) and any(_ml_get(d, k) is not None for k in
                                       ('K', 'IntrinsicMatrix', 'FocalLength', 'Intrinsics'))


def _ml_pose_to_ext(pose):
    """MATLAB rigidtform3d pose (camera -> world, premultiply) -> world -> camera 4x4."""
    if isinstance(pose, dict):
        low = {str(k).lower(): k for k in pose}
        if 'a' in low:
            A = _f64(pose[low['a']]).reshape(4, 4)
            R, t = A[:3, :3], A[:3, 3]
        else:
            R = _f64(pose[low['r']]).reshape(3, 3)
            t = _f64(pose[low['translation']]).ravel()
    else:
        A = _f64(pose)
        R, t = A[:3, :3], A[:3, 3]
    E = np.eye(4)
    E[:3, :3] = R.T
    E[:3, 3] = -R.T @ t
    return E


def _ml_camera(d, name, one_based):
    intr = _ml_get(d, 'Intrinsics')
    src = intr if isinstance(intr, dict) else d
    K = _ml_get(src, 'K')
    if K is not None:
        K = _upper_K(K)
    else:
        IM = _ml_get(src, 'IntrinsicMatrix')
        if IM is not None:
            K = _upper_K(IM)
        else:
            f = _f64(_ml_get(src, 'FocalLength')).ravel()
            c = _f64(_ml_get(src, 'PrincipalPoint')).ravel()
            s = _ml_get(src, 'Skew')
            fx, fy = (f[0], f[-1])
            K = np.array([[fx, 0.0 if s is None else float(np.ravel(s)[0]), c[0]],
                          [0, fy, c[1]], [0, 0, 1.0]])
    if one_based:
        K[0, 2] -= 1.0
        K[1, 2] -= 1.0
    rad = _ml_get(src, 'RadialDistortion')
    tan = _ml_get(src, 'TangentialDistortion')
    rad = np.zeros(0) if rad is None else _f64(rad).ravel()
    tan = np.zeros(0) if tan is None else _f64(tan).ravel()
    if len(rad) > 3:
        raise ValueError('camera {}: MATLAB radial distortion has {} terms (max 3)'.format(name, len(rad)))
    k1, k2, k3 = np.pad(rad, (0, 3 - len(rad)))
    p1, p2 = np.pad(tan[:2], (0, 2 - len(tan[:2])))
    dist = np.array([k1, k2, p1, p2, k3])
    size = _ml_get(src, 'ImageSize')
    if size is None and src is not d:
        size = _ml_get(d, 'ImageSize')
    if size is not None:
        h, w = [int(round(v)) for v in _f64(size).ravel()[:2]]
        size = (w, h)
    E = np.eye(4)
    Rm = _ml_get(d, 'RotationMatrix')
    pose = _ml_get(d, 'Pose')
    if Rm is not None:
        Rm = _f64(Rm)
        if Rm.ndim == 3:
            raise ValueError('camera {}: RotationMatrix is a stack (board poses?); pass the '
                             'camera extrinsics'.format(name))
        E[:3, :3] = Rm.reshape(3, 3).T
        tv = _ml_get(d, 'TranslationVector')
        if tv is not None:
            E[:3, 3] = _f64(tv).ravel()
    elif pose is not None:
        E = _ml_pose_to_ext(pose)
    nm = _ml_get(d, 'Name')
    name = name if name is not None else (None if nm is None else str(nm))
    return camera_from_extrinsics_mat(K, E, dist, size, name)


def _load_struct_file(path):
    path = str(path)
    if path.lower().endswith('.json'):
        with open(path) as f:
            return json.load(f)
    from scipy.io import loadmat
    try:
        d = loadmat(path, simplify_cells=True)
    except NotImplementedError as e:  # v7.3 (HDF5) files
        raise ValueError('{}: MATLAB v7.3 files are not supported by scipy; save with -v7 or '
                         'export with toStruct/jsonencode'.format(path)) from e
    d = {k: v for k, v in d.items() if not k.startswith('__')}
    if not d:
        raise ValueError('{}: no readable variables (MATLAB objects such as cameraParameters '
                         'cannot be read by scipy; save toStruct(params) instead)'.format(path))
    if len(d) == 1:
        return next(iter(d.values()))
    return d


def from_matlab(params, names=None, *, one_based=True):
    """MATLAB calibration -> :class:`~aniposelib.cameras.CameraGroup` (see module docstring).

    ``params``: a ``stereoParameters``/``cameraParameters``-like dict (``toStruct`` fields),
    a list of per-camera dicts, a ``{name: dict}`` mapping, or a path to a ``.mat``/``.json``
    file holding one of these. Per-camera extrinsics come from ``RotationMatrix`` (MATLAB,
    transposed) + ``TranslationVector``, or a ``Pose`` (camera -> world). Old behaviour of the
    hand-written converters (no 1-based shift): ``one_based=False``."""
    if isinstance(params, (str, os.PathLike)):
        params = _load_struct_file(params)
    if isinstance(params, np.ndarray) and params.dtype == object:
        params = list(params.ravel())
    cams = []
    if isinstance(params, dict) and 'CameraParameters1' in params:
        n1, n2 = (names or ['cam1', 'cam2'])[:2]
        c1 = dict(params['CameraParameters1'])
        c2 = dict(params['CameraParameters2'])
        for k in [k for k in c1 if k.lower() in ('rotationmatrix', 'translationvector', 'pose')]:
            del c1[k]
        for k in [k for k in c2 if k.lower() in ('rotationmatrix', 'translationvector', 'pose')]:
            del c2[k]
        for key in ('RotationOfCamera2', 'TranslationOfCamera2', 'PoseCamera2'):
            if key in params:
                c2[key] = params[key]
        cams = [_ml_camera(c1, n1, one_based), _ml_camera(c2, n2, one_based)]
    elif _is_ml_camera(params):
        cams = [_ml_camera(params, None if names is None else names[0], one_based)]
    elif isinstance(params, dict):
        keys = list(params)
        for i, k in enumerate(keys):
            cams.append(_ml_camera(params[k], str(k) if names is None else names[i], one_based))
    elif isinstance(params, (list, tuple)):
        for i, d in enumerate(params):
            cam = _ml_camera(d, None if names is None else names[i], one_based)
            if cam.get_name() in (None, 'None'):
                cam.set_name('cam{}'.format(i + 1))
            cams.append(cam)
    else:
        raise TypeError('unsupported MATLAB parameters of type {}'.format(type(params).__name__))
    for i, c in enumerate(cams):
        if c.get_name() in (None, 'None'):
            c.set_name('cam{}'.format(i + 1))
    return _group(cams)


def to_matlab(cgroup, path=None, *, one_based=True):
    """CameraGroup -> list of MATLAB-convention camera dicts (``IntrinsicMatrix`` = K.T,
    ``K``, ``FocalLength``, ``PrincipalPoint`` (1-based), ``Skew``, ``RadialDistortion``,
    ``TangentialDistortion``, ``ImageSize`` = [h w], ``RotationMatrix`` = R.T,
    ``TranslationVector``). With ``path`` it is also written (``.mat`` via scipy as a cell
    ``cameras``, or ``.json``)."""
    out = []
    for cam in _cameras(cgroup):
        if getattr(cam, 'fisheye', False):
            raise ValueError('camera {}: MATLAB fisheye (Scaramuzza) intrinsics are a different '
                             'model; only pinhole cameras can be exported'.format(cam.get_name()))
        K = _K(cam)
        if one_based:
            K[0, 2] += 1.0
            K[1, 2] += 1.0
        d = np.pad(_dist(cam), (0, 8))[:8]
        if np.any(d[5:] != 0):
            raise ValueError('camera {}: rational distortion (k4..k6) has no MATLAB '
                             'equivalent'.format(cam.get_name()))
        E = _extrinsics(cam)
        size = _size(cam)
        c = OrderedDict(Name=cam.get_name(), K=K.tolist(), IntrinsicMatrix=K.T.tolist(),
                        FocalLength=[K[0, 0], K[1, 1]], PrincipalPoint=[K[0, 2], K[1, 2]],
                        Skew=float(K[0, 1]), RadialDistortion=[d[0], d[1], d[4]],
                        TangentialDistortion=[d[2], d[3]],
                        RotationMatrix=E[:3, :3].T.tolist(), TranslationVector=E[:3, 3].tolist())
        if size is not None:
            c['ImageSize'] = [size[1], size[0]]
        out.append(dict(c))
    if path is not None:
        path = str(path)
        if path.lower().endswith('.json'):
            with open(path, 'w') as f:
                json.dump(out, f, indent=1)
        else:
            from scipy.io import savemat
            cell = np.empty(len(out), dtype=object)
            for i, c in enumerate(out):
                cell[i] = {k: np.asarray(v, dtype='float64') if k != 'Name' else v for k, v in c.items()}
            savemat(path, {'cameras': cell})
    return out


# ------------------------------------------------------------------ OpenCV FileStorage YAML

_NUM = re.compile(r'[-+]?(?:\.(?:nan|inf)|(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)', re.I)
_KEYLINE = re.compile(r'^(\s*)([^\s:#][^:#]*?)\s*:(?:\s+(.*?))?\s*$')


def _num(s):
    s = s.strip()
    low = s.lower().lstrip('+')
    if low in ('.nan', '-.nan'):
        return float('nan')
    if low in ('.inf', '-.inf'):
        return float('inf') if not low.startswith('-') else float('-inf')
    return float(s)


def _scalar(v):
    v = v.strip()
    if len(v) >= 2 and v[0] == v[-1] and v[0] in '"\'':
        return v[1:-1]
    if v.startswith('[') and v.endswith(']'):
        inner = v[1:-1].strip()
        return [] if not inner else [_scalar(x) for x in inner.split(',')]
    if re.fullmatch(r'[-+]?\d+', v):
        return int(v)
    if _NUM.fullmatch(v):
        return _num(v)
    return v


def _parse_fs(lines, i, indent):
    out = OrderedDict()
    n = len(lines)
    while i < n:
        raw = lines[i]
        s = raw.split('#', 1)[0].rstrip() if not raw.lstrip().startswith(('"', "'")) else raw.rstrip()
        if not s.strip() or s.startswith('%') or s.strip() == '---':
            i += 1
            continue
        ind = len(s) - len(s.lstrip())
        if ind < indent:
            break
        m = _KEYLINE.match(s)
        if not m:
            raise ValueError('cannot parse OpenCV YAML line {}: {!r}'.format(i + 1, raw))
        key, val = m.group(2).strip().strip('"\''), (m.group(3) or '')
        i += 1
        if val.startswith('!!opencv-matrix') or val.startswith('!!opencv-nd-matrix'):
            sub, i = _parse_fs(lines, i, ind + 1)
            data = sub.get('data', [])
            if isinstance(data, str):
                data = [_num(x) for x in _NUM.findall(data)]
            data = np.asarray(data if isinstance(data, list) else [data], dtype='float64')
            if 'rows' in sub:
                shape = (int(sub['rows']), int(sub['cols']))
            else:
                shape = tuple(int(x) for x in np.ravel(sub.get('sizes', [len(data)])))
            out[key] = data.reshape(shape)
        elif val.startswith('[') and not val.rstrip().endswith(']'):
            # multi-line flow sequence
            buf = [val]
            while i < n and ']' not in buf[-1]:
                buf.append(lines[i].strip())
                i += 1
            txt = ' '.join(buf)
            out[key] = [_num(x) for x in _NUM.findall(txt)]
        elif val == '':
            # nested map (or empty)
            j = i
            while j < n and not lines[j].strip():
                j += 1
            if j < n and (len(lines[j]) - len(lines[j].lstrip())) > ind:
                sub, i = _parse_fs(lines, j, len(lines[j]) - len(lines[j].lstrip()))
                out[key] = sub
            else:
                out[key] = None
        else:
            if val.startswith('['):
                vals = _scalar(val)
                out[key] = vals
            else:
                out[key] = _scalar(val)
    return out, i


def read_opencv_yaml(path_or_text):
    """Parse an OpenCV ``FileStorage`` YAML file (or its text) into a dict; matrices become
    float64 arrays. Does not use cv2 (which misreads bare integers in ``dt: d`` data)."""
    s = str(path_or_text)
    if '\n' not in s and os.path.exists(s):
        with open(s) as f:
            s = f.read()
    text = s.replace('\t', '    ').splitlines()
    out, _ = _parse_fs(text, 0, 0)
    return dict(out)


def _fmt_float(v):
    v = float(v)
    if np.isnan(v):
        return '.Nan'
    if np.isinf(v):
        return '.Inf' if v > 0 else '-.Inf'
    s = '{:.17g}'.format(v)
    if not any(c in s for c in '.eEn'):
        s += '.'
    return s


def _write_fs(f, data, indent):
    pad = ' ' * indent
    for k, v in data.items():
        if isinstance(v, dict):
            f.write('{}{}:\n'.format(pad, k))
            _write_fs(f, v, indent + 3)
        elif isinstance(v, (np.ndarray, torch.Tensor)) or (isinstance(v, (list, tuple)) and len(v)
                                                           and not isinstance(v[0], str)):
            a = np.atleast_2d(_f64(v))
            if a.ndim > 2:
                raise ValueError('only 2-D matrices are supported ({}: {})'.format(k, a.shape))
            if a.shape[0] == 1 and np.ndim(_np(v)) == 1:
                a = a.T  # vectors as column matrices, like cv2
            vals = [_fmt_float(x) for x in a.ravel()]
            rows = [', '.join(vals[i:i + 4]) for i in range(0, len(vals), 4)]
            f.write('{}{}: !!opencv-matrix\n{p}   rows: {}\n{p}   cols: {}\n{p}   dt: d\n'
                    '{p}   data: [ {} ]\n'.format(pad, k, a.shape[0], a.shape[1],
                                                 (',\n' + pad + '       ').join(rows), p=pad))
        elif isinstance(v, (list, tuple)):
            f.write('{}{}: [ {} ]\n'.format(pad, k, ', '.join('"{}"'.format(x) for x in v)))
        elif isinstance(v, (bool, np.bool_)):
            f.write('{}{}: {}\n'.format(pad, k, int(v)))
        elif isinstance(v, (int, np.integer)):
            f.write('{}{}: {}\n'.format(pad, k, int(v)))
        elif isinstance(v, (float, np.floating)):
            f.write('{}{}: {}\n'.format(pad, k, _fmt_float(v)))
        elif v is None:
            continue
        else:
            f.write('{}{}: "{}"\n'.format(pad, k, str(v)))


def write_opencv_yaml(path, data):
    """Write ``data`` (a dict of arrays, numbers, strings and nested dicts) as an OpenCV
    ``FileStorage`` YAML file (readable by ``cv2.FileStorage`` and :func:`read_opencv_yaml`)."""
    with open(str(path), 'w') as f:
        f.write('%YAML:1.0\n---\n')
        _write_fs(f, data, 0)


_CV_KEYS = {
    'K': ('camera_matrix', 'cameramatrix', 'k', 'intrinsicmatrix', 'intrinsic_matrix', 'm',
          'mtx', 'intrinsics'),
    'dist': ('distortion_coefficients', 'distortioncoefficients', 'dist_coeffs', 'distcoeffs',
             'd', 'dist', 'distortion'),
    'R': ('r', 'rotation_matrix', 'rotationmatrix'),
    'rvec': ('rvec', 'rotation', 'rotation_vector'),
    'T': ('t', 'tvec', 'translation', 'translation_vector', 'translationvector'),
    'width': ('image_width', 'width', 'imagewidth'),
    'height': ('image_height', 'height', 'imageheight'),
    'size': ('image_size', 'imagesize', 'size'),
    'name': ('camera_name', 'name', 'cameraname'),
    'model': ('distortion_model', 'model', 'cam_type'),
    'offset': ('offset', 'crop_offset'),
}


def _cv_get(d, key):
    low = {str(k).lower(): k for k in d}
    for alias in _CV_KEYS[key]:
        if alias in low and d[low[alias]] is not None:
            return d[low[alias]]
    return None


def _opencv_camera(d, name, convention, one_based):
    Kraw = _cv_get(d, 'K')
    if Kraw is None:
        raise KeyError('no camera matrix (camera_matrix / K / intrinsicMatrix) in {}'.format(list(d)))
    Kraw = _f64(Kraw).reshape(3, 3)
    transposed = bool(np.any(Kraw[2, :2] != 0) and np.all(Kraw[:2, 2] == 0))
    conv = convention
    if conv == 'auto':
        conv = 'matlab' if transposed else 'opencv'
    if conv not in ('opencv', 'matlab'):
        raise ValueError("convention must be 'auto', 'opencv' or 'matlab'")
    if one_based is None:
        one_based = conv == 'matlab'
    K = _upper_K(Kraw)
    if one_based:
        K[0, 2] -= 1.0
        K[1, 2] -= 1.0
    dist = _cv_get(d, 'dist')
    dist = np.zeros(5) if dist is None else _f64(dist).ravel()
    E = np.eye(4)
    R = _cv_get(d, 'R')
    rv = _cv_get(d, 'rvec')
    if R is not None and np.size(R) == 9:
        R = _f64(R).reshape(3, 3)
        E[:3, :3] = R.T if conv == 'matlab' else R
    elif R is not None and np.size(R) == 3:
        rv = R
    if (R is None or np.size(R) == 3) and rv is not None:
        import cv2
        E[:3, :3] = cv2.Rodrigues(_f64(rv).reshape(3, 1))[0]
    T = _cv_get(d, 'T')
    if T is not None:
        E[:3, 3] = _f64(T).ravel()[:3]
    w, h = _cv_get(d, 'width'), _cv_get(d, 'height')
    size = None
    if w is not None and h is not None and int(w) > 0 and int(h) > 0:
        size = (int(w), int(h))
    elif _cv_get(d, 'size') is not None:
        s = _f64(_cv_get(d, 'size')).ravel()
        if len(s) >= 2 and s[0] > 0:
            size = (int(s[0]), int(s[1]))
    nm = _cv_get(d, 'name')
    model = _cv_get(d, 'model')
    fisheye = isinstance(model, str) and model.lower() in ('fisheye', 'equidistant', 'kb', 'kannala_brandt')
    off = _cv_get(d, 'offset')
    off = None if off is None else _f64(off).ravel()[:2]
    return camera_from_extrinsics_mat(K, E, dist, size, name if name is not None else
                                      (None if nm is None else str(nm)), fisheye=fisheye, offset=off)


def from_opencv_yaml(paths, names=None, *, convention='auto', one_based=None):
    """OpenCV YAML calibration file(s), one camera per file, -> CameraGroup.

    ``paths``: a file, a directory (``*.yaml``/``*.yml``/``*.xml``-free, natural order), a list
    of files or ``{name: file}``. Camera names come from ``names``, the file's
    ``camera_name`` or the file stem. ``convention='matlab'`` (auto-detected when K is stored
    transposed, as in MATLAB exports) transposes K and R; ``one_based`` (default: True for the
    MATLAB convention) shifts the principal point by -1."""
    if isinstance(paths, dict):
        items = [(str(k), str(v)) for k, v in paths.items()]
    else:
        if isinstance(paths, (str, os.PathLike)):
            p = str(paths)
            if os.path.isdir(p):
                from .cameras import _order_keys
                files = [f for f in os.listdir(p) if f.lower().endswith(('.yaml', '.yml'))]
                stems = {os.path.splitext(f)[0]: f for f in files}
                paths = [os.path.join(p, stems[s]) for s in _order_keys(list(stems), 'natural')]
            else:
                paths = [p]
        items = [(None, str(p)) for p in paths]
    cams = []
    for i, (nm, p) in enumerate(items):
        d = read_opencv_yaml(p)
        if names is not None:
            nm = names[i]
        cam = _opencv_camera(d, nm, convention, one_based)
        if cam.get_name() in (None, 'None'):
            cam.set_name(os.path.splitext(os.path.basename(p))[0])
        cams.append(cam)
    return _group(cams)


def _opencv_dict(cam):
    E = _extrinsics(cam)
    d = OrderedDict(camera_name=cam.get_name())
    size = _size(cam)
    if size is not None:
        d['image_width'], d['image_height'] = size
    d['distortion_model'] = 'fisheye' if getattr(cam, 'fisheye', False) else 'plumb_bob'
    d['camera_matrix'] = _K(cam)
    d['distortion_coefficients'] = _dist(cam)[None, :]
    d['R'] = E[:3, :3]
    d['rvec'] = _f64(cam.get_rotation()).ravel()
    d['T'] = E[:3, 3]
    off = _offset(cam)
    if any(off):
        d['offset'] = np.asarray(off)[None, :]
    return d


def to_opencv_yaml(cam_or_group, path):
    """Write a Camera to one OpenCV YAML file, or a CameraGroup to ``path/<name>.yaml`` (one
    file per camera; ``path`` is created). Returns the list of files written."""
    cams = _cameras(cam_or_group)
    path = str(path)
    if not hasattr(cam_or_group, 'cameras') and len(cams) == 1 and not os.path.isdir(path):
        write_opencv_yaml(path, _opencv_dict(cams[0]))
        return [path]
    os.makedirs(path, exist_ok=True)
    files = []
    for cam in cams:
        f = os.path.join(path, '{}.yaml'.format(cam.get_name()))
        write_opencv_yaml(f, _opencv_dict(cam))
        files.append(f)
    return files


# ------------------------------------------------------------------ projection matrices / DLT

def decompose_projection(P):
    """RQ-decompose a finite 3x4 camera ``P ~ K [R | t]`` -> ``(K, R, t)`` with K
    upper-triangular, positive diagonal, ``K[2, 2] = 1`` (skew kept) and R a proper rotation."""
    from scipy.linalg import rq
    P = _f64(P).reshape(3, 4)
    M = P[:, :3]
    if abs(np.linalg.det(M)) < 1e-300 or not np.all(np.isfinite(P)):
        raise ValueError('projection matrix is singular (affine camera?)')
    if np.linalg.det(M) < 0:
        P = -P
        M = P[:, :3]
    K, R = rq(M)
    S = np.diag(np.sign(np.diag(K)))
    K, R = K @ S, S @ R
    lam = K[2, 2]
    t = np.linalg.solve(K, P[:, 3])
    K = K / lam
    if np.linalg.det(R) < 0:  # pragma: no cover (det(M) > 0 and diag(K) > 0 imply det(R) > 0)
        raise ValueError('decomposition gave an improper rotation')
    return K, R, t


def _is_affine_P(P, tol):
    P = _f64(P).reshape(3, 4)
    return np.linalg.norm(P[2, :3]) <= tol * max(abs(P[2, 3]), 1e-300)


def from_projection_matrix(P, size=None, name=None, dist=None, *, affine_tol=1e-12):
    """3x4 projection matrix -> Camera (RQ decomposition, skew kept), or an
    :class:`~aniposelib.affine.AffineCamera` when ``|P[2,:3]| <= affine_tol * |P[2,3]|``."""
    P = _f64(P).reshape(3, 4)
    if _is_affine_P(P, affine_tol):
        from .affine import AffineCamera
        return AffineCamera.from_projection_matrix(P, size=size, name=name)
    K, R, t = decompose_projection(P)
    E = np.eye(4)
    E[:3, :3], E[:3, 3] = R, t
    return camera_from_extrinsics_mat(K, E, dist, size, name)


def to_projection_matrix(cam):
    """``K [R | t]`` (3x4, ``P[2,3]`` not normalised; distortion and offset are not included)."""
    if hasattr(cam, 'get_projection_matrix'):
        return cam.get_projection_matrix()
    return _K(cam) @ _extrinsics(cam)[:3]


def _dlt_to_P(L):
    L = _f64(L).ravel()
    if len(L) == 11:
        L = np.concatenate([L, [1.0]])
    if len(L) != 12:
        raise ValueError('DLT needs 11 or 12 coefficients, got {}'.format(len(L)))
    return L.reshape(3, 4)


def from_dlt(L, size=None, name=None, dist=None, *, affine_tol=1e-12):
    """11 DLT coefficients (``L12 = 1``; 12 are accepted) -> Camera, or
    :class:`~aniposelib.affine.AffineCamera` when ``|L9..L11| <= affine_tol * |L12|``.
    The DLT is taken to be in 0-based pixels (subtract 1 from ``cx``, i.e. use
    ``L1..4 -= L9..12`` yourself, for 1-based MATLAB / Hedrick DLTdv coefficients)."""
    return from_projection_matrix(_dlt_to_P(L), size=size, name=name, dist=dist,
                                  affine_tol=affine_tol)


def to_dlt(cam):
    """Camera (or AffineCamera) -> 11 DLT coefficients (normalised so ``L12 = 1``)."""
    P = to_projection_matrix(cam)
    if abs(P[2, 3]) < 1e-300:
        raise ValueError('P[2,3] is zero (world origin on the principal plane); not a DLT')
    return (P / P[2, 3]).ravel()[:11]


# ------------------------------------------------------------------ tracktail metadata.yaml

def _mini_yaml_dump(d):
    """A YAML subset: block mappings (2 levels), flow (JSON) values. Valid YAML."""
    lines = []
    for k, v in d.items():
        if isinstance(v, dict):
            lines.append('{}:'.format(k))
            for k2, v2 in v.items():
                lines.append('  {}: {}'.format(json.dumps(str(k2)), json.dumps(v2)))
        else:
            lines.append('{}: {}'.format(k, json.dumps(v)))
    return '\n'.join(lines) + '\n'


def _mini_yaml_load(text):
    out = OrderedDict()
    cur = None
    for ln, raw in enumerate(text.splitlines()):
        if not raw.strip() or raw.lstrip().startswith('#'):
            continue
        if raw[0] not in ' \t':
            m = re.match(r'^([^:]+):\s*(.*)$', raw)
            if not m:
                raise ValueError('line {}'.format(ln + 1))
            k, v = m.group(1).strip().strip('"\''), m.group(2).strip()
            if v == '':
                cur = out[k] = OrderedDict()
            else:
                out[k] = json.loads(v)
                cur = None
        else:
            if cur is None:
                raise ValueError('line {}'.format(ln + 1))
            m = re.match(r'^\s+("(?:[^"\\]|\\.)*"|[^:]+):\s*(.+)$', raw)
            if not m:
                raise ValueError('line {}'.format(ln + 1))
            k = m.group(1).strip()
            k = json.loads(k) if k.startswith('"') else k
            cur[k] = json.loads(m.group(2))
    return out


def read_metadata_yaml(path_or_text):
    """Load a ``metadata.yaml`` (PyYAML ``safe_load`` if installed; otherwise the subset
    written by :func:`to_metadata_yaml`)."""
    s = str(path_or_text)
    if '\n' not in s and os.path.exists(s):
        with open(s) as f:
            s = f.read()
    try:
        import yaml
    except ImportError:
        try:
            return _mini_yaml_load(s)
        except (ValueError, json.JSONDecodeError) as e:
            raise ImportError('reading this metadata.yaml needs PyYAML (pip install pyyaml)') from e
    return yaml.safe_load(s)


def from_metadata_yaml(path_or_dict, *, order='tracktail', return_moving=False):
    """tracktail ``metadata.yaml`` -> CameraGroup (see module docstring).

    ``offset_dict`` becomes :attr:`Camera.offset` (aniposelib then subtracts it in ``project``,
    as tracktail does). ``cam_type: fisheye`` gives ``FisheyeCamera`` s. With
    ``moving_cams: true`` the cameras get their frame-0 pose and ``extra['moving'] = True``;
    ``return_moving=True`` returns ``(cgroup, {name: (T,4,4) world->cam})`` (otherwise a
    warning says the per-frame poses were dropped)."""
    meta = path_or_dict if isinstance(path_or_dict, dict) else read_metadata_yaml(path_or_dict)
    intr = meta['intrinsic_matrices']
    ext = meta['extrinsic_matrices']
    dist = meta.get('distortion_matrices') or {}
    heights = meta.get('camera_heights') or {}
    widths = meta.get('camera_widths') or {}
    offsets = meta.get('offset_dict') or {}
    cam_type = str(meta.get('cam_type', 'pinhole') or 'pinhole').lower()
    if cam_type not in ('pinhole', 'fisheye'):
        raise ValueError('unsupported cam_type {!r}'.format(cam_type))
    moving = bool(meta.get('moving_cams', False))

    def lookup(d, n):
        if n in d:
            return d[n]
        for k in d:  # yaml may have parsed digit names as ints
            if str(k) == n:
                return d[k]
        return None

    names = _order_names([str(k) for k in intr.keys()], order)
    cams, moving_ext = [], {}
    for n in names:
        K = _f64(lookup(intr, n))
        E = _f64(lookup(ext, n))
        d = lookup(dist, n)
        d = None if d is None else _f64(d)
        if moving:
            if E.ndim != 3 or E.shape[-2:] != (4, 4):
                raise ValueError('moving_cams is true but extrinsics of {} have shape {} '
                                 '(expected (T,4,4))'.format(n, E.shape))
            moving_ext[n] = E
            E = E[0]
            if K.ndim == 3:
                K = K[0]
            if d is not None and d.ndim == 2:
                d = d[0]
        w, h = lookup(widths, n), lookup(heights, n)
        size = None if w is None or h is None else (int(w), int(h))
        off = lookup(offsets, n)
        off = None if off is None else _f64(off).ravel()[:2]
        extra = {'moving': True} if moving else None
        cams.append(camera_from_extrinsics_mat(K, E, d, size, n, fisheye=cam_type == 'fisheye',
                                               offset=off, extra=extra))
    cg = _group(cams)
    if return_moving:
        return cg, (moving_ext if moving else None)
    if moving:
        warnings.warn('metadata has moving_cams: true; the cameras hold the frame-0 pose only '
                      '(pass return_moving=True to get the per-frame extrinsics)', RuntimeWarning)
    return cg


def to_metadata_yaml(cgroup, path=None, *, moving=None):
    """CameraGroup -> tracktail ``metadata.yaml`` dict (written to ``path`` if given).

    ``moving``: optional ``{name: (T,4,4)}`` per-frame world->cam extrinsics; sets
    ``moving_cams: true`` (cameras not in it are tiled to ``(T,4,4)``). Every camera must have
    a size; mixing pinhole and fisheye cameras raises (``cam_type`` is per file)."""
    cams = _cameras(cgroup)
    kinds = {bool(getattr(c, 'fisheye', False)) for c in cams}
    if len(kinds) > 1:
        raise ValueError('tracktail metadata has one cam_type per file; the group mixes '
                         'pinhole and fisheye cameras')
    out = OrderedDict()
    out['intrinsic_matrices'] = OrderedDict()
    out['extrinsic_matrices'] = OrderedDict()
    out['distortion_matrices'] = OrderedDict()
    out['camera_heights'] = OrderedDict()
    out['camera_widths'] = OrderedDict()
    T = None
    if moving:
        T = {int(np.shape(v)[0]) for v in moving.values()}
        if len(T) != 1:
            raise ValueError('moving extrinsics have different lengths: {}'.format(sorted(T)))
        T = T.pop()
    offs = OrderedDict()
    for c in cams:
        n = c.get_name()
        size = _size(c)
        if size is None:
            raise ValueError('camera {} has no size; tracktail needs camera_widths/heights'.format(n))
        out['intrinsic_matrices'][n] = _K(c).tolist()
        if T is not None:
            E = _f64(moving[n]) if n in moving else np.repeat(_extrinsics(c)[None], T, 0)
            if E.shape != (T, 4, 4):
                raise ValueError('moving extrinsics of {} have shape {}'.format(n, E.shape))
            out['extrinsic_matrices'][n] = E.tolist()
        else:
            out['extrinsic_matrices'][n] = _extrinsics(c).tolist()
        out['distortion_matrices'][n] = _dist(c).tolist()
        out['camera_widths'][n] = int(size[0])
        out['camera_heights'][n] = int(size[1])
        offs[n] = _offset(c)
    if any(any(v) for v in offs.values()):
        out['offset_dict'] = offs
    out['cam_type'] = 'fisheye' if kinds == {True} else 'pinhole'
    if T is not None:
        out['moving_cams'] = True
    out = dict(out)
    if path is not None:
        with open(str(path), 'w') as f:
            f.write(_mini_yaml_dump(out))
    return out


# ------------------------------------------------------------------ JSON rig

_JSON_CONVENTION = ('x_cam = R X_world + t; pixel = K [xd, yd, 1] with (xd, yd) = distort(x/z, y/z) '
                    '(K full 3x3, 0-based pixels). model "brown" = OpenCV k1 k2 p1 p2 k3 [k4 k5 k6]; '
                    'model "fisheye" = OpenCV fisheye (Kannala-Brandt) k1..k4. f = fx, aspect = fy/fx.')


def to_json_rig(cgroup, path=None, *, session=None, extra=None):
    """CameraGroup -> JSON-friendly rig dict (disco ``pose_ba_core`` layout: ``intrinsics``
    and ``stations`` keyed by camera name; also ``K``/``dist`` in full). Written to ``path``
    if given. ``extra`` keys are merged at the top level."""
    import aniposelib
    out = OrderedDict(schema='aniposelib-rig/v1', convention=_JSON_CONVENTION,
                      aniposelib_version=getattr(aniposelib, '__version__', None))
    if session is not None:
        out['session'] = session
    out['cameras'] = [c.get_name() for c in _cameras(cgroup)]
    intr, stations = OrderedDict(), OrderedDict()
    for c in _cameras(cgroup):
        n = c.get_name()
        K, d, E = _K(c), _dist(c), _extrinsics(c)
        fish = bool(getattr(c, 'fisheye', False))
        wh = _size(c)
        it = OrderedDict(model='fisheye' if fish else 'brown', f=K[0, 0], aspect=K[1, 1] / K[0, 0],
                         cx=K[0, 2], cy=K[1, 2], skew=K[0, 1])
        dd = np.pad(d, (0, 8))
        if fish:
            it.update(k1=dd[0], k2=dd[1], k3=dd[2], k4=dd[3])
        else:
            it.update(k1=dd[0], k2=dd[1], p1=dd[2], p2=dd[3], k3=dd[4])
        it['K'] = K.tolist()
        it['dist'] = d.tolist()
        if wh is not None:
            it['image_wh'] = wh
        off = _offset(c)
        if any(off):
            it['offset'] = off
        if c.extra:
            it['extra'] = json.loads(json.dumps(c.extra, default=lambda o: _np(o).tolist()))
        intr[n] = it
        R, t = E[:3, :3], E[:3, 3]
        st = OrderedDict(name=n, camera=n, model=it['model'], f=it['f'], aspect=it['aspect'],
                         cx=it['cx'], cy=it['cy'], dist=d.tolist(), K=K.tolist(), R=R.tolist(),
                         t=t.tolist(), rvec=_f64(c.get_rotation()).ravel().tolist(),
                         center_world=(-R.T @ t).tolist())
        if wh is not None:
            st['image_wh'] = wh
        stations[n] = st
    out['intrinsics'] = intr
    out['stations'] = stations
    if extra:
        out.update(extra)
    out = json.loads(json.dumps(out, default=lambda o: float(o) if np.isscalar(o) else _np(o).tolist()))
    if path is not None:
        with open(str(path), 'w') as f:
            json.dump(out, f, indent=1)
    return out


def _json_K(it):
    if 'K' in it:
        return _f64(it['K']).reshape(3, 3)
    f = float(it['f'])
    fy = f * float(it.get('aspect', 1.0))
    return np.array([[f, float(it.get('skew', 0.0)), float(it['cx'])], [0, fy, float(it['cy'])], [0, 0, 1.0]])


def _json_dist(it):
    if 'dist' in it:
        return _f64(it['dist']).ravel()
    if it.get('model', 'brown') == 'fisheye':
        return np.array([it.get('k1', 0.0), it.get('k2', 0.0), it.get('k3', 0.0), it.get('k4', 0.0)], float)
    return np.array([it.get('k1', 0.0), it.get('k2', 0.0), it.get('p1', 0.0), it.get('p2', 0.0),
                     it.get('k3', 0.0)], float)


def from_json_rig(path_or_dict, *, stations='first'):
    """JSON rig (:func:`to_json_rig` or a disco ``pose_ba_core`` rig) -> CameraGroup.

    Cameras follow ``cameras`` (if present) else the ``intrinsics`` order. Each camera takes
    the station named like it, else the first ``<camera>@...`` station (``stations='first'``;
    a warning lists the others). ``stations='all'`` returns one camera per station, named
    after the station."""
    if isinstance(path_or_dict, dict):
        d = path_or_dict
    else:
        with open(str(path_or_dict)) as f:
            d = json.load(f)
    intr = d['intrinsics']
    st = d.get('stations', {})
    names = list(d.get('cameras') or intr.keys())

    def cam_of(s):
        return st[s].get('camera', s.split('@')[0])

    def build(n, s):
        it = intr[n]
        K = _json_K(st[s]) if (s is not None and 'K' in st[s] and 'K' not in it) else _json_K(it)
        E = np.eye(4)
        if s is not None:
            E[:3, :3] = _f64(st[s]['R']).reshape(3, 3)
            E[:3, 3] = _f64(st[s]['t']).ravel()
        else:
            warnings.warn('camera {} has no station (extrinsics); using identity'.format(n), RuntimeWarning)
        wh = it.get('image_wh') or (st[s].get('image_wh') if s is not None else None)
        return camera_from_extrinsics_mat(K, E, _json_dist(it), wh, n if stations != 'all' else s,
                                          fisheye=it.get('model') == 'fisheye',
                                          offset=it.get('offset'), extra=it.get('extra'))

    cams = []
    if stations == 'all':
        for s in st:
            c = cam_of(s)
            if c in intr:
                cams.append(build(c, s))
                cams[-1].set_name(s)
    elif stations == 'first':
        for n in names:
            if n in st:
                s = n
                others = [x for x in st if x != n and cam_of(x) == n]
            else:
                cand = sorted(x for x in st if cam_of(x) == n)
                s = cand[0] if cand else None
                others = cand[1:]
            if others:
                warnings.warn('camera {} has {} more stations ({}); using {} (stations="all" keeps '
                              'them all)'.format(n, len(others), ', '.join(others[:5]), s), RuntimeWarning)
            cams.append(build(n, s))
    else:
        raise ValueError("stations must be 'first' or 'all'")
    return _group(cams)
