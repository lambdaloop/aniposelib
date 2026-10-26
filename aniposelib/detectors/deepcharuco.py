"""DeepCharuco: a learned ChArUco inner-corner detector (optional, ``[deepcharuco]`` extra).

Two small fully-convolutional nets from lambdaloop/deepcharuco (a fork of
JunkyByte/deepcharuco, MIT; an implementation of Hu et al., "Deep ChArUco: Dark ChArUco
Marker Pose Estimation", CVPR 2019), reproduced without PyTorch Lightning:

* :class:`DeepCharucoNet`: SuperPoint-style encoder (three 2x pools) with two heads:
  ``loc (N, 65, H/8, W/8)`` (which of the 64 pixels of an 8x8 cell holds a corner, plus a
  dust bin) and ``ids (N, n_ids + 1, H/8, W/8)`` (which board corner, plus a dust bin).
  One checkpoint per board geometry.
* :class:`RefineNet`: a ``(N, 1, 24, 24)`` patch around each detected corner to a
  ``(N, 1, 64, 64)`` heatmap over the central 8x8 pixels, i.e. 1/8-pixel refinement.

Weights are published at https://huggingface.co/lambdaloop/deepcharuco-weights as
``detectnets/detectnet_r{rows}_c{cols}.ckpt`` and ``refinenets/refinenet_r{rows}_c{cols}.ckpt``
for 5..9 rows x 5..9 columns (``rows = squaresY``, ``cols = squaresX``); r6_c5 has no
refinenet (detection still works, at whole-pixel precision).

Conventions (see also EvoVision ``deepcharuco_model.py``):

- **Corner ids match** ``cv2.aruco.CharucoDetector`` and :class:`aniposelib.boards.CharucoBoard`:
  row-major, x fastest, ``squaresX - 1`` per row (a property of lambdaloop's fork only).
- **Input scaling is isotropic** (``scale=``) and padded to a multiple of 8 with black, like
  training. Coordinates are mapped back with the pixel-centre convention of ``cv2.resize``:
  ``x = (x' + 0.5) / s - 0.5``.
- **Normalisation** is ``(gray - 128) / 255`` on uint8 gray values.
- The refiner sees the detector's input pixels, so the precision floor is ``1/(8 s)``
  original pixels; ``subpix=True`` adds ``cv2.cornerSubPix`` on the full-resolution frame.
- Training boards were generated with OpenCV 4.6 (the *legacy* pattern). This only matters
  for an even number of rows.

Only torch (a core dependency) is needed for ``backend='torch'``. ``huggingface_hub`` is used
for downloads when installed (otherwise a plain HTTPS download into
``~/.cache/aniposelib/deepcharuco``), and ``backend='onnx'`` needs ``onnxruntime`` (and
``onnx`` for the export).
"""

import os
import warnings
from pathlib import Path

import numpy as np
import torch
from torch import nn

from ..boards import empty_corners

HF_REPO = 'lambdaloop/deepcharuco-weights'
PUBLISHED_ROWS = range(5, 10)
PUBLISHED_COLS = range(5, 10)
MISSING_REFINENETS = {(6, 5)}  # (rows, cols)

PIXEL_OFFSET = 128.0
PIXEL_SCALE = 255.0
PATCH_SIZE = 24
REFINE_UPSCALE = 8
STRIDE = 8


# ---------------------------------------------------------------------- networks

class DeepCharucoNet(nn.Module):
    """Corner localisation + id classification (``dcModel`` upstream)."""

    def __init__(self, n_ids):
        super().__init__()
        self.n_ids = int(n_ids)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        c1, c2, c3, c4, c5 = 64, 64, 128, 128, 256
        self.conv1a = nn.Conv2d(1, c1, 3, 1, 1)
        self.bn1a = nn.BatchNorm2d(c1)
        self.conv1b = nn.Conv2d(c1, c1, 3, 1, 1)
        self.bn1b = nn.BatchNorm2d(c1)
        self.conv2a = nn.Conv2d(c1, c2, 3, 1, 1)
        self.bn2a = nn.BatchNorm2d(c2)
        self.conv2b = nn.Conv2d(c2, c2, 3, 1, 1)
        self.bn2b = nn.BatchNorm2d(c2)
        self.conv3a = nn.Conv2d(c2, c3, 3, 1, 1)
        self.bn3a = nn.BatchNorm2d(c3)
        self.conv3b = nn.Conv2d(c3, c3, 3, 1, 1)
        self.bn3b = nn.BatchNorm2d(c3)
        self.conv4a = nn.Conv2d(c3, c4, 3, 1, 1)
        self.bn4a = nn.BatchNorm2d(c4)
        self.conv4b = nn.Conv2d(c4, c4, 3, 1, 1)
        self.bn4b = nn.BatchNorm2d(c4)
        self.convPa = nn.Conv2d(c4, c5, 3, 1, 1)
        self.bnPa = nn.BatchNorm2d(c5)
        self.convPb = nn.Conv2d(c5, 65, 1, 1, 0)
        self.convDa = nn.Conv2d(c4, c5, 3, 1, 1)
        self.bnDa = nn.BatchNorm2d(c5)
        self.convDb = nn.Conv2d(c5, self.n_ids + 1, 1, 1, 0)

    def forward(self, x):
        x = self.relu(self.bn1a(self.conv1a(x)))
        x = self.relu(self.bn1b(self.conv1b(x)))
        x = self.pool(x)
        x = self.relu(self.bn2a(self.conv2a(x)))
        x = self.relu(self.bn2b(self.conv2b(x)))
        x = self.pool(x)
        x = self.relu(self.bn3a(self.conv3a(x)))
        x = self.relu(self.bn3b(self.conv3b(x)))
        x = self.pool(x)
        x = self.relu(self.bn4a(self.conv4a(x)))
        x = self.relu(self.bn4b(self.conv4b(x)))
        loc = self.convPb(self.relu(self.bnPa(self.convPa(x))))
        ids = self.convDb(self.relu(self.bnDa(self.convDa(x))))
        return loc, ids


class RefineNet(nn.Module):
    """Sub-pixel corner refinement on a 24x24 patch -> 64x64 heatmap of the central 8x8 px."""

    def __init__(self):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.up_sample = nn.UpsamplingNearest2d(scale_factor=2)
        c1, c2, c3, c4, c5 = 64, 128, 128, 128, 64
        self.conv1a = nn.Conv2d(1, c1, 3, 1, 0)
        self.bn1a = nn.BatchNorm2d(c1)
        self.conv1b = nn.Conv2d(c1, c1, 3, 1, 0)
        self.bn1b = nn.BatchNorm2d(c1)
        self.conv2a = nn.Conv2d(c1, c2, 3, 1, 0)
        self.bn2a = nn.BatchNorm2d(c2)
        self.conv2b = nn.Conv2d(c2, c2, 3, 1, 0)
        self.bn2b = nn.BatchNorm2d(c2)
        self.conv3a = nn.Conv2d(c2, c3, 3, 1, 1)
        self.bn3a = nn.BatchNorm2d(c3)
        self.conv3b = nn.Conv2d(c3, c3, 3, 1, 1)
        self.bn3b = nn.BatchNorm2d(c3)
        self.conv4a = nn.Conv2d(c3, c4, 3, 1, 1)
        self.bn4a = nn.BatchNorm2d(c4)
        self.conv4b = nn.Conv2d(c4, c4, 3, 1, 1)
        self.bn4b = nn.BatchNorm2d(c4)
        self.conv5a = nn.Conv2d(c4, c5, 3, 1, 1)
        self.bn5a = nn.BatchNorm2d(c5)
        self.conv5b = nn.Conv2d(c5, c5, 3, 1, 1)
        self.bn5b = nn.BatchNorm2d(c5)
        self.convPa = nn.Conv2d(c5, 64, 3, 1, 1)
        self.bnPa = nn.BatchNorm2d(64)
        self.convPb = nn.Conv2d(64, 1, 1, 1, 0)

    def forward(self, x):
        x = self.relu(self.bn1a(self.conv1a(x)))
        x = self.relu(self.bn1b(self.conv1b(x)))
        x = self.relu(self.bn2a(self.conv2a(x)))
        x = self.relu(self.bn2b(self.conv2b(x)))
        x = self.pool(x)
        x = self.relu(self.bn3a(self.conv3a(x)))
        x = self.relu(self.bn3b(self.conv3b(x)))
        x = self.up_sample(x)
        x = self.relu(self.bn4a(self.conv4a(x)))
        x = self.relu(self.bn4b(self.conv4b(x)))
        x = self.up_sample(x)
        x = self.relu(self.bn5a(self.conv5a(x)))
        x = self.relu(self.bn5b(self.conv5b(x)))
        x = self.up_sample(x)
        return self.convPb(self.relu(self.bnPa(self.convPa(x))))


# ---------------------------------------------------------------------- weights

def supports_board(squaresX, squaresY):
    """Is there a published checkpoint for this board geometry?"""
    return int(squaresY) in PUBLISHED_ROWS and int(squaresX) in PUBLISHED_COLS


def checkpoint_names(squaresX, squaresY):
    """``(detectnet, refinenet)`` file names (refinenet None where not published)."""
    rows, cols = int(squaresY), int(squaresX)
    if not supports_board(cols, rows):
        published = ['{}x{}'.format(c, r) for r in PUBLISHED_ROWS for c in PUBLISHED_COLS]
        raise ValueError(
            'no published DeepCharuco checkpoint for a squaresX={} x squaresY={} board '
            '(rows {}-{} x cols {}-{} only; as squaresX x squaresY: {}). '
            "Use detector='opencv'.".format(
                cols, rows, PUBLISHED_ROWS.start, PUBLISHED_ROWS.stop - 1,
                PUBLISHED_COLS.start, PUBLISHED_COLS.stop - 1, ', '.join(published)))
    ref = None if (rows, cols) in MISSING_REFINENETS else 'refinenet_r{}_c{}.ckpt'.format(rows, cols)
    return 'detectnet_r{}_c{}.ckpt'.format(rows, cols), ref


def default_cache_dir():
    return Path(os.environ.get('ANIPOSELIB_CACHE', Path.home() / '.cache' / 'aniposelib')) / 'deepcharuco'


def fetch_weights(name, weights_dir=None, repo=HF_REPO, revision=None, download=True):
    """Path to one checkpoint file, downloading it if needed.

    Looks in ``weights_dir`` (or ``$DEEPCHARUCO_WEIGHTS``), flat or under
    ``detectnets/``/``refinenets/``; then the Hugging Face cache (``huggingface_hub``) and the
    aniposelib cache; then downloads (unless ``download=False``)."""
    subdir = 'detectnets' if name.startswith('detectnet') else 'refinenets'
    dirs = [weights_dir, os.environ.get('DEEPCHARUCO_WEIGHTS')]
    for d in dirs:
        if not d:
            continue
        d = Path(d).expanduser()
        for cand in (d / name, d / subdir / name):
            if cand.is_file():
                return cand
    if weights_dir is not None:
        raise FileNotFoundError('{} not found in weights_dir={}'.format(name, weights_dir))

    local = default_cache_dir() / subdir / name
    if local.is_file():
        return local

    filename = '{}/{}'.format(subdir, name)
    try:
        import huggingface_hub
    except ImportError:
        huggingface_hub = None
    if huggingface_hub is not None:
        try:
            return Path(huggingface_hub.hf_hub_download(repo, filename, revision=revision,
                                                        local_files_only=True))
        except Exception:  # not in the HF cache (the exception type varies across versions)
            pass
        if download:
            return Path(huggingface_hub.hf_hub_download(repo, filename, revision=revision))
    if not download:
        raise FileNotFoundError('{} is not cached and download=False'.format(name))

    from urllib.request import Request, urlopen
    url = 'https://huggingface.co/{}/resolve/{}/{}'.format(repo, revision or 'main', filename)
    local.parent.mkdir(parents=True, exist_ok=True)
    tmp = local.with_suffix(local.suffix + '.part')
    with urlopen(Request(url, headers={'User-Agent': 'aniposelib'}), timeout=60) as r, \
            open(tmp, 'wb') as f:
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
    tmp.replace(local)
    return local


def load_state(path, module):
    """Load a (Lightning) checkpoint into a bare module with ``strict=True``."""
    ckpt = torch.load(str(path), map_location='cpu', weights_only=True)
    state = ckpt.get('state_dict', ckpt) if isinstance(ckpt, dict) else ckpt
    state = {(k[len('model.'):] if k.startswith('model.') else k): v for k, v in state.items()}
    module.load_state_dict(state, strict=True)
    return module


class OnnxNet:
    """An onnxruntime session that quacks like the torch module (torch tensors in and out)."""

    def __init__(self, onnx_path, device='cpu'):
        import onnxruntime as ort
        providers = (['CUDAExecutionProvider', 'CPUExecutionProvider']
                     if str(device).startswith('cuda') else ['CPUExecutionProvider'])
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.sess = ort.InferenceSession(str(onnx_path), opts, providers=providers)
        self.input_name = self.sess.get_inputs()[0].name
        self.device = torch.device(device)

    def __call__(self, x):
        arr = np.ascontiguousarray(x.detach().cpu().numpy(), dtype=np.float32)
        outs = self.sess.run(None, {self.input_name: arr})
        tensors = [torch.from_numpy(o).to(self.device) for o in outs]
        return tensors[0] if len(tensors) == 1 else tuple(tensors)


def export_onnx(module, sample, dst, input_names, output_names, dynamic_hw=True):
    """Export a net to ONNX once (cached at ``dst``); batch (and H, W) axes dynamic."""
    dst = Path(dst)
    if dst.exists():
        return dst
    axes = {input_names[0]: {0: 'batch'}}
    if dynamic_hw:
        axes[input_names[0]].update({2: 'height', 3: 'width'})
    for name in output_names:
        axes[name] = {0: 'batch'}
        if dynamic_hw:
            axes[name].update({2: 'height', 3: 'width'})
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix('.part.onnx')
    kwargs = dict(input_names=list(input_names), output_names=list(output_names),
                  dynamic_axes=axes, opset_version=17, do_constant_folding=True)
    try:
        torch.onnx.export(module, sample, str(tmp), dynamo=False, **kwargs)
    except TypeError:  # torch without the dynamo= argument
        torch.onnx.export(module, sample, str(tmp), **kwargs)
    tmp.replace(dst)
    return dst


# ---------------------------------------------------------------------- pre/post-processing

def _to_gray(image, color_order):
    img = np.asarray(image)
    if img.ndim == 2:
        gray = img
    elif img.ndim == 3 and img.shape[2] == 1:
        gray = img[..., 0]
    elif img.ndim == 3 and img.shape[2] >= 3:
        import cv2
        code = cv2.COLOR_BGR2GRAY if color_order == 'bgr' else cv2.COLOR_RGB2GRAY
        gray = cv2.cvtColor(np.ascontiguousarray(img[..., :3]), code)
    else:
        raise ValueError('image must be HxW or HxWx3, got shape {}'.format(img.shape))
    if gray.dtype != np.uint8:
        if np.issubdtype(gray.dtype, np.floating) and gray.size and gray.max() <= 1.0:
            gray = gray * 255.0
        gray = np.clip(gray, 0, 255).astype(np.uint8)
    return gray


def preprocess(gray, scale=1.0):
    """Isotropic scale, pad bottom/right to a multiple of 8 with black, normalise.

    Returns ``(float32 array (H', W'), meta)``; ``meta`` holds the achieved scales."""
    import cv2
    h, w = gray.shape[:2]
    if scale != 1.0:
        nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
        interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
        gray = cv2.resize(gray, (nw, nh), interpolation=interp)
    else:
        nw, nh = w, h
    ph, pw = (-nh) % STRIDE, (-nw) % STRIDE
    if ph or pw:
        gray = cv2.copyMakeBorder(gray, 0, ph, 0, pw, cv2.BORDER_CONSTANT, value=0)
    x = (gray.astype(np.float32) - PIXEL_OFFSET) / PIXEL_SCALE
    return x, {'sx': nw / w, 'sy': nh / h}


def to_original(xy, meta):
    """Detector-input pixel coordinates -> input-image pixels (``cv2.resize`` pixel centres)."""
    xy = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
    out = np.empty_like(xy)
    out[:, 0] = (xy[:, 0] + 0.5) / meta['sx'] - 0.5
    out[:, 1] = (xy[:, 1] + 0.5) / meta['sy'] - 0.5
    return out


def decode(loc, ids, n_ids):
    """Decode one image's heads into ``(xy (N, 2) int64, ids (N,), scores (N,))``, sorted by id.

    A cell holds a corner when neither head picks its dust bin. When two cells claim the
    same id, the one with the higher id-softmax score wins."""
    loc_arg = loc.argmax(0)
    prob = ids.softmax(0)
    score, id_arg = prob.max(0)
    mask = (loc_arg != 64) & (id_arg != n_ids)
    if not bool(mask.any()):
        return np.zeros((0, 2), np.int64), np.zeros(0, np.int64), np.zeros(0, np.float32)
    cell = torch.nonzero(mask, as_tuple=False)
    region = loc_arg[mask]
    xs = STRIDE * cell[:, 1] + region % STRIDE
    ys = STRIDE * cell[:, 0] + torch.div(region, STRIDE, rounding_mode='floor')
    xy = torch.stack((xs, ys), 1).cpu().numpy().astype(np.int64)
    found = id_arg[mask].cpu().numpy().astype(np.int64)
    scores = score[mask].float().cpu().numpy()
    order = np.lexsort((-scores, found))  # by id, best score first
    keep = order[np.unique(found[order], return_index=True)[1]]
    return xy[keep], found[keep], scores[keep]


def extract_patches(img, kpts, patch_size=PATCH_SIZE):
    """``(N, 1, 24, 24)`` zero-padded patches of ``img (H, W)`` around integer ``kpts``."""
    pad = patch_size // 2
    padded = torch.nn.functional.pad(img, (pad, pad, pad, pad), mode='constant', value=0)
    k = torch.as_tensor(np.asarray(kpts), device=padded.device, dtype=torch.long)
    r = torch.arange(patch_size, device=padded.device)
    rows = torch.index_select(padded, 0, (k[:, 1, None] + r).reshape(-1))
    rows = rows.view(k.shape[0], patch_size, padded.shape[-1])
    cols = (k[:, 0, None] + r).unsqueeze(1)
    return torch.gather(rows, 2, cols.expand(-1, patch_size, -1)).unsqueeze(1)


def heat_to_offsets(heat):
    """RefineNet heatmaps ``(N, 64, 64)`` -> sub-pixel offsets ``(N, 2)`` in input pixels."""
    flat = heat.reshape(heat.shape[0], -1).argmax(1)
    peak = torch.stack((flat % heat.shape[2],
                        torch.div(flat, heat.shape[2], rounding_mode='floor')), 1)
    return (peak.float() - (REFINE_UPSCALE * STRIDE) // 2) / REFINE_UPSCALE


# ---------------------------------------------------------------------- detector

def _resolve_device(device):
    if device in (None, 'auto'):
        return 'cuda' if torch.cuda.is_available() else 'cpu'
    return str(device)


class DeepCharucoDetector:
    """Batched DeepCharuco ChArUco corner detector.

    Parameters
    ----------
    board : CharucoBoard or (squaresX, squaresY)
        Board geometry; selects the checkpoint (``r{squaresY}_c{squaresX}``).
    weights_dir : path, optional
        Directory with the ``.ckpt`` files (flat or HF layout). Default: Hugging Face cache /
        download from ``repo``.
    device : 'auto', 'cpu', 'cuda', ...
    backend : 'torch' or 'onnx'
    batch_size : int
        Frames per network call in :meth:`detect_batch`.
    refine : bool
        Run RefineNet (1/8-pixel corners).
    scale : float
        Isotropic input scale (the board should look like it did in training: covering
        ~25-90% of a 320x240 frame). E.g. 0.25-0.5 for 1080p footage of a nearby board.
    min_score : float
        Drop corners whose id probability is below this.
    subpix : bool
        Follow with ``cv2.cornerSubPix`` on the full-resolution frame.
    color_order : 'bgr' (OpenCV frames, default) or 'rgb'

    ``detect(image)`` and ``detect_batch(images)`` return ``(corners (N, 1, 2) float32,
    ids (N, 1) int32)`` per image, the format of :meth:`CharucoBoard.detect_image`.
    """

    def __init__(self, board, *, weights_dir=None, repo=HF_REPO, revision=None,
                 device='auto', backend='torch', batch_size=16, refine=True, scale=1.0,
                 min_score=0.0, subpix=False, color_order='bgr', download=True):
        if isinstance(board, (tuple, list)):
            squaresX, squaresY = int(board[0]), int(board[1])
            legacy = None
        else:
            squaresX, squaresY = board.squaresX, board.squaresY
            legacy = getattr(board, 'legacy_pattern', None)
        self.squaresX, self.squaresY = squaresX, squaresY
        self.n_ids = (squaresX - 1) * (squaresY - 1)
        det_name, ref_name = checkpoint_names(squaresX, squaresY)
        if legacy is False and squaresY % 2 == 0:
            warnings.warn(
                'DeepCharuco was trained on legacy-pattern boards (OpenCV < 4.7); with an even '
                'number of rows (squaresY={}) the non-legacy pattern differs. Use '
                'legacy_pattern=True if your board is legacy, and check detections with '
                'validate_detection.'.format(squaresY), RuntimeWarning, stacklevel=2)
        if backend not in ('torch', 'onnx'):
            raise ValueError("backend must be 'torch' or 'onnx'")
        if color_order not in ('bgr', 'rgb'):
            raise ValueError("color_order must be 'bgr' or 'rgb'")
        self.device = _resolve_device(device)
        self.backend = backend
        self.batch_size = max(1, int(batch_size))
        self.scale = float(scale)
        self.min_score = float(min_score)
        self.subpix = bool(subpix)
        self.color_order = color_order

        det_path = fetch_weights(det_name, weights_dir, repo, revision, download)
        detector = load_state(det_path, DeepCharucoNet(self.n_ids)).eval()
        refiner = None
        if refine and ref_name is not None:
            ref_path = fetch_weights(ref_name, weights_dir, repo, revision, download)
            refiner = load_state(ref_path, RefineNet()).eval()
        elif refine:
            warnings.warn('no RefineNet published for this board; corners are whole-pixel',
                          RuntimeWarning, stacklevel=2)
        self.refine = refiner is not None

        if backend == 'torch':
            self.detector = detector.to(self.device)
            self.refiner = None if refiner is None else refiner.to(self.device)
        else:
            onnx_dir = default_cache_dir() / 'onnx'
            det_onnx = export_onnx(detector, torch.zeros(1, 1, 240, 320),
                                   onnx_dir / det_name.replace('.ckpt', '.onnx'),
                                   ['image'], ['loc', 'ids'], dynamic_hw=True)
            self.detector = OnnxNet(det_onnx, self.device)
            self.refiner = None
            if refiner is not None:
                ref_onnx = export_onnx(refiner, torch.zeros(1, 1, PATCH_SIZE, PATCH_SIZE),
                                       onnx_dir / ref_name.replace('.ckpt', '.onnx'),
                                       ['patch'], ['heat'], dynamic_hw=False)
                self.refiner = OnnxNet(ref_onnx, self.device)

    def __repr__(self):
        return 'DeepCharucoDetector(squaresX={}, squaresY={}, device={!r}, backend={!r})'.format(
            self.squaresX, self.squaresY, self.device, self.backend)

    def detect(self, image):
        return self.detect_batch([image])[0]

    @torch.no_grad()
    def detect_batch(self, images, return_scores=False):
        """Detect corners in a list of images (any mix of sizes; same-size frames are batched)."""
        grays = [_to_gray(im, self.color_order) for im in images]
        prepped = [preprocess(g, self.scale) for g in grays]
        results = [None] * len(images)
        groups = {}
        for n, (x, _) in enumerate(prepped):
            groups.setdefault(x.shape, []).append(n)
        for shape, idxs in groups.items():
            for start in range(0, len(idxs), self.batch_size):
                chunk = idxs[start:start + self.batch_size]
                batch = torch.from_numpy(np.stack([prepped[n][0] for n in chunk]))[:, None]
                batch = batch.to(self.device)
                loc, ids = self.detector(batch)
                decoded = [decode(loc[b], ids[b], self.n_ids) for b in range(len(chunk))]
                if self.min_score > 0:
                    decoded = [(xy[s >= self.min_score], i[s >= self.min_score], s[s >= self.min_score])
                               for xy, i, s in decoded]
                subs = [xy.astype(np.float64) for xy, _, _ in decoded]
                if self.refiner is not None and any(len(xy) for xy, _, _ in decoded):
                    patches = [extract_patches(batch[b, 0], decoded[b][0])
                               for b in range(len(chunk)) if len(decoded[b][0])]
                    heat = self.refiner(torch.cat(patches, 0))[:, 0]
                    off = heat_to_offsets(heat).cpu().numpy().astype(np.float64)
                    k = 0
                    for b in range(len(chunk)):
                        m = len(decoded[b][0])
                        if m:
                            subs[b] = subs[b] + off[k:k + m]
                            k += m
                for b, n in enumerate(chunk):
                    xy = to_original(subs[b], prepped[n][1])
                    results[n] = (xy, decoded[b][1], decoded[b][2])
        out = []
        for n, (xy, found, scores) in enumerate(results):
            if len(found) == 0:
                corners, ids = empty_corners()
                out.append((corners, ids, scores) if return_scores else (corners, ids))
                continue
            if self.subpix:
                xy = self._subpix(grays[n], xy)
            corners = xy.astype(np.float32).reshape(-1, 1, 2)
            ids = found.astype(np.int32).reshape(-1, 1)
            out.append((corners, ids, scores) if return_scores else (corners, ids))
        return out

    def _subpix(self, gray, xy):
        import cv2
        if len(xy) < 2:
            return xy
        # half-window: a third of the closest corner spacing, at least 2 px
        d = np.linalg.norm(xy[:, None] - xy[None], axis=-1)
        d[d == 0] = np.inf
        win = int(np.clip(np.min(d) / 3.0, 2, 11))
        crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01)
        pts = np.ascontiguousarray(xy, dtype=np.float32).reshape(-1, 1, 2)
        pts = cv2.cornerSubPix(gray, pts, (win, win), (-1, -1), crit)
        return pts.reshape(-1, 2).astype(np.float64)
