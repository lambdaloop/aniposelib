# Changelog (aniposelib-pytorch)

## 0.9.0

First release under the `aniposelib-pytorch` name, from the `pytorch` branch of
`lambdaloop/anipose-lib`. It still imports as `aniposelib`.

- The code is identical to `pytorch@05b769a`, which downstream projects previously pinned via git.
- `torch>=2.1` is now a declared dependency.
- `opencv-contrib-python` was replaced by `opencv-python>=4.7,<5`. aruco is in main OpenCV since 4.7.
- New dependency `aniposelib==0.0.1`, a file-less placeholder. It makes classic aniposelib (numpy, 0.8.x)
  impossible to install in the same environment.
- `import aniposelib` raises `ImportError` if a classic `aniposelib` distribution is also installed.
- New `aniposelib.DISTRIBUTION = 'aniposelib-pytorch'` and `aniposelib.BACKEND = 'torch'`.
- `requires-python >= 3.9`.

Differences from classic aniposelib 0.8.0, inherited from the pytorch branch:
- `Camera`, `FisheyeCamera` and `CameraGroup` are `torch.nn.Module`s with float64 parameters, and support `.to(device)`.
- Methods return torch tensors. This changes in 1.0: numpy in → numpy out, with `return_grad=True` for autograd.
- `bundle_adjust_iter` / `bundle_adjust` are torch (Adam) optimizers.
- `n_dist` / `set_n_dist` control which distortion terms are optimised, and fisheye is differentiable.
- `triangulate_possible` is GPU-batched.
