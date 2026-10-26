# aniposelib 0.0.1 (placeholder)

This release contains **no code**. It is a dependency of
[`aniposelib-pytorch`](https://pypi.org/project/aniposelib-pytorch/), the PyTorch/GPU version of
aniposelib, which provides the `aniposelib` import itself.

Because `aniposelib-pytorch` requires exactly `aniposelib==0.0.1`, pip and uv refuse to install
classic aniposelib (numpy, 0.8.x) in the same environment: the two would overwrite each other's files.

- Want classic aniposelib? `pip install aniposelib` (installs the latest real release, 0.8.x).
- Want the PyTorch version? `pip install aniposelib-pytorch`, and replace any `aniposelib`
  requirement in your project with `aniposelib-pytorch`.
