# Anipose lib

 [![License](https://img.shields.io/badge/License-BSD%202--Clause-orange.svg)](https://opensource.org/licenses/BSD-2-Clause)
 
An easy-to-use library for calibrating cameras and triangulation in Python.

This is the backend library for the [Anipose](https://github.com/lambdaloop/anipose) package.

## aniposelib-pytorch

This branch is published on PyPI as **`aniposelib-pytorch`**: the PyTorch/GPU version of aniposelib
(cameras are `torch.nn.Module`s, GPU-batched triangulation and bundle adjustment). It still
**imports as `aniposelib`**.

```bash
pip install aniposelib-pytorch
```

Install torch with the CUDA build that matches your GPU first if you need a specific one
(see https://pytorch.org/get-started/locally/).

**It cannot be installed alongside classic `aniposelib`** (numpy, 0.8.x) because both provide the
`aniposelib` import. `aniposelib-pytorch` depends on a file-less placeholder `aniposelib==0.0.1`, so
pip/uv refuse environments that also require classic aniposelib. Replace any `aniposelib`
requirement in your project with `aniposelib-pytorch`. If both somehow end up installed, `import
aniposelib` raises an error; fix it with:

```bash
pip uninstall -y aniposelib aniposelib-pytorch && pip install aniposelib-pytorch
```

Do not `pip install aniposelib==0.8.x` into an environment with aniposelib-pytorch: it overwrites
the package files. Classic aniposelib is still available with `pip install aniposelib` in a
separate environment.

The [documentation](https://anipose.readthedocs.io/en/latest/aniposelib-tutorial.html) is located in Anipose repository.

