import os
import sys

import numpy as np
import pytest
import torch
from hypothesis import settings

settings.register_profile("dev", deadline=None, max_examples=25)
settings.load_profile("dev")

# make tests/synth.py importable as `synth`
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def pytest_configure(config):
    config.addinivalue_line("markers", "integration: fast end-to-end test on a synthetic use-case rig (runs by default)")
    config.addinivalue_line("markers", "slow: longer test, excluded by default (run with -m slow)")
    config.addinivalue_line("markers", "target: 1.0 goal, marked xfail; excluded by default, run with `pixi run test-targets` (XPASS = goal met; then drop the marker)")


DEVICES = ['cpu'] + (['cuda'] if torch.cuda.is_available() else [])


@pytest.fixture(params=DEVICES)
def device(request):
    return torch.device(request.param)


@pytest.fixture(autouse=True)
def _seed():
    np.random.seed(0)
    torch.manual_seed(0)
