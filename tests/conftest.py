#!/usr/bin/env ipython

from hypothesis import settings

settings.register_profile("dev", deadline=None)
settings.load_profile("dev")
