"""The suite tests the torch pipeline and holds the Slang port to it, so the
public pixelize() runs torch here; backend selection is tested explicitly in
tests/slang/test_auto.py."""

import os

from pixeloe.backend import ENV_VAR

os.environ[ENV_VAR] = "torch"
