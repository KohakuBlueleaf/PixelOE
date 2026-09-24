"""The pixelize backend request: an explicit argument, else $PIXELOE_BACKEND,
else "auto" (see pixeloe.slang.auto)."""

import os

ENV_VAR = "PIXELOE_BACKEND"
DEFAULT = "auto"


def requested_backend(backend=None):
    return (backend or os.environ.get(ENV_VAR) or DEFAULT).lower()
