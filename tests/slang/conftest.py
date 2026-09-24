import pytest

import pixeloe.torch.env as pixeloe_env

# imports slangpy after preparing its compilers' environment (runtime/toolchain.py)
pytest.importorskip("pixeloe.slang.runtime.registry")


@pytest.fixture(autouse=True)
def disable_compile():
    old = pixeloe_env.TORCH_COMPILE
    pixeloe_env.TORCH_COMPILE = False
    yield
    pixeloe_env.TORCH_COMPILE = old
