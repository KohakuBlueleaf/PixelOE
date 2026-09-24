"""Backend selection of the public pixelize() (pixeloe.slang.auto)."""

import pytest
import torch
from slang_helpers import BACKENDS, all_backends, compare, photo

from pixeloe.slang.auto import CANDIDATES, select_context
from pixeloe.slang.pixelize import pixelize as slang_pixelize
from pixeloe.torch.pixelize import pixelize

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


def usable(device):
    """Candidates for `device` that the conformance suite runs here; skips
    when PIXELOE_SLANG_BACKENDS hides a candidate auto would still probe."""
    hidden = [b for b in CANDIDATES[device] if b not in BACKENDS]
    if hidden:
        pytest.skip(f"PIXELOE_SLANG_BACKENDS excludes {hidden}")
    return [b for b in CANDIDATES[device] if b in all_backends()]


@pytest.mark.parametrize("device", DEVICES)
def test_auto_picks_first_usable_candidate(device):
    img = photo(64, 64).to(device)
    ctx = select_context(img, "auto")
    expected = usable(device)
    if not expected:
        assert ctx is None
        return
    assert ctx is not None
    assert ctx.backend == expected[0]


@pytest.mark.parametrize("device", DEVICES)
def test_auto_output_is_the_selected_backends_output(device):
    img = photo(64, 96).to(device)
    ctx = select_context(img, "auto")
    if ctx is None:
        pytest.skip(f"no slang backend for {device} tensors here")
    out = pixelize(img, pixel_size=4, thickness=3, backend="auto")
    direct = slang_pixelize(img, pixel_size=4, thickness=3, context=ctx)
    assert compare(out, direct)["max"] == 0.0
    ref = pixelize(img, pixel_size=4, thickness=3, backend="torch")
    assert compare(out, ref)["frac_gt_1_255"] < 1e-2


def test_torch_backend_skips_slang():
    img = photo(32, 32)
    assert select_context(img, "torch") is None


@pytest.mark.parametrize(
    "img",
    [
        photo(32, 32).double(),
        photo(32, 32).requires_grad_(),
    ],
    ids=["float64", "requires_grad"],
)
def test_auto_falls_back_for_unsupported_inputs(img):
    assert select_context(img, "auto") is None


def test_unknown_backend_raises():
    with pytest.raises(ValueError):
        select_context(photo(32, 32), "metal")


def test_env_var_selects_backend(monkeypatch):
    monkeypatch.setenv("PIXELOE_BACKEND", "torch")
    assert select_context(photo(32, 32)) is None
