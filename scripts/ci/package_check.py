"""Check an installed pixeloe (run from a venv holding only the wheel).

- every shader and native source of the repo is inside the installed package
  (compared file by file against the source tree, by content hash);
- pixeloe.slang imports, and pixelize() runs on a CPU tensor through whatever
  backend "auto" selects on this machine.

Usage:
    python scripts/ci/package_check.py --source <repo root>
"""

import argparse
import hashlib
from pathlib import Path

import torch

import pixeloe
import pixeloe.slang
from pixeloe.slang.auto import select_context
from pixeloe.torch.pixelize import pixelize

PACKAGED = ("slang/shaders/**/*.slang", "slang/runtime/cpu/*.cpp")


def digests(root):
    return {
        p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for pattern in PACKAGED
        for p in root.glob(pattern)
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    args = ap.parse_args()
    installed = Path(pixeloe.__file__).parent
    source = Path(args.source).resolve() / "src" / "pixeloe"
    if installed.resolve().is_relative_to(source):
        raise SystemExit(f"{installed} is the source tree, not an installed wheel")
    want, have = digests(source), digests(installed)
    if not want:
        raise SystemExit(f"no packaged sources found under {source}")
    missing = sorted(set(want) - set(have))
    changed = sorted(k for k in set(want) & set(have) if want[k] != have[k])
    if missing or changed:
        raise SystemExit(f"wheel missing {missing}, differing {changed}")
    print(f"{len(want)} shader/native sources packaged, identical to the tree")

    img = torch.linspace(0, 1, 3 * 64 * 64).reshape(1, 3, 64, 64)
    ctx = select_context(img, "auto")
    out = pixelize(img, pixel_size=4, thickness=2)
    if out.shape != img.shape or not bool(torch.isfinite(out).all()):
        raise SystemExit(f"pixelize output invalid: {out.shape}")
    backend = "torch" if ctx is None else f"slang-{ctx.backend}"
    print(f"pixelize on CPU tensor ran through {backend}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
