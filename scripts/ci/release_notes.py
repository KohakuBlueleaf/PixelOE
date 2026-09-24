"""Release notes for a version: the commits since the previous release tag.

A plain commit list, not GitHub's generated notes: those group by pull
request, so direct pushes to main would vanish from them.

Usage:
    python scripts/ci/release_notes.py --version 1.0.0
"""

import argparse
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def git(*args):
    return subprocess.run(
        ["git", *args], capture_output=True, text=True, check=False, cwd=ROOT
    ).stdout.strip()


def previous_tag(version):
    """Newest v* tag that is not this version's own tag."""
    for tag in git("tag", "--list", "v*", "--sort=-creatordate").split():
        if tag != f"v{version}":
            return tag
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", required=True)
    args = ap.parse_args()
    prev = previous_tag(args.version)
    span = f"{prev}..HEAD" if prev else "HEAD"
    log = git("log", "--no-merges", "--pretty=* %s (%h)", span)
    since = f" since {prev}" if prev else ""
    print(f"## PixelOE {args.version}\n\n### Changes{since}\n\n{log or '* none'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
