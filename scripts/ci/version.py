"""Read, stamp and bump the single version string in pyproject.toml.

One source of truth: `[project] version`; every workflow goes through this.

Usage:
    python scripts/ci/version.py read
    python scripts/ci/version.py nightly            # X.Y.(Z+1).devYYYYmmddHHMMSS
    python scripts/ci/version.py bump patch|minor|major
    python scripts/ci/version.py write <version>    # stamp pyproject.toml
"""

import argparse
import re
import sys
from datetime import UTC, datetime
from pathlib import Path

PYPROJECT = Path(__file__).resolve().parents[2] / "pyproject.toml"
# The [project] table's own key: a line-anchored `version = "..."`, so a
# dependency pin containing "version" cannot match.
PATTERN = re.compile(r'(?m)^(version\s*=\s*")([^"]+)(")')


def read():
    match = PATTERN.search(PYPROJECT.read_text(encoding="utf-8"))
    if not match:
        raise SystemExit("no version found in pyproject.toml")
    return match.group(2)


def write(version):
    text = PYPROJECT.read_text(encoding="utf-8")
    new, count = PATTERN.subn(rf"\g<1>{version}\g<3>", text, count=1)
    if count != 1:
        raise SystemExit("no version found in pyproject.toml")
    PYPROJECT.write_text(new, encoding="utf-8")


def base(version):
    core = re.match(r"^(\d+)\.(\d+)\.(\d+)", version)
    if not core:
        raise SystemExit(f"cannot parse version {version!r}")
    return tuple(int(g) for g in core.groups())


def bump(version, part):
    major, minor, patch = base(version)
    if part == "major":
        return f"{major + 1}.0.0"
    if part == "minor":
        return f"{major}.{minor + 1}.0"
    return f"{major}.{minor}.{patch + 1}"


def nightly(version):
    """PEP 440 dev release of the next patch, X.Y.(Z+1).devYYYYmmddHHMMSS:
    uploadable to PyPI (no +local), found by `pip install --pre`, ordered by
    time and below the X.Y.(Z+1) it precedes."""
    stamp = datetime.now(UTC).strftime("%Y%m%d%H%M%S")
    major, minor, patch = base(version)
    return f"{major}.{minor}.{patch + 1}.dev{stamp}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=("read", "nightly", "bump", "write"))
    ap.add_argument("value", nargs="?")
    args = ap.parse_args()
    current = read()
    if args.action == "read":
        print(current)
    elif args.action == "nightly":
        print(nightly(current))
    elif args.action == "bump":
        print(bump(current, args.value or "patch"))
    else:
        if not args.value:
            raise SystemExit("write needs a version")
        write(args.value)
        print(args.value)
    return 0


if __name__ == "__main__":
    sys.exit(main())
