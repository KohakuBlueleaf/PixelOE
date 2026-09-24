"""Which SIMD instructions do the Slang-built CPU kernels contain?

Disassembles every kernel DLL of the newest CPU kernel cache with MSVC's
dumpbin and counts, per DLL:
  zmm / ymm / xmm   register uses (AVX-512 / AVX(2) / SSE width)
  packed float ops  (...ps: 4/8/16 lanes)  vs  scalar float ops (...ss)

usage: python isa_report.py [cache_dir]
"""

import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / "src"))
from pixeloe.slang.runtime.cpu.toolchain import vcvars64
from pixeloe.slang.runtime.paths import cache_root

FLOAT_OP = re.compile(
    r"\b(v?(?:add|sub|mul|div|min|max|fmadd\d*|fmsub\d*|sqrt)(ps|ss))\b"
)


def disasm(dll):
    cmd = f'call "{vcvars64()}" >nul && dumpbin /nologo /disasm "{dll}"'
    return subprocess.run(
        f'cmd /d /s /c "{cmd}"', capture_output=True, text=True, check=False
    ).stdout


def main():
    root = (
        Path(sys.argv[1])
        if len(sys.argv) > 1
        else max(
            (
                p
                for p in (cache_root() / "cpu").iterdir()
                if p.is_dir() and p.name != "dispatch"
            ),
            key=lambda p: p.stat().st_mtime,
        )
    )
    total = Counter()
    print(
        f"{'kernel':42s} {'zmm':>6s} {'ymm':>6s} {'xmm':>6s} {'packed':>7s} {'scalar':>7s}"
    )
    for dll in sorted(root.glob("*.dll")):
        text = disasm(dll)
        c = Counter(
            zmm=len(re.findall(r"\bzmm\d+", text)),
            ymm=len(re.findall(r"\bymm\d+", text)),
            xmm=len(re.findall(r"\bxmm\d+", text)),
        )
        for m in FLOAT_OP.finditer(text):
            c["packed" if m.group(2) == "ps" else "scalar"] += 1
        total.update(c)
        print(
            f"{dll.stem:42s} {c['zmm']:6d} {c['ymm']:6d} {c['xmm']:6d} {c['packed']:7d} {c['scalar']:7d}"
        )
    print(
        f"{'TOTAL':42s} {total['zmm']:6d} {total['ymm']:6d} {total['xmm']:6d} {total['packed']:7d} {total['scalar']:7d}"
    )


if __name__ == "__main__":
    main()
