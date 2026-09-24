"""Summarise a run_all.py report: failures, speed-up of every slang path over
the best torch path per config, memory, output agreement.

usage: python analyze.py <run_dir>   (writes <run_dir>/summary.md)
"""

import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

MIB = 2**20


def device_class(path):
    return "cpu" if path.endswith("-cpu") or "-cpu-" in path else "gpu"


def main():
    run_dir = Path(sys.argv[1])
    report = json.loads((run_dir / "report.json").read_text())
    rows = report["table"]
    lines = [f"# Summary of {run_dir.name}", "", report["meta"]["machine"], ""]

    failed = [r for r in rows if not r["ok"]]
    lines.append(f"## Failures: {len(failed)}")
    for r in failed:
        lines.append(f"- {r['path']} {r['size']}/{r['case']}: {r['error']}")
    lines.append("")

    by_cfg = defaultdict(dict)
    for r in rows:
        if r["ok"]:
            by_cfg[(r["size"], r["case"])][r["path"]] = r

    per_path = defaultdict(list)
    worst = defaultdict(lambda: (1e9, None))
    for cfg, paths in by_cfg.items():
        for p, r in paths.items():
            if not p.startswith("slang"):
                continue
            torch_paths = {
                q: t
                for q, t in paths.items()
                if q.startswith("torch") and device_class(q) == device_class(p)
            }
            if not torch_paths:
                continue
            best_p = min(torch_paths, key=lambda q: torch_paths[q]["median_ms"])
            best = torch_paths[best_p]["median_ms"]
            if True:
                s = best / r["median_ms"]
                per_path[p].append(s)
                if s < worst[p][0]:
                    worst[p] = (s, (cfg, best_p, best, r["median_ms"]))

    lines.append("## Speed-up vs the best torch path of each config")
    lines.append("")
    lines.append("| slang path | configs | min | median | max | worst config |")
    lines.append("|---|---|---|---|---|---|")
    for p, s in sorted(per_path.items()):
        _, (cfg, bp, bt, st) = worst[p]
        lines.append(
            f"| {p} | {len(s)} | {min(s):.2f}x | {statistics.median(s):.2f}x | "
            f"{max(s):.2f}x | {cfg[0]}/{cfg[1]}: {st:.2f} ms vs {bp} {bt:.2f} ms |"
        )
    lines.append("")

    lines.append("## Per config: median ms (best torch path vs slang paths)")
    lines.append("")
    slang_paths = sorted(per_path)
    lines.append("| size | case | best torch | " + " | ".join(slang_paths) + " |")
    lines.append("|---|---|---|" + "---|" * len(slang_paths))
    for cfg in sorted(by_cfg):
        paths = by_cfg[cfg]
        torch_paths = {
            p: r
            for p, r in paths.items()
            if p.startswith("torch") and device_class(p) == "gpu"
        }
        if not torch_paths:
            continue
        bp = min(torch_paths, key=lambda p: torch_paths[p]["median_ms"])
        cells = []
        for p in slang_paths:
            r = paths.get(p)
            ref = (
                paths.get("torch-cpu-fp32")
                if device_class(p) == "cpu"
                else torch_paths[bp]
            )
            cells.append(
                "-"
                if r is None or ref is None
                else f"{r['median_ms']:.2f} ({ref['median_ms'] / r['median_ms']:.1f}x)"
            )
        lines.append(
            f"| {cfg[0]} | {cfg[1]} | {torch_paths[bp]['median_ms']:.2f} ({bp}) | "
            + " | ".join(cells)
            + " |"
        )
    lines.append("")

    lines.append("## Memory (GPU: allocator peak reserved / slang pool; MiB)")
    lines.append("")
    lines.append(
        "| size | case | path | framework MiB | device delta MiB | peak private MiB |"
    )
    lines.append("|---|---|---|---|---|---|")
    for cfg in sorted(by_cfg):
        if cfg[1] != "default":
            continue
        for p, r in sorted(by_cfg[cfg].items()):
            fw = r.get("slang_peak_pool_bytes") or r.get("allocator_peak_reserved")
            dd = r.get("device_delta_bytes")
            pp = r.get("peak_private_bytes")
            lines.append(
                f"| {cfg[0]} | {cfg[1]} | {p} | "
                f"{'-' if fw is None else f'{fw / MIB:.1f}'} | "
                f"{'-' if dd is None else f'{dd / MIB:.1f}'} | "
                f"{'-' if pp is None else f'{pp / MIB:.0f}'} |"
            )
    lines.append("")

    lines.append("## Output agreement with the reference path (first batch item)")
    lines.append("")
    lines.append("| path | configs | max frac >1/255 | median frac >1/255 |")
    lines.append("|---|---|---|---|")
    agree = defaultdict(list)
    for r in rows:
        d = r.get("diff_vs_reference")
        if r["ok"] and d and "frac_gt_1_255" in d and r["case"] != "k_centroid":
            agree[r["path"]].append(d["frac_gt_1_255"])
    for p, v in sorted(agree.items()):
        lines.append(f"| {p} | {len(v)} | {max(v):.4f} | {statistics.median(v):.4f} |")
    (run_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines[:40]))


if __name__ == "__main__":
    main()
