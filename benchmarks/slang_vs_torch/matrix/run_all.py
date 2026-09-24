"""Run the full slang-vs-torch benchmark matrix and write one report.

Every implementation path runs in its own subprocess (clean allocator and
device state), sequentially (exclusive GPU). Output tree:

    outputs/bench/slang_vs_torch/<stamp>/
        runs/<path>/<size>__<case>.json   per-config record
        runs/<path>/<size>__<case>.npy    output image (first batch item)
        logs/<path>.log                   worker stdout/stderr
        report.json                       everything, aggregated
        report.md                         tables

usage: python run_all.py [--only path1,path2] [--device gpu|cpu|all]
"""

import argparse
import datetime
import json
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from configs import CPU_PATHS, GPU_PATHS, REFERENCE_PATH

ROOT = HERE.parents[2]
OUT_ROOT = ROOT / "outputs" / "bench" / "slang_vs_torch"


def run_worker(path_name, device_class, run_dir):
    log_dir = run_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    with open(log_dir / f"{path_name}.log", "w") as log:
        proc = subprocess.run(
            [
                sys.executable,
                str(HERE / "worker.py"),
                path_name,
                device_class,
                str(run_dir),
            ],
            stdout=log,
            stderr=subprocess.STDOUT,
            cwd=ROOT,
            check=False,
        )
    return proc.returncode


def load_records(run_dir):
    records = []
    for f in sorted((run_dir / "runs").glob("*/*.json")):
        with open(f) as fh:
            records.append(json.load(fh))
    return records


def output_diff(run_dir, path, ref_path, size, case):
    a = run_dir / "runs" / path / f"{size}__{case}.npy"
    b = run_dir / "runs" / ref_path / f"{size}__{case}.npy"
    if not (a.exists() and b.exists()):
        return None
    x, y = np.load(a), np.load(b)
    if x.shape != y.shape:
        return {"shape_mismatch": [list(x.shape), list(y.shape)]}
    d = np.abs(x - y)
    return {
        "max": float(d.max()),
        "mean": float(d.mean()),
        "frac_gt_1_255": float((d > 1 / 255).mean()),
        "frac_gt_8_255": float((d > 8 / 255).mean()),
    }


def aggregate(run_dir, records):
    by_key = {}
    for r in records:
        by_key.setdefault((r["size"], r["case"]), {})[r["path"]] = r
    table = []
    for (size, case), paths in sorted(by_key.items()):
        device_class = "cpu" if any(p in CPU_PATHS for p in paths) else "gpu"
        ref_path = REFERENCE_PATH[device_class]
        torch_paths = [p for p in paths if p.startswith("torch") and paths[p]["ok"]]
        best_torch = min(
            torch_paths,
            key=lambda p: paths[p]["timing"]["median_ms"],
            default=None,
        )
        for p, r in paths.items():
            row = {
                "size": size,
                "case": case,
                "path": p,
                "ok": r["ok"],
                "error": r.get("error"),
            }
            if r["ok"]:
                row["median_ms"] = r["timing"]["median_ms"]
                row["p10_ms"] = r["timing"]["p10_ms"]
                row["p90_ms"] = r["timing"]["p90_ms"]
                row["cold_ms"] = r["timing"]["cold_ms"]
                mem = r["memory"]
                row["allocator_peak_reserved"] = mem.get("allocator_peak_reserved")
                row["slang_peak_pool_bytes"] = mem.get("slang_peak_pool_bytes")
                dev_before = mem.get("device_used_before")
                dev_after = mem.get("device_used_after")
                row["device_delta_bytes"] = (
                    dev_after - dev_before
                    if dev_before is not None and dev_after is not None
                    else None
                )
                row["peak_private_bytes"] = mem["process"].get("peak_private")
                if best_torch is not None and p.startswith("slang"):
                    row["best_torch_path"] = best_torch
                    row["speedup_vs_best_torch"] = (
                        paths[best_torch]["timing"]["median_ms"] / row["median_ms"]
                    )
                    row["speedup_vs_each_torch"] = {
                        tp: paths[tp]["timing"]["median_ms"] / row["median_ms"]
                        for tp in torch_paths
                    }
                if p != ref_path:
                    row["diff_vs_reference"] = output_diff(
                        run_dir, p, ref_path, size, case
                    )
            table.append(row)
    return table


def fmt_bytes(n):
    if n is None:
        return "-"
    return f"{n / 2**20:.1f}"


def write_markdown(run_dir, table, meta):
    lines = [f"# Slang vs torch benchmark ({meta['stamp']})", ""]
    lines.append(f"Machine: {meta['machine']}")
    lines.append("")
    lines.append(
        "| size | case | path | median ms | p10 | p90 | cold ms | "
        "speedup vs best torch | alloc/pool MiB | device delta MiB | "
        "max |diff| | frac >1/255 |"
    )
    lines.append("|" + "---|" * 12)
    for r in table:
        if not r["ok"]:
            lines.append(
                f"| {r['size']} | {r['case']} | {r['path']} | FAILED: "
                f"{(r['error'] or '')[:80]} |||||||||"
            )
            continue
        mem = r.get("slang_peak_pool_bytes") or r.get("allocator_peak_reserved")
        diff = r.get("diff_vs_reference") or {}
        speed = r.get("speedup_vs_best_torch")
        lines.append(
            f"| {r['size']} | {r['case']} | {r['path']} | {r['median_ms']:.3f} | "
            f"{r['p10_ms']:.3f} | {r['p90_ms']:.3f} | {r['cold_ms']:.1f} | "
            f"{'' if speed is None else f'{speed:.2f}x'} | {fmt_bytes(mem)} | "
            f"{fmt_bytes(r.get('device_delta_bytes'))} | "
            f"{diff.get('max', '')} | {diff.get('frac_gt_1_255', '')} |"
        )
    (run_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", default="")
    parser.add_argument("--device", default="all", choices=["gpu", "cpu", "all"])
    args = parser.parse_args()
    stamp = datetime.datetime.now().astimezone().strftime("%Y%m%d-%H%M%S")
    run_dir = OUT_ROOT / stamp
    run_dir.mkdir(parents=True, exist_ok=True)

    jobs = []
    if args.device in ("gpu", "all"):
        jobs += [(p, "gpu") for p in GPU_PATHS]
    if args.device in ("cpu", "all"):
        jobs += [(p, "cpu") for p in CPU_PATHS]
    if args.only:
        keep = set(args.only.split(","))
        jobs = [j for j in jobs if j[0] in keep]

    exit_codes = {}
    for path_name, device_class in jobs:
        print(f"== {path_name} ({device_class})", flush=True)
        exit_codes[path_name] = run_worker(path_name, device_class, run_dir)

    records = load_records(run_dir)
    table = aggregate(run_dir, records)
    meta = {
        "stamp": stamp,
        "machine": platform_summary(),
        "exit_codes": exit_codes,
    }
    with open(run_dir / "report.json", "w") as f:
        json.dump({"meta": meta, "table": table, "records": records}, f, indent=1)
    write_markdown(run_dir, table, meta)
    print(f"report: {run_dir / 'report.md'}")


def platform_summary():
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none"
    return (
        f"{platform.platform()} | {platform.processor()} | "
        f"torch {torch.__version__} | GPU {gpu}"
    )


if __name__ == "__main__":
    main()
