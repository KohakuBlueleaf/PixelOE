"""Per-kernel profile of the Slang pipeline on every device and backend.

Every device runs in its own subprocess and writes one JSON with, per
(size, case):
  e2e       wall time of pixelize() from a torch tensor to a torch tensor
            (median / p10 / p90 over >= 5 runs, plain context)
  host      wall time until pixelize() returns, before the final sync: the
            host cost of recording and submitting (GPU backends with CUDA
            interop return before the GPU finishes)
  kernels   per entry point: calls, device ms per run, grids, workgroup size
            (profiling context, averaged over PROFILE_RUNS runs)
  log       every dispatch of one run, in order: name, grid, group, ms
  derived   kernel_ms (sum), gap_ms = e2e - kernel_ms (launch, sync,
            transfer and idle time), dispatches and submits per run
summary.md tabulates all devices; a cross-device table lists each kernel's
time at 1920x1080 default settings.

usage: python profile_devices.py [--only "4090 CUDA,CPU 172t"]
       -> outputs/bench/profile/<stamp>/
"""

import argparse
import datetime
import json
import statistics
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image

import pixeloe.torch.env as pixeloe_env
from pixeloe.slang.pixelize import pixelize
from pixeloe.slang.runtime.registry import create_context

ROOT = Path(__file__).resolve().parents[3]
OUT_ROOT = ROOT / "outputs" / "bench" / "profile"
IMAGE = ROOT / "img" / "snow-leopard.webp"
DEVICES = {
    "4090 CUDA": "cuda",
    "4090 Vulkan": "vulkan",
    "4090 D3D12": "d3d12",
    "B50 D3D12": "d3d12:B50",
    "B50 Vulkan": "vulkan:B50",
    "CPU 172t": "cpu:t172",
    "CPU 86t": "cpu:t86",
}
SIZES = {
    "512": (512, 512, 1),
    "1080p": (1920, 1080, 1),
    "4k": (3840, 2160, 1),
    "1024x4": (1024, 1024, 4),
}
CASES = {
    "default": {},
    "thickness0": {"thickness": 0},
    "no_color_match": {"do_color_match": False},
    "k_centroid": {"mode": "k_centroid"},
    "sliding": {"local_stats": "sliding"},
    "quant32_ordered": {"do_quant": True, "num_colors": 32},
    "quant32_ed": {
        "do_quant": True,
        "num_colors": 32,
        "dither_mode": "error_diffusion",
    },
    "repeat_kmeans": {
        "do_quant": True,
        "num_colors": 32,
        "quant_mode": "repeat-kmeans",
    },
    "lowrank1": {"blur_impl": "lowrank", "blur_rank": 1},
    "lowrank1_thickness0": {"blur_impl": "lowrank", "blur_rank": 1, "thickness": 0},
}
REFERENCE = ("1080p", "default")
MIN_RUNS, MAX_RUNS, TARGET_S = 10, 60, 2.0
WARMUP_S = 0.5  # continuous runs first: the display GPU idles at low clocks
PROFILE_RUNS = 3


def load(width, height, batch, device):
    img = Image.open(IMAGE).convert("RGB").resize((width, height), Image.BICUBIC)
    t = torch.from_numpy(np.asarray(img).copy()).permute(2, 0, 1).float() / 255
    return t[None].repeat(batch, 1, 1, 1).contiguous().to(device)


def sync_fn(x):
    return torch.cuda.synchronize if x.is_cuda else (lambda: None)


def timed(ctx, x, kw):
    sync = sync_fn(x)
    start = time.perf_counter()
    while time.perf_counter() - start < WARMUP_S:
        pixelize(x, context=ctx, **kw)
        sync()
    e2e, host = [], []
    start = time.perf_counter()
    while len(e2e) < MAX_RUNS:
        t0 = time.perf_counter()
        pixelize(x, context=ctx, **kw)
        t1 = time.perf_counter()
        sync()
        t2 = time.perf_counter()
        host.append((t1 - t0) * 1e3)
        e2e.append((t2 - t0) * 1e3)
        if len(e2e) >= MIN_RUNS and time.perf_counter() - start > TARGET_S:
            break
    q = statistics.quantiles(e2e, n=10)
    return {
        "e2e_ms": statistics.median(e2e),
        "e2e_min_ms": min(e2e),
        "e2e_p10_ms": q[0],
        "e2e_p90_ms": q[-1],
        "host_ms": statistics.median(host),
        "runs": len(e2e),
        "e2e_all_ms": e2e,
    }


def profiled(pctx, x, kw):
    start = time.perf_counter()
    while time.perf_counter() - start < WARMUP_S:
        pixelize(x, context=pctx, **kw)
        pctx.wait()
    pctx.stats.reset_peaks()
    for _ in range(PROFILE_RUNS):
        pixelize(x, context=pctx, **kw)
        pctx.wait()
    log = pctx.stats.dispatch_log
    per_run = len(log) // PROFILE_RUNS
    kernels = defaultdict(lambda: {"calls": 0, "ms": 0.0, "grids": set()})
    for d in log:
        k = kernels[d["name"]]
        k["calls"] += 1
        k["ms"] += d["ms"]
        k["group"] = list(d["group"])
        k["grids"].add(tuple(d["threads"]))
    table = [
        {
            "name": name,
            "calls": k["calls"] / PROFILE_RUNS,
            "ms": k["ms"] / PROFILE_RUNS,
            "group": k["group"],
            "grids": sorted(k["grids"]),
        }
        for name, k in kernels.items()
    ]
    table.sort(key=lambda r: -r["ms"])
    return {
        "kernels": table,
        "log": log[-per_run:],
        "kernel_ms": sum(r["ms"] for r in table),
        "dispatches": pctx.stats.dispatches / PROFILE_RUNS,
        "submits": pctx.stats.submits / PROFILE_RUNS,
    }


def configs():
    for size, (w, h, b) in SIZES.items():
        for case, extra in CASES.items():
            yield size, case, (w, h, b), {"pixel_size": 4, "thickness": 3, **extra}


def run_phase(name, ctx, device, records, fn):
    """fn(ctx, x, kw) for every config into records[(size, case)]."""
    for size, case, (w, h, b), kw in configs():
        rec = records.setdefault(
            (size, case),
            {
                "size": size,
                "case": case,
                "width": w,
                "height": h,
                "batch": b,
                "ok": True,
            },
        )
        if not rec["ok"]:
            continue
        try:
            rec.update(fn(ctx, load(w, h, b, device), kw))
        except Exception as exc:  # noqa: BLE001 - recorded, the run goes on
            rec["ok"] = False
            rec["error"] = f"{fn.__name__}: {type(exc).__name__}: {exc}"[:400]
        print(f"[{name}] {fn.__name__} {size}/{case}: ok={rec['ok']}", flush=True)


def worker(name, out_dir):
    """Timing and profiling in two phases with one live context each: two
    devices sharing the GPU in one process slow each other down."""
    spec = DEVICES[name]
    ctx = create_context(spec)
    device = "cuda" if getattr(ctx, "uses_cuda_stream", False) else "cpu"
    meta = {
        "device": name,
        "spec": spec,
        "adapter": getattr(ctx, "adapter_name", "cpu"),
        "transfer": getattr(ctx, "transfer", "host"),
        "torch_device": device,
    }
    records = {}
    run_phase(name, ctx, device, records, timed)
    if hasattr(ctx, "close"):
        ctx.close()
    del ctx
    pctx = create_context(spec, profile=True)
    run_phase(name, pctx, device, records, profiled)
    for rec in records.values():
        if rec["ok"]:
            rec["gap_ms"] = rec["e2e_ms"] - rec["kernel_ms"]
    records = list(records.values())
    path = out_dir / f"{name.replace(' ', '_')}.json"
    path.write_text(json.dumps({"meta": meta, "records": records}, indent=1))


def summarize(out_dir):
    runs = [json.loads(p.read_text()) for p in sorted(out_dir.glob("*.json"))]
    lines = [f"# Slang per-kernel profile ({out_dir.name})", ""]
    lines += [
        "## End to end",
        "",
        "| device | size | case | e2e ms | min | p10-p90 | host ms | kernel ms | gap ms | "
        "dispatches | submits | top kernels (ms) |",
        "|" + "---|" * 12,
    ]
    for run in runs:
        dev = run["meta"]["device"]
        for r in run["records"]:
            if not r["ok"]:
                lines.append(
                    f"| {dev} | {r['size']} | {r['case']} | FAILED {r['error'][:60]} |"
                )
                continue
            top = ", ".join(f"{k['name']} {k['ms']:.2f}" for k in r["kernels"][:4])
            lines.append(
                f"| {dev} | {r['size']} | {r['case']} | {r['e2e_ms']:.2f} | "
                f"{r['e2e_min_ms']:.2f} | {r['e2e_p10_ms']:.2f}-{r['e2e_p90_ms']:.2f} | {r['host_ms']:.2f} | {r['kernel_ms']:.2f} | {r['gap_ms']:.2f} | "
                f"{r['dispatches']:.0f} | {r['submits']:.0f} | {top} |"
            )
    ref = {}
    for run in runs:
        for r in run["records"]:
            if (r["size"], r["case"]) == REFERENCE and r["ok"]:
                ref[run["meta"]["device"]] = {k["name"]: k for k in r["kernels"]}
    names = sorted(
        {n for ks in ref.values() for n in ks},
        key=lambda n: -max(ks.get(n, {"ms": 0})["ms"] for ks in ref.values()),
    )
    devs = list(ref)
    lines += [
        "",
        f"## Kernels at {REFERENCE[0]} / {REFERENCE[1]} (ms per run, calls, group)",
        "",
        "| kernel | group | calls | " + " | ".join(devs) + " |",
        "|" + "---|" * (3 + len(devs)),
    ]
    for n in names:
        any_k = next(ks[n] for ks in ref.values() if n in ks)
        cells = [f"{ref[d][n]['ms']:.3f}" if n in ref[d] else "-" for d in devs]
        lines.append(
            f"| {n} | {tuple(any_k['group'])} | {any_k['calls']:.0f} | "
            + " | ".join(cells)
            + " |"
        )
    (out_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"summary: {out_dir / 'summary.md'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", nargs=2, metavar=("DEVICE", "OUT_DIR"))
    ap.add_argument("--only", default="")
    ap.add_argument("--summarize", default="")
    a = ap.parse_args()
    pixeloe_env.TORCH_COMPILE = False
    if a.worker:
        worker(a.worker[0], Path(a.worker[1]))
        return
    if a.summarize:
        summarize(Path(a.summarize))
        return
    stamp = datetime.datetime.now().astimezone().strftime("%Y%m%d-%H%M%S")
    out_dir = OUT_ROOT / stamp
    (out_dir / "logs").mkdir(parents=True)
    names = [n for n in DEVICES if not a.only or n in a.only.split(",")]
    for name in names:
        print(f"== {name}", flush=True)
        with open(out_dir / "logs" / f"{name.replace(' ', '_')}.log", "w") as log:
            subprocess.run(
                [sys.executable, __file__, "--worker", name, str(out_dir)],
                stdout=log,
                stderr=subprocess.STDOUT,
                cwd=ROOT,
                check=False,
            )
    summarize(out_dir)


if __name__ == "__main__":
    main()
