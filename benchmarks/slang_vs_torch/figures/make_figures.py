"""Paper-style benchmark figures (PNG, 300 dpi, two-column width).

usage: python make_figures.py [--sweep RUN] [--report RUN] [--out DIR]
  defaults: latest outputs/bench/sweeps/<stamp>, latest
            outputs/bench/slang_vs_torch/<stamp>, benchmarks/slang_vs_torch/figures

Figures:
  fig_gpu_scaling.png   2x3: latency / throughput / memory vs resolution,
                        per-image time vs batch, time vs pixel size, vs palette
  fig_gpu_speedup.png   1x5: Slang over the fastest torch variant, every sweep
  fig_cpu.png           1x5: CPU latency vs resolution, batch, P, palette; memory
  fig_overview.png      1x2: speed-up range per path (440 configs), 1080p memory
  fig_devices.png       1x3: Slang only - CPU workers vs RTX 4090 vs Arc B50
                        (from devices.py; skipped when no devices run exists)
"""

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]

BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN = (
    "#2a78d6",
    "#eb6834",
    "#1baf7a",
    "#eda100",
    "#e87ba4",
    "#008300",
)
INK, INK2, GRID = "#1a1c20", "#4b505c", "#e3e5ea"

# (key, label, colour, linestyle, marker)
GPU = [
    ("slang-cuda", "Slang CUDA", BLUE, "-", "o"),
    ("slang-vulkan", "Slang Vulkan", AQUA, "-", "s"),
    ("slang-d3d12", "Slang D3D12", MAGENTA, "-", "^"),
    ("torch-cuda-fp32", "torch fp32", ORANGE, "--", "o"),
    ("torch-cuda-fp16", "torch fp16", YELLOW, "--", "s"),
    ("torch-cuda-fp16-compile", "torch.compile fp16", GREEN, "--", "^"),
]
CPU = [
    ("slang-cpu", "Slang CPU (native, 64 threads)", BLUE, "-", "o"),
    ("torch-cpu-fp32", "torch CPU fp32", ORANGE, "--", "o"),
]
GPU_TORCH = ["torch-cuda-fp32", "torch-cuda-fp16", "torch-cuda-fp16-compile"]
SLANG_GPU = ["slang-cuda", "slang-vulkan", "slang-d3d12"]


def style():
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 7,
            "axes.titlesize": 7.5,
            "axes.titleweight": "bold",
            "axes.labelsize": 7,
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 6.5,
            "legend.fontsize": 6.8,
            "axes.edgecolor": INK2,
            "axes.labelcolor": INK,
            "xtick.color": INK2,
            "ytick.color": INK2,
            "axes.linewidth": 0.6,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "xtick.minor.width": 0.4,
            "ytick.minor.width": 0.4,
            "xtick.major.size": 2.5,
            "ytick.major.size": 2.5,
            "xtick.minor.size": 1.5,
            "ytick.minor.size": 1.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.5,
            "lines.linewidth": 1.1,
            "lines.markersize": 3.2,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.03,
        }
    )


def num(v, _pos=None):
    if v >= 1000:
        return f"{v / 1000:g}k"
    return f"{v:g}"


def log_axis(ax, which="both", subs=(1, 2, 5)):
    """Log axis labelled at subs x 10^k; narrow panels pass subs=(1,)."""
    for axis in ("x", "y") if which == "both" else (which,):
        getattr(ax, f"set_{axis}scale")("log")
        a = getattr(ax, f"{axis}axis")
        a.set_major_locator(LogLocator(base=10, subs=subs))
        a.set_major_formatter(FuncFormatter(num))
        a.set_minor_formatter(NullFormatter())


def panel_tag(ax, tag):
    ax.set_title(tag, loc="left")


def load_sweep(run):
    data = defaultdict(dict)  # (sweep, path) -> {x: rec}
    for line in (run / "points.jsonl").read_text().splitlines():
        r = json.loads(line) if line.strip() else None
        if r and r.get("ok"):
            data[(r["sweep"], r["path"])][r["x"]] = r
    return data


def series(ax, data, sweep, table, value, xmap=lambda x, r: x):
    for key, label, color, ls, mk in table:
        recs = data.get((sweep, key), {})
        pts = sorted((xmap(x, r), value(r)) for x, r in recs.items() if value(r))
        if pts:
            xs, ys = zip(*pts)
            ax.plot(
                xs, ys, ls=ls, marker=mk, color=color, label=label, markeredgewidth=0
            )


def best_torch(data, sweep, x):
    vals = [
        data[(sweep, t)][x]["ms"] for t in GPU_TORCH if x in data.get((sweep, t), {})
    ]
    return min(vals) if vals else None


def fig_gpu_scaling(data, out):
    fig, axs = plt.subplots(2, 3, figsize=(7.0, 4.3))
    mpx = lambda x, r: x / 1e6

    ax = axs[0, 0]
    series(ax, data, "pixels", GPU, lambda r: r["ms"], mpx)
    log_axis(ax)
    ax.set(xlabel="input size (megapixels, batch 1)", ylabel="latency (ms)")
    panel_tag(ax, "(a) latency vs resolution")

    ax = axs[0, 1]
    series(ax, data, "pixels", GPU, lambda r: r["x"] / r["ms"] / 1e3, mpx)
    log_axis(ax)
    ax.set(xlabel="input size (megapixels)", ylabel="throughput (MPix/s)")
    panel_tag(ax, "(b) throughput vs resolution")

    ax = axs[0, 2]
    # the three Slang backends share one buffer pool: identical curves, one line
    mem_table = [("slang-cuda", "Slang (all backends)", BLUE, "-", "o")] + GPU[3:]
    series(ax, data, "pixels", mem_table, lambda r: r.get("mem_mib"), mpx)
    log_axis(ax)
    ax.set(xlabel="input size (megapixels)", ylabel="peak GPU memory (MiB)")
    panel_tag(ax, "(c) memory vs resolution")

    ax = axs[1, 0]
    series(ax, data, "batch", GPU, lambda r: r["ms"] / r["batch"])
    ax.set_xscale("log", base=2)
    ax.xaxis.set_major_formatter(FuncFormatter(num))
    log_axis(ax, "y")
    ax.set(xlabel="batch size (512² each)", ylabel="latency per image (ms)")
    panel_tag(ax, "(d) batching")

    ax = axs[1, 1]
    series(ax, data, "pixel_size", GPU, lambda r: r["ms"])
    ax.set_xscale("log", base=2)
    ax.set_xticks([2, 3, 4, 6, 8, 12, 16])
    ax.xaxis.set_major_formatter(FuncFormatter(num))
    ax.xaxis.set_minor_formatter(NullFormatter())
    log_axis(ax, "y")
    ax.set(xlabel="pixel size P (1024² input)", ylabel="latency (ms)")
    panel_tag(ax, "(e) pixel size")

    ax = axs[1, 2]
    series(ax, data, "colors", GPU, lambda r: r["ms"])
    ax.set_xscale("log", base=2)
    ax.set_xticks([8, 16, 32, 64, 128, 256])
    ax.xaxis.set_major_formatter(FuncFormatter(num))
    ax.xaxis.set_minor_formatter(NullFormatter())
    log_axis(ax, "y")
    ax.set(xlabel="palette colours (k-means + dither)", ylabel="latency (ms)")
    panel_tag(ax, "(f) colour quantization")

    handles, labels = axs[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=6,
        frameon=False,
        bbox_to_anchor=(0.5, 1.035),
    )
    fig.tight_layout(h_pad=1.2, w_pad=1.0)
    fig.savefig(out / "fig_gpu_scaling.png")
    plt.close(fig)


def fig_gpu_speedup(data, out):
    sweeps = [
        ("pixels", "input size (MPix)", lambda x: x / 1e6, "log"),
        ("batch", "batch size", lambda x: x, "log2"),
        ("pixel_size", "pixel size P", lambda x: x, "log2"),
        ("thickness", "outline thickness", lambda x: x, "lin"),
        ("colors", "palette colours", lambda x: x, "log2"),
    ]
    fig, axs = plt.subplots(1, 5, figsize=(7.0, 1.75), sharey=True)
    for ax, (sweep, xlabel, xm, scale) in zip(axs, sweeps):
        for key, label, color, _ls, mk in GPU[:3]:
            pts = []
            for x, r in data.get((sweep, key), {}).items():
                b = best_torch(data, sweep, x)
                if b:
                    pts.append((xm(x), b / r["ms"]))
            if pts:
                xs, ys = zip(*sorted(pts))
                ax.plot(xs, ys, marker=mk, color=color, label=label, markeredgewidth=0)
        ax.axhline(1, color=INK2, lw=0.6)
        if scale == "log":
            log_axis(ax, "x", subs=(1,))
        elif scale == "log2":
            ax.set_xscale("log", base=2)
            xs = sorted({xm(x) for x in data.get((sweep, "slang-cuda"), {})})
            ax.set_xticks(xs)
            ax.xaxis.set_major_formatter(FuncFormatter(num))
            ax.xaxis.set_minor_formatter(NullFormatter())
        ax.set_xlabel(xlabel)
        ax.set_ylim(0, None)
    axs[0].set_ylabel("speed-up over fastest torch (×)")
    for ax, t in zip(axs, "abcde"):
        panel_tag(ax, f"({t})")
    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 1.1),
    )
    fig.tight_layout(w_pad=0.6)
    fig.savefig(out / "fig_gpu_speedup.png")
    plt.close(fig)


def fig_cpu(data, out):
    fig, axs = plt.subplots(1, 5, figsize=(7.0, 1.8))
    mpx = lambda x, r: x / 1e6
    series(axs[0], data, "pixels", CPU, lambda r: r["ms"], mpx)
    log_axis(axs[0], "x", subs=(1,))
    log_axis(axs[0], "y")
    axs[0].set(xlabel="input size (MPix)", ylabel="latency (ms)")
    series(axs[1], data, "batch", CPU, lambda r: r["ms"] / r["batch"])
    axs[1].set_xscale("log", base=2)
    axs[1].set_xticks([1, 2, 4])
    axs[1].xaxis.set_major_formatter(FuncFormatter(num))
    axs[1].set(xlabel="batch size (512²)", ylabel="ms per image")
    series(axs[2], data, "pixel_size", CPU, lambda r: r["ms"])
    axs[2].set_xscale("log", base=2)
    axs[2].set_xticks([2, 4, 8, 16])
    axs[2].xaxis.set_major_formatter(FuncFormatter(num))
    axs[2].xaxis.set_minor_formatter(NullFormatter())
    axs[2].set(xlabel="pixel size P (512²)", ylabel="latency (ms)")
    series(axs[3], data, "colors", CPU, lambda r: r["ms"])
    axs[3].set_xscale("log", base=2)
    axs[3].set_xticks([8, 32, 128])
    axs[3].xaxis.set_major_formatter(FuncFormatter(num))
    axs[3].xaxis.set_minor_formatter(NullFormatter())
    axs[3].set(xlabel="palette colours (512²)", ylabel="latency (ms)")
    series(axs[4], data, "pixels", CPU, lambda r: r.get("host_mem_mib") or None, mpx)
    log_axis(axs[4], "x", subs=(1,))
    log_axis(axs[4], "y")
    axs[4].set(xlabel="input size (MPix)", ylabel="peak host memory (MiB)")
    for ax, t in zip(axs, "abcde"):
        panel_tag(ax, f"({t})")
        ax.set_ylim(0 if ax.get_yscale() == "linear" else None, None)
    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, 1.1),
    )
    fig.tight_layout(w_pad=0.5)
    fig.savefig(out / "fig_cpu.png")
    plt.close(fig)


def fig_overview(report_run, out):
    rep = json.loads((report_run / "report.json").read_text())
    recs = {(r["path"], r["size"], r["case"]): r for r in rep["records"] if r["ok"]}
    T = {k: r["timing"]["median_ms"] for k, r in recs.items()}
    cfgs = sorted({(s, c) for (_, s, c) in T})
    torch_gpu = GPU_TORCH + ["torch-cuda-fp32-compile"]
    paths = [
        ("slang-cuda", "Slang CUDA", BLUE),
        ("slang-cuda-separable", "Slang CUDA, separable blur", BLUE),
        ("slang-vulkan", "Slang Vulkan", AQUA),
        ("slang-d3d12", "Slang D3D12", MAGENTA),
        ("slang-cpu", "Slang CPU (vs torch CPU)", INK2),
    ]
    fig, (a1, a2) = plt.subplots(
        1, 2, figsize=(7.0, 1.9), gridspec_kw={"width_ratios": [1.25, 1]}
    )
    for i, (p, label, color) in enumerate(paths):
        tp = ["torch-cpu-fp32"] if p == "slang-cpu" else torch_gpu
        xs = []
        for s, c in cfgs:
            if (p, s, c) in T:
                best = [T[(t, s, c)] for t in tp if (t, s, c) in T]
                if best:
                    xs.append(min(best) / T[(p, s, c)])
        y = len(paths) - 1 - i
        a1.plot(
            [min(xs), max(xs)],
            [y, y],
            color=GRID,
            lw=3.2,
            solid_capstyle="round",
            zorder=1,
        )
        a1.scatter(xs, [y] * len(xs), s=6, color=color, alpha=0.35, lw=0, zorder=2)
        med = statistics.median(xs)
        a1.scatter([med], [y], s=26, color=color, edgecolor="white", lw=0.8, zorder=3)
        a1.text(
            max(xs) * 1.08,
            y,
            f"median {med:.1f}×  (min {min(xs):.1f}×, n={len(xs)})",
            va="center",
            fontsize=6.3,
            color=INK,
        )
    a1.set_yticks(range(len(paths)))
    a1.set_yticklabels([p[1] for p in reversed(paths)])
    log_axis(a1, "x")
    a1.set_xlim(0.9, 60)
    a1.axvline(1, color=INK2, lw=0.6)
    a1.grid(axis="y", visible=False)
    a1.set_xlabel("speed-up over fastest torch variant, per configuration (×)")
    panel_tag(a1, "(a) all 440 benchmark configurations")

    rows = []
    for key, label, color, _ls, _mk in GPU:
        r = recs.get((key, "1080p", "default"))
        if r:
            m = r["memory"]
            b = m.get("slang_peak_pool_bytes") or m.get("allocator_peak_reserved")
            rows.append((label, b / 2**20, color, T[(key, "1080p", "default")]))
    rows.sort(key=lambda r: r[3])
    for label, mib, color, ms in rows:
        a2.scatter([ms], [mib], s=28, color=color, edgecolor="white", lw=0.6, zorder=3)
        if not label.startswith("Slang"):
            a2.annotate(
                label,
                (ms, mib),
                xytext=(4, 3),
                textcoords="offset points",
                fontsize=6,
                color=INK,
            )
    # the Slang backends share one buffer pool: one label under the group
    slang = [r for r in rows if r[0].startswith("Slang")]
    a2.annotate(
        "Slang CUDA / Vulkan / D3D12",
        (slang[0][3], slang[0][1]),
        xytext=(0, -11),
        textcoords="offset points",
        fontsize=6,
        color=INK,
    )
    log_axis(a2)
    a2.set(xlabel="latency, 1080p default (ms)", ylabel="peak GPU memory (MiB)")
    a2.set_xlim(3, 150)
    a2.set_ylim(100, 3000)
    panel_tag(a2, "(b) latency vs memory, 1920×1080")
    fig.tight_layout(w_pad=1.2)
    fig.savefig(out / "fig_overview.png")
    plt.close(fig)


DEVICE_STYLE = {
    "4090 CUDA": (BLUE, "-", "o"),
    "4090 Vulkan": (AQUA, "-", "s"),
    "4090 D3D12": (MAGENTA, "-", "^"),
    "B50 D3D12": (YELLOW, "-.", "D"),
    "B50 Vulkan": (GREEN, "-.", "v"),
    "CPU 172t": (ORANGE, "--", "o"),
    "CPU 16t": ("#4a3aa7", "--", "s"),
    "CPU 1t": (INK2, "--", "^"),
}


def fig_devices(run, out):
    pts = [
        json.loads(l)
        for l in (run / "points.jsonl").read_text().splitlines()
        if l.strip()
    ]
    cpu = sorted(
        (p["x"], p["ms"])
        for p in pts
        if p["sweep"] == "threads" and p["device"] == "CPU"
    )
    levels = {
        p["device"]: p["ms"]
        for p in pts
        if p["sweep"] == "threads" and p["device"] != "CPU"
    }
    res = defaultdict(list)
    for p in pts:
        if p["sweep"] == "resolution":
            res[p["device"]].append((p["x"] / 1e6, p["ms"]))

    fig, (a1, a2, a3) = plt.subplots(
        1, 3, figsize=(7.0, 2.25), gridspec_kw={"width_ratios": [1.15, 1.15, 0.9]}
    )
    xs, ys = zip(*cpu)
    a1.plot(
        xs,
        ys,
        color=ORANGE,
        ls="--",
        marker="o",
        markeredgewidth=0,
        label="CPU (Slang, native)",
    )
    for name, ms in sorted(levels.items(), key=lambda kv: kv[1]):
        color, ls, _ = DEVICE_STYLE[name]
        a1.axhline(ms, color=color, ls=ls, lw=1.0)
        if not name.startswith("4090"):
            a1.text(
                xs[0],
                ms * 1.08,
                f"{name}  {ms:.1f} ms",
                fontsize=5.8,
                color=color,
                va="bottom",
            )
    # the three 4090 levels sit within 1 ms: one label for the group
    rtx = sorted(
        (ms, name.split()[1]) for name, ms in levels.items() if name.startswith("4090")
    )
    if rtx:
        a1.text(
            xs[0],
            rtx[-1][0] * 1.12,
            "4090 "
            + " / ".join(n for _, n in rtx)
            + "  "
            + " / ".join(f"{m:.1f}" for m, _ in rtx)
            + " ms",
            fontsize=5.8,
            color=BLUE,
            va="bottom",
        )
    a1.set_xscale("log", base=2)
    a1.set_xticks([1, 2, 4, 8, 16, 32, 64, 172])
    a1.xaxis.set_major_formatter(FuncFormatter(num))
    a1.xaxis.set_minor_formatter(NullFormatter())
    log_axis(a1, "y")
    a1.set(
        xlabel="CPU worker threads (86 cores / 172 threads)",
        ylabel="latency, 1024² (ms)",
    )
    panel_tag(a1, "(a) CPU workers vs GPU levels")

    for name, (color, ls, mk) in DEVICE_STYLE.items():
        if name in res:
            x, y = zip(*sorted(res[name]))
            a2.plot(x, y, color=color, ls=ls, marker=mk, markeredgewidth=0, label=name)
    log_axis(a2, "x", subs=(1,))
    log_axis(a2, "y")
    a2.set(xlabel="input size (megapixels)", ylabel="latency (ms)")
    panel_tag(a2, "(b) latency vs resolution")

    t1 = dict(cpu)[1]
    a3.plot(
        xs,
        [t1 / y for y in ys],
        color=ORANGE,
        ls="--",
        marker="o",
        markeredgewidth=0,
        label="measured",
    )
    a3.plot([1, 172], [1, 172], color=INK2, lw=0.7, ls=":", label="ideal")
    a3.axvline(86, color=GRID, lw=1.0)
    a3.text(86, 1.2, " 86 cores", fontsize=5.8, color=INK2)
    for axis in ("x", "y"):
        getattr(a3, f"set_{axis}scale")("log", base=2)
        getattr(a3, f"{axis}axis").set_major_formatter(FuncFormatter(num))
        getattr(a3, f"{axis}axis").set_minor_formatter(NullFormatter())
    a3.set_xticks([1, 4, 16, 64, 172])
    a3.set_yticks([1, 4, 16, 64])
    a3.set(xlabel="CPU worker threads", ylabel="speed-up over 1 thread")
    a3.legend(loc="upper left", frameon=False)
    panel_tag(a3, "(c) CPU scaling")

    handles, labels = a2.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=8,
        frameon=False,
        bbox_to_anchor=(0.5, 1.07),
        fontsize=6.2,
    )
    fig.tight_layout(w_pad=1.0)
    fig.savefig(out / "fig_devices.png")
    plt.close(fig)


def latest(root):
    return max(p for p in root.iterdir() if p.is_dir())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", default="")
    ap.add_argument("--report", default="")
    ap.add_argument("--devices", default="")
    ap.add_argument("--out", default=str(HERE))
    a = ap.parse_args()
    sweep = Path(a.sweep) if a.sweep else latest(ROOT / "outputs" / "bench" / "sweeps")
    report = (
        Path(a.report)
        if a.report
        else latest(ROOT / "outputs" / "bench" / "slang_vs_torch")
    )
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    style()
    data = load_sweep(sweep)
    fig_gpu_scaling(data, out)
    fig_gpu_speedup(data, out)
    fig_cpu(data, out)
    fig_overview(report, out)
    devices_root = ROOT / "outputs" / "bench" / "devices"
    if devices_root.is_dir() and any(devices_root.iterdir()):
        fig_devices(Path(a.devices) if a.devices else latest(devices_root), out)
    print(f"figures -> {out} (sweep {sweep.name}, report {report.name})")


if __name__ == "__main__":
    main()
