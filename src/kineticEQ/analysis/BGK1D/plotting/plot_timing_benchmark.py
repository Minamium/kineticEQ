# kineticEQ/analysis/BGK1D/plotting/plot_timing_benchmark.py
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple
import numpy as np


def plot_timing_benchmark(
    out: Dict[str, Any],
    *,
    out_dir: str | Path = "./result",
    fname: str = "timing_benchmark_heatmap.png",
    save_fig: bool = True,
    show_gpu_time: bool = True,
    show_overhead: bool = True,
    figsize: Tuple[float, float] | None = None,
    text_fontsize: int = 8,
) -> Dict[str, Any]:
    """
    New-format timing benchmark plotter (BGK1D).

    Expected format (from run_benchmark in kineticEQ/analysis/BGK1D/benchmark.py):
      out["meta"]["bench_type"] == "time"
      out["records"] contains entries with:
        {
          "tag": ...,
          "sweep": {"nx": int, "nv": int},
          "timing": {
             "total_steps": int,
             "cpu_total_time_sec": float,
             # optional CUDA event timing:
             "gpu_total_time_sec": float,  # if measured
             "gpu_total_time_ms": float,   # if measured
          }
        }

    Output:
      out_dir/fname
    """
    import matplotlib.pyplot as plt

    if not isinstance(out, dict):
        raise TypeError("out must be a dict")
    meta = out.get("meta", {})
    if meta.get("bench_type") != "time":
        raise ValueError(f"plot_timing_benchmark requires meta['bench_type']=='time', got {meta.get('bench_type')}")

    # output path
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / fname

    # collect timing records
    records = out.get("records", [])
    t_recs = [
        r for r in records
        if isinstance(r, dict)
        and isinstance(r.get("sweep"), dict)
        and isinstance(r.get("timing"), dict)
    ]
    if len(t_recs) == 0:
        raise ValueError("No timing records found in out['records']")

    # meta labels
    device_name = meta.get("device_name", meta.get("device", "Unknown Device"))
    cpu_name = meta.get("cpu_name", "Unknown CPU")

    # grid sets
    nx_vals = sorted({int(r["sweep"].get("nx", -1)) for r in t_recs if int(r["sweep"].get("nx", -1)) > 0})
    nv_vals = sorted({int(r["sweep"].get("nv", -1)) for r in t_recs if int(r["sweep"].get("nv", -1)) > 0})
    if len(nx_vals) == 0 or len(nv_vals) == 0:
        raise ValueError("Invalid sweep values in timing records")

    nx_index = {nx: i for i, nx in enumerate(nx_vals)}
    nv_index = {nv: j for j, nv in enumerate(nv_vals)}

    # matrices: ms/step
    cpu_ms = np.full((len(nx_vals), len(nv_vals)), np.nan, dtype=float)
    gpu_ms = np.full((len(nx_vals), len(nv_vals)), np.nan, dtype=float)
    has_gpu = False

    for r in t_recs:
        nx = int(r["sweep"]["nx"])
        nv = int(r["sweep"]["nv"])
        i = nx_index[nx]
        j = nv_index[nv]

        timing = r["timing"]
        steps = int(timing.get("total_steps", 0))
        if steps <= 0:
            continue

        cpu_total = float(timing.get("cpu_total_time_sec", np.nan))
        if np.isfinite(cpu_total):
            cpu_ms[i, j] = (cpu_total / steps) * 1000.0

        # prefer *_sec if present; else derive from *_ms
        if "gpu_total_time_sec" in timing:
            gsec = float(timing.get("gpu_total_time_sec", np.nan))
            if np.isfinite(gsec):
                gpu_ms[i, j] = (gsec / steps) * 1000.0
                has_gpu = True
        elif "gpu_total_time_ms" in timing:
            gms = float(timing.get("gpu_total_time_ms", np.nan))
            if np.isfinite(gms):
                gpu_ms[i, j] = gms / steps
                has_gpu = True

    # figure layout
    subplot_num = 1
    if has_gpu and show_gpu_time:
        subplot_num += 1
    if has_gpu and show_overhead:
        subplot_num += 1

    if figsize is None:
        figsize = (4 * subplot_num, 6)

    plt.rcParams["font.family"] = "DejaVu Sans"
    fig, axes = plt.subplots(1, subplot_num, figsize=figsize, constrained_layout=True)
    fig.subplots_adjust(wspace=0.25)
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes], dtype=object)

    def _contrast_color(val: float, im) -> str:
        rgba = im.cmap(im.norm(val))
        lum = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
        return "black" if lum > 0.5 else "white"

    def _annotate(ax, mat, im):
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                v = mat[i, j]
                if not np.isfinite(v):
                    continue
                ax.text(
                    j, i, f"{v:.2f}",
                    ha="center", va="center",
                    color=_contrast_color(v, im),
                    fontsize=text_fontsize,
                )

    # CPU heatmap
    ax0 = axes[0]
    im1 = ax0.imshow(cpu_ms, cmap="viridis", aspect="auto")
    ax0.set_title(f"Wall-clock Time per Step (ms) - {cpu_name}")
    ax0.set_xlabel("nv (Velocity Grid Points)")
    ax0.set_ylabel("nx (Spatial Grid Points)")
    ax0.set_xticks(range(len(nv_vals)))
    ax0.set_xticklabels(nv_vals)
    ax0.set_yticks(range(len(nx_vals)))
    ax0.set_yticklabels(nx_vals)
    _annotate(ax0, cpu_ms, im1)
    fig.colorbar(im1, ax=ax0, label="Time (ms)")

    ax_idx = 1

    # GPU heatmap
    if has_gpu and show_gpu_time:
        ax = axes[ax_idx]
        im2 = ax.imshow(gpu_ms, cmap="plasma", aspect="auto")
        ax.set_title(f"GPU Time per Step (ms) - {device_name}")
        ax.set_xlabel("nv (Velocity Grid Points)")
        ax.set_ylabel("nx (Spatial Grid Points)")
        ax.set_xticks(range(len(nv_vals)))
        ax.set_xticklabels(nv_vals)
        ax.set_yticks(range(len(nx_vals)))
        ax.set_yticklabels(nx_vals)
        _annotate(ax, gpu_ms, im2)
        fig.colorbar(im2, ax=ax, label="Time (ms)")
        ax_idx += 1

    # Overhead (CPU - GPU)
    if has_gpu and show_overhead:
        diff = cpu_ms - gpu_ms
        ax = axes[ax_idx]
        im3 = ax.imshow(diff, cmap="magma", aspect="auto")
        ax.set_title("ΔT (CPU - GPU) per Step (ms)")
        ax.set_xlabel("nv (Velocity Grid Points)")
        ax.set_ylabel("nx (Spatial Grid Points)")
        ax.set_xticks(range(len(nv_vals)))
        ax.set_xticklabels(nv_vals)
        ax.set_yticks(range(len(nx_vals)))
        ax.set_yticklabels(nx_vals)
        _annotate(ax, diff, im3)
        fig.colorbar(im3, ax=ax, label="ΔT (ms)")

    if save_fig:
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        print(f"Saved: {out_path}")

    plt.close(fig)

    return {
        "figure_saved": str(out_path) if save_fig else None,
        "out_dir": str(out_dir),
        "has_gpu_data": bool(has_gpu),
        "nx_sorted": nx_vals,
        "nv_sorted": nv_vals,
        "cpu_ms_per_step": cpu_ms,
        "gpu_ms_per_step": gpu_ms if has_gpu else None,
    }
