# kineticEQ/analysis/BGK1D/plotting/plot_benchmark_result.py
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple
import warnings
import numpy as np


def plot_benchmark_results(
    out: Dict[str, Any],
    *,
    out_dir: str | Path = "./result",
    fname_moment: str = "moments.png",
    fname_error: str = "errors.png",
    logscale: bool = True,
    show_plots: bool = False,
) -> Dict[str, Any]:
    """
    New-format benchmark result plotter (BGK1D).

    - out_dir に指定されたディレクトリへ直接保存する
      例: out_dir="./result" -> ./result/moments.png, ./result/errors.png

    - bench_type in {"x_grid","v_grid"} のみ対応（time は別プロット推奨）
    """
    import matplotlib.pyplot as plt
    from itertools import cycle

    if not isinstance(out, dict):
        raise TypeError("out must be a dict")
    meta = out.get("meta", {})
    bench_type = meta.get("bench_type")
    if bench_type not in ("x_grid", "v_grid"):
        raise ValueError(f"plot_benchmark_results supports bench_type in {{'x_grid','v_grid'}}, got {bench_type}")

    # --- prepare output dir ---
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    moment_path = out_dir / fname_moment
    error_path = out_dir / fname_error

    # --- gather snapshots ---
    records = out.get("records", [])
    snaps = [r for r in records if isinstance(r, dict) and isinstance(r.get("fields"), dict)]
    snaps = [
        r for r in snaps
        if isinstance(r.get("sweep"), dict)
        and all(k in r["fields"] for k in ("x", "n", "u", "T"))
    ]
    if len(snaps) < 2:
        raise ValueError(f"Need at least 2 snapshot records to plot, got {len(snaps)}")

    def _nx_nv(r: Dict[str, Any]) -> Tuple[int, int]:
        sw = r.get("sweep", {})
        return int(sw.get("nx", -1)), int(sw.get("nv", -1))

    # reference (finest) consistent with append_errors()
    ref = max(snaps, key=(lambda r: _nx_nv(r)[0] if bench_type == "x_grid" else _nx_nv(r)[1]))
    ref_nx, ref_nv = _nx_nv(ref)

    # -------------------------
    # Moment comparison plot
    # -------------------------
    plt.rcParams["font.family"] = "DejaVu Sans"

    tol_colors = ["#4477AA", "#EE6677", "#228833", "#CCBB44", "#66CCEE", "#AA3377", "#BBBBBB"]
    line_styles = ["-", "--", "-.", ":"]
    color_cycle = cycle(tol_colors)
    style_cycle = cycle(line_styles)

    fig1, axes1 = plt.subplots(1, 3, figsize=(18, 5))

    if bench_type == "x_grid":
        snaps_sorted = sorted(snaps, key=lambda r: _nx_nv(r)[0], reverse=True)
    else:
        snaps_sorted = sorted(snaps, key=lambda r: _nx_nv(r)[1], reverse=True)
    snaps_sorted = [ref] + [r for r in snaps_sorted if r is not ref]

    legend_handles, legend_labels = [], []

    for r in snaps_sorted:
        color = next(color_cycle)
        linestyle = next(style_cycle)
        lw = 3 if r is ref else 2

        fields = r["fields"]
        x = np.asarray(fields["x"])
        n = np.asarray(fields["n"])
        u = np.asarray(fields["u"])
        T = np.asarray(fields["T"])

        nx, nv = _nx_nv(r)
        dx = fields.get("dx", None)
        dv = fields.get("dv", None)

        if bench_type == "x_grid":
            if dx is None:
                dx = 1.0 / max(nx, 1)
            grid_info = f"nx={nx} (dx={float(dx):.3g})"
        else:
            if dv is None:
                dv = 1.0 / max(nv, 1)
            grid_info = f"nv={nv} (dv={float(dv):.3g})"

        if r is ref:
            grid_info += " (REF)"

        ln, = axes1[0].plot(x, n, color=color, linestyle=linestyle, lw=lw, label=grid_info)
        axes1[1].plot(x, u, color=color, linestyle=linestyle, lw=lw)
        axes1[2].plot(x, T, color=color, linestyle=linestyle, lw=lw)

        legend_handles.append(ln)
        legend_labels.append(grid_info)

    titles = ["Density Distribution", "Velocity Distribution", "Temperature Distribution"]
    ylabels = ["Density n", "Mean Velocity u", "Temperature T"]
    for ax, title, ylabel in zip(axes1, titles, ylabels):
        ax.set_xlabel("Position x")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)

    fig1.legend(legend_handles, legend_labels, bbox_to_anchor=(1.02, 1), loc="upper left")
    fig1.savefig(moment_path, dpi=300, bbox_inches="tight", pad_inches=0.1, facecolor="white")
    if show_plots:
        plt.show()
    plt.close(fig1)

    # -------------------------
    # Error convergence plot
    # -------------------------
    errors = out.get("errors", [])
    if not isinstance(errors, list) or len(errors) == 0:
        raise ValueError("out['errors'] is empty. Ensure append_errors(out) is called before plotting.")

    def _is_same_ref(e: Dict[str, Any]) -> bool:
        ref_sw = e.get("ref", {})
        try:
            return int(ref_sw.get("nx", -999)) == ref_nx and int(ref_sw.get("nv", -999)) == ref_nv
        except Exception:
            return True

    entries = [e for e in errors if isinstance(e, dict) and isinstance(e.get("norms"), dict)]
    entries = [e for e in entries if _is_same_ref(e)]
    if len(entries) == 0:
        raise ValueError("No valid error entries found for plotting.")

    if bench_type == "x_grid":
        x_label = "Grid points nx"
        def _gridN(e): return int(e.get("sweep", {}).get("nx", -1))
        ref_N = ref_nx
    else:
        x_label = "Grid points nv"
        def _gridN(e): return int(e.get("sweep", {}).get("nv", -1))
        ref_N = ref_nv

    entries = [e for e in entries if _gridN(e) > 0 and _gridN(e) != ref_N]
    if len(entries) == 0:
        raise ValueError("Error entries contain only reference or invalid grid sizes.")

    entries = sorted(entries, key=_gridN)  # coarse -> fine

    fig2, axes2 = plt.subplots(1, 3, figsize=(18, 5))

    variables = ["f", "n", "u", "T"]
    norms = ["L1", "L2", "Linf"]
    var_colors = {"f": "#1f77b4", "n": "#2ca02c", "u": "#d62728", "T": "#9467bd"}
    markers = ["o", "s", "^", "D"]

    convergence_orders: Dict[str, Dict[str, float]] = {}

    for norm_idx, norm in enumerate(norms):
        ax = axes2[norm_idx]
        convergence_orders[norm] = {}

        for var_idx, var in enumerate(variables):
            counts: List[int] = []
            vals: List[float] = []

            for e in entries:
                N = _gridN(e)
                try:
                    val = float(e["norms"][norm][var])
                except Exception:
                    warnings.warn(f"Error retrieval failed: {var} {norm} sweep={e.get('sweep')}")
                    continue
                if not np.isfinite(val) or val <= 0:
                    warnings.warn(f"Invalid error value skipped: {var} {norm} N={N} error={val}")
                    continue
                counts.append(N)
                vals.append(val)

            if len(vals) < 2:
                convergence_orders[norm][var] = float("nan")
                continue

            counts_arr = np.asarray(counts, dtype=float)
            vals_arr = np.asarray(vals, dtype=float)

            slope = np.polyfit(np.log(counts_arr), np.log(vals_arr), 1)[0]
            p_mean = -float(slope)
            convergence_orders[norm][var] = p_mean

            label = f"{var} (p̅={p_mean:.2f})"
            plot = ax.loglog if logscale else ax.plot
            plot(
                counts_arr,
                vals_arr,
                marker=markers[var_idx],
                color=var_colors[var],
                linewidth=2,
                markersize=8,
                label=label,
            )

        ax.set_xlabel(f"{x_label}  (coarse → fine)")
        ax.set_ylabel(f"Error ({norm.replace('Linf', 'L∞')})")
        ax.set_title(f"{norm.replace('Linf', 'L∞')} Norm Convergence")
        ax.grid(True, alpha=0.3)
        ax.legend()
        if logscale:
            ax.set_xscale("log")
            ax.set_yscale("log")

    fig2.savefig(error_path, dpi=300, bbox_inches="tight", pad_inches=0.1, facecolor="white")
    if show_plots:
        plt.show()
    plt.close(fig2)

    print(f"Saved: {moment_path}")
    print(f"Saved: {error_path}")

    return {
        "convergence_orders": convergence_orders,
        "figures_saved": [str(moment_path), str(error_path)],
        "benchmark_type": bench_type,
        "ref_grid": {"nx": ref_nx, "nv": ref_nv},
        "out_dir": str(out_dir),
    }
