# kineticEQ/analysis/BGK1D/plotting/plot_moment_cnn_test.py
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import matplotlib.pyplot as plt


# ------------------------ data model ------------------------
@dataclass
class CaseRecord:
    path: Path
    dt: float | None
    picard_tol: float
    tau_tilde: float | None
    n_steps: int
    gpu_name: str | None

    it_base: np.ndarray  # (n_steps,)
    it_warm: np.ndarray  # (n_steps,)
    t_base: np.ndarray   # (n_steps,)
    t_warm: np.ndarray   # (n_steps,)

    n_base: np.ndarray   # (nx,)
    u_base: np.ndarray   # (nx,)
    T_base: np.ndarray   # (nx,)
    n_warm: np.ndarray   # (nx,)
    u_warm: np.ndarray   # (nx,)
    T_warm: np.ndarray   # (nx,)

    speedup_iter_sum: float
    linf_n: float
    linf_u: float
    linf_T: float

    mean_step_time_base: float
    mean_step_time_warm: float

# ------------------------ color (colorblind-safe) ------------------------
# Okabe–Ito palette (colorblind-safe), in hex.
_OKABE_ITO = [
    "#0072B2",  # blue
    "#D55E00",  # vermillion
    "#009E73",  # bluish green
    "#CC79A7",  # reddish purple
    "#E69F00",  # orange
    "#56B4E9",  # sky blue
    "#F0E442",  # yellow (use carefully on white; still OK)
    "#000000",  # black
]

_BASE_LS = "--"   # dashed
_WARM_LS = "-"    # solid


def _tol_label(tol: float) -> str:
    if tol > 0:
        return f"{tol:.0e}"
    return str(tol)


def _build_tol_color_map(tols_sorted: list[float]) -> dict[float, str]:
    """
    Deterministic mapping: tol -> color.
    If more tolerances than palette length, colors repeat (still deterministic).
    """
    cmap: dict[float, str] = {}
    for i, t in enumerate(tols_sorted):
        cmap[t] = _OKABE_ITO[i % len(_OKABE_ITO)]
    return cmap


# ------------------------ helpers ------------------------
def _as_float(x: Any, default: float | None = None) -> float | None:
    try:
        if x is None:
            return default
        return float(x)
    except Exception:
        return default


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def _get_case_root(d: dict, case_index: int = 0) -> tuple[dict, dict, dict]:
    meta = d.get("meta", {}) if isinstance(d.get("meta", {}), dict) else {}
    cases = d.get("cases", [])
    if not isinstance(cases, list) or not cases:
        raise ValueError("JSON has no 'cases' list or it is empty.")
    if not (0 <= case_index < len(cases)):
        raise ValueError(f"case_index={case_index} out of range: cases size={len(cases)}")

    case = cases[case_index]
    if not isinstance(case, dict):
        raise ValueError("cases[case_index] is not a dict.")

    detail = case.get("detail", None)
    if not isinstance(detail, dict):
        raise ValueError("cases[case_index].detail is missing or not a dict.")

    return meta, case, detail


def _extract_record(path: Path, case_index: int = 0) -> CaseRecord:
    d = _load_json(path)
    meta, case, detail = _get_case_root(d, case_index=case_index)

    picard_tol = meta.get("picard_tol", None)
    if picard_tol is None:
        raise ValueError(f"{path}: meta.picard_tol is missing.")
    picard_tol = float(picard_tol)

    gpu_name = meta.get("gpu_name", None)
    if not isinstance(gpu_name, str):
        gpu_name = None

    dt = _as_float(meta.get("dt", None), default=None)
    tau_tilde = _as_float(case.get("tau_tilde", None), default=None)

    timing = detail.get("timing", {})
    if not isinstance(timing, dict):
        raise ValueError(f"{path}: detail.timing is missing or not a dict.")
    t_base = np.asarray(timing.get("step_walltime_base_sec", []), dtype=np.float64)
    t_warm = np.asarray(timing.get("step_walltime_warm_sec", []), dtype=np.float64)
    if t_base.size == 0 or t_warm.size == 0:
        raise ValueError(f"{path}: timing arrays missing/empty (step_walltime_*).")

    pic = detail.get("picard", {})
    if not isinstance(pic, dict):
        raise ValueError(f"{path}: detail.picard is missing or not a dict.")
    it_base = np.asarray(pic.get("picard_iter_hist_base", []), dtype=np.int64)
    it_warm = np.asarray(pic.get("picard_iter_hist_warm", []), dtype=np.int64)
    if it_base.size == 0 or it_warm.size == 0:
        raise ValueError(f"{path}: picard iteration hist arrays missing/empty.")

    n_steps = int(min(t_base.size, t_warm.size, it_base.size, it_warm.size))
    t_base = t_base[:n_steps]
    t_warm = t_warm[:n_steps]
    it_base = it_base[:n_steps]
    it_warm = it_warm[:n_steps]

    # average walltime per step
    mean_step_time_base = float(np.sum(t_base) / max(n_steps, 1))
    mean_step_time_warm = float(np.sum(t_warm) / max(n_steps, 1))

    sum_base = int(pic.get("picard_iter_sum_base", int(np.sum(it_base[it_base > 0]))))
    sum_warm = int(pic.get("picard_iter_sum_warm", int(np.sum(it_warm[it_warm > 0]))))
    speedup = float(sum_base) / float(max(sum_warm, 1))

    fm_b = detail.get("final_moments_base", {})
    fm_w = detail.get("final_moments_warm", {})
    if not isinstance(fm_b, dict) or not isinstance(fm_w, dict):
        raise ValueError(f"{path}: final_moments_base/warm missing or not dict.")

    n_base = np.asarray(fm_b.get("n", []), dtype=np.float64)
    u_base = np.asarray(fm_b.get("u", []), dtype=np.float64)
    T_base = np.asarray(fm_b.get("T", []), dtype=np.float64)
    n_warm = np.asarray(fm_w.get("n", []), dtype=np.float64)
    u_warm = np.asarray(fm_w.get("u", []), dtype=np.float64)
    T_warm = np.asarray(fm_w.get("T", []), dtype=np.float64)
    if n_base.size == 0 or n_warm.size == 0:
        raise ValueError(f"{path}: final moments arrays are missing/empty.")

    nx = int(min(n_base.size, u_base.size, T_base.size, n_warm.size, u_warm.size, T_warm.size))
    n_base, u_base, T_base = n_base[:nx], u_base[:nx], T_base[:nx]
    n_warm, u_warm, T_warm = n_warm[:nx], u_warm[:nx], T_warm[:nx]

    linf_n = float(np.max(np.abs(n_warm - n_base)))
    linf_u = float(np.max(np.abs(u_warm - u_base)))
    linf_T = float(np.max(np.abs(T_warm - T_base)))

    return CaseRecord(
        path=path,
        dt=dt,
        picard_tol=picard_tol,
        tau_tilde=tau_tilde,
        n_steps=n_steps,
        gpu_name=gpu_name,
        it_base=it_base,
        it_warm=it_warm,
        t_base=t_base,
        t_warm=t_warm,
        n_base=n_base,
        u_base=u_base,
        T_base=T_base,
        n_warm=n_warm,
        u_warm=u_warm,
        T_warm=T_warm,
        speedup_iter_sum=speedup,
        linf_n=linf_n,
        linf_u=linf_u,
        linf_T=linf_T,
        mean_step_time_base=mean_step_time_base,
        mean_step_time_warm=mean_step_time_warm,
    )



def _expand_inputs(inputs: Iterable[str]) -> list[Path]:
    out: list[Path] = []
    for s in inputs:
        p = Path(s)
        if any(ch in s for ch in ["*", "?", "["]):
            out.extend(sorted(Path().glob(s)))
        else:
            out.append(p)

    out2: list[Path] = []
    for p in out:
        if p.is_dir():
            out2.extend(sorted(p.glob("*.json")))
        else:
            out2.append(p)

    seen = set()
    uniq: list[Path] = []
    for p in out2:
        rp = p.resolve()
        if rp not in seen:
            seen.add(rp)
            uniq.append(rp)
    return uniq


def _ensure_outdir(outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)


# ------------------------ public API ------------------------
def plot_moment_cnn_test(
    json_files: list[str] | tuple[str, ...],
    *,
    save: bool = False,
    show: bool = True,
    format: str = "png",
    outdir: str = ".",
    case_index: int = 0,
    plot_1_figsize: tuple[float, float] = (12, 6),
    plot_2_figsize: tuple[float, float] = (12, 6),
    plot_3_figsize: tuple[float, float] = (12, 6),
    plot_4_figsize: tuple[float, float] = (12, 6),
    mode: str = "B",  # "B" (base+warm), reserved for future: "warm_only"
    walltime_skip_first: int = 0,
    fontsize: float | None = None,
    plot_1_fontsize: float | None = None,
    plot_2_fontsize: float | None = None,
    plot_3_fontsize: float | None = None,
    plot_4_fontsize: float | None = None,
    plot_1_filename: str = "plot_1_picard_iters",
    plot_2_filename: str = "plot_2_walltime",
    plot_3_filename: str = "plot_3_final_moments",
    plot_4_filename: str = "plot_4_speedup_linf",
) -> dict:
    """
    Plots (English labels/titles/legend):
      1) Step vs Picard iterations: base(dashed) and warm(solid), same color per tol
      2) Step vs walltime per step: base(dashed) and warm(solid), same color per tol
      3) Final moments: base & warm; and Δ (warm - base), same color per tol
      4) Picard tolerance (log-x) vs:
           - left y: iteration reduction (Base/Warm) [1 line]
           - right y: final L∞ difference for n, u, T [3 lines]

    mode:
      - "B": base+warm (current)
      - "warm_only": reserved for later (not implemented yet)
    """
    if mode not in ("B", "warm_only"):
        raise ValueError("mode must be 'B' or 'warm_only'")

    fmt = str(format).lower().strip()
    if fmt not in ("png", "pdf"):
        raise ValueError("format must be 'png' or 'pdf'")

    outdir_p = Path(outdir)
    if save:
        _ensure_outdir(outdir_p)

    paths = _expand_inputs(json_files)
    if not paths:
        raise ValueError("No input JSON files found.")

    records = [_extract_record(p, case_index=int(case_index)) for p in paths]

    # deterministic order by picard_tol
    records = sorted(records, key=lambda r: float(r.picard_tol))

    # tol -> color mapping (consistent across all plots; colorblind-safe)
    unique_tols = sorted({float(r.picard_tol) for r in records})
    tol2c = _build_tol_color_map(unique_tols)

    # build common title suffix from dt / tau_tilde
    _dts = sorted({r.dt for r in records if r.dt is not None})
    _taus = sorted({r.tau_tilde for r in records if r.tau_tilde is not None})
    _title_parts: list[str] = []
    if len(_dts) == 1:
        _title_parts.append(f"dt={_dts[0]:.1e}")
    elif _dts:
        _title_parts.append(f"dt=[{', '.join(f'{v:.1e}' for v in _dts)}]")
    if len(_taus) == 1:
        _title_parts.append(f"tau={_taus[0]:.1e}")
    elif _taus:
        _title_parts.append(f"tau=[{', '.join(f'{v:.1e}' for v in _taus)}]")
    _title_suffix = f"  ({', '.join(_title_parts)})" if _title_parts else ""

    # ---------- Plot 1 ----------
    fig1, ax1 = plt.subplots(figsize=plot_1_figsize)
    for rec in sorted(records, key=lambda r: r.picard_tol):
        x = np.arange(rec.n_steps, dtype=np.int64)
        c = tol2c[float(rec.picard_tol)]
        tol_s = _tol_label(rec.picard_tol)
        if mode == "B":
            ax1.plot(x, rec.it_base, linestyle=_BASE_LS, color=c, label=f"tol={tol_s} base")
        ax1.plot(x, rec.it_warm, linestyle=_WARM_LS, color=c, label=f"tol={tol_s} warm")
    _fs1 = plot_1_fontsize or fontsize
    ax1.set_title(f"Picard Iterations per Time Step{_title_suffix}", fontsize=_fs1)
    ax1.set_xlabel("Step", fontsize=_fs1)
    ax1.set_ylabel("Picard Iterations", fontsize=_fs1)
    ax1.legend(fontsize=_fs1)
    ax1.tick_params(labelsize=_fs1)

    # ---------- Plot 2 (walltime per step) ----------
    # GPU name in title (if mixed -> "mixed")
    gpu_names = sorted({r.gpu_name for r in records if r.gpu_name})
    gpu_title = gpu_names[0] if len(gpu_names) == 1 else ("mixed" if len(gpu_names) > 1 else "unknown")

    fig2, ax2 = plt.subplots(figsize=plot_2_figsize)

    _ws = max(int(walltime_skip_first), 0)
    for rec in records:
        steps = np.arange(rec.n_steps, dtype=np.int64)[_ws:]
        c = tol2c[float(rec.picard_tol)]
        tol_s = _tol_label(rec.picard_tol)
        ax2.plot(
            steps, rec.t_base[_ws:],
            linestyle=_BASE_LS, color=c, linewidth=1.5,
            label=f"tol={tol_s} base"
        )
        ax2.plot(
            steps, rec.t_warm[_ws:],
            linestyle=_WARM_LS, color=c, linewidth=1.5,
            label=f"tol={tol_s} warm"
        )

    _fs2 = plot_2_fontsize or fontsize
    ax2.set_title(f"Walltime per Step (GPU: {gpu_title}){_title_suffix}", fontsize=_fs2)
    ax2.set_xlabel("Step", fontsize=_fs2)
    ax2.set_ylabel("Walltime [s]", fontsize=_fs2)
    ax2.tick_params(labelsize=_fs2)
    ax2.grid(True, which="both", alpha=0.3)

    # dedup legend
    handles, labels = ax2.get_legend_handles_labels()
    uniq = {}
    for h, l in zip(handles, labels):
        uniq[l] = h
    ax2.legend(list(uniq.values()), list(uniq.keys()), loc="best", fontsize=_fs2)
    ax2.grid(False)

    # ---------- Plot 3 (final moments + deltas) ----------
    fig3, axes = plt.subplots(2, 3, figsize=plot_3_figsize, squeeze=False)
    moments = ("n", "u", "T")

    for j, m in enumerate(moments):
        ax = axes[0][j]
        for rec in sorted(records, key=lambda r: r.picard_tol):
            x = np.arange(rec.n_base.size, dtype=np.int64)
            c = tol2c[float(rec.picard_tol)]
            tol_s = _tol_label(rec.picard_tol)

            if m == "n":
                if mode == "B":
                    ax.plot(x, rec.n_base, linestyle=_BASE_LS, color=c, label=f"tol={tol_s} base")
                ax.plot(x, rec.n_warm, linestyle=_WARM_LS, color=c, label=f"tol={tol_s} warm")
                ax.set_ylabel("n")
            elif m == "u":
                if mode == "B":
                    ax.plot(x, rec.u_base, linestyle=_BASE_LS, color=c, label=f"tol={tol_s} base")
                ax.plot(x, rec.u_warm, linestyle=_WARM_LS, color=c, label=f"tol={tol_s} warm")
                ax.set_ylabel("u")
            else:
                if mode == "B":
                    ax.plot(x, rec.T_base, linestyle=_BASE_LS, color=c, label=f"tol={tol_s} base")
                ax.plot(x, rec.T_warm, linestyle=_WARM_LS, color=c, label=f"tol={tol_s} warm")
                ax.set_ylabel("T")

    _fs3 = plot_3_fontsize or fontsize
    for j, m in enumerate(moments):
        ax = axes[0][j]
        ax.set_title(f"Final Moment {m}(x)", fontsize=_fs3)
        ax.set_xlabel("Cell Index", fontsize=_fs3)
        ax.legend(fontsize=_fs3)
        ax.tick_params(labelsize=_fs3)

    for j, m in enumerate(moments):
        ax = axes[1][j]
        for rec in sorted(records, key=lambda r: r.picard_tol):
            x = np.arange(rec.n_base.size, dtype=np.int64)
            c = tol2c[float(rec.picard_tol)]
            tol_s = _tol_label(rec.picard_tol)

            if m == "n":
                d = rec.n_warm - rec.n_base
                ax.plot(x, d, linestyle=_WARM_LS, color=c, label=f"tol={tol_s} Δ")
                ax.set_ylabel("Δn")
            elif m == "u":
                d = rec.u_warm - rec.u_base
                ax.plot(x, d, linestyle=_WARM_LS, color=c, label=f"tol={tol_s} Δ")
                ax.set_ylabel("Δu")
            else:
                d = rec.T_warm - rec.T_base
                ax.plot(x, d, linestyle=_WARM_LS, color=c, label=f"tol={tol_s} Δ")
                ax.set_ylabel("ΔT")

        ax.set_title(f"Final Difference Δ{m}(x) = warm - base", fontsize=_fs3)
        ax.set_xlabel("Cell Index", fontsize=_fs3)
        ax.legend(fontsize=_fs3)
        ax.tick_params(labelsize=_fs3)

    fig3.suptitle(f"Final Moments and Differences{_title_suffix}", y=1.02, fontsize=_fs3)

    # ---------- Plot 4 (two panels: speedup & mean step time; both with L∞) ----------
    recs = sorted(records, key=lambda r: r.picard_tol)
    tols = np.asarray([r.picard_tol for r in recs], dtype=np.float64)

    speed = np.asarray([r.speedup_iter_sum for r in recs], dtype=np.float64)
    linf_n = np.asarray([r.linf_n for r in recs], dtype=np.float64)
    linf_u = np.asarray([r.linf_u for r in recs], dtype=np.float64)
    linf_T = np.asarray([r.linf_T for r in recs], dtype=np.float64)

    mean_t_base = np.asarray([r.mean_step_time_base for r in recs], dtype=np.float64)
    mean_t_warm = np.asarray([r.mean_step_time_warm for r in recs], dtype=np.float64)

    # GPU name in title (if mixed -> "mixed")
    gpu_names = sorted({r.gpu_name for r in recs if r.gpu_name})
    gpu_title = gpu_names[0] if len(gpu_names) == 1 else ("mixed" if len(gpu_names) > 1 else "unknown")

    fig4, (ax4a, ax4b) = plt.subplots(
        nrows=1, ncols=2, figsize=plot_4_figsize, constrained_layout=True
    )

    # ---- left panel: speedup + L∞ ----
    ax4a_r = ax4a.twinx()
    ax4a.plot(
        tols, speed, marker="o", color="#000000",
        label="Iteration Reduction (Base / Warm)"
    )
    ax4a.set_xscale("log")
    _fs4 = plot_4_fontsize or fontsize
    ax4a.set_xlabel("Picard Tolerance", fontsize=_fs4)
    ax4a.set_ylabel("Iteration Reduction (sum iters base / warm)", fontsize=_fs4)
    ax4a.set_title(f"Iteration Reduction vs Picard Tol (GPU: {gpu_title}){_title_suffix}", fontsize=_fs4)
    ax4a.tick_params(labelsize=_fs4)
    ax4a.grid(True, which="both", alpha=0.3)

    ax4a_r.plot(tols, linf_n, marker="o", color=_OKABE_ITO[0], label="Final Difference L∞ (n)")
    ax4a_r.plot(tols, linf_u, marker="o", color=_OKABE_ITO[1], label="Final Difference L∞ (u)")
    ax4a_r.plot(tols, linf_T, marker="o", color=_OKABE_ITO[2], label="Final Difference L∞ (T)")
    ax4a_r.set_ylabel("Final Difference (L∞)", fontsize=_fs4)
    ax4a_r.tick_params(labelsize=_fs4)

    h1, l1 = ax4a.get_legend_handles_labels()
    h2, l2 = ax4a_r.get_legend_handles_labels()
    ax4a.legend(h1 + h2, l1 + l2, loc="best", fontsize=_fs4)

    # ---- right panel: mean step time (base/warm) + L∞ ----
    ax4b_r = ax4b.twinx()
    ax4b.plot(
        tols, mean_t_base, marker="o", linestyle=_BASE_LS, color="#000000",
        label="Mean Step Time (Base) [s]"
    )
    ax4b.plot(
        tols, mean_t_warm, marker="s", linestyle=_WARM_LS, color="#000000",
        label="Mean Step Time (Warm) [s]"
    )
    ax4b.set_xscale("log")
    ax4b.set_xlabel("Picard Tolerance", fontsize=_fs4)
    ax4b.set_ylabel("Mean Step Time [s]", fontsize=_fs4)
    ax4b.set_title(f"Mean Step Time vs Picard Tol (GPU: {gpu_title}){_title_suffix}", fontsize=_fs4)
    ax4b.tick_params(labelsize=_fs4)
    ax4b.grid(True, which="both", alpha=0.3)

    ax4b_r.plot(tols, linf_n, marker="o", color=_OKABE_ITO[0], label="Final Difference L∞ (n)")
    ax4b_r.plot(tols, linf_u, marker="o", color=_OKABE_ITO[1], label="Final Difference L∞ (u)")
    ax4b_r.plot(tols, linf_T, marker="o", color=_OKABE_ITO[2], label="Final Difference L∞ (T)")
    ax4b_r.set_ylabel("Final Difference (L∞)", fontsize=_fs4)
    ax4b_r.tick_params(labelsize=_fs4)

    h1, l1 = ax4b.get_legend_handles_labels()
    h2, l2 = ax4b_r.get_legend_handles_labels()
    ax4b.legend(h1 + h2, l1 + l2, loc="best", fontsize=_fs4)

    figures = {
        plot_1_filename: fig1,
        plot_2_filename: fig2,
        plot_3_filename: fig3,
        plot_4_filename: fig4,
    }

    saved: dict[str, str] = {}
    if save:
        for k, fig in figures.items():
            p = outdir_p / f"{k}.{fmt}"
            if not fig.get_constrained_layout():
                fig.tight_layout()
            fig.savefig(p, dpi=150)
            saved[k] = str(p)

    if show:
        plt.show()
    else:
        for fig in figures.values():
            plt.close(fig)

    return {
        "records": records,
        "figures": figures,
        "saved": saved,
        "tol_color_map": tol2c,
        "line_styles": {"base": _BASE_LS, "warm": _WARM_LS},
    }


# ------------------------ CLI (optional) ------------------------
def _parse_args_cli() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--inputs", type=str, nargs="+", required=True)
    p.add_argument("--outdir", type=str, default=".")
    p.add_argument("--format", type=str, default="png", choices=["png", "pdf"])
    p.add_argument("--save", action="store_true")
    p.add_argument("--show", action="store_true")
    p.add_argument("--case_index", type=int, default=0)
    p.add_argument("--mode", type=str, default="B", choices=["B", "warm_only"])

    p.add_argument("--plot_1_figsize", type=float, nargs=2, default=(12, 6))
    p.add_argument("--plot_2_figsize", type=float, nargs=2, default=(12, 6))
    p.add_argument("--plot_3_figsize", type=float, nargs=2, default=(12, 6))
    p.add_argument("--plot_4_figsize", type=float, nargs=2, default=(12, 6))

    p.add_argument("--plot_1_filename", type=str, default="plot_1_picard_iters")
    p.add_argument("--plot_2_filename", type=str, default="plot_2_walltime")
    p.add_argument("--plot_3_filename", type=str, default="plot_3_final_moments")
    p.add_argument("--plot_4_filename", type=str, default="plot_4_speedup_linf")
    return p.parse_args()


def main() -> None:
    args = _parse_args_cli()
    plot_moment_cnn_test(
        list(args.inputs),
        save=bool(args.save),
        show=bool(args.show),
        format=str(args.format),
        outdir=str(args.outdir),
        case_index=int(args.case_index),
        plot_1_figsize=(float(args.plot_1_figsize[0]), float(args.plot_1_figsize[1])),
        plot_2_figsize=(float(args.plot_2_figsize[0]), float(args.plot_2_figsize[1])),
        plot_3_figsize=(float(args.plot_3_figsize[0]), float(args.plot_3_figsize[1])),
        plot_4_figsize=(float(args.plot_4_figsize[0]), float(args.plot_4_figsize[1])),
        mode=str(args.mode),
        plot_1_filename=str(args.plot_1_filename),
        plot_2_filename=str(args.plot_2_filename),
        plot_3_filename=str(args.plot_3_filename),
        plot_4_filename=str(args.plot_4_filename),
    )


if __name__ == "__main__":
    main()
