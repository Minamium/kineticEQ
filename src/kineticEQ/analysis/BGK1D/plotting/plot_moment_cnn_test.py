# kineticEQ/analysis/BGK1D/plotting/plot_moment_cnn_test.py
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


# ------------------------ data model ------------------------
@dataclass
class CaseRecord:
    path: Path
    dt: float | None
    picard_tol: float
    tau_tilde: float | None
    n_steps: int
    gpu_name: str | None

    it_base: np.ndarray
    it_warm: np.ndarray
    t_base: np.ndarray
    t_warm: np.ndarray

    n_base: np.ndarray
    u_base: np.ndarray
    T_base: np.ndarray
    n_warm: np.ndarray
    u_warm: np.ndarray
    T_warm: np.ndarray

    speedup_iter_sum: float
    linf_n: float
    linf_u: float
    linf_T: float

    mean_step_time_base: float
    mean_step_time_warm: float
    mean_infer_time_est: float


# ------------------------ palette / styles ------------------------
_OKABE_ITO = [
    "#0072B2", "#D55E00", "#009E73", "#CC79A7",
    "#E69F00", "#56B4E9", "#F0E442", "#000000",
]
_BASE_LS = "--"
_WARM_LS = "-"

# marker cycle (monochrome: tol-distinction by marker)
_MARKERS = ["o", "s", "^", "v", "D", "P", "X", "*", "+", "x", "1", "2", "3", "4"]


# ------------------------ small utilities ------------------------
def _tol_label(tol: float) -> str:
    return f"{tol:.0e}" if tol > 0 else str(tol)


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

    mean_step_time_base = float(np.sum(t_base) / max(n_steps, 1))
    mean_step_time_warm = float(np.sum(t_warm) / max(n_steps, 1))

    t_inf_est = np.asarray(timing.get("infer_time_est_step_sec", []), dtype=np.float64)
    if t_inf_est.size > 0:
        valid = t_inf_est[~np.isnan(t_inf_est)]
        mean_infer_time_est = float(np.mean(valid)) if valid.size > 0 else float("nan")
    else:
        mean_infer_time_est = float("nan")

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
        mean_infer_time_est=mean_infer_time_est,
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


def _font_sizes(fs: float | None) -> tuple[float | None, float | None, float | None, float | None]:
    if fs is None:
        return None, None, None, None
    return fs, fs - 1, fs - 2, fs - 2


def _legend_ncol_auto(n_items: int, max_cols: int = 6) -> int:
    if n_items <= 0:
        return 1
    return max(1, min(max_cols, n_items))


def _apply_grid(ax: plt.Axes, grid_mode: str, alpha: float = 0.3) -> None:
    if grid_mode == "off":
        ax.grid(False)
        return
    if grid_mode == "both":
        ax.grid(True, which="both", alpha=alpha)
        return
    ax.grid(True, which="major", alpha=alpha)
    ax.grid(False, which="minor")


def _lab(key: str, default: str, legend_labels: dict[str, str] | None) -> str:
    if legend_labels is None:
        return default
    v = legend_labels.get(key, None)
    return default if v is None else str(v)


def _dedup_legend(handles: list[Any], labels: list[str]) -> tuple[list[Any], list[str]]:
    uniq: dict[str, Any] = {}
    for h, lab in zip(handles, labels):
        uniq[lab] = h
    return list(uniq.values()), list(uniq.keys())


def _legend_ncol_pick(global_ncol: int | None, per_plot_ncol: int | None) -> int | None:
    return per_plot_ncol if per_plot_ncol is not None else global_ncol


def _build_tol_color_map(tols_sorted: list[float]) -> dict[float, str]:
    cmap: dict[float, str] = {}
    for i, t in enumerate(tols_sorted):
        cmap[t] = _OKABE_ITO[i % len(_OKABE_ITO)]
    return cmap


def _build_tol_marker_map(tols_sorted: list[float]) -> dict[float, str]:
    mmap: dict[float, str] = {}
    for i, t in enumerate(tols_sorted):
        mmap[t] = _MARKERS[i % len(_MARKERS)]
    return mmap


def _split_legend_handles(
    *,
    mode: str,
    tols_sorted: list[float],
    tol2c: dict[float, str],
    tol2m: dict[float, str] | None,
    monochrome: bool,
    marker_size: float = 7.0,
) -> list[Line2D]:
    handles: list[Line2D] = []
    handles.append(Line2D([0], [0], ls=_WARM_LS, color="gray", lw=1.5, label="Warm"))
    if mode == "B":
        handles.append(Line2D([0], [0], ls=_BASE_LS, color="gray", lw=1.5, label="Base"))

    for t in tols_sorted:
        if monochrome:
            mk = tol2m[t] if tol2m is not None else "o"
            handles.append(
                Line2D(
                    [0], [0], ls="-", color="black", lw=1.5, marker=mk, ms=float(marker_size),
                    label=f"tol={_tol_label(t)}",
                )
            )
        else:
            handles.append(Line2D([0], [0], ls="-", color=tol2c[t], lw=1.5, label=f"tol={_tol_label(t)}"))
    return handles


def _legend_place_below(
    ax: plt.Axes,
    handles: list[Any],
    labels: list[str],
    ncol: int,
    fontsize: float | None,
    y: float = -0.18,
) -> None:
    ax.legend(handles, labels, bbox_to_anchor=(0.5, y), loc="upper center",
              ncol=ncol, fontsize=fontsize, frameon=True, columnspacing=1.0)


def _legend_place_right(
    ax: plt.Axes,
    handles: list[Any],
    labels: list[str],
    fontsize: float | None,
    x: float = 1.02,
) -> None:
    ax.legend(handles, labels, fontsize=fontsize, bbox_to_anchor=(x, 1.0), loc="upper left")


def _legend_place_in(
    ax: plt.Axes,
    handles: list[Any],
    labels: list[str],
    fontsize: float | None,
    loc: str = "best",
) -> None:
    ax.legend(handles, labels, fontsize=fontsize, loc=loc, frameon=True)


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
    plot_5_figsize: tuple[float, float] = (12, 6),
    mode: str = "B",
    walltime_skip_first: int = 0,
    ref_strictest: bool = False,
    linf_log_scale: bool = False,
    legend_position: str = "below_split",  # + "in"
    layout: str = "default",
    linf_mode: str = "separate",
    show_infer_time: bool = True,  # (global default, kept)
    fontsize: float | None = None,
    plot_1_fontsize: float | None = None,
    plot_2_fontsize: float | None = None,
    plot_3_fontsize: float | None = None,
    plot_4_fontsize: float | None = None,
    plot_5_fontsize: float | None = None,
    plot_1_title: str | None = None,
    plot_2_title: str | None = None,
    plot_3_title: str | None = None,
    plot_4_title: str | None = None,
    plot_5_title: str | None = None,
    plot_1_filename: str = "plot_1_picard_iters",
    plot_2_filename: str = "plot_2_walltime",
    plot_3_filename: str = "plot_3_final_moments",
    plot_4_filename: str = "plot_4_speedup_linf",
    plot_5_filename: str = "plot_5_steptime_linf",
    legend_labels: dict[str, str] | None = None,
    legend_ncol: int | None = None,
    legend_max_cols: int = 6,
    grid_mode: str = "both",
    grid_alpha: float = 0.3,
    save_dpi_default: int = 150,
    save_dpi_paper: int = 300,
    save_bbox_default: str | None = "tight",
    # per-plot legend ncol
    plot_1_legend_ncol: int | None = None,
    plot_2_legend_ncol: int | None = None,
    plot_3_legend_ncol: int | None = None,
    plot_4_legend_ncol: int | None = None,
    plot_5_legend_ncol: int | None = None,
    # per-plot enable
    plot_1_enable: bool = True,
    plot_2_enable: bool = True,
    plot_3_enable: bool = True,
    plot_4_enable: bool = True,
    plot_5_enable: bool = True,
    # plot5 infer toggle override
    plot_5_show_infer_time: bool | None = None,  # None -> use show_infer_time
    # plot3 compact subplot titles
    plot_3_compact_titles: bool = False,
    plot_3_top_titles: tuple[str, str, str] | None = None,     # None -> auto
    plot_3_bottom_titles: tuple[str, str, str] | None = None,  # None -> auto
    # monochrome mode
    monochrome: bool = False,
    mono_markevery_step: int = 10,   # plot1/2
    mono_markevery_x: int = 50,      # plot3
    # per-plot marker size (monochrome marking + legend tol markers)
    marker_size: float = 4.0,
    plot_1_marker_size: float | None = None,
    plot_2_marker_size: float | None = None,
    plot_3_marker_size: float | None = None,
    plot_4_marker_size: float | None = None,
    plot_5_marker_size: float | None = None,
    split_legend_marker_size: float = 7.0,
    # legend "in" location
    legend_in_loc: str = "best",
) -> dict:
    if mode not in ("B", "warm_only"):
        raise ValueError("mode must be 'B' or 'warm_only'")
    if legend_position not in ("right", "below", "below_split", "in"):
        raise ValueError("legend_position must be 'right', 'below', 'below_split', or 'in'")
    if layout not in ("default", "paper"):
        raise ValueError("layout must be 'default' or 'paper'")
    if linf_mode not in ("separate", "max"):
        raise ValueError("linf_mode must be 'separate' or 'max'")
    if grid_mode not in ("both", "major", "off"):
        raise ValueError("grid_mode must be 'both', 'major', or 'off'")

    fmt = str(format).lower().strip()
    if fmt not in ("png", "pdf"):
        raise ValueError("format must be 'png' or 'pdf'")

    outdir_p = Path(outdir)
    if save:
        _ensure_outdir(outdir_p)

    paths = _expand_inputs(json_files)
    if not paths:
        raise ValueError("No input JSON files found.")

    records = sorted([_extract_record(p, case_index=int(case_index)) for p in paths],
                     key=lambda r: float(r.picard_tol))

    ref_rec: CaseRecord | None = records[0] if ref_strictest and records else None
    ref_tol_s = _tol_label(ref_rec.picard_tol) if ref_rec is not None else ""

    unique_tols = sorted({float(r.picard_tol) for r in records})
    tol2c = _build_tol_color_map(unique_tols)
    tol2m = _build_tol_marker_map(unique_tols) if monochrome else None

    # title suffix
    _dts = sorted({r.dt for r in records if r.dt is not None})
    _taus = sorted({r.tau_tilde for r in records if r.tau_tilde is not None})
    parts: list[str] = []
    if len(_dts) == 1:
        parts.append(f"dt={_dts[0]:.1e}")
    elif _dts:
        parts.append(f"dt=[{', '.join(f'{v:.1e}' for v in _dts)}]")
    if len(_taus) == 1:
        parts.append(f"tau={_taus[0]:.1e}")
    elif _taus:
        parts.append(f"tau=[{', '.join(f'{v:.1e}' for v in _taus)}]")
    title_suffix = f"  ({', '.join(parts)})" if parts else ""

    gpu_names = sorted({r.gpu_name for r in records if r.gpu_name})
    gpu_title = gpu_names[0] if len(gpu_names) == 1 else ("mixed" if len(gpu_names) > 1 else "unknown")

    figures: dict[str, plt.Figure] = {}
    saved: dict[str, str] = {}

    def _ms(per: float | None) -> float:
        return float(per) if per is not None else float(marker_size)

    def _col(t: float) -> str:
        return "black" if monochrome else tol2c[t]

    def _mk(t: float) -> str | None:
        return (tol2m[t] if tol2m is not None else None)

    def _me_step(n: int) -> int | None:
        if not monochrome:
            return None
        return max(int(mono_markevery_step), 1) if n > 0 else None

    def _me_x(n: int) -> int | None:
        if not monochrome:
            return None
        return max(int(mono_markevery_x), 1) if n > 0 else None

    def _split_handles() -> list[Line2D]:
        return _split_legend_handles(
            mode=mode,
            tols_sorted=unique_tols,
            tol2c=tol2c,
            tol2m=tol2m,
            monochrome=monochrome,
            marker_size=float(split_legend_marker_size),
        )

    def _legend_apply(
        ax: plt.Axes,
        handles: list[Any],
        labels: list[str],
        fs_lg: float | None,
        ncol: int | None,
        below_y: float = -0.18,
        right_x: float = 1.02,
    ) -> None:
        if legend_position == "in":
            _legend_place_in(ax, handles, labels, fs_lg, loc=str(legend_in_loc))
            return
        if legend_position == "right":
            _legend_place_right(ax, handles, labels, fs_lg, x=right_x)
            return
        n = ncol if ncol is not None else _legend_ncol_auto(len(handles), legend_max_cols)
        _legend_place_below(ax, handles, labels, n, fs_lg, y=below_y)

    # ---------------- Plot 1 ----------------
    if plot_1_enable:
        fig1, ax1 = plt.subplots(figsize=plot_1_figsize)
        for rec in records:
            x = np.arange(rec.n_steps, dtype=np.int64)
            t = float(rec.picard_tol)
            c = _col(t)
            m = _mk(t)
            me = _me_step(rec.n_steps)
            ms = _ms(plot_1_marker_size)
            if mode == "B":
                ax1.plot(
                    x, rec.it_base, linestyle=_BASE_LS, color=c,
                    marker=m, markevery=me, ms=ms if monochrome else None,
                    label=f"tol={_tol_label(t)} base",
                )
            ax1.plot(
                x, rec.it_warm, linestyle=_WARM_LS, color=c,
                marker=m, markevery=me, ms=ms if monochrome else None,
                label=f"tol={_tol_label(t)} warm",
            )

        fs = plot_1_fontsize or fontsize
        fs_t, fs_l, fs_tk, fs_lg = _font_sizes(fs)
        ax1.set_title(plot_1_title or f"Picard Iterations per Time Step{title_suffix}", fontsize=fs_t)
        ax1.set_xlabel("Step", fontsize=fs_l)
        ax1.set_ylabel("Picard Iterations", fontsize=fs_l)
        ax1.tick_params(labelsize=fs_tk)

        ncol = _legend_ncol_pick(legend_ncol, plot_1_legend_ncol)
        if legend_position == "below_split":
            H = _split_handles()
            _legend_apply(ax1, H, [h.get_label() for h in H], fs_lg, ncol)
        else:
            H, L = ax1.get_legend_handles_labels()
            _legend_apply(ax1, list(H), list(L), fs_lg, ncol)

        figures[plot_1_filename] = fig1

    # ---------------- Plot 2 ----------------
    if plot_2_enable:
        fig2, ax2 = plt.subplots(figsize=plot_2_figsize)
        ws = max(int(walltime_skip_first), 0)
        for rec in records:
            steps = np.arange(rec.n_steps, dtype=np.int64)[ws:]
            t = float(rec.picard_tol)
            c = _col(t)
            m = _mk(t)
            me = _me_step(int(steps.size))
            ms = _ms(plot_2_marker_size)
            if mode == "B":
                ax2.plot(
                    steps, rec.t_base[ws:], linestyle=_BASE_LS, color=c, linewidth=1.5,
                    marker=m, markevery=me, ms=ms if monochrome else None,
                    label=f"tol={_tol_label(t)} base",
                )
            ax2.plot(
                steps, rec.t_warm[ws:], linestyle=_WARM_LS, color=c, linewidth=1.5,
                marker=m, markevery=me, ms=ms if monochrome else None,
                label=f"tol={_tol_label(t)} warm",
            )

        fs = plot_2_fontsize or fontsize
        fs_t, fs_l, fs_tk, fs_lg = _font_sizes(fs)
        ax2.set_title(plot_2_title or f"Walltime per Step (GPU: {gpu_title}){title_suffix}", fontsize=fs_t)
        ax2.set_xlabel("Step", fontsize=fs_l)
        ax2.set_ylabel("Walltime [s]", fontsize=fs_l)
        ax2.tick_params(labelsize=fs_tk)
        _apply_grid(ax2, grid_mode=grid_mode, alpha=float(grid_alpha))

        ncol = _legend_ncol_pick(legend_ncol, plot_2_legend_ncol)
        if legend_position == "below_split":
            H = _split_handles()
            _legend_apply(ax2, H, [h.get_label() for h in H], fs_lg, ncol)
        else:
            H0, L0 = ax2.get_legend_handles_labels()
            H, L = _dedup_legend(list(H0), list(L0))
            _legend_apply(ax2, H, L, fs_lg, ncol)

        figures[plot_2_filename] = fig2

    # ---------------- Plot 3 ----------------
    if plot_3_enable:
        fig3, axes = plt.subplots(2, 3, figsize=plot_3_figsize, squeeze=False)

        # ---- FIX (core): make spacing/margins robust for large fonts + bottom legend
        fig3.subplots_adjust(
            left=0.12,
            right=0.98,
            top=0.92,
            bottom=0.28,   # room for bottom legend
            wspace=0.45,
            hspace=0.55,
        )

        moments = ("n", "u", "T")

        fs = plot_3_fontsize or fontsize
        fs_t, fs_l, fs_tk, fs_lg = _font_sizes(fs)
        ms = _ms(plot_3_marker_size)

        # titles (shortening)
        if plot_3_top_titles is not None:
            top_titles = plot_3_top_titles
        else:
            top_titles = ("n(x)", "u(x)", "T(x)") if plot_3_compact_titles else (
                "Final Moment n(x)", "Final Moment u(x)", "Final Moment T(x)"
            )
        if plot_3_bottom_titles is not None:
            bot_titles = plot_3_bottom_titles
        else:
            if ref_rec is not None:
                bot_titles = ("Δn (ref)", "Δu (ref)", "ΔT (ref)") if plot_3_compact_titles else (
                    f"Δn(x) relative to reference (tol={ref_tol_s})",
                    f"Δu(x) relative to reference (tol={ref_tol_s})",
                    f"ΔT(x) relative to reference (tol={ref_tol_s})",
                )
            else:
                bot_titles = ("Δn", "Δu", "ΔT") if plot_3_compact_titles else (
                    "Final Difference Δn(x) = warm - base",
                    "Final Difference Δu(x) = warm - base",
                    "Final Difference ΔT(x) = warm - base",
                )

        # top row: final moments
        for j, mname in enumerate(moments):
            ax = axes[0][j]
            for rec in records:
                x = np.arange(rec.n_base.size, dtype=np.int64)
                t = float(rec.picard_tol)
                c = _col(t)
                mk = _mk(t)
                me = _me_x(int(x.size))
                base = {"n": rec.n_base, "u": rec.u_base, "T": rec.T_base}[mname]
                warm = {"n": rec.n_warm, "u": rec.u_warm, "T": rec.T_warm}[mname]
                if mode == "B":
                    ax.plot(
                        x, base, linestyle=_BASE_LS, color=c,
                        marker=mk, markevery=me, ms=ms if monochrome else None,
                        label=f"tol={_tol_label(t)} base",
                    )
                ax.plot(
                    x, warm, linestyle=_WARM_LS, color=c,
                    marker=mk, markevery=me, ms=ms if monochrome else None,
                    label=f"tol={_tol_label(t)} warm",
                )
            ax.set_title(top_titles[j], fontsize=fs_t)

            # ---- FIX: remove top-row x tick labels to avoid overlap with bottom-row titles
            ax.tick_params(labelbottom=False)

            # ---- FIX: ylabel fontsize + padding
            ax.set_ylabel(mname, fontsize=fs_l, labelpad=8)

            ax.tick_params(labelsize=fs_tk)
            ax.tick_params(axis="y", pad=8)
            ax.tick_params(axis="x", pad=6)

        # bottom row: differences
        for j, mname in enumerate(moments):
            ax = axes[1][j]
            for rec in records:
                t = float(rec.picard_tol)
                c = _col(t)
                mk = _mk(t)
                warm = {"n": rec.n_warm, "u": rec.u_warm, "T": rec.T_warm}[mname]
                base = {"n": rec.n_base, "u": rec.u_base, "T": rec.T_base}[mname]

                if ref_rec is not None:
                    ref_m = {"n": ref_rec.n_base, "u": ref_rec.u_base, "T": ref_rec.T_base}[mname]
                    nx = min(warm.size, ref_m.size)
                    x = np.arange(nx, dtype=np.int64)
                    me = _me_x(int(nx))
                    ax.plot(
                        x, warm[:nx] - ref_m[:nx], linestyle=_WARM_LS, color=c,
                        marker=mk, markevery=me, ms=ms if monochrome else None,
                        label=f"tol={_tol_label(t)} warm-ref",
                    )
                    if mode == "B" and rec is not ref_rec:
                        ax.plot(
                            x, base[:nx] - ref_m[:nx], linestyle=_BASE_LS, color=c,
                            marker=mk, markevery=me, ms=ms if monochrome else None,
                            label=f"tol={_tol_label(t)} base-ref",
                        )
                else:
                    x = np.arange(base.size, dtype=np.int64)
                    me = _me_x(int(x.size))
                    ax.plot(
                        x, warm - base, linestyle=_WARM_LS, color=c,
                        marker=mk, markevery=me, ms=ms if monochrome else None,
                        label=f"tol={_tol_label(t)} Δ",
                    )

            ax.set_title(bot_titles[j], fontsize=fs_t)
            ax.set_xlabel("Cell Index", fontsize=fs_l)

            # ---- FIX: ylabel fontsize + padding
            ax.set_ylabel(f"Δ{mname}", fontsize=fs_l, labelpad=8)

            ax.tick_params(labelsize=fs_tk)
            ax.tick_params(axis="y", pad=8)
            ax.tick_params(axis="x", pad=6)

        # align ylabels for cleaner look
        try:
            fig3.align_ylabels(axes[:, :])
        except Exception:
            pass

        # legend
        ncol = _legend_ncol_pick(legend_ncol, plot_3_legend_ncol)
        if legend_position == "right":
            axes[0][-1].legend(fontsize=fs_lg, bbox_to_anchor=(1.02, 1), loc="upper left")
            axes[1][-1].legend(fontsize=fs_lg, bbox_to_anchor=(1.02, 1), loc="upper left")
        elif legend_position == "in":
            H = _split_handles()
            _legend_place_in(axes[0][-1], H, [h.get_label() for h in H], fs_lg, loc=str(legend_in_loc))
        else:
            # ---- FIX: put legend lower; negative y is OK with bbox_inches="tight"
            H = _split_handles()
            n = ncol if ncol is not None else _legend_ncol_auto(len(H), legend_max_cols)
            fig3.legend(
                handles=H,
                bbox_to_anchor=(0.5, -0.02),  # push down
                loc="lower center",
                ncol=n,
                fontsize=fs_lg,
                frameon=True,
                columnspacing=1.0,
            )

        ref_tag = f" [ref tol={ref_tol_s}]" if ref_rec is not None else ""
        fig3.suptitle(plot_3_title or f"Final Moments and Differences{ref_tag}{title_suffix}",
                      y=0.96, fontsize=fs_t)

        figures[plot_3_filename] = fig3

    # ---------------- shared arrays for plot4/5 ----------------
    need_45 = plot_4_enable or plot_5_enable
    if need_45:
        recs = records
        tols = np.asarray([r.picard_tol for r in recs], dtype=np.float64)
        speed = np.asarray([r.speedup_iter_sum for r in recs], dtype=np.float64)

        if ref_rec is not None:
            def _linf_ref(attr: str, ref_attr: str) -> np.ndarray:
                ref_a = getattr(ref_rec, ref_attr)
                vals = []
                for r in recs:
                    a = getattr(r, attr)
                    nx = min(a.size, ref_a.size)
                    vals.append(float(np.max(np.abs(a[:nx] - ref_a[:nx]))))
                return np.asarray(vals, dtype=np.float64)

            linf_n = _linf_ref("n_warm", "n_base")
            linf_u = _linf_ref("u_warm", "u_base")
            linf_T = _linf_ref("T_warm", "T_base")

            ref_mask = np.array([r is not ref_rec for r in recs])
            linf_n_base = _linf_ref("n_base", "n_base")[ref_mask]
            linf_u_base = _linf_ref("u_base", "u_base")[ref_mask]
            linf_T_base = _linf_ref("T_base", "T_base")[ref_mask]
            tols_base = tols[ref_mask]
        else:
            linf_n = np.asarray([r.linf_n for r in recs], dtype=np.float64)
            linf_u = np.asarray([r.linf_u for r in recs], dtype=np.float64)
            linf_T = np.asarray([r.linf_T for r in recs], dtype=np.float64)

        mean_t_base = np.asarray([r.mean_step_time_base for r in recs], dtype=np.float64)
        mean_t_warm = np.asarray([r.mean_step_time_warm for r in recs], dtype=np.float64)

        ref_title45 = f" [ref tol={ref_tol_s}]" if ref_rec is not None else ""

    # ---------------- Plot 4 ----------------
    if plot_4_enable:
        fs = plot_4_fontsize or fontsize
        fs_t, fs_l, fs_tk, fs_lg = _font_sizes(fs)
        ms = _ms(plot_4_marker_size)

        fig4, ax4 = plt.subplots(figsize=plot_4_figsize)
        ax4_r = ax4.twinx()
        speed_wall = mean_t_base / np.where(mean_t_warm > 0, mean_t_warm, np.nan)

        ax4.plot(tols, speed, marker="o", ms=ms, color="black",
                 label=_lab("iter_reduction", "Iteration Reduction (Base / Warm)", legend_labels))
        ax4.plot(tols, speed_wall, marker="s", ms=ms, linestyle="--", color="#555555",
                 label=_lab("wall_accel", "Walltime Acceleration (Base / Warm)", legend_labels))

        ax4.set_xscale("log")
        ax4.set_xlabel("Picard Tolerance", fontsize=fs_l)
        ax4.set_ylabel("Speedup Ratio (Base / Warm)", fontsize=fs_l)
        ax4.set_title(plot_4_title or f"Speedup per Picard Tolerance (GPU: {gpu_title}){ref_title45}{title_suffix}",
                      fontsize=fs_t)
        ax4.tick_params(labelsize=fs_tk)
        _apply_grid(ax4, grid_mode=grid_mode, alpha=float(grid_alpha))

        if linf_mode == "max":
            linf_w_max = np.maximum.reduce([linf_n, linf_u, linf_T])
            if ref_rec is not None:
                linf_b_max = np.maximum.reduce([linf_n_base, linf_u_base, linf_T_base])
                ax4_r.plot(
                    tols, linf_w_max, marker="o", ms=ms,
                    color="black" if monochrome else _OKABE_ITO[0],
                    label=_lab("linf_warm_max_ref", "L∞ max (warm), ref", legend_labels),
                )
                ax4_r.plot(
                    tols_base, linf_b_max, marker="x", ms=ms, linestyle=_BASE_LS,
                    color="black" if monochrome else _OKABE_ITO[0],
                    label=_lab("linf_base_max_ref", "L∞ max (base), ref", legend_labels),
                )
            else:
                ax4_r.plot(
                    tols, linf_w_max, marker="o", ms=ms,
                    color="black" if monochrome else _OKABE_ITO[0],
                    label=_lab("linf_max", "Final Difference L∞ max", legend_labels),
                )
        else:
            if ref_rec is not None:
                ax4_r.plot(tols, linf_n, marker="o", ms=ms, color="black" if monochrome else _OKABE_ITO[0],
                           label=_lab("linf_warm_n_ref", "L∞ warm (n), ref", legend_labels))
                ax4_r.plot(tols, linf_u, marker="o", ms=ms, color="black" if monochrome else _OKABE_ITO[1],
                           label=_lab("linf_warm_u_ref", "L∞ warm (u), ref", legend_labels))
                ax4_r.plot(tols, linf_T, marker="o", ms=ms, color="black" if monochrome else _OKABE_ITO[2],
                           label=_lab("linf_warm_T_ref", "L∞ warm (T), ref", legend_labels))
                ax4_r.plot(tols_base, linf_n_base, marker="x", ms=ms, linestyle=_BASE_LS,
                           color="black" if monochrome else _OKABE_ITO[0],
                           label=_lab("linf_base_n_ref", "L∞ base (n), ref", legend_labels))
                ax4_r.plot(tols_base, linf_u_base, marker="x", ms=ms, linestyle=_BASE_LS,
                           color="black" if monochrome else _OKABE_ITO[1],
                           label=_lab("linf_base_u_ref", "L∞ base (u), ref", legend_labels))
                ax4_r.plot(tols_base, linf_T_base, marker="x", ms=ms, linestyle=_BASE_LS,
                           color="black" if monochrome else _OKABE_ITO[2],
                           label=_lab("linf_base_T_ref", "L∞ base (T), ref", legend_labels))
            else:
                ax4_r.plot(tols, linf_n, marker="o", ms=ms, color="black" if monochrome else _OKABE_ITO[0],
                           label=_lab("linf_n", "Final Difference L∞ (n)", legend_labels))
                ax4_r.plot(tols, linf_u, marker="o", ms=ms, color="black" if monochrome else _OKABE_ITO[1],
                           label=_lab("linf_u", "Final Difference L∞ (u)", legend_labels))
                ax4_r.plot(tols, linf_T, marker="o", ms=ms, color="black" if monochrome else _OKABE_ITO[2],
                           label=_lab("linf_T", "Final Difference L∞ (T)", legend_labels))

        ax4_r.set_ylabel("Final Difference (L∞)", fontsize=fs_l)
        if linf_log_scale:
            ax4_r.set_yscale("log")
        ax4_r.tick_params(labelsize=fs_tk)

        lp = "below" if legend_position == "below_split" else legend_position
        H1, L1 = ax4.get_legend_handles_labels()
        H2, L2 = ax4_r.get_legend_handles_labels()
        H = list(H1) + list(H2)
        L = list(L1) + list(L2)

        ncol = _legend_ncol_pick(legend_ncol, plot_4_legend_ncol)
        if lp == "in":
            _legend_place_in(ax4, H, L, fs_lg, loc=str(legend_in_loc))
        elif lp == "right":
            _legend_place_right(ax4, H, L, fs_lg, x=1.15)
        else:
            n = ncol if ncol is not None else _legend_ncol_auto(len(H), legend_max_cols)
            _legend_place_below(ax4, H, L, n, fs_lg, y=-0.18)

        figures[plot_4_filename] = fig4

    # ---------------- Plot 5 ----------------
    if plot_5_enable:
        fs = plot_5_fontsize or fontsize
        fs_t, fs_l, fs_tk, fs_lg = _font_sizes(fs)
        ms = _ms(plot_5_marker_size)

        fig5, ax5 = plt.subplots(figsize=plot_5_figsize)
        ax5_r = ax5.twinx()
        mean_t_inf = np.asarray([r.mean_infer_time_est for r in recs], dtype=np.float64)

        ax5.plot(tols, mean_t_base, marker="o", ms=ms, linestyle=_BASE_LS, color="black",
                 label=_lab("mean_step_base", "Mean Step Time (Base) [s]", legend_labels))
        ax5.plot(tols, mean_t_warm, marker="s", ms=ms, linestyle=_WARM_LS, color="black",
                 label=_lab("mean_step_warm", "Mean Step Time (Warm) [s]", legend_labels))

        show_infer_5 = show_infer_time if plot_5_show_infer_time is None else bool(plot_5_show_infer_time)
        if show_infer_5:
            ax5.plot(tols, mean_t_inf, marker="D", ms=ms, linestyle=":", color="#555555",
                     label=_lab("infer_time", "Mean Infer Time Est. [s]", legend_labels))

        ax5.set_xscale("log")
        ax5.set_xlabel("Picard Tolerance", fontsize=fs_l)
        ax5.set_ylabel("Mean Step Time [s]", fontsize=fs_l)
        # Walltime axis should be zero-based to avoid misleading comparisons.
        ax5.set_ylim(bottom=0.0)
        ax5.set_title(plot_5_title or f"Mean Step Time per Picard Tolerance (GPU: {gpu_title}){ref_title45}{title_suffix}",
                      fontsize=fs_t)
        ax5.tick_params(labelsize=fs_tk)
        _apply_grid(ax5, grid_mode=grid_mode, alpha=float(grid_alpha))

        if linf_mode == "max":
            linf_w_max5 = np.maximum.reduce([linf_n, linf_u, linf_T])
            if ref_rec is not None:
                linf_b_max5 = np.maximum.reduce([linf_n_base, linf_u_base, linf_T_base])
                ax5_r.plot(tols, linf_w_max5, marker="o", ms=ms,
                           color="black" if monochrome else _OKABE_ITO[0],
                           label=_lab("linf_warm_max_ref", "L∞ max (warm), ref", legend_labels))
                ax5_r.plot(tols_base, linf_b_max5, marker="x", ms=ms, linestyle=_BASE_LS,
                           color="black" if monochrome else _OKABE_ITO[0],
                           label=_lab("linf_base_max_ref", "L∞ max (base), ref", legend_labels))
            else:
                ax5_r.plot(tols, linf_w_max5, marker="o", ms=ms,
                           color="black" if monochrome else _OKABE_ITO[0],
                           label=_lab("linf_max", "Final Difference L∞ max", legend_labels))
        else:
            if ref_rec is not None:
                ax5_r.plot(tols, linf_n, marker="o", ms=ms, color="black" if monochrome else _OKABE_ITO[0],
                           label=_lab("linf_warm_n_ref", "L∞ warm (n), ref", legend_labels))
                ax5_r.plot(tols, linf_u, marker="o", ms=ms, color="black" if monochrome else _OKABE_ITO[1],
                           label=_lab("linf_warm_u_ref", "L∞ warm (u), ref", legend_labels))
                ax5_r.plot(tols, linf_T, marker="o", ms=ms, color="black" if monochrome else _OKABE_ITO[2],
                           label=_lab("linf_warm_T_ref", "L∞ warm (T), ref", legend_labels))
                ax5_r.plot(tols_base, linf_n_base, marker="x", ms=ms, linestyle=_BASE_LS,
                           color="black" if monochrome else _OKABE_ITO[0],
                           label=_lab("linf_base_n_ref", "L∞ base (n), ref", legend_labels))
                ax5_r.plot(tols_base, linf_u_base, marker="x", ms=ms, linestyle=_BASE_LS,
                           color="black" if monochrome else _OKABE_ITO[1],
                           label=_lab("linf_base_u_ref", "L∞ base (u), ref", legend_labels))
                ax5_r.plot(tols_base, linf_T_base, marker="x", ms=ms, linestyle=_BASE_LS,
                           color="black" if monochrome else _OKABE_ITO[2],
                           label=_lab("linf_base_T_ref", "L∞ base (T), ref", legend_labels))
            else:
                ax5_r.plot(tols, linf_n, marker="o", ms=ms, color="black" if monochrome else _OKABE_ITO[0],
                           label=_lab("linf_n", "Final Difference L∞ (n)", legend_labels))
                ax5_r.plot(tols, linf_u, marker="o", ms=ms, color="black" if monochrome else _OKABE_ITO[1],
                           label=_lab("linf_u", "Final Difference L∞ (u)", legend_labels))
                ax5_r.plot(tols, linf_T, marker="o", ms=ms, color="black" if monochrome else _OKABE_ITO[2],
                           label=_lab("linf_T", "Final Difference L∞ (T)", legend_labels))

        ax5_r.set_ylabel("Final Difference (L∞)", fontsize=fs_l)
        if linf_log_scale:
            ax5_r.set_yscale("log")
        ax5_r.tick_params(labelsize=fs_tk)

        lp = "below" if legend_position == "below_split" else legend_position
        H1, L1 = ax5.get_legend_handles_labels()
        H2, L2 = ax5_r.get_legend_handles_labels()
        H = list(H1) + list(H2)
        L = list(L1) + list(L2)

        ncol = _legend_ncol_pick(legend_ncol, plot_5_legend_ncol)
        if lp == "in":
            _legend_place_in(ax5, H, L, fs_lg, loc=str(legend_in_loc))
        elif lp == "right":
            _legend_place_right(ax5, H, L, fs_lg, x=1.15)
        else:
            n = ncol if ncol is not None else _legend_ncol_auto(len(H), legend_max_cols)
            _legend_place_below(ax5, H, L, n, fs_lg, y=-0.18)

        figures[plot_5_filename] = fig5

    # ---------------- layout adjustments (paper) ----------------
    if layout == "paper" and figures:
        for k, fig in figures.items():
            if legend_position == "right":
                fig.subplots_adjust(left=0.14, right=0.80, top=0.88, bottom=0.14)
            elif legend_position == "in":
                fig.subplots_adjust(left=0.14, right=0.90, top=0.88, bottom=0.14)
            else:
                # ---- FIX: do NOT shrink plot3 bottom; keep enough room for legend
                if k == plot_3_filename:
                    fig.subplots_adjust(left=0.12, right=0.98, top=0.90, bottom=0.28)
                else:
                    fig.subplots_adjust(left=0.14, right=0.86, top=0.88, bottom=0.30)

    # ---------------- save ----------------
    if save and figures:
        for k, fig in figures.items():
            p = outdir_p / f"{k}.{fmt}"
            if layout != "paper":
                if not fig.get_constrained_layout():
                    fig.tight_layout()
                if save_bbox_default is None:
                    fig.savefig(p, dpi=int(save_dpi_default))
                else:
                    fig.savefig(p, dpi=int(save_dpi_default), bbox_inches=str(save_bbox_default))
            else:
                # paper: high dpi; bbox is controlled upstream by paper margins + possible tight in default mode
                fig.savefig(p, dpi=int(save_dpi_paper), bbox_inches=str(save_bbox_default) if save_bbox_default else None)
            saved[k] = str(p)

    # ---------------- show/close ----------------
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
    p.add_argument("--ref_strictest", action="store_true")
    p.add_argument("--linf_log_scale", action="store_true")
    p.add_argument("--legend_position", type=str, default="below_split",
                   choices=["right", "below", "below_split", "in"])
    p.add_argument("--legend_in_loc", type=str, default="best")
    p.add_argument("--layout", type=str, default="default", choices=["default", "paper"])
    p.add_argument("--linf_mode", type=str, default="separate", choices=["separate", "max"])
    p.add_argument("--no_infer_time", dest="show_infer_time", action="store_false", default=True)

    p.add_argument("--legend_ncol", type=int, default=None)
    p.add_argument("--legend_max_cols", type=int, default=6)
    p.add_argument("--grid_mode", type=str, default="both", choices=["both", "major", "off"])
    p.add_argument("--grid_alpha", type=float, default=0.3)
    p.add_argument("--save_dpi_default", type=int, default=150)
    p.add_argument("--save_dpi_paper", type=int, default=300)
    p.add_argument("--save_bbox_default", type=str, default="tight")

    p.add_argument("--plot_1_legend_ncol", type=int, default=None)
    p.add_argument("--plot_2_legend_ncol", type=int, default=None)
    p.add_argument("--plot_3_legend_ncol", type=int, default=None)
    p.add_argument("--plot_4_legend_ncol", type=int, default=None)
    p.add_argument("--plot_5_legend_ncol", type=int, default=None)

    p.add_argument("--no_plot_1", dest="plot_1_enable", action="store_false", default=True)
    p.add_argument("--no_plot_2", dest="plot_2_enable", action="store_false", default=True)
    p.add_argument("--no_plot_3", dest="plot_3_enable", action="store_false", default=True)
    p.add_argument("--no_plot_4", dest="plot_4_enable", action="store_false", default=True)
    p.add_argument("--no_plot_5", dest="plot_5_enable", action="store_false", default=True)

    p.add_argument("--plot_5_no_infer", dest="plot_5_show_infer_time", action="store_false", default=None)
    p.add_argument("--plot_3_compact_titles", action="store_true", default=False)

    p.add_argument("--monochrome", action="store_true", default=False)
    p.add_argument("--mono_markevery_step", type=int, default=10)
    p.add_argument("--mono_markevery_x", type=int, default=50)

    # marker sizes
    p.add_argument("--marker_size", type=float, default=4.0)
    p.add_argument("--plot_1_marker_size", type=float, default=None)
    p.add_argument("--plot_2_marker_size", type=float, default=None)
    p.add_argument("--plot_3_marker_size", type=float, default=None)
    p.add_argument("--plot_4_marker_size", type=float, default=None)
    p.add_argument("--plot_5_marker_size", type=float, default=None)
    p.add_argument("--split_legend_marker_size", type=float, default=7.0)

    p.add_argument("--plot_1_figsize", type=float, nargs=2, default=(12, 6))
    p.add_argument("--plot_2_figsize", type=float, nargs=2, default=(12, 6))
    p.add_argument("--plot_3_figsize", type=float, nargs=2, default=(12, 6))
    p.add_argument("--plot_4_figsize", type=float, nargs=2, default=(12, 6))
    p.add_argument("--plot_5_figsize", type=float, nargs=2, default=(12, 6))

    p.add_argument("--plot_1_filename", type=str, default="plot_1_picard_iters")
    p.add_argument("--plot_2_filename", type=str, default="plot_2_walltime")
    p.add_argument("--plot_3_filename", type=str, default="plot_3_final_moments")
    p.add_argument("--plot_4_filename", type=str, default="plot_4_speedup_linf")
    p.add_argument("--plot_5_filename", type=str, default="plot_5_steptime_linf")

    p.add_argument("--plot_1_title", type=str, default=None)
    p.add_argument("--plot_2_title", type=str, default=None)
    p.add_argument("--plot_3_title", type=str, default=None)
    p.add_argument("--plot_4_title", type=str, default=None)
    p.add_argument("--plot_5_title", type=str, default=None)

    p.add_argument("--fontsize", type=float, default=None)
    p.add_argument("--plot_1_fontsize", type=float, default=None)
    p.add_argument("--plot_2_fontsize", type=float, default=None)
    p.add_argument("--plot_3_fontsize", type=float, default=None)
    p.add_argument("--plot_4_fontsize", type=float, default=None)
    p.add_argument("--plot_5_fontsize", type=float, default=None)

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
        plot_5_figsize=(float(args.plot_5_figsize[0]), float(args.plot_5_figsize[1])),
        mode=str(args.mode),
        ref_strictest=bool(args.ref_strictest),
        linf_log_scale=bool(args.linf_log_scale),
        legend_position=str(args.legend_position),
        legend_in_loc=str(args.legend_in_loc),
        layout=str(args.layout),
        linf_mode=str(args.linf_mode),
        show_infer_time=bool(args.show_infer_time),
        legend_ncol=args.legend_ncol,
        legend_max_cols=int(args.legend_max_cols),
        grid_mode=str(args.grid_mode),
        grid_alpha=float(args.grid_alpha),
        save_dpi_default=int(args.save_dpi_default),
        save_dpi_paper=int(args.save_dpi_paper),
        save_bbox_default=(None if str(args.save_bbox_default).lower() in ("none", "null") else str(args.save_bbox_default)),
        fontsize=args.fontsize,
        plot_1_fontsize=args.plot_1_fontsize,
        plot_2_fontsize=args.plot_2_fontsize,
        plot_3_fontsize=args.plot_3_fontsize,
        plot_4_fontsize=args.plot_4_fontsize,
        plot_5_fontsize=args.plot_5_fontsize,
        plot_1_filename=str(args.plot_1_filename),
        plot_2_filename=str(args.plot_2_filename),
        plot_3_filename=str(args.plot_3_filename),
        plot_4_filename=str(args.plot_4_filename),
        plot_5_filename=str(args.plot_5_filename),
        plot_1_title=args.plot_1_title,
        plot_2_title=args.plot_2_title,
        plot_3_title=args.plot_3_title,
        plot_4_title=args.plot_4_title,
        plot_5_title=args.plot_5_title,
        plot_1_legend_ncol=args.plot_1_legend_ncol,
        plot_2_legend_ncol=args.plot_2_legend_ncol,
        plot_3_legend_ncol=args.plot_3_legend_ncol,
        plot_4_legend_ncol=args.plot_4_legend_ncol,
        plot_5_legend_ncol=args.plot_5_legend_ncol,
        plot_1_enable=bool(args.plot_1_enable),
        plot_2_enable=bool(args.plot_2_enable),
        plot_3_enable=bool(args.plot_3_enable),
        plot_4_enable=bool(args.plot_4_enable),
        plot_5_enable=bool(args.plot_5_enable),
        plot_5_show_infer_time=args.plot_5_show_infer_time,
        plot_3_compact_titles=bool(args.plot_3_compact_titles),
        monochrome=bool(args.monochrome),
        mono_markevery_step=int(args.mono_markevery_step),
        mono_markevery_x=int(args.mono_markevery_x),
        marker_size=float(args.marker_size),
        plot_1_marker_size=args.plot_1_marker_size,
        plot_2_marker_size=args.plot_2_marker_size,
        plot_3_marker_size=args.plot_3_marker_size,
        plot_4_marker_size=args.plot_4_marker_size,
        plot_5_marker_size=args.plot_5_marker_size,
        split_legend_marker_size=float(args.split_legend_marker_size),
    )


if __name__ == "__main__":
    main()