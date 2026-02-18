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
    mean_infer_time_est: float


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


def _build_split_handles(tol2c: dict[float, str], mode: str) -> list[Line2D]:
    """Build split legend: line-style entries + color-per-tol entries."""
    handles: list[Line2D] = []
    handles.append(Line2D([0], [0], ls=_WARM_LS, color="gray", lw=1.5, label="Warm"))
    if mode == "B":
        handles.append(Line2D([0], [0], ls=_BASE_LS, color="gray", lw=1.5, label="Base"))
    for tol in sorted(tol2c):
        handles.append(Line2D([0], [0], ls="-", color=tol2c[tol], lw=1.5,
                              label=f"tol={_tol_label(tol)}"))
    return handles


def _font_sizes(fs: float | None) -> tuple[float | None, float | None, float | None, float | None]:
    """Return (title, label, tick, legend) font sizes from a base size."""
    if fs is None:
        return None, None, None, None
    return fs, fs - 1, fs - 2, fs - 2


def _legend_ncol_auto(n_items: int, max_cols: int = 6) -> int:
    if n_items <= 0:
        return 1
    return max(1, min(max_cols, n_items))


def _apply_grid(ax: plt.Axes, grid_mode: str, alpha: float = 0.3) -> None:
    """
    grid_mode:
      - "off"
      - "major"
      - "both"
    """
    if grid_mode == "off":
        ax.grid(False)
        return
    if grid_mode == "both":
        ax.grid(True, which="both", alpha=alpha)
        return
    ax.grid(True, which="major", alpha=alpha)
    ax.grid(False, which="minor")


def _lab(key: str, default: str, legend_labels: dict[str, str] | None) -> str:
    """
    label resolver (backward compatible):
      - if legend_labels is None -> default
      - else if key in legend_labels -> overridden
      - else -> default
    """
    if legend_labels is None:
        return default
    v = legend_labels.get(key, None)
    return default if v is None else str(v)


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
    legend_position: str = "below_split",
    layout: str = "default",
    linf_mode: str = "separate",
    show_infer_time: bool = True,
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
    # --- NEW: legend label override (backward compatible) ---
    legend_labels: dict[str, str] | None = None,
    # --- NEW: cosmetics controls (backward compatible) ---
    legend_ncol: int | None = None,
    legend_max_cols: int = 6,
    grid_mode: str = "both",   # "both" / "major" / "off"
    grid_alpha: float = 0.3,
    save_dpi_default: int = 150,
    save_dpi_paper: int = 300,
    save_bbox_default: str | None = "tight",
) -> dict:
    if mode not in ("B", "warm_only"):
        raise ValueError("mode must be 'B' or 'warm_only'")

    if legend_position not in ("right", "below", "below_split"):
        raise ValueError("legend_position must be 'right', 'below', or 'below_split'")

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

    records = [_extract_record(p, case_index=int(case_index)) for p in paths]
    records = sorted(records, key=lambda r: float(r.picard_tol))

    ref_rec: CaseRecord | None = records[0] if ref_strictest and records else None
    _ref_tol_s = _tol_label(ref_rec.picard_tol) if ref_rec is not None else ""

    unique_tols = sorted({float(r.picard_tol) for r in records})
    tol2c = _build_tol_color_map(unique_tols)

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

    gpu_names = sorted({r.gpu_name for r in records if r.gpu_name})
    gpu_title = gpu_names[0] if len(gpu_names) == 1 else ("mixed" if len(gpu_names) > 1 else "unknown")

    # ---------- Plot 1 ----------
    fig1, ax1 = plt.subplots(figsize=plot_1_figsize)
    for rec in records:
        x = np.arange(rec.n_steps, dtype=np.int64)
        c = tol2c[float(rec.picard_tol)]
        tol_s = _tol_label(rec.picard_tol)
        if mode == "B":
            ax1.plot(x, rec.it_base, linestyle=_BASE_LS, color=c, label=f"tol={tol_s} base")
        ax1.plot(x, rec.it_warm, linestyle=_WARM_LS, color=c, label=f"tol={tol_s} warm")

    _fs1 = plot_1_fontsize or fontsize
    _fs1_t, _fs1_l, _fs1_tk, _fs1_lg = _font_sizes(_fs1)
    ax1.set_title(plot_1_title or f"Picard Iterations per Time Step{_title_suffix}", fontsize=_fs1_t)
    ax1.set_xlabel("Step", fontsize=_fs1_l)
    ax1.set_ylabel("Picard Iterations", fontsize=_fs1_l)
    ax1.tick_params(labelsize=_fs1_tk)

    if legend_position == "right":
        ax1.legend(fontsize=_fs1_lg, bbox_to_anchor=(1.02, 1), loc="upper left")
    elif legend_position == "below":
        _h1, _l1 = ax1.get_legend_handles_labels()
        ncol1 = legend_ncol if legend_ncol is not None else _legend_ncol_auto(len(_h1), legend_max_cols)
        ax1.legend(_h1, _l1, bbox_to_anchor=(0.5, -0.18), loc="upper center",
                   ncol=ncol1, fontsize=_fs1_lg, frameon=True, columnspacing=1.0)
    else:
        _sh1 = _build_split_handles(tol2c, mode)
        ncol1 = legend_ncol if legend_ncol is not None else _legend_ncol_auto(len(_sh1), legend_max_cols)
        ax1.legend(handles=_sh1, bbox_to_anchor=(0.5, -0.18), loc="upper center",
                   ncol=ncol1, fontsize=_fs1_lg, frameon=True, columnspacing=1.0)

    # ---------- Plot 2 ----------
    fig2, ax2 = plt.subplots(figsize=plot_2_figsize)
    _ws = max(int(walltime_skip_first), 0)
    for rec in records:
        steps = np.arange(rec.n_steps, dtype=np.int64)[_ws:]
        c = tol2c[float(rec.picard_tol)]
        tol_s = _tol_label(rec.picard_tol)
        ax2.plot(steps, rec.t_base[_ws:], linestyle=_BASE_LS, color=c, linewidth=1.5, label=f"tol={tol_s} base")
        ax2.plot(steps, rec.t_warm[_ws:], linestyle=_WARM_LS, color=c, linewidth=1.5, label=f"tol={tol_s} warm")

    _fs2 = plot_2_fontsize or fontsize
    _fs2_t, _fs2_l, _fs2_tk, _fs2_lg = _font_sizes(_fs2)
    ax2.set_title(plot_2_title or f"Walltime per Step (GPU: {gpu_title}){_title_suffix}", fontsize=_fs2_t)
    ax2.set_xlabel("Step", fontsize=_fs2_l)
    ax2.set_ylabel("Walltime [s]", fontsize=_fs2_l)
    ax2.tick_params(labelsize=_fs2_tk)
    _apply_grid(ax2, grid_mode=grid_mode, alpha=float(grid_alpha))

    if legend_position == "below_split":
        _sh2 = _build_split_handles(tol2c, mode)
        ncol2 = legend_ncol if legend_ncol is not None else _legend_ncol_auto(len(_sh2), legend_max_cols)
        ax2.legend(handles=_sh2, bbox_to_anchor=(0.5, -0.18), loc="upper center",
                   ncol=ncol2, fontsize=_fs2_lg, frameon=True, columnspacing=1.0)
    else:
        handles, labels = ax2.get_legend_handles_labels()
        uniq = {}
        for h, l in zip(handles, labels):
            uniq[l] = h
        _h2, _l2 = list(uniq.values()), list(uniq.keys())
        if legend_position == "right":
            ax2.legend(_h2, _l2, bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=_fs2_lg)
        else:
            ncol2 = legend_ncol if legend_ncol is not None else _legend_ncol_auto(len(_h2), legend_max_cols)
            ax2.legend(_h2, _l2, bbox_to_anchor=(0.5, -0.18), loc="upper center",
                       ncol=ncol2, fontsize=_fs2_lg, frameon=True, columnspacing=1.0)

    # ---------- Plot 3 ----------
    fig3, axes = plt.subplots(2, 3, figsize=plot_3_figsize, squeeze=False)
    moments = ("n", "u", "T")

    for j, m in enumerate(moments):
        ax = axes[0][j]
        for rec in records:
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
    _fs3_t, _fs3_l, _fs3_tk, _fs3_lg = _font_sizes(_fs3)
    for j, m in enumerate(moments):
        ax = axes[0][j]
        ax.set_title(f"Final Moment {m}(x)", fontsize=_fs3_t)
        ax.set_xlabel("Cell Index", fontsize=_fs3_l)
        ax.tick_params(labelsize=_fs3_tk)

    for j, m in enumerate(moments):
        ax = axes[1][j]
        for rec in records:
            c = tol2c[float(rec.picard_tol)]
            tol_s = _tol_label(rec.picard_tol)

            warm_m = {"n": rec.n_warm, "u": rec.u_warm, "T": rec.T_warm}[m]
            base_m = {"n": rec.n_base, "u": rec.u_base, "T": rec.T_base}[m]

            if ref_rec is not None:
                ref_m = {"n": ref_rec.n_base, "u": ref_rec.u_base, "T": ref_rec.T_base}[m]
                nx_min = min(warm_m.size, ref_m.size)
                x = np.arange(nx_min, dtype=np.int64)
                ax.plot(x, warm_m[:nx_min] - ref_m[:nx_min], linestyle=_WARM_LS, color=c,
                        label=f"tol={tol_s} warm-ref")
                if mode == "B" and rec is not ref_rec:
                    ax.plot(x, base_m[:nx_min] - ref_m[:nx_min], linestyle=_BASE_LS, color=c,
                            label=f"tol={tol_s} base-ref")
            else:
                x = np.arange(base_m.size, dtype=np.int64)
                ax.plot(x, warm_m - base_m, linestyle=_WARM_LS, color=c, label=f"tol={tol_s} Δ")
            ax.set_ylabel(f"Δ{m}")

        if ref_rec is not None:
            ax.set_title(f"Δ{m}(x) relative to reference (tol={_ref_tol_s})", fontsize=_fs3_t)
        else:
            ax.set_title("Final Difference Δ{m}(x) = warm - base", fontsize=_fs3_t)
        ax.set_xlabel("Cell Index", fontsize=_fs3_l)
        ax.tick_params(labelsize=_fs3_tk)

    if legend_position == "right":
        axes[0][-1].legend(fontsize=_fs3_lg, bbox_to_anchor=(1.02, 1), loc="upper left")
        axes[1][-1].legend(fontsize=_fs3_lg, bbox_to_anchor=(1.02, 1), loc="upper left")
    else:
        _sh3 = _build_split_handles(tol2c, mode)
        ncol3 = legend_ncol if legend_ncol is not None else _legend_ncol_auto(len(_sh3), legend_max_cols)
        fig3.legend(handles=_sh3, bbox_to_anchor=(0.5, -0.02), loc="upper center",
                    ncol=ncol3, fontsize=_fs3_lg, frameon=True, columnspacing=1.0)

    _ref_tag = f" [ref tol={_ref_tol_s}]" if ref_rec is not None else ""
    fig3.suptitle(plot_3_title or f"Final Moments and Differences{_ref_tag}{_title_suffix}",
                  y=1.02, fontsize=_fs3_t)

    # ---------- Plot 4 ----------
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

        _ref_mask = np.array([r is not ref_rec for r in recs])
        linf_n_base = _linf_ref("n_base", "n_base")[_ref_mask]
        linf_u_base = _linf_ref("u_base", "u_base")[_ref_mask]
        linf_T_base = _linf_ref("T_base", "T_base")[_ref_mask]
        tols_base = tols[_ref_mask]
    else:
        linf_n = np.asarray([r.linf_n for r in recs], dtype=np.float64)
        linf_u = np.asarray([r.linf_u for r in recs], dtype=np.float64)
        linf_T = np.asarray([r.linf_T for r in recs], dtype=np.float64)

    mean_t_base = np.asarray([r.mean_step_time_base for r in recs], dtype=np.float64)
    mean_t_warm = np.asarray([r.mean_step_time_warm for r in recs], dtype=np.float64)

    _fs4 = plot_4_fontsize or fontsize
    _fs4_t, _fs4_l, _fs4_tk, _fs4_lg = _font_sizes(_fs4)
    _ref_title4 = f" [ref tol={_ref_tol_s}]" if ref_rec is not None else ""

    fig4, ax4 = plt.subplots(figsize=plot_4_figsize)
    ax4_r = ax4.twinx()
    speed_wall = mean_t_base / np.where(mean_t_warm > 0, mean_t_warm, np.nan)

    ax4.plot(
        tols, speed, marker="o", color="#000000",
        label=_lab("iter_reduction", "Iteration Reduction (Base / Warm)", legend_labels),
    )
    ax4.plot(
        tols, speed_wall, marker="s", linestyle="--", color="#555555",
        label=_lab("wall_accel", "Walltime Acceleration (Base / Warm)", legend_labels),
    )

    ax4.set_xscale("log")
    ax4.set_xlabel("Picard Tolerance", fontsize=_fs4_l)
    ax4.set_ylabel("Speedup Ratio (Base / Warm)", fontsize=_fs4_l)
    ax4.set_title(plot_4_title or f"Speedup per Picard Tolerance (GPU: {gpu_title}){_ref_title4}{_title_suffix}",
                  fontsize=_fs4_t)
    ax4.tick_params(labelsize=_fs4_tk)
    _apply_grid(ax4, grid_mode=grid_mode, alpha=float(grid_alpha))

    if linf_mode == "max":
        linf_w_max = np.maximum.reduce([linf_n, linf_u, linf_T])
        if ref_rec is not None:
            linf_b_max = np.maximum.reduce([linf_n_base, linf_u_base, linf_T_base])
            ax4_r.plot(
                tols, linf_w_max, marker="o", color=_OKABE_ITO[0],
                label=_lab("linf_warm_max_ref", "L∞ max (warm), ref", legend_labels),
            )
            ax4_r.plot(
                tols_base, linf_b_max, marker="x", linestyle=_BASE_LS, color=_OKABE_ITO[0],
                label=_lab("linf_base_max_ref", "L∞ max (base), ref", legend_labels),
            )
        else:
            ax4_r.plot(
                tols, linf_w_max, marker="o", color=_OKABE_ITO[0],
                label=_lab("linf_max", "Final Difference L∞ max", legend_labels),
            )
    else:
        if ref_rec is not None:
            ax4_r.plot(tols, linf_n, marker="o", color=_OKABE_ITO[0],
                       label=_lab("linf_warm_n_ref", "L∞ warm (n), ref", legend_labels))
            ax4_r.plot(tols, linf_u, marker="o", color=_OKABE_ITO[1],
                       label=_lab("linf_warm_u_ref", "L∞ warm (u), ref", legend_labels))
            ax4_r.plot(tols, linf_T, marker="o", color=_OKABE_ITO[2],
                       label=_lab("linf_warm_T_ref", "L∞ warm (T), ref", legend_labels))
            ax4_r.plot(tols_base, linf_n_base, marker="x", linestyle=_BASE_LS, color=_OKABE_ITO[0],
                       label=_lab("linf_base_n_ref", "L∞ base (n), ref", legend_labels))
            ax4_r.plot(tols_base, linf_u_base, marker="x", linestyle=_BASE_LS, color=_OKABE_ITO[1],
                       label=_lab("linf_base_u_ref", "L∞ base (u), ref", legend_labels))
            ax4_r.plot(tols_base, linf_T_base, marker="x", linestyle=_BASE_LS, color=_OKABE_ITO[2],
                       label=_lab("linf_base_T_ref", "L∞ base (T), ref", legend_labels))
        else:
            ax4_r.plot(tols, linf_n, marker="o", color=_OKABE_ITO[0],
                       label=_lab("linf_n", "Final Difference L∞ (n)", legend_labels))
            ax4_r.plot(tols, linf_u, marker="o", color=_OKABE_ITO[1],
                       label=_lab("linf_u", "Final Difference L∞ (u)", legend_labels))
            ax4_r.plot(tols, linf_T, marker="o", color=_OKABE_ITO[2],
                       label=_lab("linf_T", "Final Difference L∞ (T)", legend_labels))

    ax4_r.set_ylabel("Final Difference (L∞)", fontsize=_fs4_l)
    if linf_log_scale:
        ax4_r.set_yscale("log")
    ax4_r.tick_params(labelsize=_fs4_tk)

    h1, l1 = ax4.get_legend_handles_labels()
    h2, l2 = ax4_r.get_legend_handles_labels()
    if legend_position == "right":
        ax4.legend(h1 + h2, l1 + l2, bbox_to_anchor=(1.15, 1), loc="upper left", fontsize=_fs4_lg)
    else:
        ncol4 = legend_ncol if legend_ncol is not None else _legend_ncol_auto(len(h1) + len(h2), legend_max_cols)
        ax4.legend(h1 + h2, l1 + l2, bbox_to_anchor=(0.5, -0.18), loc="upper center",
                   ncol=ncol4, fontsize=_fs4_lg, frameon=True, columnspacing=1.0)

    # ---------- Plot 5 ----------
    _fs5 = plot_5_fontsize or fontsize
    _fs5_t, _fs5_l, _fs5_tk, _fs5_lg = _font_sizes(_fs5)

    fig5, ax5 = plt.subplots(figsize=plot_5_figsize)
    ax5_r = ax5.twinx()
    mean_t_inf = np.asarray([r.mean_infer_time_est for r in recs], dtype=np.float64)

    ax5.plot(
        tols, mean_t_base, marker="o", linestyle=_BASE_LS, color="#000000",
        label=_lab("mean_step_base", "Mean Step Time (Base) [s]", legend_labels),
    )
    ax5.plot(
        tols, mean_t_warm, marker="s", linestyle=_WARM_LS, color="#000000",
        label=_lab("mean_step_warm", "Mean Step Time (Warm) [s]", legend_labels),
    )
    if show_infer_time:
        ax5.plot(
            tols, mean_t_inf, marker="D", linestyle=":", color="#555555",
            label=_lab("infer_time", "Mean Infer Time Est. [s]", legend_labels),
        )

    ax5.set_xscale("log")
    ax5.set_xlabel("Picard Tolerance", fontsize=_fs5_l)
    ax5.set_ylabel("Mean Step Time [s]", fontsize=_fs5_l)
    ax5.set_title(plot_5_title or f"Mean Step Time per Picard Tolerance (GPU: {gpu_title}){_ref_title4}{_title_suffix}",
                  fontsize=_fs5_t)
    ax5.tick_params(labelsize=_fs5_tk)
    _apply_grid(ax5, grid_mode=grid_mode, alpha=float(grid_alpha))

    if linf_mode == "max":
        linf_w_max5 = np.maximum.reduce([linf_n, linf_u, linf_T])
        if ref_rec is not None:
            linf_b_max5 = np.maximum.reduce([linf_n_base, linf_u_base, linf_T_base])
            ax5_r.plot(
                tols, linf_w_max5, marker="o", color=_OKABE_ITO[0],
                label=_lab("linf_warm_max_ref", "L∞ max (warm), ref", legend_labels),
            )
            ax5_r.plot(
                tols_base, linf_b_max5, marker="x", linestyle=_BASE_LS, color=_OKABE_ITO[0],
                label=_lab("linf_base_max_ref", "L∞ max (base), ref", legend_labels),
            )
        else:
            ax5_r.plot(
                tols, linf_w_max5, marker="o", color=_OKABE_ITO[0],
                label=_lab("linf_max", "Final Difference L∞ max", legend_labels),
            )
    else:
        if ref_rec is not None:
            ax5_r.plot(tols, linf_n, marker="o", color=_OKABE_ITO[0],
                       label=_lab("linf_warm_n_ref", "L∞ warm (n), ref", legend_labels))
            ax5_r.plot(tols, linf_u, marker="o", color=_OKABE_ITO[1],
                       label=_lab("linf_warm_u_ref", "L∞ warm (u), ref", legend_labels))
            ax5_r.plot(tols, linf_T, marker="o", color=_OKABE_ITO[2],
                       label=_lab("linf_warm_T_ref", "L∞ warm (T), ref", legend_labels))
            ax5_r.plot(tols_base, linf_n_base, marker="x", linestyle=_BASE_LS, color=_OKABE_ITO[0],
                       label=_lab("linf_base_n_ref", "L∞ base (n), ref", legend_labels))
            ax5_r.plot(tols_base, linf_u_base, marker="x", linestyle=_BASE_LS, color=_OKABE_ITO[1],
                       label=_lab("linf_base_u_ref", "L∞ base (u), ref", legend_labels))
            ax5_r.plot(tols_base, linf_T_base, marker="x", linestyle=_BASE_LS, color=_OKABE_ITO[2],
                       label=_lab("linf_base_T_ref", "L∞ base (T), ref", legend_labels))
        else:
            ax5_r.plot(tols, linf_n, marker="o", color=_OKABE_ITO[0],
                       label=_lab("linf_n", "Final Difference L∞ (n)", legend_labels))
            ax5_r.plot(tols, linf_u, marker="o", color=_OKABE_ITO[1],
                       label=_lab("linf_u", "Final Difference L∞ (u)", legend_labels))
            ax5_r.plot(tols, linf_T, marker="o", color=_OKABE_ITO[2],
                       label=_lab("linf_T", "Final Difference L∞ (T)", legend_labels))

    ax5_r.set_ylabel("Final Difference (L∞)", fontsize=_fs5_l)
    if linf_log_scale:
        ax5_r.set_yscale("log")
    ax5_r.tick_params(labelsize=_fs5_tk)

    h1, l1 = ax5.get_legend_handles_labels()
    h2, l2 = ax5_r.get_legend_handles_labels()
    if legend_position == "right":
        ax5.legend(h1 + h2, l1 + l2, bbox_to_anchor=(1.15, 1), loc="upper left", fontsize=_fs5_lg)
    else:
        ncol5 = legend_ncol if legend_ncol is not None else _legend_ncol_auto(len(h1) + len(h2), legend_max_cols)
        ax5.legend(h1 + h2, l1 + l2, bbox_to_anchor=(0.5, -0.18), loc="upper center",
                   ncol=ncol5, fontsize=_fs5_lg, frameon=True, columnspacing=1.0)

    figures = {
        plot_1_filename: fig1,
        plot_2_filename: fig2,
        plot_3_filename: fig3,
        plot_4_filename: fig4,
        plot_5_filename: fig5,
    }

    # ------------------------------------------------------------------
    # Apply layout adjustments (Fix #2: lock margins including left/right)
    # ------------------------------------------------------------------
    if layout == "paper":
        if legend_position == "right":
            for fig in figures.values():
                fig.subplots_adjust(left=0.14, right=0.80, top=0.88, bottom=0.14)
        else:
            for fig in figures.values():
                fig.subplots_adjust(left=0.14, right=0.86, top=0.88, bottom=0.30)

    saved: dict[str, str] = {}
    if save:
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
                # Fix #1: do NOT use bbox_inches="tight" in paper mode.
                fig.savefig(p, dpi=int(save_dpi_paper))
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
    p.add_argument("--ref_strictest", action="store_true")
    p.add_argument("--linf_log_scale", action="store_true")
    p.add_argument("--legend_position", type=str, default="below_split",
                   choices=["right", "below", "below_split"])
    p.add_argument("--layout", type=str, default="default",
                   choices=["default", "paper"])
    p.add_argument("--linf_mode", type=str, default="separate",
                   choices=["separate", "max"])
    p.add_argument("--no_infer_time", dest="show_infer_time",
                   action="store_false", default=True)

    p.add_argument("--legend_ncol", type=int, default=None)
    p.add_argument("--legend_max_cols", type=int, default=6)
    p.add_argument("--grid_mode", type=str, default="both", choices=["both", "major", "off"])
    p.add_argument("--grid_alpha", type=float, default=0.3)
    p.add_argument("--save_dpi_default", type=int, default=150)
    p.add_argument("--save_dpi_paper", type=int, default=300)
    p.add_argument("--save_bbox_default", type=str, default="tight")

    # NOTE:
    # legend_labels is intentionally NOT exposed in CLI here (dict parsing is messy).
    # If you want, I can add a CLI option like:
    #   --legend_label iter_reduction="Iter" --legend_label wall_accel="Wall" ...
    # But for now, use python API.

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
        layout=str(args.layout),
        linf_mode=str(args.linf_mode),
        show_infer_time=bool(args.show_infer_time),
        legend_ncol=args.legend_ncol,
        legend_max_cols=int(args.legend_max_cols),
        grid_mode=str(args.grid_mode),
        grid_alpha=float(args.grid_alpha),
        save_dpi_default=int(args.save_dpi_default),
        save_dpi_paper=int(args.save_dpi_paper),
        save_bbox_default=(None if args.save_bbox_default.lower() in ("none", "null") else str(args.save_bbox_default)),
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
    )


if __name__ == "__main__":
    main()
