# kineticEQ/src/kineticEQ/analysis/BGK1D/plotting/plot_convergence_result.py
from __future__ import annotations

from typing import Any, Dict, List, Tuple
from pathlib import Path


def plot_convergence_results(
    results: List[dict] | Dict[str, Any],
    filename: str = "Conv_bench.png",
    dt: float | None = None,
    nx: int | None = None,
    nv: int | None = None,
    ho_tol: float | None = None,
    picard_tol: float | None = None,
    lo_tol: float | None = None,
    figsize: Tuple[float, float] = (10, 4.5),
    show_plots: bool = True,
) -> None:
    """
    HOLO / Picard(implicit) 収束性テスト結果の可視化（レガシー互換API）

    Figure 1（横並び 2 枚）:
      - 左 : HOLO vs Picard の反復回数（HO outer / Picard iter）
      - 右 : HOLO vs Picard の最終残差（log軸）

    Figure 2（横並び 2 枚）:
      - 左 : HOLO の外側反復回数のみ（拡大表示）
      - 右 : LO 内部反復の回数（HOLO のみ、1 step あたり）

    Figure 3（横並び 2 枚）:
      - 左 : 1 step あたりの walltime [s]（線形軸）
      - 右 : 1 step あたりの walltime [s]（対数軸）

    Parameters
    ----------
    results : list[dict] or dict
        - list[dict]: 旧仕様
        - dict: {"meta": {...}, "records": [...]} という新仕様
          records[i] は benchlog を含む場合がある
    filename : str
        Figure 1 の PNG ファイル名
        Figure 2 は "<元ファイル名>_holo.png"
        Figure 3 は "<元ファイル名>_walltime.png"
    dt, nx, nv : optional
        タイトルに表示する Δt, nx, nv（None の場合 meta から補完）
    ho_tol, picard_tol, lo_tol : optional
        収束許容誤差（タイトル表示用）
    figsize : tuple
        各 Figure のサイズ
    show_plots : bool
        True のとき表示（plt.show）
    """
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    from collections import defaultdict

    # -----------------------------
    # results の分解（新/旧両対応）
    # -----------------------------
    if isinstance(results, dict):
        meta = results.get("meta", {}) or {}
        records = results.get("records", []) or []
    else:
        meta = {}
        records = results

    if not records:
        raise ValueError("results/records が空です")

    # meta -> 引数 の順で値を解決
    def _resolve(name: str, current):
        if current is not None:
            return current
        if name in meta:
            return meta[name]
        return None

    dt = _resolve("dt", dt)
    nx = _resolve("nx", nx)
    nv = _resolve("nv", nv)

    # 旧: gpu_name / 新: device_name を許容
    gpu_name = meta.get("gpu_name", None) or meta.get("device_name", None)

    T_total_meta = meta.get("T_total", None)

    # -----------------------------
    # record 吸収（benchlog/旧直下）
    # -----------------------------
    def _get_benchlog(rec: Dict[str, Any]) -> Dict[str, Any]:
        b = rec.get("benchlog", None)
        return b if isinstance(b, dict) else {}

    def _walltime_sec(rec: Dict[str, Any]) -> float | None:
        # 新形式: walltime_sec
        if "walltime_sec" in rec and rec["walltime_sec"] is not None:
            try:
                return float(rec["walltime_sec"])
            except Exception:
                return None
        # 旧形式: walltime
        if "walltime" in rec and rec["walltime"] is not None:
            try:
                return float(rec["walltime"])
            except Exception:
                return None
        return None

    def _scheme_kind(s: Any) -> str:
        # "holo", "implicit" (new) / "implicit_picard" (old) を吸収
        ss = str(s).strip().lower()
        if ss in ("holo", "hl", "ho", "holo_nn"):
            return "holo"
        if ss in ("implicit", "imp", "implicit_picard", "picard"):
            return "picard"
        # 不明な場合はそのまま
        return ss

    # -----------------------------
    # 集計: tau_tilde ごと
    # -----------------------------
    per_tau = defaultdict(
        lambda: {
            "time_ho": [],
            "ho_iter": [],
            "ho_res": [],
            "lo_total": [],
            "time_pi": [],
            "pi_iter": [],
            "pi_res": [],
            "step_ho": [],
            "wall_ho": [],
            "step_pi": [],
            "wall_pi": [],
        }
    )

    for rec in records:
        if not isinstance(rec, dict):
            continue

        tau = float(rec.get("tau_tilde"))
        scheme = _scheme_kind(rec.get("scheme"))

        t = float(rec.get("time", 0.0))
        step = int(rec.get("step", 0))
        wall = _walltime_sec(rec)

        # benchlog があれば優先、なければ rec 直下（旧仕様）
        b = _get_benchlog(rec)

        if scheme == "holo":
            # 新: benchlog["ho_iter"], ["ho_residual"], ["lo_iter"]
            # 旧: rec["ho_iter"], rec["ho_residual"], rec["lo_iter_list"]
            ho_iter = b.get("ho_iter", rec.get("ho_iter", None))
            ho_res = b.get("ho_residual", rec.get("ho_residual", None))

            # LO は旧: lo_iter_list を合計、新: lo_iter（合計）
            if "lo_iter_list" in rec and isinstance(rec["lo_iter_list"], (list, tuple)):
                lo_total = int(sum(rec["lo_iter_list"]))
            else:
                lo_total = b.get("lo_iter", rec.get("lo_iter", None))

            if ho_iter is None or ho_res is None:
                # plot に必要なキーが無い場合はスキップ（ただし time は保持したいので最小限に）
                continue

            per_tau[tau]["time_ho"].append(t)
            per_tau[tau]["ho_iter"].append(int(ho_iter))
            per_tau[tau]["ho_res"].append(float(ho_res))
            per_tau[tau]["lo_total"].append(int(lo_total) if lo_total is not None else 0)

            if wall is not None:
                per_tau[tau]["step_ho"].append(step)
                per_tau[tau]["wall_ho"].append(float(wall))

        elif scheme == "picard":
            # 新: benchlog["picard_iter"], ["picard_residual"]
            # 旧: rec["picard_iter"], rec["picard_residual"]
            pi_iter = b.get("picard_iter", rec.get("picard_iter", None))
            pi_res = b.get("picard_residual", rec.get("picard_residual", None))

            if pi_iter is None or pi_res is None:
                continue

            per_tau[tau]["time_pi"].append(t)
            per_tau[tau]["pi_iter"].append(int(pi_iter))
            per_tau[tau]["pi_res"].append(float(pi_res))

            if wall is not None:
                per_tau[tau]["step_pi"].append(step)
                per_tau[tau]["wall_pi"].append(float(wall))

    if not per_tau:
        raise ValueError("有効な記録が見つかりません（benchlog/キー不足の可能性）")

    # t / T_total
    if T_total_meta is not None:
        T_total = float(T_total_meta)
    else:
        all_times = [tt for v in per_tau.values() for tt in (v["time_ho"] + v["time_pi"])]
        if not all_times:
            raise ValueError("time 情報が見つかりません")
        T_total = max(all_times)

    taus = sorted(per_tau.keys())

    # カラー・マーカー設定（tau ごと）
    color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    tau_colors = {tau: color_cycle[i % len(color_cycle)] for i, tau in enumerate(taus)}
    base_markers = ["o", "s", "^", "D", "v", "P", "X", "h"]
    tau_markers = {tau: base_markers[i % len(base_markers)] for i, tau in enumerate(taus)}

    # ==================================================
    # Figure 1: HOLO vs Picard（反復 & 残差）
    # ==================================================
    fig1, axes1 = plt.subplots(1, 2, figsize=figsize)
    ax_outer, ax_resid = axes1

    title_lines = ["HOLO vs Picard convergence"]
    title_info = []
    if dt is not None:
        title_info.append(r"$\Delta t$={:.4g}".format(float(dt)))
    if nx is not None:
        title_info.append(f"nx={int(nx)}")
    if nv is not None:
        title_info.append(f"nv={int(nv)}")
    if picard_tol is not None:
        title_info.append(r"tol$_P$={:.1e}".format(float(picard_tol)))
    if ho_tol is not None:
        title_info.append(r"tol$_{{HO}}$={:.1e}".format(float(ho_tol)))
    if lo_tol is not None:
        title_info.append(r"tol$_{{LO}}$={:.1e}".format(float(lo_tol)))
    if title_info:
        title_lines.append(", ".join(title_info))
    fig1.suptitle("\n".join(title_lines), fontsize=12)

    legend_handles, legend_labels = [], []

    # ==================================================
    # Figure 2: HOLO-only
    # ==================================================
    fig2, axes2 = plt.subplots(1, 2, figsize=figsize)
    ax_outer_ho, ax_lo_inner = axes2
    ho_legend_handles, ho_legend_labels = [], []

    # ==================================================
    # Figure 3: walltime
    # ==================================================
    fig3, axes3 = plt.subplots(1, 2, figsize=figsize)
    ax3_lin, ax3_log = axes3
    wall_legend_handles, wall_legend_labels = [], []

    # -----------------------------
    # plot loop
    # -----------------------------
    for tau in taus:
        info = per_tau[tau]
        color = tau_colors[tau]
        marker = tau_markers[tau]

        # ----- HOLO -----
        if info["time_ho"]:
            t_arr = np.asarray(info["time_ho"], dtype=float)
            t_norm = t_arr / T_total

            ho_iter = np.asarray(info["ho_iter"], dtype=float)
            ho_res = np.asarray(info["ho_res"], dtype=float)
            lo_total = np.asarray(info["lo_total"], dtype=float)

            npts = len(t_norm)
            mark_every = max(1, npts // 20)

            line_outer_ho, = ax_outer.plot(
                t_norm, ho_iter,
                color=color, linestyle="-",
                marker=marker, markersize=4,
                markevery=mark_every,
                label=f"HOLO, τ̃={tau:g}",
            )
            ax_resid.plot(
                t_norm, ho_res,
                color=color, linestyle="-",
                marker=marker, markersize=4,
                markevery=mark_every,
            )

            legend_handles.append(line_outer_ho)
            legend_labels.append(f"HOLO, τ̃={tau:g}")

            line_outer_ho_zoom, = ax_outer_ho.plot(
                t_norm, ho_iter,
                color=color, linestyle="-",
                marker=marker, markersize=4,
                markevery=mark_every,
                label=f"τ̃={tau:g}",
            )
            ax_lo_inner.plot(
                t_norm, lo_total,
                color=color, linestyle="-",
                marker=marker, markersize=4,
                markevery=mark_every,
            )
            ho_legend_handles.append(line_outer_ho_zoom)
            ho_legend_labels.append(f"τ̃={tau:g}")

            # walltime (HOLO)
            if info["wall_ho"]:
                # sampled times に合わせる（records が sample_interval で間引かれている想定）
                t_wall = np.asarray(info["time_ho"], dtype=float)
                t_wall_norm = t_wall / T_total
                w_ho = np.asarray(info["wall_ho"], dtype=float)

                npts_w = len(t_wall_norm)
                mark_every_w = max(1, npts_w // 20)

                h_ho_lin, = ax3_lin.plot(
                    t_wall_norm, w_ho,
                    color=color, linestyle="-",
                    marker=marker, markersize=4,
                    markerfacecolor=color, markeredgewidth=0.0,
                    markevery=mark_every_w,
                    label=f"HOLO, τ̃={tau:g}",
                )
                ax3_log.plot(
                    t_wall_norm, w_ho,
                    color=color, linestyle="-",
                    marker=marker, markersize=4,
                    markerfacecolor=color, markeredgewidth=0.0,
                    markevery=mark_every_w,
                )
                wall_legend_handles.append(h_ho_lin)
                wall_legend_labels.append(f"HOLO, τ̃={tau:g}")

        # ----- Picard -----
        if info["time_pi"]:
            t_arr = np.asarray(info["time_pi"], dtype=float)
            t_norm = t_arr / T_total

            pi_iter = np.asarray(info["pi_iter"], dtype=float)
            pi_res = np.asarray(info["pi_res"], dtype=float)

            npts = len(t_norm)
            mark_every = max(1, npts // 20)

            line_outer_pi, = ax_outer.plot(
                t_norm, pi_iter,
                color=color, linestyle="--",
                marker=marker, markersize=4,
                markevery=mark_every,
                markerfacecolor="none", markeredgewidth=1.5,
                label=f"Picard, τ̃={tau:g}",
            )
            ax_resid.plot(
                t_norm, pi_res,
                color=color, linestyle="--",
                marker=marker, markersize=4,
                markevery=mark_every,
                markerfacecolor="none", markeredgewidth=1.5,
            )

            legend_handles.append(line_outer_pi)
            legend_labels.append(f"Picard, τ̃={tau:g}")

            # walltime (Picard)
            if info["wall_pi"]:
                t_wall = np.asarray(info["time_pi"], dtype=float)
                t_wall_norm = t_wall / T_total
                w_pi = np.asarray(info["wall_pi"], dtype=float)

                npts_w = len(t_wall_norm)
                mark_every_w = max(1, npts_w // 20)

                h_pi_lin, = ax3_lin.plot(
                    t_wall_norm, w_pi,
                    color=color, linestyle="--",
                    marker=marker, markersize=4,
                    markerfacecolor="none", markeredgewidth=1.5,
                    markevery=mark_every_w,
                    label=f"Picard, τ̃={tau:g}",
                )
                ax3_log.plot(
                    t_wall_norm, w_pi,
                    color=color, linestyle="--",
                    marker=marker, markersize=4,
                    markerfacecolor="none", markeredgewidth=1.5,
                    markevery=mark_every_w,
                )
                wall_legend_handles.append(h_pi_lin)
                wall_legend_labels.append(f"Picard, τ̃={tau:g}")

    # -----------------------------
    # axis cosmetics
    # -----------------------------
    ax_outer.set_xlabel("t / T_total")
    ax_outer.set_ylabel("Number of iterations per time step")
    ax_outer.set_title("HOLO vs Picard: iteration count")
    ax_outer.grid(True, alpha=0.3)

    ax_resid.set_xlabel("t / T_total")
    ax_resid.set_ylabel("Final residual per time step")
    ax_resid.set_title("HOLO vs Picard: residual")
    ax_resid.set_yscale("log")
    ax_resid.grid(True, which="both", alpha=0.3)

    if legend_handles:
        fig1.legend(
            legend_handles, legend_labels,
            loc="center left", bbox_to_anchor=(0.99, 0.5),
            borderaxespad=0.5, fontsize=8,
        )
    fig1.tight_layout(rect=[0.02, 0.03, 0.95, 0.92])

    ax_outer_ho.set_xlabel("t / T_total")
    ax_outer_ho.set_ylabel("HOLO outer iterations")
    ax_outer_ho.set_title("HOLO outer iterations (zoom)")
    ax_outer_ho.grid(True, alpha=0.3)

    all_ho_iters = [it for v in per_tau.values() for it in v["ho_iter"]]
    if all_ho_iters:
        ax_outer_ho.set_ylim(0, max(all_ho_iters) * 1.2)

    ax_lo_inner.set_xlabel("t / T_total")
    ax_lo_inner.set_ylabel("LO inner iterations (total per step)")
    ax_lo_inner.set_title("LO inner iterations per time step (HOLO)")
    ax_lo_inner.grid(True, alpha=0.3)

    if ho_legend_handles:
        fig2.legend(
            ho_legend_handles, ho_legend_labels,
            loc="center left", bbox_to_anchor=(0.99, 0.5),
            borderaxespad=0.5, fontsize=8,
        )
    fig2.tight_layout(rect=[0.02, 0.03, 0.95, 0.95])

    ax3_lin.set_xlabel("t / T_total")
    ax3_lin.set_ylabel("Walltime per step [s]")
    ax3_lin.set_title("Per-step walltime (linear)")
    ax3_lin.grid(True, alpha=0.3)

    ax3_log.set_xlabel("t / T_total")
    ax3_log.set_ylabel("Walltime per step [s]")
    ax3_log.set_title("Per-step walltime (log)")
    ax3_log.set_yscale("log")
    ax3_log.grid(True, which="both", alpha=0.3)

    if gpu_name:
        fig3.suptitle(f"Per-step walltime (HOLO vs Picard) – Device: {gpu_name}", fontsize=12)
    else:
        fig3.suptitle("Per-step walltime (HOLO vs Picard)", fontsize=12)

    if wall_legend_handles:
        fig3.legend(
            wall_legend_handles, wall_legend_labels,
            loc="center left", bbox_to_anchor=(0.99, 0.5),
            borderaxespad=0.5, fontsize=8,
        )
    fig3.tight_layout(rect=[0.02, 0.05, 0.95, 0.9])

    # -----------------------------
    # save
    # -----------------------------
    # 出力先ディレクトリを作る
    out_path = Path(filename)
    if out_path.parent and str(out_path.parent) != ".":
        out_path.parent.mkdir(parents=True, exist_ok=True)

    fig1.savefig(str(out_path), dpi=300, bbox_inches="tight")

    root, ext = os.path.splitext(str(out_path))
    if not ext:
        ext = ".png"
    holo_filename = root + "_holo" + ext
    wall_filename = root + "_walltime" + ext

    fig2.savefig(holo_filename, dpi=300, bbox_inches="tight")
    fig3.savefig(wall_filename, dpi=300, bbox_inches="tight")

    if show_plots:
        plt.show()
    else:
        plt.close(fig1)
        plt.close(fig2)
        plt.close(fig3)

    print(f"収束性テストの図を保存: {out_path}")
    print(f"HOLO-only 図を保存: {holo_filename}")
    print(f"Walltime 図を保存: {wall_filename}")
