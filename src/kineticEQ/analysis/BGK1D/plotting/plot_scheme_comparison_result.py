# kineticEQ/src/kineticEQ/analysis/BGK1D/plotting/plot_scheme_comparison_result.py
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple


def plot_cross_scheme_results(
    results: Dict[str, Any] | List[Dict[str, Any]],
    *,
    ref_scheme: str = "explicit",
    filename: str = "scheme_compare.png",
    output_dir: str | None = None,
    show_plots: bool = False,
    figsize_moments: Tuple[float, float] = (15, 4),
    figsize_error: Tuple[float, float] = (15, 4),
) -> List[Path]:
    """
    スキーム別比較プロット（レガシー plot_cross_scheme_results の移植版）

    保存物:
      - <base>_vs_<scheme>.png          (ref vs 対象スキームの moments 比較, n/u/T の 1x3)
      - <base>_error_vs_ref_<ref>.png   (ref に対する誤差 |q-q_ref| を全スキーム重ね描き, n/u/T の 1x3)

    returns: 保存パス一覧
    """
    import numpy as np
    import matplotlib.pyplot as plt

    if isinstance(results, dict):
        meta = results.get("meta", {}) or {}
        records = results.get("records", []) or []
    else:
        meta = {}
        records = results

    if not records:
        raise ValueError("results/records が空です")

    # scheme -> record
    scheme_to = {}
    for rec in records:
        if isinstance(rec, dict) and rec.get("scheme") is not None:
            scheme_to[str(rec["scheme"])] = rec

    if ref_scheme not in scheme_to:
        raise ValueError(f"ref_scheme='{ref_scheme}' が見つかりません。available={list(scheme_to.keys())}")

    ref = scheme_to[ref_scheme]
    n_ref = np.asarray(ref["n"])
    u_ref = np.asarray(ref["u"])
    T_ref = np.asarray(ref["T"])

    # x を meta から生成（無ければ 0..1）
    nx = int(meta.get("nx", len(n_ref)))
    Lx = float(meta.get("Lx", 1.0))
    if nx != len(n_ref):
        nx = len(n_ref)
    x = np.linspace(0.0, Lx, nx)

    out_dir = Path(output_dir) if output_dir else Path(".")
    out_dir.mkdir(parents=True, exist_ok=True)

    p = Path(filename)
    base = p.stem if p.suffix else p.name
    ext = p.suffix if p.suffix else ".png"

    saved: List[Path] = []

    # まず、各スキーム vs ref の moments 比較
    err_curves = {}  # scheme -> (err_n, err_u, err_T)

    for scheme, rec in scheme_to.items():
        if scheme == ref_scheme:
            continue

        n_s = np.asarray(rec["n"])
        u_s = np.asarray(rec["u"])
        T_s = np.asarray(rec["T"])

        if not (len(n_s) == len(u_s) == len(T_s) == len(x)):
            raise RuntimeError(
                f"moments length mismatch for scheme='{scheme}': "
                f"len(x)={len(x)}, len(n)={len(n_s)}, len(u)={len(u_s)}, len(T)={len(T_s)}"
            )

        fig, axes = plt.subplots(1, 3, figsize=figsize_moments, sharex=True)
        fig.suptitle(f"Moments comparison: scheme='{scheme}' vs ref='{ref_scheme}'", fontsize=14)

        # n
        ax = axes[0]
        ax.plot(x, n_ref, label=f"{ref_scheme} (ref)")
        ax.plot(x, n_s, linestyle="--", label=scheme)
        ax.set_xlabel("x"); ax.set_ylabel("n(x)"); ax.set_title("Density n")
        ax.grid(True, linestyle=":"); ax.legend()

        # u
        ax = axes[1]
        ax.plot(x, u_ref, label=f"{ref_scheme} (ref)")
        ax.plot(x, u_s, linestyle="--", label=scheme)
        ax.set_xlabel("x"); ax.set_ylabel("u(x)"); ax.set_title("Velocity u")
        ax.grid(True, linestyle=":"); ax.legend()

        # T
        ax = axes[2]
        ax.plot(x, T_ref, label=f"{ref_scheme} (ref)")
        ax.plot(x, T_s, linestyle="--", label=scheme)
        ax.set_xlabel("x"); ax.set_ylabel("T(x)"); ax.set_title("Temperature T")
        ax.grid(True, linestyle=":"); ax.legend()

        fig.tight_layout(rect=[0, 0.0, 1, 0.95])

        fpath = out_dir / f"{base}_vs_{scheme}{ext}"
        fig.savefig(str(fpath), dpi=300, bbox_inches="tight")
        saved.append(fpath)

        if show_plots:
            plt.show()
        else:
            plt.close(fig)

        err_curves[scheme] = (np.abs(n_s - n_ref), np.abs(u_s - u_ref), np.abs(T_s - T_ref))

    # 次に、誤差重ね描き（1枚）
    if err_curves:
        fig, axes = plt.subplots(1, 3, figsize=figsize_error, sharex=True)
        fig.suptitle(f"Error of moments vs ref='{ref_scheme}'", fontsize=14)

        ax = axes[0]
        for scheme, (en, _, _) in err_curves.items():
            ax.plot(x, en, label=scheme)
        ax.set_xlabel("x"); ax.set_ylabel(r"$|n - n_{\mathrm{ref}}|$"); ax.set_title("error of n")
        ax.grid(True, linestyle=":"); ax.legend()

        ax = axes[1]
        for scheme, (_, eu, _) in err_curves.items():
            ax.plot(x, eu, label=scheme)
        ax.set_xlabel("x"); ax.set_ylabel(r"$|u - u_{\mathrm{ref}}|$"); ax.set_title("error of u")
        ax.grid(True, linestyle=":"); ax.legend()

        ax = axes[2]
        for scheme, (_, _, eT) in err_curves.items():
            ax.plot(x, eT, label=scheme)
        ax.set_xlabel("x"); ax.set_ylabel(r"$|T - T_{\mathrm{ref}}|$"); ax.set_title("error of T")
        ax.grid(True, linestyle=":"); ax.legend()

        fig.tight_layout(rect=[0, 0.0, 1, 0.95])

        fpath = out_dir / f"{base}_error_vs_ref_{ref_scheme}{ext}"
        fig.savefig(str(fpath), dpi=300, bbox_inches="tight")
        saved.append(fpath)

        if show_plots:
            plt.show()
        else:
            plt.close(fig)

    return saved
