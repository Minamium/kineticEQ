##############################
# Local端末で実行する可視化関数群 #
##############################

import torch
import numpy as np
import math
from scipy.interpolate import interp1d
from typing import Any, Union

# BGK1Dbaseクラスを継承するためのimport
from .BGK1Dsim import BGK1D
from .BGK1D_old import BGK1D_old

# 可視化関数群
class BGK1DPlotMixin:
    """可視化, 解析用の関数群"""
    #状態可視化メソッド
    def plot_state(self, filename='bgk_simulation.png'):
        """状態の可視化"""
        import matplotlib.pyplot as plt
        # CPUに転送（matplotlib用）
        f_cpu = self.f.cpu().numpy()
        x_cpu = self.x.cpu().numpy()
        v_cpu = self.v.cpu().numpy()

        # モーメント計算
        n, u, T = self.calculate_moments()
        n_cpu = n.cpu().numpy()
        u_cpu = u.cpu().numpy()
        T_cpu = T.cpu().numpy()

        # 4つのサブプロット作成
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))

        # 1. 分布関数f(x,v)のヒートマップ
        im1 = ax1.imshow(f_cpu.T, aspect='auto', origin='lower', 
                     extent=[x_cpu[0], x_cpu[-1], v_cpu[0], v_cpu[-1]],
                     cmap='viridis')
        ax1.set_xlabel('Position x')
        ax1.set_ylabel('Velocity v')
        ax1.set_title('Distribution Function f(x,v)')
        plt.colorbar(im1, ax=ax1)

        # 2. 密度分布
        ax2.plot(x_cpu, n_cpu, 'b-', linewidth=2)
        ax2.set_xlabel('Position x')
        ax2.set_ylabel('Density n')
        ax2.set_title('Density Distribution')
        ax2.grid(True, alpha=0.3)

        # 3. 速度分布
        ax3.plot(x_cpu, u_cpu, 'r-', linewidth=2)
        ax3.set_xlabel('Position x')
        ax3.set_ylabel('Mean Velocity u')
        ax3.set_title('Velocity Distribution')
        ax3.grid(True, alpha=0.3)

        # 4. 温度分布
        ax4.plot(x_cpu, T_cpu, 'g-', linewidth=2)
        ax4.set_xlabel('Position x')
        ax4.set_ylabel('Temperature T')
        ax4.set_title('Temperature Distribution')
        ax4.grid(True, alpha=0.3)

        plt.show()
        plt.savefig(filename)

        # 統計情報表示
        print(f"Density: mean={n_cpu.mean():.4f}, min={n_cpu.min():.4f}, max={n_cpu.max():.4f}")
        print(f"Velocity: mean={u_cpu.mean():.4f}, min={u_cpu.min():.4f}, max={u_cpu.max():.4f}")
        print(f"Temperature: mean={T_cpu.mean():.4f}, min={T_cpu.min():.4f}, max={T_cpu.max():.4f}")

    #GIF作成メソッド
    def create_gif(self, filename='bgk_simulation.gif', fps=2):
        """GIF作成（バグ修正版）"""
        if not self.animation_data:
            print("No animation data to create GIF")
            return

        try:
            import matplotlib.pyplot as plt
            from PIL import Image
            import io

            print(f"Creating GIF with {len(self.animation_data)} frames...")

            frames = []
            x_cpu = self.x.cpu().numpy()
            v_cpu = self.v.cpu().numpy()

            # カラーマップの範囲を事前計算
            f_max = max([np.max(d['f']) for d in self.animation_data])
            n_max = max([np.max(d['n']) for d in self.animation_data]) * 1.1
            u_max = max([np.max(np.abs(d['u'])) for d in self.animation_data]) * 1.1
            T_max = max([np.max(d['T']) for d in self.animation_data]) * 1.1

            for i, data in enumerate(self.animation_data):
                fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))

                # 1. 分布関数
                im1 = ax1.imshow(data['f'].T, aspect='auto', origin='lower',
                                extent=[x_cpu[0], x_cpu[-1], v_cpu[0], v_cpu[-1]],
                                cmap='viridis', vmin=0, vmax=f_max)
                ax1.set_xlabel('Position x')
                ax1.set_ylabel('Velocity v')
                ax1.set_title(f'Distribution f(x,v) at t={data["time"]:.3f}')

                # 2. 密度
                ax2.plot(x_cpu, data['n'], 'b-', linewidth=2)
                ax2.set_xlabel('Position x')
                ax2.set_ylabel('Density n')
                ax2.set_title('Density')
                ax2.set_ylim([0, n_max])
                ax2.grid(True, alpha=0.3)

                # 3. 速度
                ax3.plot(x_cpu, data['u'], 'r-', linewidth=2)
                ax3.set_xlabel('Position x')
                ax3.set_ylabel('Velocity u')
                ax3.set_title('Velocity')
                ax3.set_ylim([-u_max, u_max])
                ax3.grid(True, alpha=0.3)

                # 4. 温度
                ax4.plot(x_cpu, data['T'], 'g-', linewidth=2)
                ax4.set_xlabel('Position x')
                ax4.set_ylabel('Temperature T')
                ax4.set_title('Temperature')
                ax4.set_ylim([0, T_max])
                ax4.grid(True, alpha=0.3)

                # PNG画像として保存（修正版）
                buf = io.BytesIO()
                plt.savefig(buf, format='png', dpi=80, bbox_inches='tight')
                buf.seek(0)

                # 画像データを完全にメモリにコピー
                img = Image.open(buf)
                img_copy = img.copy()  # ← 重要：コピーを作成
                frames.append(img_copy)

                # リソース解放
                plt.close(fig)
                img.close()
                buf.close()

                if (i + 1) % 3 == 0:
                    print(f"  Frame {i+1}/{len(self.animation_data)} processed")

            # GIF保存
            if frames:
                frames[0].save(filename, save_all=True, append_images=frames[1:], 
                               duration=int(1000/fps), loop=0)
                print(f"GIF saved: '{filename}' ({len(frames)} frames)")
            else:
                print("No frames to save")

        except ImportError:
            print("PIL required for GIF creation: pip install pillow")
        except Exception as e:
            print(f"GIF creation failed: {e}")
            import traceback
            traceback.print_exc()  # デバッグ用

    # 実行時間ベンチマーク可視化メソッド
    def plot_timing_benchmark(self, bench_results: dict | None = None, 
                             filename: str | None = None,
                             output_filename: str | None = None,
                             save_fig: bool = True,
                             show_gpu_time: bool = True,
                             show_overhead: bool = True,
                             figsize: tuple[float, float] | None = None,
                             text_fontsize: int = 8):
        """実行時間ベンチマーク結果を可視化
        
        Parameters
        ----------
        bench_results : dict | None
            ベンチマーク結果辞書。Noneの場合はself.benchmark_resultsを使用
        filename : str | None
            pklファイルから読み込む場合のファイル名（入力用）
        output_filename : str | None
            保存する画像ファイル名（拡張子含む）。Noneの場合は'timing_benchmark_heatmap.png'
        save_fig : bool
            図を保存するかどうか
        show_gpu_time : bool
            GPU時間も表示するかどうか（利用可能な場合）
        show_overhead : bool
            オーバーヘッドΔTも表示するかどうか（利用可能な場合）
        figsize : tuple(float, float) | None
            図全体のサイズ (幅, 高さ) [inch]。None なら自動決定。
        text_fontsize : int
            各セルに描画する数値テキストのフォントサイズ。
        """
        import matplotlib.pyplot as plt
        import numpy as np
        import pickle
        import torch
        
        # matplotlib言語設定を英語に
        plt.rcParams['font.family'] = 'DejaVu Sans'
        
        # データの取得
        if filename is not None:
            with open(filename, 'rb') as f:
                data = pickle.load(f)
                if 'results' in data:
                    bench_results = data['results']
                else:
                    bench_results = data
        elif bench_results is None:
            if not hasattr(self, 'benchmark_results'):
                raise ValueError("Benchmark results not found")
            bench_results = self.benchmark_results
        
        if bench_results.get('bench_type') != 'time':
            raise ValueError("Not a timing benchmark result")
        
        timing_results = bench_results['timing_results']
        
        # デバイス情報を取得（ベンチマーク結果から）
        device_name = bench_results.get('device_name', 'Unknown Device')
        cpu_name = bench_results.get('cpu_name', 'Unknown CPU')
        
        # データの整理
        nx_values = set()
        nv_values = set()
        
        for grid_key in timing_results.keys():
            nx, nv = map(int, grid_key.split('x'))
            nx_values.add(nx)
            nv_values.add(nv)
        
        nx_sorted = sorted(nx_values)
        nv_sorted = sorted(nv_values)
        
        # 1ステップあたりの時間マトリックス作成
        cpu_step_time_matrix = np.zeros((len(nx_sorted), len(nv_sorted)))
        gpu_step_time_matrix = np.zeros((len(nx_sorted), len(nv_sorted)))
        has_gpu_data = False
        
        for i, nx in enumerate(nx_sorted):
            for j, nv in enumerate(nv_sorted):
                grid_key = f"{nx}x{nv}"
                if grid_key in timing_results:
                    result = timing_results[grid_key]
                    total_steps = result['total_steps']
                    
                    # 1ステップあたりの時間（ミリ秒）
                    cpu_step_time_matrix[i, j] = (result['cpu_total_time_sec'] / total_steps) * 1000
                    
                    if 'gpu_total_time_sec' in result:
                        gpu_step_time_matrix[i, j] = (result['gpu_total_time_sec'] / total_steps) * 1000
                        has_gpu_data = True
        
        # 図の作成
        subplot_num = 1
        if has_gpu_data and show_gpu_time:
            subplot_num += 1
        if has_gpu_data and show_overhead and torch.cuda.is_available():
            subplot_num += 1

        # figure sizeを自動またはユーザ指定で設定
        if figsize is None:
            fig_width = 4 * subplot_num
            fig_height = 6
            figsize = (fig_width, fig_height)

        fig, axes = plt.subplots(1, subplot_num, figsize=figsize, constrained_layout=True)
        # サブプロット間の水平間隔を少し空ける
        fig.subplots_adjust(wspace=0.25)
        
        if not isinstance(axes, np.ndarray):
            axes = [axes]
        
        # CPU時間ヒートマップ
        im1 = axes[0].imshow(cpu_step_time_matrix, cmap='viridis', aspect='auto')
        axes[0].set_title(f'Wall-clock Time per Step (ms) - {cpu_name}')
        axes[0].set_xlabel('nv (Velocity Grid Points)')
        axes[0].set_ylabel('nx (Spatial Grid Points)')
        axes[0].set_xticks(range(len(nv_sorted)))
        axes[0].set_xticklabels(nv_sorted)
        axes[0].set_yticks(range(len(nx_sorted)))
        axes[0].set_yticklabels(nx_sorted)
        
        # 値ごとに背景輝度を判定して文字色を自動変更（可読性向上）
        def _get_contrast_color(val: float, im):
            rgba = im.cmap(im.norm(val))
            # 相対輝度 (ITU-R BT.601)
            lum = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
            return 'black' if lum > 0.5 else 'white'
        
        for i in range(len(nx_sorted)):
            for j in range(len(nv_sorted)):
                color = _get_contrast_color(cpu_step_time_matrix[i, j], im1)
                axes[0].text(j, i, f'{cpu_step_time_matrix[i, j]:.2f}',
                              ha="center", va="center", color=color, fontsize=text_fontsize)
        
        plt.colorbar(im1, ax=axes[0], label='Time (ms)')
        
        ax_idx = 1
        if has_gpu_data and show_gpu_time:
            im2 = axes[ax_idx].imshow(gpu_step_time_matrix, cmap='plasma', aspect='auto')
            axes[ax_idx].set_title(f'GPU Time per Step (ms) - {device_name}')
            axes[ax_idx].set_xlabel('nv (Velocity Grid Points)')
            axes[ax_idx].set_ylabel('nx (Spatial Grid Points)')
            axes[ax_idx].set_xticks(range(len(nv_sorted)))
            axes[ax_idx].set_xticklabels(nv_sorted)
            axes[ax_idx].set_yticks(range(len(nx_sorted)))
            axes[ax_idx].set_yticklabels(nx_sorted)
            for i in range(len(nx_sorted)):
                for j in range(len(nv_sorted)):
                    color = _get_contrast_color(gpu_step_time_matrix[i, j], im2)
                    axes[ax_idx].text(j, i, f'{gpu_step_time_matrix[i, j]:.2f}',
                                      ha="center", va="center", color=color, fontsize=text_fontsize)
            plt.colorbar(im2, ax=axes[ax_idx], label='Time (ms)')
            ax_idx += 1

        # オーバーヘッド ΔT ヒートマップ
        if has_gpu_data and show_overhead and torch.cuda.is_available():
            diff_matrix = cpu_step_time_matrix - gpu_step_time_matrix
            im3 = axes[ax_idx].imshow(diff_matrix, cmap='magma', aspect='auto')
            axes[ax_idx].set_title('ΔT (CPU - GPU) per Step (ms)')
            axes[ax_idx].set_xlabel('nv (Velocity Grid Points)')
            axes[ax_idx].set_ylabel('nx (Spatial Grid Points)')
            axes[ax_idx].set_xticks(range(len(nv_sorted)))
            axes[ax_idx].set_xticklabels(nv_sorted)
            axes[ax_idx].set_yticks(range(len(nx_sorted)))
            axes[ax_idx].set_yticklabels(nx_sorted)
            for i in range(len(nx_sorted)):
                for j in range(len(nv_sorted)):
                    color = _get_contrast_color(diff_matrix[i, j], im3)
                    axes[ax_idx].text(j, i, f'{diff_matrix[i, j]:.2f}',
                                      ha="center", va="center", color=color, fontsize=text_fontsize)
            plt.colorbar(im3, ax=axes[ax_idx], label='ΔT (ms)')
        
        # plt.tight_layout()  # constrained_layout=True を使用しているため tight_layout は不要

        if save_fig:
            if output_filename is None:
                output_filename = 'timing_benchmark_heatmap.png'
            plt.savefig(output_filename, dpi=300, bbox_inches='tight')
            print(f"Figure saved: {output_filename}")

    # ベンチマーク結果plotメソッド
    def plot_benchmark_results(
        self,
        bench_results: dict[str, Union[str, dict[int, dict[str, Any]]]],
        error_dict: dict[int, dict[str, dict[str, float]]],
        fname_moment: str = "moments.png",
        fname_error: str = "errors.png",
        logscale: bool = True,
        show_plots: bool = False,
        ) -> dict[str, Any]:
        """
        ベンチマーク結果を可視化する（横軸＝格子点数）。

        Parameters
        ----------
        bench_results : dict
            run_benchmark の戻り値。キーは格子点数（nx / nv）。
        error_dict : dict
            compute_error の戻り値。
        fname_moment : str
            モーメント比較図の保存名。
        fname_error : str
            誤差収束図の保存名。
        logscale : bool
            True: log‐log プロット、False: 線形軸。
        show_plots : bool
            True なら画面にも表示。

        Returns
        -------
        dict
            convergence_orders, 保存ファイル名などの情報。
        """
        import matplotlib.pyplot as plt
        from itertools import cycle
        import warnings

        # --------------------------------------------------------
        # 入力検査
        # --------------------------------------------------------
        if not bench_results:
            raise ValueError("bench_results が空です")
        if not error_dict:
            raise ValueError("error_dict が空です")

        benchmark_type = bench_results.get("bench_type", "spatial")
        results = {k: v for k, v in bench_results.items() if k != "bench_type"}

        if len(results) < 2:
            raise ValueError(f"格子レベルが {len(results)} 個しかありません（最低 2 個必要）")

        # 参照解（最細格子）はキーが最大のものとする
        ref_key = max(results.keys())

        # --------------------------------------------------------
        # モーメント比較図
        # --------------------------------------------------------
        tol_colors = [
            "#4477AA",
            "#EE6677",
            "#228833",
            "#CCBB44",
            "#66CCEE",
            "#AA3377",
            "#BBBBBB",
        ]
        line_styles = ["-", "--", "-.", ":"]
        color_cycle = cycle(tol_colors)
        style_cycle = cycle(line_styles)

        fig1, axes1 = plt.subplots(1, 3, figsize=(18, 5))

        grid_keys = [ref_key] + [k for k in sorted(results.keys(), reverse=True) if k != ref_key]

        legend_handles, legend_labels = [], []

        for grid_key in grid_keys:
            color = next(color_cycle)
            linestyle = next(style_cycle)
            lw = 3 if grid_key == ref_key else 2

            res = results[grid_key]
            x = res["x"]

            if benchmark_type == "spatial":
                grid_info = f"nx={grid_key} (dx={res.get('dx', 1.0/grid_key):.3f})"
            else:
                grid_info = f"nv={grid_key} (dv={res.get('dv', 10.0/grid_key):.3f})"

            if grid_key == ref_key:
                grid_info += " (REF)"

            ln, = axes1[0].plot(x, res["n"], color=color, linestyle=linestyle, lw=lw, label=grid_info)
            axes1[1].plot(x, res["u"], color=color, linestyle=linestyle, lw=lw)
            axes1[2].plot(x, res["T"], color=color, linestyle=linestyle, lw=lw)

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
        # plt.tight_layout()  # constrained_layout=True を使用しているため tight_layout は不要
        fig1.savefig(fname_moment, dpi=300, bbox_inches="tight", pad_inches=0.1, facecolor="white")
        if show_plots:
            plt.show()
        plt.close(fig1)

        # --------------------------------------------------------
        # 誤差収束図（横軸＝格子点数）
        # --------------------------------------------------------
        fig2, axes2 = plt.subplots(1, 3, figsize=(18, 5))

        # 参照解以外のキー（粗 → 細）
        test_keys = sorted([k for k in error_dict.keys() if k != ref_key])

        # 横軸データ：格子点数
        grid_counts = [int(k) for k in test_keys]
        if benchmark_type == "spatial":
            x_label = "Grid points nx"
        else:
            x_label = "Grid points nv"

        variables = ["f", "n", "u", "T"]
        norms = ["L1", "L2", "Linf"]
        var_colors = {"f": "#1f77b4", "n": "#2ca02c", "u": "#d62728", "T": "#9467bd"}
        markers = ["o", "s", "^", "D"]

        convergence_orders: dict[str, dict[str, float]] = {}

        for norm_idx, norm in enumerate(norms):
            ax = axes2[norm_idx]
            convergence_orders[norm] = {}

            for var_idx, var in enumerate(variables):
                errors = []
                counts = []

                for i, key in enumerate(test_keys):
                    try:
                        val = error_dict[key][norm][var]
                        if val <= 0 or np.isnan(val) or np.isinf(val):
                            warnings.warn(f"Invalid error value skipped: {var} {norm} grid={key} error={val}")
                            continue
                        errors.append(val)
                        counts.append(grid_counts[i])
                    except (KeyError, TypeError):
                        warnings.warn(f"Error retrieval failed: {var} {norm} grid={key}")
                        continue

                if len(errors) < 2:
                    convergence_orders[norm][var] = np.nan
                    continue

                # 粗 → 細 でソート
                idx = np.argsort(counts)
                counts_sorted = np.array(counts)[idx]
                errs_sorted = np.array(errors)[idx]

                # 収束次数 p:  error ≈ C * N^{-p}
                slope = np.polyfit(np.log(counts_sorted), np.log(errs_sorted), 1)[0]
                p_mean = -slope
                convergence_orders[norm][var] = p_mean

                label = f"{var} (p̅={p_mean:.2f})"
                plot = ax.loglog if logscale else ax.plot
                plot(
                    counts_sorted,
                    errs_sorted,
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

        # plt.tight_layout()  # constrained_layout=True を使用しているため tight_layout は不要
        fig2.savefig(fname_error, dpi=300, bbox_inches="tight", pad_inches=0.1, facecolor="white")
        if show_plots:
            plt.show()
        plt.close(fig2)

        print(f"モーメント比較図を保存: {fname_moment}")
        print(f"誤差収束図を保存: {fname_error}")
        if show_plots:
            print("図を画面に表示しました")

        return {
            "convergence_orders": convergence_orders,
            "figures_saved": [fname_moment, fname_error],
            "benchmark_type": benchmark_type,
            "ref_grid": ref_key,
        }

    # ベンチマーク結果の分布関数プロット
    def plot_distribution_heatmaps(
        self,
        bench_results: dict,
        show_plots: bool = True,
        save_individual: bool = False,
        fname_base: str = "distribution_heatmaps"
        ) -> dict:
        """
        f(x,v) と |f − f_ref| をヒートマップで可視化。
        補間は compute_error と同じく nearest。引数と戻り値は従来どおり。
        """
        import numpy as np
        import matplotlib.pyplot as plt
        from matplotlib.colors import Normalize
        from scipy.interpolate import interp1d

        # ────────────── 入力チェック ──────────────
        if not bench_results:
            raise ValueError("ベンチマーク結果が空です")

        bench_type = bench_results["bench_type"]
        bench_dict = {k: v for k, v in bench_results.items() if isinstance(k, int)}
        if not bench_dict:
            raise ValueError("数値格子キーが見つかりません")

        ref_key     = max(bench_dict.keys())
        ref_result  = bench_dict[ref_key]
        sorted_keys = sorted(bench_dict.keys())
        n_grids     = len(sorted_keys)

        # ────────────── 補間と誤差計算 ──────────────
        def _nearest_interp(ref_arr, ref_axis, tgt_axis):
            return interp1d(ref_axis, ref_arr,
                            kind="nearest", assume_sorted=True)(tgt_axis)

        def _get_error(coarse_res):
            """粗格子に合わせた |f − f_ref|"""
            if bench_type == "spatial":
                ref_f = np.zeros_like(coarse_res["f"])
                for v_idx in range(len(coarse_res["v"])):
                    ref_f[:, v_idx] = _nearest_interp(
                        ref_result["f"][:, v_idx],
                        ref_result["x"], coarse_res["x"])
            else:  # velocity
                ref_f = np.zeros_like(coarse_res["f"])
                for x_idx in range(len(coarse_res["x"])):
                    ref_f[x_idx, :] = _nearest_interp(
                        ref_result["f"][x_idx, :],
                        ref_result["v"], coarse_res["v"])
            return np.abs(coarse_res["f"] - ref_f)

        # ────────────── 保存ファイル管理 ──────────────
        saved_files = []

        # ─────────────────────────────────────────────
        # ① 個別保存モード
        # ─────────────────────────────────────────────
        if save_individual:
            for key in sorted_keys:
                res  = bench_dict[key]
                f    = np.asarray(res["f"])
                err  = _get_error(res)

                fig, axes = plt.subplots(2, 1, figsize=(8, 10),
                                         constrained_layout=True)

                im0 = axes[0].imshow(f, origin="lower", aspect="auto",
                                     cmap="cividis")
                axes[0].set_title(f"f(x,v) – Grid {key}")
                fig.colorbar(im0, ax=axes[0])

                im1 = axes[1].imshow(err, origin="lower", aspect="auto",
                                     cmap="magma",
                                     norm=Normalize(vmin=0, vmax=err.max()))
                axes[1].set_title(r"|f − f$_{\mathrm{ref}}$|")
                fig.colorbar(im1, ax=axes[1])

                for ax in axes:
                    ax.set_xlabel("v-index")
                    ax.set_ylabel("x-index")

                if show_plots:
                    plt.show()
                fname = f"{fname_base}_grid_{key}.png"
                fig.savefig(fname, dpi=300, bbox_inches='tight')
                plt.close(fig)
                saved_files.append(fname)

        # ─────────────────────────────────────────────
        # ② 統合保存モード（3 列固定）
        # ─────────────────────────────────────────────
        else:
            # サブプロット行列サイズ
            n_cols  = 3
            n_rows  = 2 * int(np.ceil(n_grids / n_cols))   # f と err で 2 倍

            fig, axes = plt.subplots(
                n_rows, n_cols, figsize=(12, 3 * n_rows),
                constrained_layout=True
            )

            # f 用カラースケール（共有）
            f_all  = np.concatenate(
                [np.asarray(bench_dict[k]["f"]).ravel() for k in sorted_keys]
            )
            f_norm = Normalize(vmin=f_all.min(), vmax=f_all.max())

            # err 用カラースケール（共有）
            err_all  = np.concatenate(
                [_get_error(bench_dict[k]).ravel() for k in sorted_keys]
            )
            err_norm = Normalize(vmin=0, vmax=err_all.max())

            # --------------------- 描画ループ ---------------------
            for idx, key in enumerate(sorted_keys):
                pair_row = (idx // n_cols) * 2     # f 用行
                col      = idx % n_cols            # 列

                res  = bench_dict[key]
                f    = np.asarray(res["f"])
                err  = _get_error(res)

                # f(x,v)
                ax_f = axes[pair_row, col]
                im_f = ax_f.imshow(f, origin="lower", aspect="auto",
                                   cmap="cividis", norm=f_norm)
                ax_f.set_title(f"f – N={key}")
                ax_f.set_xlabel("v")
                ax_f.set_ylabel("x")

                # |f − f_ref|
                ax_e = axes[pair_row + 1, col]
                im_e = ax_e.imshow(err, origin="lower", aspect="auto",
                                   cmap="magma", norm=err_norm)
                ax_e.set_title("|f − f_ref|")
                ax_e.set_xlabel("v")
                ax_e.set_ylabel("x")

            # カラーバー（左: f, 右: 誤差）を 1 本ずつ
            fig.colorbar(im_f, ax=axes[0::2, :].ravel().tolist(),
                         location="left", shrink=0.6, pad=0.02)
            fig.colorbar(im_e, ax=axes[1::2, :].ravel().tolist(),
                         location="right", shrink=0.6, pad=0.02)

            if show_plots:
                plt.show()

            fname = f"{fname_base}.png"
            fig.savefig(fname, dpi=300, bbox_inches="tight")
            plt.close(fig)
            saved_files.append(fname)

        print("=== ヒートマップ保存完了 ===")
        for i, f in enumerate(saved_files, 1):
            print(f"{i}. {f}")

        return {"saved_files": saved_files, "grid_keys": sorted_keys}

    # ベンチマーク結果の分布関数インタラクティブ3次元プロット
    def plot_distribution_interactive(
        self,
        bench_results: dict,
        keys: list[int] | None = None,          # ← 追加：描画対象の格子キー
        show_plots: bool = True,
        save_html: bool = False,
        fname_base: str = "distribution_interactive"
        ) -> dict:
        """
        指定された格子キーについて
        * f(x,v) の 3D サーフェス
        * |f − f_ref| の 3D サーフェス
        を横並びで表示・保存する。

        Notes
        -----
        - `keys` が None の場合は最細格子（max key）のみ描画
        - 誤差は `bench_type` を参照し、compute_error と同じ最近接補間で計算
        """
        import numpy as np
        from matplotlib.colors import Normalize
        from scipy.interpolate import interp1d
        try:
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots
        except ImportError:
            raise ImportError("Plotly が必要です: pip install plotly")

        # ─── 入力チェック ───
        if not bench_results:
            raise ValueError("ベンチマーク結果が空です")

        bench_type = bench_results["bench_type"]
        bench_dict = {k: v for k, v in bench_results.items() if isinstance(k, int)}
        if not bench_dict:
            raise ValueError("数値格子キーが見つかりません")

        ref_key    = max(bench_dict.keys())
        ref_result = bench_dict[ref_key]

        # 対象キー決定
        if keys is None:
            target_keys = [ref_key]           # デフォルト: 最細格子
        else:
            target_keys = [k for k in keys if k in bench_dict]
            if not target_keys:
                raise ValueError("指定 keys が bench_results に存在しません")

        # ─── 関数定義 ───
        def _nearest_interp(ref_arr, ref_axis, tgt_axis):
            return interp1d(ref_axis, ref_arr,
                            kind="nearest", assume_sorted=True)(tgt_axis)

        def _get_error(coarse_res):
            """粗格子に合わせた |f − f_ref|"""
            if bench_type == "spatial":
                ref_f = np.zeros_like(coarse_res["f"])
                for v_idx in range(len(coarse_res["v"])):
                    ref_f[:, v_idx] = _nearest_interp(
                        ref_result["f"][:, v_idx],
                        ref_result["x"], coarse_res["x"])
            else:  # velocity
                ref_f = np.zeros_like(coarse_res["f"])
                for x_idx in range(len(coarse_res["x"])):
                    ref_f[x_idx, :] = _nearest_interp(
                        ref_result["f"][x_idx, :],
                        ref_result["v"], coarse_res["v"])
            return np.abs(coarse_res["f"] - ref_f)

        # f, err のグローバル正規化
        all_f   = np.concatenate([bench_dict[k]["f"].ravel() for k in target_keys])
        f_min   = all_f.min()
        f_max   = all_f.max()
        all_err = np.concatenate([_get_error(bench_dict[k]).ravel() for k in target_keys])
        err_max = all_err.max()

        saved_files = []

        # ─── 描画ループ ───
        for key in target_keys:
            res  = bench_dict[key]
            f    = np.asarray(res["f"])
            err  = _get_error(res)
            x    = np.asarray(res["x"])
            v    = np.asarray(res["v"])

            # Figure with 2 surfaces side-by-side
            fig = make_subplots(
                rows=1, cols=2,
                specs=[[{'type': 'surface'}, {'type': 'surface'}]],
                column_widths=[0.5, 0.5],
                horizontal_spacing=0.05,
                subplot_titles=("f(x,v)", "|f − f_ref|")
            )

            # f surface
            fig.add_trace(
                go.Surface(
                    x=x, y=v, z=f.T,
                    colorscale="Viridis",
                    cmin=f_min, cmax=f_max,
                    showscale=False,
                    hovertemplate='x:%{x:.3f}<br>v:%{y:.3f}<br>f:%{z:.6f}<extra></extra>'
                ),
                row=1, col=1
            )

            # error surface
            fig.add_trace(
                go.Surface(
                    x=x, y=v, z=err.T,
                    colorscale="Magma",
                    cmin=0, cmax=err_max,
                    showscale=False,
                    hovertemplate='x:%{x:.3f}<br>v:%{y:.3f}<br>|err|:%{z:.6e}<extra></extra>'
                ),
                row=1, col=2
            )

            # レイアウト
            fig.update_layout(
                title=f"Grid {key} – nx×nv = {f.shape[0]}×{f.shape[1]}",
                scene=dict(
                    xaxis_title='x', yaxis_title='v', zaxis_title='f',
                    aspectmode='cube'
                ),
                scene2=dict(
                    xaxis_title='x', yaxis_title='v', zaxis_title='|err|',
                    aspectmode='cube'
                ),
                width=1100, height=550,
                margin=dict(l=20, r=20, t=40, b=20)
            )

            # 保存
            if save_html:
                fname = f"{fname_base}_grid{key}.html"
                fig.write_html(fname)
                saved_files.append(fname)
                print(f"Grid {key}: {fname} を保存")

            if show_plots:
                fig.show()

    # 収束性テスト結果の可視化メソッド
    def plot_convergence_results(
        self,
        results: list[dict] | dict[str, Any],
        filename: str = "Conv_bench.png",
        dt: float | None = None,
        nx: int | None = None,
        nv: int | None = None,
        ho_tol: float | None = None,
        picard_tol: float | None = None,
        lo_tol: float | None = None,
        figsize: tuple[float, float] = (10, 4.5),
        show_plots: bool = True,
    ) -> None:
        """
        HOLO / Picard 収束性テスト結果の可視化

        Figure 1（横並び 2 枚）:
          - 左 : HOLO vs Picard の外側反復回数
          - 右 : HOLO vs Picard の最終残差

        Figure 2（横並び 2 枚）:
          - 左 : HOLO の外側反復回数のみ（拡大表示）
          - 右 : LO 内部反復の総回数（HOLO のみ）

        Figure 3（横並び 2 枚）:
          - 左 : 1 step あたりの walltime [s]（線形軸）
          - 右 : 1 step あたりの walltime [s]（対数軸）

        Parameters
        ----------
        results : list[dict] or dict
            - list[dict]: 旧仕様。run_convergence_test() が list を返す場合。
            - dict: {"meta": {...}, "records": [...]} という新仕様。
        filename : str
            Figure 1 の PNG ファイル名。
            Figure 2 は "<元ファイル名>_holo.png"、
            Figure 3 は "<元ファイル名>_walltime.png" で保存する。
        dt, nx, nv : optional
            タイトルに表示する Δt, nx, nv。None の場合 self / meta から取得。
        ho_tol, picard_tol, lo_tol : optional
            HOLO / Picard / LO の収束許容誤差（タイトル表示用）。
        figsize : tuple
            各 Figure のサイズ。
        show_plots : bool
            True のとき plt.show() 相当で Figure を表示。
        """
        import os
        import numpy as np
        import matplotlib.pyplot as plt
        from collections import defaultdict

        # --------------------------------------------------
        # results 構造の判定: list[dict] or dict(meta+records)
        # --------------------------------------------------
        if isinstance(results, dict):
            meta = results.get("meta", {}) or {}
            records = results.get("records", []) or []
        else:
            meta = {}
            records = results

        if not records:
            raise ValueError("results/records が空です")

        # meta -> 引数 -> self 属性 の順で値を解決するヘルパ
        def _resolve(name: str, current):
            if current is not None:
                return current
            if name in meta:
                return meta[name]
            return getattr(self, name, None)

        # self / meta からのデフォルト値補完
        dt = _resolve("dt", dt)
        nx = _resolve("nx", nx)
        nv = _resolve("nv", nv)
        ho_tol = _resolve("ho_tol", ho_tol)
        picard_tol = _resolve("picard_tol", picard_tol)
        lo_tol = _resolve("lo_tol", lo_tol)

        # T_total も meta にあれば使う（なければ後で time の max から決定）
        T_total_meta = meta.get("T_total", None)

        # GPU 名（walltime 図タイトル用）
        gpu_name = meta.get("gpu_name", None)

        # --------------------------------------------------
        # 結果集計: tau_tilde ごとにまとめる
        #   time_ho/time_pi: 物理時間 t
        #   step_*        : ステップ番号
        #   wall_*        : 1 step ごとの walltime [s]
        # --------------------------------------------------
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
            tau = float(rec["tau_tilde"])
            scheme = rec["scheme"]
            t = float(rec["time"])
            step = int(rec.get("step", 0))
            wall = rec.get("walltime", None)

            if scheme == "holo":
                per_tau[tau]["time_ho"].append(t)
                per_tau[tau]["ho_iter"].append(int(rec["ho_iter"]))
                per_tau[tau]["ho_res"].append(float(rec["ho_residual"]))
                lo_list = rec.get("lo_iter_list", [])
                per_tau[tau]["lo_total"].append(int(sum(lo_list)))
                if wall is not None:
                    per_tau[tau]["step_ho"].append(step)
                    per_tau[tau]["wall_ho"].append(float(wall))

            elif scheme == "implicit_picard":
                per_tau[tau]["time_pi"].append(t)
                per_tau[tau]["pi_iter"].append(int(rec["picard_iter"]))
                per_tau[tau]["pi_res"].append(float(rec["picard_residual"]))
                if wall is not None:
                    per_tau[tau]["step_pi"].append(step)
                    per_tau[tau]["wall_pi"].append(float(wall))

        # t / T_total 用に T_total を決定
        if T_total_meta is not None:
            T_total = float(T_total_meta)
        else:
            all_times = [tt for v in per_tau.values() for tt in (v["time_ho"] + v["time_pi"])]
            if not all_times:
                raise ValueError("time 情報が見つかりません")
            T_total = max(all_times)

        # tau の並び順（小さい順）
        taus = sorted(per_tau.keys())

        # カラー・マーカー設定（tau ごとに色、HOLO/Picard で塗りつぶし/抜き）
        color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        tau_colors = {tau: color_cycle[i % len(color_cycle)] for i, tau in enumerate(taus)}

        base_markers = ["o", "s", "^", "D", "v", "P", "X", "h"]
        tau_markers = {tau: base_markers[i % len(base_markers)] for i, tau in enumerate(taus)}

        # ==================================================
        # Figure 1: HOLO vs Picard（外側反復 & 残差）
        # ==================================================
        fig1, axes1 = plt.subplots(1, 2, figsize=figsize)
        ax_outer = axes1[0]
        ax_resid = axes1[1]

        # figure タイトル
        title_lines = []
        title_info = []
        if dt is not None:
            title_info.append(r"$\Delta t$={:.4g}".format(dt))
        if nx is not None:
            title_info.append(f"nx={nx}")
        if nv is not None:
            title_info.append(f"nv={nv}")
        if picard_tol is not None:
            title_info.append(r"tol$_P$={:.1e}".format(picard_tol))
        if ho_tol is not None:
            title_info.append(r"tol$_{{HO}}$={:.1e}".format(ho_tol))
        if lo_tol is not None:
            title_info.append(r"tol$_{{LO}}$={:.1e}".format(lo_tol))

        title_lines.append("HOLO vs Picard convergence")
        if title_info:
            title_lines.append(", ".join(title_info))
        fig1.suptitle("\n".join(title_lines), fontsize=12)

        legend_handles = []
        legend_labels = []

        # ==================================================
        # Figure 2: HOLO-only（外側反復 & LO 内部反復合計）
        # ==================================================
        fig2, axes2 = plt.subplots(1, 2, figsize=figsize)
        ax_outer_ho = axes2[0]
        ax_lo_inner = axes2[1]

        ho_legend_handles = []
        ho_legend_labels = []

        # ==================================================
        # Figure 3: per-step walltime（HOLO vs Picard）, 左: linear, 右: log
        # ==================================================
        fig3, axes3 = plt.subplots(1, 2, figsize=figsize)
        ax3_lin, ax3_log = axes3
        wall_legend_handles = []
        wall_legend_labels = []

        # --------------------------------------------------
        # プロットループ
        # --------------------------------------------------
        for tau in taus:
            info = per_tau[tau]
            color = tau_colors[tau]
            marker = tau_markers[tau]

            # ---------- HOLO ----------
            if info["time_ho"]:
                t_arr = np.array(info["time_ho"])
                t_norm = t_arr / T_total
                ho_iter = np.array(info["ho_iter"])
                ho_res = np.array(info["ho_res"])
                lo_total = np.array(info["lo_total"])

                npts = len(t_norm)
                mark_every = max(1, npts // 20)

                # Figure 1: outer iter & residual
                line_outer_ho, = ax_outer.plot(
                    t_norm,
                    ho_iter,
                    color=color,
                    linestyle="-",
                    marker=marker,
                    markersize=4,
                    markevery=mark_every,
                    label=f"HOLO, τ̃={tau:g}",
                )
                ax_resid.plot(
                    t_norm,
                    ho_res,
                    color=color,
                    linestyle="-",
                    marker=marker,
                    markersize=4,
                    markevery=mark_every,
                )

                legend_handles.append(line_outer_ho)
                legend_labels.append(f"HOLO, τ̃={tau:g}")

                # Figure 2: HOLO-only
                line_outer_ho_zoom, = ax_outer_ho.plot(
                    t_norm,
                    ho_iter,
                    color=color,
                    linestyle="-",
                    marker=marker,
                    markersize=4,
                    markevery=mark_every,
                    label=f"τ̃={tau:g}",
                )
                ax_lo_inner.plot(
                    t_norm,
                    lo_total,
                    color=color,
                    linestyle="-",
                    marker=marker,
                    markersize=4,
                    markevery=mark_every,
                )

                ho_legend_handles.append(line_outer_ho_zoom)
                ho_legend_labels.append(f"τ̃={tau:g}")

                # Figure 3: per-step walltime（HOLO）
                if info["wall_ho"]:
                    t_wall = np.array(info["time_ho"])
                    t_wall_norm = t_wall / T_total
                    w_ho = np.array(info["wall_ho"])
                    npts_w = len(t_wall_norm)
                    mark_every_w = max(1, npts_w // 20)

                    # 線形軸
                    h_ho_lin, = ax3_lin.plot(
                        t_wall_norm,
                        w_ho,
                        color=color,
                        linestyle="-",
                        marker=marker,
                        markersize=4,
                        markerfacecolor=color,
                        markeredgewidth=0.0,
                        markevery=mark_every_w,
                        label=f"HOLO, τ̃={tau:g}",
                    )
                    # 対数軸
                    ax3_log.plot(
                        t_wall_norm,
                        w_ho,
                        color=color,
                        linestyle="-",
                        marker=marker,
                        markersize=4,
                        markerfacecolor=color,
                        markeredgewidth=0.0,
                        markevery=mark_every_w,
                    )

                    wall_legend_handles.append(h_ho_lin)
                    wall_legend_labels.append(f"HOLO, τ̃={tau:g}")

            # ---------- Picard ----------
            if info["time_pi"]:
                t_arr = np.array(info["time_pi"])
                t_norm = t_arr / T_total
                pi_iter = np.array(info["pi_iter"])
                pi_res = np.array(info["pi_res"])

                npts = len(t_norm)
                mark_every = max(1, npts // 20)

                line_outer_pi, = ax_outer.plot(
                    t_norm,
                    pi_iter,
                    color=color,
                    linestyle="--",
                    marker=marker,
                    markersize=4,
                    markevery=mark_every,
                    markerfacecolor="none",
                    markeredgewidth=1.5,
                    label=f"Picard, τ̃={tau:g}",
                )
                ax_resid.plot(
                    t_norm,
                    pi_res,
                    color=color,
                    linestyle="--",
                    marker=marker,
                    markersize=4,
                    markevery=mark_every,
                    markerfacecolor="none",
                    markeredgewidth=1.5,
                )

                legend_handles.append(line_outer_pi)
                legend_labels.append(f"Picard, τ̃={tau:g}")

                # Figure 3: per-step walltime（Picard）
                if info["wall_pi"]:
                    t_wall = np.array(info["time_pi"])
                    t_wall_norm = t_wall / T_total
                    w_pi = np.array(info["wall_pi"])
                    npts_w = len(t_wall_norm)
                    mark_every_w = max(1, npts_w // 20)

                    # 線形軸
                    h_pi_lin, = ax3_lin.plot(
                        t_wall_norm,
                        w_pi,
                        color=color,
                        linestyle="--",
                        marker=marker,
                        markersize=4,
                        markerfacecolor="none",
                        markeredgewidth=1.5,
                        markevery=mark_every_w,
                        label=f"Picard, τ̃={tau:g}",
                    )
                    # 対数軸
                    ax3_log.plot(
                        t_wall_norm,
                        w_pi,
                        color=color,
                        linestyle="--",
                        marker=marker,
                        markersize=4,
                        markerfacecolor="none",
                        markeredgewidth=1.5,
                        markevery=mark_every_w,
                    )

                    wall_legend_handles.append(h_pi_lin)
                    wall_legend_labels.append(f"Picard, τ̃={tau:g}")

        # --------------------------------------------------
        # 軸の体裁: Figure 1
        # --------------------------------------------------
        ax_outer.set_xlabel("t / T_total")
        ax_outer.set_ylabel("Number of iterations per time step")
        ax_outer.set_title("HOLO vs Picard: iteration count (outer)")
        ax_outer.grid(True, alpha=0.3)

        ax_resid.set_xlabel("t / T_total")
        ax_resid.set_ylabel("Final residual per time step")
        ax_resid.set_title("HOLO vs Picard: residual")
        ax_resid.set_yscale("log")
        ax_resid.grid(True, which="both", alpha=0.3)

        if legend_handles:
            fig1.legend(
                legend_handles,
                legend_labels,
                loc="center left",
                bbox_to_anchor=(0.99, 0.5),
                borderaxespad=0.5,
                fontsize=8,
            )

        fig1.tight_layout(rect=[0.02, 0.03, 0.95, 0.92])

        # --------------------------------------------------
        # 軸の体裁: Figure 2（HOLO-only）
        # --------------------------------------------------
        ax_outer_ho.set_xlabel("t / T_total")
        ax_outer_ho.set_ylabel("HOLO outer iterations")
        ax_outer_ho.set_title("HOLO outer iterations (zoom)")
        ax_outer_ho.grid(True, alpha=0.3)

        all_ho_iters = [it for v in per_tau.values() for it in v["ho_iter"]]
        if all_ho_iters:
            ymin = 0
            ymax = max(all_ho_iters) * 1.2
            ax_outer_ho.set_ylim(ymin, ymax)

        ax_lo_inner.set_xlabel("t / T_total")
        ax_lo_inner.set_ylabel("Total LO inner iterations")
        ax_lo_inner.set_title("LO inner iterations per time step (HOLO)")
        ax_lo_inner.grid(True, alpha=0.3)

        if ho_legend_handles:
            fig2.legend(
                ho_legend_handles,
                ho_legend_labels,
                loc="center left",
                bbox_to_anchor=(0.99, 0.5),
                borderaxespad=0.5,
                fontsize=8,
            )

        fig2.tight_layout(rect=[0.02, 0.03, 0.95, 0.95])

        # --------------------------------------------------
        # 軸の体裁: Figure 3（walltime, linear & log）
        # --------------------------------------------------
        ax3_lin.set_xlabel("t / T_total")
        ax3_lin.set_ylabel("Walltime per step [s]")
        ax3_lin.set_title("Per-step walltime (linear)")
        ax3_lin.grid(True, alpha=0.3)

        ax3_log.set_xlabel("t / T_total")
        ax3_log.set_ylabel("Walltime per step [s]")
        ax3_log.set_title("Per-step walltime (log)")
        ax3_log.set_yscale("log")
        ax3_log.grid(True, which="both", alpha=0.3)

        # GPU 名を suptitle に反映
        if gpu_name:
            fig3.suptitle(f"Per-step walltime (HOLO vs Picard) – GPU: {gpu_name}", fontsize=12)
        else:
            fig3.suptitle("Per-step walltime (HOLO vs Picard)", fontsize=12)

        if wall_legend_handles:
            fig3.legend(
                wall_legend_handles,
                wall_legend_labels,
                loc="center left",
                bbox_to_anchor=(0.99, 0.5),
                borderaxespad=0.5,
                fontsize=8,
            )

        fig3.tight_layout(rect=[0.02, 0.05, 0.95, 0.9])

        # --------------------------------------------------
        # 表示 & 保存
        # --------------------------------------------------
        # Figure 1: メイン
        fig1.savefig(filename, dpi=300, bbox_inches="tight")

        # Figure 2: HOLO-only は _holo サフィックスで保存
        root, ext = os.path.splitext(filename)
        if not ext:
            ext = ".png"
        holo_filename = root + "_holo" + ext
        fig2.savefig(holo_filename, dpi=300, bbox_inches="tight")

        # Figure 3: walltime は _walltime サフィックスで保存
        wall_filename = root + "_walltime" + ext
        fig3.savefig(wall_filename, dpi=300, bbox_inches="tight")

        if show_plots:
            fig1.show()
            fig2.show()
            fig3.show()

        print(f"収束性テストの図を保存: {filename}")
        print(f"HOLO-only 図を保存: {holo_filename}")
        print(f"Walltime 図を保存: {wall_filename}")

    def plot_cross_scheme_results(self, results, ref_scheme: str = "explicit"):
        """
        cross_scheme_test_results に保存されたスキーム同士を比較プロットするメソッド。

        引数で与えた ref_scheme を基準として、
        - 図1群: n, u, T を 1x3 サブプロットで、ref と比較対象スキームを同じ軸にプロット
                （これは ref 以外の各スキームごとに個別に生成）
        - 図2: n, u, T の点ごとの誤差 |q - q_ref| を 1x3 サブプロットで、
               ref 以外の全スキームを重ね描きする。

        Parameters
        ----------
        results : dict または list[dict]
            run_scheme_comparison_test(...) が返す dict
            （{"meta":..., "records":[...]}）か、その中の "records" の list。
        ref_scheme : str
            records[i]["scheme"] に対応するスキーム名。
            例: "explicit", "implicit", "holo"
        """
        import matplotlib.pyplot as plt
        import numpy as np

        # results が {"meta":..., "records":[...]} 形式の場合に分解
        if isinstance(results, dict):
            meta = results.get("meta", {})
            records = results.get("records", None)
        else:
            meta = {}
            records = results

        if records is None or not isinstance(records, (list, tuple)) or len(records) == 0:
            raise RuntimeError(
                "results 内に有効な records がありません。"
                "run_scheme_comparison_test(...) の戻り値、"
                "またはその ['records'] を渡してください。"
            )

        # scheme 名で引けるように辞書化
        scheme_to_record = {}
        for rec in records:
            if not isinstance(rec, dict):
                raise RuntimeError(
                    "records 内の要素が dict ではありません。"
                    f" 要素の型: {type(rec)}"
                )
            scheme_name = rec.get("scheme", None)
            if scheme_name is None:
                continue
            # 同じ scheme 名が複数ある場合は最後を採用（通常は一意）
            scheme_to_record[scheme_name] = rec

        if ref_scheme not in scheme_to_record:
            raise ValueError(
                f"指定された ref_scheme='{ref_scheme}' に対応する結果が "
                "records に存在しません。"
                f" 利用可能なスキーム: {list(scheme_to_record.keys())}"
            )

        # 参照スキームの結果
        ref_rec = scheme_to_record[ref_scheme]
        n_ref = np.asarray(ref_rec["n"])
        u_ref = np.asarray(ref_rec["u"])
        T_ref = np.asarray(ref_rec["T"])

        # x 座標を meta から再構成（self は一切使わない）
        nx_meta = int(meta["nx"]) if "nx" in meta else len(n_ref)
        Lx_meta = float(meta["Lx"]) if "Lx" in meta else 1.0

        # nx_meta と n_ref の長さが食い違っていたら n_ref に合わせる
        if nx_meta != len(n_ref):
            nx = len(n_ref)
        else:
            nx = nx_meta
        x = np.linspace(0.0, Lx_meta, nx)

        if x.shape[0] != n_ref.shape[0]:
            raise RuntimeError(
                f"x の長さ (len(x)={x.shape[0]}) と ref n の長さ "
                f"(len(n_ref)={n_ref.shape[0]}) が一致していません。"
            )

        # 誤差曲線をまとめて描くためのバッファ
        #   err_curves[scheme_name] = (err_n, err_u, err_T)
        err_curves = {}

        # ref 以外のスキームについて順次プロット
        for scheme_name, rec in scheme_to_record.items():
            if scheme_name == ref_scheme:
                continue  # ref 自身はスキップ

            n_s = np.asarray(rec["n"])
            u_s = np.asarray(rec["u"])
            T_s = np.asarray(rec["T"])

            # 安全のため次元チェック
            if not (len(n_s) == len(u_s) == len(T_s) == len(x)):
                raise RuntimeError(
                    f"スキーム '{scheme_name}' のモーメント配列の長さが不一致です:"
                    f" len(x)={len(x)}, len(n)={len(n_s)}, len(u)={len(u_s)}, len(T)={len(T_s)}"
                )

            # =====================================================
            # 1. モーメント比較プロット (各スキームごとに 1 図)
            # =====================================================
            fig1, axes1 = plt.subplots(1, 3, figsize=(15, 4), sharex=True)
            fig1.suptitle(
                f"Moments comparison: scheme='{scheme_name}' vs ref='{ref_scheme}'",
                fontsize=14
            )

            # n
            ax = axes1[0]
            ax.plot(x, n_ref, label=f"{ref_scheme} (ref)")
            ax.plot(x, n_s, linestyle="--", label=f"{scheme_name}")
            ax.set_xlabel("x")
            ax.set_ylabel("n(x)")
            ax.set_title("Density n")
            ax.grid(True, linestyle=":")
            ax.legend()

            # u
            ax = axes1[1]
            ax.plot(x, u_ref, label=f"{ref_scheme} (ref)")
            ax.plot(x, u_s, linestyle="--", label=f"{scheme_name}")
            ax.set_xlabel("x")
            ax.set_ylabel("u(x)")
            ax.set_title("Velocity u")
            ax.grid(True, linestyle=":")
            ax.legend()

            # T
            ax = axes1[2]
            ax.plot(x, T_ref, label=f"{ref_scheme} (ref)")
            ax.plot(x, T_s, linestyle="--", label=f"{scheme_name}")
            ax.set_xlabel("x")
            ax.set_ylabel("T(x)")
            ax.set_title("Temperature T")
            ax.grid(True, linestyle=":")
            ax.legend()

            fig1.tight_layout(rect=[0, 0.0, 1, 0.95])
            plt.show()

            # =====================================================
            # 2. 誤差配列を計算してバッファに保存（プロットは後でまとめて）
            # =====================================================
            err_n = np.abs(n_s - n_ref)
            err_u = np.abs(u_s - u_ref)
            err_T = np.abs(T_s - T_ref)

            err_curves[scheme_name] = (err_n, err_u, err_T)

        # =========================================================
        # 3. 誤差 |q - q_ref| を 1 枚に重ねてプロット
        # =========================================================
        if err_curves:
            fig2, axes2 = plt.subplots(1, 3, figsize=(15, 4), sharex=True)
            fig2.suptitle(
                f"Error of moments vs ref='{ref_scheme}'",
                fontsize=14
            )

            # n 誤差
            ax = axes2[0]
            for scheme_name, (err_n, _, _) in err_curves.items():
                ax.plot(x, err_n, label=scheme_name)
            ax.set_xlabel("x")
            ax.set_ylabel(r"$|n - n_{\mathrm{ref}}|$")
            ax.set_title("error of n")
            ax.grid(True, linestyle=":")
            ax.legend()

            # u 誤差
            ax = axes2[1]
            for scheme_name, (_, err_u, _) in err_curves.items():
                ax.plot(x, err_u, label=scheme_name)
            ax.set_xlabel("x")
            ax.set_ylabel(r"$|u - u_{\mathrm{ref}}|$")
            ax.set_title("error of u")
            ax.grid(True, linestyle=":")
            ax.legend()

            # T 誤差
            ax = axes2[2]
            for scheme_name, (_, _, err_T) in err_curves.items():
                ax.plot(x, err_T, label=scheme_name)
            ax.set_xlabel("x")
            ax.set_ylabel(r"$|T - T_{\mathrm{ref}}|$")
            ax.set_title("error of T")
            ax.grid(True, linestyle=":")
            ax.legend()

            fig2.tight_layout(rect=[0, 0.0, 1, 0.95])
            plt.show()

    # ベンチマーク結果の保存・読み込みユーティリティ
    def save_benchmark_results(self, bench_results: dict | None = None, filename: str = "benchmark_results.pkl") -> str:
        """ベンチマーク結果 dict を pickle 形式で保存

        Parameters
        ----------
        bench_results : dict | None
            run_benchmark の戻り値。None のときは self.benchmark_results を使用。
        filename : str, default "benchmark_results.pkl"
            保存先ファイル名。

        Returns
        -------
        str
            実際に保存したファイルパス。
        """
        import pickle, os, datetime, platform, torch

        if bench_results is None:
            if not hasattr(self, "benchmark_results"):
                raise ValueError("bench_results が None で、self.benchmark_results も存在しません")
            bench_results = self.benchmark_results

        # 追加メタデータ
        meta = {
            "saved_at"      : datetime.datetime.now().isoformat(timespec="seconds"),
            "hostname"      : platform.node(),
            "torch_version" : torch.__version__,
            "device"        : str(self.device),
            "dtype"         : str(self.dtype),
            "solver"        : self.solver,
        }
        data_to_save = {
            "meta" : meta,
            "results" : bench_results,
        }

        # ディレクトリが無ければ作成
        os.makedirs(os.path.dirname(filename) or ".", exist_ok=True)
        with open(filename, "wb") as f:
            pickle.dump(data_to_save, f, protocol=pickle.HIGHEST_PROTOCOL)

        size_mb = os.path.getsize(filename) / (1024 ** 2)
        print(f"ベンチマーク結果を保存: {filename}  ({size_mb:.2f} MB)")
        return filename

    @staticmethod
    def load_benchmark_results(filename: str) -> dict:
        """pickle 形式のベンチマーク結果ファイルを読み込む"""
        import pickle, os
        if not os.path.isfile(filename):
            raise FileNotFoundError(filename)
        with open(filename, "rb") as f:
            data = pickle.load(f)
        print(f"ベンチマーク結果を読み込み: {filename}  (meta: {data.get('meta', {})})")
        return data.get("results", {})

    @staticmethod
    def list_benchmark_files(pattern: str = "*.pkl", directory: str | None = None) -> list[str]:
        """指定ディレクトリ内の pickle ファイル一覧を返す"""
        import glob, os
        directory = directory or os.getcwd()
        files = glob.glob(os.path.join(directory, pattern))
        for f in files:
            print(f" - {f} ({os.path.getsize(f)/(1024**2):.2f} MB)")
        return files

class BGK1DPlot(BGK1D, BGK1DPlotMixin):
    """BGK1D + プロット機能の統合クラス"""
    pass

class BGK1DPlot_old(BGK1D_old, BGK1DPlotMixin):
    """BGK1D_old + プロット機能の統合クラス"""
    pass