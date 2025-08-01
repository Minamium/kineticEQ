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

# 可視化関数群
class BGK1DPlotMixin:
    """可視化, 解析用の関数群"""
    #状態可視化メソッド
    def plot_state(self):
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

        plt.tight_layout()
        plt.show()

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

                plt.tight_layout()

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
                             save_fig: bool = True,
                             show_gpu_time: bool = True,
                             show_overhead: bool = True):
        """実行時間ベンチマーク結果を可視化
        
        Parameters
        ----------
        bench_results : dict | None
            ベンチマーク結果辞書。Noneの場合はself.benchmark_resultsを使用
        filename : str | None
            pklファイルから読み込む場合のファイル名
        save_fig : bool
            図を保存するかどうか
        show_gpu_time : bool
            GPU時間も表示するかどうか（利用可能な場合）
        show_overhead : bool
            オーバーヘッドΔTも表示するかどうか（利用可能な場合）
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

        fig_width = 4 * subplot_num
        fig, axes = plt.subplots(1, subplot_num, figsize=(fig_width, 6))
        
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
        
        # 値をテキストで表示
        for i in range(len(nx_sorted)):
            for j in range(len(nv_sorted)):
                text = axes[0].text(j, i, f'{cpu_step_time_matrix[i, j]:.2f}',
                                  ha="center", va="center", color="white", fontsize=8)
        
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
                    axes[ax_idx].text(j, i, f'{gpu_step_time_matrix[i, j]:.2f}',
                                      ha="center", va="center", color="white", fontsize=8)
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
                    axes[ax_idx].text(j, i, f'{diff_matrix[i, j]:.2f}',
                                      ha="center", va="center", color="white", fontsize=8)
            plt.colorbar(im3, ax=axes[ax_idx], label='ΔT (ms)')
        
        plt.tight_layout()
        
        if save_fig:
            base_name = 'timing_benchmark_heatmap'
            plt.savefig(base_name + '.png', dpi=300, bbox_inches='tight')
            print(f"Figure saved: {base_name + '.png'}")

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
        plt.tight_layout()
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

        plt.tight_layout()
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