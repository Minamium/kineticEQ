# kineticEQ/src/kineticEQ/plotting/bgk1d/plot_state.py
import logging
import os
from kineticEQ.core.schemes.BGK1D.bgk1d_utils.bgk1d_compute_moments import calculate_moments
logger = logging.getLogger(__name__)

#状態可視化メソッド
def plot_state(state, filename="bgk_state.png", output_dir=None):
    """状態可視化
    params:
        state: State1D1V
        filename: str
        output_dir: str
    return:
        None
    分布関数、密度分布、速度分布、温度分布を可視化する
    """
    if output_dir is None:
        output_dir = Path.cwd() / "result"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    import matplotlib.pyplot as plt
    # CPUに転送（matplotlib用）
    f_cpu = state.f.cpu().numpy()
    x_cpu = state.x.cpu().numpy()
    v_cpu = state.v.cpu().numpy()

    # モーメント計算
    n, u, T = calculate_moments(state, state.f)
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
    plt.savefig(output_dir / filename)

    # 統計情報表示
    logger.info(f"Density: mean={n_cpu.mean():.4f}, min={n_cpu.min():.4f}, max={n_cpu.max():.4f}")
    logger.info(f"Velocity: mean={u_cpu.mean():.4f}, min={u_cpu.min():.4f}, max={u_cpu.max():.4f}")
    logger.info(f"Temperature: mean={T_cpu.mean():.4f}, min={T_cpu.min():.4f}, max={T_cpu.max():.4f}")