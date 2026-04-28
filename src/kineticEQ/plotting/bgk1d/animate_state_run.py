from __future__ import annotations

import io
import logging
from pathlib import Path
from typing import Any

import torch

from kineticEQ.core.schemes.BGK1D1V.bgk1d_utils.general.bgk1d_compute_moments import (
    calculate_moments,
)
from kineticEQ.utillib.progress_bar import get_progress_bar

logger = logging.getLogger(__name__)


def _as_engine(engine_or_cfg: Any):
    from kineticEQ.api.engine import Engine

    if isinstance(engine_or_cfg, Engine):
        return engine_or_cfg
    return Engine(engine_or_cfg)


def _capture_steps(n_steps: int, frames: int) -> list[int]:
    import numpy as np

    if frames < 1:
        raise ValueError("frames must be >= 1")
    if n_steps < 0:
        raise ValueError("n_steps must be >= 0")
    return sorted({int(round(x)) for x in np.linspace(0, n_steps, int(frames))})


def _physical_slice(state, diagnostic_cells: str):
    mode = str(diagnostic_cells).strip().lower()
    if mode == "all":
        return slice(None)
    if mode == "interior":
        if state.f.shape[0] <= 2:
            return slice(None)
        return slice(1, -1)
    raise ValueError("diagnostic_cells must be 'interior' or 'all'")


@torch.no_grad()
def _diagnostics(state, step: int, time: float, diagnostic_cells: str) -> dict[str, float]:
    sl = _physical_slice(state, diagnostic_cells)
    f = state.f[sl, :]
    v = state.v
    dx = float(state.dx)
    dv = float(state.dv)

    mass = f.sum() * dv * dx
    momentum = (f * v[None, :]).sum() * dv * dx
    total_energy = 0.5 * (f * (v[None, :] ** 2)).sum() * dv * dx

    n = f.sum(dim=1) * dv
    nu = (f * v[None, :]).sum(dim=1) * dv
    u = nu / torch.clamp(n, min=torch.finfo(f.dtype).tiny)
    macro_energy = 0.5 * (n * u * u).sum() * dx
    internal_energy = total_energy - macro_energy

    return {
        "step": int(step),
        "time": float(time),
        "mass": float(mass.item()),
        "momentum": float(momentum.item()),
        "energy": float(total_energy.item()),
        "internal_energy": float(internal_energy.item()),
    }


@torch.no_grad()
def _snapshot(state, step: int, time: float) -> dict[str, Any]:
    n, u, T = calculate_moments(state, state.f)
    return {
        "step": int(step),
        "time": float(time),
        "x": state.x.detach().cpu().numpy().copy(),
        "v": state.v.detach().cpu().numpy().copy(),
        "f": state.f.detach().cpu().float().numpy().copy(),
        "n": n.detach().cpu().float().numpy().copy(),
        "u": u.detach().cpu().float().numpy().copy(),
        "T": T.detach().cpu().float().numpy().copy(),
    }


def _limits(values: list[Any], pad: float = 0.05) -> tuple[float, float]:
    import numpy as np

    lo = min(float(np.nanmin(v)) for v in values)
    hi = max(float(np.nanmax(v)) for v in values)
    if not np.isfinite(lo) or not np.isfinite(hi):
        return 0.0, 1.0
    if hi == lo:
        delta = abs(hi) * pad if hi != 0.0 else 1.0
        return lo - delta, hi + delta
    delta = (hi - lo) * pad
    return lo - delta, hi + delta


def _render_frame(snapshot: dict[str, Any], limits: dict[str, tuple[float, float]], dpi: int):
    from PIL import Image
    import matplotlib.pyplot as plt

    x = snapshot["x"]
    v = snapshot["v"]
    f = snapshot["f"]
    n = snapshot["n"]
    u = snapshot["u"]
    T = snapshot["T"]

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f"step={snapshot['step']}  t={snapshot['time']:.6g}", fontsize=13)

    im1 = ax1.imshow(
        f.T,
        aspect="auto",
        origin="lower",
        extent=[x[0], x[-1], v[0], v[-1]],
        cmap="viridis",
        vmin=limits["f"][0],
        vmax=limits["f"][1],
    )
    ax1.set_xlabel("Position x")
    ax1.set_ylabel("Velocity v")
    ax1.set_title("Distribution Function f(x,v)")
    fig.colorbar(im1, ax=ax1)

    ax2.plot(x, n, "b-", linewidth=2)
    ax2.set_xlabel("Position x")
    ax2.set_ylabel("Density n")
    ax2.set_title("Density Distribution")
    ax2.set_ylim(*limits["n"])
    ax2.grid(True, alpha=0.3)

    ax3.plot(x, u, "r-", linewidth=2)
    ax3.set_xlabel("Position x")
    ax3.set_ylabel("Mean Velocity u")
    ax3.set_title("Velocity Distribution")
    ax3.set_ylim(*limits["u"])
    ax3.grid(True, alpha=0.3)

    ax4.plot(x, T, "g-", linewidth=2)
    ax4.set_xlabel("Position x")
    ax4.set_ylabel("Temperature T")
    ax4.set_title("Temperature Distribution")
    ax4.set_ylim(*limits["T"])
    ax4.grid(True, alpha=0.3)

    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi)
    plt.close(fig)
    buf.seek(0)
    img = Image.open(buf)
    frame = img.copy()
    img.close()
    buf.close()
    return frame


def _save_gif(
    snapshots: list[dict[str, Any]],
    path: Path,
    *,
    fps: int,
    dpi: int,
) -> None:
    if not snapshots:
        raise ValueError("no snapshots captured")

    limits = {
        "f": _limits([s["f"] for s in snapshots], pad=0.0),
        "n": _limits([s["n"] for s in snapshots]),
        "u": _limits([s["u"] for s in snapshots]),
        "T": _limits([s["T"] for s in snapshots]),
    }

    frames = [_render_frame(s, limits, dpi=dpi) for s in snapshots]
    duration_ms = int(1000 / max(int(fps), 1))
    frames[0].save(
        path,
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=0,
    )
    for frame in frames:
        frame.close()


def _save_conservation_plot(records: list[dict[str, float]], path: Path, *, dpi: int) -> None:
    import matplotlib.pyplot as plt

    if not records:
        raise ValueError("no diagnostics recorded")

    steps = [r["step"] for r in records]
    mass = [r["mass"] for r in records]
    momentum = [r["momentum"] for r in records]
    energy = [r["energy"] for r in records]

    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    series = (
        ("Mass", mass, "tab:blue"),
        ("Momentum", momentum, "tab:red"),
        ("Energy", energy, "tab:green"),
    )
    for ax, (name, y, color) in zip(axes, series, strict=True):
        ax.plot(steps, y, color=color, linewidth=1.5)
        ax.set_ylabel(name)
        ax.grid(True, alpha=0.3)
        if y:
            y0 = y[0]
            y1 = y[-1]
            ax.set_title(f"{name}: initial={y0:.8e}, final={y1:.8e}, delta={y1 - y0:.3e}")
    axes[-1].set_xlabel("Step")
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def animate_state_run(
    engine_or_cfg: Any,
    *,
    frames: int = 120,
    filename: str = "bgk_state.gif",
    output_dir: str | Path | None = None,
    conservation_filename: str | None = None,
    fps: int = 10,
    dpi: int = 90,
    diagnostics_every: int = 1,
    diagnostic_cells: str = "interior",
    use_tqdm: bool | None = None,
) -> dict[str, Any]:
    """
    Run a BGK1D1V engine and save a state animation plus conservation diagnostics.

    The passed Engine is advanced in-place. If a Config is passed instead, a new Engine
    is created internally. Diagnostics default to interior cells because reflective
    boundary rows are used as boundary traces rather than physical control volumes.
    """
    from PIL import Image  # noqa: F401

    engine = _as_engine(engine_or_cfg)
    cfg = engine.config
    n_steps = int(cfg.model_cfg.time.n_steps)
    dt = float(cfg.model_cfg.time.dt)
    diagnostics_every = max(int(diagnostics_every), 1)

    if output_dir is None:
        output_dir = Path.cwd() / "result"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    gif_path = output_dir / filename
    if conservation_filename is None:
        conservation_filename = f"{Path(filename).stem}_conservation.png"
    conservation_path = output_dir / conservation_filename

    capture_set = set(_capture_steps(n_steps, int(frames)))
    snapshots: list[dict[str, Any]] = []
    diagnostics: list[dict[str, float]] = []

    def record(step_count: int) -> None:
        diagnostics.append(
            _diagnostics(
                engine.state,
                step=step_count,
                time=step_count * dt,
                diagnostic_cells=diagnostic_cells,
            )
        )

    record(0)
    if 0 in capture_set:
        snapshots.append(_snapshot(engine.state, step=0, time=0.0))

    if use_tqdm is None:
        use_tqdm = bool(cfg.use_tqdm_bool)

    with get_progress_bar(
        use_tqdm,
        total=n_steps,
        desc="Animate State Run",
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
    ) as pbar:
        for step in range(n_steps):
            engine.stepper(step)
            step_count = step + 1

            if step_count % diagnostics_every == 0 or step_count == n_steps:
                record(step_count)

            if step_count in capture_set:
                snapshots.append(_snapshot(engine.state, step=step_count, time=step_count * dt))

            pbar.update(1)

    _save_gif(snapshots, gif_path, fps=fps, dpi=dpi)
    _save_conservation_plot(diagnostics, conservation_path, dpi=dpi)

    logger.info("Saved state animation: %s", gif_path)
    logger.info("Saved conservation plot: %s", conservation_path)

    return {
        "gif": str(gif_path),
        "conservation": str(conservation_path),
        "frames": len(snapshots),
        "diagnostics": diagnostics,
    }
