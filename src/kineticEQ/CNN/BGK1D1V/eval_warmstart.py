# kineticEQ/src/kineticEQ/CNN/BGK1D1V/eval_warmstart.py
from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict

import numpy as np
import torch

from kineticEQ import Engine, Config, BGK1D
from kineticEQ.core.schemes.BGK1D.bgk1d_utils.bgk1d_compute_moments import calculate_moments

# ------------------------------------------------------------
# Model definition (must match train_moment_cnn.py)
# ここはあなたの train_moment_cnn.py の定義に合わせる必要があります。
# もし train_moment_cnn.py に MomentCNN1D があるなら、それを import して下さい。
# ------------------------------------------------------------
class MomentCNN1D(torch.nn.Module):
    """
    Minimal 1D CNN:
      input : (B, 5, nx)  [n,u,T,logdt,logtau]
      output: (B, 3, nx)  [n,u,T] predicted
    """
    def __init__(self, ch_in: int = 5, ch_hidden: int = 32, ch_out: int = 3, kernel: int = 5):
        super().__init__()
        pad = kernel // 2
        self.net = torch.nn.Sequential(
            torch.nn.Conv1d(ch_in, ch_hidden, kernel_size=kernel, padding=pad),
            torch.nn.GELU(),
            torch.nn.Conv1d(ch_hidden, ch_hidden, kernel_size=kernel, padding=pad),
            torch.nn.GELU(),
            torch.nn.Conv1d(ch_hidden, ch_out, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def load_model(ckpt_path: str, device: torch.device, hidden: int, kernel: int) -> torch.nn.Module:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    # train_moment_cnn.py の保存形式に合わせて読み分け
    if isinstance(ckpt, dict) and "model_state" in ckpt:
        state = ckpt["model_state"]
    else:
        state = ckpt

    model = MomentCNN1D(ch_in=5, ch_hidden=hidden, ch_out=3, kernel=kernel).to(device)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


@torch.no_grad()
def maxwell_1d1v_from_moments(
    n: torch.Tensor, u: torch.Tensor, T: torch.Tensor,
    v: torch.Tensor, inv_sqrt_2pi: torch.Tensor,
    eps: float = 1e-12
) -> torch.Tensor:
    """
    Build Maxwellian f(x,v) from moments.
    n,u,T: (nx,)
    v: (nv,)
    return: (nx, nv)
    """
    n = torch.clamp(n, min=eps)
    T = torch.clamp(T, min=eps)

    vv = v[None, :]          # (1, nv)
    uu = u[:, None]          # (nx, 1)
    TT = T[:, None]          # (nx, 1)

    # f = n / sqrt(2pi T) * exp(-(v-u)^2/(2T))
    f = n[:, None] * inv_sqrt_2pi / torch.sqrt(TT) * torch.exp(-(vv - uu) ** 2 / (2.0 * TT))
    return f


@torch.no_grad()
def make_nn_input(
    n: torch.Tensor, u: torch.Tensor, T: torch.Tensor,
    dt: float, tau_tilde: float
) -> torch.Tensor:
    """
    return x: (1, 5, nx)
    """
    nx = n.numel()
    logdt = torch.full((nx,), float(np.log10(dt)), device=n.device, dtype=n.dtype)
    logtau = torch.full((nx,), float(np.log10(tau_tilde)), device=n.device, dtype=n.dtype)
    x = torch.stack([n, u, T, logdt, logtau], dim=0)[None, :, :]  # (1,5,nx)
    return x


def build_engine(device: torch.device, tau_tilde: float, dt: float, T_total: float, nx: int, nv: int, v_max: float) -> Engine:
    model_cfg = BGK1D.ModelConfig(
        grid=BGK1D.Grid1D1V(nx=nx, nv=nv, Lx=1.0, v_max=v_max),
        time=BGK1D.TimeConfig(dt=dt, T_total=T_total),
        params=BGK1D.BGK1D1VParams(tau_tilde=float(tau_tilde)),
        scheme_params=BGK1D.implicit.Params(picard_iter=10_000, picard_tol=1e-6, abs_tol=1e-13),
        initial=BGK1D.InitialCondition1D(initial_regions=(
            {"x_range": (0.0, 0.5), "n": 1.0,   "u": 0.0, "T": 1.0},
            {"x_range": (0.5, 1.0), "n": 0.125, "u": 0.0, "T": 0.8},
        ))
    )

    eng = Engine(Config(
        model="BGK1D1V",
        scheme="implicit",
        backend="cuda_kernel",
        model_cfg=model_cfg,
        log_level="err",
        use_tqdm=False
    ))

    # device に配置（Engine 実装に依存。もし Engine 内で device 指定が必要なら合わせて変更）
    # 既に cfg に従って cuda になる想定だが、念のため state を確認すること
    return eng


def run_case_baseline(eng: Engine, n_steps: int) -> dict:
    iters = []
    t0 = time.perf_counter()
    for s in range(n_steps):
        eng.stepper(s)
        bench = getattr(eng.stepper, "benchlog", None) or {}
        iters.append(int(bench.get("picard_iter", -1)))
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    t1 = time.perf_counter()
    return {"picard_iter": iters, "walltime_sec": t1 - t0}


def run_case_warmstart(eng: Engine, model: torch.nn.Module, n_steps: int) -> dict:
    iters = []
    dt = float(cfg.model_cfg.time.dt)
    tau = float(cfg.model_cfg.params.tau_tilde)

    t0 = time.perf_counter()
    for s in range(n_steps):
        # current moments
        n, u, T = calculate_moments(eng.state, eng.state.f)

        # NN predict next moments (as initial guess)
        x = make_nn_input(n, u, T, dt=dt, tau_tilde=tau)
        y = model(x)  # (1,3,nx)
        n1 = y[0, 0, :]
        u1 = y[0, 1, :]
        T1 = y[0, 2, :]

        # Safety clamps (avoid negative n/T)
        n1 = torch.clamp(n1, min=1e-10)
        T1 = torch.clamp(T1, min=1e-10)

        # Maxwell -> fz0
        fz0 = maxwell_1d1v_from_moments(n1, u1, T1, eng.state.v, eng.state.inv_sqrt_2pi)

        # Inject as initial guess
        eng.stepper.ws._init_fz = fz0  # (nx,nv)

        # Step
        eng.stepper(s)
        bench = getattr(eng.stepper, "benchlog", None) or {}
        iters.append(int(bench.get("picard_iter", -1)))

    torch.cuda.synchronize() if torch.cuda.is_available() else None
    t1 = time.perf_counter()
    return {"picard_iter": iters, "walltime_sec": t1 - t0}


def summarize(iters: list[int]) -> dict:
    a = np.array(iters, dtype=np.int64)
    return {
        "n_steps": int(a.size),
        "sum": int(a.sum()),
        "mean": float(a.mean()),
        "p50": float(np.percentile(a, 50)),
        "p90": float(np.percentile(a, 90)),
        "p99": float(np.percentile(a, 99)),
        "min": int(a.min()),
        "max": int(a.max()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True, help="path to best.pt or last.pt")
    ap.add_argument("--out", type=str, default="warmstart_eval.json", help="output json")
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--tau", type=float, required=True)
    ap.add_argument("--dt", type=float, default=5e-5)
    ap.add_argument("--T_total", type=float, default=0.05)
    ap.add_argument("--n_steps", type=int, default=200, help="number of steps to evaluate")
    ap.add_argument("--nx", type=int, default=512)
    ap.add_argument("--nv", type=int, default=256)
    ap.add_argument("--v_max", type=float, default=10.0)
    ap.add_argument("--hidden", type=int, default=32)
    ap.add_argument("--kernel", type=int, default=5)
    args = ap.parse_args()

    device = torch.device(args.device)
    assert device.type == "cuda", "implicit cuda_kernel backend requires CUDA device"

    # Build baseline engine
    eng0 = build_engine(device, tau_tilde=args.tau, dt=args.dt, T_total=args.T_total, nx=args.nx, nv=args.nv, v_max=args.v_max)

    # Build warm-start engine (same settings, separate state)
    eng1 = build_engine(device, tau_tilde=args.tau, dt=args.dt, T_total=args.T_total, nx=args.nx, nv=args.nv, v_max=args.v_max)

    # Load model
    model = load_model(args.ckpt, device=device, hidden=args.hidden, kernel=args.kernel)

    # Run
    base = run_case_baseline(eng0, n_steps=args.n_steps)
    warm = run_case_warmstart(eng1, cfg=cfg1, model=model, n_steps=args.n_steps)

    base_sum = summarize(base["picard_iter"])
    warm_sum = summarize(warm["picard_iter"])

    speed_iter = (base_sum["sum"] / max(warm_sum["sum"], 1e-12))
    speed_time = (base["walltime_sec"] / max(warm["walltime_sec"], 1e-12))

    result = {
        "meta": {
            "ckpt": args.ckpt,
            "tau_tilde": args.tau,
            "dt": args.dt,
            "T_total": args.T_total,
            "nx": args.nx,
            "nv": args.nv,
            "v_max": args.v_max,
            "n_steps_eval": args.n_steps,
            "model_hidden": args.hidden,
            "model_kernel": args.kernel,
        },
        "baseline": {
            "summary": base_sum,
            "walltime_sec": float(base["walltime_sec"]),
        },
        "warmstart": {
            "summary": warm_sum,
            "walltime_sec": float(warm["walltime_sec"]),
        },
        "speedup": {
            "iter_sum_ratio": float(speed_iter),
            "walltime_ratio": float(speed_time),
        }
    }

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
