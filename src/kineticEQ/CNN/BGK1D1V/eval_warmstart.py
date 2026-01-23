# kineticEQ/src/kineticEQ/CNN/BGK1D1V/eval_warmstart.py
from __future__ import annotations

import os
import json
import time
import argparse
from dataclasses import asdict, dataclass

import torch
import numpy as np

from kineticEQ import Engine, Config, BGK1D
from kineticEQ.core.schemes.BGK1D.bgk1d_utils.bgk1d_compute_moments import calculate_moments


# -------------------------
# Model (single-file minimal)
# ※本来は models.py に移す前提。今回は eval 単体で完結する最小版。
# -------------------------
class MomentCNN1D(torch.nn.Module):
    def __init__(self, in_ch: int = 5, out_ch: int = 3, hidden: int = 64, kernel: int = 5):
        super().__init__()
        pad = kernel // 2
        self.net = torch.nn.Sequential(
            torch.nn.Conv1d(in_ch, hidden, kernel_size=kernel, padding=pad),
            torch.nn.SiLU(),
            torch.nn.Conv1d(hidden, hidden, kernel_size=kernel, padding=pad),
            torch.nn.SiLU(),
            torch.nn.Conv1d(hidden, out_ch, kernel_size=1),
        )

    def forward(self, x):
        return self.net(x)


def infer_model_shape_from_ckpt(state_dict: dict) -> tuple[int, int, int, int]:
    """
    state_dict から (in_ch, out_ch, hidden, kernel) を推定
    期待：net.0.weight shape = (hidden, in_ch, kernel)
          net.4.weight shape = (out_ch, hidden, 1)
    """
    w0 = state_dict["net.0.weight"]
    hidden = int(w0.shape[0])
    in_ch = int(w0.shape[1])
    kernel = int(w0.shape[2])
    w_last = state_dict["net.4.weight"]
    out_ch = int(w_last.shape[0])
    return in_ch, out_ch, hidden, kernel


def load_ckpt(ckpt_path: str, device: torch.device) -> MomentCNN1D:
    ckpt = torch.load(ckpt_path, map_location="cpu")

    # train_moment_cnn.py は {"model": state_dict, ...} 形式
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    in_ch, out_ch, hidden, kernel = infer_model_shape_from_ckpt(state)

    model = MomentCNN1D(in_ch=in_ch, out_ch=out_ch, hidden=hidden, kernel=kernel).to(device)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


# -------------------------
# Physics helpers
# -------------------------
@torch.no_grad()
def maxwellian_from_moments(v: torch.Tensor, inv_sqrt_2pi: torch.Tensor,
                            n: torch.Tensor, u: torch.Tensor, T: torch.Tensor) -> torch.Tensor:
    """
    v: (nv,) , n/u/T: (nx,)
    return: fM (nx, nv)
    """
    # 安全側（負は物理的に破綻するので軽くクランプ）
    eps = 1e-12
    n = torch.clamp(n, min=eps)
    T = torch.clamp(T, min=eps)

    # coeff: (nx,)
    coeff = (n * inv_sqrt_2pi) / torch.sqrt(T)
    # diff: (nx,nv)
    diff = v[None, :] - u[:, None]
    exponent = -(diff * diff) / (2.0 * T[:, None])
    fM = coeff[:, None] * torch.exp(exponent)
    return fM


@torch.no_grad()
def predict_next_moments(model: MomentCNN1D,
                         n: torch.Tensor, u: torch.Tensor, T: torch.Tensor,
                         dt: float, tau_tilde: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    入力: (nx,) を 1D CNN の (B,C,L) に詰める
    入力チャネル: [n,u,T,log10(dt),log10(tau_tilde)]
    出力: (3,nx) -> (nx,)
    """
    nx = n.shape[0]
    x = torch.empty((1, 5, nx), device=n.device, dtype=n.dtype)
    x[0, 0, :] = n
    x[0, 1, :] = u
    x[0, 2, :] = T
    x[0, 3, :].fill_(np.log10(dt))
    x[0, 4, :].fill_(np.log10(tau_tilde))

    y = model(x)[0]  # (3,nx)
    n1 = y[0]
    u1 = y[1]
    T1 = y[2]

    # ここはあなたの方針に合わせて：softplus/exp などでも良い
    # 最小安全策：n,T は正に寄せる
    n1 = torch.nn.functional.softplus(n1) + 1e-12
    T1 = torch.nn.functional.softplus(T1) + 1e-12
    return n1, u1, T1


# -------------------------
# Runner
# -------------------------
def build_engine(device: torch.device, tau: float, dt: float, T_total: float, nx: int, nv: int,
                 init_regions) -> Engine:
    model_cfg = BGK1D.ModelConfig(
        grid=BGK1D.Grid1D1V(nx=nx, nv=nv, Lx=1.0, v_max=10.0),
        time=BGK1D.TimeConfig(dt=dt, T_total=T_total),
        params=BGK1D.BGK1D1VParams(tau_tilde=float(tau)),
        scheme_params=BGK1D.implicit.Params(picard_iter=1_000, picard_tol=1e-6, abs_tol=1e-13),
        initial=BGK1D.InitialCondition1D(initial_regions=init_regions),
    )
    eng = Engine(Config(
        model="BGK1D1V",
        scheme="implicit",
        backend="cuda_kernel",
        model_cfg=model_cfg,
        log_level="err",
        use_tqdm=False,
        device=str(device),
    ))
    return eng


@torch.no_grad()
def run_baseline(eng: Engine, n_steps: int) -> dict:
    stepper = eng.stepper
    ws = getattr(stepper, "ws", None)

    iters = []
    times = []

    for s in range(n_steps):
        if ws is not None and hasattr(ws, "_init_fz"):
            ws._init_fz = None  # baseline: 前ステップから開始

        t0 = time.time()
        stepper(s)
        t1 = time.time()

        bench = getattr(stepper, "benchlog", None) or {}
        iters.append(int(bench.get("picard_iter", -1)))
        times.append(t1 - t0)

    n, u, T = calculate_moments(eng.state, eng.state.f)
    return {
        "picard_iter": iters,
        "step_time_sec": times,
        "final_moments": {
            "n": n.detach().cpu().numpy(),
            "u": u.detach().cpu().numpy(),
            "T": T.detach().cpu().numpy(),
        }
    }


@torch.no_grad()
def run_warmstart(eng: Engine, model: MomentCNN1D, n_steps: int) -> dict:
    stepper = eng.stepper
    ws = getattr(stepper, "ws", None)
    if ws is None:
        raise RuntimeError("stepper.ws が見つかりません。build_stepper で _stepper.ws = ws を付与してください。")

    dt = float(eng.config.model_cfg.time.dt)
    tau_tilde = float(eng.config.model_cfg.params.tau_tilde)

    iters = []
    times = []

    for s in range(n_steps):
        # 現在の f からモーメント
        n0, u0, T0 = calculate_moments(eng.state, eng.state.f)

        # NN で次ステップのモーメント予測
        n1, u1, T1 = predict_next_moments(model, n0, u0, T0, dt=dt, tau_tilde=tau_tilde)

        # 予測モーメントから Maxwellian を構築し、初期候補 fz とする
        f_init = maxwellian_from_moments(
            v=eng.state.v, inv_sqrt_2pi=eng.state.inv_sqrt_2pi,
            n=n1, u=u1, T=T1
        ).to(eng.state.f.dtype)

        ws._init_fz = f_init  # implicit kernel が参照

        t0 = time.time()
        stepper(s)
        t1 = time.time()

        bench = getattr(stepper, "benchlog", None) or {}
        iters.append(int(bench.get("picard_iter", -1)))
        times.append(t1 - t0)

    n, u, T = calculate_moments(eng.state, eng.state.f)
    return {
        "picard_iter": iters,
        "step_time_sec": times,
        "final_moments": {
            "n": n.detach().cpu().numpy(),
            "u": u.detach().cpu().numpy(),
            "T": T.detach().cpu().numpy(),
        }
    }


def moments_diff(a: dict, b: dict) -> dict:
    out = {}
    for k in ["n", "u", "T"]:
        x = a["final_moments"][k]
        y = b["final_moments"][k]
        diff = np.abs(x - y)
        out[k] = {
            "linf": float(diff.max()),
            "l2": float(np.sqrt((diff * diff).mean())),
            "l1": float(diff.mean()),
        }
    return out


def summarize_iters(iters: list[int]) -> dict:
    it = np.array(iters, dtype=np.float64)
    return {
        "mean": float(it.mean()),
        "min": float(it.min()),
        "max": float(it.max()),
        "p50": float(np.percentile(it, 50)),
        "p90": float(np.percentile(it, 90)),
        "p99": float(np.percentile(it, 99)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="best.pt / last.pt path")
    ap.add_argument("--out", required=True, help="output json path")
    ap.add_argument("--tau", type=float, required=True)
    ap.add_argument("--dt", type=float, default=5e-5)
    ap.add_argument("--T_total", type=float, default=0.01)
    ap.add_argument("--nx", type=int, default=512)
    ap.add_argument("--nv", type=int, default=256)
    ap.add_argument("--n_steps", type=int, default=None, help="override steps; default = T_total/dt")
    ap.add_argument("--device", type=str, default="cuda")
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # ここは最小テストとして固定（必要なら CLI 化）
    init_regions = (
        {"x_range": (0.0, 0.5), "n": 1.0,   "u": 0.0, "T": 1.0},
        {"x_range": (0.5, 1.0), "n": 0.125, "u": 0.0, "T": 0.8},
    )

    model = load_ckpt(args.ckpt, device=device)

    eng0 = build_engine(device=device, tau=args.tau, dt=args.dt, T_total=args.T_total,
                        nx=args.nx, nv=args.nv, init_regions=init_regions)
    eng1 = build_engine(device=device, tau=args.tau, dt=args.dt, T_total=args.T_total,
                        nx=args.nx, nv=args.nv, init_regions=init_regions)

    n_steps = args.n_steps
    if n_steps is None:
        n_steps = int(round(args.T_total / args.dt))

    base = run_baseline(eng0, n_steps=n_steps)
    warm = run_warmstart(eng1, model=model, n_steps=n_steps)

    result = {
        "meta": {
            "tau_tilde": args.tau,
            "dt": args.dt,
            "T_total": args.T_total,
            "nx": args.nx,
            "nv": args.nv,
            "n_steps": n_steps,
            "ckpt": args.ckpt,
            "device": str(device),
        },
        "baseline": {
            "picard_iter_summary": summarize_iters(base["picard_iter"]),
            "mean_step_time_sec": float(np.mean(base["step_time_sec"])),
        },
        "warmstart": {
            "picard_iter_summary": summarize_iters(warm["picard_iter"]),
            "mean_step_time_sec": float(np.mean(warm["step_time_sec"])),
        },
        "final_moment_diff": moments_diff(base, warm),
        "raw": {
            "baseline_picard_iter": base["picard_iter"],
            "warmstart_picard_iter": warm["picard_iter"],
        }
    }

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)

    print("[OK] wrote:", args.out)
    print("baseline iter:", result["baseline"]["picard_iter_summary"])
    print("warmstart iter:", result["warmstart"]["picard_iter_summary"])
    print("final moment diff:", result["final_moment_diff"])


if __name__ == "__main__":
    main()
