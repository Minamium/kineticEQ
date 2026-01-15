# kineticEQ/src/kineticEQ/analysis/BGK1D/scheme_comparison.py
from __future__ import annotations

from typing import Any, Dict, List, Optional
import logging
import platform
import subprocess
import time

import torch

from kineticEQ import BGK1D, Config, Engine
from kineticEQ.core.schemes.BGK1D.bgk1d_utils.bgk1d_compute_moments import calculate_moments

logger = logging.getLogger("kineticEQ")


def _get_cpu_name() -> str:
    if platform.system() == "Darwin":
        try:
            out = subprocess.check_output(
                ["sysctl", "-n", "machdep.cpu.brand_string"], text=True
            ).strip()
            if out:
                return out
        except Exception:
            pass

    if platform.system() == "Linux":
        try:
            with open("/proc/cpuinfo", "r") as f:
                for line in f:
                    if "model name" in line:
                        return line.split(":", 1)[1].strip()
        except Exception:
            pass

    p = platform.processor()
    return p if p else f"{platform.machine()} CPU"


def _get_device_name(device: str) -> str:
    if device == "cuda" and torch.cuda.is_available():
        try:
            return torch.cuda.get_device_name(0)
        except Exception:
            return "CUDA GPU"
    return device


def run_scheme_comparison_test(
    *,
    scheme_list: List[str] = ("explicit", "holo", "implicit"),
    scheme_dt_list: List[float] = (5e-7, 5e-4, 5e-7),
    nx: int = 1000,
    nv: int = 200,
    Lx: float = 1.0,
    v_max: float = 10.0,
    T_total: float = 0.05,
    tau_tilde: float = 5e-6,
    device: str = "cuda",
    backend: str = "cuda_kernel",
    dtype: str = "float64",
    log_level: str = "info",
    # scheme params（Noneなら default_scheme_params が Engine 側で入る）
    explicit_params: Any = None,
    implicit_params: Any = None,
    holo_params: Any = None,
) -> Dict[str, Any]:
    """
    スキーム別比較（explicit / implicit / holo）を同一問題設定で実行し、
    最終時刻の moments (n,u,T) を集める。

    レガシー run_scheme_comparison_test 相当の「Engine版」。
    """
    if len(scheme_list) != len(scheme_dt_list):
        raise ValueError("scheme_list と scheme_dt_list の長さが一致していません")

    out: Dict[str, Any] = {
        "meta": {
            "test_type": "scheme_comparison",
            "scheme_list": [str(s) for s in scheme_list],
            "scheme_dt_list": [float(dt) for dt in scheme_dt_list],
            "nx": int(nx),
            "nv": int(nv),
            "Lx": float(Lx),
            "v_max": float(v_max),
            "T_total": float(T_total),
            "tau_tilde": float(tau_tilde),
            "device": str(device),
            "backend": str(backend),
            "dtype": str(dtype),
            "log_level": str(log_level),
            "device_name": _get_device_name(str(device)),
            "cpu_name": _get_cpu_name(),
        },
        "records": [],
    }

    # scheme_params を scheme 名で引けるように（Noneは許容）
    sp_map = {
        "explicit": explicit_params,
        "implicit": implicit_params,
        "holo": holo_params,
    }

    is_cuda = (device == "cuda") and torch.cuda.is_available()

    for scheme, dt in zip(scheme_list, scheme_dt_list):
        scheme = str(scheme)
        dt = float(dt)

        cfg = Config(
            model="BGK1D1V",
            scheme=scheme,
            backend=backend,
            device=device,
            dtype=dtype,
            use_tqdm="false",
            log_level=log_level,
            model_cfg=BGK1D.ModelConfig(
                grid=BGK1D.Grid1D1V(nx=nx, nv=nv, Lx=Lx, v_max=v_max),
                time=BGK1D.TimeConfig(dt=dt, T_total=T_total),
                params=BGK1D.BGK1D1VParams(tau_tilde=tau_tilde),
                scheme_params=sp_map.get(scheme, None),
            ),
        )

        engine = Engine(cfg, apply_logging_flag=True)

        n_steps = int(engine.config.model_cfg.time.n_steps)

        if is_cuda:
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        # progress bar を使わず手動ステップ
        for step in range(n_steps):
            engine.stepper(step)

        if is_cuda:
            torch.cuda.synchronize()
        wall_total = time.perf_counter() - t0

        # 最終 moments
        n, u, T = calculate_moments(engine.state, engine.state.f)

        # CPUへ（plot/保存/比較の安定のため）
        n_cpu = n.detach().cpu().numpy().copy()
        u_cpu = u.detach().cpu().numpy().copy()
        T_cpu = T.detach().cpu().numpy().copy()

        out["records"].append(
            {
                "scheme": scheme,
                "dt": dt,
                "n": n_cpu,
                "u": u_cpu,
                "T": T_cpu,
                "walltime_total_sec": float(wall_total),
                "n_steps": int(n_steps),
                # 参考：最終ステップ時点の benchlog（存在すれば）
                "benchlog_last": getattr(engine.stepper, "benchlog", None),
            }
        )

    return out
