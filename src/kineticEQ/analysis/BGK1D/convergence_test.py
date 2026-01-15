# kineticEQ/src/kineticEQ/analysis/BGK1D/convergence_test.py
from __future__ import annotations

from dataclasses import replace
from typing import Any, Dict, List, Optional

import logging
import platform
import subprocess
import time

import torch

from kineticEQ import BGK1D, Config, Engine

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


def run_convergence_test(
    *,
    tau_tilde_list: List[float] = (5e-4, 5e-5, 5e-6, 5e-7),
    nx: int = 1000,
    nv: int = 200,
    Lx: float = 1.0,
    v_max: float = 10.0,
    dt: float = 5e-5,
    T_total: float = 0.05,
    device: str = "cuda",
    backend: str = "cuda_kernel",
    dtype: str = "float64",
    log_level: str = "info",
    # Picard (implicit)
    imp_params: Any = None,
    # HOLO
    holo_params: Any = None,
    # 記録密度：100 なら 1% 間隔
    sample_percent: int = 100,
) -> Dict[str, Any]:
    """
    HOLO 反復 vs Implicit(Picard) 反復の収束性を比較する。
    出力形式は meta + records の新形式。

    records には、指定間隔で
      - step, time, walltime
      - stepper.benchlog（実装が付与している反復回数・残差等）
    を格納する。
    """

    if imp_params is None:
        imp_params = BGK1D.implicit.Params(picard_tol=1e-6, picard_iter=10_000)
    if holo_params is None:
        holo_params = BGK1D.holo.Params(ho_tol=1e-6, ho_iter=10_000, lo_tol=1e-6, lo_iter=10_000)

    n_steps = int(BGK1D.TimeConfig(dt=dt, T_total=T_total).n_steps)
    sample_interval = max(1, n_steps // max(1, sample_percent))

    out: Dict[str, Any] = {
        "meta": {
            "test_type": "convergence",
            "nx": int(nx),
            "nv": int(nv),
            "Lx": float(Lx),
            "v_max": float(v_max),
            "dt": float(dt),
            "T_total": float(T_total),
            "n_steps": int(n_steps),
            "tau_tilde_list": [float(t) for t in tau_tilde_list],
            "device": str(device),
            "backend": str(backend),
            "dtype": str(dtype),
            "log_level": str(log_level),
            "device_name": _get_device_name(str(device)),
            "cpu_name": _get_cpu_name(),
            "sample_interval": int(sample_interval),
            "sample_percent": int(sample_percent),
        },
        "records": [],
    }

    def _run_one_scheme(*, tau_tilde: float, scheme: str, scheme_params: Any) -> None:
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
                scheme_params=scheme_params,
            ),
        )

        engine = Engine(cfg, apply_logging_flag=True)

        is_cuda = (device == "cuda") and torch.cuda.is_available()

        for step in range(n_steps):
            if is_cuda:
                torch.cuda.synchronize()
            t0 = time.perf_counter()

            engine.stepper(step)

            if is_cuda:
                torch.cuda.synchronize()
            wall = time.perf_counter() - t0

            if (step % sample_interval == 0) or (step == n_steps - 1):
                benchlog = getattr(engine.stepper, "benchlog", None)
                out["records"].append(
                    {
                        "scheme": scheme,
                        "tau_tilde": float(tau_tilde),
                        "step": int(step),
                        "time": float(step * dt),
                        "walltime_sec": float(wall),
                        "benchlog": benchlog if isinstance(benchlog, dict) else None,
                    }
                )

    logger.info("--- Convergence Test Start ---")
    for tau in tau_tilde_list:
        logger.info(f"tau_tilde={tau}")
        _run_one_scheme(tau_tilde=tau, scheme="holo", scheme_params=holo_params)
        _run_one_scheme(tau_tilde=tau, scheme="implicit", scheme_params=imp_params)

    logger.info("--- Convergence Test Completed ---")
    return out
