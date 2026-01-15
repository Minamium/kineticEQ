# kineticEQ/analysis/BGK1D/benchmark.py
from kineticEQ import Config, Engine, BGK1D
from .utils.snapshot import snapshot_from_engine
from .utils.compute_err import append_errors
from .utils.swap_grid_params import with_grid
from dataclasses import replace
import torch
import time
import platform
import subprocess
from typing import Any

def run_benchmark(bench_type: str, error_interp: str = "nearest",
                  scheme: str = "explicit", backend: str = "cuda_kernel", device: str = "cuda",
                  use_tqdm: str = "true", log_level: str = "info",
                  dt: float = 5e-6, T_total: float = 5e-4, tau_tilde: float = 5e-5,
                  Lx: float = 1.0, v_max: float = 10.0,ini_nx: int = 128,ini_nv: int = 64,
                  nx_list: list = [64, 128, 256, 512, 1024],
                  nv_list: list = [32, 64, 128, 256, 516],
                  scheme_params: Any = None):

    base_cfg = Config(
        model="BGK1D",
        scheme=scheme,
        backend=backend,
        device=device,
        dtype="float64",
        use_tqdm=use_tqdm,
        log_level=log_level,
        model_cfg=BGK1D.ModelConfig(
            grid=BGK1D.Grid1D1V(nx=ini_nx, nv=ini_nv, Lx=Lx, v_max=v_max),
            time=BGK1D.TimeConfig(dt=dt, T_total=T_total),
            params=BGK1D.BGK1D1VParams(tau_tilde=tau_tilde),
            scheme_params=scheme_params
        )
    )

    def _get_cpu_name() -> str:
        # macOS: sysctl
        if platform.system() == "Darwin":
            try:
                out = subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
                if out:
                    return out
            except Exception:
                pass

        # Linux: /proc/cpuinfo
        if platform.system() == "Linux":
            try:
                with open("/proc/cpuinfo", "r") as f:
                    for line in f:
                        if "model name" in line:
                            return line.split(":", 1)[1].strip()
            except Exception:
                pass

        # fallback
        p = platform.processor()
        if p:
            return p
        return f"{platform.machine()} CPU"

    def _get_device_name(cfg: Config) -> str:
        if cfg.device == "cuda" and torch.cuda.is_available():
            try:
                return torch.cuda.get_device_name(0)
            except Exception:
                return "CUDA GPU"
        return str(cfg.device)

    out = {
        "meta": {
            "bench_type": bench_type,
            "error_interp": error_interp,
            "scheme": scheme,
            "backend": backend,
            "device": device,
            "dtype": "float64",
            "dt": dt,
            "T_total": T_total,
            "tau_tilde": tau_tilde,
            "Lx": Lx,
            "v_max": v_max,
            "device_name": _get_device_name(base_cfg),
            "cpu_name": _get_cpu_name(),
        },
        "records": []
    }

    def run_one(cfg1: Config, tag: str):
        engine = Engine(cfg1, apply_logging_flag=False)
        engine.run()
        snap = snapshot_from_engine(engine)
        out["records"].append({
            "tag": tag,
            "sweep": {"nx": cfg1.model_cfg.grid.nx, "nv": cfg1.model_cfg.grid.nv},
            "fields": snap,
        })

    def run_timebenchmark(cfg1: Config, tag: str):
        """
        Engine.stepper(steps) を直接回して timing を測定する。
        - tqdm / logging のオーバーヘッドを避けるため Engine.run() は使わない
        - warmup + CPU wall + (CUDAなら) GPU event time
        """
        # time ベンチでは tqdm を強制OFFにする（I/O混入を避ける）
        cfg1 = replace(cfg1, use_tqdm="false")

        engine = Engine(cfg1, apply_logging_flag=False)

        n_steps = int(cfg1.model_cfg.time.n_steps)
        warmup_steps = min(5, max(0, n_steps // 4))

        is_cuda = (cfg1.device == "cuda") and torch.cuda.is_available()

        # CUDA timing 準備
        start_event = end_event = None
        if is_cuda:
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)

        # Warm-up（JIT/初回メモリ確保/キャッシュの影響を抑える）
        for s in range(warmup_steps):
            engine.stepper(s)

        if is_cuda:
            torch.cuda.synchronize()

        # 本計測
        t0 = time.perf_counter()
        if is_cuda:
            start_event.record()

        for s in range(n_steps):
            engine.stepper(s)

        if is_cuda:
            end_event.record()
            torch.cuda.synchronize()

        cpu_total = time.perf_counter() - t0

        timing = {
            "nx": int(cfg1.model_cfg.grid.nx),
            "nv": int(cfg1.model_cfg.grid.nv),
            "total_grid_points": int(cfg1.model_cfg.grid.nx) * int(cfg1.model_cfg.grid.nv),
            "device": str(cfg1.device),
            "backend": cfg1.backend_name,
            "scheme": cfg1.scheme_name,
            "total_steps": n_steps,
            "warmup_steps": warmup_steps,
            "cpu_total_time_sec": cpu_total,                 # GPU同期込み walltime
            "cpu_time_per_step_sec": cpu_total / max(1, n_steps),
        }

        if is_cuda and start_event is not None and end_event is not None:
            gpu_ms = start_event.elapsed_time(end_event)
            timing.update({
                "gpu_total_time_ms": gpu_ms,
                "gpu_total_time_sec": gpu_ms / 1000.0,
                "gpu_time_per_step_ms": gpu_ms / max(1, n_steps),
            })

        out["records"].append({
            "tag": tag,
            "sweep": {"nx": cfg1.model_cfg.grid.nx, "nv": cfg1.model_cfg.grid.nv},
            "timing": timing,
        })

    if bench_type == "x_grid":
        for nx in nx_list:
            cfg1 = with_grid(base_cfg, nx=nx)
            print(f"[bench:{bench_type}] nx={cfg1.model_cfg.grid.nx}, nv={cfg1.model_cfg.grid.nv}")
            run_one(cfg1, tag=scheme)

    elif bench_type == "v_grid":
        for nv in nv_list:
            cfg1 = with_grid(base_cfg, nv=nv)
            print(f"[bench:{bench_type}] nx={cfg1.model_cfg.grid.nx}, nv={cfg1.model_cfg.grid.nv}")
            run_one(cfg1, tag=scheme)

    elif bench_type == "time":
        # time は snapshot を取らず timing のみ
        for nv in nv_list:
            for nx in nx_list:
                cfg1 = with_grid(base_cfg, nx=nx, nv=nv)
                print(f"[bench:{bench_type}] nx={cfg1.model_cfg.grid.nx}, nv={cfg1.model_cfg.grid.nv}")
                run_timebenchmark(cfg1, tag=scheme)

    else:
        raise ValueError(f"Unknown benchmark type: {bench_type}")

    out = append_errors(out, kind=error_interp)
    return out