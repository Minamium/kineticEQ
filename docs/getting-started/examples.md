---
title: Examples
parent: Getting Started
nav_order: 13
---

# Examples

## 1. BGK1D explicit (CPU / torch)

```python
from kineticEQ import Config, Engine, BGK1D

cfg = Config(
    model="BGK1D1V",
    scheme="explicit",
    backend="torch",
    device="cpu",
    dtype="float64",
    use_tqdm="false",
    model_cfg=BGK1D.ModelConfig(
        grid=BGK1D.Grid1D1V(nx=256, nv=128, Lx=1.0, v_max=10.0),
        time=BGK1D.TimeConfig(dt=5e-6, T_total=5e-4),
        params=BGK1D.BGK1D1VParams(tau_tilde=5e-5),
    ),
)

engine = Engine(cfg)
result = engine.run()
```

## 2. BGK1D implicit (CUDA kernel + AA)

```python
from kineticEQ import Config, Engine, BGK1D

cfg = Config(
    model="BGK1D1V",
    scheme="implicit",
    backend="cuda_kernel",
    device="cuda",
    dtype="float64",  # 現行の CUDA 拡張実装と整合
    model_cfg=BGK1D.ModelConfig(
        grid=BGK1D.Grid1D1V(nx=512, nv=256, Lx=1.0, v_max=10.0),
        time=BGK1D.TimeConfig(dt=5e-5, T_total=5e-3),
        params=BGK1D.BGK1D1VParams(tau_tilde=5e-7),
        scheme_params=BGK1D.implicit.Params(
            picard_iter=200,
            picard_tol=1e-6,
            abs_tol=1e-13,
            conv_type="f",     # "f" or "w"
            aa_enable=True,
            aa_m=6,
            aa_beta=1.0,
            aa_stride=1,
            aa_start_iter=2,
            aa_reg=1e-10,
            aa_alpha_max=50.0,
        ),
    ),
)

engine = Engine(cfg)
engine.run()
```

## 3. CNN warmstart を有効化

```python
from kineticEQ import Config, Engine, BGK1D

cfg = Config(
    model="BGK1D1V",
    scheme="implicit",
    backend="cuda_kernel",
    device="cuda",
    dtype="float64",
    model_cfg=BGK1D.ModelConfig(
        scheme_params=BGK1D.implicit.Params(
            picard_iter=200,
            picard_tol=1e-6,
            moments_cnn_modelpath="/path/to/best_speed.pt",
        )
    ),
)

Engine(cfg).run()
```

## 4. 分析モジュールの例

```python
from kineticEQ.analysis.BGK1D.benchmark import run_benchmark
from kineticEQ.analysis.BGK1D.plotting import plot_timing_benchmark

out = run_benchmark(
    bench_type="time",
    scheme="explicit",
    backend="cuda_kernel",
    device="cuda",
    nx_list=[65, 129, 257],
    nv_list=[65, 129, 257],
)
plot_timing_benchmark(out, out_dir="./results/benchmarks")
```

## 5. 現在は使えない構成

- `model="BGK2D2V"` は現行 Engine 実装では実行不可
- `scheme="holo_nn"` は Config 受理するが stepper 未登録
