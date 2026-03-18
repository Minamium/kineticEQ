import pytest
from kineticEQ import Config, Engine, BGK1D

@pytest.mark.parametrize("model", ["BGK1D1V"])
@pytest.mark.parametrize(
    ("scheme", "backend"),
    [
        pytest.param("explicit", "torch", id="torch-explicit"),
        pytest.param("implicit", "cpu_kernel", id="cpu_kernel-implicit"),
    ],
)
def test_smoke_cpu(model, scheme, backend):
    if backend == "torch":
        model_cfg = BGK1D.ModelConfig(
            grid=BGK1D.Grid1D1V(nx=128, nv=64, Lx=1.0, v_max=10.0),
            time=BGK1D.TimeConfig(dt=5e-6, T_total=0.05),
            params=BGK1D.BGK1D1VParams(tau_tilde=5e-5),
        )
    else:
        model_cfg = BGK1D.ModelConfig(
            grid=BGK1D.Grid1D1V(nx=65, nv=33, Lx=1.0, v_max=10.0),
            time=BGK1D.TimeConfig(dt=5e-5, T_total=1e-4),
            params=BGK1D.BGK1D1VParams(tau_tilde=5e-2),
            scheme_params=BGK1D.implicit.Params(
                picard_iter=32,
                picard_tol=1e-3,
                abs_tol=1e-13,
            ),
        )

    cfg = Config(
        model=model,
        scheme=scheme,
        backend=backend,
        device="cpu",
        dtype="float64",
        use_tqdm="false",
        log_level="info",
        model_cfg=model_cfg,
    )
    Engine(cfg, apply_logging_flag=True).run()
    
