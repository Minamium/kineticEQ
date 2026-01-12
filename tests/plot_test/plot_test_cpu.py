import pytest
from kineticEQ import Config, Engine, BGK1D

@pytest.mark.parametrize("model", ["BGK1D1V"])
@pytest.mark.parametrize("scheme", ["explicit"])
def test_smoke_cpu(model, scheme):
    cfg = Config(
        model=model,
        scheme=scheme,
        backend="torch",
        device="cpu",
        dtype="float64",
        use_tqdm="false",
        log_level="debug",
        model_cfg=BGK1D.ModelConfig(
            grid=BGK1D.Grid1D1V(nx=32, nv=8, Lx=1.0, v_max=5.0),
            time=BGK1D.TimeConfig(dt=5e-4, T_total=5e-5),
            params=BGK1D.BGK1D1VParams(tau_tilde=5e-2),
        )
    )
    Engine(cfg, apply_logging_flag=True).run()