import pytest
from kineticEQ import Config, Engine

@pytest.mark.parametrize("model", ["BGK1D1V"])
@pytest.mark.parametrize("scheme", ["explicit"])
def test_smoke_cpu(model, scheme):
    cfg = Config(model=model, scheme=scheme, backend="torch", device="cpu", use_tqdm="false", log_level="DEBUG",
                nx=124, nv=64, Lx=1.0, v_max=10.0, dt=5e-4, T_total=5e-3, tau_tilde=5e-1,
                initial_regions=(
                    {"x_range": (0.0, 0.5), "n": 1.0, "u": 0.0, "T": 1.0},
                    {"x_range": (0.5, 1.0), "n": 0.125, "u": 0.0, "T": 0.8},
                ))
    Engine(cfg, apply_logging_flag=True).run()