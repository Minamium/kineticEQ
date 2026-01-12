import pytest
from kineticEQ import Config, Engine, BGK1D
from kineticEQ.plotting.bgk1d.plot_state import plot_state

@pytest.mark.parametrize("model", ["BGK1D1V"])
@pytest.mark.parametrize("scheme", ["explicit"])
@pytest.mark.parametrize("backend", ["torch"])
def test_plot_cpu(model, scheme, backend):
    cfg = Config(
        model=model,
        scheme=scheme,
        backend=backend,
        device="cpu",
        dtype="float64",
        use_tqdm="false",
        log_level="info",
        model_cfg=BGK1D.ModelConfig(
            grid=BGK1D.Grid1D1V(nx=128, nv=64, Lx=1.0, v_max=10.0),
            time=BGK1D.TimeConfig(dt=5e-6, T_total=0.05),
            params=BGK1D.BGK1D1VParams(tau_tilde=5e-5),
        )
    )
    Engine(cfg, apply_logging_flag=True).run()
    