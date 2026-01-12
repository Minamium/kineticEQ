import pytest
import torch
from kineticEQ.utillib.device_util import resolve_device
from kineticEQ import Config, Engine, BGK1D
from kineticEQ.plotting.bgk1d.plot_state import plot_state

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")

def test_resolve_device_cuda_ok():
    assert resolve_device("cuda") == "cuda"

@pytest.mark.parametrize("model", ["BGK1D1V"])
@pytest.mark.parametrize("scheme", ["explicit"])
@pytest.mark.parametrize("backend", ["torch", "cuda_kernel"])
def test_plot_cuda(model, scheme, backend):
    cfg = Config(
        model=model,
        scheme=scheme,
        backend=backend,
        device="cuda",
        dtype="float64",
        use_tqdm="false",
        log_level="info",
        model_cfg=BGK1D.ModelConfig(
            grid=BGK1D.Grid1D1V(nx=512, nv=128, Lx=1.0, v_max=10.0),
            time=BGK1D.TimeConfig(dt=5e-6, T_total=0.05),
            params=BGK1D.BGK1D1VParams(tau_tilde=5e-5),
        )
    )
    simulation_engine = Engine(cfg, apply_logging_flag=True)
    simulation_engine.run()
    plot_state(simulation_engine.state)