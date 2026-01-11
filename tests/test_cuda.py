import pytest
import torch
from kineticEQ.utillib.device_util import resolve_device
from kineticEQ import Config, Engine

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")

def test_resolve_device_cuda_ok():
    assert resolve_device("cuda") == "cuda"

@pytest.mark.parametrize("model", ["BGK1D1V"])
@pytest.mark.parametrize("scheme", ["explicit"])
@pytest.mark.parametrize("backend", ["torch"])
def test_smoke_cuda(model, scheme, backend):
    cfg = Config(
        model=model,
        scheme=scheme,
        backend=backend,
        device="cuda",
        dtype="float64",
        use_tqdm="false",
        log_level="debug",
        model_cfg=ModelConfig(
            grid=Grid1D1V(nx=500, nv=200, Lx=1.0, v_max=10.0),
            time=TimeConfig(dt=5e-6, T_total=5e-5),
            params=BGK1D1VParams(tau_tilde=5e-1),
            initial=InitialCondition1D(
                initial_regions=(
                    {"x_range": (0.0, 0.5), "n": 1.0, "u": 0.0, "T": 1.0},
                    {"x_range": (0.5, 1.0), "n": 0.125, "u": 0.0, "T": 0.8},
                )
            ),
        ),
    )
    Engine(cfg, apply_logging_flag=True).run()