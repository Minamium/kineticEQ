import pytest
import torch
from kineticEQ.utillib.device_util import resolve_device
from kineticEQ import Config, Engine, BGK1D

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")

def test_resolve_device_cuda_ok():
    assert resolve_device("cuda") == "cuda"

@pytest.mark.parametrize("model", ["BGK1D1V"])
@pytest.mark.parametrize("scheme", ["explicit", "implicit"])
@pytest.mark.parametrize("backend", ["torch", "cuda_kernel"])
def test_smoke_cuda(model, scheme, backend):
    cfg = Config(
        model=model,
        scheme=scheme,
        backend=backend,
        device="cuda",
        dtype="float64",
        use_tqdm="false",
        log_level="debug",
        model_cfg=BGK1D.ModelConfig(
            grid=BGK1D.Grid1D1V(nx=256, nv=64, Lx=1.0, v_max=10.0),
            time=BGK1D.TimeConfig(dt=5e-5, T_total=5e-4),
            params=BGK1D.BGK1D1VParams(tau_tilde=5e-2),
        )
    )
    Engine(cfg, apply_logging_flag=True).run()