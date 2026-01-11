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
    cfg = Config(model=model, scheme=scheme, backend=backend, device="cuda", use_tqdm="false", log_level="DEBUG")
    Engine(cfg, apply_logging_flag=True).run()