import pytest
from kineticEQ import Config, Engine

@pytest.mark.parametrize("model", ["BGK1D1V", "BGK2D2V"])
@pytest.mark.parametrize("scheme", ["explicit"])
def test_smoke_cpu(model, scheme):
    cfg = Config(model=model, scheme=scheme, backend="torch", device="cpu", use_tqdm="false")
    Engine(cfg, apply_logging_flag=True).run()