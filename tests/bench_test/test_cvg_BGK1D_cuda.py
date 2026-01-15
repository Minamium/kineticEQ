import pytest, torch
from kineticEQ.analysis.BGK1D.convergence_test import run_convergence_test
from kineticEQ.utillib.device_util import resolve_device
from kineticEQ import BGK1D

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")

def test_resolve_device_cuda_ok():
    assert resolve_device("cuda") == "cuda"

def test_bgk1d_convergence_test_cuda():
    out = run_convergence_test()