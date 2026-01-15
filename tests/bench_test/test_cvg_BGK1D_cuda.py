import pytest, torch
from kineticEQ.analysis.BGK1D.convergence_test import run_convergence_test
from kineticEQ.analysis.BGK1D.plotting.plot_convergence_result import plot_convergence_results
from kineticEQ.utillib.device_util import resolve_device
from kineticEQ import BGK1D

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")

def test_resolve_device_cuda_ok():
    assert resolve_device("cuda") == "cuda"

def test_bgk1d_convergence_test_cuda():
    out = run_convergence_test()
    plot_convergence_results(out, filename="test_cvg_BGK1D_cuda.png", output_dir="./results")