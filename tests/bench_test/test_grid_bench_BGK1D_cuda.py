import pytest, torch
from kineticEQ.analysis.BGK1D.benchmark import run_benchmark
from kineticEQ.utillib.device_util import resolve_device
from kineticEQ.analysis.BGK1D.plotting.plot_benchmark_result import plot_benchmark_results
from kineticEQ.analysis.BGK1D.plotting.plot_timing_benchmark import plot_timing_benchmark
from kineticEQ import BGK1D

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")

def test_resolve_device_cuda_ok():
    assert resolve_device("cuda") == "cuda"

@pytest.mark.parametrize("bench_type", ["x_grid", "v_grid"])
@pytest.mark.parametrize("scheme", ["explicit", "implicit"])
@pytest.mark.parametrize("backend", ["cuda_kernel"])
def test_bgk1d_benchmark_cuda(bench_type, scheme, backend):
    if scheme == "holo":
        scheme_params = BGK1D.holo.Params(ho_tol=1e-6, ho_iter=64,
                                          lo_tol=1e-6, lo_iter=64)
    elif scheme == "implicit":
        scheme_params=BGK1D.implicit.Params(picard_iter=64, picard_tol=1e-6)
    elif scheme == "explicit":
        scheme_params=None

    out = run_benchmark(bench_type=bench_type, scheme=scheme,
                        T_total=0.05,
                        backend=backend, device="cuda",
                        use_tqdm="true", log_level="info",
                        ini_nx=500,ini_nv=200,
                        nv_list=[33, 65, 129, 257, 513, 1025],
                        nx_list=[33, 65, 129, 257, 513])

    plot_benchmark_results(out, out_dir="./results/benchmarks", 
                           fname_moment=f"{bench_type}_{scheme}_{backend}_cuda_moments.png", 
                           fname_error=f"{bench_type}_{scheme}_{backend}_cuda_errors.png")