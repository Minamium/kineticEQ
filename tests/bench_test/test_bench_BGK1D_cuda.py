import pytest, torch
from kineticEQ.analysis.BGK1D.benchmark import run_benchmark
from kineticEQ.utillib.device_util import resolve_device
from kineticEQ.analysis.BGK1D.plotting.plot_benchmark_result import plot_benchmark_results
from kineticEQ.analysis.BGK1D.plotting.plot_timing_benchmark import plot_timing_benchmark

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")

def test_resolve_device_cuda_ok():
    assert resolve_device("cuda") == "cuda"

@pytest.mark.parametrize("bench_type", ["x_grid", "v_grid", "time"])
@pytest.mark.parametrize("scheme", ["explicit", "implicit", "holo"])
@pytest.mark.parametrize("backend", ["cuda_kernel"])
def test_bgk1d_benchmark_cuda(bench_type, scheme, backend):
    out = run_benchmark(bench_type=bench_type, scheme=scheme,
                        T_total=0.05,
                        backend=backend, device="cuda",
                        use_tqdm="true", log_level="info",
                        nv_list=[32, 64, 128, 256, 512, 1024],
                        nx_list=[32, 64, 128, 256])

    if bench_type == "x_grid" or bench_type == "v_grid":
        plot_benchmark_results(out, out_dir="./results/benchmarks", 
                               fname_moment=f"{bench_type}_{scheme}_{backend}_cuda_moments.png", 
                               fname_error=f"{bench_type}_{scheme}_{backend}_cuda_errors.png")

    if bench_type == "time":
        plot_timing_benchmark(out, out_dir="./results/benchmarks", 
                              fname=f"{bench_type}_{scheme}_{backend}_cuda_timing.png")