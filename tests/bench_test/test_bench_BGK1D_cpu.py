import pytest
from kineticEQ.analysis.BGK1D.benchmark import run_benchmark
from kineticEQ.analysis.BGK1D.plotting.plot_benchmark_result import plot_benchmark_results
from kineticEQ.analysis.BGK1D.plotting.plot_timing_benchmark import plot_timing_benchmark

@pytest.mark.parametrize("bench_type", ["x_grid", "v_grid", "time"])
def test_bgk1d_benchmark_cpu(bench_type):
    out = run_benchmark(bench_type=bench_type, scheme="explicit", 
                        backend="torch", device="cpu",
                        use_tqdm="true", log_level="info",
                        nv_list=[64, 128, 256],
                        nx_list=[64, 128, 256])

    if bench_type == "x_grid" or bench_type == "v_grid":
        plot_benchmark_results(out, out_dir="./bechmark_result", 
                               fname_moment=f"{bench_type}_{scheme}_{backend}_cpu_moments.png", 
                               fname_error=f"{bench_type}_{scheme}_{backend}_cpu_errors.png")

    if bench_type == "time":
        plot_timing_benchmark(out, out_dir="./bechmark_result", 
                              fname=f"{bench_type}_{scheme}_{backend}_cpu_timing.png")