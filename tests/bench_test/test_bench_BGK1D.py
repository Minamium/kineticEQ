import pytest
from kineticEQ import Config, Engine, BGK1D
from kineticEQ.plotting.bgk1d.plot_state import plot_state
from kineticEQ.analysis.BGK1D.benchmark import run_benchmark

@pytest.mark.parametrize("bench_type", ["x_grid", "v_grid", "time"])
def test_bgk1d_benchmark_cpu(bench_type):
    out = run_benchmark(bench_type=bench_type, scheme="explicit", 
                        backend="torch", device="cpu",
                        use_tqdm="true", log_level="info",
                        nv_list=[64, 128, 256],
                        nx_list=[64, 128, 256])