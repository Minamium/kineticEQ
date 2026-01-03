import os
import tempfile
import numpy as np
from kineticEQ import BGK1DPlot


def make_sim():
    return BGK1DPlot(
        solver="implicit",
        implicit_solver="holo",
        tau_tilde=1e-3,
        v_max=1.0,
        dt=1e-3,
        nv=4,
        T_total=0.0,
        dtype="float64",
        use_tqdm=False,
        device="cpu",
        auto_compile=False,
    )


def _dummy_benchmark_results():
    v = np.linspace(-1.0, 1.0, 3)
    x = np.linspace(0.0, 1.0, 4)
    record = {
        "x": x,
        "v": v,
        "f": np.zeros((x.size, v.size)),
        "n": np.zeros_like(x),
        "u": np.zeros_like(x),
        "T": np.zeros_like(x),
        "dx": x[1] - x[0],
        "dv": v[1] - v[0],
    }
    return {"bench_type": "spatial", 4: record, 8: record}


def test_can_save_and_load_benchmark_results():
    sim = make_sim()
    bench_results = _dummy_benchmark_results()

    with tempfile.TemporaryDirectory() as tmpdir:
        file_path = os.path.join(tmpdir, "results.pkl")
        saved_path = sim.save_benchmark_results(bench_results=bench_results, filename=file_path)
        assert os.path.isfile(saved_path)

        loaded = sim.load_benchmark_results(saved_path)
        assert loaded["bench_type"] == bench_results["bench_type"]
        for grid in (4, 8):
            assert loaded[grid]["dx"] == bench_results[grid]["dx"]
            assert loaded[grid]["dv"] == bench_results[grid]["dv"]
            np.testing.assert_array_equal(loaded[grid]["x"], bench_results[grid]["x"])
            np.testing.assert_array_equal(loaded[grid]["v"], bench_results[grid]["v"])
            np.testing.assert_array_equal(loaded[grid]["f"], bench_results[grid]["f"])
            np.testing.assert_array_equal(loaded[grid]["n"], bench_results[grid]["n"])
            np.testing.assert_array_equal(loaded[grid]["u"], bench_results[grid]["u"])
            np.testing.assert_array_equal(loaded[grid]["T"], bench_results[grid]["T"])
