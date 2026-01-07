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


def _dummy_result(grid_points: int):
    x = np.linspace(0.0, 1.0, grid_points)
    v = np.linspace(-1.0, 1.0, 3)
    f = np.ones((grid_points, v.size))
    n = np.ones_like(x)
    u = np.zeros_like(x)
    T = np.ones_like(x)
    return {
        "x": x,
        "v": v,
        "f": f,
        "n": n,
        "u": u,
        "T": T,
        "dx": x[1] - x[0],
        "dv": v[1] - v[0],
    }


def test_compute_error_returns_zero_for_identical_profiles():
    sim = make_sim()

    fine_result = _dummy_result(8)
    coarse_result = _dummy_result(4)
    bench_result = {
        "bench_type": "spatial",
        4: coarse_result,
        8: fine_result,
    }

    error = sim.compute_error(bench_result)
    metrics = error[4]

    assert metrics["L1"]["f"] == 0.0
    assert metrics["L2"]["n"] == 0.0
    assert metrics["Linf"]["T"] == 0.0
