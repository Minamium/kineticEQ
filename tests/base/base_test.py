import numpy as np
from kineticEQ import BGK1DPlot


def make_config(**overrides):
    config = {
        "solver": "implicit",
        "implicit_solver": "holo",
        "picard_iter": 4,
        "picard_tol": 1e-3,
        "ho_iter": 2,
        "lo_iter": 2,
        "ho_tol": 1e-3,
        "lo_tol": 1e-3,
        "Con_Terms_do": False,
        "flux_consistency_do": False,
        "SVdown": False,
        "tau_tilde": 1e-3,
        "nx": 6,
        "nv": 6,
        "v_max": 1.0,
        "dt": 1e-3,
        "T_total": 0.0,
        "dtype": "float64",
        "use_tqdm": False,
        "device": "cpu",
        "auto_compile": False,
        "initial_regions": [
            {"x_range": (0.0, 0.5), "n": 1.0, "u": 0.0, "T": 1.0},
            {"x_range": (0.5, 1.0), "n": 0.5, "u": 0.0, "T": 0.8},
        ],
    }
    config.update(overrides)
    return config


def test_allocation_and_initialization_on_cpu():
    sim = BGK1DPlot(**make_config())

    sim.Array_allocation()
    sim.set_initial_condition()
    sim.apply_boundary_condition()

    assert sim.f.shape == (sim.nx, sim.nv)
    assert np.allclose(sim.x.cpu().numpy()[0], 0.0)
    assert np.allclose(sim.v.cpu().numpy()[0], -sim.v_max)
