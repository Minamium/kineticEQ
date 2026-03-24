import torch

from kineticEQ.core.schemes.BGK1D.bgk1d_utils.implicit.bgk1d_momentCNN_util import (
    predict_next_moments_delta,
)


class ConstantDeltaModel:
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        _, _, nx = x.shape
        return torch.ones((1, 3, nx), device=x.device, dtype=x.dtype)


def test_predict_next_moments_delta_preserves_original_mode():
    model = ConstantDeltaModel()
    n0 = torch.ones(7, dtype=torch.float64)
    u0 = torch.zeros(7, dtype=torch.float64)
    T0 = torch.ones(7, dtype=torch.float64)

    _, _, _, dn, dmid, dT = predict_next_moments_delta(
        model,
        n0,
        u0,
        T0,
        logdt=-3.0,
        logtau=-6.0,
        delta_type="dw",
        warm_delta_weight_mode="none",
    )

    assert torch.allclose(dn, torch.ones_like(n0))
    assert torch.allclose(dmid, torch.ones_like(n0))
    assert torch.allclose(dT, torch.ones_like(n0))


def test_predict_next_moments_delta_w_grad_damps_flat_region_more_than_shock():
    model = ConstantDeltaModel()
    n0 = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0], dtype=torch.float64)
    u0 = torch.zeros(11, dtype=torch.float64)
    T0 = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0], dtype=torch.float64)

    _, _, _, dn, _, _ = predict_next_moments_delta(
        model,
        n0,
        u0,
        T0,
        logdt=-3.0,
        logtau=-6.0,
        delta_type="dw",
        warm_delta_weight_mode="w_grad",
        warm_delta_weight_floor=0.2,
        warm_delta_weight_center=0.5,
        warm_delta_weight_sharpness=10.0,
        warm_delta_weight_sigma=1.5,
    )

    assert float(dn[1]) > 0.2
    assert float(dn[3]) > float(dn[1])
    assert float(dn[4]) > float(dn[3])
    assert float(dn[5]) > float(dn[3])
    assert float(torch.max(dn)) <= 1.0


def test_predict_next_moments_delta_excludes_boundary_adjacent_cells():
    model = ConstantDeltaModel()
    n0 = torch.ones(9, dtype=torch.float64)
    u0 = torch.zeros(9, dtype=torch.float64)
    T0 = torch.ones(9, dtype=torch.float64)

    n1_int, u1_int, T1_int, dn, dmid, dT = predict_next_moments_delta(
        model,
        n0,
        u0,
        T0,
        logdt=-3.0,
        logtau=-6.0,
        delta_type="dw",
        warm_delta_weight_mode="none",
        warm_delta_exclude_cells=2,
    )

    expected_full = torch.tensor([0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0], dtype=torch.float64)
    assert torch.allclose(dn, expected_full)
    assert torch.allclose(dmid, expected_full)
    assert torch.allclose(dT, expected_full)

    expected_int = torch.tensor([1.0, 1.0, 2.0, 2.0, 2.0, 1.0, 1.0], dtype=torch.float64)
    assert torch.allclose(n1_int, expected_int)
    assert torch.allclose(u1_int, expected_full[1:-1])
    assert torch.allclose(T1_int, expected_int)
