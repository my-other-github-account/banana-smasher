import pytest
import torch

from repair_api.balanced64 import ArtifactError
from repair_api.modern_green_resident import _fp64_state_adam


def test_fp64_state_adam_keeps_squared_extreme_finite_without_clipping():
    parameter = torch.nn.Parameter(torch.tensor([1.0], dtype=torch.float32))
    optimizer = _fp64_state_adam(torch, [{"params": [parameter], "lr": 1.0e-3}])
    parameter.grad = torch.tensor([1.0e30], dtype=torch.float32)
    optimizer.step()
    state = optimizer.state[parameter]
    assert state["exp_avg"].dtype == torch.float64
    assert state["exp_avg_sq"].dtype == torch.float64
    assert torch.isfinite(state["exp_avg_sq"]).all()
    assert torch.isfinite(parameter).all()
    assert parameter.item() != 1.0


def test_fp64_state_adam_matches_standard_adam_at_ordinary_scale():
    standard_parameter = torch.nn.Parameter(torch.tensor([1.0, -2.0], dtype=torch.float32))
    stable_parameter = torch.nn.Parameter(standard_parameter.detach().clone())
    standard = torch.optim.Adam([standard_parameter], lr=1.0e-3, foreach=False)
    stable = _fp64_state_adam(torch, [{"params": [stable_parameter], "lr": 1.0e-3}])
    for gradient in ([0.25, -0.5], [0.125, 0.75], [-0.4, 0.2]):
        standard_parameter.grad = torch.tensor(gradient, dtype=torch.float32)
        stable_parameter.grad = torch.tensor(gradient, dtype=torch.float32)
        standard.step()
        stable.step()
    torch.testing.assert_close(stable_parameter, standard_parameter, rtol=1e-6, atol=1e-7)


def test_fp64_state_adam_unscales_power_of_two_gradients_without_scientific_delta():
    reference_parameter = torch.nn.Parameter(torch.tensor([1.0, -2.0], dtype=torch.float32))
    scaled_parameter = torch.nn.Parameter(reference_parameter.detach().clone())
    reference = _fp64_state_adam(torch, [{"params": [reference_parameter], "lr": 1.0e-3}])
    scale = 2.0 ** -64
    scaled = _fp64_state_adam(
        torch,
        [{"params": [scaled_parameter], "lr": 1.0e-3}],
        gradient_scale=scale,
    )
    for gradient in ([1.0e20, -2.0e20], [3.0e20, 4.0e20]):
        reference_parameter.grad = torch.tensor(gradient, dtype=torch.float32)
        scaled_parameter.grad = torch.tensor(gradient, dtype=torch.float32) * scale
        reference.step()
        scaled.step()
    torch.testing.assert_close(scaled_parameter, reference_parameter, rtol=0.0, atol=0.0)


def test_fp64_state_adam_reports_nonfinite_gradient_before_moment_mutation():
    parameter = torch.nn.Parameter(torch.tensor([1.0], dtype=torch.float32))
    optimizer = _fp64_state_adam(
        torch,
        [{"params": [parameter], "lr": 1.0e-3, "group_name": "outputs"}],
        gradient_scale=2.0 ** -96,
    )
    parameter.grad = torch.tensor([float("inf")], dtype=torch.float32)

    with pytest.raises(
        ArtifactError,
        match=r"nonfinite gradient before FP64 Adam update: group=outputs parameter_index=0",
    ):
        optimizer.step()

    assert optimizer.state[parameter] == {}
    assert parameter.item() == 1.0


def test_fp64_state_adam_keeps_equivalent_scaled_domain_moments_finite():
    scale = 2.0 ** -96
    parameter = torch.nn.Parameter(torch.tensor([1.0], dtype=torch.float64))
    optimizer = _fp64_state_adam(
        torch,
        [{"params": [parameter], "lr": 1.0e-3, "group_name": "outputs"}],
        gradient_scale=scale,
    )
    # Finite loss-scaled gradient; unscaling before squaring overflows FP64.
    parameter.grad = torch.tensor([1.0e140], dtype=torch.float64)

    optimizer.step()

    state = optimizer.state[parameter]
    assert torch.isfinite(state["exp_avg"]).all()
    assert torch.isfinite(state["exp_avg_sq"]).all()
    assert torch.isfinite(parameter).all()


def test_fp64_state_adam_names_first_nonfinite_scaled_moment():
    scale = 2.0 ** -96
    parameter = torch.nn.Parameter(torch.tensor([1.0], dtype=torch.float64))
    optimizer = _fp64_state_adam(
        torch,
        [{"params": [parameter], "lr": 1.0e-3, "group_name": "outputs"}],
        gradient_scale=scale,
    )
    parameter.grad = torch.tensor([1.0e200], dtype=torch.float64)

    with pytest.raises(
        ArtifactError,
        match=r"nonfinite FP64 Adam exp_avg_sq during mutation: group=outputs parameter_index=0",
    ):
        optimizer.step()


def test_fp64_state_adam_rescales_saved_fp64_moments_before_parameter_dtype_cast():
    source_parameter = torch.nn.Parameter(torch.tensor([1.0], dtype=torch.float64))
    source = _fp64_state_adam(
        torch,
        [{"params": [source_parameter], "lr": 1.0e-3, "group_name": "luts"}],
    )
    source.state[source_parameter] = {
        "step": torch.tensor(14.0, dtype=torch.float64),
        "exp_avg": torch.tensor([7.44e23], dtype=torch.float64),
        "exp_avg_sq": torch.tensor([5.54e46], dtype=torch.float64),
        "gradient_scale": 1.0,
    }

    scale = 2.0 ** -96
    target_parameter = torch.nn.Parameter(torch.tensor([1.0], dtype=torch.float32))
    target = _fp64_state_adam(
        torch,
        [{"params": [target_parameter], "lr": 1.0e-3, "group_name": "luts"}],
        gradient_scale=scale,
    )
    target.load_state_dict(source.state_dict())

    state = target.state[target_parameter]
    assert state["exp_avg"].dtype == torch.float64
    assert state["exp_avg_sq"].dtype == torch.float64
    assert torch.isfinite(state["exp_avg"]).all()
    assert torch.isfinite(state["exp_avg_sq"]).all()
    torch.testing.assert_close(
        state["exp_avg"], torch.tensor([7.44e23 * scale], dtype=torch.float64)
    )
    torch.testing.assert_close(
        state["exp_avg_sq"], torch.tensor([5.54e46 * scale**2], dtype=torch.float64)
    )


def test_fp64_state_adam_load_preserves_dormant_parameter_sparse_state():
    source_active = torch.nn.Parameter(torch.tensor([1.0], dtype=torch.float64))
    source_dormant = torch.nn.Parameter(torch.tensor([2.0], dtype=torch.float64))
    source = _fp64_state_adam(
        torch,
        [{"params": [source_active, source_dormant], "lr": 1.0e-3, "group_name": "luts"}],
    )
    source.state[source_active] = {
        "step": torch.tensor(14.0, dtype=torch.float64),
        "exp_avg": torch.tensor([1.0], dtype=torch.float64),
        "exp_avg_sq": torch.tensor([2.0], dtype=torch.float64),
        "gradient_scale": 1.0,
    }

    target_active = torch.nn.Parameter(torch.tensor([1.0], dtype=torch.float32))
    target_dormant = torch.nn.Parameter(torch.tensor([2.0], dtype=torch.float32))
    target = _fp64_state_adam(
        torch,
        [{"params": [target_active, target_dormant], "lr": 1.0e-3, "group_name": "luts"}],
        gradient_scale=2.0 ** -96,
    )
    target.load_state_dict(source.state_dict())

    assert target_active in target.state
    assert target_dormant not in target.state
