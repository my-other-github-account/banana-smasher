import pytest

from repair_api.balanced64 import ArtifactError
from repair_api.modern_green_resident import (
    _published_pre_controlled_schedule,
    _weighted_window_loss,
)


def _schedule(weights):
    windows = [28, 56, 20, 21]
    return {
        "expected_first_four_update_boundary": {
            "updates": [
                {"global_update": update, "windows": windows}
                for update in [21, 22, 23, 24]
            ]
        },
        "exact_parameter_delta": {
            "to": {
                "windows_per_optimizer_update": 4,
                "category_loss_weight": 0.25,
                "pipeline_microbatch": 2,
                "pipeline_groups_per_update": 2,
                "group_gradient_scale": 0.5,
                "per_window_teacher_kl_weights": weights,
                "per_window_teacher_kl_window_order": windows,
            }
        },
        "unchanged_fields": {
            "train_bank_membership_sha256": (
                "3553fce00efdb6d452171e6d5c429adc31580dedbf63eb821f81bc82406983b3"
            )
        },
    }


def _config(weights):
    return {
        "controlled_window_schedule_source_rows": [21, 22, 23, 24],
        "controlled_windows_per_update": 4,
        "pipeline_microbatch": 2,
        "per_window_teacher_kl_weights": weights,
        "per_window_teacher_kl_window_order": [28, 56, 20, 21],
    }


def test_anchor_weighted_schedule_is_exactly_bound_and_normalized():
    weights = [0.375, 0.375, 0.125, 0.125]
    rows, labels, count, membership = _published_pre_controlled_schedule(
        _schedule(weights), _config(weights)
    )
    assert [row["windows"] for row in rows] == [[28, 56, 20, 21]] * 4
    assert labels == [21, 22, 23, 24]
    assert count == 4
    assert membership.startswith("3553fce0")


@pytest.mark.parametrize(
    "weights,order",
    [
        ([0.5, 0.5, 0.0, 0.0], [28, 56, 20, 21]),
        ([0.375, 0.375, 0.125, 0.124], [28, 56, 20, 21]),
        ([0.375, 0.375, 0.125, 0.125], [56, 28, 20, 21]),
    ],
)
def test_anchor_weighted_schedule_fails_closed_on_weight_or_order_drift(weights, order):
    config = _config(weights)
    config["per_window_teacher_kl_window_order"] = order
    with pytest.raises(ArtifactError, match="per-window teacher KL"):
        _published_pre_controlled_schedule(_schedule(weights), config)


def test_anchor_weighted_group_scaling_preserves_exact_global_weight_vector():
    torch = pytest.importorskip("torch")
    config = _config([0.375, 0.375, 0.125, 0.125])
    anchor_group = _weighted_window_loss(
        torch,
        [torch.tensor([1.0]), torch.tensor([2.0])],
        [28, 56],
        config,
    )
    support_group = _weighted_window_loss(
        torch,
        [torch.tensor([4.0]), torch.tensor([8.0])],
        [20, 21],
        config,
    )
    observed = (anchor_group + support_group) / 2.0
    expected = 0.375 * 1.0 + 0.375 * 2.0 + 0.125 * 4.0 + 0.125 * 8.0
    assert observed.item() == pytest.approx(expected)


def test_uniform_known_control_is_bitwise_formula_equivalent():
    torch = pytest.importorskip("torch")
    values = [torch.tensor([1.0, 3.0]), torch.tensor([2.0, 6.0])]
    config = _config([0.25, 0.25, 0.25, 0.25])
    observed = _weighted_window_loss(torch, values, [28, 56], config)
    control = torch.stack([value.mean() for value in values]).mean()
    assert torch.equal(observed, control)
