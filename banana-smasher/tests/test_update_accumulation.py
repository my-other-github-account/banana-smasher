from __future__ import annotations

import pytest
import torch

from banana_smasher.accumulation import backward_logical_mean, exact_accumulation_step


def test_exact_accumulation_weights_unequal_slices_by_total_items() -> None:
    parameter = torch.nn.Parameter(torch.tensor(2.0, dtype=torch.float64))
    optimizer = torch.optim.SGD([parameter], lr=0.1)
    segments = [torch.tensor([1.0, 2.0]), torch.tensor([4.0])]

    receipt = exact_accumulation_step(
        optimizer=optimizer,
        segments=segments,
        item_count=lambda values: int(values.numel()),
        loss_sum=lambda values: ((parameter - values) ** 2).sum(),
    )

    assert parameter.item() == pytest.approx(2.0 + 0.1 * 2.0 / 3.0)
    assert receipt["segment_items"] == [2, 1]
    assert receipt["logical_items"] == 3
    assert receipt["optimizer_steps"] == 1


def test_exact_accumulation_rejects_empty_segments_before_optimizer_mutation() -> None:
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.SGD([parameter], lr=0.1)

    with pytest.raises(ValueError, match="non-empty"):
        exact_accumulation_step(
            optimizer=optimizer,
            segments=[torch.empty(0)],
            item_count=lambda values: int(values.numel()),
            loss_sum=lambda values: parameter * values.sum(),
        )

    assert parameter.grad is None


@pytest.mark.parametrize("invalid", [True, 1.0, 1.5])
def test_backward_logical_mean_requires_integer_item_count(invalid: object) -> None:
    parameter = torch.nn.Parameter(torch.tensor(1.0))

    with pytest.raises(TypeError, match="logical_items must be an integer"):
        backward_logical_mean(parameter.square(), invalid)


@pytest.mark.parametrize("invalid", [False, 1.0, 1.5])
def test_accumulation_segment_counts_require_integers(invalid: object) -> None:
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.SGD([parameter], lr=0.1)

    with pytest.raises(TypeError, match=r"segment_items\[0\] must be an integer"):
        exact_accumulation_step(
            optimizer=optimizer,
            segments=[torch.ones(1)],
            item_count=lambda _values: invalid,
            loss_sum=lambda values: (parameter * values).sum(),
        )
    assert parameter.grad is None
