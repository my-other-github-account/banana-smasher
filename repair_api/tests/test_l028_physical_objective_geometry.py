import torch

from repair_api.modern_green_resident import _physical_training_row


def test_l028_static_w28_keeps_2048_physical_rows_and_1024_objective_span() -> None:
    ids = torch.arange(1024, dtype=torch.int64)

    physical, objective_span = _physical_training_row(
        ids,
        requested_objective_span=1024,
        required_physical_rows=2048,
        pad_token_id=7,
    )

    assert physical.shape == (2048,)
    assert torch.equal(physical[:1024], ids)
    assert torch.equal(physical[1024:], torch.full((1024,), 7, dtype=torch.int64))
    assert objective_span == 1024
