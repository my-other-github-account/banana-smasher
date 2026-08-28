from pathlib import Path

import torch


SOURCE = Path(__file__).resolve().parents[1] / "modern_green_resident.py"


def _bf16_leaf() -> torch.Tensor:
    return torch.tensor(1.0, dtype=torch.bfloat16, requires_grad=True)


def test_bf16_pre_backward_underflow_is_removed_from_resident_path() -> None:
    control = _bf16_leaf()
    (control * (2.0 ** -192)).backward()
    assert torch.isfinite(control.grad).all()
    assert control.grad.item() == 0.0

    candidate = _bf16_leaf()
    candidate.backward()
    assert torch.isfinite(candidate.grad).all()
    assert candidate.grad.item() != 0.0

    source = SOURCE.read_text()
    assert "loss * self.equivalent_gradient_scale" not in source
    assert "gradient_scale=self.equivalent_gradient_scale" in source
    assert "self.equivalent_gradient_scale = 1.0" in source


def test_public_api_admits_only_the_exact_w56_repair_boundary() -> None:
    api_source = (SOURCE.parent / "api.py").read_text()
    assert "valid_authenticated_u40_u41_w56_repair" in api_source
    assert '== "pre-backward-underflow-removal-v1"' in api_source
    assert '== "c908dfef579e6c47dafea508fde13730ba3286d40fc19d4f161432f48082e8f6"' in api_source
    assert 'config.get("diagnostic_train_windows") == [56, 28]' in api_source
    assert 'config.get("train_windows") == [56, 28]' in api_source
