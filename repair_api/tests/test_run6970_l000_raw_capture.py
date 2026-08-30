from __future__ import annotations

import torch

from repair_api.diagnostics.run6970_l000_moe_comparator import (
    _persist_l000_raw_tensors,
)


def test_persist_l000_raw_tensors_preserves_same_forward_values(tmp_path):
    mlp_input = torch.arange(12, dtype=torch.bfloat16).reshape(1, 3, 4)
    control = mlp_input + 1
    product = mlp_input + 2

    receipt = _persist_l000_raw_tensors(
        tmp_path,
        0,
        mlp_input=mlp_input,
        control_mlp_output=control,
        product_mlp_output=product,
    )

    expected = {
        "pre_layer_input": mlp_input,
        "true_builder_mlp_output": control,
        "product_mlp_output": product,
    }
    assert receipt.keys() == expected.keys()
    for name, tensor in expected.items():
        path = tmp_path / "raw_l000_same_forward" / f"{name}.rank0.pt"
        assert torch.equal(torch.load(path, weights_only=True), tensor)
        assert receipt[name]["path"] == str(path)
        assert len(receipt[name]["file_sha256"]) == 64