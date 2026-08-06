from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import pytest
import torch
from safetensors.torch import save_file

from banana_smasher_plugin.contract import PackContractError, load_runtime_contract
from banana_smasher_plugin.repair import (
    apply_dense_norm_repair,
    apply_runtime_repairs,
    load_output_log_gains,
)


def _pack(root: Path) -> Path:
    (root / "planes").mkdir(parents=True)
    (root / "repair").mkdir()
    repair_state = root / "repair/repair_state.safetensors"
    save_file(
        {
            "norms/model.norm": torch.arange(4, dtype=torch.float32),
            "norms/model.layers.0.input_layernorm": torch.arange(
                4, dtype=torch.float32
            )
            + 10,
            "norms/model.layers.0.post_attention_layernorm": torch.arange(
                4, dtype=torch.float32
            )
            + 20,
            "norms/model.layers.0.self_attn.q_a_norm": torch.arange(
                4, dtype=torch.float32
            )
            + 30,
            "outputs/model.layers.0.self_attn.o_b_proj.output_log_gain": torch.tensor(
                0.125, dtype=torch.float32
            ),
        },
        repair_state,
    )
    state_sha = hashlib.sha256(repair_state.read_bytes()).hexdigest()
    repair_manifest = {
        "schema": "bs-repair-materialization-v1",
        "status": "MATERIALIZED",
        "format": "bs-basic-repair-v1",
        "update": 12,
        "dense_state": {
            "path": "repair/repair_state.safetensors",
            "sha256": state_sha,
            "norms": 4,
            "outputs": 1,
            "tensors": [],
        },
    }
    repair_path = root / "repair/REPAIR_MANIFEST.json"
    repair_path.write_text(json.dumps(repair_manifest))
    repair_sha = hashlib.sha256(repair_path.read_bytes()).hexdigest()
    manifest = {
        "schema": "bs-pack",
        "schema_version": 1,
        "source_format": "p1016-true-c-native-planes-v1",
        "quant_method": "banana_smasher",
        "instance_id": "fixture",
        "layers": [0],
        "tensor_layout_sha256": "0dae88283affb718f7b9cd7d6b2f9bd11016fb9b792ecf98ea96dce426ee4cc8",
        "repair": {
            "format": "bs-basic-repair-v1",
            "manifest": "repair/REPAIR_MANIFEST.json",
            "manifest_sha256": repair_sha,
            "state": "repair/repair_state.safetensors",
            "state_sha256": state_sha,
            "norms": 4,
            "outputs": 1,
            "update": 12,
        },
    }
    (root / "BANANA_PACK_MANIFEST.json").write_text(json.dumps(manifest))
    (root / "config.json").write_text(
        json.dumps(
            {
                "quantization_config": {
                    "quant_method": "banana_smasher",
                    "format": "bs-pack",
                    "format_version": 1,
                    "pack_manifest": "BANANA_PACK_MANIFEST.json",
                    "pack_root": ".",
                    "repair_format": "bs-basic-repair-v1",
                    "repair_manifest": "repair/REPAIR_MANIFEST.json",
                    "repair_state": "repair/repair_state.safetensors",
                    "repair_update": 12,
                }
            }
        )
    )
    return root


def test_runtime_contract_accepts_exact_native_repair_pack(tmp_path: Path) -> None:
    pack = _pack(tmp_path / "pack")
    contract = load_runtime_contract(pack)
    assert contract.pack_root == pack.resolve()
    assert contract.layers == (0,)
    assert contract.repair_update == 12


def test_runtime_contract_rejects_layout_lie(tmp_path: Path) -> None:
    pack = _pack(tmp_path / "pack")
    path = pack / "BANANA_PACK_MANIFEST.json"
    value = json.loads(path.read_text())
    value["source_format"] = "iq3-layout"
    path.write_text(json.dumps(value))
    with pytest.raises(PackContractError, match="source_format"):
        load_runtime_contract(pack)


def test_dense_repair_and_output_gains_are_exact(tmp_path: Path) -> None:
    contract = load_runtime_contract(_pack(tmp_path / "pack"))
    module = torch.nn.Module()
    module.model = torch.nn.Module()
    module.model.norm = torch.nn.LayerNorm(4, elementwise_affine=True)
    module.model.layers = torch.nn.ModuleList([torch.nn.Module()])
    module.model.layers[0].attn_norm = torch.nn.LayerNorm(4, elementwise_affine=True)
    module.model.layers[0].ffn_norm = torch.nn.LayerNorm(4, elementwise_affine=True)
    module.model.layers[0].attn = torch.nn.Module()
    module.model.layers[0].attn.q_norm = torch.nn.LayerNorm(4, elementwise_affine=True)
    applied = apply_dense_norm_repair(module, contract)
    assert applied == (
        "model.layers.0.attn.q_norm.weight",
        "model.layers.0.attn_norm.weight",
        "model.layers.0.ffn_norm.weight",
        "model.norm.weight",
    )
    assert torch.equal(module.model.norm.weight, torch.arange(4, dtype=torch.float32))
    assert torch.equal(
        module.model.layers[0].attn_norm.weight,
        torch.arange(4, dtype=torch.float32) + 10,
    )
    assert torch.equal(
        module.model.layers[0].ffn_norm.weight,
        torch.arange(4, dtype=torch.float32) + 20,
    )
    assert torch.equal(
        module.model.layers[0].attn.q_norm.weight,
        torch.arange(4, dtype=torch.float32) + 30,
    )
    gains = load_output_log_gains(contract)
    assert gains == {"model.layers.0.self_attn.o_b_proj": pytest.approx(0.125)}


class RMSNorm(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(4))


def _repair_module(wo_b: Any) -> Any:
    module = torch.nn.Module()
    module.model = torch.nn.Module()
    module.model.norm = RMSNorm()
    module.model.layers = torch.nn.ModuleList([torch.nn.Module()])
    layer = module.model.layers[0]
    layer.attn_norm = RMSNorm()
    layer.ffn_norm = RMSNorm()
    layer.attn = torch.nn.Module()
    layer.attn.q_norm = RMSNorm()
    layer.attn.wo_b = wo_b
    return module


def _block_fp8_linear(
    scale_dtype: torch.dtype,
    *,
    rows: int = 128,
    columns: int = 128,
    block_size: tuple[int, int] = (128, 128),
) -> Any:
    block_n, block_k = block_size
    target = torch.nn.Module()
    target.register_parameter(
        "weight",
        torch.nn.Parameter(
            torch.eye(rows, columns, dtype=torch.float32).to(torch.float8_e4m3fn),
            requires_grad=False,
        ),
    )
    target.register_parameter(
        "weight_scale_inv",
        torch.nn.Parameter(
            torch.ones(math.ceil(rows / block_n), math.ceil(columns / block_k)).to(
                scale_dtype
            ),
            requires_grad=False,
        ),
    )
    setattr(target, "weight_block_size", list(block_size))
    return target


def test_runtime_repairs_fold_output_gain_into_fp8_block_scale_once(
    tmp_path: Path,
) -> None:
    contract = load_runtime_contract(_pack(tmp_path / "pack"))
    wo_b = _block_fp8_linear(torch.float32)
    module = _repair_module(wo_b)
    original_weight = wo_b.weight.detach().clone()

    first = apply_runtime_repairs(module, contract)
    second = apply_runtime_repairs(module, contract)

    assert first == {
        "norms": (
            "model.layers.0.attn.q_norm.weight",
            "model.layers.0.attn_norm.weight",
            "model.layers.0.ffn_norm.weight",
            "model.norm.weight",
        ),
        "output_log_gains": ("model.layers.0.self_attn.o_b_proj",),
    }
    assert second == first
    assert torch.equal(module.model.norm.weight, torch.arange(4, dtype=torch.float32))
    assert torch.equal(
        module.model.layers[0].attn_norm.weight,
        torch.arange(4, dtype=torch.float32) + 10,
    )
    assert torch.equal(
        module.model.layers[0].ffn_norm.weight,
        torch.arange(4, dtype=torch.float32) + 20,
    )
    assert torch.equal(
        module.model.layers[0].attn.q_norm.weight,
        torch.arange(4, dtype=torch.float32) + 30,
    )
    assert torch.equal(wo_b.weight, original_weight)
    assert torch.allclose(
        wo_b.weight_scale_inv,
        torch.full((1, 1), math.exp(0.125)),
    )


def test_runtime_repairs_requantize_fp8_weight_for_e8m0_scale(
    tmp_path: Path,
) -> None:
    contract = load_runtime_contract(_pack(tmp_path / "pack"))
    wo_b = _block_fp8_linear(torch.float8_e8m0fnu)
    wo_b.weight.data.fill_(416.0)
    original = wo_b.weight.float().clone()
    module = _repair_module(wo_b)

    apply_runtime_repairs(module, contract)
    actual = wo_b.weight.float() * wo_b.weight_scale_inv.float().item()
    expected = original * torch.exp(torch.tensor(0.125))
    folded_weight = wo_b.weight.detach().clone()
    folded_scale = wo_b.weight_scale_inv.detach().clone()
    apply_runtime_repairs(module, contract)

    assert wo_b.weight_scale_inv.float().item() == 2.0
    assert torch.equal(wo_b.weight, folded_weight)
    assert torch.equal(wo_b.weight_scale_inv, folded_scale)
    assert torch.all(torch.isfinite(actual))
    assert torch.allclose(actual, expected, rtol=0.03, atol=0.0)


def test_runtime_repairs_requantize_partial_fp8_edge_blocks(tmp_path: Path) -> None:
    contract = load_runtime_contract(_pack(tmp_path / "pack"))
    wo_b = _block_fp8_linear(
        torch.float8_e8m0fnu,
        rows=5,
        columns=6,
        block_size=(4, 4),
    )
    wo_b.weight.data.fill_(416.0)
    original = wo_b.weight.float().clone()

    apply_runtime_repairs(_repair_module(wo_b), contract)
    expanded_scale = wo_b.weight_scale_inv.float().repeat_interleave(
        4, 0
    ).repeat_interleave(4, 1)[:5, :6]
    actual = wo_b.weight.float() * expanded_scale
    expected = original * math.exp(0.125)

    assert tuple(wo_b.weight_scale_inv.shape) == (2, 2)
    assert wo_b.weight_scale_inv.float().unique().tolist() == [2.0]
    assert torch.allclose(actual, expected, rtol=0.03, atol=0.0)
