from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import torch

from banana_smasher.ff0731_gate_runtime import FF0731GateRuntimeAdapter
from banana_smasher.ff0731_gate_smoke import _manifest_component_expectations
from banana_smasher.gate_only_trainer import (
    final_logit_teacher_kld,
    straight_through_categorical,
)


def test_qtip2_byte_only_component_list_uses_container_fallback() -> None:
    assert _manifest_component_expectations(
        [{"name": "SU", "bytes": 4096}, {"name": "trellis", "bytes": 2097152}]
    ) == {}


class _Layer(torch.nn.Module):
    def forward(
        self,
        activation: torch.Tensor,
        gates: torch.Tensor,
        hard_tiers: torch.Tensor,
    ) -> torch.Tensor:
        del hard_tiers
        branch_values = torch.tensor(
            [[0.0, 1.0, -2.0], [0.0, -1.0, 2.0]],
            dtype=activation.dtype,
            device=activation.device,
        )
        return activation + torch.sum(gates * branch_values)


class _Head(torch.nn.Module):
    def forward(self, activation: torch.Tensor) -> torch.Tensor:
        return torch.stack((activation, -activation))


class _BFloat16Head(torch.nn.Module):
    weight: torch.Tensor

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("weight", torch.tensor([[1.0], [-1.0]], dtype=torch.bfloat16))

    def forward(self, activation: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.linear(activation, self.weight)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _file(path: Path, payload: bytes) -> dict[str, object]:
    path.write_bytes(payload)
    return {"path": str(path), "bytes": len(payload), "sha256": _sha256(path)}


def _runtime_inputs(
    tmp_path: Path,
) -> tuple[dict[str, object], Path, dict[str, object]]:
    layer_path = tmp_path / "layer.pt"
    head_path = tmp_path / "head.pt"
    example_activation = torch.tensor(0.0)
    example_gates = torch.zeros((2, 3))
    example_tiers = torch.zeros(2, dtype=torch.long)
    layer_module = torch.jit.trace(
        _Layer(), (example_activation, example_gates, example_tiers)
    )
    head_module = torch.jit.trace(_Head(), example_activation)
    torch.jit.save(layer_module, str(layer_path))
    torch.jit.save(head_module, str(head_path))

    activation_path = tmp_path / "activation.pt"
    teacher_path = tmp_path / "teacher.pt"
    torch.save(torch.tensor(0.0), activation_path)
    torch.save(torch.tensor([2.0, -2.0]), teacher_path)

    rows = []
    for cell_index, projection in enumerate(("down", "fused13")):
        tiers = {}
        for tier_index, tier in enumerate(("native_mxfp4", "qtip2", "qtip3")):
            artifact = _file(
                tmp_path / f"cell-{cell_index}-{tier}.bin",
                f"{cell_index}:{tier}".encode(),
            )
            tiers[tier] = {
                "wire_bytes": tier_index + 1,
                "artifacts": [artifact],
            }
        rows.append(
            {
                "cell_id": f"L000.E000.{projection}",
                "layer": 0,
                "expert": 0,
                "projection": projection,
                "tiers": tiers,
            }
        )
    physical_manifest = tmp_path / "physical.json"
    physical_manifest.write_text(
        json.dumps(
            {
                "schema": "banana-smasher-ff0731-three-tier-cells-v1",
                "status": "PASS",
                "basis_sha256": "a" * 64,
                "tiers": ["native_mxfp4", "qtip2", "qtip3"],
                "cells": rows,
            },
            sort_keys=True,
        )
        + "\n"
    )

    parameters: dict[str, object] = {
        "schema": "banana-smasher-ff0731-torchscript-gate-runtime-v1",
        "strict_geometry": False,
        "physical_manifest": {
            "path": str(physical_manifest),
            "sha256": _sha256(physical_manifest),
        },
        "layers": [
            {
                "layer": 0,
                "cell_ids": [row["cell_id"] for row in rows],
                "module": {"path": str(layer_path), "sha256": _sha256(layer_path)},
            }
        ],
        "final_head": {"path": str(head_path), "sha256": _sha256(head_path)},
        "verify_payloads": True,
    }
    batch = {
        "window_id": "train-agentic",
        "class": "agentic",
        "activation": {"path": str(activation_path), "sha256": _sha256(activation_path)},
        "teacher_logits": {"path": str(teacher_path), "sha256": _sha256(teacher_path)},
    }
    return parameters, physical_manifest, batch


def test_final_head_casts_fp32_activation_to_frozen_bfloat16_dtype(tmp_path: Path) -> None:
    parameters, _, _ = _runtime_inputs(tmp_path)
    head_path = tmp_path / "bfloat16-head.pt"
    head_module = torch.jit.trace(_BFloat16Head(), torch.zeros(1, dtype=torch.bfloat16))
    torch.jit.save(head_module, str(head_path))
    parameters["final_head"] = {"path": str(head_path), "sha256": _sha256(head_path)}
    runtime = FF0731GateRuntimeAdapter(
        model_root=tmp_path,
        basis_sha256="a" * 64,
        parameters=parameters,
    )

    logits = runtime.final_logits(torch.ones(1, dtype=torch.float32), window_id="mixed")

    assert logits.dtype == torch.bfloat16


def test_concrete_runtime_executes_hard_branches_and_backpropagates_kld(
    tmp_path: Path,
) -> None:
    parameters, _, batch = _runtime_inputs(tmp_path)
    runtime = FF0731GateRuntimeAdapter(
        model_root=tmp_path,
        basis_sha256="a" * 64,
        parameters=parameters,
    )
    logits = torch.zeros((2, 3), requires_grad=True)
    gates, _ = straight_through_categorical(logits, temperature=1.0)
    hard_tiers = gates.detach().argmax(dim=-1)

    activation = runtime.initial(batch)
    with runtime.layer_stage(0) as layer_forward:
        activation = layer_forward(
            activation,
            gates=gates,
            hard_tiers=hard_tiers,
            window_id=batch["window_id"],
        )
    loss = final_logit_teacher_kld(
        runtime.final_logits(activation, window_id=batch["window_id"]),
        runtime.teacher_logits(batch),
    )
    loss.backward()

    assert runtime.cell_ids == ("L000.E000.down", "L000.E000.fused13")
    assert runtime.tier_bytes.tolist() == [[1, 2, 3], [1, 2, 3]]
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()
    assert torch.linalg.vector_norm(logits.grad) > 0
    assert runtime.execution_trace == [
        {
            "layer": 0,
            "window_id": "train-agentic",
            "hard_forward": True,
            "tier_counts": {"native_mxfp4": 2, "qtip2": 0, "qtip3": 0},
        }
    ]
    assert runtime.frozen_state()


def test_runtime_rejects_changed_physical_payload(tmp_path: Path) -> None:
    parameters, physical_manifest, _ = _runtime_inputs(tmp_path)
    document = json.loads(physical_manifest.read_text())
    payload_path = Path(document["cells"][0]["tiers"]["qtip3"]["artifacts"][0]["path"])
    payload_path.write_bytes(b"changed")

    with pytest.raises(ValueError, match=r"artifact 0 SHA-256 mismatch"):
        FF0731GateRuntimeAdapter(
            model_root=tmp_path,
            basis_sha256="a" * 64,
            parameters=parameters,
        )


def test_production_runtime_requires_the_sole_ff0731_model_root(tmp_path: Path) -> None:
    parameters, _, _ = _runtime_inputs(tmp_path)
    parameters["strict_geometry"] = True

    with pytest.raises(ValueError, match="sole model root"):
        FF0731GateRuntimeAdapter(
            model_root=tmp_path,
            basis_sha256="a" * 64,
            parameters=parameters,
        )
