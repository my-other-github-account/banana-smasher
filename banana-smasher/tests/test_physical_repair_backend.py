from __future__ import annotations

import json
from pathlib import Path

import torch

from banana_smasher.token_sizing import MemoryBudget
from banana_smasher.update_backends.physical_repair import PhysicalRepairBackend


class _PhysicalLayer(torch.nn.Module):
    def __init__(self, counters: dict[str, int]) -> None:
        super().__init__()
        self.codebook = torch.nn.Parameter(torch.tensor(1.0))
        self.counters = counters

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        for name in self.counters:
            self.counters[name] += 1
        return hidden * self.codebook


class _FixtureRuntime:
    def __init__(self) -> None:
        self.calls = {
            "authenticate_aot": 0,
            "decode_packed_indices": 0,
            "build_persistent_layer_layouts": 0,
            "stage_inputs": 0,
            "configure_depth_checkpointing": 0,
        }
        self.sentinels = {
            "kmajor_batch": 0,
            "kmajor_fused": 0,
            "grouped_vjp": 0,
            "layer_graph": 0,
            "fwht": 0,
        }
        self.layer = _PhysicalLayer(self.sentinels)
        self.frozen = torch.nn.Module()

    def authenticate_aot(self) -> dict[str, object]:
        self.calls["authenticate_aot"] += 1
        return {"status": "PASS_AUTHENTICATED_AOT", "sha256": "a" * 64}

    def decode_packed_indices(self) -> dict[str, object]:
        self.calls["decode_packed_indices"] += 1
        packed = torch.tensor([0, 1, 1, 0], dtype=torch.int32)
        self.frozen.register_buffer("packed_indices", packed)
        return {
            "status": "PASS_DECODED_ONCE",
            "tensors": [packed],
            "decoded_bytes": packed.numel() * packed.element_size(),
        }

    def build_persistent_layer_layouts(self, packed_indices) -> dict[str, object]:
        self.calls["build_persistent_layer_layouts"] += 1
        return {
            "status": "PASS_PERSISTENT_LAYOUTS",
            "layers": 1,
            "persistent": True,
            "packed_tensor_ids": [id(value) for value in packed_indices],
        }

    def stage_inputs(self, *, largest_first: bool) -> dict[str, object]:
        self.calls["stage_inputs"] += 1
        assert largest_first is True
        tokens = torch.arange(4, dtype=torch.float32).reshape(1, 4)
        targets = torch.zeros((1, 4, 1), dtype=torch.float32)
        return {
            "status": "PASS_STAGED_LARGEST_FIRST",
            "stage_order_nbytes": [targets.numel() * targets.element_size(), tokens.numel() * tokens.element_size()],
            "input_ids": tokens,
            "teacher_targets": targets,
            "teacher_mask": torch.ones((1, 4), dtype=torch.bool),
            "positions": torch.arange(4).reshape(1, 4),
        }

    def configure_depth_checkpointing(self, *, required: bool) -> dict[str, object]:
        self.calls["configure_depth_checkpointing"] += 1
        assert required is True
        return {"status": "PASS_DEPTH_CHECKPOINTING", "depth_groups": 1}

    def update_bundle(self) -> dict[str, object]:
        return {
            "layers": [self.layer],
            "codebooks": [self.layer.codebook],
            "frozen_modules": [self.frozen],
            "encode": lambda segment: segment["input_ids"].unsqueeze(-1),
            "loss_sum": lambda hidden, segment: (
                (hidden - segment["teacher_targets"])
                .masked_select(segment["teacher_mask"].unsqueeze(-1))
                .square()
                .sum()
            ),
            "optimizer_factory": lambda parameters: torch.optim.SGD(parameters, lr=0.01),
            "backend_sentinels": lambda: self.sentinels,
            "peak_memory_bytes": 1024,
            "synchronize": lambda: None,
        }


def fixture_runtime_factory(_request: dict[str, object], _context: dict[str, object]):
    return _FixtureRuntime()


def _identity() -> dict[str, str]:
    return {
        "content_sha256": "1" * 64,
        "config_sha256": "2" * 64,
        "assignment_sha256": "3" * 64,
        "aot_sha256": "4" * 64,
        "runtime_sha256": "5" * 64,
        "code_sha256": "6" * 64,
    }


def test_physical_backend_initializes_once_and_runs_one_physical_cycle(tmp_path: Path) -> None:
    request = {
        "schema": "banana-smasher-physical-repair-request-v1",
        "runtime_factory": f"{__name__}:fixture_runtime_factory",
    }
    context = {
        "output": tmp_path / "updated.pt",
        "receipt": tmp_path / "updated.receipt.json",
        "identity": _identity(),
        "requested_tokens": 2,
        "physical_tokens": 2,
        "segments": 2,
        "batch_size": 1,
        "memory_sizing": {"physical_tokens": 2},
        "memory_budget": MemoryBudget(
            available_bytes=8 * 1024**3,
            resident_frozen_bytes=0,
            trainable_bytes=0,
            optimizer_bytes=0,
            staging_bytes=0,
            calibrated_activation_bytes_per_token=1,
        ),
        "resume": True,
        "restart": False,
    }
    backend = PhysicalRepairBackend(request, context)

    worker = backend.initialize()
    result = backend.cycle(worker, request)

    runtime = worker["runtime"]
    assert runtime.calls == {
        "authenticate_aot": 1,
        "decode_packed_indices": 1,
        "build_persistent_layer_layouts": 1,
        "stage_inputs": 1,
        "configure_depth_checkpointing": 1,
    }
    assert worker["init_receipt"]["init_seconds"] < 180
    assert worker["init_receipt"]["decoded_once"] is True
    assert worker["init_receipt"]["persistent_layouts"] is True
    assert result["status"] == "PASS_UPDATE"
    assert result["optimizer_steps"] == 1
    assert result["physical_repair"]["codebooks_changed"] is True
    assert result["physical_repair"]["packed_indices_frozen"] is True
    assert result["physical_repair"]["init_reused"] is True
    assert json.loads((tmp_path / "updated.receipt.json").read_text())["status"] == "PASS_UPDATE"


def test_package_registers_physical_repair_update_backend() -> None:
    project = (Path(__file__).parents[1] / "pyproject.toml").read_text()

    assert '[project.entry-points."banana_smasher.update_backends"]' in project
    assert (
        'physical-repair = "banana_smasher.update_backends.physical_repair:'
        'run_physical_repair"'
    ) in project
