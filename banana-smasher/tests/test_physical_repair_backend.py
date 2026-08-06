from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import torch

import banana_smasher.update as update_module
from banana_smasher.cli import main
from banana_smasher.token_sizing import MemoryBudget
from banana_smasher.update_backends.physical_repair import (
    PhysicalRepairBackend,
    run_physical_repair,
)


def _identity() -> dict[str, str]:
    return {
        "content_sha256": "1" * 64,
        "config_sha256": "2" * 64,
        "assignment_sha256": "3" * 64,
        "aot_sha256": "4" * 64,
        "runtime_sha256": "5" * 64,
        "code_sha256": "6" * 64,
    }


def _write_bundle(tmp_path: Path) -> tuple[Path, str]:
    torch.manual_seed(7)
    hidden = 32
    code_dim = 2
    codes = 4
    tokens = 4

    def projection(rows: int) -> dict[str, torch.Tensor]:
        return {
            "codebook": torch.randn(codes, code_dim, dtype=torch.float32),
            "packed_codes": torch.randint(
                codes, (1, rows, hidden // code_dim), dtype=torch.int16
            ),
            "scales": torch.full((1, rows, hidden // 32), 127, dtype=torch.uint8),
        }

    bundle = {
        "schema": "banana-smasher-physical-repair-bundle-v1",
        "input_ids": torch.zeros(1, tokens, dtype=torch.int64),
        "activation_inputs": torch.randn(1, tokens, hidden, dtype=torch.float32),
        "teacher_targets": torch.zeros(1, tokens, hidden, dtype=torch.float32),
        "teacher_mask": torch.ones(1, tokens, dtype=torch.bool),
        "positions": torch.arange(tokens).reshape(1, tokens),
        "layers": [
            {
                "top_k_index": torch.zeros(tokens, 1, dtype=torch.int32),
                "top_k_weights": torch.ones(tokens, 1, dtype=torch.float32),
                "projections": {"13": projection(hidden * 2), "2": projection(hidden)},
            }
        ],
        "optimizer": {"name": "sgd", "learning_rate": 0.01},
    }
    path = tmp_path / "repair-bundle.pt"
    torch.save(bundle, path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return path, digest


def _context(tmp_path: Path) -> dict[str, object]:
    return {
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


def test_physical_backend_initializes_once_and_runs_one_physical_cycle(tmp_path: Path) -> None:
    bundle, bundle_sha256 = _write_bundle(tmp_path)
    request = {
        "schema": "banana-smasher-physical-repair-request-v1",
        "bundle": str(bundle),
        "bundle_sha256": bundle_sha256,
    }
    backend = PhysicalRepairBackend(request, _context(tmp_path))

    worker = backend.initialize()
    result = backend.cycle(worker, request)

    assert worker["init_receipt"]["init_seconds"] < 180
    assert worker["init_receipt"]["decoded_once"] is True
    assert worker["init_receipt"]["persistent_layouts"] is True
    assert worker["init_receipt"]["source_retired"] is True
    assert result["status"] == "PASS_UPDATE"
    assert result["optimizer_steps"] == 1
    assert result["physical_repair"]["codebooks_changed"] is True
    assert result["physical_repair"]["packed_indices_frozen"] is True
    assert result["physical_repair"]["init_reused"] is True
    assert all(
        value > 0
        for value in result["production_runtime"]["backend_sentinels"].values()
    )
    assert json.loads((tmp_path / "updated.receipt.json").read_text())["status"] == "PASS_UPDATE"


def test_physical_backend_rejects_mission_private_runtime_imports(tmp_path: Path) -> None:
    request = {
        "schema": "banana-smasher-physical-repair-request-v1",
        "runtime_factory": "legacy_trainer:factory",
    }

    with pytest.raises(ValueError, match="bundle"):
        PhysicalRepairBackend(request, _context(tmp_path))


def test_public_cli_runs_standalone_backend_without_old_tree_imports(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    bundle, bundle_sha256 = _write_bundle(tmp_path)
    request = tmp_path / "request.json"
    request.write_text(
        json.dumps(
            {
                "schema": "banana-smasher-physical-repair-request-v1",
                "bundle": str(bundle),
                "bundle_sha256": bundle_sha256,
            }
        )
    )
    identity = tmp_path / "identity.json"
    identity.write_text(json.dumps(_identity()))

    class Entry:
        @staticmethod
        def load():
            return run_physical_repair

    monkeypatch.setattr(update_module, "_update_entry_point", lambda _name: Entry())
    argv = [
        "update",
        "--backend",
        "physical-repair",
        "--request",
        str(request),
        "--identity",
        str(identity),
        "--output",
        str(tmp_path / "cli-updated.pt"),
        "--tokens",
        "2",
        "--segments",
        "2",
        "--available-bytes",
        str(8 * 1024**3),
        "--resident-frozen-bytes",
        "0",
        "--trainable-bytes",
        "0",
        "--optimizer-bytes",
        "0",
        "--staging-bytes",
        "0",
        "--activation-bytes-per-token",
        "1",
    ]

    assert main(argv) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "PASS_UPDATE"
    assert result["physical_repair"]["fallback_used"] is False
    sources = "\n".join(
        (Path(__file__).parents[1] / "src" / "banana_smasher" / "update_backends" / name).read_text()
        for name in ("physical_repair.py", "physical_bundle.py")
    )
    for forbidden in (".hermes", "glm52-humming-w3", "spark-bench-reproducers", "runtime_factory"):
        assert forbidden not in sources


def test_package_registers_physical_repair_update_backend() -> None:
    project = (Path(__file__).parents[1] / "pyproject.toml").read_text()

    assert '[project.entry-points."banana_smasher.update_backends"]' in project
    assert (
        'physical-repair = "banana_smasher.update_backends.physical_repair:'
        'run_physical_repair"'
    ) in project
