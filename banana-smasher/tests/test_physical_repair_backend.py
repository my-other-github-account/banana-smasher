from __future__ import annotations

import hashlib
import json
import math
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


def _reference_fwht(value: torch.Tensor) -> torch.Tensor:
    width = int(value.shape[-1])
    result = value.contiguous()
    stride = 1
    while stride < width:
        stage = result.reshape(*result.shape[:-1], -1, 2, stride)
        left = stage[..., 0, :]
        right = stage[..., 1, :]
        result = torch.cat((left + right, left - right), dim=-1).reshape_as(result)
        stride *= 2
    return result / math.sqrt(width)


def _reference_qtip2_weight(unit: dict[str, object]) -> torch.Tensor:
    m, k = (int(value) for value in unit["shape"])
    compressed = unit["trellis"].contiguous().view(torch.uint16)
    bits_per_block = 2 * 16 * 16
    compressed = (
        compressed.view(torch.uint8)
        .reshape(m // 32, k // 32, 32, 2, 2, 2)
        .permute(0, -2, 1, -3, 2, -1)
        .flip((-1,))
        .reshape(m // 16, k // 16, bits_per_block // 16, 2)
        .flip((-1,))
        .contiguous()
        .view(torch.uint16)
        .reshape(m // 16, k // 16, bits_per_block // 16)
    )
    blocked = compressed.reshape(2 * m * k // bits_per_block, -1, 1)
    blocked_roll = torch.roll(blocked.to(torch.int32), -1, -2).to(blocked.dtype)
    blocked32 = (
        torch.cat((blocked_roll, blocked), dim=-1)
        .reshape(blocked.shape[0], -1)
        .contiguous()
        .view(torch.uint32)
    )
    expanded32 = blocked32.reshape(*blocked32.shape, 1).expand(
        *blocked32.shape, 16
    ).view(torch.int32)
    shifts = torch.arange(16, dtype=torch.int32).reshape(1, 1, -1)
    states = ((expanded32 >> (16 - shifts)).reshape(expanded32.shape[0], -1)[:, 0::4])
    states = torch.bitwise_and(states, (1 << 16) - 1)
    quadratic = (states + 1) * states
    lut_index = (quadratic >> 6) & ((1 << 9) - 1)
    expanded = unit["tlut"].float()[lut_index]
    expanded[..., 0] *= 1 - ((quadratic >> 15) & 1) * 2
    raw = (
        expanded.reshape(m // 16, k // 16, 16, 16)
        .reshape(m // 16, k // 16, 8, 4, 2, 2, 2)
        .permute(0, -2, 2, 1, -3, 3, -1)
        .reshape(m, k)
    )
    weight = raw * unit["Wscale"].float()
    weight = _reference_fwht(weight.transpose(0, 1)).transpose(0, 1)
    weight = weight * unit["SV"].float()[:, None]
    return _reference_fwht(weight) * unit["SU"].float()


def _write_qtip2_bundle(tmp_path: Path) -> tuple[Path, str]:
    torch.manual_seed(11)
    width = tokens = 32
    unit: dict[str, object] = {
        "schema": "banana-smasher-qtip2-public-unit-v1",
        "shape": [width, width],
        "trellis": torch.randint(
            0,
            1 << 16,
            ((width // 16) * (width // 16), 32),
            dtype=torch.int32,
        ).to(torch.uint16),
        "SU": torch.ones(width, dtype=torch.float16),
        "SV": torch.ones(width, dtype=torch.float16),
        "Wscale": torch.tensor(0.25),
        "tlut": torch.randn(512, 2, dtype=torch.float32),
        "geometry": {
            "L": 16,
            "K": 2,
            "V": 2,
            "tlut_bits": 9,
            "decode_mode": "quantlut_sym",
            "td_x": 16,
            "td_y": 16,
        },
    }
    unit["reconstructed_weight"] = _reference_qtip2_weight(unit).half()
    activations = torch.randn(1, tokens, width)
    bundle = {
        "schema": "banana-smasher-physical-repair-bundle-v1",
        "input_ids": torch.zeros(1, tokens, dtype=torch.int64),
        "activation_inputs": activations,
        "teacher_targets": torch.zeros(1, tokens, width),
        "teacher_mask": torch.ones(1, tokens, dtype=torch.bool),
        "positions": torch.arange(tokens).reshape(1, tokens),
        "layers": [unit],
        "optimizer": {"name": "sgd", "learning_rate": 1e-4},
    }
    path = tmp_path / "qtip2-repair-bundle.pt"
    torch.save(bundle, path)
    return path, hashlib.sha256(path.read_bytes()).hexdigest()


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


def test_physical_backend_repairs_genuine_qtip2_tlut_without_changing_trellis(
    tmp_path: Path,
) -> None:
    bundle, bundle_sha256 = _write_qtip2_bundle(tmp_path)
    packed_wire = torch.load(bundle, weights_only=False)["layers"][0]["trellis"]
    request = {
        "schema": "banana-smasher-physical-repair-request-v1",
        "bundle": str(bundle),
        "bundle_sha256": bundle_sha256,
    }
    backend = PhysicalRepairBackend(request, _context(tmp_path))

    worker = backend.initialize()
    before = [value.detach().clone() for value in worker["packed_indices"]]
    resident_wire = worker["packed_indices"][0].detach().cpu()
    result = backend.cycle(worker, request)

    assert resident_wire.dtype == packed_wire.dtype == torch.uint16
    assert tuple(resident_wire.shape) == tuple(packed_wire.shape)
    assert torch.equal(
        resident_wire.contiguous().view(torch.uint8).reshape(-1),
        packed_wire.contiguous().view(torch.uint8).reshape(-1),
    )
    assert result["status"] == "PASS_UPDATE"
    assert result["physical_repair"]["codebook_tensors"] == 1
    assert result["physical_repair"]["packed_indices_frozen"] is True
    assert all(
        torch.equal(expected, observed)
        for expected, observed in zip(before, worker["packed_indices"])
    )
    sentinels = result["production_runtime"]["backend_sentinels"]
    assert sentinels["fwht"] > 0
    assert sentinels["kmajor_batch"] == 0
    assert sentinels["kmajor_fused"] == 0
    assert sentinels["grouped_vjp"] == 0
    assert sentinels["layer_graph"] == 0


@pytest.mark.parametrize("dtype", [torch.uint8, torch.int32, torch.int64])
def test_physical_backend_rejects_noncanonical_qtip2_trellis_dtype(
    tmp_path: Path, dtype: torch.dtype
) -> None:
    bundle, _ = _write_qtip2_bundle(tmp_path)
    document = torch.load(bundle, weights_only=False)
    document["layers"][0]["trellis"] = document["layers"][0]["trellis"].to(dtype)
    torch.save(document, bundle)
    bundle_sha256 = hashlib.sha256(bundle.read_bytes()).hexdigest()
    request = {
        "schema": "banana-smasher-physical-repair-request-v1",
        "bundle": str(bundle),
        "bundle_sha256": bundle_sha256,
    }

    with pytest.raises(ValueError, match="canonical torch.uint16 wire"):
        PhysicalRepairBackend(request, _context(tmp_path)).initialize()


def test_qtip2_backend_sentinels_are_reset_at_physical_cycle_boundary(
    tmp_path: Path,
) -> None:
    def run_cycle(directory: Path, *, pollute: bool) -> tuple[dict[str, int], int]:
        directory.mkdir()
        bundle, bundle_sha256 = _write_qtip2_bundle(directory)
        request = {
            "schema": "banana-smasher-physical-repair-request-v1",
            "bundle": str(bundle),
            "bundle_sha256": bundle_sha256,
        }
        backend = PhysicalRepairBackend(request, _context(directory))
        worker = backend.initialize()
        reset_calls = 0
        reset_backend_sentinels = worker["bundle"]["reset_backend_sentinels"]

        def record_reset() -> None:
            nonlocal reset_calls
            reset_calls += 1
            reset_backend_sentinels()

        worker["bundle"]["reset_backend_sentinels"] = record_reset
        if pollute:
            layer = worker["runtime"].layers[0]
            for _ in range(3):
                layer(torch.randn(1, 1, 32)).sum().backward()
                layer.tlut.grad = None
        result = backend.cycle(worker, request)
        return result["production_runtime"]["backend_sentinels"], reset_calls

    clean, clean_reset_calls = run_cycle(tmp_path / "clean", pollute=False)
    polluted, polluted_reset_calls = run_cycle(tmp_path / "polluted", pollute=True)

    assert clean_reset_calls == polluted_reset_calls == 1
    assert polluted == clean
    assert clean["fwht"] > 0
    assert all(
        clean[name] == 0
        for name in ("kmajor_batch", "kmajor_fused", "grouped_vjp", "layer_graph")
    )


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
