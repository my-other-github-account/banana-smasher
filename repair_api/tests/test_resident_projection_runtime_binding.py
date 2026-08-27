from __future__ import annotations

import hashlib
from types import SimpleNamespace

import pytest
import torch

from repair_api import ArtifactError, ResidentRepairAPI
from repair_api.modern_green_resident import (
    _bind_installed_projection_runtime,
    _bind_sealed_gate_up_projection,
)
from repair_api.tests.test_resident_projection_wrapper import (
    Immutable942cProjectionProvider,
    PROVIDER_SHA256,
)


def _bound_config() -> dict[str, object]:
    return ResidentRepairAPI.bind_combined_gate_up_projection(
        {"basis_sha256": "98efab455cf08dfbbbaaba6f570e1bf10bf927d2b4c3c453a59c2f6f0e3be92b"},
        provider_expert_sha256=PROVIDER_SHA256,
        capture_witness=True,
    )


def _wrapped_type(config: dict[str, object]):
    def combined_projection(*args: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = args[0]
        return torch.ones_like(x), torch.full_like(x, 2.0)

    return _bind_sealed_gate_up_projection(
        Immutable942cProjectionProvider,
        config,
        combined_projection=combined_projection,
    )


def test_bound_runtime_replaces_stale_static_trainer_symbol_before_construction() -> None:
    config = _bound_config()
    wrapped = _wrapped_type(config)
    trainer = SimpleNamespace(FullyResidentGroupedV7Experts=Immutable942cProjectionProvider)

    binding = _bind_installed_projection_runtime(trainer, wrapped, config)

    assert trainer.FullyResidentGroupedV7Experts is wrapped
    assert binding == {
        "status": "BOUND_TO_ORDINARY_TRAINER_GLOBAL",
        "implementation": "combined_4096_bf16_f_linear_v1",
        "provider_expert_sha256": PROVIDER_SHA256,
        "runtime_class_marker": "sealed_combined_gate_up_projection_v1",
    }


def test_bound_runtime_refuses_an_unmarked_or_missing_installed_class() -> None:
    config = _bound_config()
    trainer = SimpleNamespace(FullyResidentGroupedV7Experts=Immutable942cProjectionProvider)

    with pytest.raises(ArtifactError, match="installed runtime expert is missing"):
        _bind_installed_projection_runtime(trainer, None, config)
    with pytest.raises(ArtifactError, match="marker mismatch"):
        _bind_installed_projection_runtime(
            trainer, Immutable942cProjectionProvider, config
        )


def test_bound_runtime_emits_fail_closed_activation_and_exact_gate_up_hashes() -> None:
    config = _bound_config()
    wrapped = _wrapped_type(config)
    provider = wrapped()
    hidden = torch.zeros((1, 1), dtype=torch.bfloat16)
    indices = torch.tensor([[0]], dtype=torch.int64)
    weights = torch.ones((1, 1), dtype=torch.bfloat16)

    with pytest.raises(RuntimeError, match="ACTIVATION_MISSING"):
        provider.sealed_gate_up_runtime_witness(require_activation=True)

    provider.forward(hidden, indices, weights)
    witness = provider.sealed_gate_up_runtime_witness(require_activation=True)

    assert witness["activation_count"] == 1
    assert witness["gate"] == {
        "dtype": "torch.bfloat16",
        "shape": [1, 1],
        "sha256": "b9c205bdac187f20bf876cea369cb6032ad1bf69043b31d716b36b8defbffdf2",
    }
    assert witness["up"] == {
        "dtype": "torch.bfloat16",
        "shape": [1, 1],
        "sha256": "b8811852747cfa3620c3dd2af5d59498c240f208e689b4052bac934c29faf094",
    }


def test_bound_runtime_emits_aligned_active_route_gate_up_activated_and_w2_hashes() -> None:
    config = ResidentRepairAPI.bind_combined_gate_up_projection(
        {"basis_sha256": "98efab455cf08dfbbbaaba6f570e1bf10bf927d2b4c3c453a59c2f6f0e3be92b"},
        provider_expert_sha256=PROVIDER_SHA256,
        capture_witness=True,
        active_row_expert=204,
    )
    wrapped = _wrapped_type(config)
    provider = wrapped()
    hidden = torch.tensor([[0.0], [0.0]], dtype=torch.bfloat16)
    indices = torch.tensor([[7, 204], [204, 9]], dtype=torch.int64)
    weights = torch.ones((2, 2), dtype=torch.bfloat16)

    provider.forward(hidden, indices, weights)
    witness = provider.sealed_gate_up_runtime_witness(require_activation=True)
    aligned = witness["aligned_active_rows"]

    def tensor_witness(value: torch.Tensor) -> dict[str, object]:
        contiguous = value.contiguous()
        raw = contiguous.view(torch.uint8).numpy().tobytes()
        return {
            "dtype": str(contiguous.dtype),
            "shape": list(contiguous.shape),
            "sha256": hashlib.sha256(raw).hexdigest(),
        }

    # A27 truth is slot-major then token-major. The immutable provider executes
    # token-major, so the capture must realign rows without changing arithmetic.
    assert aligned["expert"] == 204
    assert aligned["route_key"] == tensor_witness(
        torch.tensor([[0, 1], [1, 0]], dtype=torch.int64)
    )
    assert aligned["gate"] == tensor_witness(torch.ones((2, 1), dtype=torch.bfloat16))
    assert aligned["up"] == tensor_witness(torch.full((2, 1), 2.0, dtype=torch.bfloat16))
    activated = torch.nn.functional.silu(torch.ones((2, 1), dtype=torch.bfloat16)) * 2
    assert aligned["activated"] == tensor_witness(activated)
    assert aligned["w2_down"] == tensor_witness(activated)


def test_a27_aligned_active_row_adjudication_names_first_unequal_boundary() -> None:
    from repair_api.resident_full64_accept import _adjudicate_a27_active_rows

    control = {
        "route_key": "77f7c2dea33f31f549f1170d41b90c6b620e8db99e70807aab5367d50b3ae1ae",
        "gate": "e6a418dbb208c4e83fcb25aa55f4246ff1387aa5abdfa2c15835d10764221e6c",
        "up": "c2350b61f0d24b0b2b2b812df20e5693ebfc699b4afab74f19120c7dba220463",
        "activated": "377fedca101670cf7d35a51579aec7a383d473e98ad21e4bc692c3a1b7b7a150",
        "w2_down": "07e386fb4a72f7ef31f9603e93c26f3bf758e03aec886720875c3516f9447e31",
    }
    witness = {
        "expert": 204,
        **{name: {"sha256": digest} for name, digest in control.items()},
    }

    exact = _adjudicate_a27_active_rows(witness)
    assert exact["first_unequal_boundary"] is None
    assert exact["status"] == "A27_ALIGNED_ACTIVE_ROW_PARITY"

    witness["activated"] = {"sha256": "0" * 64}
    unequal = _adjudicate_a27_active_rows(witness)
    assert unequal["first_unequal_boundary"] == "activated"
    assert unequal["boundaries"]["activated"] == {
        "control_sha256": control["activated"],
        "variant_sha256": "0" * 64,
        "exact": False,
    }


def test_aligned_active_row_tensor_rail_uses_the_declared_cgroup_only_memory_gate() -> None:
    source = (
        __import__("pathlib").Path(__file__).parents[1] / "resident_full64_accept.py"
    ).read_text()
    assert (
        "sealed_runtime_tensor_ab_only\n"
        "        or aligned_active_row_capture_only\n"
        "        or sealed_runtime_expert_trace_ab_only"
    ) in source
    assert "# The accepted R20 tensor rail" in source
