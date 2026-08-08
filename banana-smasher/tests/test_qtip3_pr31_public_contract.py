from __future__ import annotations

import hashlib
import importlib
import importlib.util
from pathlib import Path

import pytest
import torch


PROVIDER_ID = "periodic-qtip3@3.00"
LEGACY_PROVIDER_ID = "qtip-native-v6@3.00"
LUT_ID = "pr31-affine-gaussian-edge-v1"
BASIS_SHA = "98efab455cf08dfbbbaaba6f570e1bf10bf927d2b4c3c453a59c2f6f0e3be92b"


def _qtip3_fixed_module():
    spec = importlib.util.find_spec("banana_smasher.qtip3_fixed")
    assert spec is not None, "public fixed-assignment QTIP3 module is missing"
    return importlib.import_module("banana_smasher.qtip3_fixed")


def _write_member(
    tmp_path: Path, *, provider_id: str = PROVIDER_ID
) -> tuple[Path, Path]:
    lut = torch.linspace(-1.0, 1.0, 1024, dtype=torch.float16)
    lut_path = tmp_path / "pr31-lut.pt"
    torch.save(lut, lut_path)
    member = {
        "schema": "banana-smasher-qtip3-fixed-member-v1",
        "codec_provider_id": provider_id,
        "basis_index_sha256": BASIS_SHA,
        "source_weight_sha256": "1" * 64,
        "hessian_sha256": "2" * 64,
        "geometry": {
            "L": 16,
            "B": 12,
            "V": 4,
            "layout": "homogeneous",
            "phase_widths": [3, 3, 3, 3],
        },
        "lut": {
            "identity": LUT_ID,
            "tensor_sha256": hashlib.sha256(lut.numpy().tobytes()).hexdigest(),
            "data_bytes": lut.numel() * lut.element_size(),
        },
        "codes": torch.arange(48, dtype=torch.uint8),
        "SU": torch.ones(8, dtype=torch.float16),
        "SV": torch.ones(6, dtype=torch.float16),
        "Wscale": torch.tensor(0.75, dtype=torch.float32),
    }
    member_path = tmp_path / "QTIP3_MEMBER.pt"
    torch.save(member, member_path)
    return member_path, lut_path


def test_qtip3_member_binds_owned_pr31_lut_and_exact_fixed_geometry(tmp_path: Path) -> None:
    qtip3 = _qtip3_fixed_module()
    member_path, lut_path = _write_member(tmp_path)

    loaded = qtip3.load_qtip3_fixed_member(member_path, lut_path=lut_path)

    assert loaded.codec_provider_id == PROVIDER_ID
    assert loaded.geometry == {
        "L": 16,
        "B": 12,
        "V": 4,
        "layout": "homogeneous",
        "phase_widths": [3, 3, 3, 3],
    }
    assert loaded.lut_identity == LUT_ID
    assert loaded.lut_tensor_sha256 == hashlib.sha256(loaded.lut.numpy().tobytes()).hexdigest()
    assert loaded.codes.dtype == torch.uint8
    assert loaded.codes.requires_grad is False
    assert loaded.Wscale.dtype == torch.float32

    tampered = torch.load(lut_path, weights_only=True)
    tampered[0] += 1
    torch.save(tampered, lut_path)
    with pytest.raises(ValueError, match="LUT tensor SHA-256 mismatch"):
        qtip3.load_qtip3_fixed_member(member_path, lut_path=lut_path)


def test_qtip3_legacy_identity_is_accepted_without_reinterpretation(tmp_path: Path) -> None:
    qtip3 = _qtip3_fixed_module()
    member_path, lut_path = _write_member(tmp_path, provider_id=LEGACY_PROVIDER_ID)
    payload = torch.load(member_path, weights_only=True)
    payload["geometry"].pop("layout")
    torch.save(payload, member_path)

    loaded = qtip3.load_qtip3_fixed_member(member_path, lut_path=lut_path)

    assert loaded.codec_provider_id == LEGACY_PROVIDER_ID
    assert "layout" not in loaded.geometry


def test_public_backpack_registry_exposes_qtip3_fixed_assignment_lifecycle() -> None:
    _qtip3_fixed_module()
    from banana_smasher import (
        backpack_provider_from_declaration,
        builtin_backpack_family_providers,
    )

    bindings = builtin_backpack_family_providers()
    assert PROVIDER_ID in bindings
    assert LEGACY_PROVIDER_ID in bindings
    binding = bindings[PROVIDER_ID]
    assert binding.provider_id == PROVIDER_ID
    assert binding.kind == "fixed_qtip"
    assert binding.runtime_family == "periodic_qtip3"
    assert backpack_provider_from_declaration(PROVIDER_ID) == binding
    assert (
        backpack_provider_from_declaration(LEGACY_PROVIDER_ID).provider_id
        == LEGACY_PROVIDER_ID
    )
    assert all(
        callable(value)
        for value in (
            binding.generate,
            binding.materialize,
            binding.price,
            binding.predict,
            binding.verify,
        )
    )


def test_qtip3_public_repair_microdose_changes_only_authorized_continuous_state(
    tmp_path: Path,
) -> None:
    qtip3 = _qtip3_fixed_module()
    member_path, lut_path = _write_member(tmp_path)
    member = qtip3.load_qtip3_fixed_member(member_path, lut_path=lut_path)
    runtime = qtip3.Qtip3FixedRepairRuntime(
        members=[member], learning_rate=0.05, device="cpu"
    )
    codes_before = member.codes.clone()
    geometry_before = dict(member.geometry)
    lut_before = runtime.shared_lut.detach().clone()

    receipt = runtime.microdose(
        activation_inputs=torch.ones((1, 4, 8), dtype=torch.float32),
        teacher_targets=torch.zeros((1, 4, 6), dtype=torch.float32),
        teacher_mask=torch.ones((1, 4), dtype=torch.bool),
    )

    assert receipt["status"] == "PASS_UPDATE"
    assert receipt["finite_nonzero_gradients"] is True
    assert receipt["authorized_parameter_delta"] > 0
    assert receipt["packed_codes_unchanged"] is True
    assert receipt["geometry_unchanged"] is True
    assert receipt["acceleration_counters"]["periodic_qtip3_lut_vjp_calls"] > 0
    assert receipt["acceleration_counters"]["fallback_calls"] == 0
    assert torch.equal(member.codes, codes_before)
    assert member.geometry == geometry_before
    assert not torch.equal(runtime.shared_lut.detach(), lut_before)

    project = (Path(__file__).parents[1] / "pyproject.toml").read_text()
    assert '[project.entry-points."banana_smasher.update_backends"]' in project
    assert "periodic-qtip3" in project
