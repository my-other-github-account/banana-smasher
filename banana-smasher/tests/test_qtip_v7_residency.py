from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

import banana_smasher.qtip_v7_residency as residency
from banana_smasher.qtip_v7_residency import qtip_v7_resident_weight


def _accounting(tmp_path: Path, *, duplicate: bool = False) -> Path:
    rows = []
    first_wire: Path | None = None
    for layer in range(43):
        wire = tmp_path / f"L{layer:03d}.wire"
        if duplicate and layer == 42:
            assert first_wire is not None
            os.link(first_wire, wire)
        else:
            wire.write_bytes(bytes([layer]) * 10)
            first_wire = first_wire or wire
        receipt = tmp_path / f"L{layer:03d}.receipt.json"
        receipt.write_text(
            json.dumps(
                {
                    "schema": "banana-smasher-qtip-v7-layer-wire-receipt-v1",
                    "layer": layer,
                    "wire": str(wire),
                    "physical_bytes": 10,
                }
            )
        )
        rows.append({"layer": layer, "path": str(receipt), "sha256": f"{layer:064x}"})
    accounting = tmp_path / "accounting.json"
    accounting.write_text(
        json.dumps(
            {
                "schema": "banana-smasher-qtip-v7-model-accounting-v1",
                "status": "PASS",
                "verified_layer_receipts": 43,
                "layer_receipts": rows,
                "qtip_routed_stored_bytes": 430,
                "native_base_bytes": 100,
            }
        )
    )
    return accounting


@pytest.fixture(autouse=True)
def tiny_model_constants(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(residency, "_ROUTED_WIRE_BYTES", 430)
    monkeypatch.setattr(residency, "_NATIVE_BASE_BYTES", 100)
    monkeypatch.setattr(residency, "_EXL_FULL_BYTES", 530)
    monkeypatch.setattr(residency, "_SEPARATE_LUT_BYTES", 86)
    monkeypatch.setattr(residency, "_PROJECTED_RUNTIME_METADATA_BYTES", 24)


def test_projected_receipt_is_conservative_and_never_proven(tmp_path: Path) -> None:
    result = qtip_v7_resident_weight(accounting=_accounting(tmp_path))

    assert result["status"] == "PROJECTED"
    assert result["hardware_gate"] == "OPEN"
    assert result["stored_wire_bytes"] == 430
    assert result["unique_physical_mapped_resident_weight_bytes"] == 430
    assert result["persistent_runtime_metadata_bytes"] == 24
    assert result["separate_lut_tensor_bytes"] == 0
    assert result["duplicate_packed_bytes"] == 0
    assert result["persistent_decoded_state_bytes"] == 0
    assert result["persistent_dense_weight_bytes"] == 0
    assert result["native_base_bytes"] == 100
    assert result["routed_bytes"] == 430
    assert result["full_resident_weight_bytes"] == 530
    assert result["resident_parity_gap_bytes"] == 0
    assert result["resident_parity"] == "UNPROVEN"
    assert result["cuda_allocated_bytes"] is None
    assert result["process_pss_bytes"] is None


def test_mock_readback_structure_can_close_zero_copy_arithmetic(tmp_path: Path) -> None:
    accounting = _accounting(tmp_path)
    readback = tmp_path / "readback.json"
    readback.write_text(
        json.dumps(
            {
                "hardware_readback": True,
                "direct_kernel_dispatch": residency._DIRECT_DISPATCH,
                "direct_dispatch_calls": 43 * 3,
                "native_base_bytes": 100,
                "unique_physical_mapped_resident_weight_bytes": 430,
                "duplicate_packed_bytes": 0,
                "persistent_decoded_state_bytes": 0,
                "persistent_dense_weight_bytes": 0,
                "generic_fallback_calls": 0,
                "separate_lut_tensor_bytes": 0,
                "persistent_runtime_metadata_bytes": 0,
                "transient_workspace_peak_bytes": 12,
                "lut_alias_storage_identity": True,
                "cuda_allocated_bytes": 1,
                "cuda_reserved_bytes": 2,
                "process_rss_bytes": 3,
                "process_pss_bytes": 4,
                "nvml_process_bytes": 5,
                "resident_page_touch_count": 43,
            }
        )
    )

    result = qtip_v7_resident_weight(
        accounting=accounting, hardware_readback=readback
    )
    assert result["status"] == "PROVEN"
    assert result["routed_bytes"] == 430
    assert result["full_resident_weight_bytes"] == 530
    assert result["resident_parity"] == "GREEN"
    assert result["transient_workspace_peak_bytes"] == 12
    assert result["nvml_process_bytes"] == 5


def test_unique_physical_file_dedupe_rejects_duplicate_layer(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="43 unique fixed envelopes"):
        qtip_v7_resident_weight(accounting=_accounting(tmp_path, duplicate=True))


def test_obsolete_total_is_rejected(tmp_path: Path) -> None:
    accounting = _accounting(tmp_path)
    value = json.loads(accounting.read_text())
    value["historical_total"] = 100_000_000_000 + 636_011_256
    accounting.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="obsolete QTIP total"):
        qtip_v7_resident_weight(accounting=accounting)