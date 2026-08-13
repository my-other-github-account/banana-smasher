"""Honest projected or hardware-read QTIP V7 resident-weight accounting."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Sequence


_SCHEMA = "banana-smasher-qtip-v7-resident-weight-v1"
_ACCOUNTING_SCHEMA = "banana-smasher-qtip-v7-model-accounting-v1"
_NATIVE_BASE_BYTES = 19_708_797_688
_ROUTED_WIRE_BYTES = 69_662_278_656
_EXL_FULL_BYTES = 89_371_076_344
_SEPARATE_LUT_BYTES = 43 * 2_048
_PROJECTED_RUNTIME_METADATA_BYTES = 43 * 3 * 256 * 8
_STALE_TOTAL = 100_000_000_000 + 636_011_256
_DIRECT_DISPATCH = "banana_smasher_plugin._v4_moe.qtip2_v7_direct"


def _contains_stale_total(value: object) -> bool:
    if value == _STALE_TOTAL:
        return True
    if isinstance(value, dict):
        return any(_contains_stale_total(item) for item in value.values())
    if isinstance(value, list):
        return any(_contains_stale_total(item) for item in value)
    return False


def _load_object(path: str | Path) -> tuple[Path, dict[str, Any]]:
    resolved = Path(path).expanduser().resolve()
    value = json.loads(resolved.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"QTIP V7 JSON input must contain an object: {resolved}")
    if _contains_stale_total(value):
        raise ValueError("obsolete QTIP total is forbidden")
    return resolved, value


def _unique_wire_bytes(rows: Sequence[dict[str, Any]]) -> tuple[int, int]:
    identities: dict[tuple[int, int], int] = {}
    for row in rows:
        wire = row.get("wire")
        if not isinstance(wire, str):
            raise ValueError("QTIP V7 layer receipt is missing its physical wire")
        path = Path(wire).expanduser().resolve()
        stat = path.stat()
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"QTIP V7 physical wire is missing or unsafe: {path}")
        declared = row.get("physical_bytes")
        if declared != stat.st_size:
            raise ValueError(f"QTIP V7 physical wire byte count drift: {path}")
        identity = (stat.st_dev, stat.st_ino)
        prior = identities.setdefault(identity, stat.st_size)
        if prior != stat.st_size:
            raise ValueError(f"QTIP V7 physical identity changed size: {path}")
    return sum(identities.values()), len(identities)


def qtip_v7_resident_weight(
    *,
    accounting: str | Path,
    output: str | Path | None = None,
    hardware_readback: str | Path | None = None,
) -> dict[str, Any]:
    """Reconcile unique mapped bytes; PROVEN requires explicit hardware readback."""
    accounting_path, model = _load_object(accounting)
    rows = model.get("layer_receipts")
    if (
        model.get("schema") != _ACCOUNTING_SCHEMA
        or model.get("status") != "PASS"
        or model.get("verified_layer_receipts") != 43
        or model.get("qtip_routed_stored_bytes") != _ROUTED_WIRE_BYTES
        or model.get("native_base_bytes") != _NATIVE_BASE_BYTES
        or not isinstance(rows, list)
        or len(rows) != 43
        or [row.get("layer") for row in rows if isinstance(row, dict)] != list(range(43))
    ):
        raise ValueError("QTIP V7 residency requires exact PASS physical layers 0..42")
    typed_rows = []
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("path"), str):
            raise ValueError("QTIP V7 model accounting has an invalid layer receipt row")
        receipt_path, receipt = _load_object(row["path"])
        if (
            receipt.get("layer") != row.get("layer")
            or receipt.get("schema")
            != "banana-smasher-qtip-v7-layer-wire-receipt-v1"
        ):
            raise ValueError(f"QTIP V7 layer receipt binding drift: {receipt_path}")
        typed_rows.append(receipt)
    unique_wire_bytes, unique_files = _unique_wire_bytes(typed_rows)
    if unique_wire_bytes != _ROUTED_WIRE_BYTES or unique_files != 43:
        raise ValueError(
            "QTIP V7 residency requires 43 unique fixed envelopes totaling "
            f"{_ROUTED_WIRE_BYTES} bytes"
        )

    measured: dict[str, Any] | None = None
    if hardware_readback is not None:
        _, measured = _load_object(hardware_readback)
        required = {
            "hardware_readback": True,
            "direct_kernel_dispatch": _DIRECT_DISPATCH,
            "direct_dispatch_calls": 43 * 3,
            "native_base_bytes": _NATIVE_BASE_BYTES,
            "unique_physical_mapped_resident_weight_bytes": unique_wire_bytes,
            "duplicate_packed_bytes": 0,
            "persistent_decoded_state_bytes": 0,
            "persistent_dense_weight_bytes": 0,
            "generic_fallback_calls": 0,
        }
        for field, expected in required.items():
            if measured.get(field) != expected:
                raise ValueError(
                    f"QTIP V7 hardware readback {field} mismatch: "
                    f"{measured.get(field)!r} != {expected!r}"
                )

    separate_lut_bytes = (
        0
        if measured is None
        else int(measured.get("separate_lut_tensor_bytes", -1))
    )
    if separate_lut_bytes not in (0, _SEPARATE_LUT_BYTES):
        raise ValueError("QTIP V7 separate LUT readback must be zero or exactly 88,064 bytes")
    runtime_metadata = (
        _PROJECTED_RUNTIME_METADATA_BYTES
        if measured is None
        else int(measured.get("persistent_runtime_metadata_bytes", -1))
    )
    transient_peak = (
        3 * 768 * 12_292
        if measured is None
        else int(measured.get("transient_workspace_peak_bytes", -1))
    )
    routed = unique_wire_bytes + separate_lut_bytes
    full = _NATIVE_BASE_BYTES + routed

    def telemetry(name: str) -> int | None:
        if measured is None or measured.get(name) is None:
            return None
        value = int(measured[name])
        if value < 0:
            raise ValueError(f"QTIP V7 hardware readback {name} must be nonnegative")
        return value

    telemetry_fields = (
        "cuda_allocated_bytes",
        "cuda_reserved_bytes",
        "process_rss_bytes",
        "process_pss_bytes",
        "nvml_process_bytes",
    )
    telemetry_values = {name: telemetry(name) for name in telemetry_fields}
    if measured is not None and (
        runtime_metadata < 0
        or transient_peak <= 0
        or int(measured.get("resident_page_touch_count", 0)) <= 0
        or any(value is None for value in telemetry_values.values())
    ):
        raise ValueError("QTIP V7 hardware readback is missing runtime memory telemetry")

    result = {
        "schema": _SCHEMA,
        "status": "PROJECTED" if measured is None else "PROVEN",
        "hardware_gate": "OPEN" if measured is None else "READBACK_COMPLETE",
        "accounting": str(accounting_path),
        "stored_wire_bytes": unique_wire_bytes,
        "unique_physical_mapped_resident_weight_bytes": unique_wire_bytes,
        "unique_physical_file_count": unique_files,
        "persistent_runtime_metadata_bytes": runtime_metadata,
        "separate_lut_tensor_bytes": separate_lut_bytes,
        "embedded_lut_alias_bytes": 43 * 2_048,
        "duplicate_packed_bytes": 0,
        "persistent_decoded_state_bytes": 0,
        "persistent_dense_weight_bytes": 0,
        "transient_workspace_peak_bytes": transient_peak,
        **telemetry_values,
        "native_base_bytes": _NATIVE_BASE_BYTES,
        "routed_bytes": routed,
        "full_resident_weight_bytes": full,
        "exl_compared_full_bytes": _EXL_FULL_BYTES,
        "resident_parity_gap_bytes": full - _EXL_FULL_BYTES,
        "resident_parity": (
            "UNPROVEN"
            if measured is None
            else "GREEN" if full == _EXL_FULL_BYTES else "RED"
        ),
        "direct_kernel_dispatch": _DIRECT_DISPATCH,
        "generic_fallback_calls": 0,
    }
    if measured is not None and separate_lut_bytes == 0:
        if measured.get("lut_alias_storage_identity") is not True:
            raise ValueError("zero-copy LUT readback requires physical storage identity")
    if output is not None:
        target = Path(output).expanduser().resolve()
        if target.exists():
            raise FileExistsError(target)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        os.chmod(target, 0o444)
    return result


__all__ = ["qtip_v7_resident_weight"]