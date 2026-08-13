"""Physical fixed-envelope QTIP V7 layer wire and model accounting."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from decimal import Decimal, localcontext
import hashlib
import json
import mmap
import os
from pathlib import Path
import re
import struct
import tempfile
from typing import Any, Sequence
import zlib

_MAGIC = b"BSQTIPV7WIRE1\0\0"
_LAYER_SCHEMA = "banana-smasher-qtip-v7-layer-wire-receipt-v1"
_MODEL_SCHEMA = "banana-smasher-qtip-v7-model-accounting-v1"
_PROJECTIONS = ("w1", "w2", "w3")
_NATIVE_BASE_BYTES = 19_708_797_688
_EXL_K2_ROUTED_BYTES = 69_662_278_656
_REQUIRED_LAYERS = 43
_BLOCK = 8 << 20


@dataclass(frozen=True)
class WireGeometry:
    experts: int = 256
    projections: tuple[str, ...] = _PROJECTIONS
    packed_bytes: int = 2_097_152
    control_bytes: int = 12_292
    lut_bytes: int = 2_048
    header_bytes: int = 128 << 10
    envelope_bytes: int = 768 * 2_109_444

    @property
    def member_count(self) -> int:
        return self.experts * len(self.projections)

    @property
    def source_bytes(self) -> int:
        return self.member_count * (self.packed_bytes + self.control_bytes)


_DEFAULT_GEOMETRY = WireGeometry()


class QtipV7LayerMapping:
    """Read-only views over one fixed envelope; packed bytes and LUT are never copied."""

    def __init__(
        self,
        wire: str | Path,
        *,
        _geometry: WireGeometry = _DEFAULT_GEOMETRY,
    ) -> None:
        self.geometry = _geometry
        self.path = Path(wire).expanduser().resolve()
        if self.path.stat().st_size != self.geometry.envelope_bytes:
            raise ValueError("QTIP V7 layer wire physical byte count drift")
        self._handle = self.path.open("rb")
        try:
            self.buffer = mmap.mmap(self._handle.fileno(), 0, access=mmap.ACCESS_READ)
            metadata = _read_header(self.buffer, self.geometry)
            _validate_metadata(metadata, self.geometry)
            self.metadata = metadata
            self._roster_index = {
                name: index for index, name in enumerate(metadata["roster"])
            }
        except Exception:
            self._handle.close()
            raise

    def packed_view(self, expert: int, projection: str) -> memoryview:
        """Alias one direct-kernel trellis member inside the mapped envelope."""
        try:
            index = self._roster_index[_member_name(expert, projection)]
        except (KeyError, TypeError) as exc:
            raise ValueError(
                f"unknown QTIP V7 member expert={expert!r} projection={projection!r}"
            ) from exc
        start = self.metadata["layout"]["packed_offset"] + index * self.geometry.packed_bytes
        return memoryview(self.buffer)[start : start + self.geometry.packed_bytes]

    def lut_view(self) -> memoryview:
        """Alias the embedded layer-shared FP16[1024] LUT."""
        start = self.metadata["layout"]["lut_offset"]
        return memoryview(self.buffer)[start : start + self.geometry.lut_bytes]

    def transient_controls(self) -> bytearray:
        """Expand controls into caller-owned bounded workspace, never resident state."""
        layout = self.metadata["layout"]
        start = layout["control_offset"]
        compressed = self.buffer[start : start + layout["compressed_control_bytes"]]
        controls = bytearray(zlib.decompress(compressed))
        if len(controls) != layout["uncompressed_control_bytes"]:
            raise ValueError("QTIP V7 control reconstruction byte count drift")
        if hashlib.sha256(controls).hexdigest() != self.metadata["control_sha256"]:
            raise ValueError("QTIP V7 control reconstruction mismatch")
        return controls

    @property
    def transient_workspace_peak_bytes(self) -> int:
        return int(self.metadata["layout"]["uncompressed_control_bytes"])

    def close(self) -> None:
        self.buffer.close()
        self._handle.close()

    def __enter__(self) -> "QtipV7LayerMapping":
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(_BLOCK), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(_canonical(value))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def _member_name(expert: int, projection: str) -> str:
    return f"E{expert:03d}_{projection}.q2v7wire"


def _roster(root: Path, geometry: WireGeometry) -> list[Path]:
    expected = [
        root / _member_name(expert, projection)
        for expert in range(geometry.experts)
        for projection in geometry.projections
    ]
    observed = sorted(path.name for path in root.glob("*.q2v7wire"))
    expected_names = [path.name for path in expected]
    if observed != sorted(expected_names):
        missing = sorted(set(expected_names).difference(observed))
        extra = sorted(set(observed).difference(expected_names))
        raise ValueError(f"QTIP V7 layer roster drift: missing={missing} extra={extra}")
    member_bytes = geometry.packed_bytes + geometry.control_bytes
    for path in expected:
        if not path.is_file() or path.is_symlink() or path.stat().st_size != member_bytes:
            raise ValueError(
                f"QTIP V7 member requires exactly {member_bytes} bytes: {path}"
            )
    return expected


def _header_bytes(metadata: dict[str, Any], geometry: WireGeometry) -> bytes:
    payload = _canonical(metadata)
    prefix = _MAGIC + struct.pack("<Q", len(payload))
    if len(prefix) + len(payload) > geometry.header_bytes:
        raise ValueError("QTIP V7 wire metadata exceeds the fixed header envelope")
    return prefix + payload + bytes(geometry.header_bytes - len(prefix) - len(payload))


def _read_header(handle: Any, geometry: WireGeometry) -> dict[str, Any]:
    header = handle.read(geometry.header_bytes)
    if len(header) != geometry.header_bytes or header[: len(_MAGIC)] != _MAGIC:
        raise ValueError("QTIP V7 wire header/magic drift")
    length = struct.unpack("<Q", header[len(_MAGIC) : len(_MAGIC) + 8])[0]
    start = len(_MAGIC) + 8
    if length <= 0 or start + length > geometry.header_bytes:
        raise ValueError("QTIP V7 wire header length drift")
    value = json.loads(header[start : start + length])
    if not isinstance(value, dict):
        raise ValueError("QTIP V7 wire header must contain an object")
    return value


def _validate_metadata(metadata: dict[str, Any], geometry: WireGeometry) -> None:
    expected_geometry = asdict(geometry)
    expected_geometry["projections"] = list(geometry.projections)
    if metadata.get("schema") != "banana-smasher-qtip-v7-layer-wire-v1":
        raise ValueError("QTIP V7 wire schema drift")
    if metadata.get("geometry") != expected_geometry:
        raise ValueError("QTIP V7 wire geometry drift")
    roster = metadata.get("roster")
    expected_roster = [
        _member_name(expert, projection)
        for expert in range(geometry.experts)
        for projection in geometry.projections
    ]
    if roster != expected_roster:
        raise ValueError("QTIP V7 wire roster/order drift")
    layout = metadata.get("layout")
    if not isinstance(layout, dict):
        raise ValueError("QTIP V7 wire layout is missing")
    expected_packed = geometry.member_count * geometry.packed_bytes
    if (
        layout.get("packed_offset") != geometry.header_bytes
        or layout.get("packed_bytes") != expected_packed
        or layout.get("control_offset") != geometry.header_bytes + expected_packed
        or layout.get("lut_offset")
        != layout.get("control_offset", -1) + layout.get("compressed_control_bytes", -1)
        or layout.get("padding_offset") != layout.get("lut_offset", -1) + geometry.lut_bytes
        or layout.get("padding_bytes")
        != geometry.envelope_bytes - layout.get("padding_offset", -1)
        or layout.get("padding_bytes", -1) < 0
    ):
        raise ValueError("QTIP V7 wire layout drift")


def pack_qtip_v7_layer(
    *,
    source_root: str | Path,
    lut: str | Path,
    layer: int,
    output: str | Path,
    receipt: str | Path | None = None,
    _geometry: WireGeometry = _DEFAULT_GEOMETRY,
) -> dict[str, Any]:
    """Pack one exact expert-major w1/w2/w3 layer into its fixed physical envelope."""
    geometry = _geometry
    if isinstance(layer, bool) or not isinstance(layer, int) or layer < 0:
        raise ValueError("QTIP V7 layer must be a nonnegative integer")
    source = Path(source_root).expanduser().resolve()
    members = _roster(source, geometry)
    lut_path = Path(lut).expanduser().resolve()
    if not lut_path.is_file() or lut_path.is_symlink() or lut_path.stat().st_size != geometry.lut_bytes:
        raise ValueError(f"QTIP V7 layer LUT requires exactly {geometry.lut_bytes} bytes")
    lut_payload = lut_path.read_bytes()
    output_path = Path(output).expanduser().resolve()
    receipt_path = (
        output_path.with_name(f"{output_path.name}.receipt.json")
        if receipt is None
        else Path(receipt).expanduser().resolve()
    )
    if output_path.exists() or receipt_path.exists():
        raise FileExistsError(output_path if output_path.exists() else receipt_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    controls = bytearray()
    reconstructed = hashlib.sha256()
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output_path.name}.", dir=output_path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w+b") as wire:
            wire.write(bytes(geometry.header_bytes))
            for path in members:
                with path.open("rb") as member:
                    remaining = geometry.packed_bytes
                    while remaining:
                        block = member.read(min(_BLOCK, remaining))
                        if not block:
                            raise ValueError(f"truncated QTIP V7 packed plane: {path}")
                        wire.write(block)
                        reconstructed.update(block)
                        remaining -= len(block)
                    control = member.read(geometry.control_bytes)
                    if len(control) != geometry.control_bytes or member.read(1):
                        raise ValueError(f"QTIP V7 control plane geometry drift: {path}")
                    controls.extend(control)
                    reconstructed.update(control)
            compressed = zlib.compress(bytes(controls), level=9)
            control_offset = wire.tell()
            wire.write(compressed)
            lut_offset = wire.tell()
            wire.write(lut_payload)
            padding_offset = wire.tell()
            padding_bytes = geometry.envelope_bytes - padding_offset
            if padding_bytes < 0:
                raise ValueError(
                    "compressed QTIP V7 controls and metadata do not fit the fixed layer envelope"
                )
            zeros = bytes(min(_BLOCK, padding_bytes))
            remaining = padding_bytes
            while remaining:
                block = zeros[: min(len(zeros), remaining)]
                wire.write(block)
                remaining -= len(block)
            metadata = {
                "schema": "banana-smasher-qtip-v7-layer-wire-v1",
                "layer": layer,
                "geometry": {
                    **asdict(geometry),
                    "projections": list(geometry.projections),
                },
                "roster": [path.name for path in members],
                "reconstructed_stream_sha256": reconstructed.hexdigest(),
                "lut_sha256": hashlib.sha256(lut_payload).hexdigest(),
                "control_sha256": hashlib.sha256(controls).hexdigest(),
                "layout": {
                    "packed_offset": geometry.header_bytes,
                    "packed_bytes": geometry.member_count * geometry.packed_bytes,
                    "control_offset": control_offset,
                    "compressed_control_bytes": len(compressed),
                    "uncompressed_control_bytes": len(controls),
                    "lut_offset": lut_offset,
                    "lut_bytes": geometry.lut_bytes,
                    "padding_offset": padding_offset,
                    "padding_bytes": padding_bytes,
                },
            }
            wire.seek(0)
            wire.write(_header_bytes(metadata, geometry))
            wire.flush()
            os.fsync(wire.fileno())
        os.replace(temporary, output_path)
        wire_sha256 = _sha256(output_path)
        result = {
            "schema": _LAYER_SCHEMA,
            "status": "PASS",
            "layer": layer,
            "member_count": geometry.member_count,
            "roster": metadata["roster"],
            "source_member_bytes": geometry.source_bytes,
            "physical_bytes": output_path.stat().st_size,
            "wire_size_delta": output_path.stat().st_size - geometry.source_bytes,
            "packed_bytes": metadata["layout"]["packed_bytes"],
            "control_bytes": len(controls),
            "compressed_control_bytes": len(compressed),
            "compression_saved_bytes": len(controls) - len(compressed),
            "embedded_lut_bytes": geometry.lut_bytes,
            "padding_bytes": padding_bytes,
            "reconstructed_stream_sha256": metadata["reconstructed_stream_sha256"],
            "lut_sha256": metadata["lut_sha256"],
            "wire_sha256": wire_sha256,
            "wire": str(output_path),
            "reconstructed_stream_authenticated": True,
            "embedded_lut_authenticated": True,
            "roster_authenticated": True,
            "physical_bytes_authenticated": True,
        }
        if result["physical_bytes"] != geometry.envelope_bytes or result["wire_size_delta"] != 0:
            raise RuntimeError("QTIP V7 fixed-envelope physical byte invariant failed")
        _write_json(receipt_path, result)
        os.chmod(receipt_path, 0o444)
        return {**result, "receipt": str(receipt_path), "receipt_sha256": _sha256(receipt_path)}
    except Exception:
        temporary.unlink(missing_ok=True)
        output_path.unlink(missing_ok=True)
        if receipt_path.exists():
            os.chmod(receipt_path, 0o644)
            receipt_path.unlink()
        raise


def verify_qtip_v7_layer(
    *,
    wire: str | Path,
    receipt: str | Path,
    reconstructed_output: str | Path | None = None,
    _geometry: WireGeometry = _DEFAULT_GEOMETRY,
) -> dict[str, Any]:
    """Stream-verify and optionally reconstruct every original member and embedded LUT."""
    geometry = _geometry
    wire_path = Path(wire).expanduser().resolve()
    receipt_path = Path(receipt).expanduser().resolve()
    expected = json.loads(receipt_path.read_text())
    if not isinstance(expected, dict) or expected.get("schema") != _LAYER_SCHEMA:
        raise ValueError("QTIP V7 layer receipt schema drift")
    if wire_path.stat().st_size != geometry.envelope_bytes:
        raise ValueError("QTIP V7 layer wire physical byte count drift")
    if expected.get("wire_sha256") != _sha256(wire_path):
        raise ValueError("QTIP V7 layer wire SHA-256 mismatch")
    output = None if reconstructed_output is None else Path(reconstructed_output).expanduser().resolve()
    if output is not None:
        if output.exists():
            raise FileExistsError(output)
        output.mkdir(parents=True)
    try:
        with wire_path.open("rb") as handle:
            metadata = _read_header(handle, geometry)
            _validate_metadata(metadata, geometry)
            layout = metadata["layout"]
            handle.seek(layout["control_offset"])
            compressed = handle.read(layout["compressed_control_bytes"])
            controls = zlib.decompress(compressed)
            if (
                len(controls) != layout["uncompressed_control_bytes"]
                or hashlib.sha256(controls).hexdigest() != metadata["control_sha256"]
            ):
                raise ValueError("QTIP V7 control reconstruction mismatch")
            handle.seek(layout["lut_offset"])
            lut_payload = handle.read(geometry.lut_bytes)
            lut_sha256 = hashlib.sha256(lut_payload).hexdigest()
            if lut_sha256 != metadata["lut_sha256"]:
                raise ValueError("QTIP V7 embedded LUT SHA-256 mismatch")
            if output is not None:
                (output / "embedded_lut.f16").write_bytes(lut_payload)
            reconstructed = hashlib.sha256()
            handle.seek(layout["packed_offset"])
            control_offset = 0
            for name in metadata["roster"]:
                target = None if output is None else output / name
                target_handle = None if target is None else target.open("wb")
                try:
                    remaining = geometry.packed_bytes
                    while remaining:
                        block = handle.read(min(_BLOCK, remaining))
                        if not block:
                            raise ValueError("truncated QTIP V7 packed plane")
                        reconstructed.update(block)
                        if target_handle is not None:
                            target_handle.write(block)
                        remaining -= len(block)
                    control = controls[
                        control_offset : control_offset + geometry.control_bytes
                    ]
                    control_offset += geometry.control_bytes
                    reconstructed.update(control)
                    if target_handle is not None:
                        target_handle.write(control)
                finally:
                    if target_handle is not None:
                        target_handle.close()
            reconstructed_sha256 = reconstructed.hexdigest()
            if reconstructed_sha256 != metadata["reconstructed_stream_sha256"]:
                raise ValueError("QTIP V7 reconstructed member stream SHA-256 mismatch")
            for field, observed in (
                ("layer", metadata["layer"]),
                ("member_count", geometry.member_count),
                ("physical_bytes", geometry.envelope_bytes),
                ("wire_size_delta", 0),
                ("roster", metadata["roster"]),
                ("reconstructed_stream_sha256", reconstructed_sha256),
                ("lut_sha256", lut_sha256),
            ):
                if expected.get(field) != observed:
                    raise ValueError(f"QTIP V7 layer receipt {field} mismatch")
        result = {
            **expected,
            "status": "PASS",
            "reconstructed_stream_sha256": reconstructed_sha256,
            "lut_sha256": lut_sha256,
            "wire_sha256": _sha256(wire_path),
            "reconstructed_stream_authenticated": True,
            "embedded_lut_authenticated": True,
            "roster_authenticated": True,
            "physical_bytes_authenticated": True,
            "receipt": str(receipt_path),
            "receipt_sha256": _sha256(receipt_path),
        }
        return result
    except Exception:
        if output is not None and output.exists():
            for child in output.iterdir():
                child.unlink()
            output.rmdir()
        raise


def account_qtip_v7_model(
    *,
    receipts: Sequence[str | Path],
    output: str | Path,
    weight_denominator: int,
    weight_denominator_label: str,
) -> dict[str, Any]:
    """Derive canonical zero-gap full-model accounting from 43 verified layer receipts."""
    if (
        isinstance(weight_denominator, bool)
        or not isinstance(weight_denominator, int)
        or weight_denominator <= 0
        or not weight_denominator_label.strip()
    ):
        raise ValueError("stored-wire BPW requires a positive declared weight denominator")
    rows: dict[int, dict[str, Any]] = {}
    sources = []
    required_flags = (
        "reconstructed_stream_authenticated",
        "embedded_lut_authenticated",
        "roster_authenticated",
        "physical_bytes_authenticated",
    )
    for value in receipts:
        path = Path(value).expanduser().resolve()
        declared = json.loads(path.read_text())
        if not isinstance(declared, dict) or not isinstance(declared.get("wire"), str):
            raise ValueError(f"QTIP V7 layer receipt does not bind physical wire: {path}")
        row = verify_qtip_v7_layer(wire=declared["wire"], receipt=path)
        if (
            not isinstance(row, dict)
            or row.get("schema") != _LAYER_SCHEMA
            or row.get("status") != "PASS"
            or row.get("member_count") != _DEFAULT_GEOMETRY.member_count
            or row.get("physical_bytes") != _DEFAULT_GEOMETRY.envelope_bytes
            or row.get("wire_size_delta") != 0
            or not all(row.get(flag) is True for flag in required_flags)
            or not isinstance(row.get("wire_sha256"), str)
            or not re.fullmatch(r"[0-9a-f]{64}", row["wire_sha256"])
        ):
            raise ValueError(f"unverified QTIP V7 physical layer receipt: {path}")
        layer = row.get("layer")
        if isinstance(layer, bool) or not isinstance(layer, int) or layer in rows:
            raise ValueError(f"duplicate/invalid QTIP V7 layer receipt: {path}")
        rows[layer] = row
        sources.append({"layer": layer, "path": str(path), "sha256": _sha256(path)})
    if set(rows) != set(range(_REQUIRED_LAYERS)):
        raise ValueError("QTIP V7 model accounting requires exact verified layers 0..42")
    routed = sum(rows[layer]["physical_bytes"] for layer in range(_REQUIRED_LAYERS))
    if routed != _EXL_K2_ROUTED_BYTES:
        raise RuntimeError("QTIP V7 routed physical total does not close the EXL K2 envelope")
    full = routed + _NATIVE_BASE_BYTES
    exl_full = _EXL_K2_ROUTED_BYTES + _NATIVE_BASE_BYTES
    numerator = full * 8
    divisor = __import__("math").gcd(numerator, weight_denominator)
    with localcontext() as context:
        context.prec = 40
        decimal_bpw = str(Decimal(numerator) / Decimal(weight_denominator))
    result = {
        "schema": _MODEL_SCHEMA,
        "status": "PASS",
        "verified_layer_receipts": _REQUIRED_LAYERS,
        "layer_receipts": sorted(sources, key=lambda row: row["layer"]),
        "qtip_routed_stored_bytes": routed,
        "exl_k2_routed_stored_bytes": _EXL_K2_ROUTED_BYTES,
        "routed_gap_bytes": routed - _EXL_K2_ROUTED_BYTES,
        "native_base_bytes": _NATIVE_BASE_BYTES,
        "native_base_inclusions_per_full_model": 1,
        "qtip_full_stored_bytes": full,
        "exl_full_stored_bytes": exl_full,
        "full_gap_bytes": full - exl_full,
        "stored_wire_bpw": {
            "meaning": "stored wire bits per declared weight",
            "numerator_bits": numerator,
            "weight_denominator": weight_denominator,
            "weight_denominator_label": weight_denominator_label.strip(),
            "exact_fraction": f"{numerator // divisor}/{weight_denominator // divisor}",
            "decimal": decimal_bpw,
        },
        "decoded_dtype_claim": None,
    }
    output_path = Path(output).expanduser().resolve()
    if output_path.exists():
        raise FileExistsError(output_path)
    _write_json(output_path, result)
    os.chmod(output_path, 0o444)
    return result


__all__ = [
    "QtipV7LayerMapping",
    "WireGeometry",
    "account_qtip_v7_model",
    "pack_qtip_v7_layer",
    "verify_qtip_v7_layer",
]
