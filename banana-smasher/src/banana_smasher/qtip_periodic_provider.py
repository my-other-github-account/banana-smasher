from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .qtip_periodic import (
    PERIODIC_QTIP25_FORMAT,
    decode_packed,
    pack_symbols,
    periodic_wire_accounting,
    solve_periodic,
    unpack_symbols,
)

_RECEIPT_NAME = "QTIP25_PERIODIC_RECEIPT.json"
_CODES_NAME = "codes.npy"


@dataclass(frozen=True)
class PeriodicWirePrice:
    cell_payload_bytes: int
    auxiliary_bytes: int
    routing_bytes: int
    assignment_map_bytes: int
    shared_tlut_bytes: int

    @property
    def full_wire_bytes(self) -> int:
        return self.cell_payload_bytes + self.auxiliary_bytes + self.shared_tlut_bytes


@dataclass(frozen=True)
class PeriodicQTIP25Provider:
    provider_id: str
    kind: str
    runtime_family: str
    codec_form: str
    rate_num: int
    rate_den: int
    encode: Callable[..., dict[str, Any]]
    generate: Callable[..., dict[str, Any]]
    materialize: Callable[..., dict[str, Any]]
    price: Callable[..., PeriodicWirePrice]
    predict: Callable[..., np.ndarray]
    verify: Callable[..., bool]


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _root(value: str | Path) -> Path:
    return Path(value).expanduser().resolve()


def _receipt(root: Path) -> dict[str, Any]:
    value = json.loads((root / _RECEIPT_NAME).read_text())
    if not isinstance(value, dict):
        raise ValueError("QTIP2.5-PERIODIC receipt must contain an object")
    return value


def generate_periodic_candidate(
    output: str | Path,
    *,
    symbols: np.ndarray,
    intended_basis_sha256: str,
    transform_bytes: int = 0,
    scale_bytes: int = 0,
    shared_tlut_bytes: int = 0,
) -> dict[str, Any]:
    """Generate one standalone provider artifact from periodic QTIP transitions."""
    if (
        not isinstance(intended_basis_sha256, str)
        or len(intended_basis_sha256) != 64
        or any(character not in "0123456789abcdef" for character in intended_basis_sha256)
    ):
        raise ValueError("QTIP2.5-PERIODIC requires a lowercase 64-hex intended basis")
    root = _root(output)
    if root.exists():
        raise FileExistsError(f"output already exists: {root}")
    packed = pack_symbols(symbols)
    transition_count = int(np.asarray(symbols).size)
    position_count = transition_count * int(PERIODIC_QTIP25_FORMAT["values_per_transition"])
    accounting = periodic_wire_accounting(
        position_count=position_count,
        transform_bytes=transform_bytes,
        scale_bytes=scale_bytes,
        shared_tlut_bytes=shared_tlut_bytes,
    )
    root.mkdir(parents=True)
    try:
        np.save(root / _CODES_NAME, packed, allow_pickle=False)
        receipt = {
            "schema": "banana-smasher-qtip25-periodic-candidate-v1",
            "status": "PASS",
            **PERIODIC_QTIP25_FORMAT,
            "provider_id": "qtip25-periodic",
            "runtime_family": "qtip25_periodic",
            "intended_basis_sha256": intended_basis_sha256,
            "transition_count": transition_count,
            **accounting,
            "cell_payload_bytes": int(packed.nbytes),
            "unique_physical_tree_bytes_excluding_receipt": int(
                (root / _CODES_NAME).stat().st_size
            ),
            "whole_model_bytes": None,
            "whole_model_gb": None,
            "whole_model_bpw": None,
            "fp8_control": None,
            "codes": {
                "file": _CODES_NAME,
                "dtype": str(packed.dtype),
                "shape": list(packed.shape),
                "data_bytes": int(packed.nbytes),
                "data_sha256": _sha256_bytes(packed.tobytes()),
            },
            "activation_artifacts": (
                [{"id": "qtip-tlut", "bytes": int(shared_tlut_bytes)}]
                if shared_tlut_bytes
                else []
            ),
        }
        (root / _RECEIPT_NAME).write_text(
            json.dumps(receipt, sort_keys=True, separators=(",", ":")) + "\n"
        )
        if not verify_periodic_candidate(root):
            raise ValueError("QTIP2.5-PERIODIC generated artifact failed readback")
        return receipt
    except Exception:
        for path in root.iterdir():
            path.unlink()
        root.rmdir()
        raise


def verify_periodic_candidate(value: str | Path) -> bool:
    """Verify format identity, exact code bytes, and transition roundtrip."""
    try:
        root = _root(value)
        receipt = _receipt(root)
        for key, expected in PERIODIC_QTIP25_FORMAT.items():
            if receipt.get(key) != expected:
                return False
        if (
            receipt.get("schema") != "banana-smasher-qtip25-periodic-candidate-v1"
            or receipt.get("status") != "PASS"
            or receipt.get("provider_id") != "qtip25-periodic"
            or receipt.get("runtime_family") != "qtip25_periodic"
            or receipt.get("assignment_map_bytes") != 0
            or receipt.get("routing_bytes") != 0
        ):
            return False
        basis = receipt.get("intended_basis_sha256")
        if (
            not isinstance(basis, str)
            or len(basis) != 64
            or any(character not in "0123456789abcdef" for character in basis)
        ):
            return False
        transition_count = int(receipt["transition_count"])
        codes = np.load(root / _CODES_NAME, allow_pickle=False)
        spec = receipt["codes"]
        if (
            codes.dtype != np.uint8
            or list(codes.shape) != spec.get("shape")
            or int(codes.nbytes) != spec.get("data_bytes")
            or int(codes.nbytes) != receipt.get("cell_payload_bytes")
            or _sha256_bytes(codes.tobytes()) != spec.get("data_sha256")
        ):
            return False
        return np.array_equal(
            pack_symbols(unpack_symbols(codes, transition_count)), codes
        )
    except (FileNotFoundError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return False


def price_periodic_candidate(value: str | Path) -> PeriodicWirePrice:
    root = _root(value)
    if not verify_periodic_candidate(root):
        raise ValueError(f"invalid QTIP2.5-PERIODIC candidate: {root}")
    receipt = _receipt(root)
    return PeriodicWirePrice(
        cell_payload_bytes=int(receipt["cell_payload_bytes"]),
        auxiliary_bytes=int(receipt["auxiliary_bytes"]),
        routing_bytes=int(receipt["routing_bytes"]),
        assignment_map_bytes=int(receipt["assignment_map_bytes"]),
        shared_tlut_bytes=int(receipt["deduplicated_shared_tlut_bytes"]),
    )


def materialize_periodic_candidate(value: str | Path) -> dict[str, Any]:
    root = _root(value)
    if not verify_periodic_candidate(root):
        raise ValueError(f"invalid QTIP2.5-PERIODIC candidate: {root}")
    receipt = _receipt(root)
    return {
        "runtime_family": "qtip25_periodic",
        "codec_form": "qtip25_periodic_23",
        "rate_num": 5,
        "rate_den": 2,
        "transition_count": int(receipt["transition_count"]),
        "codes": np.asarray(np.load(root / _CODES_NAME, allow_pickle=False)),
        "intended_basis_sha256": receipt["intended_basis_sha256"],
    }


def predict_periodic_candidate(
    value: str | Path, *, lut: np.ndarray
) -> np.ndarray:
    materialized = materialize_periodic_candidate(value)
    return decode_packed(
        materialized["codes"], materialized["transition_count"], lut
    )


def periodic_qtip25_provider() -> PeriodicQTIP25Provider:
    """Return the clean public provider lifecycle for the periodic codec."""
    return PeriodicQTIP25Provider(
        provider_id="qtip25-periodic",
        kind="qtip_periodic",
        runtime_family="qtip25_periodic",
        codec_form="qtip25_periodic_23",
        rate_num=5,
        rate_den=2,
        encode=solve_periodic,
        generate=generate_periodic_candidate,
        materialize=materialize_periodic_candidate,
        price=price_periodic_candidate,
        predict=predict_periodic_candidate,
        verify=verify_periodic_candidate,
    )
