"""Public provider lifecycle for homogeneous native QTIP2.5 L16/B10/V4."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .qtip25_native_v4 import (
    NATIVE_QTIP25_GEOMETRY,
    decode_native_v4,
    native_v4_wire_accounting,
    pack_native_v4_states,
    solve_native_v4,
    states_from_native_v4_packed,
)

PROVIDER_ID = "qtip25_native_v4"
_RECEIPT = "QTIP25_NATIVE_V4_RECEIPT.json"
_CODES = "codes.npy"


@dataclass(frozen=True)
class NativeV4WirePrice:
    code_bytes: int
    auxiliary_bytes: int
    shared_tlut_bytes: int

    @property
    def full_wire_bytes(self) -> int:
        return self.code_bytes + self.auxiliary_bytes + self.shared_tlut_bytes


@dataclass(frozen=True)
class NativeV4Provider:
    provider_id: str
    public_name: str
    kind: str
    runtime_family: str
    codec_form: str
    rate_num: int
    rate_den: int
    encode: Callable[..., Any]
    generate: Callable[..., dict[str, Any]]
    materialize: Callable[..., dict[str, Any]]
    price: Callable[..., NativeV4WirePrice]
    predict: Callable[..., np.ndarray]
    verify: Callable[..., bool]


def _root(value: str | Path) -> Path:
    return Path(value).expanduser().resolve()


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _basis(value: str) -> str:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError("native QTIP2.5 V4 requires a lowercase 64-hex intended basis")
    return value


def generate_native_v4_candidate(
    output: str | Path,
    *,
    states: np.ndarray,
    intended_basis_sha256: str,
    scale_bytes: int = 0,
    transform_bytes: int = 0,
    shared_tlut_bytes: int = 0,
) -> dict[str, Any]:
    """Seal exact transition states into a standalone provider artifact."""
    basis = _basis(intended_basis_sha256)
    source = np.asarray(states)
    if source.ndim != 2 or source.shape[1] < 2:
        raise ValueError("native QTIP2.5 V4 states must have shape [sequences,steps>=2]")
    packed = pack_native_v4_states(source)
    positions = int(source.size) * 4
    accounting = native_v4_wire_accounting(
        position_count=positions,
        sequence_count=int(source.shape[0]),
        scale_bytes=scale_bytes,
        transform_bytes=transform_bytes,
        shared_tlut_bytes=shared_tlut_bytes,
    )
    root = _root(output)
    if root.exists():
        raise FileExistsError(f"output already exists: {root}")
    root.mkdir(parents=True)
    try:
        np.save(root / _CODES, packed, allow_pickle=False)
        receipt = {
            "schema": "banana-smasher-qtip25-native-v4-candidate-v1",
            "status": "PASS",
            "provider_id": PROVIDER_ID,
            "runtime_family": PROVIDER_ID,
            "codec_form": "homogeneous_l16_b10_v4",
            **NATIVE_QTIP25_GEOMETRY.as_mapping(),
            "phase_count": 1,
            "unique_transition_bits": [10],
            "alternation": False,
            "intended_basis_sha256": basis,
            "sequence_count": int(source.shape[0]),
            "transition_count": int(source.size),
            **accounting,
            "cell_payload_bytes": int(packed.nbytes),
            "codes": {
                "file": _CODES,
                "dtype": str(packed.dtype),
                "shape": list(packed.shape),
                "data_bytes": int(packed.nbytes),
                "data_sha256": _sha(packed.tobytes()),
            },
            "activation_artifacts": (
                [{"id": "qtip-tlut", "bytes": int(shared_tlut_bytes)}]
                if shared_tlut_bytes
                else []
            ),
        }
        (root / _RECEIPT).write_text(
            json.dumps(receipt, sort_keys=True, separators=(",", ":")) + "\n"
        )
        if not verify_native_v4_candidate(root):
            raise ValueError("native QTIP2.5 V4 generated artifact failed readback")
        return receipt
    except Exception:
        for path in root.iterdir():
            path.unlink()
        root.rmdir()
        raise


def verify_native_v4_candidate(value: str | Path) -> bool:
    try:
        root = _root(value)
        receipt = json.loads((root / _RECEIPT).read_text())
        geometry = NATIVE_QTIP25_GEOMETRY.as_mapping()
        if any(receipt.get(key) != expected for key, expected in geometry.items()):
            return False
        if (
            receipt.get("schema") != "banana-smasher-qtip25-native-v4-candidate-v1"
            or receipt.get("status") != "PASS"
            or receipt.get("provider_id") != PROVIDER_ID
            or receipt.get("phase_count") != 1
            or receipt.get("unique_transition_bits") != [10]
            or receipt.get("alternation") is not False
            or receipt.get("routing_bytes") != 0
            or receipt.get("assignment_map_bytes") != 0
        ):
            return False
        _basis(receipt["intended_basis_sha256"])
        steps = int(receipt["transition_count"]) // int(receipt["sequence_count"])
        packed = np.load(root / _CODES, allow_pickle=False)
        spec = receipt["codes"]
        return (
            packed.dtype == np.uint8
            and list(packed.shape) == spec.get("shape")
            and int(packed.nbytes) == spec.get("data_bytes")
            and int(packed.nbytes) == receipt.get("cell_payload_bytes")
            and _sha(packed.tobytes()) == spec.get("data_sha256")
            and np.array_equal(
                pack_native_v4_states(states_from_native_v4_packed(packed, steps=steps)),
                packed,
            )
        )
    except (FileNotFoundError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return False


def materialize_native_v4_candidate(value: str | Path) -> dict[str, Any]:
    root = _root(value)
    if not verify_native_v4_candidate(root):
        raise ValueError(f"invalid native QTIP2.5 V4 candidate: {root}")
    receipt = json.loads((root / _RECEIPT).read_text())
    return {
        "runtime_family": PROVIDER_ID,
        "codec_form": "homogeneous_l16_b10_v4",
        "geometry": NATIVE_QTIP25_GEOMETRY.as_mapping(),
        "sequence_count": int(receipt["sequence_count"]),
        "transition_count": int(receipt["transition_count"]),
        "codes": np.asarray(np.load(root / _CODES, allow_pickle=False)),
        "intended_basis_sha256": receipt["intended_basis_sha256"],
    }


def price_native_v4_candidate(value: str | Path) -> NativeV4WirePrice:
    root = _root(value)
    if not verify_native_v4_candidate(root):
        raise ValueError(f"invalid native QTIP2.5 V4 candidate: {root}")
    receipt = json.loads((root / _RECEIPT).read_text())
    return NativeV4WirePrice(
        code_bytes=int(receipt["cell_payload_bytes"]),
        auxiliary_bytes=int(receipt["auxiliary_bytes"]),
        shared_tlut_bytes=int(receipt["deduplicated_shared_tlut_bytes"]),
    )


def predict_native_v4_candidate(
    value: str | Path, *, scales: np.ndarray, tlut: np.ndarray
) -> np.ndarray:
    materialized = materialize_native_v4_candidate(value)
    steps = materialized["transition_count"] // materialized["sequence_count"]
    return decode_native_v4(
        materialized["codes"], scales, positions=steps * 4, tlut=tlut
    )


def native_v4_provider() -> NativeV4Provider:
    return NativeV4Provider(
        provider_id=PROVIDER_ID,
        public_name="QTIP2.5-NATIVE-V4",
        kind="qtip_native_v4",
        runtime_family=PROVIDER_ID,
        codec_form="homogeneous_l16_b10_v4",
        rate_num=5,
        rate_den=2,
        encode=solve_native_v4,
        generate=generate_native_v4_candidate,
        materialize=materialize_native_v4_candidate,
        price=price_native_v4_candidate,
        predict=predict_native_v4_candidate,
        verify=verify_native_v4_candidate,
    )


__all__ = [
    "NativeV4Provider",
    "NativeV4WirePrice",
    "generate_native_v4_candidate",
    "materialize_native_v4_candidate",
    "native_v4_provider",
    "predict_native_v4_candidate",
    "price_native_v4_candidate",
    "verify_native_v4_candidate",
]
