from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

RECEIPT_SCHEMA = "banana-smasher-checkpoint-identity-rebind-v1"
_TRANSIENT_FIELDS = frozenset(
    {
        "claim_sha256",
        "base_stat_fingerprint_sha256",
        "ctime_ns",
        "device",
        "inode",
    }
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _identity_sha256(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _immutable_identity(value: object) -> object:
    """Return content identity with host placement and lease fields removed."""
    if isinstance(value, dict):
        return {
            key: _immutable_identity(item)
            for key, item in value.items()
            if key not in _TRANSIENT_FIELDS
        }
    if isinstance(value, list):
        return [_immutable_identity(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_immutable_identity(item) for item in value)
    if isinstance(value, str) and os.path.isabs(value):
        return "<relocated-absolute-path>"
    return value


def _identity_changes(old: object, new: object, path: str = "") -> list[dict[str, object]]:
    if type(old) is not type(new):
        return [{"path": path, "old": old, "new": new}]
    if isinstance(old, dict):
        changes: list[dict[str, object]] = []
        for key in sorted(set(old) | set(new)):
            child = f"{path}.{key}" if path else str(key)
            if key not in old:
                changes.append({"path": child, "old": "<missing>", "new": new[key]})
            elif key not in new:
                changes.append({"path": child, "old": old[key], "new": "<missing>"})
            else:
                changes.extend(_identity_changes(old[key], new[key], child))
        return changes
    if isinstance(old, (list, tuple)):
        changes = []
        if len(old) != len(new):
            changes.append({"path": f"{path}.length", "old": len(old), "new": len(new)})
        for index, (old_item, new_item) in enumerate(zip(old, new, strict=False)):
            changes.extend(_identity_changes(old_item, new_item, f"{path}[{index}]"))
        return changes
    return [] if old == new else [{"path": path, "old": old, "new": new}]


def _require_sha256(label: str, value: str) -> str:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{label} must be 64 lowercase hexadecimal characters")
    return value


def build_checkpoint_identity_rebind_receipt(
    *,
    old_identity: dict[str, object],
    current_identity: dict[str, object],
    sidecar_sha256: str,
    checkpoint_sha256: str,
    task_id: str,
) -> dict[str, object]:
    """Build an exact authorization for placement-only checkpoint relocation."""
    _require_sha256("sidecar_sha256", sidecar_sha256)
    _require_sha256("checkpoint_sha256", checkpoint_sha256)
    if not task_id:
        raise ValueError("task_id must be non-empty")
    old_immutable = _immutable_identity(old_identity)
    current_immutable = _immutable_identity(current_identity)
    if old_immutable != current_immutable:
        raise RuntimeError("immutable checkpoint identity drift")
    return {
        "schema": RECEIPT_SCHEMA,
        "status": "PASS_APPROVED_TRANSIENT_REBIND",
        "task_id": task_id,
        "sidecar_sha256": sidecar_sha256,
        "checkpoint_sha256": checkpoint_sha256,
        "source_identity_sha256": _identity_sha256(old_identity),
        "target_identity_sha256": _identity_sha256(current_identity),
        "immutable_identity_sha256": _identity_sha256(old_immutable),
        "approved_claim_sha256": current_identity.get("claim_sha256"),
        "transient_changes": _identity_changes(old_identity, current_identity),
    }


def validate_checkpoint_identity_rebind(
    *,
    old_identity: dict[str, object],
    current_identity: dict[str, object],
    receipt: dict[str, object] | None,
    sidecar_sha256: str,
    checkpoint_sha256: str,
) -> None:
    """Accept equal identity or a receipt-bound, byte-stable relocation only."""
    if old_identity == current_identity:
        return
    old_immutable = _immutable_identity(old_identity)
    if old_immutable != _immutable_identity(current_identity):
        raise RuntimeError("immutable checkpoint identity drift")
    if receipt is None:
        raise RuntimeError("approved checkpoint identity rebind receipt required")
    expected = {
        "schema": RECEIPT_SCHEMA,
        "status": "PASS_APPROVED_TRANSIENT_REBIND",
        "sidecar_sha256": sidecar_sha256,
        "checkpoint_sha256": checkpoint_sha256,
        "source_identity_sha256": _identity_sha256(old_identity),
        "target_identity_sha256": _identity_sha256(current_identity),
        "immutable_identity_sha256": _identity_sha256(old_immutable),
        "approved_claim_sha256": current_identity.get("claim_sha256"),
        "transient_changes": _identity_changes(old_identity, current_identity),
    }
    drift = {key: (receipt.get(key), value) for key, value in expected.items() if receipt.get(key) != value}
    if not isinstance(receipt.get("task_id"), str) or not receipt["task_id"]:
        drift["task_id"] = (receipt.get("task_id"), "non-empty task id")
    if drift:
        raise RuntimeError(f"checkpoint identity rebind receipt drift: {drift}")


def authorize_checkpoint_identity_rebind(
    *,
    sidecar_path: str | Path,
    checkpoint_path: str | Path,
    old_identity: dict[str, object],
    current_identity: dict[str, object],
) -> Path:
    sidecar = Path(sidecar_path).resolve()
    checkpoint = Path(checkpoint_path).resolve()
    receipt_path = sidecar.with_suffix(".rebind.json")
    receipt = json.loads(receipt_path.read_text()) if receipt_path.is_file() else None
    validate_checkpoint_identity_rebind(
        old_identity=old_identity,
        current_identity=current_identity,
        receipt=receipt,
        sidecar_sha256=_sha256_file(sidecar),
        checkpoint_sha256=_sha256_file(checkpoint),
    )
    return receipt_path
