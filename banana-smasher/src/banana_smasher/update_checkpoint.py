from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any

CHECKPOINT_SCHEMA = "banana-smasher-update-checkpoint-v2"
MANIFEST_SCHEMA = "banana-smasher-update-checkpoint-manifest-v2"
REQUIRED_IDENTITY_HASHES = (
    "content_sha256",
    "config_sha256",
    "assignment_sha256",
    "code_sha256",
)
_SHA256 = re.compile(r"[0-9a-f]{64}")


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _contains_absolute_path(value: Any) -> bool:
    if isinstance(value, dict):
        return any(_contains_absolute_path(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return any(_contains_absolute_path(item) for item in value)
    return isinstance(value, str) and Path(value).is_absolute()


def validate_identity(identity: dict[str, Any]) -> dict[str, Any]:
    try:
        canonical = json.loads(json.dumps(identity, sort_keys=True, allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"checkpoint identity must be canonical JSON: {exc}") from exc
    if not isinstance(canonical, dict):
        raise ValueError("checkpoint identity must be a JSON object")
    for name in REQUIRED_IDENTITY_HASHES:
        value = canonical.get(name)
        if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
            raise ValueError(f"checkpoint identity requires immutable {name}")
    if _contains_absolute_path(canonical):
        raise ValueError("absolute paths cannot participate in checkpoint identity")
    return canonical


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    data = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
    with temporary.open("xb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    _fsync_directory(path.parent)


def atomic_torch_save(path: Path, value: Any) -> dict[str, Any]:
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("xb") as stream:
        torch.save(value, stream)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    _fsync_directory(path.parent)
    return {"sha256": _sha256(path), "bytes": path.stat().st_size}


def _local_payload(root: Path, relative: str) -> Path:
    candidate = Path(relative)
    if candidate.is_absolute() or candidate.name != relative or not relative.startswith("payload-"):
        raise RuntimeError("checkpoint payload path is not a safe root-relative member")
    path = root / candidate
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"checkpoint payload is missing or not regular: {relative}")
    return path


def commit_segment_checkpoint(
    checkpoint_dir: str | Path,
    payload: dict[str, Any],
    *,
    identity: dict[str, Any],
    backend: str,
    segment_plan: list[int],
) -> dict[str, Any]:
    root = Path(checkpoint_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    identity = validate_identity(identity)
    manifest_path = root / "manifest.json"
    previous_payload: Path | None = None
    if manifest_path.is_file():
        previous = json.loads(manifest_path.read_text()).get("payload_member")
        if isinstance(previous, str):
            previous_payload = root / previous
    payload_member = (
        f"payload-{payload['run_id']}-{int(payload['next_segment_index']):04d}-"
        f"{payload['state']}.pt"
    )
    payload_path = root / payload_member
    payload_record = atomic_torch_save(
        payload_path, {"schema": CHECKPOINT_SCHEMA, **payload}
    )
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "status": "COMPLETE" if payload.get("state") == "complete" else "IN_PROGRESS",
        "backend": backend,
        "identity": identity,
        "identity_sha256": hashlib.sha256(
            json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "segment_plan": [int(count) for count in segment_plan],
        "logical_items": sum(int(count) for count in segment_plan),
        "next_segment_index": int(payload["next_segment_index"]),
        "completed_segments": list(payload["completed_segments"]),
        "optimizer_steps": int(payload.get("optimizer_steps", 0)),
        "transaction_state": str(payload["state"]),
        "payload_member": payload_member,
        "payload_sha256": payload_record["sha256"],
        "payload_bytes": payload_record["bytes"],
    }
    atomic_json(manifest_path, manifest)
    if previous_payload is not None and previous_payload != payload_path:
        previous_payload.unlink(missing_ok=True)
        _fsync_directory(root)
    return manifest


def load_checkpoint(
    checkpoint_dir: str | Path,
    *,
    expected_identity: dict[str, Any],
    expected_backend: str,
    expected_segment_plan: list[int] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    import torch

    root = Path(checkpoint_dir).resolve()
    manifest_path = root / "manifest.json"
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise FileNotFoundError(f"update checkpoint manifest does not exist: {manifest_path}")
    try:
        manifest = json.loads(manifest_path.read_text())
    except Exception as exc:
        raise RuntimeError(f"update checkpoint manifest is corrupt: {exc}") from exc
    if manifest.get("schema") != MANIFEST_SCHEMA:
        raise RuntimeError(
            f"unsupported update checkpoint manifest schema: {manifest.get('schema')!r}"
        )
    expected_identity = validate_identity(expected_identity)
    if manifest.get("identity") != expected_identity:
        raise RuntimeError("checkpoint identity mismatch")
    if manifest.get("backend") != expected_backend:
        raise RuntimeError(
            f"checkpoint backend mismatch: {manifest.get('backend')!r} != {expected_backend!r}"
        )
    if expected_segment_plan is not None and manifest.get("segment_plan") != expected_segment_plan:
        raise RuntimeError("checkpoint segment geometry mismatch")
    completed = manifest.get("completed_segments")
    next_index = int(manifest.get("next_segment_index", -1))
    if completed != list(range(next_index)):
        raise RuntimeError("checkpoint completed segments are not a contiguous prefix")
    payload_path = _local_payload(root, str(manifest.get("payload_member", "")))
    if payload_path.stat().st_size != int(manifest["payload_bytes"]):
        raise RuntimeError("checkpoint payload byte count mismatch")
    if _sha256(payload_path) != manifest["payload_sha256"]:
        raise RuntimeError("checkpoint payload SHA-256 mismatch")
    try:
        payload = torch.load(payload_path, map_location="cpu", weights_only=False)
    except Exception as exc:
        raise RuntimeError(f"checkpoint payload cannot be loaded: {exc}") from exc
    if payload.get("schema") != CHECKPOINT_SCHEMA:
        raise RuntimeError(
            f"unsupported update checkpoint payload schema: {payload.get('schema')!r}"
        )
    if int(payload.get("next_segment_index", -1)) != next_index:
        raise RuntimeError("checkpoint payload/manifest next-segment mismatch")
    if payload.get("completed_segments") != completed:
        raise RuntimeError("checkpoint payload/manifest completed-prefix mismatch")
    atomic_json(
        root / "rebind-receipt.json",
        {
            "schema": "banana-smasher-checkpoint-rebind-v1",
            "status": "ATOMIC_REBIND",
            "identity": expected_identity,
            "payload_member": payload_path.name,
            "payload_sha256": manifest["payload_sha256"],
        },
    )
    return payload, manifest


def finalize_checkpoint(
    checkpoint_dir: str | Path,
    *,
    receipt: str | Path,
    output_record: dict[str, Any],
) -> dict[str, Any]:
    root = Path(checkpoint_dir).resolve()
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest.update(
        {
            "status": "COMPLETE",
            "transaction_state": "complete",
            "receipt_member": Path(receipt).name,
            "output": {
                "sha256": output_record["sha256"],
                "bytes": int(output_record["bytes"]),
            },
        }
    )
    atomic_json(manifest_path, manifest)
    return manifest
