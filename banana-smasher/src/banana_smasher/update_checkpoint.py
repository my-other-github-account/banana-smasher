from __future__ import annotations

import hashlib
import hmac
import json
import os
import re
import secrets
from pathlib import Path
from typing import Any

from .token_sizing import require_integer

CHECKPOINT_SCHEMA = "banana-smasher-update-checkpoint-v2"
MANIFEST_SCHEMA = "banana-smasher-update-checkpoint-manifest-v2"
REBIND_SCHEMA = "banana-smasher-update-checkpoint-rebind-v1"
REQUIRED_IDENTITY_SHA_FIELDS = (
    "content_sha256",
    "config_sha256",
    "assignment_sha256",
    "aot_sha256",
    "runtime_sha256",
    "code_sha256",
)
AUTHENTICATION_SCHEMA = "banana-smasher-update-checkpoint-auth-v1"
_AUTH_KEY_NAME = ".checkpoint-auth-key-v1"
_AUTH_EXCLUDED_FIELDS = {"authentication"}


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


def _root_sha256(root: Path) -> str:
    return hashlib.sha256(os.fsencode(str(root.resolve()))).hexdigest()


def _authentication_key(root: Path, *, create: bool) -> bytes:
    path = root / _AUTH_KEY_NAME
    if create and not path.exists():
        key = secrets.token_bytes(32)
        try:
            descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        except FileExistsError:
            pass
        else:
            try:
                with os.fdopen(descriptor, "wb", closefd=True) as stream:
                    stream.write(key)
                    stream.flush()
                    os.fsync(stream.fileno())
            except BaseException:
                path.unlink(missing_ok=True)
                raise
            _fsync_directory(root)
    try:
        key = path.read_bytes()
    except FileNotFoundError as exc:
        raise RuntimeError("checkpoint authentication key is missing") from exc
    if len(key) != 32:
        raise RuntimeError("checkpoint authentication key is invalid")
    return key


def _authentication_message(manifest: dict[str, Any]) -> bytes:
    authenticated = {
        name: value
        for name, value in manifest.items()
        if name not in _AUTH_EXCLUDED_FIELDS
    }
    return json.dumps(
        authenticated,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()


def _authenticate_manifest(
    root: Path, manifest: dict[str, Any], *, create_key: bool
) -> None:
    key = _authentication_key(root, create=create_key)
    tag = hmac.new(key, _authentication_message(manifest), hashlib.sha256).hexdigest()
    manifest["authentication"] = {
        "schema": AUTHENTICATION_SCHEMA,
        "algorithm": "hmac-sha256",
        "key_path": _AUTH_KEY_NAME,
        "tag": tag,
    }


def _verify_manifest_authentication(root: Path, manifest: dict[str, Any]) -> None:
    record = manifest.get("authentication")
    if not isinstance(record, dict):
        raise RuntimeError("checkpoint manifest authentication is missing")
    if (
        record.get("schema") != AUTHENTICATION_SCHEMA
        or record.get("algorithm") != "hmac-sha256"
        or record.get("key_path") != _AUTH_KEY_NAME
    ):
        raise RuntimeError("checkpoint manifest authentication metadata is invalid")
    actual = record.get("tag")
    if not isinstance(actual, str):
        raise RuntimeError("checkpoint manifest authentication tag is invalid")
    key = _authentication_key(root, create=False)
    expected = hmac.new(
        key, _authentication_message(manifest), hashlib.sha256
    ).hexdigest()
    if not hmac.compare_digest(actual, expected):
        raise RuntimeError("checkpoint manifest authentication mismatch")


def canonical_identity(identity: dict[str, Any]) -> dict[str, Any]:
    try:
        result = json.loads(json.dumps(identity, sort_keys=True, allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"update identity must be canonical JSON: {exc}") from exc
    if not isinstance(result, dict):
        raise ValueError("update identity must be a JSON object")
    for name in REQUIRED_IDENTITY_SHA_FIELDS:
        value = result.get(name)
        if not isinstance(value, str) or len(value) != 64:
            raise ValueError(f"immutable identity requires 64-hex {name}")
        try:
            int(value, 16)
        except ValueError as exc:
            raise ValueError(f"immutable identity requires 64-hex {name}") from exc
    return result


def atomic_json(path: Path, value: dict[str, Any]) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    data = (
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode()
    with temporary.open("wb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    _fsync_directory(path.parent)
    return {
        "path": str(path),
        "sha256": hashlib.sha256(data).hexdigest(),
        "bytes": len(data),
    }


def atomic_torch_save(path: Path, value: Any) -> dict[str, Any]:
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as stream:
        torch.save(value, stream)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    _fsync_directory(path.parent)
    return {"path": str(path), "sha256": _sha256(path), "bytes": path.stat().st_size}


def _relative(root: Path, path: Path) -> str:
    return os.path.relpath(path.resolve(), root.resolve())


def _resolve_relative(root: Path, value: str, *, inside_root: bool) -> Path:
    relative = Path(value)
    if relative.is_absolute():
        raise RuntimeError("checkpoint manifest contains an absolute payload path")
    resolved = (root / relative).resolve()
    if inside_root and resolved.parent != root and root not in resolved.parents:
        raise RuntimeError("checkpoint payload path escapes its root")
    return resolved


def _verified_record(
    root: Path, record: dict[str, Any], *, label: str, inside_root: bool
) -> Path:
    path = _resolve_relative(root, str(record["path"]), inside_root=inside_root)
    if not path.is_file():
        raise RuntimeError(f"checkpoint {label} is missing: {path}")
    if path.stat().st_size != int(record["bytes"]):
        raise RuntimeError(f"checkpoint {label} byte count mismatch")
    if _sha256(path) != record["sha256"]:
        raise RuntimeError(f"checkpoint {label} SHA-256 mismatch")
    return path


def commit_segment_checkpoint(
    checkpoint_dir: str | Path,
    payload: dict[str, Any],
    *,
    identity: dict[str, Any],
    backend: str,
    segment_plan: list[int],
) -> dict[str, Any]:
    """Publish payload first, then atomically publish its checksummed manifest."""
    root = Path(checkpoint_dir).resolve()
    identity = canonical_identity(identity)
    segment_plan = [
        require_integer(f"segment_plan[{index}]", value)
        for index, value in enumerate(segment_plan)
    ]
    if not segment_plan or any(value <= 0 for value in segment_plan):
        raise ValueError(
            f"checkpoint segment_plan must be non-empty and positive: {segment_plan}"
        )
    manifest_path = root / "manifest.json"
    previous_payload: Path | None = None
    previous: dict[str, Any] = {}
    if manifest_path.is_file():
        previous = json.loads(manifest_path.read_text())
        previous_payload = _resolve_relative(
            root, str(previous["payload_path"]), inside_root=True
        )

    run_id = payload.get("run_id")
    if not isinstance(run_id, str) or re.fullmatch(r"[0-9a-f]{32}", run_id) is None:
        raise ValueError(
            "checkpoint run_id must be 32 lowercase hexadecimal characters"
        )
    next_segment_index = require_integer(
        "next_segment_index", payload.get("next_segment_index")
    )
    state = payload.get("state")
    if state not in {"accumulating", "optimizer_pending", "optimizer_done"}:
        raise ValueError(f"unsupported checkpoint transaction state: {state!r}")
    completed_segments = [
        require_integer(f"completed_segments[{index}]", value)
        for index, value in enumerate(payload.get("completed_segments", []))
    ]
    if completed_segments != list(range(next_segment_index)):
        raise ValueError("checkpoint completed segments are not a contiguous prefix")
    optimizer_steps = require_integer(
        "optimizer_steps", payload.get("optimizer_steps", 0)
    )
    payload_path = root / f"payload-{run_id}-{next_segment_index:04d}-{state}.pt"
    payload_value = {
        **payload,
        "schema": CHECKPOINT_SCHEMA,
        "identity": identity,
        "backend": backend,
        "segment_plan": segment_plan,
        "run_id": run_id,
        "next_segment_index": next_segment_index,
        "completed_segments": completed_segments,
        "optimizer_steps": optimizer_steps,
        "state": state,
    }
    root.mkdir(parents=True, exist_ok=True)
    _authentication_key(root, create=True)
    payload_record = atomic_torch_save(payload_path, payload_value)
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "status": "IN_PROGRESS",
        "backend": backend,
        "identity": identity,
        "segment_plan": segment_plan,
        "logical_items": sum(segment_plan),
        "next_segment_index": next_segment_index,
        "completed_segments": completed_segments,
        "optimizer_steps": optimizer_steps,
        "transaction_state": state,
        "payload_path": payload_path.name,
        "payload_sha256": payload_record["sha256"],
        "payload_bytes": payload_record["bytes"],
        "root_binding_sha256": _root_sha256(root),
        "rebind_count": int(previous.get("rebind_count", 0))
        if manifest_path.is_file()
        else 0,
    }
    if manifest_path.is_file() and previous.get("last_rebind_receipt"):
        manifest["last_rebind_receipt"] = previous["last_rebind_receipt"]
    _authenticate_manifest(root, manifest, create_key=False)
    atomic_json(manifest_path, manifest)
    if previous_payload is not None and previous_payload != payload_path:
        previous_payload.unlink(missing_ok=True)
        _fsync_directory(root)
    return manifest


def _atomic_rebind(root: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    new_root_sha = _root_sha256(root)
    old_root_sha = str(manifest.get("root_binding_sha256", ""))
    if old_root_sha == new_root_sha:
        return manifest
    count = int(manifest.get("rebind_count", 0)) + 1
    name = f"rebind-{count:04d}.json"
    receipt = {
        "schema": REBIND_SCHEMA,
        "status": "PASS_REBIND",
        "sequence": count,
        "old_root_sha256": old_root_sha,
        "new_root_sha256": new_root_sha,
        "identity": manifest["identity"],
        "payload_sha256": manifest["payload_sha256"],
    }
    record = atomic_json(root / name, receipt)
    rebound = dict(manifest)
    rebound["root_binding_sha256"] = new_root_sha
    rebound["rebind_count"] = count
    rebound["last_rebind_receipt"] = {
        "path": name,
        "sha256": record["sha256"],
        "bytes": record["bytes"],
    }
    _authenticate_manifest(root, rebound, create_key=False)
    atomic_json(root / "manifest.json", rebound)
    return rebound


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
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"update checkpoint manifest does not exist: {manifest_path}"
        )
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema") != MANIFEST_SCHEMA:
        raise RuntimeError(
            f"unsupported update checkpoint manifest schema: {manifest.get('schema')!r}"
        )
    identity = canonical_identity(expected_identity)
    if manifest.get("identity") != identity:
        raise RuntimeError("checkpoint identity mismatch")
    if manifest.get("backend") != expected_backend:
        raise RuntimeError("checkpoint backend mismatch")
    if (
        expected_segment_plan is not None
        and manifest.get("segment_plan") != expected_segment_plan
    ):
        raise RuntimeError("checkpoint segment geometry mismatch")
    completed = manifest.get("completed_segments")
    next_index = int(manifest.get("next_segment_index", -1))
    if completed != list(range(next_index)):
        raise RuntimeError("checkpoint completed segments are not a contiguous prefix")
    rebind_record = manifest.get("last_rebind_receipt")
    if rebind_record is not None:
        rebind_path = _verified_record(
            root, rebind_record, label="rebind receipt", inside_root=True
        )
        rebind = json.loads(rebind_path.read_text())
        if (
            rebind.get("schema") != REBIND_SCHEMA
            or rebind.get("status") != "PASS_REBIND"
            or rebind.get("identity") != identity
            or rebind.get("new_root_sha256") != manifest.get("root_binding_sha256")
        ):
            raise RuntimeError("checkpoint rebind receipt identity mismatch")

    payload_path = _resolve_relative(
        root, str(manifest["payload_path"]), inside_root=True
    )
    if not payload_path.is_file():
        raise RuntimeError(f"checkpoint payload is missing: {payload_path}")
    if payload_path.stat().st_size != int(manifest["payload_bytes"]):
        raise RuntimeError("checkpoint payload byte count mismatch")
    if _sha256(payload_path) != manifest["payload_sha256"]:
        raise RuntimeError("checkpoint payload SHA-256 mismatch")
    try:
        payload = torch.load(payload_path, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise RuntimeError(f"checkpoint payload cannot be loaded: {exc}") from exc
    _verify_manifest_authentication(root, manifest)
    if payload.get("schema") != CHECKPOINT_SCHEMA:
        raise RuntimeError("unsupported update checkpoint payload schema")
    if payload.get("identity") != identity:
        raise RuntimeError("checkpoint payload identity mismatch")
    if payload.get("backend") != expected_backend:
        raise RuntimeError("checkpoint payload backend mismatch")
    if payload.get("segment_plan") != manifest.get("segment_plan"):
        raise RuntimeError("checkpoint payload segment plan mismatch")
    if int(payload.get("next_segment_index", -1)) != next_index:
        raise RuntimeError("checkpoint payload/manifest next-segment mismatch")
    if payload.get("completed_segments") != completed:
        raise RuntimeError("checkpoint payload/manifest completed-prefix mismatch")

    manifest = _atomic_rebind(root, manifest)
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
    output_path = Path(str(output_record["path"])).resolve()
    receipt_path = Path(receipt).resolve()
    receipt_record = {
        "path": _relative(root, receipt_path),
        "sha256": _sha256(receipt_path),
        "bytes": receipt_path.stat().st_size,
    }
    manifest.update(
        {
            "status": "COMPLETE",
            "transaction_state": "complete",
            "receipt": receipt_record,
            "output": {
                "path": _relative(root, output_path),
                "sha256": output_record["sha256"],
                "bytes": output_record["bytes"],
            },
        }
    )
    _authenticate_manifest(root, manifest, create_key=False)
    atomic_json(manifest_path, manifest)
    return manifest


def verify_completed_files(root: Path, manifest: dict[str, Any]) -> tuple[Path, Path]:
    output = _verified_record(
        root, manifest["output"], label="output", inside_root=False
    )
    receipt = _verified_record(
        root, manifest["receipt"], label="receipt", inside_root=False
    )
    return output, receipt
