from __future__ import annotations

import argparse
import hashlib
import ipaddress
import json
import os
import subprocess
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable

from .locality import require_local_path

SCHEMA = "banana-smasher-qsfp-stage-v1"
BATCH_SCHEMA = "banana-smasher-qsfp-stage-v2"
QSFP_NETWORK = ipaddress.ip_network("192.168.200.0/24")


def _canonical(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, value: object) -> None:
    payload = _canonical(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def _qsfp_host(value: object) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError("stage row requires source_host")
    address_text = value.rsplit("@", 1)[-1]
    try:
        address = ipaddress.ip_address(address_text)
    except ValueError as exc:
        raise ValueError(
            f"source_host must be a direct QSFP address, not an alias: {value!r}"
        ) from exc
    if address not in QSFP_NETWORK:
        raise ValueError(f"source_host is not on the QSFP fabric: {value!r}")
    return value


def _load_manifest(path: Path) -> tuple[dict[str, Any], bytes]:
    raw = path.read_bytes()
    value = json.loads(raw)
    if value.get("schema") not in {SCHEMA, BATCH_SCHEMA} or value.get("status") != "READY":
        raise ValueError(
            f"stage manifest schema/status must be {SCHEMA} or {BATCH_SCHEMA}, and READY"
        )
    items = value.get("items")
    if not isinstance(items, list) or not items:
        raise ValueError("stage manifest requires non-empty items")
    return value, raw


def _validate_item(item: object, output_root: Path) -> dict[str, Any]:
    if not isinstance(item, dict):
        raise ValueError("stage item must be an object")
    host = _qsfp_host(item.get("source_host"))
    source = Path(str(item.get("source_path", "")))
    destination = Path(str(item.get("destination", "")))
    if not source.is_absolute():
        raise ValueError(f"source_path must be absolute: {source}")
    if destination.is_absolute() or not destination.parts or ".." in destination.parts:
        raise ValueError(f"destination must be a safe relative path: {destination}")
    target = (output_root / destination).resolve(strict=False)
    try:
        target.relative_to(output_root)
    except ValueError as exc:
        raise ValueError(f"destination escapes output root: {destination}") from exc
    expected_bytes = item.get("bytes")
    if expected_bytes is not None and (
        isinstance(expected_bytes, bool)
        or not isinstance(expected_bytes, int)
        or expected_bytes < 0
    ):
        raise ValueError("stage item bytes must be a nonnegative integer")
    return {
        "source_host": host,
        "source_path": str(source),
        "destination": destination.as_posix(),
        "bytes": expected_bytes,
    }


def _validate_batch_item(item: object, output_root: Path) -> dict[str, Any]:
    if not isinstance(item, dict):
        raise ValueError("stage item must be an object")
    host = _qsfp_host(item.get("source_host"))
    source_root = Path(str(item.get("source_root", "")))
    destination = Path(str(item.get("destination", "")))
    relative_paths = item.get("relative_paths")
    if not source_root.is_absolute():
        raise ValueError(f"source_root must be absolute: {source_root}")
    if destination.is_absolute() or not destination.parts or ".." in destination.parts:
        raise ValueError(f"destination must be a safe relative path: {destination}")
    target = (output_root / destination).resolve(strict=False)
    try:
        target.relative_to(output_root)
    except ValueError as exc:
        raise ValueError(f"destination escapes output root: {destination}") from exc
    if not isinstance(relative_paths, list) or not relative_paths:
        raise ValueError("batched stage item requires non-empty relative_paths")
    checked: list[str] = []
    for value in relative_paths:
        relative = Path(str(value))
        if relative.is_absolute() or not relative.parts or ".." in relative.parts:
            raise ValueError(f"relative path is unsafe: {relative}")
        checked.append(relative.as_posix())
    expected_bytes = item.get("bytes")
    if expected_bytes is not None and (
        isinstance(expected_bytes, bool)
        or not isinstance(expected_bytes, int)
        or expected_bytes < 0
    ):
        raise ValueError("stage item bytes must be a nonnegative integer")
    return {
        "source_host": host,
        "source_root": str(source_root),
        "destination": destination.as_posix(),
        "relative_paths": checked,
        "bytes": expected_bytes,
    }


def _tree_bytes(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    return sum(entry.stat().st_size for entry in path.rglob("*") if entry.is_file())


def _rsync_item(item: dict[str, Any], output_root: Path) -> dict[str, Any]:
    destination = output_root / item["destination"]
    destination.mkdir(parents=True, exist_ok=True)
    source = item["source_path"].rstrip("/") + "/"
    command = [
        "rsync",
        "-a",
        "--partial",
        "--inplace",
        "-e",
        "ssh -o BatchMode=yes -o ConnectTimeout=10",
        f"{item['source_host']}:{source}",
        str(destination) + "/",
    ]
    started = time.monotonic()
    completed = subprocess.run(command, capture_output=True, text=True)
    if completed.returncode != 0:
        raise RuntimeError(
            f"QSFP rsync failed for {item['source_host']}:{source}: "
            f"{completed.stderr[-2000:]}"
        )
    actual_bytes = _tree_bytes(destination)
    expected_bytes = item.get("bytes")
    if expected_bytes is not None and actual_bytes != expected_bytes:
        raise ValueError(
            f"staged byte mismatch for {item['destination']}: "
            f"expected={expected_bytes} actual={actual_bytes}"
        )
    return {
        **item,
        "actual_bytes": actual_bytes,
        "elapsed_seconds": time.monotonic() - started,
        "status": "PASS",
    }


def _rsync_batch_item(item: dict[str, Any], output_root: Path) -> dict[str, Any]:
    destination = output_root / item["destination"]
    destination.mkdir(parents=True, exist_ok=True)
    command = [
        "rsync",
        "-a",
        "--partial",
        "--relative",
        "--files-from=-",
        "-e",
        "ssh -o BatchMode=yes -o ConnectTimeout=10",
        f"{item['source_host']}:{item['source_root'].rstrip('/')}/",
        str(destination) + "/",
    ]
    started = time.monotonic()
    completed = subprocess.run(
        command,
        input="\n".join(item["relative_paths"]) + "\n",
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"QSFP batched rsync failed for {item['source_host']}:{item['source_root']}: "
            f"{completed.stderr[-2000:]}"
        )
    missing = [
        relative
        for relative in item["relative_paths"]
        if not (destination / relative).is_file()
    ]
    if missing:
        raise RuntimeError(
            f"QSFP batched stage left {len(missing)} files missing; first={missing[0]}"
        )
    actual_bytes = sum((destination / relative).stat().st_size for relative in item["relative_paths"])
    expected_bytes = item.get("bytes")
    if expected_bytes is not None and actual_bytes != expected_bytes:
        raise ValueError(
            f"staged byte mismatch for {item['destination']}: "
            f"expected={expected_bytes} actual={actual_bytes}"
        )
    return {
        **item,
        "files": len(item["relative_paths"]),
        "actual_bytes": actual_bytes,
        "elapsed_seconds": time.monotonic() - started,
        "status": "PASS",
    }


def stage_qsfp_manifest(
    manifest_path: str | Path,
    output_root: str | Path,
    *,
    parallelism: int = 8,
    transfer: Callable[[dict[str, Any], Path], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Explicitly pre-stage payload directories over direct QSFP SSH.

    Compute APIs never invoke this function. Callers stage first, inspect the PASS
    receipt, and then pass only the local output paths to compute.
    """

    if parallelism < 1:
        raise ValueError("parallelism must be positive")
    manifest_path = Path(manifest_path).expanduser().resolve()
    output_root = Path(output_root).expanduser().resolve(strict=False)
    require_local_path(output_root, label="stage_output")
    manifest, manifest_raw = _load_manifest(manifest_path)
    batched = manifest["schema"] == BATCH_SCHEMA
    validator = _validate_batch_item if batched else _validate_item
    items = [validator(item, output_root) for item in manifest["items"]]
    if batched:
        targets = [
            (item["destination"], relative)
            for item in items
            for relative in item["relative_paths"]
        ]
        if len(set(targets)) != len(targets):
            raise ValueError("batched stage manifest has duplicate destination files")
    elif len({item["destination"] for item in items}) != len(items):
        raise ValueError("stage manifest has duplicate destinations")
    output_root.mkdir(parents=True, exist_ok=True)
    runner = transfer or (_rsync_batch_item if batched else _rsync_item)
    started = time.monotonic()
    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=min(parallelism, len(items))) as executor:
        futures = {executor.submit(runner, item, output_root): item for item in items}
        for future in as_completed(futures):
            results.append(future.result())
    results.sort(key=lambda row: row["destination"])
    elapsed = time.monotonic() - started
    total_bytes = sum(row["actual_bytes"] for row in results)
    receipt = {
        "schema": "banana-smasher-qsfp-stage-receipt-v1",
        "status": "PASS",
        "manifest_sha256": hashlib.sha256(manifest_raw).hexdigest(),
        "manifest_schema": manifest["schema"],
        "output_root": str(output_root),
        "parallelism": min(parallelism, len(items)),
        "items": results,
        "bytes": total_bytes,
        "elapsed_seconds": elapsed,
        "bytes_per_second": total_bytes / elapsed if elapsed else None,
        "transport": "direct-qsfp-ssh-rsync",
    }
    _atomic_json(output_root / "STAGE_RECEIPT.json", receipt)
    return receipt


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Explicitly stage local Backpack inputs over QSFP")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--parallelism", type=int, default=8)
    args = parser.parse_args(argv)
    print(
        json.dumps(
            stage_qsfp_manifest(
                args.manifest, args.output, parallelism=args.parallelism
            ),
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
