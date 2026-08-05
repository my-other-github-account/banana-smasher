#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import importlib
import importlib.metadata as metadata
import json
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any, Iterable

SCHEMA = "banana-smasher-plugin-derivative-v1"
RUNTIME_PREFIX = "banana-smasher-plugin/src/banana_smasher_plugin/"
TEST_PREFIX = "banana-smasher-plugin/tests/"
SOURCE_INVENTORY = "provenance/SOURCE_INVENTORY.json"
PURE_SUFFIXES = {".py", ".json", ".npy"}
HEX40 = re.compile(r"[0-9a-f]{40}")
IMAGE_ID = re.compile(r"sha256:[0-9a-f]{64}")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical_json(value: dict[str, Any]) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def manifest_digest(manifest: dict[str, Any]) -> str:
    unsigned = dict(manifest)
    unsigned.pop("manifest_sha256", None)
    return sha256_bytes(canonical_json(unsigned))


def _safe_relative(path: str) -> PurePosixPath:
    relative = PurePosixPath(path)
    if relative.is_absolute() or ".." in relative.parts or "." in relative.parts:
        raise RuntimeError(f"DERIVATIVE_CHANGE_REFUSE unsafe path: {path}")
    return relative


def classify_change(status: str, path: str) -> tuple[str, str | None]:
    if status not in {"A", "M"}:
        raise RuntimeError(f"DERIVATIVE_STATUS_REFUSE status={status} path={path}")
    relative = _safe_relative(path)
    suffix = relative.suffix.lower()
    if path.startswith(RUNTIME_PREFIX) and suffix in PURE_SUFFIXES:
        distribution_path = path[len("banana-smasher-plugin/src/") :]
        if "/csrc/" in f"/{distribution_path}" or distribution_path.endswith(".so"):
            raise RuntimeError(f"DERIVATIVE_CHANGE_REFUSE native asset: {path}")
        return "runtime", distribution_path
    if path.startswith(TEST_PREFIX) and suffix == ".py":
        return "test", None
    if path == SOURCE_INVENTORY:
        return "provenance", None
    raise RuntimeError(f"DERIVATIVE_CHANGE_REFUSE non-pure-plugin path: {path}")


def _git(repo: Path, *args: str, binary: bool = False) -> str | bytes:
    completed = subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=not binary,
    )
    return completed.stdout


def _commit(repo: Path, revision: str) -> str:
    value = str(_git(repo, "rev-parse", f"{revision}^{{commit}}")).strip()
    if HEX40.fullmatch(value) is None:
        raise RuntimeError(f"DERIVATIVE_COMMIT_REFUSE {revision} -> {value}")
    return value


def _tree(repo: Path, commit: str) -> str:
    value = str(_git(repo, "rev-parse", f"{commit}^{{tree}}")).strip()
    if HEX40.fullmatch(value) is None:
        raise RuntimeError(f"DERIVATIVE_TREE_REFUSE {commit} -> {value}")
    return value


def _git_object(repo: Path, commit: str, path: str) -> bytes:
    data = _git(repo, "show", f"{commit}:{path}", binary=True)
    assert isinstance(data, bytes)
    return data


def build_manifest(
    repo: Path,
    base_revision: str,
    target_revision: str,
    parent_image_id: str,
) -> tuple[dict[str, Any], dict[str, bytes]]:
    repo = repo.resolve()
    base = _commit(repo, base_revision)
    target = _commit(repo, target_revision)
    if IMAGE_ID.fullmatch(parent_image_id) is None:
        raise RuntimeError("DERIVATIVE_PARENT_IMAGE_REFUSE expected sha256 image id")
    raw = str(
        _git(
            repo,
            "diff",
            "--name-status",
            "--no-renames",
            base,
            target,
        )
    )
    if not raw.strip():
        raise RuntimeError("DERIVATIVE_EMPTY_REFUSE no changed files")

    payloads: dict[str, bytes] = {}
    changes: list[dict[str, Any]] = []
    runtime_assets: list[dict[str, Any]] = []
    source_inventory: dict[str, Any] | None = None
    for line in raw.splitlines():
        fields = line.split("\t")
        if len(fields) != 2:
            raise RuntimeError(f"DERIVATIVE_DIFF_REFUSE malformed row: {line!r}")
        status, path = fields
        role, distribution_path = classify_change(status, path)
        data = _git_object(repo, target, path)
        payloads[path] = data
        row: dict[str, Any] = {
            "status": status,
            "role": role,
            "source_path": path,
            "bytes": len(data),
            "sha256": sha256_bytes(data),
        }
        if distribution_path is not None:
            row["distribution_path"] = distribution_path
            runtime_assets.append(dict(row))
        elif role == "provenance":
            source_inventory = {
                "source_path": path,
                "staged_path": SOURCE_INVENTORY,
                "bytes": len(data),
                "sha256": sha256_bytes(data),
            }
        changes.append(row)

    if not runtime_assets:
        raise RuntimeError("DERIVATIVE_RUNTIME_ASSET_REFUSE no installed plugin assets changed")
    manifest: dict[str, Any] = {
        "schema": SCHEMA,
        "base_commit": base,
        "source_commit": target,
        "source_tree": _tree(repo, target),
        "parent_image_id": parent_image_id,
        "allowed_runtime_suffixes": sorted(PURE_SUFFIXES),
        "changes": changes,
        "runtime_assets": runtime_assets,
        "source_inventory": source_inventory,
    }
    manifest["manifest_sha256"] = manifest_digest(manifest)
    return manifest, payloads


def _atomic_write(path: Path, data: bytes, mode: int = 0o644) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temporary, mode)
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _record_path(distribution: Any) -> Path:
    matches = [
        Path(distribution.locate_file(member))
        for member in distribution.files or ()
        if str(member).endswith(".dist-info/RECORD")
    ]
    if len(matches) != 1 or not matches[0].is_file():
        raise RuntimeError(f"DERIVATIVE_RECORD_REFUSE matches={matches}")
    return matches[0]


def _record_digest(data: bytes) -> str:
    return base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode()


def _rewrite_record(record: Path, replacements: dict[str, bytes]) -> None:
    rows = list(csv.reader(record.read_text().splitlines()))
    seen: set[str] = set()
    for row in rows:
        if len(row) != 3:
            raise RuntimeError(f"DERIVATIVE_RECORD_REFUSE malformed row={row}")
        data = replacements.get(row[0])
        if data is not None:
            row[1] = f"sha256={_record_digest(data)}"
            row[2] = str(len(data))
            seen.add(row[0])
    missing = set(replacements) - seen
    if missing:
        raise RuntimeError(f"DERIVATIVE_RECORD_MEMBER_REFUSE missing={sorted(missing)}")
    output = []
    for row in rows:
        output.append(",".join(row))
    _atomic_write(record, ("\n".join(output) + "\n").encode())


def _validate_manifest(manifest: dict[str, Any]) -> None:
    if manifest.get("schema") != SCHEMA:
        raise RuntimeError("DERIVATIVE_MANIFEST_SCHEMA_REFUSE")
    for key in ("base_commit", "source_commit", "source_tree"):
        if HEX40.fullmatch(str(manifest.get(key, ""))) is None:
            raise RuntimeError(f"DERIVATIVE_MANIFEST_IDENTITY_REFUSE {key}")
    if IMAGE_ID.fullmatch(str(manifest.get("parent_image_id", ""))) is None:
        raise RuntimeError("DERIVATIVE_MANIFEST_IDENTITY_REFUSE parent_image_id")
    if manifest.get("manifest_sha256") != manifest_digest(manifest):
        raise RuntimeError("DERIVATIVE_MANIFEST_SHA_REFUSE")
    assets = manifest.get("runtime_assets")
    if not isinstance(assets, list) or not assets:
        raise RuntimeError("DERIVATIVE_RUNTIME_ASSET_REFUSE")
    for row in assets:
        role, distribution_path = classify_change(row.get("status"), row.get("source_path"))
        if role != "runtime" or distribution_path != row.get("distribution_path"):
            raise RuntimeError("DERIVATIVE_RUNTIME_MAPPING_REFUSE")


def _verify_imported_assets(package_root: Path, assets: Iterable[dict[str, Any]]) -> None:
    importlib.invalidate_caches()
    importlib.import_module("banana_smasher_plugin")
    for row in assets:
        distribution_path = row["distribution_path"]
        path = package_root.parent / distribution_path
        if sha256_bytes(path.read_bytes()) != row["sha256"]:
            raise RuntimeError(f"DERIVATIVE_POST_APPLY_SHA_REFUSE {distribution_path}")
        if path.suffix == ".py":
            module = distribution_path[:-3].replace("/", ".")
            if module.endswith(".__init__"):
                module = module[: -len(".__init__")]
            importlib.import_module(module)


def _remove_pycache(package_root: Path) -> None:
    for cache in sorted(package_root.rglob("__pycache__"), reverse=True):
        if cache.is_dir() and not cache.is_symlink():
            shutil.rmtree(cache)
    remaining = sorted(package_root.rglob("__pycache__"))
    if remaining:
        raise RuntimeError(f"DERIVATIVE_PYCACHE_REFUSE remaining={remaining}")


def apply_derivative(
    manifest: dict[str, Any],
    staged_root: Path,
    provenance_root: Path,
    *,
    distribution: Any | None = None,
    verify_imports: bool = True,
) -> dict[str, Any]:
    _validate_manifest(manifest)
    staged_root = staged_root.resolve()
    provenance_root = provenance_root.resolve()
    if distribution is None:
        distribution = metadata.distribution("banana-smasher-plugin")
    if str(getattr(distribution, "version", "")) != "0.2.0":
        raise RuntimeError("DERIVATIVE_DISTRIBUTION_VERSION_REFUSE")
    package_root = Path(distribution.locate_file("banana_smasher_plugin")).resolve()
    if not package_root.is_dir() or package_root.is_symlink():
        raise RuntimeError(f"DERIVATIVE_DISTRIBUTION_LOCATION_REFUSE {package_root}")

    replacements: dict[str, bytes] = {}
    applied: list[dict[str, Any]] = []
    for row in manifest["runtime_assets"]:
        member = row["distribution_path"]
        source = staged_root / member
        if source.is_symlink() or not source.is_file():
            raise RuntimeError(f"DERIVATIVE_STAGED_ASSET_REFUSE {source}")
        data = source.read_bytes()
        actual = {"bytes": len(data), "sha256": sha256_bytes(data)}
        expected = {"bytes": row["bytes"], "sha256": row["sha256"]}
        if actual != expected:
            raise RuntimeError(
                f"DERIVATIVE_STAGED_ASSET_SHA_REFUSE {member} actual={actual} expected={expected}"
            )
        destination = package_root.parent / member
        try:
            destination.resolve().relative_to(package_root)
        except ValueError as exc:
            raise RuntimeError(f"DERIVATIVE_DESTINATION_REFUSE {destination}") from exc
        _atomic_write(destination, data)
        replacements[member] = data
        applied.append({"distribution_path": member, **actual})

    _remove_pycache(package_root)
    record = _record_path(distribution)
    _rewrite_record(record, replacements)

    inventory = manifest.get("source_inventory")
    if inventory is not None:
        source = staged_root / inventory["staged_path"]
        data = source.read_bytes()
        if {"bytes": len(data), "sha256": sha256_bytes(data)} != {
            "bytes": inventory["bytes"],
            "sha256": inventory["sha256"],
        }:
            raise RuntimeError("DERIVATIVE_SOURCE_INVENTORY_SHA_REFUSE")
        _atomic_write(provenance_root / "SOURCE_INVENTORY.json", data)

    source_path = provenance_root / "source.json"
    source_receipt = json.loads(source_path.read_text())
    source_receipt.update(
        {
            "banana_smasher_source_commit": manifest["source_commit"],
            "banana_smasher_source_tree": manifest["source_tree"],
            "derived_from_image": manifest["parent_image_id"],
            "plugin_derivative_manifest_sha256": manifest["manifest_sha256"],
            "plugin_derivative_assets": applied,
        }
    )
    _atomic_write(
        source_path,
        (json.dumps(source_receipt, indent=2, sort_keys=True) + "\n").encode(),
    )
    if verify_imports:
        _verify_imported_assets(package_root, manifest["runtime_assets"])
        _remove_pycache(package_root)
    return {
        "status": "PASS_PLUGIN_DERIVATIVE_APPLIED",
        "distribution_root": str(package_root),
        "record": str(record),
        "record_sha256": sha256_bytes(record.read_bytes()),
        "manifest_sha256": manifest["manifest_sha256"],
        "assets": applied,
        "source_commit": manifest["source_commit"],
        "source_tree": manifest["source_tree"],
        "parent_image_id": manifest["parent_image_id"],
        "imports_verified": verify_imports,
    }


def _load_manifest(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise RuntimeError("DERIVATIVE_MANIFEST_OBJECT_REFUSE")
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    apply_parser = subparsers.add_parser("apply")
    apply_parser.add_argument("--manifest", type=Path, required=True)
    apply_parser.add_argument("--assets-root", type=Path, required=True)
    apply_parser.add_argument("--provenance-root", type=Path, required=True)
    apply_parser.add_argument("--manifest-sha256", required=True)
    apply_parser.add_argument("--source-commit", required=True)
    apply_parser.add_argument("--source-tree", required=True)
    apply_parser.add_argument("--parent-image-id", required=True)
    arguments = parser.parse_args()
    manifest = _load_manifest(arguments.manifest)
    expected = {
        "manifest_sha256": arguments.manifest_sha256,
        "source_commit": arguments.source_commit,
        "source_tree": arguments.source_tree,
        "parent_image_id": arguments.parent_image_id,
    }
    actual = {key: manifest.get(key) for key in expected}
    if actual != expected:
        raise RuntimeError(
            f"DERIVATIVE_BUILD_ARG_REFUSE actual={actual} expected={expected}"
        )
    result = apply_derivative(
        manifest,
        arguments.assets_root,
        arguments.provenance_root,
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
