#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ctypes
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import re
from importlib import import_module
from pathlib import Path
from typing import Any

EXPECTED_PACKAGES = {
    "banana-smasher": "1.0.0",
    "banana-smasher-plugin": "0.2.0",
    "deep-gemm": "2.6.1",
    "flashinfer-jit-cache": "0.6.17+cu130",
    "flashinfer-python": "0.6.17",
    "tilelang": "0.1.9",
    "vllm": "0.24.0",
}
PROVENANCE_NAMES = (
    "ASSET_MANIFEST.json",
    "ACCELERATION_MANIFEST.json",
    "KERNEL_PRODUCERS.json",
    "SOURCE_INVENTORY.json",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except Exception as exc:
        raise RuntimeError(f"invalid provenance JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"provenance JSON is not an object: {path}")
    return value


def _validate_source_commit(source_commit: str) -> str:
    if re.fullmatch(r"[0-9a-f]{40}", source_commit) is None:
        raise RuntimeError("BANANA_SMASHER_SOURCE_COMMIT must be a full lowercase Git commit")
    return source_commit


def stamp_provenance_source_commit(provenance_root: Path, source_commit: str) -> None:
    source_commit = _validate_source_commit(source_commit)
    for name in PROVENANCE_NAMES:
        path = provenance_root / name
        payload = _load_object(path)
        payload["source_commit"] = source_commit
        payload.pop("source_status", None)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        temporary.replace(path)


def _identity(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"asset is not a local regular file: {path}")
    return {"bytes": path.stat().st_size, "name": path.name, "sha256": sha256(path)}


def _exact_group(directory: Path, expected: list[dict[str, Any]], group: str) -> list[dict[str, Any]]:
    if directory.is_symlink() or not directory.is_dir():
        raise RuntimeError(f"asset group is not a local directory: {directory}")
    entries = sorted(directory.iterdir(), key=lambda item: item.name)
    if any(entry.is_symlink() for entry in entries):
        raise RuntimeError(f"expected one exact asset set for {group}; symlink member present")
    if any(not entry.is_file() for entry in entries):
        raise RuntimeError(f"expected one exact asset set for {group}; non-regular member present")
    actual = [_identity(entry) for entry in entries]
    expected_sorted = sorted(expected, key=lambda item: item["name"])
    if [item["name"] for item in actual] != [item["name"] for item in expected_sorted]:
        raise RuntimeError(
            f"expected one exact asset set for {group}: "
            f"actual={[item['name'] for item in actual]} "
            f"expected={[item['name'] for item in expected_sorted]}"
        )
    for got, wanted in zip(actual, expected_sorted):
        if got != wanted:
            raise RuntimeError(
                f"asset identity mismatch for {group}/{got['name']}: "
                f"actual={got} expected={wanted}"
            )
    return actual


def verify_asset_set(
    manifest_path: Path,
    aot_root: Path,
    tlut_path: Path,
) -> dict[str, Any]:
    manifest = _load_object(manifest_path)
    if manifest.get("schema") != "banana-smasher-active-assets-v1":
        raise RuntimeError("active asset manifest schema mismatch")

    groups = manifest.get("groups")
    if not isinstance(groups, dict) or set(groups) != {"sm120_cubins", "e43_cubins"}:
        raise RuntimeError("active asset manifest groups mismatch")
    actual = {
        "sm120_cubins": _exact_group(
            aot_root / "cubins-sm120", groups["sm120_cubins"], "sm120_cubins"
        ),
        "e43_cubins": _exact_group(
            aot_root / "cubins-e43", groups["e43_cubins"], "e43_cubins"
        ),
    }
    tlut = _identity(tlut_path)
    if tlut != manifest.get("qtip_tlut"):
        raise RuntimeError(
            f"asset identity mismatch for qtip_tlut: actual={tlut} "
            f"expected={manifest.get('qtip_tlut')}"
        )
    return {
        "status": "PASS",
        "counts": {name: len(records) for name, records in actual.items()},
        "assets": actual,
        "qtip_tlut": tlut,
    }


def _asset_mapping(asset_manifest: dict[str, Any]) -> dict[str, dict[str, str]]:
    return {
        group: {item["name"]: item["sha256"] for item in records}
        for group, records in asset_manifest["groups"].items()
    }


def verify_provenance_manifests(
    provenance_root: Path,
    source_commit: str,
) -> dict[str, Any]:
    source_commit = _validate_source_commit(source_commit)
    required = {
        name: _load_object(provenance_root / name)
        for name in PROVENANCE_NAMES
    }
    schemas = {
        "ASSET_MANIFEST.json": "banana-smasher-active-assets-v1",
        "ACCELERATION_MANIFEST.json": "banana-smasher-acceleration-manifest-v1",
        "KERNEL_PRODUCERS.json": "banana-smasher-kernel-producers-v1",
    }
    for name, schema in schemas.items():
        if required[name].get("schema") != schema:
            raise RuntimeError(f"provenance schema mismatch: {name}")
    for name, payload in required.items():
        if payload.get("source_commit") != source_commit:
            raise RuntimeError(f"provenance source commit mismatch: {name}")

    asset_manifest = required["ASSET_MANIFEST.json"]
    expected = _asset_mapping(asset_manifest)
    producers = required["KERNEL_PRODUCERS.json"].get("producers")
    if not isinstance(producers, dict) or set(producers) != {"sm120", "e43"}:
        raise RuntimeError("kernel producer groups mismatch")
    producer_mapping = {
        "sm120_cubins": {
            item["name"]: item["sha256"] for item in producers["sm120"]["assets"]
        },
        "e43_cubins": {
            item["name"]: item["sha256"] for item in producers["e43"]["assets"]
        },
    }
    if producer_mapping != expected:
        raise RuntimeError(
            f"producer asset mapping mismatch: actual={producer_mapping} expected={expected}"
        )

    acceleration = required["ACCELERATION_MANIFEST.json"].get("accelerations")
    cache = next(
        (item for item in acceleration or [] if item.get("id") == "flashinfer-autotune-cache"),
        None,
    )
    if cache is None or {
        "status": cache.get("status"),
        "active_cache_baked": cache.get("active_cache_baked"),
        "required_version": cache.get("required_version"),
        "architecture": cache.get("architecture"),
    } != {
        "status": "gpu-regeneration-outstanding",
        "active_cache_baked": False,
        "required_version": "0.6.17",
        "architecture": "121a",
    }:
        raise RuntimeError("FlashInfer cache provenance is not fail-closed")

    inventory = required["SOURCE_INVENTORY.json"]
    records = list(inventory.get("files", [])) + list(inventory.get("generated_files", []))
    inventory_by_path = {item["path"]: item for item in records}
    for group, directory in (
        ("sm120_cubins", "banana-smasher/kernels/cubins-sm120"),
        ("e43_cubins", "banana-smasher/kernels/cubins-e43"),
    ):
        for name, digest in expected[group].items():
            record = inventory_by_path.get(f"{directory}/{name}")
            if record is None or record.get("output_sha256") != digest:
                raise RuntimeError(f"source inventory asset mismatch: {group}/{name}")
    tlut_record = inventory_by_path.get(
        "banana-smasher-plugin/src/banana_smasher_plugin/qtip_tlut.npy"
    )
    if tlut_record is None or tlut_record.get("output_sha256") != asset_manifest["qtip_tlut"]["sha256"]:
        raise RuntimeError("source inventory TLUT mismatch")
    if any("flashinfer-autotune/0.6.14" in path for path in inventory_by_path):
        raise RuntimeError("source inventory retains excluded FlashInfer cache payload")

    return {
        "status": "PASS",
        "manifest_sha256": {
            name: sha256(provenance_root / name) for name in sorted(required)
        },
        "producer_assets": sum(len(value) for value in producer_mapping.values()),
    }


def one(pattern: str) -> Path:
    matches = sorted(Path("/").glob(pattern.lstrip("/")))
    if len(matches) != 1:
        raise RuntimeError(f"expected one {pattern}, found {matches}")
    return matches[0]


def verify_flashinfer_aot() -> dict[str, Any]:
    flashinfer_jit_cache = import_module("flashinfer_jit_cache")
    root = Path(flashinfer_jit_cache.get_jit_cache_dir())
    modules = sorted(path.parent.name for path in root.glob("*/*.so"))
    required_exact = {"sampling", "sparse_mla_sm120"}
    missing = sorted(required_exact - set(modules))
    if missing:
        raise RuntimeError(f"FlashInfer AOT modules missing: {missing}")
    head_dim_512 = sorted(
        name for name in modules if "head_dim_qk_512_head_dim_vo_512" in name
    )
    if not head_dim_512:
        raise RuntimeError("FlashInfer AOT head-dimension-512 module is missing")
    return {
        "root": str(root),
        "module_count": len(modules),
        "required": sorted(required_exact),
        "head_dim_512": head_dim_512,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stamp-provenance", action="store_true")
    args = parser.parse_args()
    source_commit = _validate_source_commit(
        os.environ.get("BANANA_SMASHER_SOURCE_COMMIT", "")
    )
    provenance_root = Path("/opt/banana-smasher/provenance")
    if args.stamp_provenance:
        stamp_provenance_source_commit(provenance_root, source_commit)

    actual = {name: importlib.metadata.version(name) for name in EXPECTED_PACKAGES}
    if actual != EXPECTED_PACKAGES:
        raise RuntimeError(
            f"package identity mismatch: actual={actual} expected={EXPECTED_PACKAGES}"
        )

    entries = importlib.metadata.entry_points(group="vllm.general_plugins")
    if not any(
        entry.name == "banana_smasher_plugin"
        and entry.value == "banana_smasher_plugin:register"
        for entry in entries
    ):
        raise RuntimeError("banana-smasher vLLM general plugin entry point is missing")

    spec = importlib.util.find_spec("tilelang")
    if spec is None or spec.origin is None:
        raise RuntimeError("tilelang package is missing")
    stub = Path(spec.origin).parent / "lib/libcudart_stub.so"
    real = one(
        "/usr/local/lib/python3.12/dist-packages/nvidia/cu13/lib/libcudart.so.13"
    )
    if not stub.is_symlink() or stub.resolve() != real.resolve():
        raise RuntimeError(f"TileLang libcudart link mismatch: {stub} -> {stub.resolve()}")
    getattr(ctypes.CDLL(real, mode=ctypes.RTLD_GLOBAL), "cudaDeviceReset")
    import flashinfer.comm  # noqa: F401

    plugin = importlib.util.find_spec("banana_smasher_plugin")
    if plugin is None or plugin.origin is None:
        raise RuntimeError("banana-smasher plugin package is missing")
    provenance = verify_provenance_manifests(provenance_root, source_commit)
    assets = verify_asset_set(
        Path("/opt/banana-smasher/provenance/ASSET_MANIFEST.json"),
        Path("/opt/banana-smasher/aot"),
        Path(plugin.origin).parent / "qtip_tlut.npy",
    )
    flashinfer_aot = verify_flashinfer_aot()
    print(
        {
            "status": "PASS",
            "packages": actual,
            "real_libcudart": str(real),
            "assets": assets,
            "flashinfer_aot": flashinfer_aot,
            "provenance": provenance,
        }
    )


if __name__ == "__main__":
    main()
