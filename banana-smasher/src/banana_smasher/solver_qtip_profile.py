#!/usr/bin/env python3
"""Fresh exact QTIP unit profiler used by the public ``smash profile`` verb.

This is a profiling adapter around the sealed QTIP builder and exact-prefix
Viterbi kernels.  It does not change the objective, prune states, or reuse a
prior assignment.  The reference unit is read only after the fresh solve and
is used solely for an exact assignment/trellis digest gate.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import time
import types
from types import MappingProxyType
from typing import Any, Mapping
import weakref

import torch
from safetensors import safe_open

from .qtip_materialize import EXPLICIT_RHT_SEED_POLICY, validate_qtip_projection
from .qtip_matrix_lifetime import build_qtip_bounded
from .qtip_rings import (
    PERSISTENT_BACKENDS,
    TRELLIS_V2_BACKEND,
    assign_ring_geometries,
    backend_for_geometry,
    canonical_qtip_packed_shape,
    known_qtip_geometries,
    resolve_qtip_ring,
    validate_qtip_ring_manifest,
)


QTIP_RHT_DOMAIN = "qtip-rht-manifest-v1"
_TRUSTED_PUBLIC_QTIP_RUNNER_SHA256 = (
    "8c45536a9f2bf8e26d324ed07a474da733f0bc144d56a456944445f71f1717af"
)

# A config-directory solve is one public process. These caches remove repeated
# import, capture-bank, manifest, TLUT, and index staging between independent
# exact units. Candidate state, objectives, codebooks, weights, and assignments
# remain unit-local.
_MODULE_CACHE: dict[Path, Any] = {}
_CAPTURE_CACHE: dict[tuple[Path, int, int], list[dict[str, Any]]] = {}
_HESSIAN_BINDING_CACHE: dict[
    tuple[Path, str, Path, int, int], tuple[Path, int, dict[str, Any]]
] = {}
_TLUT_CACHE: weakref.WeakValueDictionary[Path, torch.Tensor] = (
    weakref.WeakValueDictionary()
)
_MODEL_INDEX_CACHE: dict[Path, dict[str, str]] = {}
_MODEL_SHAPE_CACHE: dict[Path, tuple[int, int]] = {}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _md5(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


_CONFIG_DIR_KEY = "_banana_smasher_config_dir"


def _read_qtip_config(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"QTIP config must be a JSON object: {path}")
    value[_CONFIG_DIR_KEY] = str(path.resolve().parent)
    return value


def _resolve_config_path(config: Mapping[str, Any], value: object) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError("QTIP config path must be a non-empty string")
    path = Path(value)
    if not path.is_absolute():
        base = Path(str(config.get(_CONFIG_DIR_KEY, "."))).resolve()
        path = base / path
    return path.resolve()


def _config_path(config: Mapping[str, Any], key: str) -> Path:
    return _resolve_config_path(config, config.get(key))


def _public_receipt(value: Any) -> Any:
    """Remove host identity and absolute local paths from shareable receipts."""
    if isinstance(value, dict):
        return {
            key: _public_receipt(item)
            for key, item in value.items()
            if key != "host" and key != _CONFIG_DIR_KEY
        }
    if isinstance(value, list):
        return [_public_receipt(item) for item in value]
    if isinstance(value, tuple):
        return [_public_receipt(item) for item in value]
    if isinstance(value, str) and Path(value).is_absolute():
        return Path(value).name
    return value


def _basis_sha(value: Any) -> str | None:
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        for key in ("index_sha256", "source_model_index_sha256", "sha256"):
            candidate = value.get(key)
            if isinstance(candidate, str):
                return candidate
    return None


def _verify_basis(config: dict[str, Any], run_root: Path) -> dict[str, Any]:
    model_root = _config_path(config, "model_root")
    index_path = model_root / "model.safetensors.index.json"
    actual = _sha256(index_path)
    identity = config.get("input_identity")
    configured = (
        _basis_sha(identity.get("model_index"))
        if isinstance(identity, dict)
        else None
    )
    if configured is None:
        configured = _basis_sha(config.get("model_index"))
    if configured is None:
        raise ValueError("QTIP config lacks a SHA-bound model index identity")
    if configured != actual:
        raise ValueError(f"QTIP config model-index mismatch: {actual} != {configured}")

    shards_path = run_root.resolve() / "SHARDS.json"
    shards = json.loads(shards_path.read_text())
    intended = _basis_sha(shards.get("intended_basis"))
    if intended is None:
        raise ValueError(f"SHARDS.json lacks intended_basis: {shards_path}")
    if intended != actual:
        raise ValueError(f"QTIP basis mismatch: {actual} != {intended}")
    return {
        "schema": "banana-smasher-qtip-basis-gate-v1",
        "status": "PASS",
        "index_path": str(index_path),
        "index_sha256": actual,
        "intended_basis": intended,
        "shards_manifest": str(shards_path),
        "shards_manifest_sha256": _sha256(shards_path),
    }


def _canonical_rht_seed(
    seed_material: str,
    layer: int,
    expert: int,
    projection: str,
) -> int:
    if not seed_material:
        raise ValueError("manifest-bound RHT seed material must be non-empty")
    identity = f"L{layer:03d}_E{expert:03d}_{projection}"
    digest = hashlib.sha256(
        f"{QTIP_RHT_DOMAIN}|{seed_material}|{identity}".encode()
    ).digest()
    return int.from_bytes(digest[:8], "big") & ((1 << 63) - 1)


def _resolve_rht_seed(
    config: dict[str, Any],
    reference: dict[str, Any],
    *,
    layer: int,
    expert: int,
    projection: str,
) -> tuple[int, str]:
    policy = str(config.get("rht_seed_policy", "reference-unit-v1"))
    if policy == QTIP_RHT_DOMAIN:
        seed_material = config.get("rht_seed_material")
        if not isinstance(seed_material, str) or not seed_material:
            raise ValueError("manifest-bound RHT seed lacks rht_seed_material")
        expected = _canonical_rht_seed(
            seed_material,
            layer,
            expert,
            projection,
        )
        configured = config.get("rht_seed")
        if not isinstance(configured, int) or configured != expected:
            raise ValueError(
                "canonical RHT seed mismatch: "
                f"configured={configured!r} expected={expected} "
                f"identity=L{layer:03d}_E{expert:03d}_{projection}"
            )
        return expected, policy
    if policy == EXPLICIT_RHT_SEED_POLICY:
        materialization = config.get("materialization")
        if not isinstance(materialization, dict) or any(
            not _is_sha256_digest(materialization.get(key))
            for key in ("run_manifest_sha256", "source_config_sha256")
        ):
            raise ValueError("explicit RHT seed lacks hash-bound materialization")
        configured = config.get("rht_seed")
        if type(configured) is not int or not 0 <= configured < (1 << 63):
            raise ValueError(
                f"explicit RHT seed must be an integer in [0, 2^63): {configured!r}"
            )
        return configured, policy
    if policy != "reference-unit-v1":
        raise ValueError(f"unsupported RHT seed policy: {policy}")
    reference_seed = reference.get("rht_seed")
    if not isinstance(reference_seed, int):
        raise ValueError("reference unit is missing an integer rht_seed")
    configured = config.get("rht_seed", reference_seed)
    if not isinstance(configured, int) or configured != reference_seed:
        raise ValueError(
            f"reference-unit RHT seed mismatch: {configured!r} != {reference_seed}"
        )
    return reference_seed, policy


def _tensor_sha256(value: torch.Tensor) -> str:
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def _fsync_parent(path: Path) -> None:
    directory_fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name("." + path.name + ".tmp")
    with tmp.open("w") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)
    _fsync_parent(path)


def _atomic_torch(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name("." + path.name + ".tmp")
    with tmp.open("wb") as handle:
        torch.save(value, handle)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)
    _fsync_parent(path)


_QTIP_UNIT_PAYLOAD_SCHEMA = "banana-smasher-qtip-unit-v1"
_LEGACY_QTIP_UNIT_PAYLOAD_SCHEMA = "ds4-qtip-hyb-bounded36-unit-v1"
_QTIP_SOLVE_RECEIPT_SCHEMA = "banana-smasher-qtip-solve-v1"
_QTIP_UNIT_REQUIRED_TENSORS = ("trellis", "SU", "SV", "Wscale", "tlut")


def _is_sha256_digest(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _run_intended_basis(root: Path) -> str:
    """Read the run's intended model basis so existing units bind to THIS run."""
    shards_path = root.resolve() / "SHARDS.json"
    try:
        shards = json.loads(shards_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            f"existing QTIP unit cannot bind run basis: {shards_path}"
        ) from exc
    intended = _basis_sha(shards.get("intended_basis"))
    if not _is_sha256_digest(intended):
        raise RuntimeError(
            f"existing QTIP unit cannot bind run basis: {shards_path}"
        )
    assert isinstance(intended, str)
    return intended


def _validated_existing_unit(
    config_path: Path,
    root: Path,
    layer: int,
    *,
    profile_mode: bool,
) -> dict[str, Any] | None:
    """Return the receipt of an immutable, hash-valid existing PASS unit.

    Returns ``None`` when the unit has never been solved (nothing durable
    exists), so the caller computes it fresh.  Any partial, divergent,
    corrupt, or internally inconsistent existing state raises instead of
    silently rerunning or overwriting sealed bytes.  Profiling never
    resumes: a profile receipt is a measurement, not a solve artifact.
    """
    if profile_mode:
        return None
    try:
        config = _read_qtip_config(config_path)
        configured_layer = config["layer"]
        expert = config["expert"]
        projection = validate_qtip_projection(config["projection"])
        geometry = config.get("geometry", {"L": 16, "K": 3, "V": 2})
        sealed_geometry = (geometry["L"], geometry["K"], geometry["V"])
    except (KeyError, TypeError, ValueError, OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            f"existing QTIP unit has invalid config: {config_path}"
        ) from exc
    if (
        isinstance(configured_layer, bool)
        or not isinstance(configured_layer, int)
        or configured_layer != layer
        or isinstance(expert, bool)
        or not isinstance(expert, int)
        or expert < 0
        or any(
            isinstance(value, bool) or not isinstance(value, int)
            for value in sealed_geometry
        )
        or sealed_geometry not in known_qtip_geometries()
    ):
        raise RuntimeError(
            f"existing QTIP unit has invalid config identity: {config_path}"
        )
    out = root / "solve" / f"L{layer:03d}" / f"E{expert:03d}_{projection}"
    artifact_path = out / "QTIP_UNIT.pt"
    receipt_path = out / "QTIP_SOLVE_RECEIPT.json"
    artifact_exists = artifact_path.is_file()
    receipt_exists = receipt_path.is_file()
    if not artifact_exists and not receipt_exists:
        return None
    if artifact_exists != receipt_exists:
        raise RuntimeError(
            "existing QTIP unit is partial: "
            f"payload={artifact_exists} receipt={receipt_exists} unit={out}"
        )
    try:
        receipt = json.loads(receipt_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            f"existing QTIP unit receipt is corrupt: {receipt_path}"
        ) from exc
    if not isinstance(receipt, dict):
        raise RuntimeError(
            f"existing QTIP unit receipt is corrupt: {receipt_path}"
        )
    expected_identity = {
        "schema": _QTIP_SOLVE_RECEIPT_SCHEMA,
        "status": "PASS",
        "layer": layer,
        "expert": expert,
        "projection": projection,
    }
    drift = {
        key: (receipt.get(key), expected)
        for key, expected in expected_identity.items()
        if receipt.get(key) != expected
    }
    if drift:
        raise RuntimeError(
            f"existing QTIP unit identity drift at {receipt_path}: {drift}"
        )
    if receipt.get("config_sha256") != _sha256(config_path):
        raise RuntimeError(
            f"existing QTIP unit config hash drift: {config_path}"
        )
    run_basis = _run_intended_basis(root)
    configured_basis = None
    identity = config.get("input_identity")
    if isinstance(identity, dict):
        configured_basis = _basis_sha(identity.get("model_index"))
    if configured_basis is None:
        configured_basis = _basis_sha(config.get("model_index"))
    try:
        model_index = _config_path(config, "model_root") / "model.safetensors.index.json"
    except (KeyError, OSError) as exc:
        raise RuntimeError(
            f"existing QTIP unit lacks model root: {config_path}"
        ) from exc
    if not model_index.is_file() or _sha256(model_index) != run_basis:
        raise RuntimeError(
            f"existing QTIP unit live model basis drift: {model_index}"
        )
    gate = receipt.get("basis_gate")
    if (
        not isinstance(gate, dict)
        or gate.get("schema") != "banana-smasher-qtip-basis-gate-v1"
        or gate.get("status") != "PASS"
        or gate.get("index_sha256") != run_basis
        or gate.get("intended_basis") != run_basis
        or configured_basis != run_basis
    ):
        raise RuntimeError(f"existing QTIP unit basis drift: {receipt_path}")
    try:
        recorded_artifact = Path(str(receipt["artifact"]))
        if not recorded_artifact.is_absolute():
            recorded_artifact = receipt_path.parent / recorded_artifact
        recorded_artifact = recorded_artifact.resolve()
    except (KeyError, OSError) as exc:
        raise RuntimeError(
            f"existing QTIP unit lacks an artifact path: {receipt_path}"
        ) from exc
    if recorded_artifact != artifact_path.resolve():
        raise RuntimeError(
            "existing QTIP unit artifact path drift: "
            f"{recorded_artifact} != {artifact_path.resolve()}"
        )
    if not _is_sha256_digest(receipt.get("artifact_sha256")):
        raise RuntimeError(
            f"existing QTIP unit lacks a payload hash: {receipt_path}"
        )
    if receipt["artifact_sha256"] != _sha256(artifact_path):
        raise RuntimeError(
            f"existing QTIP unit payload hash drift: {artifact_path}"
        )
    try:
        artifact = torch.load(
            artifact_path,
            map_location="cpu",
            mmap=True,
            weights_only=True,
        )
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        raise RuntimeError(
            f"existing QTIP unit payload is unreadable: {artifact_path}"
        ) from exc
    expected_payload_geometry = {
        "L": sealed_geometry[0],
        "K": sealed_geometry[1],
        "V": sealed_geometry[2],
        "tlut_bits": 9,
        "decode_mode": "quantlut_sym",
        "td_x": 16,
        "td_y": 16,
    }
    if (
        not isinstance(artifact, dict)
        or artifact.get("schema")
        not in {_QTIP_UNIT_PAYLOAD_SCHEMA, _LEGACY_QTIP_UNIT_PAYLOAD_SCHEMA}
        or artifact.get("geometry") != expected_payload_geometry
        or any(
            not isinstance(artifact.get(key), torch.Tensor)
            for key in _QTIP_UNIT_REQUIRED_TENSORS
        )
    ):
        raise RuntimeError(
            f"existing QTIP unit payload schema is invalid: {artifact_path}"
        )
    if not _is_sha256_digest(receipt.get("assignment_sha256")):
        raise RuntimeError(
            f"existing QTIP unit lacks an assignment digest: {receipt_path}"
        )
    if receipt["assignment_sha256"] != _tensor_sha256(artifact["trellis"]):
        raise RuntimeError(
            f"existing QTIP unit assignment digest drift: {artifact_path}"
        )
    total_wall_seconds = receipt.get("total_wall_seconds")
    if (
        isinstance(total_wall_seconds, bool)
        or not isinstance(total_wall_seconds, (int, float))
        or not math.isfinite(total_wall_seconds)
        or total_wall_seconds < 0
    ):
        raise RuntimeError(
            f"existing QTIP unit timing is invalid: {receipt_path}"
        )
    return receipt


def _process_receipt() -> dict[str, int]:
    stat = Path("/proc/self/stat")
    return {
        "pid": os.getpid(),
        "startticks": int(stat.read_text().split()[21]) if stat.is_file() else 0,
    }


def _load_module(name: str, path: Path):
    path = path.resolve()
    cached = _MODULE_CACHE.get(path)
    if cached is not None:
        return cached
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    _MODULE_CACHE[path] = module
    return module


def _manifest_bound_public_qtip_pack(pack):
    """Adapt a legacy runner's pack seam to the manifest-owned wire layout."""

    def manifest_pack(cb, states: torch.Tensor, m: int, k: int):
        binding = getattr(cb, "_banana_smasher_public_runner_pack_contract", None)
        if binding is None:
            raise RuntimeError("public QTIP runner manifest pack binding missing")
        fields = (
            "schema",
            "geometry",
            "matrix_shape",
            "input_tile",
            "dtype",
            "packed_words_per_tile_per_k",
            "output_rows",
            "expected_shape",
        )
        input_tile = binding.get("input_tile") if isinstance(binding, Mapping) else None
        words_per_k = (
            binding.get("packed_words_per_tile_per_k")
            if isinstance(binding, Mapping)
            else None
        )
        raw_geometry = tuple(getattr(cb, key) for key in ("L", "K", "V"))
        geometry = tuple(int(value) for value in raw_geometry)
        vector_width = geometry[2]
        expected_state_shape = (
            (m, k // vector_width)
            if vector_width > 0 and k % vector_width == 0
            else None
        )
        if (
            not isinstance(states, torch.Tensor)
            or expected_state_shape is None
            or tuple(states.shape) != expected_state_shape
        ):
            raise RuntimeError(
                "public QTIP runner manifest input state shape mismatch: "
                f"{getattr(states, 'shape', None)} != {expected_state_shape}"
            )
        if (
            not isinstance(binding, Mapping)
            or len(binding) != len(fields)
            or set(binding) != set(fields)
            or any(
                isinstance(value, bool) or not isinstance(value, int) or value < 1
                for value in raw_geometry
            )
            or binding.get("schema")
            != "banana-smasher-public-runner-pack-contract-v1"
            or binding.get("geometry") != geometry
            or binding.get("matrix_shape") != (m, k)
            or not isinstance(input_tile, tuple)
            or len(input_tile) != 2
            or any(
                isinstance(value, bool) or not isinstance(value, int) or value < 1
                for value in input_tile
            )
            or binding.get("dtype") != "uint16"
            or isinstance(words_per_k, bool)
            or not isinstance(words_per_k, int)
            or words_per_k < 1
            or binding.get("output_rows") != "input_tile_grid"
            or m % input_tile[0]
            or k % input_tile[1]
            or input_tile[1] % geometry[2]
        ):
            raise RuntimeError("public QTIP runner manifest packed-layout binding drift")
        output_rows = (m // input_tile[0]) * (k // input_tile[1])
        expected_shape = (output_rows, words_per_k * geometry[1])
        if binding.get("expected_shape") != expected_shape:
            raise RuntimeError("public QTIP runner manifest expected shape drift")
        tiled = (
            states.reshape(
                m // input_tile[0],
                input_tile[0],
                k // input_tile[1],
                input_tile[1] // geometry[2],
            )
            .transpose(1, 2)
            .reshape(-1, input_tile[0] * input_tile[1] // geometry[2])
        )
        packed = cb.pack_trellis(tiled).contiguous()
        if tuple(packed.shape) != expected_shape or packed.dtype != torch.uint16:
            raise RuntimeError(
                "public QTIP runner manifest packed shape/dtype mismatch: "
                f"{tuple(packed.shape)} {packed.dtype} != "
                f"{expected_shape} torch.uint16"
            )
        unpacked = cb.unpack_trellis(packed, input_tile[0] * input_tile[1])
        if (
            not isinstance(unpacked, torch.Tensor)
            or tuple(unpacked.shape) != tuple(tiled.shape)
            or not bool(unpacked.to(tiled.dtype).eq(tiled).all())
        ):
            raise RuntimeError("public QTIP runner manifest pack roundtrip mismatch")
        roundtrip = unpacked.to(tiled.dtype).eq(tiled)
        packed_sha = _tensor_sha256(packed)
        return packed, {
            "tile_states_shape": list(tiled.shape),
            "canonical_packed_shape": list(packed.shape),
            "canonical_packed_dtype": str(packed.dtype),
            "canonical_packed_sha256": packed_sha,
            "canonical_unpack_state_sha256": _tensor_sha256(unpacked),
            "input_state_sha256": _tensor_sha256(tiled),
            "canonical_pack_roundtrip_fraction": float(roundtrip.float().mean()),
            "canonical_pack_roundtrip_exact": bool(roundtrip.all()),
            "kernel_swizzle": "manifest-canonical-direct",
            "kernel_packed_shape": list(packed.shape),
            "kernel_packed_sha256": packed_sha,
            "kernel_packed_bytes": packed.numel() * packed.element_size(),
        }

    return manifest_pack


def _load_public_qtip_runner(path: Path, expected_sha256: str):
    """Load only the runner whose declared SHA owns the physical pack path."""
    path = path.resolve()
    source = path.read_bytes()
    actual_sha256 = hashlib.sha256(source).hexdigest()
    if not _is_sha256_digest(expected_sha256) or actual_sha256 != expected_sha256:
        raise ValueError(
            f"public QTIP runner SHA mismatch: {actual_sha256} != {expected_sha256}"
        )
    if actual_sha256 != _TRUSTED_PUBLIC_QTIP_RUNNER_SHA256:
        raise ValueError(
            "public QTIP runner differs from the trusted package anchor: "
            f"{actual_sha256} != {_TRUSTED_PUBLIC_QTIP_RUNNER_SHA256}"
        )
    spec = importlib.util.spec_from_file_location("banana_smasher_qtip_runner", path)
    if spec is None:
        raise ImportError(f"cannot load public QTIP runner {path}")
    runner = importlib.util.module_from_spec(spec)
    exec(compile(source, str(path), "exec"), runner.__dict__)
    if _sha256(path) != expected_sha256:
        raise ValueError("public QTIP runner changed while loading")
    if Path(str(runner.__file__)).resolve() != path:
        raise RuntimeError(f"public QTIP runner loaded from divergent path: {runner.__file__}")
    pack = getattr(runner, "pack_kernel_layout", None)
    build = getattr(runner, "build_qtip", None)
    if not isinstance(pack, types.FunctionType):
        raise RuntimeError("public QTIP runner lacks canonical pack_kernel_layout")
    if (
        not isinstance(build, types.FunctionType)
        or "pack_kernel_layout" not in build.__code__.co_names
        or build.__globals__.get("pack_kernel_layout") is not pack
    ):
        raise RuntimeError("public QTIP runner build_qtip does not own canonical pack path")
    validate_pack = getattr(runner, "validate_manifest_packed_layout", None)
    owns_manifest_shape = (
        isinstance(validate_pack, types.FunctionType)
        and "validate_manifest_packed_layout" in pack.__code__.co_names
        and pack.__globals__.get("validate_manifest_packed_layout") is validate_pack
    )
    if not owns_manifest_shape:
        pack = _manifest_bound_public_qtip_pack(pack)
        setattr(runner, "pack_kernel_layout", pack)
        build.__globals__["pack_kernel_layout"] = pack
    return runner


def _declared_public_qtip_runner(config: dict[str, Any]) -> tuple[Path, str]:
    """Resolve the runner path/SHA directly from the config's bound run manifest."""
    materialization = config.get("materialization")
    if not isinstance(materialization, dict):
        raise ValueError("QTIP config lacks hash-bound materialization")
    manifest_path = _resolve_config_path(config, materialization.get("run_manifest"))
    raw = manifest_path.read_bytes()
    actual_manifest_sha = hashlib.sha256(raw).hexdigest()
    if actual_manifest_sha != materialization.get("run_manifest_sha256"):
        raise ValueError("QTIP config run-manifest SHA mismatch")
    manifest = json.loads(raw)
    rows = manifest.get("tiers") if isinstance(manifest, dict) else None
    matches = (
        [
            row
            for row in rows
            if isinstance(row, dict) and row.get("name") == config.get("tier")
        ]
        if isinstance(rows, list)
        else []
    )
    if len(matches) != 1:
        raise ValueError("QTIP config tier is not unique in its bound run manifest")
    bindings = matches[0].get("bindings")
    runner = bindings.get("qtip_runner") if isinstance(bindings, dict) else None
    if not isinstance(runner, dict):
        raise ValueError("QTIP run manifest lacks public runner binding")
    path = Path(str(runner.get("path", "")))
    if not path.is_absolute():
        path = manifest_path.parent / path
    path = path.resolve()
    configured_path = _config_path(config, "qtip_runner")
    expected_sha = runner.get("sha256")
    if path != configured_path or not _is_sha256_digest(expected_sha):
        raise ValueError("QTIP public runner binding differs from materialized config")
    assert isinstance(expected_sha, str)
    return path, expected_sha


def _load_captures(
    root: Path,
    layer: int,
    windows: int,
) -> list[dict[str, Any]]:
    root = root.resolve()
    cache_key = (root, layer, windows)
    cached = _CAPTURE_CACHE.get(cache_key)
    if cached is not None:
        return cached
    rows = []
    for window in range(windows):
        path = root / f"xmoe_L{layer:03d}_win{window:04d}.pt"
        done_path = path.with_suffix(path.suffix + ".DONE.json")
        if not path.is_file() or not done_path.is_file():
            raise FileNotFoundError(f"missing fit capture or receipt: {path}")
        done = json.loads(done_path.read_text())
        expected_md5 = done.get("md5") if isinstance(done, dict) else None
        actual_md5 = _md5(path)
        if (
            not isinstance(expected_md5, str)
            or len(expected_md5) != 32
            or any(character not in "0123456789abcdef" for character in expected_md5)
            or expected_md5 != actual_md5
        ):
            raise RuntimeError(
                f"capture MD5 mismatch: {path}: {actual_md5} != {expected_md5!r}"
            )
        data = torch.load(path, map_location="cpu", mmap=True, weights_only=True)
        if int(data["layer"]) != layer or int(data["win"]) != window:
            raise RuntimeError(f"capture identity mismatch: {path}")
        rows.append({
            "window": window,
            "x": data["x"].to(torch.bfloat16).contiguous(),
            "topk": data["topk"].to(torch.int64).contiguous(),
            "route": data["w"].float().contiguous(),
            "receipt_md5": done.get("md5"),
        })
    _CAPTURE_CACHE[cache_key] = rows
    return rows


def _release_capture_bank(
    root: Path,
    layer: int,
    windows: int,
    captures: list[dict[str, Any]],
) -> None:
    """Drop the full capture bank after routed fit windows are materialized.

    Keeping both surfaces alive through source dequantization and LDLQ was the
    dominant cause of the seven-matrix peak. The fit-window tensors own the
    routed rows needed by the builder, so the source capture bank is no longer
    part of the exact solve after this boundary.
    """
    _CAPTURE_CACHE.pop((root.resolve(), layer, windows), None)
    captures.clear()


def _bind_hessian_layer_manifest(
    config: dict[str, Any],
    *,
    layer: int,
) -> tuple[Path, int, dict[str, Any]]:
    manifest_path = _config_path(config, "hessian_layer_manifest")
    windows = int(config["fit_windows"])
    expected_sha = str(config["hessian_layer_manifest_sha256"])
    configured_root = _config_path(config, "fit_capture_root")
    cache_key = (manifest_path, expected_sha, configured_root, layer, windows)
    cached = _HESSIAN_BINDING_CACHE.get(cache_key)
    if cached is not None:
        return cached
    raw = manifest_path.read_bytes()
    manifest = json.loads(raw)
    expected = {
        "schema": "banana-smasher-hessian-layer-manifest-v1",
        "status": "PASS",
        "layer": layer,
        "windows": windows,
    }
    if any(manifest.get(key) != value for key, value in expected.items()):
        raise ValueError(f"QTIP Hessian layer-manifest binding mismatch: {manifest_path}")
    actual_sha = hashlib.sha256(raw).hexdigest()
    if actual_sha != expected_sha:
        raise ValueError(
            f"QTIP Hessian layer-manifest hash drift: {actual_sha} != {expected_sha}"
        )
    manifest_capture_root = Path(str(manifest["capture_root"])).resolve()
    members = manifest.get("members")
    if not isinstance(members, list) or len(members) != windows:
        raise ValueError(f"QTIP Hessian member population mismatch: {manifest_path}")
    for window, member in enumerate(members):
        if member.get("window") != window:
            raise ValueError(f"QTIP Hessian member order mismatch at window {window}")
        capture_name = f"xmoe_L{layer:03d}_win{window:04d}.pt"
        done_name = capture_name + ".DONE.json"
        for key, name in (("capture", capture_name), ("capture_done", done_name)):
            artifact = member.get(key)
            if not isinstance(artifact, dict):
                raise ValueError(f"QTIP Hessian member lacks {key}: window {window}")
            source_path = Path(str(artifact.get("path", ""))).resolve()
            expected_source = (manifest_capture_root / name).resolve()
            if source_path != expected_source or not source_path.is_file():
                raise ValueError(f"QTIP Hessian {key} path mismatch: {source_path}")
            expected_size = artifact.get("bytes")
            expected_digest = artifact.get("sha256")
            if source_path.stat().st_size != expected_size:
                raise ValueError(f"QTIP Hessian {key} size drift: {source_path}")
            has_digest = _is_sha256_digest(expected_digest)
            if not has_digest:
                raise ValueError(
                    f"QTIP Hessian {key} lacks SHA-256: window {window}"
                )
            if _sha256(source_path) != expected_digest:
                raise ValueError(f"QTIP Hessian {key} hash drift: {source_path}")
            configured_path = (configured_root / name).resolve()
            if not configured_path.is_file():
                raise ValueError(f"missing prefetched QTIP Hessian {key}: {configured_path}")
            if configured_path.stat().st_size != expected_size:
                raise ValueError(f"prefetched QTIP Hessian {key} size drift: {configured_path}")
            if _sha256(configured_path) != expected_digest:
                raise ValueError(f"prefetched QTIP Hessian {key} hash drift: {configured_path}")
    binding = {
        "path": str(manifest_path),
        "bytes": len(raw),
        "sha256": actual_sha,
        "windows": windows,
        "capture_root": str(configured_root),
        "manifest_capture_root": str(manifest_capture_root),
        "relocated_capture_root": configured_root != manifest_capture_root,
    }
    value = (configured_root, windows, binding)
    _HESSIAN_BINDING_CACHE[cache_key] = value
    return value


def _load_tlut(path: Path) -> torch.Tensor:
    path = path.resolve()
    cached = _TLUT_CACHE.get(path)
    if cached is None:
        payload = torch.load(path, map_location="cpu", mmap=True, weights_only=True)
        cached = payload["tlut"].float().contiguous()
        _TLUT_CACHE[path] = cached
    return cached


_E2M1 = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
     -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
    dtype=torch.float32,
)


def _model_moe_shape(model_root: Path) -> tuple[int, int]:
    """Read hidden/intermediate dimensions from the selected model metadata."""
    config_path = (model_root / "config.json").resolve()
    cached = _MODEL_SHAPE_CACHE.get(config_path)
    if cached is not None:
        return cached
    try:
        config = json.loads(config_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"model shape metadata is unavailable: {config_path}") from exc
    hidden = config.get("hidden_size") if isinstance(config, dict) else None
    intermediate = (
        config.get("moe_intermediate_size") if isinstance(config, dict) else None
    )
    if (
        isinstance(hidden, bool)
        or not isinstance(hidden, int)
        or hidden < 1
        or isinstance(intermediate, bool)
        or not isinstance(intermediate, int)
        or intermediate < 1
    ):
        raise RuntimeError(
            f"model config lacks positive hidden_size/moe_intermediate_size: {config_path}"
        )
    value = (hidden, intermediate)
    _MODEL_SHAPE_CACHE[config_path] = value
    return value


def _load_weight(model_root: Path, layer: int, expert: int, projection: str) -> tuple[torch.Tensor, dict[str, Any]]:
    projection = validate_qtip_projection(projection)
    index_path = model_root / "model.safetensors.index.json"
    resolved_index = index_path.resolve()
    mapping = _MODEL_INDEX_CACHE.get(resolved_index)
    if mapping is None:
        mapping = json.loads(index_path.read_text())["weight_map"]
        _MODEL_INDEX_CACHE[resolved_index] = mapping
    names = ("w1", "w3") if projection == "fused13" else ("w2",)
    matrices = []
    source = []
    for name in names:
        weight_key = f"layers.{layer}.ffn.experts.{expert}.{name}.weight"
        scale_key = f"layers.{layer}.ffn.experts.{expert}.{name}.scale"
        shard = model_root / mapping[weight_key]
        if mapping[scale_key] != mapping[weight_key]:
            raise RuntimeError(f"weight/scale shard split: {weight_key}")
        with safe_open(str(shard), framework="pt", device="cpu") as handle:
            packed = handle.get_tensor(weight_key).view(torch.uint8)
            scales = handle.get_tensor(scale_key).view(torch.uint8)
        nibbles = torch.stack((packed & 15, packed >> 4), dim=-1).flatten(-2)
        matrices.append(
            (_E2M1[nibbles.long()] * torch.exp2(scales.float() - 127.0).repeat_interleave(32, dim=1)).contiguous()
        )
        source.append({"path": str(shard), "bytes": shard.stat().st_size, "weight_key": weight_key})
    value = torch.cat(matrices, dim=0) if len(matrices) == 2 else matrices[0]
    hidden_size, intermediate_size = _model_moe_shape(model_root)
    expected = (
        (2 * intermediate_size, hidden_size)
        if projection == "fused13"
        else (hidden_size, intermediate_size)
    )
    if tuple(value.shape) != expected:
        raise RuntimeError(f"source shape mismatch: {tuple(value.shape)} != {expected}")
    return value.contiguous(), {
        "index_path": str(index_path),
        "index_sha256": _sha256(index_path),
        "shards": source,
    }


def _bind_candidate_geometry(
    candidate: dict[str, Any],
    config: dict[str, Any],
) -> None:
    """Replace vendor-fixed payload metadata with the selected manifest geometry."""
    geometry = config["geometry"]
    codebook = config["codebook"]
    candidate["geometry"] = {
        "L": int(geometry["L"]),
        "K": int(geometry["K"]),
        "V": int(geometry["V"]),
        "tlut_bits": int(codebook["tlut_bits"]),
        "decode_mode": str(codebook["decode_mode"]),
        "td_x": int(codebook["td_x"]),
        "td_y": int(codebook["td_y"]),
    }


def _bind_public_runner_pack_contract(
    cb: Any,
    config: dict[str, Any],
    source_weight: torch.Tensor,
) -> Mapping[str, object] | None:
    """Bind the manifest's canonical packed layout to the physical runner."""
    geometry = config["geometry"]
    sealed = tuple(int(geometry[key]) for key in ("L", "K", "V"))
    codebook = config["codebook"]
    contract = codebook.get("pack_contract") if isinstance(codebook, dict) else None
    if not isinstance(contract, dict):
        return None
    expected_shape = canonical_qtip_packed_shape(
        codebook=codebook,
        geometry=sealed,  # type: ignore[arg-type]
        matrix_shape=tuple(source_weight.shape),  # type: ignore[arg-type]
    )
    input_tile = contract["input_tile"]
    binding: Mapping[str, object] = MappingProxyType({
        "schema": "banana-smasher-public-runner-pack-contract-v1",
        "geometry": sealed,
        "matrix_shape": tuple(source_weight.shape),
        "input_tile": tuple(int(value) for value in input_tile),
        "dtype": str(contract["dtype"]),
        "packed_words_per_tile_per_k": int(
            contract["packed_words_per_tile_per_k"]
        ),
        "output_rows": str(contract["output_rows"]),
        "expected_shape": expected_shape,
    })
    cb._banana_smasher_public_runner_pack_contract = binding
    return binding


def _validate_candidate_packed_shape(
    candidate: dict[str, Any],
    config: dict[str, Any],
    source_weight: torch.Tensor,
) -> tuple[int, int] | None:
    """Bind canonical packed bytes to the selected manifest geometry contract."""
    geometry = config["geometry"]
    sealed = tuple(int(geometry[key]) for key in ("L", "K", "V"))
    codebook = config["codebook"]
    contract = codebook.get("pack_contract") if isinstance(codebook, dict) else None
    if not isinstance(contract, dict):
        return None
    expected = canonical_qtip_packed_shape(
        codebook=codebook,
        geometry=sealed,  # type: ignore[arg-type]
        matrix_shape=tuple(source_weight.shape),  # type: ignore[arg-type]
    )
    trellis = candidate.get("trellis")
    observed = tuple(trellis.shape) if isinstance(trellis, torch.Tensor) else None
    observed_dtype = trellis.dtype if isinstance(trellis, torch.Tensor) else None
    if observed != expected or observed_dtype != torch.uint16:
        raise RuntimeError(
            "QTIP packed shape/dtype mismatch: "
            f"{observed} {observed_dtype} != {expected} torch.uint16"
        )
    return expected


def _prepare_fit_windows(
    runner: Any,
    captures: list[Any],
    *,
    model_root: Path,
    layer: int,
    expert: int,
    projection: str,
    device: torch.device,
) -> tuple[list[Any], dict[str, Any]]:
    routed = runner.expert_windows(captures, expert)
    if projection != "down":
        return routed, {"mode": "routed-source-activation"}
    source_fused13, source_ref = _load_weight(
        model_root,
        layer,
        expert,
        "fused13",
    )
    try:
        windows = runner.down_windows(routed, source_fused13, device)
    finally:
        del source_fused13
    return windows, {
        "mode": "source-fused13",
        "source_weight": source_ref,
    }


def _bind_builder_memory_contract(
    cb: Any, source_weight: torch.Tensor
) -> Mapping[str, int | str]:
    """Bind exact upstream Qidxs and final contiguous-output storage."""
    vector_width = int(cb.V)
    index_dtype = getattr(cb, "idx_dtype", None)
    if (
        source_weight.ndim != 2
        or vector_width < 1
        or source_weight.numel() % vector_width
        or not isinstance(index_dtype, torch.dtype)
    ):
        raise ValueError(
            "QTIP source matrix/codebook does not close the builder memory contract"
        )
    state_elements = source_weight.numel() // vector_width
    state_storage_bytes = state_elements * torch.empty(
        (), dtype=index_dtype
    ).element_size()
    retained_output_bytes = source_weight.numel() * torch.empty(
        (), dtype=torch.float32
    ).element_size()
    contract: Mapping[str, int | str] = MappingProxyType({
        "schema": "banana-smasher-qtip-builder-memory-v2",
        "state_elements": state_elements,
        "state_storage_bytes": state_storage_bytes,
        "retained_output_bytes": retained_output_bytes,
    })
    cb._banana_smasher_memory_contract = contract
    cb._banana_smasher_memory_contract_binding = tuple(contract.values())
    cb._banana_smasher_observed_state_elements = 0
    return contract


def _verify_builder_memory_contract(cb: Any) -> Mapping[str, int | str]:
    contract = getattr(cb, "_banana_smasher_memory_contract", None)
    binding = getattr(cb, "_banana_smasher_memory_contract_binding", None)
    observed = getattr(cb, "_banana_smasher_observed_state_elements", None)
    fields = (
        "schema",
        "state_elements",
        "state_storage_bytes",
        "retained_output_bytes",
    )
    if (
        not isinstance(contract, Mapping)
        or tuple(contract) != fields
        or not isinstance(binding, tuple)
        or len(binding) != len(fields)
        or tuple(contract.get(field) for field in fields) != binding
        or binding[0] != "banana-smasher-qtip-builder-memory-v2"
    ):
        raise RuntimeError("QTIP builder memory contract drift")
    expected = binding[1]
    if observed != expected:
        raise RuntimeError(
            f"QTIP builder state output closure mismatch: {observed} != {expected}"
        )
    return contract


class _ExactTimers:
    def __init__(self) -> None:
        self.codebook_distance_seconds = 0.0
        self.transition_seconds = 0.0
        self.calls = 0
        self.sequences = 0


def _install_profiled_exact_viterbi(
    cb: Any,
    exact: Any,
    timers: _ExactTimers,
    *,
    profile_mode: bool,
) -> dict[str, Any]:
    """Install exact Viterbi; instrumentation is profile-only, never solve overhead."""
    def solve(self: Any, x: torch.Tensor, overlap: torch.Tensor | None = None) -> torch.Tensor:
        if not x.is_cuda or x.ndim != 2 or x.shape[0] != 256:
            raise ValueError(f"exact prefix Viterbi expects CUDA [256,B], got {tuple(x.shape)}")
        batch = int(x.shape[1])
        if not 1 <= batch <= 8192:
            raise ValueError(f"batch outside 1..8192: {batch}")
        if profile_mode:
            with torch.profiler.record_function("qtip.viterbi_transition_scoring"):
                states = exact.exact_prefix_viterbi(self, x, overlap)
        else:
            states = exact.exact_prefix_viterbi(self, x, overlap)
        timers.calls += 1
        timers.sequences += batch
        return states

    def quantize_seq(self: Any, x: torch.Tensor, overlap: torch.Tensor | None = None, **_: Any):
        return solve(self, x, overlap)

    cb.viterbi = types.MethodType(solve, cb)
    cb.quantize_seq = types.MethodType(quantize_seq, cb)
    metadata = dict(exact.geometry(cb))
    return {
        **metadata,
        "branch_sampling": metadata.get("branch_sampling", "full"),
        "ordering": metadata.get(
            "ordering", "one persistent launch per independent sequence batch"
        ),
        "production_default": True,
    }


def _install_configured_viterbi(
    cb: Any,
    exact: Any,
    timers: _ExactTimers,
    config: dict[str, Any],
    *,
    profile_mode: bool,
) -> dict[str, Any]:
    geometry = config.get("geometry", {"L": 16, "K": 3, "V": 2})
    sealed = (int(geometry["L"]), int(geometry["K"]), int(geometry["V"]))
    if sealed not in known_qtip_geometries():
        bpw = config.get("bpw", "<bpw>")
        raise ValueError(
            f"QTIP geometry {sealed} not in compiled set; "
            f"run `smash kernels build --tier qtip --bpw {bpw}`"
        )
    actual = (int(cb.L), int(cb.K), int(cb.V))
    if actual != sealed:
        raise ValueError(f"QTIP codebook geometry mismatch: {actual} != {sealed}")
    expected_backend = backend_for_geometry(sealed)
    backend = config.get("backend", expected_backend)
    if backend != expected_backend:
        raise ValueError(
            f"QTIP backend {backend!r} differs from qtip_rings.json recipe "
            f"{expected_backend!r} for geometry {sealed}"
        )
    if backend in PERSISTENT_BACKENDS:
        return _install_profiled_exact_viterbi(
            cb, exact, timers, profile_mode=profile_mode
        )
    if backend != TRELLIS_V2_BACKEND:
        raise ValueError(
            f"QTIP backend {backend!r} has no compiled installer; run "
            f"`smash kernels build --tier qtip --bpw {config.get('bpw', '<bpw>')}`"
        )
    from .trellis_v2 import install_trellis_v2

    metadata = install_trellis_v2(cb)
    cb._trellis_v2_collect_stats = False
    base = cb.quantize_seq

    def solve(self: Any, x: torch.Tensor, overlap: torch.Tensor | None = None) -> torch.Tensor:
        if profile_mode:
            with torch.profiler.record_function("qtip.viterbi_transition_scoring"):
                states = base(x, overlap)
        else:
            states = base(x, overlap)
        timers.calls += 1
        timers.sequences += int(x.shape[1])
        return states

    cb.viterbi = types.MethodType(solve, cb)
    cb.quantize_seq = types.MethodType(solve, cb)
    return {
        **metadata,
        "stats_collection_during_timing": False,
    }


def _top_ops(profile: Any) -> list[dict[str, Any]]:
    rows = []
    for event in profile.key_averages():
        cpu_us = float(getattr(event, "self_cpu_time_total", 0.0))
        device_us = float(
            getattr(event, "self_device_time_total", getattr(event, "self_cuda_time_total", 0.0))
        )
        rows.append({
            "op": str(event.key),
            "calls": int(event.count),
            "self_cpu_seconds": cpu_us / 1e6,
            "self_device_seconds": device_us / 1e6,
            "rank_seconds": (cpu_us + device_us) / 1e6,
        })
    return sorted(rows, key=lambda row: row["rank_seconds"], reverse=True)[:50]


def _split_solve_and_conformance_seconds(
    build_seconds: float,
    build_receipt: Mapping[str, Any],
) -> tuple[float, float]:
    """Keep mandatory packed-wire audit time in same-work solve latency."""
    phase_seconds = build_receipt.get("phase_seconds")
    conformance = (
        phase_seconds.get("packed_decode_conformance")
        if isinstance(phase_seconds, Mapping)
        else None
    )
    if (
        isinstance(conformance, bool)
        or not isinstance(conformance, (int, float))
        or not math.isfinite(conformance)
        or conformance < 0
        or conformance > build_seconds
    ):
        raise RuntimeError(
            "invalid packed decode conformance timing: "
            f"conformance={conformance!r} build={build_seconds!r}"
        )
    return build_seconds, float(conformance)


def main(
    config_path: Path,
    root: Path,
    layer: int,
    *,
    profile_mode: bool = True,
    kernel_cache_root: Path | None = None,
) -> dict[str, Any]:
    config = _read_qtip_config(config_path)
    if int(config["layer"]) != layer:
        raise ValueError("QTIP config layer differs from --layer")
    expert = int(config["expert"])
    projection = validate_qtip_projection(config["projection"])
    basis_gate = _verify_basis(config, root)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")

    mode = "profile" if profile_mode else "solve"
    out = root / mode / f"L{layer:03d}" / f"E{expert:03d}_{projection}"
    out.mkdir(parents=True, exist_ok=True)
    outer_started = time.perf_counter()
    epoch_started = time.time()

    qv_path, qv_sha256 = _declared_public_qtip_runner(config)
    qv = _load_public_qtip_runner(qv_path, qv_sha256)
    qv.QTIP = _config_path(config, "qtip_root")
    bitshift, ldlq, math_utils, kernel_decode = qv.load_official_qtip()
    from . import qtip_viterbi as exact

    reference_path = _config_path(config, "reference_unit")
    reference = torch.load(reference_path, map_location="cpu", mmap=True, weights_only=True)
    seed, seed_policy = _resolve_rht_seed(
        config,
        reference,
        layer=layer,
        expert=expert,
        projection=projection,
    )
    pinned_tlut = _load_tlut(_config_path(config, "tlut_source"))
    if _tensor_sha256(pinned_tlut) != str(reference["tlut_sha256"]):
        raise RuntimeError("TLUT digest differs from sealed reference unit")
    geometry = config.get("geometry", {"L": 16, "K": 3, "V": 2})
    codebook = config.get("codebook")
    if not isinstance(codebook, dict):
        codebook = {
            "tlut_bits": 9,
            "decode_mode": "quantlut_sym",
        }
    cb = bitshift.bitshift_codebook(
        L=int(geometry["L"]),
        K=int(geometry["K"]),
        V=int(geometry["V"]),
        tlut_bits=int(codebook["tlut_bits"]),
        decode_mode=str(codebook["decode_mode"]),
        tlut=pinned_tlut.to("cuda"),
    ).to("cuda")
    kernel_prepare_started = time.perf_counter()
    from .qtip_kernel_cache import build_qtip_kernels

    kernel_bpw = config.get("bpw")
    if not isinstance(kernel_bpw, str):
        kernel_bpw = f"{int(geometry['K']):.2f}"
    kernel_cache = build_qtip_kernels(kernel_bpw, cache_root=kernel_cache_root)
    timers = _ExactTimers()
    solver = _install_configured_viterbi(
        cb,
        exact,
        timers,
        config,
        profile_mode=profile_mode,
    )
    kernel_prepare_seconds = time.perf_counter() - kernel_prepare_started

    model_root = _config_path(config, "model_root")
    capture_root, fit_window_count, hessian_binding = _bind_hessian_layer_manifest(
        config,
        layer=layer,
    )
    captures = _load_captures(capture_root, layer, fit_window_count)
    fit_windows, fit_source = _prepare_fit_windows(
        qv,
        captures,
        model_root=model_root,
        layer=layer,
        expert=expert,
        projection=projection,
        device=torch.device("cuda"),
    )
    _release_capture_bank(capture_root, layer, fit_window_count, captures)
    del captures
    source_weight, source_ref = _load_weight(model_root, layer, expert, projection)
    _bind_public_runner_pack_contract(cb, config, source_weight)
    if solver.get("implementation") in PERSISTENT_BACKENDS:
        _bind_builder_memory_contract(cb, source_weight)
    staging_seconds = (
        time.perf_counter() - outer_started - kernel_prepare_seconds
    )

    dequant_seconds = 0.0
    original_decode = qv.decode_packed
    def timed_decode(*args: Any, **kwargs: Any):
        nonlocal dequant_seconds
        torch.cuda.synchronize()
        started = time.perf_counter()
        with torch.profiler.record_function("qtip.dequant_and_reread"):
            value = original_decode(*args, **kwargs)
        torch.cuda.synchronize()
        dequant_seconds += time.perf_counter() - started
        return value
    qv.decode_packed = timed_decode

    first_gpu_phase = {
        "schema": "banana-smasher-qtip-first-gpu-phase-v1",
        "phase": "fresh_exact_qtip_build",
        "pid": os.getpid(),
        "startticks": int(Path("/proc/self/stat").read_text().split()[21]),
        "layer": layer,
        "expert": expert,
        "projection": projection,
        "staging_seconds": staging_seconds,
        "epoch": time.time(),
    }
    progress_receipt_started = time.perf_counter()
    _atomic_json(out / "FIRST_GPU_PHASE.json", first_gpu_phase)
    progress_receipt_fsync_seconds = (
        time.perf_counter() - progress_receipt_started
    )
    print(
        "FIRST_GPU_PHASE " + json.dumps(first_gpu_phase, sort_keys=True),
        file=sys.stderr,
        flush=True,
    )

    if not profile_mode:
        qv.decode_packed = original_decode
        torch.cuda.reset_peak_memory_stats()
        build_started = time.perf_counter()
        candidate, build = build_qtip_bounded(
            qv,
            source_weight, fit_windows, cb, ldlq, math_utils, kernel_decode,
            torch.device("cuda"), seed,
        )
        torch.cuda.synchronize()
        build_seconds = time.perf_counter() - build_started
        solve_seconds, packed_decode_conformance_seconds = (
            _split_solve_and_conformance_seconds(build_seconds, build)
        )
        if hasattr(cb, "_banana_smasher_memory_contract"):
            _verify_builder_memory_contract(cb)
        _validate_candidate_packed_shape(candidate, config, source_weight)
        _bind_candidate_geometry(candidate, config)
        reconstructed = candidate.pop("reconstructed_weight", None)
        if reconstructed is None:
            raise RuntimeError(
                "QTIP builder omitted reconstructed_weight before public wire seal"
            )
        artifact_path = out / "QTIP_UNIT.pt"
        candidate["schema"] = _QTIP_UNIT_PAYLOAD_SCHEMA
        artifact_fsync_started = time.perf_counter()
        _atomic_torch(artifact_path, candidate)
        artifact_fsync_seconds = time.perf_counter() - artifact_fsync_started
        total_seconds = time.perf_counter() - outer_started
        phase_seconds = {
            "staging": staging_seconds,
            "kernel_prepare": kernel_prepare_seconds,
            "progress_receipt_fsync": progress_receipt_fsync_seconds,
            "solve": solve_seconds,
            "solve_core_excluding_packed_decode_conformance": (
                solve_seconds - packed_decode_conformance_seconds
            ),
            "packed_decode_conformance": packed_decode_conformance_seconds,
            "artifact_fsync": artifact_fsync_seconds,
            "receipt_fsync": 0.0,
        }
        phase_seconds["remainder"] = max(
            0.0, total_seconds - sum(phase_seconds.values())
        )
        receipt = {
            "schema": "banana-smasher-qtip-solve-v1",
            "status": "PASS",
            "host": os.uname().nodename,
            "layer": layer,
            "expert": expert,
            "projection": projection,
            "fresh_no_warm_start": True,
            "public_command_config": str(config_path.resolve()),
            "config_sha256": _sha256(config_path),
            "basis_gate": basis_gate,
            "epoch_started": epoch_started,
            "epoch_ended": time.time(),
            "total_wall_seconds": total_seconds,
            "staging_seconds": staging_seconds,
            "solve_seconds": solve_seconds,
            "phase_seconds": phase_seconds,
            "assignment_sha256": _tensor_sha256(candidate["trellis"]),
            "artifact": str(artifact_path),
            "artifact_sha256": _sha256(artifact_path),
            "viterbi_launches": timers.calls,
            "viterbi_sequences": timers.sequences,
            "transition_decisions": (
                timers.sequences
                * (int(solver["steps"]) - 1)
                * int(solver["branches_per_prefix"])
            ),
            "solver": solver,
            "kernel_cache": kernel_cache,
            "build": build,
            "source_weight": source_ref,
            "fit_source": fit_source,
            "fit_windows": fit_window_count,
            "hessian_layer_manifest": hessian_binding,
            "rht_seed": seed,
            "rht_seed_policy": seed_policy,
            "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(),
        }
        receipt = _public_receipt(receipt)
        receipt_path = out / "QTIP_SOLVE_RECEIPT.json"
        receipt_fsync_started = time.perf_counter()
        _atomic_json(receipt_path, receipt)
        phase_seconds["receipt_fsync"] = time.perf_counter() - receipt_fsync_started
        receipt["total_wall_seconds"] = time.perf_counter() - outer_started
        phase_seconds["remainder"] = max(
            0.0,
            receipt["total_wall_seconds"]
            - sum(
                seconds
                for name, seconds in phase_seconds.items()
                if name != "remainder"
            ),
        )
        receipt["epoch_ended"] = time.time()
        _atomic_json(receipt_path, receipt)
        return receipt

    activities = [torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
    with torch.profiler.profile(activities=activities, record_shapes=False) as profile:
        with torch.profiler.record_function("qtip.fresh_exact_build"):
            candidate, build = build_qtip_bounded(
                qv,
                source_weight, fit_windows, cb, ldlq, math_utils, kernel_decode,
                torch.device("cuda"), seed,
            )
    torch.cuda.synchronize()
    if hasattr(cb, "_banana_smasher_memory_contract"):
        _verify_builder_memory_contract(cb)
    _validate_candidate_packed_shape(candidate, config, source_weight)
    transition_keys = {
        "_persistent_prefix_viterbi",
        "_persistent_k2_viterbi",
        "aten::argmin",
    }
    timers.transition_seconds = sum(
        float(getattr(event, "self_device_time_total", getattr(event, "self_cuda_time_total", 0.0)))
        for event in profile.key_averages()
        if (
            str(event.key) in transition_keys
            or "pair_kernel" in str(event.key)
            or "backtrack_kernel" in str(event.key)
        )
    ) / 1e6
    outer_seconds = time.perf_counter() - outer_started

    assignment_sha = _tensor_sha256(candidate["trellis"])
    bucket_seconds = {
        "trellis_viterbi_transition_scoring": timers.transition_seconds,
        "codebook_distances": timers.codebook_distance_seconds,
        "staging": staging_seconds,
        "dequant": dequant_seconds,
    }
    bucket_seconds["remainder"] = max(0.0, outer_seconds - sum(bucket_seconds.values()))
    census = {key: int(value) for key, value in config["layer_census"].items()}
    pack_counts = {key: int(value) for key, value in config["pack_counts"].items()}
    qtip_population = sum(
        value for key, value in pack_counts.items() if key.startswith("qtip")
    )
    qtip_fraction = qtip_population / sum(pack_counts.values())
    receipt = {
        "schema": "banana-smasher-qtip-profile-v1",
        "status": "PASS",
        "host": os.uname().nodename,
        "layer": layer,
        "expert": expert,
        "projection": projection,
        "fresh_no_warm_start": True,
        "public_command_config": str(config_path.resolve()),
        "config_sha256": _sha256(config_path),
        "basis_gate": basis_gate,
        "epoch_started": epoch_started,
        "epoch_ended": time.time(),
        "outer_wall_seconds": outer_seconds,
        "bucket_seconds": bucket_seconds,
        "bucket_percent": {key: 100.0 * value / outer_seconds for key, value in bucket_seconds.items()},
        "bucket_definition": {
            "trellis_viterbi_transition_scoring": "all 127 exact prefix-DP advance steps plus final argmin/backtrack; each advance kernel fuses predecessor transition minimization with that step's codebook-distance term",
            "codebook_distances": "exclusive initial-state exact distance kernel; later distance terms are fused into the transition bucket",
            "staging": "capture receipt identity reads, local model/TLUT/reference reads, QTIP import, and source dequant; full payload hashing remains in tests/CI",
            "dequant": "canonical packed-wire decode plus inverse FWHT reread conformance",
            "remainder": "Hessian/FWHT/LDLQ matrix work, packing, Python dispatch, compile overhead, profiler overhead, serialization, and closed residual",
        },
        "viterbi_calls": timers.calls,
        "solver": solver,
        "kernel_cache": kernel_cache,
        "build": build,
        "top_10_ops": _top_ops(profile),
        "assignment_sha256": assignment_sha,
        "acceptance_provenance": {
            "source_commit": "48dd3443d86eae585c2c1b41e49f47912c50170f",
            "receipt": "P2C_QTIP3_PUBLIC_FIRST64_PASS",
            "ordered_assignment_sha256": (
                "96e0fd6c689cb1af387dce9843dc96ca52a086f85cc7e0caf7101d6ad92dfb26"
            ),
            "mean_public_outer_seconds": 1.9163911582144217,
        },
        "reference_unit": {"path": str(reference_path), "file_sha256": _sha256(reference_path)},
        "source_weight": source_ref,
        "fit_source": fit_source,
        "rht_seed": seed,
        "rht_seed_policy": seed_policy,
        "layer_census": census,
        "layer_qtip3_fraction": census["qtip3"] / sum(census.values()),
        "pack_projection": {
            "counts": pack_counts,
            "qtip_count_fraction": qtip_fraction,
            "qtip_dominated": qtip_fraction > 0.5,
            "profiled_geometry": {key: int(value) for key, value in geometry.items()},
            "profiled_unit_wall_seconds": outer_seconds,
            "projected_matching_geometry_pack_wall_seconds": (
                pack_counts["qtip2"]
                if int(geometry["K"]) == 2
                else pack_counts["qtip3"]
            ) * outer_seconds,
            "method": "physical fresh unit wall multiplied only by the sealed count for the matching QTIP geometry",
        },
        "next_kernel_recommendation": "The measured exact DP floor is now the persistent kernel itself; the next legal throughput lever is one resident layer solve that batches independent projection units and amortizes capture/model/TLUT staging without sharing assignment or objective state.",
    }
    receipt = _public_receipt(receipt)
    receipt_path = out / "QTIP_PROFILE_RECEIPT.json"
    _atomic_json(receipt_path, receipt)
    return receipt


def _validated_root_ring_manifest(
    config_root: Path,
    ring: Any,
) -> tuple[Path, str]:
    """Bind ring configs to the canonical manifest at the selected config root."""
    manifest_path = config_root.resolve() / "QTIP_RUN_MANIFEST.json"
    if not manifest_path.is_file():
        raise ValueError(
            f"manifest tier {ring.tier} canonical run manifest missing: {manifest_path}"
        )
    raw = manifest_path.read_bytes()
    try:
        manifest = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"manifest tier {ring.tier} canonical run manifest is invalid: {manifest_path}"
        ) from exc
    if (
        not isinstance(manifest, dict)
        or manifest.get("schema") != "banana-smasher-qtip-run-manifest-v1"
        or manifest.get("status") != "PASS"
    ):
        raise ValueError(
            f"manifest tier {ring.tier} canonical run manifest is invalid: {manifest_path}"
        )
    rows = manifest.get("tiers")
    matches = (
        [row for row in rows if isinstance(row, dict) and row.get("name") == ring.tier]
        if isinstance(rows, list)
        else []
    )
    if len(matches) != 1:
        raise ValueError(
            f"manifest tier {ring.tier} requires exactly one canonical ring manifest row"
        )
    validate_qtip_ring_manifest(matches[0].get("ring"), ring)
    return manifest_path, hashlib.sha256(raw).hexdigest()


def _validated_ring_members(
    config_root: Path,
    ring: Any,
    *,
    layer: int,
    run_manifest_sha: str,
) -> dict[tuple[int, str], dict[str, Any]]:
    """Return the hash-bound materialized member population for one layer."""
    manifest_path = config_root.resolve() / "QTIP_CONFIG_MANIFEST.json"
    if not manifest_path.is_file():
        raise ValueError(
            f"manifest tier {ring.tier} member manifest missing: {manifest_path}"
        )
    try:
        manifest = json.loads(manifest_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"manifest tier {ring.tier} member manifest is invalid: {manifest_path}"
        ) from exc
    ring_identity = manifest.get("ring") if isinstance(manifest, dict) else None
    if (
        not isinstance(manifest, dict)
        or manifest.get("schema") != "banana-smasher-qtip-config-manifest-v1"
        or manifest.get("status") != "PASS"
        or manifest.get("tier") != ring.tier
        or manifest.get("run_manifest_sha256") != run_manifest_sha
        or not isinstance(ring_identity, dict)
        or ring_identity.get("bpw") != ring.canonical_bpw
    ):
        raise ValueError(
            f"manifest tier {ring.tier} member manifest identity mismatch: {manifest_path}"
        )
    records = manifest.get("member_records")
    if not isinstance(records, list):
        raise ValueError(
            f"manifest tier {ring.tier} member manifest lacks records: {manifest_path}"
        )
    members: dict[tuple[int, str], dict[str, Any]] = {}
    for record in records:
        if not isinstance(record, dict) or record.get("layer") != layer:
            continue
        expert = record.get("expert")
        projection = record.get("projection")
        if (
            isinstance(expert, bool)
            or not isinstance(expert, int)
            or expert < 0
            or not isinstance(projection, str)
            or not projection
        ):
            raise ValueError(
                f"manifest tier {ring.tier} member identity is invalid: {record!r}"
            )
        identity = (expert, projection)
        if identity in members:
            raise ValueError(
                f"manifest tier {ring.tier} duplicate member identity: {identity!r}"
            )
        members[identity] = record
    if not members:
        raise ValueError(
            f"manifest tier {ring.tier} member manifest lacks L{layer:03d} records"
        )
    return members


def _ordered_qtip_configs(
    config_root: Path,
    layer: int,
    *,
    tier: str | None = None,
    all_cells: bool = False,
) -> list[Path]:
    """Order one manifest-declared tier without a package-global tier menu."""
    projection_order = {"fused13": 0, "down": 1}
    rows: list[tuple[int, int, Path]] = []
    identities: set[tuple[int, str]] = set()
    selected_geometry: tuple[int, int, int] | None = None
    candidates: list[tuple[Path, dict[str, Any]]] = []
    for path in config_root.rglob("E*_*.json"):
        config = _read_qtip_config(path)
        if int(config["layer"]) == layer:
            candidates.append((path, config))
    if tier is None:
        manifest_tiers = {
            configured_tier
            for _path, config in candidates
            if isinstance((configured_tier := config.get("tier")), str)
            and configured_tier.startswith("qtip@")
        }
        if len(manifest_tiers) > 1:
            raise ValueError(
                f"multiple manifest QTIP tiers for L{layer:03d}: "
                f"{sorted(manifest_tiers)}"
            )
        if manifest_tiers:
            tier = manifest_tiers.pop()
    ring = (
        resolve_qtip_ring(tier.removeprefix("qtip@"))
        if tier is not None and tier.startswith("qtip@")
        else None
    )
    ring_assignments: dict[tuple[int, str], tuple[int, int, int]] = {}
    root_manifest_path: Path | None = None
    root_manifest_sha: str | None = None
    ring_members: dict[tuple[int, str], dict[str, Any]] | None = None
    if ring is not None:
        root_manifest_path, root_manifest_sha = _validated_root_ring_manifest(
            config_root, ring
        )
        ring_members = _validated_ring_members(
            config_root,
            ring,
            layer=layer,
            run_manifest_sha=root_manifest_sha,
        )
    for path, config in candidates:
        configured_tier = config.get("tier")
        if configured_tier is not None:
            if not isinstance(configured_tier, str) or not configured_tier:
                raise ValueError(f"invalid QTIP tier in {path}: {configured_tier!r}")
        if tier is not None and configured_tier is not None and configured_tier != tier:
            continue
        geometry = config.get("geometry")
        if not isinstance(geometry, dict) or set(geometry) != {"L", "K", "V"}:
            raise ValueError(f"QTIP config lacks exact L/K/V geometry: {path}")
        sealed_values = tuple(geometry[key] for key in ("L", "K", "V"))
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 1
            for value in sealed_values
        ):
            raise ValueError(f"invalid QTIP geometry in {path}: {sealed_values}")
        sealed = (
            int(sealed_values[0]),
            int(sealed_values[1]),
            int(sealed_values[2]),
        )
        if ring is not None:
            if configured_tier != ring.tier:
                raise ValueError(
                    f"manifest tier {ring.tier} tier mismatch in {path}: "
                    f"{configured_tier!r} != {ring.tier!r}"
                )
            if sealed not in ring.geometries:
                raise ValueError(
                    f"manifest tier {ring.tier} geometry {sealed} is outside its ring"
                )
            configured_bpw = config.get("bpw")
            if configured_bpw != ring.canonical_bpw:
                raise ValueError(
                    f"manifest tier {ring.tier} bpw mismatch in {path}: "
                    f"{configured_bpw!r} != {ring.canonical_bpw!r}"
                )
            expected_backend = ring.backend_for(sealed)
            if config.get("backend") != expected_backend:
                raise ValueError(
                    f"manifest tier {ring.tier} backend mismatch in {path}: "
                    f"{config.get('backend')!r} != {expected_backend!r}"
                )
            if config.get("codebook") != dict(ring.codebook):
                raise ValueError(
                    f"manifest tier {ring.tier} codebook mismatch in {path}"
                )
            if config.get("aot") != dict(ring.aot):
                raise ValueError(
                    f"manifest tier {ring.tier} AOT identity mismatch in {path}"
                )
            materialization = config.get("materialization")
            if (
                not isinstance(materialization, dict)
                or materialization.get("schema")
                != "banana-smasher-qtip-config-materialization-v1"
                or materialization.get("qtip_ring_bpw") != ring.canonical_bpw
                or not isinstance(materialization.get("run_manifest"), str)
                or not isinstance(materialization.get("run_manifest_sha256"), str)
            ):
                raise ValueError(
                    f"manifest tier {ring.tier} materialization identity mismatch in {path}"
                )
            manifest_path = _resolve_config_path(
                config, materialization["run_manifest"]
            )
            declared_manifest_sha = materialization["run_manifest_sha256"]
            if manifest_path != root_manifest_path:
                raise ValueError(
                    f"manifest tier {ring.tier} config is not bound to canonical run manifest "
                    f"{root_manifest_path}: {manifest_path}"
                )
            if declared_manifest_sha != root_manifest_sha:
                raise ValueError(
                    f"manifest tier {ring.tier} materialization SHA mismatch in {path}: "
                    f"{declared_manifest_sha!r} != {root_manifest_sha!r}"
                )
        elif selected_geometry is None:
            selected_geometry = sealed
        elif sealed != selected_geometry:
            raise ValueError(
                f"manifest tier {tier or configured_tier or 'QTIP'} has mixed geometries: "
                f"{selected_geometry} != {sealed} in {path}"
            )
        expert = int(config["expert"])
        projection = str(config["projection"])
        if expert < 0:
            raise ValueError(f"QTIP expert must be non-negative in {path}: {expert}")
        if projection not in projection_order:
            raise ValueError(f"unsupported QTIP projection in {path}: {projection}")
        identity = (expert, projection)
        if identity in identities:
            raise ValueError(
                f"duplicate resident QTIP config for E{expert:03d}_{projection}"
            )
        identities.add(identity)
        ring_assignments[identity] = sealed
        rows.append((expert, projection_order[projection], path))
    if not rows:
        label = tier or "QTIP"
        raise ValueError(
            f"no L{layer:03d} {label} configs under {config_root}; "
            "run public producer `smash qtip-configs`"
        )
    ordered = [path for _expert, _projection, path in sorted(rows)]
    if ring is not None:
        assert ring_members is not None
        if identities != set(ring_members):
            missing = sorted(set(ring_members) - identities)
            extra = sorted(identities - set(ring_members))
            raise ValueError(
                f"manifest tier {ring.tier} member population mismatch for L{layer:03d}: "
                f"missing={missing[:5]} extra={extra[:5]}"
            )
        expected = assign_ring_geometries(
            ring,
            ((layer, expert, projection) for expert, projection in identities),
        )
        for identity, actual in ring_assignments.items():
            manifest_identity = (layer, identity[0], identity[1])
            if actual != expected[manifest_identity]:
                raise ValueError(
                    f"manifest tier {ring.tier} ring assignment mismatch for "
                    f"E{identity[0]:03d}_{identity[1]}: "
                    f"{actual} != {expected[manifest_identity]}"
                )
        projection_by_order = {order: name for name, order in projection_order.items()}
        selected_paths = {
            (expert, projection_by_order[projection_index]): path.resolve()
            for expert, projection_index, path in rows
        }
        for identity, record in ring_members.items():
            path = selected_paths[identity]
            record_path = Path(str(record.get("path", "")))
            if not record_path.is_absolute():
                record_path = config_root / record_path
            record_path = record_path.resolve()
            if record_path != path or record.get("sha256") != _sha256(path):
                raise ValueError(
                    f"manifest tier {ring.tier} member hash/path mismatch for {identity!r}"
                )
    if all_cells:
        if ring_members is not None:
            expected_population = len(ring_members)
        else:
            populations = set()
            for path in ordered:
                config = json.loads(path.read_text())
                census = config.get("layer_census")
                configured_tier = config.get("tier")
                population = (
                    census.get(configured_tier)
                    if isinstance(census, dict)
                    and isinstance(configured_tier, str)
                    else None
                )
                if (
                    isinstance(population, bool)
                    or not isinstance(population, int)
                    or population < 1
                ):
                    raise ValueError(
                        f"public {tier} --all-cells requires manifest population in {path}"
                    )
                populations.add(population)
            if len(populations) != 1:
                raise ValueError(
                    f"public {tier} --all-cells has inconsistent manifest populations: "
                    f"{sorted(populations)}"
                )
            expected_population = populations.pop()
        if len(ordered) != expected_population:
            raise ValueError(
                f"public {tier} --all-cells requires exactly {expected_population} "
                f"manifest configs for L{layer:03d}, got {len(ordered)}"
            )
    return ordered


def main_many(
    config_root: Path,
    root: Path,
    layer: int,
    *,
    limit: int | None = None,
    tier: str | None = None,
    all_cells: bool = False,
    profile_mode: bool = False,
    resume: bool = True,
    resume_flag_explicit: bool = False,
    kernel_cache_root: Path | None = None,
) -> dict[str, Any]:
    """Solve an ordered config directory in one resident public process."""
    batch_started = time.perf_counter()
    epoch_started = time.time()
    if limit is not None and limit < 1:
        raise ValueError("--qtip-units must be positive")
    if all_cells and limit is not None:
        raise ValueError("--all-cells refuses a QTIP unit limit")
    if not resume:
        raise ValueError("resident QTIP solve requires hash-validating resume")
    paths = _ordered_qtip_configs(
        config_root,
        layer,
        tier=tier,
        all_cells=all_cells,
    )
    if limit is not None:
        paths = paths[:limit]
    ordered_assignments = []
    unit_receipts = []
    resumed_units = 0
    computed_units = 0
    # Idempotent resume preflight: every pre-existing unit is hash-validated
    # BEFORE any new compute so a divergent/corrupt/partial unit fails loudly
    # instead of being rerun or overwritten.  Valid PASS units are skipped
    # byte-for-byte (no content, metadata, or mtime rewrite); execution then
    # continues at the first missing unit.
    resume_preflight_started = time.perf_counter()
    existing_units = [
        _validated_existing_unit(
            path,
            root,
            layer,
            profile_mode=profile_mode,
        )
        for path in paths
    ]
    resume_preflight_seconds = time.perf_counter() - resume_preflight_started
    unit_dispatch_started = time.perf_counter()
    for path, existing in zip(paths, existing_units, strict=True):
        if existing is None:
            main_kwargs: dict[str, Any] = {"profile_mode": profile_mode}
            if kernel_cache_root is not None:
                main_kwargs["kernel_cache_root"] = kernel_cache_root
            receipt = main(path, root, layer, **main_kwargs)
            computed_units += 1
        else:
            receipt = existing
            resumed_units += 1
        if not str(receipt.get("status", "")).startswith("PASS"):
            raise RuntimeError(f"resident QTIP unit failed: {path}")
        ordered_assignments.append(
            {
                "layer": int(receipt["layer"]),
                "expert": int(receipt["expert"]),
                "projection": str(receipt["projection"]),
                "assignment_sha256": str(receipt["assignment_sha256"]),
            }
        )
        unit_receipts.append(receipt)
    unit_dispatch_seconds = time.perf_counter() - unit_dispatch_started
    assignment_payload = json.dumps(
        ordered_assignments, separators=(",", ":"), sort_keys=True
    ).encode()
    unit_wall_key = "outer_wall_seconds" if profile_mode else "total_wall_seconds"
    unit_wall_seconds = [float(receipt[unit_wall_key]) for receipt in unit_receipts]
    process = _process_receipt()
    boundary_before_receipt = max(
        0.0,
        time.perf_counter()
        - batch_started
        - resume_preflight_seconds
        - unit_dispatch_seconds,
    )
    phase_seconds = {
        "resume_preflight": resume_preflight_seconds,
        "unit_dispatch": unit_dispatch_seconds,
        "batch_boundary_overhead": boundary_before_receipt,
        "batch_receipt_fsync": 0.0,
    }
    batch_wall_seconds = time.perf_counter() - batch_started
    batch = {
        "schema": "banana-smasher-qtip-resident-batch-v1",
        "status": "PASS",
        "host": os.uname().nodename,
        "layer": layer,
        "tier": tier,
        "all_cells": all_cells,
        "mode": "profile" if profile_mode else "solve",
        "fresh_no_warm_start": True,
        "unit_state_isolation": "independent objectives/codebooks/weights/assignments",
        "shared_staging": [
            "python modules",
            "capture bank",
            "hessian manifest binding",
            "TLUT",
            "model index",
        ],
        "process": process,
        "epoch_started": epoch_started,
        "epoch_ended": time.time(),
        "units": len(unit_receipts),
        "resumed_units": resumed_units,
        "computed_units": computed_units,
        "resume": {
            "enabled": resume,
            "explicit_flag": resume_flag_explicit,
            "policy": "hash-validate-pass-skip",
        },
        "kernel_cache_root": (
            str(kernel_cache_root.resolve()) if kernel_cache_root is not None else None
        ),
        "batch_wall_seconds": batch_wall_seconds,
        "phase_seconds": phase_seconds,
        "mean_public_outer_seconds": batch_wall_seconds / len(unit_receipts),
        "mean_unit_receipt_outer_seconds": sum(unit_wall_seconds) / len(unit_wall_seconds),
        "min_unit_receipt_outer_seconds": min(unit_wall_seconds),
        "max_unit_receipt_outer_seconds": max(unit_wall_seconds),
        "ordered_assignment_sha256": hashlib.sha256(assignment_payload).hexdigest(),
        "ordered_assignment_encoding": "canonical-json-sort-keys-compact-v1",
        "ordered_assignments": ordered_assignments,
        "config_root": str(config_root.resolve()),
        "config_paths": [str(path.resolve()) for path in paths],
    }
    batch = _public_receipt(batch)
    receipt_path = root / "solve" / f"L{layer:03d}" / "QTIP_BATCH_RECEIPT.json"
    receipt_fsync_started = time.perf_counter()
    _atomic_json(receipt_path, batch)
    phase_seconds["batch_receipt_fsync"] = (
        time.perf_counter() - receipt_fsync_started
    )
    batch["batch_wall_seconds"] = time.perf_counter() - batch_started
    batch["mean_public_outer_seconds"] = (
        batch["batch_wall_seconds"] / len(unit_receipts)
    )
    batch["epoch_ended"] = time.time()
    _atomic_json(receipt_path, batch)
    return batch
