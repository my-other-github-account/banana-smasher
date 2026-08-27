from __future__ import annotations

import hashlib
import json
import math
import os
import re
import struct
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol

HF_SOURCE_ADMISSION_SCHEMA = "banana-smasher-hf-source-admission-v1"
HF_UNIFORM_PLAN_SCHEMA = "banana-smasher-hf-moe-uniform-plan-v1"
_REVISION_RE = re.compile(r"[0-9a-f]{40}(?:[0-9a-f]{24})?\Z")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _regular_file(path: Path, label: str) -> Path:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"HF source {label} is missing or non-regular: {path}")
    return path


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def admit_hf_source(
    model: str | Path,
    *,
    revision: str,
    receipt_path: str | Path,
) -> dict[str, Any]:
    """Admit one immutable, revision-pinned sharded Hugging Face source tree."""

    if not _REVISION_RE.fullmatch(str(revision)):
        raise ValueError("HF source revision must be a lowercase 40- or 64-digit hex identity")
    root = Path(model).expanduser().resolve()
    config_path = _regular_file(root / "config.json", "config.json")
    index_path = _regular_file(
        root / "model.safetensors.index.json", "model.safetensors.index.json"
    )
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
        index = json.loads(index_path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("HF source config/index is not valid UTF-8 JSON") from exc
    if not isinstance(config, dict):
        raise TypeError("HF source config must be a JSON object")
    weight_map = index.get("weight_map") if isinstance(index, dict) else None
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError("HF source index requires a non-empty weight_map")
    if any(not isinstance(name, str) or not name for name in weight_map):
        raise ValueError("HF source index contains an invalid tensor name")
    shards: set[str] = set()
    for shard in weight_map.values():
        if not isinstance(shard, str) or Path(shard).name != shard:
            raise ValueError(f"HF source index contains an unsafe shard binding: {shard!r}")
        _regular_file(root / shard, f"shard {shard}")
        shards.add(shard)
    receipt = {
        "schema": HF_SOURCE_ADMISSION_SCHEMA,
        "status": "PASS",
        "model_root": str(root),
        "revision": str(revision),
        "config_sha256": _sha256(config_path),
        "model_index_sha256": _sha256(index_path),
        "tensor_count": len(weight_map),
        "shards": sorted(shards),
        "shard_count": len(shards),
        "source_mutated": False,
    }
    destination = Path(receipt_path).expanduser().resolve()
    _atomic_json(destination, receipt)
    return receipt


class HFMoeAdapter(Protocol):
    """Architecture-neutral contract for classifying HF MoE tensor names."""

    adapter_id: str

    def matches(self, config: Mapping[str, Any], tensor_names: Sequence[str]) -> bool: ...

    def routed_weight_names(
        self, config: Mapping[str, Any], tensor_names: Sequence[str]
    ) -> set[str]: ...


def _nested_positive_int(document: Mapping[str, Any], key: str) -> int | None:
    found: list[int] = []

    def visit(value: Any) -> None:
        if isinstance(value, Mapping):
            for name, child in value.items():
                if name == key and isinstance(child, int) and not isinstance(child, bool) and child > 0:
                    found.append(child)
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    visit(document)
    unique = sorted(set(found))
    if len(unique) > 1:
        raise ValueError(f"HF config declares conflicting {key} values: {unique}")
    return unique[0] if unique else None


class NumericExpertsAdapter:
    """Generic adapter for ``experts.<numeric-id>.*.weight`` HF MoE layouts."""

    adapter_id = "hf-numeric-experts-v1"
    _routed = re.compile(r"(?:^|\.)experts\.(\d+)\..+\.weight\Z")

    def matches(self, config: Mapping[str, Any], tensor_names: Sequence[str]) -> bool:
        count = _nested_positive_int(config, "n_routed_experts")
        return count is not None and any(self._routed.search(name) for name in tensor_names)

    def routed_weight_names(
        self, config: Mapping[str, Any], tensor_names: Sequence[str]
    ) -> set[str]:
        count = _nested_positive_int(config, "n_routed_experts")
        selected: set[str] = set()
        for tensor_name in tensor_names:
            match = self._routed.search(tensor_name)
            if match and count is not None and int(match.group(1)) < count:
                selected.add(tensor_name)
        return selected


HF_MOE_ADAPTERS: tuple[HFMoeAdapter, ...] = (NumericExpertsAdapter(),)


def _safetensors_header(path: Path) -> dict[str, Any]:
    with path.open("rb") as stream:
        prefix = stream.read(8)
        if len(prefix) != 8:
            raise ValueError(f"safetensors shard has no complete header length: {path}")
        length = struct.unpack("<Q", prefix)[0]
        if length <= 1 or length > 256 << 20:
            raise ValueError(f"safetensors shard header length is invalid: {path}: {length}")
        payload = stream.read(length)
    if len(payload) != length:
        raise ValueError(f"safetensors shard header is truncated: {path}")
    try:
        header = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"safetensors shard header is invalid JSON: {path}") from exc
    if not isinstance(header, dict):
        raise TypeError(f"safetensors shard header must be an object: {path}")
    header.pop("__metadata__", None)
    return header


def plan_hf_moe_uniform(
    model: str | Path,
    *,
    revision: str,
    tier: str,
    scope: str,
    native_rest: bool,
    receipt_path: str | Path,
) -> dict[str, Any]:
    """Plan a metadata-only routed-Q2/native-rest build without loading tensors."""

    if tier not in {"q2", "qtip2"}:
        raise ValueError("HF MoE uniform planner currently requires tier='q2'")
    if scope != "routed_only" or native_rest is not True:
        raise ValueError("HF MoE uniform planner requires routed_only with native_rest=True")
    destination = Path(receipt_path).expanduser().resolve()
    source = admit_hf_source(
        model,
        revision=revision,
        receipt_path=destination.with_name("SOURCE_ADMISSION.json"),
    )
    root = Path(source["model_root"])
    config = json.loads((root / "config.json").read_text(encoding="utf-8"))
    index = json.loads(
        (root / "model.safetensors.index.json").read_text(encoding="utf-8")
    )
    weight_map = index["weight_map"]
    tensor_names = sorted(weight_map)
    matches = [adapter for adapter in HF_MOE_ADAPTERS if adapter.matches(config, tensor_names)]
    if len(matches) != 1:
        raise ValueError(
            "HF MoE adapter selection must resolve exactly once: "
            f"matched={[adapter.adapter_id for adapter in matches]}"
        )
    adapter = matches[0]
    headers = {
        shard: _safetensors_header(root / shard) for shard in source["shards"]
    }
    rows: list[dict[str, Any]] = []
    missing: list[str] = []
    duplicate: list[str] = []
    seen: set[str] = set()
    for name in tensor_names:
        shard = weight_map[name]
        metadata = headers[shard].get(name)
        if not isinstance(metadata, dict):
            missing.append(name)
            continue
        if name in seen:
            duplicate.append(name)
            continue
        seen.add(name)
        shape = metadata.get("shape")
        offsets = metadata.get("data_offsets")
        dtype = metadata.get("dtype")
        if (
            not isinstance(shape, list)
            or any(not isinstance(value, int) or value < 0 for value in shape)
            or not isinstance(offsets, list)
            or len(offsets) != 2
            or any(not isinstance(value, int) or value < 0 for value in offsets)
            or offsets[1] < offsets[0]
            or not isinstance(dtype, str)
        ):
            raise ValueError(f"invalid safetensors metadata for {name}")
        rows.append(
            {
                "name": name,
                "shard": shard,
                "dtype": dtype,
                "parameters": math.prod(shape),
                "source_bytes": offsets[1] - offsets[0],
            }
        )
    indexed = set(tensor_names)
    unindexed = sorted(
        name for header in headers.values() for name in header if name not in indexed
    )
    duplicate.extend(unindexed)
    routed_names = adapter.routed_weight_names(config, tensor_names)
    routed = [row for row in rows if row["name"] in routed_names]
    native = [row for row in rows if row["name"] not in routed_names]
    if not routed:
        raise ValueError("HF MoE adapter selected zero routed expert weights")
    layer_pattern = re.compile(r"(?:^|\.)layers\.(\d+)\.")
    model_layer_ids = sorted(
        {
            int(match.group(1))
            for name in tensor_names
            if (match := layer_pattern.search(name)) is not None
        }
    )
    routed_layer_ids = sorted(
        {
            int(match.group(1))
            for name in routed_names
            if (match := layer_pattern.search(name)) is not None
        }
    )
    expected_model_layers = _nested_positive_int(config, "num_hidden_layers")
    if expected_model_layers is None:
        raise ValueError("HF MoE config does not declare num_hidden_layers")
    model_layer_gaps = sorted(set(range(expected_model_layers)) - set(model_layer_ids))
    plan = {
        "schema": HF_UNIFORM_PLAN_SCHEMA,
        "status": (
            "PASS" if not missing and not duplicate and not model_layer_gaps else "FAILED"
        ),
        "api": {"method": "plan_hf_moe_uniform", "version": 1},
        "source": source,
        "intent": {"tier": "q2", "scope": scope, "native_rest": True},
        "adapter": {"id": adapter.adapter_id},
        "geometry": {
            "expected_model_layers": expected_model_layers,
            "model_layer_ids": model_layer_ids,
            "routed_layer_ids": routed_layer_ids,
            "model_layer_gaps": model_layer_gaps,
        },
        "routed_tensors": routed,
        "native_tensors": native,
        "accounting": {
            "source_tensor_count": len(tensor_names),
            "routed_tensor_count": len(routed),
            "native_tensor_count": len(native),
            "routed_parameters": sum(row["parameters"] for row in routed),
            "native_parameters": sum(row["parameters"] for row in native),
            "routed_source_bytes": sum(row["source_bytes"] for row in routed),
            "native_source_bytes": sum(row["source_bytes"] for row in native),
        },
        "coverage": {"gaps": sorted(missing), "duplicates": sorted(set(duplicate))},
        "mechanisms": {"fallback": 0},
    }
    _atomic_json(destination, plan)
    if plan["status"] != "PASS":
        raise ValueError("HF MoE plan has tensor coverage gaps or duplicates")
    return plan
