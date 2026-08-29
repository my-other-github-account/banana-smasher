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
HF_UNIFORM_ARTIFACT_SCHEMA = "banana-smasher-hf-moe-uniform-artifact-v1"
HF_UNIFORM_SHARD_SCHEMA = "banana-smasher-hf-moe-uniform-shard-v1"
HF_ROUTED_SCOPE_SCHEMA = "banana-smasher-hf-moe-routed-scope-v1"
_REVISION_RE = re.compile(r"[0-9a-f]{40}(?:[0-9a-f]{24})?\Z")

#: Client-side bookkeeping subtrees a standard ``hf_hub`` download writes beside the
#: repository content.  They are excluded from the authoritative repository roster and
#: the exclusion is always published in the admission receipt so a caller can reproduce
#: the documented file/byte identity without inferring the rule.
HF_CLIENT_BOOKKEEPING_PREFIXES: tuple[str, ...] = (".cache/huggingface/",)

#: Data-driven, architecture-neutral statement of how routed scope is bounded.  Layer
#: ids at or above ``num_hidden_layers`` belong to auxiliary heads (multi-token
#: prediction / "nextn") whose experts are never routed and always stay native rest.
HF_AUXILIARY_LAYER_RULE = (
    "routed layer ids are [first_k_dense_replace, num_hidden_layers); layer ids >= "
    "num_hidden_layers are auxiliary prediction heads (e.g. num_nextn_predict_layers) "
    "whose expert tensors remain native rest and are never routed"
)

#: Public calls that need the heavyweight ``[solve]`` extra rather than the base install.
HF_SOLVE_EXTRA_REQUIREMENT = (
    "install './banana-smasher[solve]' — this call encodes tensors and requires the "
    "solve extra (torch); the base install only supports the metadata-only calls "
    "admit_hf_source, discover_hf_moe_routed_scope, plan_hf_moe_uniform and "
    "preflight_hf_moe_output_fit"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    offset = 0
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
            posix_fadvise = getattr(os, "posix_fadvise", None)
            dontneed = getattr(os, "POSIX_FADV_DONTNEED", None)
            if posix_fadvise is not None and dontneed is not None:
                try:
                    posix_fadvise(stream.fileno(), offset, len(block), dontneed)
                except OSError:
                    pass
            offset += len(block)
    return digest.hexdigest()


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


def _bound_file(root: Path, relative: str, label: str) -> dict[str, Any]:
    """Bind one repository member by CONTENT identity, resolving HF cache symlinks.

    The canonical ``hf download`` / ``snapshot_download`` layout materializes every
    repository file as a symlink into a sibling ``blobs/`` store.  Identity is therefore
    proven by the resolved target's content hash, inode and size — never by the
    filesystem link type.  Only unresolvable, non-regular, or escaping targets are
    rejected.
    """

    path = root / relative
    try:
        resolved = path.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise ValueError(f"HF source {label} is missing or unresolvable: {path}") from exc
    if not resolved.is_file():
        raise ValueError(f"HF source {label} is not a regular file: {path} -> {resolved}")
    status = resolved.stat()
    return {
        "path": relative,
        "realpath": str(resolved),
        "symlink": path.is_symlink(),
        "inode": status.st_ino,
        "bytes": status.st_size,
        "sha256": _sha256(resolved),
    }


def _repository_roster(root: Path) -> dict[str, Any]:
    """Partition the staged tree into authoritative repository content vs bookkeeping.

    A tree fetched with the standard ``hf_hub`` client carries the repository files plus
    a client-side ``.cache/huggingface/`` subtree of per-file download stamps.  Both
    partitions are reported so a caller can reproduce the documented file/byte identity
    without having to infer which paths are bookkeeping.
    """

    repository: list[str] = []
    excluded: list[str] = []
    repository_bytes = 0
    excluded_bytes = 0
    for path in sorted(root.rglob("*")):
        if path.is_dir() and not path.is_symlink():
            continue
        relative = path.relative_to(root).as_posix()
        try:
            size = path.resolve(strict=True).stat().st_size
        except (OSError, RuntimeError):
            size = 0
        if relative.startswith(HF_CLIENT_BOOKKEEPING_PREFIXES):
            excluded.append(relative)
            excluded_bytes += size
        else:
            repository.append(relative)
            repository_bytes += size
    return {
        "excluded_prefixes": list(HF_CLIENT_BOOKKEEPING_PREFIXES),
        "repository_files": repository,
        "repository_file_count": len(repository),
        "repository_bytes": repository_bytes,
        "excluded_files": excluded,
        "excluded_file_count": len(excluded),
        "excluded_bytes": excluded_bytes,
    }


def admit_hf_source(
    model: str | Path,
    *,
    revision: str,
    receipt_path: str | Path,
) -> dict[str, Any]:
    """Admit one immutable, revision-pinned sharded Hugging Face source tree.

    Accepts the canonical HF cache/snapshot layout: each declared member is resolved
    through any symlink and bound by realpath, inode and content SHA-256.  The receipt
    publishes the authoritative repository roster and the excluded client-side
    bookkeeping subtree so file/byte identity is reproducible by the caller.
    """

    if not _REVISION_RE.fullmatch(str(revision)):
        raise ValueError("HF source revision must be a lowercase 40- or 64-digit hex identity")
    root = Path(model).expanduser().resolve()
    config_member = _bound_file(root, "config.json", "config.json")
    index_member = _bound_file(
        root, "model.safetensors.index.json", "model.safetensors.index.json"
    )
    config_path = Path(config_member["realpath"])
    index_path = Path(index_member["realpath"])
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
    shard_members: list[dict[str, Any]] = []
    for shard in weight_map.values():
        if not isinstance(shard, str) or Path(shard).name != shard:
            raise ValueError(f"HF source index contains an unsafe shard binding: {shard!r}")
        if shard not in shards:
            shard_members.append(_bound_file(root, shard, f"shard {shard}"))
        shards.add(shard)
    receipt = {
        "schema": HF_SOURCE_ADMISSION_SCHEMA,
        "status": "PASS",
        "model_root": str(root),
        "revision": str(revision),
        "config_sha256": config_member["sha256"],
        "model_index_sha256": index_member["sha256"],
        "tensor_count": len(weight_map),
        "shards": sorted(shards),
        "shard_count": len(shards),
        "source_mutated": False,
        "binding": {
            "identity": "content-sha256",
            "symlinks_resolved": any(
                member["symlink"]
                for member in (config_member, index_member, *shard_members)
            ),
            "members": sorted(
                (config_member, index_member, *shard_members), key=lambda row: row["path"]
            ),
        },
        "roster_boundary": _repository_roster(root),
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
    _layer = re.compile(r"(?:^|\.)layers\.(\d+)\.")

    def matches(self, config: Mapping[str, Any], tensor_names: Sequence[str]) -> bool:
        count = _nested_positive_int(config, "n_routed_experts")
        return count is not None and any(self._routed.search(name) for name in tensor_names)

    def routed_weight_names(
        self, config: Mapping[str, Any], tensor_names: Sequence[str]
    ) -> set[str]:
        count = _nested_positive_int(config, "n_routed_experts")
        layer_count = _nested_positive_int(config, "num_hidden_layers")
        selected: set[str] = set()
        for tensor_name in tensor_names:
            match = self._routed.search(tensor_name)
            layer = self._layer.search(tensor_name)
            if (
                match
                and layer
                and count is not None
                and layer_count is not None
                and int(match.group(1)) < count
                and int(layer.group(1)) < layer_count
            ):
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


def _derive_routed_scope(
    config: Mapping[str, Any], tensor_names: Sequence[str]
) -> tuple[Any, set[str], dict[str, Any]]:
    """Select exactly one adapter and derive routed/native scope plus its geometry.

    The layer bound is data-driven and architecture-neutral: routed layer ids are
    ``[first_k_dense_replace, num_hidden_layers)`` and any observed layer id at or above
    ``num_hidden_layers`` is an auxiliary prediction head whose experts stay native.
    """

    matches = [adapter for adapter in HF_MOE_ADAPTERS if adapter.matches(config, tensor_names)]
    if len(matches) != 1:
        raise ValueError(
            "HF MoE adapter selection must resolve exactly once: "
            f"matched={[adapter.adapter_id for adapter in matches]}"
        )
    adapter = matches[0]
    routed_names = adapter.routed_weight_names(config, tensor_names)
    if not routed_names:
        raise ValueError("HF MoE adapter selected zero routed expert weights")
    layer_pattern = re.compile(r"(?:^|\.)layers\.(\d+)\.")
    observed_layer_ids = sorted(
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
    model_layer_ids = [
        layer for layer in observed_layer_ids if layer < expected_model_layers
    ]
    auxiliary_layer_ids = [
        layer for layer in observed_layer_ids if layer >= expected_model_layers
    ]
    geometry = {
        "expected_model_layers": expected_model_layers,
        "dense_prefix_layers": _nested_positive_int(config, "first_k_dense_replace") or 0,
        "routed_experts": _nested_positive_int(config, "n_routed_experts"),
        "model_layer_ids": model_layer_ids,
        "auxiliary_layer_ids": auxiliary_layer_ids,
        "auxiliary_layer_rule": HF_AUXILIARY_LAYER_RULE,
        "auxiliary_layer_deciding_config_keys": [
            "num_hidden_layers",
            "first_k_dense_replace",
            "n_routed_experts",
            "num_nextn_predict_layers",
        ],
        "routed_layer_ids": routed_layer_ids,
        "model_layer_gaps": sorted(set(range(expected_model_layers)) - set(model_layer_ids)),
        "routed_auxiliary_layers": sorted(set(routed_layer_ids) & set(auxiliary_layer_ids)),
    }
    return adapter, routed_names, geometry


def discover_hf_moe_routed_scope(
    model: str | Path,
    *,
    revision: str,
    receipt_path: str | Path,
) -> dict[str, Any]:
    """Answer 'which tensors would routed_only select' from config + index alone.

    Read-only routed-scope discovery.  This is a metadata-only call available under the
    base install: it admits the pinned source, selects exactly one registered MoE
    adapter, and returns the adapter id plus sorted routed/native tensor name
    inventories.  It reads no tensor bytes, plans no build, and performs no output-fit
    projection, so a caller can inspect routed scope before staging a large source.
    """

    destination = Path(receipt_path).expanduser().resolve()
    source = admit_hf_source(
        model,
        revision=revision,
        receipt_path=destination.with_name("ROUTED_SCOPE_SOURCE_ADMISSION.json"),
    )
    root = Path(source["model_root"])
    config = json.loads((root / "config.json").read_text(encoding="utf-8"))
    index = json.loads((root / "model.safetensors.index.json").read_text(encoding="utf-8"))
    tensor_names = sorted(index["weight_map"])
    adapter, routed_names, geometry = _derive_routed_scope(config, tensor_names)
    native_names = [name for name in tensor_names if name not in routed_names]
    receipt = {
        "schema": HF_ROUTED_SCOPE_SCHEMA,
        "status": "PASS" if not geometry["model_layer_gaps"] and not geometry["routed_auxiliary_layers"] else "FAILED",
        "api": {"method": "discover_hf_moe_routed_scope", "version": 1},
        "reads_tensor_bytes": False,
        "source": source,
        "scope": "routed_only",
        "adapter": {"id": adapter.adapter_id},
        "geometry": geometry,
        "routed_tensor_names": sorted(routed_names),
        "native_tensor_names": native_names,
        "accounting": {
            "source_tensor_count": len(tensor_names),
            "routed_tensor_count": len(routed_names),
            "native_tensor_count": len(native_names),
        },
        "mechanisms": {"fallback": 0},
    }
    _atomic_json(destination, receipt)
    if receipt["status"] != "PASS":
        raise ValueError("HF MoE routed-scope discovery found layer coverage defects")
    return receipt


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
    adapter, routed_names, geometry = _derive_routed_scope(config, tensor_names)
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
                "shape": shape,
                "parameters": math.prod(shape),
                "source_bytes": offsets[1] - offsets[0],
            }
        )
    indexed = set(tensor_names)
    unindexed = sorted(
        name for header in headers.values() for name in header if name not in indexed
    )
    duplicate.extend(unindexed)
    routed = [row for row in rows if row["name"] in routed_names]
    native = [row for row in rows if row["name"] not in routed_names]
    if not routed:
        raise ValueError("HF MoE adapter selected zero routed expert weights")
    model_layer_gaps = geometry["model_layer_gaps"]
    routed_auxiliary_layers = geometry["routed_auxiliary_layers"]
    plan = {
        "schema": HF_UNIFORM_PLAN_SCHEMA,
        "status": (
            "PASS"
            if not missing
            and not duplicate
            and not model_layer_gaps
            and not routed_auxiliary_layers
            else "FAILED"
        ),
        "api": {"method": "plan_hf_moe_uniform", "version": 1},
        "source": source,
        "intent": {"tier": "q2", "scope": scope, "native_rest": True},
        "adapter": {"id": adapter.adapter_id},
        "geometry": geometry,
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
        raise ValueError("HF MoE plan has tensor coverage or routed-scope defects")
    return plan


def preflight_hf_moe_output_fit(
    plan: Mapping[str, Any],
    *,
    free_bytes: int | None = None,
    output_root: str | Path | None = None,
    reserve_bytes: int = 8 << 30,
    native_spill_root: str | Path | None = None,
    native_spill_free_bytes: int | None = None,
    native_spill_reserve_bytes: int = 4 << 30,
    receipt_path: str | Path,
) -> dict[str, Any]:
    """Project exact Q2/code/native bytes across admitted local storage roots."""

    if plan.get("status") != "PASS":
        raise ValueError("output-fit preflight requires a PASS HF MoE plan")
    if plan.get("intent") != {
        "tier": "q2",
        "scope": "routed_only",
        "native_rest": True,
    }:
        raise ValueError("output-fit preflight requires routed-only Q2 with native rest")
    if free_bytes is None:
        if output_root is None:
            raise ValueError("output-fit preflight requires output_root or measured free_bytes")
        import shutil

        probe = Path(output_root).expanduser().resolve()
        probe.mkdir(parents=True, exist_ok=True)
        free_bytes = shutil.disk_usage(probe).free
    for value, label in ((free_bytes, "free_bytes"), (reserve_bytes, "reserve_bytes")):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{label} must be a positive integer")
    q2_code_bytes = 0
    q2_scale_bytes = 0
    for row in plan["routed_tensors"]:
        shape = row.get("shape")
        if (
            not isinstance(shape, list)
            or len(shape) != 2
            or any(not isinstance(value, int) or value < 1 for value in shape)
        ):
            raise ValueError(f"routed tensor lacks matrix shape: {row.get('name')}")
        rows, width = shape
        q2_code_bytes += rows * math.ceil(width * 2 / 16) * 2
        q2_scale_bytes += rows * 4
    native_bytes = int(plan["accounting"]["native_source_bytes"])
    metadata_bytes = (
        len(json.dumps(plan, sort_keys=True, separators=(",", ":")).encode())
        + 256 * (2 * len(plan["routed_tensors"]) + len(plan.get("native_tensors", [])))
        + 4096
    )
    payload_bytes = native_bytes + q2_code_bytes + q2_scale_bytes + metadata_bytes
    required_bytes = payload_bytes + reserve_bytes
    primary_required_bytes = q2_code_bytes + q2_scale_bytes + metadata_bytes + reserve_bytes
    storage_mode = "primary-only-v1"
    spill_path: Path | None = None
    spill_free: int | None = None
    spill_required: int | None = None
    if (
        native_spill_root is None
        and free_bytes < required_bytes
        and output_root is not None
        and Path("/dev/shm").is_dir()
    ):
        native_spill_root = Path("/dev/shm/banana-smasher-hf-moe-native")
    if native_spill_root is not None:
        import shutil

        spill_path = Path(native_spill_root).expanduser().resolve()
        spill_probe = spill_path if spill_path.exists() else spill_path.parent
        spill_probe.mkdir(parents=True, exist_ok=True)
        spill_free = (
            native_spill_free_bytes
            if native_spill_free_bytes is not None
            else shutil.disk_usage(spill_probe).free
        )
        if (
            isinstance(spill_free, bool)
            or not isinstance(spill_free, int)
            or spill_free <= 0
            or isinstance(native_spill_reserve_bytes, bool)
            or not isinstance(native_spill_reserve_bytes, int)
            or native_spill_reserve_bytes <= 0
        ):
            raise ValueError("native spill free/reserve bytes must be positive integers")
        spill_required = native_bytes + native_spill_reserve_bytes
        if free_bytes >= primary_required_bytes and spill_free >= spill_required:
            storage_mode = "split-native-local-v1"
    passed = free_bytes >= required_bytes or storage_mode == "split-native-local-v1"
    receipt = {
        "schema": "banana-smasher-hf-moe-output-fit-v1",
        "status": "PASS" if passed else "FAILED",
        "storage_mode": storage_mode,
        "free_bytes": free_bytes,
        "native_payload_bytes": native_bytes,
        "q2_code_bytes": q2_code_bytes,
        "q2_scale_bytes": q2_scale_bytes,
        "metadata_bytes": metadata_bytes,
        "projected_payload_bytes": payload_bytes,
        "reserve_bytes": reserve_bytes,
        "required_bytes": required_bytes,
        "margin_bytes": free_bytes - required_bytes,
        "primary_required_bytes": primary_required_bytes,
        "primary_margin_bytes": free_bytes - primary_required_bytes,
        "native_spill_root": str(spill_path) if spill_path is not None else None,
        "native_spill_free_bytes": spill_free,
        "native_spill_reserve_bytes": (
            native_spill_reserve_bytes if spill_path is not None else None
        ),
        "native_spill_required_bytes": spill_required,
        "native_spill_margin_bytes": (
            spill_free - spill_required
            if spill_free is not None and spill_required is not None
            else None
        ),
    }
    _atomic_json(Path(receipt_path).expanduser().resolve(), receipt)
    return receipt


def _require_solve_extra(method: str) -> None:
    """Fail closed with the declared dependency boundary before any expensive work.

    The base install supports the metadata-only planning tier; calls that encode
    production-sized tensors need the ``[solve]`` extra and say so explicitly instead of
    raising a bare ``ModuleNotFoundError`` from deep inside the encoder stack.
    """

    import importlib.util

    if importlib.util.find_spec("torch") is None:
        raise RuntimeError(
            f"{method} requires the banana-smasher [solve] extra: {HF_SOLVE_EXTRA_REQUIREMENT}"
        )


def estimate_hf_moe_uniform(
    model: str | Path,
    *,
    revision: str,
    tier: str,
    scope: str,
    native_rest: bool,
    receipt_path: str | Path,
) -> dict[str, Any]:
    """Encode one representative routed tensor and project a complete build.

    Dependency boundary: the metadata-only portion runs under the base install.  Any
    routed tensor large enough to need the CUDA encoder fails closed naming the
    ``[solve]`` extra (``HF_SOLVE_EXTRA_REQUIREMENT``) rather than raising a bare
    ``ModuleNotFoundError`` from inside the encoder stack.
    """

    import time
    import tracemalloc

    from .qtip1 import QTIP2_GEOMETRY, gaussian_tlut

    destination = Path(receipt_path).expanduser().resolve()
    plan = plan_hf_moe_uniform(
        model,
        revision=revision,
        tier=tier,
        scope=scope,
        native_rest=native_rest,
        receipt_path=destination.with_name("CANARY_PLAN.json"),
    )
    ordered = sorted(plan["routed_tensors"], key=lambda row: (row["source_bytes"], row["name"]))
    selected = ordered[len(ordered) // 2]
    root = Path(plan["source"]["model_root"])
    tlut = gaussian_tlut(bits=QTIP2_GEOMETRY.tlut_bits, columns=QTIP2_GEOMETRY.V)
    try:
        import torch
    except ModuleNotFoundError:
        torch = None
    if torch is not None and torch.cuda.is_available():
        from .trellis_v2.exact import prepare_exact_cuda

        prepare_exact_cuda()
    tracemalloc.start()
    started = time.perf_counter()
    matrix = _load_safetensors_matrix(root / selected["shard"], selected)
    encoded, encoder = _encode_hf_q2(matrix, geometry=QTIP2_GEOMETRY, tlut=tlut)
    wall_seconds = time.perf_counter() - started
    _, traced_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    peak_memory_bytes = int(traced_peak + matrix.nbytes + encoded.packed.nbytes + encoded.scales.nbytes)
    total_parameters = int(plan["accounting"]["routed_parameters"])
    complete_wall_seconds = wall_seconds * total_parameters / int(selected["parameters"])
    q2_code_bytes = 0
    q2_scale_bytes = 0
    for row in plan["routed_tensors"]:
        rows, width = row["shape"]
        q2_code_bytes += rows * math.ceil(width * 2 / 16) * 2
        q2_scale_bytes += rows * 4
    complete_payload_bytes = (
        int(plan["accounting"]["native_source_bytes"])
        + q2_code_bytes
        + q2_scale_bytes
    )
    receipt = {
        "schema": "banana-smasher-hf-moe-build-estimate-v1",
        "status": "PASS_DIAGNOSTIC",
        "artifact_admissible": False,
        "artifact_created": False,
        "api": {"method": "estimate_hf_moe_uniform", "version": 1},
        "source": plan["source"],
        "intent": plan["intent"],
        "canary": {
            "routed_tensor_count": 1,
            "name": selected["name"],
            "shape": selected["shape"],
            "parameters": selected["parameters"],
            "source_bytes": selected["source_bytes"],
            "source_dtype": selected["dtype"],
            "wall_seconds": wall_seconds,
            "peak_memory_bytes": peak_memory_bytes,
            "q2_code_bytes": encoded.packed.nbytes,
            "q2_scale_bytes": encoded.scales.nbytes,
            "encoder": encoder,
        },
        "projection": {
            "complete_routed_tensor_count": len(plan["routed_tensors"]),
            "complete_routed_parameters": total_parameters,
            "complete_wall_seconds": complete_wall_seconds,
            "complete_q2_code_bytes": q2_code_bytes,
            "complete_q2_scale_bytes": q2_scale_bytes,
            "complete_native_bytes": int(plan["accounting"]["native_source_bytes"]),
            "complete_payload_bytes": complete_payload_bytes,
        },
        "mechanisms": {"fallback": 0},
    }
    _atomic_json(destination, receipt)
    return receipt


def _member_name(tensor_name: str, suffix: str) -> str:
    return f"{hashlib.sha256(tensor_name.encode()).hexdigest()}.{suffix}"


def _copy_tensor_data_bytes(
    *, source: Path, metadata: Mapping[str, Any], destination: Path
) -> str:
    offsets = metadata["data_offsets"]
    with source.open("rb") as stream:
        header_length = struct.unpack("<Q", stream.read(8))[0]
        stream.seek(8 + header_length + int(offsets[0]))
        payload = stream.read(int(offsets[1]) - int(offsets[0]))
    if len(payload) != int(offsets[1]) - int(offsets[0]):
        raise ValueError(f"safetensors tensor data is truncated: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    return hashlib.sha256(payload).hexdigest()


def _load_safetensors_matrix(source: Path, row: Mapping[str, Any]):
    import numpy as np

    if row["dtype"] != "F8_E4M3":
        from safetensors import safe_open

        with safe_open(source, framework="numpy") as handle:
            return np.asarray(handle.get_tensor(row["name"]), dtype=np.float32)
    metadata = _safetensors_header(source)[row["name"]]
    offsets = metadata["data_offsets"]
    with source.open("rb") as stream:
        header_length = struct.unpack("<Q", stream.read(8))[0]
        stream.seek(8 + header_length + int(offsets[0]))
        raw = np.frombuffer(
            stream.read(int(offsets[1]) - int(offsets[0])), dtype=np.uint8
        )
    bits = np.arange(256, dtype=np.uint16)
    exponent = (bits >> 3) & 0xF
    mantissa = bits & 0x7
    magnitude = np.where(
        exponent == 0,
        np.ldexp(mantissa.astype(np.float32) / 8.0, -6),
        np.ldexp(
            1.0 + mantissa.astype(np.float32) / 8.0,
            exponent.astype(int) - 7,
        ),
    ).astype(np.float32)
    magnitude[(exponent == 15) & (mantissa == 7)] = np.nan
    lookup = np.where(bits & 0x80, -magnitude, magnitude).astype(np.float32)
    matrix = lookup[raw].reshape(tuple(row["shape"]))
    if not np.isfinite(matrix).all():
        raise ValueError(
            f"F8_E4M3 routed tensor contains non-finite values: {row['name']}"
        )
    return matrix


def _encode_hf_q2(matrix: Any, *, geometry: Any, tlut: Any):
    """Use bounded CUDA for production tensors and keep tiny fixture coverage.

    Declared dependency boundary: the numpy reference path covers only bounded fixtures
    (<= 1 MiB).  Anything larger requires the ``[solve]`` extra's CUDA encoder and fails
    closed naming it, rather than silently taking a slower fallback.
    """
    from .qtip1 import encode_qtip, encode_qtip2_bounded_cuda

    try:
        import torch
    except ModuleNotFoundError:
        torch = None
    if torch is not None and torch.cuda.is_available():
        return encode_qtip2_bounded_cuda(matrix, geometry=geometry, tlut=tlut)
    if int(matrix.nbytes) > 1 << 20:
        raise RuntimeError(
            "QTIP2 CUDA fast path is unavailable; refusing slower fallback. "
            f"{HF_SOLVE_EXTRA_REQUIREMENT}"
        )
    encoded = encode_qtip(matrix, geometry=geometry, tlut=tlut)
    return encoded, {
        "backend": "numpy-reference-small-fixture",
        "fixture_max_bytes": 1 << 20,
        "fallback": 0,
    }


def build_hf_moe_uniform(
    model: str | Path,
    *,
    revision: str,
    tier: str,
    scope: str,
    native_rest: bool,
    output: str | Path,
    native_spill_root: str | Path | None = None,
    _routed_ordinal_range: tuple[int, int] | None = None,
) -> dict[str, Any]:
    """Materialize a routed-Q2/native-rest HF MoE artifact and seal its receipt.

    Production-sized routed tensors require the ``[solve]`` extra's CUDA encoder; the
    encoder refuses any slower fallback and keeps only a bounded small-fixture path.
    """

    from .qtip1 import EncodedQtip, QTIP2_GEOMETRY, gaussian_tlut

    destination = Path(output).expanduser().resolve()
    if destination.exists():
        raise FileExistsError(f"HF MoE artifact output already exists: {destination}")

    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent))
    requested_spill = (
        Path(native_spill_root).expanduser().resolve()
        if native_spill_root is not None
        else Path("/dev/shm/banana-smasher-hf-moe-native").resolve() / destination.name
    )
    spill_staging: Path | None = None
    spill_promoted = False
    try:
        plan = plan_hf_moe_uniform(
            model,
            revision=revision,
            tier=tier,
            scope=scope,
            native_rest=native_rest,
            receipt_path=staging / "UNIFORM_PLAN.json",
        )
        root = Path(plan["source"]["model_root"])
        total_routed = len(plan["routed_tensors"])
        if _routed_ordinal_range is None:
            ordinal_start, ordinal_end = 0, total_routed
        else:
            ordinal_start, ordinal_end = _routed_ordinal_range
            if (
                isinstance(ordinal_start, bool)
                or isinstance(ordinal_end, bool)
                or not isinstance(ordinal_start, int)
                or not isinstance(ordinal_end, int)
                or ordinal_start < 0
                or ordinal_start >= ordinal_end
                or ordinal_end > total_routed
            ):
                raise ValueError(
                    f"routed ordinal range must be a non-empty subset of [0, {total_routed})"
                )
        selected_routed = plan["routed_tensors"][ordinal_start:ordinal_end]
        selected_native = plan["native_tensors"] if ordinal_start == 0 else []
        fit_plan = dict(plan)
        fit_plan["routed_tensors"] = selected_routed
        fit_plan["native_tensors"] = selected_native
        fit_plan["accounting"] = {
            **plan["accounting"],
            "native_source_bytes": sum(row["source_bytes"] for row in selected_native),
            "routed_parameters": sum(row["parameters"] for row in selected_routed),
        }
        fit = preflight_hf_moe_output_fit(
            fit_plan,
            output_root=destination.parent,
            native_spill_root=(
                requested_spill
                if native_spill_root is not None or Path("/dev/shm").is_dir()
                else None
            ),
            receipt_path=staging / "OUTPUT_FIT.json",
        )
        if fit["status"] != "PASS":
            raise ValueError("HF MoE artifact does not fit admitted local storage roots")
        split_native = fit["storage_mode"] == "split-native-local-v1"
        if split_native:
            if requested_spill.exists():
                raise FileExistsError(f"HF MoE native spill output already exists: {requested_spill}")
            requested_spill.parent.mkdir(parents=True, exist_ok=True)
            spill_staging = Path(
                tempfile.mkdtemp(prefix=f".{requested_spill.name}.", dir=requested_spill.parent)
            )
        headers = {
            shard: _safetensors_header(root / shard) for shard in plan["source"]["shards"]
        }
        tlut = gaussian_tlut(bits=QTIP2_GEOMETRY.tlut_bits, columns=QTIP2_GEOMETRY.V)
        routed_rows: list[dict[str, Any]] = []
        native_rows: list[dict[str, Any]] = []
        max_batch_tensors = 10
        routed_batches: list[list[dict[str, Any]]] = []
        open_batch_by_width: dict[int, list[dict[str, Any]]] = {}
        for row in selected_routed:
            width = int(row["shape"][1])
            batch = open_batch_by_width.get(width)
            if batch is None or len(batch) >= max_batch_tensors:
                batch = []
                routed_batches.append(batch)
                open_batch_by_width[width] = batch
            batch.append(row)
        import numpy as np

        routed_row_by_name: dict[str, dict[str, Any]] = {}
        for batch in routed_batches:
            matrices = []
            for row in batch:
                matrix = _load_safetensors_matrix(root / row["shard"], row)
                if matrix.ndim != 2:
                    raise ValueError(f"Q2 routed tensor must be a matrix: {row['name']}")
                matrices.append(matrix)
            combined = np.concatenate(matrices, axis=0)
            batch_encoded, batch_encoder = _encode_hf_q2(
                combined, geometry=QTIP2_GEOMETRY, tlut=tlut
            )
            row_offset = 0
            for batch_ordinal, (row, matrix) in enumerate(zip(batch, matrices, strict=True)):
                row_end = row_offset + int(matrix.shape[0])
                encoded = EncodedQtip(
                    geometry=batch_encoded.geometry,
                    shape=(int(matrix.shape[0]), int(matrix.shape[1])),
                    states=batch_encoded.states[row_offset:row_end],
                    packed=batch_encoded.packed[row_offset:row_end],
                    scales=batch_encoded.scales[row_offset:row_end],
                )
                encoder = {
                    **batch_encoder,
                    "packed_sha256": hashlib.sha256(encoded.packed.tobytes()).hexdigest(),
                    "scales_sha256": hashlib.sha256(encoded.scales.tobytes()).hexdigest(),
                    "batch_tensor_count": len(batch),
                    "batch_tensor_ordinal": batch_ordinal,
                    "batch_rows": int(combined.shape[0]),
                }
                trellis = staging / "routed" / _member_name(row["name"], "trellis.npy")
                scales = staging / "routed" / _member_name(row["name"], "scales.npy")
                trellis.parent.mkdir(parents=True, exist_ok=True)
                np.save(trellis, encoded.packed, allow_pickle=False)
                np.save(scales, encoded.scales, allow_pickle=False)
                routed_row_by_name[row["name"]] = {
                        **row,
                        "wire": {
                            "geometry": QTIP2_GEOMETRY.as_mapping(),
                            "code_bpw": encoded.code_bpw,
                            "trellis": {
                                "path": trellis.relative_to(staging).as_posix(),
                                "bytes": trellis.stat().st_size,
                                "sha256": _sha256(trellis),
                            },
                            "scales": {
                                "path": scales.relative_to(staging).as_posix(),
                                "bytes": scales.stat().st_size,
                                "sha256": _sha256(scales),
                            },
                            "encoder": encoder,
                        },
                    }
                row_offset = row_end
        routed_rows = [routed_row_by_name[row["name"]] for row in selected_routed]
        for row in selected_native:
            native_base = spill_staging if split_native else staging
            assert native_base is not None
            member = native_base / "native" / _member_name(row["name"], "native.bin")
            digest = _copy_tensor_data_bytes(
                source=root / row["shard"],
                metadata=headers[row["shard"]][row["name"]],
                destination=member,
            )
            native_rows.append(
                {
                    **row,
                    "representation": "exact-source-data-bytes",
                    "storage_root": "native" if split_native else "primary",
                    "path": member.relative_to(native_base).as_posix(),
                    "source_sha256": digest,
                    "artifact_sha256": _sha256(member),
                }
            )
        receipt = {
            "schema": (
                HF_UNIFORM_ARTIFACT_SCHEMA
                if _routed_ordinal_range is None
                else HF_UNIFORM_SHARD_SCHEMA
            ),
            "status": "STAGED",
            "reload_verified": False,
            "api": {
                "method": (
                    "build_hf_moe_uniform"
                    if _routed_ordinal_range is None
                    else "build_hf_moe_uniform_shard"
                ),
                "version": 1,
            },
            "source": plan["source"],
            "intent": plan["intent"],
            "adapter": plan["adapter"],
            "geometry": plan["geometry"],
            "routed_tensors": routed_rows,
            "native_tensors": native_rows,
            "acceleration": {
                "routed_encode_batches": len(routed_batches),
                "routed_tensors_batched": len(selected_routed),
                "max_batch_tensors": max_batch_tensors,
                "same_width_batching": True,
            },
            "accounting": {
                "routed_tensor_count": len(routed_rows),
                "planned_routed_tensor_count": len(selected_routed),
                "native_tensor_count": len(native_rows),
                "planned_native_tensor_count": len(selected_native),
                "complete_routed_tensor_count": plan["accounting"]["routed_tensor_count"],
                "complete_native_tensor_count": plan["accounting"]["native_tensor_count"],
                "routed_parameters": sum(row["parameters"] for row in routed_rows),
                "native_parameters": sum(row["parameters"] for row in native_rows),
                "routed_source_bytes": sum(row["source_bytes"] for row in routed_rows),
                "native_source_bytes": sum(row["source_bytes"] for row in native_rows),
            },
            "coverage": plan["coverage"],
            "mechanisms": {
                "fallback": 0,
                "reconstruction": 0,
                "relay": 0,
                "streaming": 0,
            },
            "storage": {
                "mode": fit["storage_mode"],
                "primary_root": str(destination),
                "native_root": str(requested_spill) if split_native else str(destination),
                "fit": fit,
            },
        }
        if _routed_ordinal_range is not None:
            receipt["shard"] = {
                "routed_ordinals": [ordinal_start, ordinal_end],
                "complete_routed_tensor_count": total_routed,
                "native_policy": "complete-on-ordinal-zero-shard-v1",
            }
        _atomic_json(staging / "ARTIFACT.json", receipt)
        if split_native:
            assert spill_staging is not None
            os.replace(spill_staging, requested_spill)
            spill_promoted = True
        destination.parent.mkdir(parents=True, exist_ok=True)
        os.replace(staging, destination)
        _verify_hf_moe_members(destination, receipt)
        receipt["status"] = "PASS"
        receipt["reload_verified"] = True
        _atomic_json(destination / "ARTIFACT.json", receipt)
        return (
            open_hf_moe_uniform(destination)
            if _routed_ordinal_range is None
            else open_hf_moe_uniform_shard(destination)
        )
    finally:
        import shutil

        shutil.rmtree(staging, ignore_errors=True)
        if spill_staging is not None:
            shutil.rmtree(spill_staging, ignore_errors=True)
        if spill_promoted and not destination.exists():
            shutil.rmtree(requested_spill, ignore_errors=True)


def _verify_hf_moe_members(root: Path, receipt: Mapping[str, Any]) -> None:
    accounting = receipt.get("accounting")
    if not isinstance(accounting, Mapping):
        raise ValueError("HF MoE artifact lacks accounting")
    if (
        len(receipt["routed_tensors"]) != accounting.get("planned_routed_tensor_count")
        or len(receipt["native_tensors"]) != accounting.get("planned_native_tensor_count")
        or accounting.get("routed_tensor_count") != accounting.get("planned_routed_tensor_count")
        or accounting.get("native_tensor_count") != accounting.get("planned_native_tensor_count")
        or receipt.get("coverage") != {"duplicates": [], "gaps": []}
    ):
        raise ValueError("HF MoE artifact does not cover its complete planned inventory")
    for row in receipt["routed_tensors"]:
        for key in ("trellis", "scales"):
            member = root / row["wire"][key]["path"]
            if _sha256(member) != row["wire"][key]["sha256"]:
                raise ValueError(f"HF MoE routed member hash mismatch: {member}")
    for row in receipt["native_tensors"]:
        storage = receipt.get("storage", {})
        native_root = Path(storage.get("native_root", root))
        member_root = native_root if row.get("storage_root", "primary") == "native" else root
        member = member_root / row["path"]
        if _sha256(member) != row["artifact_sha256"] or row["artifact_sha256"] != row["source_sha256"]:
            raise ValueError(f"HF MoE native member hash mismatch: {member}")


def open_hf_moe_uniform(output: str | Path) -> dict[str, Any]:
    """Reload and hash-verify a serialized routed-Q2/native-rest artifact."""

    root = Path(output).expanduser().resolve()
    receipt = json.loads((root / "ARTIFACT.json").read_text(encoding="utf-8"))
    if (
        receipt.get("schema") != HF_UNIFORM_ARTIFACT_SCHEMA
        or receipt.get("status") != "PASS"
        or receipt.get("reload_verified") is not True
    ):
        raise ValueError("HF MoE artifact receipt is not an admitted reloaded PASS")
    _verify_hf_moe_members(root, receipt)
    return {**receipt, "artifact_root": str(root)}


def build_hf_moe_uniform_shard(
    model: str | Path,
    *,
    revision: str,
    tier: str,
    scope: str,
    native_rest: bool,
    routed_ordinal_start: int,
    routed_ordinal_end: int,
    output: str | Path,
    native_spill_root: str | Path | None = None,
) -> dict[str, Any]:
    """Build one canonical half-open routed-tensor ordinal range."""

    return build_hf_moe_uniform(
        model,
        revision=revision,
        tier=tier,
        scope=scope,
        native_rest=native_rest,
        output=output,
        native_spill_root=native_spill_root,
        _routed_ordinal_range=(routed_ordinal_start, routed_ordinal_end),
    )


def open_hf_moe_uniform_shard(output: str | Path) -> dict[str, Any]:
    """Reload and hash-verify one partial horizontal build shard."""

    root = Path(output).expanduser().resolve()
    receipt = json.loads((root / "ARTIFACT.json").read_text(encoding="utf-8"))
    if (
        receipt.get("schema") != HF_UNIFORM_SHARD_SCHEMA
        or receipt.get("status") != "PASS"
        or receipt.get("reload_verified") is not True
        or not isinstance(receipt.get("shard"), Mapping)
    ):
        raise ValueError("HF MoE shard receipt is not an admitted reloaded PASS")
    _verify_hf_moe_members(root, receipt)
    return {**receipt, "artifact_root": str(root)}


def union_hf_moe_uniform_shards(
    shards: Sequence[str | Path], *, output: str | Path
) -> dict[str, Any]:
    """Deterministically union a disjoint, gap-free routed shard set."""

    import shutil

    opened = [open_hf_moe_uniform_shard(path) for path in shards]
    if not opened:
        raise ValueError("HF MoE shard union requires at least one shard")
    ordered = sorted(opened, key=lambda row: row["shard"]["routed_ordinals"])
    ranges = [row["shard"]["routed_ordinals"] for row in ordered]
    for previous, current in zip(ranges, ranges[1:]):
        if current[0] < previous[1]:
            raise ValueError("HF MoE shard ranges must be disjoint")
    total = ordered[0]["shard"]["complete_routed_tensor_count"]
    if ranges[0][0] != 0 or ranges[-1][1] != total or any(
        left[1] != right[0] for left, right in zip(ranges, ranges[1:])
    ):
        raise ValueError("HF MoE shard ranges must be gap-free and cover the complete plan")
    identity_keys = ("source", "intent", "adapter", "geometry")
    if any(
        row["shard"]["complete_routed_tensor_count"] != total
        or any(row.get(key) != ordered[0].get(key) for key in identity_keys)
        for row in ordered[1:]
    ):
        raise ValueError("HF MoE shard source/plan identity mismatch")
    routed = sorted(
        [member for row in ordered for member in row["routed_tensors"]],
        key=lambda member: member["name"],
    )
    native = [member for row in ordered for member in row["native_tensors"]]
    complete_native = ordered[0]["accounting"]["complete_native_tensor_count"]
    if len(routed) != total or len({row["name"] for row in routed}) != total:
        raise ValueError("HF MoE shard union routed inventory is incomplete or duplicated")
    if len(native) != complete_native or len({row["name"] for row in native}) != complete_native:
        raise ValueError("HF MoE shard union native inventory is incomplete or duplicated")
    destination = Path(output).expanduser().resolve()
    if destination.exists():
        raise FileExistsError(f"HF MoE artifact output already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent))
    try:
        owner_by_name = {
            member["name"]: Path(row["artifact_root"])
            for row in ordered
            for member in [*row["routed_tensors"], *row["native_tensors"]]
        }
        for member in routed:
            owner = owner_by_name[member["name"]]
            for kind in ("trellis", "scales"):
                relative = member["wire"][kind]["path"]
                target = staging / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(owner / relative, target)
        for member in native:
            owner = owner_by_name[member["name"]]
            source = owner / member["path"]
            target = staging / member["path"]
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
        member_hashes = [
            member["wire"][kind]["sha256"]
            for member in routed
            for kind in ("trellis", "scales")
        ] + [member["artifact_sha256"] for member in native]
        first = ordered[0]
        receipt = {
            "schema": HF_UNIFORM_ARTIFACT_SCHEMA,
            "status": "PASS",
            "reload_verified": True,
            "api": {"method": "union_hf_moe_uniform_shards", "version": 1},
            **{key: first[key] for key in identity_keys},
            "routed_tensors": routed,
            "native_tensors": native,
            "acceleration": {
                "routed_encode_batches": sum(
                    int(row["acceleration"]["routed_encode_batches"]) for row in ordered
                ),
                "routed_tensors_batched": sum(
                    int(row["acceleration"]["routed_tensors_batched"]) for row in ordered
                ),
                "max_batch_tensors": min(
                    int(row["acceleration"]["max_batch_tensors"]) for row in ordered
                ),
                "same_width_batching": all(
                    row["acceleration"]["same_width_batching"] is True for row in ordered
                ),
            },
            "accounting": {
                "routed_tensor_count": len(routed),
                "planned_routed_tensor_count": total,
                "native_tensor_count": len(native),
                "planned_native_tensor_count": complete_native,
                "complete_routed_tensor_count": total,
                "complete_native_tensor_count": complete_native,
                "routed_parameters": sum(row["parameters"] for row in routed),
                "native_parameters": sum(row["parameters"] for row in native),
                "routed_source_bytes": sum(row["source_bytes"] for row in routed),
                "native_source_bytes": sum(row["source_bytes"] for row in native),
            },
            "coverage": {"duplicates": [], "gaps": []},
            "mechanisms": {"fallback": 0, "reconstruction": 0, "relay": 0, "streaming": 0},
            "storage": {"mode": "primary-only-v1", "primary_root": str(destination), "native_root": str(destination)},
            "union": {
                "input_ranges": ranges,
                "ordered_member_sha256": hashlib.sha256("".join(member_hashes).encode()).hexdigest(),
            },
        }
        _atomic_json(staging / "ARTIFACT.json", receipt)
        _verify_hf_moe_members(staging, receipt)
        os.replace(staging, destination)
        return open_hf_moe_uniform(destination)
    finally:
        shutil.rmtree(staging, ignore_errors=True)
