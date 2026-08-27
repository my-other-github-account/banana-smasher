"""Fully resident official-K2 Balanced64 scoring backend.

The production entry point is ``ResidentRepairAPI.score(checkpoint, windows)``.
This module loads one checkpoint and one local model/payload closure per rank,
then executes every requested window without opening model, payload, teacher, or
corpus files after the ``resident_ready`` boundary.
"""
from __future__ import annotations

from collections.abc import MutableMapping
from dataclasses import replace
from datetime import timedelta
import copy
import ctypes
import gc
import hashlib
import importlib.util
import json
import math
import os
import pickle
from pathlib import Path
import subprocess
import sys
import time
from types import MappingProxyType
from typing import Any, Callable, Iterable, Mapping, cast

from .balanced64 import (
    ArtifactError,
    BALANCED64_SPEC,
    POSITIONS_PER_WINDOW,
    SUPPORT,
    RepairArtifact,
    ScoreResult,
    _load_torch,
)

BASIS_SHA256 = "98efab455cf08dfbbbaaba6f570e1bf10bf927d2b4c3c453a59c2f6f0e3be92b"
# Canonical grouped implementation aligned to the sealed DeepseekV4 expert
# boundary: FP32 projection/clamp arithmetic with one final output cast. Keep
# the executing asset hash explicit and synchronized with the pinned source.
OFFICIAL_PHYSICAL_LAYER_SHA256 = "791d90cb43b068f5f58f1e3049b434ffe8f235af8c8279670b7a5fff047298ba"
LP4_PACK_SOURCE_SHA256 = "7a8e48547824a87a48db4c7142ec53f73303a91ce6a0c95cf1a88b1b87d22350"
LP4_TRAIN_SOURCE_SHA256 = "10abc4b04a9bc88bf348cd121d3d072456a54de1cd801a1425edc15b104e4523"
T8192_BUILDER_SOURCE_SHA256 = "ed6a1d0f0666027372a726ea96d7d6f7c3487b60da8c5d8f8be591330ccb7137"
CANONICAL_U0_CHECKPOINT_SHA256 = "7978d1002d7e4ecfa280f646f70cc76638c0e7bd833cc3cc13a2de999050133f"
CANONICAL_U0_IDENTITY_SHA256 = "d602de92d998c0e649b0bc4fdf35a857384ff3cf6d1021bdbb76a8070af73a88"
CANONICAL_U0_LOCK_SHA256 = "7eb5edeb8583abba450a6f94de3cfe4fee0ab053c962bfcc1d035bd2d0c30fc2"
CANONICAL_U0_TRAJECTORY_SHA256 = "ddcdebacefb002c39e8c0c66636cfbc24f13997b8a56d02d1023e15ce42cd9cf"
CANONICAL_U0_SOURCE_PATH = "/home/dnola/missions/MODERN_GREEN_t_6bc398da/run_clean_u0_attempt4/checkpoints/UPDATE_000.pt"
BUILDER_EVAL_CORPUS_SHA256 = "5aadaacbb486ae4f528c5e51ae70beff863337bd908fc727e6e49fc3ac520ebd"
SCORE_TRAIN_CORPUS_SHA256 = "16575db7fd180ca193aa13c4e642400b9ed416dbd0c36c3c5302422b31f5cbae"
TEACHER_INVENTORY_SHA256 = "017c7e9261b3e3701bd2f2dd53a03e46466b1dd2a3c5b4ecfb55b4c0aad04a92"
# The immutable clean-U0 lock predates the explicit builder/score split. Its
# trajectory envelope remains bound to this historical corpus identity while
# scoring itself is unconditionally routed to SCORE_TRAIN_CORPUS_SHA256.
CANONICAL_U0_LOCK_CORPUS_SHA256 = "434a3f9eec14e54d348efde3265998c9521bb3579cba0d976b3e0a9b93d184c5"
# Compatibility alias for older callers. Canonical admission below binds each
# distinct corpus role explicitly and never compares both roles to this alias.
CANONICAL_CORPUS_SHA256 = SCORE_TRAIN_CORPUS_SHA256
CANONICAL_U1_CHECKPOINT_SHA256 = "65cedb1c46f7bb57ad42dcc44686d00410f955f4e1b8e38a18a4520c68b3b865"
CANONICAL_U1_IDENTITY_SHA256 = "a7ea864bb810af15cf75a03935b335020b5394f7637974e777c96319683b5ecd"
# This exact predecessor differs only by the U1 raw-identity adapter.  Its U0
# binary64 resume rows therefore remain byte-for-byte valid after that repair.
U0_RESUME_COMPATIBLE_IMPLEMENTATION_SHA256 = "ba94e819badadeace56ff0c48b780a1f4129f0d58daffdd2759de1d25bd98236"
# This serialized PRE is retained solely for explicit quarantine tests and
# historical evidence; it is never a production admission prerequisite.
ALTERNATE_PRE_CHECKPOINT_SHA256 = "f9bffe04c6e1ee03ea2eefe838f68ed773179e05363d08ac509602cb740f9f70"
PRE_CHECKPOINT_SHA256 = CANONICAL_U0_CHECKPOINT_SHA256
PUBLIC_API_METHOD = "ResidentRepairAPI.score"
PUBLIC_API_VERSION = "official-k2-resident-v2"
ROUTED_K2_API_METHOD = "ResidentRepairAPI.score_routed_k2"
ROUTED_K2_API_VERSION = "official-k2-routed-resident-v1"


_ORDINARY_FORK_PAYLOADS: dict[str, dict[str, Any]] = {}


def _freeze_checkpoint_mappings(value: Any) -> Any:
    """Make inherited checkpoint containers rank-local read-only views."""
    if isinstance(value, Mapping):
        return MappingProxyType({
            key: _freeze_checkpoint_mappings(child) for key, child in value.items()
        })
    return value


def _serialize_resident_storage_ipc(payload: Mapping[str, Any]) -> bytes:
    """Serialize only tensor IPC descriptors, never packed wire payload bytes."""
    from multiprocessing.reduction import ForkingPickler

    encoded = bytes(ForkingPickler.dumps(dict(payload)))
    if len(encoded) > 1_048_576:
        raise ArtifactError("resident storage IPC descriptor unexpectedly contains payload bytes")
    return encoded


def _deserialize_resident_storage_ipc(payload: bytes) -> Mapping[str, Any]:
    value = pickle.loads(payload)
    if not isinstance(value, Mapping):
        raise ArtifactError("resident storage IPC descriptor is not a mapping")
    return value


def _unique_tensor_storage_bytes(value: Any) -> int:
    """Count unique tensor storages reachable through a broker descriptor."""
    seen: set[tuple[str, int]] = set()
    total = 0

    def visit(current: Any) -> None:
        nonlocal total
        if isinstance(current, Mapping):
            for child in current.values():
                visit(child)
            return
        if isinstance(current, (tuple, list)):
            for child in current:
                visit(child)
            return
        storage_method = getattr(current, "untyped_storage", None)
        if not callable(storage_method):
            return
        storage: Any = storage_method()
        key = (str(getattr(current, "device", "unknown")), int(storage.data_ptr()))
        if key not in seen:
            seen.add(key)
            total += int(storage.nbytes())

    visit(value)
    return total


def _install_ordinary_fork_payload(path: Path, expected_sha256: str) -> Mapping[str, Any]:
    """Materialize one hash-bound ordinary payload in the pre-fork broker."""
    if len(expected_sha256) != 64 or any(
        character not in "0123456789abcdef" for character in expected_sha256
    ):
        raise ArtifactError("checkpoint source SHA is malformed")
    resolved = path.expanduser().resolve(strict=True)
    if not resolved.is_file() or resolved.is_symlink():
        raise ArtifactError(f"checkpoint source must be a regular file: {resolved}")
    observed_sha256 = _sha256_file(resolved)
    if observed_sha256 != expected_sha256:
        raise ArtifactError(
            f"checkpoint source SHA mismatch: {resolved}: {observed_sha256} != {expected_sha256}"
        )
    existing = _ORDINARY_FORK_PAYLOADS.get(expected_sha256)
    if existing is None:
        payload = _load_torch(resolved)
        if not isinstance(payload, Mapping):
            raise ArtifactError(f"{resolved} must contain a mapping")
        existing = {
            "payload": _freeze_checkpoint_mappings(payload),
            "materialization_pid": os.getpid(),
            "source_path": str(resolved),
            "checkpoint_sha256": expected_sha256,
        }
        _ORDINARY_FORK_PAYLOADS[expected_sha256] = existing
    return MappingProxyType({
        key: value for key, value in existing.items() if key != "payload"
    })


def _load_hash_bound_torch_mmap(path: Path, expected_sha256: str) -> Mapping[str, Any]:
    """Load one immutable checkpoint through a hash-bound private mmap.

    ``MAP_PRIVATE`` makes the checkpoint file a read-only backing source: tensor
    writes, if any legacy adapter performs them, are copy-on-write and cannot
    alter the sealed bytes shared by the staggered peer rank.
    """
    if len(expected_sha256) != 64 or any(
        character not in "0123456789abcdef" for character in expected_sha256
    ):
        raise ArtifactError("checkpoint mmap source SHA is malformed")
    resolved = path.expanduser().resolve(strict=True)
    if not resolved.is_file() or resolved.is_symlink():
        raise ArtifactError(f"checkpoint mmap source must be a regular file: {resolved}")
    observed_sha256 = _sha256_file(resolved)
    if observed_sha256 != expected_sha256:
        raise ArtifactError(
            "checkpoint mmap source SHA mismatch: "
            f"{resolved}: {observed_sha256} != {expected_sha256}"
        )
    try:
        import mmap
        import torch
    except ImportError as exc:  # pragma: no cover - exercised on runtime hosts
        raise ArtifactError("torch is required to score .pt candidate artifacts") from exc
    with torch.serialization.set_default_mmap_options(mmap.MAP_PRIVATE):
        value = torch.load(resolved, map_location="cpu", mmap=True, weights_only=True)
    if not isinstance(value, Mapping):
        raise ArtifactError(f"{resolved} must contain a mapping")
    return value


def _load_score_checkpoint(
    path: Path, expected_sha256: str, config: Mapping[str, Any]
) -> Mapping[str, Any]:
    """Select the hash-bound loader explicitly for candidate isolation."""
    if bool(config.get("ordinary_load_fork_broker", False)):
        if bool(config.get("checkpoint_mmap", True)):
            raise ArtifactError("ordinary-load fork broker requires checkpoint_mmap=false")
        resolved = path.expanduser().resolve(strict=True)
        observed_sha256 = _sha256_file(resolved)
        if observed_sha256 != expected_sha256:
            raise ArtifactError(
                f"checkpoint source SHA mismatch: {resolved}: {observed_sha256} != {expected_sha256}"
            )
        inherited = _ORDINARY_FORK_PAYLOADS.get(expected_sha256)
        if inherited is None:
            raise ArtifactError("ordinary-load fork broker payload was not inherited")
        if (
            int(inherited["materialization_pid"]) == os.getpid()
            and not bool(config.get("same_process_dual_shard", False))
        ):
            raise ArtifactError("ordinary-load fork broker payload must be consumed by a fork child")
        return cast(Mapping[str, Any], inherited["payload"])
    if bool(config.get("checkpoint_mmap", True)):
        return _load_hash_bound_torch_mmap(path, expected_sha256)
    observed_sha256 = _sha256_file(path)
    if observed_sha256 != expected_sha256:
        raise ArtifactError(
            f"checkpoint source SHA mismatch: {path}: {observed_sha256} != {expected_sha256}"
        )
    value = _load_torch(path)
    if not isinstance(value, Mapping):
        raise ArtifactError(f"{path} must contain a mapping")
    return value


def _release_or_retain_checkpoint_payload(
    payload: Mapping[str, Any], *, ordinary_load_fork_broker: bool,
    checkpoint_sha256: str | None = None,
) -> str:
    """Retain shared broker pages; clear only a process-owned checkpoint mapping."""
    if ordinary_load_fork_broker:
        if checkpoint_sha256:
            inherited = _ORDINARY_FORK_PAYLOADS.get(checkpoint_sha256)
            registered = inherited.get("payload") if inherited is not None else None
            # Canonical raw checkpoints receive an in-memory identity envelope
            # through a shallow outer mapping copy.  The copy is still bound to
            # the broker only when every non-envelope value is the exact frozen
            # object registered by the hash-bound ordinary load.  This admits
            # the canonical adapter without accepting a copied/replaced state.
            broker_bound = registered is payload
            if isinstance(registered, Mapping) and isinstance(payload, Mapping):
                registered_keys = set(registered) - {"identity"}
                payload_keys = set(payload) - {"identity"}
                broker_bound = broker_bound or (
                    registered_keys == payload_keys
                    and all(payload[key] is registered[key] for key in registered_keys)
                )
            if not broker_bound:
                raise ArtifactError("ordinary-load fork broker payload identity mismatch")
        elif not isinstance(payload, MappingProxyType):
            raise ArtifactError("ordinary-load fork broker payload must be read-only")
        return "inherited_read_only"
    if not isinstance(payload, MutableMapping):
        raise ArtifactError("official-K2 checkpoint payload must support ownership transfer")
    payload.clear()
    return "cleared_child_owned"


def _write_q_lp_capture(path: str | Path, array: Any) -> dict[str, Any]:
    """Immutably publish the complete diagnostic q_lp_at_ref tensor."""
    target = Path(path)
    if target.exists():
        raise ArtifactError(f"q_lp_at_ref capture already exists: {target}")
    if tuple(array.shape) != (POSITIONS_PER_WINDOW, SUPPORT) or str(array.dtype) != "float16":
        raise ArtifactError("q_lp_at_ref capture requires float16[1024,8192]")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.tmp.{os.getpid()}")
    try:
        with temporary.open("xb") as stream:
            __import__("numpy").save(stream, array, allow_pickle=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, target)
        temporary.unlink()
        os.chmod(target, 0o444)
        descriptor = os.open(target, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        directory = os.open(target.parent, os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if temporary.exists():
            temporary.unlink()
    raw = target.read_bytes()
    return {
        "path": str(target),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "bytes": len(raw),
        "dtype": "float16",
        "shape": [POSITIONS_PER_WINDOW, SUPPORT],
    }
ROUTED_K2_ROUTE_KIND = "routed_k2_official_k2_l034_selected_wire_v1"
# Immutable closure identity from t_36e9ce8e.  Alternate PRE is admitted only
# through the explicitly routed specialization, never through score().
ROUTED_K2_CLOSURE = {
    "manifest_sha256": "51f62c7c63d49da6115e139dda37728cc75215abcd02360bbe5b2cb5e4341f49",
    "package_root": "/home/dnola/missions/ROUTED_K2_VALIDATION_PACKAGE_t_36e9ce8e_L034_SELECTED_WIRE_v1",
    "package_identity_sha256": "3ea2d829b003c0d8c3fcae300e0ac70447d34e7dc784c378cd12900597605ffc",
    "selected_binding_sha256": "418c1cd803413fb0cfad3ae93eae6ac93095de00e106114a64c6b5f7983286a5",
    "selected_roster_sha256": "cea2d8aa9cf8ba8dde0d4b699acc24295a03d0ab0dddae1950e20f4b0e8e269e",
    "official_source_package": "/home/dnola/missions/ROUTED_K2_VALIDATION_PACKAGE_t_cf7ed633_v5",
    # The routed published-PRE package has its own immutable physical class.
    # Keep it distinct from the later canonical raw-U0 executing source.
    "official_class_sha256": "7687e39fc5b6bb34b30e8d4a79771affb472497f4d2f323adbe1e8e277746729",
    "basis_model_index_sha256": BASIS_SHA256,
    "pre_checkpoint_sha256": ALTERNATE_PRE_CHECKPOINT_SHA256,
    "post_checkpoint_sha256": "23cbc9f5fcc79117f70a45c404ee15a576cf4ad62f8a6332107f3c2ca85a5819",
    "windows_count": 64,
    "positions_per_window": 1024,
    "support": 8192,
}
CANONICAL_CALIBRATION_SCHEMA = "official-k2-resident-canonical-calibration-v1"
PRE_KLD = 0.22939197531977115
PRE_TOP1 = 56533
PRE_KLD_ABS_TOLERANCE = 1.0e-15
EXPECTED_RESIDENT_BYTES = {0: 41_201_023_756, 1: 43_146_481_404}
DEFAULT_CUDA_RESERVE_BYTES = 4 << 30
# Keep the historical scalar name as a compatibility alias for callers that
# import it, while public manifests should use an explicit rank-local mapping.
CUDA_RESERVE_BYTES = DEFAULT_CUDA_RESERVE_BYTES


def _configured_expert_source_sha256(config: Mapping[str, Any]) -> str:
    value = str(config.get("official_expert_source_sha256", OFFICIAL_PHYSICAL_LAYER_SHA256))
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ArtifactError("official-K2 resident expert source SHA configuration is malformed")
    return value


def _configured_attention_implementation(config: Mapping[str, Any]) -> str:
    # The sealed builder constructs DeepseekV4 with attn_implementation="eager".
    # Resident scoring must preserve that public reference unless an artifact
    # explicitly binds another admitted implementation.
    value = str(config.get(
        "attention_implementation_override",
        config.get("attention_implementation", "eager"),
    ))
    if value not in ("eager", "sdpa"):
        raise ArtifactError("official resident attention implementation must be eager or sdpa")
    return value


def resolve_rank_local_bytes(
    value: Any,
    rank: int,
    *,
    default: int,
    field: str,
) -> tuple[int, str]:
    """Resolve a positive byte policy without silently disabling the reserve."""
    policy = "scalar_default"
    selected: Any = default if value is None else value
    if isinstance(selected, Mapping):
        policy = "rank_local_explicit"
        if rank in selected:
            selected = selected[rank]
        elif str(rank) in selected:
            selected = selected[str(rank)]
        else:
            raise ArtifactError(f"{field} rank-local policy is missing rank{rank}")
    try:
        selected = int(selected)
    except (TypeError, ValueError) as exc:
        raise ArtifactError(f"{field} must be a positive integer or rank-local mapping") from exc
    if selected <= 0:
        raise ArtifactError(f"{field} must be positive; zero disables the reserve")
    return selected, policy


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _prune_loaded_parent_members(*, parent_root: Path, sources: Mapping[int, Any]) -> tuple[int, int]:
    """Unlink task-local cold-load members only after they are fully resident.

    This is a storage-lifetime optimization, not layer streaming: scoring retains
    every grouped expert in memory and performs zero payload reads after the
    resident-ready boundary.  The fail-closed root check prevents a manifest
    from turning this opt-in cleanup into deletion outside the staged parent.
    """
    root = parent_root.resolve(strict=True)
    files: set[Path] = set()
    for source in sources.values():
        if int(source.layer) == 34:
            continue
        for member in source.member_paths.values():
            path = Path(member).resolve(strict=True)
            try:
                path.relative_to(root)
            except ValueError as exc:
                raise ArtifactError(f"resident parent member escapes staged root: {path}") from exc
            if path.is_symlink() or not path.is_file():
                raise ArtifactError(f"resident parent member is not a regular local file: {path}")
            files.add(path)
    byte_count = sum(path.stat().st_size for path in files)
    for path in sorted(files):
        path.unlink()
    return len(files), byte_count


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    payload = (json.dumps(dict(value), indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    directory_fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
    if path.read_bytes() != payload:
        raise ArtifactError(f"resident receipt readback mismatch: {path}")


def _aggregate_score_phase_profiles(
    rank_profiles: Iterable[Iterable[Mapping[str, Any]]],
    *,
    ordered_windows: tuple[int, ...],
    post_load_wall_seconds: float,
    configured_batch_size: int,
) -> dict[str, Any]:
    """Fan in exact per-batch phase telemetry and evaluate the 300s gate."""
    profiles = [list(rows) for rows in rank_profiles]
    if len(profiles) != 2 or any(
        not isinstance(row, Mapping) for rows in profiles for row in rows
    ):
        raise ArtifactError("official-K2 score phase profile rank fan-in drift")
    rank0, rank1 = profiles
    rank0_windows = [int(window) for row in rank0 for window in row.get("batch_windows", [])]
    rank1_windows = [int(window) for row in rank1 for window in row.get("batch_windows", [])]
    expected = list(ordered_windows)
    if rank0_windows != expected or rank1_windows != expected or len(rank0) != len(rank1):
        raise ArtifactError(
            "official-K2 score phase profile window coverage drift: "
            f"rank0={rank0_windows} rank1={rank1_windows} expected={expected}"
        )
    paired: list[dict[str, Any]] = []
    for ordinal, (left, right) in enumerate(zip(rank0, rank1)):
        left_windows = [int(value) for value in left.get("batch_windows", [])]
        right_windows = [int(value) for value in right.get("batch_windows", [])]
        if left_windows != right_windows:
            raise ArtifactError("official-K2 score phase profile batch alignment drift")
        paired.append({
            "batch_ordinal": ordinal,
            "batch_windows": left_windows,
            "rank0": dict(left),
            "rank1": dict(right),
        })

    def total(rows: list[Mapping[str, Any]], field: str) -> float:
        return sum(float(row.get(field, 0.0)) for row in rows)

    phases = {
        "rank0_embedding": total(rank0, "embedding_ms"),
        "rank0_layer_forward": total(rank0, "layer_forward_ms"),
        "rank0_consumer_wait": total(rank0, "consumer_wait_ms"),
        "rank1_activation_wait": total(rank1, "activation_wait_ms"),
        "rank1_layer_forward": total(rank1, "layer_forward_ms"),
        "rank1_readout": total(rank1, "readout_ms"),
        "rank1_logits": total(rank1, "logits_ms"),
        "rank1_teacher_gather": total(rank1, "teacher_gather_ms"),
        "rank1_binary64_reduce": total(rank1, "binary64_reduce_ms"),
        "rank1_glue": total(rank1, "glue_ms"),
    }
    wall = float(post_load_wall_seconds)
    passed = wall < 300.0 and len(expected) == 64
    return {
        "schema": "official-k2-resident-score-phase-profile-v1",
        "status": "PASS" if passed else "PROFILE_ONLY",
        "window_count": len(expected),
        "configured_batch_size": int(configured_batch_size),
        "batch_count": len(paired),
        "post_load_wall_seconds": wall,
        "post_load_threshold_seconds": 300.0,
        "post_load_under_300_seconds": wall < 300.0,
        "full64_gate_pass": passed,
        "phase_milliseconds": phases,
        "per_batch": paired,
    }


def _trim_host_allocator() -> bool:
    """Return freed checkpoint arenas to Linux after forked materialization."""
    try:
        trim = ctypes.CDLL(None).malloc_trim
    except AttributeError:
        return False
    trim.argtypes = [ctypes.c_size_t]
    trim.restype = ctypes.c_int
    return bool(trim(0))


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError) as exc:
        raise ArtifactError(f"cannot read resident JSON input {path}: {exc}") from exc


def _rebase_admission_lut_sources(admission: Mapping[str, Any], root: str | Path) -> dict[str, Any]:
    """Rebind reboot-volatile LUT provenance paths to an exact durable mirror.

    The admission document remains the scientific authority for every expected
    digest.  Only its absolute source paths are replaced, and the replacement
    bytes must match before model construction begins.
    """
    rebound = copy.deepcopy(dict(admission))
    try:
        rows = rebound["trainable_roster"]["luts"]
    except (KeyError, TypeError) as exc:
        raise ArtifactError("official-K2 admission is missing the LUT roster") from exc
    if not isinstance(rows, list) or not rows:
        raise ArtifactError("official-K2 admission LUT roster is empty")
    durable_root = Path(root).expanduser().resolve()
    seen: set[int] = set()
    for row in rows:
        try:
            layer = int(row["layer"])
            manifest_sha = str(row["source_manifest"]["sha256"])
            wire_sha = str(row["wire"]["sha256"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ArtifactError("official-K2 admission LUT row is malformed") from exc
        if layer in seen or not 0 <= layer <= 42:
            raise ArtifactError("official-K2 admission LUT layer roster is invalid")
        seen.add(layer)
        parent = durable_root / f"L{layer:03d}" / "parent"
        replacements = (
            (row["source_manifest"], parent / "QTIP_V7_MANIFEST.json", manifest_sha),
            (row["wire"], parent / f"L{layer:03d}.tlut.f16", wire_sha),
        )
        for target, path, expected in replacements:
            if not path.is_file():
                raise ArtifactError(f"official-K2 durable LUT source is missing: {path}")
            observed = _sha256_file(path)
            if observed != expected:
                raise ArtifactError(
                    f"official-K2 durable LUT source SHA mismatch: {path}: {observed} != {expected}"
                )
            target["path" if "path" in target else "source_path"] = str(path)
    return rebound


def _drop_cold_file_cache(paths: Iterable[Path]) -> tuple[int, int]:
    """Evict reproducible source pages without deleting immutable inputs."""
    if not hasattr(os, "posix_fadvise") or not hasattr(os, "POSIX_FADV_DONTNEED"):
        raise ArtifactError("official-K2 cold-source cache eviction is unavailable")
    file_count = 0
    byte_count = 0
    for path in paths:
        size = path.stat().st_size
        fd = os.open(path, os.O_RDONLY)
        try:
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        finally:
            os.close(fd)
        file_count += 1
        byte_count += size
    return file_count, byte_count


def _drop_cold_model_cache(model_root: Path) -> tuple[int, int]:
    """Evict loaded model-shard pages before allocating score inputs."""
    root = model_root.resolve(strict=True)
    files = sorted(path for path in root.glob("*.safetensors") if path.is_file())
    return _drop_cold_file_cache(files)


def _windows_sha256(windows: Iterable[int]) -> str:
    payload = json.dumps([int(value) for value in windows], separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _validate_public_score_windows(selected: tuple[int, ...], balanced64: tuple[int, ...]) -> str:
    """Admit the exact W28 physical canary or the production Balanced64 geometry."""
    if selected == (28,):
        return "W28_CANARY"
    if len(selected) == 64 and selected == balanced64:
        return "FULL64"
    raise ArtifactError(
        "official-K2 public score requires exact W28 canary or 64 ordered Balanced64 windows"
    )


def _effective_score_window_batch_size(configured: int, window_count: int) -> int:
    """Retain the full-rail batch shape while permitting the one-window W28 canary."""
    if configured < 1 or window_count < 1:
        raise ArtifactError("official-K2 resident score window counts must be positive")
    return min(configured, window_count)


def _sealed_pair_groups(
    windows: Iterable[int], concurrency: int
) -> tuple[tuple[tuple[int, int], ...], ...]:
    """Group the immutable sealed mb=2 walk without changing any pair."""
    ordered = tuple(int(window) for window in windows)
    if concurrency < 1:
        raise ArtifactError("official-K2 sealed pair concurrency must be positive")
    if len(ordered) % 2:
        raise ArtifactError("official-K2 sealed pair roster must contain whole pairs")
    pairs = tuple(
        (ordered[index], ordered[index + 1])
        for index in range(0, len(ordered), 2)
    )
    return tuple(
        pairs[index : index + concurrency]
        for index in range(0, len(pairs), concurrency)
    )


def _physical_canary_batch_windows(
    selected: tuple[int, ...], configured: int, balanced64: tuple[int, ...]
) -> tuple[int, ...]:
    """Execute W28 in its aligned full-rail batch while reporting only W28."""
    if selected != (28,) or configured == 1:
        return selected
    if configured < 1 or 28 not in balanced64:
        raise ArtifactError("official-K2 W28 physical batch configuration is invalid")
    offset = balanced64.index(28)
    start = offset - (offset % configured)
    physical = balanced64[start : start + configured]
    if len(physical) != configured or 28 not in physical:
        raise ArtifactError("official-K2 W28 canary requires a complete aligned batch")
    return physical


SOURCE_CONTEXT_TOKENS = 2048
QSFP_INTERFACE = "enp1s0f1np1"


def _validate_qsfp_pin(config: Mapping[str, Any], *, rank: int) -> dict[str, str]:
    """Fail closed unless the two-rank route is pinned to the 200G fabric."""
    mapping = config.get("qsfp_host_ip_by_rank")
    if not isinstance(mapping, Mapping):
        raise ArtifactError("official-K2 resident QSFP rank map is required")
    try:
        rank0_value = mapping[0] if 0 in mapping else mapping["0"]
        local_value = mapping[rank] if rank in mapping else mapping[str(rank)]
        rank0 = str(rank0_value)
        local = str(local_value)
    except (KeyError, TypeError) as exc:
        raise ArtifactError(f"official-K2 resident QSFP rank map is missing rank{rank}") from exc
    if not rank0.startswith("192.168.200.") or not local.startswith("192.168.200."):
        raise ArtifactError("official-K2 resident QSFP addresses must use 192.168.200.0/24")
    if str(config.get("master_addr", "")) != rank0:
        raise ArtifactError("official-K2 resident QSFP master_addr must equal rank0 QSFP address")
    interface = str(config.get("distributed_socket_interface", ""))
    if interface != QSFP_INTERFACE:
        raise ArtifactError(f"official-K2 resident QSFP interface must be {QSFP_INTERFACE}")
    return {"master_addr": rank0, "local_qsfp_ip": local, "interface": interface}


def _canonical_causal_score_tokens(tokens: Any, *, real_len: int, pad_token_id: int) -> list[int]:
    """Materialize the frozen 2,048-token source context.

    Balanced64 reduces only positions 0..1,023, but model execution retains the
    canonical 2,048-position context. Short source rows are padded explicitly;
    they are never silently shortened to the scored prefix.
    """
    if not isinstance(tokens, list):
        raise ArtifactError("official-K2 resident corpus tokens must be a list")
    real = int(real_len)
    if real < POSITIONS_PER_WINDOW or len(tokens) < POSITIONS_PER_WINDOW:
        raise ArtifactError(
            f"official-K2 resident corpus has fewer than {POSITIONS_PER_WINDOW} real tokens"
        )
    if real > SOURCE_CONTEXT_TOKENS:
        raise ArtifactError("official-K2 resident corpus exceeds the 2048-token source context")
    if len(tokens) < real:
        raise ArtifactError("official-K2 resident corpus token_ids are shorter than real_len")
    selected = [int(token) for token in tokens[:real]]
    selected.extend([int(pad_token_id)] * (SOURCE_CONTEXT_TOKENS - real))
    return selected


def _nested_field(value: Any, names: tuple[str, ...]) -> Any:
    """Return the first named field from a lock's shallow nested mappings."""
    if not isinstance(value, Mapping):
        return None
    for name in names:
        if name in value:
            return value[name]
    for child in value.values():
        if isinstance(child, Mapping):
            found = _nested_field(child, names)
            if found is not None:
                return found
    return None


def _normalized_hash_field(value: Any) -> Any:
    if isinstance(value, Mapping):
        return _nested_field(value, ("sha256", "identity_sha256", "hash"))
    return value


def _relative_manifest_path(root: Path, value: Any, field: str) -> Path:
    if not isinstance(value, str) or not value or Path(value).is_absolute():
        raise ArtifactError(f"canonical raw U0 {field} must be a non-empty relative path")
    path = (root / value).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as exc:
        raise ArtifactError(f"canonical raw U0 {field} escapes artifact root") from exc
    return path


def _require_equal(observed: Any, expected: Any, field: str) -> None:
    if observed != expected:
        raise ArtifactError(f"canonical raw U0 {field} drift: {observed!r} != {expected!r}")


def _validate_raw_u0_gates(payload: Mapping[str, Any], lock: Mapping[str, Any]) -> None:
    """Check clean-U0 gates without requiring metadata in the raw checkpoint."""
    embedded = payload.get("identity")
    if isinstance(embedded, Mapping):
        input_checkpoint = _nested_field(embedded, ("input_checkpoint_sha256", "checkpoint_sha256"))
        parent_checkpoint = _nested_field(embedded, ("parent_checkpoint_sha256", "parent_sha256"))
        model_index = _nested_field(embedded, ("model_index_sha256", "basis_sha256"))
        corpus = _nested_field(embedded, ("corpus_sha256", "train_score_corpus_sha256"))
        if input_checkpoint not in (None, ""):
            raise ArtifactError("canonical raw U0 input checkpoint identity drift")
        if parent_checkpoint not in (None, ""):
            raise ArtifactError("canonical raw U0 parent checkpoint identity drift")
        if model_index is not None:
            _require_equal(_normalized_hash_field(model_index), BASIS_SHA256, "payload model-index SHA")
        if corpus is not None:
            _require_equal(_normalized_hash_field(corpus), CANONICAL_U0_LOCK_CORPUS_SHA256, "payload corpus SHA")
    top_identity = payload.get("identity_sha256")
    if top_identity not in (None, ""):
        _require_equal(_normalized_hash_field(top_identity), CANONICAL_U0_IDENTITY_SHA256, "payload identity SHA")
    embedded_optimizer_entries = _nested_field(embedded, ("optimizer_state_entries",))
    if embedded_optimizer_entries is not None:
        _require_equal(embedded_optimizer_entries, 0, "payload optimizer state entries")
    top_optimizer_entries = payload.get("optimizer_state_entries")
    if top_optimizer_entries is not None:
        _require_equal(top_optimizer_entries, 0, "payload optimizer state entries")
    if "checkpoint_identity_sha256" in payload and payload.get("checkpoint_identity_sha256") not in (None, ""):
        _require_equal(
            _normalized_hash_field(payload.get("checkpoint_identity_sha256")),
            CANONICAL_U0_IDENTITY_SHA256,
            "payload checkpoint identity SHA",
        )
    if not isinstance(payload.get("state"), Mapping):
        raise ArtifactError("canonical raw U0 checkpoint is missing its state mapping")
    if payload.get("checkpoint_loaded", False) is not False:
        raise ArtifactError("canonical raw U0 checkpoint_loaded gate drift")
    for key in ("optimizer_state", "optimizer"):
        value = payload.get(key)
        if isinstance(value, Mapping):
            # A fresh Adam optimizer legitimately serializes configured parameter
            # groups before its first step. Scientific optimizer state is empty
            # when the state mapping has zero entries; parameter-group metadata
            # is not an optimizer-history entry.
            if value.get("state", {}):
                raise ArtifactError("canonical raw U0 optimizer state is not empty")
        elif value not in (None, {}):
            raise ArtifactError("canonical raw U0 optimizer state is not empty")
    for key in ("scheduler_state", "scheduler"):
        value = payload.get(key)
        if not isinstance(value, Mapping):
            continue
        for epoch_key in ("last_epoch", "epoch", "scheduler_epoch"):
            if epoch_key in value:
                _require_equal(value[epoch_key], 0, f"payload {epoch_key}")
    _require_equal(
        _nested_field(lock, ("checkpoint_loaded", "loaded_checkpoint", "input_checkpoint_loaded")),
        False,
        "lock checkpoint_loaded",
    )
    optimizer = _nested_field(lock, ("optimizer_state", "optimizer", "optimizer_state_entries"))
    if optimizer is None:
        raise ArtifactError("canonical raw U0 lock is missing empty optimizer state")
    if isinstance(optimizer, Mapping):
        if optimizer.get("state", {}) or optimizer.get("param_groups", []):
            raise ArtifactError("canonical raw U0 lock optimizer state is not empty")
    else:
        _require_equal(optimizer, 0, "lock optimizer state entries")
    scheduler_epoch = _nested_field(lock, ("scheduler_epoch", "scheduler_last_epoch", "last_epoch", "epoch"))
    if scheduler_epoch is None:
        raise ArtifactError("canonical raw U0 lock is missing scheduler epoch")
    _require_equal(scheduler_epoch, 0, "lock scheduler epoch")


def adapt_canonical_raw_u0_payload(
    payload: Mapping[str, Any],
    *,
    artifact_root: str | Path,
    manifest: Mapping[str, Any],
    checkpoint_path: str | Path | None = None,
    checkpoint_key: str = "UPDATE_000",
    lock_path: str | Path | None = None,
    config: Mapping[str, Any] | None = None,
) -> Mapping[str, Any]:
    """Adapt an authentic raw canonical U0 payload in memory only.

    The source file is never rewritten.  The returned copy has an identity
    envelope whose ``checkpoint_loaded`` flag describes this runtime load, not
    metadata embedded in the source checkpoint.
    """
    root = Path(artifact_root).expanduser().resolve()
    if checkpoint_key != "UPDATE_000":
        raise ArtifactError("canonical raw U0 adapter admits only UPDATE_000")
    if not isinstance(manifest, Mapping) or manifest.get("schema") != "repair-artifact-v1":
        raise ArtifactError("canonical raw U0 artifact manifest schema drift")
    checkpoints = manifest.get("checkpoints")
    meta = checkpoints.get(checkpoint_key) if isinstance(checkpoints, Mapping) else None
    if not isinstance(meta, Mapping):
        raise ArtifactError("canonical raw U0 manifest entry is missing")
    declared_path = meta.get("path")
    if declared_path != "checkpoints/UPDATE_000.pt":
        raise ArtifactError("canonical raw U0 root-relative checkpoint path drift")
    path = _relative_manifest_path(root, declared_path, "checkpoint path")
    if checkpoint_path is not None and Path(checkpoint_path).expanduser().resolve() != path:
        raise ArtifactError("canonical raw U0 checkpoint path does not match the manifest")
    if not path.is_file():
        raise ArtifactError(f"canonical raw U0 checkpoint is missing: {path}")
    _require_equal(meta.get("sha256"), CANONICAL_U0_CHECKPOINT_SHA256, "manifest checkpoint SHA")
    _require_equal(_sha256_file(path), CANONICAL_U0_CHECKPOINT_SHA256, "raw checkpoint SHA")
    _require_equal(meta.get("next_update", meta.get("update")), 0, "manifest next_update")
    if meta.get("parent_sha256") or meta.get("parent_checkpoint_sha256"):
        raise ArtifactError("canonical raw U0 must not declare a parent checkpoint")
    if "canonical_source_path" in meta:
        _require_equal(meta.get("canonical_source_path"), CANONICAL_U0_SOURCE_PATH, "canonical source path")

    supplied_config = config if isinstance(config, Mapping) else {}
    identity = manifest.get("identity")
    if not isinstance(identity, Mapping):
        raise ArtifactError("canonical raw U0 manifest identity is missing")
    _require_equal(identity.get("basis_sha256"), BASIS_SHA256, "basis SHA")
    sealed_corpus_sha = supplied_config.get("corpus_sha256")
    if sealed_corpus_sha is None:
        _require_equal(identity.get("builder_eval_corpus_sha256"), BUILDER_EVAL_CORPUS_SHA256, "builder corpus SHA")
        _require_equal(identity.get("train_score_corpus_sha256"), SCORE_TRAIN_CORPUS_SHA256, "score corpus SHA")
    else:
        _require_equal(identity.get("builder_eval_corpus_sha256"), sealed_corpus_sha, "builder corpus/config SHA")
        _require_equal(identity.get("train_score_corpus_sha256"), sealed_corpus_sha, "score corpus/config SHA")
    _require_equal(identity.get("teacher_inventory_sha256"), TEACHER_INVENTORY_SHA256, "teacher inventory SHA")
    _require_equal(meta.get("identity_sha256"), CANONICAL_U0_IDENTITY_SHA256, "checkpoint identity SHA")

    raw_config = manifest.get("canonical_raw_u0")
    raw_config = raw_config if isinstance(raw_config, Mapping) else {}
    lock_value = lock_path or supplied_config.get("clean_u0_lock_path") or raw_config.get("clean_u0_lock_path")
    lock_value = lock_value or "receipts/CLEAN_U0_LOCK.json"
    lock = _relative_manifest_path(root, lock_value, "CLEAN_U0_LOCK path")
    if not lock.is_file():
        raise ArtifactError(f"canonical raw U0 CLEAN_U0_LOCK is missing: {lock}")
    lock_sha = _sha256_file(lock)
    _require_equal(lock_sha, CANONICAL_U0_LOCK_SHA256, "CLEAN_U0_LOCK SHA")
    declared_lock_sha = raw_config.get("clean_u0_lock_sha256", identity.get("clean_u0_lock_sha256"))
    if declared_lock_sha is not None:
        _require_equal(declared_lock_sha, CANONICAL_U0_LOCK_SHA256, "manifest CLEAN_U0_LOCK SHA")
    try:
        lock_payload = json.loads(lock.read_text())
    except (OSError, ValueError) as exc:
        raise ArtifactError(f"canonical raw U0 CLEAN_U0_LOCK is invalid: {lock}") from exc
    if not isinstance(lock_payload, Mapping):
        raise ArtifactError("canonical raw U0 CLEAN_U0_LOCK must contain a mapping")
    _require_equal(
        _normalized_hash_field(
            _nested_field(lock_payload, ("checkpoint_sha256", "u0_checkpoint_sha256", "source_checkpoint_sha256", "input_checkpoint_sha256"))
        ),
        CANONICAL_U0_CHECKPOINT_SHA256,
        "lock checkpoint SHA",
    )
    _require_equal(_normalized_hash_field(_nested_field(lock_payload, ("basis_sha256", "model_index_sha256"))), BASIS_SHA256, "lock basis SHA")
    _require_equal(
        _normalized_hash_field(_nested_field(lock_payload, ("corpus_sha256", "train_score_corpus_sha256", "corpus_identity_sha256"))),
        CANONICAL_U0_LOCK_CORPUS_SHA256,
        "lock corpus SHA",
    )
    trajectory = _normalized_hash_field(
        _nested_field(lock_payload, ("trajectory_sha256", "trajectory_identity_sha256", "trajectory_id", "trajectory"))
    )
    expected_trajectory = raw_config.get("trajectory_sha256", CANONICAL_U0_TRAJECTORY_SHA256)
    _require_equal(trajectory, expected_trajectory, "lock trajectory SHA")
    _require_equal(expected_trajectory, CANONICAL_U0_TRAJECTORY_SHA256, "manifest trajectory SHA")
    _validate_raw_u0_gates(payload, lock_payload)

    manifest_path = root / "ARTIFACT.json"
    manifest_sha = _sha256_file(manifest_path) if manifest_path.is_file() else None
    envelope = {
        "schema": "canonical-raw-u0-identity-envelope-v1",
        "source": "canonical_raw_manifest_adapter",
        "embedded_identity_fields": False,
        "artifact_root": str(root),
        "artifact_manifest_sha256": manifest_sha,
        "checkpoint_path": str(path.relative_to(root)),
        "checkpoint_sha256": CANONICAL_U0_CHECKPOINT_SHA256,
        "identity_sha256": CANONICAL_U0_IDENTITY_SHA256,
        "next_update": 0,
        "checkpoint_loaded": True,
        "runtime_load_provenance": {
            "checkpoint_loaded": True,
            "source": "canonical_raw_manifest_adapter",
            "raw_checkpoint_embedded_identity": False,
            "clean_u0_lock_sha256": CANONICAL_U0_LOCK_SHA256,
            "trajectory_sha256": CANONICAL_U0_TRAJECTORY_SHA256,
        },
    }
    adapted = dict(payload)
    adapted["identity"] = envelope
    return adapted


def adapt_canonical_raw_u1_payload(
    payload: Mapping[str, Any],
    *,
    artifact_root: str | Path,
    manifest: Mapping[str, Any],
    checkpoint_path: str | Path | None = None,
    checkpoint_key: str = "UPDATE_001",
) -> Mapping[str, Any]:
    """Adapt the authentic canonical U1 checkpoint identity in memory only."""
    root = Path(artifact_root).expanduser().resolve()
    if checkpoint_key != "UPDATE_001":
        raise ArtifactError("canonical raw U1 adapter admits only UPDATE_001")
    if not isinstance(manifest, Mapping) or manifest.get("schema") != "repair-artifact-v1":
        raise ArtifactError("canonical raw U1 artifact manifest schema drift")
    checkpoints = manifest.get("checkpoints")
    meta = checkpoints.get(checkpoint_key) if isinstance(checkpoints, Mapping) else None
    if not isinstance(meta, Mapping):
        raise ArtifactError("canonical raw U1 manifest entry is missing")
    if meta.get("path") != "checkpoints/UPDATE_001.pt":
        raise ArtifactError("canonical raw U1 root-relative checkpoint path drift")
    path = _relative_manifest_path(root, meta["path"], "checkpoint path")
    if checkpoint_path is not None and Path(checkpoint_path).expanduser().resolve() != path:
        raise ArtifactError("canonical raw U1 checkpoint path does not match the manifest")
    if not path.is_file():
        raise ArtifactError(f"canonical raw U1 checkpoint is missing: {path}")
    _require_equal(meta.get("sha256"), CANONICAL_U1_CHECKPOINT_SHA256, "U1 manifest checkpoint SHA")
    _require_equal(_sha256_file(path), CANONICAL_U1_CHECKPOINT_SHA256, "raw U1 checkpoint SHA")
    _require_equal(meta.get("identity_sha256"), CANONICAL_U1_IDENTITY_SHA256, "U1 checkpoint identity SHA")
    _require_equal(meta.get("next_update", meta.get("update")), 1, "U1 manifest next_update")
    parent = meta.get("parent_sha256") or meta.get("parent_checkpoint_sha256")
    _require_equal(parent, CANONICAL_U0_CHECKPOINT_SHA256, "U1 parent checkpoint SHA")
    manifest_identity = manifest.get("identity")
    if not isinstance(manifest_identity, Mapping):
        raise ArtifactError("canonical raw U1 manifest identity is missing")
    _require_equal(manifest_identity.get("basis_sha256"), BASIS_SHA256, "U1 basis SHA")
    if not isinstance(payload.get("state"), Mapping) or set(payload["state"]) != {"luts", "norms", "outputs"}:
        raise ArtifactError("canonical raw U1 checkpoint state geometry drift")
    _require_equal(payload.get("identity_sha256"), CANONICAL_U1_IDENTITY_SHA256, "raw U1 identity SHA")
    _require_equal(payload.get("next_update"), 1, "raw U1 next_update")
    source_identity = payload.get("identity")
    if not isinstance(source_identity, Mapping):
        raise ArtifactError("canonical raw U1 source identity is missing")
    _require_equal(source_identity.get("framework"), "banana-smasher", "raw U1 framework")
    _require_equal(source_identity.get("model_index_sha256"), BASIS_SHA256, "raw U1 model-index SHA")
    input_checkpoint = source_identity.get("input_checkpoint_sha256")
    if input_checkpoint not in (None, "", CANONICAL_U0_CHECKPOINT_SHA256):
        raise ArtifactError("canonical raw U1 input checkpoint identity drift")
    _require_equal(
        source_identity.get("continuous_parent_checkpoint_sha256"),
        CANONICAL_U0_CHECKPOINT_SHA256,
        "raw U1 continuous parent SHA",
    )
    envelope = {
        "schema": "canonical-raw-u1-identity-envelope-v1",
        "source": "canonical_raw_u1_manifest_adapter",
        "embedded_identity_fields": True,
        "artifact_root": str(root),
        "artifact_manifest_sha256": _sha256_file(root / "ARTIFACT.json"),
        "checkpoint_path": str(path.relative_to(root)),
        "checkpoint_sha256": CANONICAL_U1_CHECKPOINT_SHA256,
        "identity_sha256": CANONICAL_U1_IDENTITY_SHA256,
        "parent_checkpoint_sha256": CANONICAL_U0_CHECKPOINT_SHA256,
        "next_update": 1,
        "checkpoint_loaded": True,
        "runtime_load_provenance": {
            "checkpoint_loaded": True,
            "source": "canonical_raw_u1_manifest_adapter",
            "raw_checkpoint_embedded_identity": True,
            "source_identity": dict(source_identity),
            "parent_checkpoint_sha256": CANONICAL_U0_CHECKPOINT_SHA256,
        },
    }
    adapted = dict(payload)
    adapted["identity"] = envelope
    return adapted


def load_canonical_raw_u0(
    artifact_root: str | Path,
    *,
    manifest: Mapping[str, Any] | None = None,
    checkpoint_key: str = "UPDATE_000",
    lock_path: str | Path | None = None,
    config: Mapping[str, Any] | None = None,
) -> Mapping[str, Any]:
    """Load and adapt canonical raw U0 without mutating its checkpoint file."""
    root = Path(artifact_root).expanduser().resolve()
    if manifest is None:
        try:
            manifest = json.loads((root / "ARTIFACT.json").read_text())
        except (OSError, ValueError) as exc:
            raise ArtifactError(f"cannot read canonical raw U0 artifact manifest: {root / 'ARTIFACT.json'}") from exc
    checkpoints = manifest.get("checkpoints") if isinstance(manifest, Mapping) else None
    meta = checkpoints.get(checkpoint_key) if isinstance(checkpoints, Mapping) else None
    path = meta.get("path") if isinstance(meta, Mapping) else None
    checkpoint_path = _relative_manifest_path(root, path, "checkpoint path") if path is not None else None
    payload = _load_torch(checkpoint_path) if checkpoint_path is not None and checkpoint_path.is_file() else {}
    if not payload:
        raise ArtifactError("canonical raw U0 checkpoint could not be loaded")
    return adapt_canonical_raw_u0_payload(
        payload,
        artifact_root=root,
        manifest=manifest,
        checkpoint_path=checkpoint_path,
        checkpoint_key=checkpoint_key,
        lock_path=lock_path,
        config=config,
    )


def enforce_pre_canary(
    *,
    checkpoint_sha256: str,
    kld: float,
    top1: int,
    receipt_path: str | Path,
) -> dict[str, Any]:
    """Accept only the exact frozen PRE score; quarantine every mismatch."""
    kld_ok = math.isclose(float(kld), PRE_KLD, rel_tol=0.0, abs_tol=PRE_KLD_ABS_TOLERANCE)
    row = {
        "schema": "official-k2-resident-pre-canary-v1",
        "status": "PASS" if checkpoint_sha256 == PRE_CHECKPOINT_SHA256 and kld_ok and int(top1) == PRE_TOP1 else "QUARANTINED",
        "quality_status": "ACCEPTED_PRE_CANARY" if checkpoint_sha256 == PRE_CHECKPOINT_SHA256 and kld_ok and int(top1) == PRE_TOP1 else "PRE_CANARY_MISMATCH",
        "checkpoint_sha256": checkpoint_sha256,
        "expected_checkpoint_sha256": PRE_CHECKPOINT_SHA256,
        "kld_mean": float(kld),
        "expected_kld_mean": PRE_KLD,
        "kld_abs_tolerance": PRE_KLD_ABS_TOLERANCE,
        "top1": int(top1),
        "expected_top1": PRE_TOP1,
        "windows": 64,
        "positions": 65536,
        "support": 8192,
        "direction": "KL(teacher||candidate)",
        "reduction": "binary64/math.fsum in ordered window/position order",
    }
    _atomic_json(Path(receipt_path), row)
    if row["status"] != "PASS":
        raise ArtifactError(
            "PRE canary mismatch; resident mechanism is quarantined and no quality score may be promoted"
        )
    return row


def require_canonical_calibration(receipt_path: str | Path) -> Mapping[str, Any]:
    """Require an explicit public calibration receipt for non-canonical updates."""
    path = Path(receipt_path)
    if not path.is_file():
        raise ArtifactError("canonical resident calibration receipt is required for non-canonical scoring")
    row = _load_json(path)
    expected = {
        "schema": CANONICAL_CALIBRATION_SCHEMA,
        "status": "PASS",
        "checkpoint_sha256": CANONICAL_U0_CHECKPOINT_SHA256,
        "lane": "official-k2-resident",
    }
    drift = {key: (row.get(key), value) for key, value in expected.items() if row.get(key) != value}
    if drift:
        raise ArtifactError(f"canonical resident calibration drift: {drift}")
    return row


def require_pre_calibration(receipt_path: str | Path) -> Mapping[str, Any]:
    """Validate historical alternate-PRE evidence for quarantine inspection only."""
    path = Path(receipt_path)
    if not path.is_file():
        raise ArtifactError("historical alternate PRE quarantine receipt is missing")
    row = _load_json(path)
    expected = {
        "status": "PASS",
        "quality_status": "ACCEPTED_PRE_CANARY",
        "checkpoint_sha256": PRE_CHECKPOINT_SHA256,
        "kld_mean": PRE_KLD,
        "top1": PRE_TOP1,
        "windows": 64,
        "positions": 65536,
        "support": 8192,
    }
    drift = {key: (row.get(key), value) for key, value in expected.items() if row.get(key) != value}
    if drift:
        raise ArtifactError(f"exact PRE calibration receipt drift: {drift}")
    return row


def validate_payload_identity(
    payload: Mapping[str, Any],
    *,
    checkpoint_sha256: str,
    checkpoint_identity_sha256: str,
    next_update: int,
) -> None:
    """Bind the one torch-loaded payload to manifest checkpoint identity.

    ``checkpoint_sha256`` is an external file identity and cannot generally be
    embedded in the file whose bytes it hashes.  ``RepairArtifact.checkpoint_path``
    recomputes that SHA before this payload is loaded.  Therefore an embedded
    value is checked when present, while the non-circular identity fields remain
    mandatory in every payload.
    """
    raw = payload.get("identity")
    if not isinstance(raw, Mapping):
        # Legacy accepted checkpoints are bound by the file SHA recomputed by
        # RepairArtifact plus immutable manifest metadata.  They predate the
        # embedded identity envelope; absence is therefore admissible, while a
        # present envelope remains strict and cannot partially disagree.
        embedded_checkpoint_sha = payload.get("checkpoint_sha256")
        if embedded_checkpoint_sha not in (None, "") and embedded_checkpoint_sha != checkpoint_sha256:
            raise ArtifactError(
                "payload checkpoint identity drift: "
                f"{{'checkpoint_sha256': ({embedded_checkpoint_sha!r}, {checkpoint_sha256!r})}}"
            )
        return
    identity = dict(raw)
    for key in ("identity_sha256", "next_update", "checkpoint_loaded", "checkpoint_sha256"):
        if key not in identity and key in payload:
            identity[key] = payload[key]
    if (
        "checkpoint_loaded" not in identity
        and identity.get("identity_sha256")
        and identity.get("next_update") is not None
    ):
        identity["checkpoint_loaded"] = True
    # Historical checkpoints placed the identity digest and update cursor at
    # the top level while retaining descriptive provenance under ``identity``.
    # Treat those locations as one envelope; contradictory nested values still
    # win and fail the strict comparison below.
    expected = {
        "identity_sha256": checkpoint_identity_sha256,
        "next_update": int(next_update),
        "checkpoint_loaded": True,
    }
    drift = {
        key: (identity.get(key), value)
        for key, value in expected.items()
        if identity.get(key) != value
    }
    if drift:
        raise ArtifactError(f"payload checkpoint identity drift: {drift}")
    embedded_checkpoint_sha = identity.get("checkpoint_sha256")
    if embedded_checkpoint_sha not in (None, "") and embedded_checkpoint_sha != checkpoint_sha256:
        raise ArtifactError(
            "payload checkpoint identity drift: "
            f"{{'checkpoint_sha256': ({embedded_checkpoint_sha!r}, {checkpoint_sha256!r})}}"
        )


def validate_routed_k2_closure(route: Mapping[str, Any]) -> None:
    """Require the sealed routed-only K2 package closure, without weakening score()."""
    if not isinstance(route, Mapping) or route.get("route_kind") != ROUTED_K2_ROUTE_KIND:
        raise ArtifactError("routed-K2 route kind must be explicit and exact")
    drift = {
        key: (route.get(key), expected)
        for key, expected in ROUTED_K2_CLOSURE.items()
        if route.get(key) != expected
    }
    if drift:
        raise ArtifactError(f"routed-K2 sealed package closure drift: {drift}")
    required = (
        "pre_checkpoint_identity_sha256", "post_checkpoint_identity_sha256",
        "post_parent_checkpoint_sha256", "teacher_manifest_sha256",
        "corpus_manifest_sha256", "window_manifest_sha256",
    )
    missing = [key for key in required if not isinstance(route.get(key), str) or not route[key]]
    if missing:
        raise ArtifactError("routed-K2 exact checkpoint/teacher/corpus/window manifests are required: " + ", ".join(missing))


def authorize_routed_k2_score(
    checkpoint_update: int,
    *,
    checkpoint_sha256: str,
    checkpoint_identity_sha256: str,
    checkpoint_parent_sha256: str | None,
    route: Mapping[str, Any],
) -> dict[str, Any]:
    """Admit only the exact t_36e9ce8e routed PRE/POST checkpoint pair."""
    validate_routed_k2_closure(route)
    update = int(checkpoint_update)
    if update == 0:
        expected_sha = ROUTED_K2_CLOSURE["pre_checkpoint_sha256"]
        expected_identity = route["pre_checkpoint_identity_sha256"]
        if checkpoint_sha256 != expected_sha or checkpoint_identity_sha256 != expected_identity:
            raise ArtifactError("routed-K2 PRE checkpoint identity drift")
        if checkpoint_parent_sha256 is not None:
            raise ArtifactError("routed-K2 PRE must not declare a parent checkpoint")
        scope = "ROUTED_K2_PRE"
    elif update == 1:
        expected_sha = ROUTED_K2_CLOSURE["post_checkpoint_sha256"]
        expected_identity = route["post_checkpoint_identity_sha256"]
        if checkpoint_sha256 != expected_sha or checkpoint_identity_sha256 != expected_identity:
            raise ArtifactError("routed-K2 POST checkpoint identity drift")
        if checkpoint_parent_sha256 != route["post_parent_checkpoint_sha256"]:
            raise ArtifactError("routed-K2 POST parent checkpoint identity drift")
        scope = "ROUTED_K2_POST"
    else:
        raise ArtifactError("routed-K2 admits only exact PRE and UPDATE_001 checkpoints")
    return {
        "scope": scope,
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_identity_sha256": checkpoint_identity_sha256,
        "checkpoint_parent_sha256": checkpoint_parent_sha256,
        "route_kind": ROUTED_K2_ROUTE_KIND,
        "package": dict(ROUTED_K2_CLOSURE),
        "public_api": {"method": ROUTED_K2_API_METHOD, "version": ROUTED_K2_API_VERSION},
    }


def authorize_production_score(
    checkpoint_update: int,
    *,
    pre_calibration_receipt: str | Path | None = None,
    scientific_question_receipt: str | Path | None = None,
    checkpoint_sha256: str | None = None,
    checkpoint_parent_sha256: str | None = None,
    ordered_windows_sha256: str | None = None,
    allow_alternate_pre_diagnostic: bool = False,
) -> dict[str, Any]:
    """Admit the canonical U0→U1 lane and gate all other updates publicly.

    Canonical U0 is admitted by immutable SHA. Canonical U1 requires update 1
    and the immediate canonical U0 parent. Alternate serialized PRE evidence is
    quarantine-only; non-canonical updates require a public calibration receipt
    and, for U3+, a pre-registered scientific question.
    """
    update = int(checkpoint_update)
    if checkpoint_sha256 == ALTERNATE_PRE_CHECKPOINT_SHA256:
        if not allow_alternate_pre_diagnostic:
            raise ArtifactError(
                "alternate serialized PRE is quarantine-only and cannot enter the canonical resident lane"
            )
        if update != 0 or checkpoint_parent_sha256 is not None:
            raise ArtifactError("alternate PRE diagnostic requires exact parentless update 0")
        return {
            "scope": "ALTERNATE_PRE_DIAGNOSTIC_ONLY",
            "checkpoint_sha256": checkpoint_sha256,
            "checkpoint_parent_sha256": None,
            "public_api": {"method": PUBLIC_API_METHOD, "version": PUBLIC_API_VERSION},
        }
    if checkpoint_sha256 == CANONICAL_U0_CHECKPOINT_SHA256:
        if update != 0:
            raise ArtifactError("canonical U0 checkpoint must declare update 0")
        if pre_calibration_receipt is None:
            raise ArtifactError("canonical resident calibration receipt is required for canonical U0 scoring")
        pre = require_canonical_calibration(pre_calibration_receipt)
        return {
            "scope": "CANONICAL_U0",
            "checkpoint_sha256": checkpoint_sha256,
            "checkpoint_parent_sha256": None,
            "calibration": dict(pre),
            "public_api": {"method": PUBLIC_API_METHOD, "version": PUBLIC_API_VERSION},
        }
    if checkpoint_sha256 == CANONICAL_U1_CHECKPOINT_SHA256:
        if update != 1:
            raise ArtifactError("canonical U1 checkpoint must declare update 1")
        if checkpoint_parent_sha256 != CANONICAL_U0_CHECKPOINT_SHA256:
            raise ArtifactError("canonical U1 requires the immediate canonical U0 parent")
        return {
            "scope": "CANONICAL_U1_IMMEDIATE_PARENT",
            "checkpoint_sha256": checkpoint_sha256,
            "checkpoint_parent_sha256": checkpoint_parent_sha256,
            "public_api": {"method": PUBLIC_API_METHOD, "version": PUBLIC_API_VERSION},
        }
    if checkpoint_sha256 is not None and update <= 1:
        raise ArtifactError("non-canonical U0/U1 checkpoint is not admitted to the canonical resident lane")
    if pre_calibration_receipt is None:
        raise ArtifactError("canonical resident calibration receipt is required for non-canonical scoring")
    pre = require_canonical_calibration(pre_calibration_receipt)
    if update <= 1:
        return {"scope": "CALIBRATED_PRE_OR_U1", "calibration": dict(pre)}
    if scientific_question_receipt is None or not Path(scientific_question_receipt).is_file():
        raise ArtifactError(
            "U3+ production scoring requires a separate pre-registered scientific question "
            "with matched parent, ordered windows, and dose"
        )
    question = _load_json(Path(scientific_question_receipt))
    expected = {
        "schema": "official-k2-resident-scientific-question-v1",
        "status": "PRE_REGISTERED",
        "checkpoint_update": update,
        "matched_parent_sha256": checkpoint_parent_sha256,
        "ordered_windows_sha256": ordered_windows_sha256,
        "dose": update,
    }
    drift = {key: (question.get(key), value) for key, value in expected.items() if question.get(key) != value}
    if drift:
        raise ArtifactError(f"pre-registered scientific question identity drift: {drift}")
    return {
        "scope": "SEPARATE_PRE_REGISTERED_QUESTION",
        "calibration": dict(pre),
        "scientific_question": question,
    }


def _bind_resident_expert_class(
    module: Any,
    route_kind: str | None,
    diagnostic_mode: str | None,
) -> Any:
    """Keep production resident execution distinct from routed diagnostics."""
    if route_kind != ROUTED_K2_ROUTE_KIND:
        return module
    if diagnostic_mode != "sealed_reference":
        if getattr(module, "FullyResidentGroupedV7Experts", None) is None:
            raise ArtifactError(
                "routed-K2 production source is missing FullyResidentGroupedV7Experts"
            )
        return module
    routed = getattr(module, "JointV7ExpertBase", None)
    if routed is None:
        raise ArtifactError("routed-K2 physical source is missing JointV7ExpertBase")

    class FullyResidentGroupedV7Experts(routed):
        def __init__(self, *args: Any, swiglu_limit: float | None = None, **kwargs: Any) -> None:
            # Routed sealed-reference diagnostics use their own frozen expert
            # implementation; accept the resident constructor's public seam
            # without mutating that oracle's arithmetic.
            del swiglu_limit
            super().__init__(*args, **kwargs)

        @property
        def resident_bytes(self) -> int:
            # The immutable routed-PRE oracle decodes member projections lazily;
            # this compatibility count is diagnostic metadata, not a residency claim.
            return 0

    FullyResidentGroupedV7Experts.__module__ = getattr(module, "__name__", routed.__module__)
    module.FullyResidentGroupedV7Experts = FullyResidentGroupedV7Experts
    return module


def _wait_for_cold_load_gate(config: Mapping[str, Any], rank: int) -> None:
    """Keep rank 1's checkpoint payload off CUDA until rank 0 is bounded."""
    value = config.get("cold_load_gate_dir")
    if value is None or rank == 0:
        return
    generation = str(config.get("cold_load_generation", "")).strip()
    if not generation:
        raise ArtifactError("official-K2 cold-load gate requires cold_load_generation")
    path = Path(str(value)).expanduser().resolve() / "COLD_LOAD_RANK0_PRUNED.json"
    timeout = float(config.get("cold_load_gate_timeout_seconds", 900.0))
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            row = _load_json(path)
        except (ArtifactError, FileNotFoundError, json.JSONDecodeError, OSError):
            row = {}
        if (
            row.get("status") == "PASS"
            and row.get("generation") == generation
            and (
                int(row.get("cold_source_bytes_pruned", 0))
                + int(row.get("cold_source_cache_drop_bytes", 0))
            ) > 0
        ):
            return
        time.sleep(min(0.25, max(timeout / 10.0, 0.001)))
    raise ArtifactError(
        f"official-K2 cold-load gate timeout rank{rank}: {path} generation={generation}"
    )


def _support_mass_diagnostic(torch: Any, ref_logprob: Any, q_logprob: Any) -> dict[str, float]:
    """Mirror the sealed scorer's float32 support-mass diagnostics."""
    mass_p = ref_logprob.float().exp().sum(-1)
    mass_q = q_logprob.float().exp().sum(-1)
    return {
        "mass_p_mean": float(mass_p.mean().item()),
        "mass_p_sum": float(mass_p.sum().item()),
        "mass_q_mean": float(mass_q.mean().item()),
        "mass_q_sum": float(mass_q.sum().item()),
    }


def _validate_parity_terminal(
    terminals: Iterable[Mapping[str, Any]], *, allow_source_reads: bool = False
) -> None:
    """Keep production resident closure strict while admitting the lazy sealed oracle."""
    zero_fields = (
        "fallback_calls", "reconstruction_calls", "reference_fwht_calls",
        "cpu_relay_bytes", "layer_streaming_calls",
    )
    if not allow_source_reads:
        zero_fields = ("timed_model_payload_reads", *zero_fields)
    if any(any(int(row.get(name, -1)) != 0 for name in zero_fields) for row in terminals):
        raise ArtifactError(f"parity_tap terminal closure failed: {terminals}")


class PayloadModelReadCounter:
    """Audit Python file opens beneath immutable score-input roots.

    The hook is installed before loading but activated only at resident_ready.
    Consequently the terminal delta counts forbidden payload/model/teacher/
    corpus reopens during scoring, not the intentional one-time resident load.
    """

    def __init__(self, roots: Iterable[str | Path]):
        self.roots = tuple(Path(value).expanduser().resolve() for value in roots)
        self.active = False
        self.reads = 0
        self.paths: list[str] = []

        def audit(event: str, args: tuple[Any, ...]) -> None:
            if not self.active or event != "open" or not args:
                return
            raw = args[0]
            if not isinstance(raw, (str, bytes, os.PathLike)):
                return
            try:
                path = Path(os.fsdecode(raw)).expanduser().resolve()
            except (OSError, TypeError, ValueError):
                return
            if any(path == root or root in path.parents for root in self.roots):
                self.reads += 1
                if len(self.paths) < 16:
                    self.paths.append(str(path))

        self._audit = audit
        sys.addaudithook(audit)

    def mark_resident_ready(self) -> int:
        self.active = True
        return self.reads

    def delta(self, start: int) -> int:
        return self.reads - start


class OfficialK2ResidentRankEngine:
    """One rank of the two-rank, fully resident official-K2 scorer."""

    def __init__(
        self,
        *,
        payload: Mapping[str, Any],
        checkpoint_sha256: str,
        checkpoint_identity_sha256: str,
        windows: tuple[int, ...],
        config: Mapping[str, Any],
    ) -> None:
        started = time.perf_counter()
        try:
            import numpy as np
            import torch
            import torch.distributed as dist
        except ImportError as exc:
            raise ArtifactError("official-K2 resident scoring requires NumPy and PyTorch") from exc
        self.np = np
        self.torch = torch
        self.dist = dist
        self.config = dict(config)
        self.local_dual_shard = bool(config.get("same_process_dual_shard", False))
        self.local_coordinator = config.get("local_dual_shard_coordinator")
        self.rank = int(config.get("rank", 0)) if self.local_dual_shard else int(
            os.environ.get("RANK") or config.get("rank", 0)
        )
        self.world_size = int(os.environ.get("WORLD_SIZE", config.get("world_size", 2)))
        if self.rank not in (0, 1) or self.world_size != 2:
            raise ArtifactError("official-K2 resident scoring requires exactly ranks 0 and 1")
        self.device = torch.device(str(config.get("device", "cuda")))
        if self.device.type != "cuda" or not torch.cuda.is_available():
            raise ArtifactError("official-K2 resident scoring requires a CUDA device")
        torch.cuda.set_device(int(config.get("cuda_device", 0)))
        self.windows = windows
        self.checkpoint_sha256 = checkpoint_sha256
        self.checkpoint_identity_sha256 = checkpoint_identity_sha256
        self.model_root = Path(str(config["model_root"])).expanduser().resolve()
        self.asset_root = Path(str(config["asset_root"])).expanduser().resolve()
        self.parent_root = Path(str(config["parent_root"])).expanduser().resolve()
        self.teacher_root = Path(str(config["teacher_root"])).expanduser().resolve()
        self.corpus_path = Path(str(config["corpus"])).expanduser().resolve()
        self.trainer_path = Path(str(config["trainer_source"])).expanduser().resolve()
        self.lp4_pack_path = Path(str(config.get(
            "lp4_pack_source", "/home/dnola/missions/RESIDENT_FULL64_t_8a38f1b8/repo/runtime/v7/vendor/src_lp4/lp4_pack.py"
        ))).expanduser().resolve()
        self.lp4_train_path = Path(str(config.get(
            "lp4_train_source", "/home/dnola/missions/RESIDENT_FULL64_t_8a38f1b8/repo/runtime/v7/vendor/src_lp4/lp4_train.py"
        ))).expanduser().resolve()
        self.builder_source_path = Path(str(config.get(
            "t8192_builder_source", "/home/dnola/missions/LP4_REPAIR/src/t8192_ds4_build_v3.py"
        ))).expanduser().resolve()
        self.fast_k2_extension = Path(str(config["fast_k2_extension"])).expanduser().resolve()
        self.fast_k2_module_name = str(config["fast_k2_module_name"])
        self.fast_k2_wrapper_source = Path(str(config["fast_k2_wrapper_source"])).expanduser().resolve()
        # The routed closure binds two distinct expert sources: the official
        # selected-wire implementation for scientific identity, and the
        # fully-resident grouped implementation imported by the trainer under
        # its historical ``fast_v7_expert_base`` module name.  Loading the
        # official source under that name makes the public route fail before
        # model construction because it does not define
        # FullyResidentGroupedV7Experts.
        self.official_expert_source = Path(str(config["official_expert_source"])).expanduser().resolve()
        self.expert_source = Path(str(config["resident_expert_source"])).expanduser().resolve()
        self.l034_roster = Path(str(config["l034_roster"])).expanduser().resolve()
        self.basis_sha256 = str(config.get("basis_sha256", BASIS_SHA256))
        if self.basis_sha256 != BASIS_SHA256:
            raise ArtifactError("official-K2 resident basis configuration drift")
        if not bool(config.get("shared_cuda_device_process_group", False)):
            self.qsfp_pin = _validate_qsfp_pin(config, rank=self.rank)
            os.environ["GLOO_SOCKET_IFNAME"] = self.qsfp_pin["interface"]
            os.environ["NCCL_SOCKET_IFNAME"] = self.qsfp_pin["interface"]
        else:
            self.qsfp_pin = {"mode": "host-local-shared-device"}
        expert_source_sha256 = _configured_expert_source_sha256(config)
        resident_expert_source_sha256 = str(config.get("resident_expert_source_sha256", ""))
        if len(resident_expert_source_sha256) != 64 or any(
            character not in "0123456789abcdef" for character in resident_expert_source_sha256
        ):
            raise ArtifactError("resident grouped expert source SHA configuration is malformed")
        required = {
            self.model_root / "model.safetensors.index.json": BASIS_SHA256,
            self.official_expert_source: expert_source_sha256,
            self.expert_source: resident_expert_source_sha256,
            self.lp4_pack_path: LP4_PACK_SOURCE_SHA256,
            self.lp4_train_path: LP4_TRAIN_SOURCE_SHA256,
            self.builder_source_path: T8192_BUILDER_SOURCE_SHA256,
            self.fast_k2_extension: str(config["fast_k2_extension_sha256"]),
            self.fast_k2_wrapper_source: str(config["fast_k2_wrapper_source_sha256"]),
        }
        if config.get("trainer_source_sha256"):
            required[self.trainer_path] = str(config["trainer_source_sha256"])
        for path, expected in required.items():
            if not path.is_file():
                raise ArtifactError(f"official-K2 resident immutable input is missing: {path}")
            observed = _sha256_file(path)
            if observed != expected:
                raise ArtifactError(f"official-K2 resident immutable SHA mismatch: {path}: {observed} != {expected}")
        # Arm the rank-0 TCPStore listener before rank-local model loading.
        # Rank load times differ by more than c10d's default connection window;
        # creating the rendezvous after ShardStudent made the faster rank time
        # out and orphaned the slower rank before resident_ready.
        if self.local_dual_shard:
            self.rendezvous_preflight = {
                "status": "PASS",
                "mode": "same-process-local-cuda",
                "rank": self.rank,
                "world_size": 2,
            }
        else:
            self._init_rendezvous()
        # On unified-memory hosts, loading both rank-local parents concurrently
        # can exceed RAM before either rank reaches the post-load prune.  Keep
        # the TCPStore armed, then let rank 0 convert+prune first; rank 1 starts
        # only after the generation-bound durable marker is visible.
        if not self.local_dual_shard:
            self._wait_for_cold_load_turn()
        tracked_roots = (
            self.model_root,
            self.asset_root,
            self.parent_root,
            self.teacher_root,
            self.corpus_path,
            Path(str(config["checkpoint_path"])),
        )
        self.read_counter = PayloadModelReadCounter(tracked_roots)
        self._preflight_memory()
        self._prepare_import_paths()
        self._configure_base_environment()
        # Bind the exact admitted API-core expert module before the trainer
        # imports it by its historical top-level name. This prevents an older
        # asset/source directory from shadowing the corrected sealed-parity
        # implementation on sys.path.
        os.environ["FAST_K2_EXTENSION"] = str(self.fast_k2_extension)
        os.environ["FAST_K2_EXTENSION_SHA256"] = str(config["fast_k2_extension_sha256"])
        os.environ["FAST_K2_MODULE_NAME"] = self.fast_k2_module_name
        self._load_module("fast_k2_grouped", self.fast_k2_wrapper_source)
        expert_module = self._load_module("fast_v7_expert_base", self.expert_source)
        _bind_resident_expert_class(
            expert_module,
            config.get("route_kind"),
            config.get("parity_tap_mode"),
        )
        self.expert_parallel_all_layers = bool(
            config.get("expert_parallel_all_layers", False)
        )
        if self.expert_parallel_all_layers:
            ranges = {0: (0, 42), 1: (0, 42)}
            self.first, self.last = (0, 42)
        else:
            ranges = self._layer_ranges(config.get("layer_split"))
            self.first, self.last = ranges[self.rank]
        self.gpu_resident_storage_broker = bool(
            config.get("gpu_resident_storage_broker", False)
        )
        if self.gpu_resident_storage_broker:
            if not bool(config.get("ordinary_load_fork_broker", False)):
                raise ArtifactError("GPU resident storage broker requires ordinary-load fork broker")
            if bool(config.get("checkpoint_mmap", True)):
                raise ArtifactError("GPU resident storage broker requires checkpoint_mmap=false")
            expert_module._resident_storage_provider = (
                self._consume_brokered_resident_storage if self.rank == 1 else None
            )
        self.expert_module = expert_module
        self.trainer = self._load_module(
            f"banana_smasher_resident_score_trainer_{os.getpid()}_{self.rank}", self.trainer_path
        )
        if getattr(self.trainer, "MODEL_INDEX_SHA256", None) != BASIS_SHA256:
            raise ArtifactError("official resident trainer model-index identity drift")
        self.base = self._load_base()
        try:
            from banana_smasher import qtip_k2 as official_k2
        except Exception as exc:
            raise ArtifactError(f"official grouped-K2 backend is unavailable: {exc}") from exc
        self._configure_base()
        admission_path = self.asset_root / "code" / "JOINT_REPAIR_ADMISSION.json"
        admission = _load_json(admission_path)
        if config.get("lut_parent_root"):
            admission = _rebase_admission_lut_sources(admission, config["lut_parent_root"])
        if admission.get("framework") != "banana-smasher":
            raise ArtifactError("official-K2 resident admission framework drift")
        self.status: dict[str, Any] = {}
        self.cold_source_files_pruned = 0
        self.cold_source_bytes_pruned = 0
        self.cold_source_cache_drop_files = 0
        self.cold_source_cache_drop_bytes = 0
        self.student = self.trainer.ShardStudent(
            torch=torch,
            np=np,
            base=self.base,
            official_k2=official_k2,
            model_root=self.model_root,
            admission=admission,
            parent_root=self.parent_root,
            l034_roster=self.l034_roster,
            input_state=payload,
            rank=self.rank,
            first=self.first,
            last=self.last,
            status_cb=self._status,
            defer_dense_l034=False,
        )
        if self.expert_parallel_all_layers:
            for expert in self.student.experts.values():
                expert.expert_parallel_rank = self.rank
                expert.expert_parallel_world_size = self.world_size
                expert.expert_parallel_group = None
            if self.rank == 1:
                from torch import nn
                self.student.model.model.embed_tokens.weight = nn.Parameter(
                    self.student.get_tensor("embed.weight")
                    .to(self.device).to(torch.bfloat16),
                    requires_grad=False,
                )
        self._gpu_storage_broker_owned: dict[int, tuple[Any, Any]] = {}
        if self.gpu_resident_storage_broker and self.rank == 0:
            self._publish_brokered_resident_storage(admission, ranges[1])
        if bool(config.get(
            "drop_model_cache_after_resident_load",
            config.get("drop_parent_cache_incrementally_after_layer_load", False),
        )):
            model_files, model_bytes = _drop_cold_model_cache(self.model_root)
            self.cold_source_cache_drop_files += model_files
            self.cold_source_cache_drop_bytes += model_bytes
        if (
            bool(config.get("prune_parent_after_resident_load", False))
            and not bool(config.get("prune_parent_incrementally_after_layer_load", False))
        ):
            final_files, final_bytes = _prune_loaded_parent_members(
                parent_root=self.parent_root,
                sources=self.student.sources,
            )
            self.cold_source_files_pruned += final_files
            self.cold_source_bytes_pruned += final_bytes
        self._bind_checkpoint_state(payload, admission)
        self._assert_fully_resident_grouped_experts()
        self._load_inputs()
        # Resolve the child-side payload lifetime before rank 0 opens the load
        # gate. Ordinary per-rank loads transfer ownership by clearing their
        # mapping. Fork-broker children retain only the inherited read-only view;
        # its CPU pages stay physically shared with the broker and peer rank.
        _release_or_retain_checkpoint_payload(
            payload,
            ordinary_load_fork_broker=bool(config.get("ordinary_load_fork_broker", False)),
            checkpoint_sha256=self.checkpoint_sha256,
        )
        self._release_post_bind_checkpoint_workspace()
        # Rank 1's heavy resident conversion must not overlap rank 0's
        # checkpoint/input bind or rank 0's now-dead conversion allocations.
        # On coherent-memory CUDA, allocator-reserved blocks remain charged to
        # the host until empty_cache(), even after their tensors are dead.
        self._release_transient_resident_load_workspace()
        self._publish_cold_load_pruned()
        if not self.local_dual_shard:
            self._init_distributed()
        torch.cuda.synchronize()
        self.resident_bytes = self._resident_bytes()
        self.resident_load_seconds = time.perf_counter() - started
        self.ready_counter = self.read_counter.mark_resident_ready()
        local_ready = {
            "event": "resident_ready",
            "rank": self.rank,
            "resident_bytes": self.resident_bytes,
            "checkpoint_sha256": self.checkpoint_sha256,
            "checkpoint_identity_sha256": self.checkpoint_identity_sha256,
            "model_index_sha256": BASIS_SHA256,
            "basis_sha256": BASIS_SHA256,
            "official_physical_layer_sha256": OFFICIAL_PHYSICAL_LAYER_SHA256,
            "payload_model_file_reads": self.ready_counter,
            "cold_source_files_pruned": self.cold_source_files_pruned,
            "cold_source_bytes_pruned": self.cold_source_bytes_pruned,
            "cold_source_cache_drop_files": self.cold_source_cache_drop_files,
            "cold_source_cache_drop_bytes": self.cold_source_cache_drop_bytes,
            "transient_load_memory_release": dict(self.transient_load_memory_release),
            "resident_load_seconds": self.resident_load_seconds,
            "rendezvous_preflight": dict(self.rendezvous_preflight),
            "memory_preflight": dict(self.memory_preflight),
            "qsfp_pin": dict(self.qsfp_pin),
            "layer_range": [self.first, self.last],
        }
        if self.local_dual_shard:
            self.local_ready = local_ready
            self.resident_ready = [local_ready]
        else:
            rows: list[Any] = [None, None]
            self.dist.all_gather_object(rows, local_ready)
            self.resident_ready = rows
        callback = config.get("event_callback")
        if callable(callback):
            callback(dict(local_ready))

    @staticmethod
    def _load_module(name: str, path: Path) -> Any:
        spec = importlib.util.spec_from_file_location(name, path)
        if spec is None or spec.loader is None:
            raise ArtifactError(f"cannot import official resident source: {path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        source_dir = str(path.parent)
        inserted = source_dir not in sys.path
        if inserted:
            sys.path.insert(0, source_dir)
        try:
            spec.loader.exec_module(module)
        finally:
            if inserted:
                sys.path.remove(source_dir)
        return module

    @staticmethod
    def _broker_storage_key(layer: int) -> str:
        return f"resident-storage-L{int(layer):03d}"

    def _consume_brokered_resident_storage(
        self, layer: int, plane_source: Any
    ) -> Mapping[str, Any]:
        del plane_source
        if self.local_dual_shard:
            payload = self.local_coordinator.storage[int(layer)]
            if int(payload.get("layer", -1)) != int(layer):
                raise ArtifactError(f"local resident storage layer drift: {layer}")
            return payload
        key = self._broker_storage_key(layer)
        payload = _deserialize_resident_storage_ipc(bytes(self.rendezvous_store.get(key)))
        if int(payload.get("layer", -1)) != int(layer):
            raise ArtifactError(f"brokered resident storage layer drift: {layer}")
        self.rendezvous_store.set(key + "-consumed", b"1")
        return payload

    def _publish_brokered_resident_storage(
        self, admission: Mapping[str, Any], peer_range: tuple[int, int]
    ) -> None:
        """Own rank-1's immutable expert storages and export CUDA-IPC views."""
        rows = {int(row["layer"]): row for row in admission["trainable_roster"]["luts"]}
        first, last = map(int, peer_range)
        expert_class = self.expert_module.FullyResidentGroupedV7Experts
        provider = getattr(self.expert_module, "_resident_storage_provider", None)
        self.expert_module._resident_storage_provider = None
        try:
            for layer in range(first, last + 1):
                source = self.trainer.PlaneSource(
                    torch=self.torch,
                    np=self.np,
                    row=rows[layer],
                    parent_root=self.parent_root,
                    l034_roster=self.l034_roster,
                    device=self.device,
                )
                expert = expert_class(
                    layer=layer,
                    pilot=True,
                    plane_source=source,
                    swiglu_limit=float(self.student.model.model.layers[layer].mlp.experts.limit),
                )
                projections = {
                    name: (
                        getattr(expert, f"packed_{name}"),
                        getattr(expert, f"su_{name}"),
                        getattr(expert, f"sv_{name}"),
                    )
                    for name in ("w1", "w2", "w3")
                }
                payload = {
                    "layer": layer,
                    "master": source.master.detach(),
                    "projections": projections,
                }
                # Retain the allocating objects until both ranks finish. CUDA
                # IPC consumers are invalid as soon as the producer frees them.
                self._gpu_storage_broker_owned[layer] = (source, expert)
                if self.local_dual_shard:
                    self.local_coordinator.storage[layer] = payload
                    continue
                key = self._broker_storage_key(layer)
                self.rendezvous_store.set(key, _serialize_resident_storage_ipc(payload))
                self.rendezvous_store.get(key + "-consumed")
        finally:
            self.expert_module._resident_storage_provider = provider

    def _cold_load_gate(self) -> tuple[Path, str] | None:
        value = self.config.get("cold_load_gate_dir")
        if value is None:
            return None
        generation = str(self.config.get("cold_load_generation", "")).strip()
        if not generation:
            raise ArtifactError("official-K2 cold-load gate requires cold_load_generation")
        return Path(str(value)).expanduser().resolve() / "COLD_LOAD_RANK0_PRUNED.json", generation

    def _wait_for_cold_load_turn(self) -> None:
        _wait_for_cold_load_gate(self.config, self.rank)

    def _release_transient_resident_load_workspace(self) -> None:
        """Return dead load-time CUDA banks before the peer starts loading."""
        cuda = self.torch.cuda
        cuda.synchronize()
        before_reserved = int(cuda.memory_reserved(self.device))
        before_allocated = int(cuda.memory_allocated(self.device))
        gc.collect()
        _trim_host_allocator()
        cuda.empty_cache()
        cuda.synchronize()
        self.transient_load_memory_release = {
            "reserved_before_bytes": before_reserved,
            "allocated_before_bytes": before_allocated,
            "reserved_after_bytes": int(cuda.memory_reserved(self.device)),
            "allocated_after_bytes": int(cuda.memory_allocated(self.device)),
        }

    def _release_post_bind_checkpoint_workspace(self) -> None:
        """Return checkpoint tensors no longer owned after resident binding."""
        cuda = self.torch.cuda
        cuda.synchronize()
        before_reserved = int(cuda.memory_reserved(self.device))
        before_allocated = int(cuda.memory_allocated(self.device))
        gc.collect()
        _trim_host_allocator()
        cuda.empty_cache()
        cuda.synchronize()
        self.post_bind_checkpoint_memory_release = {
            "reserved_before_bytes": before_reserved,
            "allocated_before_bytes": before_allocated,
            "reserved_after_bytes": int(cuda.memory_reserved(self.device)),
            "allocated_after_bytes": int(cuda.memory_allocated(self.device)),
        }

    def _publish_cold_load_pruned(self) -> None:
        gate = self._cold_load_gate()
        if gate is None or self.rank != 0:
            return
        path, generation = gate
        bounded_source_bytes = (
            int(self.cold_source_bytes_pruned)
            + int(self.cold_source_cache_drop_bytes)
        )
        if bounded_source_bytes <= 0:
            raise ArtifactError(
                "official-K2 rank0 cold-load gate requires positive pruned or cache-dropped bytes"
            )
        _atomic_json(
            path,
            {
                "status": "PASS",
                "generation": generation,
                "rank": self.rank,
                "pid": os.getpid(),
                "cold_source_files_pruned": int(self.cold_source_files_pruned),
                "cold_source_bytes_pruned": int(self.cold_source_bytes_pruned),
                "cold_source_cache_drop_files": int(self.cold_source_cache_drop_files),
                "cold_source_cache_drop_bytes": int(self.cold_source_cache_drop_bytes),
                "unix": time.time(),
            },
        )

    @staticmethod
    def _layer_ranges(value: Any) -> dict[int, tuple[int, int]]:
        if not isinstance(value, Mapping):
            raise ArtifactError("official-K2 resident score requires an explicit two-rank layer_split")
        try:
            ranges = {int(key): tuple(int(item) for item in row) for key, row in value.items()}
        except (TypeError, ValueError) as exc:
            raise ArtifactError("official-K2 resident layer_split is invalid") from exc
        if set(ranges) != {0, 1} or any(len(row) != 2 for row in ranges.values()):
            raise ArtifactError("official-K2 resident layer_split must assign ranks 0 and 1")
        covered: set[int] = set()
        for lo, hi in ranges.values():
            current = set(range(lo, hi + 1))
            if lo < 0 or hi > 42 or lo > hi or covered & current:
                raise ArtifactError("official-K2 resident layer_split must be disjoint and within 0..42")
            covered |= current
        if covered != set(range(43)):
            raise ArtifactError("official-K2 resident layer_split must cover all 43 layers")
        return ranges

    def _preflight_memory(self) -> None:
        expected_map = self.config.get("estimated_resident_bytes_by_rank", EXPECTED_RESIDENT_BYTES)
        expected, _ = resolve_rank_local_bytes(
            expected_map,
            self.rank,
            default=EXPECTED_RESIDENT_BYTES[self.rank],
            field="estimated_resident_bytes_by_rank",
        )
        peak_map = self.config.get("estimated_peak_bytes_by_rank", expected_map)
        peak, _ = resolve_rank_local_bytes(
            peak_map,
            self.rank,
            default=expected,
            field="estimated_peak_bytes_by_rank",
        )
        if peak < expected:
            raise ArtifactError(
                f"estimated_peak_bytes_by_rank rank{self.rank} is below resident estimate"
            )
        brokered_resident_bytes = 0
        if (
            self.rank == 1
            and bool(self.config.get("gpu_resident_storage_broker", False))
            and self.local_dual_shard
        ):
            coordinator = self.local_coordinator
            if coordinator is None:
                raise ArtifactError("rank1 brokered resident coordinator is missing")
            brokered_resident_bytes = _unique_tensor_storage_bytes(coordinator.storage)
            if brokered_resident_bytes <= 0:
                raise ArtifactError("rank1 brokered resident storage byte inventory is empty")
        incremental_expected = max(0, expected - brokered_resident_bytes)
        incremental_peak = max(incremental_expected, peak - brokered_resident_bytes)
        reserve, reserve_policy = resolve_rank_local_bytes(
            self.config.get("cuda_reserve_bytes"),
            self.rank,
            default=DEFAULT_CUDA_RESERVE_BYTES,
            field="cuda_reserve_bytes",
        )
        # GB10 unified memory: cuda.mem_get_info() mirrors Linux MemFree and
        # EXCLUDES reclaimable page cache, so a host with 100GB+ actually
        # available can report ~40GB "free" after heavy file I/O and fail this
        # gate spuriously (v10 incident, 2026-08-18). Drop caches once and
        # re-read before failing; the kernel reclaims cache on demand anyway,
        # so this only makes the preflight measurement honest.
        free_bytes, total_bytes = self.torch.cuda.mem_get_info()
        free_bytes = int(free_bytes)
        required = incremental_peak + reserve
        if free_bytes < required:
            try:
                subprocess.run(
                    ["sudo", "-n", "sh", "-c",
                     "sync && echo 3 > /proc/sys/vm/drop_caches"],
                    capture_output=True, timeout=60, check=False,
                )
                free_bytes = int(self.torch.cuda.mem_get_info()[0])
            except Exception:
                pass  # fall through with original reading; gate below decides
        margin = free_bytes - required
        prunable_cold_source_bytes = 0
        effective_free_bytes = free_bytes
        if bool(self.config.get("prune_parent_after_resident_load", False)):
            prunable_cold_source_bytes = sum(
                path.stat().st_size
                for pattern in ("*.q2v7wire", "*.k2wire")
                for path in self.parent_root.rglob(pattern)
                if path.is_file() and not path.is_symlink()
            )
            effective_free_bytes += prunable_cold_source_bytes
            margin = effective_free_bytes - required
        self.memory_preflight = {
            "rank": self.rank,
            "cuda_free_bytes": free_bytes,
            "effective_free_after_cold_source_prune_bytes": effective_free_bytes,
            "prunable_cold_source_bytes": prunable_cold_source_bytes,
            "cuda_total_bytes": int(total_bytes),
            "estimated_resident_bytes": expected,
            "peak_estimate_bytes": peak,
            "brokered_resident_bytes": brokered_resident_bytes,
            "incremental_resident_bytes": incremental_expected,
            "incremental_peak_bytes": incremental_peak,
            "reserve_bytes": reserve,
            "reserve_policy": reserve_policy,
            "required_cuda_free_bytes": required,
            "margin_bytes": margin,
            "predicate": "incremental_peak_bytes + reserve_bytes <= cuda_free_bytes + task_local_prunable_cold_source_bytes",
        }
        if margin < 0:
            error = ArtifactError(
                f"official-K2 resident CUDA preflight refused rank{self.rank}: "
                f"free={free_bytes} required={required} resident={expected} "
                f"peak={peak} brokered={brokered_resident_bytes} "
                f"incremental_peak={incremental_peak} reserve={reserve} margin={margin}"
            )
            error.__dict__["memory_preflight"] = dict(self.memory_preflight)
            raise error

    def _prepare_import_paths(self) -> None:
        for path in (
            self.trainer_path.parent,
            self.lp4_pack_path.parent,
            self.lp4_train_path.parent,
            self.builder_source_path.parent,
            self.expert_source.parent,
            self.asset_root / "source",
            self.asset_root / "source" / "site",
            self.parent_root,
        ):
            value = str(path)
            if value not in sys.path:
                sys.path.insert(0, value)

    def _configure_base_environment(self) -> None:
        """Bind the legacy base loader's required env to the public manifest."""
        attention = _configured_attention_implementation(self.config)
        os.environ["BR_ATTN_IMPL"] = attention
        path_values = {
            "BR_MANIFEST": self.config.get(
                "binrepair_manifest", self.asset_root / "code" / "DUALVQ_K4096MENU_IQ3_BIN_MANIFEST.json"
            ),
            "BR_DELTA_DIR": self.config.get("binrepair_delta_dir", self.asset_root / "delta"),
            "BR_VQ3B_DIR": self.config.get("binrepair_vq3b_dir"),
        }
        if path_values["BR_VQ3B_DIR"] is None:
            raise ArtifactError("official resident manifest is missing binrepair_vq3b_dir")
        for key, value in path_values.items():
            path = Path(str(value)).expanduser().resolve()
            if not path.exists():
                raise ArtifactError(f"official resident base input is missing for {key}: {path}")
            os.environ[key] = str(path)
        os.environ["BR_TRAIN"] = str(
            self.config.get("binrepair_train_windows", ",".join(map(str, self.windows)))
        )
        os.environ["BR_PROBE"] = str(
            self.config.get("binrepair_probe_windows", ",".join(map(str, self.windows)))
        )

    def _load_base(self) -> Any:
        path = self.asset_root / "source" / "base_binrepair_e2e.py"
        if not path.is_file():
            raise ArtifactError(f"official resident base source is missing: {path}")
        module = self._load_module(f"banana_smasher_resident_score_base_{os.getpid()}_{self.rank}", path)
        module.T.CKPT = str(self.model_root)
        module.T.DEV = "cuda"
        return module

    def _configure_base(self) -> None:
        os.environ["BR_CORPUS"] = str(self.corpus_path)
        os.environ["BR_TEACH"] = str(self.teacher_root)
        os.environ.setdefault("BR_ATTN_IMPL", "eager")
        os.environ.setdefault("BR_FAST_STACK", "1")
        self.base.T.CKPT = str(self.model_root)
        self.base.T.DEV = "cuda"
        import random
        random.seed(1701)
        self.torch.manual_seed(1701)
        self.torch.cuda.manual_seed_all(1701)

    def _status(self, **fields: Any) -> None:
        self.status.update(fields)
        if (
            bool(self.config.get("drop_parent_cache_incrementally_after_layer_load", False))
            and fields.get("phase") == "loading"
            and "loaded_layer" in fields
        ):
            layer = int(fields["loaded_layer"])
            root = self.parent_root.resolve(strict=True)
            candidate = root / f"L{layer:03d}"
            # L034 is admitted through its separately hash-bound roster and is
            # intentionally absent from the staged parent root.  There are no
            # staged parent pages to evict for that layer.
            if not candidate.exists():
                return
            layer_root = candidate.resolve(strict=True)
            if root not in layer_root.parents:
                raise ArtifactError("incremental parent cache-drop layer escapes staged root")
            files = sorted(list(layer_root.glob("*.q2v7wire")) + list(layer_root.glob("*.k2wire")))
            file_count, byte_count = _drop_cold_file_cache(files)
            self.cold_source_cache_drop_files += file_count
            self.cold_source_cache_drop_bytes += byte_count
        if (
            bool(self.config.get("prune_parent_incrementally_after_layer_load", False))
            and fields.get("phase") == "loading"
            and "loaded_layer" in fields
        ):
            layer = int(fields["loaded_layer"])
            root = self.parent_root.resolve(strict=True)
            candidate = root / f"L{layer:03d}"
            if not candidate.exists():
                return
            layer_root = candidate.resolve(strict=True)
            if root not in layer_root.parents:
                raise ArtifactError("incremental parent prune layer escapes staged root")
            files = sorted(list(layer_root.glob("*.q2v7wire")) + list(layer_root.glob("*.k2wire")))
            byte_count = sum(path.stat().st_size for path in files)
            for path in files:
                path.unlink()
            self.cold_source_files_pruned += len(files)
            self.cold_source_bytes_pruned += byte_count

    def _bind_checkpoint_state(self, payload: Mapping[str, Any], admission: Mapping[str, Any]) -> None:
        state = payload.get("state")
        if not isinstance(state, Mapping) or set(state) != {"luts", "norms", "outputs"}:
            raise ArtifactError("official-K2 checkpoint must contain exactly luts, norms, and outputs")
        if not hasattr(self, "_local_dense"):
            self._local_dense = self.trainer.expose_local_dense(self.torch, self.student, admission)
        luts, norms, outputs = self._local_dense
        for surface, rows in (("luts", luts), ("norms", norms), ("outputs", outputs)):
            saved = state.get(surface)
            if not isinstance(saved, Mapping):
                raise ArtifactError(f"official-K2 checkpoint is missing {surface}")
            self.trainer.load_local_state(rows, saved, self.student.device)

    def rebind_checkpoint(
        self,
        *,
        payload: Mapping[str, Any],
        checkpoint_sha256: str,
        checkpoint_identity_sha256: str,
        admission: Mapping[str, Any],
    ) -> None:
        """Rebind trainable state while retaining the complete resident model."""
        started = time.perf_counter()
        self._bind_checkpoint_state(payload, admission)
        self.torch.cuda.synchronize()
        self.checkpoint_sha256 = checkpoint_sha256
        self.checkpoint_identity_sha256 = checkpoint_identity_sha256
        rebind_seconds = time.perf_counter() - started
        local_ready = {
            "event": "resident_ready",
            "resident_reused": True,
            "rank": self.rank,
            "resident_bytes": self.resident_bytes,
            "checkpoint_sha256": checkpoint_sha256,
            "checkpoint_identity_sha256": checkpoint_identity_sha256,
            "model_index_sha256": BASIS_SHA256,
            "basis_sha256": BASIS_SHA256,
            "official_physical_layer_sha256": OFFICIAL_PHYSICAL_LAYER_SHA256,
            "payload_model_file_reads": self.read_counter.reads,
            "cold_source_files_pruned": self.cold_source_files_pruned,
            "cold_source_bytes_pruned": self.cold_source_bytes_pruned,
            "cold_source_cache_drop_files": self.cold_source_cache_drop_files,
            "cold_source_cache_drop_bytes": self.cold_source_cache_drop_bytes,
            "resident_load_seconds": 0.0,
            "resident_rebind_seconds": rebind_seconds,
            "rendezvous_preflight": dict(self.rendezvous_preflight),
            "memory_preflight": dict(self.memory_preflight),
            "layer_range": [self.first, self.last],
        }
        rows: list[Any] = [None, None]
        self.dist.all_gather_object(rows, local_ready)
        self.resident_ready = rows

    def _assert_fully_resident_grouped_experts(self) -> None:
        local_layers = self.student.model.model.layers[self.first : self.last + 1]
        names = [type(module).__name__ for layer in local_layers for module in layer.modules()]
        count = sum(name == "FullyResidentGroupedV7Experts" for name in names)
        if count != self.last - self.first + 1:
            raise ArtifactError(
                "official-K2 resident scorer did not construct one FullyResidentGroupedV7Experts per local layer"
            )
        meta = [name for layer in local_layers for name, tensor in list(layer.named_parameters()) + list(layer.named_buffers()) if getattr(tensor, "is_meta", False)]
        if meta:
            raise ArtifactError(f"official-K2 resident local layer closure still contains meta tensors: {meta[:8]}")

    def _load_inputs(self) -> None:
        corpus = _load_json(self.corpus_path)
        if not isinstance(corpus, list):
            raise ArtifactError("official-K2 resident corpus must be a list")
        seq_len = SOURCE_CONTEXT_TOKENS
        pad_token = int(self.config.get("pad_token_id", 1))
        self.ids_cache: dict[int, Any] = {}
        self.real_lengths: dict[int, int] = {}
        self.teacher_cache: dict[int, tuple[Any, Any]] = {}
        load_windows = tuple(int(value) for value in self.config.get(
            "physical_canary_windows", self.windows
        ))
        if not set(self.windows).issubset(load_windows):
            raise ArtifactError("official-K2 physical canary batch omits a reported window")
        for window in load_windows:
            if window < 0 or window >= len(corpus) or not isinstance(corpus[window], Mapping):
                raise ArtifactError(f"official-K2 resident corpus is missing window {window}")
            row = corpus[window]
            tokens = row.get("token_ids")
            real_len = int(row.get("real_len", len(tokens) if isinstance(tokens, list) else 0))
            tokens = _canonical_causal_score_tokens(tokens, real_len=real_len, pad_token_id=pad_token)
            ids = self.torch.full((1, seq_len), pad_token, dtype=self.torch.long, device=self.device)
            ids[0, : len(tokens)] = self.torch.tensor(tokens, dtype=self.torch.long, device=self.device)
            self.ids_cache[window] = ids
            self.real_lengths[window] = real_len
            if self.rank == 1:
                teacher_path = self.teacher_root / f"t8192_win{window}.pt"
                teacher = _load_torch(teacher_path)
                idx = teacher.get("idx")
                logprob = teacher.get("logprob")
                if not hasattr(idx, "shape") or not hasattr(logprob, "shape"):
                    raise ArtifactError(f"official-K2 resident teacher window {window} is malformed")
                idx = idx[:POSITIONS_PER_WINDOW, :SUPPORT].to(dtype=self.torch.int64, device="cpu").contiguous()
                logprob = logprob[:POSITIONS_PER_WINDOW, :SUPPORT].to(dtype=self.torch.float16, device="cpu").contiguous()
                if tuple(idx.shape) != (POSITIONS_PER_WINDOW, SUPPORT) or tuple(logprob.shape) != (POSITIONS_PER_WINDOW, SUPPORT):
                    raise ArtifactError(f"official-K2 resident teacher window {window} geometry drift")
                self.teacher_cache[window] = (idx, logprob)

    def _init_distributed(self) -> None:
        if self.dist.is_initialized():
            if self.dist.get_world_size() != 2 or self.dist.get_rank() != self.rank:
                raise ArtifactError("existing process group does not match official-K2 resident rank")
        else:
            store = getattr(self, "rendezvous_store", None)
            if store is None:
                raise ArtifactError("official-K2 resident TCPStore rendezvous was not armed before model load")
            timeout_seconds = int(self.config.get("process_group_timeout_seconds", 3600))
            if timeout_seconds < 1 or timeout_seconds > 7200:
                raise ArtifactError("official-K2 resident process-group timeout must be within 1..7200 seconds")
            backend = str(self.config.get("distributed_backend", "nccl"))
            if bool(self.config.get("shared_cuda_device_process_group", False)):
                master_addr = str(self.config.get("master_addr", os.environ.get("MASTER_ADDR", "127.0.0.1")))
                if master_addr not in ("127.0.0.1", "localhost", "::1"):
                    raise ArtifactError("shared CUDA-device process group requires a host-local rendezvous")
                # NCCL rejects two processes intentionally bound to one physical GPU.
                # Keep only tiny control/object collectives on Gloo; activations use
                # CUDA IPC handles in _score_window and never relay tensor bytes via CPU.
                backend = "gloo"
            try:
                self.dist.init_process_group(
                    backend=backend,
                    store=store,
                    rank=self.rank,
                    world_size=2,
                    timeout=timedelta(seconds=timeout_seconds),
                )
            except Exception as exc:
                raise ArtifactError(f"official-K2 resident process-group initialization failed: {exc}") from exc
        if not bool(self.config.get("shared_cuda_device_process_group", False)):
            self._warm_p2p_communicator()

    def _warm_p2p_communicator(self) -> None:
        """Collectively create the exact two-rank NCCL P2P communicator."""
        peer = 1 - self.rank
        outgoing = self.torch.full(
            (1,), self.rank, dtype=self.torch.int32, device=self.student.device
        )
        incoming = self.torch.empty_like(outgoing)
        operations = [
            self.dist.P2POp(self.dist.isend, outgoing, peer),
            self.dist.P2POp(self.dist.irecv, incoming, peer),
        ]
        started = time.perf_counter()
        try:
            requests = self.dist.batch_isend_irecv(operations)
            for request in requests:
                request.wait()
            self.torch.cuda.synchronize()
        except Exception as exc:
            raise ArtifactError(
                f"official-K2 resident P2P communicator warmup failed: {exc}"
            ) from exc
        if int(incoming.item()) != peer:
            raise ArtifactError("official-K2 resident P2P communicator warmup identity drift")
        self.p2p_communicator_warmup = {
            "status": "PASS",
            "rank": self.rank,
            "peer": peer,
            "elapsed_seconds": time.perf_counter() - started,
        }

    def _batch_p2p_isend(self, tensor: Any, *, dst: int) -> Any:
        """Launch one send through the communicator-compatible batched API."""
        operations = [self.dist.P2POp(self.dist.isend, tensor, dst)]
        requests = self.dist.batch_isend_irecv(operations)
        if len(requests) != 1:
            raise ArtifactError("official-K2 resident batched P2P send request drift")
        return requests[0]

    def _batch_p2p_recv(self, tensor: Any, *, src: int) -> None:
        """Receive one activation through the already-warmed batched API."""
        operations = [self.dist.P2POp(self.dist.irecv, tensor, src)]
        requests = self.dist.batch_isend_irecv(operations)
        if len(requests) != 1:
            raise ArtifactError("official-K2 resident batched P2P receive request drift")
        requests[0].wait()

    def _init_rendezvous(self) -> None:
        master_addr = str(self.config.get("master_addr", os.environ.get("MASTER_ADDR", "127.0.0.1")))
        master_port = int(self.config.get("master_port", os.environ.get("MASTER_PORT", 29598)))
        timeout_seconds = int(self.config.get("rendezvous_timeout_seconds", 120))
        if timeout_seconds < 1 or timeout_seconds > 600:
            raise ArtifactError("official-K2 resident rendezvous timeout must be within 1..600 seconds")
        try:
            self.rendezvous_store = self.dist.TCPStore(
                master_addr,
                master_port,
                2,
                self.rank == 0,
                timeout=timedelta(seconds=timeout_seconds),
                # Rank 1 is deliberately held before checkpoint materialization
                # until rank 0 has converted and pruned its cold sources.  The
                # rank-0 TCPStore must therefore arm without waiting for the
                # gated client; rank 1 connects after the durable prune marker.
                wait_for_workers=self.rank != 0,
            )
        except Exception as exc:
            raise ArtifactError(
                f"official-K2 resident TCPStore rendezvous preflight failed: {exc}"
            ) from exc
        self.rendezvous_preflight = {
            "status": "PASS",
            "master_addr": master_addr,
            "master_port": master_port,
            "rank": self.rank,
            "world_size": 2,
            "rank0_listener": self.rank == 0,
            "timeout_seconds": timeout_seconds,
        }

    def _resident_bytes(self) -> int:
        tensors = list(self.student.model.parameters()) + list(self.student.model.buffers()) + list(self.ids_cache.values())
        seen: set[tuple[str, int]] = set()
        total = 0
        for tensor in tensors:
            if not hasattr(tensor, "untyped_storage") or getattr(tensor, "is_meta", False):
                continue
            storage = tensor.untyped_storage()
            key = (str(tensor.device), int(storage.data_ptr()))
            if key not in seen:
                seen.add(key)
                total += int(storage.nbytes())
        if self.rank == 1:
            total += sum(int(value.numel() * value.element_size()) for pair in self.teacher_cache.values() for value in pair)
        return total

    def _score_resume_path(self) -> Path:
        config = getattr(self, "config", {})
        attention = _configured_attention_implementation(config).upper()
        configured_root = config.get("score_resume_root")
        root = (
            Path(str(configured_root)).expanduser().resolve()
            if configured_root is not None
            else self.parent_root.parent
        )
        return root / (
            f"SCORE_RESUME_{self.checkpoint_sha256}_{attention}_RANK{self.rank}.json"
        )

    @staticmethod
    def _score_implementation_sha256() -> str:
        return _sha256_file(Path(__file__).resolve())

    def _persist_score_resume(
        self,
        *,
        completed_windows: int,
        terms: list[float],
        top1: int,
        per_window: list[dict[str, Any]],
        cumulative_scoring_wall_seconds: float,
    ) -> None:
        completed = int(completed_windows)
        if completed < 0 or completed > len(self.windows):
            raise ArtifactError("official-K2 score resume completed-window count is invalid")
        if self.rank == 1:
            if len(terms) != completed * POSITIONS_PER_WINDOW or len(per_window) != completed:
                raise ArtifactError("official-K2 score resume rank1 reduction geometry drift")
        elif terms or top1 or per_window:
            raise ArtifactError("official-K2 score resume rank0 must not persist rank1 reductions")
        _atomic_json(
            self._score_resume_path(),
            {
                "schema": "official-k2-resident-score-resume-v1",
                "status": "COMPLETE" if completed == len(self.windows) else "PARTIAL",
                "basis_sha256": BASIS_SHA256,
                "implementation_sha256": self._score_implementation_sha256(),
                "attention_implementation": _configured_attention_implementation(
                    getattr(self, "config", {})
                ),
                "checkpoint_sha256": self.checkpoint_sha256,
                "ordered_windows_sha256": _windows_sha256(self.windows),
                "rank": self.rank,
                "completed_windows": completed,
                "terms": list(terms),
                "top1": int(top1),
                "per_window": list(per_window),
                "cumulative_scoring_wall_seconds": float(cumulative_scoring_wall_seconds),
            },
        )

    def _load_score_resume(self) -> dict[str, Any]:
        path = self._score_resume_path()
        empty = {
            "completed_windows": 0,
            "terms": [],
            "top1": 0,
            "per_window": [],
            "cumulative_scoring_wall_seconds": 0.0,
            "implementation_sha256": self._score_implementation_sha256(),
        }
        if not path.is_file():
            return empty
        row = _load_json(path)
        current_implementation = self._score_implementation_sha256()
        expected = {
            "schema": "official-k2-resident-score-resume-v1",
            "basis_sha256": BASIS_SHA256,
            "attention_implementation": _configured_attention_implementation(
                getattr(self, "config", {})
            ),
            "checkpoint_sha256": self.checkpoint_sha256,
            "ordered_windows_sha256": _windows_sha256(self.windows),
            "rank": self.rank,
        }
        if not isinstance(row, Mapping) or any(row.get(key) != value for key, value in expected.items()):
            raise ArtifactError(f"official-K2 score resume identity drift: {path}")
        source_implementation = row.get("implementation_sha256")
        if source_implementation != current_implementation:
            compatible_u0_adapter_only_change = (
                self.checkpoint_sha256 == CANONICAL_U0_CHECKPOINT_SHA256
                and source_implementation == U0_RESUME_COMPATIBLE_IMPLEMENTATION_SHA256
            )
            if not compatible_u0_adapter_only_change:
                raise ArtifactError(f"official-K2 score resume identity drift: {path}")
            row = dict(row)
            row["source_implementation_sha256"] = source_implementation
            row["implementation_sha256"] = current_implementation
            _atomic_json(path, row)
        completed = int(row.get("completed_windows", -1))
        terms = row.get("terms")
        top1 = int(row.get("top1", -1))
        per_window = row.get("per_window")
        cumulative = float(row.get("cumulative_scoring_wall_seconds", -1.0))
        if completed < 0 or completed > len(self.windows) or cumulative < 0.0:
            raise ArtifactError("official-K2 score resume progress drift")
        if not isinstance(terms, list) or not isinstance(per_window, list):
            raise ArtifactError("official-K2 score resume reductions are malformed")
        if self.rank == 1:
            if len(terms) != completed * POSITIONS_PER_WINDOW or len(per_window) != completed or top1 < 0:
                raise ArtifactError("official-K2 score resume rank1 reduction geometry drift")
        elif terms or top1 != 0 or per_window:
            raise ArtifactError("official-K2 score resume rank0 reduction drift")
        expected_status = "COMPLETE" if completed == len(self.windows) else "PARTIAL"
        if row.get("status") != expected_status:
            raise ArtifactError("official-K2 score resume status drift")
        return dict(row)

    def _positional(self, ids: Any, template: Any, cache: Any) -> tuple[Any, Any, Any]:
        """Build the exact sealed-builder positional state for one prefill."""
        from transformers.masking_utils import create_sliding_window_causal_mask
        pos = self.torch.arange(ids.shape[1], device=self.student.device).unsqueeze(0)
        embeddings = self.student.model.model.rotary_emb
        pe = {
            "main": embeddings(template, position_ids=pos, layer_type="main"),
            "compress": embeddings(template, position_ids=pos, layer_type="compress"),
        }
        mask = create_sliding_window_causal_mask(
            config=self.student.config,
            inputs_embeds=template,
            attention_mask=None,
            past_key_values=cache,
            position_ids=pos,
        )
        return pos, pe, mask

    @staticmethod
    def _tensor_tap(value: Any) -> dict[str, Any]:
        """Describe one tensor with stable bytes and a bounded numeric sample."""
        tensor = value.detach().contiguous()
        cpu = tensor.to(device="cpu")
        raw = cpu.view(dtype=__import__("torch").uint8).numpy().tobytes()
        flat = cpu.reshape(-1)[:8]
        if flat.dtype.is_floating_point:
            sample = [float(item) for item in flat.float().tolist()]
        else:
            sample = [int(item) for item in flat.to(dtype=__import__("torch").int64).tolist()]
        return {
            "sha256": hashlib.sha256(raw).hexdigest(),
            "dtype": str(tensor.dtype),
            "shape": list(tensor.shape),
            "sample": sample,
        }

    @staticmethod
    def _attention_workspace_key(
        query: Any, key: Any, chunk_size: int, logits_dtype: Any
    ) -> tuple[Any, ...]:
        return (
            str(query.device),
            str(query.dtype),
            tuple(int(value) for value in query.shape),
            int(key.shape[-2]),
            int(chunk_size),
            str(logits_dtype),
        )

    def _attention_workspace_for(
        self, query: Any, key: Any, chunk_size: int, logits_dtype: Any
    ) -> tuple[Any, Any, Any]:
        """Own one reusable workspace per CUDA stream.

        Concurrent sealed pairs must never alias attention scratch.  CPU tests
        and the single-stream path retain one stable cache key.
        """
        batch, heads, query_rows, width = query.shape
        key_rows = int(key.shape[-2])
        rows = min(int(chunk_size), int(query_rows))
        device = getattr(query, "device", None)
        stream_key: Any = "cpu"
        if getattr(device, "type", None) == "cuda":
            stream_key = int(self.torch.cuda.current_stream(device=device).cuda_stream)
        workspace_key = (
            stream_key,
            *self._attention_workspace_key(query, key, rows, logits_dtype),
        )
        workspaces = getattr(self, "_attention_workspaces", None)
        if workspaces is None:
            workspaces = {}
            self._attention_workspaces = workspaces
        current = workspaces.get(workspace_key)
        if current is None:
            output = query.new_empty((batch, query_rows, heads, width))
            weights = query.new_empty((batch, heads, rows, key_rows))
            logits = self.torch.empty(
                (batch, heads, rows, key_rows + 1), device=query.device, dtype=logits_dtype
            )
            current = (workspace_key, output, weights, logits)
            workspaces[workspace_key] = current
        return current[1], current[2], current[3]

    @staticmethod
    def _chunked_eager_attention_forward(
        module: Any,
        query: Any,
        key: Any,
        value: Any,
        attention_mask: Any,
        scaling: float,
        dropout: float = 0.0,
        **kwargs: Any,
    ) -> tuple[Any, None]:
        """Run exact eager attention a bounded number of query rows at a time."""
        torch = __import__("torch")
        chunk_size = int(kwargs.pop("query_chunk_size", 128))
        observer = kwargs.pop("_chunk_observer", None)
        workspace_observer = kwargs.pop("_workspace_observer", None)
        workspace_factory = kwargs.pop("_official_k2_workspace_factory", None)
        if chunk_size <= 0:
            raise ArtifactError("official-K2 attention query chunk must be positive")
        if not callable(workspace_factory):
            raise ArtifactError("official-K2 attention caller workspace is required")

        def repeat_kv(states: Any, repeats: int) -> Any:
            batch, heads, length, width = states.shape
            if repeats == 1:
                return states
            states = states[:, :, None, :, :].expand(batch, heads, repeats, length, width)
            return states.reshape(batch, heads * repeats, length, width)

        batch, heads, query_rows, width = query.shape
        repeats = int(module.num_key_value_groups)
        if int(key.shape[1]) == 1 and repeats == heads:
            # ``matmul`` broadcasts the single shared KV head without materializing
            # the two full repeated K/V tensors retained by stock eager attention.
            key_states = key
            value_states = value
        else:
            key_states = repeat_kv(key, repeats)
            value_states = repeat_kv(value, repeats)
        logits_dtype = torch.promote_types(query.dtype, module.sinks.dtype)
        factory = cast(Callable[..., tuple[Any, Any, Any]], workspace_factory)
        output, weight_workspace, logits_workspace = factory(
            query, key, chunk_size, logits_dtype
        )
        for start in range(0, query_rows, chunk_size):
            end = min(start + chunk_size, query_rows)
            rows = end - start
            if observer is not None:
                observer(rows)
            query_chunk = query[:, :, start:end]
            logits = logits_workspace[:, :, :rows]
            if workspace_observer is not None:
                workspace_observer(output, weight_workspace, logits_workspace)
            weights = weight_workspace[:, :, :rows]
            torch.matmul(query_chunk, key_states.transpose(2, 3), out=weights)
            weights.mul_(scaling)
            if attention_mask is not None:
                mask = attention_mask
                if int(mask.shape[-2]) == query_rows:
                    mask = mask[..., start:end, :]
                weights.add_(mask)
            logits[..., :-1].copy_(weights)
            logits[..., -1:].copy_(
                module.sinks.reshape(1, -1, 1, 1).expand(batch, -1, rows, -1)
            )
            logits.sub_(logits.max(dim=-1, keepdim=True).values)
            # ATen's exact eager softmax accepts an aliased ``out`` tensor.  Reuse
            # the logits bank after its final read instead of retaining a second
            # full probability bank for every resident layer forward.
            torch.ops.aten._softmax.out(logits, -1, False, out=logits)
            scores = logits[..., :-1]
            scores = torch.nn.functional.dropout(
                scores, p=dropout, training=bool(getattr(module, "training", False))
            ).to(value_states.dtype)
            torch.matmul(scores, value_states, out=output[:, start:end].transpose(1, 2))
            del query_chunk, weights, logits, scores
        return output, None

    def _call_chunked_self_attention(self, attention: Any, hidden: Any, **kwargs: Any) -> Any:
        """Select the model's public attention forward with a bounded eager backend."""
        forward = attention.forward
        function = getattr(forward, "__func__", forward)
        namespace = getattr(function, "__globals__", None)
        config = attention.config
        interface = namespace.get("ALL_ATTENTION_FUNCTIONS") if isinstance(namespace, dict) else None
        register = getattr(interface, "register", None)
        if not callable(register):
            raise ArtifactError("official-K2 public attention registration seam drift")
        implementation = "official_k2_chunked_eager"
        register(implementation, self._chunked_eager_attention_forward)
        original_implementation = config._attn_implementation
        config._attn_implementation = implementation
        try:
            return attention(
                hidden,
                query_chunk_size=int(
                    self.config.get("attention_query_chunk_size", 512)
                ),
                _official_k2_workspace_factory=self._attention_workspace_for,
                **kwargs,
            )
        finally:
            config._attn_implementation = original_implementation

    def _streamed_decoder_layer(
        self,
        layer: Any,
        hidden: Any,
        ids: Any,
        position_embeddings: Any,
        position_ids: Any,
        attention_mask: Any,
        past_key_values: Any,
        output: Any,
        residual: Any,
    ) -> Any:
        """Execute the public decoder arithmetic with three full-shape banks.

        The stock expression briefly retains its input, product, residual
        matmul, and sum.  Product-first ``out=`` operations preserve that exact
        operation order while reusing caller-owned output/residual storage.
        Once the attention residual is complete, the original input bank is
        dead and becomes the MoE output bank.
        """
        dtype = hidden.dtype
        post, comb, collapsed = layer.attn_hc(hidden)
        attention_output, _ = self._call_chunked_self_attention(
            layer.self_attn,
            layer.input_layernorm(collapsed),
            position_embeddings=position_embeddings,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
        )
        self.torch.mul(
            post.to(dtype).unsqueeze(-1), attention_output.unsqueeze(-2), out=output
        )
        self.torch.matmul(
            comb.to(dtype).transpose(-1, -2), hidden, out=residual
        )
        output.add_(residual)
        del post, comb, collapsed, attention_output

        post, comb, collapsed = layer.ffn_hc(output)
        mlp_output = layer.mlp(
            layer.post_attention_layernorm(collapsed), input_ids=ids
        )
        self.torch.mul(
            post.to(dtype).unsqueeze(-1), mlp_output.unsqueeze(-2), out=hidden
        )
        self.torch.matmul(
            comb.to(dtype).transpose(-1, -2), output, out=residual
        )
        hidden.add_(residual)
        return hidden

    def _decoder_workspace_for(
        self, hidden: Any, *, stream_key: Any | None = None
    ) -> tuple[Any, Any]:
        """Own one exact streamed-decoder workspace per pair CUDA stream."""
        if stream_key is None:
            device = getattr(hidden, "device", None)
            stream_key = "cpu"
            if getattr(device, "type", None) == "cuda":
                stream_key = int(
                    self.torch.cuda.current_stream(device=device).cuda_stream
                )
        key = (
            stream_key,
            str(getattr(hidden, "device", "cpu")),
            str(hidden.dtype),
            tuple(int(value) for value in hidden.shape),
        )
        workspaces = getattr(self, "_decoder_workspaces", None)
        if workspaces is None:
            workspaces = {}
            self._decoder_workspaces = workspaces
        current = workspaces.get(key)
        if current is None:
            current = (self.torch.empty_like(hidden), self.torch.empty_like(hidden))
            workspaces[key] = current
        return current

    @staticmethod
    def _release_completed_layer_cache(cache: Any, index: int) -> None:
        """Drop dead prefill K/V tensors while preserving the sealed cache object."""
        layers = getattr(cache, "layers", None)
        if not isinstance(layers, (list, tuple)) or not 0 <= index < len(layers):
            raise ArtifactError("official-K2 shared layer cache geometry drift")
        entry = layers[index]
        if not bool(getattr(entry, "is_initialized", False)):
            return
        keys: Any = getattr(entry, "keys", None)
        values: Any = getattr(entry, "values", None)
        if not hasattr(keys, "new_empty") or not hasattr(values, "new_empty"):
            raise ArtifactError("official-K2 shared layer cache tensor drift")
        # A later decoder layer receives the same DynamicCache object, matching
        # the sealed builder, but never consumes an earlier layer's K/V entry in
        # this single full-prefill pass.  Retaining those entries only scales
        # workspace with layer count.
        entry.keys = keys.new_empty((0,))
        entry.values = values.new_empty((0,))
        entry.is_initialized = False

    def _run_layers(self, hidden: Any, ids: Any, taps: dict[str, Any] | None = None) -> Any:
        from transformers.cache_utils import DynamicCache

        template = hidden[:, :, 0, :] if hidden.ndim == 4 else hidden
        # Match the accepted A1 and the sealed builder mechanically: one cache
        # carried through every ordered layer.
        layer_cache = DynamicCache(config=self.student.config)
        pos, pe, mask = self._positional(ids, template, layer_cache)
        hidden = hidden.detach()
        output, residual = self._decoder_workspace_for(hidden)
        for index in range(self.first, self.last + 1):
            hidden = self._streamed_decoder_layer(
                self.student.model.model.layers[index], hidden, ids,
                pe, pos, mask, layer_cache, output, residual,
            )
            self._release_completed_layer_cache(layer_cache, index)
            if taps is not None:
                taps[f"L{index:03d}"] = self._tensor_tap(hidden)
        del layer_cache, output, residual
        return hidden

    def parity_tap(self, window: int) -> dict[str, Any]:
        """Run one resident window and collect the exact public parity trace."""
        torch = self.torch
        selected = int(window)
        if self.windows != (selected,):
            raise ArtifactError("parity_tap engine must be resident for exactly one requested window")
        ids = self.ids_cache[selected]
        local: dict[str, Any] = {}
        local_diagnostic: dict[str, Any] | None = None
        local_capture: dict[str, Any] | None = None
        shape = (
            1, ids.shape[1], int(self.student.config.hc_mult),
            int(self.student.config.hidden_size),
        )
        started = time.perf_counter()
        with torch.no_grad():
            if self.rank == 0:
                local["ids"] = self._tensor_tap(ids)
                embeddings = self.student.model.model.embed_tokens(ids)
                local["embeddings"] = self._tensor_tap(embeddings)
                hidden = embeddings.unsqueeze(2).expand(
                    -1, -1, self.student.config.hc_mult, -1
                ).contiguous()
                hidden = self._run_layers(hidden, ids, local)
                if tuple(hidden.shape) != shape or hidden.dtype != torch.bfloat16:
                    raise ArtifactError("parity_tap rank0 activation geometry drift")
                self.dist.send(hidden.contiguous(), dst=1)
            else:
                hidden = torch.empty(shape, dtype=torch.bfloat16, device=self.student.device)
                self.dist.recv(hidden, src=0)
                hidden = self._run_layers(hidden, ids, local)
                hc = self.student.model.model.hc_head(hidden)
                local["hc_head"] = self._tensor_tap(hc)
                final = self.student.model.model.norm(hc)
                local["norm"] = self._tensor_tap(final)
                logits = self.student.model.lm_head(
                    final[0, :POSITIONS_PER_WINDOW].to(torch.bfloat16)
                ).float()
                local["logits"] = self._tensor_tap(logits)
                logprob = torch.log_softmax(logits, dim=-1)
                teacher_idx, teacher_logprob = self.teacher_cache[selected]
                idx_device = teacher_idx.to(device=self.student.device, non_blocking=False)
                q_lp = logprob.gather(1, idx_device).to(torch.float16)
                q_argmax = logprob.argmax(-1).to(torch.int32)
                local["q_lp_at_ref"] = self._tensor_tap(q_lp)
                local["q_argmax"] = self._tensor_tap(q_argmax)
                q_capture = q_lp.cpu().numpy()
                capture_path = self.config.get("q_lp_at_ref_w28_path") if selected == 28 else None
                if capture_path is not None:
                    local_capture = _write_q_lp_capture(capture_path, q_capture)
                q_np = q_capture.astype(self.np.float64, copy=False)
                ref_np = teacher_logprob.numpy().astype(self.np.float64, copy=False)
                ref_max = self.np.max(ref_np, axis=1, keepdims=True)
                cand_max = self.np.max(q_np, axis=1, keepdims=True)
                ref_norm = ref_np - (
                    ref_max + self.np.log(self.np.sum(
                        self.np.exp(ref_np - ref_max), axis=1,
                        dtype=self.np.float64, keepdims=True,
                    ))
                )
                cand_norm = q_np - (
                    cand_max + self.np.log(self.np.sum(
                        self.np.exp(q_np - cand_max), axis=1,
                        dtype=self.np.float64, keepdims=True,
                    ))
                )
                terms = self.np.sum(
                    self.np.exp(ref_norm) * (ref_norm - cand_norm),
                    axis=1, dtype=self.np.float64,
                )
                term_values = [float(value) for value in terms.tolist()]
                local_diagnostic = {
                    "window": selected,
                    "positions": POSITIONS_PER_WINDOW,
                    "support": SUPPORT,
                    "kld_sum": math.fsum(term_values),
                    "kld_mean": math.fsum(term_values) / POSITIONS_PER_WINDOW,
                    "top1": int(self.np.count_nonzero(
                        q_argmax.cpu().numpy() == teacher_idx[:, 0].numpy()
                    )),
                    **_support_mass_diagnostic(torch, teacher_logprob, q_lp.cpu()),
                }
        torch.cuda.synchronize()
        gathered: list[Any] = [None, None]
        self.dist.all_gather_object(gathered, local)
        merged: dict[str, Any] = {}
        for row in gathered:
            if not isinstance(row, Mapping):
                raise ArtifactError("parity_tap rank trace fan-in failed")
            overlap = set(merged) & set(row)
            if overlap:
                raise ArtifactError(f"parity_tap duplicate rank taps: {sorted(overlap)}")
            merged.update(row)
        required = (
            "ids", "embeddings", *(f"L{layer:03d}" for layer in range(43)),
            "hc_head", "norm", "logits", "q_lp_at_ref", "q_argmax",
        )
        if set(merged) != set(required):
            raise ArtifactError("parity_tap did not close all required tensors")
        diagnostics: list[Any] = [None, None]
        self.dist.all_gather_object(diagnostics, local_diagnostic)
        diagnostic = diagnostics[1]
        if diagnostics[0] is not None or not isinstance(diagnostic, Mapping):
            raise ArtifactError("parity_tap diagnostic metric fan-in failed")
        captures: list[Any] = [None, None]
        self.dist.all_gather_object(captures, local_capture)
        capture = captures[1]
        if captures == [None, None]:
            capture = None
        elif captures[0] is not None or not isinstance(capture, Mapping):
            raise ArtifactError("parity_tap q_lp_at_ref capture fan-in failed")
        read_delta = self.read_counter.delta(self.ready_counter)
        local_terminal = {
            "rank": self.rank,
            "timed_model_payload_reads": read_delta,
            "fallback_calls": int(self.status.get("fallback_calls", 0)),
            "reconstruction_calls": int(self.status.get("reconstruction_calls", 0)),
            "reference_fwht_calls": int(self.status.get("reference_fwht_calls", 0)),
            "cpu_relay_bytes": int(self.status.get("cpu_relay_bytes", 0)),
            "layer_streaming_calls": 0,
        }
        terminals: list[Any] = [None, None]
        self.dist.all_gather_object(terminals, local_terminal)
        zero_fields = (
            "timed_model_payload_reads", "fallback_calls", "reconstruction_calls",
            "reference_fwht_calls", "cpu_relay_bytes", "layer_streaming_calls",
        )
        allow_source_reads = self.config.get("route_kind") == ROUTED_K2_ROUTE_KIND
        _validate_parity_terminal(terminals, allow_source_reads=allow_source_reads)
        result = {
            "taps": {name: merged[name] for name in required},
            "diagnostic_metrics": dict(diagnostic),
            "runtime_counters": {
                **{
                    name: (
                        sum(int(row[name]) for row in terminals)
                        if name == "timed_model_payload_reads" and allow_source_reads
                        else 0
                    )
                    for name in zero_fields
                },
                "resident_ready": self.resident_ready,
                "rank_terminal": terminals,
                "wall_seconds": time.perf_counter() - started,
            },
        }
        if capture is not None:
            result["q_lp_at_ref_capture"] = dict(capture)
        return result

    def _drain_pipeline_inflight(self) -> float:
        """Retire rank-0 activation owners after rank 1 has consumed them."""
        waited = 0.0
        while getattr(self, "_pipeline_inflight", []):
            previous_key, previous_hidden, previous_work = self._pipeline_inflight.pop(0)
            wait_started = time.perf_counter()
            self.rendezvous_store.get(previous_key + "-consumed")
            if previous_work is not None:
                previous_work.wait()
            waited += time.perf_counter() - wait_started
            # The producer must retain this CUDA storage until the consumer ACK.
            del previous_hidden
        return waited

    def _score_window(self, windows: int | Iterable[int]) -> list[tuple[list[float], int]] | None:
        torch = self.torch
        window_started = time.perf_counter()
        reported_windows = (windows,) if isinstance(windows, int) else tuple(int(value) for value in windows)
        if not reported_windows:
            raise ArtifactError("official-K2 resident score batch must not be empty")
        batch_windows = tuple(int(value) for value in self.config.get(
            "physical_canary_windows", reported_windows
        )) if reported_windows == (28,) else reported_windows
        if not set(reported_windows).issubset(batch_windows):
            raise ArtifactError("official-K2 physical score batch omits a reported window")
        pair_stream_concurrency = int(
            self.config.get("score_pair_stream_concurrency", 1)
        )
        pair_parallel = pair_stream_concurrency > 1 and len(batch_windows) > 2
        if pair_parallel:
            if len(batch_windows) % 2 or len(batch_windows) > 2 * pair_stream_concurrency:
                raise ArtifactError(
                    "official-K2 concurrent score group must contain only whole sealed pairs"
                )
            pair_windows = tuple(
                batch_windows[index : index + 2]
                for index in range(0, len(batch_windows), 2)
            )
        else:
            pair_windows = (batch_windows,)
        pair_ids = tuple(
            torch.cat([self.ids_cache[window] for window in pair], dim=0)
            for pair in pair_windows
        )
        ids = torch.cat([self.ids_cache[window] for window in batch_windows], dim=0)
        shared_cuda = bool(self.config.get("shared_cuda_device_process_group", False))
        local_cuda = bool(getattr(self, "local_dual_shard", False))
        pipeline_overlap = bool(self.config.get("score_pipeline_overlap", False))
        expert_parallel = self.expert_parallel_all_layers
        network_pipeline = (
            pipeline_overlap and not shared_cuda and not local_cuda
            and not expert_parallel
        )
        ipc_key = f"cuda-ipc-{self.checkpoint_sha256}-{'-'.join(map(str, batch_windows))}"
        shape = (
            len(batch_windows),
            ids.shape[1],
            int(self.student.config.hc_mult),
            int(self.student.config.hidden_size),
        )
        with torch.no_grad():
            if self.rank == 0 and not expert_parallel:
                if pair_parallel:
                    launch_stream = torch.cuda.current_stream(device=self.student.device)
                    pair_streams = [
                        torch.cuda.Stream(device=self.student.device)
                        for _ in pair_windows
                    ]
                    hidden_pairs = []
                    for pair_id_tensor, stream in zip(pair_ids, pair_streams):
                        stream.wait_stream(launch_stream)
                        with torch.cuda.stream(stream):
                            pair_embeds = self.student.model.model.embed_tokens(pair_id_tensor)
                            pair_hidden = pair_embeds.unsqueeze(2).expand(
                                -1, -1, self.student.config.hc_mult, -1
                            ).contiguous()
                            pair_hidden = self._run_layers(pair_hidden, pair_id_tensor)
                            hidden_pairs.append(pair_hidden)
                    embedding_done = time.perf_counter()
                    for stream in pair_streams:
                        launch_stream.wait_stream(stream)
                    hidden = torch.cat(hidden_pairs, dim=0)
                else:
                    embeds = self.student.model.model.embed_tokens(ids)
                    hidden = embeds.unsqueeze(2).expand(
                        -1, -1, self.student.config.hc_mult, -1
                    ).contiguous()
                    torch.cuda.synchronize()
                    embedding_done = time.perf_counter()
                    hidden = self._run_layers(hidden, ids)
                torch.cuda.synchronize()
                layers_done = time.perf_counter()
                if tuple(hidden.shape) != shape or hidden.dtype != torch.bfloat16:
                    raise ArtifactError(f"official-K2 resident activation geometry drift: {tuple(hidden.shape)} {hidden.dtype}")
                hidden = hidden.contiguous()
                if local_cuda:
                    self.local_coordinator.activations[ipc_key] = hidden
                    consumer_wait = 0.0
                elif shared_cuda:
                    from multiprocessing.reduction import ForkingPickler
                    payload = bytes(ForkingPickler.dumps(hidden))
                    if len(payload) > 65536:
                        raise ArtifactError("shared CUDA IPC descriptor unexpectedly contains payload bytes")
                    self.rendezvous_store.set(ipc_key, payload)
                    # Keep one prior activation alive while this rank computes the
                    # next independent batch.  Rank 1 consumes the previous batch
                    # concurrently, turning the two resident halves into a 1F1B
                    # pipeline without changing window order or binary64 fan-in.
                    if pipeline_overlap:
                        if not hasattr(self, "_pipeline_inflight"):
                            self._pipeline_inflight = []
                        self._pipeline_inflight.append((ipc_key, hidden, None))
                        consumer_wait = 0.0
                        if len(self._pipeline_inflight) > 1:
                            previous_key, previous_hidden, previous_work = self._pipeline_inflight.pop(0)
                            wait_started = time.perf_counter()
                            self.rendezvous_store.get(previous_key + "-consumed")
                            if previous_work is not None:
                                previous_work.wait()
                            consumer_wait = time.perf_counter() - wait_started
                            del previous_hidden
                    else:
                        wait_started = time.perf_counter()
                        self.rendezvous_store.get(ipc_key + "-consumed")
                        consumer_wait = time.perf_counter() - wait_started
                elif network_pipeline:
                    work = self._batch_p2p_isend(hidden, dst=1)
                    if not hasattr(self, "_pipeline_inflight"):
                        self._pipeline_inflight = []
                    self._pipeline_inflight.append((ipc_key, hidden, work))
                    consumer_wait = 0.0
                    if len(self._pipeline_inflight) > 1:
                        previous_key, previous_hidden, previous_work = self._pipeline_inflight.pop(0)
                        wait_started = time.perf_counter()
                        self.rendezvous_store.get(previous_key + "-consumed")
                        previous_work.wait()
                        consumer_wait = time.perf_counter() - wait_started
                        del previous_hidden
                else:
                    work = self._batch_p2p_isend(hidden, dst=1)
                    work.wait()
                    consumer_wait = time.perf_counter() - layers_done
                self.last_window_profile = {
                    "batch_windows": list(batch_windows),
                    "rank": self.rank,
                    "embedding_ms": (embedding_done - window_started) * 1000.0,
                    "layer_forward_ms": (layers_done - embedding_done) * 1000.0,
                    "consumer_wait_ms": consumer_wait * 1000.0,
                    "pipeline_inflight_batches": len(getattr(self, "_pipeline_inflight", [])),
                    "network_pipeline": network_pipeline,
                    "sealed_pair_stream_concurrency": len(pair_windows),
                    "wall_seconds": time.perf_counter() - window_started,
                }
                return None
            receive_started = time.perf_counter()
            if expert_parallel:
                embeds = self.student.model.model.embed_tokens(ids)
                hidden = embeds.unsqueeze(2).expand(
                    -1, -1, self.student.config.hc_mult, -1
                ).contiguous()
            elif local_cuda:
                self.local_coordinator.rank0._score_window(reported_windows)
                hidden = self.local_coordinator.activations.pop(ipc_key)
            elif shared_cuda:
                import pickle
                hidden = pickle.loads(self.rendezvous_store.get(ipc_key))
                if not isinstance(hidden, torch.Tensor) or not hidden.is_cuda:
                    raise ArtifactError("shared CUDA IPC transport produced a non-CUDA activation")
                if tuple(hidden.shape) != shape or hidden.dtype != torch.bfloat16:
                    raise ArtifactError("shared CUDA IPC activation geometry drift")
            else:
                hidden = torch.empty(shape, dtype=torch.bfloat16, device=self.student.device)
                self._batch_p2p_recv(hidden, src=0)
            receive_done = time.perf_counter()
            if expert_parallel:
                hidden = self._run_layers(hidden, ids)
                if self.rank == 0:
                    torch.cuda.synchronize()
                    layers_done = time.perf_counter()
                    self.last_window_profile = {
                        "batch_windows": list(batch_windows),
                        "rank": self.rank,
                        "activation_wait_ms": 0.0,
                        "layer_forward_ms": (layers_done - receive_done) * 1000.0,
                        "expert_parallel_all_layers": True,
                        "wall_seconds": layers_done - window_started,
                    }
                    return None
                final = self.student.model.model.norm(
                    self.student.model.model.hc_head(hidden)
                )
            elif pair_parallel:
                launch_stream = torch.cuda.current_stream(device=self.student.device)
                pair_streams = [
                    torch.cuda.Stream(device=self.student.device)
                    for _ in pair_windows
                ]
                final_pairs = []
                for pair_hidden, pair_id_tensor, stream in zip(
                    hidden.split(2, dim=0), pair_ids, pair_streams
                ):
                    stream.wait_stream(launch_stream)
                    with torch.cuda.stream(stream):
                        pair_output = self._run_layers(pair_hidden, pair_id_tensor)
                        pair_final = self.student.model.model.norm(
                            self.student.model.model.hc_head(pair_output)
                        )
                        final_pairs.append(pair_final)
                for stream in pair_streams:
                    launch_stream.wait_stream(stream)
                final = torch.cat(final_pairs, dim=0)
            else:
                hidden = self._run_layers(hidden, ids)
                final = self.student.model.model.norm(
                    self.student.model.model.hc_head(hidden)
                )
            torch.cuda.synchronize()
            layers_done = time.perf_counter()
            readout_done = layers_done
            score_rows: list[tuple[list[float], int]] = []
            logits_seconds = 0.0
            teacher_gather_seconds = 0.0
            binary64_reduce_seconds = 0.0
            reduce_done_previous = readout_done
            for batch_index, window in enumerate(batch_windows):
                teacher_idx, teacher_logprob = self.teacher_cache[window]
                count = POSITIONS_PER_WINDOW
                # Keep the vocabulary projection rank-1-at-a-time to bound peak
                # memory; the expensive resident layer stack above still reuses
                # each layer's weights across the independent window batch.
                logits = self.student.model.lm_head(final[batch_index, :count].to(torch.bfloat16)).float()
                logprob = torch.log_softmax(logits, dim=-1)
                q_argmax = logprob.argmax(-1).to(torch.int64).cpu().numpy()
                torch.cuda.synchronize()
                logits_done = time.perf_counter()
                idx_device = teacher_idx.to(device=self.student.device, non_blocking=False)
                q_lp = logprob.gather(1, idx_device).to(torch.float16).cpu().numpy().astype(self.np.float64, copy=False)
                ref_lp = teacher_logprob.numpy().astype(self.np.float64, copy=False)
                idx0 = teacher_idx[:, 0].numpy()
                teacher_done = time.perf_counter()
                ref_max = self.np.max(ref_lp, axis=1, keepdims=True)
                cand_max = self.np.max(q_lp, axis=1, keepdims=True)
                ref_norm = ref_lp - (ref_max + self.np.log(self.np.sum(self.np.exp(ref_lp - ref_max), axis=1, dtype=self.np.float64, keepdims=True)))
                cand_norm = q_lp - (cand_max + self.np.log(self.np.sum(self.np.exp(q_lp - cand_max), axis=1, dtype=self.np.float64, keepdims=True)))
                terms = self.np.sum(self.np.exp(ref_norm) * (ref_norm - cand_norm), axis=1, dtype=self.np.float64)
                if not self.np.isfinite(terms).all():
                    raise ArtifactError(f"official-K2 resident non-finite KLD at window {window}")
                term_values = [float(value) for value in self.np.asarray(terms).reshape(-1).tolist()]
                score_rows.append((term_values, int(self.np.count_nonzero(q_argmax == idx0))))
                reduce_done = time.perf_counter()
                logits_seconds += (
                    logits_done - readout_done
                    if batch_index == 0
                    else logits_done - reduce_done_previous
                )
                teacher_gather_seconds += teacher_done - logits_done
                binary64_reduce_seconds += reduce_done - teacher_done
                reduce_done_previous = reduce_done
            glue_started = time.perf_counter()
            if (shared_cuda and not local_cuda) or network_pipeline:
                torch.cuda.synchronize()
                self.rendezvous_store.set(ipc_key + "-consumed", "1")
            glue_done = time.perf_counter()
            self.last_window_profile = {
                "batch_windows": list(batch_windows),
                "rank": self.rank,
                "activation_wait_ms": (receive_done - receive_started) * 1000.0,
                "layer_forward_ms": (layers_done - receive_done) * 1000.0,
                "readout_ms": (readout_done - layers_done) * 1000.0,
                "logits_ms": logits_seconds * 1000.0,
                "teacher_gather_ms": teacher_gather_seconds * 1000.0,
                "binary64_reduce_ms": binary64_reduce_seconds * 1000.0,
                "glue_ms": (glue_done - glue_started) * 1000.0,
                "sealed_pair_stream_concurrency": len(pair_windows),
                "wall_seconds": glue_done - window_started,
            }
            row_by_window = dict(zip(batch_windows, score_rows))
            return [row_by_window[window] for window in reported_windows]

    def score(self) -> dict[str, Any]:
        resume = self._load_score_resume()
        completed_before = int(resume["completed_windows"])
        resume_identity = {
            "rank": self.rank,
            "completed_windows": completed_before,
            "implementation_sha256": resume["implementation_sha256"],
        }
        if bool(getattr(self, "local_dual_shard", False)):
            peer_resume = self.local_coordinator.rank0._load_score_resume()
            resume_rows = [
                {
                    "rank": 0,
                    "completed_windows": int(peer_resume["completed_windows"]),
                    "implementation_sha256": peer_resume["implementation_sha256"],
                },
                resume_identity,
            ]
        else:
            resume_rows: list[Any] = [None, None]
            self.dist.all_gather_object(resume_rows, resume_identity)
        if (
            any(not isinstance(row, Mapping) for row in resume_rows)
            or {int(row["completed_windows"]) for row in resume_rows} != {completed_before}
            or {str(row["implementation_sha256"]) for row in resume_rows}
            != {self._score_implementation_sha256()}
        ):
            raise ArtifactError(f"official-K2 score resume rank drift: {resume_rows}")
        started = time.perf_counter()
        all_terms: list[float] = [float(value) for value in resume["terms"]]
        top1 = int(resume["top1"])
        per_window = list(resume["per_window"])
        previous_cumulative_wall = float(resume["cumulative_scoring_wall_seconds"])
        shared_cuda = bool(self.config.get("shared_cuda_device_process_group", False))
        batch_size = _effective_score_window_batch_size(
            int(self.config.get("score_window_batch_size", 1)), len(self.windows)
        )
        pair_stream_concurrency = int(
            self.config.get("score_pair_stream_concurrency", 1)
        )
        if pair_stream_concurrency > 1:
            if batch_size != 2 or completed_before % 2:
                raise ArtifactError(
                    "official-K2 pair concurrency requires the exact sealed mb=2 frontier"
                )
            scheduled_batches = tuple(
                tuple(window for pair in group for window in pair)
                for group in _sealed_pair_groups(
                    self.windows[completed_before:], pair_stream_concurrency
                )
            )
        else:
            scheduled_batches = tuple(
                self.windows[start : start + batch_size]
                for start in range(completed_before, len(self.windows), batch_size)
            )
        batch_phase_profiles: list[dict[str, Any]] = []
        rank0_local_phase_profiles: list[dict[str, Any]] = []
        start = completed_before
        for batch_windows in scheduled_batches:
            rows = self._score_window(batch_windows)
            self.torch.cuda.synchronize()
            batch_phase_profiles.append(dict(self.last_window_profile))
            if bool(getattr(self, "local_dual_shard", False)):
                coordinator = self.local_coordinator
                if coordinator is None or coordinator.rank0 is None:
                    raise ArtifactError("local dual-shard phase profile coordinator drift")
                rank0_local_phase_profiles.append(
                    dict(coordinator.rank0.last_window_profile)
                )
            if self.rank == 1:
                if rows is None or len(rows) != len(batch_windows):
                    raise ArtifactError("official-K2 resident rank1 produced no score row")
                for offset, (terms, matches) in enumerate(rows):
                    window = batch_windows[offset]
                    all_terms.extend(terms)
                    top1 += matches
                    per_window.append({
                        "ordinal": start + offset,
                        "window": window,
                        "positions": len(terms),
                        "support": SUPPORT,
                        "kld_sum_binary64": math.fsum(terms),
                        "top1": matches,
                    })
            completed_now = start + len(batch_windows)
            cumulative_now = previous_cumulative_wall + (time.perf_counter() - started)
            self._persist_score_resume(
                completed_windows=completed_now,
                terms=all_terms if self.rank == 1 else [],
                top1=top1 if self.rank == 1 else 0,
                per_window=per_window if self.rank == 1 else [],
                cumulative_scoring_wall_seconds=cumulative_now,
            )
            if bool(getattr(self, "local_dual_shard", False)):
                self.local_coordinator.rank0._persist_score_resume(
                    completed_windows=completed_now,
                    terms=[],
                    top1=0,
                    per_window=[],
                    cumulative_scoring_wall_seconds=cumulative_now,
                )
            _atomic_json(
                self.parent_root.parent / f"SCORE_PROGRESS_RANK{self.rank}.json",
                {
                    "schema": "official-k2-resident-score-progress-v1",
                    "status": "RUNNING",
                    "public_api_method": PUBLIC_API_METHOD,
                    "public_api_version": PUBLIC_API_VERSION,
                    "rank": self.rank,
                    "checkpoint_sha256": self.checkpoint_sha256,
                    "completed_windows": completed_now,
                    "resumed_windows": completed_before,
                    "total_windows": len(self.windows),
                    "last_window": batch_windows[-1],
                    "last_window_profile": dict(self.last_window_profile),
                    "elapsed_seconds": time.perf_counter() - started,
                    "cumulative_scoring_wall_seconds": cumulative_now,
                },
            )
            start = completed_now
        if self.rank == 0 and bool(self.config.get("score_pipeline_overlap", False)):
            self._drain_pipeline_inflight()
        self.torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        if bool(getattr(self, "local_dual_shard", False)):
            rank_phase_profiles = [rank0_local_phase_profiles, batch_phase_profiles]
        else:
            rank_phase_envelopes: list[Any] = [None, None]
            self.dist.all_gather_object(
                rank_phase_envelopes, {"profiles": batch_phase_profiles}
            )
            if any(
                not isinstance(row, Mapping) or not isinstance(row.get("profiles"), list)
                for row in rank_phase_envelopes
            ):
                raise ArtifactError("official-K2 score phase profile envelope drift")
            rank_phase_profiles = [row["profiles"] for row in rank_phase_envelopes]
        phase_profile = _aggregate_score_phase_profiles(
            rank_phase_profiles,
            ordered_windows=self.windows[completed_before:],
            post_load_wall_seconds=elapsed,
            configured_batch_size=batch_size,
        )
        result: dict[str, Any] | None = None
        if self.rank == 1:
            positions = len(all_terms)
            if positions != len(self.windows) * POSITIONS_PER_WINDOW:
                raise ArtifactError(f"official-K2 resident position closure drift: {positions}")
            result = {
                "kld_mean": math.fsum(all_terms) / positions,
                "top1": top1,
                "positions": positions,
                "support": SUPPORT,
                "windows": list(self.windows),
                "per_window": per_window,
                "scoring_wall_seconds": elapsed,
                "cumulative_scoring_wall_seconds": previous_cumulative_wall + elapsed,
                "resumed_windows": completed_before,
                "phase_profile": phase_profile,
            }
        if not bool(getattr(self, "local_dual_shard", False)):
            rows = [result]
            self.dist.broadcast_object_list(rows, src=1)
            result = rows[0]
        if not isinstance(result, Mapping):
            raise ArtifactError("official-K2 resident rank fan-in produced no result")
        read_delta = self.read_counter.delta(self.ready_counter)
        local_terminal = {
            "rank": self.rank,
            "timed_score_file_reads": read_delta,
            "file_read_paths": list(self.read_counter.paths),
            "fallback_calls": int(self.status.get("fallback_calls", 0)),
            "reconstruction_calls": int(self.status.get("reconstruction_calls", 0)),
            "reference_fwht_calls": int(self.status.get("reference_fwht_calls", 0)),
            "cpu_relay_bytes": int(self.status.get("cpu_relay_bytes", 0)),
        }
        if bool(getattr(self, "local_dual_shard", False)):
            rank0 = self.local_coordinator.rank0
            terminals = [{
                "rank": 0,
                "timed_score_file_reads": rank0.read_counter.delta(rank0.ready_counter),
                "file_read_paths": list(rank0.read_counter.paths),
                "fallback_calls": int(rank0.status.get("fallback_calls", 0)),
                "reconstruction_calls": int(rank0.status.get("reconstruction_calls", 0)),
                "reference_fwht_calls": int(rank0.status.get("reference_fwht_calls", 0)),
                "cpu_relay_bytes": int(rank0.status.get("cpu_relay_bytes", 0)),
            }, local_terminal]
        else:
            terminals: list[Any] = [None, None]
            self.dist.all_gather_object(terminals, local_terminal)
        forbidden = [row for row in terminals if any(int(row[key]) != 0 for key in (
            "timed_score_file_reads", "fallback_calls", "reconstruction_calls", "reference_fwht_calls", "cpu_relay_bytes"
        ))]
        if forbidden:
            raise ArtifactError(f"official-K2 resident terminal closure failed: {forbidden}")
        return {
            **dict(result),
            "execution_mode": "resident_in_memory",
            "resident_load_seconds": max(float(row["resident_load_seconds"]) for row in self.resident_ready),
            "resident_ready": self.resident_ready,
            "memory_preflight": [dict(row["memory_preflight"]) for row in self.resident_ready],
            "phase_profile": dict(result["phase_profile"]),
            "rank_terminal": terminals,
            "timed_score_file_reads": 0,
        }

    def close(self) -> None:
        if self.dist.is_initialized() and self.config.get("destroy_process_group", False):
            self.dist.destroy_process_group()


class _LocalDualShardCoordinator:
    """Own two local shard objects and pass CUDA activations by Python reference."""

    def __init__(self) -> None:
        self.storage: dict[int, Mapping[str, Any]] = {}
        self.activations: dict[str, Any] = {}
        self.rank0: OfficialK2ResidentRankEngine | None = None
        self.post_rank0_workspace_release: dict[str, Any] = {}


class OfficialK2LocalDualShardEngine:
    """Two resident layer shards in one process on one local CUDA device."""

    def __init__(
        self,
        *,
        payload: Mapping[str, Any],
        checkpoint_sha256: str,
        checkpoint_identity_sha256: str,
        windows: tuple[int, ...],
        config: Mapping[str, Any],
    ) -> None:
        coordinator = _LocalDualShardCoordinator()
        self.coordinator = coordinator
        common = dict(config)
        common.update({
            "same_process_dual_shard": True,
            "local_dual_shard_coordinator": coordinator,
            "shared_cuda_device_process_group": True,
            "ordinary_load_fork_broker": True,
            "checkpoint_mmap": False,
            "gpu_resident_storage_broker": True,
            "score_pipeline_overlap": False,
        })
        rank0_config = dict(common, rank=0)
        rank1_resume_root = Path(str(config["score_resume_root"])).expanduser().resolve()
        rank0_config["score_resume_root"] = str(config.get(
            "local_rank0_score_resume_root", rank1_resume_root.parent / "rank0"
        ))
        self.rank0 = OfficialK2ResidentRankEngine(
            payload=payload,
            checkpoint_sha256=checkpoint_sha256,
            checkpoint_identity_sha256=checkpoint_identity_sha256,
            windows=windows,
            config=rank0_config,
        )
        coordinator.rank0 = self.rank0
        # The rank-0 constructor already drops its transient workspace, but the
        # completed constructor frame can be the final owner of a few allocator
        # blocks on unified-memory CUDA.  Release once more at the exact
        # rank-boundary before rank 1 evaluates its strict 4 GiB reserve gate.
        self.rank0._release_transient_resident_load_workspace()
        cuda = self.rank0.torch.cuda
        free_bytes, total_bytes = cuda.mem_get_info(self.rank0.device)
        expected, expected_policy = resolve_rank_local_bytes(
            common.get("estimated_resident_bytes_by_rank", EXPECTED_RESIDENT_BYTES),
            1,
            default=EXPECTED_RESIDENT_BYTES[1],
            field="estimated_resident_bytes_by_rank",
        )
        reserve, reserve_policy = resolve_rank_local_bytes(
            common.get("cuda_reserve_bytes"),
            1,
            default=DEFAULT_CUDA_RESERVE_BYTES,
            field="cuda_reserve_bytes",
        )
        coordinator.post_rank0_workspace_release = {
            "status": "PASS",
            "cuda_free_bytes": int(free_bytes),
            "cuda_total_bytes": int(total_bytes),
            "required_cuda_free_bytes": int(expected + reserve),
            "margin_bytes": int(free_bytes - expected - reserve),
            "estimated_resident_bytes": int(expected),
            "estimated_resident_policy": expected_policy,
            "reserve_bytes": int(reserve),
            "reserve_policy": reserve_policy,
            "allocator_release": dict(self.rank0.transient_load_memory_release),
        }
        self.rank1 = OfficialK2ResidentRankEngine(
            payload=payload,
            checkpoint_sha256=checkpoint_sha256,
            checkpoint_identity_sha256=checkpoint_identity_sha256,
            windows=windows,
            config=dict(common, rank=1),
        )
        ready = [self.rank0.local_ready, self.rank1.local_ready]
        self.rank0.resident_ready = ready
        self.rank1.resident_ready = ready

    def score(self) -> dict[str, Any]:
        result = self.rank1.score()
        result["execution_mode"] = "resident_in_memory"
        result["activation_transport"] = "local_cuda_reference"
        result["post_rank0_workspace_release"] = dict(
            self.coordinator.post_rank0_workspace_release
        )
        return result

    def parity_tap(self, window: int) -> dict[str, Any]:
        return self.rank1.parity_tap(window)

    def _release_post_bind_checkpoint_workspace(self) -> None:
        return None

    def _score_implementation_sha256(self) -> str:
        return self.rank1._score_implementation_sha256()

    def rebind_checkpoint(self, **kwargs: Any) -> None:
        raise ArtifactError("same-process dual shard rebind requires a fresh score process")

    def close(self) -> None:
        self.rank1.close()
        self.rank0.close()


class OfficialK2ResidentScorer:
    """Artifact-bound backend used automatically by ResidentRepairAPI.score."""

    def __init__(self, artifact: RepairArtifact, config: Mapping[str, Any]):
        forbidden = {
            "remote", "shard_buf", "planes_dir", "builder", "builder_template",
            "candidate_output", "safe_open", "reconstruction", "fallback",
            "reference_fwht", "cpu_relay", "subprocess", "launcher",
        }
        present = sorted(forbidden & set(config))
        if present:
            raise ArtifactError(
                f"forbidden non-resident official-K2 score configuration: {present}"
            )
        self.artifact = artifact
        self.config = dict(config)
        self._checkpoint_loads = 0
        self._engine: Any | None = None

    def _engine_type(self) -> type[Any]:
        if bool(self.config.get("same_process_dual_shard", False)) or os.environ.get(
            "BANANA_SMASHER_SAME_PROCESS_DUAL_SHARD"
        ) == "1":
            return OfficialK2LocalDualShardEngine
        return OfficialK2ResidentRankEngine

    def bind_routed_k2(self, route: Mapping[str, Any]) -> None:
        """Bind exact routed admission without reconstructing the resident rail."""
        validate_routed_k2_closure(route)
        if self._engine is None:
            raise ArtifactError("routed-K2 reuse requires an already resident canonical rail")
        self.config.update(dict(route))
        self.config["route_kind"] = ROUTED_K2_ROUTE_KIND

    def parity_tap(self, checkpoint: str, window: int) -> dict[str, Any]:
        """Load one exact resident window and emit diagnostic tensor taps."""
        selected = (int(window),)
        if selected[0] not in self.artifact.windows:
            raise ArtifactError("parity_tap window is not declared by the artifact")
        meta = self.artifact.manifest["checkpoints"][checkpoint]
        checkpoint_path = self.artifact.checkpoint_path(checkpoint)
        checkpoint_sha = str(meta.get("sha256", ""))
        checkpoint_identity_sha = str(meta.get("identity_sha256", ""))
        update = int(meta.get("next_update", meta.get("update", -1)))
        if not checkpoint_sha or not checkpoint_identity_sha or update < 0:
            raise ArtifactError("parity_tap checkpoint identity is incomplete")
        if str(self.config.get("basis_sha256", BASIS_SHA256)) != BASIS_SHA256:
            raise ArtifactError("parity_tap basis mismatch")
        self.config["checkpoint_path"] = str(checkpoint_path)
        pre_receipt = Path(str(self.config.get(
            "pre_calibration_receipt",
            self.artifact.root / "receipts" / "CANONICAL_RESIDENT_CALIBRATION.json",
        )))
        parent_sha = meta.get("parent_sha256") or meta.get("parent_checkpoint_sha256")
        if self.config.get("route_kind") == ROUTED_K2_ROUTE_KIND:
            authorize_routed_k2_score(
                update,
                checkpoint_sha256=checkpoint_sha,
                checkpoint_identity_sha256=checkpoint_identity_sha,
                checkpoint_parent_sha256=parent_sha,
                route=self.config,
            )
        else:
            authorize_production_score(
                update,
                checkpoint_sha256=checkpoint_sha,
                checkpoint_parent_sha256=parent_sha,
                pre_calibration_receipt=pre_receipt if pre_receipt.is_file() else None,
                scientific_question_receipt=self.config.get("scientific_question_receipt"),
                ordered_windows_sha256=_windows_sha256(selected),
                allow_alternate_pre_diagnostic=(
                    self.config.get("parity_tap_mode") == "sealed_reference"
                ),
            )
        payload = _load_score_checkpoint(checkpoint_path, checkpoint_sha, self.config)
        self._checkpoint_loads += 1
        payload_identity = payload.get("identity") if isinstance(payload, Mapping) else None
        identity_missing = not isinstance(payload_identity, Mapping) or any(
            payload_identity.get(field) in (None, "")
            for field in ("identity_sha256", "checkpoint_sha256", "next_update", "checkpoint_loaded")
        )
        if update == 0 and checkpoint_sha == CANONICAL_U0_CHECKPOINT_SHA256 and identity_missing:
            payload = adapt_canonical_raw_u0_payload(
                payload,
                artifact_root=self.artifact.root,
                manifest=self.artifact.manifest,
                checkpoint_path=checkpoint_path,
                checkpoint_key=checkpoint,
                config=self.config,
            )
        elif update == 1 and checkpoint_sha == CANONICAL_U1_CHECKPOINT_SHA256 and identity_missing:
            payload = adapt_canonical_raw_u1_payload(
                payload,
                artifact_root=self.artifact.root,
                manifest=self.artifact.manifest,
                checkpoint_path=checkpoint_path,
                checkpoint_key=checkpoint,
            )
        validate_payload_identity(
            payload,
            checkpoint_sha256=checkpoint_sha,
            checkpoint_identity_sha256=checkpoint_identity_sha,
            next_update=update,
        )
        if self._engine is None:
            self._engine = self._engine_type()(
                payload=payload,
                checkpoint_sha256=checkpoint_sha,
                checkpoint_identity_sha256=checkpoint_identity_sha,
                windows=selected,
                config=self.config,
            )
        else:
            raise ArtifactError("parity_tap backend is one-window/one-checkpoint only")
        return self._engine.parity_tap(selected[0])

    def score(self, checkpoint: str, windows: Iterable[int]) -> ScoreResult:
        selected = tuple(int(value) for value in windows)
        admitted_ep_canary = tuple(
            int(value) for value in self.config.get("expert_parallel_canary_windows", ())
        )
        if (
            bool(self.config.get("expert_parallel_all_layers", False))
            and len(admitted_ep_canary) == 2
            and 28 not in admitted_ep_canary
            and selected == admitted_ep_canary
        ):
            geometry = "EXPERT_PARALLEL_CANARY"
        else:
            geometry = _validate_public_score_windows(selected, self.artifact.windows)
        if geometry == "W28_CANARY":
            self.config["physical_canary_windows"] = list(_physical_canary_batch_windows(
                selected,
                int(self.config.get("score_window_batch_size", 1)),
                self.artifact.windows,
            ))
        else:
            self.config.pop("physical_canary_windows", None)
        meta = self.artifact.manifest["checkpoints"][checkpoint]
        checkpoint_path = self.artifact.checkpoint_path(checkpoint)
        checkpoint_sha = str(meta.get("sha256", ""))
        checkpoint_identity_sha = str(meta.get("identity_sha256", ""))
        update = int(meta.get("next_update", meta.get("update", -1)))
        if not checkpoint_sha or not checkpoint_identity_sha or update < 0:
            raise ArtifactError("official-K2 resident checkpoint identity is incomplete")
        if str(self.config.get("basis_sha256", BASIS_SHA256)) != BASIS_SHA256:
            raise ArtifactError("official-K2 resident manifest basis mismatch")
        declared_pre = self.config.get("pre_checkpoint_sha256")
        if (
            declared_pre
            and str(declared_pre) != CANONICAL_U0_CHECKPOINT_SHA256
            and self.config.get("route_kind") != ROUTED_K2_ROUTE_KIND
        ):
            raise ArtifactError(
                "alternate serialized PRE is quarantine-only and cannot be a production prerequisite"
            )
        self.config["checkpoint_path"] = str(checkpoint_path)
        pre_receipt = Path(str(self.config.get(
            "pre_calibration_receipt", self.artifact.root / "receipts" / "CANONICAL_RESIDENT_CALIBRATION.json"
        )))
        parent_sha = meta.get("parent_sha256") or meta.get("parent_checkpoint_sha256")
        if self.config.get("route_kind") == ROUTED_K2_ROUTE_KIND:
            admission = authorize_routed_k2_score(
                update,
                checkpoint_sha256=checkpoint_sha,
                checkpoint_identity_sha256=checkpoint_identity_sha,
                checkpoint_parent_sha256=parent_sha,
                route=self.config,
            )
        else:
            admission = authorize_production_score(
                update,
                checkpoint_sha256=checkpoint_sha,
                checkpoint_parent_sha256=parent_sha,
                pre_calibration_receipt=pre_receipt if pre_receipt.is_file() else None,
                scientific_question_receipt=self.config.get("scientific_question_receipt"),
                ordered_windows_sha256=_windows_sha256(selected),
            )
        if self._engine is None and self._engine_type() is OfficialK2ResidentRankEngine:
            # Checkpoint adaptation can materialize CUDA tensors before the
            # rank engine exists. Gate that allocation as well as the later
            # resident conversion so rank-local peaks cannot overlap.
            _wait_for_cold_load_gate(self.config, int(self.config["rank"]))
        payload = _load_score_checkpoint(checkpoint_path, checkpoint_sha, self.config)
        self._checkpoint_loads += 1
        payload_identity = payload.get("identity") if isinstance(payload, Mapping) else None
        raw_u0_identity_missing = not isinstance(payload_identity, Mapping) or any(
            payload_identity.get(field) in (None, "")
            for field in ("identity_sha256", "checkpoint_sha256", "next_update", "checkpoint_loaded")
        )
        if update == 0 and checkpoint_sha == CANONICAL_U0_CHECKPOINT_SHA256 and raw_u0_identity_missing:
            payload = adapt_canonical_raw_u0_payload(
                payload,
                artifact_root=self.artifact.root,
                manifest=self.artifact.manifest,
                checkpoint_path=checkpoint_path,
                checkpoint_key=checkpoint,
                config=self.config,
            )
        elif update == 1 and checkpoint_sha == CANONICAL_U1_CHECKPOINT_SHA256 and raw_u0_identity_missing:
            payload = adapt_canonical_raw_u1_payload(
                payload,
                artifact_root=self.artifact.root,
                manifest=self.artifact.manifest,
                checkpoint_path=checkpoint_path,
                checkpoint_key=checkpoint,
            )
        validate_payload_identity(
            payload,
            checkpoint_sha256=checkpoint_sha,
            checkpoint_identity_sha256=checkpoint_identity_sha,
            next_update=update,
        )
        if self._engine is None:
            self._engine = self._engine_type()(
                payload=payload,
                checkpoint_sha256=checkpoint_sha,
                checkpoint_identity_sha256=checkpoint_identity_sha,
                windows=selected,
                config=self.config,
            )
        else:
            self._engine.rebind_checkpoint(
                payload=payload,
                checkpoint_sha256=checkpoint_sha,
                checkpoint_identity_sha256=checkpoint_identity_sha,
                admission=admission,
            )
        engine = self._engine
        payload_identity = payload.get("identity") if isinstance(payload, Mapping) else None
        # The rank engine has copied or adopted every checkpoint tensor it owns.
        # Drop the scorer's outer payload mapping before scoring so rank 0 does
        # not retain a second full checkpoint bank beside resident weights.
        del payload
        engine._release_post_bind_checkpoint_workspace()
        measured = engine.score()
        if measured.get("execution_mode") != "resident_in_memory" or measured.get("timed_score_file_reads") != 0:
            raise ArtifactError("official-K2 resident backend did not close execution mode/file-read gates")
        quality_status = {
            "CANONICAL_U0": "CANONICAL_U0_ADMITTED",
            "CANONICAL_U1_IMMEDIATE_PARENT": "CANONICAL_U1_ADMITTED",
        }.get(admission["scope"], "MECHANISM_PASS_QUALITY_UNPROMOTED")
        positions = int(measured["positions"])
        identity_provenance = (
            dict(payload_identity.get("runtime_load_provenance", {}))
            if isinstance(payload_identity, Mapping)
            else {}
        )
        identity = {
            "basis_sha256": BASIS_SHA256,
            "model_index_sha256": BASIS_SHA256,
            "checkpoint_sha256": checkpoint_sha,
            "checkpoint_identity_sha256": checkpoint_identity_sha,
            "official_physical_layer_sha256": OFFICIAL_PHYSICAL_LAYER_SHA256,
            "builder_eval_corpus_sha256": BUILDER_EVAL_CORPUS_SHA256,
            "train_score_corpus_sha256": SCORE_TRAIN_CORPUS_SHA256,
            "teacher_inventory_sha256": TEACHER_INVENTORY_SHA256,
            "source_context_tokens": SOURCE_CONTEXT_TOKENS,
            "scored_positions_per_window": POSITIONS_PER_WINDOW,
            "ordered_balanced64_windows": list(selected),
            "ordered_windows_sha256": _windows_sha256(selected),
            "quality_status": quality_status,
            "production_admission": admission,
            "public_api": {"method": PUBLIC_API_METHOD, "version": PUBLIC_API_VERSION},
        }
        if isinstance(payload_identity, Mapping) and payload_identity.get("source"):
            identity["checkpoint_identity_source"] = payload_identity["source"]
            identity["checkpoint_identity_provenance"] = identity_provenance
        return ScoreResult(
            checkpoint=checkpoint,
            windows=selected,
            positions=positions,
            support=SUPPORT,
            kld=float(measured["kld_mean"]),
            top1=int(measured["top1"]),
            top1_rate=int(measured["top1"]) / positions,
            artifact_root=str(self.artifact.root),
            spec=BALANCED64_SPEC,
            candidate_dir="fully-resident-official-k2",
            execution_mode="resident_in_memory",
            resident_load_seconds=float(measured["resident_load_seconds"]),
            timed_wall_seconds=float(measured["scoring_wall_seconds"]),
            identity=identity,
            runtime_counters={
                "checkpoint_loads": self._checkpoint_loads,
                "resident_engine_loads": 1,
                "resident_checkpoint_rebinds": max(0, self._checkpoint_loads - 1),
                "timed_score_file_reads": 0,
                "score_resumed_windows": int(measured["resumed_windows"]),
                "cumulative_scoring_wall_seconds": float(measured["cumulative_scoring_wall_seconds"]),
                "scoring_phase_profile": dict(measured["phase_profile"]),
                "score_resume_implementation_sha256": engine._score_implementation_sha256(),
                "attention_implementation": _configured_attention_implementation(
                    getattr(self, "config", {})
                ),
                "file_reads_during_timed_score": 0,
                "payload_model_file_read_delta": 0,
                "fallback_calls": 0,
                "reconstruction_calls": 0,
                "reference_fwht_calls": 0,
                "cpu_relay_bytes": 0,
                "layer_streaming_calls": 0,
                "cold_source_prune_mode": "post-resident-load-pre-score",
                "resident_ready": measured["resident_ready"],
                "memory_preflight": measured["memory_preflight"],
                "rank_terminal": measured["rank_terminal"],
            },
        )


__all__ = [
    "OfficialK2ResidentRankEngine",
    "OfficialK2ResidentScorer",
    "ALTERNATE_PRE_CHECKPOINT_SHA256",
    "CANONICAL_U0_CHECKPOINT_SHA256",
    "CANONICAL_U0_LOCK_CORPUS_SHA256",
    "CANONICAL_U0_IDENTITY_SHA256",
    "CANONICAL_U0_LOCK_SHA256",
    "CANONICAL_U0_TRAJECTORY_SHA256",
    "CANONICAL_U0_SOURCE_PATH",
    "CANONICAL_CORPUS_SHA256",
    "CANONICAL_U1_CHECKPOINT_SHA256",
    "CANONICAL_U1_IDENTITY_SHA256",
    "ROUTED_K2_API_METHOD",
    "ROUTED_K2_API_VERSION",
    "ROUTED_K2_CLOSURE",
    "ROUTED_K2_ROUTE_KIND",
    "PUBLIC_API_METHOD",
    "PUBLIC_API_VERSION",
    "DEFAULT_CUDA_RESERVE_BYTES",
    "CUDA_RESERVE_BYTES",
    "EXPECTED_RESIDENT_BYTES",
    "_aggregate_score_phase_profiles",
    "resolve_rank_local_bytes",
    "authorize_routed_k2_score",
    "authorize_production_score",
    "adapt_canonical_raw_u0_payload",
    "adapt_canonical_raw_u1_payload",
    "load_canonical_raw_u0",
    "enforce_pre_canary",
    "require_canonical_calibration",
    "require_pre_calibration",
    "validate_payload_identity",
    "validate_routed_k2_closure",
]
