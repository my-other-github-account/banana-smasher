"""Deterministic public admission for physical resident artifacts."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Mapping

from .artifact_identity import ArtifactIdentity
from .production_rails import ALL_LAYERS, PIPELINE_MICROBATCH, PRODUCTION_RAILS_SCHEMA, ProductionRails
from .resident_balanced64 import RepairArtifact

ADMISSION_SPEC_SCHEMA = "banana-smasher-resident-admission-spec-v1"
ADMISSION_RECEIPT_SCHEMA = "banana-smasher-resident-admission-v1"
SHARED_CONTINUATION_BINDING_FIELDS = (
    "basis_sha256",
    "base_source_sha256",
    "trainer_source_sha256",
    "resident_expert_source_sha256",
    "fast_k2_wrapper_source_sha256",
    "fast_k2_extension_sha256",
    "fast_k2_module_name",
    "member_roster_sha256",
    "corpus_sha256",
    "train_corpus_sha256",
    "score_corpus_sha256",
    "shared_optimizer_scheduler_lineage",
    "world_size",
    "layer_split",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha(value: object, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return value


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    payload = (json.dumps(dict(value), indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    path.parent.mkdir(parents=True, exist_ok=True)
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


def prebuilt_uniform_builder(**_: Any) -> None:
    raise RuntimeError("admitted resident artifact is prebuilt; uniform construction is unavailable")


def prebuilt_backpack_mixer(**_: Any) -> None:
    raise RuntimeError("admitted resident artifact is prebuilt; Backpack mixing is unavailable")


def provider_binding(spec: Mapping[str, Any]) -> tuple[dict[str, Any], str]:
    provider = spec.get("provider", {})
    if not isinstance(provider, Mapping):
        raise ValueError("spec.provider must be an object")
    continuations = spec.get("continuations", {})
    if not isinstance(continuations, Mapping):
        raise ValueError("spec.continuations must be an object")
    rank_rows: list[Mapping[str, Any]] = []
    for rank in (0, 1):
        row = continuations.get(str(rank))
        if not isinstance(row, Mapping):
            raise ValueError("continuations must bind both ranks")
        rank_rows.append(row)
    continuation_science = {
        field: rank_rows[0].get(field) for field in SHARED_CONTINUATION_BINDING_FIELDS
    }
    if any(
        row.get(field) != continuation_science[field]
        for row in rank_rows[1:]
        for field in SHARED_CONTINUATION_BINDING_FIELDS
    ):
        raise ValueError("continuations scientific binding mismatch")
    fields = {
        "schema": PRODUCTION_RAILS_SCHEMA,
        "pipeline_microbatch": PIPELINE_MICROBATCH,
        "layers": list(ALL_LAYERS),
        "uniform_builder": str(provider.get("uniform_builder", "banana_smasher.resident_admission:prebuilt_uniform_builder")),
        "backpack_mixer": str(provider.get("backpack_mixer", "banana_smasher.resident_admission:prebuilt_backpack_mixer")),
        "score_contract": dict(spec.get("score", {})),
        "continuation_science": continuation_science,
    }
    digest = hashlib.sha256(json.dumps(fields, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return fields, digest


def admit_resident_artifact(
    spec_path: str | Path,
    output_root: str | Path,
    *,
    checkpoint: str | Path,
    checkpoint_sha256: str,
) -> dict[str, Any]:
    """Generate and verify one identity plus rank configs from authenticated bytes."""
    spec_file = Path(spec_path).expanduser().resolve()
    spec_bytes = spec_file.read_bytes()
    spec = json.loads(spec_bytes)
    if not isinstance(spec, Mapping) or spec.get("schema") != ADMISSION_SPEC_SCHEMA:
        raise ValueError(f"admission spec must use {ADMISSION_SPEC_SCHEMA}")
    source_checkpoint = Path(checkpoint).expanduser().resolve()
    expected_checkpoint_sha = _sha(checkpoint_sha256, "checkpoint_sha256")
    if not source_checkpoint.is_file() or _sha256(source_checkpoint) != expected_checkpoint_sha:
        raise ValueError("checkpoint bytes do not match explicit checkpoint SHA")
    authenticated_sha_by_path: dict[Path, str] = {}
    for index, row in enumerate(spec.get("authenticated_inputs", [])):
        if not isinstance(row, Mapping):
            raise ValueError("authenticated_inputs rows must be objects")
        path = Path(str(row.get("path", ""))).expanduser().resolve()
        expected = _sha(row.get("sha256"), f"authenticated_inputs[{index}].sha256")
        if not path.is_file() or _sha256(path) != expected:
            raise ValueError(f"authenticated input mismatch: {path}")
        authenticated_sha_by_path[path] = expected
    root = Path(output_root).expanduser().resolve()
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(f"resident artifact output is not empty: {root}")
    (root / "checkpoints").mkdir(parents=True, exist_ok=True)
    checkpoint_row = spec.get("checkpoint")
    corpora = spec.get("corpora")
    composition = spec.get("composition")
    canary = spec.get("canary")
    score = spec.get("score")
    continuations = spec.get("continuations")
    if not all(isinstance(row, Mapping) for row in (checkpoint_row, corpora, composition, canary, score, continuations)):
        raise ValueError("checkpoint/corpora/composition/canary/score/continuations must be objects")
    assert isinstance(continuations, Mapping)
    bound_continuations: dict[str, dict[str, Any]] = {}
    for rank in (0, 1):
        continuation = continuations.get(str(rank))
        if not isinstance(continuation, Mapping) or continuation.get("rank") != rank:
            raise ValueError(f"continuations.{rank} must bind rank {rank}")
        bound = dict(continuation)
        configured_expert = bound.get("resident_expert_source")
        if configured_expert:
            expert = Path(str(configured_expert)).expanduser().resolve()
            expert_sha = authenticated_sha_by_path.get(expert)
            if expert_sha is None or bound.get("resident_expert_source_sha256") != expert_sha:
                raise ValueError(
                    f"continuations.{rank} requires an authenticated resident expert source"
                )
            configured_wrapper = bound.get("fast_k2_wrapper_source")
            wrapper = (
                Path(str(configured_wrapper)).expanduser().resolve()
                if configured_wrapper
                else expert.with_name("fast_k2_grouped.py")
            )
            wrapper_sha = authenticated_sha_by_path.get(wrapper)
            if wrapper_sha is None:
                raise ValueError(
                    f"continuations.{rank} requires an authenticated grouped-K2 wrapper"
                )
            explicit_wrapper_sha = bound.get("fast_k2_wrapper_source_sha256")
            if explicit_wrapper_sha is not None and explicit_wrapper_sha != wrapper_sha:
                raise ValueError(
                    f"continuations.{rank}.fast_k2_wrapper_source_sha256 contradicts authenticated input"
                )
            bound["fast_k2_wrapper_source"] = str(wrapper)
            bound["fast_k2_wrapper_source_sha256"] = wrapper_sha
        bound_continuations[str(rank)] = bound
    binding_spec = dict(spec)
    binding_spec["continuations"] = bound_continuations
    checkpoint_name = str(checkpoint_row.get("name", ""))
    if not checkpoint_name:
        raise ValueError("checkpoint.name is required")
    destination = root / "checkpoints" / f"{checkpoint_name}.pt"
    with source_checkpoint.open("rb") as source, destination.open("xb") as target:
        shutil.copyfileobj(source, target, length=8 << 20)
        target.flush()
        os.fsync(target.fileno())
    if _sha256(destination) != expected_checkpoint_sha:
        raise RuntimeError("copied checkpoint failed read-back verification")
    identity_fields, binding_sha = provider_binding(binding_spec)
    manifest = {
        "schema": "repair-artifact-v1",
        "artifact_id": str(spec.get("artifact_id", "canonical-resident-artifact")),
        "identity": {
            "basis_sha256": _sha(spec.get("basis_sha256"), "basis_sha256"),
            "builder_eval_corpus_sha256": _sha(corpora.get("builder_eval_sha256"), "corpora.builder_eval_sha256"),
            "train_score_corpus_sha256": _sha(corpora.get("train_score_sha256"), "corpora.train_score_sha256"),
            "teacher_inventory_sha256": _sha(corpora.get("teacher_inventory_sha256"), "corpora.teacher_inventory_sha256"),
        },
        "checkpoints": {
            checkpoint_name: {
                "path": f"checkpoints/{checkpoint_name}.pt",
                "sha256": expected_checkpoint_sha,
                "identity_sha256": _sha(checkpoint_row.get("identity_sha256"), "checkpoint.identity_sha256"),
                "next_update": int(checkpoint_row.get("next_update", 0)),
                "lock_sha256": _sha(checkpoint_row.get("lock_sha256"), "checkpoint.lock_sha256"),
                "trajectory_sha256": _sha(checkpoint_row.get("trajectory_sha256"), "checkpoint.trajectory_sha256"),
            }
        },
        "score": {
            "spec": "balanced64-v1",
            "teacher_dir": "teacher",
            "candidate_dir_template": "score/candidates/{checkpoint}",
            "window_ids": list(score.get("window_ids", [])),
            "positions_per_window": 1024,
            "support": 8192,
        },
    }
    _atomic_json(root / "ARTIFACT.json", manifest)
    manifest_sha = _sha256(root / "ARTIFACT.json")
    identity = {
        "schema": "banana-smasher-artifact-identity-v1",
        "basis": {"model_index_sha256": manifest["identity"]["basis_sha256"]},
        "corpora": {
            "builder_eval_sha256": manifest["identity"]["builder_eval_corpus_sha256"],
            "train_score_sha256": manifest["identity"]["train_score_corpus_sha256"],
            "u0_lock_sha256": manifest["checkpoints"][checkpoint_name]["lock_sha256"],
            "teacher_inventory_sha256": manifest["identity"]["teacher_inventory_sha256"],
        },
        "checkpoints": {checkpoint_name: manifest["checkpoints"][checkpoint_name]},
        "composition": dict(composition),
        "canary": dict(canary),
        "runtime": {"production_rails": {"provider_binding_sha256": binding_sha}},
        "provenance": {
            "admission_spec_sha256": hashlib.sha256(spec_bytes).hexdigest(),
            "checkpoint_sha256": expected_checkpoint_sha,
        },
    }
    _atomic_json(root / "identity.json", identity)
    loaded_identity = ArtifactIdentity.load(root)
    RepairArtifact.open(root)
    configs: dict[str, str] = {}
    for rank in (0, 1):
        bound_continuation = dict(bound_continuations[str(rank)])
        asset_root = bound_continuation.get("asset_root")
        if asset_root:
            admission_path = Path(str(asset_root)).expanduser().resolve() / "code" / "JOINT_REPAIR_ADMISSION.json"
            authenticated_admission_sha = authenticated_sha_by_path.get(admission_path)
            if authenticated_admission_sha is not None:
                explicit_admission_sha = bound_continuation.get("admission_sha256")
                if explicit_admission_sha is not None and explicit_admission_sha != authenticated_admission_sha:
                    raise ValueError(f"continuations.{rank}.admission_sha256 contradicts authenticated input")
                bound_continuation["admission_sha256"] = authenticated_admission_sha
        config = {
            **identity_fields,
            "allowed_artifacts": {
                loaded_identity.sha256: {
                    "basis_sha256": loaded_identity.basis_sha256,
                    "checkpoint": checkpoint_name,
                    "artifact_manifest_sha256": manifest_sha,
                    "checkpoint_sha256": expected_checkpoint_sha,
                }
            },
            "continuation": bound_continuation,
        }
        path = root / f"production-rails.rank{rank}.json"
        _atomic_json(path, config)
        rails = ProductionRails.from_file(path, run_root=root / f".verify-rank{rank}")
        if rails.provider_binding_sha256 != binding_sha:
            raise RuntimeError("generated provider binding failed verification")
        shutil.rmtree(root / f".verify-rank{rank}")
        configs[str(rank)] = str(path)
    receipt = {
        "schema": ADMISSION_RECEIPT_SCHEMA,
        "status": "PASS",
        "checkpoint_path": str(source_checkpoint),
        "checkpoint_sha256": expected_checkpoint_sha,
        "artifact_root": str(root),
        "artifact_manifest_sha256": manifest_sha,
        "artifact_identity_sha256": loaded_identity.sha256,
        "provider_binding_sha256": binding_sha,
        "rank_configs": configs,
        "spec_path": str(spec_file),
        "spec_sha256": hashlib.sha256(spec_bytes).hexdigest(),
    }
    _atomic_json(root / "ADMISSION.json", receipt)
    return receipt


__all__ = ["ADMISSION_RECEIPT_SCHEMA", "ADMISSION_SPEC_SCHEMA", "admit_resident_artifact", "provider_binding"]
