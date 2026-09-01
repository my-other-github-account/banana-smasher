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
MIXED_ADMISSION_SPEC_SCHEMA = "banana-smasher-mixed-resident-admission-spec-v1"
MIXED_ADMISSION_RECEIPT_SCHEMA = "banana-smasher-mixed-resident-admission-v1"
MIXED_ARTIFACT_MODE = "mixed-backpack-virtual-v1"
MIXED_PHYSICAL_TIERS = frozenset({"native_mxfp4", "qtip2", "qtip3"})
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
    "mixed_provider_factory",
    "mixed_provider_source_sha256",
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


def _mixed_index_composition(index_raw: bytes) -> list[dict[str, Any]]:
    """Validate and summarize the canonical 43x256x2 physical cell roster."""

    counts = {
        layer: {tier: 0 for tier in MIXED_PHYSICAL_TIERS}
        for layer in ALL_LAYERS
    }
    cells: set[tuple[int, int, str]] = set()
    try:
        rows = [json.loads(line) for line in index_raw.splitlines() if line.strip()]
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("mixed chain materialization index is not valid JSONL") from exc
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError("mixed chain requires an exact 43x256x2 cell roster")
        layer, expert, projection, tier = (
            row.get("layer"),
            row.get("expert"),
            row.get("projection"),
            row.get("source_key"),
        )
        if (
            isinstance(layer, bool)
            or not isinstance(layer, int)
            or layer not in counts
            or isinstance(expert, bool)
            or not isinstance(expert, int)
            or expert not in range(256)
            or projection not in {"down", "fused13"}
            or tier not in MIXED_PHYSICAL_TIERS
            or (row.get("tier") is not None and row.get("tier") != tier)
        ):
            raise ValueError("mixed chain requires an exact 43x256x2 cell roster")
        cell = (layer, expert, str(projection))
        if cell in cells:
            raise ValueError("mixed chain requires an exact 43x256x2 cell roster")
        cells.add(cell)
        counts[layer][str(tier)] += 1
    expected = {
        (layer, expert, projection)
        for layer in ALL_LAYERS
        for expert in range(256)
        for projection in ("down", "fused13")
    }
    if cells != expected:
        raise ValueError("mixed chain requires an exact 43x256x2 cell roster")
    return [
        {"layer": layer, "tiers": dict(sorted(counts[layer].items()))}
        for layer in ALL_LAYERS
    ]


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


def _atomic_bytes(path: Path, payload: bytes) -> None:
    """Install authenticated bytes without JSON reserialization."""

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
    composition = spec.get("composition")
    layer_rows = (
        composition.get("layers")
        if isinstance(composition, Mapping)
        and isinstance(composition.get("layers"), list)
        else None
    )
    layers = (
        [int(row["layer"]) for row in layer_rows if isinstance(row, Mapping) and "layer" in row]
        if layer_rows
        else list(ALL_LAYERS)
    )
    fields = {
        "schema": PRODUCTION_RAILS_SCHEMA,
        "pipeline_microbatch": PIPELINE_MICROBATCH,
        "model_layer_count": len(layers),
        "layers": layers,
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


def admit_mixed_resident_artifact(
    spec_path: str | Path,
    artifact_root: str | Path,
) -> dict[str, Any]:
    """Admit a sealed mixed virtual chain without rewriting its identity or wire."""

    spec_file = Path(spec_path).expanduser().resolve()
    spec_raw = spec_file.read_bytes()
    spec = json.loads(spec_raw)
    if not isinstance(spec, Mapping) or spec.get("schema") != MIXED_ADMISSION_SPEC_SCHEMA:
        raise ValueError(f"mixed admission spec must use {MIXED_ADMISSION_SPEC_SCHEMA}")

    root = Path(artifact_root).expanduser().resolve()
    identity_path = root / "identity.json"
    virtual_path = root / "BACKPACK_VIRTUAL_MANIFEST.json"
    index_path = root / "MATERIALIZATION_INDEX.jsonl"
    virtual_raw = virtual_path.read_bytes()
    index_raw = index_path.read_bytes()
    if hashlib.sha256(virtual_raw).hexdigest() != _sha(
        spec.get("virtual_manifest_sha256"), "virtual_manifest_sha256"
    ):
        raise ValueError("mixed chain virtual manifest identity mismatch")
    if hashlib.sha256(index_raw).hexdigest() != _sha(
        spec.get("materialization_index_sha256"), "materialization_index_sha256"
    ):
        raise ValueError("mixed chain materialization index identity mismatch")

    identity_manifest = spec.get("identity_manifest")
    install_identity = False
    identity_source: Path | None = None
    if identity_path.is_file():
        if identity_manifest is not None:
            raise ValueError(
                "identity_manifest handoff requires an identity-less sealed chain"
            )
        identity_raw = identity_path.read_bytes()
        expected_identity_sha = _sha(spec.get("identity_sha256"), "identity_sha256")
    else:
        if not isinstance(identity_manifest, Mapping) or set(identity_manifest) != {
            "path",
            "sha256",
        }:
            raise ValueError(
                "identity-less mixed chain requires identity_manifest with exactly path and sha256"
            )
        if spec.get("identity_sha256") is not None:
            raise ValueError(
                "identity-less mixed chain must bind identity only through identity_manifest"
            )
        identity_source = Path(str(identity_manifest.get("path", ""))).expanduser().resolve()
        expected_identity_sha = _sha(
            identity_manifest.get("sha256"), "identity_manifest.sha256"
        )
        if not identity_source.is_file():
            raise ValueError("authenticated mixed identity manifest is not a file")
        identity_raw = identity_source.read_bytes()
        install_identity = True
    if hashlib.sha256(identity_raw).hexdigest() != expected_identity_sha:
        raise ValueError("mixed chain identity manifest identity mismatch")

    with tempfile.TemporaryDirectory(prefix="banana-smasher-mixed-identity-") as temporary:
        temporary_root = Path(temporary)
        (temporary_root / "identity.json").write_bytes(identity_raw)
        identity = ArtifactIdentity.load(temporary_root)
    tiers = {
        str(tier)
        for layer in identity.composition
        for tier, count in layer["tiers"].items()
        if int(count) > 0
    }
    if (
        identity.composition_kind != "mixed-per-layer-per-expert"
        or tiers != MIXED_PHYSICAL_TIERS
        or [row.get("layer") for row in identity.composition] != list(ALL_LAYERS)
    ):
        raise ValueError(
            "mixed chain composition must cover ordered layers 0..42 with exactly "
            "native_mxfp4+qtip2+qtip3"
        )
    virtual = json.loads(virtual_raw)
    index_binding = virtual.get("materialization_index", {})
    if (
        virtual.get("schema") != "banana-smasher-backpack-virtual-assignment-v1"
        or virtual.get("status") != "PASS_LOGICAL_FULL_WIRE"
        or virtual.get("basis_sha256") != identity.basis_sha256
        or not isinstance(index_binding, Mapping)
        or index_binding.get("file") != index_path.name
        or index_binding.get("bytes") != len(index_raw)
        or index_binding.get("sha256") != hashlib.sha256(index_raw).hexdigest()
    ):
        raise ValueError("mixed chain virtual/index binding mismatch")
    score = spec.get("score")
    if (
        not isinstance(score, Mapping)
        or score.get("positions") != 64 * 1024
        or score.get("support") != 8192
        or not isinstance(score.get("window_ids"), list)
        or len(score["window_ids"]) != 64
        or len(set(score["window_ids"])) != 64
        or any(
            isinstance(window, bool) or not isinstance(window, int)
            for window in score["window_ids"]
        )
    ):
        raise ValueError("mixed admission requires scorer-aligned 64x1024/t8192 fixture")
    checkpoint = str(spec.get("checkpoint", ""))
    checkpoint_row = identity.checkpoints.get(checkpoint)
    if not isinstance(checkpoint_row, Mapping):
        raise ValueError("mixed admission checkpoint is absent from sealed identity")

    configs: dict[str, str] = {}
    identity_sha = identity.sha256
    binding = {
        "artifact_mode": MIXED_ARTIFACT_MODE,
        "basis_sha256": identity.basis_sha256,
        "checkpoint": checkpoint,
        "checkpoint_sha256": str(checkpoint_row["sha256"]),
        "identity_sha256": identity_sha,
        "virtual_manifest_sha256": hashlib.sha256(virtual_raw).hexdigest(),
        "materialization_index_sha256": hashlib.sha256(index_raw).hexdigest(),
        "physical_tiers": sorted(MIXED_PHYSICAL_TIERS),
    }
    continuations = spec.get("continuations")
    assert isinstance(continuations, Mapping)
    allow_test_provider = spec.get("allow_test_mixed_provider") is True
    bound_continuations: dict[str, dict[str, Any]] = {}
    if allow_test_provider:
        bound_continuations = {
            str(rank): {
                **dict(continuations[str(rank)]),
                "test_fixture_provider": True,
            }
            for rank in (0, 1)
        }
    else:
        from . import mixed_physical_provider
        from .mixed_physical_provider import (
            CANONICAL_BASIS_SHA256,
            CANONICAL_FACTORY,
            CANONICAL_LAYER_SPLIT,
        )

        canonical_source = Path(str(mixed_physical_provider.__file__)).resolve()
        canonical_sha = _sha256(canonical_source)
        supplied_factories = {
            continuations.get(str(rank), {}).get("mixed_provider_factory")
            for rank in (0, 1)
            if isinstance(continuations.get(str(rank)), Mapping)
        }
        if any(
            factory not in (None, CANONICAL_FACTORY)
            for factory in supplied_factories
        ):
            raise ValueError(
                "production admission requires the canonical physical mixed provider"
            )
        if identity.basis_sha256 != CANONICAL_BASIS_SHA256:
            raise ValueError("canonical physical mixed provider basis identity mismatch")
        index_composition = _mixed_index_composition(index_raw)
        index_tier_counts = {
            tier: sum(row["tiers"][tier] for row in index_composition)
            for tier in sorted(MIXED_PHYSICAL_TIERS)
        }
        if (
            virtual.get("source_component_counts") != index_tier_counts
            or virtual.get("tier_counts") != index_tier_counts
        ):
            raise ValueError(
                "mixed virtual manifest tier counts do not match exact materialization roster"
            )
        identity_composition = [
            {
                "layer": row.get("layer"),
                "tiers": dict(sorted(dict(row.get("tiers", {})).items())),
            }
            for row in identity.composition
        ]
        if identity_composition != index_composition:
            raise ValueError(
                "mixed identity composition does not match exact materialization roster"
            )
        source_bindings = virtual.get("source_bindings")
        if (
            not isinstance(source_bindings, Mapping)
            or set(source_bindings) != MIXED_PHYSICAL_TIERS
        ):
            raise ValueError("mixed virtual manifest requires exact physical source bindings")
        for tier in sorted(MIXED_PHYSICAL_TIERS):
            source_binding = source_bindings.get(tier)
            if (
                not isinstance(source_binding, Mapping)
                or source_binding.get("basis_sha256") != identity.basis_sha256
            ):
                raise ValueError("mixed virtual manifest source binding basis mismatch")
            _sha(
                source_binding.get("identity_sha256"),
                f"source_bindings.{tier}.identity_sha256",
            )
        for rank in (0, 1):
            continuation = continuations.get(str(rank))
            if not isinstance(continuation, Mapping) or continuation.get("rank") != rank:
                raise ValueError(f"mixed continuations.{rank} must bind rank {rank}")
            supplied = continuation.get("mixed_provider_factory")
            if supplied not in (None, CANONICAL_FACTORY):
                raise ValueError("production admission requires the canonical physical mixed provider")
            try:
                split = {
                    int(key): tuple(int(item) for item in value)
                    for key, value in continuation.get("layer_split", {}).items()
                }
            except (AttributeError, TypeError, ValueError) as exc:
                raise ValueError("canonical physical mixed provider requires exact rank geometry") from exc
            if split != CANONICAL_LAYER_SPLIT:
                raise ValueError(
                    "canonical physical mixed provider requires rank0 [0,20] and rank1 [21,42]"
                )
            bound_continuations[str(rank)] = {
                **dict(continuation),
                "mixed_provider_factory": CANONICAL_FACTORY,
                "mixed_provider_source": str(canonical_source),
                "mixed_provider_source_sha256": canonical_sha,
                "layer_split": {
                    str(key): list(value) for key, value in CANONICAL_LAYER_SPLIT.items()
                },
            }
    bound_spec = {**dict(spec), "continuations": bound_continuations}
    identity_fields, binding_sha = provider_binding(bound_spec)
    provider_sources = []
    for rank in (0, 1):
        continuation = bound_continuations.get(str(rank))
        if not isinstance(continuation, Mapping) or continuation.get("rank") != rank:
            raise ValueError(f"mixed continuations.{rank} must bind rank {rank}")
        source = Path(
            str(continuation.get("mixed_provider_source", ""))
        ).expanduser().resolve()
        expected = _sha(
            continuation.get("mixed_provider_source_sha256"),
            f"continuations.{rank}.mixed_provider_source_sha256",
        )
        if (
            not isinstance(continuation.get("mixed_provider_factory"), str)
            or not source.is_file()
            or _sha256(source) != expected
        ):
            raise ValueError(f"mixed continuations.{rank} provider identity mismatch")
        provider_sources.append((source, expected))
    if provider_sources[0] != provider_sources[1]:
        raise ValueError("mixed continuations provider source mismatch")
    if install_identity:
        _atomic_bytes(identity_path, identity_raw)
        installed_identity = ArtifactIdentity.load(root)
        if installed_identity.sha256 != identity.sha256:
            raise RuntimeError("installed mixed identity manifest failed read-back verification")
        identity = installed_identity
    for rank in (0, 1):
        continuation = bound_continuations.get(str(rank))
        if not isinstance(continuation, Mapping) or continuation.get("rank") != rank:
            raise ValueError(f"mixed continuations.{rank} must bind rank {rank}")
        config = {
            **identity_fields,
            "allowed_artifacts": {identity_sha: binding},
            "continuation": dict(continuation),
        }
        path = root / f"production-rails.rank{rank}.json"
        _atomic_json(path, config)
        rails = ProductionRails.from_file(path, run_root=root / f".verify-rank{rank}")
        if rails.provider_binding_sha256 != binding_sha:
            raise RuntimeError("generated mixed provider binding failed verification")
        shutil.rmtree(root / f".verify-rank{rank}")
        configs[str(rank)] = str(path)
    receipt = {
        "schema": MIXED_ADMISSION_RECEIPT_SCHEMA,
        "status": "PASS",
        "artifact_mode": MIXED_ARTIFACT_MODE,
        "artifact_root": str(root),
        "artifact_identity_sha256": identity_sha,
        "identity_manifest_source": (
            None if identity_source is None else str(identity_source)
        ),
        "virtual_manifest_sha256": hashlib.sha256(virtual_raw).hexdigest(),
        "materialization_index_sha256": hashlib.sha256(index_raw).hexdigest(),
        "checkpoint": checkpoint,
        "checkpoint_sha256": str(checkpoint_row["sha256"]),
        "provider_binding_sha256": binding_sha,
        "rank_configs": configs,
        "spec_path": str(spec_file),
        "spec_sha256": hashlib.sha256(spec_raw).hexdigest(),
    }
    _atomic_json(root / "MIXED_ADMISSION.json", receipt)
    return receipt


__all__ = [
    "ADMISSION_RECEIPT_SCHEMA",
    "ADMISSION_SPEC_SCHEMA",
    "MIXED_ADMISSION_RECEIPT_SCHEMA",
    "MIXED_ADMISSION_SPEC_SCHEMA",
    "admit_mixed_resident_artifact",
    "admit_resident_artifact",
    "provider_binding",
]
