"""Concrete production rails for the resident repair facade.

The provider is deliberately configuration-bound: executable hooks and every
accepted artifact identity live in one SHA-pinned document.  An artifact not in
that document cannot be loaded, swapped, scored, or trained.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import tempfile
import time
from typing import Any, Callable, Mapping, Sequence

from .artifact_identity import ArtifactIdentity
from .resident_proven_api import ResidentRepairAPI as _ProvenResidentAPI
from .resident_repair_api import BackpackArtifact, UniformBuild

PRODUCTION_RAILS_SCHEMA = "banana-smasher-production-resident-rails-v1"
PIPELINE_MICROBATCH = 4
DEFAULT_IMPROVE_LR_SCALE = 0.1
VALIDATED_REPAIR_RECIPE = {
    "training_recipe": "u45_validated_v1",
    "sampling_mode": "broad_rotation_v1",
    "windows_per_update": 16,
    "pipeline_microbatch": 4,
    "loss_reduction_dtype": "float32",
    "optimizer_moment_dtype": "float64",
    "base_lrs": {"luts": 1.0e-2, "norms": 1.0e-4, "outputs": 1.0e-2},
    "lr_scale": DEFAULT_IMPROVE_LR_SCALE,
    "heldout_validation_interval": 4,
    "heldout_kill_patience": 2,
    "accepted_update_cadence": 1,
    # Keep the documented public recipe on the low-memory reentrant path.
    # Disabling this retains the full rank-local layer graph and repeatedly
    # exhausted DGX Spark unified memory before the first pipeline send, even
    # with MemoryMax=105 GiB and 80 GiB total swap on rank 0.
    "activation_checkpointing": True,
}
ALL_LAYERS = tuple(range(43))
FORBIDDEN_SLOW_CONTROL_FIELDS = frozenset(
    {
        "fallback",
        "slow_path",
        "notification_source",
        "rate_low",
        "offline_path",
        "replay_path",
        "staged_file_path",
        "reload_path",
    }
)


class ProductionRailsError(RuntimeError):
    """The production provider or its pinned artifact binding is invalid."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _callable(reference: object, field: str) -> Callable[..., Any]:
    if not isinstance(reference, str) or reference.count(":") != 1:
        raise ProductionRailsError(f"{field} must be a module:callable reference")
    module_name, attribute = reference.split(":", 1)
    try:
        value = getattr(importlib.import_module(module_name), attribute)
    except (ImportError, AttributeError) as exc:
        raise ProductionRailsError(f"cannot load {field}={reference!r}") from exc
    if not callable(value):
        raise ProductionRailsError(f"{field}={reference!r} is not callable")
    return value


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    payload = (json.dumps(dict(value), indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass


def _forbidden_slow_control(value: object) -> str | None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            normalized = str(key).lower()
            if normalized in FORBIDDEN_SLOW_CONTROL_FIELDS:
                return normalized
            nested = _forbidden_slow_control(item)
            if nested is not None:
                return nested
    elif isinstance(value, (list, tuple)):
        for item in value:
            nested = _forbidden_slow_control(item)
            if nested is not None:
                return nested
    return None


def _heldout_decision(
    previous_kld: float, current_kld: float, streak: int, *, patience: int
) -> dict[str, Any]:
    if not all(math.isfinite(value) for value in (previous_kld, current_kld)):
        raise ProductionRailsError("held-out gate received non-finite KLD")
    if patience < 1 or streak < 0:
        raise ProductionRailsError("held-out gate patience/streak is invalid")
    improved = current_kld < previous_kld
    next_streak = 0 if improved else streak + 1
    return {
        "improved": improved,
        "non_improving_streak": next_streak,
        "halt": next_streak >= patience,
    }


@dataclass(frozen=True)
class _ArtifactBinding:
    identity_sha256: str
    basis_sha256: str
    checkpoint: str
    score_checkpoints: Mapping[str, str]
    artifact_manifest_sha256: str
    checkpoint_sha256: str
    artifact_mode: str = "repair-artifact-v1"
    virtual_manifest_sha256: str | None = None
    materialization_index_sha256: str | None = None


def _require_distributed_pair_binding(
    provider_binding_sha256: str,
    config: Mapping[str, Any],
    *,
    distributed: Any | None = None,
) -> None:
    world_size = config.get("world_size")
    if not isinstance(world_size, int) or world_size <= 1:
        return
    if distributed is None:
        try:
            import torch.distributed as distributed
        except ImportError as exc:
            raise ProductionRailsError(
                "distributed pair binding requires torch.distributed"
            ) from exc
    if (
        not distributed.is_available()
        or not distributed.is_initialized()
        or distributed.get_world_size() != world_size
    ):
        raise ProductionRailsError("distributed pair binding requires initialized world_size")
    value = {
        "provider_binding_sha256": provider_binding_sha256,
        "basis_sha256": config.get("basis_sha256"),
    }
    gathered: list[object] = [None] * world_size
    distributed.all_gather_object(gathered, value)
    if any(row != value for row in gathered):
        raise ProductionRailsError("distributed pair scientific binding mismatch")


def _construct_resident_engine(
    api: _ProvenResidentAPI,
    binding: _ArtifactBinding,
    config: Mapping[str, Any],
) -> Any:
    """Construct the one physical model owned by a resident arm session."""
    method = getattr(api, "construct_resident_engine", None)
    if not callable(method):
        raise ProductionRailsError("resident implementation cannot construct a physical engine")
    options = dict(config)
    for field, expected in (
        ("basis_sha256", binding.basis_sha256),
        ("checkpoint_sha256", binding.checkpoint_sha256),
    ):
        if field in options and options[field] != expected:
            raise ProductionRailsError(
                f"continuation {field} does not match admitted artifact bytes"
            )
        options[field] = expected
    return method(binding.checkpoint, config=options)


def _construct_resident_score_engine(
    api: _ProvenResidentAPI,
    binding: _ArtifactBinding,
    config: Mapping[str, Any],
) -> Any:
    """Construct one score-only physical model for a single API phase."""
    method = getattr(api, "construct_resident_score_engine", None)
    if not callable(method):
        raise ProductionRailsError("resident implementation cannot construct a score engine")
    options = dict(config)
    options["basis_sha256"] = binding.basis_sha256
    options["checkpoint_sha256"] = binding.checkpoint_sha256
    checkpoint_path = api.artifact.checkpoint_path(binding.checkpoint)
    return method(
        checkpoint_path,
        binding.checkpoint_sha256,
        config=options,
    )


class _MixedProviderSession:
    """Adapter for one authenticated provider over immutable mixed physical bytes."""

    def __init__(
        self,
        artifact: BackpackArtifact,
        binding: _ArtifactBinding,
        *,
        continuation_config: Mapping[str, Any],
        receipt_root: Path,
    ) -> None:
        reference = continuation_config.get("mixed_provider_factory")
        expected_source_sha = continuation_config.get("mixed_provider_source_sha256")
        if not isinstance(reference, str) or not isinstance(expected_source_sha, str):
            raise ProductionRailsError(
                "mixed resident continuation requires an authenticated mixed_provider_factory"
            )
        factory = _callable(reference, "mixed_provider_factory")
        module = importlib.import_module(factory.__module__)
        source = Path(str(getattr(module, "__file__", ""))).resolve()
        if not source.is_file() or _sha256(source) != expected_source_sha:
            raise ProductionRailsError("mixed resident provider source identity mismatch")
        self.root = artifact.root.resolve()
        self.binding = binding
        self.provider = factory(
            artifact_root=self.root,
            identity_sha256=artifact.identity.sha256,
            basis_sha256=artifact.identity.basis_sha256,
            checkpoint=binding.checkpoint,
            checkpoint_sha256=binding.checkpoint_sha256,
            virtual_manifest=self.root / "BACKPACK_VIRTUAL_MANIFEST.json",
            materialization_index=self.root / "MATERIALIZATION_INDEX.jsonl",
            rank=continuation_config.get("rank"),
            run_root=receipt_root,
            config=dict(continuation_config),
        )
        for method in ("score", "train", "restore_pre_score", "restore_training"):
            if not callable(getattr(self.provider, method, None)):
                raise ProductionRailsError(
                    f"mixed resident physical provider lacks required {method}()"
                )
        if (
            getattr(self.provider, "physical_mixed_provider", None) is not True
            and continuation_config.get("test_fixture_provider") is not True
        ):
            raise ProductionRailsError(
                "mixed resident continuation rejected a fixture-only provider"
            )

    def hot_swap(self, artifact: BackpackArtifact, binding: _ArtifactBinding) -> None:
        if (
            artifact.root.resolve() != self.root
            or binding.identity_sha256 != self.binding.identity_sha256
        ):
            raise ProductionRailsError("mixed resident hot swap changed physical provider identity")
        self.binding = binding

    def score(self, phase: str) -> Mapping[str, Any]:
        result = dict(self.provider.score(phase))
        if "support" not in result and "support_width" in result:
            result["support"] = result["support_width"]
        result.setdefault("checkpoint", self.binding.checkpoint)
        result.setdefault("physical_checkpoint", result["checkpoint"])
        return result

    def score_probe(self, windows: Sequence[int]) -> Mapping[str, Any]:
        method = getattr(self.provider, "score_probe", None)
        if not callable(method):
            raise ProductionRailsError("mixed resident provider lacks score_probe()")
        return dict(method(tuple(int(value) for value in windows)))

    def train(self, updates: int) -> Mapping[str, Any]:
        result = dict(self.provider.train(updates))
        if result.get("updates") != updates:
            raise ProductionRailsError("mixed resident provider update count mismatch")
        checkpoint = result.get("checkpoint")
        checkpoint_sha = result.get("checkpoint_sha256")
        if not isinstance(checkpoint, str) or not isinstance(checkpoint_sha, str):
            raise ProductionRailsError("mixed resident training did not return a bound checkpoint")
        self.binding = _ArtifactBinding(
            identity_sha256=self.binding.identity_sha256,
            basis_sha256=self.binding.basis_sha256,
            checkpoint=checkpoint,
            score_checkpoints={"post": checkpoint},
            artifact_manifest_sha256="",
            checkpoint_sha256=checkpoint_sha,
            artifact_mode=self.binding.artifact_mode,
            virtual_manifest_sha256=self.binding.virtual_manifest_sha256,
            materialization_index_sha256=self.binding.materialization_index_sha256,
        )
        return result

    def restore_pre_score(self, pre: Mapping[str, Any]) -> None:
        self.provider.restore_pre_score(dict(pre))

    def restore_training(
        self, pre: Mapping[str, Any], training: Mapping[str, Any]
    ) -> None:
        self.provider.restore_training(dict(pre), dict(training))
        self.binding = _ArtifactBinding(
            identity_sha256=self.binding.identity_sha256,
            basis_sha256=self.binding.basis_sha256,
            checkpoint=str(training["checkpoint"]),
            score_checkpoints={"post": str(training["checkpoint"])},
            artifact_manifest_sha256="",
            checkpoint_sha256=str(training["checkpoint_sha256"]),
            artifact_mode=self.binding.artifact_mode,
            virtual_manifest_sha256=self.binding.virtual_manifest_sha256,
            materialization_index_sha256=self.binding.materialization_index_sha256,
        )


class _ProvenSession:
    """Adapter over the proven resident scorer/continuation implementation.

    One object owns the public arm, but each score/train phase owns a distinct
    physical engine.  This matches the validated production topology: score
    caches are destroyed before optimizer construction, and training residency
    is destroyed before the post score is constructed.
    """

    def __init__(
        self,
        artifact: BackpackArtifact,
        binding: _ArtifactBinding,
        *,
        continuation_config: Mapping[str, Any],
        receipt_root: Path,
        provider_binding_sha256: str | None = None,
    ) -> None:
        self.root = artifact.root.resolve()
        self.api = _ProvenResidentAPI.open(self.root)
        self.binding = binding
        self.continuation_config = dict(continuation_config)
        # The public arm owns the validated U45 recipe. Callers cannot select
        # sampling, numeric, learning-rate, or held-out gate variants.
        self.continuation_config.update(
            {
                key: dict(value) if isinstance(value, Mapping) else value
                for key, value in VALIDATED_REPAIR_RECIPE.items()
            }
        )
        self.continuation_config.setdefault(
            "checkpoint_lut_root",
            str((receipt_root / "checkpoint-luts").resolve()),
        )
        self.continuation_config.setdefault(
            "cold_start_phase_receipt",
            str((receipt_root / "cold-start-phase.rank{rank}.jsonl").resolve()),
        )
        self.receipt_root = receipt_root
        # Physical residency is phase-lazy: a sealed PRE may proceed directly to
        # training without constructing an unused scorer, while score_pre()
        # constructs and later destroys its own score-only engine.
        self.engine: Any | None = None
        if provider_binding_sha256 is not None:
            _require_distributed_pair_binding(
                provider_binding_sha256,
                self.continuation_config,
            )
        self._pre_kld: float | None = None
        self.phase_releases: list[Mapping[str, Any]] = []

    def _release_engine(self, phase: str) -> Mapping[str, Any]:
        if self.engine is None:
            raise ProductionRailsError(f"resident {phase} engine is already released")
        close = getattr(self.engine, "close", None)
        if not callable(close):
            raise ProductionRailsError("physical resident engine lacks phase teardown")
        raw_release = close(phase=phase)
        if not isinstance(raw_release, Mapping):
            raise ProductionRailsError("physical resident teardown must return a mapping")
        release: dict[str, Any] = dict(raw_release)
        allocated = release.get("post_release_allocated_bytes")
        limit = 10 * 1024**3
        if not isinstance(allocated, int) or allocated < 0 or allocated >= limit:
            raise ProductionRailsError(
                f"resident {phase} teardown did not return below 10 GiB: {allocated}"
            )
        release["limit_bytes"] = limit
        self.phase_releases.append(release)
        self.engine = None
        return release

    def hot_swap(self, artifact: BackpackArtifact, binding: _ArtifactBinding) -> None:
        if artifact.root.resolve() != self.root:
            raise ProductionRailsError(
                "resident hot swap must remain inside the loaded repair artifact root"
            )
        # Re-activating the admitted artifact after training must not roll the
        # live model back to the checkpoint named in the static config.
        if binding.identity_sha256 != self.binding.identity_sha256:
            raise ProductionRailsError(
                "resident hot swap cannot change artifact identity after model construction"
            )

    def score(self, phase: str) -> Mapping[str, Any]:
        if self.engine is None:
            self.engine = _construct_resident_score_engine(
                self.api, self.binding, self.continuation_config
            )
        method = getattr(self.engine, "score_balanced64", None)
        if not callable(method):
            raise ProductionRailsError("physical resident engine cannot score in memory")
        raw_result: Any = method(self.api.artifact.windows)
        result: dict[str, Any] = dict(raw_result)
        # The sealed Balanced64 scorer names this field ``support_width``.
        # Normalize that receipt-backed value at the production-rail seam;
        # never synthesize it from the artifact declaration.
        if "support" not in result and "support_width" in result:
            result["support"] = result["support_width"]
        scored_checkpoint = result.get("checkpoint")
        if scored_checkpoint != self.binding.checkpoint:
            try:
                checkpoint_row = self.api.artifact.manifest["checkpoints"][
                    self.binding.checkpoint
                ]
                expected_physical = f"UPDATE_{int(checkpoint_row['next_update']):03d}"
            except (KeyError, TypeError, ValueError) as exc:
                raise ProductionRailsError(
                    "resident checkpoint alias has no authenticated update cursor"
                ) from exc
            if scored_checkpoint != expected_physical:
                raise ProductionRailsError(
                    "physical resident score checkpoint does not match authenticated alias"
                )
        result["physical_checkpoint"] = scored_checkpoint
        result["checkpoint"] = self.binding.checkpoint
        if phase == "pre":
            try:
                self._pre_kld = float(result["mean_kld"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ProductionRailsError("resident pre-score lacks a loss-guard baseline") from exc
        result["phase_release"] = dict(self._release_engine(f"score_{phase}"))
        return result

    def score_probe(self, windows: Sequence[int]) -> Mapping[str, Any]:
        ordered = tuple(int(value) for value in windows)
        if not ordered or len(set(ordered)) != len(ordered):
            raise ProductionRailsError("resident score probe requires unique windows")
        if self.engine is None:
            self.engine = _construct_resident_score_engine(
                self.api, self.binding, self.continuation_config
            )
        method = getattr(self.engine, "score_probe", None)
        if not callable(method):
            raise ProductionRailsError(
                "physical resident engine cannot run bounded score probe"
            )
        raw = method(ordered)
        if not isinstance(raw, Mapping):
            raise ProductionRailsError("resident score probe returned a non-mapping")
        result = dict(raw)
        if "support" not in result and "support_width" in result:
            result["support"] = result["support_width"]
        result["phase_release"] = dict(self._release_engine("score_probe"))
        return result

    def train(self, updates: int) -> Mapping[str, Any]:
        if updates == 0:
            return {"updates": 0, "checkpoint": self.binding.checkpoint}
        try:
            start_update = int(self.api.artifact.manifest["checkpoints"][self.binding.checkpoint]["next_update"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ProductionRailsError("selected checkpoint has no integer next_update") from exc
        target = start_update + updates
        config = dict(self.continuation_config)
        rank = self.continuation_config.get("rank")
        rank_suffix = f".rank{rank}" if rank in (0, 1) else ""
        if self._pre_kld is None:
            raise ProductionRailsError("resident training requires a measured pre-score baseline")
        if self.engine is not None:
            raise ProductionRailsError("resident score engine must be released before training")
        self.engine = _construct_resident_engine(
            self.api, self.binding, self.continuation_config
        )
        assert self.engine is not None
        method = getattr(self.api, "advance_resident_engine", None)
        if not callable(method):
            raise ProductionRailsError("resident implementation cannot advance the live engine")
        interval = int(config["heldout_validation_interval"])
        patience = int(config["heldout_kill_patience"])
        accepted_update_cadence = int(config["accepted_update_cadence"])
        if interval != 4:
            raise ProductionRailsError("validated resident recipe requires four-update boundaries")
        if accepted_update_cadence != 1:
            raise ProductionRailsError("validated resident recipe requires every update to be durable")
        previous_kld = self._pre_kld
        streak = 0
        boundaries: list[dict[str, Any]] = []
        receipts: list[str] = []
        result: Mapping[str, Any] = {}
        current = start_update
        while current < target:
            boundary = min(current + accepted_update_cadence, target)
            receipt = self.receipt_root / (
                f"CONTINUATION_U{current:03d}_U{boundary:03d}{rank_suffix}.json"
            )
            loss_guard_receipt = receipt.with_name(
                f"{receipt.stem}.LOSS_GUARD.json"
            )
            result = method(
                self.engine,
                self.binding.checkpoint,
                boundary,
                config=config,
                receipt_path=receipt,
                loss_guard_baseline=self._pre_kld,
                loss_guard_receipt_path=loss_guard_receipt,
            )
            checkpoint = result.get("checkpoint")
            checkpoint_sha = result.get("checkpoint_sha256")
            if not isinstance(checkpoint, str) or not isinstance(checkpoint_sha, str):
                raise ProductionRailsError("resident training did not return a bound checkpoint")
            self.binding = _ArtifactBinding(
                identity_sha256=self.binding.identity_sha256,
                basis_sha256=self.binding.basis_sha256,
                checkpoint=checkpoint,
                score_checkpoints={"post": checkpoint},
                artifact_manifest_sha256=self.binding.artifact_manifest_sha256,
                checkpoint_sha256=checkpoint_sha,
            )
            receipts.append(str(receipt))
            current = boundary
            if (boundary - start_update) % interval:
                continue
            validation = dict(self.engine.score_balanced64(self.api.artifact.windows))
            try:
                current_kld = float(validation["mean_kld"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ProductionRailsError("held-out boundary score lacks mean_kld") from exc
            decision = _heldout_decision(
                previous_kld, current_kld, streak, patience=patience
            )
            streak = int(decision["non_improving_streak"])
            boundary_row = {
                "update": boundary,
                "checkpoint": checkpoint,
                "checkpoint_sha256": checkpoint_sha,
                "mean_kld": current_kld,
                **decision,
            }
            boundaries.append(boundary_row)
            previous_kld = current_kld
            gate_path = self.receipt_root / (
                f"HELDOUT_GATES_U{start_update:03d}_U{target:03d}{rank_suffix}.json"
            )
            _atomic_json(
                gate_path,
                {
                    "schema": "banana-smasher-heldout-kill-gates-v1",
                    "status": "HALTED" if decision["halt"] else "IN_PROGRESS",
                    "start_kld": self._pre_kld,
                    "interval_updates": interval,
                    "patience": patience,
                    "boundaries": boundaries,
                },
            )
            if decision["halt"]:
                raise ProductionRailsError(
                    "held-out KLD was flat/rising at two consecutive boundaries; "
                    f"halted at U{boundary}; receipt={gate_path}"
                )
        gate_path = self.receipt_root / (
            f"HELDOUT_GATES_U{start_update:03d}_U{target:03d}{rank_suffix}.json"
        )
        _atomic_json(
            gate_path,
            {
                "schema": "banana-smasher-heldout-kill-gates-v1",
                "status": "PASS",
                "start_kld": self._pre_kld,
                "interval_updates": interval,
                "patience": patience,
                "boundaries": boundaries,
            },
        )
        completed = {
            **dict(result),
            "updates": updates,
            "accepted_update_cadence": accepted_update_cadence,
            "receipts": receipts,
            "heldout_gates": str(gate_path),
            "heldout_boundaries": boundaries,
        }
        completed["phase_release"] = dict(self._release_engine("repair_train"))
        return completed


class ProductionRails:
    """Production implementation of :class:`PipelineRails`.

    The model/session factory is invoked exactly once.  Every later activation
    uses ``hot_swap`` on that object.  Lifecycle state is durably published so a
    physical acceptance can prove this property independently of process logs.
    """

    def __init__(
        self,
        config: str | Path | Mapping[str, Any],
        *,
        run_root: str | Path,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.run_root = Path(run_root).expanduser().resolve()
        self.run_root.mkdir(parents=True, exist_ok=True)
        self._clock = clock
        if isinstance(config, Mapping):
            self.config_path = None
            self.config = dict(config)
            encoded = json.dumps(self.config, sort_keys=True, separators=(",", ":")).encode()
            self.config_sha256 = hashlib.sha256(encoded).hexdigest()
        else:
            self.config_path = Path(config).expanduser().resolve()
            try:
                raw = self.config_path.read_bytes()
                value = json.loads(raw)
            except (OSError, ValueError) as exc:
                raise ProductionRailsError(f"cannot read production rails config: {exc}") from exc
            if not isinstance(value, Mapping):
                raise ProductionRailsError("production rails config root must be an object")
            self.config = dict(value)
            self.config_sha256 = hashlib.sha256(raw).hexdigest()
        self._validate_config()
        binding_fields = {
            key: self.config.get(key)
            for key in (
                "schema",
                "pipeline_microbatch",
                "layers",
                "uniform_builder",
                "backpack_mixer",
                "score_contract",
                "continuation_science",
            )
        }
        self.provider_binding_sha256 = hashlib.sha256(
            json.dumps(binding_fields, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        self._builder = _callable(self.config["uniform_builder"], "uniform_builder")
        self._mixer = _callable(self.config["backpack_mixer"], "backpack_mixer")
        continuation = self.config.get("continuation", {})
        rank = continuation.get("rank") if isinstance(continuation, Mapping) else None
        self._rank = rank if rank in (0, 1) else None
        self._session: Any | None = None
        self._active: BackpackArtifact | None = None
        self._active_binding: _ArtifactBinding | None = None
        self._phase_state = "initialized"
        self._pre_checkpoint: str | None = None
        self._requested_updates: int | None = None
        self._started = self._clock()
        self._events: list[dict[str, Any]] = []
        self._counts = {
            "model_constructions": 0,
            "resident_loads": 0,
            "hot_swaps": 0,
            "scores": 0,
            "canary_passes": 0,
            "training_calls": 0,
            "updates": 0,
        }
        self._publish("provider_initialized", config_sha256=self.config_sha256)

    @classmethod
    def from_file(
        cls, config: str | Path, *, run_root: str | Path, clock: Callable[[], float] = time.monotonic
    ) -> "ProductionRails":
        return cls(config, run_root=run_root, clock=clock)

    def _validate_config(self) -> None:
        forbidden = _forbidden_slow_control(self.config)
        if forbidden is not None:
            raise ProductionRailsError(
                f"production resident rail forbids slowness control {forbidden}"
            )
        if self.config.get("schema") != PRODUCTION_RAILS_SCHEMA:
            raise ProductionRailsError(f"production rails schema must be {PRODUCTION_RAILS_SCHEMA}")
        if self.config.get("pipeline_microbatch") != PIPELINE_MICROBATCH:
            raise ProductionRailsError("production rails require sealed PIPELINE_MICROBATCH=4")
        layers = self.config.get("layers")
        if layers != list(ALL_LAYERS):
            raise ProductionRailsError("production rails must declare generic ordered layers 0..42")
        if "session_factory" in self.config:
            raise ProductionRailsError("production session_factory is forbidden")
        for field in ("uniform_builder", "backpack_mixer"):
            if field not in self.config:
                raise ProductionRailsError(f"production rails config is missing {field}")
        allowed = self.config.get("allowed_artifacts")
        if not isinstance(allowed, Mapping) or not allowed:
            raise ProductionRailsError("production rails require a non-empty allowed_artifacts map")
        for identity_sha, row in allowed.items():
            common_invalid = (
                not isinstance(identity_sha, str)
                or len(identity_sha) != 64
                or not isinstance(row, Mapping)
                or not isinstance(row.get("basis_sha256"), str)
                or not isinstance(row.get("checkpoint"), str)
                or not isinstance(row.get("checkpoint_sha256"), str)
            )
            mixed = (
                isinstance(row, Mapping)
                and row.get("artifact_mode") == "mixed-backpack-virtual-v1"
            )
            if mixed:
                mode_invalid = (
                    not isinstance(row.get("identity_sha256"), str)
                    or row.get("identity_sha256") != identity_sha
                    or not isinstance(row.get("virtual_manifest_sha256"), str)
                    or not isinstance(row.get("materialization_index_sha256"), str)
                    or row.get("physical_tiers")
                    != ["native_mxfp4", "qtip2", "qtip3"]
                )
            else:
                mode_invalid = not isinstance(
                    row.get("artifact_manifest_sha256"), str
                )
            if common_invalid or mode_invalid:
                raise ProductionRailsError("allowed_artifacts contains an invalid identity binding")

    def _publish(self, event: str, **fields: Any) -> None:
        row = {
            "event": event,
            "sequence": len(self._events),
            "elapsed_seconds": self._clock() - self._started,
            **fields,
        }
        self._events.append(row)
        complete_counts = {
            "model_constructions": 1,
            "resident_loads": 1,
            "hot_swaps": 2,
            "scores": 2,
            "canary_passes": 2,
            "training_calls": 1,
            "updates": self._requested_updates or 4,
        }
        status = (
            "PASS"
            if self._counts == complete_counts and self._phase_state == "completed"
            else "IN_PROGRESS"
        )
        payload = {
            "schema": "banana-smasher-resident-lifecycle-v1",
            "status": status,
            "rank": self._rank,
            "config_sha256": self.config_sha256,
            "provider_binding_sha256": self.provider_binding_sha256,
            "phase_state": self._phase_state,
            "counts": dict(self._counts),
            "events": list(self._events),
        }
        _atomic_json(self.lifecycle_path, payload)
        if self._rank is not None:
            rank_paths = [
                self.run_root / f"RESIDENT_LIFECYCLE.rank{rank}.json"
                for rank in (0, 1)
            ]
            if all(path.is_file() for path in rank_paths):
                rows = [json.loads(path.read_text()) for path in rank_paths]
                if any(
                    row.get("rank") != rank
                    or row.get("status") != "PASS"
                    or row.get("provider_binding_sha256") != self.provider_binding_sha256
                    or row.get("counts") != complete_counts
                    for rank, row in enumerate(rows)
                ):
                    return
                _atomic_json(
                    self.run_root / "RESIDENT_LIFECYCLE.json",
                    {
                        "schema": "banana-smasher-resident-lifecycle-pair-v1",
                        "status": "PASS",
                        "provider_binding_sha256": self.provider_binding_sha256,
                        "ranks": rows,
                    },
                )

    @property
    def lifecycle_path(self) -> Path:
        if self._rank is None:
            return self.run_root / "RESIDENT_LIFECYCLE.json"
        return self.run_root / f"RESIDENT_LIFECYCLE.rank{self._rank}.json"

    def _binding(self, artifact: BackpackArtifact) -> _ArtifactBinding:
        try:
            current_identity = ArtifactIdentity.load(artifact.root)
        except Exception as exc:
            raise ProductionRailsError(
                "artifact identity bytes are unreadable or invalid"
            ) from exc
        if current_identity.sha256 != artifact.identity.sha256:
            raise ProductionRailsError("artifact identity.json bytes changed after selection")
        raw = self.config["allowed_artifacts"].get(artifact.identity.sha256)
        if not isinstance(raw, Mapping):
            raise ProductionRailsError("unknown artifact identity; refusing resident operation")
        if raw.get("basis_sha256") != artifact.identity.basis_sha256:
            raise ProductionRailsError("artifact basis does not match pinned provider binding")
        if raw.get("checkpoint_sha256") != artifact.checkpoint_sha256:
            raise ProductionRailsError("artifact checkpoint does not match pinned provider binding")
        mixed = raw.get("artifact_mode") == "mixed-backpack-virtual-v1"
        if mixed:
            virtual_path = artifact.root.resolve() / "BACKPACK_VIRTUAL_MANIFEST.json"
            index_path = artifact.root.resolve() / "MATERIALIZATION_INDEX.jsonl"
            try:
                virtual_raw = virtual_path.read_bytes()
                index_raw = index_path.read_bytes()
                virtual = json.loads(virtual_raw)
                index_binding = virtual["materialization_index"]
            except (OSError, ValueError, KeyError, TypeError) as exc:
                raise ProductionRailsError("mixed artifact virtual chain is unreadable") from exc
            tiers = {
                str(tier)
                for layer in artifact.identity.composition
                for tier, count in layer["tiers"].items()
                if int(count) > 0
            }
            checkpoint = str(raw["checkpoint"])
            checkpoint_row = artifact.identity.checkpoints.get(checkpoint)
            if (
                artifact.identity.composition_kind != "mixed-per-layer-per-expert"
                or tiers != {"native_mxfp4", "qtip2", "qtip3"}
                or [row.get("layer") for row in artifact.identity.composition]
                != list(ALL_LAYERS)
                or raw.get("identity_sha256") != artifact.identity.sha256
                or hashlib.sha256(virtual_raw).hexdigest()
                != raw.get("virtual_manifest_sha256")
                or hashlib.sha256(index_raw).hexdigest()
                != raw.get("materialization_index_sha256")
                or virtual.get("basis_sha256") != artifact.identity.basis_sha256
                or index_binding.get("file") != index_path.name
                or index_binding.get("bytes") != len(index_raw)
                or index_binding.get("sha256") != hashlib.sha256(index_raw).hexdigest()
                or not isinstance(checkpoint_row, Mapping)
                or checkpoint_row.get("sha256") != raw.get("checkpoint_sha256")
            ):
                raise ProductionRailsError("mixed artifact sealed-chain identity mismatch")
            if (
                self._active is not None
                and self._active_binding is not None
                and self._active_binding.artifact_mode == "mixed-backpack-virtual-v1"
                and artifact.root.resolve() == self._active.root.resolve()
                and artifact.identity.sha256 == self._active.identity.sha256
            ):
                return self._active_binding
            return _ArtifactBinding(
                identity_sha256=artifact.identity.sha256,
                basis_sha256=artifact.identity.basis_sha256,
                checkpoint=checkpoint,
                score_checkpoints={},
                artifact_manifest_sha256="",
                checkpoint_sha256=str(raw["checkpoint_sha256"]),
                artifact_mode="mixed-backpack-virtual-v1",
                virtual_manifest_sha256=str(raw["virtual_manifest_sha256"]),
                materialization_index_sha256=str(
                    raw["materialization_index_sha256"]
                ),
            )
        runtime = artifact.identity.runtime.get("production_rails")
        if not isinstance(runtime, Mapping):
            raise ProductionRailsError("artifact identity has no production_rails binding")
        if runtime.get("provider_binding_sha256") != self.provider_binding_sha256:
            raise ProductionRailsError("artifact production_rails provider identity mismatch")
        if (
            self._active is not None
            and self._active_binding is not None
            and artifact.root.resolve() == self._active.root.resolve()
            and artifact.identity.sha256 == self._active.identity.sha256
        ):
            # The initial ARTIFACT/checkpoint bytes were admitted before model
            # construction. Continuation legitimately advances the manifest;
            # later phases use only the already-resident, identity-bound state.
            return self._active_binding
        manifest_path = artifact.root.resolve() / "ARTIFACT.json"
        try:
            manifest_raw = manifest_path.read_bytes()
            manifest = json.loads(manifest_raw)
            checkpoint_row = manifest["checkpoints"][raw["checkpoint"]]
            checkpoint_path = (
                artifact.root.resolve() / Path(checkpoint_row["path"])
            ).resolve()
            checkpoint_path.relative_to(artifact.root.resolve())
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise ProductionRailsError("artifact checkpoint binding is invalid") from exc
        if hashlib.sha256(manifest_raw).hexdigest() != raw.get("artifact_manifest_sha256"):
            if set(manifest.get("checkpoints", {})) == {str(raw["checkpoint"])}:
                raise ProductionRailsError(
                    "artifact ARTIFACT.json bytes do not match pinned binding"
                )
            base_manifest = dict(manifest)
            base_manifest["checkpoints"] = {str(raw["checkpoint"]): checkpoint_row}
            serialized = (
                json.dumps(base_manifest, sort_keys=True),
                json.dumps(base_manifest, sort_keys=True) + "\n",
                json.dumps(base_manifest, indent=2, sort_keys=True) + "\n",
                json.dumps(base_manifest, sort_keys=True, separators=(",", ":")) + "\n",
            )
            if raw.get("artifact_manifest_sha256") not in {
                hashlib.sha256(value.encode()).hexdigest() for value in serialized
            }:
                raise ProductionRailsError(
                    "artifact ARTIFACT.json is not an additive extension of the pinned binding"
                )
            try:
                prior_sha = str(raw["checkpoint_sha256"])
                prior_update = int(checkpoint_row["next_update"])
                additions = sorted(
                    (
                        int(row["next_update"]),
                        str(name),
                        row,
                    )
                    for name, row in manifest["checkpoints"].items()
                    if name != raw["checkpoint"]
                )
                for next_update, _name, row in additions:
                    candidate = (artifact.root.resolve() / Path(row["path"])).resolve()
                    candidate.relative_to(artifact.root.resolve())
                    if (
                        next_update <= prior_update
                        or row.get("parent_sha256") != prior_sha
                        or row.get("sha256") != _sha256(candidate)
                    ):
                        raise ValueError("checkpoint continuation chain mismatch")
                    prior_update = next_update
                    prior_sha = str(row["sha256"])
            except (OSError, ValueError, KeyError, TypeError) as exc:
                raise ProductionRailsError(
                    "artifact additive checkpoint continuation is invalid"
                ) from exc
        if (
            not checkpoint_path.is_file()
            or checkpoint_row.get("sha256") != raw.get("checkpoint_sha256")
            or _sha256(checkpoint_path) != raw.get("checkpoint_sha256")
        ):
            raise ProductionRailsError("artifact checkpoint bytes do not match pinned binding")
        layers = [row.get("layer") for row in artifact.identity.composition]
        if layers != list(ALL_LAYERS):
            raise ProductionRailsError(
                "resident artifact composition must cover ordered generic layers 0..42"
            )
        checkpoints = raw.get("score_checkpoints", {})
        if not isinstance(checkpoints, Mapping) or any(
            phase not in {"pre", "post"} or not isinstance(value, str)
            for phase, value in checkpoints.items()
        ):
            raise ProductionRailsError("artifact score checkpoint binding is invalid")
        return _ArtifactBinding(
            identity_sha256=artifact.identity.sha256,
            basis_sha256=artifact.identity.basis_sha256,
            checkpoint=str(raw["checkpoint"]),
            score_checkpoints={str(key): str(value) for key, value in checkpoints.items()},
            artifact_manifest_sha256=str(raw["artifact_manifest_sha256"]),
            checkpoint_sha256=str(raw["checkpoint_sha256"]),
        )

    @staticmethod
    def _require_live_checkpoint_bytes(
        artifact: BackpackArtifact, binding: _ArtifactBinding
    ) -> None:
        if binding.artifact_mode == "mixed-backpack-virtual-v1":
            return
        try:
            manifest = json.loads((artifact.root.resolve() / "ARTIFACT.json").read_text())
            row = manifest["checkpoints"][binding.checkpoint]
            checkpoint_path = (artifact.root.resolve() / Path(row["path"])).resolve()
            checkpoint_path.relative_to(artifact.root.resolve())
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise ProductionRailsError("live resident checkpoint binding is invalid") from exc
        if (
            not checkpoint_path.is_file()
            or row.get("sha256") != binding.checkpoint_sha256
            or _sha256(checkpoint_path) != binding.checkpoint_sha256
        ):
            raise ProductionRailsError(
                "live resident checkpoint bytes do not match active binding"
            )

    def build_uniform(self, model: Path, tier: str, output: Path) -> str | Path:
        return self._builder(model=model, tier=tier, output=output, config=dict(self.config))

    def mix(
        self, builds: Sequence[UniformBuild], bpw_target: float, output: Path
    ) -> str | Path:
        return self._mixer(
            builds=tuple(builds), bpw_target=float(bpw_target), output=output, config=dict(self.config)
        )

    def load_resident(self, artifact: BackpackArtifact) -> None:
        if self._session is not None:
            raise ProductionRailsError("resident model/session is already constructed")
        binding = self._binding(artifact)
        continuation = dict(self.config.get("continuation", {}))
        if binding.artifact_mode == "mixed-backpack-virtual-v1":
            self._session = _MixedProviderSession(
                artifact,
                binding,
                continuation_config=continuation,
                receipt_root=self.run_root / "receipts",
            )
        else:
            self._session = _ProvenSession(
                artifact,
                binding,
                continuation_config=continuation,
                receipt_root=self.run_root / "receipts",
                provider_binding_sha256=self.provider_binding_sha256,
            )
        self._active = artifact
        self._active_binding = binding
        self._counts["model_constructions"] += 1
        self._counts["resident_loads"] += 1
        self._phase_state = "loaded"
        self._publish("model_constructed", artifact_identity_sha256=artifact.identity.sha256)

    def hot_swap(self, artifact: BackpackArtifact) -> None:
        if self._session is None:
            raise ProductionRailsError("hot_swap requires a resident model/session")
        binding = self._binding(artifact)
        method = getattr(self._session, "hot_swap", None)
        if not callable(method):
            raise ProductionRailsError("resident session does not implement in-memory hot_swap")
        method(artifact, binding)
        self._active = artifact
        live_binding = getattr(self._session, "binding", binding)
        self._active_binding = live_binding if isinstance(live_binding, _ArtifactBinding) else binding
        self._counts["hot_swaps"] += 1
        self._publish("checkpoint_hot_swap", artifact_identity_sha256=artifact.identity.sha256)

    def restore_pre_score(
        self, artifact: BackpackArtifact, pre: Mapping[str, Any]
    ) -> None:
        """Bind sealed PRE to the newest authenticated checkpoint for training."""
        if self._session is not None:
            raise ProductionRailsError("restoring PRE requires a fresh process")
        raw = self.config["allowed_artifacts"].get(artifact.identity.sha256)
        if not isinstance(raw, Mapping):
            raise ProductionRailsError("unknown artifact identity; refusing PRE restore")
        if raw.get("basis_sha256") != artifact.identity.basis_sha256:
            raise ProductionRailsError("restored PRE basis does not match admission")
        if raw.get("artifact_mode") == "mixed-backpack-virtual-v1":
            try:
                pre_kld = float(pre["mean_kld"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ProductionRailsError("restored mixed PRE receipt is invalid") from exc
            if not math.isfinite(pre_kld):
                raise ProductionRailsError("restored mixed PRE KLD is non-finite")
            binding = self._binding(artifact)
            self._session = _MixedProviderSession(
                artifact,
                binding,
                continuation_config=dict(self.config.get("continuation", {})),
                receipt_root=self.run_root / "receipts",
            )
            self._session.restore_pre_score(pre)
            self._active = artifact
            self._active_binding = binding
            self._pre_checkpoint = binding.checkpoint
            self._phase_state = "pre_scored"
            self._counts.update(
                {
                    "model_constructions": 1,
                    "resident_loads": 1,
                    "scores": 1,
                    "canary_passes": 1,
                }
            )
            self._publish(
                "pre_score_restored",
                mean_kld=pre_kld,
                checkpoint=binding.checkpoint,
                checkpoint_sha256=binding.checkpoint_sha256,
            )
            return
        try:
            pre_kld = float(pre["mean_kld"])
            manifest = json.loads((artifact.root / "ARTIFACT.json").read_text())
            rows = [
                (int(row["next_update"]), str(name), row)
                for name, row in manifest["checkpoints"].items()
                if isinstance(row, Mapping)
            ]
            next_update, checkpoint, checkpoint_row = max(rows)
            checkpoint_sha = str(checkpoint_row["sha256"])
            checkpoint_path = (artifact.root / checkpoint_row["path"]).resolve()
            checkpoint_path.relative_to(artifact.root.resolve())
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise ProductionRailsError("restored PRE checkpoint frontier is invalid") from exc
        if (
            not math.isfinite(pre_kld)
            or next_update < 0
            or len(checkpoint_sha) != 64
            or not checkpoint_path.is_file()
            or _sha256(checkpoint_path) != checkpoint_sha
        ):
            raise ProductionRailsError("restored PRE checkpoint frontier bytes mismatch")
        binding = _ArtifactBinding(
            identity_sha256=artifact.identity.sha256,
            basis_sha256=artifact.identity.basis_sha256,
            checkpoint=checkpoint,
            score_checkpoints={"post": checkpoint},
            artifact_manifest_sha256=str(raw["artifact_manifest_sha256"]),
            checkpoint_sha256=checkpoint_sha,
        )
        self._session = _ProvenSession(
            artifact,
            binding,
            continuation_config=dict(self.config.get("continuation", {})),
            receipt_root=self.run_root / "receipts",
            provider_binding_sha256=self.provider_binding_sha256,
        )
        self._session._pre_kld = pre_kld
        self._active = artifact
        self._active_binding = binding
        self._pre_checkpoint = checkpoint
        self._phase_state = "pre_scored"
        self._counts.update(
            {
                "model_constructions": 1,
                "resident_loads": 1,
                "scores": 1,
                "canary_passes": 1,
            }
        )
        self._publish(
            "pre_score_restored",
            mean_kld=pre_kld,
            checkpoint=checkpoint,
            checkpoint_sha256=checkpoint_sha,
            next_update=next_update,
        )

    def restore_training(
        self,
        artifact: BackpackArtifact,
        pre: Mapping[str, Any],
        training: Mapping[str, Any],
    ) -> None:
        """Bind an authenticated trained checkpoint to a fresh POST process."""
        if self._session is not None:
            raise ProductionRailsError("restoring training requires a fresh process")
        raw = self.config["allowed_artifacts"].get(artifact.identity.sha256)
        if not isinstance(raw, Mapping):
            raise ProductionRailsError("unknown artifact identity; refusing training restore")
        if raw.get("basis_sha256") != artifact.identity.basis_sha256:
            raise ProductionRailsError("restored training basis does not match admission")
        if raw.get("artifact_mode") == "mixed-backpack-virtual-v1":
            try:
                pre_kld = float(pre["mean_kld"])
                updates = int(training["updates"])
                checkpoint = str(training["checkpoint"])
                checkpoint_sha = str(training["checkpoint_sha256"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ProductionRailsError(
                    "restored mixed training receipt is invalid"
                ) from exc
            if not math.isfinite(pre_kld) or updates <= 0 or len(checkpoint_sha) != 64:
                raise ProductionRailsError("restored mixed training identity mismatch")
            binding = self._binding(artifact)
            self._session = _MixedProviderSession(
                artifact,
                binding,
                continuation_config=dict(self.config.get("continuation", {})),
                receipt_root=self.run_root / "receipts",
            )
            self._session.restore_training(pre, training)
            live_binding = self._session.binding
            self._active = artifact
            self._active_binding = live_binding
            self._pre_checkpoint = binding.checkpoint
            self._requested_updates = updates
            self._phase_state = "trained"
            self._counts.update(
                {
                    "model_constructions": 1,
                    "resident_loads": 1,
                    "scores": 1,
                    "canary_passes": 1,
                    "training_calls": 1,
                    "updates": updates,
                }
            )
            self._publish(
                "training_restored",
                updates=updates,
                checkpoint=checkpoint,
                checkpoint_sha256=checkpoint_sha,
            )
            return
        try:
            pre_kld = float(pre["mean_kld"])
            updates = int(training["updates"])
            checkpoint = str(training["checkpoint"])
            checkpoint_sha = str(training["checkpoint_sha256"])
            manifest = json.loads((artifact.root / "ARTIFACT.json").read_text())
            checkpoint_row = manifest["checkpoints"][checkpoint]
            checkpoint_path = (artifact.root / checkpoint_row["path"]).resolve()
            checkpoint_path.relative_to(artifact.root.resolve())
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise ProductionRailsError("restored training checkpoint receipt is invalid") from exc
        if (
            not math.isfinite(pre_kld)
            or updates <= 0
            or len(checkpoint_sha) != 64
            or checkpoint_row.get("sha256") != checkpoint_sha
            or not checkpoint_path.is_file()
            or _sha256(checkpoint_path) != checkpoint_sha
        ):
            raise ProductionRailsError("restored training checkpoint bytes do not match receipt")
        binding = _ArtifactBinding(
            identity_sha256=artifact.identity.sha256,
            basis_sha256=artifact.identity.basis_sha256,
            checkpoint=checkpoint,
            score_checkpoints={"post": checkpoint},
            artifact_manifest_sha256=str(raw["artifact_manifest_sha256"]),
            checkpoint_sha256=checkpoint_sha,
        )
        self._session = _ProvenSession(
            artifact,
            binding,
            continuation_config=dict(self.config.get("continuation", {})),
            receipt_root=self.run_root / "receipts",
            provider_binding_sha256=self.provider_binding_sha256,
        )
        self._session._pre_kld = pre_kld
        self._active = artifact
        self._active_binding = binding
        self._pre_checkpoint = str(raw["checkpoint"])
        self._requested_updates = updates
        self._phase_state = "trained"
        self._counts.update(
            {
                "model_constructions": 1,
                "resident_loads": 1,
                "scores": 1,
                "canary_passes": 1,
                "training_calls": 1,
                "updates": updates,
            }
        )
        self._publish(
            "training_restored",
            updates=updates,
            checkpoint=checkpoint,
            checkpoint_sha256=checkpoint_sha,
        )

    def score(self, artifact: BackpackArtifact, phase: str) -> Mapping[str, Any]:
        if self._session is None or self._active is None:
            raise ProductionRailsError("score requires a resident model/session")
        if phase not in {"pre", "post"}:
            raise ProductionRailsError("score phase must be pre or post")
        required_state = "loaded" if phase == "pre" else "trained"
        if self._phase_state != required_state:
            raise ProductionRailsError(
                f"score_{phase} requires resident phase state {required_state}"
            )
        binding = self._binding(artifact)
        self._require_live_checkpoint_bytes(artifact, binding)
        if (
            self._active_binding is None
            or binding.identity_sha256 != self._active_binding.identity_sha256
            or binding.basis_sha256 != self._active_binding.basis_sha256
        ):
            raise ProductionRailsError("score artifact is not the active pinned checkpoint")
        method = getattr(self._session, "score", None)
        if not callable(method):
            raise ProductionRailsError("resident session does not implement score")
        result = dict(method(phase))
        try:
            kld = float(result["mean_kld"])
            top1 = int(result["top1_matches"])
            positions = int(result["positions"])
            counters = result["runtime_counters"]
            scored_checkpoint = result["checkpoint"]
        except (KeyError, TypeError, ValueError) as exc:
            raise ProductionRailsError("resident scorer returned an incomplete full64 receipt") from exc
        if not math.isfinite(kld):
            raise ProductionRailsError("resident scorer returned non-finite KLD")
        if binding.artifact_mode == "mixed-backpack-virtual-v1" and result.get(
            "support"
        ) != 8192:
            raise ProductionRailsError("mixed resident scorer did not prove t8192 support")
        if (
            positions != 64 * 1024
            or result.get("execution_mode") != "resident_model_in_memory"
            or not isinstance(counters, Mapping)
            or counters.get("windows") != 64
            or counters.get("checkpoint_loads_during_score") != 0
            or counters.get("candidate_file_reads_during_score") != 0
            or scored_checkpoint != binding.checkpoint
        ):
            raise ProductionRailsError("resident scorer did not prove physical full64 execution")
        score_attempt = {
            "schema": "banana-smasher-resident-score-attempt-v1",
            "status": "MEASURED_UNACCEPTED",
            "phase": phase,
            "rank": self._rank,
            "artifact_identity_sha256": artifact.identity.sha256,
            "provider_binding_sha256": self.provider_binding_sha256,
            "checkpoint": binding.checkpoint,
            "checkpoint_sha256": binding.checkpoint_sha256,
            "mean_kld": kld,
            "top1_matches": top1,
            "positions": positions,
            "timed_wall_seconds": float(result.get("timed_wall_seconds", 0.0)),
            "execution_mode": result.get("execution_mode"),
            "runtime_counters": dict(counters),
        }
        suffix = f".rank{self._rank}" if self._rank is not None else ""
        _atomic_json(
            self.run_root / f"RESIDENT_SCORE_ATTEMPT.{phase}{suffix}.json",
            score_attempt,
        )
        # The canary is selected by the exact identity admitted above.  Publish
        # no score event until the artifact-declared values pass.
        ArtifactIdentity.load(artifact.root).require_canary(
            kld=kld,
            top1=top1,
            allow_kld_improvement=phase == "post",
        )
        if phase == "pre":
            self._pre_checkpoint = binding.checkpoint
            self._phase_state = "pre_scored"
        else:
            self._phase_state = "completed"
        self._counts["scores"] += 1
        self._counts["canary_passes"] += 1
        self._publish(
            "score_published",
            phase=phase,
            artifact_identity_sha256=artifact.identity.sha256,
            mean_kld=kld,
            top1_matches=top1,
            checkpoint=binding.checkpoint,
            checkpoint_sha256=binding.checkpoint_sha256,
        )
        return result

    def score_probe(
        self, artifact: BackpackArtifact, windows: Sequence[int]
    ) -> Mapping[str, Any]:
        if self._session is None or self._active is None:
            raise ProductionRailsError("score_probe requires a resident model/session")
        binding = self._binding(artifact)
        self._require_live_checkpoint_bytes(artifact, binding)
        method = getattr(self._session, "score_probe", None)
        if not callable(method):
            raise ProductionRailsError("resident session does not implement score_probe")
        ordered = tuple(int(value) for value in windows)
        raw = method(ordered)
        if not isinstance(raw, Mapping):
            raise ProductionRailsError("resident score probe returned a non-mapping")
        result = dict(raw)
        counters = result.get("runtime_counters")
        if (
            int(result.get("positions", -1)) != len(ordered) * 1024
            or result.get("execution_mode") != "resident_model_in_memory"
            or not isinstance(counters, Mapping)
            or int(counters.get("windows", -1)) != len(ordered)
            or int(counters.get("checkpoint_loads_during_score", -1)) != 0
            or int(counters.get("candidate_file_reads_during_score", -1)) != 0
        ):
            raise ProductionRailsError(
                "resident score probe did not prove bounded in-memory execution"
            )
        return result

    def train(self, artifact: BackpackArtifact, updates: int) -> Mapping[str, Any]:
        if self._session is None or self._active is None:
            raise ProductionRailsError("train requires a resident model/session")
        if (
            isinstance(updates, bool)
            or not isinstance(updates, int)
            or updates <= 0
        ):
            raise ProductionRailsError(
                "production resident repair requires a positive update count"
            )
        if self._phase_state != "pre_scored":
            raise ProductionRailsError("resident training requires one published pre-score")
        binding = self._binding(artifact)
        if (
            self._active_binding is None
            or binding.identity_sha256 != self._active_binding.identity_sha256
            or binding.basis_sha256 != self._active_binding.basis_sha256
        ):
            raise ProductionRailsError("train artifact is not the active pinned checkpoint")
        method = getattr(self._session, "train", None)
        if not callable(method):
            raise ProductionRailsError("resident session does not implement continuation training")
        result = dict(method(updates))
        if int(result.get("updates", updates)) != updates:
            raise ProductionRailsError("resident continuation update count mismatch")
        self._counts["training_calls"] += 1
        self._counts["updates"] += updates
        self._requested_updates = updates
        live_binding = getattr(self._session, "binding", None)
        if isinstance(live_binding, _ArtifactBinding):
            self._active_binding = live_binding
        if (
            self._active_binding is None
            or self._pre_checkpoint is None
            or self._active_binding.checkpoint == self._pre_checkpoint
        ):
            raise ProductionRailsError(
                "resident training did not advance the active in-memory checkpoint"
            )
        self._phase_state = "trained"
        self._publish("resident_training_complete", updates=updates)
        return result


__all__ = [
    "ALL_LAYERS",
    "DEFAULT_IMPROVE_LR_SCALE",
    "PIPELINE_MICROBATCH",
    "PRODUCTION_RAILS_SCHEMA",
    "VALIDATED_REPAIR_RECIPE",
    "ProductionRails",
    "ProductionRailsError",
]
