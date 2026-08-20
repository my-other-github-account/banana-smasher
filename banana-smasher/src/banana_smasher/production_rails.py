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


@dataclass(frozen=True)
class _ArtifactBinding:
    identity_sha256: str
    basis_sha256: str
    checkpoint: str
    score_checkpoints: Mapping[str, str]
    artifact_manifest_sha256: str
    checkpoint_sha256: str


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


class _ProvenSession:
    """Adapter over the proven resident scorer/continuation implementation.

    One object owns the scorer caches for the whole arm.  ``hot_swap`` changes
    only the selected checkpoint; it never reopens or reconstructs the session.
    """

    def __init__(
        self,
        artifact: BackpackArtifact,
        binding: _ArtifactBinding,
        *,
        continuation_config: Mapping[str, Any],
        receipt_root: Path,
    ) -> None:
        self.root = artifact.root.resolve()
        self.api = _ProvenResidentAPI.open(self.root)
        self.binding = binding
        self.continuation_config = dict(continuation_config)
        self.receipt_root = receipt_root
        self.engine = _construct_resident_engine(
            self.api, self.binding, self.continuation_config
        )

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
        del phase
        method = getattr(self.engine, "score_balanced64", None)
        if not callable(method):
            raise ProductionRailsError("physical resident engine cannot score in memory")
        return dict(method(self.api.artifact.windows))

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
        receipt = self.receipt_root / (
            f"CONTINUATION_U{start_update:03d}_U{target:03d}{rank_suffix}.json"
        )
        method = getattr(self.api, "advance_resident_engine", None)
        if not callable(method):
            raise ProductionRailsError("resident implementation cannot advance the live engine")
        result = method(
            self.engine,
            self.binding.checkpoint,
            target,
            config=config,
            receipt_path=receipt,
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
        return {**dict(result), "updates": updates, "receipt": str(receipt)}


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
            if (
                not isinstance(identity_sha, str)
                or len(identity_sha) != 64
                or not isinstance(row, Mapping)
                or not isinstance(row.get("basis_sha256"), str)
                or not isinstance(row.get("checkpoint"), str)
                or not isinstance(row.get("artifact_manifest_sha256"), str)
                or not isinstance(row.get("checkpoint_sha256"), str)
            ):
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
            "updates": 4,
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
        runtime = artifact.identity.runtime.get("production_rails")
        if not isinstance(runtime, Mapping):
            raise ProductionRailsError("artifact identity has no production_rails binding")
        if runtime.get("provider_binding_sha256") != self.provider_binding_sha256:
            raise ProductionRailsError("artifact production_rails provider identity mismatch")
        raw = self.config["allowed_artifacts"].get(artifact.identity.sha256)
        if not isinstance(raw, Mapping):
            raise ProductionRailsError("unknown artifact identity; refusing resident operation")
        if raw.get("basis_sha256") != artifact.identity.basis_sha256:
            raise ProductionRailsError("artifact basis does not match pinned provider binding")
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
        if not manifest_path.is_file() or _sha256(manifest_path) != raw.get(
            "artifact_manifest_sha256"
        ):
            raise ProductionRailsError("artifact ARTIFACT.json bytes do not match pinned binding")
        try:
            manifest = json.loads(manifest_path.read_text())
            checkpoint_row = manifest["checkpoints"][raw["checkpoint"]]
            checkpoint_path = (
                artifact.root.resolve() / Path(checkpoint_row["path"])
            ).resolve()
            checkpoint_path.relative_to(artifact.root.resolve())
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise ProductionRailsError("artifact checkpoint binding is invalid") from exc
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
        self._session = _ProvenSession(
            artifact,
            binding,
            continuation_config=dict(self.config.get("continuation", {})),
            receipt_root=self.run_root / "receipts",
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
        # The canary is selected by the exact identity admitted above.  Publish
        # no score event until the artifact-declared values pass.
        ArtifactIdentity.load(artifact.root).require_canary(kld=kld, top1=top1)
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

    def train(self, artifact: BackpackArtifact, updates: int) -> Mapping[str, Any]:
        if self._session is None or self._active is None:
            raise ProductionRailsError("train requires a resident model/session")
        if isinstance(updates, bool) or updates != 4:
            raise ProductionRailsError("production resident arm requires exactly four updates")
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
    "PIPELINE_MICROBATCH",
    "PRODUCTION_RAILS_SCHEMA",
    "ProductionRails",
    "ProductionRailsError",
]
