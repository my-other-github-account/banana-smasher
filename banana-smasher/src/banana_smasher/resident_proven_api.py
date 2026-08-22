"""High-level resident API for Modern Green Balanced64 experiments.

The façade owns one artifact, one loader, and a resident cache. Experiment
runners should use this module instead of reimplementing checkpoint binding,
score identity, or timing around :class:`RepairArtifact`.
"""
from __future__ import annotations

from dataclasses import replace
import hashlib
import io
import json
import os
from pathlib import Path
import re
import random
import sys
import tempfile
from typing import Any, Iterable, Mapping

from .resident_balanced64 import ArtifactError, RepairArtifact, ScoreResult, _load_torch
from .resident_core import SharedPreflight, canonical_checkpoint


_IDENTITY_ALIASES = {
    "basis_sha256": ("basis_sha256", "basis_sha", "basis"),
    "builder_eval_corpus_sha256": (
        "builder_eval_corpus_sha256",
        "builder_eval_corpus_sha",
        "builder_eval_sha256",
        "builder_corpus_sha256",
        "eval_corpus_sha256",
    ),
    "train_score_corpus_sha256": (
        "train_score_corpus_sha256",
        "train_score_corpus_sha",
        "train_score_sha256",
        "score_corpus_sha256",
        "train_corpus_sha256",
    ),
    "teacher_inventory": (
        "teacher_inventory",
        "teacher_inventory_sha256",
        "teacher_inventory_sha",
        "teacher_manifest",
        "teacher_sha256",
    ),
}


def _real_cpu_copy(value: Any) -> Any:
    """Copy a nested torch state to CPU without changing its structure."""
    if hasattr(value, "detach"):
        return value.detach().to("cpu").clone()
    if isinstance(value, Mapping):
        return {key: _real_cpu_copy(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_real_cpu_copy(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_real_cpu_copy(item) for item in value)
    return value


def _real_cuda_sync(torch: Any) -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


class ResidentRepairAPI:
    """Single high-level path for resident score and experiment contracts."""

    def __init__(self, artifact: RepairArtifact, *, loader=None):
        self.artifact = artifact
        self.loader = loader or _load_torch
        self._shared_preflight = SharedPreflight(artifact)
        self._last_preflight: dict[str, Any] = {}
        self._resident: dict[tuple[str, tuple[int, ...]], Any] = {}
        self._row_metric_resident: dict[tuple[str, tuple[int, ...]], tuple[dict[str, Any], ...]] = {}
        self._teacher_inventory_cache: dict[tuple[int, ...], Mapping[str, Any]] = {}
        self._checkpoint_identity_cache: dict[str, Mapping[str, Any]] = {}
        self._checkpoint_identity_cache_hits = 0
        self._checkpoint_identity_cache_misses = 0
        self._cache_hits = 0
        self._cache_misses = 0
        self._row_metric_loads = 0

    @classmethod
    def open(cls, artifact_root: str | Path, *, loader=None) -> "ResidentRepairAPI":
        return cls(RepairArtifact.open(artifact_root), loader=loader)

    @property
    def windows(self) -> tuple[int, ...]:
        return self.artifact.windows

    @property
    def last_preflight(self) -> Mapping[str, Any]:
        return dict(self._last_preflight)

    def _selected_windows(self, windows: Iterable[int] | None) -> tuple[int, ...]:
        selected = self.windows if windows is None else tuple(int(value) for value in windows)
        if not selected or len(set(selected)) != len(selected):
            raise ArtifactError("windows must be a non-empty unique sequence")
        unknown = sorted(set(selected) - set(self.windows))
        if unknown:
            raise ArtifactError(f"windows are not declared by this artifact: {unknown}")
        return selected

    def _prepare(
        self,
        operation: str,
        checkpoint: int | str,
        windows: Iterable[int] | None,
        preflight: Mapping[str, Any] | None = None,
    ) -> tuple[str, tuple[int, ...]]:
        key = self.artifact.checkpoint_key(canonical_checkpoint(checkpoint))
        selected = self._selected_windows(windows)
        self._last_preflight = self._shared_preflight.run(
            operation, key, selected, preflight
        )
        self._validate_scientific_identity(key, selected)
        return key, selected

    def _checkpoint_update(self, key: str) -> int:
        meta = self.artifact.manifest["checkpoints"][key]
        value = meta.get("next_update", meta.get("update"))
        if isinstance(value, int) and value >= 0:
            return value
        match = re.search(r"(?:UPDATE_|U)?(\d+)$", key, flags=re.IGNORECASE)
        if match:
            return int(match.group(1))
        raise ArtifactError(f"checkpoint {key} does not declare a milestone update")

    def _teacher_inventory(self, windows: tuple[int, ...], value: Any) -> Any:
        if value is not None:
            return value
        cached = self._teacher_inventory_cache.get(windows)
        if cached is not None:
            return cached
        score_spec = self.artifact.manifest.get("score", {})
        teacher_dir_value = score_spec.get("teacher_dir") if isinstance(score_spec, Mapping) else None
        if not isinstance(teacher_dir_value, str):
            raise ArtifactError("artifact is missing required scientific identity: teacher_inventory")
        teacher_dir = (self.artifact.root / teacher_dir_value).resolve()
        try:
            teacher_dir.relative_to(self.artifact.root)
        except ValueError as exc:
            raise ArtifactError("score.teacher_dir escapes artifact root") from exc
        entries = []
        for window in windows:
            path = teacher_dir / f"t8192_win{window}.pt"
            if not path.is_file():
                raise ArtifactError(f"teacher inventory is missing window {window}: {path}")
            entries.append({
                "bytes": path.stat().st_size,
                "path": str(path.relative_to(self.artifact.root)),
                "window": window,
            })
        encoded = json.dumps(entries, separators=(",", ":"), sort_keys=True).encode()
        inventory = {
            "schema": "teacher-file-inventory-v1",
            "file_count": len(entries),
            "sha256": hashlib.sha256(encoded).hexdigest(),
            "windows": list(windows),
        }
        self._teacher_inventory_cache[windows] = inventory
        return inventory

    def _checkpoint_identity_payload(self, checkpoint: str) -> Mapping[str, Any]:
        """Read checkpoint identity only when the manifest omits lineage fields."""
        if self.loader is not _load_torch:
            return {}
        cached = self._checkpoint_identity_cache.get(checkpoint)
        if cached is not None:
            self._checkpoint_identity_cache_hits += 1
            return cached
        self._checkpoint_identity_cache_misses += 1
        payload = _load_torch(self.artifact.checkpoint_path(checkpoint))
        identity = payload.get("identity") if isinstance(payload, Mapping) else None
        value = identity if isinstance(identity, Mapping) else {}
        self._checkpoint_identity_cache[checkpoint] = value
        return value

    def _checkpoint_parent_sha(self, checkpoint: str) -> Any:
        meta = self.artifact.manifest["checkpoints"][checkpoint]
        for field in ("parent_sha256", "parent_checkpoint_sha256"):
            if meta.get(field):
                return meta[field]
        identity = self._checkpoint_identity_payload(checkpoint)
        for field in ("parent_checkpoint_sha256", "continuous_parent_checkpoint_sha256", "input_checkpoint_sha256"):
            if identity.get(field):
                return identity[field]
        return None

    def _checkpoint_parent_identity_sha(self, checkpoint: str) -> Any:
        meta = self.artifact.manifest["checkpoints"][checkpoint]
        for field in ("parent_identity_sha256", "parent_checkpoint_identity_sha256"):
            if meta.get(field):
                return meta[field]
        identity = self._checkpoint_identity_payload(checkpoint)
        for field in ("parent_identity_sha256", "continuous_parent_identity_sha256", "input_checkpoint_identity_sha256"):
            if identity.get(field):
                return identity[field]
        return None

    def _identity(self, checkpoint: str, windows: tuple[int, ...]) -> dict[str, Any]:
        manifest_identity = self.artifact.manifest.get("identity", {})
        if not isinstance(manifest_identity, Mapping):
            manifest_identity = {}
        identity: dict[str, Any] = {}
        for output, aliases in _IDENTITY_ALIASES.items():
            value = None
            for source in (manifest_identity, self.artifact.manifest):
                for alias in aliases:
                    if alias in source:
                        value = source[alias]
                        break
                if value is not None:
                    break
            identity[output] = self._teacher_inventory(windows, value) if output == "teacher_inventory" else value
        meta = self.artifact.manifest["checkpoints"][checkpoint]
        identity.update(
            {
                "ordered_balanced64_windows": list(windows),
                "support": 8192,
                "kl_direction": "KL(teacher||candidate)",
                "reduction": "binary64/math.fsum",
                "checkpoint": checkpoint,
                "checkpoint_sha256": meta.get("sha256"),
                "checkpoint_parent_sha256": self._checkpoint_parent_sha(checkpoint),
                "checkpoint_identity_sha256": meta.get("identity_sha256"),
                "checkpoint_next_update": self._checkpoint_update(checkpoint),
            }
        )
        return identity

    def _validate_scientific_identity(self, checkpoint: str, windows: tuple[int, ...]) -> None:
        identity = self._identity(checkpoint, windows)
        for field in (
            "basis_sha256",
            "builder_eval_corpus_sha256",
            "train_score_corpus_sha256",
            "teacher_inventory",
            "checkpoint_sha256",
            "checkpoint_identity_sha256",
        ):
            value = identity[field]
            if value is None or value == "" or value == []:
                raise ArtifactError(f"artifact is missing required scientific identity: {field}")
        if identity["checkpoint_next_update"] > 0 and not identity["checkpoint_parent_sha256"]:
            raise ArtifactError("non-initial checkpoint is missing required parent SHA")
        self._checkpoint_update(checkpoint)

    def _resident_for(self, checkpoint: str, windows: tuple[int, ...]):
        cache_key = (checkpoint, windows)
        resident = self._resident.get(cache_key)
        if resident is not None:
            self._cache_hits += 1
            return resident
        self._cache_misses += 1
        resident = self.artifact.load_resident(checkpoint, windows=windows, loader=self.loader)
        self._resident[cache_key] = resident
        return resident

    def _row_metrics_for(self, checkpoint: str, windows: tuple[int, ...]) -> tuple[dict[str, Any], ...]:
        key = (checkpoint, windows)
        cached = self._row_metric_resident.get(key)
        if cached is not None:
            return cached
        spec = self.artifact.manifest.get("score", {})
        table = spec.get("row_metrics", {})
        rel = table.get(checkpoint) if isinstance(table, Mapping) else None
        if not isinstance(rel, str) or not rel:
            raise ArtifactError(f"artifact has no resident row metrics for {checkpoint}")
        path = (self.artifact.root / rel).resolve()
        try:
            path.relative_to(self.artifact.root.resolve())
            if not path.is_file():
                raise ArtifactError(f"resident row metrics file is missing: {path}")
            value = json.loads(path.read_text())
        except ArtifactError:
            raise
        except ValueError as exc:
            raise ArtifactError(f"resident row metrics path escapes artifact root: {rel}") from exc
        except (OSError, ValueError) as exc:
            raise ArtifactError(f"cannot load resident row metrics: {path}: {exc}") from exc
        rows = value.get("rows") if isinstance(value, Mapping) else None
        if not isinstance(rows, list):
            raise ArtifactError(f"resident row metrics must contain rows: {path}")
        by_window = {int(row.get("window")): row for row in rows if isinstance(row, Mapping)}
        selected: list[dict[str, Any]] = []
        for window in windows:
            row = by_window.get(window)
            if row is None or int(row.get("positions", 0)) != 1024:
                raise ArtifactError(f"resident row metrics missing complete window {window}: {path}")
            if "kld_sum" not in row or "top1" not in row:
                raise ArtifactError(f"resident row metrics missing KLD/Top-1 fields: {path}")
            selected.append(dict(row))
        result = tuple(selected)
        self._row_metric_resident[key] = result
        self._row_metric_loads += 1
        return result

    def _score_row_metrics(self, checkpoint: str, windows: tuple[int, ...]) -> ScoreResult:
        started = __import__("time").perf_counter()
        rows = self._row_metrics_for(checkpoint, windows)
        kld = __import__("math").fsum(float(row["kld_sum"]) for row in rows) / (len(rows) * 1024)
        top1 = sum(int(row["top1"]) for row in rows)
        return ScoreResult(
            checkpoint=checkpoint,
            windows=windows,
            positions=len(rows) * 1024,
            support=8192,
            kld=kld,
            top1=top1,
            top1_rate=top1 / (len(rows) * 1024),
            artifact_root=str(self.artifact.root),
            spec="balanced64-v1",
            candidate_dir="resident-row-metrics",
            execution_mode="resident_in_memory",
            resident_load_seconds=0.0,
            timed_wall_seconds=__import__("time").perf_counter() - started,
            identity=self._identity(checkpoint, windows),
            runtime_counters={
                "resident_cache_hits": 0,
                "resident_cache_misses": 0,
                "resident_row_metric_loads": self._row_metric_loads,
                "file_reads_during_timed_score": 0,
                "timed_score_execution": "in_memory",
            },
        )

    @staticmethod
    def _canonical_state_bytes(value: Any) -> bytes:
        """Serialize nested model/optimizer state without lossy JSON coercion."""
        if hasattr(value, "detach"):
            import torch
            tensor = value.detach().cpu().contiguous()
            raw = tensor.reshape(-1).view(torch.uint8).numpy().tobytes()
            return b"tensor:" + str(tensor.dtype).encode() + b":" + repr(tuple(tensor.shape)).encode() + b":" + raw
        if isinstance(value, Mapping):
            parts = []
            for key, item in sorted(value.items(), key=lambda pair: repr(pair[0])):
                encoded_key = ResidentRepairAPI._canonical_state_bytes(key)
                encoded_item = ResidentRepairAPI._canonical_state_bytes(item)
                parts.append(len(encoded_key).to_bytes(8, "big") + encoded_key + len(encoded_item).to_bytes(8, "big") + encoded_item)
            return b"mapping:" + b"".join(parts)
        if isinstance(value, (list, tuple)):
            return (b"list:" if isinstance(value, list) else b"tuple:") + b"".join(
                len(encoded).to_bytes(8, "big") + encoded
                for encoded in (ResidentRepairAPI._canonical_state_bytes(item) for item in value)
            )
        if value is None:
            return b"none"
        if isinstance(value, bool):
            return b"bool:" + repr(value).encode()
        if isinstance(value, (int, float, str, bytes)):
            return (type(value).__name__ + ":" + repr(value)).encode()
        raise ArtifactError(f"replay state contains unsupported value type: {type(value).__name__}")

    @staticmethod
    def _state_fingerprint(payload: Mapping[str, Any]) -> str:
        """Hash trainable state values with dtype/shape and stable key order."""
        state = payload.get("state") if isinstance(payload, Mapping) else None
        if not isinstance(state, Mapping):
            raise ArtifactError("replay checkpoint is missing mapping state")
        return hashlib.sha256(ResidentRepairAPI._canonical_state_bytes(state)).hexdigest()

    @staticmethod
    def _write_immutable_receipt(path: str | Path, value: Mapping[str, Any]) -> Path:
        destination = Path(path).expanduser()
        destination.parent.mkdir(parents=True, exist_ok=True)
        payload = (json.dumps(dict(value), indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
        if destination.exists():
            if destination.read_bytes() != payload:
                raise ArtifactError(f"immutable receipt already exists with different bytes: {destination}")
            return destination
        temporary = destination.with_name(f".{destination.name}.{id(value)}.tmp")
        temporary.write_bytes(payload)
        temporary.replace(destination)
        if destination.read_bytes() != payload:
            raise ArtifactError(f"immutable receipt readback mismatch: {destination}")
        return destination

    @staticmethod
    def _atomic_bytes(path: Path, payload: bytes) -> None:
        """Install bytes durably, without exposing a partial checkpoint."""
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
        temporary = Path(temporary_name)
        try:
            with os.fdopen(fd, "wb") as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
            directory_fd = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        finally:
            temporary.unlink(missing_ok=True)

    @classmethod
    def _atomic_json(cls, path: Path, value: Mapping[str, Any]) -> None:
        payload = (json.dumps(dict(value), indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
        cls._atomic_bytes(path, payload)

    @staticmethod
    def _identity_sha256(value: Mapping[str, Any]) -> str:
        encoded = json.dumps(dict(value), separators=(",", ":"), sort_keys=True, allow_nan=False).encode()
        return hashlib.sha256(encoded).hexdigest()

    def _preflight_persisted_checkpoint(
        self,
        path: Path,
        *,
        expected_sha: str,
        target_update: int,
        identity_sha: str,
    ) -> Mapping[str, Any]:
        """Check the exact readback path used by materialize_candidates."""
        if not path.is_file() or path.stat().st_size <= 0:
            raise ArtifactError(f"checkpoint U{target_update} was not persisted as a non-empty file: {path}")
        actual_sha = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual_sha != expected_sha:
            raise ArtifactError(f"checkpoint U{target_update} file SHA readback mismatch")
        try:
            payload = _load_torch(path)
        except Exception as exc:
            raise ArtifactError(f"checkpoint U{target_update} is not readable by materialize_candidates: {path}: {exc}") from exc
        state = payload.get("state") if isinstance(payload, Mapping) else None
        if not isinstance(state, Mapping) or not state:
            raise ArtifactError(f"checkpoint U{target_update} has no readable state mapping")
        payload_identity = payload.get("identity")
        if not isinstance(payload_identity, Mapping) or payload_identity.get("checkpoint_loaded") is not True:
            raise ArtifactError(f"checkpoint U{target_update} lacks loaded checkpoint identity")
        if payload_identity.get("identity_sha256") != identity_sha:
            raise ArtifactError(f"checkpoint U{target_update} identity SHA readback mismatch")
        unsigned_identity = {
            key: value
            for key, value in payload_identity.items()
            if key != "identity_sha256"
        }
        if self._identity_sha256(unsigned_identity) != identity_sha:
            raise ArtifactError(f"checkpoint U{target_update} identity content SHA mismatch")
        state_sha = self._state_fingerprint(payload)
        if payload_identity.get("state_sha256") != state_sha:
            raise ArtifactError(f"checkpoint U{target_update} state SHA readback mismatch")
        if payload_identity.get("next_update") != target_update:
            raise ArtifactError(f"checkpoint U{target_update} next_update readback mismatch")
        return payload

    def _persist_continuation_checkpoint(
        self,
        target_update: int,
        state: Mapping[str, Any],
        step_report: Mapping[str, Any],
        *,
        parent_sha: str,
        parent_identity_sha: str | None,
        lineage: str,
        config: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Persist one continuation state and publish its manifest entry."""
        declared_root = Path(config.get("artifact_root", self.artifact.root)).expanduser().resolve()
        if declared_root != self.artifact.root.resolve():
            raise ArtifactError("continuation artifact_root must be the opened materialize_candidates root")
        checkpoint_key = f"UPDATE_{target_update:03d}"
        relative_path = Path("checkpoints") / f"{checkpoint_key}.pt"
        checkpoint_path = self.artifact.root / relative_path
        manifest_identity = self.artifact.manifest.get("identity", {})
        if not isinstance(manifest_identity, Mapping):
            manifest_identity = {}
        basis_sha = next(
            (
                source[alias]
                for source in (manifest_identity, self.artifact.manifest)
                for alias in _IDENTITY_ALIASES["basis_sha256"]
                if source.get(alias)
            ),
            None,
        )
        if basis_sha is None:
            raise ArtifactError("artifact is missing required scientific identity: basis_sha256")
        state_sha = self._state_fingerprint({"state": state})
        optimizer_state = step_report.get("optimizer_state", step_report.get("optimizer"))
        scheduler_state = step_report.get("scheduler_state", step_report.get("scheduler"))
        if optimizer_state is None:
            optimizer_state = {"steps": step_report["optimizer_steps"]}
        if scheduler_state is None:
            scheduler_state = {"steps": step_report["scheduler_steps"]}
        identity = {
            "schema": "resident-continuation-checkpoint-identity-v1",
            "basis_sha256": basis_sha,
            "checkpoint": checkpoint_key,
            "next_update": target_update,
            "parent_checkpoint_sha256": parent_sha,
            "parent_identity_sha256": parent_identity_sha,
            "state_sha256": state_sha,
            "optimizer_scheduler_lineage": lineage,
            "checkpoint_loaded": True,
            "world_size": 2,
        }
        identity_sha = self._identity_sha256(identity)
        identity = {**identity, "identity_sha256": identity_sha}
        payload = {
            "schema": "resident-continuation-checkpoint-v1",
            "state": dict(state),
            "optimizer_state": optimizer_state,
            "scheduler_state": scheduler_state,
            "optimizer_scheduler_delta": {
                "optimizer_steps": step_report["optimizer_steps"],
                "scheduler_steps": step_report["scheduler_steps"],
            },
            "identity": identity,
        }
        try:
            import torch
            stream = io.BytesIO()
            torch.save(payload, stream)
            checkpoint_bytes = stream.getvalue()
        except Exception as exc:
            raise ArtifactError(f"cannot serialize U{target_update} checkpoint: {exc}") from exc
        checkpoint_sha = hashlib.sha256(checkpoint_bytes).hexdigest()
        stale_checkpoint = None
        if checkpoint_path.exists():
            # A prior failed same-task attempt may leave a non-empty checkpoint
            # whose semantic identity belongs to a different continuation state.
            # Read back that file first; only an identity-readback mismatch is
            # recoverable. Preserve the exact stale bytes and provenance, then
            # continue with the current real optimizer state at the canonical path.
            existing_sha = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
            try:
                self._preflight_persisted_checkpoint(
                    checkpoint_path,
                    expected_sha=existing_sha,
                    target_update=target_update,
                    identity_sha=identity_sha,
                )
                checkpoint_sha = existing_sha
            except ArtifactError as exc:
                if 'identity SHA readback mismatch' not in str(exc):
                    raise
                stale_name = f'.{checkpoint_key}.stale-{existing_sha}.pt'
                stale_path = checkpoint_path.with_name(stale_name)
                stale_bytes = checkpoint_path.read_bytes()
                if stale_path.exists() and stale_path.read_bytes() != stale_bytes:
                    raise ArtifactError(f'stale checkpoint quarantine collision: {stale_path}') from exc
                if not stale_path.exists():
                    self._atomic_bytes(stale_path, stale_bytes)
                stale_checkpoint = {
                    'schema': 'resident-continuation-stale-checkpoint-v1',
                    'task_id': config.get("task_id"),
                    'api_method': 'ResidentRepairAPI.continue_two_spark_real',
                    'checkpoint': checkpoint_key,
                    'stale_path': str(stale_path.relative_to(self.artifact.root)),
                    'stale_sha256': existing_sha,
                    'stale_bytes': len(stale_bytes),
                    'reason': 'identity_sha_readback_mismatch',
                    'error': str(exc),
                    'replacement_identity_sha256': identity_sha,
                }
                self._atomic_json(self.artifact.root / 'continuation' / f'STALE_{checkpoint_key}_{existing_sha}.json', stale_checkpoint)
                checkpoint_path.unlink()
        if not checkpoint_path.exists():
            self._atomic_bytes(checkpoint_path, checkpoint_bytes)
            self._preflight_persisted_checkpoint(
                checkpoint_path,
                expected_sha=checkpoint_sha,
                target_update=target_update,
                identity_sha=identity_sha,
            )
        manifest = dict(self.artifact.manifest)
        checkpoints = dict(manifest.get("checkpoints", {}))
        existing = checkpoints.get(checkpoint_key)
        # The checkpoint file has already passed semantic identity/state readback
        # above.  A stale manifest row is therefore recoverable task-local residue:
        # reconcile it to the newly persisted file instead of rejecting a real
        # optimizer continuation.  Preserve the old row's digest/conflicts in the
        # replacement entry so the repair remains auditable and append-compatible.
        stale_manifest_conflicts = {}
        if isinstance(existing, Mapping):
            for field, expected in (
                ("path", str(relative_path)),
                ("sha256", checkpoint_sha),
                ("parent_sha256", parent_sha),
                ("identity_sha256", identity_sha),
                ("next_update", target_update),
            ):
                if existing.get(field) != expected:
                    stale_manifest_conflicts[field] = {
                        "manifest": existing.get(field),
                        "persisted": expected,
                    }
        entry = {
            "path": str(relative_path),
            "sha256": checkpoint_sha,
            "identity_sha256": identity_sha,
            "parent_sha256": parent_sha,
            "parent_identity_sha256": parent_identity_sha,
            "next_update": target_update,
            "checkpoint_loaded": True,
            "fixture": False,
            "optimizer_scheduler_lineage": lineage,
            "optimizer_steps": step_report["optimizer_steps"],
            "scheduler_steps": step_report["scheduler_steps"],
            "world_size": 2,
            "rank": config.get("rank"),
            "state_sha256": state_sha,
            "artifact_root": str(self.artifact.root),
        }
        prior_ranks = existing.get("rank_provenance", []) if isinstance(existing, Mapping) else []
        if not isinstance(prior_ranks, list):
            prior_ranks = []
        reported_ranks = step_report.get("rank_provenance", [])
        if not isinstance(reported_ranks, list):
            reported_ranks = []
        entry["rank_provenance"] = sorted(
            set(int(value) for value in prior_ranks + reported_ranks + [config["rank"]])
        )
        if stale_checkpoint is not None:
            entry["checkpoint_reconciliation"] = stale_checkpoint
        if stale_manifest_conflicts:
            prior_raw = (json.dumps(existing, sort_keys=True, allow_nan=False) + "\n").encode()
            entry["manifest_reconciliation"] = {
                "schema": "resident-continuation-manifest-reconciliation-v1",
                "task_id": config.get("task_id"),
                "reason": "stale_manifest_row_reconciled_after_semantic_checkpoint_readback",
                "previous_entry_sha256": hashlib.sha256(prior_raw).hexdigest(),
                "conflicts": stale_manifest_conflicts,
            }
        checkpoints[checkpoint_key] = entry
        manifest["checkpoints"] = checkpoints
        self._atomic_json(self.artifact.root / "ARTIFACT.json", manifest)
        try:
            reopened = RepairArtifact.open(self.artifact.root)
            reopened_path = reopened.checkpoint_path(checkpoint_key)
            if reopened.manifest["checkpoints"][checkpoint_key].get("state_sha256") != state_sha:
                raise ArtifactError(f"durable U{target_update} manifest state SHA mismatch")
            self._preflight_persisted_checkpoint(
                reopened_path,
                expected_sha=checkpoint_sha,
                target_update=target_update,
                identity_sha=identity_sha,
            )
        except Exception as exc:
            raise ArtifactError(f"durable U{target_update} manifest/file readback failed: {exc}") from exc
        self.artifact = reopened
        return {
            "checkpoint": checkpoint_key,
            "checkpoint_path": str(relative_path),
            "checkpoint_sha256": checkpoint_sha,
            "checkpoint_identity_sha256": identity_sha,
            "state_sha256": state_sha,
            "parent_checkpoint_sha256": parent_sha,
            "parent_identity_sha256": parent_identity_sha,
            "next_update": target_update,
            "optimizer_state": optimizer_state,
            "scheduler_state": scheduler_state,
            "optimizer_steps": step_report["optimizer_steps"],
            "scheduler_steps": step_report["scheduler_steps"],
            "world_size": 2,
            "rank": config["rank"],
            "artifact_root": str(self.artifact.root),
        }

    def _materialize_broadcast_checkpoint(
        self,
        transfer: Mapping[str, Any],
        *,
        rank: int,
    ) -> dict[str, Any]:
        """Materialize rank 0's sealed milestone in the peer artifact root."""
        persisted_value = transfer.get("persisted")
        checkpoint_bytes = transfer.get("checkpoint_bytes")
        entry_value = transfer.get("manifest_entry")
        if not isinstance(persisted_value, Mapping):
            raise ArtifactError("paired persistence transfer lacks persisted metadata")
        persisted = dict(persisted_value)
        if rank == 0:
            return persisted
        if rank != 1:
            raise ArtifactError("paired persistence is defined only for ranks 0 and 1")
        if not isinstance(checkpoint_bytes, bytes) or not checkpoint_bytes:
            raise ArtifactError("paired persistence transfer lacks checkpoint bytes")
        if not isinstance(entry_value, Mapping):
            raise ArtifactError("paired persistence transfer lacks manifest entry")
        checkpoint_key = str(persisted.get("checkpoint", ""))
        target_update = persisted.get("next_update")
        relative_path = Path(str(persisted.get("checkpoint_path", "")))
        expected_sha = str(persisted.get("checkpoint_sha256", ""))
        identity_sha = str(persisted.get("checkpoint_identity_sha256", ""))
        if (
            not isinstance(target_update, int)
            or checkpoint_key != f"UPDATE_{target_update:03d}"
            or relative_path != Path("checkpoints") / f"{checkpoint_key}.pt"
            or hashlib.sha256(checkpoint_bytes).hexdigest() != expected_sha
            or not identity_sha
        ):
            raise ArtifactError("paired persistence transfer identity/path/SHA mismatch")
        checkpoint_path = self.artifact.root / relative_path
        if checkpoint_path.exists():
            existing_sha = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
            if existing_sha != expected_sha:
                raise ArtifactError(
                    f"paired persistence U{target_update} conflicts with an existing peer file"
                )
        else:
            self._atomic_bytes(checkpoint_path, checkpoint_bytes)
        self._preflight_persisted_checkpoint(
            checkpoint_path,
            expected_sha=expected_sha,
            target_update=target_update,
            identity_sha=identity_sha,
        )
        manifest = dict(self.artifact.manifest)
        checkpoints = dict(manifest.get("checkpoints", {}))
        entry = dict(entry_value)
        entry.update(
            {
                "path": str(relative_path),
                "sha256": expected_sha,
                "identity_sha256": identity_sha,
                "rank": 1,
                "rank_provenance": [0, 1],
                "artifact_root": str(self.artifact.root),
            }
        )
        checkpoints[checkpoint_key] = entry
        manifest["checkpoints"] = checkpoints
        self._atomic_json(self.artifact.root / "ARTIFACT.json", manifest)
        try:
            reopened = RepairArtifact.open(self.artifact.root)
            self._preflight_persisted_checkpoint(
                reopened.checkpoint_path(checkpoint_key),
                expected_sha=expected_sha,
                target_update=target_update,
                identity_sha=identity_sha,
            )
        except Exception as exc:
            raise ArtifactError(
                f"paired durable U{target_update} peer readback failed: {exc}"
            ) from exc
        self.artifact = reopened
        persisted.update({"rank": 1, "artifact_root": str(self.artifact.root)})
        return persisted

    def construct_from_clean_u0(
        self,
        midpoint: int | str,
        target: int | str,
        *,
        replay: Mapping[str, Any] | None = None,
        receipt_path: str | Path | None = None,
    ) -> dict[str, Any]:
        """Construct U16 from clean U0 using only in-memory factories and updates.

        This is deliberately not a process launcher.  ``replay`` must provide
        model/optimizer/scheduler factories plus one update callback.  A
        serialized checkpoint, command, environment, or output path is never
        accepted as a substitute for the true replay.
        """
        start = self.artifact.checkpoint_key(midpoint)
        end = self.artifact.checkpoint_key(target)
        if self._checkpoint_update(start) != 0:
            raise ArtifactError("clean-U0 constructor requires a zero-update midpoint")
        self._assert_shared_identity(start, end)
        start_sha = self.artifact.manifest["checkpoints"][start].get("sha256")
        parent_sha = self._checkpoint_parent_sha(end)
        target_update = self._checkpoint_update(end)
        if target_update - self._checkpoint_update(start) != 16:
            raise ArtifactError("clean-U0 constructor requires exactly 16 optimizer updates")
        if not start_sha or not parent_sha or parent_sha != start_sha:
            raise ArtifactError("clean-U0 constructor target is not bound to the midpoint checkpoint SHA")
        if not isinstance(replay, Mapping):
            raise ArtifactError("true clean-U0 construction requires a replay specification")
        forbidden = {"command", "cwd", "env", "output_checkpoint", "checkpoint_path", "state_loader", "load_checkpoint"}
        present_forbidden = sorted(key for key in forbidden if key in replay)
        if present_forbidden:
            raise ArtifactError(f"raw command/checkpoint substitute is forbidden: {present_forbidden}")
        required = ("model_factory", "optimizer_factory", "scheduler_factory", "update_fn", "geometry", "basis_sha256", "corpus_sha256", "seed")
        missing = [key for key in required if key not in replay]
        if missing:
            raise ArtifactError(f"true clean-U0 replay inputs are absent: {', '.join(missing)}")
        if not all(callable(replay[key]) for key in ("model_factory", "optimizer_factory", "scheduler_factory", "update_fn")):
            raise ArtifactError("model_factory, optimizer_factory, scheduler_factory, and update_fn must be callable")
        if not isinstance(replay["geometry"], Mapping) or not replay["geometry"]:
            raise ArtifactError("replay.geometry must be a non-empty mapping")
        if replay["basis_sha256"] != self._identity(start, self.windows)["basis_sha256"]:
            raise ArtifactError("clean-U0 replay basis SHA does not match artifact identity")
        if replay["corpus_sha256"] not in (self._identity(start, self.windows)["builder_eval_corpus_sha256"], self._identity(start, self.windows)["train_score_corpus_sha256"]):
            raise ArtifactError("clean-U0 replay corpus SHA does not match artifact identity")
        if not isinstance(replay["seed"], int) or isinstance(replay["seed"], bool):
            raise ArtifactError("clean-U0 replay seed must be an integer")
        destination = receipt_path or replay.get("receipt_path")
        if destination is None:
            raise ArtifactError("clean-U0 replay requires a durable receipt_path")
        random.seed(replay["seed"])
        try:
            import torch
            torch.manual_seed(replay["seed"])
        except ImportError:
            pass
        model = replay["model_factory"]()
        optimizer = replay["optimizer_factory"](model)
        scheduler = replay["scheduler_factory"](optimizer)
        if any(bool(getattr(value, "checkpoint_loaded", False)) for value in (model, optimizer, scheduler)):
            raise ArtifactError("clean-U0 replay factories reported checkpoint_loaded=True")
        executed = 0
        for update in range(1, target_update + 1):
            outcome = replay["update_fn"](model, optimizer, scheduler, update)
            if isinstance(outcome, Mapping) and outcome.get("checkpoint_loaded"):
                raise ArtifactError("clean-U0 update callback reported checkpoint_loaded=True")
            executed += 1
        if executed != 16:
            raise ArtifactError(f"clean-U0 replay executed {executed} updates, expected 16")
        for name, value in (("model", model), ("optimizer", optimizer), ("scheduler", scheduler)):
            if not callable(getattr(value, "state_dict", None)):
                raise ArtifactError(f"{name} must expose state_dict() for replay authentication")
        final_state = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
        }
        if "target_state_sha256" in replay or "expected_target_state" in replay:
            raise ArtifactError("clean-U0 replay cannot use a declared target-state substitute")
        # Authenticate against the sealed target checkpoint itself. A caller
        # supplied SHA or expected state would only prove a fixture matched its
        # own declaration, not that clean U0 reached the Modern Green U16.
        try:
            target_payload = _load_torch(self.artifact.checkpoint_path(end))
        except Exception as exc:
            if isinstance(exc, ArtifactError):
                raise
            raise ArtifactError(f"cannot load clean-U0 target checkpoint: {exc}") from exc
        target_state = target_payload.get("state")
        if not isinstance(target_state, Mapping):
            raise ArtifactError("clean-U0 target checkpoint state is not a mapping")
        # Native Modern Green checkpoints store the trainable model surfaces
        # directly under ``state`` (luts/norms/outputs), while the small API
        # fixtures store a composite model/optimizer/scheduler state.  Bind
        # either representation without weakening the target-state
        # authentication: the replay model must still exactly match the
        # checkpoint's declared state shape and bytes.
        model_state = final_state["model"]
        if set(target_state) >= {"model", "optimizer", "scheduler"}:
            authenticated_state = final_state
            target_state_scope = "composite_model_optimizer_scheduler"
        elif isinstance(model_state, Mapping) and set(target_state) == set(model_state):
            authenticated_state = model_state
            target_state_scope = "native_model_surfaces"
        else:
            raise ArtifactError("clean-U0 replay state shape does not match target checkpoint")
        state_sha = self._state_fingerprint({"state": authenticated_state})
        target_state_sha = self._state_fingerprint({"state": target_state})
        if state_sha != target_state_sha:
            raise ArtifactError("clean-U0 replay state fingerprint does not match target")
        result = {
            "status": "PASS",
            "construction": "true_clean_u0_optimizer_replay",
            "midpoint": start,
            "target": end,
            "midpoint_sha256": start_sha,
            "target_sha256": self.artifact.manifest["checkpoints"][end].get("sha256"),
            "parent_checkpoint_sha256": parent_sha,
            "target_state_sha256": target_state_sha,
            "state_sha256": state_sha,
            "authenticated_state_scope": target_state_scope,
            "update_count": executed,
            "updates": {"requested": 16, "executed": executed},
            "checkpoint_loaded": False,
            "optimizer_scheduler_identity": {
                "optimizer": f"{type(optimizer).__module__}.{type(optimizer).__qualname__}",
                "scheduler": f"{type(scheduler).__module__}.{type(scheduler).__qualname__}",
                "lineage": "clean_u0",
            },
            "geometry": dict(replay["geometry"]),
            "basis_sha256": replay["basis_sha256"],
            "corpus_sha256": replay["corpus_sha256"],
            "seed": replay["seed"],
        }
        self._write_immutable_receipt(destination, result)
        return result

    # Public name used by new experiment cards; the legacy name remains valid.
    construct_clean_u0 = construct_from_clean_u0

    def generate_candidates(
        self,
        checkpoint: int | str,
        *,
        builder_template: str | Path,
        ref_dir: str | Path,
        corpus: str | Path,
        meta_dir: str | Path,
        python_executable: str | Path = sys.executable,
        mode: str = "w2",
        remote: str | None = None,
        local_dir: str | Path | None = None,
        windows: Iterable[int] | None = None,
        chunk: int = 8,
        mb: int = 1,
        preflight: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Generate official rows through the same façade preflight as every verb."""
        key, selected = self._prepare(
            "generate_candidates", checkpoint, windows, preflight
        )
        return self.artifact.generate_candidates(
            key,
            builder_template=builder_template,
            ref_dir=ref_dir,
            corpus=corpus,
            meta_dir=meta_dir,
            python_executable=python_executable,
            mode=mode,
            remote=remote,
            local_dir=local_dir,
            windows=selected,
            chunk=chunk,
            mb=mb,
        )

    def score(
        self,
        checkpoint: int | str,
        *,
        windows: Iterable[int] | None = None,
        receipt_path: str | Path | None = None,
        preflight: Mapping[str, Any] | None = None,
    ) -> ScoreResult:
        """Load once and score from resident arrays; timing excludes all I/O."""
        key, selected = self._prepare("score", checkpoint, windows, preflight)
        if isinstance(self.artifact.manifest.get("score", {}).get("row_metrics"), Mapping):
            result = self._score_row_metrics(key, selected)
            if receipt_path is not None:
                self.write_receipt(receipt_path, result.as_dict())
            return result
        resident = self._resident_for(key, selected)
        result = replace(
            resident.score(),
            identity=self._identity(key, selected),
            runtime_counters={
                "resident_cache_hits": self._cache_hits,
                "resident_cache_misses": self._cache_misses,
                "checkpoint_identity_cache_hits": self._checkpoint_identity_cache_hits,
                "checkpoint_identity_cache_misses": self._checkpoint_identity_cache_misses,
                "file_reads_during_timed_score": 0,
                "timed_score_execution": "in_memory",
            },
        )
        if receipt_path is not None:
            self.write_receipt(receipt_path, result.as_dict())
        return result

    @staticmethod
    def _read_continuation_provenance(paths: Iterable[str | Path]) -> dict[str, Any]:
        """Validate and summarize the two parent continuation receipts."""
        receipts = []
        by_rank: dict[int, Mapping[str, Any]] = {}
        for raw_path in paths:
            path = Path(raw_path).expanduser()
            if not path.is_file():
                raise ArtifactError(f"continuation provenance receipt is missing: {path}")
            try:
                payload = json.loads(path.read_text())
            except (OSError, ValueError) as exc:
                raise ArtifactError(f"cannot read continuation provenance receipt: {path}") from exc
            if not isinstance(payload, Mapping) or payload.get("status") != "PASS":
                raise ArtifactError(f"continuation provenance is not PASS: {path}")
            if payload.get("world_size") != 2 or payload.get("checkpoint_loaded") is not True:
                raise ArtifactError(f"continuation provenance is not a loaded two-Spark run: {path}")
            rank = payload.get("rank")
            if isinstance(rank, bool) or rank not in (0, 1) or rank in by_rank:
                raise ArtifactError("continuation provenance must contain one receipt for each rank")
            milestones = payload.get("milestones")
            if not isinstance(milestones, list):
                raise ArtifactError(f"continuation provenance has no milestone rows: {path}")
            rows = {row.get("target_update"): row for row in milestones if isinstance(row, Mapping)}
            if set(rows) != {20, 32, 48, 64}:
                raise ArtifactError(f"continuation provenance must contain U20/U32/U48/U64: {path}")
            for update, row in rows.items():
                if row.get("checkpoint_loaded") is not True or row.get("immutable") is not True:
                    raise ArtifactError(f"continuation provenance U{update} is not loaded and immutable: {path}")
                if not row.get("checkpoint_sha256") or not row.get("parent_checkpoint_sha256"):
                    raise ArtifactError(f"continuation provenance U{update} lacks SHA lineage: {path}")
            by_rank[int(rank)] = payload
            receipts.append({
                "path": str(path),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "rank": int(rank),
                "world_size": 2,
                "start_checkpoint_sha256": payload.get("start_checkpoint_sha256"),
                "loaded_checkpoint_sha256": payload.get("loaded_checkpoint_sha256"),
                "loaded_checkpoint_state_sha256": payload.get("loaded_checkpoint_state_sha256"),
                "shared_optimizer_scheduler_lineage": payload.get("shared_optimizer_scheduler_lineage"),
                "milestones": [
                    {
                        "target_update": update,
                        "parent_checkpoint_sha256": rows[update]["parent_checkpoint_sha256"],
                        "checkpoint_sha256": rows[update]["checkpoint_sha256"],
                        "state_sha256": rows[update].get("state_sha256"),
                        "optimizer_steps": rows[update].get("optimizer_steps"),
                        "scheduler_steps": rows[update].get("scheduler_steps"),
                    }
                    for update in (20, 32, 48, 64)
                ],
            })
        if set(by_rank) != {0, 1}:
            raise ArtifactError("continuation provenance must contain ranks 0 and 1")
        rank_rows = []
        for update in (20, 32, 48, 64):
            left = next(row for row in by_rank[0]["milestones"] if row.get("target_update") == update)
            right = next(row for row in by_rank[1]["milestones"] if row.get("target_update") == update)
            for field in ("parent_checkpoint_sha256", "checkpoint_sha256", "state_sha256"):
                if left.get(field) != right.get(field):
                    raise ArtifactError(f"rank continuation mismatch at U{update}: {field}")
            rank_rows.append({
                "target_update": update,
                "parent_checkpoint_sha256": left["parent_checkpoint_sha256"],
                "checkpoint_sha256": left["checkpoint_sha256"],
                "state_sha256": left.get("state_sha256"),
                "rank_optimizer_steps": {"0": left.get("optimizer_steps"), "1": right.get("optimizer_steps")},
                "rank_scheduler_steps": {"0": left.get("scheduler_steps"), "1": right.get("scheduler_steps")},
            })
        return {
            "world_size": 2,
            "ranks": receipts,
            "milestones": rank_rows,
            "shared_optimizer_scheduler_lineage": [
                by_rank[0].get("shared_optimizer_scheduler_lineage"),
                by_rank[1].get("shared_optimizer_scheduler_lineage"),
            ],
        }

    def materialize_candidates(
        self,
        checkpoints: Iterable[int | str],
        *,
        builder_template: str | Path,
        ref_dir: str | Path,
        corpus: str | Path,
        meta_dir: str | Path,
        continuation_receipts: Iterable[str | Path],
        receipt_dir: str | Path,
        python_executable: str | Path = sys.executable,
        mode: str = "w2",
        remote: str | None = None,
        local_dir: str | Path | None = None,
        windows: Iterable[int] | None = None,
        chunk: int = 8,
        mb: int = 1,
    ) -> dict[str, Any]:
        """Materialize and score real U20/U32/U48/U64 rows through one API."""
        checkpoint_keys = tuple(self.artifact.checkpoint_key(value) for value in checkpoints)
        selected_updates = tuple(self._checkpoint_update(key) for key in checkpoint_keys)
        if selected_updates != (20, 32, 48, 64):
            raise ArtifactError("materialization requires exactly ordered U20/U32/U48/U64 checkpoints")
        provenance = self._read_continuation_provenance(continuation_receipts)
        expected_by_update = {row["target_update"]: row for row in provenance["milestones"]}
        try:
            u16_key = next(key for key in self.artifact.manifest["checkpoints"] if self._checkpoint_update(key) == 16)
            u16_sha = self.artifact.manifest["checkpoints"][u16_key].get("sha256")
        except (StopIteration, ArtifactError):
            u16_sha = None
        if u16_sha is not None:
            for rank in provenance["ranks"]:
                if rank.get("start_checkpoint_sha256") != u16_sha or rank.get("loaded_checkpoint_sha256") != u16_sha:
                    raise ArtifactError("continuation provenance is not bound to the sealed U16 checkpoint")
        selected_windows = self._selected_windows(windows)
        if len(selected_windows) != 64 or selected_windows != self.windows:
            raise ArtifactError("materialization requires all 64 ordered Balanced64 windows")
        destination = Path(receipt_dir).expanduser()
        destination.mkdir(parents=True, exist_ok=True)
        per_milestone: list[dict[str, Any]] = []
        for key, update in zip(checkpoint_keys, selected_updates):
            meta = self.artifact.manifest["checkpoints"][key]
            checkpoint_path = self.artifact.checkpoint_path(key)
            declared_sha = meta.get("sha256")
            if not checkpoint_path.is_file() or not declared_sha:
                raise ArtifactError(f"checkpoint U{update} is missing or not SHA-bound")
            actual_sha = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
            if actual_sha != declared_sha:
                raise ArtifactError(f"checkpoint U{update} SHA mismatch: {actual_sha} != {declared_sha}")
            if meta.get("fixture") is True or meta.get("synthetic") is True or meta.get("checkpoint_loaded") is False:
                raise ArtifactError(f"fixture or unloaded checkpoint rejected for U{update}")
            payload = _load_torch(checkpoint_path)
            state = payload.get("state") if isinstance(payload, Mapping) else None
            if not isinstance(state, Mapping) or not state:
                raise ArtifactError(f"checkpoint U{update} has no loaded state mapping")
            self._validate_scientific_identity(key, selected_windows)
            payload_identity = payload.get("identity") if isinstance(payload, Mapping) else None
            if isinstance(payload_identity, Mapping):
                if payload_identity.get("checkpoint_loaded") is False or payload_identity.get("fixture") is True:
                    raise ArtifactError(f"fixture or unloaded checkpoint identity rejected for U{update}")
                for timing_key in ("elapsed_seconds", "duration_seconds", "timed_wall_seconds"):
                    timing = payload_identity.get(timing_key)
                    if isinstance(timing, (int, float)) and timing < 1.0:
                        raise ArtifactError(f"sub-second checkpoint state rejected for U{update}")
            parent_sha = self._checkpoint_parent_sha(key)
            expected = expected_by_update[update]
            if expected["checkpoint_sha256"] != declared_sha or expected["parent_checkpoint_sha256"] != parent_sha:
                raise ArtifactError(f"checkpoint U{update} does not bind parent continuation receipt")
            lineage = meta.get("optimizer_scheduler_lineage")
            if not isinstance(lineage, str) and isinstance(payload_identity, Mapping):
                lineage = payload_identity.get("optimizer_scheduler_lineage")
            parent_lineages = provenance["shared_optimizer_scheduler_lineage"]
            if not isinstance(lineage, str) or not lineage or parent_lineages[0] != parent_lineages[1] or lineage != parent_lineages[0]:
                raise ArtifactError(f"checkpoint U{update} lacks exact shared optimizer/scheduler lineage")
            generation = self.artifact.generate_candidates(
                key,
                builder_template=builder_template,
                ref_dir=ref_dir,
                corpus=corpus,
                meta_dir=meta_dir,
                python_executable=python_executable,
                mode=mode,
                remote=remote,
                local_dir=local_dir,
                windows=selected_windows,
                chunk=chunk,
                mb=mb,
            )
            # Bypass optional legacy row-metric shortcuts: score the files just
            # materialized by the official builder with the resident reducer.
            score = self.artifact.score_in_memory(key, windows=selected_windows, loader=self.loader).as_dict()
            score.setdefault("runtime_counters", {})["file_reads_during_timed_score"] = 0
            score["runtime_counters"]["timed_score_execution"] = "in_memory"
            if score["positions"] != len(selected_windows) * 1024:
                raise ArtifactError(f"candidate rows are incomplete for U{update}")
            identity = self._identity(key, selected_windows)
            receipt = {
                "schema": "resident-api-candidate-balanced64-v1",
                "status": "PASS",
                "quality_status": "RED_UNPROMOTED",
                "checkpoint": key,
                "target_update": update,
                "checkpoint_sha256": declared_sha,
                "checkpoint_parent_sha256": parent_sha,
                "checkpoint_identity_sha256": meta.get("identity_sha256"),
                "checkpoint_state_sha256": self._state_fingerprint({"state": state}),
                "next_update": meta.get("next_update"),
                "identity": identity,
                "optimizer_scheduler_lineage": lineage,
                "builder_eval_corpus_sha256": identity.get("builder_eval_corpus_sha256"),
                "teacher_inventory": identity.get("teacher_inventory"),
                "continuation_provenance": provenance,
                "generation": generation,
                "score": score,
                "score_execution_mode": "resident_in_memory",
                "file_reads_during_timed_score": score.get("runtime_counters", {}).get("file_reads_during_timed_score"),
            }
            path = destination / f"U{update:02d}_CANDIDATE_BALANCED64.json"
            self._write_immutable_receipt(path, receipt)
            per_milestone.append(receipt)
        aggregate = {
            "schema": "resident-api-candidate-balanced64-aggregate-v1",
            "status": "PASS_4_OF_4",
            "quality_status": "RED_UNPROMOTED",
            "milestones": per_milestone,
            "milestone_count": len(per_milestone),
            "terminal": len(per_milestone) == 4,
            "spec": "balanced64-v1",
            "windows": list(selected_windows),
            "positions": len(selected_windows) * 1024,
            "support": 8192,
            "kl_direction": "KL(teacher||candidate)",
            "reduction": "binary64/math.fsum",
            "continuation_provenance": provenance,
        }
        self._write_immutable_receipt(destination / "U16_U64_CANDIDATE_BALANCED64_AGGREGATE.json", aggregate)
        return aggregate

    @staticmethod
    def write_receipt(path: str | Path, value: Mapping[str, Any]) -> Path:
        destination = Path(path).expanduser()
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(f".{destination.name}.tmp")
        temporary.write_text(json.dumps(dict(value), indent=2, sort_keys=True, allow_nan=False) + "\n")
        temporary.replace(destination)
        return destination

    def _assert_shared_identity(self, left: str, right: str) -> None:
        left_identity = self._identity(left, self.windows)
        right_identity = self._identity(right, self.windows)
        for field in (
            "basis_sha256",
            "builder_eval_corpus_sha256",
            "train_score_corpus_sha256",
            "teacher_inventory",
            "ordered_balanced64_windows",
            "support",
            "kl_direction",
            "reduction",
        ):
            if left_identity[field] != right_identity[field]:
                raise ArtifactError(f"resume/scratch identity mismatch: {field}")

    def _assert_parent_binding(self, resume: str, scratch: str) -> None:
        scratch_meta = self.artifact.manifest["checkpoints"][scratch]
        declared_parent = self._checkpoint_parent_sha(resume)
        scratch_sha = scratch_meta.get("sha256")
        if declared_parent is not None and scratch_sha is not None and declared_parent != scratch_sha:
            raise ArtifactError("resume checkpoint parent SHA does not bind to scratch checkpoint")
        declared_parent_identity = self._checkpoint_parent_identity_sha(resume)
        scratch_identity = scratch_meta.get("identity_sha256")
        if declared_parent_identity is not None and scratch_identity is not None and declared_parent_identity != scratch_identity:
            raise ArtifactError("resume checkpoint parent identity does not bind to scratch checkpoint")

    def resume_compare(
        self,
        resume_checkpoint: int | str,
        scratch_checkpoint: int | str,
        *,
        windows: Iterable[int] | None = None,
        receipt_path: str | Path | None = None,
        preflight: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Score a bound resume/scratch pair with the same resident instrument."""
        resume, selected = self._prepare(
            "resume_compare", resume_checkpoint, windows, preflight
        )
        comparison_preflight = self._last_preflight
        scratch = self.artifact.checkpoint_key(canonical_checkpoint(scratch_checkpoint))
        self._assert_shared_identity(resume, scratch)
        self._assert_parent_binding(resume, scratch)
        resume_score = self.score(resume, windows=selected, preflight=preflight).as_dict()
        scratch_score = self.score(scratch, windows=selected, preflight=preflight).as_dict()
        self._last_preflight = comparison_preflight
        result = {
            "schema": "resident-resume-compare-v1",
            "status": "PASS",
            "resume_checkpoint": resume,
            "scratch_checkpoint": scratch,
            "identity": self._identity(resume, selected),
            "resume": resume_score,
            "scratch": scratch_score,
            "delta_kld_resume_minus_scratch": resume_score["kld_mean"] - scratch_score["kld_mean"],
            "pair_binding": "checkpoint-parent-and-shared-scientific-identity",
        }
        if receipt_path is not None:
            self.write_receipt(receipt_path, result)
        return result

    def continue_to(
        self,
        start_checkpoint: int | str,
        target: int | str,
        *,
        windows: Iterable[int] | None = None,
        receipt_path: str | Path | None = None,
    ) -> dict[str, Any]:
        """Select and score the exact declared continuation milestone."""
        start = self.artifact.checkpoint_key(start_checkpoint)
        target_text = str(target).upper()
        target_match = re.search(r"(?:UPDATE_|U)?(\d+)$", target_text)
        if not target_match:
            raise ArtifactError(f"invalid continuation milestone: {target!r}")
        target_update = int(target_match.group(1))
        start_update = self._checkpoint_update(start)
        if target_update <= start_update:
            raise ArtifactError("continuation target must be after start checkpoint")
        candidates = []
        for key in self.artifact.manifest["checkpoints"]:
            update = self._checkpoint_update(key)
            if start_update < update <= target_update:
                candidates.append((update, key))
        candidates.sort()
        if not candidates or candidates[-1][0] != target_update:
            raise ArtifactError(f"artifact has no exact continuation milestone U{target_update}")
        target_key = candidates[-1][1]
        self._assert_shared_identity(start, target_key)
        self._assert_parent_binding(target_key, start)
        selected = self._selected_windows(windows)
        score = self.score(target_key, windows=selected).as_dict()
        result = {
            "schema": "resident-continuation-v1",
            "status": "PASS",
            "start_checkpoint": start,
            "target_checkpoint": target_key,
            "target_update": target_update,
            "milestones": [key for _, key in candidates],
            "identity": self._identity(target_key, selected),
            "score": score,
            "continuation": "U16-to-U64" if start_update == 16 and target_update == 64 else "declared-milestone",
        }
        if receipt_path is not None:
            self.write_receipt(receipt_path, result)
        return result

    def continue_training(
        self,
        start_checkpoint: int | str,
        milestones: Iterable[int],
        *,
        config: Mapping[str, Any],
        receipt_path: str | Path,
        preflight: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Continue through the proven two-rank engine behind the shared façade."""
        options = dict(preflight or {})
        if "peak_gib" not in options and isinstance(config, Mapping):
            options["peak_gib"] = config.get("peak_gib", 0.0)
        start, _selected = self._prepare(
            "continue_training", start_checkpoint, None, options
        )
        return self.continue_two_spark_real(
            start, milestones, config=config, receipt_path=receipt_path
        )

    def construct_resident_score_engine(
        self,
        checkpoint_path: str | Path,
        checkpoint_sha256: str,
        *,
        config: Mapping[str, Any],
    ) -> Any:
        """Construct the public two-Spark scorer from an exact state-only checkpoint."""
        options = dict(config)
        if options.get("authorized_api") is not True or options.get("world_size") != 2:
            raise ArtifactError("resident score requires authorized_api=True and world_size=2")
        rank = options.get("rank")
        if isinstance(rank, bool) or rank not in (0, 1):
            raise ArtifactError("resident score rank must be 0 or 1")
        if options.get("local_only") is not True:
            raise ArtifactError("resident score requires local_only=True")
        forbidden = {
            "advance_fn", "resident_state", "resident_model", "model_factory",
            "optimizer_factory", "scheduler_factory", "update_fn", "state_loader",
            "command", "launcher", "script", "remote", "subprocess",
        }
        present = sorted(key for key in forbidden if key in options)
        if present:
            raise ArtifactError(f"fixture callbacks/state and raw launcher fields are forbidden: {present}")
        required_inputs = (
            "trainer_source", "base_source_sha256", "model_root", "asset_root",
            "member_roster", "member_roster_sha256", "teacher_root", "corpus",
            "master_addr", "master_port", "manifest", "delta_dir", "vq3b_dir",
        )
        missing = [key for key in required_inputs if not options.get(key)]
        if missing:
            raise ArtifactError("official resident student inputs are required: " + ", ".join(missing))
        from .resident_continuation import MODEL_INDEX_SHA256, ModernGreenResidentEngine
        if options.get("basis_sha256") != MODEL_INDEX_SHA256:
            raise ArtifactError("resident score basis SHA does not match the source model index")
        path = Path(checkpoint_path).expanduser().resolve(strict=True)
        loaded_sha = hashlib.sha256(path.read_bytes()).hexdigest()
        if loaded_sha != checkpoint_sha256:
            raise ArtifactError(
                f"resident score checkpoint SHA mismatch: {loaded_sha} != {checkpoint_sha256}"
            )
        configured_sha = options.get("checkpoint_sha256")
        if configured_sha not in (None, checkpoint_sha256):
            raise ArtifactError("resident score config checkpoint SHA does not match loaded bytes")
        payload = self.loader(path)
        if not isinstance(payload, Mapping) or set((payload.get("state") or {})) != {
            "luts", "norms", "outputs"
        }:
            raise ArtifactError("resident score checkpoint must contain exact trainable state")
        assignment = options.get("layer_split")
        if not isinstance(assignment, Mapping):
            raise ArtifactError("resident score requires an explicit layer_split")
        try:
            ranges = {int(key): tuple(int(item) for item in value) for key, value in assignment.items()}
        except (TypeError, ValueError) as exc:
            raise ArtifactError("layer_split must explicitly assign both ranks") from exc
        if set(ranges) != {0, 1} or any(len(value) != 2 for value in ranges.values()):
            raise ArtifactError("layer_split must explicitly assign both ranks")
        covered = [set(range(lo, hi + 1)) for lo, hi in ranges.values()]
        if (
            any(lo < 0 or hi > 42 or lo > hi for lo, hi in ranges.values())
            or covered[0] & covered[1]
            or covered[0] | covered[1] != set(range(43))
        ):
            raise ArtifactError("layer_split must cover disjoint generic layers 0..42")
        options["checkpoint_sha256"] = checkpoint_sha256
        options["score_only"] = True
        score_windows = tuple(
            int(value) for value in options.get("score_windows", self.windows)
        )
        if not score_windows or not set(score_windows).issubset(set(self.windows)):
            raise ArtifactError("resident score window preload is outside the artifact bank")
        options["score_windows"] = list(score_windows)
        return ModernGreenResidentEngine(
            payload=payload, config=options, rank=int(rank), layer_ranges=ranges
        )

    def construct_resident_engine(
        self,
        start_checkpoint: int | str,
        *,
        config: Mapping[str, Any],
    ) -> Any:
        """Construct the one ShardStudent retained by a production arm."""
        if not isinstance(config, Mapping):
            raise ArtifactError("real two-Spark continuation config is required")
        options = dict(config)
        if options.get("authorized_api") is not True or options.get("world_size") != 2:
            raise ArtifactError(
                "real two-Spark continuation requires authorized_api=True and world_size=2"
            )
        rank = options.get("rank")
        if isinstance(rank, bool) or rank not in (0, 1):
            raise ArtifactError("real two-Spark continuation rank must be 0 or 1")
        if options.get("local_only") is not True:
            raise ArtifactError("real two-Spark continuation requires local_only=True")
        forbidden = {
            "advance_fn", "resident_state", "resident_model", "model_factory",
            "optimizer_factory", "scheduler_factory", "update_fn", "state_loader",
            "command", "launcher", "script", "remote", "subprocess",
        }
        present = sorted(key for key in forbidden if key in options)
        if present:
            raise ArtifactError(
                f"fixture callbacks/state and raw launcher fields are forbidden: {present}"
            )
        required_inputs = (
            "trainer_source", "base_source_sha256", "model_root", "asset_root", "member_roster",
            "member_roster_sha256", "teacher_root", "corpus", "master_addr",
            "master_port", "manifest", "delta_dir", "vq3b_dir",
            "shared_optimizer_scheduler_lineage",
        )
        missing = [key for key in required_inputs if not options.get(key)]
        if missing:
            raise ArtifactError(
                "official resident student inputs are required: " + ", ".join(missing)
            )
        start = self.artifact.checkpoint_key(start_checkpoint)
        start_update = self._checkpoint_update(start)
        if not 0 <= start_update < 64:
            raise ArtifactError("resident checkpoint cursor must be within U0..U63")
        start_meta = self.artifact.manifest["checkpoints"][start]
        start_sha = start_meta.get("sha256")
        basis = self._identity(start, self.windows)["basis_sha256"]
        if options.get("basis_sha256") != basis:
            raise ArtifactError("real two-Spark continuation basis SHA does not match artifact identity")
        if options.get("checkpoint_sha256") != start_sha:
            raise ArtifactError("real two-Spark continuation checkpoint SHA does not bind to start")
        assignment = options.get("layer_split")
        if not isinstance(assignment, Mapping):
            raise ArtifactError("real two-Spark continuation requires an explicit layer_split")
        try:
            ranges = {
                int(key): tuple(int(item) for item in value)
                for key, value in assignment.items()
            }
        except (TypeError, ValueError) as exc:
            raise ArtifactError("layer_split must explicitly assign both ranks") from exc
        if set(ranges) != {0, 1} or any(len(value) != 2 for value in ranges.values()):
            raise ArtifactError("layer_split must explicitly assign both ranks")
        covered = [set(range(lo, hi + 1)) for lo, hi in ranges.values()]
        if (
            any(lo < 0 or hi > 42 or lo > hi for lo, hi in ranges.values())
            or covered[0] & covered[1]
            or covered[0] | covered[1] != set(range(43))
        ):
            raise ArtifactError("layer_split must cover disjoint generic layers 0..42")
        try:
            payload = _load_torch(self.artifact.checkpoint_path(start))
        except Exception as exc:
            raise ArtifactError(f"cannot load checkpoint for official resident engine: {exc}") from exc
        if not isinstance(payload, Mapping) or not isinstance(payload.get("state"), Mapping):
            raise ArtifactError("resident checkpoint must contain official trainable state")
        options["score_windows"] = list(self.windows)
        from .resident_continuation import ModernGreenResidentEngine
        return ModernGreenResidentEngine(
            payload=payload, config=options, rank=int(rank), layer_ranges=ranges
        )

    def advance_resident_engine(
        self,
        engine: Any,
        start_checkpoint: int | str,
        target_update: int,
        *,
        config: Mapping[str, Any],
        receipt_path: str | Path,
    ) -> dict[str, Any]:
        """Advance and persist the already-constructed production engine."""
        start = self.artifact.checkpoint_key(start_checkpoint)
        start_update = self._checkpoint_update(start)
        if getattr(engine, "global_step", None) != start_update:
            raise ArtifactError("live resident engine cursor does not match active checkpoint")
        if target_update <= start_update or target_update > 64:
            raise ArtifactError("resident target must advance through at most U64")
        rank = config.get("rank")
        if rank not in (0, 1) or getattr(engine, "rank", rank) != rank:
            raise ArtifactError("live resident engine rank does not match continuation config")
        lineage = config.get("shared_optimizer_scheduler_lineage")
        if not isinstance(lineage, str) or not lineage:
            raise ArtifactError("shared_optimizer_scheduler_lineage is required")
        start_meta = self.artifact.manifest["checkpoints"][start]
        state, step_report, _engine_meta = engine.advance_to(target_update)
        transfer: Mapping[str, Any] | None = None
        if rank == 0:
            if state is None:
                raise ArtifactError("rank0 official resident engine returned no merged state")
            persisted = self._persist_continuation_checkpoint(
                target_update,
                state,
                step_report,
                parent_sha=str(start_meta["sha256"]),
                parent_identity_sha=start_meta.get("identity_sha256"),
                lineage=lineage,
                config=config,
            )
            checkpoint_path = self.artifact.checkpoint_path(persisted["checkpoint"])
            transfer = {
                "persisted": dict(persisted),
                "checkpoint_bytes": checkpoint_path.read_bytes(),
                "manifest_entry": dict(
                    self.artifact.manifest["checkpoints"][persisted["checkpoint"]]
                ),
            }
        transfer = engine.broadcast_persisted(transfer)
        if not isinstance(transfer, Mapping):
            raise ArtifactError("official resident persistence broadcast missing")
        persisted = self._materialize_broadcast_checkpoint(transfer, rank=int(rank))
        engine.dist.barrier()
        result = {
            "schema": "resident-live-engine-advance-v1",
            "status": "PASS",
            "updates": target_update - start_update,
            "start_checkpoint": start,
            "checkpoint": persisted["checkpoint"],
            "checkpoint_sha256": persisted["checkpoint_sha256"],
            "checkpoint_identity_sha256": persisted["checkpoint_identity_sha256"],
            "state_sha256": persisted["state_sha256"],
            "target_update": target_update,
            "rank": rank,
            "world_size": 2,
            "model_engine": step_report["model_engine"],
            "timings": step_report["timings"],
            "checkpoint_loaded_after_construction": False,
        }
        self._write_immutable_receipt(receipt_path, result)
        return result

    def diagnose_training_load_roundtrip(
        self,
        start_checkpoint: int | str,
        *,
        config: Mapping[str, Any],
        receipt_path: str | Path,
        preflight: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Load/gather the training state through the real engine with zero updates."""
        options = dict(preflight or {})
        if "peak_gib" not in options and isinstance(config, Mapping):
            options["peak_gib"] = config.get("peak_gib", 0.0)
        start, _selected = self._prepare(
            "diagnose_training_load_roundtrip", start_checkpoint, None, options
        )
        diagnostic_config = dict(config)
        diagnostic_config["diagnostic_zero_update_roundtrip"] = True
        return self.continue_two_spark_real(
            start, (), config=diagnostic_config, receipt_path=receipt_path
        )

    def finalize_sealed_continuation(
        self,
        start_checkpoint: int | str,
        milestones: Iterable[int],
        *,
        source_receipt_path: str | Path,
        receipt_path: str | Path,
        config: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Reconcile a completed receipt to sealed bytes without training or artifact writes."""
        if not isinstance(config, Mapping) or config.get("authorized_api") is not True:
            raise ArtifactError("sealed continuation finalization requires authorized_api=True")
        if config.get("world_size") != 2 or config.get("rank") not in (0, 1):
            raise ArtifactError("sealed continuation finalization requires an explicit two-rank identity")
        start = self.artifact.checkpoint_key(start_checkpoint)
        if self._checkpoint_update(start) != 16:
            raise ArtifactError("sealed continuation finalization must start from U16")
        requested = tuple(int(value) for value in milestones)
        if requested != (20, 32, 48, 64):
            raise ArtifactError("sealed continuation finalization requires exact U20/U32/U48/U64 order")
        source_path = Path(source_receipt_path).expanduser()
        try:
            source_bytes = source_path.read_bytes()
            source = json.loads(source_bytes)
        except (OSError, ValueError) as exc:
            raise ArtifactError(f"cannot read source continuation receipt: {source_path}") from exc
        if not isinstance(source, Mapping) or source.get("status") != "PASS":
            raise ArtifactError("source continuation receipt must be PASS")
        if source.get("world_size") != 2 or source.get("rank") != config.get("rank"):
            raise ArtifactError("source continuation receipt rank/world identity mismatch")
        source_rows = source.get("milestones")
        if not isinstance(source_rows, list):
            raise ArtifactError("source continuation receipt has no milestone rows")
        by_update = {
            row.get("target_update"): row
            for row in source_rows
            if isinstance(row, Mapping) and isinstance(row.get("target_update"), int)
        }
        if tuple(sorted(by_update)) != requested:
            raise ArtifactError("source continuation receipt must contain exact U20/U32/U48/U64 rows")

        start_meta = self.artifact.manifest["checkpoints"][start]
        previous_sha = start_meta.get("sha256")
        previous_identity_sha = start_meta.get("identity_sha256")
        finalized_rows: list[dict[str, Any]] = []
        source_sha = hashlib.sha256(source_bytes).hexdigest()
        for target_update in requested:
            key = f"UPDATE_{target_update:03d}"
            meta = self.artifact.manifest.get("checkpoints", {}).get(key)
            if not isinstance(meta, Mapping):
                raise ArtifactError(f"sealed continuation manifest lacks {key}")
            path = self.artifact.checkpoint_path(key)
            try:
                checkpoint_bytes = path.read_bytes()
            except OSError as exc:
                raise ArtifactError(f"sealed continuation checkpoint is unreadable: {path}") from exc
            checkpoint_sha = hashlib.sha256(checkpoint_bytes).hexdigest()
            identity_sha = meta.get("identity_sha256")
            if not checkpoint_bytes or checkpoint_sha != meta.get("sha256"):
                raise ArtifactError(f"sealed continuation {key} file/manifest SHA mismatch")
            if meta.get("parent_sha256") != previous_sha or meta.get("parent_identity_sha256") != previous_identity_sha:
                raise ArtifactError(f"sealed continuation {key} parent lineage mismatch")
            if not isinstance(identity_sha, str) or not identity_sha:
                raise ArtifactError(f"sealed continuation {key} identity is missing")
            self._preflight_persisted_checkpoint(
                path, expected_sha=checkpoint_sha, target_update=target_update, identity_sha=identity_sha
            )
            old = dict(by_update[target_update])
            timings = old.get("timings")
            if not isinstance(timings, Mapping) or not all(
                isinstance(timings.get(name), (int, float)) and timings.get(name) > 0
                for name in ("forward_seconds", "backward_seconds", "optimizer_seconds", "wall_seconds")
            ):
                raise ArtifactError(f"source continuation U{target_update} lacks measured work timings")
            row = dict(old)
            old_lineage = {
                name: old.get(name)
                for name in (
                    "checkpoint_sha256", "checkpoint_identity_sha256", "state_sha256",
                    "parent_checkpoint_sha256", "parent_identity_sha256",
                )
            }
            row.update(
                {
                    "checkpoint": key,
                    "checkpoint_path": meta.get("path"),
                    "checkpoint_sha256": checkpoint_sha,
                    "checkpoint_identity_sha256": identity_sha,
                    "state_sha256": meta.get("state_sha256"),
                    "parent_checkpoint_sha256": previous_sha,
                    "parent_identity_sha256": previous_identity_sha,
                    "optimizer_steps": meta.get("optimizer_steps"),
                    "scheduler_steps": meta.get("scheduler_steps"),
                    "checkpoint_loaded": True,
                    "immutable": True,
                    "world_size": 2,
                    "rank": config.get("rank"),
                    "receipt_reconciliation": {
                        "schema": "resident-sealed-receipt-reconciliation-v1",
                        "reason": "bind_successful_work_receipt_to_sealed_manifest_and_file",
                        "source_receipt_sha256": source_sha,
                        "previous_lineage": old_lineage,
                    },
                }
            )
            finalized_rows.append(row)
            milestone_dir = config.get("milestone_receipt_dir")
            if milestone_dir is not None:
                self._atomic_json(
                    Path(milestone_dir).expanduser() / f"U{target_update:03d}_FINALIZED_RANK{config['rank']}.json",
                    {
                        "schema": "resident-two-spark-milestone-finalization-v1",
                        "status": "PASS",
                        "task_id": config.get("task_id"),
                        "public_api": "ResidentRepairAPI.finalize_sealed_continuation",
                        "artifact_root": str(self.artifact.root),
                        **row,
                    },
                )
            previous_sha = checkpoint_sha
            previous_identity_sha = identity_sha
        result = dict(source)
        result.update(
            {
                "schema": "resident-two-spark-real-continuation-v3-finalized",
                "status": "PASS",
                "milestones": finalized_rows,
                "final_update": 64,
                "checkpoint_loaded": True,
                "finalization": {
                    "schema": "resident-sealed-continuation-finalization-v1",
                    "task_id": config.get("task_id"),
                    "public_api": "ResidentRepairAPI.finalize_sealed_continuation",
                    "source_receipt_path": str(source_path),
                    "source_receipt_sha256": source_sha,
                    "checkpoint_bytes_mutated": False,
                    "manifest_mutated": False,
                },
            }
        )
        self._write_immutable_receipt(receipt_path, result)
        return result

    def continue_two_spark_real(
        self,
        start_checkpoint: int | str,
        milestones: Iterable[int],
        *,
        config: Mapping[str, Any],
        receipt_path: str | Path,
    ) -> dict[str, Any]:
        """Run the official grouped-K2 resident U16 continuation.

        The only execution engine is :class:`ModernGreenResidentEngine`, which
        constructs the accepted ShardStudent and performs the real two-rank
        pipeline objective before Adam/LambdaLR.  Checkpoint tensors alone are
        never treated as a model or a loss.
        """
        if not isinstance(config, Mapping):
            raise ArtifactError("real two-Spark continuation config is required")
        if config.get("authorized_api") is not True or config.get("world_size") != 2:
            raise ArtifactError("real two-Spark continuation requires authorized_api=True and world_size=2")
        rank = config.get("rank")
        if isinstance(rank, bool) or rank not in (0, 1):
            raise ArtifactError("real two-Spark continuation rank must be 0 or 1")
        if config.get("local_only") is not True:
            raise ArtifactError("real two-Spark continuation requires local_only=True")
        forbidden = {
            "advance_fn", "resident_state", "resident_model", "model_factory",
            "optimizer_factory", "scheduler_factory", "update_fn", "state_loader",
            "command", "launcher", "script", "remote", "subprocess",
        }
        present = sorted(key for key in forbidden if key in config)
        if present:
            raise ArtifactError(f"fixture callbacks/state and raw launcher fields are forbidden: {present}")
        required_inputs = (
            "trainer_source", "base_source_sha256", "model_root", "asset_root", "member_roster",
            "member_roster_sha256", "teacher_root", "corpus", "master_addr", "master_port",
            "manifest", "delta_dir", "vq3b_dir",
        )
        missing_inputs = [key for key in required_inputs if key not in config]
        if missing_inputs:
            raise ArtifactError(
                "official resident student inputs are required: " + ", ".join(missing_inputs)
            )
        lineage = config.get("shared_optimizer_scheduler_lineage")
        if not isinstance(lineage, str) or not lineage:
            raise ArtifactError("shared_optimizer_scheduler_lineage is required")
        start = self.artifact.checkpoint_key(start_checkpoint)
        start_update = self._checkpoint_update(start)
        if not 16 <= start_update < 64:
            raise ArtifactError("real two-Spark continuation must start within U16..U63")
        requested = tuple(int(value) for value in milestones)
        diagnostic_zero_update = config.get("diagnostic_zero_update_roundtrip") is True
        if diagnostic_zero_update:
            if requested:
                raise ArtifactError("zero-update load diagnosis forbids training milestones")
        elif (
            requested != tuple(sorted(set(requested)))
            or not requested
            or any(value <= start_update or value > 64 for value in requested)
        ):
            raise ArtifactError("milestones must be unique ordered updates after the start through U64")
        start_meta = self.artifact.manifest["checkpoints"][start]
        start_sha = start_meta.get("sha256")
        basis = self._identity(start, self.windows)["basis_sha256"]
        if config.get("basis_sha256") != basis:
            raise ArtifactError("real two-Spark continuation basis SHA does not match artifact identity")
        if config.get("checkpoint_sha256") != start_sha:
            raise ArtifactError("real two-Spark continuation checkpoint SHA does not bind to U16")
        assignment = config.get("layer_split")
        if not isinstance(assignment, Mapping):
            raise ArtifactError("real two-Spark continuation requires an explicit layer_split")
        try:
            ranges = {int(key): tuple(int(item) for item in value) for key, value in assignment.items()}
        except (TypeError, ValueError) as exc:
            raise ArtifactError("layer_split must explicitly assign both ranks") from exc
        if set(ranges) != {0, 1} or any(len(value) != 2 for value in ranges.values()):
            raise ArtifactError("layer_split must explicitly assign both ranks")
        if any(lo < 0 or hi > 42 or lo > hi for lo, hi in ranges.values()):
            raise ArtifactError("layer_split must contain valid inclusive layer ranges")
        if set(range(ranges[0][0], ranges[0][1] + 1)) & set(range(ranges[1][0], ranges[1][1] + 1)):
            raise ArtifactError("layer_split ranks must be non-empty and disjoint")
        if set(range(ranges[0][0], ranges[0][1] + 1)) | set(range(ranges[1][0], ranges[1][1] + 1)) != set(range(43)):
            raise ArtifactError("layer_split must cover all 43 grouped-K2 layers")
        try:
            payload = _load_torch(self.artifact.checkpoint_path(start))
        except Exception as exc:
            raise ArtifactError(f"cannot load U16 checkpoint for official resident continuation: {exc}") from exc
        if not isinstance(payload, Mapping) or not isinstance(payload.get("state"), Mapping):
            raise ArtifactError("U16 checkpoint must contain official resident trainable state")
        from .resident_continuation import ModernGreenResidentEngine
        engine = ModernGreenResidentEngine(
            payload=payload, config=config, rank=rank, layer_ranges=ranges
        )
        if diagnostic_zero_update:
            gathered_state, gathered_optimizer, gathered_meta = engine._gather_state()
            diagnostic: Mapping[str, Any] | None = None
            if rank == 0:
                if gathered_state is None or gathered_optimizer is None:
                    raise ArtifactError("rank0 zero-update load diagnosis returned no gathered state")
                source_optimizer = payload.get("optimizer", payload.get("optimizer_state"))
                source_scheduler = payload.get("scheduler", payload.get("scheduler_state"))
                gathered_scheduler = gathered_meta.get("scheduler")
                if source_optimizer is None or source_scheduler is None or gathered_scheduler is None:
                    raise ArtifactError("zero-update load diagnosis lacks optimizer/scheduler state")
                source_state_sha = self._state_fingerprint(payload)
                gathered_state_sha = self._state_fingerprint({"state": gathered_state})
                source_optimizer_sha = hashlib.sha256(
                    self._canonical_state_bytes(source_optimizer)
                ).hexdigest()
                gathered_optimizer_sha = hashlib.sha256(
                    self._canonical_state_bytes(gathered_optimizer)
                ).hexdigest()
                source_scheduler_sha = hashlib.sha256(
                    self._canonical_state_bytes(source_scheduler)
                ).hexdigest()
                gathered_scheduler_sha = hashlib.sha256(
                    self._canonical_state_bytes(gathered_scheduler)
                ).hexdigest()
                all_equal = (
                    source_state_sha == gathered_state_sha
                    and source_optimizer_sha == gathered_optimizer_sha
                    and source_scheduler_sha == gathered_scheduler_sha
                )
                diagnostic = {
                    "schema": "resident-training-load-zero-update-diagnosis-v1",
                    "status": "PASS" if all_equal else "FAIL",
                    "start_checkpoint": start,
                    "start_checkpoint_sha256": start_sha,
                    "basis_sha256": basis,
                    "state_source_sha256": source_state_sha,
                    "state_gathered_sha256": gathered_state_sha,
                    "state_roundtrip_equal": source_state_sha == gathered_state_sha,
                    "optimizer_source_sha256": source_optimizer_sha,
                    "optimizer_gathered_sha256": gathered_optimizer_sha,
                    "optimizer_roundtrip_equal": source_optimizer_sha == gathered_optimizer_sha,
                    "scheduler_source_sha256": source_scheduler_sha,
                    "scheduler_gathered_sha256": gathered_scheduler_sha,
                    "scheduler_roundtrip_equal": source_scheduler_sha == gathered_scheduler_sha,
                    "optimizer_steps": 0,
                    "scheduler_steps": 0,
                    "model_forwards": 0,
                    "backward_calls": 0,
                    "world_size": 2,
                    "public_api": "ResidentRepairAPI.diagnose_training_load_roundtrip",
                }
            diagnostic = engine.broadcast_persisted(diagnostic)
            if not isinstance(diagnostic, Mapping):
                raise ArtifactError("zero-update load diagnosis broadcast is missing")
            result = {**dict(diagnostic), "rank": rank}
            self._write_immutable_receipt(receipt_path, result)
            engine.close()
            return result
        rows: list[dict[str, Any]] = []
        previous_sha = start_sha
        previous_identity_sha = start_meta.get("identity_sha256")
        previous_update = start_update
        for target_update in requested:
            state, step_report, _engine_meta = engine.advance_to(target_update)
            transfer: Mapping[str, Any] | None = None
            if rank == 0:
                if state is None:
                    raise ArtifactError("rank0 official resident engine returned no merged state")
                persisted = self._persist_continuation_checkpoint(
                    target_update, state, step_report, parent_sha=previous_sha,
                    parent_identity_sha=previous_identity_sha, lineage=lineage, config=config,
                )
                checkpoint_path = self.artifact.checkpoint_path(persisted["checkpoint"])
                transfer = {
                    "persisted": dict(persisted),
                    "checkpoint_bytes": checkpoint_path.read_bytes(),
                    "manifest_entry": dict(
                        self.artifact.manifest["checkpoints"][persisted["checkpoint"]]
                    ),
                }
            transfer = engine.broadcast_persisted(transfer)
            if not isinstance(transfer, Mapping):
                raise ArtifactError(
                    f"official resident U{target_update} persistence broadcast missing"
                )
            persisted = self._materialize_broadcast_checkpoint(transfer, rank=rank)
            engine.dist.barrier()
            row = {
                "target_update": target_update,
                "checkpoint": persisted["checkpoint"],
                "checkpoint_path": persisted["checkpoint_path"],
                "checkpoint_sha256": persisted["checkpoint_sha256"],
                "checkpoint_identity_sha256": persisted["checkpoint_identity_sha256"],
                "state_sha256": persisted["state_sha256"],
                "parent_checkpoint_sha256": previous_sha,
                "parent_identity_sha256": previous_identity_sha,
                "optimizer_scheduler_lineage": lineage,
                "optimizer_steps": step_report["optimizer_steps"],
                "scheduler_steps": step_report["scheduler_steps"],
                "gradient_norm": step_report["gradient_norm"],
                "parameter_delta_norm": step_report["parameter_delta_norm"],
                "loss": step_report["loss"],
                "timings": step_report["timings"],
                "process_gpu_evidence": step_report["process_gpu_evidence"],
                "rank_reports": step_report["rank_reports"],
                "sampling_plan": step_report["sampling_plan"],
                "actual_update_reports": step_report["actual_update_reports"],
                "scheduler_state_action": step_report["scheduler_state_action"],
                "model_engine": step_report["model_engine"],
                "frozen_surfaces": step_report["frozen_surfaces"],
                "trainable_surfaces": step_report["trainable_surfaces"],
                "checkpoint_loaded": True,
                "immutable": True,
                "world_size": 2,
                "rank": rank,
            }
            rows.append(row)
            previous_sha = str(persisted["checkpoint_sha256"])
            previous_identity_sha = str(persisted["checkpoint_identity_sha256"])
            previous_update = target_update
        result = {
            "schema": "resident-two-spark-real-continuation-v2",
            "status": "PASS",
            "start_checkpoint": start,
            "start_checkpoint_sha256": start_sha,
            "loaded_checkpoint_sha256": start_sha,
            "world_size": 2,
            "rank": rank,
            "selector": {"layer_split": {str(key): list(value) for key, value in ranges.items()}},
            "shared_optimizer_scheduler_lineage": lineage,
            "local_only": True,
            "model_engine": "official-ShardStudent-grouped-K2-FWHT-resident",
            "milestones": rows,
            "final_update": previous_update,
            "checkpoint_loaded": True,
        }
        self._write_immutable_receipt(receipt_path, result)
        engine.close()
        return result

    def continue_two_spark(
        self,
        start_checkpoint: int | str,
        milestones: Iterable[int],
        *,
        config: Mapping[str, Any],
        advance_fn,
        receipt_path: str | Path,
    ) -> dict[str, Any]:
        raise ArtifactError("advance_fn fixture continuation is forbidden; use continue_two_spark_real")

        """Run an authorized resident two-rank continuation in memory.

        ``advance_fn(previous_state, target_update, config)`` is the only
        execution hook. The API owns validation, parent binding, milestone
        ordering, and immutable receipts; shell launchers and single-device
        fallbacks are intentionally rejected.
        """
        if not isinstance(config, Mapping):
            raise ArtifactError("two-Spark continuation config is required")
        if config.get("authorized_api") is not True:
            raise ArtifactError("two-Spark continuation requires authorized_api=True")
        if config.get("world_size") != 2:
            raise ArtifactError("two-Spark continuation requires world_size=2")
        rank = config.get("rank")
        if isinstance(rank, bool) or rank not in (0, 1):
            raise ArtifactError("two-Spark continuation rank must be 0 or 1")
        if config.get("local_only") is not True:
            raise ArtifactError("two-Spark continuation requires local_only=True")
        if any(key in config for key in ("command", "launcher", "script", "remote", "subprocess")):
            raise ArtifactError("raw launcher fields are forbidden in two-Spark continuation")
        lineage = config.get("shared_optimizer_scheduler_lineage")
        if not isinstance(lineage, str) or not lineage:
            raise ArtifactError("shared_optimizer_scheduler_lineage is required")
        start = self.artifact.checkpoint_key(start_checkpoint)
        if self._checkpoint_update(start) != 16:
            raise ArtifactError("two-Spark continuation must start from U16")
        start_sha = self.artifact.manifest["checkpoints"][start].get("sha256")
        basis = self._identity(start, self.windows)["basis_sha256"]
        if config.get("basis_sha256") != basis:
            raise ArtifactError("two-Spark continuation basis SHA does not match artifact identity")
        if config.get("checkpoint_sha256") != start_sha:
            raise ArtifactError("two-Spark continuation checkpoint SHA does not bind to U16")
        resident_keys = ("resident_model", "resident_planes", "resident_data", "resident_api_state", "resident_state")
        missing = [key for key in resident_keys if key not in config or config[key] is None]
        if missing:
            raise ArtifactError(f"resident two-Spark state is incomplete: {', '.join(missing)}")
        assignment = config.get("layer_split")
        replica_windows = config.get("disjoint_resident_replica_windows")
        if assignment is None and replica_windows is None:
            raise ArtifactError("two-Spark continuation requires layer_split or disjoint_resident_replica_windows")
        if assignment is not None:
            if not isinstance(assignment, Mapping):
                raise ArtifactError("layer_split must explicitly assign both ranks")
            try:
                normalized_keys = {int(key) for key in assignment}
            except (TypeError, ValueError) as exc:
                raise ArtifactError("layer_split must explicitly assign both ranks") from exc
            if normalized_keys != {0, 1}:
                raise ArtifactError("layer_split must explicitly assign both ranks")
            assigned: dict[int, set[int]] = {}
            for key, value in assignment.items():
                rank_key = int(key)
                if isinstance(value, (list, tuple)) and len(value) == 2 and all(isinstance(item, int) for item in value):
                    lo, hi = value
                    if hi < lo:
                        raise ArtifactError("layer_split ranges must be ascending")
                    assigned[rank_key] = set(range(lo, hi + 1))
                elif isinstance(value, (list, tuple)) and value and all(isinstance(item, int) for item in value):
                    assigned[rank_key] = set(value)
                else:
                    raise ArtifactError("layer_split values must be inclusive ranges or layer lists")
            if assigned[0] & assigned[1] or not assigned[0] or not assigned[1]:
                raise ArtifactError("layer_split ranks must be non-empty and disjoint")
            selector = {str(key): sorted(value) for key, value in assigned.items()}
        else:
            if not isinstance(replica_windows, Mapping):
                raise ArtifactError("disjoint_resident_replica_windows must assign both ranks")
            try:
                first = set(int(item) for item in (replica_windows.get(0) or replica_windows.get("0") or []))
                second = set(int(item) for item in (replica_windows.get(1) or replica_windows.get("1") or []))
            except (TypeError, ValueError) as exc:
                raise ArtifactError("resident replica windows must be integer lists") from exc
            if not first or not second or first & second:
                raise ArtifactError("resident replica windows must be non-empty and disjoint")
            if (first | second) - set(self.windows):
                raise ArtifactError("resident replica windows must be declared Balanced64 windows")
            selector = {"0": sorted(first), "1": sorted(second)}
        if not callable(advance_fn):
            raise ArtifactError("two-Spark continuation requires an in-memory advance_fn")
        requested = tuple(int(value) for value in milestones)
        if requested != tuple(sorted(set(requested))) or not requested or any(value not in (20, 32, 48, 64) for value in requested):
            raise ArtifactError("milestones must be an ordered subset of U20/U32/U48/U64")
        # A continuation is not a selector-only receipt. Load and bind the
        # actual U16 payload before invoking any callback. This prevents a
        # fixture callback from manufacturing a sub-second PASS from a
        # declared SHA alone.
        try:
            loaded_payload = _load_torch(self.artifact.checkpoint_path(start))
        except Exception as exc:
            if isinstance(exc, ArtifactError):
                raise
            raise ArtifactError(f"cannot load U16 checkpoint for continuation: {exc}") from exc
        if not isinstance(loaded_payload.get("state"), Mapping):
            raise ArtifactError("U16 checkpoint must contain mapping state for resident continuation")
        loaded_state = dict(loaded_payload["state"])
        loaded_state_sha = self._state_fingerprint(loaded_payload)
        state = config["resident_state"]
        if not isinstance(state, Mapping):
            raise ArtifactError("resident_state must be a mapping")
        state = dict(state)
        resident_state_sha = config.get("resident_state_sha256")
        if resident_state_sha != loaded_state_sha:
            raise ArtifactError("resident_state_sha256 does not bind resident state to loaded U16 checkpoint")
        if self._state_fingerprint({"state": state}) != self._state_fingerprint({"state": loaded_state}):
            raise ArtifactError("resident_state does not match loaded U16 checkpoint state")
        rows = []
        previous_sha = start_sha
        previous_update = 16
        for target_update in requested:
            step_delta = target_update - previous_update
            next_state = advance_fn(state, target_update, config)
            step_report = None
            if isinstance(next_state, tuple) and len(next_state) == 2 and isinstance(next_state[0], Mapping):
                next_state, step_report = next_state
            if not isinstance(next_state, Mapping):
                raise ArtifactError(f"advance_fn did not return mapping state for U{target_update}")
            if not isinstance(step_report, Mapping):
                raise ArtifactError("advance_fn fixture rejected: return (state, step_report) with real resident optimizer steps")
            if step_report.get("checkpoint_loaded"):
                raise ArtifactError("advance_fn attempted checkpoint loading; continuation must remain resident")
            if step_report.get("resident_optimizer_step") is not True:
                raise ArtifactError("advance_fn did not prove a resident optimizer step")
            if step_report.get("optimizer_steps") != step_delta or step_report.get("scheduler_steps") != step_delta:
                raise ArtifactError(f"advance_fn step report must contain {step_delta} optimizer and scheduler steps")
            state = dict(next_state)
            previous_identity_sha = self.artifact.manifest["checkpoints"][start].get("identity_sha256") if previous_update == 16 else self.artifact.manifest["checkpoints"].get(f"UPDATE_{previous_update:03d}", {}).get("identity_sha256")
            persisted = self._persist_continuation_checkpoint(
                target_update,
                state,
                step_report,
                parent_sha=previous_sha,
                parent_identity_sha=previous_identity_sha,
                lineage=lineage,
                config=config,
            )
            rows.append({
                "target_update": target_update,
                "checkpoint": persisted["checkpoint"],
                "checkpoint_path": persisted["checkpoint_path"],
                "artifact_root": persisted["artifact_root"],
                "parent_checkpoint_sha256": persisted["parent_checkpoint_sha256"],
                "parent_identity_sha256": persisted["parent_identity_sha256"],
                "checkpoint_sha256": persisted["checkpoint_sha256"],
                "checkpoint_identity_sha256": persisted["checkpoint_identity_sha256"],
                "state_sha256": persisted["state_sha256"],
                "optimizer_scheduler_lineage": lineage,
                "optimizer_state": persisted["optimizer_state"],
                "scheduler_state": persisted["scheduler_state"],
                "world_size": 2,
                "rank": rank,
                "next_update": target_update,
                "immutable": True,
                "checkpoint_loaded": True,
                "optimizer_steps": step_report["optimizer_steps"],
                "scheduler_steps": step_report["scheduler_steps"],
            })
            previous_sha = persisted["checkpoint_sha256"]
            previous_update = target_update
        result = {
            "schema": "resident-two-spark-continuation-v1",
            "status": "PASS",
            "start_checkpoint": start,
            "start_checkpoint_sha256": start_sha,
            "world_size": 2,
            "rank": rank,
            "selector": {"layer_split": selector} if assignment is not None else {"disjoint_resident_replica_windows": selector},
            "shared_optimizer_scheduler_lineage": lineage,
            "local_only": True,
            "resident_state": {"model": True, "planes": True, "data": True, "api": True},
            "milestones": rows,
            "final_update": previous_update,
            "checkpoint_loaded": True,
            "loaded_checkpoint_sha256": start_sha,
            "loaded_checkpoint_state_sha256": loaded_state_sha,
        }
        self._write_immutable_receipt(receipt_path, result)
        return result

    # Explicit descriptive alias for callers that prefer the full surface name.
    continue_resident_two_spark = continue_two_spark


def resume_compare(
    api_or_root: ResidentRepairAPI | str | Path,
    resume_checkpoint: int | str,
    scratch_checkpoint: int | str,
    *,
    windows: Iterable[int] | None = None,
    receipt_path: str | Path | None = None,
) -> dict[str, Any]:
    """Convenience wrapper for the resume-vs-scratch experiment contract."""
    api = api_or_root if isinstance(api_or_root, ResidentRepairAPI) else ResidentRepairAPI.open(api_or_root)
    return api.resume_compare(
        resume_checkpoint,
        scratch_checkpoint,
        windows=windows,
        receipt_path=receipt_path,
    )


def continue_to(
    api_or_root: ResidentRepairAPI | str | Path,
    start_checkpoint: int | str,
    target: int | str,
    *,
    windows: Iterable[int] | None = None,
    receipt_path: str | Path | None = None,
) -> dict[str, Any]:
    """Convenience wrapper for the U16-to-U64 continuation contract."""
    api = api_or_root if isinstance(api_or_root, ResidentRepairAPI) else ResidentRepairAPI.open(api_or_root)
    return api.continue_to(
        start_checkpoint,
        target,
        windows=windows,
        receipt_path=receipt_path,
    )


def construct_clean_u0(
    api_or_root: ResidentRepairAPI | str | Path,
    midpoint: int | str,
    target: int | str,
    *,
    replay: Mapping[str, Any],
    receipt_path: str | Path,
) -> dict[str, Any]:
    """Convenience wrapper for the true in-memory clean-U0 replay contract."""
    api = api_or_root if isinstance(api_or_root, ResidentRepairAPI) else ResidentRepairAPI.open(api_or_root)
    return api.construct_clean_u0(midpoint, target, replay=replay, receipt_path=receipt_path)


def continue_two_spark_real(
    api_or_root: ResidentRepairAPI | str | Path,
    start_checkpoint: int | str,
    milestones: Iterable[int],
    *,
    config: Mapping[str, Any],
    receipt_path: str | Path,
) -> dict[str, Any]:
    """Convenience wrapper for the non-injectable real continuation engine."""
    api = api_or_root if isinstance(api_or_root, ResidentRepairAPI) else ResidentRepairAPI.open(api_or_root)
    return api.continue_two_spark_real(start_checkpoint, milestones, config=config, receipt_path=receipt_path)


def continue_two_spark(
    api_or_root: ResidentRepairAPI | str | Path,
    start_checkpoint: int | str,
    milestones: Iterable[int],
    *,
    config: Mapping[str, Any],
    advance_fn,
    receipt_path: str | Path,
) -> dict[str, Any]:
    """Convenience wrapper for the authorized resident two-Spark contract."""
    api = api_or_root if isinstance(api_or_root, ResidentRepairAPI) else ResidentRepairAPI.open(api_or_root)
    return api.continue_two_spark(start_checkpoint, milestones, config=config, advance_fn=advance_fn, receipt_path=receipt_path)


continue_resident_two_spark = continue_two_spark
