"""Thin two-stage orchestration over the canonical Backpack and resident rails.

This module intentionally owns no solver, scorer, or trainer implementation.  It
orders uniform V7 builds, receipt-bound Backpack mixing, exact scoring, and the
existing resident trainer while enforcing artifact-declared identity at each
boundary.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import tempfile
import time
from typing import Any, Callable, Mapping, Protocol, Sequence, TypeVar

from .artifact_identity import ArtifactIdentity


V7_UNIFORM_TIERS = frozenset({"qtip1_v7", "qtip2_v7", "qtip3_v7", "qtip4_v7"})
_NATIVE_TIERS = frozenset({"native", "native_mxfp4"})
SCORE_BUDGET_SECONDS = 300.0
TRAIN_BUDGET_SECONDS = 2_100.0
ARM_BUDGET_SECONDS = 2_700.0
PHASE_BUDGET_SECONDS = {
    "zero_update_score": SCORE_BUDGET_SECONDS,
    "four_resident_updates": TRAIN_BUDGET_SECONDS,
    "post_update_score": SCORE_BUDGET_SECONDS,
}
_PHASE_PUBLIC_NAMES = {
    "zero_update_score": "score_pre",
    "four_resident_updates": "repair_train",
    "post_update_score": "score_post",
}
_T = TypeVar("_T")


class ResidentPhaseTimeout(RuntimeError):
    """The resident rail exceeded a hard phase budget."""


class PipelineRails(Protocol):
    """Provider boundary implemented by canonical package build/score/train rails."""

    def build_uniform(self, model: Path, tier: str, output: Path) -> str | Path: ...

    def mix(
        self, builds: Sequence["UniformBuild"], bpw_target: float, output: Path
    ) -> str | Path: ...

    def load_resident(self, artifact: "BackpackArtifact") -> None: ...

    def hot_swap(self, artifact: "BackpackArtifact") -> None: ...

    def score(self, artifact: "BackpackArtifact", phase: str) -> Mapping[str, Any]: ...

    def train(
        self, artifact: "BackpackArtifact", updates: int
    ) -> Mapping[str, Any]: ...


@dataclass(frozen=True)
class BackpackArtifact:
    root: Path
    identity: ArtifactIdentity


@dataclass(frozen=True)
class UniformBuild(BackpackArtifact):
    tier: str


@dataclass(frozen=True)
class PipelineResult:
    uniforms: tuple[UniformBuild, ...]
    mixed: BackpackArtifact
    pre: Mapping[str, Any]
    training: Mapping[str, Any]
    post: Mapping[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {
            "uniforms": self.uniforms,
            "mixed": self.mixed,
            "pre": self.pre,
            "training": self.training,
            "post": self.post,
        }


def _finite_positive(value: object, label: str) -> float:
    import math

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{label} must be finite and positive")
    return result


def _composition_tiers(identity: ArtifactIdentity) -> set[str]:
    result: set[str] = set()
    for row in identity.composition:
        tiers = row.get("tiers")
        if isinstance(tiers, Mapping):
            result.update(str(name) for name, count in tiers.items() if int(count) > 0)
    return result


class ResidentRepairAPI:
    """One-path facade for uniform-build -> mix -> score -> train -> score.

    ``rails`` must delegate to the package's established implementations.  The
    facade only validates ordering and artifact identity; it never computes
    candidates, scores logits, or applies optimizer updates itself.
    """

    def __init__(
        self,
        *,
        rails: PipelineRails,
        run_root: str | Path,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.rails = rails
        self.run_root = Path(run_root).expanduser().resolve()
        self.run_root.mkdir(parents=True, exist_ok=True)
        self._clock = clock
        self._uniforms: dict[str, UniformBuild] = {}
        self._mixed: BackpackArtifact | None = None
        self._resident_loaded = False
        self._phase_state = "initialized"
        self._arm_started: float | None = None
        self._phase_timings: list[dict[str, Any]] = []

    @property
    def timing_path(self) -> Path:
        return self.run_root / "RESIDENT_ARM_TIMING.json"

    def _timing_receipt(
        self, *, status: str, failed_phase: str | None = None
    ) -> dict[str, Any]:
        total_elapsed = (
            0.0
            if self._arm_started is None
            else self._clock() - self._arm_started
        )
        return {
            "schema": "banana-smasher-resident-arm-timing-v1",
            "status": status,
            "failed_phase": failed_phase,
            "total_budget_seconds": ARM_BUDGET_SECONDS,
            "total_elapsed_seconds": total_elapsed,
            "phases": [dict(row) for row in self._phase_timings],
        }

    def _publish_timing(self, *, status: str, failed_phase: str | None = None) -> dict[str, Any]:
        receipt = self._timing_receipt(status=status, failed_phase=failed_phase)
        payload = (json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
        fd, temporary_name = tempfile.mkstemp(
            prefix=f".{self.timing_path.name}.", dir=self.run_root
        )
        try:
            with os.fdopen(fd, "wb") as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary_name, self.timing_path)
        finally:
            try:
                os.unlink(temporary_name)
            except FileNotFoundError:
                pass
        return receipt

    def _timed(self, phase: str, call: Callable[[], _T]) -> _T:
        if phase not in PHASE_BUDGET_SECONDS:
            raise ValueError(f"unknown resident arm phase: {phase}")
        if self._arm_started is None:
            self._arm_started = self._clock()
        started = self._clock()
        try:
            result = call()
        except Exception as exc:
            elapsed = self._clock() - started
            self._phase_timings.append(
                {
                    "phase": phase,
                    "public_operation": _PHASE_PUBLIC_NAMES[phase],
                    "budget_seconds": PHASE_BUDGET_SECONDS[phase],
                    "elapsed_seconds": elapsed,
                    "status": "FAILED",
                    "error_type": type(exc).__name__,
                }
            )
            self._publish_timing(status="FAILED", failed_phase=phase)
            raise
        elapsed = self._clock() - started
        total_elapsed = self._clock() - self._arm_started
        phase_exceeded = elapsed > PHASE_BUDGET_SECONDS[phase]
        total_exceeded = total_elapsed > ARM_BUDGET_SECONDS
        self._phase_timings.append(
            {
                "phase": phase,
                "public_operation": _PHASE_PUBLIC_NAMES[phase],
                "budget_seconds": PHASE_BUDGET_SECONDS[phase],
                "elapsed_seconds": elapsed,
                "status": "TIMEOUT" if phase_exceeded or total_exceeded else "PASS",
            }
        )
        if phase_exceeded or total_exceeded:
            self._publish_timing(status="FAILED", failed_phase=phase)
            public_name = _PHASE_PUBLIC_NAMES[phase]
            if phase_exceeded and total_exceeded:
                detail = (
                    f"phase budget {PHASE_BUDGET_SECONDS[phase]:g}s and arm_cycle total "
                    f"budget {ARM_BUDGET_SECONDS:g}s"
                )
            elif phase_exceeded:
                detail = f"phase budget {PHASE_BUDGET_SECONDS[phase]:g}s"
            else:
                detail = f"arm_cycle total budget {ARM_BUDGET_SECONDS:g}s"
            raise ResidentPhaseTimeout(
                f"{phase} ({public_name}) exceeded {detail}: "
                f"phase={elapsed:.3f}s total={total_elapsed:.3f}s"
            )
        self._publish_timing(status="IN_PROGRESS")
        return result

    def _activate(self, artifact: BackpackArtifact) -> None:
        if not self._resident_loaded:
            self.rails.load_resident(artifact)
            self._resident_loaded = True
            return
        self.rails.hot_swap(artifact)

    def build_uniform(self, model: str | Path, tier: str) -> UniformBuild:
        if tier not in V7_UNIFORM_TIERS:
            raise ValueError(
                "uniform build requires an integer QTIP-V7 tier: "
                + ", ".join(sorted(V7_UNIFORM_TIERS))
            )
        if tier in self._uniforms:
            return self._uniforms[tier]
        destination = self.run_root / "uniform" / tier
        root = Path(
            self.rails.build_uniform(
                Path(model).expanduser().resolve(), tier, destination
            )
        ).resolve()
        identity = ArtifactIdentity.load(root)
        declared = _composition_tiers(identity)
        forbidden = declared - {tier} - _NATIVE_TIERS
        if forbidden or tier not in declared:
            raise ValueError(
                f"uniform QTIP-V7 identity tier drift: expected={tier!r} declared={sorted(declared)}"
            )
        if identity.composition_kind != "uniform-qtip-v7":
            raise ValueError(
                "uniform build identity must declare uniform-qtip-v7 composition"
            )
        result = UniformBuild(root=root, identity=identity, tier=tier)
        self._uniforms[tier] = result
        return result

    def backpack_mix(
        self, builds: Sequence[UniformBuild], bpw_target: float
    ) -> BackpackArtifact:
        target = _finite_positive(bpw_target, "bpw_target")
        rows = tuple(builds)
        if not rows:
            raise ValueError("Backpack mixing requires at least one uniform build")
        tiers = [row.tier for row in rows]
        if len(set(tiers)) != len(tiers) or any(
            tier not in V7_UNIFORM_TIERS for tier in tiers
        ):
            raise ValueError(
                "Backpack mixing requires unique integer QTIP-V7 uniform tiers"
            )
        bases = {row.identity.basis_sha256 for row in rows}
        if len(bases) != 1:
            raise ValueError("uniform builds do not share one model basis")
        destination = self.run_root / "mixed"
        root = Path(self.rails.mix(rows, target, destination)).resolve()
        identity = ArtifactIdentity.load(root)
        if identity.basis_sha256 != rows[0].identity.basis_sha256:
            raise ValueError("mixed Backpack basis differs from its uniform builds")
        if identity.composition_kind != "mixed-qtip-v7-backpack":
            raise ValueError(
                "mixed artifact identity must declare mixed-qtip-v7-backpack"
            )
        declared = _composition_tiers(identity)
        if declared - set(tiers) - _NATIVE_TIERS:
            raise ValueError(
                "mixed Backpack identity declares a tier outside its uniform builds"
            )
        provenance = identity.document.get("provenance")
        expected = [
            {"tier": row.tier, "identity_sha256": row.identity.sha256} for row in rows
        ]
        if (
            not isinstance(provenance, Mapping)
            or provenance.get("uniform_builds") != expected
        ):
            raise ValueError("mixed artifact uniform-build provenance mismatch")
        if float(provenance.get("bpw_target", -1.0)) != target:
            raise ValueError("mixed artifact BPW target provenance mismatch")
        self._mixed = BackpackArtifact(root=root, identity=identity)
        return self._mixed

    def _score(self, artifact: BackpackArtifact, phase: str) -> Mapping[str, Any]:
        phase_name = "zero_update_score" if phase == "pre" else "post_update_score"

        def execute() -> dict[str, Any]:
            self._activate(artifact)
            result = dict(self.rails.score(artifact, phase))
            try:
                kld = float(result["mean_kld"])
                top1 = int(result["top1_matches"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(
                    "canonical scorer result lacks mean_kld/top1_matches"
                ) from exc
            artifact.identity.require_canary(kld=kld, top1=top1)
            return result

        return self._timed(
            phase_name,
            execute,
        )

    def score_pre(self, artifact: BackpackArtifact | None = None) -> Mapping[str, Any]:
        selected = artifact or self._mixed
        if selected is None:
            raise ValueError("score_pre requires a mixed Backpack")
        if self._phase_state != "initialized":
            raise ValueError("score_pre must be the first resident arm phase")
        result = self._score(selected, "pre")
        self._phase_state = "pre_scored"
        return result

    def repair_train(
        self, artifact: BackpackArtifact | None = None, *, updates: int
    ) -> Mapping[str, Any]:
        selected = artifact or self._mixed
        if selected is None:
            raise ValueError("repair_train requires a mixed Backpack")
        if isinstance(updates, bool) or updates != 4:
            raise ValueError("production resident arm requires exactly four updates")
        if self._phase_state != "pre_scored":
            raise ValueError("repair_train requires one completed pre-score")
        def execute() -> dict[str, Any]:
            self._activate(selected)
            return dict(self.rails.train(selected, updates))

        result = self._timed(
            "four_resident_updates",
            execute,
        )
        self._phase_state = "trained"
        return result

    def score_post(self, artifact: BackpackArtifact | None = None) -> Mapping[str, Any]:
        selected = artifact or self._mixed
        if selected is None:
            raise ValueError("score_post requires a mixed Backpack")
        if self._phase_state != "trained":
            raise ValueError("score_post requires completed resident training")
        result = self._score(selected, "post")
        self._phase_state = "completed"
        self._publish_timing(status="PASS")
        return result

    def run_arm(
        self, artifact: BackpackArtifact | None = None, *, updates: int = 4
    ) -> dict[str, Any]:
        """Execute exactly zero-update score -> four updates -> post-update score."""
        selected = artifact or self._mixed
        if selected is None:
            raise ValueError("run_arm requires a mixed Backpack")
        pre = self.score_pre(selected)
        training = self.repair_train(selected, updates=updates)
        post = self.score_post(selected)
        return {
            "pre": pre,
            "training": training,
            "post": post,
            "timing": self._timing_receipt(status="PASS"),
        }

    def run(
        self,
        *,
        model: str | Path,
        uniform_tiers: Sequence[str],
        bpw_target: float,
        repair_updates: int,
    ) -> dict[str, Any]:
        uniforms = tuple(self.build_uniform(model, tier) for tier in uniform_tiers)
        mixed = self.backpack_mix(uniforms, bpw_target)
        arm = self.run_arm(mixed, updates=repair_updates)
        result = PipelineResult(
            uniforms=uniforms,
            mixed=mixed,
            pre=arm["pre"],
            training=arm["training"],
            post=arm["post"],
        ).as_dict()
        result["timing"] = arm["timing"]
        return result


__all__ = [
    "BackpackArtifact",
    "PipelineRails",
    "PipelineResult",
    "ResidentPhaseTimeout",
    "ResidentRepairAPI",
    "SCORE_BUDGET_SECONDS",
    "TRAIN_BUDGET_SECONDS",
    "ARM_BUDGET_SECONDS",
    "PHASE_BUDGET_SECONDS",
    "UniformBuild",
    "V7_UNIFORM_TIERS",
]
