"""Thin two-stage orchestration over the canonical Backpack and resident rails.

This module intentionally owns no solver, scorer, or trainer implementation.  It
orders uniform V7 builds, receipt-bound Backpack mixing, exact scoring, and the
existing resident trainer while enforcing artifact-declared identity at each
boundary.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import importlib
import json
import os
from pathlib import Path
import shutil
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


def _ensure_ninja_available() -> Path:
    """Make the solve extra's Ninja executable visible to PyTorch extensions."""
    resolved = shutil.which("ninja")
    if resolved is not None:
        return Path(resolved)
    try:
        ninja = importlib.import_module("ninja")
    except ImportError as exc:
        raise RuntimeError(
            "resident Q2 execution requires the solve extra: pip install 'banana-smasher[solve]'"
        ) from exc
    bin_dir = Path(str(getattr(ninja, "BIN_DIR", ""))).resolve()
    candidate = bin_dir / "ninja"
    if not candidate.is_file():
        raise RuntimeError(f"solve extra did not provide a Ninja executable under {bin_dir}")
    os.environ["PATH"] = str(bin_dir) + os.pathsep + os.environ.get("PATH", "")
    resolved = shutil.which("ninja")
    if resolved is None:
        raise RuntimeError(f"could not expose solve-extra Ninja executable: {candidate}")
    return Path(resolved)


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"resident arm receipt contains unsupported {type(value).__name__}")


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
    checkpoint_sha256: str


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


def _checkpoint_sha(
    identity: ArtifactIdentity, expected: str, *, operation: str
) -> str:
    if (
        not isinstance(expected, str)
        or len(expected) != 64
        or any(character not in "0123456789abcdef" for character in expected)
    ):
        raise ValueError(f"{operation} checkpoint SHA must be a lowercase SHA-256")
    declared = {
        row.get("sha256")
        for row in identity.checkpoints.values()
        if isinstance(row, Mapping)
    }
    if expected not in declared:
        raise ValueError(
            f"{operation} checkpoint SHA mismatch: expected={expected} "
            f"declared={sorted(value for value in declared if isinstance(value, str))}"
        )
    return expected


def _checkpoint_receipt(
    value: Mapping[str, Any], checkpoint_sha: str, *, operation: str
) -> dict[str, Any]:
    result = dict(value)
    observed = result.get("input_checkpoint_sha256")
    if observed is not None and observed != checkpoint_sha:
        raise ValueError(
            f"{operation} receipt checkpoint SHA mismatch: "
            f"expected={checkpoint_sha} observed={observed}"
        )
    result["input_checkpoint_sha256"] = checkpoint_sha
    return result


def _verified_checkpoint_path(
    artifact: BackpackArtifact, checkpoint_path: str | Path, checkpoint_sha: str
) -> str:
    selected = Path(checkpoint_path).expanduser().resolve()
    rows = [
        row
        for row in artifact.identity.checkpoints.values()
        if isinstance(row, Mapping) and row.get("sha256") == checkpoint_sha
    ]
    if len(rows) != 1 or not isinstance(rows[0].get("path"), str):
        raise ValueError("explicit checkpoint path is not uniquely declared by artifact identity")
    declared = (artifact.root.resolve() / rows[0]["path"]).resolve()
    try:
        declared.relative_to(artifact.root.resolve())
    except ValueError as exc:
        raise ValueError("explicit checkpoint path escapes artifact root") from exc
    if selected != declared or not selected.is_file():
        raise ValueError(
            f"explicit checkpoint path mismatch: selected={selected} declared={declared}"
        )
    digest = hashlib.sha256()
    with selected.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    if digest.hexdigest() != checkpoint_sha:
        raise ValueError("explicit checkpoint bytes do not match checkpoint SHA")
    return str(selected)


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
        enforce_improvement: bool = False,
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
        self._default_checkpoint_sha: str | None = None
        self._enforce_improvement = bool(enforce_improvement)
        self._pre_result: Mapping[str, Any] | None = None
        self._training_result: Mapping[str, Any] | None = None

    def _selected_checkpoint_sha(self, value: str | None, operation: str) -> str:
        selected = value or self._default_checkpoint_sha
        if selected is None:
            raise ValueError(
                f"{operation} requires checkpoint_sha unless build_uniform bound it"
            )
        return selected

    @property
    def timing_path(self) -> Path:
        return self.run_root / "RESIDENT_ARM_TIMING.json"

    @property
    def result_path(self) -> Path:
        return self.run_root / "RESIDENT_ARM_RESULT.json"

    def _publish_result(self, value: Mapping[str, Any]) -> None:
        payload = (
            json.dumps(_json_safe(value), indent=2, sort_keys=True, allow_nan=False) + "\n"
        ).encode()
        fd, temporary_name = tempfile.mkstemp(
            prefix=f".{self.result_path.name}.", dir=self.run_root
        )
        try:
            with os.fdopen(fd, "wb") as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary_name, self.result_path)
        finally:
            try:
                os.unlink(temporary_name)
            except FileNotFoundError:
                pass

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

    def build_uniform(
        self_or_model,
        model: str | Path | None = None,
        tier: str | None = None,
        *,
        checkpoint_sha: str | None = None,
        run_root: str | Path | None = None,
        scope: str | None = None,
        native_rest: bool | None = None,
        revision: str | None = None,
        output: str | Path | None = None,
        native_spill_root: str | Path | None = None,
    ) -> "UniformBuild | ResidentRepairAPI | dict[str, Any]":
        """Build through an injected provider, or open an admitted Q2 artifact.

        Calling ``ResidentRepairAPI.build_uniform(model, tier="q2", ...)`` is
        the documented production path. Calling the same method on an instance
        preserves the lower-level provider seam used by integrations and tests.
        ``scope='routed_only', native_rest=True`` makes the routed/native intent
        explicit while preserving compatibility with already admitted artifacts.
        """
        if scope is not None and scope != "routed_only":
            raise ValueError("build_uniform scope must be 'routed_only'")
        if native_rest is not None and native_rest is not True:
            raise ValueError("build_uniform native_rest must be True")
        if isinstance(self_or_model, ResidentRepairAPI):
            if model is None or tier is None:
                raise TypeError("instance build_uniform requires model and tier")
            if checkpoint_sha is None:
                raise TypeError("instance build_uniform requires checkpoint_sha")
            if revision is not None or output is not None:
                raise TypeError("instance build_uniform does not accept revision/output")
            return self_or_model._build_uniform(
                model, tier, checkpoint_sha=checkpoint_sha
            )
        if model is not None:
            raise TypeError("class build_uniform accepts the model as its first argument")
        artifact_root = Path(self_or_model).expanduser().resolve()
        normalized_tier = {"q2": "qtip2_v7", "qtip2": "qtip2_v7"}.get(
            str(tier), str(tier)
        )
        if normalized_tier != "qtip2_v7":
            raise ValueError("documented resident production path currently requires tier='q2'")
        if revision is not None or output is not None:
            if checkpoint_sha is not None:
                raise ValueError("HF source build does not accept an artifact checkpoint SHA")
            if revision is None or output is None:
                raise ValueError("HF source build requires both revision and output")
            if scope != "routed_only" or native_rest is not True:
                raise ValueError("HF source build requires routed_only with native_rest=True")
            from .hf_moe import build_hf_moe_uniform

            return build_hf_moe_uniform(
                artifact_root,
                revision=revision,
                tier="q2",
                scope=scope,
                native_rest=native_rest,
                output=output,
                native_spill_root=native_spill_root,
            )
        if checkpoint_sha is None:
            raise TypeError("admitted artifact build_uniform requires checkpoint_sha")
        identity = ArtifactIdentity.load(artifact_root)
        _checkpoint_sha(identity, checkpoint_sha, operation="build")
        declared = _composition_tiers(identity) - _NATIVE_TIERS
        if declared != {"qtip2_v7"}:
            raise ValueError(
                "admitted production artifact is not routed-only uniform Q2: "
                f"declared={sorted(declared)}"
            )
        try:
            rank = int(os.environ["RANK"])
        except (KeyError, ValueError) as exc:
            raise ValueError("build_uniform requires distributed RANK=0 or RANK=1") from exc
        if rank not in (0, 1):
            raise ValueError("build_uniform requires distributed RANK=0 or RANK=1")
        selected_run_root = Path(
            run_root
            or os.environ.get("BANANA_SMASHER_RUN_ROOT", "banana-smasher-resident-run")
        ).expanduser().resolve()
        config = artifact_root / f"production-rails.rank{rank}.json"
        if not config.is_file():
            raise ValueError(f"admitted artifact is missing rank config: {config}")
        from .production_rails import ProductionRails

        rails = ProductionRails.from_file(config, run_root=selected_run_root)
        if isinstance(rails, ProductionRails):
            _ensure_ninja_available()
        api = ResidentRepairAPI(
            rails=rails,
            run_root=selected_run_root / "facade" / f"rank{rank}",
            enforce_improvement=True,
        )
        api._mixed = BackpackArtifact(
            root=artifact_root,
            identity=identity,
            checkpoint_sha256=checkpoint_sha,
        )
        api._default_checkpoint_sha = checkpoint_sha
        return api

    def _build_uniform(
        self, model: str | Path, tier: str, *, checkpoint_sha: str
    ) -> UniformBuild:
        if tier not in V7_UNIFORM_TIERS:
            raise ValueError(
                "uniform build requires an integer QTIP-V7 tier: "
                + ", ".join(sorted(V7_UNIFORM_TIERS))
            )
        if tier in self._uniforms:
            cached = self._uniforms[tier]
            if cached.checkpoint_sha256 != checkpoint_sha:
                raise ValueError(
                    "build checkpoint SHA mismatch for cached uniform: "
                    f"expected={checkpoint_sha} cached={cached.checkpoint_sha256}"
                )
            return cached
        destination = self.run_root / "uniform" / tier
        root = Path(
            self.rails.build_uniform(
                Path(model).expanduser().resolve(), tier, destination
            )
        ).resolve()
        identity = ArtifactIdentity.load(root)
        verified_checkpoint_sha = _checkpoint_sha(
            identity, checkpoint_sha, operation="build"
        )
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
        result = UniformBuild(
            root=root,
            identity=identity,
            checkpoint_sha256=verified_checkpoint_sha,
            tier=tier,
        )
        self._uniforms[tier] = result
        return result

    def backpack_mix(
        self,
        builds: Sequence[UniformBuild],
        bpw_target: float,
        *,
        checkpoint_sha: str,
    ) -> BackpackArtifact:
        target = _finite_positive(bpw_target, "bpw_target")
        rows = tuple(builds)
        if not rows:
            raise ValueError("Backpack mixing requires at least one uniform build")
        if any(row.checkpoint_sha256 != checkpoint_sha for row in rows):
            raise ValueError("mix checkpoint SHA mismatch across uniform builds")
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
        verified_checkpoint_sha = _checkpoint_sha(
            identity, checkpoint_sha, operation="mix"
        )
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
        self._mixed = BackpackArtifact(
            root=root,
            identity=identity,
            checkpoint_sha256=verified_checkpoint_sha,
        )
        return self._mixed

    def _score(
        self, artifact: BackpackArtifact, phase: str, *, checkpoint_sha: str
    ) -> Mapping[str, Any]:
        phase_name = "zero_update_score" if phase == "pre" else "post_update_score"
        _checkpoint_sha(artifact.identity, checkpoint_sha, operation=f"score_{phase}")
        if artifact.checkpoint_sha256 != checkpoint_sha:
            raise ValueError(f"score_{phase} checkpoint SHA mismatch for selected artifact")

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
            return _checkpoint_receipt(
                result, checkpoint_sha, operation=f"score_{phase}"
            )

        return self._timed(
            phase_name,
            execute,
        )

    def score_pre(
        self,
        artifact: BackpackArtifact | None = None,
        *,
        checkpoint_sha: str | None = None,
    ) -> Mapping[str, Any]:
        checkpoint_sha = self._selected_checkpoint_sha(checkpoint_sha, "score_pre")
        selected = artifact or self._mixed
        if selected is None:
            raise ValueError("score_pre requires a mixed Backpack")
        if self._phase_state != "initialized":
            raise ValueError("score_pre must be the first resident arm phase")
        result = self._score(selected, "pre", checkpoint_sha=checkpoint_sha)
        self._phase_state = "pre_scored"
        self._pre_result = result
        return result

    def repair_train(
        self,
        artifact: BackpackArtifact | None = None,
        *,
        updates: int,
        checkpoint_sha: str | None = None,
    ) -> Mapping[str, Any]:
        checkpoint_sha = self._selected_checkpoint_sha(checkpoint_sha, "repair_train")
        selected = artifact or self._mixed
        if selected is None:
            raise ValueError("repair_train requires a mixed Backpack")
        if (
            isinstance(updates, bool)
            or not isinstance(updates, int)
            or updates <= 0
        ):
            raise ValueError("production resident repair requires a positive update count")
        if self._phase_state != "pre_scored":
            raise ValueError("repair_train requires one completed pre-score")
        _checkpoint_sha(selected.identity, checkpoint_sha, operation="repair_train")
        if selected.checkpoint_sha256 != checkpoint_sha:
            raise ValueError("repair_train checkpoint SHA mismatch for selected artifact")

        def execute() -> dict[str, Any]:
            self._activate(selected)
            return _checkpoint_receipt(
                self.rails.train(selected, updates),
                checkpoint_sha,
                operation="repair_train",
            )

        result = self._timed(
            "four_resident_updates",
            execute,
        )
        self._phase_state = "trained"
        self._training_result = result
        return result

    def score_post(
        self,
        artifact: BackpackArtifact | None = None,
        *,
        checkpoint_sha: str | None = None,
    ) -> Mapping[str, Any]:
        checkpoint_sha = self._selected_checkpoint_sha(checkpoint_sha, "score_post")
        selected = artifact or self._mixed
        if selected is None:
            raise ValueError("score_post requires a mixed Backpack")
        if self._phase_state != "trained":
            raise ValueError("score_post requires completed resident training")
        result = self._score(selected, "post", checkpoint_sha=checkpoint_sha)
        self._phase_state = "completed"
        self._publish_timing(status="PASS")
        if self._enforce_improvement:
            if self._pre_result is None or self._training_result is None:
                raise RuntimeError("improvement verdict lacks pre-score or training receipt")
            pre_kld = float(self._pre_result["mean_kld"])
            post_kld = float(result["mean_kld"])
            improvement = {
                "pre_kld": pre_kld,
                "post_kld": post_kld,
                "delta_kld": post_kld - pre_kld,
                "improved": post_kld < pre_kld,
            }
            receipt = {
                "schema": "banana-smasher-resident-arm-result-v1",
                "status": "PASS" if improvement["improved"] else "FAILED",
                "input_checkpoint_path": None,
                "input_checkpoint_sha256": checkpoint_sha,
                "pre": dict(self._pre_result),
                "training": dict(self._training_result),
                "post": dict(result),
                "improvement": improvement,
                "timing": self._timing_receipt(status="PASS"),
            }
            self._publish_result(receipt)
            if not improvement["improved"]:
                raise ValueError(
                    "resident KLD did not improve: "
                    f"pre={pre_kld:.17g} post={post_kld:.17g}; "
                    f"receipt={self.result_path}"
                )
        return result

    def run_arm(
        self,
        artifact: BackpackArtifact | None = None,
        *,
        updates: int = 4,
        checkpoint_sha: str,
        checkpoint_path: str | Path | None = None,
    ) -> dict[str, Any]:
        """Execute exactly zero-update score -> four updates -> post-update score."""
        selected = artifact or self._mixed
        if selected is None:
            raise ValueError("run_arm requires a mixed Backpack")
        verified_checkpoint_path = (
            None
            if checkpoint_path is None
            else _verified_checkpoint_path(selected, checkpoint_path, checkpoint_sha)
        )
        pre = self.score_pre(selected, checkpoint_sha=checkpoint_sha)
        training = self.repair_train(
            selected, updates=updates, checkpoint_sha=checkpoint_sha
        )
        post = self.score_post(selected, checkpoint_sha=checkpoint_sha)
        pre_kld = float(pre["mean_kld"])
        post_kld = float(post["mean_kld"])
        improvement = {
            "pre_kld": pre_kld,
            "post_kld": post_kld,
            "delta_kld": post_kld - pre_kld,
            "improved": post_kld < pre_kld,
        }
        result = {
            "schema": "banana-smasher-resident-arm-result-v1",
            "status": "PASS" if improvement["improved"] else "FAILED",
            "input_checkpoint_path": verified_checkpoint_path,
            "input_checkpoint_sha256": checkpoint_sha,
            "pre": pre,
            "training": training,
            "post": post,
            "improvement": improvement,
            "timing": self._timing_receipt(status="PASS"),
        }
        self._publish_result(result)
        if not improvement["improved"]:
            raise ValueError(
                "resident KLD did not improve: "
                f"pre={pre_kld:.17g} post={post_kld:.17g}; "
                f"receipt={self.result_path}"
            )
        return result

    def run(
        self,
        *,
        model: str | Path,
        uniform_tiers: Sequence[str],
        bpw_target: float,
        repair_updates: int,
        checkpoint_sha: str,
    ) -> dict[str, Any]:
        uniforms = tuple(
            self.build_uniform(model, tier, checkpoint_sha=checkpoint_sha)
            for tier in uniform_tiers
        )
        mixed = self.backpack_mix(
            uniforms, bpw_target, checkpoint_sha=checkpoint_sha
        )
        arm = self.run_arm(
            mixed, updates=repair_updates, checkpoint_sha=checkpoint_sha
        )
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
