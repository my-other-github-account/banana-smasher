"""Thin two-stage orchestration over the canonical Backpack and resident rails.

This module intentionally owns no solver, scorer, or trainer implementation.  It
orders uniform V7 builds, receipt-bound Backpack mixing, exact scoring, and the
existing resident trainer while enforcing artifact-declared identity at each
boundary.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

from .artifact_identity import ArtifactIdentity


V7_UNIFORM_TIERS = frozenset({"qtip1_v7", "qtip2_v7", "qtip3_v7", "qtip4_v7"})
_NATIVE_TIERS = frozenset({"native", "native_mxfp4"})


class PipelineRails(Protocol):
    """Provider boundary implemented by canonical package build/score/train rails."""

    def build_uniform(self, model: Path, tier: str, output: Path) -> str | Path: ...

    def mix(
        self, builds: Sequence["UniformBuild"], bpw_target: float, output: Path
    ) -> str | Path: ...

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

    def __init__(self, *, rails: PipelineRails, run_root: str | Path) -> None:
        self.rails = rails
        self.run_root = Path(run_root).expanduser().resolve()
        self._uniforms: dict[str, UniformBuild] = {}
        self._mixed: BackpackArtifact | None = None

    def build_uniform(self, model: str | Path, tier: str) -> UniformBuild:
        if tier not in V7_UNIFORM_TIERS:
            raise ValueError(
                "uniform build requires an integer QTIP-V7 tier: "
                + ", ".join(sorted(V7_UNIFORM_TIERS))
            )
        if tier in self._uniforms:
            return self._uniforms[tier]
        destination = self.run_root / "uniform" / tier
        root = Path(self.rails.build_uniform(Path(model).expanduser().resolve(), tier, destination)).resolve()
        identity = ArtifactIdentity.load(root)
        declared = _composition_tiers(identity)
        forbidden = declared - {tier} - _NATIVE_TIERS
        if forbidden or tier not in declared:
            raise ValueError(
                f"uniform QTIP-V7 identity tier drift: expected={tier!r} declared={sorted(declared)}"
            )
        if identity.composition_kind != "uniform-qtip-v7":
            raise ValueError("uniform build identity must declare uniform-qtip-v7 composition")
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
        if len(set(tiers)) != len(tiers) or any(tier not in V7_UNIFORM_TIERS for tier in tiers):
            raise ValueError("Backpack mixing requires unique integer QTIP-V7 uniform tiers")
        bases = {row.identity.basis_sha256 for row in rows}
        if len(bases) != 1:
            raise ValueError("uniform builds do not share one model basis")
        destination = self.run_root / "mixed"
        root = Path(self.rails.mix(rows, target, destination)).resolve()
        identity = ArtifactIdentity.load(root)
        if identity.basis_sha256 != rows[0].identity.basis_sha256:
            raise ValueError("mixed Backpack basis differs from its uniform builds")
        if identity.composition_kind != "mixed-qtip-v7-backpack":
            raise ValueError("mixed artifact identity must declare mixed-qtip-v7-backpack")
        declared = _composition_tiers(identity)
        if declared - set(tiers) - _NATIVE_TIERS:
            raise ValueError("mixed Backpack identity declares a tier outside its uniform builds")
        provenance = identity.document.get("provenance")
        expected = [
            {"tier": row.tier, "identity_sha256": row.identity.sha256} for row in rows
        ]
        if not isinstance(provenance, Mapping) or provenance.get("uniform_builds") != expected:
            raise ValueError("mixed artifact uniform-build provenance mismatch")
        if float(provenance.get("bpw_target", -1.0)) != target:
            raise ValueError("mixed artifact BPW target provenance mismatch")
        self._mixed = BackpackArtifact(root=root, identity=identity)
        return self._mixed

    def _score(self, artifact: BackpackArtifact, phase: str) -> Mapping[str, Any]:
        result = dict(self.rails.score(artifact, phase))
        try:
            kld = float(result["mean_kld"])
            top1 = int(result["top1_matches"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("canonical scorer result lacks mean_kld/top1_matches") from exc
        artifact.identity.require_canary(kld=kld, top1=top1)
        return result

    def score_pre(self, artifact: BackpackArtifact | None = None) -> Mapping[str, Any]:
        selected = artifact or self._mixed
        if selected is None:
            raise ValueError("score_pre requires a mixed Backpack")
        return self._score(selected, "pre")

    def repair_train(
        self, artifact: BackpackArtifact | None = None, *, updates: int
    ) -> Mapping[str, Any]:
        selected = artifact or self._mixed
        if selected is None:
            raise ValueError("repair_train requires a mixed Backpack")
        if isinstance(updates, bool) or not isinstance(updates, int) or updates < 0:
            raise ValueError("repair updates must be a nonnegative integer")
        return dict(self.rails.train(selected, updates))

    def score_post(self, artifact: BackpackArtifact | None = None) -> Mapping[str, Any]:
        selected = artifact or self._mixed
        if selected is None:
            raise ValueError("score_post requires a mixed Backpack")
        return self._score(selected, "post")

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
        pre = self.score_pre(mixed)
        training = self.repair_train(mixed, updates=repair_updates)
        post = self.score_post(mixed)
        return PipelineResult(
            uniforms=uniforms,
            mixed=mixed,
            pre=pre,
            training=training,
            post=post,
        ).as_dict()


__all__ = [
    "BackpackArtifact",
    "PipelineRails",
    "PipelineResult",
    "ResidentRepairAPI",
    "UniformBuild",
    "V7_UNIFORM_TIERS",
]
