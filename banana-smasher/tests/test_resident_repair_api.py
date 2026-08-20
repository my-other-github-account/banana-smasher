from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from banana_smasher.resident_repair_api import ResidentPhaseTimeout, ResidentRepairAPI


def sha(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def identity(
    root: Path, *, kind: str, tiers: list[dict], provenance: dict | None = None
) -> str:
    root.mkdir(parents=True, exist_ok=True)
    value = {
        "schema": "banana-smasher-artifact-identity-v1",
        "basis": {"model_index_sha256": sha("basis")},
        "corpora": {
            "builder_eval_sha256": sha("builder"),
            "train_score_sha256": sha("score"),
            "u0_lock_sha256": sha("lock"),
            "teacher_inventory_sha256": sha("teacher"),
        },
        "checkpoints": {"u0": {"sha256": sha("u0"), "identity_sha256": sha("u0-id")}},
        "composition": {"kind": kind, "layers": tiers},
        "canary": {
            "reference": {"kld": 0.25, "top1": 7},
            "tolerance": {"kld_abs": 0.0, "top1_abs": 0},
        },
        "runtime": {},
    }
    if provenance is not None:
        value["provenance"] = provenance
    path = root / "identity.json"
    path.write_text(json.dumps(value, sort_keys=True))
    return hashlib.sha256(path.read_bytes()).hexdigest()


class Rails:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.calls: list[tuple] = []

    def build_uniform(self, model: Path, tier: str, output: Path) -> Path:
        self.calls.append(("build_uniform", model, tier, output))
        identity(
            output,
            kind="uniform-qtip-v7",
            tiers=[{"layer": 0, "tiers": {tier: 2, "native": 1}}],
        )
        return output

    def mix(self, builds, bpw_target: float, output: Path) -> Path:
        self.calls.append(
            ("mix", tuple(row.tier for row in builds), bpw_target, output)
        )
        provenance = {
            "uniform_builds": [
                {"tier": row.tier, "identity_sha256": row.identity.sha256}
                for row in builds
            ],
            "bpw_target": bpw_target,
        }
        identity(
            output,
            kind="mixed-qtip-v7-backpack",
            tiers=[{"layer": 0, "tiers": {"qtip1_v7": 1, "qtip2_v7": 1, "native": 1}}],
            provenance=provenance,
        )
        return output

    def load_resident(self, artifact):
        self.calls.append(("load_resident", artifact.root))

    def hot_swap(self, artifact):
        self.calls.append(("hot_swap", artifact.root))

    def score(self, artifact, phase: str):
        self.calls.append(("score", phase, artifact.root))
        return {"mean_kld": 0.25, "top1_matches": 7, "phase": phase}

    def train(self, artifact, updates: int):
        self.calls.append(("train", updates, artifact.root))
        return {"updates": updates, "artifact_root": artifact.root}


def test_full_pipeline_builds_uniforms_then_mixes_without_resolving(
    tmp_path: Path,
) -> None:
    rails = Rails(tmp_path)
    api = ResidentRepairAPI(rails=rails, run_root=tmp_path / "run")
    result = api.run(
        model=tmp_path / "model",
        uniform_tiers=("qtip1_v7", "qtip2_v7"),
        bpw_target=1.5,
        repair_updates=0,
    )
    assert [row[0] for row in rails.calls] == [
        "build_uniform",
        "build_uniform",
        "mix",
        "load_resident",
        "score",
        "hot_swap",
        "train",
        "hot_swap",
        "score",
    ]
    assert result["pre"]["phase"] == "pre"
    assert result["post"]["phase"] == "post"
    assert result["mixed"].identity.composition_kind == "mixed-qtip-v7-backpack"


def test_tier_space_is_integer_v7_only_and_rejects_d4_or_fractional(
    tmp_path: Path,
) -> None:
    api = ResidentRepairAPI(rails=Rails(tmp_path), run_root=tmp_path / "run")
    for tier in ("d4", "qtip2_5", "qtip1.5_v7", "native"):
        with pytest.raises(ValueError, match="QTIP-V7 tier"):
            api.build_uniform(tmp_path / "model", tier)


def test_mix_fails_closed_if_declared_provenance_does_not_bind_builds(
    tmp_path: Path,
) -> None:
    class BadRails(Rails):
        def mix(self, builds, bpw_target: float, output: Path) -> Path:
            identity(
                output,
                kind="mixed-qtip-v7-backpack",
                tiers=[{"layer": 0, "tiers": {"qtip1_v7": 2}}],
                provenance={"uniform_builds": [], "bpw_target": bpw_target},
            )
            return output

    api = ResidentRepairAPI(rails=BadRails(tmp_path), run_root=tmp_path / "run")
    build = api.build_uniform(tmp_path / "model", "qtip1_v7")
    with pytest.raises(ValueError, match="uniform-build provenance"):
        api.backpack_mix((build,), 1.0)


def test_canary_is_read_from_artifact_not_source_constant(tmp_path: Path) -> None:
    rails = Rails(tmp_path)
    api = ResidentRepairAPI(rails=rails, run_root=tmp_path / "run")
    build = api.build_uniform(tmp_path / "model", "qtip1_v7")
    rails.score = lambda artifact, phase: {"mean_kld": 0.251, "top1_matches": 7}
    with pytest.raises(ValueError, match="declared tolerance"):
        api.score_pre(build)


def test_score_phase_raises_named_error_when_fast_budget_is_exceeded(
    tmp_path: Path,
) -> None:
    now = 0.0

    def clock() -> float:
        return now

    class SlowRails(Rails):
        def score(self, artifact, phase: str):
            nonlocal now
            now += 301.0
            return super().score(artifact, phase)

    api = ResidentRepairAPI(
        rails=SlowRails(tmp_path), run_root=tmp_path / "run", clock=clock
    )
    build = api.build_uniform(tmp_path / "model", "qtip1_v7")
    with pytest.raises(ResidentPhaseTimeout, match=r"score_pre.*300"):
        api.score_pre(build)


def test_model_loads_once_and_every_later_phase_hot_swaps_in_memory(
    tmp_path: Path,
) -> None:
    rails = Rails(tmp_path)
    api = ResidentRepairAPI(rails=rails, run_root=tmp_path / "run")
    api.run(
        model=tmp_path / "model",
        uniform_tiers=("qtip1_v7", "qtip2_v7"),
        bpw_target=1.5,
        repair_updates=4,
    )
    resident_calls = [
        row[0] for row in rails.calls if row[0] in {"load_resident", "hot_swap"}
    ]
    assert resident_calls == ["load_resident", "hot_swap", "hot_swap"]
