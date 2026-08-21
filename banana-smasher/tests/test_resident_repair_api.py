from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path

import pytest

from banana_smasher.resident_repair_api import (
    ARM_BUDGET_SECONDS,
    PHASE_BUDGET_SECONDS,
    ResidentPhaseTimeout,
    ResidentRepairAPI,
)


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
        repair_updates=4,
        checkpoint_sha=sha("u0"),
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


def test_every_checkpoint_loading_operation_requires_one_explicit_sha() -> None:
    for operation in (
        "build_uniform",
        "backpack_mix",
        "score_pre",
        "repair_train",
        "score_post",
        "run_arm",
        "run",
    ):
        parameter = inspect.signature(getattr(ResidentRepairAPI, operation)).parameters[
            "checkpoint_sha"
        ]
        assert parameter.default is inspect.Parameter.empty
        assert parameter.kind is inspect.Parameter.KEYWORD_ONLY


def test_checkpoint_sha_is_refused_on_mismatch_and_echoed_in_every_receipt(
    tmp_path: Path,
) -> None:
    expected = sha("u0")
    rails = Rails(tmp_path)
    api = ResidentRepairAPI(rails=rails, run_root=tmp_path / "run")
    build = api.build_uniform(
        tmp_path / "model", "qtip1_v7", checkpoint_sha=expected
    )
    second = api.build_uniform(
        tmp_path / "model", "qtip2_v7", checkpoint_sha=expected
    )
    assert build.checkpoint_sha256 == expected

    with pytest.raises(ValueError, match="checkpoint SHA mismatch"):
        api.backpack_mix((build, second), 1.0, checkpoint_sha=sha("wrong"))

    mixed = api.backpack_mix((build, second), 1.0, checkpoint_sha=expected)
    assert mixed.checkpoint_sha256 == expected
    assert (
        api.score_pre(mixed, checkpoint_sha=expected)["input_checkpoint_sha256"]
        == expected
    )
    assert (
        api.repair_train(mixed, updates=4, checkpoint_sha=expected)[
            "input_checkpoint_sha256"
        ]
        == expected
    )
    assert (
        api.score_post(mixed, checkpoint_sha=expected)["input_checkpoint_sha256"]
        == expected
    )


def test_checkpoint_receipt_echo_cannot_overwrite_backend_mismatch(tmp_path: Path) -> None:
    expected = sha("u0")

    class BadReceiptRails(Rails):
        def score(self, artifact, phase: str):
            return {
                "mean_kld": 0.25,
                "top1_matches": 7,
                "input_checkpoint_sha256": sha("wrong"),
            }

    api = ResidentRepairAPI(
        rails=BadReceiptRails(tmp_path), run_root=tmp_path / "run"
    )
    build = api.build_uniform(
        tmp_path / "model", "qtip1_v7", checkpoint_sha=expected
    )
    with pytest.raises(ValueError, match="receipt checkpoint SHA mismatch"):
        api.score_pre(build, checkpoint_sha=expected)


def test_tier_space_is_integer_v7_only_and_rejects_d4_or_fractional(
    tmp_path: Path,
) -> None:
    api = ResidentRepairAPI(rails=Rails(tmp_path), run_root=tmp_path / "run")
    for tier in ("d4", "qtip2_5", "qtip1.5_v7", "native"):
        with pytest.raises(ValueError, match="QTIP-V7 tier"):
            api.build_uniform(tmp_path / "model", tier, checkpoint_sha=sha("u0"))


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
    build = api.build_uniform(
        tmp_path / "model", "qtip1_v7", checkpoint_sha=sha("u0")
    )
    with pytest.raises(ValueError, match="uniform-build provenance"):
        api.backpack_mix((build,), 1.0, checkpoint_sha=sha("u0"))


def test_canary_is_read_from_artifact_not_source_constant(tmp_path: Path) -> None:
    rails = Rails(tmp_path)
    api = ResidentRepairAPI(rails=rails, run_root=tmp_path / "run")
    build = api.build_uniform(
        tmp_path / "model", "qtip1_v7", checkpoint_sha=sha("u0")
    )
    rails.score = lambda artifact, phase: {"mean_kld": 0.251, "top1_matches": 7}
    with pytest.raises(ValueError, match="declared tolerance"):
        api.score_pre(build, checkpoint_sha=sha("u0"))


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
    build = api.build_uniform(
        tmp_path / "model", "qtip1_v7", checkpoint_sha=sha("u0")
    )
    with pytest.raises(ResidentPhaseTimeout, match=r"score_pre.*300"):
        api.score_pre(build, checkpoint_sha=sha("u0"))


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
        checkpoint_sha=sha("u0"),
    )
    resident_calls = [
        row[0] for row in rails.calls if row[0] in {"load_resident", "hot_swap"}
    ]
    assert resident_calls == ["load_resident", "hot_swap", "hot_swap"]


class FaultClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class TimedRails(Rails):
    def __init__(self, root: Path, clock: FaultClock, durations: dict[str, float]) -> None:
        super().__init__(root)
        self.clock = clock
        self.durations = durations

    def load_resident(self, artifact):
        self.clock.advance(self.durations.get("resident_load", 0.0))
        return super().load_resident(artifact)

    def hot_swap(self, artifact):
        self.clock.advance(self.durations.get("hot_swap", 0.0))
        return super().hot_swap(artifact)

    def score(self, artifact, phase: str):
        self.clock.advance(self.durations.get(f"score_{phase}", 0.0))
        return super().score(artifact, phase)

    def train(self, artifact, updates: int):
        self.clock.advance(self.durations.get("train", 0.0))
        return super().train(artifact, updates)


@pytest.mark.parametrize(
    ("durations", "expected_phase"),
    [
        ({"score_pre": PHASE_BUDGET_SECONDS["zero_update_score"] + 1}, "zero_update_score"),
        ({"train": PHASE_BUDGET_SECONDS["four_resident_updates"] + 1}, "four_resident_updates"),
        ({"score_post": PHASE_BUDGET_SECONDS["post_update_score"] + 1}, "post_update_score"),
    ],
)
def test_each_named_arm_phase_fails_hard_and_records_fault_timing(
    tmp_path: Path, durations: dict[str, float], expected_phase: str
) -> None:
    clock = FaultClock()
    api = ResidentRepairAPI(
        rails=TimedRails(tmp_path, clock, durations),
        run_root=tmp_path / "run",
        clock=clock,
    )
    build = api.build_uniform(
        tmp_path / "model", "qtip1_v7", checkpoint_sha=sha("u0")
    )

    with pytest.raises(ResidentPhaseTimeout, match=expected_phase):
        api.run_arm(build, updates=4, checkpoint_sha=sha("u0"))

    receipt = json.loads((tmp_path / "run" / "RESIDENT_ARM_TIMING.json").read_text())
    assert receipt["status"] == "FAILED"
    assert receipt["failed_phase"] == expected_phase
    assert receipt["phases"][-1]["phase"] == expected_phase
    assert receipt["phases"][-1]["status"] == "TIMEOUT"


def test_total_arm_budget_fails_hard_and_identifies_current_phase(tmp_path: Path) -> None:
    clock = FaultClock()
    durations = {
        "score_pre": PHASE_BUDGET_SECONDS["zero_update_score"],
        "train": PHASE_BUDGET_SECONDS["four_resident_updates"],
    }
    api = ResidentRepairAPI(
        rails=TimedRails(tmp_path, clock, durations),
        run_root=tmp_path / "run",
        clock=clock,
    )
    build = api.build_uniform(
        tmp_path / "model", "qtip1_v7", checkpoint_sha=sha("u0")
    )
    api.score_pre(build, checkpoint_sha=sha("u0"))
    clock.advance(301.0)

    with pytest.raises(
        ResidentPhaseTimeout,
        match=rf"four_resident_updates.*arm_cycle.*{ARM_BUDGET_SECONDS:g}",
    ):
        api.repair_train(build, updates=4, checkpoint_sha=sha("u0"))

    receipt = json.loads((tmp_path / "run" / "RESIDENT_ARM_TIMING.json").read_text())
    assert receipt["status"] == "FAILED"
    assert receipt["failed_phase"] == "four_resident_updates"
    assert receipt["total_elapsed_seconds"] == ARM_BUDGET_SECONDS + 1.0


def test_successful_arm_records_all_named_phases_and_exact_cycle(tmp_path: Path) -> None:
    clock = FaultClock()
    durations = {"resident_load": 2.0, "score_pre": 3.0, "hot_swap": 1.0, "train": 7.0, "score_post": 5.0}
    rails = TimedRails(tmp_path, clock, durations)
    api = ResidentRepairAPI(rails=rails, run_root=tmp_path / "run", clock=clock)
    build = api.build_uniform(
        tmp_path / "model", "qtip1_v7", checkpoint_sha=sha("u0")
    )

    result = api.run_arm(build, updates=4, checkpoint_sha=sha("u0"))

    assert result["training"]["updates"] == 4
    assert [row[0] for row in rails.calls] == [
        "build_uniform", "load_resident", "score", "hot_swap", "train", "hot_swap", "score"
    ]
    timing = result["timing"]
    assert timing["status"] == "PASS"
    assert [row["phase"] for row in timing["phases"]] == [
        "zero_update_score", "four_resident_updates", "post_update_score"
    ]
    assert [row["elapsed_seconds"] for row in timing["phases"]] == [5.0, 8.0, 6.0]
    assert timing["total_elapsed_seconds"] == 19.0
