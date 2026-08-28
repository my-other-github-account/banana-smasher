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
            "tolerance": {"kld_abs": 0.02, "top1_abs": 0},
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
        return {
            "mean_kld": 0.25 if phase == "pre" else 0.24,
            "top1_matches": 7,
            "phase": phase,
        }

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


def test_build_binds_checkpoint_once_and_later_operations_default_to_it() -> None:
    for operation in (
        "backpack_mix",
        "run_arm",
        "run",
    ):
        parameter = inspect.signature(getattr(ResidentRepairAPI, operation)).parameters[
            "checkpoint_sha"
        ]
        assert parameter.default is inspect.Parameter.empty
        assert parameter.kind is inspect.Parameter.KEYWORD_ONLY
    build_checkpoint = inspect.signature(ResidentRepairAPI.build_uniform).parameters[
        "checkpoint_sha"
    ]
    assert build_checkpoint.default is None
    assert build_checkpoint.kind is inspect.Parameter.KEYWORD_ONLY
    for operation in ("score_pre", "repair_train", "score_post"):
        parameter = inspect.signature(getattr(ResidentRepairAPI, operation)).parameters[
            "checkpoint_sha"
        ]
        assert parameter.default is None
        assert parameter.kind is inspect.Parameter.KEYWORD_ONLY


def test_documented_class_call_opens_admitted_q2_with_internal_defaults(
    tmp_path: Path, monkeypatch
) -> None:
    checkpoint_sha = sha("u0")
    artifact_root = tmp_path / "admitted-q2"
    identity(
        artifact_root,
        kind="uniform-qtip-v7",
        tiers=[{"layer": 0, "tiers": {"qtip2_v7": 2, "native": 1}}],
    )
    (artifact_root / "production-rails.rank0.json").write_text("{}")
    rails = Rails(tmp_path)
    monkeypatch.setenv("RANK", "0")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        "banana_smasher.production_rails.ProductionRails.from_file",
        lambda config, *, run_root: rails,
    )

    api = ResidentRepairAPI.build_uniform(
        artifact_root,
        tier="q2",
        scope="routed_only",
        native_rest=True,
        checkpoint_sha=checkpoint_sha,
    )

    assert isinstance(api, ResidentRepairAPI)
    assert api.score_pre()["input_checkpoint_sha256"] == checkpoint_sha
    assert api.repair_train(updates=4)["input_checkpoint_sha256"] == checkpoint_sha
    assert api.score_post()["input_checkpoint_sha256"] == checkpoint_sha
    assert api.result_path.parent.name == "rank0"


def test_documented_separate_calls_fail_and_seal_when_post_is_not_better(
    tmp_path: Path,
) -> None:
    class FlatRails(Rails):
        def score(self, artifact, phase: str):
            return {"mean_kld": 0.25, "top1_matches": 7, "phase": phase}

    checkpoint_sha = sha("u0")
    rails = FlatRails(tmp_path)
    api = ResidentRepairAPI(
        rails=rails, run_root=tmp_path / "run", enforce_improvement=True
    )
    build = api.build_uniform(
        tmp_path / "model", "qtip1_v7", checkpoint_sha=checkpoint_sha
    )
    api.score_pre(build, checkpoint_sha=checkpoint_sha)
    api.repair_train(build, updates=4, checkpoint_sha=checkpoint_sha)

    with pytest.raises(ValueError, match="did not improve"):
        api.score_post(build, checkpoint_sha=checkpoint_sha)

    receipt = json.loads(api.result_path.read_text())
    assert receipt["status"] == "FAILED"
    assert receipt["improvement"] == {
        "pre_kld": 0.25,
        "post_kld": 0.25,
        "delta_kld": 0.0,
        "improved": False,
    }


def test_isolated_process_state_restore_delegates_bound_receipts(tmp_path: Path) -> None:
    class IsolatedRails(Rails):
        def restore_pre_score(self, artifact, pre):
            self.calls.append(("restore_pre_score", artifact.root, pre["mean_kld"]))

        def restore_training(self, artifact, pre, training):
            self.calls.append(
                ("restore_training", artifact.root, pre["mean_kld"], training["updates"])
            )

    checkpoint_sha = sha("u0")
    rails = IsolatedRails(tmp_path)
    api = ResidentRepairAPI(rails=rails, run_root=tmp_path / "run")
    build = api.build_uniform(
        tmp_path / "model", "qtip1_v7", checkpoint_sha=checkpoint_sha
    )
    pre = {
        "mean_kld": 0.25,
        "top1_matches": 7,
        "input_checkpoint_sha256": checkpoint_sha,
    }
    api.restore_pre_score(pre, build, checkpoint_sha=checkpoint_sha)
    assert api.repair_train(build, updates=4, checkpoint_sha=checkpoint_sha)["updates"] == 4
    assert ("restore_pre_score", build.root, 0.25) in rails.calls

    post_rails = IsolatedRails(tmp_path)
    post_api = ResidentRepairAPI(rails=post_rails, run_root=tmp_path / "post")
    training = {"updates": 4, "input_checkpoint_sha256": checkpoint_sha}
    post_api.restore_training(
        pre, training, build, checkpoint_sha=checkpoint_sha
    )
    assert post_api.score_post(build, checkpoint_sha=checkpoint_sha)["phase"] == "post"
    assert ("restore_training", build.root, 0.25, 4) in post_rails.calls


def test_repair_train_accepts_any_positive_update_count(tmp_path: Path) -> None:
    checkpoint_sha = sha("u0")
    rails = Rails(tmp_path)
    api = ResidentRepairAPI(rails=rails, run_root=tmp_path / "run")
    build = api.build_uniform(
        tmp_path / "model", "qtip1_v7", checkpoint_sha=checkpoint_sha
    )
    api.score_pre(build, checkpoint_sha=checkpoint_sha)

    result = api.repair_train(build, updates=45, checkpoint_sha=checkpoint_sha)

    assert result["updates"] == 45
    assert ("train", 45, build.root) in rails.calls
    for index, invalid in enumerate((0, -1, True, 1.5)):
        other = ResidentRepairAPI(
            rails=Rails(tmp_path), run_root=tmp_path / f"bad-{index}"
        )
        with pytest.raises(ValueError, match="positive update count"):
            other.repair_train(build, updates=invalid, checkpoint_sha=checkpoint_sha)


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
    rails.score = lambda artifact, phase: {"mean_kld": 0.30, "top1_matches": 7}
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
            now += 1_201.0
            return super().score(artifact, phase)

    api = ResidentRepairAPI(
        rails=SlowRails(tmp_path), run_root=tmp_path / "run", clock=clock
    )
    build = api.build_uniform(
        tmp_path / "model", "qtip1_v7", checkpoint_sha=sha("u0")
    )
    with pytest.raises(ResidentPhaseTimeout, match=r"score_pre.*1200"):
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
    clock.advance(
        ARM_BUDGET_SECONDS
        - PHASE_BUDGET_SECONDS["zero_update_score"]
        - PHASE_BUDGET_SECONDS["four_resident_updates"]
        + 1.0
    )

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


def test_arm_requires_measured_kld_improvement_and_seals_both_numbers(
    tmp_path: Path, monkeypatch
) -> None:
    class ScoredRails(Rails):
        post_kld = 0.24

        def score(self, artifact, phase: str):
            result = super().score(artifact, phase)
            result["mean_kld"] = 0.25 if phase == "pre" else self.post_kld
            return result

        def train(self, artifact, updates: int):
            return {"updates": updates, "artifact_root": str(artifact.root)}

    monkeypatch.setattr(
        "banana_smasher.artifact_identity.ArtifactIdentity.require_canary",
        lambda self, *, kld, top1: None,
    )
    rails = ScoredRails(tmp_path)
    api = ResidentRepairAPI(rails=rails, run_root=tmp_path / "pass")
    build = api.build_uniform(
        tmp_path / "model", "qtip1_v7", checkpoint_sha=sha("u0")
    )

    result = api.run_arm(build, checkpoint_sha=sha("u0"))

    assert result["improvement"] == {
        "pre_kld": 0.25,
        "post_kld": 0.24,
        "delta_kld": pytest.approx(-0.01),
        "improved": True,
    }
    receipt = json.loads((tmp_path / "pass" / "RESIDENT_ARM_RESULT.json").read_text())
    assert receipt["status"] == "PASS"
    assert receipt["improvement"]["post_kld"] < receipt["improvement"]["pre_kld"]

    rails = ScoredRails(tmp_path)
    rails.post_kld = 0.25
    api = ResidentRepairAPI(rails=rails, run_root=tmp_path / "fail")
    build = api.build_uniform(
        tmp_path / "model-2", "qtip1_v7", checkpoint_sha=sha("u0")
    )
    with pytest.raises(ValueError, match="did not improve"):
        api.run_arm(build, checkpoint_sha=sha("u0"))
    receipt = json.loads((tmp_path / "fail" / "RESIDENT_ARM_RESULT.json").read_text())
    assert receipt["status"] == "FAILED"
    assert receipt["improvement"] == {
        "pre_kld": 0.25,
        "post_kld": 0.25,
        "delta_kld": 0.0,
        "improved": False,
    }
