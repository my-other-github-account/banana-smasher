from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

import banana_smasher.production_rails as production_rails
from banana_smasher.artifact_identity import ArtifactIdentity
from banana_smasher.cli import _parser
from banana_smasher.production_rails import (
    ALL_LAYERS,
    PIPELINE_MICROBATCH,
    PRODUCTION_RAILS_SCHEMA,
    ProductionRails,
    ProductionRailsError,
)
from banana_smasher.resident_balanced64 import ArtifactError, RepairArtifact
from banana_smasher.resident_continuation import _window_microbatches
from banana_smasher.resident_repair_api import BackpackArtifact, ResidentRepairAPI


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def fixture_builder(**kwargs):
    return kwargs["output"]


def fixture_mixer(**kwargs):
    return kwargs["output"]


def test_heldout_gate_kills_after_two_flat_or_rising_boundaries() -> None:
    first = production_rails._heldout_decision(0.229, 0.220, 0, patience=2)
    assert first == {"improved": True, "non_improving_streak": 0, "halt": False}
    second = production_rails._heldout_decision(0.220, 0.220, 0, patience=2)
    assert second == {"improved": False, "non_improving_streak": 1, "halt": False}
    third = production_rails._heldout_decision(0.220, 0.221, 1, patience=2)
    assert third == {"improved": False, "non_improving_streak": 2, "halt": True}


class FixtureSession:
    constructions = 0

    def __init__(self, artifact, binding, **kwargs):
        del artifact, kwargs
        type(self).constructions += 1
        self.swaps = 0
        self.binding = binding

    def hot_swap(self, artifact, binding):
        del artifact, binding
        self.swaps += 1

    def score(self, phase):
        return {
            "mean_kld": 0.25,
            "top1_matches": 7,
            "positions": 64 * 1024,
            "phase": phase,
            "checkpoint": self.binding.checkpoint,
            "timed_wall_seconds": 0.01,
            "execution_mode": "resident_model_in_memory",
            "runtime_counters": {
                "checkpoint_loads_during_score": 0,
                "candidate_file_reads_during_score": 0,
                "windows": 64,
            },
        }

    def train(self, updates):
        current = int(self.binding.checkpoint.rsplit("_", 1)[1])
        target = current + updates
        checkpoint = f"UPDATE_{target:03d}"
        self.binding = production_rails._ArtifactBinding(
            identity_sha256=self.binding.identity_sha256,
            basis_sha256=self.binding.basis_sha256,
            checkpoint=checkpoint,
            score_checkpoints={"post": checkpoint},
            artifact_manifest_sha256=self.binding.artifact_manifest_sha256,
            checkpoint_sha256=_sha(f"checkpoint-{target}"),
        )
        return {"updates": updates}


def _base_config() -> dict:
    return {
        "schema": PRODUCTION_RAILS_SCHEMA,
        "pipeline_microbatch": PIPELINE_MICROBATCH,
        "layers": list(ALL_LAYERS),
        "uniform_builder": "test_production_rails:fixture_builder",
        "backpack_mixer": "test_production_rails:fixture_mixer",

        "allowed_artifacts": {
            _sha("placeholder"): {
                "basis_sha256": _sha("basis"),
                "checkpoint": "UPDATE_000",
                "artifact_manifest_sha256": _sha("manifest"),
                "checkpoint_sha256": _sha("checkpoint"),
            }
        },
    }


def _binding_sha(config: dict) -> str:
    fields = {
        key: config.get(key)
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
    return hashlib.sha256(
        json.dumps(fields, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _artifact(root: Path, provider_binding_sha256: str) -> BackpackArtifact:
    root.mkdir(parents=True, exist_ok=True)
    (root / "checkpoints").mkdir()
    checkpoint = root / "checkpoints" / "UPDATE_000.pt"
    checkpoint.write_bytes(b"resident-checkpoint")
    checkpoint_sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    manifest = {
        "schema": "repair-artifact-v1",
        "score": {
            "spec": "balanced64-v1",
            "positions_per_window": 1024,
            "support": 8192,
            "window_ids": list(range(64)),
            "teacher_dir": "teacher",
            "candidate_dir_template": "candidates/{checkpoint}",
        },
        "checkpoints": {
            "UPDATE_000": {
                "path": "checkpoints/UPDATE_000.pt",
                "sha256": checkpoint_sha,
                "identity_sha256": _sha("checkpoint-identity"),
                "next_update": 0,
            }
        },
    }
    (root / "ARTIFACT.json").write_text(json.dumps(manifest, sort_keys=True))
    document = {
        "schema": "banana-smasher-artifact-identity-v1",
        "basis": {"model_index_sha256": _sha("basis")},
        "corpora": {
            "builder_eval_sha256": _sha("builder"),
            "train_score_sha256": _sha("score"),
            "u0_lock_sha256": _sha("lock"),
            "teacher_inventory_sha256": _sha("teacher"),
        },
        "checkpoints": {
            "u0": {"sha256": checkpoint_sha, "identity_sha256": _sha("u0-id")}
        },
        "composition": {
            "kind": "mixed-qtip-v7-backpack",
            "layers": [
                {"layer": layer, "tiers": {"qtip2_v7": 512}}
                for layer in ALL_LAYERS
            ],
        },
        "canary": {
            "reference": {"kld": 0.25, "top1": 7},
            "tolerance": {"kld_abs": 0.0, "top1_abs": 0},
        },
        "runtime": {
            "production_rails": {
                "provider_binding_sha256": provider_binding_sha256
            }
        },
    }
    (root / "identity.json").write_text(json.dumps(document, sort_keys=True))
    return BackpackArtifact(
        root=root,
        identity=ArtifactIdentity.load(root),
        checkpoint_sha256=checkpoint_sha,
    )


def _admit(config: dict, artifact: BackpackArtifact) -> None:
    manifest = artifact.root / "ARTIFACT.json"
    checkpoint = artifact.root / "checkpoints" / "UPDATE_000.pt"
    config["allowed_artifacts"] = {
        artifact.identity.sha256: {
            "basis_sha256": artifact.identity.basis_sha256,
            "checkpoint": "UPDATE_000",
            "artifact_manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
            "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        }
    }


def test_production_rails_one_construction_across_score_updates_swap_and_post(
    tmp_path, monkeypatch
):
    FixtureSession.constructions = 0
    monkeypatch.setattr(production_rails, "_ProvenSession", FixtureSession)
    monkeypatch.setattr(
        ProductionRails, "_require_live_checkpoint_bytes", staticmethod(lambda *args: None)
    )
    config = _base_config()
    artifact = _artifact(tmp_path / "artifact", _binding_sha(config))
    _admit(config, artifact)
    rails = ProductionRails(
        config,
        run_root=tmp_path / "run",
    )
    # Exercise the public phase surface with the already mixed pinned artifact.
    facade = ResidentRepairAPI(rails=rails, run_root=tmp_path / "facade")
    pre = facade.score_pre(artifact, checkpoint_sha=artifact.checkpoint_sha256)
    trained = facade.repair_train(
        artifact, updates=4, checkpoint_sha=artifact.checkpoint_sha256
    )
    post = facade.score_post(artifact, checkpoint_sha=artifact.checkpoint_sha256)

    lifecycle = json.loads((tmp_path / "run" / "RESIDENT_LIFECYCLE.json").read_text())
    assert FixtureSession.constructions == 1
    assert pre["phase"] == "pre" and post["phase"] == "post"
    assert trained["updates"] == 4
    for phase in ("pre", "post"):
        attempt = json.loads(
            (tmp_path / "run" / f"RESIDENT_SCORE_ATTEMPT.{phase}.json").read_text()
        )
        assert attempt["status"] == "MEASURED_UNACCEPTED"
        assert attempt["mean_kld"] == 0.25
        assert attempt["positions"] == 64 * 1024
        assert attempt["runtime_counters"]["candidate_file_reads_during_score"] == 0
    assert lifecycle["counts"] == {
        "model_constructions": 1,
        "resident_loads": 1,
        "hot_swaps": 2,
        "scores": 2,
        "canary_passes": 2,
        "training_calls": 1,
        "updates": 4,
    }
    assert [row["event"] for row in lifecycle["events"]][-5:] == [
        "score_published",
        "checkpoint_hot_swap",
        "resident_training_complete",
        "checkpoint_hot_swap",
        "score_published",
    ]


def test_isolated_process_restore_runs_pre_train_post_without_shared_session(
    tmp_path, monkeypatch
):
    FixtureSession.constructions = 0
    monkeypatch.setattr(production_rails, "_ProvenSession", FixtureSession)
    monkeypatch.setattr(
        ProductionRails, "_require_live_checkpoint_bytes", staticmethod(lambda *args: None)
    )
    config = _base_config()
    artifact = _artifact(tmp_path / "artifact", _binding_sha(config))
    _admit(config, artifact)

    pre_rails = ProductionRails(config, run_root=tmp_path / "pre-run")
    pre_api = ResidentRepairAPI(rails=pre_rails, run_root=tmp_path / "pre-facade")
    pre = pre_api.score_pre(artifact, checkpoint_sha=artifact.checkpoint_sha256)

    trained_checkpoint = artifact.root / "checkpoints" / "UPDATE_004.pt"
    trained_checkpoint.write_bytes(b"preserved-u4-checkpoint")
    trained_sha = hashlib.sha256(trained_checkpoint.read_bytes()).hexdigest()
    manifest_path = artifact.root / "ARTIFACT.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["checkpoints"]["UPDATE_004"] = {
        "path": "checkpoints/UPDATE_004.pt",
        "sha256": trained_sha,
        "identity_sha256": _sha("checkpoint-4-identity"),
        "parent_sha256": artifact.checkpoint_sha256,
        "next_update": 4,
    }
    manifest_path.write_text(json.dumps(manifest, sort_keys=True))

    repeat_pre_rails = ProductionRails(config, run_root=tmp_path / "repeat-pre-run")
    repeat_pre_api = ResidentRepairAPI(
        rails=repeat_pre_rails, run_root=tmp_path / "repeat-pre-facade"
    )
    assert repeat_pre_api.score_pre(
        artifact, checkpoint_sha=artifact.checkpoint_sha256
    )["phase"] == "pre"

    train_rails = ProductionRails(config, run_root=tmp_path / "train-run")
    train_api = ResidentRepairAPI(rails=train_rails, run_root=tmp_path / "train-facade")
    train_api.restore_pre_score(pre, artifact, checkpoint_sha=artifact.checkpoint_sha256)
    assert train_rails._active_binding.checkpoint == "UPDATE_004"
    training = dict(train_api.repair_train(
        artifact, updates=4, checkpoint_sha=artifact.checkpoint_sha256
    ))
    assert train_rails._active_binding.checkpoint == "UPDATE_008"

    trained_checkpoint = artifact.root / "checkpoints" / "UPDATE_008.pt"
    trained_checkpoint.write_bytes(b"trained-checkpoint")
    trained_sha = hashlib.sha256(trained_checkpoint.read_bytes()).hexdigest()
    training.update(
        {
            "checkpoint": "UPDATE_008",
            "checkpoint_sha256": trained_sha,
        }
    )
    manifest = json.loads(manifest_path.read_text())
    manifest["checkpoints"]["UPDATE_008"] = {
        "path": "checkpoints/UPDATE_008.pt",
        "sha256": trained_sha,
        "identity_sha256": _sha("checkpoint-8-identity"),
        "parent_sha256": manifest["checkpoints"]["UPDATE_004"]["sha256"],
        "next_update": 8,
    }
    manifest_path.write_text(json.dumps(manifest, sort_keys=True))

    post_rails = ProductionRails(config, run_root=tmp_path / "post-run")
    post_api = ResidentRepairAPI(rails=post_rails, run_root=tmp_path / "post-facade")
    post_api.restore_training(
        pre, training, artifact, checkpoint_sha=artifact.checkpoint_sha256
    )
    post = post_api.score_post(artifact, checkpoint_sha=artifact.checkpoint_sha256)

    assert post["phase"] == "post"
    assert FixtureSession.constructions == 4


def test_unknown_artifact_and_geometry_drift_fail_closed(tmp_path):
    config = _base_config()
    artifact = _artifact(tmp_path / "artifact", _binding_sha(config))
    rails = ProductionRails(config, run_root=tmp_path / "run")
    with pytest.raises(ProductionRailsError, match="unknown artifact identity"):
        rails.load_resident(artifact)

    config["pipeline_microbatch"] = 2
    with pytest.raises(ProductionRailsError, match="PIPELINE_MICROBATCH=4"):
        ProductionRails(config, run_root=tmp_path / "bad-microbatch")
    config["pipeline_microbatch"] = 4
    config["layers"] = list(range(42))
    with pytest.raises(ProductionRailsError, match="layers 0..42"):
        ProductionRails(config, run_root=tmp_path / "bad-layers")


def test_one_window_resident_score_matches_sealed_fixture_oracle(tmp_path):
    root = tmp_path / "repair"
    (root / "checkpoints").mkdir(parents=True)
    (root / "teacher").mkdir()
    (root / "candidates" / "UPDATE_000").mkdir(parents=True)
    checkpoint = root / "checkpoints" / "UPDATE_000.pt"
    checkpoint.write_bytes(b"sealed-fixture")
    (root / "teacher" / "t8192_win0.pt").write_bytes(b"teacher")
    (root / "candidates" / "UPDATE_000" / "q8192_win0.pt").write_bytes(b"candidate")
    manifest = {
        "schema": "repair-artifact-v1",
        "score": {
            "spec": "balanced64-v1",
            "positions_per_window": 1024,
            "support": 8192,
            "window_ids": list(range(64)),
            "teacher_dir": "teacher",
            "candidate_dir_template": "candidates/{checkpoint}",
        },
        "checkpoints": {
            "UPDATE_000": {
                "path": "checkpoints/UPDATE_000.pt",
                "sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            }
        },
    }
    (root / "ARTIFACT.json").write_text(json.dumps(manifest))
    ref = np.zeros((1024, 8192), dtype=np.float16)
    idx = np.zeros((1024, 8192), dtype=np.int64)
    argmax = np.zeros((1024,), dtype=np.int64)

    def loader(path: Path):
        if path.name.startswith("t8192"):
            return {"idx": idx, "logprob": ref}
        return {"q_lp_at_ref": ref, "q_argmax": argmax}

    artifact = RepairArtifact.open(root)
    file_result = artifact.score("UPDATE_000", windows=(0,), loader=loader)
    resident_result = artifact.score_in_memory("UPDATE_000", windows=(0,), loader=loader)
    assert file_result.kld == resident_result.kld == 0.0
    assert file_result.top1 == resident_result.top1 == 1024
    assert file_result.positions == resident_result.positions == 1024
    assert resident_result.execution_mode == "resident_in_memory"


def test_cli_exposes_only_one_process_resident_arm():
    parser = _parser()
    args = parser.parse_args(
        [
            "resident",
            "arm",
            "--artifact-root",
            "/artifact",
            "--rails-config",
            "/rails.json",
            "--run-root",
            "/run",
            "--checkpoint-sha",
            "a" * 64,
        ]
    )
    assert args.command == "resident"
    assert args.resident_command == "arm"
    assert args.updates == 4
    assert args.checkpoint_sha == "a" * 64

    improve = parser.parse_args(
        [
            "resident",
            "improve",
            "--artifact-root",
            "/artifact",
            "--run-root",
            "/run",
            "--checkpoint",
            "/artifact/checkpoints/UPDATE_000.pt",
            "--checkpoint-sha",
            "b" * 64,
        ]
    )
    assert improve.resident_command == "improve"
    assert improve.checkpoint == Path("/artifact/checkpoints/UPDATE_000.pt")
    assert improve.checkpoint_sha == "b" * 64
    assert not hasattr(improve, "rails_config")


def test_continuation_geometry_is_sealed_to_pipeline_microbatch_four():
    assert _window_microbatches({}, 16) == [[20, 21, 22, 23]]
    assert _window_microbatches(
        {"windows_per_update": 16, "pipeline_microbatch": 4}, 16
    ) == [
        [20, 21, 22, 23],
        [24, 25, 26, 27],
        [28, 29, 30, 31],
        [32, 33, 34, 35],
    ]
    with pytest.raises(ArtifactError, match="geometry"):
        _window_microbatches(
            {"windows_per_update": 16, "pipeline_microbatch": 2}, 16
        )


def test_artifact_admission_binds_manifest_and_checkpoint_bytes(tmp_path):
    config = _base_config()
    artifact = _artifact(tmp_path / "artifact", _binding_sha(config))
    _admit(config, artifact)
    rails = ProductionRails(config, run_root=tmp_path / "run")

    (artifact.root / "checkpoints" / "UPDATE_000.pt").write_bytes(b"retargeted")
    with pytest.raises(ProductionRailsError, match="checkpoint bytes"):
        rails.load_resident(artifact)

    artifact = _artifact(tmp_path / "artifact-2", _binding_sha(config))
    _admit(config, artifact)
    rails = ProductionRails(config, run_root=tmp_path / "run-2")
    manifest = artifact.root / "ARTIFACT.json"
    manifest.write_text(manifest.read_text() + "\n")
    with pytest.raises(ProductionRailsError, match="ARTIFACT.json bytes"):
        rails.load_resident(artifact)


def test_public_session_defaults_checkpoint_lut_materialization_inside_run_root(
    tmp_path, monkeypatch
):
    observed = {}

    class FakeProvenAPI:
        artifact = type(
            "Artifact",
            (),
            {"windows": tuple(range(64)), "manifest": {"checkpoints": {"UPDATE_000": {"next_update": 0}}}},
        )()

    def construct(api, binding, config):
        del api, binding
        observed.update(config)
        return object()

    monkeypatch.setattr(production_rails._ProvenResidentAPI, "open", lambda root: FakeProvenAPI())
    monkeypatch.setattr(production_rails, "_construct_resident_score_engine", construct)
    config = _base_config()
    artifact = _artifact(tmp_path / "artifact", _binding_sha(config))
    receipt_root = tmp_path / "run" / "resident-receipts"

    session = production_rails._ProvenSession(
        artifact,
        production_rails._ArtifactBinding(
            identity_sha256=artifact.identity.sha256,
            basis_sha256=artifact.identity.basis_sha256,
            checkpoint="UPDATE_000",
            score_checkpoints={"pre": "UPDATE_000"},
            artifact_manifest_sha256=hashlib.sha256((artifact.root / "ARTIFACT.json").read_bytes()).hexdigest(),
            checkpoint_sha256=artifact.checkpoint_sha256,
        ),
        continuation_config={},
        receipt_root=receipt_root,
    )

    assert session.continuation_config["checkpoint_lut_root"] == str(
        (receipt_root / "checkpoint-luts").resolve()
    )
    assert session.continuation_config["cold_start_phase_receipt"] == str(
        (receipt_root / "cold-start-phase.rank{rank}.jsonl").resolve()
    )


def test_resident_score_normalizes_authenticated_checkpoint_alias(
    tmp_path, monkeypatch
):
    class FakeEngine:
        def close(self, *, phase):
            return {"phase": phase, "post_release_allocated_bytes": 0}

        def score_balanced64(self, windows):
            return {
                "mean_kld": 0.2292069946743951,
                "top1_matches": 56534,
                "positions": len(tuple(windows)) * 1024,
                "checkpoint": "UPDATE_000",
                "timed_wall_seconds": 1.0,
                "execution_mode": "resident_model_in_memory",
                "runtime_counters": {
                    "model_constructions": 1,
                    "windows": 64,
                    "checkpoint_loads_during_score": 0,
                    "candidate_file_reads_during_score": 0,
                },
            }

    class FakeProvenAPI:
        artifact = type(
            "Artifact",
            (),
            {
                "windows": tuple(range(64)),
                "manifest": {"checkpoints": {"PRE": {"next_update": 0}}},
            },
        )()

    monkeypatch.setattr(
        production_rails._ProvenResidentAPI, "open", lambda root: FakeProvenAPI()
    )
    monkeypatch.setattr(
        production_rails,
        "_construct_resident_score_engine",
        lambda api, binding, config: FakeEngine(),
    )
    config = _base_config()
    artifact = _artifact(tmp_path / "artifact", _binding_sha(config))
    session = production_rails._ProvenSession(
        artifact,
        production_rails._ArtifactBinding(
            identity_sha256=artifact.identity.sha256,
            basis_sha256=artifact.identity.basis_sha256,
            checkpoint="PRE",
            score_checkpoints={"pre": "PRE"},
            artifact_manifest_sha256=hashlib.sha256(
                (artifact.root / "ARTIFACT.json").read_bytes()
            ).hexdigest(),
            checkpoint_sha256=artifact.checkpoint_sha256,
        ),
        continuation_config={},
        receipt_root=tmp_path / "receipts",
    )

    result = session.score("pre")

    assert result["checkpoint"] == "PRE"
    assert result["physical_checkpoint"] == "UPDATE_000"


def test_default_provider_releases_each_phase_engine_and_scores_trained_state(
    tmp_path, monkeypatch
):
    class FakeEngine:
        constructions = 0
        closures = 0

        def __init__(self, update=0):
            type(self).constructions += 1
            self.update = update

        def close(self, *, phase):
            type(self).closures += 1
            return {
                "phase": phase,
                "post_release_allocated_bytes": 0,
                "limit_bytes": 10 * 1024**3,
            }

        def score_balanced64(self, windows):
            return {
                "mean_kld": 0.25 - 0.001 * self.update,
                "top1_matches": 7,
                "positions": len(tuple(windows)) * 1024,
                "checkpoint": f"UPDATE_{self.update:03d}",
                "timed_wall_seconds": 0.01,
                "execution_mode": "resident_model_in_memory",
                "runtime_counters": {
                    "model_constructions": 1,
                    "windows": 64,
                    "checkpoint_loads_during_score": 0,
                    "candidate_file_reads_during_score": 0,
                },
            }

    class FakeProvenAPI:
        def __init__(self):
            self.advance_kwargs = None
            self.advance_targets = []
            self.artifact = type(
                "Artifact",
                (),
                {"windows": tuple(range(64)), "manifest": {"checkpoints": {"UPDATE_000": {"next_update": 0}}}},
            )()

        def advance_resident_engine(self, engine, start_checkpoint, target_update, **kwargs):
            del start_checkpoint
            self.advance_kwargs = kwargs
            self.advance_targets.append(target_update)
            engine.update = target_update
            return {
                "updates": target_update,
                "checkpoint": f"UPDATE_{target_update:03d}",
                "checkpoint_sha256": _sha(f"checkpoint-{target_update}"),
            }

    fake_api = FakeProvenAPI()
    FakeEngine.constructions = 0
    FakeEngine.closures = 0
    monkeypatch.setattr(production_rails._ProvenResidentAPI, "open", lambda root: fake_api)
    monkeypatch.setattr(
        production_rails,
        "_construct_resident_score_engine",
        lambda api, binding, config: FakeEngine(
            int(binding.checkpoint.rsplit("_", 1)[-1])
        ),
    )
    monkeypatch.setattr(
        production_rails,
        "_construct_resident_engine",
        lambda api, binding, config: FakeEngine(
            int(binding.checkpoint.rsplit("_", 1)[-1])
        ),
    )
    monkeypatch.setattr(
        ProductionRails, "_require_live_checkpoint_bytes", staticmethod(lambda *args: None)
    )
    canary_calls = []

    def record_canary(self, **kwargs):
        canary_calls.append(kwargs)

    monkeypatch.setattr(ArtifactIdentity, "require_canary", record_canary)
    monkeypatch.setattr(
        production_rails,
        "_require_distributed_pair_binding",
        lambda *args, **kwargs: None,
    )
    config = _base_config()
    config["continuation"] = {"authorized_api": True, "world_size": 2, "rank": 0}
    artifact = _artifact(tmp_path / "artifact", _binding_sha(config))
    _admit(config, artifact)
    rails = ProductionRails(config, run_root=tmp_path / "run")
    facade = ResidentRepairAPI(rails=rails, run_root=tmp_path / "facade")

    pre = facade.score_pre(artifact, checkpoint_sha=artifact.checkpoint_sha256)
    trained = facade.repair_train(
        artifact, updates=8, checkpoint_sha=artifact.checkpoint_sha256
    )
    post = facade.score_post(artifact, checkpoint_sha=artifact.checkpoint_sha256)

    assert canary_calls[0]["allow_kld_improvement"] is False
    assert canary_calls[-1]["allow_kld_improvement"] is True
    assert FakeEngine.constructions == 3
    assert FakeEngine.closures == 3
    assert pre["checkpoint"] == "UPDATE_000"
    assert trained["checkpoint"] == "UPDATE_008"
    assert post["checkpoint"] == "UPDATE_008"
    assert [row["update"] for row in trained["heldout_boundaries"]] == [4, 8]
    assert fake_api.advance_targets == list(range(1, 9))
    assert len(trained["receipts"]) == 8
    assert trained["accepted_update_cadence"] == 1
    assert fake_api.advance_kwargs["loss_guard_baseline"] == 0.25
    recipe = fake_api.advance_kwargs["config"]
    assert recipe["training_recipe"] == "u45_validated_v1"
    assert recipe["sampling_mode"] == "broad_rotation_v1"
    assert recipe["windows_per_update"] == 16
    assert recipe["pipeline_microbatch"] == 4
    assert recipe["loss_reduction_dtype"] == "float32"
    assert recipe["optimizer_moment_dtype"] == "float64"
    assert recipe["base_lrs"] == {
        "luts": 1.0e-2,
        "norms": 1.0e-4,
        "outputs": 1.0e-2,
    }
    assert recipe["lr_scale"] == 0.1
    assert recipe["heldout_validation_interval"] == 4
    assert recipe["heldout_kill_patience"] == 2
    assert recipe["accepted_update_cadence"] == 1
    # The public default must actually activate the low-memory reentrant
    # checkpoint path; otherwise unified-memory hosts retain the full layer
    # graph and can OOM before the first pipeline send.
    assert recipe["activation_checkpointing"] is True
    assert str(fake_api.advance_kwargs["loss_guard_receipt_path"]).endswith(
        "CONTINUATION_U007_U008.rank0.LOSS_GUARD.json"
    )
    lifecycle = json.loads(rails.lifecycle_path.read_text())
    assert lifecycle["counts"]["model_constructions"] == 1


def test_production_config_rejects_session_factory_bypass(tmp_path):
    config = _base_config()
    with pytest.raises(TypeError, match="session_factory"):
        ProductionRails(
            config,
            run_root=tmp_path / "run",
            session_factory=lambda **kwargs: FixtureSession(**kwargs),
        )


def test_distributed_pair_binding_rejects_different_rank_science() -> None:
    class FakeDistributed:
        @staticmethod
        def is_available():
            return True

        @staticmethod
        def is_initialized():
            return True

        @staticmethod
        def get_world_size():
            return 2

        @staticmethod
        def all_gather_object(output, value):
            output[:] = [value, {**value, "provider_binding_sha256": _sha("different-rank")}]

    with pytest.raises(ProductionRailsError, match="distributed pair scientific binding mismatch"):
        production_rails._require_distributed_pair_binding(
            "a" * 64,
            {"world_size": 2, "basis_sha256": "b" * 64},
            distributed=FakeDistributed(),
        )


@pytest.mark.parametrize(
    "field",
    (
        "fallback",
        "slow_path",
        "notification_source",
        "rate_low",
        "offline_path",
        "replay_path",
        "staged_file_path",
        "reload_path",
    ),
)
def test_production_config_rejects_notification_and_slow_path_controls(
    tmp_path, field
):
    config = _base_config()
    config[field] = "callable-or-path"
    with pytest.raises(ProductionRailsError, match=field):
        ProductionRails(config, run_root=tmp_path / field)


def test_lifecycle_is_not_pass_before_complete_and_training_accepts_positive_updates(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(production_rails, "_ProvenSession", FixtureSession)
    config = _base_config()
    artifact = _artifact(tmp_path / "artifact", _binding_sha(config))
    _admit(config, artifact)
    rails = ProductionRails(config, run_root=tmp_path / "run")
    assert json.loads(rails.lifecycle_path.read_text())["status"] == "IN_PROGRESS"
    rails.load_resident(artifact)
    with pytest.raises(ProductionRailsError, match="positive update count"):
        rails.train(artifact, 0)

    rails = ProductionRails(config, run_root=tmp_path / "run-8")
    rails.load_resident(artifact)
    rails.score(artifact, "pre")
    result = rails.train(artifact, 8)
    assert result["updates"] == 8
    assert json.loads(rails.lifecycle_path.read_text())["counts"]["updates"] == 8


def test_identity_retarget_and_stale_score_checkpoint_fail_closed(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(production_rails, "_ProvenSession", FixtureSession)
    config = _base_config()
    artifact = _artifact(tmp_path / "artifact", _binding_sha(config))
    _admit(config, artifact)
    rails = ProductionRails(config, run_root=tmp_path / "run")
    identity_path = artifact.root / "identity.json"
    identity_path.write_text(identity_path.read_text() + "\n")
    with pytest.raises(ProductionRailsError, match="identity.json bytes changed"):
        rails.load_resident(artifact)

    artifact = _artifact(tmp_path / "artifact-2", _binding_sha(config))
    _admit(config, artifact)
    rails = ProductionRails(config, run_root=tmp_path / "run-2")
    rails.load_resident(artifact)
    rails._session.score = lambda phase: {
        "mean_kld": 0.25,
        "top1_matches": 7,
        "positions": 64 * 1024,
        "checkpoint": "UPDATE_999",
        "execution_mode": "resident_model_in_memory",
        "runtime_counters": {
            "windows": 64,
            "checkpoint_loads_during_score": 0,
            "candidate_file_reads_during_score": 0,
        },
    }
    with pytest.raises(ProductionRailsError, match="physical full64"):
        rails.score(artifact, "pre")
