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
        self.binding = production_rails._ArtifactBinding(
            identity_sha256=self.binding.identity_sha256,
            basis_sha256=self.binding.basis_sha256,
            checkpoint="UPDATE_004",
            score_checkpoints={"post": "UPDATE_004"},
            artifact_manifest_sha256=self.binding.artifact_manifest_sha256,
            checkpoint_sha256=_sha("checkpoint-4"),
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


def test_default_provider_reuses_one_physical_engine_and_scores_trained_state(
    tmp_path, monkeypatch
):
    class FakeEngine:
        constructions = 0

        def __init__(self):
            type(self).constructions += 1
            self.update = 0

        def score_balanced64(self, windows):
            return {
                "mean_kld": 0.25,
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
            self.artifact = type(
                "Artifact",
                (),
                {"windows": tuple(range(64)), "manifest": {"checkpoints": {"UPDATE_000": {"next_update": 0}}}},
            )()

        def advance_resident_engine(self, engine, start_checkpoint, target_update, **kwargs):
            del start_checkpoint
            self.advance_kwargs = kwargs
            engine.update = target_update
            return {
                "updates": target_update,
                "checkpoint": f"UPDATE_{target_update:03d}",
                "checkpoint_sha256": _sha(f"checkpoint-{target_update}"),
            }

    fake_api = FakeProvenAPI()
    FakeEngine.constructions = 0
    monkeypatch.setattr(production_rails._ProvenResidentAPI, "open", lambda root: fake_api)
    monkeypatch.setattr(
        production_rails,
        "_construct_resident_engine",
        lambda api, binding, config: FakeEngine(),
    )
    monkeypatch.setattr(
        ProductionRails, "_require_live_checkpoint_bytes", staticmethod(lambda *args: None)
    )
    config = _base_config()
    config["continuation"] = {"authorized_api": True, "world_size": 2, "rank": 0}
    artifact = _artifact(tmp_path / "artifact", _binding_sha(config))
    _admit(config, artifact)
    rails = ProductionRails(config, run_root=tmp_path / "run")
    facade = ResidentRepairAPI(rails=rails, run_root=tmp_path / "facade")

    pre = facade.score_pre(artifact, checkpoint_sha=artifact.checkpoint_sha256)
    trained = facade.repair_train(
        artifact, updates=4, checkpoint_sha=artifact.checkpoint_sha256
    )
    post = facade.score_post(artifact, checkpoint_sha=artifact.checkpoint_sha256)

    assert FakeEngine.constructions == 1
    assert pre["checkpoint"] == "UPDATE_000"
    assert trained["checkpoint"] == "UPDATE_004"
    assert post["checkpoint"] == "UPDATE_004"
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
    assert str(fake_api.advance_kwargs["loss_guard_receipt_path"]).endswith(
        "CONTINUATION_U000_U004.rank0.LOSS_GUARD.json"
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


def test_lifecycle_is_not_pass_before_complete_and_training_is_exactly_four(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(production_rails, "_ProvenSession", FixtureSession)
    config = _base_config()
    artifact = _artifact(tmp_path / "artifact", _binding_sha(config))
    _admit(config, artifact)
    rails = ProductionRails(config, run_root=tmp_path / "run")
    assert json.loads(rails.lifecycle_path.read_text())["status"] == "IN_PROGRESS"
    rails.load_resident(artifact)
    with pytest.raises(ProductionRailsError, match="exactly four"):
        rails.train(artifact, 3)


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
