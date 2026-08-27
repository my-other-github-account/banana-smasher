from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

from repair_api import ResidentRepairAPI
from repair_api.cli import main as cli_main
from repair_api.modern_green_resident import ModernGreenResidentEngine, _install_runtime_modules


BASIS = "98efab455cf08dfbbbaaba6f570e1bf10bf927d2b4c3c453a59c2f6f0e3be92b"
CHECKPOINT_BYTES = b"exact-u20"
CHECKPOINT_SHA = hashlib.sha256(CHECKPOINT_BYTES).hexdigest()


class FakeEngine:
    instances = []

    def __init__(self, *, payload, config, rank, layer_ranges):
        self.payload = payload
        self.config = config
        self.rank = rank
        self.layer_ranges = layer_ranges
        self.advance_calls = []
        self.validate_calls = []
        self.closed = False
        self.torch = SimpleNamespace(cuda=SimpleNamespace(memory_allocated=lambda: 1234))
        self.__class__.instances.append(self)

    def advance_to(self, target):
        self.advance_calls.append(target)
        raise AssertionError("release-only resident stage must not train")

    def validate(self, windows, teacher_root):
        self.validate_calls.append((tuple(windows), Path(teacher_root)))
        return {
            "kld_mean": 0.13712959240533734,
            "top1": 877,
            "positions": 1024,
            "runtime_counters": {
                "timed_model_payload_reads": 0,
                "timed_score_file_reads": 0,
            },
        }

    def close(self):
        self.closed = True


def test_resident_stage_release_loads_exact_state_without_training():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        (root / "checkpoints").mkdir()
        (root / "checkpoints" / "UPDATE_020.pt").write_bytes(CHECKPOINT_BYTES)
        manifest = {
            "schema": "repair-artifact-v1",
            "artifact_id": "u20-stage-test",
            "identity": {
                "basis_sha256": BASIS,
                "builder_eval_corpus_sha256": "builder",
                "train_score_corpus_sha256": "corpus",
                "teacher_inventory": "teacher",
            },
            "checkpoints": {
                "UPDATE_020": {
                    "path": "checkpoints/UPDATE_020.pt",
                    "sha256": CHECKPOINT_SHA,
                    "identity_sha256": "identity-u20",
                    "next_update": 20,
                }
            },
            "score": {
                "spec": "balanced64-v1",
                "teacher_dir": "teacher",
                "candidate_dir_template": "rows/{checkpoint}",
                "window_ids": list(range(64)),
                "positions_per_window": 1024,
                "support": 8192,
            },
        }
        (root / "ARTIFACT.json").write_text(json.dumps(manifest))
        ready = root / "READY.json"
        control = root / "CONTROL.json"
        control.write_text(json.dumps({"action": "release"}))
        payload = {
            "state": {"luts": {"a": 1}, "norms": {"b": 2}, "outputs": {"c": 3}},
            "optimizer": {"state": {"x": 1}},
            "scheduler": {"last_epoch": 4},
        }
        config = {
            "authorized_api": True,
            "world_size": 2,
            "rank": 0,
            "local_only": True,
            "trainer_source": "/trainer.py",
            "model_root": "/model",
            "asset_root": "/asset",
            "parent_root": "/parent",
            "l034_roster": "/roster.json",
            "teacher_root": "/training-teacher",
            "validation_teacher_root": "/validation-teacher",
            "corpus": "/corpus.json",
            "manifest": "/manifest.json",
            "delta_dir": "/delta",
            "vq3b_dir": "/vq3b",
            "master_addr": "192.168.200.7",
            "master_port": 29991,
            "layer_split": {"0": [0, 20], "1": [21, 42]},
            "basis_sha256": BASIS,
            "checkpoint_sha256": CHECKPOINT_SHA,
            "shared_optimizer_scheduler_lineage": "modern-green-u16-lower-lr-global-cosine-lrscale-0.125",
            "canonical_code_commit": "test-pin",
        }
        FakeEngine.instances.clear()
        with patch("repair_api.api._load_torch", return_value=payload), patch(
            "repair_api.modern_green_resident.ModernGreenResidentEngine", FakeEngine
        ):
            result = ResidentRepairAPI.open(root).stage_two_spark_real(
                "UPDATE_020", config=config, ready_path=ready, control_path=control, poll_seconds=0.01
            )
        engine = FakeEngine.instances[-1]
        assert result["status"] == "RELEASED_WITHOUT_TRAINING"
        assert engine.advance_calls == []
        assert engine.closed is True
        receipt = json.loads(ready.read_text())
        assert receipt["status"] == "RESIDENT_READY"
        assert receipt["checkpoint_sha256"] == CHECKPOINT_SHA
        assert receipt["optimizer_state_nonempty"] is True
        assert receipt["scheduler_state_nonempty"] is True
        assert receipt["state_counts"] == {"luts": 1, "norms": 1, "outputs": 1}
        assert receipt["training_launched"] is False
        assert receipt["scoring_launched"] is False

        validate_ready = root / "VALIDATE_READY.json"
        validate_control = root / "VALIDATE_CONTROL.json"
        validate_receipt = root / "VALIDATE.json"
        validate_control.write_text(json.dumps({
            "action": "validate_release",
            "windows": [28],
            "teacher_root": "/validation-teacher",
            "receipt_path": str(validate_receipt),
        }))
        with patch("repair_api.api._load_torch", return_value=payload), patch(
            "repair_api.modern_green_resident.ModernGreenResidentEngine", FakeEngine
        ):
            validated = ResidentRepairAPI.open(root).stage_two_spark_real(
                "UPDATE_020",
                config=config,
                ready_path=validate_ready,
                control_path=validate_control,
                poll_seconds=0.01,
            )
        validate_engine = FakeEngine.instances[-1]
        assert validated["status"] == "VALIDATED_AND_RELEASED_WITHOUT_TRAINING"
        assert validated["validation"]["kld_mean"] == 0.13712959240533734
        assert validate_engine.validate_calls == [((28,), Path("/validation-teacher"))]
        assert validate_engine.advance_calls == []
        assert validate_engine.closed is True
        assert json.loads(validate_receipt.read_text())["kld_mean"] == 0.13712959240533734


def test_resident_stage_cli_preserves_explicit_distributed_identity():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        config_path = root / "config.json"
        config_path.write_text(json.dumps({"rank": 1, "world_size": 2, "authorized_api": True}))
        observed = {}

        class FakeAPI:
            def stage_two_spark_real(self, checkpoint, *, config, ready_path, control_path):
                observed.update(config)
                return {"status": "RELEASED_WITHOUT_TRAINING"}

        with patch("repair_api.cli.ResidentRepairAPI.open", return_value=FakeAPI()):
            assert cli_main([
                "resident-stage", "--artifact-root", str(root), "--checkpoint", "UPDATE_020",
                "--config", str(config_path), "--ready", str(root / "ready.json"),
                "--control", str(root / "control.json"),
            ]) == 0
        assert observed["rank"] == 1
        assert observed["world_size"] == 2


def test_resident_engine_binds_base_loader_window_environment(monkeypatch):
    engine = ModernGreenResidentEngine.__new__(ModernGreenResidentEngine)
    engine.config = {
        "train_windows": [28, 56],
        "probe_windows": [68, 71],
    }
    engine.manifest_path = Path("/manifest.json")
    engine.delta_dir = Path("/delta")
    engine.vq3b_dir = Path("/vq3b")
    engine.corpus_path = Path("/corpus.json")
    engine.teacher_root = Path("/teacher")
    for key in ("BR_TRAIN", "BR_PROBE"):
        monkeypatch.delenv(key, raising=False)

    engine._configure_import_environment()

    assert __import__("os").environ["BR_TRAIN"] == "28,56"
    assert __import__("os").environ["BR_PROBE"] == "68,71"


def test_resident_engine_defaults_base_loader_windows_to_canonical_bank(monkeypatch):
    engine = ModernGreenResidentEngine.__new__(ModernGreenResidentEngine)
    engine.config = {}
    engine.manifest_path = Path("/manifest.json")
    engine.delta_dir = Path("/delta")
    engine.vq3b_dir = Path("/vq3b")
    engine.corpus_path = Path("/corpus.json")
    engine.teacher_root = Path("/teacher")
    for key in ("BR_TRAIN", "BR_PROBE"):
        monkeypatch.delenv(key, raising=False)

    engine._configure_import_environment()

    expected = ",".join(str(value) for value in range(20, 84))
    assert __import__("os").environ["BR_TRAIN"] == expected
    assert __import__("os").environ["BR_PROBE"] == expected


def test_runtime_modules_are_hash_bound_and_installed_under_trainer_names():
    import sys
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        extension = root / "compiled_name.so"
        wrapper = root / "wrapper.py"
        expert = root / "expert.py"
        extension.write_bytes(b"compiled-extension-placeholder")
        wrapper.write_text("VALUE = 41\n")
        expert.write_text("from fast_k2_grouped import VALUE\nRESULT = VALUE + 1\n")
        config = {
            "fast_k2_extension": str(extension),
            "fast_k2_extension_sha256": hashlib.sha256(extension.read_bytes()).hexdigest(),
            "fast_k2_module_name": "compiled_name",
            "fast_k2_wrapper_source": str(wrapper),
            "fast_k2_wrapper_source_sha256": hashlib.sha256(wrapper.read_bytes()).hexdigest(),
            "fast_v7_expert_source": str(expert),
            "fast_v7_expert_source_sha256": hashlib.sha256(expert.read_bytes()).hexdigest(),
        }
        try:
            _install_runtime_modules(config)
            assert sys.modules["fast_k2_grouped"].VALUE == 41
            assert sys.modules["fast_v7_expert_base"].RESULT == 42
        finally:
            for name in ("compiled_name", "fast_k2_grouped", "fast_v7_expert_base"):
                sys.modules.pop(name, None)


def test_runtime_expert_accepts_sealed_historical_constructor_via_model_limit():
    import sys
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        model_root = root / "model"
        model_root.mkdir()
        (model_root / "config.json").write_text(json.dumps({"swiglu_limit": 10.0}))
        extension = root / "compiled_name.so"
        wrapper = root / "wrapper.py"
        expert = root / "expert.py"
        extension.write_bytes(b"compiled-extension-placeholder")
        wrapper.write_text("VALUE = 41\n")
        expert.write_text(
            "class FullyResidentGroupedV7Experts:\n"
            "    def __init__(self, layer, pilot=True, *, plane_source, swiglu_limit):\n"
            "        self.limit = float(swiglu_limit)\n"
        )
        config = {
            "model_root": str(model_root),
            "fast_k2_extension": str(extension),
            "fast_k2_extension_sha256": hashlib.sha256(extension.read_bytes()).hexdigest(),
            "fast_k2_module_name": "compiled_name",
            "fast_k2_wrapper_source": str(wrapper),
            "fast_k2_wrapper_source_sha256": hashlib.sha256(wrapper.read_bytes()).hexdigest(),
            "fast_v7_expert_source": str(expert),
            "fast_v7_expert_source_sha256": hashlib.sha256(expert.read_bytes()).hexdigest(),
        }
        try:
            _install_runtime_modules(config)
            cls = sys.modules["fast_v7_expert_base"].FullyResidentGroupedV7Experts
            resident = cls(layer=0, pilot=True, plane_source=object())
            assert resident.limit == 10.0
        finally:
            for name in ("compiled_name", "fast_k2_grouped", "fast_v7_expert_base"):
                sys.modules.pop(name, None)
