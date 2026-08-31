import hashlib
import json
from pathlib import Path
import sys

from repair_api import modern_green_resident, sealed_pre_forward


class _WriteTrackingConfig(dict):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.writes: list[str] = []

    def __setitem__(self, key, value) -> None:
        self.writes.append(key)
        super().__setitem__(key, value)


def test_sealed_pre_binding_preserves_explicit_accepted_static_provider(monkeypatch) -> None:
    monkeypatch.setattr(sealed_pre_forward, "source_binding", lambda: {"status": "PASS"})
    config = _WriteTrackingConfig(
        resident_validation_expert_implementation="accepted_static_w28",
        recipe_id=modern_green_resident.PUBLISHED_PRE_RECIPE_ID,
        resident_validation_proof=True,
    )

    sealed_pre_forward.bind_sealed_pre_resident_config(config)

    assert "resident_validation_expert_implementation" not in config.writes
    resolved = modern_green_resident._resolve_runtime_provider_files(config)
    assert resolved["wrapper_sha256"] == (
        "ec681dd1ac35d5c4368071db12c8bb0801cbf78c3677c51ef9a56d0cacdf3454"
    )
    assert resolved["expert_sha256"] == (
        "4ba1411601b186dd0d6a3a89c829320f1b50e3112a40db40034e9fbadfb5d552"
    )


def test_sealed_pre_w28_keeps_target_producer_mb2_geometry(monkeypatch) -> None:
    binding = {
        "status": "PASS",
        "known_value_fixture": {
            "window": 28,
            "kld_mean": 0.1364830042977786,
            "top1": 880,
        },
    }
    monkeypatch.setattr(sealed_pre_forward, "source_binding", lambda: binding)
    monkeypatch.setattr(modern_green_resident, "_uses_static_w28_provider", lambda config: True)
    config = {"resident_validation_expert_implementation": "accepted_static_w28"}

    observed = sealed_pre_forward.bind_sealed_pre_resident_config(config)

    assert observed is binding
    assert config["provider_resolution_mode"] == "STATIC_W28_GROUPED"
    assert config["resident_validation_expert_implementation"] == "accepted_static_w28"
    assert config["score_window_batch_size"] == 2
    assert config["sealed_builder_window_microbatch"] == 2

    resolved = modern_green_resident._resolve_runtime_provider_files(config)
    assert resolved["wrapper_path"].name == "static_w28_fast_k2_grouped.py"
    assert resolved["wrapper_sha256"] == modern_green_resident.STATIC_W28_GROUPED_WRAPPER_SHA256
    assert resolved["expert_path"].name == "static_w28_fast_v7_expert_base.py"
    assert resolved["expert_sha256"] == modern_green_resident.STATIC_W28_GROUPED_EXPERT_SHA256
    trainer = Path(modern_green_resident.__file__).parent / "assets" / "static_w28_modern_green_clean_u0.py"
    assert hashlib.sha256(trainer.read_bytes()).hexdigest() == "126c11f306a12ed35c1234bd12952a32662c3bd81fc2e74361f0a55ebdc21fc0"


def test_u20_continuation_resolves_commit_owned_serial_provider_descendant() -> None:
    config = {
        "recipe_id": modern_green_resident.PUBLISHED_PRE_RECIPE_ID,
        "resident_validation_proof": True,
        "fast_k2_wrapper_source_sha256": (
            "fb8f66b20f3fa61b9304d5f874d90c7e6a5c55149bfaa44e7784d6683cbd67ef"
        ),
        "fast_v7_expert_source_sha256": (
            "0b673aaa31dedaaf604488bb71543e92560167cdef7e6bade50b65b4568b9f81"
        ),
    }

    resolved = modern_green_resident._resolve_runtime_provider_files(config)

    assert resolved["wrapper_path"].parts[-2:] == (
        "u20_resident_provider",
        "fast_k2_grouped.py",
    )
    assert resolved["expert_path"].parts[-2:] == (
        "u20_resident_provider",
        "fast_v7_expert_base.py",
    )
    assert resolved["wrapper_sha256"] == hashlib.sha256(
        resolved["wrapper_path"].read_bytes()
    ).hexdigest()
    assert resolved["expert_sha256"] == hashlib.sha256(
        resolved["expert_path"].read_bytes()
    ).hexdigest()
    expert_source = resolved["expert_path"].read_text()
    assert "one live expert workspace" in expert_source
    assert "torch.cuda.Stream" not in expert_source


def test_sealed_pre_binding_preserves_explicit_singleton_geometry(monkeypatch) -> None:
    monkeypatch.setattr(sealed_pre_forward, "source_binding", lambda: {"status": "PASS"})
    config = {
        "score_window_batch_size": 1,
        "sealed_builder_window_microbatch": 1,
    }

    sealed_pre_forward.bind_sealed_pre_resident_config(config)

    assert config["score_window_batch_size"] == 1
    assert config["sealed_builder_window_microbatch"] == 1


def test_builder_uses_configured_paired_geometry(monkeypatch, tmp_path) -> None:
    observed = {}

    class Builder:
        __file__ = __file__

        @staticmethod
        def main():
            observed["argv"] = list(sys.argv)
            return 0

    monkeypatch.setattr(sealed_pre_forward, "_score_outputs", lambda *args: [])
    config = {
        "rank": 1,
        "validation_teacher_root": str(tmp_path / "teacher"),
        "validation_corpus": str(tmp_path / "corpus.json"),
        "sealed_builder_window_microbatch": 2,
        "sealed_builder_chunk": 64,
        "sealed_pre_use_local_model": True,
        "model_root": str(tmp_path / "warm-model"),
    }
    sealed_pre_forward._run_builder(
        Builder, root=tmp_path, config=config, windows=(28, 56), label="PAIR"
    )
    argv = observed["argv"]
    assert argv[argv.index("--mb") + 1] == "2"
    assert argv[argv.index("--chunk") + 1] == "64"
    assert argv[argv.index("--windows") + 1] == "28,56"
    assert argv[argv.index("--local-dir") + 1] == str(tmp_path / "warm-model")
    assert argv[argv.index("--meta-dir") + 1] == str(tmp_path / "warm-model")


def test_static_w28_acceptance_is_paired_and_never_full64(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(sealed_pre_forward, "bind_sealed_pre_resident_config", lambda config: {"status": "PASS"})
    progress = tmp_path / "progress.json"
    progress.write_text(json.dumps({
        "runtime_counters": {
            "layer_decode_seconds": [
                {"layer": layer, "seconds": 1.0} for layer in range(43)
            ]
        }
    }))
    planesource = type("PlaneSourceModule", (), {"PROGRESS": progress})()
    monkeypatch.setattr(
        sealed_pre_forward,
        "_prepare_exact_modules",
        lambda **kwargs: (object(), planesource, {}),
    )
    seen = {}

    def run_builder(*args, **kwargs):
        seen.update(kwargs)
        return ([
            {"window": 28, "kld_mean": sealed_pre_forward.W28_KLD, "top1": sealed_pre_forward.W28_TOP1},
            {"window": 56, "kld_mean": 0.0, "top1": 0},
        ], 1.0)

    monkeypatch.setattr(sealed_pre_forward, "_run_builder", run_builder)
    receipt = sealed_pre_forward.run_static_w28_acceptance(
        task="t_test", rank=0, root=tmp_path, config={}, checkpoint=tmp_path / "pre.pt",
        canonical_pin="deadbeef",
    )
    assert seen["windows"] == (28, 56)
    assert receipt["producer"] == {"mode": "planes", "mb": 2, "chunk": 64, "windows": [28, 56]}
    assert receipt["resident_measurement"] == {
        "budget_seconds": 300.0,
        "layer_count": 43,
        "seconds": 43.0,
    }
    assert receipt["static_measurement_budget_seconds"] == 900.0
    assert receipt["full64_launched"] is False
    assert receipt["status"] == "PASS"


def test_static_w28_acceptance_is_a_public_api_path() -> None:
    import repair_api

    assert repair_api.run_static_w28_acceptance is sealed_pre_forward.run_static_w28_acceptance
    assert "run_static_w28_acceptance" in repair_api.__all__


def test_l034_binding_returns_verified_paths_to_batched_decoder(tmp_path) -> None:
    selected = tmp_path / "E000_w1.bin"
    selected.write_bytes(b"selected wire")
    selected_sha = hashlib.sha256(selected.read_bytes()).hexdigest()
    members = []
    for expert in range(256):
        for projection in ("w1", "w2", "w3"):
            members.append({
                "expert": expert,
                "projection": projection,
                "path": selected.name if (expert, projection) == (0, "w1") else "unused",
                "bytes": selected.stat().st_size,
                "sha256": selected_sha,
            })
    roster = tmp_path / "roster.json"
    roster.write_text(json.dumps({
        "basis_sha256": sealed_pre_forward.BASIS_SHA256,
        "member_count": 768,
        "members": members,
    }))

    class PlaneSource:
        def __init__(self) -> None:
            self.counters = {
                "compact_layers_touched": [],
                "local_staged_layers": [],
                "local_staged_count": 0,
            }

        def _write_progress(self, **kwargs) -> None:
            pass

        def _decode(self, path, projection):
            raise AssertionError("L034 binder must not decode before _decode_batch")

    planesource = type(
        "PlaneSourceModule",
        (),
        {"PlaneSource": PlaneSource, "candidate_lut": staticmethod(lambda layer: layer)},
    )
    sealed_pre_forward._bind_l034(planesource, roster)

    read = PlaneSource()._load_complete34()

    assert read(0, "w1") == selected


def test_public_planes_predecode_uses_vectorized_batched_k2_and_is_hash_bound() -> None:
    binding = sealed_pre_forward.source_binding()
    source = Path(binding["planesource_path"]).read_text()

    assert binding["status"] == "PASS"
    assert "def _predecode_layer(self, read):" in source
    assert "from repair_api.batched_k2 import" in source
    assert "decode_k2_matrix_batched" in source
    assert "inverse_transform_batched" in source
    assert "batch_size = 64" in source
