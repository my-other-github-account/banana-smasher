import hashlib
from pathlib import Path

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
        "782665057b122b42937542bcbb32aea51907f6f71d11a3782dd239859c3aef45"
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
    assert Path(config["fast_k2_wrapper_source"]).name == "static_w28_fast_k2_grouped.py"
    assert config["fast_k2_wrapper_source_sha256"] == (
        modern_green_resident.STATIC_W28_GROUPED_WRAPPER_SHA256
    )
    assert Path(config["resident_expert_source"]).name == "static_w28_fast_v7_expert_base.py"
    assert config["resident_expert_source_sha256"] == (
        modern_green_resident.STATIC_W28_GROUPED_EXPERT_SHA256
    )

    resolved = modern_green_resident._resolve_runtime_provider_files(config)
    assert resolved["wrapper_path"].name == "static_w28_fast_k2_grouped.py"
    assert resolved["wrapper_sha256"] == modern_green_resident.STATIC_W28_GROUPED_WRAPPER_SHA256
    assert resolved["expert_path"].name == "static_w28_fast_v7_expert_base.py"
    assert resolved["expert_sha256"] == modern_green_resident.STATIC_W28_GROUPED_EXPERT_SHA256
    trainer = Path(modern_green_resident.__file__).parent / "assets" / "static_w28_modern_green_clean_u0.py"
    assert hashlib.sha256(trainer.read_bytes()).hexdigest() == "af10284d12dc391b81d00978993d2a8680547c7d2fbc7f261ae411dbaf3e43b4"
    trainer_source = trainer.read_text()
    assert '"wire/E{expert:03d}/{projection}.q2v7wire"' in trainer_source
    assert '"wire/E{expert:03d}/{projection}.k2wire"' in trainer_source
    assert "wire_templates = active_wire_templates(root)" in trainer_source


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
