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
        "64403d3e9b9761c3fcc636ba24d4d65c635f57675c1f749af312d441d55407c4"
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
    assert hashlib.sha256(trainer.read_bytes()).hexdigest() == "a55c2f5104b8d9dd06d845684d168be6f6e9dae637bac08443bd6ddbaf94201a"
