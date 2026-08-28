from repair_api import modern_green_resident, sealed_pre_forward


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
