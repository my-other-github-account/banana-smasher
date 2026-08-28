from repair_api import sealed_pre_forward


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
    config = {"resident_validation_expert_implementation": "accepted_static_w28"}

    observed = sealed_pre_forward.bind_sealed_pre_resident_config(config)

    assert observed is binding
    assert config["provider_resolution_mode"] == "SEALED_BF16_FULL_WEIGHT"
    assert config["resident_validation_expert_implementation"] == "sealed_bf16_full_weight"
    assert config["score_window_batch_size"] == 2
    assert config["sealed_builder_window_microbatch"] == 2
