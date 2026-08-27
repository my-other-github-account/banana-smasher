from pathlib import Path
from types import SimpleNamespace

from repair_api.modern_green_resident import ModernGreenResidentEngine


def test_accepted_static_w28_preserves_producer_full_weight_bf16(monkeypatch) -> None:
    import repair_api.official_k2_resident_score as sealed_builder_binding

    monkeypatch.setattr(
        sealed_builder_binding,
        "_configured_attention_implementation",
        lambda _config: "eager",
    )
    for name in (
        "FAST_K2_SEALED_FULL_WEIGHT_BF16",
        "FAST_K2_SEALED_PROJECTION_BF16",
        "FAST_K2_SEALED_NO_SWIGLU_CLAMP",
    ):
        monkeypatch.delenv(name, raising=False)

    engine = ModernGreenResidentEngine.__new__(ModernGreenResidentEngine)
    engine.config = {
        "resident_validation_proof": True,
        "recipe_id": "published_pre_lower_lr_warmup16_cosine64_v1",
        "resident_validation_expert_implementation": "accepted_static_w28",
    }
    engine.published_pre_recipe = True
    engine.corpus_path = Path("corpus")
    engine.teacher_root = Path("teacher")
    engine.model_root = Path("model")
    engine.base = SimpleNamespace(T=SimpleNamespace(CKPT=None, DEV=None))
    engine.torch = SimpleNamespace(
        manual_seed=lambda _seed: None,
        cuda=SimpleNamespace(manual_seed_all=lambda _seed: None),
    )

    engine._configure_base()

    assert __import__("os").environ["FAST_K2_SEALED_FULL_WEIGHT_BF16"] == "1"
    assert __import__("os").environ["FAST_K2_SEALED_PROJECTION_BF16"] == "0"
    assert engine.sealed_builder_binding["provider_expert_sha256"] == (
        "64403d3e9b9761c3fcc636ba24d4d65c635f57675c1f749af312d441d55407c4"
    )
