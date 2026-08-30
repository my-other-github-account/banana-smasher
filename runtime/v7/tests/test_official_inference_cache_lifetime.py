from pathlib import Path


def test_no_grad_fast_forward_reuses_official_dynamic_cache_across_layers():
    source = (Path(__file__).parents[1] / "runner/base_binrepair_e2e.py").read_text()
    start = source.index("def fast_forward(")
    end = source.index("\ndef loss_window(", start)
    body = source[start:end]

    create = body.index("active_cache = DynamicCache(config=config)")
    mask = body.index("past_key_values=active_cache, position_ids=pos", create)
    layer = body.index("active_cache if not requires_grad else DynamicCache(config=config)", mask)
    loop = body.index("for Li in range(", layer)
    assert create < mask < layer < loop
    assert "clear_inference_cache" not in body