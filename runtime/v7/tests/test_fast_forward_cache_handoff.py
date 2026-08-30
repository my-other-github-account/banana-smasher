from pathlib import Path


def test_no_grad_fast_forward_reuses_one_dynamic_cache_without_clearing():
    source = (Path(__file__).parents[1] / "runner/base_binrepair_e2e.py").read_text()
    start = source.index("def fast_forward(")
    end = source.index("\ndef loss_window(", start)
    body = source[start:end]

    assert "active_cache = DynamicCache(config=config)" in body
    assert "past_key_values=active_cache, position_ids=pos" in body
    assert "active_cache if not requires_grad else DynamicCache(config=config)" in body
    assert "clear_inference_cache" not in body
    assert "entry.keys = entry.keys.new_empty((0,))" not in body