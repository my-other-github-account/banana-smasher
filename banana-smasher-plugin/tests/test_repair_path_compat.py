from __future__ import annotations

import pytest

from banana_smasher_plugin import repair


@pytest.mark.parametrize(
    ("source", "expected"),
    (
        ("model.layers.0.input_layernorm", "model.layers.0.attn_norm"),
        ("model.layers.0.post_attention_layernorm", "model.layers.0.ffn_norm"),
        ("model.layers.0.self_attn.q_a_norm", "model.layers.0.attn.q_norm"),
        ("model.layers.0.self_attn.kv_norm", "model.layers.0.attn.kv_norm"),
        (
            "model.layers.2.self_attn.compressor.kv_norm",
            "model.layers.2.attn.compressor.norm",
        ),
        (
            "model.layers.2.self_attn.compressor.indexer.kv_norm",
            "model.layers.2.attn.indexer.compressor.norm",
        ),
        ("model.layers.0.self_attn.o_b_proj", "model.layers.0.attn.wo_b"),
        ("model.norm", "model.norm"),
    ),
)
def test_translates_repair_paths_to_stock_vllm_deepseek_v4_names(
    source: str,
    expected: str,
) -> None:
    translate = getattr(repair, "_translate_stock_dsv4_repair_path", None)
    assert callable(translate), "stock DeepSeek-V4 repair-path adapter is missing"
    assert translate(source) == expected
