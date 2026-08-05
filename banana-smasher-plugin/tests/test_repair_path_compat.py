from __future__ import annotations

from types import SimpleNamespace

from banana_smasher_plugin import repair


def test_resolves_exported_input_norm_on_stock_deepseek_v4_decoder_shape() -> None:
    stock_attn_norm = object()
    decoder = SimpleNamespace(attn_norm=stock_attn_norm)
    root = SimpleNamespace(
        model=SimpleNamespace(layers=SimpleNamespace(**{"0": decoder}))
    )

    assert (
        repair._resolve(root, "model.layers.0.input_layernorm")
        is stock_attn_norm
    )
