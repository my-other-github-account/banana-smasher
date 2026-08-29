from pathlib import Path


def test_l001_attention_payload_is_public_tap_only_and_fsynced() -> None:
    source = (Path(__file__).parents[1] / "resident_full64_accept.py").read_text()
    assert 'LAW4_L001_ATTENTION_PAYLOAD_ONLY' in source
    assert 'wrap(model.model.layers[1].self_attn, "L001_attention_return", _first_tensor)' in source
    assert 'RESIDENT_L001_ATTENTION_RETURN.pt' in source
    assert 'os.fsync(handle.fileno())' in source
    assert 'layer_taps.append("L001_attention_return")' in source
    assert 'layer_taps.append(f"L{index:03d}")' in source
    assert source.index('if singleton_public_parity_tap_only:') < source.rindex('_law4_public_product_taps(')


def test_l001_attention_input_payload_is_a_distinct_single_boundary_diagnostic() -> None:
    source = (Path(__file__).parents[1] / "resident_full64_accept.py").read_text()
    assert 'LAW4_L001_ATTENTION_INPUT_PAYLOAD_ONLY' in source
    assert 'wrap(model.model.layers[1].input_layernorm, "L001_attention_input")' in source
    assert 'RESIDENT_L001_ATTENTION_INPUT.pt' in source
    assert 'payload_name = "L001_attention_input" if attention_input_only else "L001_attention_return"' in source
    assert 'payload = retained_payloads.get(payload_name)' in source
    assert 'layer_taps.append("L001_attention_input")' in source
    assert source.index('layer_taps.append("L001_attention_input")') < source.index('layer_taps.append("L001_attention_return")')
