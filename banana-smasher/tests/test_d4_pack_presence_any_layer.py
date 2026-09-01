"""The D4 pack-presence gate must accept selected-cell packs whose first layer
has no D4 selections (full-menu exact102 selects zero D4 cells in L000/L001)."""


def _has_declared_d4(tensor_index, source_key):
    # Mirrors hf_deepseek_v4_backpack_adapter presence check.
    expected_marker = f".truevq_d4.{source_key}."
    return any(
        name.startswith("layers.") and expected_marker in name for name in tensor_index
    )


def test_d4_presence_accepts_pack_starting_at_layer_2():
    idx = {
        "layers.2.truevq_d4.d4_k4096.down.codes": {},
        "layers.10.truevq_d4.d4_k4096.down.codebooks": {},
    }
    assert _has_declared_d4(idx, "d4_k4096")


def test_d4_presence_rejects_pack_without_that_tier():
    idx = {"layers.2.truevq_d4.d4_k2048.down.codes": {}}
    assert not _has_declared_d4(idx, "d4_k4096")


def test_d4_presence_source_matches_adapter():
    import inspect
    from banana_smasher import hf_deepseek_v4_backpack_adapter as m

    src = inspect.getsource(m)
    assert 'f"layers.0.truevq_d4' not in src
    assert '".truevq_d4.{source_key}."' in src
