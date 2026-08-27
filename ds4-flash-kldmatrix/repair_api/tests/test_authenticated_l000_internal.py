from pathlib import Path

ROOT = Path(__file__).parents[1]


def test_authenticated_l000_internal_instrument_is_basis_and_control_bound():
    source = (ROOT / "resident_full64_accept.py").read_text()
    assert 'AUTHENTICATED_L000_INTERNAL_ONLY' in source
    assert 'ACCEPTED_L000_INTERNAL_SHA256' in source
    assert 'ACCEPTED_L000_INTERNAL_PATH' in source
    assert '11ead706db562197e76cdc320d5d13044bb254a411b6412326667f524ddf29ed' in source
    for boundary in (
        '"layer_input"', '"attention_return"', '"post_attention_residual"',
        '"router_input"', '"router_output"', '"moe_return"', '"final_residual"',
    ):
        assert boundary in source


def test_accepted_builder_instrument_authenticates_final_l000():
    source = (ROOT / "assets" / "builder_B2_PUBLISHED_PRE_internal.py").read_text()
    assert 'source_builder_parent_sha256' in source
    assert '11ead706db562197e76cdc320d5d13044bb254a411b6412326667f524ddf29ed' in source
    assert '62b2e4027be14946b581cfddaccca09b9f3baa7c459dd91c45f41a805d7c01ec' in source
    assert 'ACCEPTED_L000_FINAL_AUTHENTICATION_RED' in source


def test_authenticated_l000_probe_supports_the_authorized_single_rank_harness():
    source = (ROOT / "resident_full64_accept.py").read_text()
    assert "[None] * torch.distributed.get_world_size()" in source
    assert 'AUTHENTICATED_L000_SINGLE_RANK_ONLY' in source
    assert "expected_world_size = 1 if l000_single_rank else 2" in source
