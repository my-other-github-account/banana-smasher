from pathlib import Path
import tempfile

from repair_api.resident_full64_accept import atomic


def test_resident_receipt_atomic_creates_missing_parent() -> None:
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "receipts" / "receipt.json"
        digest = atomic(path, {"status": "PASS"})
        assert path.exists()
        assert len(digest) == 64


def test_resident_full64_accept_is_single_load_admission_then_production() -> None:
    source = (Path(__file__).parents[1] / "resident_full64_accept.py").read_text()
    assert source.count('checkpoint_path("PRE")') == 1
    assert source.count("torch.load(checkpoint_path") == 1
    assert source.count("ModernGreenResidentEngine(") == 1
    assert 'config["sealed_builder_window_microbatch"] = 2' in source
    assert 'config["sealed_builder_window_microbatch"] = 4' in source
    eager = source.index('config["resident_validation_attention_implementation"] = "eager"')
    sdpa = source.index('config["resident_validation_attention_implementation"] = "sdpa"')
    admission = source.index("api.validate(engine, (28,)")
    production = source.index("api.validate(engine, windows")
    assert eager < admission < sdpa < production
    assert 'W28_KLD = 0.13712959240533734' in source
    assert "W28_TOP1 = 877" in source
    assert 'post_load_wall >= 300.0' in source
    assert 'banana-smasher-resident-full64-rate-low-v2' in source
    assert 'FULL64_REQUIRES_ACCEPTED_PROVIDER' in source
    assert 'len(rows) != 64' in source
    assert 'checkpoint_reloads' in source
    assert 'per_window_diff' in source
    assert 'reference_terminal_sha256' in source


def test_sdpa_repair_retains_attention_sink_denominator() -> None:
    source = (Path(__file__).parents[1] / "modern_green_resident.py").read_text()
    assert "scaled_dot_product_attention" in source
    assert "torch.logsumexp(scores, dim=-1)" in source
    assert "torch.sigmoid(logsumexp - sinks.to(logsumexp.dtype))" in source
    assert "official_k2_sink_corrected_sdpa" in source
