from pathlib import Path


def test_v5_worker_requires_launch_bound_manifest_sha_and_count() -> None:
    source = (Path(__file__).parents[1] / "tools" / "sensitivity_probe_worker_v5.py").read_text()
    assert 'os.environ.get("BANANA_SMASHER_PROBE_MANIFEST_SHA256")' in source
    assert 'raise RuntimeError("PROBE_MANIFEST_EXPECTED_SHA_MISSING")' in source
    assert "PROBE_MANIFEST_V5_COMPLETE20_COUNT = 20" in source
    assert 'raise RuntimeError("PROBE_MANIFEST_V5_COMPLETE20_SHA_RED")' in source