from pathlib import Path


def test_v5_worker_binds_sealed_manifest_and_count() -> None:
    source = (Path(__file__).parents[1] / "tools" / "sensitivity_probe_worker_v5.py").read_text()
    assert 'PROBE_MANIFEST_V5_COMPLETE20_SHA = "21113fc48a370cbaf479071c96d46143b0158abe938f13968e297b536b7ec9df"' in source
    assert "PROBE_MANIFEST_V5_COMPLETE20_COUNT = 20" in source
    assert 'raise RuntimeError("PROBE_MANIFEST_V5_COMPLETE20_SHA_RED")' in source