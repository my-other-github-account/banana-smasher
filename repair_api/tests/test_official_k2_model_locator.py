import hashlib
from pathlib import Path

import pytest

from repair_api.api import _resolve_official_k2_config_locators
from repair_api.balanced64 import ArtifactError


STALE = "/home/dnola/missions/STAGE_U20_t_3a6f22a5_spark-5-work/sparse-model-rank0-v1"


def test_localizes_only_missing_sealed_locator_with_exact_basis(monkeypatch, tmp_path: Path) -> None:
    candidate = tmp_path / "model-ro"
    candidate.mkdir()
    index = candidate / "model.safetensors.index.json"
    index.write_bytes(b"basis-index")
    basis = hashlib.sha256(index.read_bytes()).hexdigest()
    monkeypatch.setenv("BANANA_SMASHER_OFFICIAL_MODEL_ROOT", str(candidate))
    monkeypatch.setattr("repair_api.api.BASIS_SHA256", basis)
    monkeypatch.setattr("repair_api.api.bind_sealed_pre_resident_config", lambda config: {})
    monkeypatch.setattr("pathlib.Path.exists", lambda self: self != Path(STALE))
    monkeypatch.setattr("pathlib.Path.is_file", lambda self: True)
    original_read_bytes = Path.read_bytes
    monkeypatch.setattr(
        "pathlib.Path.read_bytes",
        lambda self: original_read_bytes(self) if self == index else b"asset",
    )
    asset_sha = hashlib.sha256(b"asset").hexdigest()

    resolved = _resolve_official_k2_config_locators(
        {
            "model_root": STALE,
            "basis_sha256": basis,
            "fast_k2_extension_sha256": asset_sha,
            "fast_k2_wrapper_source_sha256": asset_sha,
            "official_expert_source_sha256": asset_sha,
            "resident_expert_source_sha256": asset_sha,
            "trainer_source_sha256": asset_sha,
        }
    )

    assert resolved["model_root"] == str(candidate.resolve())

    with pytest.raises(ArtifactError, match="sealed stale Spark-5 locator"):
        _resolve_official_k2_config_locators(
            {"model_root": str(tmp_path / "other"), "basis_sha256": basis}
        )