import hashlib
from pathlib import Path

import pytest

from repair_api.api import (
    _resolve_exact_parent_manifest,
    _resolve_official_k2_config_locators,
    _select_exact_manifest_member,
)
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
    parent = tmp_path / "full-parent"
    lut_parent = tmp_path / "lut-parent"
    monkeypatch.setattr("repair_api.api._SPARK3_SEALED_PARENT_ROOT", parent)
    monkeypatch.setattr("repair_api.api._SPARK3_SEALED_PARENT_MANIFEST_ROOT", lut_parent)
    monkeypatch.setattr("repair_api.api._validate_sealed_parent_root", lambda **kwargs: None)
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
    assert resolved["parent_root"] == str(parent.resolve())
    assert resolved["lut_parent_root"] == str(lut_parent.resolve())

    with pytest.raises(ArtifactError, match="sealed stale Spark-5 locator"):
        _resolve_official_k2_config_locators(
            {"model_root": str(tmp_path / "other"), "basis_sha256": basis}
        )


def test_manifest_member_selector_accepts_identical_duplicate_and_rejects_conflict(
    tmp_path: Path,
) -> None:
    q2 = tmp_path / "E000_w1.q2v7wire"
    k2 = tmp_path / "E000_w1.k2wire"
    q2.write_bytes(b"sealed-member")
    k2.write_bytes(b"sealed-member")
    expected = hashlib.sha256(b"sealed-member").hexdigest()

    assert _select_exact_manifest_member(
        (q2, k2), expected_sha256=expected, label="L000 E000/w1"
    ) == q2.resolve()

    k2.write_bytes(b"conflicting-member")
    with pytest.raises(ArtifactError, match="non-identical ambiguity"):
        _select_exact_manifest_member(
            (q2, k2), expected_sha256=expected, label="L000 E000/w1"
        )


def test_parent_manifest_localizes_only_to_identity_exact_copy(tmp_path: Path) -> None:
    declared = tmp_path / "stale" / "QTIP_V7_MANIFEST.json"
    localized = tmp_path / "localized" / "QTIP_V7_MANIFEST.json"
    localized.parent.mkdir()
    localized.write_bytes(b"sealed-manifest")
    expected = hashlib.sha256(b"sealed-manifest").hexdigest()

    assert _resolve_exact_parent_manifest(
        declared, localized=localized, expected_sha256=expected, label="L000"
    ) == localized.resolve()

    localized.write_bytes(b"drifted-manifest")
    with pytest.raises(ArtifactError, match="identity drift: L000"):
        _resolve_exact_parent_manifest(
            declared, localized=localized, expected_sha256=expected, label="L000"
        )