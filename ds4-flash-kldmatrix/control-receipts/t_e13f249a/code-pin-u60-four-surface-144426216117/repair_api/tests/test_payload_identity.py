from __future__ import annotations

import pytest

from repair_api.balanced64 import ArtifactError
from repair_api.official_k2_resident_score import validate_payload_identity


EXPECTED = {
    "checkpoint_sha256": "a" * 64,
    "checkpoint_identity_sha256": "b" * 64,
    "next_update": 16,
}


def test_legacy_payload_without_identity_uses_external_file_and_manifest_binding():
    validate_payload_identity({"state": {"luts": {}}}, **EXPECTED)


def test_legacy_payload_rejects_conflicting_embedded_checkpoint_sha():
    with pytest.raises(ArtifactError, match="checkpoint_sha256"):
        validate_payload_identity(
            {"state": {}, "checkpoint_sha256": "c" * 64},
            **EXPECTED,
        )


def test_present_identity_envelope_remains_strict():
    with pytest.raises(ArtifactError, match="identity_sha256"):
        validate_payload_identity(
            {"identity": {"next_update": 16, "checkpoint_loaded": True}},
            **EXPECTED,
        )


def test_legacy_descriptive_identity_uses_top_level_binding_fields():
    validate_payload_identity(
        {
            "identity": {"schema": "legacy-provenance-v1"},
            "identity_sha256": EXPECTED["checkpoint_identity_sha256"],
            "next_update": 16,
        },
        **EXPECTED,
    )


def test_nested_binding_field_overrides_and_rejects_conflicting_top_level_value():
    with pytest.raises(ArtifactError, match="next_update"):
        validate_payload_identity(
            {
                "identity": {
                    "identity_sha256": EXPECTED["checkpoint_identity_sha256"],
                    "next_update": 15,
                    "checkpoint_loaded": True,
                },
                "next_update": 16,
            },
            **EXPECTED,
        )
