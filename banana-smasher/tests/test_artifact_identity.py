from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from banana_smasher.artifact_identity import ArtifactIdentity
from banana_smasher.contract import PackValidationError


def sha(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def document() -> dict:
    return {
        "schema": "banana-smasher-artifact-identity-v1",
        "basis": {
            "model_index_sha256": sha("basis"),
            "official_physical_layer_sha256": sha("physical"),
        },
        "corpora": {
            "builder_eval_sha256": sha("builder"),
            "train_score_sha256": sha("train"),
            "u0_lock_sha256": sha("lock-corpus"),
            "teacher_inventory_sha256": sha("teacher"),
        },
        "checkpoints": {
            "pre": {
                "sha256": sha("pre"),
                "identity_sha256": sha("pre-identity"),
                "lock_sha256": sha("lock"),
                "trajectory_sha256": sha("trajectory"),
            }
        },
        "composition": {
            "kind": "mixed-backpack",
            "layers": [
                {"layer": 0, "tiers": {"qtip2_v7": 400, "native_mxfp4": 112}},
                {"layer": 1, "tiers": {"qtip3": 256, "d4_k2048": 256}},
            ],
        },
        "canary": {
            "reference": {"kld": 0.22939197531977115, "top1": 56533},
            "tolerance": {"kld_abs": 0.004587839506395423, "top1_abs": 0},
        },
        "runtime": {
            "qtip2_v7": {
                "route_census_sha256": sha("route-census"),
                "shared_lut_sha256": sha("shared-lut"),
                "dense_roster_sha256": sha("dense-roster"),
            }
        },
    }


def test_identity_loads_mixed_composition_canary_and_v7_bindings(tmp_path: Path) -> None:
    path = tmp_path / "identity.json"
    path.write_text(json.dumps(document()))
    value = ArtifactIdentity.load(tmp_path)
    assert value.basis_sha256 == sha("basis")
    assert value.composition[0]["tiers"]["qtip2_v7"] == 400
    assert value.canary.reference_kld == 0.22939197531977115
    assert value.runtime["qtip2_v7"]["route_census_sha256"] == sha("route-census")
    assert value.sha256 == hashlib.sha256(path.read_bytes()).hexdigest()


def test_identity_fails_closed_instead_of_defaulting_artifact_constants(tmp_path: Path) -> None:
    with pytest.raises(PackValidationError, match="identity.json"):
        ArtifactIdentity.load(tmp_path)
    value = document()
    del value["canary"]["tolerance"]["kld_abs"]
    (tmp_path / "identity.json").write_text(json.dumps(value))
    with pytest.raises(PackValidationError, match="canary.tolerance.kld_abs"):
        ArtifactIdentity.load(tmp_path)


def test_canary_kld_failure_reports_actual_reference_delta_and_tolerance(tmp_path: Path) -> None:
    (tmp_path / "identity.json").write_text(json.dumps(document()))
    value = ArtifactIdentity.load(tmp_path)

    with pytest.raises(PackValidationError) as failure:
        value.require_canary(kld=0.5, top1=56533)

    message = str(failure.value)
    assert "actual=0.5" in message
    assert "reference=0.22939197531977115" in message
    assert "abs_delta=0.27060802468022882" in message
    assert "abs_tolerance=0.0045878395063954228" in message


def test_post_canary_accepts_kld_better_than_lower_tolerance_bound(tmp_path: Path) -> None:
    (tmp_path / "identity.json").write_text(json.dumps(document()))
    value = ArtifactIdentity.load(tmp_path)

    value.require_canary(kld=0.1971105878793163, top1=56533, allow_kld_improvement=True)


def test_post_canary_still_rejects_kld_worse_than_upper_tolerance_bound(tmp_path: Path) -> None:
    (tmp_path / "identity.json").write_text(json.dumps(document()))
    value = ArtifactIdentity.load(tmp_path)

    with pytest.raises(PackValidationError, match="outside declared tolerance"):
        value.require_canary(kld=0.5, top1=56533, allow_kld_improvement=True)
