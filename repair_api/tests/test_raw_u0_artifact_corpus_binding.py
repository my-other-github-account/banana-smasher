from __future__ import annotations

import hashlib
import json

from repair_api import official_k2_resident_score as scorer


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_raw_u0_adapter_binds_corpus_to_sealed_artifact_and_config(monkeypatch, tmp_path):
    root = tmp_path
    checkpoint = root / "checkpoints" / "UPDATE_000.pt"
    checkpoint.parent.mkdir()
    checkpoint.write_bytes(b"authentic-checkpoint-fixture")

    lock = root / "receipts" / "CLEAN_U0_LOCK.json"
    lock.parent.mkdir()
    lock.write_text(json.dumps({
        "u0_checkpoint_sha256": _sha256(checkpoint),
        "basis_sha256": "basis-fixture",
        "corpus_sha256": "sealed-corpus-fixture",
        "trajectory_sha256": "trajectory-fixture",
    }))

    manifest = {
        "schema": "repair-artifact-v1",
        "checkpoints": {
            "UPDATE_000": {
                "path": "checkpoints/UPDATE_000.pt",
                "sha256": _sha256(checkpoint),
                "identity_sha256": "identity-fixture",
                "next_update": 0,
                "parent_sha256": None,
            }
        },
        "identity": {
            "basis_sha256": "basis-fixture",
            "builder_eval_corpus_sha256": "sealed-corpus-fixture",
            "train_score_corpus_sha256": "sealed-corpus-fixture",
            "teacher_inventory_sha256": "teacher-fixture",
        },
        "canonical_raw_u0": {
            "clean_u0_lock_path": "receipts/CLEAN_U0_LOCK.json",
            "clean_u0_lock_sha256": _sha256(lock),
            "trajectory_sha256": "trajectory-fixture",
        },
    }
    (root / "ARTIFACT.json").write_text(json.dumps(manifest))

    monkeypatch.setattr(scorer, "BASIS_SHA256", "basis-fixture")
    monkeypatch.setattr(scorer, "CANONICAL_U0_CHECKPOINT_SHA256", _sha256(checkpoint))
    monkeypatch.setattr(scorer, "CANONICAL_U0_IDENTITY_SHA256", "identity-fixture")
    monkeypatch.setattr(scorer, "CANONICAL_U0_LOCK_SHA256", _sha256(lock))
    monkeypatch.setattr(scorer, "CANONICAL_U0_LOCK_CORPUS_SHA256", "sealed-corpus-fixture")
    monkeypatch.setattr(scorer, "CANONICAL_U0_TRAJECTORY_SHA256", "trajectory-fixture")
    monkeypatch.setattr(scorer, "TEACHER_INVENTORY_SHA256", "teacher-fixture")
    monkeypatch.setattr(scorer, "BUILDER_EVAL_CORPUS_SHA256", "unrelated-static-builder")
    monkeypatch.setattr(scorer, "SCORE_TRAIN_CORPUS_SHA256", "unrelated-static-score")
    monkeypatch.setattr(scorer, "_validate_raw_u0_gates", lambda payload, lock_payload: None)

    adapted = scorer.adapt_canonical_raw_u0_payload(
        {},
        artifact_root=root,
        manifest=manifest,
        config={"corpus_sha256": "sealed-corpus-fixture"},
    )

    assert adapted["identity"]["checkpoint_sha256"] == _sha256(checkpoint)
