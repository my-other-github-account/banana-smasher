from __future__ import annotations

import hashlib
import json
import math

from banana_smasher.backpack_contextual_measure import record_contextual_swap_measurement
from banana_smasher.cli import main


def test_record_contextual_swap_measurement_is_public_api() -> None:
    import banana_smasher

    assert (
        banana_smasher.record_contextual_swap_measurement
        is record_contextual_swap_measurement
    )


def _write(path, value) -> None:
    path.write_text(json.dumps(value, sort_keys=True) + "\n")


def test_record_contextual_swap_measurement_is_paired_and_manifest_backed(
    tmp_path,
) -> None:
    identities = {
        "bank_sha256": "1" * 64,
        "basis_sha256": "2" * 64,
        "model_id": "generic-model",
        "teacher_manifest_sha256": "3" * 64,
        "teacher_sha256": "4" * 64,
    }
    anchor_score = {
        "schema": "banana-smasher-anchor-sidecar-score-v1",
        "status": "PASS",
        "claimable": True,
        "support_width": 8192,
        "windows": 2,
        "positions": 20,
        "mean_kld": 0.2,
        "top1_matches": 15,
        "identities": {**identities, "pack_sha256": "5" * 64},
        "per_window": [
            {"window_id": "w0", "positions": 10, "mean_kld": 0.1, "top1_matches": 8},
            {"window_id": "w1", "positions": 10, "mean_kld": 0.3, "top1_matches": 7},
        ],
    }
    candidate_score = {
        **anchor_score,
        "mean_kld": 0.15,
        "top1_matches": 15,
        "identities": {**identities, "pack_sha256": "6" * 64},
        "per_window": [
            {"window_id": "w0", "positions": 10, "mean_kld": 0.2, "top1_matches": 7},
            {"window_id": "w1", "positions": 10, "mean_kld": 0.1, "top1_matches": 8},
        ],
    }
    anchor_score_path = tmp_path / "anchor-score.json"
    candidate_score_path = tmp_path / "candidate-score.json"
    _write(anchor_score_path, anchor_score)
    _write(candidate_score_path, candidate_score)
    anchor_score_sha = hashlib.sha256(anchor_score_path.read_bytes()).hexdigest()
    anchor_path = tmp_path / "anchor.json"
    _write(
        anchor_path,
        {
            "schema": "banana-smasher-contextual-anchor-v1",
            "status": "PASS",
            "assignment_sha256": "a" * 64,
            "physical_score_receipt_sha256": anchor_score_sha,
            "cells": [
                {
                    "cell": "cell",
                    "option": "old",
                    "physical_identity": "b" * 64,
                    "payload_bytes": 10,
                }
            ],
        },
    )
    change_path = tmp_path / "change.json"
    _write(
        change_path,
        {
            "schema": "banana-smasher-contextual-change-v1",
            "status": "READY",
            "anchor_assignment_sha256": "a" * 64,
            "candidate_assignment_sha256": "c" * 64,
            "candidate_pack_sha256": "6" * 64,
            "scope": "exact64",
            "change": {"cell": "cell", "physical_identity": "d" * 64},
        },
    )
    manifest_path = tmp_path / "measurements.json"
    _write(
        manifest_path,
        {
            "schema": "banana-smasher-contextual-measurement-manifest-v1",
            "status": "READY",
            "anchor_assignment_sha256": "a" * 64,
            "measurements": [],
        },
    )
    output_path = tmp_path / "measurement.json"

    summary = record_contextual_swap_measurement(
        anchor_path,
        change_path,
        anchor_score_path,
        candidate_score_path,
        measurement_manifest_path=manifest_path,
        output_path=output_path,
    )

    receipt = json.loads(output_path.read_text())
    manifest = json.loads(manifest_path.read_text())
    assert summary["status"] == "PASS"
    assert math.isclose(receipt["delta_mean_kld"], -0.05, abs_tol=1e-12)
    assert math.isclose(receipt["stderr_mean_kld"], 0.15, abs_tol=1e-12)
    assert receipt["delta_top1_matches"] == 0
    assert receipt["windows"] == 2
    assert manifest["measurements"] == [receipt]
    assert len(receipt["receipt_sha256"]) == 64


def test_cli_delegates_record_contextual_to_public_process(
    tmp_path, capsys, monkeypatch
) -> None:
    observed = {}

    def fake_record(anchor, change, anchor_score, candidate_score, **parameters):
        observed.update(
            anchor=anchor,
            change=change,
            anchor_score=anchor_score,
            candidate_score=candidate_score,
            **parameters,
        )
        return {"status": "PASS", "receipt_sha256": "f" * 64}

    monkeypatch.setattr(
        "banana_smasher.backpack_contextual_measure.record_contextual_swap_measurement",
        fake_record,
    )
    status = main(
        [
            "backpack",
            "record-contextual",
            "--anchor",
            str(tmp_path / "anchor.json"),
            "--change",
            str(tmp_path / "change.json"),
            "--anchor-score",
            str(tmp_path / "anchor-score.json"),
            "--candidate-score",
            str(tmp_path / "candidate-score.json"),
            "--measurements",
            str(tmp_path / "measurements.json"),
            "--output",
            str(tmp_path / "receipt.json"),
        ]
    )

    emitted = json.loads(capsys.readouterr().out)
    assert status == 0
    assert emitted["command"] == "backpack record-contextual"
    assert observed["measurement_manifest_path"] == tmp_path / "measurements.json"
