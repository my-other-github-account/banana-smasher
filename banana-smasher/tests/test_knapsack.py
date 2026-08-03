from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from banana_smasher.cli import main
from banana_smasher.knapsack import KnapsackValidationError, run_knapsack

BASIS = "a" * 64


def _canonical(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


def _write_json(path: Path, value: object) -> str:
    payload = _canonical(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


def _fixture(root: Path) -> Path:
    anchors: dict[str, dict[str, str]] = {}
    for tier, byte_counts in {"qtip2": (1, 1), "qtip3": (2, 2)}.items():
        path = root / "anchors" / tier / "MANIFEST.json"
        digest = _write_json(
            path,
            {
                "schema": "banana-smasher-anchor-manifest-v1",
                "status": "SEALED",
                "basis_sha256": BASIS,
                "tier": tier,
                "cells": [
                    {"cell_id": "L000.E000.down", "bytes": byte_counts[0]},
                    {"cell_id": "L000.E000.fused13", "bytes": byte_counts[1]},
                ],
            },
        )
        anchors[tier] = {
            "path": path.relative_to(root).as_posix(),
            "sha256": digest,
            "producer_command": "immutable test fixture",
        }
    damage_path = root / "damage" / "ROWS.json"
    damage_sha = _write_json(
        damage_path,
        {
            "schema": "banana-smasher-damage-rows-v1",
            "status": "SEALED",
            "basis_sha256": BASIS,
            "rows": [
                {"cell_id": "L000.E000.down", "tier": "qtip2", "damage": 0.4},
                {"cell_id": "L000.E000.down", "tier": "qtip3", "damage": 0.1},
                {"cell_id": "L000.E000.fused13", "tier": "qtip2", "damage": 0.3},
                {"cell_id": "L000.E000.fused13", "tier": "qtip3", "damage": 0.2},
            ],
        },
    )
    _write_json(
        root / "MANIFEST.json",
        {
            "schema": "banana-smasher-knapsack-input-index-v1",
            "status": "PASS",
            "intended_basis_sha256": BASIS,
            "intended_tiers": ["qtip2", "qtip3"],
            "envelope_bytes": 3,
            "missing_inputs": [],
            "anchor_manifests": anchors,
            "damage_rows": {
                "path": damage_path.relative_to(root).as_posix(),
                "sha256": damage_sha,
                "producer_command": "immutable test fixture",
            },
        },
    )
    return root


def test_exact_integer_envelope_serialization_and_deterministic_roundtrip(tmp_path: Path) -> None:
    root = _fixture(tmp_path / "run")
    first = run_knapsack(run_root=root, envelope_bytes=3)
    assignment_before = (root / "knapsack/ASSIGNMENT.json").read_bytes()
    receipt_before = (root / "knapsack/RECEIPT.json").read_bytes()

    second = run_knapsack(run_root=root, envelope_bytes=3)

    assignment = json.loads(assignment_before)
    assert first == second
    assert (root / "knapsack/ASSIGNMENT.json").read_bytes() == assignment_before
    assert (root / "knapsack/RECEIPT.json").read_bytes() == receipt_before
    assert assignment["byte_accounting"] == {
        "assigned_bytes": 3,
        "envelope_bytes": 3,
        "slack_bytes": 0,
    }
    assert sum(row["bytes"] for row in assignment["assignments"]) == 3


def test_rerun_repairs_missing_receipt_without_rewriting_assignment(tmp_path: Path) -> None:
    root = _fixture(tmp_path / "run")
    run_knapsack(run_root=root, envelope_bytes=3)
    assignment_path = root / "knapsack/ASSIGNMENT.json"
    receipt_path = root / "knapsack/RECEIPT.json"
    assignment_before = assignment_path.read_bytes()
    receipt_before = receipt_path.read_bytes()
    receipt_path.unlink()

    result = run_knapsack(run_root=root, envelope_bytes=3)

    assert result["status"] == "PASS"
    assert assignment_path.read_bytes() == assignment_before
    assert receipt_path.read_bytes() == receipt_before


def test_cli_exact_knapsack_roundtrip(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    root = _fixture(tmp_path / "run")
    assert main(["knapsack", "--run-root", str(root), "--envelope-bytes", "3"]) == 0
    emitted = json.loads(capsys.readouterr().out)
    assert emitted["status"] == "PASS"
    assert emitted["byte_accounting"]["assigned_bytes"] == 3


def test_infeasible_physical_envelope_fails_loudly(tmp_path: Path) -> None:
    root = _fixture(tmp_path / "run")
    with pytest.raises(KnapsackValidationError, match="minimum required 2 bytes exceeds"):
        run_knapsack(run_root=root, envelope_bytes=1)


def test_manifest_basis_mismatch_fails_before_solve(tmp_path: Path) -> None:
    root = _fixture(tmp_path / "run")
    anchor = root / "anchors/qtip2/MANIFEST.json"
    value = json.loads(anchor.read_text())
    value["basis_sha256"] = "b" * 64
    digest = _write_json(anchor, value)
    manifest_path = root / "MANIFEST.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["anchor_manifests"]["qtip2"]["sha256"] = digest
    _write_json(manifest_path, manifest)

    with pytest.raises(KnapsackValidationError, match="basis mismatch"):
        run_knapsack(run_root=root, envelope_bytes=3)


def test_manifest_rejects_wrong_anchor_sha_without_partial_outputs(tmp_path: Path) -> None:
    root = _fixture(tmp_path / "run")
    manifest_path = root / "MANIFEST.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["anchor_manifests"]["qtip2"]["sha256"] = "f" * 64
    _write_json(manifest_path, manifest)

    with pytest.raises(KnapsackValidationError, match="SHA-256 mismatch"):
        run_knapsack(run_root=root, envelope_bytes=3)
    assert not (root / "knapsack/ASSIGNMENT.json").exists()
    assert not (root / "knapsack/RECEIPT.json").exists()
