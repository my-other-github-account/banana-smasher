from __future__ import annotations

import hashlib
import json
from pathlib import Path

from banana_smasher.backpack_contextual_prepare import prepare_contextual_iteration
from banana_smasher.cli import main


def test_prepare_contextual_iteration_is_public_api() -> None:
    import banana_smasher

    assert banana_smasher.prepare_contextual_iteration is prepare_contextual_iteration


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, sort_keys=True) + "\n")


def _descriptor(path: Path, *, key: str = "path") -> dict[str, object]:
    payload = path.read_bytes()
    return {
        key: path.name,
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def test_prepare_contextual_iteration_derives_physical_aliases_from_artifacts(
    tmp_path,
) -> None:
    basis = "b" * 64
    assignment = tmp_path / "ASSIGNMENT.json"
    materialization = tmp_path / "MATERIALIZATION_INDEX.jsonl"
    option_ledger = tmp_path / "OPTION_LEDGER.jsonl"
    solve_input = tmp_path / "SOLVE_INPUT.json"
    virtual_manifest = tmp_path / "BACKPACK_VIRTUAL_MANIFEST.json"
    terminal = tmp_path / "TERMINAL.json"

    _write_json(assignment, {"0": {"0": {"down": "qtip2"}}})
    materialization.write_text(
        json.dumps(
            {
                "activation_artifact_ids": ["shared"],
                "cell_id": "L000:E000:down",
                "expert": 0,
                "layer": 0,
                "physical_bytes": 10,
                "projection": "down",
                "source_key": "qtip2",
                "tier": "qtip2",
            },
            sort_keys=True,
        )
        + "\n"
    )
    rows = [
        {
            "activation_artifact_ids": ["shared"],
            "basis_sha256": basis,
            "cell_id": "L000:E000:down",
            "expert": 0,
            "layer": 0,
            "physical_bytes": 10,
            "prediction_by_class": {"code": 1.0},
            "projection": "down",
            "schema": "banana-smasher-backpack-option-row-v1",
            "tier": "qtip2",
        },
        {
            "activation_artifact_ids": ["shared"],
            "basis_sha256": basis,
            "cell_id": "L000:E000:down",
            "expert": 0,
            "layer": 0,
            "physical_bytes": 10,
            "prediction_by_class": {"code": -999.0},
            "prediction_components": {"qtip_k": 2},
            "projection": "down",
            "schema": "banana-smasher-backpack-option-row-v1",
            "tier": "qtip2_5",
        },
    ]
    option_ledger.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
    )
    _write_json(
        solve_input,
        {
            "schema": "banana-smasher-backpack-exact-matched-full-wire-input-v2",
            "status": "PASS",
            "basis_sha256": basis,
            "budget": {
                "fixed_nonexpert_bytes": 5,
                "total_package_bytes": 15,
            },
            "activation_artifacts": [{"artifact_id": "shared", "bytes": 1}],
        },
    )
    _write_json(
        virtual_manifest,
        {
            "schema": "banana-smasher-backpack-virtual-assignment-v1",
            "status": "PASS_LOGICAL_FULL_WIRE",
            "basis_sha256": basis,
            "assignment_map_sha256": _descriptor(assignment)["sha256"],
            "assignment": _descriptor(assignment, key="file"),
            "materialization_index": _descriptor(materialization, key="file"),
            "option_ledger": _descriptor(option_ledger),
            "solve_input": _descriptor(solve_input),
            "byte_accounting": {
                "fixed_nonexpert_bytes": 5,
                "assigned_package_bytes": 15,
            },
        },
    )
    pack_files = []
    for path in (assignment, virtual_manifest, materialization):
        pack_files.append(
            {
                "file": path.name,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "bytes": path.stat().st_size,
            }
        )
    pack_files.sort(key=lambda row: row["file"])
    pack_sha = hashlib.sha256(
        (json.dumps(pack_files, sort_keys=True, separators=(",", ":")) + "\n").encode()
    ).hexdigest()
    _write_json(
        terminal,
        {
            "schema": "banana-smasher-backpack-exact64-terminal-v1",
            "status": "PASS",
            "basis_sha256": basis,
            "pack_sha256": pack_sha,
            "windows": 64,
            "positions": 65536,
            "support_width": 8192,
            "mean_kld": 0.1,
            "top1_matches": 100,
            "score_receipt_sha256": "c" * 64,
        },
    )

    receipt = prepare_contextual_iteration(
        virtual_manifest,
        terminal,
        output_root=tmp_path / "prepared",
    )

    anchor = json.loads((tmp_path / "prepared" / "ANCHOR.json").read_text())
    inventory = json.loads((tmp_path / "prepared" / "OPTION_INVENTORY.json").read_text())
    measurements = json.loads(
        (tmp_path / "prepared" / "MEASUREMENTS.json").read_text()
    )
    qtip25 = next(row for row in inventory["options"] if row["option"] == "qtip2_5")
    assert receipt["status"] == "PASS"
    assert qtip25["physical_identity"] == anchor["cells"][0]["physical_identity"]
    assert "prediction_by_class" not in qtip25
    assert anchor["fixed_bytes"] == 5
    assert anchor["package_cap_bytes"] == 15
    assert anchor["physical_score_receipt_sha256"] == "c" * 64
    assert measurements["measurements"] == []


def test_cli_delegates_prepare_contextual_to_public_process(
    tmp_path, capsys, monkeypatch
) -> None:
    observed = {}

    def fake_prepare(virtual_manifest, score_receipt, **parameters):
        observed.update(
            virtual_manifest=virtual_manifest,
            score_receipt=score_receipt,
            **parameters,
        )
        return {"status": "PASS", "cells": 1, "options": 2}

    monkeypatch.setattr(
        "banana_smasher.backpack_contextual_prepare.prepare_contextual_iteration",
        fake_prepare,
    )
    status = main(
        [
            "backpack",
            "prepare-contextual",
            "--virtual-manifest",
            str(tmp_path / "virtual.json"),
            "--score-receipt",
            str(tmp_path / "terminal.json"),
            "--output",
            str(tmp_path / "prepared"),
        ]
    )

    emitted = json.loads(capsys.readouterr().out)
    assert status == 0
    assert emitted["command"] == "backpack prepare-contextual"
    assert observed["output_root"] == tmp_path / "prepared"
