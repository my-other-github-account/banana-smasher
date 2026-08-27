from __future__ import annotations

import hashlib
import json

from banana_smasher.provenance_wire import (
    build_full_wire_provenance_ledger,
    preflight_provenance_physical,
    run_configured_provenance_solve,
    run_full_wire_provenance_solve,
)


def _sha(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_build_full_wire_provenance_ledger_reprices_d4(tmp_path) -> None:
    provenance = tmp_path / "provenance.jsonl"
    rows = [
        {
            "cell_id": "L000:E000:down",
            "tier": "d4_k2048",
            "physical_bytes": 10,
            "activation_ids": ["wrong-pack-id"],
        },
        {
            "cell_id": "L000:E000:fused13",
            "tier": "qtip2",
            "physical_bytes": 20,
            "activation_ids": ["duplicated-unit-id"],
        },
    ]
    provenance.write_text("".join(json.dumps(row) + "\n" for row in rows))
    artifact_id = "sha256:" + "1" * 64
    full_wire = tmp_path / "full-wire.jsonl"
    full_wire.write_text(
        json.dumps(
            {
                "schema": "banana-smasher-backpack-option-row-v1",
                "cell_id": "L000:E000:down",
                "tier": "d4_k2048",
                "physical_bytes": 14,
                "activation_artifact_ids": [artifact_id],
                "byte_definition": "complete per-cell selectable D4 wire",
            }
        )
        + "\n"
    )
    registry = tmp_path / "registry.json"
    registry.write_text(
        json.dumps(
            {
                "activation_artifacts": [
                    {"artifact_id": artifact_id, "bytes": 3, "role": "d4_codebook"}
                ]
            }
        )
        + "\n"
    )

    receipt = build_full_wire_provenance_ledger(
        provenance,
        full_wire,
        registry,
        tmp_path / "output.jsonl",
        tmp_path / "receipt.json",
        expected_provenance_sha256=_sha(provenance),
        expected_full_wire_sha256=_sha(full_wire),
        expected_activation_registry_sha256=_sha(registry),
    )

    output = [json.loads(line) for line in (tmp_path / "output.jsonl").read_text().splitlines()]
    assert output[0]["physical_bytes"] == 14
    assert output[0]["activation_ids"] == [artifact_id]
    assert output[0]["activation_artifacts"] == [
        {"artifact_id": artifact_id, "id": artifact_id, "bytes": 3, "role": "d4_codebook"}
    ]
    assert output[1]["physical_bytes"] == 20
    assert output[1]["activation_ids"] == []
    assert output[1]["activation_artifacts"] == []
    assert receipt["status"] == "PASS"
    assert receipt["output"]["d4_rows"] == 1


def test_run_full_wire_provenance_solve_charges_shared_activation_once(tmp_path) -> None:
    identity = {
        "model_id": "test/model",
        "model_revision": "r1",
        "basis_sha256": "a" * 64,
        "bank_sha256": "b" * 64,
        "teacher_sha256": "c" * 64,
        "scorer_sha256": "d" * 64,
    }
    artifact = {"id": "shared-d4-codebook", "bytes": 3}
    rows = []
    for cell in ("L000:E000:down", "L000:E001:down"):
        rows.extend(
            [
                {
                    **identity,
                    "cell_id": cell,
                    "tier": "d4_k2048",
                    "physical_bytes": 2,
                    "prediction_by_class": {"chat": 0.0},
                    "activation_ids": [artifact["id"]],
                    "activation_artifacts": [artifact],
                },
                {
                    **identity,
                    "cell_id": cell,
                    "tier": "native_mxfp4",
                    "physical_bytes": 4,
                    "prediction_by_class": {"chat": 1.0},
                    "activation_ids": [],
                    "activation_artifacts": [],
                },
            ]
        )
    ledger = tmp_path / "full-wire-provenance.jsonl"
    ledger.write_text("".join(json.dumps(row) + "\n" for row in rows))
    fixed = tmp_path / "fixed.json"
    fixed.write_text(
        json.dumps(
            {
                "components": {
                    "dense_nonrouted_bytes": 3,
                    "repair_bytes": 0,
                    "metadata_bytes": 2,
                }
            }
        )
        + "\n"
    )

    receipt = run_full_wire_provenance_solve(
        ledger,
        fixed,
        tmp_path / "assignment.json",
        tmp_path / "solve-receipt.json",
        expected_option_ledger_sha256=_sha(ledger),
        expected_fixed_accounting_sha256=_sha(fixed),
        shipping_bytes_cap=12,
        class_weights={"chat": 1.0},
    )

    assignment = json.loads((tmp_path / "assignment.json").read_text())
    assert [row["tier"] for row in assignment["assignments"]] == [
        "d4_k2048",
        "d4_k2048",
    ]
    assert assignment["activation_artifacts"] == [artifact]
    assert receipt["whole_model_accounting"]["selected_cell_payload_bytes"] == 4
    assert receipt["whole_model_accounting"]["selected_activation_bytes"] == 3
    assert receipt["whole_model_accounting"]["whole_shipping_bytes"] == 12


def test_configured_provenance_solve_filters_tiers_and_hits_exact_cap(tmp_path) -> None:
    identity = {
        "model_id": "test/model",
        "model_revision": "r1",
        "basis_sha256": "a" * 64,
        "bank_sha256": "b" * 64,
        "teacher_sha256": "c" * 64,
        "scorer_sha256": "d" * 64,
    }
    rows = []
    for cell in ("L000:E000:down", "L000:E001:down"):
        for tier, physical_bytes, damage in (
            ("qtip2", 2, 2.0),
            ("qtip3", 3, 1.0),
            ("native_mxfp4", 4, 0.0),
        ):
            rows.append(
                {
                    **identity,
                    "cell_id": cell,
                    "tier": tier,
                    "physical_bytes": physical_bytes,
                    "prediction_by_class": {"chat": damage},
                    "activation_ids": [],
                    "activation_artifacts": [],
                }
            )
    ledger = tmp_path / "options.jsonl"
    ledger.write_text("".join(json.dumps(row) + "\n" for row in rows))
    fixed = tmp_path / "fixed.json"
    fixed.write_text(
        json.dumps(
            {
                "components": {
                    "dense_nonrouted_bytes": 3,
                    "repair_bytes": 0,
                    "metadata_bytes": 2,
                }
            }
        )
        + "\n"
    )
    config = tmp_path / "config.json"
    config.write_text(
        json.dumps(
            {
                "schema": "banana-smasher-provenance-solve-config-v1",
                "inputs": {
                    "option_ledger": {"path": ledger.name, "sha256": _sha(ledger)},
                    "fixed_accounting": {"path": fixed.name, "sha256": _sha(fixed)},
                },
                "target": {"whole_model_bytes": 11, "exact": True},
                "allowed_tiers": ["qtip2", "qtip3"],
                "class_weights": {"chat": 1.0},
            }
        )
        + "\n"
    )

    receipt = run_configured_provenance_solve(config, output=tmp_path / "solve")
    assignment = json.loads((tmp_path / "solve" / "ASSIGNMENT.json").read_text())
    assert assignment["allowed_tiers"] == ["qtip2", "qtip3"]
    assert assignment["exact_envelope"] is True
    assert {row["tier"] for row in assignment["assignments"]} == {"qtip3"}
    assert receipt["whole_model_accounting"]["whole_shipping_bytes"] == 11


def test_physical_preflight_durably_refuses_missing_cells(tmp_path) -> None:
    assignment = tmp_path / "ASSIGNMENT.json"
    assignment.write_text(json.dumps({
        "schema": "banana-smasher-provenance-weighted-assignment-v1",
        "basis_sha256": "a" * 64,
        "assignments": [
            {"cell_id": "L000:E000:down", "tier": "qtip3", "bytes": 3},
            {"cell_id": "L000:E000:fused13", "tier": "qtip2", "bytes": 2},
        ],
    }) + "\n")
    locate = tmp_path / "LOCATE_MANIFEST.json"
    locate.write_text(json.dumps({
        "schema": "banana-smasher-mixed-backpack-locate-manifest-v1",
        "basis_sha256": "a" * 64,
        "assignment": {"sha256": _sha(assignment)},
        "rows": [{
            "cell_id": "L000:E000:down",
            "layer": 0,
            "basis_sha256": "a" * 64,
            "selected_physical_bytes": 3,
            "status": "ARCHIVED_AUTHORITY_ONLY_NOT_LIVE_LOCATED",
        }],
    }) + "\n")

    result = preflight_provenance_physical(
        assignment, locate, tmp_path / "PHYSICAL_PREFLIGHT.json"
    )

    assert result["status"] == "REFUSED_MISSING_PHYSICAL"
    assert result["selected_qtip3_cells"] == 1
    assert result["live_qtip3_cells"] == 0
    assert result["missing_qtip3_cells"] == 1
    assert result["missing_qtip3_bytes"] == 3
    assert result["missing_by_layer"] == {"L000": 1}
