from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from banana_smasher.backpack_dimensions import (
    CLASSES,
    DynamicDimensionsError,
    build_solved_dimension_sidecars,
    build_dynamic_dimensions,
)
from banana_smasher.cli import main

BASIS = "a" * 64
AUTHORITY = {
    "six_class_predictions_sha256": "1" * 64,
    "routing_importance_sha256": "2" * 64,
    "projection_correction_sha256": "3" * 64,
    "physical_bytes_sha256": "4" * 64,
}


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))


def _inputs(tmp_path: Path) -> tuple[Path, Path, Path]:
    ledger = tmp_path / "ledger.jsonl"
    dimensions = tmp_path / "dimensions.jsonl"
    ceilings = tmp_path / "ceilings.json"
    candidate_rows = []
    dimension_rows = []
    for projection, physical_bytes, importance in (("down", 101, 2.0), ("fused13", 203, 3.0)):
        candidate_id = f"d4_k4096:L000:E000:{projection}"
        common = {
            "basis_sha256": BASIS,
            "candidate_id": candidate_id,
            "layer": 0,
            "expert": 0,
            "projection": projection,
            "tier": "d4_k4096",
        }
        candidate_rows.append(
            {
                "schema": "banana-smasher-dynamic-backpack-candidate-ledger-row-v1",
                **common,
                "physical_bytes": physical_bytes,
                "source_class_features": {"routing_importance": importance},
                "missing_dimensions": ["six_class_predictions", "six_class_ceilings", "projection_correction"],
                "allocation_eligible": False,
                "status": "ADMITTED_PARTIAL_ALLOCATION_FORBIDDEN",
            }
        )
        dimension_rows.append(
            {
                "schema": "banana-smasher-dynamic-backpack-explicit-dimension-row-v1",
                **common,
                "physical_bytes": physical_bytes,
                "routing_importance": importance,
                "projection_weight": 0.75 if projection == "down" else 1.25,
                "projection_correction": -0.01 if projection == "down" else 0.02,
                "six_class_predictions": {name: 0.1 + index / 100 for index, name in enumerate(CLASSES)},
                "authority": AUTHORITY,
            }
        )
    _write_jsonl(ledger, candidate_rows)
    _write_jsonl(dimensions, dimension_rows)
    ceilings.write_text(
        json.dumps(
            {
                "schema": "banana-smasher-dynamic-backpack-class-ceilings-v1",
                "status": "SEALED",
                "basis_sha256": BASIS,
                "six_class_ceilings": {name: 1.0 for name in CLASSES},
            }
        )
    )
    return ledger, dimensions, ceilings


def test_complete_explicit_dimensions_seal_allocation_eligible_ledger(tmp_path: Path) -> None:
    ledger, dimensions, ceilings = _inputs(tmp_path)
    output, receipt = tmp_path / "complete.jsonl", tmp_path / "receipt.json"
    result = build_dynamic_dimensions(
        ledger=ledger,
        dimensions=dimensions,
        class_ceilings=ceilings,
        basis_sha256=BASIS,
        output=output,
        receipt=receipt,
    )
    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert result["status"] == "PASS_EXPLICIT_DIMENSIONS_COMPLETE"
    assert len(rows) == 2
    assert all(row["allocation_eligible"] is True and row["missing_dimensions"] == [] for row in rows)
    assert all(set(row["six_class_predictions"]) == set(CLASSES) for row in rows)
    assert all(set(row["six_class_ceilings"]) == set(CLASSES) for row in rows)
    assert json.loads(receipt.read_text())["inference_policy"].startswith("explicit-per-candidate-only")


def test_missing_candidate_dimensions_fail_closed(tmp_path: Path) -> None:
    ledger, dimensions, ceilings = _inputs(tmp_path)
    dimensions.write_text(dimensions.read_text().splitlines()[0] + "\n")
    with pytest.raises(DynamicDimensionsError, match="missing explicit dimensions"):
        build_dynamic_dimensions(
            ledger=ledger,
            dimensions=dimensions,
            class_ceilings=ceilings,
            basis_sha256=BASIS,
            output=tmp_path / "out.jsonl",
            receipt=tmp_path / "receipt.json",
        )


def test_aggregate_to_cell_prediction_is_forbidden(tmp_path: Path) -> None:
    ledger, dimensions, ceilings = _inputs(tmp_path)
    rows = [json.loads(line) for line in dimensions.read_text().splitlines()]
    rows[0]["six_class_predictions"] = {"mean": 0.1}
    _write_jsonl(dimensions, rows)
    with pytest.raises(DynamicDimensionsError, match="aggregate inference forbidden"):
        build_dynamic_dimensions(
            ledger=ledger,
            dimensions=dimensions,
            class_ceilings=ceilings,
            basis_sha256=BASIS,
            output=tmp_path / "out.jsonl",
            receipt=tmp_path / "receipt.json",
        )


def test_physical_byte_mismatch_is_forbidden(tmp_path: Path) -> None:
    ledger, dimensions, ceilings = _inputs(tmp_path)
    rows = [json.loads(line) for line in dimensions.read_text().splitlines()]
    rows[0]["physical_bytes"] += 1
    _write_jsonl(dimensions, rows)
    with pytest.raises(DynamicDimensionsError, match="physical_bytes mismatch"):
        build_dynamic_dimensions(
            ledger=ledger,
            dimensions=dimensions,
            class_ceilings=ceilings,
            basis_sha256=BASIS,
            output=tmp_path / "out.jsonl",
            receipt=tmp_path / "receipt.json",
        )


def test_mixed_append_only_ledger_resolves_v2_physical_binding(tmp_path: Path) -> None:
    ledger, dimensions, ceilings = _inputs(tmp_path)
    candidate_rows = [json.loads(line) for line in ledger.read_text().splitlines()]
    dimension_rows = [json.loads(line) for line in dimensions.read_text().splitlines()]
    old_id = candidate_rows[1]["candidate_id"]
    new_id = "d4_k4096:L000:E000:13"
    candidate_rows[1].update(
        {
            "schema": "banana-smasher-dynamic-backpack-candidate-ledger-row-v2",
            "candidate_id": new_id,
            "projection": "13",
            "physical_bytes": None,
        }
    )
    dimension_rows[1].update({"candidate_id": new_id, "projection": "13"})
    candidate_rows.append(
        {
            "schema": "banana-smasher-dynamic-backpack-dimension-binding-v1",
            "basis_sha256": BASIS,
            "candidate_id": new_id,
            "layer": 0,
            "expert": 0,
            "projection": "13",
            "tier": "d4_k4096",
            "physical_bytes": 203,
            "source_physical_sidecar_sha256": "5" * 64,
            "remaining_missing_dimensions": ["six_class_predictions"],
            "status": "PASS_AUTHORITATIVE_PHYSICAL_BYTES_ONLY_ALLOCATION_FORBIDDEN",
        }
    )
    assert old_id != new_id
    _write_jsonl(ledger, candidate_rows)
    _write_jsonl(dimensions, dimension_rows)

    result = build_dynamic_dimensions(
        ledger=ledger,
        dimensions=dimensions,
        class_ceilings=ceilings,
        basis_sha256=BASIS,
        output=tmp_path / "out.jsonl",
        receipt=tmp_path / "receipt.json",
    )

    output_rows = [json.loads(line) for line in (tmp_path / "out.jsonl").read_text().splitlines()]
    by_id = {row["candidate_id"]: row for row in output_rows}
    assert result["candidate_count"] == 2
    assert by_id[new_id]["physical_bytes"] == 203
    assert by_id[new_id]["projection"] == "13"
    assert by_id[new_id]["schema"] == "banana-smasher-dynamic-backpack-candidate-ledger-row-v2"


def test_conflicting_physical_bindings_fail_closed(tmp_path: Path) -> None:
    ledger, dimensions, ceilings = _inputs(tmp_path)
    candidate_rows = [json.loads(line) for line in ledger.read_text().splitlines()]
    dimension_rows = [json.loads(line) for line in dimensions.read_text().splitlines()]
    candidate_id = "d4_k4096:L000:E000:13"
    candidate_rows[1].update(
        {
            "schema": "banana-smasher-dynamic-backpack-candidate-ledger-row-v2",
            "candidate_id": candidate_id,
            "projection": "13",
            "physical_bytes": None,
        }
    )
    dimension_rows[1].update(
        {"candidate_id": candidate_id, "projection": "13", "physical_bytes": 204}
    )
    binding = {
        "schema": "banana-smasher-dynamic-backpack-dimension-binding-v1",
        "basis_sha256": BASIS,
        "candidate_id": candidate_id,
        "layer": 0,
        "expert": 0,
        "projection": "13",
        "tier": "d4_k4096",
        "source_physical_sidecar_sha256": "5" * 64,
        "remaining_missing_dimensions": ["six_class_predictions"],
        "status": "PASS_AUTHORITATIVE_PHYSICAL_BYTES_ONLY_ALLOCATION_FORBIDDEN",
    }
    candidate_rows.extend(
        [
            {**binding, "physical_bytes": 203},
            {**binding, "physical_bytes": 204},
        ]
    )
    _write_jsonl(ledger, candidate_rows)
    _write_jsonl(dimensions, dimension_rows)

    with pytest.raises(DynamicDimensionsError, match="conflicting dimension bindings"):
        build_dynamic_dimensions(
            ledger=ledger,
            dimensions=dimensions,
            class_ceilings=ceilings,
            basis_sha256=BASIS,
            output=tmp_path / "out.jsonl",
            receipt=tmp_path / "receipt.json",
        )


def test_public_cli_emits_dimensions(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    ledger, dimensions, ceilings = _inputs(tmp_path)
    output, receipt = tmp_path / "cli.jsonl", tmp_path / "cli-receipt.json"
    rc = main(
        [
            "backpack-dimensions",
            "--ledger", str(ledger),
            "--dimensions", str(dimensions),
            "--class-ceilings", str(ceilings),
            "--basis-sha256", BASIS,
            "--output", str(output),
            "--receipt", str(receipt),
        ]
    )
    emitted = json.loads(capsys.readouterr().out)
    assert rc == 0
    assert emitted["command"] == "backpack-dimensions"
    assert emitted["candidate_count"] == 2
    assert output.is_file() and receipt.is_file()


def _solved_sidecar_inputs(tmp_path: Path) -> tuple[Path, Path]:
    objective = tmp_path / "OBJECTIVE.json"
    profile = tmp_path / "PROFILE_ROWS.jsonl"
    objective.write_text(
        json.dumps(
            {
                "assignment": {
                    "L040.E000.P2": {
                        "tier": "d4_k4096",
                        "variant": "base",
                        "codeword_assignment_count": 4,
                        "codeword_assignment_dtype": "int16-le",
                        "codeword_assignment_sha256": "1" * 64,
                    },
                    "L040.E000.P13": {
                        "tier": "d4_k4096",
                        "variant": "base",
                        "codeword_assignment_count": 8,
                        "codeword_assignment_dtype": "int16-le",
                        "codeword_assignment_sha256": "2" * 64,
                    },
                }
            }
        )
    )
    profile.write_text(json.dumps({"layer": 40, "expert": 0, "routed_rows": 7}) + "\n")
    handoff = tmp_path / "PRODUCER_HANDOFF.json"
    handoff.write_text(
        json.dumps(
            {
                "status": "PASS_ADMISSION_READY_NOT_QUARANTINED",
                "basis_sha256": BASIS,
                "tier": "d4_k4096",
                "layers": [40],
                "layer_rows": [
                    {
                        "layer": 40,
                        "members": {
                            "OBJECTIVE": {
                                "path": str(objective),
                                "sha256": hashlib.sha256(objective.read_bytes()).hexdigest(),
                            },
                            "PROFILE_ROWS": {
                                "path": str(profile),
                                "sha256": hashlib.sha256(profile.read_bytes()).hexdigest(),
                            },
                        },
                    }
                ],
            }
        )
    )
    expectations = tmp_path / "AUTHORITY_EXPECTATIONS.json"
    expectations.write_text(
        json.dumps(
            {
                "schema": "banana-smasher-dynamic-backpack-authority-expectations-v1",
                "basis_sha256": BASIS,
                "blocked_groups": {
                    group: {
                        "required_key": group,
                        "expected_authority_path": f"/authority/{group}.json",
                        "expected_authority_sha256": None,
                        "searched_sources": [
                            {"path": str(handoff), "sha256": hashlib.sha256(handoff.read_bytes()).hexdigest()}
                        ],
                    }
                    for group in [
                        "projection_correction",
                        "projection_weight",
                        "six_class_ceilings",
                        "six_class_predictions",
                    ]
                },
            }
        )
    )
    return handoff, expectations


def test_solved_sidecars_publish_only_authenticated_non_inferred_bindings(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    handoff, expectations = _solved_sidecar_inputs(tmp_path)
    rc = main(
        [
            "backpack-solved-sidecars",
            "--handoff",
            str(handoff),
            "--handoff-sha256",
            hashlib.sha256(handoff.read_bytes()).hexdigest(),
            "--basis-sha256",
            BASIS,
            "--layers",
            "40",
            "--output-dir",
            str(tmp_path / "sidecars"),
            "--authority-expectations",
            str(expectations),
        ]
    )
    result = json.loads(capsys.readouterr().out)

    physical = [
        json.loads(line)
        for line in (tmp_path / "sidecars/PACKED_WIRE_BYTES.jsonl").read_text().splitlines()
    ]
    routing = [
        json.loads(line)
        for line in (tmp_path / "sidecars/EXPERT_ROUTING_IMPORTANCE.jsonl").read_text().splitlines()
    ]
    manifest = json.loads((tmp_path / "sidecars/DIMENSION_SIDECAR_MANIFEST.json").read_text())
    assert rc == 0
    assert result["candidate_count"] == 2
    # Objective insertion order is P2/P13; publication order is canonical candidate_id order.
    assert [row["packed_wire_bytes"] for row in physical] == [16, 8]
    assert {row["routing_importance"] for row in routing} == {7}
    assert manifest["bound_groups"] == [
        "candidate_identity",
        "expert_routing_importance",
        "packed_wire_bytes",
    ]
    assert manifest["blocked_groups"] == [
        "projection_correction",
        "projection_weight",
        "six_class_ceilings",
        "six_class_predictions",
    ]
    assert manifest["allocation_finalized"] is False
    blockers = json.loads((tmp_path / "sidecars/AUTHORITATIVE_PRODUCER_BLOCKERS.json").read_text())
    assert blockers["blocked_groups"]["six_class_predictions"]["expected_authority_path"] == (
        "/authority/six_class_predictions.json"
    )
    assert blockers["blocked_groups"]["six_class_predictions"]["expected_authority_sha256"] is None
    assert blockers["blocked_groups"]["six_class_predictions"]["searched_sources"] == [
        {"path": str(handoff), "sha256": hashlib.sha256(handoff.read_bytes()).hexdigest()}
    ]


def test_solved_sidecars_refuse_handoff_digest_mismatch(tmp_path: Path) -> None:
    handoff = tmp_path / "PRODUCER_HANDOFF.json"
    handoff.write_text("{}")
    with pytest.raises(DynamicDimensionsError, match="handoff SHA-256 mismatch"):
        build_solved_dimension_sidecars(
            handoff=handoff,
            handoff_sha256="0" * 64,
            basis_sha256=BASIS,
            layers=[40],
            output_dir=tmp_path / "sidecars",
        )


def test_solved_sidecars_refuse_missing_packed_wire_metadata(tmp_path: Path) -> None:
    handoff, _ = _solved_sidecar_inputs(tmp_path)
    handoff_value = json.loads(handoff.read_text())
    objective = Path(handoff_value["layer_rows"][0]["members"]["OBJECTIVE"]["path"])
    objective_value = json.loads(objective.read_text())
    del objective_value["assignment"]["L040.E000.P2"]["codeword_assignment_count"]
    objective.write_text(json.dumps(objective_value))
    handoff_value["layer_rows"][0]["members"]["OBJECTIVE"]["sha256"] = hashlib.sha256(
        objective.read_bytes()
    ).hexdigest()
    handoff.write_text(json.dumps(handoff_value))

    with pytest.raises(DynamicDimensionsError, match="wire layout is invalid"):
        build_solved_dimension_sidecars(
            handoff=handoff,
            handoff_sha256=hashlib.sha256(handoff.read_bytes()).hexdigest(),
            basis_sha256=BASIS,
            layers=[40],
            output_dir=tmp_path / "sidecars",
        )


def test_solved_sidecars_require_exact_layer_row_coverage(tmp_path: Path) -> None:
    handoff, _ = _solved_sidecar_inputs(tmp_path)
    second_objective = tmp_path / "OBJECTIVE_SECOND_L040.json"
    second_profile = tmp_path / "PROFILE_ROWS_SECOND_L040.jsonl"
    second_objective.write_text(
        json.dumps(
            {
                "assignment": {
                    projection: {
                        "tier": "d4_k4096",
                        "codeword_assignment_count": 4,
                        "codeword_assignment_dtype": "int16-le",
                        "codeword_assignment_sha256": digest * 64,
                    }
                    for projection, digest in (
                        ("L040.E001.P2", "3"),
                        ("L040.E001.P13", "4"),
                    )
                }
            }
        )
    )
    second_profile.write_text(
        json.dumps({"layer": 40, "expert": 1, "routed_rows": 5}) + "\n"
    )
    handoff_value = json.loads(handoff.read_text())
    handoff_value["layers"] = [40, 41]
    handoff_value["layer_rows"].append(
        {
            "layer": 40,
            "members": {
                "OBJECTIVE": {
                    "path": str(second_objective),
                    "sha256": hashlib.sha256(second_objective.read_bytes()).hexdigest(),
                },
                "PROFILE_ROWS": {
                    "path": str(second_profile),
                    "sha256": hashlib.sha256(second_profile.read_bytes()).hexdigest(),
                },
            },
        }
    )
    handoff.write_text(json.dumps(handoff_value))

    with pytest.raises(DynamicDimensionsError, match="layer rows must cover requested layers exactly"):
        build_solved_dimension_sidecars(
            handoff=handoff,
            handoff_sha256=hashlib.sha256(handoff.read_bytes()).hexdigest(),
            basis_sha256=BASIS,
            layers=[40, 41],
            output_dir=tmp_path / "sidecars",
        )


def test_solved_sidecars_reject_noncanonical_cell_identity(tmp_path: Path) -> None:
    handoff, _ = _solved_sidecar_inputs(tmp_path)
    handoff_value = json.loads(handoff.read_text())
    objective = Path(handoff_value["layer_rows"][0]["members"]["OBJECTIVE"]["path"])
    objective_value = json.loads(objective.read_text())
    objective_value["assignment"]["040.000.P2"] = objective_value["assignment"].pop(
        "L040.E000.P2"
    )
    objective.write_text(json.dumps(objective_value))
    handoff_value["layer_rows"][0]["members"]["OBJECTIVE"]["sha256"] = hashlib.sha256(
        objective.read_bytes()
    ).hexdigest()
    handoff.write_text(json.dumps(handoff_value))

    with pytest.raises(DynamicDimensionsError, match="invalid objective cell identity"):
        build_solved_dimension_sidecars(
            handoff=handoff,
            handoff_sha256=hashlib.sha256(handoff.read_bytes()).hexdigest(),
            basis_sha256=BASIS,
            layers=[40],
            output_dir=tmp_path / "sidecars",
        )


def test_solved_sidecars_require_both_projections_for_each_profile_expert(
    tmp_path: Path,
) -> None:
    handoff, _ = _solved_sidecar_inputs(tmp_path)
    handoff_value = json.loads(handoff.read_text())
    objective = Path(handoff_value["layer_rows"][0]["members"]["OBJECTIVE"]["path"])
    objective_value = json.loads(objective.read_text())
    del objective_value["assignment"]["L040.E000.P13"]
    objective.write_text(json.dumps(objective_value))
    handoff_value["layer_rows"][0]["members"]["OBJECTIVE"]["sha256"] = hashlib.sha256(
        objective.read_bytes()
    ).hexdigest()
    handoff.write_text(json.dumps(handoff_value))

    with pytest.raises(DynamicDimensionsError, match="objective assignment coverage mismatch"):
        build_solved_dimension_sidecars(
            handoff=handoff,
            handoff_sha256=hashlib.sha256(handoff.read_bytes()).hexdigest(),
            basis_sha256=BASIS,
            layers=[40],
            output_dir=tmp_path / "sidecars",
        )


def test_solved_sidecars_validate_expectations_before_publishing(tmp_path: Path) -> None:
    handoff, expectations = _solved_sidecar_inputs(tmp_path)
    expectation_value = json.loads(expectations.read_text())
    expectation_value["schema"] = "invalid"
    expectations.write_text(json.dumps(expectation_value))
    output = tmp_path / "sidecars"

    with pytest.raises(DynamicDimensionsError, match="authority expectations schema/basis mismatch"):
        build_solved_dimension_sidecars(
            handoff=handoff,
            handoff_sha256=hashlib.sha256(handoff.read_bytes()).hexdigest(),
            basis_sha256=BASIS,
            layers=[40],
            output_dir=output,
            authority_expectations=expectations,
        )
    assert not output.exists()


def test_solved_sidecars_reject_boolean_handoff_layer_identity(tmp_path: Path) -> None:
    handoff, _ = _solved_sidecar_inputs(tmp_path)
    handoff_value = json.loads(handoff.read_text())
    handoff_value["layers"] = [True]
    handoff_value["layer_rows"][0]["layer"] = True
    handoff.write_text(json.dumps(handoff_value))

    with pytest.raises(DynamicDimensionsError, match="layer mismatch"):
        build_solved_dimension_sidecars(
            handoff=handoff,
            handoff_sha256=hashlib.sha256(handoff.read_bytes()).hexdigest(),
            basis_sha256=BASIS,
            layers=[1],
            output_dir=tmp_path / "sidecars",
        )


def test_solved_sidecars_reject_duplicate_objective_keys(tmp_path: Path) -> None:
    handoff, _ = _solved_sidecar_inputs(tmp_path)
    handoff_value = json.loads(handoff.read_text())
    objective = Path(handoff_value["layer_rows"][0]["members"]["OBJECTIVE"]["path"])
    assignments = json.loads(objective.read_text())["assignment"]
    p2 = json.dumps(assignments["L040.E000.P2"])
    p13 = json.dumps(assignments["L040.E000.P13"])
    objective.write_text(
        '{"assignment":{'
        f'"L040.E000.P2":{p2},'
        f'"L040.E000.P2":{p2},'
        f'"L040.E000.P13":{p13}'
        '}}'
    )
    handoff_value["layer_rows"][0]["members"]["OBJECTIVE"]["sha256"] = hashlib.sha256(
        objective.read_bytes()
    ).hexdigest()
    handoff.write_text(json.dumps(handoff_value))

    with pytest.raises(DynamicDimensionsError, match="duplicate JSON key"):
        build_solved_dimension_sidecars(
            handoff=handoff,
            handoff_sha256=hashlib.sha256(handoff.read_bytes()).hexdigest(),
            basis_sha256=BASIS,
            layers=[40],
            output_dir=tmp_path / "sidecars",
        )


def test_solved_sidecars_cli_structures_non_string_member_path_failure(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    handoff, _ = _solved_sidecar_inputs(tmp_path)
    handoff_value = json.loads(handoff.read_text())
    handoff_value["layer_rows"][0]["members"]["OBJECTIVE"]["path"] = 7
    handoff.write_text(json.dumps(handoff_value))

    rc = main(
        [
            "backpack-solved-sidecars",
            "--handoff",
            str(handoff),
            "--handoff-sha256",
            hashlib.sha256(handoff.read_bytes()).hexdigest(),
            "--basis-sha256",
            BASIS,
            "--layers",
            "40",
            "--output-dir",
            str(tmp_path / "sidecars"),
        ]
    )
    failure = json.loads(capsys.readouterr().err)
    assert rc == 2
    assert failure["status"] == "FAIL"
    assert failure["error_type"] == "DynamicDimensionsError"
    assert "path must be a non-empty string" in failure["error"]
