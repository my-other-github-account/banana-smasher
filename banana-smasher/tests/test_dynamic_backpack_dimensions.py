from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from banana_smasher.backpack_dimensions import (
    CLASSES,
    DynamicDimensionsError,
    build_dynamic_dimensions,
)
from banana_smasher.cli import main

BASIS = "a" * 64


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))


def _source_reference(path: Path) -> dict[str, str]:
    return {
        "path": str(path.resolve()),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def _inputs(tmp_path: Path) -> tuple[Path, Path, Path]:
    ledger = tmp_path / "ledger.jsonl"
    dimensions = tmp_path / "dimensions.jsonl"
    ceilings = tmp_path / "ceilings.json"
    prediction_source = tmp_path / "six-class-predictions.jsonl"
    projection_source = tmp_path / "projection-corrections.jsonl"
    physical_source = tmp_path / "packed-physical-bytes.jsonl"
    prediction_source.write_text('{"source":"six-class-predictions"}\n')
    projection_source.write_text('{"source":"projection-corrections"}\n')
    physical_source.write_text('{"source":"packed-physical-bytes"}\n')
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
    authority = {
        "six_class_predictions": _source_reference(prediction_source),
        "projection_correction": _source_reference(projection_source),
        "physical_bytes": _source_reference(physical_source),
        "six_class_ceilings": _source_reference(ceilings),
        "routing_importance_sha256": "2" * 64,
    }
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
                "authority": authority,
            }
        )
    _write_jsonl(ledger, candidate_rows)
    _write_jsonl(dimensions, dimension_rows)
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
    assert all(
        set(row["dimension_authority"][group]) == {"path", "sha256"}
        for row in rows
        for group in (
            "six_class_predictions",
            "six_class_ceilings",
            "projection_correction",
            "physical_bytes",
        )
    )
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


def test_missing_authority_source_regular_file_fails_closed(tmp_path: Path) -> None:
    ledger, dimensions, ceilings = _inputs(tmp_path)
    rows = [json.loads(line) for line in dimensions.read_text().splitlines()]
    source = Path(rows[0]["authority"]["six_class_predictions"]["path"])
    source.unlink()
    _write_jsonl(dimensions, rows)

    with pytest.raises(DynamicDimensionsError, match="six_class_predictions authority source"):
        build_dynamic_dimensions(
            ledger=ledger,
            dimensions=dimensions,
            class_ceilings=ceilings,
            basis_sha256=BASIS,
            output=tmp_path / "out.jsonl",
            receipt=tmp_path / "receipt.json",
        )


def test_authority_source_hash_mismatch_fails_closed(tmp_path: Path) -> None:
    ledger, dimensions, ceilings = _inputs(tmp_path)
    rows = [json.loads(line) for line in dimensions.read_text().splitlines()]
    rows[0]["authority"]["projection_correction"]["sha256"] = "0" * 64
    _write_jsonl(dimensions, rows)

    with pytest.raises(DynamicDimensionsError, match="authority source SHA-256 mismatch"):
        build_dynamic_dimensions(
            ledger=ledger,
            dimensions=dimensions,
            class_ceilings=ceilings,
            basis_sha256=BASIS,
            output=tmp_path / "out.jsonl",
            receipt=tmp_path / "receipt.json",
        )


def test_symlink_authority_source_fails_closed(tmp_path: Path) -> None:
    ledger, dimensions, ceilings = _inputs(tmp_path)
    rows = [json.loads(line) for line in dimensions.read_text().splitlines()]
    reference = rows[0]["authority"]["physical_bytes"]
    source = Path(reference["path"])
    target = source.with_suffix(".target")
    source.rename(target)
    source.symlink_to(target)
    _write_jsonl(dimensions, rows)

    with pytest.raises(DynamicDimensionsError, match="not a readable regular file"):
        build_dynamic_dimensions(
            ledger=ledger,
            dimensions=dimensions,
            class_ceilings=ceilings,
            basis_sha256=BASIS,
            output=tmp_path / "out.jsonl",
            receipt=tmp_path / "receipt.json",
        )


def test_duplicate_json_key_in_authority_row_fails_closed(tmp_path: Path) -> None:
    ledger, dimensions, ceilings = _inputs(tmp_path)
    lines = dimensions.read_text().splitlines()
    lines[0] = lines[0].replace("{", '{"candidate_id":"forged",', 1)
    dimensions.write_text("\n".join(lines) + "\n")

    with pytest.raises(DynamicDimensionsError, match="duplicate JSON key"):
        build_dynamic_dimensions(
            ledger=ledger,
            dimensions=dimensions,
            class_ceilings=ceilings,
            basis_sha256=BASIS,
            output=tmp_path / "out.jsonl",
            receipt=tmp_path / "receipt.json",
        )


def test_public_authority_schema_requires_path_and_sha_for_each_dimension() -> None:
    schema_path = (
        Path(__file__).parents[1]
        / "schema"
        / "bs-dynamic-backpack-authority-v1.schema.json"
    )
    schema = json.loads(schema_path.read_text())

    assert schema["$id"].endswith("bs-dynamic-backpack-authority-v1.schema.json")
    assert set(schema["required"]) == {
        "six_class_predictions",
        "six_class_ceilings",
        "projection_correction",
        "physical_bytes",
        "routing_importance_sha256",
    }
    for group in (
        "six_class_predictions",
        "six_class_ceilings",
        "projection_correction",
        "physical_bytes",
    ):
        reference = schema["properties"][group]
        assert reference["required"] == ["path", "sha256"]
        assert reference["properties"]["path"]["pattern"] == "^/"
        assert reference["properties"]["sha256"]["pattern"] == "^[0-9a-f]{64}$"


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
