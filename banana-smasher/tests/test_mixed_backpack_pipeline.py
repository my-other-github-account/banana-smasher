from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from banana_smasher.backpack_dimensions import (
    preflight_mixed_backpack_config,
    solve_mixed_backpack_config,
)
from banana_smasher import solve_mixed_backpack_config as public_solve_mixed_backpack_config
from banana_smasher import (
    preflight_mixed_backpack_config as public_preflight_mixed_backpack_config,
)
from banana_smasher.cli import main


BASIS = "a" * 64
CLASSES = ("agentic", "chat", "code", "multilingual", "prose", "reasoning")


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n")


def _dimension_rows(path: Path) -> None:
    rows = []
    for layer, tiers in ((0, ("qtip2", "qtip3")), (1, ("qtip2",))):
        for tier in tiers:
            rows.append(
                {
                    "schema": "banana-smasher-dynamic-backpack-candidate-ledger-row-v2",
                    "status": "ADMITTED_COMPLETE_ALLOCATION_ELIGIBLE",
                    "allocation_eligible": True,
                    "basis_sha256": BASIS,
                    "candidate_id": f"L{layer:03d}.E000.down.{tier}",
                    "layer": layer,
                    "expert": 0,
                    "projection": "down",
                    "tier": tier,
                    "physical_bytes": 1 if tier == "qtip2" else 2,
                    "six_class_predictions": {
                        name: 1.0 if tier == "qtip2" else 0.1 for name in CLASSES
                    },
                    "activation_artifacts": [],
                }
            )
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))


def _config(tmp_path: Path, *, target: int) -> Path:
    dimensions = tmp_path / "dimensions.jsonl"
    _dimension_rows(dimensions)
    config = tmp_path / f"mixed-{target}.json"
    _write_json(
        config,
        {
            "schema": "banana-smasher-mixed-backpack-config-v1",
            "basis_sha256": BASIS,
            "target": {
                "whole_model_bytes": target,
                "fixed_nonexpert_bytes": 10,
                "exact": True,
            },
            "allowed_tiers": ["qtip2", "qtip3"],
            "fallback_tier": "qtip2",
            "dimensions": {
                "path": str(dimensions),
                "sha256": hashlib.sha256(dimensions.read_bytes()).hexdigest(),
            },
            "class_caps": {name: 10.0 for name in CLASSES},
        },
    )
    return config


def test_mixed_config_forces_q2_where_q3_inventory_is_missing(tmp_path: Path) -> None:
    assert public_solve_mixed_backpack_config is solve_mixed_backpack_config
    config = _config(tmp_path, target=13)
    output = tmp_path / "solve"

    receipt = solve_mixed_backpack_config(config, output=output)

    assert receipt["byte_accounting"] == {
        "fixed_nonexpert_bytes": 10,
        "candidate_payload_bytes": 3,
        "whole_model_bytes": 13,
        "target_whole_model_bytes": 13,
        "slack_bytes": 0,
    }
    identity = json.loads((output / "identity.json").read_text())
    assert identity["assignment"] == {
        "L000.E000": "qtip3",
        "L001.E000": "qtip2",
    }
    assert identity["coverage"]["qtip3"] == {
        "available_cells": 1,
        "missing_layers": [1],
    }
    assert identity["composition"]["layers"] == [
        {"layer": 0, "tiers": {"qtip3": 1}},
        {"layer": 1, "tiers": {"qtip2": 1}},
    ]


def test_mixed_config_selects_one_tier_per_expert_across_both_projections(
    tmp_path: Path,
) -> None:
    dimensions = tmp_path / "dimensions.jsonl"
    rows = []
    for layer, tiers in ((0, ("qtip2", "qtip3")), (1, ("qtip2",))):
        for projection in ("down", "fused13"):
            for tier in tiers:
                rows.append(
                    {
                        "schema": "banana-smasher-dynamic-backpack-candidate-ledger-row-v2",
                        "status": "ADMITTED_COMPLETE_ALLOCATION_ELIGIBLE",
                        "allocation_eligible": True,
                        "basis_sha256": BASIS,
                        "candidate_id": f"L{layer:03d}.E000.{projection}.{tier}",
                        "layer": layer,
                        "expert": 0,
                        "projection": projection,
                        "tier": tier,
                        "physical_bytes": 1 if tier == "qtip2" else 2,
                        "six_class_predictions": {
                            name: 1.0 if tier == "qtip2" else 0.1 for name in CLASSES
                        },
                        "activation_artifacts": [],
                    }
                )
    dimensions.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
    )
    config = tmp_path / "mixed.json"
    _write_json(
        config,
        {
            "schema": "banana-smasher-mixed-backpack-config-v1",
            "basis_sha256": BASIS,
            "target": {
                "whole_model_bytes": 16,
                "fixed_nonexpert_bytes": 10,
                "exact": True,
            },
            "allowed_tiers": ["qtip2", "qtip3"],
            "fallback_tier": "qtip2",
            "dimensions": {
                "path": str(dimensions),
                "sha256": hashlib.sha256(dimensions.read_bytes()).hexdigest(),
            },
            "class_caps": {name: 10.0 for name in CLASSES},
        },
    )

    solve_mixed_backpack_config(config, output=tmp_path / "solve")

    identity = json.loads((tmp_path / "solve/identity.json").read_text())
    assert identity["assignment"] == {
        "L000.E000": "qtip3",
        "L001.E000": "qtip2",
    }
    assignment = json.loads((tmp_path / "solve/ASSIGNMENT.json").read_text())
    assert assignment["materialization_assignments"] == [
        {"cell_id": "L000.E000.down", "tier": "qtip3"},
        {"cell_id": "L000.E000.fused13", "tier": "qtip3"},
        {"cell_id": "L001.E000.down", "tier": "qtip2"},
        {"cell_id": "L001.E000.fused13", "tier": "qtip2"},
    ]


def test_budget_is_a_config_only_change_and_cli_uses_same_api(
    tmp_path: Path, capsys
) -> None:
    config = _config(tmp_path, target=12)
    output = tmp_path / "solve"

    assert (
        main(
            [
                "backpack",
                "solve-mixed",
                "--config",
                str(config),
                "--output",
                str(output),
            ]
        )
        == 0
    )
    emitted = json.loads(capsys.readouterr().out)
    assert emitted["command"] == "backpack solve-mixed"
    assert emitted["byte_accounting"]["whole_model_bytes"] == 12
    assert json.loads((output / "identity.json").read_text())["assignment"] == {
        "L000.E000": "qtip2",
        "L001.E000": "qtip2",
    }


def test_mixed_config_has_a_public_json_schema(tmp_path: Path) -> None:
    jsonschema = pytest.importorskip("jsonschema")
    schema = json.loads(
        (
            Path(__file__).resolve().parents[1]
            / "schema/banana-smasher-mixed-backpack-config-v1.schema.json"
        ).read_text()
    )

    jsonschema.validate(json.loads(_config(tmp_path, target=13).read_text()), schema)


def test_preflight_accepts_partial_dimensions_and_reports_pending_locator(
    tmp_path: Path, capsys,
) -> None:
    dimensions = tmp_path / "partial.jsonl"
    _dimension_rows(dimensions)
    config = tmp_path / "mixed.json"
    _write_json(
        config,
        {
            "schema": "banana-smasher-mixed-backpack-config-v1",
            "basis_sha256": BASIS,
            "target": {
                "whole_model_bytes": 13,
                "fixed_nonexpert_bytes": 10,
                "exact": True,
            },
            "allowed_tiers": ["qtip2", "qtip3"],
            "fallback_tier": "qtip2",
            "topology": {
                "layers": [0, 1, 2],
                "experts_per_layer": 1,
                "projections": ["down"],
            },
            "dimensions": {
                "sources": [
                    {
                        "path": str(dimensions),
                        "sha256": hashlib.sha256(dimensions.read_bytes()).hexdigest(),
                    },
                    {"locator_path": "final-dimensions-locator.json"},
                ]
            },
            "class_caps": {name: 10.0 for name in CLASSES},
        },
    )

    schema = json.loads(
        (
            Path(__file__).resolve().parents[1]
            / "schema/banana-smasher-mixed-backpack-config-v1.schema.json"
        ).read_text()
    )
    pytest.importorskip("jsonschema").validate(json.loads(config.read_text()), schema)

    receipt = preflight_mixed_backpack_config(config)

    assert public_preflight_mixed_backpack_config is preflight_mixed_backpack_config
    assert receipt["status"] == "WAITING_FOR_DIMENSION_LOCATORS"
    assert receipt["ready_to_solve"] is False
    assert receipt["sources"] == {"admitted": 1, "pending": 1}
    assert receipt["coverage"]["qtip2"]["available_projection_cells"] == 2
    assert receipt["coverage"]["qtip2"]["missing_layers"] == [2]
    assert receipt["missing_fallback_projection_cells"] == ["L002.E000.down"]
    assert main(["backpack", "preflight-mixed", "--config", str(config)]) == 0
    emitted = json.loads(capsys.readouterr().out)
    assert emitted["command"] == "backpack preflight-mixed"
    assert emitted["status"] == "WAITING_FOR_DIMENSION_LOCATORS"


def test_solve_auto_consumes_a_sealed_dimension_locator(tmp_path: Path) -> None:
    partial = tmp_path / "partial.jsonl"
    _dimension_rows(partial)
    final = tmp_path / "final.jsonl"
    final_row = {
        "schema": "banana-smasher-dynamic-backpack-candidate-ledger-row-v2",
        "status": "ADMITTED_COMPLETE_ALLOCATION_ELIGIBLE",
        "allocation_eligible": True,
        "basis_sha256": BASIS,
        "candidate_id": "L002.E000.down.qtip2",
        "layer": 2,
        "expert": 0,
        "projection": "down",
        "tier": "qtip2",
        "physical_bytes": 1,
        "six_class_predictions": {name: 1.0 for name in CLASSES},
        "activation_artifacts": [],
    }
    final.write_text(json.dumps(final_row, sort_keys=True) + "\n")
    locator = tmp_path / "final-locator.json"
    _write_json(
        locator,
        {
            "schema": "banana-smasher-mixed-backpack-dimensions-locator-v1",
            "status": "SEALED",
            "basis_sha256": BASIS,
            "dimensions": {
                "path": final.name,
                "sha256": hashlib.sha256(final.read_bytes()).hexdigest(),
            },
        },
    )
    config = tmp_path / "mixed.json"
    _write_json(
        config,
        {
            "schema": "banana-smasher-mixed-backpack-config-v1",
            "basis_sha256": BASIS,
            "target": {
                "whole_model_bytes": 14,
                "fixed_nonexpert_bytes": 10,
                "exact": True,
            },
            "allowed_tiers": ["qtip2", "qtip3"],
            "fallback_tier": "qtip2",
            "topology": {
                "layers": [0, 1, 2],
                "experts_per_layer": 1,
                "projections": ["down"],
            },
            "dimensions": {
                "sources": [
                    {
                        "path": partial.name,
                        "sha256": hashlib.sha256(partial.read_bytes()).hexdigest(),
                    },
                    {"locator_path": locator.name},
                ]
            },
            "class_caps": {name: 10.0 for name in CLASSES},
        },
    )

    receipt = solve_mixed_backpack_config(config, output=tmp_path / "solve")

    assert receipt["sources"]["admitted"] == 2
    assert receipt["sources"]["pending"] == 0
    assert json.loads((tmp_path / "solve/identity.json").read_text())["assignment"] == {
        "L000.E000": "qtip3",
        "L001.E000": "qtip2",
        "L002.E000": "qtip2",
    }
