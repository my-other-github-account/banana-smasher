from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from banana_smasher.backpack_dimensions import solve_mixed_backpack_config
from banana_smasher import solve_mixed_backpack_config as public_solve_mixed_backpack_config
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
