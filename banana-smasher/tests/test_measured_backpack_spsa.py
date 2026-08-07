from __future__ import annotations

import json
from pathlib import Path

from banana_smasher.measured_backpack_spsa import (
    CLASSES,
    TIERS,
    build_hierarchical_groups,
    load_full_wire_menu,
    load_routing_usage,
    project_group_logits,
    run_measured_spsa,
)


def _inputs(tmp_path: Path):
    basis = "9" * 64
    ledger = tmp_path / "wire.jsonl"
    rows = []
    for expert in range(24):
        cell_id = f"L000:E{expert:03d}:down"
        for tier_index, tier in enumerate(TIERS):
            artifact = (
                [{"id": "d4-l0-down", "bytes": 7}]
                if tier.startswith("d4_")
                else []
            )
            rows.append(
                {
                    "schema": "banana-smasher-provenance-option-row-v1",
                    "model_id": "deepseek-ai/DeepSeek-V4-Flash-0731",
                    "model_revision": "0731",
                    "basis_sha256": basis,
                    "cell_id": cell_id,
                    "layer": 0,
                    "expert": expert,
                    "projection": "down",
                    "tier": tier,
                    "physical_bytes": 10 + tier_index * 3,
                    "activation_ids": [item["id"] for item in artifact],
                    "activation_artifacts": artifact,
                    # This forbidden historical proxy is present in the source and
                    # must not enter the measured optimizer's in-memory wire menu.
                    "prediction_by_class": {name: 999.0 for name in CLASSES},
                }
            )
    ledger.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
    routing_path = tmp_path / "routing.json"
    routing_path.write_text(
        json.dumps(
            {
                "schema": "banana-smasher-ff0731-routing-usage-v1",
                "status": "PASS",
                "basis_sha256": basis,
                "cells": [
                    {
                        "cell_id": f"L000:E{expert:03d}:down",
                        "usage_by_class": {
                            name: float(20 if index == expert % len(CLASSES) else 1)
                            + expert / 100
                            for index, name in enumerate(CLASSES)
                        },
                    }
                    for expert in range(24)
                ],
            },
            sort_keys=True,
        )
        + "\n"
    )
    menu, activation_bytes, ledger_evidence = load_full_wire_menu(
        ledger, expected_basis_sha256=basis
    )
    routing, routing_evidence = load_routing_usage(
        routing_path,
        expected_basis_sha256=basis,
        expected_cells=set(menu),
    )
    return menu, activation_bytes, routing, ledger_evidence, routing_evidence


def test_projection_uses_measured_group_logits_and_full_wire_only(tmp_path: Path) -> None:
    menu, activation_bytes, routing, ledger_evidence, routing_evidence = _inputs(tmp_path)
    groups = build_hierarchical_groups(routing)
    logits = {group: [float(index) for index in range(len(TIERS))] for group in set(groups.values())}

    result = project_group_logits(
        menu,
        activation_bytes,
        groups,
        logits,
        shipping_bytes=500,
        fixed_nonexpert_bytes=100,
        routing=routing,
    )

    assert ledger_evidence["quality_coefficients_loaded"] is False
    assert routing_evidence["use"] == "grouping-metadata-only"
    assert result["whole_model_accounting"]["whole_shipping_bytes"] <= 500
    assert result["whole_model_accounting"]["repair_bytes"] == 0
    assert sum(result["tier_counts"].values()) == 24
    assert result["projection"]["quality_signal"] == "measured-group-logits-only"
    assert all("prediction_by_class" not in row for row in result["assignments"])


def test_spsa_runs_antithetic_pairs_rotates_train_and_freezes_best(tmp_path: Path) -> None:
    menu, activation_bytes, routing, _, _ = _inputs(tmp_path)
    slices = [
        {
            "slice_id": f"train-{index}",
            "window_ids": [f"w{index}-{slot}" for slot in range(8)],
            "holdout_used": False,
        }
        for index in range(2)
    ]
    calls = []

    def evaluator(assignment_path, train_slice, output_path):
        assignment = json.loads(assignment_path.read_text())
        counts = assignment["tier_counts"]
        measured = sum((index + 1) * counts[tier] for index, tier in enumerate(TIERS)) / 10_000
        calls.append((assignment["assignment_sha256"], train_slice["slice_id"]))
        value = {
            "schema": "test-eight-window-measurement-v1",
            "status": "PASS",
            "assignment_sha256": assignment["assignment_sha256"],
            "slice_id": train_slice["slice_id"],
            "window_ids": train_slice["window_ids"],
            "windows": 8,
            "mean_kld": measured,
            "class_kld": {
                name: measured + class_index / 100_000
                for class_index, name in enumerate(CLASSES)
            },
            "top1_agreement": 0.95 - measured / 100,
        }
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(value, sort_keys=True) + "\n")
        return value

    result = run_measured_spsa(
        menu,
        activation_bytes,
        routing,
        slices,
        evaluator,
        tmp_path / "run",
        coarse_iterations=1,
        refine_iterations=1,
        seed=17,
    )

    assert result["status"] == "PASS_FROZEN_PRE_REPAIR"
    assert result["evaluations"] == 4
    assert [slice_id for _, slice_id in calls] == [
        "train-0",
        "train-0",
        "train-1",
        "train-1",
    ]
    frozen = json.loads((tmp_path / "run" / "FROZEN_ASSIGNMENT.json").read_text())
    assert frozen["status"] == "PASS_FROZEN_PRE_REPAIR"
    assert frozen["whole_model_accounting"]["shipping_bytes_cap"] == 102_000_000_000
    assert frozen["whole_model_accounting"]["fixed_nonexpert_bytes"] == 9_032_112_614
    assert frozen["whole_model_accounting"]["repair_bytes"] == 0
    assert (tmp_path / "run" / "SEARCH_TERMINAL.json").is_file()
