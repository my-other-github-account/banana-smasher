from __future__ import annotations

import json

from banana_smasher.backpack_contextual import (
    build_contextual_delta_ledger,
    run_contextual_value_update,
    run_contextual_trust_solve,
    solve_contextual_trust_region,
)
from banana_smasher.cli import main


def test_contextual_value_update_is_public_api() -> None:
    import banana_smasher

    assert banana_smasher.run_contextual_value_update is run_contextual_value_update
    assert banana_smasher.run_contextual_trust_solve is run_contextual_trust_solve


def test_physical_alias_has_zero_contextual_delta_without_measurement() -> None:
    anchor = {
        "schema": "banana-smasher-contextual-anchor-v1",
        "status": "PASS",
        "assignment_sha256": "a" * 64,
        "physical_score_receipt_sha256": "b" * 64,
        "cells": [
            {
                "cell": "L00/E000_fused13",
                "option": "qtip2",
                "physical_identity": "c" * 64,
                "payload_bytes": 1024,
            }
        ],
    }
    options = [
        {
            "cell": "L00/E000_fused13",
            "option": "qtip2.5",
            "physical_identity": "c" * 64,
            "payload_bytes": 1024,
        }
    ]

    ledger = build_contextual_delta_ledger(anchor, options, measurements=[])

    assert ledger["rows"] == [
        {
            "cell": "L00/E000_fused13",
            "option": "qtip2.5",
            "physical_identity": "c" * 64,
            "payload_bytes": 1024,
            "eligible": True,
            "delta_mean_kld": 0.0,
            "delta_top1_matches": 0,
            "valuation_source": "physical-alias-invariance",
            "measurement_receipt_sha256": None,
        }
    ]


def test_unmeasured_distinct_option_is_not_solver_eligible() -> None:
    anchor = {
        "schema": "banana-smasher-contextual-anchor-v1",
        "status": "PASS",
        "assignment_sha256": "a" * 64,
        "physical_score_receipt_sha256": "b" * 64,
        "cells": [
            {
                "cell": "L00/E000_fused13",
                "option": "qtip2",
                "physical_identity": "c" * 64,
                "payload_bytes": 1024,
            }
        ],
    }
    options = [
        {
            "cell": "L00/E000_fused13",
            "option": "qtip3",
            "physical_identity": "d" * 64,
            "payload_bytes": 1536,
        }
    ]

    ledger = build_contextual_delta_ledger(anchor, options, measurements=[])

    assert ledger["rows"][0]["eligible"] is False
    assert ledger["rows"][0]["delta_mean_kld"] is None
    assert ledger["rows"][0]["valuation_source"] == "unmeasured"


def test_matching_physical_swap_receipt_supplies_contextual_delta() -> None:
    anchor = {
        "schema": "banana-smasher-contextual-anchor-v1",
        "status": "PASS",
        "assignment_sha256": "a" * 64,
        "physical_score_receipt_sha256": "b" * 64,
        "cells": [
            {
                "cell": "L00/E000_fused13",
                "option": "qtip2",
                "physical_identity": "c" * 64,
                "payload_bytes": 1024,
            }
        ],
    }
    options = [
        {
            "cell": "L00/E000_fused13",
            "option": "qtip3",
            "physical_identity": "d" * 64,
            "payload_bytes": 1536,
            "activations": [{"id": "shared", "bytes": 4096}],
        }
    ]
    measurements = [
        {
            "schema": "banana-smasher-contextual-swap-measurement-v1",
            "status": "PASS",
            "receipt_sha256": "e" * 64,
            "anchor_assignment_sha256": "a" * 64,
            "scope": "screen",
            "windows": 8,
            "positions": 8192,
            "support_width": 8192,
            "change": {
                "cell": "L00/E000_fused13",
                "physical_identity": "d" * 64,
            },
            "delta_mean_kld": -0.002,
            "delta_top1_matches": 3,
            "stderr_mean_kld": 0.0002,
        }
    ]

    ledger = build_contextual_delta_ledger(anchor, options, measurements)

    row = ledger["rows"][0]
    assert row["eligible"] is True
    assert row["delta_mean_kld"] == -0.002
    assert row["delta_top1_matches"] == 3
    assert row["stderr_mean_kld"] == 0.0002
    assert row["valuation_source"] == "physical-swap-receipt"
    assert row["measurement_receipt_sha256"] == "e" * 64
    assert row["activations"] == [{"id": "shared", "bytes": 4096}]


def test_file_api_is_byte_repeatable_from_versioned_inputs(tmp_path) -> None:
    anchor = {
        "schema": "banana-smasher-contextual-anchor-v1",
        "status": "PASS",
        "assignment_sha256": "a" * 64,
        "physical_score_receipt_sha256": "b" * 64,
        "cells": [
            {
                "cell": "any-cell",
                "option": "incumbent",
                "physical_identity": "c" * 64,
                "payload_bytes": 10,
            }
        ],
    }
    inventory = {
        "schema": "banana-smasher-contextual-option-inventory-v1",
        "status": "READY",
        "options": [
            {
                "cell": "any-cell",
                "option": "new-logical-label",
                "physical_identity": "c" * 64,
                "payload_bytes": 10,
            }
        ],
    }
    measurements = {
        "schema": "banana-smasher-contextual-measurement-manifest-v1",
        "status": "READY",
        "measurements": [],
    }
    paths = []
    for name, value in (
        ("anchor.json", anchor),
        ("inventory.json", inventory),
        ("measurements.json", measurements),
    ):
        path = tmp_path / name
        path.write_text(json.dumps(value, sort_keys=True) + "\n")
        paths.append(path)

    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    run_contextual_value_update(*paths, output_path=first)
    run_contextual_value_update(*paths, output_path=second)

    assert first.read_bytes() == second.read_bytes()
    assert json.loads(first.read_text())["status"] == "PASS"


def test_cli_runs_the_same_contextual_value_api(tmp_path, capsys) -> None:
    values = {
        "anchor.json": {
            "schema": "banana-smasher-contextual-anchor-v1",
            "status": "PASS",
            "assignment_sha256": "a" * 64,
            "physical_score_receipt_sha256": "b" * 64,
            "cells": [
                {
                    "cell": "cell",
                    "option": "old",
                    "physical_identity": "c" * 64,
                    "payload_bytes": 10,
                }
            ],
        },
        "inventory.json": {
            "schema": "banana-smasher-contextual-option-inventory-v1",
            "status": "READY",
            "options": [
                {
                    "cell": "cell",
                    "option": "alias",
                    "physical_identity": "c" * 64,
                    "payload_bytes": 10,
                }
            ],
        },
        "measurements.json": {
            "schema": "banana-smasher-contextual-measurement-manifest-v1",
            "status": "READY",
            "measurements": [],
        },
    }
    for name, value in values.items():
        (tmp_path / name).write_text(json.dumps(value, sort_keys=True) + "\n")
    output = tmp_path / "ledger.json"

    status = main(
        [
            "backpack",
            "value-contextual",
            "--anchor",
            str(tmp_path / "anchor.json"),
            "--options",
            str(tmp_path / "inventory.json"),
            "--measurements",
            str(tmp_path / "measurements.json"),
            "--output",
            str(output),
        ]
    )

    emitted = json.loads(capsys.readouterr().out)
    assert status == 0
    assert emitted["status"] == "PASS"
    assert emitted["command"] == "backpack value-contextual"
    assert output.is_file()


def test_trust_region_respects_exact_bytes_and_retains_incumbents() -> None:
    anchor = {
        "schema": "banana-smasher-contextual-anchor-v1",
        "status": "PASS",
        "assignment_sha256": "a" * 64,
        "physical_score_receipt_sha256": "b" * 64,
        "fixed_bytes": 10,
        "package_cap_bytes": 100,
        "cells": [
            {
                "cell": "A",
                "option": "old-a",
                "physical_identity": "1" * 64,
                "payload_bytes": 40,
            },
            {
                "cell": "B",
                "option": "old-b",
                "physical_identity": "2" * 64,
                "payload_bytes": 40,
            },
        ],
    }
    ledger = {
        "schema": "banana-smasher-contextual-delta-ledger-v1",
        "status": "PASS",
        "anchor_assignment_sha256": "a" * 64,
        "rows": [
            {
                "cell": "A",
                "option": "new-a",
                "physical_identity": "3" * 64,
                "payload_bytes": 45,
                "eligible": True,
                "delta_mean_kld": -0.1,
                "stderr_mean_kld": 0.0,
            },
            {
                "cell": "B",
                "option": "new-b",
                "physical_identity": "4" * 64,
                "payload_bytes": 60,
                "eligible": True,
                "delta_mean_kld": -0.2,
                "stderr_mean_kld": 0.0,
            },
        ],
    }

    result = solve_contextual_trust_region(
        anchor,
        ledger,
        max_changes=1,
        uncertainty_multiplier=0.0,
        time_limit_seconds=5.0,
    )

    selected = {row["cell"]: row["option"] for row in result["assignment"]}
    assert selected == {"A": "new-a", "B": "old-b"}
    assert result["changed_cells"] == 1
    assert result["package_bytes"] == 95
    assert result["predicted_delta_mean_kld"] == -0.1


def test_trust_solve_file_api_writes_content_bound_receipt(tmp_path) -> None:
    anchor = {
        "schema": "banana-smasher-contextual-anchor-v1",
        "status": "PASS",
        "assignment_sha256": "a" * 64,
        "fixed_bytes": 0,
        "package_cap_bytes": 10,
        "cells": [
            {
                "cell": "generic-cell",
                "option": "incumbent",
                "physical_identity": "1" * 64,
                "payload_bytes": 10,
            }
        ],
    }
    ledger = {
        "schema": "banana-smasher-contextual-delta-ledger-v1",
        "status": "PASS",
        "anchor_assignment_sha256": "a" * 64,
        "rows": [
            {
                "cell": "generic-cell",
                "option": "measured-substitute",
                "physical_identity": "2" * 64,
                "payload_bytes": 8,
                "eligible": True,
                "delta_mean_kld": -0.01,
                "stderr_mean_kld": 0.001,
            }
        ],
    }
    anchor_path = tmp_path / "anchor.json"
    ledger_path = tmp_path / "ledger.json"
    output_path = tmp_path / "solve.json"
    anchor_path.write_text(json.dumps(anchor, sort_keys=True) + "\n")
    ledger_path.write_text(json.dumps(ledger, sort_keys=True) + "\n")

    receipt = run_contextual_trust_solve(
        anchor_path,
        ledger_path,
        output_path=output_path,
        max_changes=1,
        uncertainty_multiplier=2.0,
        time_limit_seconds=5.0,
    )

    solved = json.loads(output_path.read_text())
    assert receipt["status"] == "PASS"
    assert solved["status"] == "PASS"
    assert solved["assignment"][0]["option"] == "measured-substitute"
    assert [row["role"] for row in solved["input_bindings"]] == ["anchor", "ledger"]


def test_cli_delegates_contextual_trust_solve_to_file_api(
    tmp_path, capsys, monkeypatch
) -> None:
    observed = {}

    def fake_run(anchor, ledger, **parameters):
        observed.update(
            anchor=anchor,
            ledger=ledger,
            **parameters,
        )
        return {"status": "PASS", "assignment_sha256": "f" * 64}

    monkeypatch.setattr(
        "banana_smasher.backpack_contextual.run_contextual_trust_solve",
        fake_run,
    )
    status = main(
        [
            "backpack",
            "solve-contextual",
            "--anchor",
            str(tmp_path / "anchor.json"),
            "--ledger",
            str(tmp_path / "ledger.json"),
            "--max-changes",
            "7",
            "--uncertainty-multiplier",
            "1.5",
            "--time-limit-seconds",
            "9",
            "--output",
            str(tmp_path / "solve.json"),
        ]
    )

    emitted = json.loads(capsys.readouterr().out)
    assert status == 0
    assert emitted["command"] == "backpack solve-contextual"
    assert observed["max_changes"] == 7
    assert observed["uncertainty_multiplier"] == 1.5
    assert observed["time_limit_seconds"] == 9.0


def test_trust_region_charges_shared_activation_once() -> None:
    anchor = {
        "schema": "banana-smasher-contextual-anchor-v1",
        "status": "PASS",
        "assignment_sha256": "a" * 64,
        "fixed_bytes": 0,
        "package_cap_bytes": 20,
        "cells": [
            {
                "cell": cell,
                "option": "old",
                "physical_identity": identity * 64,
                "payload_bytes": 5,
            }
            for cell, identity in (("A", "1"), ("B", "2"))
        ],
    }
    ledger = {
        "schema": "banana-smasher-contextual-delta-ledger-v1",
        "status": "PASS",
        "anchor_assignment_sha256": "a" * 64,
        "rows": [
            {
                "cell": cell,
                "option": "new",
                "physical_identity": identity * 64,
                "payload_bytes": 5,
                "activation": {"id": "shared-codebook", "bytes": 8},
                "eligible": True,
                "delta_mean_kld": -0.2,
                "stderr_mean_kld": 0.0,
            }
            for cell, identity in (("A", "3"), ("B", "4"))
        ],
    }

    result = solve_contextual_trust_region(
        anchor,
        ledger,
        max_changes=2,
        uncertainty_multiplier=0.0,
        time_limit_seconds=5.0,
    )

    assert result["changed_cells"] == 2
    assert result["package_bytes"] == 18
    assert {row["activation"]["id"] for row in result["assignment"]} == {
        "shared-codebook"
    }


def test_trust_region_supports_multiple_activations_per_option() -> None:
    anchor = {
        "schema": "banana-smasher-contextual-anchor-v1",
        "status": "PASS",
        "assignment_sha256": "a" * 64,
        "fixed_bytes": 0,
        "package_cap_bytes": 12,
        "cells": [
            {
                "cell": "cell",
                "option": "old",
                "physical_identity": "1" * 64,
                "payload_bytes": 5,
            }
        ],
    }
    ledger = {
        "schema": "banana-smasher-contextual-delta-ledger-v1",
        "status": "PASS",
        "anchor_assignment_sha256": "a" * 64,
        "rows": [
            {
                "cell": "cell",
                "option": "new",
                "physical_identity": "2" * 64,
                "payload_bytes": 5,
                "activations": [
                    {"id": "first", "bytes": 3},
                    {"id": "second", "bytes": 4},
                ],
                "eligible": True,
                "delta_mean_kld": -0.1,
                "stderr_mean_kld": 0.0,
            }
        ],
    }

    result = solve_contextual_trust_region(
        anchor,
        ledger,
        max_changes=1,
        uncertainty_multiplier=0.0,
        time_limit_seconds=5.0,
    )

    assert result["package_bytes"] == 12
    assert {row["id"] for row in result["assignment"][0]["activations"]} == {
        "first",
        "second",
    }
