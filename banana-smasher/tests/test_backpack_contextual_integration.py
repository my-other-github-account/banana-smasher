from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from banana_smasher import (
    bq23_backpack_family_providers,
    build_backpack,
    build_contextual_delta_ledger,
    materialize_virtual_backpack,
    select_measured_nonworse,
    solve_contextual_trust_region,
)
from banana_smasher.cli import main
from banana_smasher.locality import require_local_path
from banana_smasher.staging import stage_qsfp_manifest
from test_backpack_framework import _fixture_plan


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, sort_keys=True) + "\n")


def test_bq23_taxonomy_uses_canonical_provider_objects() -> None:
    providers = bq23_backpack_family_providers()

    assert list(providers) == [
        "native-mxfp4",
        "qtip@2.00",
        "qtip@3.00",
        "d4-k2048",
        "d4-k4096",
    ]
    assert providers["qtip@2.00"].runtime_family == "qtip2"
    assert providers["qtip@3.00"].runtime_family == "qtip3"
    assert all(provider.generate and provider.materialize for provider in providers.values())


def test_locality_rejects_remote_mount_and_explicit_stage_fan_in_is_collision_free(
    tmp_path: Path,
) -> None:
    remote = tmp_path / "remote"
    remote.mkdir()
    mountinfo = tmp_path / "mountinfo"
    mountinfo.write_text(
        f"1 0 0:1 / {tmp_path} rw - ext4 /dev/nvme0n1 rw\n"
        f"2 1 0:2 / {remote} rw - fuse.sshfs user@192.0.2.1:/source rw\n"
    )
    with pytest.raises(ValueError, match="explicit QSFP staging API"):
        require_local_path(remote, label="payload", mountinfo_path=mountinfo)

    manifest = tmp_path / "stage.json"
    _write_json(
        manifest,
        {
            "schema": "banana-smasher-qsfp-stage-v2",
            "status": "READY",
            "fabric_cidr": "192.0.2.0/24",
            "items": [
                {
                    "source_host": "user@192.0.2.1",
                    "source_root": "/source-a",
                    "destination": "fan-in",
                    "relative_paths": ["L000/E000_down/QTIP_UNIT.pt"],
                    "bytes": 1,
                },
                {
                    "source_host": "user@192.0.2.2",
                    "source_root": "/source-b",
                    "destination": "fan-in",
                    "relative_paths": ["L001/E000_down/QTIP_UNIT.pt"],
                    "bytes": 1,
                },
            ],
        },
    )

    def transfer(item: dict[str, object], output: Path) -> dict[str, object]:
        for relative in item["relative_paths"]:  # type: ignore[index]
            target = output / str(item["destination"]) / str(relative)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(b"x")
        return {**item, "actual_bytes": 1, "status": "PASS"}

    receipt = stage_qsfp_manifest(manifest, tmp_path / "staged", transfer=transfer)
    assert receipt["status"] == "PASS"
    assert receipt["bytes"] == 2
    assert len(receipt["items"]) == 2

    collision = json.loads(manifest.read_text())
    collision["items"][1]["relative_paths"] = ["L000/E000_down/QTIP_UNIT.pt"]
    _write_json(manifest, collision)
    with pytest.raises(ValueError, match="duplicate destination files"):
        stage_qsfp_manifest(manifest, tmp_path / "collision", transfer=transfer)

    collision["items"][0]["destination"] = "fan-in"
    collision["items"][0]["relative_paths"] = ["nested/QTIP_UNIT.pt"]
    collision["items"][1]["destination"] = "fan-in/nested"
    collision["items"][1]["relative_paths"] = ["QTIP_UNIT.pt"]
    _write_json(manifest, collision)
    with pytest.raises(ValueError, match="duplicate destination files"):
        stage_qsfp_manifest(manifest, tmp_path / "nested-collision", transfer=transfer)

    collision["items"][0]["destination"] = "fan"
    collision["items"][0]["relative_paths"] = ["node"]
    collision["items"][1]["destination"] = "fan/node"
    collision["items"][1]["relative_paths"] = ["child"]
    _write_json(manifest, collision)
    with pytest.raises(ValueError, match="overlapping destination files"):
        stage_qsfp_manifest(manifest, tmp_path / "prefix-collision", transfer=transfer)


def test_measured_selection_retains_baseline_on_proxy_reversal(tmp_path: Path) -> None:
    basis = "a" * 64
    solve = {
        "basis_sha256": basis,
        "arms": {
            "baseline": {
                "tiers": ["native_mxfp4", "qtip2", "qtip3"],
                "assignment_map_sha256": "b" * 64,
                "objective": {"value": 0.04},
            },
            "expanded": {
                "tiers": ["native_mxfp4", "qtip2", "qtip2_5", "qtip3"],
                "assignment_map_sha256": "c" * 64,
                "objective": {"value": 0.03},
            },
        },
    }
    common = {
        "status": "PASS",
        "basis_sha256": basis,
        "bank_sha256": "d" * 64,
        "windows": 64,
        "positions": 65536,
        "support_width": 8192,
    }
    paths = []
    for name, value in (
        ("solve.json", solve),
        (
            "baseline.json",
            {
                **common,
                "assignment_map_sha256": "b" * 64,
                "mean_kld": 0.10,
                "top1_matches": 50000,
            },
        ),
        (
            "expanded.json",
            {
                **common,
                "assignment_map_sha256": "c" * 64,
                "mean_kld": 0.12,
                "top1_matches": 49900,
            },
        ),
    ):
        path = tmp_path / name
        _write_json(path, value)
        paths.append(path)

    receipt = select_measured_nonworse(
        paths[0],
        paths[1],
        paths[2],
        tmp_path / "selection.json",
        baseline_arm="baseline",
        expanded_arm="expanded",
    )

    assert receipt["decision"] == "RETAIN_BASELINE"
    assert receipt["chosen_assignment_map_sha256"] == "b" * 64
    assert receipt["proxy"]["ordering_agrees_with_measurement"] is False

    unbound = json.loads(paths[2].read_text())
    unbound["assignment_map_sha256"] = "b" * 64
    _write_json(paths[2], unbound)
    with pytest.raises(ValueError, match="does not bind its solve-arm assignment"):
        select_measured_nonworse(
            paths[0],
            paths[1],
            paths[2],
            tmp_path / "unbound-selection.json",
            baseline_arm="baseline",
            expanded_arm="expanded",
        )


def test_contextual_ledger_and_trust_region_use_physical_measurements() -> None:
    anchor = {
        "schema": "banana-smasher-contextual-anchor-v1",
        "status": "PASS",
        "basis_sha256": "0" * 64,
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
    options = [
        {
            "cell": "A",
            "option": "new-a",
            "physical_identity": "3" * 64,
            "payload_bytes": 45,
        },
        {
            "cell": "B",
            "option": "alias-b",
            "physical_identity": "2" * 64,
            "payload_bytes": 40,
        },
    ]
    measurement = {
        "schema": "banana-smasher-contextual-swap-measurement-v1",
        "status": "PASS",
        "anchor_assignment_sha256": "a" * 64,
        "candidate_assignment_sha256": "c" * 64,
        "anchor_score_sha256": "b" * 64,
        "candidate_score_sha256": "d" * 64,
        "candidate_pack_sha256": "e" * 64,
        "change": {"cell": "A", "physical_identity": "3" * 64},
        "delta_mean_kld": -0.1,
        "delta_top1_matches": 2,
        "stderr_mean_kld": 0.0,
    }
    measurement["receipt_sha256"] = hashlib.sha256(
        json.dumps(measurement, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    measurements = [measurement]

    ledger = build_contextual_delta_ledger(anchor, options, measurements)
    assert [row["valuation_source"] for row in ledger["rows"]] == [
        "physical-swap-receipt",
        "physical-alias-invariance",
    ]

    solved = solve_contextual_trust_region(
        anchor,
        ledger,
        max_changes=1,
        uncertainty_multiplier=0.0,
        time_limit_seconds=5.0,
    )
    assert {row["cell"]: row["option"] for row in solved["assignment"]} == {
        "A": "new-a",
        "B": "old-b",
    }
    assert solved["package_bytes"] == 95


def test_public_build_backpack_composes_with_cli_contextual_prepare(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    plan = _fixture_plan(tmp_path, exact_bytes=53344)
    run_root = tmp_path / "run"
    built = build_backpack(plan, run_root=run_root)
    assert built["status"] == "PASS"

    virtual_root = tmp_path / "virtual"
    virtual = materialize_virtual_backpack(run_root, virtual_root)
    assert virtual["status"] == "PASS"
    manifest_path = virtual_root / "BACKPACK_VIRTUAL_MANIFEST.json"
    manifest = json.loads(manifest_path.read_text())

    score_identities = {
        "bank_sha256": "1" * 64,
        "basis_sha256": manifest["basis_sha256"],
        "model_id": "fixture-model",
        "teacher_manifest_sha256": "2" * 64,
        "teacher_sha256": "3" * 64,
    }
    anchor_score_path = tmp_path / "anchor-score.json"
    _write_json(
        anchor_score_path,
        {
            "schema": "banana-smasher-anchor-sidecar-score-v1",
            "status": "PASS",
            "claimable": True,
            "support_width": 8192,
            "windows": 64,
            "positions": 65536,
            "mean_kld": 0.2,
            "top1_matches": 51200,
            "identities": {
                **score_identities,
                "pack_sha256": virtual["artifact_sha256"],
            },
            "per_window": [
                {
                    "window_id": f"w{index:02d}",
                    "positions": 1024,
                    "mean_kld": 0.2,
                    "top1_matches": 800,
                }
                for index in range(64)
            ],
        },
    )
    score_path = tmp_path / "exact64.json"
    assert main(
        [
            "backpack",
            "bind-exact64",
            "--virtual-manifest",
            str(manifest_path),
            "--score-receipt",
            str(anchor_score_path),
            "--output",
            str(score_path),
        ]
    ) == 0
    exact64_emitted = json.loads(capsys.readouterr().out)
    assert exact64_emitted["status"] == "PASS"
    assert exact64_emitted["command"] == "backpack bind-exact64"
    prepared = tmp_path / "prepared"
    status = main(
        [
            "backpack",
            "prepare-contextual",
            "--virtual-manifest",
            str(manifest_path),
            "--score-receipt",
            str(score_path),
            "--output",
            str(prepared),
        ]
    )

    emitted = json.loads(capsys.readouterr().out)
    assert status == 0
    assert emitted["status"] == "PASS"
    assert emitted["command"] == "backpack prepare-contextual"
    anchor = json.loads((prepared / "ANCHOR.json").read_text())
    inventory = json.loads((prepared / "OPTION_INVENTORY.json").read_text())
    assert anchor["assignment_sha256"] == manifest["assignment_map_sha256"]
    assert len(anchor["cells"]) == 2
    assert len(inventory["options"]) == 6
    assert hashlib.sha256(anchor_score_path.read_bytes()).hexdigest() == anchor[
        "physical_score_receipt_sha256"
    ]

    incumbent = {row["cell"]: row["option"] for row in anchor["cells"]}
    target = next(
        row
        for row in inventory["options"]
        if row["option"] != incumbent[row["cell"]]
    )
    request_path = tmp_path / "change-request.json"
    _write_json(
        request_path,
        {
            "schema": "banana-smasher-contextual-change-request-v1",
            "status": "READY",
            "anchor_assignment_sha256": anchor["assignment_sha256"],
            "scope": "exact64",
            "change": {
                "cell": target["cell"],
                "physical_identity": target["physical_identity"],
            },
        },
    )
    candidate_root = tmp_path / "contextual-candidate"
    assert main(
        [
            "backpack",
            "materialize-contextual",
            "--virtual-manifest",
            str(manifest_path),
            "--inventory",
            str(prepared / "OPTION_INVENTORY.json"),
            "--request",
            str(request_path),
            "--output",
            str(candidate_root),
        ]
    ) == 0
    candidate_emitted = json.loads(capsys.readouterr().out)
    assert candidate_emitted["status"] == "PASS"
    candidate_index = [
        json.loads(line)
        for line in (candidate_root / "MATERIALIZATION_INDEX.jsonl").read_text().splitlines()
    ]
    target_members = {member["cell"]: member for member in target["members"]}
    changed_index = [
        row for row in candidate_index if row["cell_id"] in target_members
    ]
    assert {row["cell_id"] for row in changed_index} == set(target_members)
    for row in changed_index:
        assert row["physical_receipt_sha256"] == target_members[row["cell_id"]][
            "physical_receipt_sha256"
        ]
    change_path = candidate_root / "CHANGE.json"
    change = json.loads(change_path.read_text())

    candidate_score_path = tmp_path / "candidate-score.json"
    candidate_score = json.loads(anchor_score_path.read_text())
    candidate_score.update(
        {
            "mean_kld": 0.15,
            "top1_matches": 51200,
            "identities": {
                **score_identities,
                "pack_sha256": change["candidate_pack_sha256"],
            },
            "per_window": [
                {
                    "window_id": f"w{index:02d}",
                    "positions": 1024,
                    "mean_kld": 0.15,
                    "top1_matches": 800,
                }
                for index in range(64)
            ],
        }
    )
    _write_json(candidate_score_path, candidate_score)
    measurement_path = tmp_path / "measurement.json"
    assert main(
        [
            "backpack",
            "record-contextual",
            "--anchor",
            str(prepared / "ANCHOR.json"),
            "--change",
            str(change_path),
            "--anchor-score",
            str(anchor_score_path),
            "--candidate-score",
            str(candidate_score_path),
            "--measurements",
            str(prepared / "MEASUREMENTS.json"),
            "--output",
            str(measurement_path),
        ]
    ) == 0
    measured = json.loads(capsys.readouterr().out)
    assert measured["status"] == "PASS"
    assert measured["delta_mean_kld"] < 0

    ledger_path = tmp_path / "contextual-ledger.json"
    assert main(
        [
            "backpack",
            "value-contextual",
            "--anchor",
            str(prepared / "ANCHOR.json"),
            "--options",
            str(prepared / "OPTION_INVENTORY.json"),
            "--measurements",
            str(prepared / "MEASUREMENTS.json"),
            "--output",
            str(ledger_path),
        ]
    ) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "PASS"

    contextual_solve = tmp_path / "contextual-solve.json"
    assert main(
        [
            "backpack",
            "solve-contextual",
            "--anchor",
            str(prepared / "ANCHOR.json"),
            "--ledger",
            str(ledger_path),
            "--max-changes",
            "1",
            "--uncertainty-multiplier",
            "0",
            "--time-limit-seconds",
            "5",
            "--output",
            str(contextual_solve),
        ]
    ) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "PASS"
