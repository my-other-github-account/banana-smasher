from __future__ import annotations

import hashlib
import json
from pathlib import Path

from banana_smasher.backpack_contextual_candidate import materialize_contextual_change
from banana_smasher.cli import main


def test_materialize_contextual_change_is_public_api() -> None:
    import banana_smasher

    assert banana_smasher.materialize_contextual_change is materialize_contextual_change


def _canonical(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def _write(path: Path, value: object) -> None:
    path.write_bytes(_canonical(value))


def _descriptor(path: Path, *, key: str) -> dict[str, object]:
    payload = path.read_bytes()
    return {key: str(path), "bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}


def test_materialize_contextual_change_builds_candidate_pack_from_artifacts(
    tmp_path,
) -> None:
    basis = "a" * 64
    baseline = tmp_path / "baseline"
    baseline.mkdir()
    assignment_path = baseline / "ASSIGNMENT.json"
    index_path = baseline / "MATERIALIZATION_INDEX.jsonl"
    ledger_path = tmp_path / "options.jsonl"
    solve_path = tmp_path / "solve.json"
    _write(assignment_path, {"0": {"0": {"down": "qtip2"}}})
    index_path.write_bytes(
        _canonical(
            {
                "cell_id": "L000:E000:down",
                "layer": 0,
                "expert": 0,
                "projection": "down",
                "tier": "qtip2",
                "source_key": "qtip2",
                "physical_bytes": 10,
                "activation_artifact_ids": ["shared"],
            }
        )
    )
    ledger_path.write_text("{}\n")
    _write(solve_path, {"basis_sha256": basis})
    source_bindings = {}
    for source in ("qtip2", "qtip3"):
        root = tmp_path / source
        root.mkdir()
        identity = root / "IDENTITY.json"
        _write(identity, {"source": source})
        payload = identity.read_bytes()
        source_bindings[source] = {
            "root": str(root),
            "identity": identity.name,
            "identity_sha256": hashlib.sha256(payload).hexdigest(),
            "identity_bytes": len(payload),
            "basis_sha256": basis,
        }
    assignment_sha = hashlib.sha256(assignment_path.read_bytes()).hexdigest()
    virtual_path = baseline / "BACKPACK_VIRTUAL_MANIFEST.json"
    _write(
        virtual_path,
        {
            "schema": "banana-smasher-backpack-virtual-assignment-v1",
            "status": "PASS_LOGICAL_FULL_WIRE",
            "storage": {
                "kind": "external-family-roots-v1",
                "tensor_payload_copy_bytes": 0,
                "source_roots_bound_once": True,
            },
            "basis_sha256": basis,
            "arm_name": "baseline",
            "assignment_map_sha256": assignment_sha,
            "assignment": {
                "file": assignment_path.name,
                "sha256": assignment_sha,
                "bytes": assignment_path.stat().st_size,
                "rows": 1,
            },
            "materialization_index": {
                "file": index_path.name,
                "sha256": hashlib.sha256(index_path.read_bytes()).hexdigest(),
                "bytes": index_path.stat().st_size,
                "rows": 1,
            },
            "option_ledger": _descriptor(ledger_path, key="path"),
            "solve_input": _descriptor(solve_path, key="path"),
            "source_bindings": source_bindings,
            "source_component_counts": {"qtip2": 1},
            "tier_counts": {"qtip2": 1, "qtip3": 0},
            "byte_accounting": {
                "payload_bytes": 10,
                "activation_bytes": 1,
                "assigned_expert_bytes": 11,
                "fixed_nonexpert_bytes": 5,
                "assigned_package_bytes": 16,
                "tier_payload_bytes": {"qtip2": 10, "qtip3": 0},
            },
            "activated_artifacts": [{"artifact_id": "shared", "bytes": 1}],
            "expert_parameter_denominator": 20,
            "expert_wire_bpw": 4.4,
            "geometry": {"cells": 1, "layers": 1, "experts_per_layer": 1, "projections": ["down"]},
        },
    )
    inventory_path = tmp_path / "inventory.json"
    _write(
        inventory_path,
        {
            "schema": "banana-smasher-contextual-option-inventory-v1",
            "status": "READY",
            "basis_sha256": basis,
            "anchor_assignment_sha256": assignment_sha,
            "options": [
                {
                    "cell": "L000:E000:down",
                    "option": "qtip3",
                    "physical_source_key": "qtip3",
                    "physical_identity": "b" * 64,
                    "payload_bytes": 15,
                    "activations": [{"id": "shared", "bytes": 1}],
                }
            ],
        },
    )
    request_path = tmp_path / "request.json"
    _write(
        request_path,
        {
            "schema": "banana-smasher-contextual-change-request-v1",
            "status": "READY",
            "anchor_assignment_sha256": assignment_sha,
            "scope": "exact64",
            "change": {
                "cell": "L000:E000:down",
                "physical_identity": "b" * 64,
            },
        },
    )

    receipt = materialize_contextual_change(
        virtual_path,
        inventory_path,
        request_path,
        output_root=tmp_path / "candidate",
    )

    candidate = tmp_path / "candidate"
    assignment = json.loads((candidate / "ASSIGNMENT.json").read_text())
    index = json.loads((candidate / "MATERIALIZATION_INDEX.jsonl").read_text())
    change = json.loads((candidate / "CHANGE.json").read_text())
    manifest = json.loads((candidate / "BACKPACK_VIRTUAL_MANIFEST.json").read_text())
    assert receipt["status"] == "PASS"
    assert assignment["0"]["0"]["down"] == "qtip3"
    assert index["source_key"] == "qtip3"
    assert manifest["byte_accounting"]["assigned_package_bytes"] == 21
    assert change["candidate_assignment_sha256"] == manifest["assignment_map_sha256"]
    assert change["candidate_pack_sha256"] == receipt["candidate_pack_sha256"]


def test_cli_delegates_materialize_contextual_to_public_process(
    tmp_path, capsys, monkeypatch
) -> None:
    observed = {}

    def fake_materialize(virtual, inventory, request, **parameters):
        observed.update(
            virtual=virtual,
            inventory=inventory,
            request=request,
            **parameters,
        )
        return {"status": "PASS", "candidate_pack_sha256": "f" * 64}

    monkeypatch.setattr(
        "banana_smasher.backpack_contextual_candidate.materialize_contextual_change",
        fake_materialize,
    )
    status = main(
        [
            "backpack",
            "materialize-contextual",
            "--virtual-manifest",
            str(tmp_path / "virtual.json"),
            "--inventory",
            str(tmp_path / "inventory.json"),
            "--request",
            str(tmp_path / "request.json"),
            "--output",
            str(tmp_path / "candidate"),
        ]
    )

    emitted = json.loads(capsys.readouterr().out)
    assert status == 0
    assert emitted["command"] == "backpack materialize-contextual"
    assert observed["output_root"] == tmp_path / "candidate"
